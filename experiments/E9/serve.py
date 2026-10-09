"""Local reranking HTTP API: one persistent model worker and a bounded queue."""
import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
import time

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, model_validator

from .backend import Backend


class Document(BaseModel):
    document_id: str = Field(min_length=1, max_length=256)
    text: str = Field(min_length=1, max_length=20000)


class Request(BaseModel):
    request_id: str = Field(min_length=1, max_length=256)
    query: str = Field(min_length=1, max_length=2000)
    documents: list[Document] = Field(min_length=1, max_length=256)
    top_k: int = Field(default=10, ge=1, le=256)

    @model_validator(mode='after')
    def unique_ids(self):
        if len({d.document_id for d in self.documents}) != len(self.documents):
            raise ValueError('Document IDs must be unique')
        return self


class ModelWorker:
    def __init__(self, backend, batch_size=8, queue_size=32):
        self.backend_name = backend
        self.batch_size = batch_size
        self.queue = asyncio.Queue(maxsize=queue_size)
        # Load and execute on the same dedicated thread (including MLX stream context).
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='reranker')
        self.backend = None
        self.task = None

    def load(self):
        self.backend = Backend(self.backend_name)
        self.backend.score(['warmup'], ['warmup document'])

    async def start(self):
        await asyncio.get_running_loop().run_in_executor(self.executor, self.load)
        self.task = asyncio.create_task(self.run())

    def infer(self, request):
        docs = sorted(request.documents, key=lambda d: d.document_id)
        ranking = self.backend.rerank(request.query, [d.text for d in docs], self.batch_size)
        return [{'document_id': docs[r['index']].document_id, 'score': r['score']}
                for r in ranking[:request.top_k]]

    async def run(self):
        while True:
            request, future, queued_at = await self.queue.get()
            try:
                if future.cancelled():
                    continue
                started = time.perf_counter()
                ranking = await asyncio.get_running_loop().run_in_executor(
                    self.executor, self.infer, request)
                if not future.done():
                    future.set_result({'request_id': request.request_id,
                        'backend': self.backend_name, 'ranking': ranking,
                        'queue_ms': (started-queued_at)*1000,
                        'processing_ms': (time.perf_counter()-started)*1000})
            except Exception as error:
                if not future.done():
                    future.set_exception(error)
            finally:
                self.queue.task_done()

    async def submit(self, request, timeout=30):
        future = asyncio.get_running_loop().create_future()
        try:
            self.queue.put_nowait((request, future, time.perf_counter()))
        except asyncio.QueueFull:
            raise HTTPException(status_code=503, detail='Rerank queue is full')
        try:
            return await asyncio.wait_for(future, timeout)
        except TimeoutError:
            raise HTTPException(status_code=504, detail='Rerank deadline exceeded')
        except asyncio.CancelledError:
            future.cancel()
            raise

    async def close(self):
        # Drain accepted jobs before stopping the worker.
        await self.queue.join()
        if self.task:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
        self.executor.shutdown(wait=True)


def create_app(backend, batch_size=8, queue_size=32, timeout=30):
    worker = ModelWorker(backend, batch_size, queue_size)

    @asynccontextmanager
    async def lifespan(app):
        await worker.start()
        try:
            yield
        finally:
            await worker.close()

    app = FastAPI(title='tiny-reranker', lifespan=lifespan)

    @app.get('/health')
    async def health():
        return {'ready': worker.backend is not None, 'backend': backend,
                'queue_depth': worker.queue.qsize(), 'pair_batch_size': batch_size}

    @app.post('/rerank')
    async def rerank(request: Request):
        return await worker.submit(request, timeout)

    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend', choices=['pytorch', 'mlx'], required=True)
    parser.add_argument('--port', type=int, default=8099)
    parser.add_argument('--pair-batch-size', type=int, choices=range(1,65), default=8)
    parser.add_argument('--queue-size', type=int, default=32)
    args = parser.parse_args()
    if args.queue_size < 1:
        parser.error('queue-size must be positive')
    import uvicorn
    uvicorn.run(create_app(args.backend, args.pair_batch_size, args.queue_size),
                host='127.0.0.1', port=args.port, access_log=False)


if __name__ == '__main__':
    main()
