"""Python HTTP reference using exactly the same child worker as Rust."""
import argparse
import asyncio
from contextlib import asynccontextmanager
import json
import math
from pathlib import Path
import sys
import time

from fastapi import FastAPI, Request as HTTPRequest
from fastapi.responses import JSONResponse
from .schema import Request
from .common import ROOT
from .worker import MAX_FRAME


class Failure(Exception):
    def __init__(self,status,code,detail,request_id=None):
        self.status,self.code,self.detail,self.request_id=status,code,detail,request_id


class Manager:
    def __init__(self,backend,batch,queue_size,timeout,worker_timeout,worker_script=None):
        self.backend,self.batch=backend,batch
        self.queue=asyncio.Queue(maxsize=queue_size)
        self.timeout,self.worker_timeout=timeout,worker_timeout
        self.script=worker_script
        self.ready=False
        self.process=None
        self.task=None
        self.monitor=None
        self.metadata={}

    async def frame(self):
        wire=await self.process.stdout.readline()
        if not wire or len(wire)>MAX_FRAME or not wire.endswith(b'\n'):
            raise RuntimeError('Invalid frame or worker exited')
        return json.loads(wire)

    async def start(self):
        command=[sys.executable,'-u']
        command+= [str(self.script)] if self.script else ['-m','experiments.E18.worker']
        command+=['--backend',self.backend,'--pair-batch-size',str(self.batch)]
        self.process=await asyncio.create_subprocess_exec(*command,cwd=ROOT,
            stdin=asyncio.subprocess.PIPE,stdout=asyncio.subprocess.PIPE,limit=MAX_FRAME+1)
        try:
            ready=await asyncio.wait_for(self.frame(),900)
            if ready.get('type')!='ready' or ready.get('protocol_version')!=1 or \
                ready.get('backend')!=self.backend or ready.get('pair_batch_size')!=self.batch or \
                ready.get('max_length')!=512 or len(ready.get('checkpoint_sha256',''))!=64 or \
                any(c not in '0123456789abcdefABCDEF' for c in ready['checkpoint_sha256']):
                raise RuntimeError('Worker readiness metadata mismatch')
            self.metadata={'coordinator':'python','backend':self.backend,'pair_batch_size':self.batch,
                'max_length':512,'checkpoint_sha256':ready['checkpoint_sha256'],
                'worker_pid':self.process.pid,'queue_capacity':self.queue.maxsize}
            self.ready=True
            self.task=asyncio.create_task(self.run())
            self.monitor=asyncio.create_task(self.watch_exit())
        except BaseException:
            await self.stop_child()
            raise

    async def watch_exit(self):
        await self.process.wait()
        self.ready=False

    async def infer(self,seq,request):
        wire=json.dumps({'type':'rerank','seq':seq,'request':request.model_dump()},
                        ensure_ascii=False,allow_nan=False).encode()+b'\n'
        if len(wire)>MAX_FRAME:
            raise RuntimeError('Request exceeds frame limit')
        self.process.stdin.write(wire)
        await self.process.stdin.drain()
        response=await self.frame()
        if response.get('type')!='result' or response.get('seq')!=seq or response.get('request_id')!=request.request_id or \
            (('error' in response)==('ranking' in response)):
            raise RuntimeError('Worker response sequence/ID/type mismatch')
        if 'error' in response:
            raise Failure(500,'inference_failed','Model inference failed',request.request_id)
        ranking=response['ranking']
        ids={d.document_id for d in request.documents}
        if len(ranking)!=min(request.top_k,len(ids)) or \
            len({r['document_id'] for r in ranking})!=len(ranking) or \
            any(r['document_id'] not in ids or not math.isfinite(r['score']) for r in ranking) or \
            ranking!=sorted(ranking,key=lambda r:(-r['score'],r['document_id'])):
            raise RuntimeError('Invalid worker ranking')
        return ranking

    async def run(self):
        seq=0
        fatal=False
        while True:
            item=await self.queue.get()
            try:
                if item is None:
                    break
                request,future,queued=item
                if future.done():
                    continue
                seq+=1
                started=time.perf_counter()
                try:
                    ranking=await asyncio.wait_for(self.infer(seq,request),self.worker_timeout)
                    result={'request_id':request.request_id,'backend':self.backend,'ranking':ranking,
                            'queue_ms':(started-queued)*1000,'processing_ms':(time.perf_counter()-started)*1000}
                    if not future.done(): future.set_result(result)
                except Failure as error:
                    if not future.done(): future.set_exception(error)
                except Exception:
                    fatal=True
                    if not future.done(): future.set_exception(Failure(502,'worker_failed','Model worker failed',request.request_id))
                    break
            finally:
                self.queue.task_done()
        self.ready=False
        if fatal:
            while not self.queue.empty():
                item=self.queue.get_nowait()
                if item is not None and not item[1].done():
                    item[1].set_exception(Failure(503,'worker_unavailable','Model worker is unavailable',item[0].request_id))
                self.queue.task_done()
        await self.stop_child()

    async def submit(self,request):
        if not self.ready:
            raise Failure(503,'worker_unavailable','Model worker is unavailable',request.request_id)
        future=asyncio.get_running_loop().create_future()
        try:
            self.queue.put_nowait((request,future,time.perf_counter()))
        except asyncio.QueueFull:
            raise Failure(503,'queue_full','Rerank queue is full',request.request_id)
        try:
            return await asyncio.wait_for(future,self.timeout)
        except TimeoutError:
            raise Failure(504,'deadline_exceeded','Rerank deadline exceeded',request.request_id)

    async def stop_child(self):
        if self.process is None: return
        self.process.stdin.close()
        try:
            await asyncio.wait_for(self.process.wait(),5)
        except TimeoutError:
            self.process.kill()
            await self.process.wait()

    async def close(self):
        self.ready=False
        if self.task and not self.task.done():
            await self.queue.put(None)
            await self.task
        if self.monitor: await self.monitor


def create_app(backend='mlx',batch=8,queue_size=32,timeout=30,worker_timeout=60,worker_script=None):
    manager=Manager(backend,batch,queue_size,timeout,worker_timeout,worker_script)
    @asynccontextmanager
    async def lifespan(app):
        await manager.start()
        try: yield
        finally: await manager.close()
    app=FastAPI(lifespan=lifespan)
    @app.exception_handler(Failure)
    async def failure(request,error):
        return JSONResponse(status_code=error.status,content={'code':error.code,'detail':error.detail,'request_id':error.request_id})
    @app.get('/health')
    async def health():
        return JSONResponse(status_code=200 if manager.ready else 503,
            content={**manager.metadata,'ready':manager.ready,'queue_depth':manager.queue.qsize()})
    @app.post('/rerank')
    async def rerank(request:HTTPRequest):
        body=bytearray()
        async for chunk in request.stream():
            body.extend(chunk)
            if len(body)>MAX_FRAME-1024:
                raise Failure(413,'invalid_request','Request body too large')
        try:
            payload=Request.model_validate_json(body)
        except ValueError:
            raise Failure(422,'invalid_request','Invalid rerank request')
        return await manager.submit(payload)
    return app


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--backend',choices=['pytorch','mlx'],default='mlx')
    p.add_argument('--port',type=int,default=8101)
    p.add_argument('--pair-batch-size',type=int,choices=range(1,65),default=8)
    p.add_argument('--queue-size',type=int,default=32)
    p.add_argument('--timeout',type=float,default=30)
    p.add_argument('--worker-timeout',type=float,default=60)
    p.add_argument('--worker-script',type=Path)
    a=p.parse_args()
    if not 1<=a.queue_size<=1024 or not 0<a.timeout<=a.worker_timeout<=3600:
        p.error('Invalid queue size or timeouts')
    import uvicorn
    uvicorn.run(create_app(a.backend,a.pair_batch_size,a.queue_size,a.timeout,a.worker_timeout,a.worker_script),
                host='127.0.0.1',port=a.port,access_log=False)


if __name__=='__main__': main()
