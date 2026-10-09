"""Queue behavior tests using a controlled slow worker without GPU access."""
import asyncio
import time

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from .serve import ModelWorker, Request


def request(name):
    return Request(request_id=name, query='query', documents=[{'document_id':'d','text':'doc'}])


class SlowWorker(ModelWorker):
    def load(self):
        self.backend = object()

    def infer(self, req):
        time.sleep(.03)
        return [{'document_id': req.request_id, 'score': 1.0}]


def test_queue_matching_limits_timeout_and_cancellation():
    async def check():
        worker = SlowWorker('mlx', queue_size=1)
        await worker.start()
        try:
            first = asyncio.create_task(worker.submit(request('first')))
            await asyncio.sleep(.005)
            second = asyncio.create_task(worker.submit(request('second')))
            await asyncio.sleep(.005)
            with pytest.raises(HTTPException) as exc:
                await worker.submit(request('overflow'))
            assert exc.value.status_code == 503
            a,b = await asyncio.gather(first, second)
            assert a['request_id'] == a['ranking'][0]['document_id'] == 'first'
            assert b['request_id'] == b['ranking'][0]['document_id'] == 'second'
            with pytest.raises(HTTPException) as exc:
                await worker.submit(request('timeout'), timeout=.001)
            assert exc.value.status_code == 504
            await worker.queue.join()
            cancelled = asyncio.create_task(worker.submit(request('cancelled')))
            await asyncio.sleep(.005)
            cancelled.cancel()
            with pytest.raises(asyncio.CancelledError):
                await cancelled
            await worker.queue.join()
            assert (await worker.submit(request('recovery')))['request_id'] == 'recovery'
        finally:
            await worker.close()
    asyncio.run(check())


def test_invalid_documents():
    with pytest.raises(ValidationError):
        Request(request_id='id', query='q', documents=[])
    with pytest.raises(ValidationError):
        Request(request_id='id', query='q', documents=[{'document_id':'x','text':'a'},
                                                     {'document_id':'x','text':'b'}])


def test_worker_recovers_after_inference_failure():
    class FailingWorker(SlowWorker):
        def infer(self, req):
            if req.request_id == 'fail':
                raise RuntimeError('controlled failure')
            return super().infer(req)
    async def check():
        worker = FailingWorker('pytorch')
        await worker.start()
        try:
            with pytest.raises(RuntimeError, match='controlled failure'):
                await worker.submit(request('fail'))
            assert (await worker.submit(request('good')))['request_id'] == 'good'
        finally:
            await worker.close()
    asyncio.run(check())
