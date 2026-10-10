"""Real HTTP and child-process failure tests for both coordinators, without GPU."""
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

import httpx
import pytest

ROOT=Path(__file__).resolve().parents[1]
BINARY=ROOT/'experiments/E18/rust-server/target/release/tiny-reranker-server'


def payload(name='normal'):
    return {'request_id':name,'query':'query','documents':[
        {'document_id':'b','text':'second'},{'document_id':'a','text':'first'}],'top_k':2}


@pytest.fixture(params=['rust','python'])
def service(request,tmp_path):
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0)); port=sock.getsockname()[1]
    if request.param=='rust':
        assert BINARY.exists(), 'Build the debug Rust binary before these integration tests'
        command=[str(BINARY),'--root',str(ROOT),'--python',sys.executable]
    else:
        command=[sys.executable,'-m','experiments.E18.python_server']
    command+=['--backend','mlx','--port',str(port),'--queue-size','1','--timeout','.12',
              '--worker-timeout','1','--worker-script',str(ROOT/'tests/fixtures/e18_worker.py')]
    events=tmp_path/'events.txt'
    log=tmp_path/'server.log'
    with log.open('w') as output:
        proc=subprocess.Popen(command,cwd=ROOT,stdout=output,stderr=output,
                              env={**os.environ,'E18_FIXTURE_EVENTS':str(events)})
    client=httpx.Client(base_url=f'http://127.0.0.1:{port}',timeout=5,trust_env=False)
    try:
        for _ in range(200):
            assert proc.poll() is None,log.read_text()
            try:
                response=client.get('/health')
                if response.status_code==200: break
            except httpx.ConnectError: pass
            time.sleep(.02)
        else: pytest.fail('Readiness timeout')
        pid=response.json()['worker_pid']
        yield client,proc,events
    finally:
        client.close()
        proc.terminate()
        proc.wait(timeout=8)
        # Child must be reaped by the coordinator's shutdown path.
        if 'pid' in locals():
            with pytest.raises(ProcessLookupError): os.kill(pid,0)


def test_schema_matching_limits_and_recovery(service):
    client,_,_=service
    response=client.post('/rerank',json=payload())
    assert response.status_code==200
    assert [r['document_id'] for r in response.json()['ranking']]==['a','b']
    assert response.json()['request_id']=='normal'
    assert response.json()['queue_ms']>=0 and response.json()['processing_ms']>=0
    for change in ({'documents':[]},{'query':'x'*2001},{'top_k':0},
                   {'documents':[{'document_id':'a','text':'x'},{'document_id':'a','text':'y'}]}):
        assert client.post('/rerank',json={**payload(),**change}).status_code==422
    assert client.post('/rerank',content=b'{' ,headers={'Content-Type':'application/json'}).status_code==422
    assert client.post('/rerank',content=b' '*(8*1024*1024),headers={'Content-Type':'application/json'}).status_code==413
    assert client.post('/rerank',json=payload('error')).status_code==500
    assert client.post('/rerank',json=payload('recovery')).status_code==200


def test_queue_full_timeout_and_protocol_remains_aligned(service):
    client,_,events=service
    with ThreadPoolExecutor(max_workers=2) as pool:
        active=pool.submit(client.post,'/rerank',json=payload('slow'))
        for _ in range(100):
            if events.exists() and 'slow' in events.read_text(): break
            time.sleep(.005)
        else: pytest.fail('Slow request did not start')
        queued=pool.submit(client.post,'/rerank',json=payload('expired_queued'))
        for _ in range(100):
            if client.get('/health').json()['queue_depth']==1: break
            time.sleep(.005)
        else: pytest.fail('Second request did not queue')
        assert client.post('/rerank',json=payload('overflow')).status_code==503
        assert active.result().status_code==504
        assert queued.result().status_code==504
    time.sleep(.25)
    assert client.post('/rerank',json=payload('after_timeout')).status_code==200
    assert events.read_text().splitlines()==['slow','after_timeout']


@pytest.mark.parametrize('mode',['crash','badseq','malformed','badids','oversized'])
def test_worker_failure_is_closed_and_structured(service,mode):
    client,_,_=service
    response=client.post('/rerank',json=payload(mode))
    assert response.status_code==502
    assert response.json()['request_id']==mode
    assert client.get('/health').status_code==503
    assert client.post('/rerank',json=payload('after_failure')).status_code==503


def test_idle_worker_death_affects_readiness(service):
    client,_,_=service
    pid=client.get('/health').json()['worker_pid']
    os.kill(pid,9)
    for _ in range(100):
        if client.get('/health').status_code==503: break
        time.sleep(.01)
    else: pytest.fail('Idle worker death did not affect health')
    assert client.post('/rerank',json=payload()).status_code==503
