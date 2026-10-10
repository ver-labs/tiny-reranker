"""Fixed-concurrency closed-loop loopback with per-response correctness audits."""
import argparse
import asyncio
import gzip
import json
import os
import random
import socket
import subprocess
import sys
import time
import traceback
import httpx
import numpy as np
from .common import ROOT,HERE,save,sha,verify_plan,environment
from .direct_benchmark import selected
from .accuracy import read

BINARY=HERE/'rust-server/target/release/tiny-reranker-server'


def payload(row,k,index):
    return {'request_id':f"{row['dataset']}:{row['query_id']}:K{k}:{index}",'query':row['query'],
            'documents':[{'document_id':d,'text':t} for d,t in zip(row['document_ids'][:k],row['texts'][:k])],'top_k':k}


def validate(body,request,ref):
    if body['request_id']!=request['request_id']:raise ValueError('Wrong request ID')
    ranking=body['ranking'];ids=[r['document_id'] for r in ranking]
    if ids!=ref['ranking']:raise ValueError('Ranking differs from same-path direct audit')
    expected=dict(zip(ref['ids'],ref['scores']));values=np.array([r['score'] for r in ranking]);truth=np.array([expected[d] for d in ids])
    if not np.isfinite(values).all() or not np.all(np.abs(values-truth)<=2e-5+1e-4*np.abs(truth)):
        raise ValueError('HTTP score parity failure')
    if any(not np.isfinite(body[f]) or body[f]<0 for f in ('queue_ms','processing_ms')):raise ValueError('Invalid timing')
    return float(np.max(np.abs(values-truth)))


def aggregate(obs,seconds,k):
    good=[r for r in obs if r['success']]
    if seconds<=0:raise ValueError('Nonpositive elapsed time')
    result={'successful_requests':len(good),'successful_pairs':len(good)*k,'failures':len(obs)-len(good),'timeouts':sum(r.get('timeout',False) for r in obs),'seconds':seconds,'requests_per_second':len(good)/seconds,'pairs_per_second':len(good)*k/seconds}
    for name,field in [('latency','latency_ms'),('queue','queue_ms'),('processing','processing_ms')]:
        result[name+'_ms']={f'p{p}':float(np.percentile([r[field] for r in good],p)) for p in (50,95,99)} if good else None
    result['validation_seconds']=sum(r.get('validation_seconds',0) for r in obs)
    return result


async def load(client,requests,c,k,refs):
    q=asyncio.Queue()
    for index,(request,ref) in enumerate(requests):q.put_nowait((index,request,ref))
    obs=[];start=time.perf_counter();cpu=time.process_time()
    async def consumer():
        while not q.empty():
            index,request,ref=q.get_nowait();begin=time.perf_counter()
            record={'index':index,'request_id':request['request_id'],'start_offset':begin-start}
            try:
                response=await client.post('/rerank',json=request);end=time.perf_counter();record.update(latency_ms=1000*(end-begin),end_offset=end-start)
                validation=time.perf_counter();response.raise_for_status();body=response.json();error=validate(body,request,ref)
                record.update(success=True,queue_ms=body['queue_ms'],processing_ms=body['processing_ms'],ranking=body['ranking'],max_score_error=error,validation_seconds=time.perf_counter()-validation)
            except Exception as error:
                record.update(success=False,error=repr(error),timeout=isinstance(error,httpx.TimeoutException));record.setdefault('latency_ms',1000*(time.perf_counter()-begin))
            obs.append(record)
    await asyncio.gather(*(consumer() for _ in range(c)))
    seconds=time.perf_counter()-start;obs.sort(key=lambda r:r['index'])
    assert [r['request_id'] for r in obs]==[p['request_id'] for p,ref in requests]
    return obs,{**aggregate(obs,seconds,k),'client_cpu_seconds':time.process_time()-cpu,'client_cpu_fraction_of_one_core':(time.process_time()-cpu)/seconds}


def rss(pids):
    result={}
    for pid in pids:
        try:result[str(pid)]=int(subprocess.check_output(['ps','-o','rss=','-p',str(pid)],text=True).strip())*1024
        except Exception:result[str(pid)]=None
    return {'bytes':result,'scope':'Coordinator and worker process RSS snapshots; shared memory may be counted twice; not total system peak'}


async def measure(session,coordinator,path,study='primary',primary_only=False):
    plan=verify_plan();prefix='http' if study=='primary' else study+'_http'
    dest=HERE/'raw'/f'{prefix}_s{session}_{coordinator}_{path}.json'
    if dest.exists() and json.loads(dest.read_text()).get('status')=='COMPLETE':
        assert json.loads(dest.read_text())['plan_sha256']==sha(HERE/'plan.json');return
    audit=json.loads((HERE/f'mps_{path}_accuracy.json').read_text());assert audit['status'] in ('VERIFIED','FAILED_ACCURACY') and audit['plan_sha256']==sha(HERE/'plan.json')
    assert sha(HERE/audit['artifact'])==audit['artifact_sha256']
    refs={(r['dataset'],r['query_id'],r['k']):r for r in read(HERE/audit['artifact'])}
    data=selected(plan['timing_ids']);backend='mlx' if path in ('mlx','mlx_compiled','metal','metal_compiled') else 'pytorch'
    with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    command=[str(BINARY),'--python',sys.executable,'--root',str(ROOT),'--worker-script',str(HERE/'worker.py')] if coordinator=='rust' else [sys.executable,'-u','-m','experiments.E18.python_server']
    command+=['--backend',backend,'--port',str(port),'--pair-batch-size','16','--queue-size','32','--timeout','120','--worker-timeout','120']
    env={**os.environ,'E18_INFERENCE_PATH':path};log=HERE/f'{prefix}_s{session}_{coordinator}_{path}_server.log'
    start=time.perf_counter()
    with log.open('w') as f:proc=subprocess.Popen(command,cwd=ROOT,env=env,stdout=f,stderr=f)
    result={'session':session,'coordinator':coordinator,'path':path,'plan_sha256':sha(HERE/'plan.json'),'environment':environment(),'accuracy_status':audit['status'],'study':study,'primary_only':primary_only,'runs':[]}
    raw=HERE/'raw'/f'{prefix}_s{session}_{coordinator}_{path}_responses.jsonl.gz';raw.parent.mkdir(exist_ok=True)
    try:
        async with httpx.AsyncClient(base_url=f'http://127.0.0.1:{port}',timeout=120,trust_env=False,limits=httpx.Limits(max_connections=8,max_keepalive_connections=8)) as client:
            while time.perf_counter()-start<900:
                if proc.poll() is not None:raise RuntimeError(log.read_text()[-3000:])
                try:
                    response=await client.get('/health')
                    if response.status_code==200 and response.json()['ready']:break
                except httpx.ConnectError:pass
                await asyncio.sleep(.1)
            else:raise TimeoutError('Server startup >900s')
            health=response.json();assert health['max_length']==512 and health['checkpoint_sha256']==plan['model']['file_sha256']['model.safetensors'] and health['pair_batch_size']==16
            result.update(startup_seconds=time.perf_counter()-start,health=health,rss_before=rss([proc.pid,health['worker_pid']]))
            invalid=await client.post('/rerank',json={**payload(data[0],10,0),'documents':[]});assert invalid.status_code==422
            cells=[(k,c) for k in ([100] if primary_only else plan['document_counts']) for c in plan['concurrency']];random.Random(1801+session).shuffle(cells)
            with gzip.open(raw,'wt') as f:
                for rep in range(2):
                    for k,c in cells[rep*3:]+cells[:rep*3]:
                        order=list(data);random.Random(1801+session*100+rep).shuffle(order)
                        requests=[(payload(r,k,i),refs[r['dataset'],r['query_id'],k]) for i,r in enumerate(order)]
                        obs,stats=await load(client,requests,c,k,refs)
                        cell={'session':session,'repetition':rep,'path':coordinator+'/'+path,'k':k,'concurrency':c,'batch':16,'length':512}
                        for record in obs:f.write(json.dumps({**cell,**record},allow_nan=False)+'\n')
                        f.flush();result['runs'].append({**cell,**stats});save(dest,result)
                        print(session,coordinator,path,k,c,rep,stats['successful_requests'],flush=True)
            result.update(status='COMPLETE',rss_after=rss([proc.pid,health['worker_pid']]),response_artifact=str(raw.relative_to(HERE)),response_sha256=sha(raw))
    except Exception as error:result.update(status='FAILED_EXECUTION',error=repr(error),traceback=traceback.format_exc())
    finally:
        proc.terminate()
        try:proc.wait(timeout=20)
        except subprocess.TimeoutExpired:proc.kill();proc.wait()
        result['end_environment']=environment();save(dest,result)
    print(result['status'],flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--session',type=int,required=True);p.add_argument('--coordinator',choices=['python','rust'],required=True);p.add_argument('--path',required=True);p.add_argument('--study',default='primary');p.add_argument('--primary-only',action='store_true');a=p.parse_args();asyncio.run(measure(a.session,a.coordinator,a.path,a.study,a.primary_only))

if __name__=='__main__':main()
