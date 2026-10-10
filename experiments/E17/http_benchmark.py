"""Matched existing E12 coordinators, realistic requests, raw response audits."""
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
import numpy as np
import httpx
from .common import ROOT, HERE, rows, key, payload, sha, save, verify_plan, environment

BINARY=ROOT/'rust-server/target/release/tiny-reranker-server'
PATHS=[('python','pytorch'),('rust','pytorch'),('python','mlx'),('rust','mlx')]


def load_audit(device='mps'):
    report=json.loads((HERE/f'{device}_accuracy.json').read_text())
    assert report['plan_sha256']==sha(HERE/'plan.json')
    assert sha(HERE/report['audit_file'])==report['audit_sha256']
    with gzip.open(HERE/report['audit_file'],'rt') as f:
        records={key(r,r['k']):r for r in map(json.loads,f)}
    return report,records


def validate_response(body, request, reference, path, tolerance):
    if body['request_id']!=request['request_id']:raise ValueError('Wrong request ID')
    ranking=body['ranking'];ids=[d['document_id'] for d in ranking]
    if ids!=reference['rankings'][path]:raise ValueError('Ranking differs from direct inference')
    values=[d['score'] for d in ranking]
    original=dict(zip(reference['candidate_ids'],reference['scores'][path]))
    expected=[original[d] for d in ids]
    if not np.isfinite(values).all() or not np.allclose(values,expected,atol=tolerance['http_atol'],rtol=tolerance['http_rtol']):
        raise ValueError('Returned scores fail direct-inference tolerance')
    if not all(np.isfinite(body[f]) and body[f]>=0 for f in ('queue_ms','processing_ms')):
        raise ValueError('Invalid server timings')
    return float(np.max(np.abs(np.asarray(values)-expected)))


def aggregate(observations, seconds, k):
    good=[r for r in observations if r['success']]
    if not good:raise RuntimeError('No successful responses')
    latency=[r['latency_ms'] for r in good]
    return {'successful_requests':len(good),'successful_pairs':len(good)*k,
            'failures':len(observations)-len(good),'timeouts':sum(r.get('timeout',False) for r in observations),
            'wall_seconds':seconds,'requests_per_second':len(good)/seconds,
            'pairs_per_second':len(good)*k/seconds,
            'p50_ms':float(np.percentile(latency,50)),'p95_ms':float(np.percentile(latency,95)),
            'p99_ms':float(np.percentile(latency,99)),
            'queue_p50_ms':float(np.median([r['queue_ms'] for r in good])),
            'processing_p50_ms':float(np.median([r['processing_ms'] for r in good]))}


async def load(client,payloads,c,k,refs,path,tolerance):
    queue=asyncio.Queue()
    for index,p in enumerate(payloads):queue.put_nowait((index,p))
    observations=[];started=time.perf_counter()
    async def consumer():
        while not queue.empty():
            index,p=queue.get_nowait();begin=time.perf_counter()
            try:
                response=await client.post('/rerank',json=p);end=time.perf_counter()
                response.raise_for_status();body=response.json()
                error=validate_response(body,p,refs[p['request_id']],path,tolerance)
                row={'index':index,'request_id':p['request_id'],'success':True,
                     'start_offset_seconds':begin-started,'end_offset_seconds':end-started,
                     'latency_ms':(end-begin)*1000,'queue_ms':body['queue_ms'],
                     'processing_ms':body['processing_ms'],'max_score_error':error,
                     'ranking':body['ranking']}
            except Exception as e:
                row={'index':index,'request_id':p['request_id'],'success':False,
                     'latency_ms':(time.perf_counter()-begin)*1000,'error':str(e),
                     'timeout':isinstance(e,httpx.TimeoutException)}
            observations.append(row)
    await asyncio.gather(*(consumer() for _ in range(c)))
    seconds=time.perf_counter()-started
    observations.sort(key=lambda r:r['index'])
    assert [r['request_id'] for r in observations]==[p['request_id'] for p in payloads]
    return {'observations':observations,**aggregate(observations,seconds,k)}


def rss(pids):
    result={}
    for pid in pids:
        try:result[str(pid)]=int(subprocess.check_output(['ps','-o','rss=','-p',str(pid)],text=True).strip())*1024
        except (OSError,ValueError,subprocess.SubprocessError):result[str(pid)]=None
    return result


async def measure(session,coordinator,backend,selected,plan,refs):
    with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    command=[str(BINARY)] if coordinator=='rust' else [sys.executable,'-u','-m','experiments.E12.python_server']
    command+=['--backend',backend,'--port',str(port),'--pair-batch-size',str(plan['pair_microbatch']),
              '--queue-size','32','--timeout','30','--worker-timeout','60']
    logfile=HERE/f'session_{session}_{coordinator}_{backend}_server.log'
    started=time.perf_counter()
    with logfile.open('w') as f:proc=subprocess.Popen(command,cwd=ROOT,stdout=f,stderr=f)
    health=None
    result={'session':session,'coordinator':coordinator,'backend':backend,'runs':[]}
    direct_path='stock' if backend=='pytorch' else 'mlx'
    try:
        async with httpx.AsyncClient(base_url=f'http://127.0.0.1:{port}',timeout=30,trust_env=False,
                    limits=httpx.Limits(max_connections=8,max_keepalive_connections=8)) as client:
            for _ in range(600):
                if proc.poll() is not None:raise RuntimeError(logfile.read_text())
                try:
                    response=await client.get('/health')
                    if response.status_code==200 and response.json()['ready']:
                        health=response.json();break
                except httpx.ConnectError:pass
                await asyncio.sleep(.1)
            else:raise RuntimeError('Startup timeout')
            assert health['checkpoint_sha256']==plan['checkpoint_sha256'] and health['pair_batch_size']==plan['pair_microbatch'] and health['max_length']==128
            result.update(ready_seconds=time.perf_counter()-started,health=health,rss_before=rss([proc.pid,health['worker_pid']]))
            invalid=await client.post('/rerank',json={**payload(selected[0],10),'documents':[]})
            assert invalid.status_code==422
            for row in (selected[0],selected[8]):
                for k in plan['document_counts']:
                    p=payload(row,k);response=await client.post('/rerank',json=p);response.raise_for_status()
                    validate_response(response.json(),p,refs[p['request_id']],direct_path,plan['accuracy_tolerance'])
            cells=[(k,c) for k in plan['document_counts'] for c in plan['concurrency']]
            random.Random(plan['seed']+session).shuffle(cells)
            for repeat in range(plan['repeats_per_session']):
                shift=repeat*3
                for k,c in cells[shift:]+cells[:shift]:
                    payloads=[payload(r,k) for r in selected]
                    run=await load(client,payloads,c,k,refs,direct_path,plan['accuracy_tolerance'])
                    run.update(repeat=repeat,documents=k,concurrency=c)
                    result['runs'].append(run)
                    save(HERE/f'http_s{session}_{coordinator}_{backend}.json',result)
                    print(f'session={session} {coordinator}/{backend} repeat={repeat} K={k} C={c} '
                          f'{run["p50_ms"]:.1f}ms {run["requests_per_second"]:.1f}req/s failures={run["failures"]}',flush=True)
            result['rss_after']=rss([proc.pid,health['worker_pid']])
            result['status']='complete'
            save(HERE/f'http_s{session}_{coordinator}_{backend}.json',result)
    finally:
        proc.terminate()
        try:await asyncio.to_thread(proc.wait,10)
        except subprocess.TimeoutExpired:proc.kill();await asyncio.to_thread(proc.wait)
        if health:
            for _ in range(50):
                try:os.kill(health['worker_pid'],0)
                except ProcessLookupError:break
                await asyncio.sleep(.1)
            else:
                os.kill(health['worker_pid'],9)
                raise RuntimeError('Coordinator did not reap worker')
    return result


async def main():
    p=argparse.ArgumentParser();p.add_argument('--session',type=int,required=True,choices=range(4));a=p.parse_args()
    plan=verify_plan();report,refs=load_audit()
    assert report['accuracy']['stock']['accuracy_qualified'] and report['accuracy']['mlx']['accuracy_qualified']
    data={(r['dataset'],r['query_id']):r for r in rows()};selected=[data[tuple(k)] for k in plan['http_subset']]
    order=PATHS.copy();random.Random(plan['seed']).shuffle(order);order=order[a.session:]+order[:a.session]
    conditions={'plan_sha256':sha(HERE/'plan.json'),'session':a.session,'order':order,
                'binary_sha256':sha(BINARY),'before':environment(),
                'memory_scope':'Coordinator/worker RSS sampled before/after; not peak, unified GPU memory may overlap RSS.',
                'throughput_scope':'Entire client wave including response validation and task scheduling; per-request latency stops before validation.'}
    save(HERE/f'http_session_{a.session}.json',conditions)
    for coordinator,backend in order:await measure(a.session,coordinator,backend,selected,plan,refs)
    conditions['after']=environment();conditions['status']='complete'
    save(HERE/f'http_session_{a.session}.json',conditions)


if __name__=='__main__':asyncio.run(main())
