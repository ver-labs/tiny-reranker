"""Compare HTTP coordinators while holding the child-worker protocol fixed."""
import asyncio
import itertools
import json
import os
from pathlib import Path
import socket
import statistics
import subprocess
import sys
import time

import httpx
import numpy as np

from experiments.S1.backend import ROOT, CHECKPOINT, REFERENCE, sha256, write_json
from experiments.S1.benchmark_serving import run_requests
from scripts.evaluate_initial import CANDIDATES

HERE=Path(__file__).resolve().parent
BINARY=ROOT/'rust-server/target/release/tiny-reranker-server'


async def measure(coordinator,backend,round_index,payloads,refs):
    with socket.socket() as s:
        s.bind(('127.0.0.1',0));port=s.getsockname()[1]
    command=[str(BINARY)] if coordinator=='rust' else [sys.executable,'-u','-m','experiments.E12.python_server']
    command+=['--backend',backend,'--port',str(port),'--pair-batch-size','8','--queue-size','32',
              '--timeout','30','--worker-timeout','60']
    log_path=HERE/f'round_{round_index}_{coordinator}_{backend}_server.log'
    with log_path.open('w') as log:
        proc=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    expected={r['query_id']:[d['document_id'] for d in r['ranking']] for r in refs}
    reference_scores={r['query_id']:{d['document_id']:d['score'] for d in r['ranking']} for r in refs}
    health=None
    try:
        async with httpx.AsyncClient(base_url=f'http://127.0.0.1:{port}',timeout=30,
                trust_env=False,limits=httpx.Limits(max_connections=16)) as client:
            started=time.perf_counter()
            for _ in range(600):
                if proc.poll() is not None: raise RuntimeError(log_path.read_text())
                try:
                    response=await client.get('/health')
                    if response.status_code==200 and response.json()['ready']:
                        health=response.json();break
                except httpx.ConnectError: pass
                await asyncio.sleep(.1)
            else: raise RuntimeError('Server startup timed out')
            if health['checkpoint_sha256']!=sha256(CHECKPOINT) or health['max_length']!=128 or health['pair_batch_size']!=8:
                raise RuntimeError('Worker checkpoint/configuration mismatch')
            ready_seconds=time.perf_counter()-started
            invalid=await client.post('/rerank',json={**payloads[0],'documents':[]})
            assert invalid.status_code==422
            max_error=0.
            for payload in payloads[:16]:
                response=await client.post('/rerank',json=payload)
                response.raise_for_status()
                actual=response.json()['ranking']
                assert [d['document_id'] for d in actual]==expected[payload['request_id']]
                scores=[d['score'] for d in actual]
                baseline=[reference_scores[payload['request_id']][d['document_id']] for d in actual]
                np.testing.assert_allclose(scores,baseline,atol=2e-5,rtol=1e-4)
                max_error=max(max_error,max(abs(a-b) for a,b in zip(scores,baseline)))
            top=await client.post('/rerank',json={**payloads[0],'top_k':1})
            assert top.status_code==200 and len(top.json()['ranking'])==1
            results={str(c):[] for c in (1,4,8)}
            for repeat in range(5):
                for c in ((1,4,8) if repeat%2==0 else (8,4,1)):
                    row=await run_requests(client,payloads,c,expected)
                    results[str(c)].append(row)
                    print(f'round={round_index} {coordinator}/{backend} repeat={repeat+1} concurrency={c} '
                          f'{row["requests_per_second"]:.1f} requests/s',flush=True)
            return {'coordinator':coordinator,'backend':backend,'round':round_index,
                    'ready_seconds':ready_seconds,'worker_metadata':health,'warmup_requests':16,
                    'max_warmup_score_difference_from_validation':max_error,'results':results}
    finally:
        proc.terminate()
        try: proc.wait(timeout=10)
        except subprocess.TimeoutExpired: proc.kill();proc.wait()
        if health:
            for _ in range(50):
                try: os.kill(health['worker_pid'],0)
                except ProcessLookupError: break
                await asyncio.sleep(.1)
            else:
                os.kill(health['worker_pid'],9)
                raise RuntimeError('Coordinator failed to reap worker at shutdown')


async def main():
    if not BINARY.exists(): raise RuntimeError('Build the release Rust binary before benchmarking')
    with CANDIDATES.open() as f:
        rows=[json.loads(s) for s in itertools.islice(f,128)]
    with (REFERENCE/'epoch_0003_rankings.jsonl').open() as f:
        refs=[json.loads(s) for s in itertools.islice(f,128)]
    assert [r['query_id'] for r in rows]==[r['query_id'] for r in refs]
    payloads=[{'request_id':r['query_id'],'query':r['query'],'top_k':len(r['candidates']),
        'documents':[{'document_id':c['document_id'],'text':c['text']} for c in r['candidates']]} for r in rows]
    rounds=[]
    first=[('python','mlx'),('rust','mlx'),('python','pytorch'),('rust','pytorch')]
    for round_index,order in enumerate((first,list(reversed(first))),1):
        for coordinator,backend in order:
            result=await measure(coordinator,backend,round_index,payloads,refs)
            rounds.append(result)
            write_json(HERE/'benchmark_raw.json',{'rounds':rounds})
    summary=[]
    for backend in ('mlx','pytorch'):
        for c in (1,4,8):
            row={'backend':backend,'concurrency':c}
            for coordinator in ('python','rust'):
                observations=[x for r in rounds if r['coordinator']==coordinator and r['backend']==backend
                              for x in r['results'][str(c)]]
                row[coordinator]={k:statistics.median([x[k] for x in observations])
                    for k in ('requests_per_second','median_ms','p95_ms','median_queue_ms','median_processing_ms')}
                row[coordinator]['throughput_min']=min(x['requests_per_second'] for x in observations)
                row[coordinator]['throughput_max']=max(x['requests_per_second'] for x in observations)
                row[coordinator]['successful_requests']=sum(x['successful_requests'] for x in observations)
                row[coordinator]['failures']=sum(x['failures'] for x in observations)
            row['rust_throughput_speedup']=row['rust']['requests_per_second']/row['python']['requests_per_second']
            summary.append(row)
    result={'experiment':'E12: Rust vs Python HTTP coordinators; identical child-worker protocol',
            'checkpoint':str(CHECKPOINT.relative_to(ROOT)),'checkpoint_sha256':sha256(CHECKPOINT),
            'rust_binary_sha256':sha256(BINARY),'cargo_lock_sha256':sha256(ROOT/'rust-server/Cargo.lock'),
            'code_sha256':{str(p.relative_to(ROOT)):sha256(p) for p in [
                *sorted((ROOT/'rust-server/src').glob('*.rs')),HERE/'worker.py',HERE/'python_server.py',HERE/'benchmark.py']},
            'dataset_split':'validation','candidate_sha256':sha256(CANDIDATES),'test_split_touched':False,
            'requests_per_repeat':128,'documents_per_request':2,'repeats_per_configuration':10,
            'precision':'FP32; MLX_ENABLE_TF32=0; Apple M5 GPU','pair_batch_size':8,'max_length':128,
            'queue_capacity':32,'model_workers':1,'cross_request_batching':False,
            'scope':'loopback HTTP, queueing, JSON pipe round trip, CPU tokenization, GPU scoring, sorting and JSON response',
            'fairness':'Both coordinators launch experiments.E12.worker; model loads once; same pipe protocol, weights, batching and workloads.',
            'ordering':'Two sequential isolated rounds; reverse coordinator/backend order in round two.',
            'limitations':['Short local two-document workload; variation is retained, not a universal speed ranking.',
                           'New Python reference includes IPC; comparison with S1 would combine coordinator and worker-transport changes.',
                           'HTTP timeouts discard responses but do not cancel already running GPU inference.',
                           'Worker failures make service unavailable; automatic restart is not implemented.'],
            'results':summary}
    write_json(HERE/'comparison.json',result)
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__': asyncio.run(main())
