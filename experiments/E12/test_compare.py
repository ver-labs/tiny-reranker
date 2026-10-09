"""Fixed final-test evaluation: four HTTP configurations and direct batch sweep."""
import asyncio
from collections import Counter
import itertools
import json
import math
import os
from pathlib import Path
import socket
import statistics
import subprocess
import sys
import time

import httpx

from experiments.S1.backend import ROOT, CHECKPOINT, HERE as S1, sha256, write_json
from experiments.S1.sweep import summarize
from scripts.data import split_for_query
from scripts.evaluate_initial import CANDIDATE_MANIFEST, ranking_metrics, check_metrics
from .benchmark import BINARY

HERE=Path(__file__).resolve().parent
CANDIDATES=ROOT/'data/processed/eval_test_candidates.jsonl'
BATCHES=[1,2,4,8,16,32,64,128]
CONCURRENCY=4


def payload(row):
    return {'request_id':row['query_id'],'query':row['query'],'top_k':len(row['candidates']),
            'documents':[{'document_id':c['document_id'],'text':c['text']} for c in row['candidates']]}


async def evaluate(coordinator,backend,rows,plan,baseline_path):
    name=f'test_{coordinator}_{backend}'
    summary_path=HERE/f'{name}_summary.json'
    rankings_path=HERE/f'{name}_rankings.jsonl'
    if summary_path.exists():
        saved=json.loads(summary_path.read_text())
        if saved['plan_sha256']!=sha256(HERE/'test_plan.json') or sha256(rankings_path)!=saved['rankings_sha256']:
            raise RuntimeError('Saved test evaluation differs from fixed plan or rankings')
        print(f'Reuse completed {name}',flush=True)
        return saved
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    command=[str(BINARY)] if coordinator=='rust' else [sys.executable,'-u','-m','experiments.E12.python_server']
    command+=['--backend',backend,'--port',str(port),'--pair-batch-size','8','--queue-size','32',
              '--timeout','30','--worker-timeout','60']
    log_path=HERE/f'{name}_server.log'
    with log_path.open('w') as log:
        proc=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    metadata=None
    baseline=None
    try:
        async with httpx.AsyncClient(base_url=f'http://127.0.0.1:{port}',timeout=30,trust_env=False,
                limits=httpx.Limits(max_connections=CONCURRENCY)) as client:
            for _ in range(600):
                if proc.poll() is not None: raise RuntimeError(log_path.read_text())
                try:
                    health=await client.get('/health')
                    if health.status_code==200 and health.json()['ready']:
                        metadata=health.json();break
                except httpx.ConnectError: pass
                await asyncio.sleep(.1)
            else: raise RuntimeError('Server readiness timeout')
            assert metadata['checkpoint_sha256']==plan['checkpoint_sha256']
            assert metadata['pair_batch_size']==8 and metadata['max_length']==128
            for row in rows[:16]:
                response=await client.post('/rerank',json=payload(row));response.raise_for_status()
            if baseline_path: baseline=baseline_path.open()
            semaphore=asyncio.Semaphore(CONCURRENCY)
            latencies=[]
            queues=[]
            processing=[]
            async def one(row):
                async with semaphore:
                    begin=time.perf_counter()
                    response=await client.post('/rerank',json=payload(row))
                    elapsed=time.perf_counter()-begin
                    response.raise_for_status()
                    data=response.json()
                    if data['request_id']!=row['query_id'] or data['backend']!=backend:
                        raise RuntimeError('HTTP response does not match request/backend')
                    return row,data,elapsed
            count=changed=no_relevant=0
            mrr_total=ndcg_total=max_error=0.
            temp=rankings_path.with_suffix('.jsonl.tmp')
            start=time.perf_counter()
            with temp.open('w') as output:
                for offset in range(0,len(rows),256):
                    responses=await asyncio.gather(*(one(row) for row in rows[offset:offset+256]))
                    for row,data,elapsed in responses:
                        label_map={c['document_id']:c['label'] for c in row['candidates']}
                        ranking=[{**r,'label':label_map[r['document_id']]} for r in data['ranking']]
                        if len(ranking)!=len(label_map) or len({r['document_id'] for r in ranking})!=len(label_map) or \
                            any(not math.isfinite(r['score']) for r in ranking) or \
                            ranking!=sorted(ranking,key=lambda r:(-r['score'],r['document_id'])):
                            raise RuntimeError('Invalid final-test ranking')
                        labels=[r['label'] for r in ranking]
                        no_relevant+=int(not any(labels))
                        mrr,ndcg=ranking_metrics(labels)
                        mrr_total+=mrr;ndcg_total+=ndcg;count+=1
                        if baseline:
                            reference=json.loads(next(baseline))
                            assert reference['query_id']==row['query_id']
                            changed+=int([r['document_id'] for r in ranking]!=[r['document_id'] for r in reference['ranking']])
                            expected_scores={r['document_id']:r['score'] for r in reference['ranking']}
                            max_error=max(max_error,max(abs(r['score']-expected_scores[r['document_id']]) for r in ranking))
                        latencies.append(elapsed*1000);queues.append(data['queue_ms']);processing.append(data['processing_ms'])
                        output.write(json.dumps({'query_id':row['query_id'],'ranking':ranking,'mrr_at_10':mrr,'ndcg_at_10':ndcg})+'\n')
                    if offset%4096==0 or count==len(rows):
                        progress={'phase':'full_test','configuration':name,'queries_scored':count,
                                  'queries_total':len(rows),'elapsed_seconds':time.perf_counter()-start}
                        write_json(HERE/'test_progress.json',progress)
                        print(f'{name}: {count:,}/{len(rows):,} queries',flush=True)
            elapsed=time.perf_counter()-start
            if baseline: assert next(baseline,None) is None
            assert count==plan['test_manifest']['evaluable_queries']
            temp.replace(rankings_path)
            report={'coordinator':coordinator,'backend':backend,'dataset_split':'test',
                'plan_sha256':sha256(HERE/'test_plan.json'),'checkpoint_sha256':plan['checkpoint_sha256'],
                'queries':count,'pairs':plan['test_manifest']['candidates'],
                'source_query_ids':plan['test_manifest']['source_query_ids'],
                'excluded_queries_with_no_clean_pairs':plan['test_manifest']['queries_with_no_clean_pairs'],
                'queries_without_relevant_candidate':no_relevant,
                'mrr_at_10':mrr_total/count,'ndcg_at_10':ndcg_total/count,
                'baseline':str(baseline_path.relative_to(ROOT)) if baseline_path else None,
                'changed_rankings_vs_python_pytorch':changed if baseline_path else None,
                'max_score_error_vs_python_pytorch':max_error if baseline_path else None,
                'pair_batch_limit':8,'actual_documents_per_request':plan['documents_per_query'],
                'http_concurrency':CONCURRENCY,'successful_requests':count,'failures':0,
                'full_quality_pass_seconds':elapsed,'full_quality_pass_queries_per_second':count/elapsed,
                'timing_scope':'one full pass includes HTTP, metric calculation and ranking JSONL writes; not a repeated speed benchmark',
                'median_request_ms':statistics.median(latencies),'p95_request_ms':sorted(latencies)[int(.95*len(latencies))],
                'median_queue_ms':statistics.median(queues),'median_processing_ms':statistics.median(processing),
                'rankings_file':str(rankings_path.relative_to(ROOT)),'rankings_sha256':sha256(rankings_path)}
            write_json(summary_path,report)
            print(f'{name}: MRR={report["mrr_at_10"]:.6f} nDCG={report["ndcg_at_10"]:.6f} changed={changed}',flush=True)
            return report
    finally:
        if baseline: baseline.close()
        proc.terminate()
        try: proc.wait(timeout=10)
        except subprocess.TimeoutExpired: proc.kill();proc.wait()
        if metadata:
            for _ in range(50):
                try: os.kill(metadata['worker_pid'],0)
                except ProcessLookupError: break
                await asyncio.sleep(.1)
            else:
                os.kill(metadata['worker_pid'],9)
                raise RuntimeError('Worker not reaped after test server shutdown')


async def main():
    check_metrics()
    expected=json.loads(CANDIDATE_MANIFEST.read_text())['splits']['test']
    if sha256(CANDIDATES)!=expected['sha256']: raise RuntimeError('Test candidates differ from manifest')
    rows=[json.loads(line) for line in CANDIDATES.open()]
    if len(rows)!=expected['evaluable_queries'] or sum(len(r['candidates']) for r in rows)!=expected['candidates']:
        raise RuntimeError('Test counts differ from manifest')
    if any(split_for_query(r['query'])!='test' for r in rows): raise RuntimeError('Non-test query in test evaluation')
    plan={'checkpoint':str(CHECKPOINT.relative_to(ROOT)),'checkpoint_sha256':sha256(CHECKPOINT),
          'dataset_split':'test','test_manifest':expected,'pair_batch_sizes':BATCHES,
          'documents_per_query':dict(sorted(Counter(len(r['candidates']) for r in rows).items())),
          'direct_timing_sample':'first 128 fixed test queries; same sample in every direct round',
          'direct_repeats_per_point':6,'http_full_quality_concurrency':CONCURRENCY,
          'model_selection':'epoch 3 selected from validation before viewing test results; no test-driven tuning',
          'rust_binary_sha256':sha256(BINARY),
          'code_sha256':{str(p.relative_to(ROOT)):sha256(p) for p in [Path(__file__),S1/'benchmark.py',
                         HERE/'worker.py',HERE/'python_server.py',*sorted((ROOT/'rust-server/src').glob('*.rs'))]}}
    if (HERE/'test_plan.json').exists():
        if json.loads((HERE/'test_plan.json').read_text())!=plan: raise RuntimeError('Fixed test plan changed')
    else: write_json(HERE/'test_plan.json',plan)
    quality=[]
    baseline_path=None
    for coordinator,backend in [('python','pytorch'),('rust','pytorch'),('python','mlx'),('rust','mlx')]:
        report=await evaluate(coordinator,backend,rows,plan,baseline_path)
        quality.append(report)
        if baseline_path is None: baseline_path=HERE/'test_python_pytorch_rankings.jsonl'
    direct=[]
    for round_index,order in enumerate((('pytorch','mlx'),('mlx','pytorch')),1):
        for backend in order:
            output=f'test_direct_round_{round_index}_{backend}.json'
            command=[sys.executable,'-u','-m','experiments.S1.benchmark','--backend',backend,'--split','test',
                     '--batch-sizes',*[str(x) for x in BATCHES],'--repeats','3','--output-name',output]
            if round_index==2: command.append('--reverse')
            print(f'TEST BATCH SWEEP round={round_index} backend={backend}',flush=True)
            subprocess.run(command,cwd=ROOT,check=True)
            result=json.loads((S1/output).read_text())
            assert result['candidate_sha256']==expected['sha256'] and result['checkpoint_sha256']==plan['checkpoint_sha256']
            direct.append(result)
    assert len({r['sample_sha256'] for r in direct})==1
    batches=[]
    for size in BATCHES:
        row={'pair_batch_size':size}
        for backend in ('pytorch','mlx'):
            reports=[r for report in direct if report['backend']==backend for r in report['results'] if r['pair_batch_size']==size]
            row[backend]={'pairs_per_second':summarize([r['pairs']/t for r in reports for t in r['scoring']['repeat_seconds']]),
                          'median_batch_ms':summarize([t for r in reports for t in r['scoring']['repeat_median_ms']])}
        batches.append(row)
    result={'plan':plan,'full_test_quality':quality,'direct_batch_sweep':batches,
            'direct_sample_sha256':direct[0]['sample_sha256'],
            'limitations':['Constructed test lists, not a realistic retrieval benchmark or leaderboard.',
                           'HTTP quality uses all queries; batch timing uses a fixed 128-query test sample.',
                           'Direct batches combine pairs across queries; HTTP does not batch across requests.',
                           'HTTP full-pass timing is one observation per configuration; no inference of a consistent coordinator speedup.',
                           'No model or batch grid selected using these test results.']}
    write_json(HERE/'test_comparison.json',result)
    write_json(HERE/'test_progress.json',{'phase':'complete','test_comparison':str((HERE/'test_comparison.json').relative_to(ROOT))})
    print('Final test and batch comparisons saved.',flush=True)


if __name__=='__main__': asyncio.run(main())
