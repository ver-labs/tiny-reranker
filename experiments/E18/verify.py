"""Independent saved-evidence audit; strict completeness remains visible."""
import gzip
import json
import math
from collections import Counter
import numpy as np
from .common import HERE,ROOT,rows,sha,save,verify_plan
from experiments.E15.metrics import assert_trec_parity


def metric(ids,qrels):
    gains=[qrels.get(d,0) for d in ids[:10]]
    ideal=sorted((v for v in qrels.values() if v>0),reverse=True)[:10]
    ndcg=sum(g/math.log2(i+2) for i,g in enumerate(gains))/sum(g/math.log2(i+2) for i,g in enumerate(ideal))
    first=next((i+1 for i,g in enumerate(gains) if g>0),None)
    return ndcg,1/first if first else 0.


def main():
    plan=verify_plan();lookup={(r['dataset'],r['query_id']):r for r in rows()};result={'plan_sha256':sha(HERE/'plan.json'),'scores':{},'http':{},'direct':{},'limitations':[]}
    for p in HERE.glob('*_accuracy.json'):
        report=json.loads(p.read_text())
        if report['status']=='FAILED_EXECUTION':result['scores'][p.stem]={'status':'FAILED_EXECUTION','error':report['error']};continue
        artifact=HERE/report['artifact'];assert sha(artifact)==report['artifact_sha256']
        with gzip.open(artifact,'rt') as f:records=[json.loads(line) for line in f]
        assert len(records)==1455 and sum(r['k'] for r in records)==77600
        identities=[]
        for r in records:
            data=lookup[r['dataset'],r['query_id']];assert r['ids']==data['document_ids'][:r['k']] and r['qrels']==data['qrels']
            assert len(r['scores'])==r['k'] and np.isfinite(r['scores']).all()
            order=sorted(r['ids'],key=lambda d:(-r['scores'][r['ids'].index(d)],d));assert order==r['ranking']
            ndcg,mrr=metric(order,r['qrels']);assert abs(ndcg-r['metrics']['ndcg_at_10'])<1e-12 and abs(mrr-r['metrics']['mrr_at_10'])<1e-12
            identities.append((r['dataset'],r['query_id'],r['k']))
        assert len(set(identities))==1455
        for dataset in ('scifact','nfcorpus'):
            for k in (10,50,100):
                subset=[r for r in records if r['dataset']==dataset and r['k']==k]
                assert_trec_parity({r['query_id']:r['qrels'] for r in subset},{r['query_id']:r['ranking'] for r in subset})
        device=report['device']
        with gzip.open(HERE/'scores'/f'{device}_stock.jsonl.gz','rt') as f:
            reference={(v['dataset'],v['query_id'],v['k']):v for v in map(json.loads,f)}
        absolute=[];relative=[];strict=True;cell_records={}
        for r in records:
            a=reference[r['dataset'],r['query_id'],r['k']]
            x=np.array(a['scores']);y=np.array(r['scores']);error=np.abs(x-y)
            absolute.extend(error);relative.extend(error/np.maximum(np.abs(x),1e-3));strict &= bool(np.all(error<=2e-5+1e-4*np.abs(x)))
            cell_records.setdefault(f"{r['dataset']}/K{r['k']}",[]).append((a,r))
        for field,values in [('absolute_error',absolute),('relative_error_floor_1e-3',relative)]:
            quantiles=np.quantile(values,[0,.25,.5,.75,.95,.99,1])
            assert all(abs(report[field][name]-v)<1e-12 for name,v in zip(('min','p25','median','p75','p95','p99','max'),quantiles))
        assert bool(strict)==report['strict_numerical_pass']
        application=max(absolute)<=.01
        for key,items in cell_records.items():
            top1=sum(a['ranking'][0]!=r['ranking'][0] for a,r in items)
            membership=sum(set(a['ranking'][:10])!=set(r['ranking'][:10]) for a,r in items)
            assert report['cells'][key]['top1_changed']==top1 and report['cells'][key]['top10_membership_changed']==membership
            passed=top1/len(items)<=.01 and membership/len(items)<=.01
            for name,field in [('ndcg_at_10','ndcg_delta'),('mrr_at_10','mrr_delta')]:
                delta=np.array([r['metrics'][name]-a['metrics'][name] for a,r in items])
                generator=np.random.default_rng(1801);means=delta[generator.integers(0,len(delta),(10000,len(delta)))].mean(axis=1);ci=np.quantile(means,[.025,.975])
                assert np.allclose(ci,report['cells'][key][field]['ci95'],atol=1e-12,rtol=0)
                passed &= ci[0]>=-.001
            assert bool(passed)==report['cells'][key]['application_pass'];application &= passed
        assert bool(application)==report['application_pass']
        result['scores'][p.stem]={'status':'VERIFIED','records':1455,'pairs':77600,'metric_oracle':'Independent direct formulas and pytrec_eval nDCG/recall'}
    for p in (HERE/'raw').glob('*http_s*.json'):
        report=json.loads(p.read_text())
        if report.get('status')!='COMPLETE':result['http'][p.stem]={'status':'INCOMPLETE'};continue
        artifact=HERE/report['response_artifact'];assert sha(artifact)==report['response_sha256']
        with gzip.open(artifact,'rt') as f:obs=[json.loads(line) for line in f]
        assert len(obs)==(96 if report.get('primary_only') else 288)
        audit=HERE/'scores'/f"mps_{report['path']}.jsonl.gz"
        with gzip.open(audit,'rt') as f:refs={(r['dataset'],r['query_id'],r['k']):r for r in map(json.loads,f)}
        for run in report['runs']:
            selected=[r for r in obs if (r['repetition'],r['k'],r['concurrency'])==(run['repetition'],run['k'],run['concurrency'])]
            assert len(selected)==16 and {r['index'] for r in selected}==set(range(16)) and len({r['request_id'] for r in selected})==16
            good=[r for r in selected if r['success']];assert len(good)==run['successful_requests']
            assert run['successful_pairs']==len(good)*run['k']
            assert abs(run['requests_per_second']-len(good)/run['seconds'])<1e-12
            for r in good:
                dataset,qid,kk,index=r['request_id'].split(':');ref=refs[dataset,qid,int(kk[1:])]
                ids=[d['document_id'] for d in r['ranking']];assert ids==ref['ranking'] and len(ids)==run['k']
                expected=dict(zip(ref['ids'],ref['scores']))
                assert all(math.isfinite(d['score']) and abs(d['score']-expected[d['document_id']])<=2e-5+1e-4*abs(expected[d['document_id']]) for d in r['ranking'])
            for percentile in (50,95,99):assert abs(run['latency_ms'][f'p{percentile}']-np.percentile([r['latency_ms'] for r in good],percentile))<1e-8
        result['http'][p.stem]={'status':'VERIFIED','requests':len(obs),'successful_pairs':sum(r['successful_pairs'] for r in report['runs'])}
    for p in (HERE/'raw').glob('*_direct_s*.json'):
        r=json.loads(p.read_text())
        if r.get('status')!='COMPLETE':result['direct'][p.stem]={'status':r.get('status','INCOMPLETE')};continue
        assert len(r['observations'])==60
        assert len({(v['k'],v['batch'],v['length'],v['scope'],v['repetition'],v['group']) for v in r['observations']})==60
        for v in r['observations']:
            assert v['pairs']==v['queries']*v['k'] and v['seconds']>0 and abs(v['pairs_per_second']-v['pairs']/v['seconds'])<1e-8
        result['direct'][p.stem]={'status':'VERIFIED','observations':60}
    result['status']='VERIFIED_SAVED_ARTIFACTS'
    result['completeness']={'expected_primary_http_sessions':32,'complete_http_sessions':sum(v['status']=='VERIFIED' for k,v in result['http'].items() if k.startswith('http_s')),'expected_mps_direct_sessions':36,'complete_mps_direct_sessions':sum(v['status']=='VERIFIED' for k,v in result['direct'].items() if k.startswith('mps')),'expected_cuda_direct_sessions':25,'complete_cuda_direct_sessions':sum(v['status']=='VERIFIED' for k,v in result['direct'].items() if k.startswith('cuda'))}
    save(HERE/'verification.json',result)
    files=[p for p in HERE.rglob('*') if p.is_file() and '__pycache__' not in str(p) and not any(part in p.parts for part in ('cache','target')) and p.name not in ('checksums.json',) and not p.name.endswith('_server.log') and p.suffix not in ('.so',)]
    save(HERE/'checksums.json',{'files':{str(p.relative_to(HERE)):sha(p) for p in sorted(files)}});print(result['completeness'],flush=True)

if __name__=='__main__':main()
