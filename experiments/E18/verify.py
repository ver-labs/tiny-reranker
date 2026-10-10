"""Independent saved-evidence audit; strict completeness remains visible."""
import gzip
import ast
import json
import math
from collections import Counter
import numpy as np
from .common import HERE,ROOT,rows,sha,save,verify_plan,fixtures
from experiments.E15.metrics import assert_trec_parity


def metric(ids,qrels):
    gains=[qrels.get(d,0) for d in ids[:10]]
    ideal=sorted((v for v in qrels.values() if v>0),reverse=True)[:10]
    ndcg=sum(g/math.log2(i+2) for i,g in enumerate(gains))/sum(g/math.log2(i+2) for i,g in enumerate(ideal))
    first=next((i+1 for i,g in enumerate(gains) if g>0),None)
    return ndcg,1/first if first else 0.


def verify_records(artifact, digest, expected, lookup):
    """Confirmation and changed-context files must preserve every candidate."""
    assert sha(artifact)==digest
    with gzip.open(artifact,'rt') as f: records=[json.loads(line) for line in f]
    identities=[(r['dataset'],r['query_id'],r['k']) for r in records]
    assert len(identities)==len(set(identities))==len(expected)
    assert set(identities)==set(expected)
    for r in records:
        data=lookup[r['dataset'],r['query_id']]
        assert r['ids']==data['document_ids'][:r['k']] and r['qrels']==data['qrels']
        assert len(r['scores'])==r['k'] and np.isfinite(r['scores']).all()
        scores=dict(zip(r['ids'],r['scores']))
        order=sorted(r['ids'],key=lambda d:(-scores[d],d))
        assert order==r['ranking']
        ndcg,mrr=metric(order,r['qrels'])
        assert abs(ndcg-r['metrics']['ndcg_at_10'])<1e-12
        assert abs(mrr-r['metrics']['mrr_at_10'])<1e-12
    return records


def verify_ratio(observations,baseline,candidate,reported):
    paired={}
    for r in observations:
        if r['path'] in (baseline,candidate):
            key=r['session'],r['repetition'];cell=paired.setdefault(key,{})
            assert r['path'] not in cell;cell[r['path']]=r['seconds']
    assert paired and all(set(v)=={baseline,candidate} for v in paired.values())
    sessions={}
    for (session,rep),v in paired.items():sessions.setdefault(session,[]).append(math.log(v[baseline]/v[candidate]))
    units=np.array([np.mean(sessions[s]) for s in sorted(sessions)])
    rng=np.random.default_rng(1801)
    ci=np.exp(np.quantile(units[rng.integers(0,len(units),(10000,len(units)))].mean(axis=1),[.025,.975]))
    assert reported['units']==len(units) and reported['paired_repetitions']==len(paired)
    assert abs(reported['ratio']-math.exp(units.mean()))<1e-12
    assert np.allclose(reported['ci95'],ci,atol=1e-12,rtol=0)
    assert reported['practical_primary_gain']==bool(ci[0]>1.05)


def main():
    plan=verify_plan();lookup={(r['dataset'],r['query_id']):r for r in rows()};result={'plan_sha256':sha(HERE/'plan.json'),'scores':{},'http':{},'direct':{},'limitations':[]}
    manifest=json.loads((HERE/'data_manifest.json').read_text())
    for relative,digest in manifest['source_sha256'].items():assert sha(ROOT/relative)==digest
    result['source_inputs']='VERIFIED candidate file, assets, frozen manifests and source corpora/qrels'
    for p in list(HERE.glob('*_accuracy.json'))+list(HERE.glob('*_reserve.json')):
        report=json.loads(p.read_text());target=report.get('accuracy',report)
        if 'per_query_artifact' in target:
            artifact=HERE/target['per_query_artifact'];assert sha(artifact)==target['per_query_artifact_sha256']
            with gzip.open(artifact,'rt') as f:diagnostics=json.load(f)
            assert sum(len(v) for v in diagnostics.values())==(421 if p.name.endswith('_reserve.json') else 1455)
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
            assert run['failures']==len(selected)-len(good) and run['timeouts']==sum(v.get('timeout',False) for v in selected)
            assert abs(run['pairs_per_second']-run['successful_pairs']/run['seconds'])<1e-10
            for scope in ('latency','queue','processing'):
                for percentile in (50,95,99):
                    assert all(math.isfinite(v[scope+'_ms']) and v[scope+'_ms']>=0 for v in good)
                    assert abs(run[scope+'_ms'][f'p{percentile}']-np.percentile([v[scope+'_ms'] for v in good],percentile))<1e-8
        result['http'][p.stem]={'status':'VERIFIED','requests':len(obs),'successful_pairs':sum(r['successful_pairs'] for r in report['runs'])}
    for p in (HERE/'raw').glob('*_direct_s*.json'):
        r=json.loads(p.read_text())
        if r.get('status')!='COMPLETE':result['direct'][p.stem]={'status':r.get('status','INCOMPLETE')};continue
        assert len(r['observations'])==60
        assert len({(v['k'],v['batch'],v['length'],v['scope'],v['repetition'],v['group']) for v in r['observations']})==60
        expected_cells={(*cell,'primary' if cell==[100,16,512] else 'sweep') for cell in plan['direct']['cells']}|{(100,16,512,'sweep')}
        assert {(v['k'],v['batch'],v['length'],v['group']) for v in r['observations']}==expected_cells
        if r['device']=='mps':
            assert r['apple_forward_addendum_sha256']==sha(HERE/'apple_forward_addendum.json')
            if r['path'] in ('mlx','mlx_compiled','metal','metal_compiled'):
                assert len(r['async_checks'])==10 and {(v['k'],v['batch'],v['length'],v['group']) for v in r['async_checks']}==expected_cells
                assert all(v['max_abs_error']==0 for v in r['async_checks'])
        for v in r['observations']:
            assert v['queries']==(16 if v['group']=='primary' else 4) and v['session']==r['session'] and v['path']==r['path']
            assert v['pairs']==v['queries']*v['k'] and v['seconds']>0 and abs(v['pairs_per_second']-v['pairs']/v['seconds'])<1e-8
        result['direct'][p.stem]={'status':'VERIFIED','observations':60}
    result['confirmation']={};result['context']={};result['operations']={}
    expected_reserve={(r['dataset'],r['query_id'],100) for r in fixtures(True)}
    for p in HERE.glob('*_reserve.json'):
        r=json.loads(p.read_text())
        records=verify_records(HERE/r['artifact'],r['artifact_sha256'],expected_reserve,lookup)
        assert r['queries']==len(records)==421 and r['pairs']==42100 and r['seconds']>0
        result['confirmation'][p.stem]={'status':'VERIFIED','queries':421,'pairs':42100}
    expected_context={(dataset,qid,100) for dataset,qid in plan['timing_ids']}
    for p in HERE.glob('*_context_check.json'):
        report=json.loads(p.read_text())
        with gzip.open(HERE/'scores'/f"{report['device']}_stock.jsonl.gz",'rt') as f:
            baseline={(r['dataset'],r['query_id'],r['k']):r for r in map(json.loads,f)}
        for cell,r in report['cells'].items():
            records=verify_records(HERE/r['artifact'],r['artifact_sha256'],expected_context,lookup)
            for field,entry in r['quality_changes_vs_stock_L512'].items():
                delta=np.array([v['metrics'][field]-baseline[v['dataset'],v['query_id'],100]['metrics'][field] for v in records])
                rng=np.random.default_rng(1801)
                ci=np.quantile(delta[rng.integers(0,len(delta),(10000,len(delta)))].mean(axis=1),[.025,.975])
                assert entry['units']==16 and abs(entry['mean']-delta.mean())<1e-12 and np.allclose(entry['ci95'],ci,rtol=0,atol=1e-12)
            result['context'][p.stem+'/'+cell]={'status':'VERIFIED','queries':16,'pairs':1600}
    for p in (HERE/'raw').glob('*_operation_s*.json'):
        r=json.loads(p.read_text());paths={'native','metal'} if r['device']=='mps' else {'native','cuda','triton'}
        expected={(path,n,rep) for path in paths for n in (2048,4096,8192) for rep in range(3)}
        obs=r['observations'];assert len(obs)==len(expected)
        assert {(v['path'],v['rows'],v['repetition']) for v in obs}==expected
        assert all(v['session']==r['session'] and v['hidden']==384 and math.isfinite(v['seconds']) and v['seconds']>0 for v in obs)
        result['operations'][p.stem]={'status':'VERIFIED','observations':len(obs)}
    result['timing_intervals']={}
    if (HERE/'summary.json').exists():
        summary=json.loads((HERE/'summary.json').read_text())
        for key,cell in summary['direct'].items():
            device,shape=key.split('/',1);scope,k,b,length,group=ast.literal_eval(shape)
            observations=[]
            for p in (HERE/'raw').glob(f'{device}_direct_s*.json'):
                raw=json.loads(p.read_text())
                if raw.get('status')=='COMPLETE':observations.extend(v for v in raw['observations'] if (v['scope'],v['k'],v['batch'],v['length'],v['group'])==(scope,k,b,length,group))
            for comparison,r in cell['comparisons'].items():
                if 'ratio' not in r:continue
                baseline,candidate=comparison.split('/');verify_ratio(observations,baseline,candidate,r)
                result['timing_intervals'][key+'/'+comparison]='VERIFIED'
        for key,cell in summary['operation'].items():
            device,n=key.split('/');observations=[]
            for p in (HERE/'raw').glob(f'{device}_operation_s*.json'):
                observations.extend(v for v in json.loads(p.read_text())['observations'] if v['rows']==int(n))
            for comparison,r in cell['comparisons'].items():
                baseline,candidate=comparison.split('/');verify_ratio(observations,baseline,candidate,r)
                result['timing_intervals'][key+'/'+comparison]='VERIFIED'
        http_observations=[]
        for p in (HERE/'raw').glob('*http_s*.json'):
            raw=json.loads(p.read_text())
            if raw.get('status')=='COMPLETE':http_observations.extend({**v,'study':raw.get('study','primary')} for v in raw['runs'])
        for family in ('http','optimized_http'):
            for key,cell in summary[family].items():
                if family=='http':study='primary';kk,cc=key.split('/')
                else:study,kk,cc=key.split('/')
                observations=[v for v in http_observations if (v['study'],v['k'],v['concurrency'])==(study,int(kk[1:]),int(cc[1:]))]
                for comparison,r in cell['comparisons'].items():
                    if 'ratio' not in r:continue
                    if family=='http':
                        parts=comparison.split('/');baseline='/'.join(parts[:2]);candidate='/'.join(parts[2:])
                    else:baseline='python/'+study.removeprefix('optimized_');candidate=comparison
                    verify_ratio(observations,baseline,candidate,r)
                    result['timing_intervals'][family+'/'+key+'/'+comparison]='VERIFIED'
    result['status']='VERIFIED_SAVED_ARTIFACTS'
    result['completeness']={'expected_primary_http_sessions':32,'complete_http_sessions':sum(v['status']=='VERIFIED' for k,v in result['http'].items() if k.startswith('http_s')),'expected_mps_direct_sessions':36,'complete_mps_direct_sessions':sum(v['status']=='VERIFIED' for k,v in result['direct'].items() if k.startswith('mps')),'expected_cuda_direct_sessions':25,'complete_cuda_direct_sessions':sum(v['status']=='VERIFIED' for k,v in result['direct'].items() if k.startswith('cuda'))}
    save(HERE/'verification.json',result)
    files=[p for p in HERE.rglob('*') if p.is_file() and '__pycache__' not in str(p) and not any(part in p.parts for part in ('cache','target')) and p.name not in ('checksums.json',) and not p.name.endswith('_server.log') and p.suffix not in ('.so',)]
    save(HERE/'checksums.json',{'files':{str(p.relative_to(HERE)):sha(p) for p in sorted(files)}});print(result['completeness'],flush=True)

if __name__=='__main__':main()
