"""Independent saved-record audit; no model execution or hardware substitution."""
import gzip
import json
import math
import numpy as np
from .common import HERE, ROOT, rows, key, sha, save, verify_plan
from experiments.E15.metrics import assert_trec_parity


def independent_metrics(ids,qrels):
    gains=[qrels.get(d,0) for d in ids[:10]]
    ideal=sorted((v for v in qrels.values() if v>0),reverse=True)[:10]
    dcg=sum(v/math.log2(i+2) for i,v in enumerate(gains))
    idcg=sum(v/math.log2(i+2) for i,v in enumerate(ideal))
    first=next((i+1 for i,v in enumerate(gains) if v>0),None)
    positive=sum(v>0 for v in qrels.values())
    return {'ndcg_at_10':dcg/idcg,'mrr_at_10':1/first if first else 0.,
            'recall_at_10':sum(v>0 for v in gains)/positive,
            'recall_at_100':sum(qrels.get(d,0)>0 for d in ids[:100])/positive}


def main():
    plan=verify_plan();source={(r['dataset'],r['query_id']):r for r in rows()}
    for p,h in plan['dataset_source_sha256'].items():assert sha(ROOT/p)==h
    from .accuracy import summarize
    report=json.loads((HERE/'mps_accuracy.json').read_text());records=[];seen=set();trec={}
    assert sha(HERE/report['audit_file'])==report['audit_sha256']
    with gzip.open(HERE/report['audit_file'],'rt') as f:
        for r in map(json.loads,f):
            expected=source[(r['dataset'],r['query_id'])];identity=key(r,r['k'])
            assert identity not in seen;seen.add(identity)
            assert r['candidate_ids']==expected['document_ids'][:r['k']] and r['qrels']==expected['qrels'] and r['query']==expected['query']
            for path,scores in r['scores'].items():
                assert len(scores)==r['k'] and np.isfinite(scores).all()
                ids=[d for d,s in sorted(zip(r['candidate_ids'],scores),key=lambda pair:(-pair[1],pair[0]))]
                assert ids==r['rankings'][path]
                m=independent_metrics(ids,r['qrels'])
                assert all(abs(v-r['metrics'][path][k])<1e-12 for k,v in m.items())
                group=trec.setdefault((path,r['dataset'],r['k']),{'rels':{},'runs':{}})
                group['rels'][r['query_id']]=r['qrels'];group['runs'][r['query_id']]=ids
            records.append(r)
    assert seen=={key(r,k) for r in source.values() for k in plan['document_counts']}
    for group in trec.values():assert_trec_parity(group['rels'],group['runs'])
    assert summarize(records,plan)==report['accuracy']
    refs={key(r,r['k']):r for r in records};requests=pairs=0;files=0
    from .http_benchmark import validate_response
    for file in HERE.glob('http_s[0-3]_*.json'):
        data=json.loads(file.read_text());assert data['status']=='complete';files+=1
        assert len(data['runs'])==27
        runkeys=set()
        for r in data['runs']:
            rk=(r['repeat'],r['documents'],r['concurrency']);assert rk not in runkeys;runkeys.add(rk)
            obs=r['observations'];assert len(obs)==16
            assert [o['request_id'] for o in obs]==[f'{d}:{q}:{r["documents"]}' for d,q in plan['http_subset']]
            good=[o for o in obs if o['success']]
            assert len(good)==r['successful_requests'] and len(obs)-len(good)==r['failures']
            for o in good:
                reference=refs[o['request_id']];path='stock' if data['backend']=='pytorch' else 'mlx'
                body={**o,'request_id':o['request_id']}
                validate_response(body,{'request_id':o['request_id']},reference,path,plan['accuracy_tolerance'])
                assert math.isclose(o['latency_ms'],(o['end_offset_seconds']-o['start_offset_seconds'])*1000,abs_tol=1e-7)
            for p in (50,95,99):assert math.isclose(r[f'p{p}_ms'],np.percentile([o['latency_ms'] for o in good],p),abs_tol=1e-9)
            assert math.isclose(r['requests_per_second'],len(good)/r['wall_seconds'],rel_tol=1e-12)
            assert r['successful_pairs']==len(good)*r['documents']
            assert math.isclose(r['pairs_per_second'],r['requests_per_second']*r['documents'],rel_tol=1e-12)
            requests+=len(good);pairs+=r['successful_pairs']
    assert files==16 and requests==6912 and pairs==368640
    direct_files=list(HERE.glob('mps_direct_s*.json'));assert len(direct_files)==4
    expected_runs=18*len(report['accuracy'])
    for file in direct_files:
        data=json.loads(file.read_text());assert len(data['runs'])==expected_runs and data['status']=='complete'
        assert len({(r['path'],r['scope'],r['documents'],r['repeat']) for r in data['runs']})==expected_runs
        for r in data['runs']:
            assert sum(r['batch_pairs'])==r['pairs']==16*r['documents']
            assert math.isclose(sum(r['batch_ms'])/1000,r['seconds'],rel_tol=1e-12)
            assert math.isclose(r['pairs']/r['seconds'],r['pairs_per_second'],rel_tol=1e-12)
    summary=json.loads((HERE/'summary.json').read_text())
    assert summary['total_http_requests']==requests and summary['total_http_pairs']==pairs
    # Independently reconstruct matched throughput ratios from each raw session.
    for row in summary['http']:
        cluster=[]
        for session in range(4):
            a=json.loads((HERE/f'http_s{session}_python_{row["backend"]}.json').read_text())
            b=json.loads((HERE/f'http_s{session}_rust_{row["backend"]}.json').read_text())
            filtered=lambda data:{r['repeat']:r['requests_per_second'] for r in data['runs'] if r['documents']==row['documents'] and r['concurrency']==row['concurrency']}
            x,y=filtered(a),filtered(b);assert set(x)==set(y)=={0,1,2}
            cluster.append(float(np.mean([math.log(y[i]/x[i]) for i in range(3)])))
        target=row['paired_rust_vs_python']['requests_per_second']
        assert math.isclose(math.exp(np.mean(cluster)),target['ratio_geomean'],rel_tol=1e-12)
        rng=np.random.default_rng(1701)
        ci=np.exp(np.quantile(np.mean(rng.choice(cluster,(10000,4),replace=True),axis=1),[.025,.975]))
        assert np.allclose(ci,target['ratio_ci95'],atol=1e-12,rtol=0)
    results=[*HERE.glob('*.json'),*HERE.glob('*scores.jsonl.gz'),*HERE.glob('*.png'),HERE/'REPORT.md']
    results=[p for p in results if p.name not in ('checksums.json','verification.json')]
    checksums={p.name:sha(p) for p in sorted(results)}
    save(HERE/'checksums.json',checksums)
    save(HERE/'verification.json',{'status':'verified','plan_sha256':sha(HERE/'plan.json'),
        'checksums_sha256':sha(HERE/'checksums.json'),'accuracy_records':len(records),
        'accuracy_pairs_per_path':sum(r['k'] for r in records),'trec_metric_groups_verified':len(trec),
        'successful_http_requests':requests,'successful_http_pairs':pairs,
        'raw_http_files':files,'nvidia':'pending authentication; no substitute measurements'})
    print('Verified',len(records),'accuracy records and',requests,'HTTP requests',flush=True)


if __name__=='__main__':main()
