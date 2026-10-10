"""CUDA-only summaries; never import Apple results into an NVIDIA comparison."""
import gzip
import json
import math
from .common import HERE, rows, key, verify_plan, sha, save, describe
from .summarize import paired
from .accuracy import summarize
from .verify import independent_metrics
from experiments.E15.metrics import assert_trec_parity


def main():
    plan=verify_plan();audit=json.loads((HERE/'cuda_accuracy.json').read_text())
    assert sha(HERE/audit['audit_file'])==audit['audit_sha256']
    expected={(r['dataset'],r['query_id']):r for r in rows()};seen=set();records=[];trec={}
    with gzip.open(HERE/audit['audit_file'],'rt') as f:
        for r in map(json.loads,f):
            identity=key(r,r['k']);assert identity not in seen;seen.add(identity)
            source=expected[(r['dataset'],r['query_id'])]
            assert r['candidate_ids']==source['document_ids'][:r['k']] and r['qrels']==source['qrels']
            for path,scores in r['scores'].items():
                assert len(scores)==r['k'] and all(math.isfinite(s) for s in scores)
                ids=[d for d,s in sorted(zip(r['candidate_ids'],scores),key=lambda v:(-v[1],v[0]))]
                assert ids==r['rankings'][path]
                assert all(abs(v-r['metrics'][path][k])<1e-12 for k,v in independent_metrics(ids,r['qrels']).items())
                group=trec.setdefault((path,r['dataset'],r['k']),{'rels':{},'runs':{}})
                group['rels'][r['query_id']]=r['qrels'];group['runs'][r['query_id']]=ids
            records.append(r)
    assert seen=={key(r,k) for r in expected.values() for k in plan['document_counts']}
    for group in trec.values():assert_trec_parity(group['rels'],group['runs'])
    assert summarize(records,plan)==audit['accuracy']
    runs=[]
    for session in range(4):
        data=json.loads((HERE/f'cuda_direct_s{session}.json').read_text());assert data['status']=='complete'
        for r in data['runs']:
            assert sum(r['batch_pairs'])==r['pairs']==16*r['documents']
            assert math.isclose(sum(r['batch_ms'])/1000,r['seconds'],rel_tol=1e-12)
            runs.append({**r,'session':session})
    output={'device':'NVIDIA only','plan_sha256':sha(HERE/'plan.json'),'accuracy':audit['accuracy'],
            'comparisons':[],'verification':'All input IDs, scores, rankings, quality metrics and raw timing totals independently checked; TREC parity verified.'}
    paths=list(audit['accuracy'])
    for scope in ('forward','direct'):
        for k in plan['document_counts']:
            groups={p:[r for r in runs if r['scope']==scope and r['documents']==k and r['path']==p] for p in paths}
            assert all(len(v)==12 for v in groups.values())
            output['comparisons'].append({'scope':scope,'documents':k,
                'paths':{p:{'accuracy_qualified':audit['accuracy'][p]['accuracy_qualified'],
                            'pairs_per_second':describe([r['pairs_per_second'] for r in v])} for p,v in groups.items()},
                'against_stock':{p:{**paired(groups['stock'],groups[p],'pairs_per_second'),
                                    'diagnostic_only':not audit['accuracy'][p]['accuracy_qualified'] or not audit['accuracy']['stock']['accuracy_qualified']} for p in paths if p!='stock'}})
    save(HERE/'cuda_summary.json',output)
    save(HERE/'cuda_checksums.json',{p.name:sha(p) for p in sorted(HERE.glob('cuda_*')) if p.name!='cuda_checksums.json'})


if __name__=='__main__':main()
