"""All 485 queries, all K; incremental per-path artifacts survive interruption."""
import argparse
import gzip
import json
import time
import traceback
from .common import HERE,rows,save,sha,ranked,environment,verify_plan
from .inference import Inference
from .statistics import accuracy_gate
from experiments.E15.metrics import metrics


def read(path):
    with gzip.open(path,'rt') as f:return [json.loads(line) for line in f]


def main():
    p=argparse.ArgumentParser();p.add_argument('--device',required=True);p.add_argument('--path',required=True);a=p.parse_args()
    plan=verify_plan();out=HERE/'scores'/f'{a.device}_{a.path}.jsonl.gz';out.parent.mkdir(exist_ok=True)
    summary=HERE/f'{a.device}_{a.path}_accuracy.json'
    if summary.exists() and json.loads(summary.read_text()).get('status') in ('VERIFIED','FAILED_ACCURACY'):
        assert json.loads(summary.read_text())['artifact_sha256']==sha(out);print('Already audited',a.path);return
    started=time.perf_counter();report={'device':a.device,'path':a.path,'plan_sha256':sha(HERE/'plan.json'),'environment':environment()}
    data=rows();expected=[(r,k) for k in plan['accuracy']['document_counts'] for r in data]
    existing=read(out) if out.exists() else []
    assert len(existing)<=len(expected)
    for old,(r,k) in zip(existing,expected):assert (old['dataset'],old['query_id'],old['k'])==(r['dataset'],r['query_id'],k)
    try:
        model=Inference(a.device,a.path)
        report['dispatch']=model.dispatch([(data[0]['query'],t) for t in data[0]['texts'][:16]])
        with gzip.open(out,'at') as f:
            for index,(r,k) in enumerate(expected[len(existing):],len(existing)):
                scores=model.request(r,k,16);ids=r['document_ids'][:k];ranking=ranked(ids,scores)
                record={'dataset':r['dataset'],'query_id':r['query_id'],'k':k,'ids':ids,'scores':scores,
                        'ranking':ranking,'metrics':metrics(ranking,r['qrels']),'qrels':r['qrels']}
                f.write(json.dumps(record,allow_nan=False)+'\n');f.flush()
                if index%50==0:print(a.device,a.path,index+1,len(expected),flush=True)
        records=read(out);assert len(records)==1455 and sum(r['k'] for r in records)==77600
        reference=records if a.path=='stock' else read(HERE/'scores'/f'{a.device}_stock.jsonl.gz')
        report.update(accuracy_gate(reference,records));report.update(status='VERIFIED' if report['application_pass'] else 'FAILED_ACCURACY',
            records=len(records),pair_occurrences=sum(r['k'] for r in records),artifact=str(out.relative_to(HERE)),artifact_sha256=sha(out),memory=model.memory())
    except Exception as error:
        report.update(status='FAILED_EXECUTION',error=repr(error),traceback=traceback.format_exc())
    report['wall_seconds']=time.perf_counter()-started;save(summary,report);print(report['status'],flush=True)


if __name__=='__main__':main()
