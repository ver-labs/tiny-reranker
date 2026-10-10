"""Complete development accuracy audits at each request size; no retraining."""
import argparse
import gzip
import json
import time
import numpy as np
from .common import HERE, rows, key, sha, save, verify_plan, ranked, errors, interval, acceptance, environment
from .inference import Inference
from experiments.E15.metrics import metrics


def summarize(records, plan):
    paths = list(records[0]['scores'])
    output = {}
    for path in paths:
        cells = []
        for dataset in ('scifact','nfcorpus'):
            for k in plan['document_counts']:
                items = [r for r in records if r['dataset']==dataset and r['k']==k]
                error = errors([s for r in items for s in r['scores']['strict']],
                               [s for r in items for s in r['scores'][path]])
                changes = {name: sum(f(r['rankings']['strict'],r['rankings'][path]) for r in items)
                           for name,f in {'full_ranking':lambda a,b:a!=b,'top1':lambda a,b:a[0]!=b[0],
                                          'top10_membership':lambda a,b:set(a[:10])!=set(b[:10]),
                                          'top10_order':lambda a,b:a[:10]!=b[:10]}.items()}
                changes.update(top1_fraction=changes['top1']/len(items),
                               top10_membership_fraction=changes['top10_membership']/len(items))
                deltas = {m:interval([r['metrics'][path][m]-r['metrics']['strict'][m] for r in items])
                          for m in ('ndcg_at_10','mrr_at_10')}
                cells.append({'dataset':dataset,'documents':k,'queries':len(items),'error':error,
                              'changes':changes,'metric_delta':deltas,
                              'metrics':{m:float(np.mean([r['metrics'][path][m] for r in items])) for m in ('ndcg_at_10','mrr_at_10')},
                              'acceptance':acceptance(error,changes,deltas,plan['accuracy_tolerance'])})
        output[path] = {'cells':cells,'accuracy_qualified':all(c['acceptance']['accuracy_qualified'] for c in cells),
                        'strict_numerical_equivalence':all(c['acceptance']['strict_numerical_equivalence'] for c in cells)}
    return output


def main():
    p=argparse.ArgumentParser();p.add_argument('--device',choices=['mps','cuda'],required=True)
    args=p.parse_args();plan=verify_plan();data=rows()
    paths = ('strict','stock','mlx') if args.device=='mps' else ('strict','stock','cuda','triton')
    probe=HERE/f'{args.device}_compile.json'
    if probe.exists() and json.loads(probe.read_text())['status']=='compatible':
        paths=(*paths,'compiled')
    records = [{'dataset':r['dataset'],'query_id':r['query_id'],'query':r['query'],'k':k,
                'candidate_ids':r['document_ids'][:k],'qrels':r['qrels'],'scores':{},'rankings':{},'metrics':{}}
               for r in data for k in plan['document_counts']]
    report = {'plan_sha256':sha(HERE/'plan.json'),'environment':environment(),'device':args.device,
              'paths':{},'queries':len(data),'records_per_path':len(records),'pairs_per_path':77600}
    for path in paths:
        model=Inference(args.device,path)
        report['paths'][path]={}
        started=time.perf_counter()
        try:
            report['paths'][path]['dispatch']=model.dispatch([(data[0]['query'],t) for t in data[0]['texts'][:32]])
            for index,r in enumerate(data):
                for j,k in enumerate(plan['document_counts']):
                    item=records[index*3+j];values=model.request(r,k,plan['pair_microbatch'])
                    item['scores'][path]=values;item['rankings'][path]=ranked(item['candidate_ids'],values)
                    item['metrics'][path]=metrics(item['rankings'][path],r['qrels'])
                if (index+1)%50==0: print(f'accuracy {path}: {index+1}/{len(data)} queries',flush=True)
        except Exception as error:
            if path!='compiled':raise
            partial=[r for r in records if path in r['scores']]
            artifact=HERE/f'{args.device}_compiled_partial_scores.jsonl.gz'
            with gzip.open(artifact,'wt') as f:
                for r in partial:f.write(json.dumps(r,allow_nan=False)+'\n')
            report['paths'][path].update(status='failed_heterogeneous_workload',diagnostic_only=True,
                accuracy_qualified=False,error_type=type(error).__name__,error=str(error),
                completed_records=len(partial),partial_audit=artifact.name,partial_audit_sha256=sha(artifact))
            for r in records:
                for field in ('scores','rankings','metrics'):r[field].pop(path,None)
            print('Compiled workload failed; diagnostic retained:',type(error).__name__,flush=True)
        report['paths'][path]['audit_wall_seconds']=time.perf_counter()-started
        del model
        import gc;gc.collect()
    audit=HERE/f'{args.device}_scores.jsonl.gz'
    with gzip.open(audit,'wt') as f:
        for r in records:f.write(json.dumps(r,allow_nan=False)+'\n')
    report.update(status='complete',audit_file=audit.name,audit_sha256=sha(audit),
                  accuracy=summarize(records,plan),environment_after=environment())
    save(HERE/f'{args.device}_accuracy.json',report)
    print(json.dumps({p:{k:v for k,v in r.items() if k!='cells'} for p,r in report['accuracy'].items()},indent=2))


if __name__=='__main__':main()
