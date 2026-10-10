"""Reduced secondary quality check; shortening context changes task inputs."""
import argparse
import gzip
import json
from .common import HERE,save,sha,ranked,verify_plan
from .direct_benchmark import selected
from .accuracy import read
from .inference import Inference
from .statistics import accuracy_gate,interval
from experiments.E15.metrics import metrics


def main():
    p=argparse.ArgumentParser();p.add_argument('--device',required=True);a=p.parse_args();plan=verify_plan();data=selected(plan['timing_ids']);output={'device':a.device,'scope':'Secondary 16-query / K100 check; no all-485 shortened-context qualification','plan_sha256':sha(HERE/'plan.json'),'cells':{}}
    baseline=read(HERE/'scores'/f'{a.device}_stock.jsonl.gz');lookup={(r['dataset'],r['query_id'],r['k']):r for r in baseline};paths=['stock','mlx'] if a.device=='mps' else ['stock']
    records_by={}
    for length in (128,256):
        for path in paths:
            model=Inference(a.device,path,length);records=[]
            for row in data:
                scores=model.request(row,100);ids=row['document_ids'];ranking=ranked(ids,scores)
                records.append({'dataset':row['dataset'],'query_id':row['query_id'],'k':100,'ids':ids,'scores':scores,'ranking':ranking,'qrels':row['qrels'],'metrics':metrics(ranking,row['qrels'])})
            artifact=HERE/'scores'/f'{a.device}_{path}_L{length}_context.jsonl.gz'
            with gzip.open(artifact,'wt') as f:
                for r in records:f.write(json.dumps(r)+'\n')
            entry={'artifact':str(artifact.relative_to(HERE)),'artifact_sha256':sha(artifact),'quality_changes_vs_stock_L512':{}}
            for metric in ('ndcg_at_10','mrr_at_10'):
                entry['quality_changes_vs_stock_L512'][metric]=interval([r['metrics'][metric]-lookup[r['dataset'],r['query_id'],100]['metrics'][metric] for r in records])
            if path!='stock':entry['same_context_framework_accuracy']=accuracy_gate(records_by[length,'stock'],records)
            records_by[length,path]=records;output['cells'][f'L{length}/{path}']=entry
    save(HERE/f'{a.device}_context_check.json',output)

if __name__=='__main__':main()
