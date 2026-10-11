"""One confirmation pass on the untouched 421 queries; no pseudo CI."""
import argparse
import gzip
import json
import time
from .common import HERE,fixtures,ranked,save,sha,verify_plan
from .inference import Inference
from .statistics import accuracy_gate
from .accuracy import read
from .metrics import metrics


def main():
    p=argparse.ArgumentParser();p.add_argument('--device',required=True,choices=['mps','cuda']);p.add_argument('--path',required=True);a=p.parse_args();verify_plan()
    model=Inference(a.device,a.path);data=fixtures(True);artifact=HERE/'scores'/f'{a.device}_{a.path}_reserve.jsonl.gz';artifact.parent.mkdir(exist_ok=True)
    started=time.perf_counter();records=[]
    with gzip.open(artifact,'wt') as f:
        for i,row in enumerate(data):
            values=model.request(row,100);ids=row['document_ids'];order=ranked(ids,values)
            record={'dataset':row['dataset'],'query_id':row['query_id'],'k':100,'ids':ids,'scores':values,'ranking':order,'metrics':metrics(order,row['qrels']),'qrels':row['qrels']};records.append(record);f.write(json.dumps(record)+'\n')
            if i%50==0:print(a.device,a.path,'reserve',i,flush=True)
    elapsed=time.perf_counter()-started
    keys={(r['dataset'],r['query_id']) for r in records};reference=[r for r in read(HERE/'scores'/f'{a.device}_stock.jsonl.gz') if r['k']==100 and (r['dataset'],r['query_id']) in keys]
    lookup={(r['dataset'],r['query_id']):r for r in reference};reference=[lookup[r['dataset'],r['query_id']] for r in records]
    save(HERE/f'{a.device}_{a.path}_reserve.json',{'status':'COMPLETE','device':a.device,'path':a.path,'queries':421,'pairs':42100,'seconds':elapsed,'plan_sha256':sha(HERE/'plan.json'),'artifact':str(artifact.relative_to(HERE)),'artifact_sha256':sha(artifact),'interpretation':'One confirmation pass; file writes and ranking included in wall time; not a repeated direct performance endpoint','accuracy':accuracy_gate(reference,records)})

if __name__=='__main__':main()
