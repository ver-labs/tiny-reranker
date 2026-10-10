"""Post-hoc diagnostic of an unexplained eager flag gap; not a primary result.

Reuses one model, tokenizer and prepared tensors for both settings to remove
model-loading/path-state differences. Does not amend the original contract.
"""
import json
import random
import time
import numpy as np
from .common import HERE, rows, sha, save, verify_plan, environment, describe, interval
from .inference import Inference


def main():
    plan=verify_plan();data={(r['dataset'],r['query_id']):r for r in rows()}
    selected=[data[tuple(k)] for k in plan['http_subset']]
    pairs=[(r['query'],t) for r in selected for d,t in sorted(zip(r['document_ids'][:10],r['texts'][:10]))]
    chunks=[pairs[i:i+32] for i in range(0,len(pairs),32)]
    model=Inference('mps','strict');resident=[model.prepare(model.encode(c)) for c in chunks]
    report={'classification':'Post-hoc same-hot-model diagnostic, not independent primary sessions',
            'plan_sha256':sha(HERE/'plan.json'),'source_sha256':sha(__file__),'pairs':160,'documents':10,
            'microbatch':32,'repeats':24,'seed':1701,'environment_before':environment(),
            'scope':'Complete scoring, including tokenization/transfers/CPU-readable scores; same model/tokenizer across flags',
            'observations':[]}
    for path in ('strict','stock'):
        model.path=path
        for _ in range(10):
            for chunk in chunks:model.score(chunk)
    orders=[['strict','stock'] for _ in range(12)]+[['stock','strict'] for _ in range(12)]
    random.Random(1701).shuffle(orders)
    for repeat,order in enumerate(orders):
        values={}
        for path in order:
            model.path=path;model.sync();start=time.perf_counter();scores=[]
            for chunk in chunks:scores+=model.score(chunk)
            model.sync();elapsed=time.perf_counter()-start
            values[path]=scores
            report['observations'].append({'repeat':repeat,'path':path,'order':order,'seconds':elapsed,'pairs_per_second':160/elapsed})
        if values['strict']!=values['stock']:raise RuntimeError('Diagnostic eager outputs changed')
    grouped={p:{r['repeat']:r['seconds'] for r in report['observations'] if r['path']==p} for p in ('strict','stock')}
    logs=[np.log(grouped['strict'][i]/grouped['stock'][i]) for i in range(24)]
    v=interval(logs)
    report['summary']={'stock_over_strict_ratio':float(np.exp(v['mean'])),
        'within_process_ci95':np.exp(v['ci95']).tolist(),
        'scope_of_interval':'Within one hot process only; does not estimate independent-session uncertainty',
        'seconds':{p:describe(list(d.values())) for p,d in grouped.items()},'exact_score_parity':True}
    report['environment_after']=environment();save(HERE/'flag_diagnostic.json',report)
    print(json.dumps(report['summary'],indent=2))


if __name__=='__main__':main()
