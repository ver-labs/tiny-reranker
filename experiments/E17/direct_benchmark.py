"""Separate synchronized forward and complete-scoring timings in fresh sessions."""
import argparse
import gc
import json
import random
import time
import numpy as np
from .common import HERE, rows, sha, save, verify_plan, environment
from .inference import Inference
from .http_benchmark import load_audit


def main():
    p=argparse.ArgumentParser();p.add_argument('--device',choices=['mps','cuda'],required=True)
    p.add_argument('--session',type=int,required=True,choices=range(4));a=p.parse_args()
    plan=verify_plan();accuracy,refs=load_audit(a.device)
    allrows={(r['dataset'],r['query_id']):r for r in rows()}; selected=[allrows[tuple(k)] for k in plan['http_subset']]
    paths=['strict','stock','mlx'] if a.device=='mps' else ['strict','stock','cuda','triton']
    compile_report=HERE/f'{a.device}_compile.json'
    if compile_report.exists() and json.loads(compile_report.read_text())['status']=='compatible':
        # A successful probe must first receive its full accuracy audit.
        if 'compiled' in accuracy['accuracy']:paths.append('compiled')
    random.Random(plan['seed']).shuffle(paths)
    paths=paths[a.session%len(paths):]+paths[:a.session%len(paths)]
    report={'device':a.device,'session':a.session,'plan_sha256':sha(HERE/'plan.json'),'order':paths,
            'environment_before':environment(),'runs':[]}
    for path in paths:
        model=Inference(a.device,path)
        for k in plan['document_counts']:
            pairs=[(r['query'],t) for r in selected for d,t in sorted(zip(r['document_ids'][:k],r['texts'][:k]))]
            chunks=[pairs[i:i+plan['pair_microbatch']] for i in range(0,len(pairs),plan['pair_microbatch'])]
            inputs=[model.prepare(model.encode(chunk)) for chunk in chunks]
            for _ in range(10):model.forward(inputs[0])
            model.sync()
            # Materialize every prepared shape and complete-scoring shape before
            # timing, including any Inductor compilation/recompilation.
            for chunk,resident in zip(chunks,inputs):
                model.forward(resident);model.score(chunk)
            model.sync()
            for repeat in range(plan['repeats_per_session']):
                for scope in (('forward','direct') if repeat%2==0 else ('direct','forward')):
                    timings=[];scores=[]
                    for chunk,resident in zip(chunks,inputs):
                        model.sync();start=time.perf_counter()
                        if scope=='forward':
                            values=model.forward(resident);model.sync()
                            elapsed=time.perf_counter()-start
                            output=model.cpu(values)
                        else:
                            output=model.score(chunk);model.sync();elapsed=time.perf_counter()-start
                        timings.append(elapsed*1000);scores+=output
                    assert len(scores)==len(pairs) and np.isfinite(scores).all()
                    # Global flattening changes microbatch boundaries vs HTTP. Score
                    # sanity is strict tolerance checked against a one-off same-chunk result.
                    if repeat==0 and scope=='forward':baseline=scores
                    error=np.max(np.abs(np.asarray(scores)-baseline)) if 'baseline' in locals() else 0.
                    if 'baseline' in locals() and not np.allclose(scores,baseline,atol=1e-5,rtol=1e-4):raise RuntimeError('Unstable repeated scores')
                    seconds=sum(timings)/1000
                    report['runs'].append({'path':path,'accuracy_qualified':accuracy['accuracy'][path]['accuracy_qualified'],
                        'documents':k,'repeat':repeat,'scope':scope,'pairs':len(pairs),'seconds':seconds,
                        'pairs_per_second':len(pairs)/seconds,'batch_ms':timings,
                        'batch_pairs':[len(c) for c in chunks],'max_repeat_score_error':float(error)})
                    save(HERE/f'{a.device}_direct_s{a.session}.json',report)
            del inputs
        del model;gc.collect()
    report.update(status='complete',environment_after=environment())
    save(HERE/f'{a.device}_direct_s{a.session}.json',report)


if __name__=='__main__':main()
