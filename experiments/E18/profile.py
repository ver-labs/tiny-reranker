"""Separate synchronized stage-wall diagnostics; never claimed as kernel time."""
import argparse
import json
import time
import numpy as np
from .common import HERE,pairs,ranked,save,verify_plan,sha
from .direct_benchmark import selected
from .inference import Inference


def main():
    p=argparse.ArgumentParser();p.add_argument('--device',required=True);p.add_argument('--path',required=True);a=p.parse_args();plan=verify_plan();model=Inference(a.device,a.path)
    row=selected(plan['sweep_ids'])[0];pp=pairs(row,100)
    model.request(row,100);stages=[]
    for _ in range(3):
        elapsed={'tokenization':0.,'transfer_or_array':0.,'forward':0.,'cpu_scores':0.,'sorting':0.,'serialization':0.};scores=[]
        for o in range(0,100,16):
            model.sync();start=time.perf_counter();encoded=model.encode(pp[o:o+16]);elapsed['tokenization']+=time.perf_counter()-start
            start=time.perf_counter();inputs=model.prepare(encoded);model.sync();elapsed['transfer_or_array']+=time.perf_counter()-start
            start=time.perf_counter();out=model.forward(inputs);model.sync();elapsed['forward']+=time.perf_counter()-start
            start=time.perf_counter();scores+=model.cpu(out);elapsed['cpu_scores']+=time.perf_counter()-start
        ids=sorted(row['document_ids']);start=time.perf_counter();order=ranked(ids,scores);elapsed['sorting']=time.perf_counter()-start
        mapping=dict(zip(ids,scores));start=time.perf_counter();json.dumps([{'document_id':d,'score':mapping[d]} for d in order]);elapsed['serialization']=time.perf_counter()-start
        stages.append(elapsed)
    save(HERE/f'{a.device}_{a.path}_profile.json',{'device':a.device,'path':a.path,'plan_sha256':sha(HERE/'plan.json'),'k':100,'batch':16,'length':512,'scope':'Synchronized stage wall time, three diagnostic passes; not GPU kernel time','stages_seconds':stages,'dispatch':model.dispatch(pp[:16])})

if __name__=='__main__':main()
