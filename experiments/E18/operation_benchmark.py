"""Resident residual+LayerNorm scope, independent operation sessions."""
import argparse
import random
import time
import numpy as np
from .common import HERE,save,sha,verify_plan,environment


def main():
    p=argparse.ArgumentParser();p.add_argument('--device',required=True,choices=['mps','cuda']);p.add_argument('--session',type=int,required=True);a=p.parse_args();verify_plan()
    rng=np.random.default_rng(1801);result={'session':a.session,'device':a.device,'plan_sha256':sha(HERE/'plan.json'),'environment':environment(),'observations':[]}
    paths=['native','metal'] if a.device=='mps' else ['native','cuda','triton']
    if a.session==0 and not (HERE/f'{a.device}_kernel_guards.json').exists():
        from .kernel_guards import check
        check(a.device)
    for rows in (16*128,16*256,16*512):
        values=[rng.normal(size=shape).astype(np.float32) for shape in ((rows,384),(rows,384),(384,),(384,))]
        if a.device=='mps':
            import mlx.core as mx
            from .metal_kernel import residual_norm
            tensors=[mx.array(v) for v in values];mx.eval(tensors)
            def call(path):return residual_norm(*tensors) if path=='metal' else mx.fast.layer_norm(tensors[0]+tensors[1],tensors[2],tensors[3],1e-12)
        else:
            import torch
            from .torch_kernel import residual_norm
            tensors=[torch.from_numpy(v).cuda() for v in values]
            def call(path):return residual_norm(*tensors,path=path) if path!='native' else torch.nn.functional.layer_norm(tensors[0]+tensors[1],(384,),tensors[2],tensors[3],1e-12)
        order=paths[a.session%len(paths):]+paths[:a.session%len(paths)]
        for path in order:
            for _ in range(20):
                out=call(path)
                if a.device=='mps':mx.eval(out)
            for rep in range(3):
                loops=100
                if a.device=='mps':
                    mx.synchronize();start=time.perf_counter()
                    for _ in range(loops):out=call(path);mx.eval(out)
                    mx.synchronize();seconds=(time.perf_counter()-start)/loops;scope='Synchronized amortized dispatch+GPU wall time; not isolated GPU event time'
                else:
                    torch.cuda.synchronize();start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                    start.record()
                    for _ in range(loops):out=call(path)
                    end.record();end.synchronize();seconds=start.elapsed_time(end)/1000/loops;scope='CUDA event interval over 100 eager launches; includes GPU-visible launch gaps, no graph replay'
                result['observations'].append({'session':a.session,'repetition':rep,'path':path,'rows':rows,'hidden':384,'seconds':seconds,'scope':scope})
    result['status']='COMPLETE';save(HERE/'raw'/f'{a.device}_operation_s{a.session}.json',result)

if __name__=='__main__':main()
