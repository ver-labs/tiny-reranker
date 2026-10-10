"""GPU checks for documented layout semantics, outside performance timers."""
import argparse
import numpy as np
from .common import HERE, save
from .kernel_check import oracle


def check(device):
    rng=np.random.default_rng(1801)
    arrays=[rng.normal(size=s).astype(np.float32) for s in ((7,768),(7,768),(768,),(768,))]
    report={'device':device,'scope':'Correctness only, no performance measurements','cases':[]}
    def record(name,actual,values,path):
        expected=oracle(*values)
        error=np.abs(actual-expected)
        passed=bool(np.isfinite(actual).all() and np.all(error<=2e-5+1e-4*np.abs(expected)))
        report['cases'].append({'case':name,'path':path,'passed':passed,'max_abs_error':float(error.max())})
        assert passed,(name,path,float(error.max()))
    if device=='mps':
        import mlx.core as mx
        from .metal_kernel import residual_norm
        views=[mx.array(v)[...,::2] for v in arrays]
        record('strided_inputs_explicitly_packed',np.array(residual_norm(*views)),[v[...,::2] for v in arrays],'metal')
        try:residual_norm(*[v.astype(mx.float16) for v in views])
        except ValueError:report['cases'].append({'case':'fp16_rejected','passed':True})
        else:raise AssertionError('FP16 accepted')
    else:
        import torch
        from .torch_kernel import residual_norm
        with torch.inference_mode():
            for path in ('cuda','triton'):
                views=[torch.from_numpy(v).cuda()[...,::2] for v in arrays]
                try:residual_norm(*views,path=path)
                except RuntimeError:report['cases'].append({'case':'noncontiguous_rejected','path':path,'passed':True})
                else:raise AssertionError('Noncontiguous inputs accepted')
                values=[np.ascontiguousarray(v[...,::2]) for v in arrays]
                # A contiguous tensor may begin one scalar after an aligned allocation.
                shifted=[]
                for value in values:
                    storage=torch.empty(value.size+1,device='cuda',dtype=torch.float32)
                    tensor=storage[1:].reshape(value.shape);tensor.copy_(torch.from_numpy(value))
                    assert tensor.is_contiguous() and tensor.data_ptr()%16==4
                    shifted.append(tensor)
                record('scalar_offset_contiguous',residual_norm(*shifted,path=path).cpu().numpy(),values,path)
                try:residual_norm(*[v.half() for v in shifted],path=path)
                except RuntimeError:report['cases'].append({'case':'fp16_rejected','path':path,'passed':True})
                else:raise AssertionError('FP16 accepted')
    report['status']='VERIFIED';save(HERE/f'{device}_kernel_guards.json',report)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--device',required=True,choices=['mps','cuda'])
    print(check(parser.parse_args().device))
