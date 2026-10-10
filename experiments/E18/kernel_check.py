"""Predeclared operation oracle checks, including real BERT activations."""
import argparse
import numpy as np
from .common import HERE,fixtures,save
from .inference import Inference


def oracle(r,x,w,b,eps=1e-12):
    values=(r+x).astype(np.float32).astype(np.float64)
    centered=values-values.mean(axis=-1,keepdims=True)
    return centered/np.sqrt(np.mean(centered**2,axis=-1,keepdims=True)+eps)*w+b


def main():
    p=argparse.ArgumentParser();p.add_argument('--device',choices=['mps','cuda'],required=True);a=p.parse_args()
    rng=np.random.default_rng(1801)
    cases=[]
    for shape in ((1,384),(5,384),(2,37,384)):
        for kind in ('random','zero','constant','near_constant','cancellation','mixed_magnitude'):
            r=rng.normal(size=shape).astype(np.float32);x=rng.normal(size=shape).astype(np.float32)
            if kind=='zero':r.fill(0);x.fill(0)
            if kind=='constant':r.fill(10000);x.fill(-9999)
            if kind=='near_constant':r=1+r*1e-6;x*=1e-8
            if kind=='cancellation':r*=1e5;x=-r+x
            if kind=='mixed_magnitude':r[...,::2]*=1e5;x[...,1::2]*=1e-5
            cases.append((kind,r,x,np.ones(384,np.float32),np.zeros(384,np.float32)))
    model=Inference(a.device,'stock')
    hooks=[]
    def capture(site,args):
        import torch
        with torch.inference_mode():
            branch=site.dense(args[0]);residual=args[1]
            cases.append(('real_activation',residual.detach().cpu().numpy().reshape(-1,384)[:17],
                branch.detach().cpu().numpy().reshape(-1,384)[:17],
                site.LayerNorm.weight.detach().cpu().numpy(),site.LayerNorm.bias.detach().cpu().numpy()))
    for layer in model.model.bert.encoder.layer:
        hooks += [site.register_forward_pre_hook(capture) for site in (layer.attention.output,layer.output)]
    row=fixtures()[0];model.score([(row['query'],t) for t in row['texts'][:16]])
    for hook in hooks:hook.remove()
    report={'device':a.device,'epsilon':1e-12,'tolerance':'abs <= 2e-5 + 1e-4*abs(float64_oracle)',
            'oracle':'Residual addition rounded FP32, centered statistics/affine in CPU FP64',
            'cases':[]}
    for path in (('metal',) if a.device=='mps' else ('cuda','triton')):
        for kind,r,x,w,b in cases:
            expected=oracle(r,x,w,b)
            if a.device=='mps':
                import mlx.core as mx
                from .metal_kernel import residual_norm
                actual=np.array(residual_norm(*[mx.array(v) for v in (r,x,w,b)]))
                native=np.array(mx.fast.layer_norm(mx.array(r)+mx.array(x),mx.array(w),mx.array(b),1e-12))
            else:
                import torch
                from .torch_kernel import residual_norm
                tensors=[torch.from_numpy(np.ascontiguousarray(v)).cuda() for v in (r,x,w,b)]
                with torch.inference_mode():
                    actual=residual_norm(*tensors,path=path).cpu().numpy()
                    native=torch.nn.functional.layer_norm(tensors[0]+tensors[1],(384,),tensors[2],tensors[3],1e-12).cpu().numpy()
            error=np.abs(actual-expected)
            report['cases'].append({'path':path,'case':kind,'shape':list(r.shape),'max_abs_error':float(error.max()),
                 'native_max_abs_error':float(np.max(np.abs(native-expected))),
                 'passed':bool(np.isfinite(actual).all() and np.all(error<=2e-5+1e-4*np.abs(expected)))})
    report['paths']={path:{'status':'VERIFIED' if all(c['passed'] for c in report['cases'] if c['path']==path) else 'FAILED_ACCURACY'} for path in {c['path'] for c in report['cases']}}
    save(HERE/f'{a.device}_kernel_check.json',report)
    print(report['paths'])


if __name__=='__main__':main()
