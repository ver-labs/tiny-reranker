"""Inference-only 384-wide CUDA/Triton and matched explicit BERT graph."""
import ctypes
import functools
import math
from pathlib import Path
import types
import torch


@functools.lru_cache(maxsize=1)
def library():
    lib=ctypes.CDLL(str(Path(__file__).with_name('residual_norm.so')))
    function=lib.launch_e18
    function.argtypes=[ctypes.c_void_p]*5+[ctypes.c_int,ctypes.c_float,ctypes.c_void_p]
    function.restype=ctypes.c_int
    return function


def residual_norm(residual,branch,weight,bias,eps=1e-12,path='cuda'):
    arrays=(residual,branch,weight,bias)
    if residual.shape!=branch.shape or residual.ndim not in (2,3) or residual.shape[-1]!=384:
        raise ValueError('Matching rank-2/3 hidden-384 inputs required')
    if weight.shape!=(384,) or bias.shape!=(384,) or residual.numel()==0:
        raise ValueError('Nonempty inputs and affine vectors required')
    if not math.isfinite(eps) or not torch.finfo(torch.float32).tiny<=eps<=torch.finfo(torch.float32).max:
        raise ValueError('Positive normal FP32 epsilon required')
    if not residual.is_cuda or torch.version.hip is not None or any(a.device!=residual.device or a.dtype!=torch.float32 or not a.is_contiguous() for a in arrays):
        raise RuntimeError('Contiguous NVIDIA FP32 tensors required; no fallback')
    if torch.is_grad_enabled() and any(a.requires_grad for a in arrays):
        raise RuntimeError('Inference only')
    if path not in ('cuda','triton'):
        raise ValueError(path)
    output=torch.empty_like(residual)
    count=residual.numel()//384
    if path=='cuda':
        with torch.cuda.device(residual.device):
            status=library()(*(a.data_ptr() for a in (*arrays,output)),count,float(eps),torch.cuda.current_stream().cuda_stream)
        if status:raise RuntimeError(f'CUDA launch failed: {status}')
    else:
        from .triton_kernel import e18_residual_norm_384
        e18_residual_norm_384[(count,)](*arrays,output,EPS=float(eps),num_warps=4,enable_fp_fusion=False)
    return output


def install(model,path,owner):
    for layer in model.bert.encoder.layer:
        for site in (layer.attention.output,layer.output):
            def forward(self,hidden_states,input_tensor):
                branch=self.dense(hidden_states)
                if path=='explicit':
                    return self.LayerNorm(branch+input_tensor)
                owner.custom_calls+=1
                return residual_norm(input_tensor,branch,self.LayerNorm.weight,self.LayerNorm.bias,self.LayerNorm.eps,path)
            site.forward=types.MethodType(forward,site)
