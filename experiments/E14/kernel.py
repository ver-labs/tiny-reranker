"""Handwritten CUDA port of the Metal algorithm; ctypes binding, no autograd."""
import ctypes
import functools
import math
from pathlib import Path
import torch
from experiments.E13.kernel import reference


@functools.lru_cache(maxsize=1)
def library():
    lib = ctypes.CDLL(str(Path(__file__).with_name('residual_layer_norm.so')))
    launch = lib.launch_residual_layer_norm
    launch.argtypes = [ctypes.c_void_p] * 6 + [ctypes.c_int, ctypes.c_int,
                                              ctypes.c_float, ctypes.c_void_p]
    launch.restype = ctypes.c_int
    return launch


def residual_layer_norm(residual, branch, weight, bias, eps=1e-5, *, fallback=True, num_warps=1):
    arrays = (residual, branch, weight, bias)
    if residual.ndim not in (2, 3) or residual.shape != branch.shape:
        raise ValueError('Residual and branch require matching rank-2/3 shapes')
    hidden = residual.shape[-1]
    if hidden < 1 or weight.shape != (hidden,) or bias.shape != (hidden,):
        raise ValueError('Affine vectors must match hidden dimension')
    if any(a.device != residual.device for a in arrays):
        raise ValueError('All inputs must be on the same device')
    if not math.isfinite(eps) or not torch.finfo(torch.float32).tiny <= eps <= torch.finfo(torch.float32).max:
        raise ValueError('epsilon must be positive and representable as normal FP32')
    supported = (residual.is_cuda and torch.version.hip is None and residual.numel() > 0
                 and hidden in (32, 64, 128, 256)
                 and all(a.dtype == torch.float32 and a.is_contiguous() for a in arrays))
    if not supported:
        if fallback: return reference(*arrays, eps)
        raise RuntimeError('Custom CUDA requires contiguous NVIDIA FP32 inputs; hidden 32/64/128/256')
    if torch.is_grad_enabled() and any(a.requires_grad for a in arrays):
        raise RuntimeError('Custom CUDA path is inference-only')
    rows = residual.numel() // hidden
    if rows > 2**31 - 1: raise ValueError('Too many rows')
    summed, normalized = torch.empty_like(residual), torch.empty_like(residual)
    with torch.cuda.device(residual.device):
        status = library()(*(a.data_ptr() for a in (*arrays, summed, normalized)),
                           rows, hidden, float(eps), torch.cuda.current_stream().cuda_stream)
    if status: raise RuntimeError(f'CUDA launch failed with runtime error {status}')
    return summed, normalized
