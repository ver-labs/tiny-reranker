"""Inference-only FP32 CUDA/Triton residual addition + LayerNorm wrapper."""
import importlib.util
import math

import torch
import torch.nn.functional as F


def reference(residual, branch, weight, bias, eps=1e-5):
    summed = residual + branch
    return summed, F.layer_norm(summed, (summed.shape[-1],), weight, bias, eps)


def residual_layer_norm(residual, branch, weight, bias, eps=1e-5, *, fallback=True, num_warps=4):
    """Return (sum, normalized sum). Unsupported contracts fall back or raise.

    Custom path: NVIDIA CUDA, contiguous nonempty rank-2/3 FP32 arrays, matching
    residual shapes, FP32 affine vectors, hidden size 1..1024. Use finite inputs
    with representable sums and reduction intermediates. No backward support.
    """
    arrays = (residual, branch, weight, bias)
    if residual.ndim not in (2, 3) or residual.shape != branch.shape:
        raise ValueError('Residual and branch require matching rank-2/3 shapes')
    hidden = residual.shape[-1]
    if hidden < 1 or weight.shape != (hidden,) or bias.shape != (hidden,):
        raise ValueError('Affine vectors must match a nonzero hidden dimension')
    if any(a.device != residual.device for a in arrays):
        raise ValueError('All inputs must be on the same device')
    if not math.isfinite(eps) or not torch.finfo(torch.float32).tiny <= eps <= torch.finfo(torch.float32).max:
        raise ValueError('epsilon must be positive and representable as normal FP32')
    if num_warps not in (1, 2, 4):
        raise ValueError('Supported launch configurations use 1, 2 or 4 warps')
    supported = (residual.is_cuda and torch.version.hip is None and
                 torch.cuda.get_device_capability(residual.device) >= (8, 0) and
                 residual.numel() > 0 and hidden <= 1024 and
                 all(a.dtype == torch.float32 and a.is_contiguous() for a in arrays) and
                 importlib.util.find_spec('triton') is not None)
    if not supported:
        if fallback:
            return reference(*arrays, eps)
        raise RuntimeError('Custom path requires NVIDIA CUDA + Triton, contiguous FP32 inputs, hidden size 1..1024')
    if torch.is_grad_enabled() and any(a.requires_grad for a in arrays):
        raise RuntimeError('Custom Triton path is inference-only; use torch.inference_mode()')
    from .triton_impl import fused_residual_layer_norm
    summed = torch.empty_like(residual)
    normalized = torch.empty_like(residual)
    rows = residual.numel() // hidden
    if rows > 2**31 - 1:
        raise ValueError('Too many rows for the launch grid')
    block = max(32, 1 << (hidden - 1).bit_length())
    with torch.cuda.device(residual.device):
        fused_residual_layer_norm[(rows,)](
            *arrays, summed, normalized, hidden, float(eps), block,
            num_warps=num_warps, enable_fp_fusion=False)
    return summed, normalized
