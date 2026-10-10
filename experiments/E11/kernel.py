"""Inference-only fused residual addition + LayerNorm, with ordinary MLX fallback."""
from functools import lru_cache
import math
from pathlib import Path

import mlx.core as mx

SUPPORTED_HIDDEN = (32, 64, 128, 256)
SOURCE = Path(__file__).with_name('residual_layer_norm.metal')


@lru_cache(maxsize=1)
def _kernel():
    return mx.fast.metal_kernel(
        name='tiny_reranker_residual_layer_norm_fp32',
        input_names=['residual', 'branch', 'weight', 'bias', 'epsilon'],
        output_names=['summed', 'normalized'], source=SOURCE.read_text(),
        ensure_row_contiguous=True, compile_options={'math_mode': 'safe'})


@lru_cache(maxsize=16)
def _epsilon(value):
    return mx.array([value], dtype=mx.float32)


def reference(residual, branch, weight, bias, eps=1e-5):
    summed = residual + branch
    return summed, mx.fast.layer_norm(summed, weight, bias, eps)


def residual_layer_norm(residual, branch, weight, bias, eps=1e-5, *, fallback=True):
    """Return (residual sum, normalized sum); never use this forward kernel to train.

    Custom path: nonempty rank-2/3 FP32 arrays with matching shapes and a supported
    hidden size, FP32 affine vectors. MLX packs strided arrays before dispatch;
    this copy is included in caller timing. Other dtypes/sizes use the reference
    unless fallback=False, which raises. No broadcasting of residuals is accepted.
    """
    if residual.shape != branch.shape or residual.ndim not in (2, 3):
        raise ValueError('Residual and branch must have matching rank-2/3 shapes')
    hidden = residual.shape[-1]
    if weight.shape != (hidden,) or bias.shape != (hidden,):
        raise ValueError('Affine parameters must match the hidden dimension')
    if not math.isfinite(eps) or eps <= 0 or eps > 3.4028235e38 or eps < 1.1754944e-38:
        raise ValueError('epsilon must be finite, positive and representable as normal FP32')
    supported = (hidden in SUPPORTED_HIDDEN and residual.size > 0 and
                 all(a.dtype == mx.float32 for a in (residual, branch, weight, bias)))
    if not supported:
        if fallback:
            return reference(residual, branch, weight, bias, eps)
        raise ValueError('Custom kernel supports nonempty FP32 inputs and hidden sizes 32/64/128/256')
    rows = residual.size // hidden
    if rows > 2**31 - 1:
        raise ValueError('Too many rows for Metal grid')
    outputs = _kernel()(
        inputs=[residual, branch, weight, bias, _epsilon(float(eps))],
        template=[('HIDDEN', hidden), ('ROWS', rows)],
        grid=(32, rows, 1), threadgroup=(32, 1, 1),
        output_shapes=[residual.shape, residual.shape],
        output_dtypes=[mx.float32, mx.float32])
    return tuple(outputs)
