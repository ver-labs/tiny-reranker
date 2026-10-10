"""Experimental, inference-only Triton -> Metal -> MLX launch wrapper.

Uses triton-msl's documented MLX interface. Its compiler still requires the
Metal Toolchain, even though MLX compiles the emitted source at runtime.
This wrapper has not passed GPU correctness checks on the current machine.
"""
import mlx.core as mx
from triton_msl.mlx import triton_call

from experiments.E13.triton_impl import fused_residual_layer_norm


def residual_layer_norm(residual, branch, weight, bias, eps=1e-5, *, num_warps=4):
    if residual.shape != branch.shape or residual.ndim not in (2, 3):
        raise ValueError("Inputs must have matching rank 2/3 shapes")
    hidden = residual.shape[-1]
    if hidden != 128 or num_warps not in (1, 2, 4):
        raise ValueError("Apple pilot supports hidden=128 and 1/2/4 warps only")
    if weight.shape != (hidden,) or bias.shape != (hidden,):
        raise ValueError("Affine parameters must match hidden size")
    if any(x.dtype != mx.float32 for x in (residual, branch, weight, bias)):
        raise ValueError("Apple pilot requires FP32")
    if not 0 < eps < float("inf"):
        raise ValueError("eps must be finite and positive")
    rows = residual.size // hidden
    if not 0 < residual.size < 2**31:
        raise ValueError("Nonempty inputs with fewer than 2^31 elements required")
    # Shape/dtype placeholders only: the MLX launcher allocates fresh outputs.
    placeholder = mx.zeros(residual.shape, dtype=mx.float32)
    return tuple(triton_call(
        fused_residual_layer_norm, residual, branch, weight, bias, placeholder, placeholder,
        grid=(rows,), N=hidden, EPS=float(eps), BLOCK=hidden, num_warps=num_warps,
    ))
