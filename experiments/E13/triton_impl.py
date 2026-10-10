"""GPU implementation, loaded lazily so CPU/Mac tooling needs no Triton install.

Uses the reduction pattern in Triton's official LayerNorm tutorial, adapted to
add the residual in registers and return both the sum and normalized values.
No backward implementation is provided.
"""
import triton
import triton.language as tl


@triton.jit
def fused_residual_layer_norm(R, X, W, B, S, Y,
                             N: tl.constexpr, EPS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    valid = cols < N
    # int64 offsets avoid overflow for large row counts.
    offsets = row.to(tl.int64) * N + cols
    residual = tl.load(R + offsets, mask=valid, other=0.).to(tl.float32)
    branch = tl.load(X + offsets, mask=valid, other=0.).to(tl.float32)
    summed = residual + branch
    mean = tl.sum(summed, axis=0) / N
    centered = tl.where(valid, summed - mean, 0.)
    variance = tl.sum(centered * centered, axis=0) / N
    inverse_std = tl.rsqrt(variance + EPS)
    weight = tl.load(W + cols, mask=valid, other=0.).to(tl.float32)
    bias = tl.load(B + cols, mask=valid, other=0.).to(tl.float32)
    normalized = (centered * inverse_std) * weight + bias
    tl.store(S + offsets, summed, mask=valid)
    tl.store(Y + offsets, normalized, mask=valid)
