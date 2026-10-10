"""384-wide FP32 Metal operation; no fallback and explicit packing semantics."""
from functools import lru_cache
import math
from pathlib import Path
import mlx.core as mx

CALLS = 0


@lru_cache(maxsize=1)
def kernel():
    return mx.fast.metal_kernel(name='e18_residual_norm_384_fp32',
        input_names=['residual','branch','weight','bias','epsilon'],output_names=['normalized'],
        source=Path(__file__).with_name('residual_norm.metal').read_text(),
        ensure_row_contiguous=True,compile_options={'math_mode':'safe'})


@lru_cache(maxsize=4)
def epsilon(value):
    return mx.array([value],dtype=mx.float32)


def residual_norm(residual,branch,weight,bias,eps=1e-12):
    global CALLS
    if residual.shape!=branch.shape or residual.ndim not in (2,3) or residual.shape[-1]!=384:
        raise ValueError('Matching rank-2/3, hidden 384 required')
    if weight.shape!=(384,) or bias.shape!=(384,) or residual.size==0:
        raise ValueError('Nonempty inputs and 384-wide affine vectors required')
    if any(a.dtype!=mx.float32 for a in (residual,branch,weight,bias)):
        raise ValueError('FP32 required; no fallback')
    if not math.isfinite(eps) or not 1.1754944e-38<=eps<=3.4028235e38:
        raise ValueError('Positive normal FP32 epsilon required')
    rows=residual.size//384
    CALLS += 1
    # MLX packs noncontiguous views inside dispatch; that work is included in
    # caller time. This is explicit materialization, never a native fallback.
    return kernel()(inputs=[residual,branch,weight,bias,epsilon(float(eps))],
                    template=[('ROWS',rows)],grid=(32,rows,1),threadgroup=(32,1,1),
                    output_shapes=[residual.shape],output_dtypes=[mx.float32])[0]
