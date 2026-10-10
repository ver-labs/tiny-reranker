import triton
import triton.language as tl


@triton.jit
def e18_residual_norm_384(R,X,W,B,Y,EPS:tl.constexpr):
    row=tl.program_id(0)
    col=tl.arange(0,512)
    valid=col<384
    base=row.to(tl.int64)*384
    anchor=tl.load(R+base)+tl.load(X+base)
    values=tl.load(R+base+col,mask=valid,other=0.).to(tl.float32)+tl.load(X+base+col,mask=valid,other=0.).to(tl.float32)
    shifted=tl.where(valid,values-anchor,0.)
    mean=tl.sum(shifted,0)/384
    centered=tl.where(valid,shifted-mean,0.)
    variance=tl.sum(centered*centered,0)/384
    inverse=tl.rsqrt(variance+EPS)
    w=tl.load(W+col,mask=valid,other=0.)
    b=tl.load(B+col,mask=valid,other=0.)
    tl.store(Y+base+col,w*(centered*inverse)+b,mask=valid)
