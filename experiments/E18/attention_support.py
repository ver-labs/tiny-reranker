"""Runtime capability diagnostic; does not change the benchmark attention path."""
import mlx.core as mx
from importlib.metadata import version
from .common import HERE,save


def main():
    q=mx.zeros((16,12,512,32),dtype=mx.float32);mask=mx.ones((16,1,1,512),dtype=mx.bool_)
    result={'mlx_version':version('mlx'),'dtype':'FP32','qkv_shape':[16,12,512,32],'stream':'GPU','force_fused':'Capability diagnostic only; benchmark keeps default routing','source':'https://github.com/ml-explore/mlx/blob/v0.32.3/mlx/backend/metal/scaled_dot_product_attention.cpp#L697'}
    try:
        out=mx.fast.scaled_dot_product_attention(q,q,q,scale=32**-.5,mask=mask,force_fused=True,stream=mx.gpu);mx.eval(out);mx.synchronize()
        result['status']='FUSED_SUPPORTED'
    except (ValueError,RuntimeError) as e:result.update(status='FUSED_UNSUPPORTED',error=str(e))
    save(HERE/'mlx_attention_support.json',result);print(result,flush=True)

if __name__=='__main__':main()
