#include <cuda_runtime.h>
#include <cstddef>
__device__ float reduce(float x) {
    for (int delta=16; delta; delta/=2) x += __shfl_xor_sync(0xffffffffu,x,delta);
    return x;
}
__global__ void e18_residual_norm_384(const float* r,const float* x,const float* w,
    const float* b,float* y,float epsilon) {
    const int lane=threadIdx.x;
    const size_t base=size_t(blockIdx.x)*384;
    const float anchor=r[base]+x[base];
    float values[12],sum=0.0f;
    #pragma unroll
    for (int i=0;i<12;++i) { values[i]=(r[base+lane+i*32]+x[base+lane+i*32])-anchor; sum+=values[i]; }
    const float mean=reduce(sum)/384.0f;
    float squares=0.0f;
    #pragma unroll
    for (int i=0;i<12;++i) { values[i]-=mean; squares+=values[i]*values[i]; }
    const float inverse=1.0f/sqrtf(reduce(squares)/384.0f+epsilon);
    #pragma unroll
    for (int i=0;i<12;++i) { const int col=lane+i*32; y[base+col]=w[col]*(values[i]*inverse)+b[col]; }
}
extern "C" int launch_e18(const float* r,const float* x,const float* w,const float* b,
    float* y,int rows,float eps,void* stream) {
    e18_residual_norm_384<<<rows,32,0,reinterpret_cast<cudaStream_t>(stream)>>>(r,x,w,b,y,eps);
    return int(cudaGetLastError());
}
