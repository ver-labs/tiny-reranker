// CUDA C++ port of E11/residual_layer_norm.metal: one warp per row,
// strided lane values, two-pass centered population variance, two outputs.
#include <cuda_runtime.h>
#include <cstddef>

__device__ float warp_sum(float value) {
    for (int offset = 16; offset; offset /= 2)
        value += __shfl_xor_sync(0xffffffffu, value, offset);
    return value;
}

template<int H>
__global__ void fused(const float* residual, const float* branch,
                      const float* weight, const float* bias,
                      float* summed, float* output, float epsilon) {
    const int lane = threadIdx.x;
    const size_t base = size_t(blockIdx.x) * H;
    float values[H / 32];
    float partial_mean = 0.0f;
    #pragma unroll
    for (int i = 0; i < H / 32; ++i) {
        const int col = lane + i * 32;
        values[i] = residual[base + col] + branch[base + col];
        summed[base + col] = values[i];
        partial_mean += values[i];
    }
    const float mean = warp_sum(partial_mean) / H;
    float partial_variance = 0.0f;
    #pragma unroll
    for (int i = 0; i < H / 32; ++i) {
        values[i] -= mean;
        partial_variance += values[i] * values[i];
    }
    const float variance = warp_sum(partial_variance) / H;
    // IEEE division/sqrt; do not enable fast math or fused multiply-add.
    const float inverse_std = 1.0f / sqrtf(variance + epsilon);
    #pragma unroll
    for (int i = 0; i < H / 32; ++i) {
        const int col = lane + i * 32;
        const float normalized = values[i] * inverse_std;
        output[base + col] = weight[col] * normalized + bias[col];
    }
}

extern "C" int launch_residual_layer_norm(const float* residual, const float* branch,
    const float* weight, const float* bias, float* summed, float* output,
    int rows, int hidden, float epsilon, void* stream_pointer) {
    const cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_pointer);
    #define LAUNCH(H) fused<H><<<rows, 32, 0, stream>>>(residual, branch, weight, bias, summed, output, epsilon)
    switch (hidden) {
        case 32: LAUNCH(32); break;
        case 64: LAUNCH(64); break;
        case 128: LAUNCH(128); break;
        case 256: LAUNCH(256); break;
        default: return int(cudaErrorInvalidValue);
    }
    return int(cudaGetLastError());
}
