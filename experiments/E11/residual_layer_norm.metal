// Kernel body for MLX's custom Metal interface. One 32-thread SIMD group per row.
// Population variance in FP32, computed in two passes as in MLX LayerNorm.
const uint lane = thread_index_in_simdgroup;
const uint row = threadgroup_position_in_grid.y;
if (row >= ROWS) return;
const size_t base = size_t(row) * HIDDEN;
float values[HIDDEN / 32];
float partial_mean = 0.0f;
for (uint i = 0; i < HIDDEN / 32; ++i) {
    const uint column = lane + i * 32;
    const float value = residual[base + column] + branch[base + column];
    values[i] = value;
    summed[base + column] = value;
    partial_mean += value;
}
const float mean = simd_sum(partial_mean) / float(HIDDEN);
float partial_variance = 0.0f;
for (uint i = 0; i < HIDDEN / 32; ++i) {
    values[i] -= mean;
    partial_variance += values[i] * values[i];
}
const float variance = simd_sum(partial_variance) / float(HIDDEN);
const float inverse_std = metal::precise::rsqrt(variance + epsilon[0]);
for (uint i = 0; i < HIDDEN / 32; ++i) {
    const uint column = lane + i * 32;
    const float normalized_value = values[i] * inverse_std;
    normalized[base + column] = weight[column] * normalized_value + bias[column];
}
