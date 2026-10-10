// One SIMD group per 384-wide row. Shift by an actual value before reduction
// to preserve small differences in nearly constant rows with epsilon=1e-12.
const uint lane = thread_index_in_simdgroup;
const uint row = threadgroup_position_in_grid.y;
if (row >= ROWS) return;
const size_t base = size_t(row) * 384;
const float anchor = residual[base] + branch[base];
float values[12];
float sum = 0.0f;
for (uint i = 0; i < 12; ++i) {
    const uint col = lane + i * 32;
    values[i] = (residual[base+col] + branch[base+col]) - anchor;
    sum += values[i];
}
const float mean = simd_sum(sum) / 384.0f;
float squares = 0.0f;
for (uint i = 0; i < 12; ++i) {
    values[i] -= mean;
    squares += values[i] * values[i];
}
const float variance = simd_sum(squares) / 384.0f;
const float inverse = metal::precise::rsqrt(variance + epsilon[0]);
for (uint i = 0; i < 12; ++i) {
    const uint col = lane + i * 32;
    normalized[base+col] = weight[col] * (values[i] * inverse) + bias[col];
}
