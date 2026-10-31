// SPDX-License-Identifier: MIT
// Compile with FSR411_SPECIALIZE=1. Inspect the actual production Winograd
// convolution BEFORE bias, INT8 rounding or saturation can hide an error.
#include "fsr411_simd.metal"

kernel void fsr411_simd_winograd_raw_test(
    device const uchar *scratch [[buffer(0)]], device const half *wg [[buffer(1)]],
    constant fsr411::PassParams &raw [[buffer(2)]], device int *out [[buffer(3)]],
    uint2 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    FSR411_LOCAL_PARAMS;
    threadgroup half tile[34 * 6 * 16];
    const uint x0 = group.x * 32u, y0 = group.y * 4u;
    fsr411_simd::load_window<16, 34, 6, 128>(tile, scratch, p.input, int(x0) - 1, int(y0) - 1, tid);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint sx = (sg & 1u) * 16u, sy = (sg >> 1u) * 2u;
    if (y0 + sy >= p.input.height) return;
    const ushort2 lo = fsr411_simd::lane_origin(lane);
    const fsr411_simd::WinogradFragments values = fsr411_simd::winograd3x3<16, 34>(tile, wg + size_t(p.weights[0]) * 2u, 0u, sx, sy, lo);
#pragma unroll
    for (uint m = 0; m < 4u; ++m) {
        const uint x = x0 + sx + 2u * lo.y + (m & 1u), y = y0 + sy + (m >> 1u);
        if (x >= p.input.width || y >= p.input.height) continue;
        const float4 value = m == 0u ? values.p00 : m == 1u ? values.p01 : m == 2u ? values.p10 : values.p11;
        const uint index = (y * p.input.width + x) * 16u + lo.x;
        *(device int2 *)(out + index) = int2(value.xy);
        *(device int2 *)(out + index + 8u) = int2(value.zw);
    }
}
