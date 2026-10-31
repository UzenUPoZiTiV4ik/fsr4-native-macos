// SPDX-License-Identifier: MIT
// Compile the actual Apple7+ production int2 epilogue, not a copied formula.
#include "../src/fsr411_simd.metal"
struct QuantizeCase {
    int4 accumulator;
    int4 bias;
    int4 residual;
    fsr411::Epilogue q;
};
kernel void fsr411_simd_quantize_test(device const QuantizeCase *cases [[buffer(0)]],
                                     device uint *results [[buffer(1)]],
                                     constant uint &count [[buffer(2)]],
                                     uint index [[thread_position_in_grid]]) {
    if (index >= count) return;
    const QuantizeCase c = cases[index];
    const int2 lo = fsr411_simd::finish2(c.accumulator.xy,c.bias.xy,c.residual.xy,c.q);
    const int2 hi = fsr411_simd::finish2(c.accumulator.zw,c.bias.zw,c.residual.zw,c.q);
    results[index] = uint(fsr411_simd::pack2(lo.x,lo.y)) | (uint(fsr411_simd::pack2(hi.x,hi.y)) << 16u);
}
