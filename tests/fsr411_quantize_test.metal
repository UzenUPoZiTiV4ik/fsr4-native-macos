// SPDX-License-Identifier: MIT
// Compile the production helper itself, with its production feature defaults.
// Do not copy the vector arithmetic or force it on from the test.
#include "../src/fsr411_mpp.metal"

#if !FSR411_VECTOR_QUANTIZE
#error "The vector quantize regression requires the production vector path"
#endif

struct QuantizeCase {
    int4 accumulator;
    int4 bias;
    int4 residual;
    fsr411::Epilogue q;
};

kernel void fsr411_quantize_test(device const QuantizeCase *cases [[buffer(0)]],
                                device uint *results [[buffer(1)]],
                                constant uint &count [[buffer(2)]],
                                uint index [[thread_position_in_grid]]) {
    if (index >= count) return;
    const QuantizeCase c = cases[index];
    results[index] = fsr411_gpu::quantize(c.accumulator, c.bias, c.residual, c.q);
}

// Both template variants must also preserve arbitrary epilogues through the
// generic fallback. CNB intermediates have no residual input in either path.
kernel void fsr411_quantize_cnb_linear_test(device const QuantizeCase *cases [[buffer(0)]],
                                           device uint *results [[buffer(1)]],
                                           constant uint &count [[buffer(2)]],
                                           uint index [[thread_position_in_grid]]) {
    if (index >= count) return;
    const QuantizeCase c = cases[index];
    results[index] = fsr411_gpu::quantize_cnb_intermediate<false>(c.accumulator, c.bias, c.q);
}

kernel void fsr411_quantize_cnb_relu_test(device const QuantizeCase *cases [[buffer(0)]],
                                         device uint *results [[buffer(1)]],
                                         constant uint &count [[buffer(2)]],
                                         uint index [[thread_position_in_grid]]) {
    if (index >= count) return;
    const QuantizeCase c = cases[index];
    results[index] = fsr411_gpu::quantize_cnb_intermediate<true>(c.accumulator, c.bias, c.q);
}
