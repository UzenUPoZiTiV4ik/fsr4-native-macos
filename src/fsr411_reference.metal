// SPDX-License-Identifier: MIT
// Portable FSR 4.1.1 integer reference. One spatial pixel per thread, with
// ordinary scalar loops and no Metal Performance Primitives dependency.
// Normal and Ultra Performance select their own verified packed model blobs.
// The arithmetic is exact INT8/INT32, including the provider's residual tails.
// Compile with -std=metal3.0 -I src.
#include <metal_stdlib>
#include "fsr411_model.h"

using namespace metal;

namespace fsr411_reference {

inline int weight(device const uchar *weights, uint offset, uint index) {
    return int(((device const char *)weights)[offset + index]);
}

inline int bias(device const uchar *weights, uint offset, uint channel) {
    return offset == fsr411::kNoOffset ? 0 : ((device const int *)(weights + offset))[channel];
}

inline int pixel(device const uchar *scratch, fsr411::Tensor t, int x, int y, uint channel) {
    if (x < 0 || y < 0 || uint(x) >= t.width || uint(y) >= t.height) return 0;
    return int(((device const char *)scratch)[fsr411::tensor_address(t, uint(x), uint(y), channel)]);
}

inline void write(device uchar *scratch, fsr411::Tensor t, uint x, uint y, uint channel, int value) {
    ((device char *)scratch)[fsr411::tensor_address(t, x, y, channel)] = char(value);
}

inline int conv3x3(device const uchar *scratch, device const uchar *weights,
                   constant fsr411::PassParams &p, uint2 pos, uint oc) {
    const uint bank = oc / 16u;
    int acc = 0;
#pragma clang loop unroll(disable)
    for (uint tap = 0; tap < 9; ++tap) {
        const int ix = int(pos.x) + int(tap % 3u) - 1;
        const int iy = int(pos.y) + int(tap / 3u) - 1;
#pragma clang loop unroll(disable)
        for (uint ic = 0; ic < 16; ++ic) {
            const int activation = pixel(scratch, p.input, ix, iy, bank * 16u + ic);
            const int coefficient = weight(weights, p.weights[0], bank * 9u * 16u * 16u + (tap * 16u + ic) * 16u + oc % 16u);
            acc += activation * coefficient;
        }
    }
    return fsr411::finish(acc, bias(weights, p.biases[0], oc), 0, p.q[0]);
}

inline void fnb_block(device const uchar *scratch, device const uchar *weights,
                      constant fsr411::PassParams &p, uint2 pos, thread char *mid) {
    const uint nf = p.input.channels;
    char first[64], expanded[128];
#pragma clang loop unroll(disable)
    for (uint oc = 0; oc < nf / 2u; ++oc)
        first[oc] = char(conv3x3(scratch, weights, p, pos, oc));
#pragma clang loop unroll(disable)
    for (uint ic = nf / 2u; ic < nf; ++ic)
        first[ic] = char(pixel(scratch, p.input, int(pos.x), int(pos.y), ic));
#pragma clang loop unroll(disable)
    for (uint oc = 0; oc < 2u * nf; ++oc) {
        int acc = 0;
#pragma clang loop unroll(disable)
        for (uint ic = 0; ic < nf; ++ic)
            acc += int(first[ic]) * weight(weights, p.weights[1], ic * 2u * nf + oc);
        expanded[oc] = char(fsr411::finish(acc, bias(weights, p.biases[1], oc), 0, p.q[1]));
    }
#pragma clang loop unroll(disable)
    for (uint oc = 0; oc < nf; ++oc) {
        int acc = 0;
#pragma clang loop unroll(disable)
        for (uint ic = 0; ic < 2u * nf; ++ic)
            acc += int(expanded[ic]) * weight(weights, p.weights[2], ic * nf + oc);
        const int residual = pixel(scratch, p.input, int(pos.x), int(pos.y), oc);
        mid[oc] = char(fsr411::finish(acc, bias(weights, p.biases[2], oc), residual, p.q[oc < nf / 2u ? 2 : 3]));
    }
}

} // namespace fsr411_reference

// Bindings are identical to fsr411_mpp.metal: scratch, weights, PassParams.
// Dispatch ceil(logicalWidth/32) x logicalHeight groups, 32x1 threads.
// CNB/DS/FNB dispatch in output space; CT dispatches in input space.
kernel void fsr411_ref_cnb(device uchar *scratch [[buffer(0)]],
                           device const uchar *weights [[buffer(1)]],
                           constant fsr411::PassParams &p [[buffer(2)]],
                           uint2 pos [[thread_position_in_grid]]) {
    if (pos.x >= p.output.width || pos.y >= p.output.height) return;
    char first[16], expanded[32];
#pragma clang loop unroll(disable)
    for (uint oc = 0; oc < 16; ++oc)
        first[oc] = char(fsr411_reference::conv3x3(scratch, weights, p, pos, oc));
#pragma clang loop unroll(disable)
    for (uint oc = 0; oc < 32; ++oc) {
        int acc = 0;
#pragma clang loop unroll(disable)
        for (uint ic = 0; ic < 16; ++ic)
            acc += int(first[ic]) * fsr411_reference::weight(weights, p.weights[1], ic * 32u + oc);
        expanded[oc] = char(fsr411::finish(acc, fsr411_reference::bias(weights, p.biases[1], oc), 0, p.q[1]));
    }
#pragma clang loop unroll(disable)
    for (uint oc = 0; oc < 16; ++oc) {
        int acc = 0;
#pragma clang loop unroll(disable)
        for (uint ic = 0; ic < 32; ++ic)
            acc += int(expanded[ic]) * fsr411_reference::weight(weights, p.weights[2], ic * 16u + oc);
        const int residual = fsr411_reference::pixel(scratch, p.input, int(pos.x), int(pos.y), oc);
        const int value = fsr411::finish(acc, fsr411_reference::bias(weights, p.biases[2], oc), residual, p.q[oc < 8u ? 2 : 3]);
        fsr411_reference::write(scratch, p.output, pos.x, pos.y, oc, value);
    }
}

kernel void fsr411_ref_ds(device uchar *scratch [[buffer(0)]],
                          device const uchar *weights [[buffer(1)]],
                          constant fsr411::PassParams &p [[buffer(2)]],
                          uint2 pos [[thread_position_in_grid]]) {
    if (pos.x >= p.output.width || pos.y >= p.output.height) return;
    const uint cin = p.input.channels, cout = p.output.channels;
#pragma clang loop unroll(disable)
    for (uint oc = 0; oc < cout; ++oc) {
        int acc = 0;
#pragma clang loop unroll(disable)
        for (uint tap = 0; tap < 4; ++tap) {
            const int ix = int(pos.x * 2u + tap % 2u), iy = int(pos.y * 2u + tap / 2u);
#pragma clang loop unroll(disable)
            for (uint ic = 0; ic < cin; ++ic)
                acc += fsr411_reference::pixel(scratch, p.input, ix, iy, ic) *
                       fsr411_reference::weight(weights, p.weights[0], (tap * cin + ic) * cout + oc);
        }
        const int value = fsr411::finish(acc, fsr411_reference::bias(weights, p.biases[0], oc), 0, p.q[oc < cout / 2u ? 4 : 5]);
        fsr411_reference::write(scratch, p.output, pos.x, pos.y, oc, value);
    }
}

kernel void fsr411_ref_fnb(device uchar *scratch [[buffer(0)]],
                           device const uchar *weights [[buffer(1)]],
                           constant fsr411::PassParams &p [[buffer(2)]],
                           uint2 pos [[thread_position_in_grid]]) {
    if (pos.x >= p.output.width || pos.y >= p.output.height) return;
    char mid[64];
    fsr411_reference::fnb_block(scratch, weights, p, pos, mid);
#pragma clang loop unroll(disable)
    for (uint oc = 0; oc < p.output.channels; ++oc)
        fsr411_reference::write(scratch, p.output, pos.x, pos.y, oc, int(mid[oc]));
}

kernel void fsr411_ref_ct(device uchar *scratch [[buffer(0)]],
                          device const uchar *weights [[buffer(1)]],
                          constant fsr411::PassParams &p [[buffer(2)]],
                          uint2 pos [[thread_position_in_grid]]) {
    if (pos.x >= p.input.width || pos.y >= p.input.height) return;
    char mid[64];
    fsr411_reference::fnb_block(scratch, weights, p, pos, mid);
    const uint cin = p.input.channels, cout = p.output.channels;
#pragma clang loop unroll(disable)
    for (uint tap = 0; tap < 4; ++tap) {
        const uint x = pos.x * 2u + tap % 2u, y = pos.y * 2u + tap / 2u;
        if (x >= p.output.width || y >= p.output.height) continue;
#pragma clang loop unroll(disable)
        for (uint oc = 0; oc < cout; ++oc) {
            int acc = 0;
#pragma clang loop unroll(disable)
            for (uint ic = 0; ic < cin; ++ic)
                acc += int(mid[ic]) * fsr411_reference::weight(weights, p.weights[3], ic * 4u * cout + tap * cout + oc);
            const int residual = fsr411_reference::pixel(scratch, p.skip, int(x), int(y), oc);
            const int value = fsr411::finish(acc, fsr411_reference::bias(weights, p.biases[3], oc), residual, p.q[oc < cout / 2u ? 4 : 5]);
            fsr411_reference::write(scratch, p.output, x, y, oc, value);
        }
    }
}
