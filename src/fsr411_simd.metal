// SPDX-License-Identifier: MIT
// Apple7+ (M1–M4) FSR 4.1.1 integer ConvNext blocks on simdgroup matrices.
//
// INT8 activations and weights are exact in half. 8x8 half x half products
// accumulate in float: every model accumulator is bounded by
// 128 * sum|w| < 2^24 (474,880 for the pinned Normal/UP, Exact/Lossy models),
// so the float sum is the exact integer in any association order. Biases,
// residuals and the INT32 epilogue (fsr411::finish) are applied in integers,
// so the result is byte-identical to the scalar backend.
//
// The epilogue addresses the two elements each lane owns in an 8x8 matrix.
// That ownership is the Apple7+ layout used by lane_origin(); the host runs
// fsr411_simd_probe once per device and selects this backend only if both
// the layout and the exact float accumulation are confirmed.
//
// Bindings: 0 scratch, 1 packed INT8 weights + INT32 biases, 2 PassParams,
// 3 the same weights blob widened element-wise to half (same indices).
// Compile with -std=metal3.0 -I src; FSR411_SPECIALIZE=1 for per-pass
// function constants.
#include <metal_stdlib>
#include "fsr411_model.h"

using namespace metal;

namespace fsr411_simd {

typedef simdgroup_matrix<half, 8, 8> hmat;
typedef simdgroup_matrix<float, 8, 8> fmat;

// (column, row) of the first of two adjacent elements owned by this lane.
__attribute__((always_inline)) inline ushort2 lane_origin(ushort lane) {
    const ushort q = lane / 4u;
    return ushort2((q & 2u) * 2u + (lane % 2u) * 2u, (q & 4u) + (lane / 2u) % 4u);
}

__attribute__((always_inline)) inline int2 load_bias2(device const uchar *weights, uint offset, uint channel) {
    return offset == fsr411::kNoOffset ? int2(0) : *(device const int2 *)(weights + offset + channel * 4u);
}

__attribute__((always_inline)) inline ushort pack2(int a, int b) {
    return ushort(uchar(a & 255)) | ushort(uchar(b & 255) << 8);
}

__attribute__((always_inline)) inline fmat zero() {
    return make_filled_simdgroup_matrix<float, 8, 8>(0.f);
}

// Requantize one accumulator fragment into an INT8-valued half fragment.
__attribute__((always_inline)) inline hmat requant(thread fmat &acc, int2 bias, fsr411::Epilogue q) {
    hmat r;
    r.thread_elements()[0] = half(fsr411::finish(int(acc.thread_elements()[0]), bias.x, 0, q));
    r.thread_elements()[1] = half(fsr411::finish(int(acc.thread_elements()[1]), bias.y, 0, q));
    return r;
}

// Copy a W x H pixel window of C channels (multiple of 16) as half, zero
// outside the logical tensor. Layout [row][column][channel].
template <uint C, uint W, uint H, uint THREADS>
__attribute__((always_inline)) inline void load_window(threadgroup half *tile, device const uchar *scratch,
                                                       fsr411::Tensor t, int x0, int y0, uint tid) {
    constexpr uint UNITS = C / 16u;
    for (uint i = tid; i < W * H * UNITS; i += THREADS) {
        const uint px = i / UNITS, ch = (i % UNITS) * 16u;
        const int x = x0 + int(px % W), y = y0 + int(px / W);
        uint4 v(0);
        if (x >= 0 && y >= 0 && uint(x) < t.width && uint(y) < t.height)
            v = *(device const uint4 *)(scratch + fsr411::tensor_address(t, uint(x), uint(y), ch));
        threadgroup half4 *dst = (threadgroup half4 *)(tile + px * C + ch);
        dst[0] = half4(as_type<char4>(v.x));
        dst[1] = half4(as_type<char4>(v.y));
        dst[2] = half4(as_type<char4>(v.z));
        dst[3] = half4(as_type<char4>(v.w));
    }
}

// CNB (NF=16, one bank), FNB and CT blocks. Each SIMD group computes 8*MB
// consecutive pixels of one row; the threadgroup covers 8*MB x SG pixels and
// shares one input window with a one-pixel halo. CNB/FNB dispatch in output
// space, CT in input space, exactly like the scalar kernels.
template <uint NF, uint BANKS, uint MB, uint SG, bool UPSAMPLE>
__attribute__((always_inline)) inline void block(device uchar *scratch, device const uchar *weights,
                                                 device const half *wh, FSR411_PARAMS_ADDRESS fsr411::PassParams &p,
                                                 threadgroup half *tile, uint2 group, uint tid, ushort sg, ushort lane) {
    static_assert(NF % 16u == 0 && 16u * BANKS <= NF, "conv banks must fit the block width");
    constexpr uint TW = 8u * MB, WW = TW + 2u, WH = SG + 2u;
    constexpr uint KB = NF / 8u, N1 = 2u * NF;
    const int x0 = int(group.x * TW), y0 = int(group.y * SG);
    load_window<NF, WW, WH, SG * 32u>(tile, scratch, p.input, x0 - 1, y0 - 1, tid);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint y = uint(y0) + sg;
    if (y >= (UPSAMPLE ? p.input.height : p.output.height)) return;
    const ushort2 lo = lane_origin(lane);
    threadgroup const half *centre = tile + ((sg + 1u) * WW + 1u) * NF;

    // Grouped 3x3 convolutions (16 -> 16 per bank), then the bypass channels.
    hmat a1[MB][KB];
#pragma unroll
    for (uint b = 0; b < BANKS; ++b) {
        fmat acc[MB][2];
#pragma unroll
        for (uint m = 0; m < MB; ++m) acc[m][0] = acc[m][1] = zero();
        device const half *w0 = wh + p.weights[0] + b * 2304u;
#pragma unroll
        for (uint tap = 0; tap < 9u; ++tap) {
            const uint ky = tap / 3u, kx = tap % 3u;
#pragma unroll
            for (uint kb = 0; kb < 2u; ++kb) {
                hmat lower, upper;
                simdgroup_load(lower, w0 + (tap * 16u + kb * 8u) * 16u, 16);
                simdgroup_load(upper, w0 + (tap * 16u + kb * 8u) * 16u + 8u, 16);
#pragma unroll
                for (uint m = 0; m < MB; ++m) {
                    hmat a;
                    simdgroup_load(a, tile + ((sg + ky) * WW + m * 8u + kx) * NF + b * 16u + kb * 8u, NF);
                    simdgroup_multiply_accumulate(acc[m][0], a, lower, acc[m][0]);
                    simdgroup_multiply_accumulate(acc[m][1], a, upper, acc[m][1]);
                }
            }
        }
#pragma unroll
        for (uint n = 0; n < 2u; ++n) {
            const int2 bias = load_bias2(weights, p.biases[0], b * 16u + n * 8u + lo.x);
#pragma unroll
            for (uint m = 0; m < MB; ++m) a1[m][2u * b + n] = requant(acc[m][n], bias, p.q[0]);
        }
    }
#pragma unroll
    for (uint cb = 2u * BANKS; cb < KB; ++cb)
#pragma unroll
        for (uint m = 0; m < MB; ++m) simdgroup_load(a1[m][cb], centre + m * 8u * NF + cb * 8u, NF);

    // ReLU expansion (NF -> 2NF), streamed one 8-channel block at a time
    // straight into the projection (2NF -> NF) accumulators.
    fmat out[MB][KB];
#pragma unroll
    for (uint m = 0; m < MB; ++m)
#pragma unroll
        for (uint ob = 0; ob < KB; ++ob) out[m][ob] = zero();
    device const half *w1 = wh + p.weights[1], *w2 = wh + p.weights[2];
#pragma unroll
    for (uint nb = 0; nb < N1 / 8u; ++nb) {
        hmat w[KB];
#pragma unroll
        for (uint kb = 0; kb < KB; ++kb) simdgroup_load(w[kb], w1 + kb * 8u * N1 + nb * 8u, N1);
        const int2 bias = load_bias2(weights, p.biases[1], nb * 8u + lo.x);
        hmat a2[MB];
#pragma unroll
        for (uint m = 0; m < MB; ++m) {
            fmat e = zero();
#pragma unroll
            for (uint kb = 0; kb < KB; ++kb) simdgroup_multiply_accumulate(e, a1[m][kb], w[kb], e);
            a2[m] = requant(e, bias, p.q[1]);
        }
#pragma unroll
        for (uint ob = 0; ob < KB; ++ob) {
            hmat v;
            simdgroup_load(v, w2 + nb * 8u * NF + ob * 8u, NF);
#pragma unroll
            for (uint m = 0; m < MB; ++m) simdgroup_multiply_accumulate(out[m][ob], a2[m], v, out[m][ob]);
        }
    }

    // Residual contraction. FNB/CNB store it; CT keeps it as the transposed
    // convolution's input.
    // Dead (and eliminated) for FNB/CNB; kept sized so the code stays C++14.
    hmat a3[MB][KB];
#pragma unroll
    for (uint ob = 0; ob < KB; ++ob) {
        const uint ch = ob * 8u + lo.x;
        const int2 bias = load_bias2(weights, p.biases[2], ch);
        const fsr411::Epilogue q = ch < NF / 2u ? p.q[2] : p.q[3];
#pragma unroll
        for (uint m = 0; m < MB; ++m) {
            const uint px = m * 8u + lo.y;
            threadgroup const half *r = centre + px * NF + ch;
            const int v0 = fsr411::finish(int(out[m][ob].thread_elements()[0]), bias.x, int(r[0]), q);
            const int v1 = fsr411::finish(int(out[m][ob].thread_elements()[1]), bias.y, int(r[1]), q);
            if (UPSAMPLE) {
                a3[m][ob].thread_elements()[0] = half(v0);
                a3[m][ob].thread_elements()[1] = half(v1);
            } else {
                const uint x = uint(x0) + px;
                if (x < p.output.width)
                    *(device ushort *)(scratch + fsr411::tensor_address(p.output, x, y, ch)) = pack2(v0, v1);
            }
        }
    }
    if (!UPSAMPLE) return;

    // Transposed 2x2 convolution: columns are [tap][output channel].
    constexpr uint OUT = NF / 2u, N3 = 4u * OUT;
    device const half *w3 = wh + p.weights[3];
#pragma unroll
    for (uint nb = 0; nb < N3 / 8u; ++nb) {
        hmat w[KB];
#pragma unroll
        for (uint kb = 0; kb < KB; ++kb) simdgroup_load(w[kb], w3 + kb * 8u * N3 + nb * 8u, N3);
        const uint n = nb * 8u + lo.x, tap = n / OUT, oc = n % OUT;
        const int2 bias = load_bias2(weights, p.biases[3], oc);
        const fsr411::Epilogue q = oc < OUT / 2u ? p.q[4] : p.q[5];
#pragma unroll
        for (uint m = 0; m < MB; ++m) {
            fmat t = zero();
#pragma unroll
            for (uint kb = 0; kb < KB; ++kb) simdgroup_multiply_accumulate(t, a3[m][kb], w[kb], t);
            const uint ix = uint(x0) + m * 8u + lo.y;
            const uint ox = ix * 2u + (tap & 1u), oy = y * 2u + (tap >> 1u);
            if (ix >= p.input.width || ox >= p.output.width || oy >= p.output.height) continue;
            int2 residual(0);
            if (ox < p.skip.width && oy < p.skip.height)
                residual = int2(as_type<char2>(*(device const ushort *)(scratch + fsr411::tensor_address(p.skip, ox, oy, oc))));
            const int v0 = fsr411::finish(int(t.thread_elements()[0]), bias.x, residual.x, q);
            const int v1 = fsr411::finish(int(t.thread_elements()[1]), bias.y, residual.y, q);
            *(device ushort *)(scratch + fsr411::tensor_address(p.output, ox, oy, oc)) = pack2(v0, v1);
        }
    }
}

// Stride-2 2x2 downsample CIN -> 2*CIN. K = [tap][input channel].
template <uint CIN, uint MB, uint SG>
__attribute__((always_inline)) inline void downsample(device uchar *scratch, device const uchar *weights,
                                                      device const half *wh, FSR411_PARAMS_ADDRESS fsr411::PassParams &p,
                                                      threadgroup half *tile, uint2 group, uint tid, ushort sg, ushort lane) {
    constexpr uint COUT = 2u * CIN, TW = 8u * MB, WW = 2u * TW, WH = 2u * SG, NB = COUT / 8u;
    const int x0 = int(group.x * TW), y0 = int(group.y * SG);
    load_window<CIN, WW, WH, SG * 32u>(tile, scratch, p.input, 2 * x0, 2 * y0, tid);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint y = uint(y0) + sg;
    if (y >= p.output.height) return;
    const ushort2 lo = lane_origin(lane);
    fmat acc[MB][NB];
#pragma unroll
    for (uint m = 0; m < MB; ++m)
#pragma unroll
        for (uint nb = 0; nb < NB; ++nb) acc[m][nb] = zero();
    device const half *w0 = wh + p.weights[0];
#pragma unroll
    for (uint tap = 0; tap < 4u; ++tap) {
        const uint ty = tap / 2u, tx = tap % 2u;
#pragma unroll
        for (uint kb = 0; kb < CIN / 8u; ++kb) {
            hmat a[MB];
#pragma unroll
            for (uint m = 0; m < MB; ++m)
                simdgroup_load(a[m], tile + ((2u * sg + ty) * WW + 16u * m + tx) * CIN + kb * 8u, 2u * CIN);
#pragma unroll
            for (uint nb = 0; nb < NB; ++nb) {
                hmat w;
                simdgroup_load(w, w0 + (tap * CIN + kb * 8u) * COUT + nb * 8u, COUT);
#pragma unroll
                for (uint m = 0; m < MB; ++m) simdgroup_multiply_accumulate(acc[m][nb], a[m], w, acc[m][nb]);
            }
        }
    }
#pragma unroll
    for (uint nb = 0; nb < NB; ++nb) {
        const uint ch = nb * 8u + lo.x;
        const int2 bias = load_bias2(weights, p.biases[0], ch);
        const fsr411::Epilogue q = ch < COUT / 2u ? p.q[4] : p.q[5];
#pragma unroll
        for (uint m = 0; m < MB; ++m) {
            const uint x = uint(x0) + m * 8u + lo.y;
            if (x >= p.output.width) continue;
            const int v0 = fsr411::finish(int(acc[m][nb].thread_elements()[0]), bias.x, 0, q);
            const int v1 = fsr411::finish(int(acc[m][nb].thread_elements()[1]), bias.y, 0, q);
            *(device ushort *)(scratch + fsr411::tensor_address(p.output, x, y, ch)) = pack2(v0, v1);
        }
    }
}

} // namespace fsr411_simd

#define FSR411_SIMD_ARGS \
    device uchar *scratch [[buffer(0)]], device const uchar *weights [[buffer(1)]], \
    constant fsr411::PassParams &raw [[buffer(2)]], device const half *wh [[buffer(3)]], \
    uint2 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], \
    ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]

// Tile = 8*MB x SG pixels, 32*SG threads. Manifest "simd" entries must match.
#define FSR411_SIMD_BLOCK(NAME, NF, BANKS, MB, SG, UPSAMPLE) \
kernel void NAME(FSR411_SIMD_ARGS) { \
    FSR411_LOCAL_PARAMS; \
    threadgroup half tile[(8 * (MB) + 2) * ((SG) + 2) * (NF)]; \
    fsr411_simd::block<NF, BANKS, MB, SG, UPSAMPLE>(scratch, weights, wh, p, tile, group, tid, sg, lane); \
}
FSR411_SIMD_BLOCK(fsr411_simd_cnb, 16, 1, 4, 4, false)
FSR411_SIMD_BLOCK(fsr411_simd_fnb32, 32, 1, 4, 4, false)
FSR411_SIMD_BLOCK(fsr411_simd_fnb64, 64, 2, 2, 4, false)
FSR411_SIMD_BLOCK(fsr411_simd_ct32, 32, 1, 4, 4, true)
FSR411_SIMD_BLOCK(fsr411_simd_ct64, 64, 2, 2, 4, true)
#undef FSR411_SIMD_BLOCK

#define FSR411_SIMD_DS(NAME, CIN, MB, SG) \
kernel void NAME(FSR411_SIMD_ARGS) { \
    FSR411_LOCAL_PARAMS; \
    threadgroup half tile[16 * (MB) * 2 * (SG) * (CIN)]; \
    fsr411_simd::downsample<CIN, MB, SG>(scratch, weights, wh, p, tile, group, tid, sg, lane); \
}
FSR411_SIMD_DS(fsr411_simd_ds16, 16, 4, 4)
FSR411_SIMD_DS(fsr411_simd_ds32, 32, 2, 4)
#undef FSR411_SIMD_DS
#undef FSR411_SIMD_ARGS

// Host capability probe, one SIMD32 group. out[0..63]: C = A*B + A*B as
// integers (exact float accumulation of INT8 extremes); out[64+lane]: 1 if
// this lane owns elements lane_origin() .. +1 for loads and for writes.
kernel void fsr411_simd_probe(device const half *a [[buffer(0)]], device const half *b [[buffer(1)]],
                              device int *out [[buffer(2)]], ushort lane [[thread_index_in_simdgroup]]) {
    using namespace fsr411_simd;
    hmat A, B;
    simdgroup_load(A, a, 8);
    simdgroup_load(B, b, 8);
    fmat C = zero();
    simdgroup_multiply_accumulate(C, A, B, C);
    simdgroup_multiply_accumulate(C, A, B, C);
    threadgroup float result[64];
    threadgroup half iota[64], written[64];
    simdgroup_store(C, result, 8);
    for (uint i = lane; i < 64u; i += 32u) iota[i] = half(i);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const ushort2 lo = lane_origin(lane);
    const uint index = uint(lo.y) * 8u + lo.x;
    hmat I;
    simdgroup_load(I, iota, 8);
    bool ok = uint(I.thread_elements()[0]) == index && uint(I.thread_elements()[1]) == index + 1u;
    hmat W;
    W.thread_elements()[0] = half(index);
    W.thread_elements()[1] = half(index + 1u);
    simdgroup_store(W, written, 8);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = lane; i < 64u; i += 32u) out[i] = int(result[i]);
    ok = ok && uint(written[index]) == index && uint(written[index + 1u]) == index + 1u;
    out[64u + lane] = ok ? 1 : 0;
}
