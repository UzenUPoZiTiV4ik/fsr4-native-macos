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

#ifndef FSR411_FLOAT_EPILOGUE
#define FSR411_FLOAT_EPILOGUE 0
#endif
// fsr411::finish on an exact integer float accumulator, returned as float.
// The host defines FSR411_FLOAT_EPILOGUE only after proving, for every pass,
// |accumulator| + |bias| + (128 << residualShift) < 2^24: every float sum is
// then the exact integer, no 32-bit wrap can occur, scaling by 2^-k is exact,
// rint is round-half-even and floor(x + 0.5) round-half-up. Shifted or
// INT16-narrowed epilogues keep the integer sequence (folded per pass).
__attribute__((always_inline)) inline float finishf(float acc, int bias, int residual, fsr411::Epilogue q) {
#if FSR411_FLOAT_EPILOGUE
    if (q.accumulatorShift == 0u && !(q.flags & fsr411::kNarrowInt16)) {
        float v = acc + float(bias) + float(residual) * float(1u << q.residualShift);
        if (q.flags & fsr411::kReluBeforeRound) v = max(v, 0.f);
        const float s = as_type<float>((127u - q.rightShift) << 23u);
        v = (q.flags & fsr411::kRoundHalfUp) ? floor(v * s + 0.5f) : rint(v * s);
        return clamp(v, -128.f, 127.f);
    }
#endif
    return float(fsr411::finish(int(acc), bias, residual, q));
}

// Requantize one accumulator fragment into an INT8-valued half fragment.
__attribute__((always_inline)) inline hmat requant(thread fmat &acc, int2 bias, fsr411::Epilogue q) {
    hmat r;
    r.thread_elements()[0] = half(finishf(acc.thread_elements()[0], bias.x, 0, q));
    r.thread_elements()[1] = half(finishf(acc.thread_elements()[1], bias.y, 0, q));
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

// Tile-local (column, row) of A-matrix row r in pixel block m. Direct: block m
// is 8 consecutive pixels of row sg. Winograd: row r is 2x2 tile r of the
// SIMD group's two rows and block m = 2*i + j its sub-position (j, i).
template <bool WINO>
__attribute__((always_inline)) inline uint2 pixel(uint m, uint r, uint sg) {
    return WINO ? uint2(2u * r + (m & 1u), 2u * sg + (m >> 1u)) : uint2(m * 8u + r, sg);
}

template <bool WINO> struct Conv3x3;

// Direct grouped 3x3 convolution (16 -> 16 per bank): 9 taps x 2 K blocks.
template <> struct Conv3x3<false> {
    template <uint NF, uint BANKS, uint MB, uint SG>
    static __attribute__((always_inline)) inline void run(thread hmat (&a1)[MB][NF / 8u], threadgroup const half *tile,
                                                          device const half *wh, device const uchar *weights,
                                                          FSR411_PARAMS_ADDRESS fsr411::PassParams &p, ushort sg, ushort2 lo) {
        constexpr uint WW = 8u * MB + 2u;
#pragma clang loop unroll(full)
        for (uint b = 0; b < BANKS; ++b) {
            fmat acc[MB][2];
#pragma clang loop unroll(full)
            for (uint m = 0; m < MB; ++m) acc[m][0] = acc[m][1] = zero();
            device const half *w0 = wh + p.weights[0] + b * 2304u;
#pragma clang loop unroll(full)
            for (uint tap = 0; tap < 9u; ++tap) {
                const uint ky = tap / 3u, kx = tap % 3u;
#pragma clang loop unroll(full)
                for (uint kb = 0; kb < 2u; ++kb) {
                    hmat lower, upper;
                    simdgroup_load(lower, w0 + (tap * 16u + kb * 8u) * 16u, 16);
                    simdgroup_load(upper, w0 + (tap * 16u + kb * 8u) * 16u + 8u, 16);
#pragma clang loop unroll(full)
                    for (uint m = 0; m < MB; ++m) {
                        hmat a;
                        simdgroup_load(a, tile + ((sg + ky) * WW + m * 8u + kx) * NF + b * 16u + kb * 8u, NF);
                        simdgroup_multiply_accumulate(acc[m][0], a, lower, acc[m][0]);
                        simdgroup_multiply_accumulate(acc[m][1], a, upper, acc[m][1]);
                    }
                }
            }
#pragma clang loop unroll(full)
            for (uint n = 0; n < 2u; ++n) {
                const int2 bias = load_bias2(weights, p.biases[0], b * 16u + n * 8u + lo.x);
#pragma clang loop unroll(full)
                for (uint m = 0; m < MB; ++m) a1[m][2u * b + n] = requant(acc[m][n], bias, p.q[0]);
            }
        }
    }
};

// Integer Winograd F(2x2, 3x3): 16 products per 2x2 output instead of 36.
// d' = B^T d B and w' = (2G) g (2G)^T are integers (|d'| <= 512, |w'| <= 2048
// checked by the generator), exact in half; y = A^T (d' . w') A = 4 * conv
// exactly while every partial sum stays below 2^24 (generator bound, host
// probe). Weights are [bank][position 4*xi+nu][input 16][output 16].
template <> struct Conv3x3<true> {
    static constexpr int at(uint i, uint k) { return i == 0u ? (k < 3u ? 1 : 0) : (k == 0u ? 0 : k == 1u ? 1 : -1); }
    template <uint NF, uint BANKS, uint MB, uint SG>
    static __attribute__((always_inline)) inline void run(thread hmat (&a1)[MB][NF / 8u], threadgroup const half *tile,
                                                          device const half *wh, device const uchar *weights,
                                                          FSR411_PARAMS_ADDRESS fsr411::PassParams &p, ushort sg, ushort2 lo) {
        static_assert(MB == 4u, "Winograd blocks are the four 2x2 sub-positions");
        constexpr uint WW = 18u;
#pragma clang loop unroll(full)
        for (uint b = 0; b < BANKS; ++b) {
            // This lane's 4x4 input patch of 2x2 tile lo.y, channels lo.x, lo.x+1.
            threadgroup const half *base = tile + (2u * sg * WW + 2u * lo.y) * NF + b * 16u + lo.x;
            half2 d[2][4][4];
#pragma clang loop unroll(full)
            for (uint kb = 0; kb < 2u; ++kb)
#pragma clang loop unroll(full)
                for (uint r = 0; r < 4u; ++r)
#pragma clang loop unroll(full)
                    for (uint c = 0; c < 4u; ++c)
                        d[kb][r][c] = *(threadgroup const half2 *)(base + (r * WW + c) * NF + kb * 8u);
            fmat y[4][2];
#pragma clang loop unroll(full)
            for (uint m = 0; m < 4u; ++m) y[m][0] = y[m][1] = zero();
            device const half *w0 = wh + p.weights[0] + b * 4096u;
#pragma clang loop unroll(full)
            for (uint xi = 0; xi < 4u; ++xi) {
                half2 e[2][4];  // rows of B^T d
#pragma clang loop unroll(full)
                for (uint kb = 0; kb < 2u; ++kb)
#pragma clang loop unroll(full)
                    for (uint c = 0; c < 4u; ++c)
                        e[kb][c] = xi == 0u ? d[kb][0][c] - d[kb][2][c] : xi == 1u ? d[kb][1][c] + d[kb][2][c]
                                 : xi == 2u ? d[kb][2][c] - d[kb][1][c] : d[kb][1][c] - d[kb][3][c];
#pragma clang loop unroll(full)
                for (uint nu = 0; nu < 4u; ++nu) {
                    fmat M[2] = {zero(), zero()};
#pragma clang loop unroll(full)
                    for (uint kb = 0; kb < 2u; ++kb) {
                        const half2 v = nu == 0u ? e[kb][0] - e[kb][2] : nu == 1u ? e[kb][1] + e[kb][2]
                                      : nu == 2u ? e[kb][2] - e[kb][1] : e[kb][1] - e[kb][3];
                        hmat a;
                        a.thread_elements()[0] = v.x;
                        a.thread_elements()[1] = v.y;
#pragma clang loop unroll(full)
                        for (uint n = 0; n < 2u; ++n) {
                            hmat w;
                            simdgroup_load(w, w0 + ((xi * 4u + nu) * 16u + kb * 8u) * 16u + n * 8u, 16);
                            simdgroup_multiply_accumulate(M[n], a, w, M[n]);
                        }
                    }
#pragma clang loop unroll(full)
                    for (uint m = 0; m < 4u; ++m) {
                        const int c = at(m >> 1u, xi) * at(m & 1u, nu);
                        if (c == 0) continue;
#pragma clang loop unroll(full)
                        for (uint n = 0; n < 2u; ++n)
#pragma clang loop unroll(full)
                            for (uint k = 0; k < 2u; ++k)
                                y[m][n].thread_elements()[k] += float(c) * M[n].thread_elements()[k];
                    }
                }
            }
#pragma clang loop unroll(full)
            for (uint n = 0; n < 2u; ++n) {
                const int2 bias = load_bias2(weights, p.biases[0], b * 16u + n * 8u + lo.x);
#pragma clang loop unroll(full)
                for (uint m = 0; m < 4u; ++m) {
                    // y is an exact multiple of 4; the scale is exact.
                    y[m][n].thread_elements()[0] *= 0.25f;
                    y[m][n].thread_elements()[1] *= 0.25f;
                    a1[m][2u * b + n] = requant(y[m][n], bias, p.q[0]);
                }
            }
        }
    }
};

// CNB (NF=16, one bank), FNB and CT blocks. Direct: each SIMD group computes
// 8*MB consecutive pixels of one row, the threadgroup 8*MB x SG pixels.
// Winograd (MB=4): each SIMD group computes 8 2x2 tiles (16 x 2 pixels), the
// threadgroup 16 x 2*SG. One shared input window with a one-pixel halo.
// CNB/FNB dispatch in output space, CT in input space, like the scalar kernels.
template <uint NF, uint BANKS, uint MB, uint SG, bool UPSAMPLE, bool WINO = false>
__attribute__((always_inline)) inline void block(device uchar *scratch, device const uchar *weights,
                                                 device const half *wh, FSR411_PARAMS_ADDRESS fsr411::PassParams &p,
                                                 threadgroup half *tile, uint2 group, uint tid, ushort sg, ushort lane) {
    static_assert(NF % 16u == 0 && 16u * BANKS <= NF, "conv banks must fit the block width");
    constexpr uint TW = WINO ? 16u : 8u * MB, TH = WINO ? 2u * SG : SG, WW = TW + 2u, WH = TH + 2u;
    constexpr uint KB = NF / 8u, N1 = 2u * NF, ROW = WINO ? 2u : 1u;
    const int x0 = int(group.x * TW), y0 = int(group.y * TH);
    load_window<NF, WW, WH, SG * 32u>(tile, scratch, p.input, x0 - 1, y0 - 1, tid);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Uniform per SIMD group; Winograd checks its second row per element.
    const uint rows = UPSAMPLE ? p.input.height : p.output.height;
    if (uint(y0) + pixel<WINO>(0u, 0u, sg).y >= rows) return;
    const ushort2 lo = lane_origin(lane);
    // Centre pixel of A-matrix row r in block m inside the haloed window.
#define FSR411_WINDOW(m, r) (tile + ((pixel<WINO>(m, r, sg).y + 1u) * WW + pixel<WINO>(m, r, sg).x + 1u) * NF)

    hmat a1[MB][KB];
    Conv3x3<WINO>::template run<NF, BANKS, MB, SG>(a1, tile, wh, weights, p, sg, lo);
#pragma clang loop unroll(full)
    for (uint cb = 2u * BANKS; cb < KB; ++cb)
#pragma clang loop unroll(full)
        for (uint m = 0; m < MB; ++m) simdgroup_load(a1[m][cb], FSR411_WINDOW(m, 0u) + cb * 8u, ROW * NF);

    // ReLU expansion (NF -> 2NF), streamed one 8-channel block at a time
    // straight into the projection (2NF -> NF) accumulators.
    fmat out[MB][KB];
#pragma clang loop unroll(full)
    for (uint m = 0; m < MB; ++m)
#pragma clang loop unroll(full)
        for (uint ob = 0; ob < KB; ++ob) out[m][ob] = zero();
    device const half *w1 = wh + p.weights[1], *w2 = wh + p.weights[2];
#pragma clang loop unroll(full)
    for (uint nb = 0; nb < N1 / 8u; ++nb) {
        hmat w[KB];
#pragma clang loop unroll(full)
        for (uint kb = 0; kb < KB; ++kb) simdgroup_load(w[kb], w1 + kb * 8u * N1 + nb * 8u, N1);
        const int2 bias = load_bias2(weights, p.biases[1], nb * 8u + lo.x);
        hmat a2[MB];
#pragma clang loop unroll(full)
        for (uint m = 0; m < MB; ++m) {
            fmat e = zero();
#pragma clang loop unroll(full)
            for (uint kb = 0; kb < KB; ++kb) simdgroup_multiply_accumulate(e, a1[m][kb], w[kb], e);
            a2[m] = requant(e, bias, p.q[1]);
        }
#pragma clang loop unroll(full)
        for (uint ob = 0; ob < KB; ++ob) {
            hmat v;
            simdgroup_load(v, w2 + nb * 8u * NF + ob * 8u, NF);
#pragma clang loop unroll(full)
            for (uint m = 0; m < MB; ++m) simdgroup_multiply_accumulate(out[m][ob], a2[m], v, out[m][ob]);
        }
    }

    // Residual contraction. FNB/CNB store it; CT keeps it as the transposed
    // convolution's input.
    // Dead (and eliminated) for FNB/CNB; kept sized so the code stays C++14.
    hmat a3[MB][KB];
#pragma clang loop unroll(full)
    for (uint ob = 0; ob < KB; ++ob) {
        const uint ch = ob * 8u + lo.x;
        const int2 bias = load_bias2(weights, p.biases[2], ch);
        const fsr411::Epilogue q2 = ch < NF / 2u ? p.q[2] : p.q[3];
#pragma clang loop unroll(full)
        for (uint m = 0; m < MB; ++m) {
            const uint2 q = pixel<WINO>(m, lo.y, sg);
            threadgroup const half *r = FSR411_WINDOW(m, lo.y) + ch;
            const int v0 = int(finishf(out[m][ob].thread_elements()[0], bias.x, int(r[0]), q2));
            const int v1 = int(finishf(out[m][ob].thread_elements()[1], bias.y, int(r[1]), q2));
            if (UPSAMPLE) {
                a3[m][ob].thread_elements()[0] = half(v0);
                a3[m][ob].thread_elements()[1] = half(v1);
            } else {
                const uint x = uint(x0) + q.x, y = uint(y0) + q.y;
                if (x < p.output.width && y < p.output.height)
                    *(device ushort *)(scratch + fsr411::tensor_address(p.output, x, y, ch)) = pack2(v0, v1);
            }
        }
    }
    if (!UPSAMPLE) return;

    // Transposed 2x2 convolution: columns are [tap][output channel].
    constexpr uint OUT = NF / 2u, N3 = 4u * OUT;
    device const half *w3 = wh + p.weights[3];
#pragma clang loop unroll(full)
    for (uint nb = 0; nb < N3 / 8u; ++nb) {
        hmat w[KB];
#pragma clang loop unroll(full)
        for (uint kb = 0; kb < KB; ++kb) simdgroup_load(w[kb], w3 + kb * 8u * N3 + nb * 8u, N3);
        const uint n = nb * 8u + lo.x, tap = n / OUT, oc = n % OUT;
        const int2 bias = load_bias2(weights, p.biases[3], oc);
        const fsr411::Epilogue q = oc < OUT / 2u ? p.q[4] : p.q[5];
#pragma clang loop unroll(full)
        for (uint m = 0; m < MB; ++m) {
            fmat t = zero();
#pragma clang loop unroll(full)
            for (uint kb = 0; kb < KB; ++kb) simdgroup_multiply_accumulate(t, a3[m][kb], w[kb], t);
            const uint2 pq = pixel<WINO>(m, lo.y, sg);
            const uint ix = uint(x0) + pq.x, iy = uint(y0) + pq.y;
            const uint ox = ix * 2u + (tap & 1u), oy = iy * 2u + (tap >> 1u);
            if (ix >= p.input.width || iy >= p.input.height || ox >= p.output.width || oy >= p.output.height) continue;
            int2 residual(0);
            if (ox < p.skip.width && oy < p.skip.height)
                residual = int2(as_type<char2>(*(device const ushort *)(scratch + fsr411::tensor_address(p.skip, ox, oy, oc))));
            const int v0 = int(finishf(t.thread_elements()[0], bias.x, residual.x, q));
            const int v1 = int(finishf(t.thread_elements()[1], bias.y, residual.y, q));
            *(device ushort *)(scratch + fsr411::tensor_address(p.output, ox, oy, oc)) = pack2(v0, v1);
        }
    }
}
#undef FSR411_WINDOW

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
#pragma clang loop unroll(full)
    for (uint m = 0; m < MB; ++m)
#pragma clang loop unroll(full)
        for (uint nb = 0; nb < NB; ++nb) acc[m][nb] = zero();
    device const half *w0 = wh + p.weights[0];
#pragma clang loop unroll(full)
    for (uint tap = 0; tap < 4u; ++tap) {
        const uint ty = tap / 2u, tx = tap % 2u;
#pragma clang loop unroll(full)
        for (uint kb = 0; kb < CIN / 8u; ++kb) {
            hmat a[MB];
#pragma clang loop unroll(full)
            for (uint m = 0; m < MB; ++m)
                simdgroup_load(a[m], tile + ((2u * sg + ty) * WW + 16u * m + tx) * CIN + kb * 8u, 2u * CIN);
#pragma clang loop unroll(full)
            for (uint nb = 0; nb < NB; ++nb) {
                hmat w;
                simdgroup_load(w, w0 + (tap * CIN + kb * 8u) * COUT + nb * 8u, COUT);
#pragma clang loop unroll(full)
                for (uint m = 0; m < MB; ++m) simdgroup_multiply_accumulate(acc[m][nb], a[m], w, acc[m][nb]);
            }
        }
    }
#pragma clang loop unroll(full)
    for (uint nb = 0; nb < NB; ++nb) {
        const uint ch = nb * 8u + lo.x;
        const int2 bias = load_bias2(weights, p.biases[0], ch);
        const fsr411::Epilogue q = ch < COUT / 2u ? p.q[4] : p.q[5];
#pragma clang loop unroll(full)
        for (uint m = 0; m < MB; ++m) {
            const uint x = uint(x0) + m * 8u + lo.y;
            if (x >= p.output.width) continue;
            const int v0 = int(finishf(acc[m][nb].thread_elements()[0], bias.x, 0, q));
            const int v1 = int(finishf(acc[m][nb].thread_elements()[1], bias.y, 0, q));
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

// Tile = 8*MB x SG pixels, 32*SG threads (Winograd: 16 x 2*SG). Manifest
// "simd" / "winograd" entries must match.
#define FSR411_SIMD_BLOCK(NAME, NF, BANKS, MB, SG, UPSAMPLE, WINO) \
kernel void NAME(FSR411_SIMD_ARGS) { \
    FSR411_LOCAL_PARAMS; \
    threadgroup half tile[((WINO) ? 18 * (2 * (SG) + 2) : (8 * (MB) + 2) * ((SG) + 2)) * (NF)]; \
    fsr411_simd::block<NF, BANKS, MB, SG, UPSAMPLE, WINO>(scratch, weights, wh, p, tile, group, tid, sg, lane); \
}
FSR411_SIMD_BLOCK(fsr411_simd_cnb, 16, 1, 4, 4, false, false)
FSR411_SIMD_BLOCK(fsr411_simd_fnb32, 32, 1, 4, 4, false, false)
FSR411_SIMD_BLOCK(fsr411_simd_fnb64, 64, 2, 2, 4, false, false)
FSR411_SIMD_BLOCK(fsr411_simd_ct32, 32, 1, 4, 4, true, false)
FSR411_SIMD_BLOCK(fsr411_simd_ct64, 64, 2, 2, 4, true, false)
FSR411_SIMD_BLOCK(fsr411_simd_wcnb, 16, 1, 4, 4, false, true)
FSR411_SIMD_BLOCK(fsr411_simd_wfnb32, 32, 1, 4, 4, false, true)
FSR411_SIMD_BLOCK(fsr411_simd_wct32, 32, 1, 4, 4, true, true)
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
