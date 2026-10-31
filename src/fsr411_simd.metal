// SPDX-License-Identifier: MIT
// Apple7+ (M1-M4) FSR 4.1.1 / 4.1.2 integer blocks on simdgroup matrices.
//
// INT8 activations and weights are exact in half. 8x8 half x half products
// accumulate in float: model importers verify 128 * sum|w| < 2^24 per
// output column (Lite: 410,112 Normal / 636,672 Ultra), so every partial
// float sum is the exact integer in any association order. Biases,
// residuals and the INT32 epilogue (fsr411::finish) are applied in integers,
// so the result is byte-identical to the scalar backend.
//
// The epilogue addresses the two elements each lane owns in an 8x8 matrix.
// That ownership is the Apple7+ layout used by lane_origin(); the host runs
// fsr411_simd_probe once per device and selects this backend only if both
// the layout and the exact float accumulation are confirmed.
//
// Bindings: 0 scratch, 1 packed INT8 weights + INT32 biases, 2 PassParams,
// 3 the same weights blob widened element-wise to half (same indices),
// 4 (*_winograd entries only) integer H*g*H^T per eligible 3x3 bank, at half
// element offset 2*weights[0] (src/fsr411_simd_weights.h).
// Compile with -std=metal3.0 -I src; FSR411_SPECIALIZE=1 for per-pass
// function constants.
#include <metal_stdlib>
#include "fsr411_model.h"

// Compile-time tile overrides are for the isolated SIMD benchmark. Defaults
// keep the manifest ABI: CNB/NF32 32x4, NF64/DS32 16x4, 128 threads.
#ifndef FSR411_SIMD_MB16
#define FSR411_SIMD_MB16 4
#endif
#ifndef FSR411_SIMD_MB32
#define FSR411_SIMD_MB32 4
#endif
#ifndef FSR411_SIMD_MB64
#define FSR411_SIMD_MB64 2
#endif

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

// Same unsigned wrapping/vector rounding as the optimized M5 INT8 path,
// adapted to the two adjacent channels owned by an Apple7+ matrix lane.
__attribute__((always_inline)) inline int2 finish2(int2 acc, int2 bias, int2 residual, fsr411::Epilogue q) {
    uint2 bits = ((as_type<uint2>(acc) + as_type<uint2>(bias)) << q.accumulatorShift)
               + (as_type<uint2>(residual) << q.residualShift);
    int2 value = as_type<int2>(bits);
    if (q.flags & fsr411::kReluBeforeRound) value = max(value, int2(0));
    if (q.rightShift) {
        bits = as_type<uint2>(value);
        const uint halfStep = 1u << (q.rightShift - 1u);
        const uint2 increment = (q.flags & fsr411::kRoundHalfUp)
            ? uint2(halfStep) : uint2(halfStep - 1u) + ((bits >> q.rightShift) & 1u);
        value = as_type<int2>(bits + increment) >> q.rightShift;
    }
    if (q.flags & fsr411::kNarrowInt16)
        value = as_type<int2>(as_type<uint2>(value) << 16u) >> 16u;
    return clamp(value, int2(-128), int2(127));
}

#ifndef FSR411_FLOAT_EPILOGUE
#define FSR411_FLOAT_EPILOGUE 0
#endif
// finish2 on exact integer float accumulators, returned as float. The host
// defines FSR411_FLOAT_EPILOGUE only after proving, for every pass,
// |accumulator| + |bias| + (128 << residualShift) < 2^24: every float sum is
// then the exact integer, no 32-bit wrap can occur, scaling by 2^-k is exact,
// rint is round-half-even and floor(x + 0.5) round-half-up. Shifted or
// INT16-narrowed epilogues keep the integer sequence (folded per pass).
__attribute__((always_inline)) inline float2 finish2f(float2 acc, int2 bias, int2 residual, fsr411::Epilogue q) {
#if FSR411_FLOAT_EPILOGUE
    if (q.accumulatorShift == 0u && !(q.flags & fsr411::kNarrowInt16)) {
        float2 v = acc + float2(bias) + float2(residual) * float(1u << q.residualShift);
        if (q.flags & fsr411::kReluBeforeRound) v = max(v, float2(0.f));
        const float scale = as_type<float>((127u - q.rightShift) << 23u);
        v = (q.flags & fsr411::kRoundHalfUp) ? floor(v * scale + 0.5f) : rint(v * scale);
        return clamp(v, float2(-128.f), float2(127.f));
    }
#endif
    return float2(finish2(int2(acc), bias, residual, q));
}

// Requantize one accumulator fragment into an INT8-valued half fragment.
__attribute__((always_inline)) inline hmat requant(thread fmat &acc, int2 bias, fsr411::Epilogue q) {
    const half2 value = half2(finish2f(float2(acc.thread_elements()[0], acc.thread_elements()[1]), bias, int2(0), q));
    hmat r;
    r.thread_elements()[0] = value.x;
    r.thread_elements()[1] = value.y;
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

// Downsample consumes each staged INT8 element exactly once per SIMD tile.
// Keep it packed until the lane loads its two matrix elements: the number of
// conversions is unchanged, but shared storage and staging writes are halved.
template <uint C, uint W, uint H, uint THREADS>
__attribute__((always_inline)) inline void load_window_s8(threadgroup uchar *tile, device const uchar *scratch,
                                                          fsr411::Tensor t, int x0, int y0, uint tid) {
    constexpr uint UNITS = C / 16u;
    for (uint i = tid; i < W * H * UNITS; i += THREADS) {
        const uint px = i / UNITS, ch = (i % UNITS) * 16u;
        const int x = x0 + int(px % W), y = y0 + int(px / W);
        uint4 value(0);
        if (x >= 0 && y >= 0 && uint(x) < t.width && uint(y) < t.height)
            value = *(device const uint4 *)(scratch + fsr411::tensor_address(t, uint(x), uint(y), ch));
        *(threadgroup uint4 *)(tile + px * C + ch) = value;
    }
}

// Only convolved banks need a halo. Bypass banks need centre pixels only.
// Compact layout: [halo row][halo pixel][16*BANKS], then
// [centre row][centre pixel][NF-16*BANKS]. No uninitialized lane is read.
template <uint NF, uint BANKS, uint TW, uint SG>
__attribute__((always_inline)) inline void load_block_window(threadgroup half *tile, device const uchar *scratch,
                                                            fsr411::Tensor t, int x0, int y0, uint tid) {
    constexpr uint BC = 16u * BANKS, BY = NF - BC, WW = TW + 2u, WH = SG + 2u;
    constexpr uint CONV_UNITS = WW * WH * BANKS, BY_UNITS = BY / 16u;
    for (uint i = tid; i < CONV_UNITS + TW * SG * BY_UNITS; i += SG * 32u) {
        uint px, ch; int x, y; threadgroup half *dst;
        if (i < CONV_UNITS) {
            px = i / BANKS; ch = (i % BANKS) * 16u;
            x = x0 + int(px % WW) - 1; y = y0 + int(px / WW) - 1;
            dst = tile + px * BC + ch;
        } else {
            // max keeps this expression valid for CNB (BY_UNITS == 0);
            // that specialization never enters the bypass branch.
            const uint j = i - CONV_UNITS;
            px = j / max(BY_UNITS, 1u); ch = BC + (j % max(BY_UNITS, 1u)) * 16u;
            x = x0 + int(px % TW); y = y0 + int(px / TW);
            dst = tile + WW * WH * BC + px * BY + (ch - BC);
        }
        uint4 v(0);
        if (x >= 0 && y >= 0 && uint(x) < t.width && uint(y) < t.height)
            v = *(device const uint4 *)(scratch + fsr411::tensor_address(t, uint(x), uint(y), ch));
        threadgroup half4 *out = (threadgroup half4 *)dst;
        out[0] = half4(as_type<char4>(v.x)); out[1] = half4(as_type<char4>(v.y));
        out[2] = half4(as_type<char4>(v.z)); out[3] = half4(as_type<char4>(v.w));
    }
}

// Construct a fresh fragment instead of read/modify/write of a matrix
// array element. Only the lane-owned pair is meaningful after distribution.
__attribute__((always_inline)) inline hmat half_fragment(half2 value) {
    hmat result = make_filled_simdgroup_matrix<half, 8, 8>(half(0));
    result.thread_elements()[0] = value.x; result.thread_elements()[1] = value.y;
    return result;
}
__attribute__((always_inline)) inline hmat requant_values(float2 acc, int2 bias, fsr411::Epilogue q) {
    return half_fragment(half2(finish2f(acc, bias, int2(0), q)));
}

// Exact F(2x2,3x3). Compile-time transform coordinates avoid dynamically
// indexed private arrays; keep the inverse in ordinary lane-owned float4s.
// V=B^T*d*B is integral, |V|<=512. T4=H*g*H^T=4G*g*G^T is an exact half
// integer, |T4|<=1152. The host bounds EVERY scaled inverse partial <2^24.
// No extra shared memory, intermediate quantization or approximate math.
struct WinogradRow { float4 left, right; };
struct WinogradFragments { float4 p00, p01, p10, p11; };

template <uint BC, uint WW, uint WY, uint KB, uint X>
__attribute__((always_inline)) inline half2 winograd_vertical(threadgroup const half *tile,
                                                              uint bank, uint sx, uint sy, ushort2 lo) {
    constexpr uint FIRST = WY == 0u ? 0u : WY == 2u ? 2u : 1u;
    constexpr uint SECOND = WY < 2u ? 2u : WY == 2u ? 1u : 3u;
    const uint tx = sx + 2u * lo.y + X, ch = bank * 16u + KB * 8u + lo.x;
    const half2 a = *(threadgroup const half2 *)(tile + ((sy + FIRST) * WW + tx) * BC + ch);
    const half2 b = *(threadgroup const half2 *)(tile + ((sy + SECOND) * WW + tx) * BC + ch);
    return WY == 1u ? a + b : a - b;
}

template <uint WY, uint WX>
__attribute__((always_inline)) inline float4 winograd_dot(half2 v0, half2 v1, device const half *w0) {
    constexpr uint OFF = (WY * 4u + WX) * 256u;
    fmat lowerDot = zero(), upperDot = zero();
    hmat lower, upper, a0 = half_fragment(v0), a1 = half_fragment(v1);
    simdgroup_load(lower, w0 + OFF, 16);
    simdgroup_load(upper, w0 + OFF + 8u, 16);
    simdgroup_multiply_accumulate(lowerDot, a0, lower, lowerDot);
    simdgroup_multiply_accumulate(upperDot, a0, upper, upperDot);
    simdgroup_load(lower, w0 + OFF + 128u, 16);
    simdgroup_load(upper, w0 + OFF + 136u, 16);
    simdgroup_multiply_accumulate(lowerDot, a1, lower, lowerDot);
    simdgroup_multiply_accumulate(upperDot, a1, upper, upperDot);
    return float4(lowerDot.thread_elements()[0], lowerDot.thread_elements()[1],
                  upperDot.thread_elements()[0], upperDot.thread_elements()[1]);
}

template <uint BC, uint WW, uint WY>
__attribute__((always_inline)) inline WinogradRow winograd_row(threadgroup const half *tile,
                                                               device const half *w0, uint bank, uint sx,
                                                               uint sy, ushort2 lo) {
    const half2 v00 = winograd_vertical<BC, WW, WY, 0, 0>(tile,bank,sx,sy,lo);
    const half2 v01 = winograd_vertical<BC, WW, WY, 0, 1>(tile,bank,sx,sy,lo);
    const half2 v02 = winograd_vertical<BC, WW, WY, 0, 2>(tile,bank,sx,sy,lo);
    const half2 v03 = winograd_vertical<BC, WW, WY, 0, 3>(tile,bank,sx,sy,lo);
    const half2 v10 = winograd_vertical<BC, WW, WY, 1, 0>(tile,bank,sx,sy,lo);
    const half2 v11 = winograd_vertical<BC, WW, WY, 1, 1>(tile,bank,sx,sy,lo);
    const half2 v12 = winograd_vertical<BC, WW, WY, 1, 2>(tile,bank,sx,sy,lo);
    const half2 v13 = winograd_vertical<BC, WW, WY, 1, 3>(tile,bank,sx,sy,lo);
    float4 left = winograd_dot<WY,0>(v00-v02,v10-v12,w0);
    float4 d = winograd_dot<WY,1>(v01+v02,v11+v12,w0);
    left += d; float4 right = d;
    d = winograd_dot<WY,2>(v02-v01,v12-v11,w0); left += d; right -= d;
    d = winograd_dot<WY,3>(v01-v03,v11-v13,w0); right -= d;
    return {left,right};
}

template <uint BC, uint WW>
__attribute__((always_inline)) inline WinogradFragments winograd3x3(threadgroup const half *tile,
                                                                    device const half *w0, uint bank,
                                                                    uint sx, uint sy, ushort2 lo) {
    WinogradRow r = winograd_row<BC,WW,0>(tile,w0,bank,sx,sy,lo);
    float4 p00 = r.left, p01 = r.right;
    r = winograd_row<BC,WW,1>(tile,w0,bank,sx,sy,lo);
    p00 += r.left; p01 += r.right; float4 p10 = r.left, p11 = r.right;
    r = winograd_row<BC,WW,2>(tile,w0,bank,sx,sy,lo);
    p00 += r.left; p01 += r.right; p10 -= r.left; p11 -= r.right;
    r = winograd_row<BC,WW,3>(tile,w0,bank,sx,sy,lo);
    p10 -= r.left; p11 -= r.right;
    // Remove H's factor of four only after the COMPLETE integer inverse.
    // Every final numerator is divisible by four: this cannot round.
    return {p00*0.25f,p01*0.25f,p10*0.25f,p11*0.25f};
}

// CNB (NF=16, one bank), FNB and CT blocks. Each SIMD group computes 8*MB
// consecutive pixels of one row; the threadgroup covers 8*MB x SG pixels and
// shares one input window with a one-pixel halo. CNB/FNB dispatch in output
// space, CT in input space, exactly like the scalar kernels.
template <uint NF, uint BANKS, uint MB, uint SG, bool UPSAMPLE, bool WINOGRAD = false>
__attribute__((always_inline)) inline void block(device uchar *scratch, device const uchar *weights,
                                                 device const half *wh, device const half *wg, FSR411_PARAMS_ADDRESS fsr411::PassParams &p,
                                                 threadgroup half *tile, uint2 group, uint tid, ushort sg, ushort lane) {
    static_assert(NF % 16u == 0 && 16u * BANKS <= NF, "conv banks must fit the block width");
    static_assert(!WINOGRAD || (MB == 4u && SG == 4u && BANKS == 1u), "Winograd requires the 32x4 tile and one conv bank");
    constexpr uint TW = 8u * MB, WW = TW + 2u, WH = SG + 2u;
    constexpr uint BC = 16u * BANKS, BY = NF - BC, KB = NF / 8u, N1 = 2u * NF;
    const int x0 = int(group.x * TW), y0 = int(group.y * SG);
    load_block_window<NF, BANKS, TW, SG>(tile, scratch, p.input, x0, y0, tid);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Winograd assigns eight 2x2 tiles to each SIMD group, retaining the
    // original 32x4/128-thread dispatch and the same logical tensor layout.
    const uint sx = WINOGRAD ? (sg & 1u) * 16u : 0u;
    const uint sy = WINOGRAD ? (sg >> 1u) * 2u : sg;
    const uint y = uint(y0) + sy;
    if (y >= (UPSAMPLE ? p.input.height : p.output.height)) return;
    const ushort2 lo = lane_origin(lane);
    threadgroup const half *centre = tile + ((sy + 1u) * WW + sx + 1u) * BC;
    threadgroup const half *bypass = tile + WW * WH * BC + (sy * TW + sx) * BY;

    // Grouped 3x3 convolutions (16 -> 16 per bank), then the bypass channels.
    hmat a1[MB][KB];
#pragma unroll
    for (uint b = 0; b < BANKS; ++b) {
        if (WINOGRAD) {
            const WinogradFragments v = winograd3x3<BC, WW>(tile, wg + size_t(p.weights[0]) * 2u + b * 4096u, b, sx, sy, lo);
#pragma unroll
            for (uint n = 0; n < 2u; ++n) {
                const int2 bias = load_bias2(weights, p.biases[0], b * 16u + n * 8u + lo.x);
                a1[0][2u*b+n] = requant_values(n == 0u ? v.p00.xy : v.p00.zw, bias, p.q[0]);
                a1[1][2u*b+n] = requant_values(n == 0u ? v.p01.xy : v.p01.zw, bias, p.q[0]);
                a1[2][2u*b+n] = requant_values(n == 0u ? v.p10.xy : v.p10.zw, bias, p.q[0]);
                a1[3][2u*b+n] = requant_values(n == 0u ? v.p11.xy : v.p11.zw, bias, p.q[0]);
            }
        } else {
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
                        simdgroup_load(a, tile + ((sg + ky) * WW + m * 8u + kx) * BC + b * 16u + kb * 8u, BC);
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
    }
#pragma unroll
    for (uint cb = 2u * BANKS; cb < KB; ++cb)
#pragma unroll
        for (uint m = 0; m < MB; ++m) {
            const uint px = WINOGRAD ? (m & 1u) : m * 8u, py = WINOGRAD ? m / 2u : 0u;
            simdgroup_load(a1[m][cb], bypass + (py * TW + px) * BY + (cb * 8u - BC), (WINOGRAD ? 2u : 1u) * BY);
        }

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
            const uint px = WINOGRAD ? 2u * lo.y + (m & 1u) : m * 8u + lo.y;
            const uint py = WINOGRAD ? m / 2u : 0u;
            threadgroup const half *r = ch < BC ? centre + (py * WW + px) * BC + ch : bypass + (py * TW + px) * BY + (ch - BC);
            const int2 value = int2(finish2f(float2(out[m][ob].thread_elements()[0], out[m][ob].thread_elements()[1]),
                                      bias, int2(r[0], r[1]), q));
            if (UPSAMPLE) {
                a3[m][ob].thread_elements()[0] = half(value.x);
                a3[m][ob].thread_elements()[1] = half(value.y);
            } else {
                const uint x = uint(x0) + sx + px, oy = y + py;
                if (x < p.output.width && oy < p.output.height)
                    *(device ushort *)(scratch + fsr411::tensor_address(p.output, x, oy, ch)) = pack2(value.x, value.y);
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
            const uint ix = uint(x0) + sx + (WINOGRAD ? 2u * lo.y + (m & 1u) : m * 8u + lo.y);
            const uint iy = y + (WINOGRAD ? m / 2u : 0u);
            const uint ox = ix * 2u + (tap & 1u), oy = iy * 2u + (tap >> 1u);
            if (ix >= p.input.width || iy >= p.input.height || ox >= p.output.width || oy >= p.output.height) continue;
            int2 residual(0);
            if (ox < p.skip.width && oy < p.skip.height)
                residual = int2(as_type<char2>(*(device const ushort *)(scratch + fsr411::tensor_address(p.skip, ox, oy, oc))));
            const int2 value = int2(finish2f(float2(t.thread_elements()[0], t.thread_elements()[1]), bias, residual, q));
            *(device ushort *)(scratch + fsr411::tensor_address(p.output, ox, oy, oc)) = pack2(value.x, value.y);
        }
    }
}

// Stride-2 2x2 downsample CIN -> 2*CIN. K = [tap][input channel].
template <uint CIN, uint MB, uint SG, bool COMPACT>
__attribute__((always_inline)) inline void downsample(device uchar *scratch, device const uchar *weights,
                                                      device const half *wh, FSR411_PARAMS_ADDRESS fsr411::PassParams &p,
                                                      threadgroup uchar *storage, uint2 group, uint tid, ushort sg, ushort lane) {
    constexpr uint COUT = 2u * CIN, TW = 8u * MB, WW = 2u * TW, WH = 2u * SG, NB = COUT / 8u;
    const int x0 = int(group.x * TW), y0 = int(group.y * SG);
    threadgroup half *tile = (threadgroup half *)storage;
    if (COMPACT) load_window_s8<CIN, WW, WH, SG * 32u>(storage, scratch, p.input, 2 * x0, 2 * y0, tid);
    else load_window<CIN, WW, WH, SG * 32u>(tile, scratch, p.input, 2 * x0, 2 * y0, tid);
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
            for (uint m = 0; m < MB; ++m) {
                const uint off = ((2u * sg + ty) * WW + 16u * m + tx) * CIN + kb * 8u;
                if (COMPACT) {
                    const half2 value = half2(as_type<char2>(*(threadgroup const ushort *)(storage + off + 2u * lo.y * CIN + lo.x)));
                    a[m] = half_fragment(value);
                } else simdgroup_load(a[m], tile + off, 2u * CIN);
            }
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
            const int2 value = int2(finish2f(float2(acc[m][nb].thread_elements()[0], acc[m][nb].thread_elements()[1]), bias, int2(0), q));
            *(device ushort *)(scratch + fsr411::tensor_address(p.output, x, y, ch)) = pack2(value.x, value.y);
        }
    }
}

} // namespace fsr411_simd

#define FSR411_SIMD_ARGS \
    device uchar *scratch [[buffer(0)]], device const uchar *weights [[buffer(1)]], \
    constant fsr411::PassParams &raw [[buffer(2)]], device const half *wh [[buffer(3)]], \
    device const half *wg [[buffer(4)]], \
    uint2 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], \
    ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]

// Tile = 8*MB x SG pixels, 32*SG threads. Manifest "simd" entries must match.
#define FSR411_SIMD_BLOCK(NAME, NF, BANKS, MB, SG, UPSAMPLE, WINOGRAD) \
kernel void NAME(FSR411_SIMD_ARGS) { \
    FSR411_LOCAL_PARAMS; \
    threadgroup half tile[(8 * (MB) + 2) * ((SG) + 2) * (16 * (BANKS)) + (8 * (MB)) * (SG) * ((NF) - 16 * (BANKS))]; \
    fsr411_simd::block<NF, BANKS, MB, SG, UPSAMPLE, WINOGRAD>(scratch, weights, wh, wg, p, tile, group, tid, sg, lane); \
}
FSR411_SIMD_BLOCK(fsr411_simd_cnb, 16, 1, FSR411_SIMD_MB16, 4, false, false)
FSR411_SIMD_BLOCK(fsr411_simd_fnb32, 32, 1, FSR411_SIMD_MB32, 4, false, false)
FSR411_SIMD_BLOCK(fsr411_simd_fnb64, 64, 2, FSR411_SIMD_MB64, 4, false, false)
FSR411_SIMD_BLOCK(fsr411_simd_ct32, 32, 1, FSR411_SIMD_MB32, 4, true, false)
FSR411_SIMD_BLOCK(fsr411_simd_ct64, 64, 2, FSR411_SIMD_MB64, 4, true, false)
// Optional exact transform variants. The host must supply bound-checked
// transformed weights at buffer(4); existing manifest entries stay valid.
FSR411_SIMD_BLOCK(fsr411_simd_cnb_winograd, 16, 1, 4, 4, false, true)
FSR411_SIMD_BLOCK(fsr411_simd_fnb32_winograd, 32, 1, 4, 4, false, true)
// CT32 Winograd is exact but measured slower on M5; the runtime never selects
// it. Kept for `fsr411_simd_bench --winograd` comparisons on M1-M4.
FSR411_SIMD_BLOCK(fsr411_simd_ct32_winograd, 32, 1, 4, 4, true, true)
#undef FSR411_SIMD_BLOCK

#define FSR411_SIMD_DS(NAME, CIN, MB, SG, COMPACT) \
kernel void NAME(FSR411_SIMD_ARGS) { \
    FSR411_LOCAL_PARAMS; \
    threadgroup uint4 tile[16 * (MB) * 2 * (SG) * (CIN) * ((COMPACT) ? 1 : 2) / 16]; \
    fsr411_simd::downsample<CIN, MB, SG, COMPACT>(scratch, weights, wh, p, (threadgroup uchar *)tile, group, tid, sg, lane); \
}
FSR411_SIMD_DS(fsr411_simd_ds16, 16, FSR411_SIMD_MB32, 4, false)
FSR411_SIMD_DS(fsr411_simd_ds32, 32, FSR411_SIMD_MB64, 4, false)
// Experimental compact staging, identical dispatch ABI.
FSR411_SIMD_DS(fsr411_simd_ds16_compact, 16, FSR411_SIMD_MB32, 4, true)
FSR411_SIMD_DS(fsr411_simd_ds32_compact, 32, FSR411_SIMD_MB64, 4, true)
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
