// SPDX-License-Identifier: MIT
// Native M5 FSR 4.1.1 integer ConvNext blocks. Integer matrix products use
// Metal Performance Primitives; epilogues preserve the 4.1.1 round-even,
// int32 bias and int16-before-int8 tail semantics. Fixed pass metadata can
// be specialized at pipeline creation without changing the parameter ABI.
//
// The weights come only from the verified 4.1.1 import, packed by
// tools/gen_fsr411_native.py. There is no v07 model fallback in this file.
// Compile with -std=metal4.0, -I src, and FSR411_MAXW/FSR411_MAXH for the tier.
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include "fsr411_model.h"

using namespace metal;
using namespace mpp::tensor_ops;

#ifndef FSR411_MAXW
#define FSR411_MAXW 960
#endif
#ifndef FSR411_MAXH
#define FSR411_MAXH 540
#endif
#ifndef FSR411_VECTOR_QUANTIZE
#define FSR411_VECTOR_QUANTIZE 1
#endif

// FSR411_SPECIALIZE, FSR411_PARAMS_ADDRESS and FSR411_LOCAL_PARAMS come from
// fsr411_model.h.

namespace fsr411_gpu {

__attribute__((always_inline)) inline int4 load_bias(device const uchar *weights, uint offset, uint channel) {
    return offset == fsr411::kNoOffset ? int4(0) :
        *(device const int4 *)(weights + offset + channel * sizeof(int));
}

__attribute__((always_inline)) inline int4 load_acc(threadgroup int *acc, uint offset) {
    return *(threadgroup int4 *)(acc + offset);
}

__attribute__((always_inline)) inline uint quantize(int4 acc, int4 bias, int4 residual, fsr411::Epilogue q) {
#if FSR411_VECTOR_QUANTIZE
    // Keep wrapping operations unsigned. Metal's signed right shift is
    // arithmetic, so it directly implements the sign fill in finish().
    // The half-way increment uses the pre-rounded integer, including at
    // int32 endpoints; narrowing still happens before the int8 clamp.
    uint4 bits = ((as_type<uint4>(acc) + as_type<uint4>(bias)) << q.accumulatorShift)
               + (as_type<uint4>(residual) << q.residualShift);
    int4 value = as_type<int4>(bits);
    if (q.flags & fsr411::kReluBeforeRound) value = max(value, int4(0));
    if (q.rightShift) {
        bits = as_type<uint4>(value);
        const uint halfStep = 1u << (q.rightShift - 1u);
        const uint4 increment = (q.flags & fsr411::kRoundHalfUp)
            ? uint4(halfStep) : uint4(halfStep - 1u) + ((bits >> q.rightShift) & 1u);
        value = as_type<int4>(bits + increment) >> q.rightShift;
    }
    if (q.flags & fsr411::kNarrowInt16)
        value = as_type<int4>(as_type<uint4>(value) << 16u) >> 16u;
    return as_type<uint>(char4(clamp(value, int4(-128), int4(127))));
#else
    int4 result;
#pragma unroll
    for (uint c = 0; c < 4; ++c)
        result[c] = fsr411::finish(acc[c], bias[c], residual[c], q);
    return as_type<uint>(char4(result));
#endif
}

template <bool RELU>
__attribute__((always_inline)) inline uint quantize_cnb_intermediate(int4 acc, int4 bias, fsr411::Epilogue q) {
    constexpr uint flags = RELU ? fsr411::kReluBeforeRound : 0u;
    // Every imported intermediate has no accumulator shift and no int16
    // narrowing. Keep a generic fallback for other ABI-valid epilogues.
    if (!q.accumulatorShift && q.flags == flags)
        return quantize(acc, bias, int4(0), fsr411::Epilogue{0, 0, q.rightShift, flags});
    if (!q.accumulatorShift && q.flags == (flags | fsr411::kRoundHalfUp))
        return quantize(acc, bias, int4(0), fsr411::Epilogue{0, 0, q.rightShift, flags | fsr411::kRoundHalfUp});
    return quantize(acc, bias, int4(0), q);
}

// Fixed NHWC tiles retain the storage reduction already proven on M5: the
// expansion reads all a1 before a2 replaces it. Both matrix products complete
// before the next stage writes the shared activation allocation.
template <int TW, int TH, int SG = 4>
__attribute__((always_inline)) inline void conv3x3(device uchar *scratch, device uchar *weights,
                    FSR411_PARAMS_ADDRESS fsr411::PassParams &p, threadgroup int *acc, uint2 group, uint bank = 0) {
    auto input = tensor<device int8_t, dextents<int32_t, 4>, tensor_inline>(
        (device int8_t *)(scratch + fsr411::tensor_address(p.input, 0, 0, bank * 16u)),
        dextents<int32_t, 4>(16, int(p.input.width), int(p.input.height), 1),
        array<int, 4>{1, int(p.input.strideX), int(p.input.strideY), int(p.input.strideY * p.input.height)});
    auto weight = tensor<device int8_t, extents<int32_t, 16, 16, 3, 3>, tensor_inline>(
        (device int8_t *)(weights + p.weights[0] + bank * 9u * 16u * 16u), extents<int32_t, 16, 16, 3, 3>());
    auto output = tensor<threadgroup int, extents<int32_t, 16, TW, TH, 1>, tensor_inline>(
        acc, extents<int32_t, 16, TW, TH, 1>());
    constexpr auto descriptor = convolution2d_descriptor(
        int4(16, TW, TH, 1), int4(16, FSR411_MAXW, FSR411_MAXH, 1), int2(3),
        convolution2d_activation_layout::nhwc, convolution2d_weights_layout::hwio,
        int2(1), int2(1), 1, false, convolution2d_descriptor::mode::multiply);
    convolution2d<descriptor, execution_simdgroups<SG>> op;
    op.set_offsets(int2(group * uint2(TW, TH)));
    op.run(input, weight, output);
}

template <int T, int N, int K, bool SPLIT_ROWS = false, int SG = 4>
__attribute__((always_inline)) inline void pointwise(threadgroup int8_t *activations, device int8_t *weights,
                      threadgroup int *acc, uint tid) {
    // DS16 and ordinary FNB32 benefit from four independent row slices.
    // Other blocks retain the full-threadgroup operation. Caller barriers
    // still make all output rows visible before the common epilogue.
    if constexpr (SPLIT_ROWS) {
        // The Apple10 native ABI uses four SIMD32 groups (128 threads).
        static_assert(fsr411::kThreads == 128, "Row splitting requires four SIMD32 groups");
        static_assert(T % 4 == 0, "Each SIMD group needs an equal row slice");
        constexpr int ROWS = T / 4;
        const uint row = (tid / 32u) * ROWS;
        auto a = tensor<threadgroup int8_t, extents<int32_t, K, ROWS>, tensor_inline>(activations + row * K, extents<int32_t, K, ROWS>());
        auto b = tensor<device int8_t, extents<int32_t, N, K>, tensor_inline>(weights, extents<int32_t, N, K>());
        auto c = tensor<threadgroup int, extents<int32_t, N, ROWS>, tensor_inline>(acc + row * N, extents<int32_t, N, ROWS>());
        constexpr auto descriptor = matmul2d_descriptor(ROWS, N, K, false, false, false, matmul2d_descriptor::mode::multiply);
        matmul2d<descriptor, execution_simdgroup> op;
        op.run(a, b, c);
    } else {
        auto a = tensor<threadgroup int8_t, extents<int32_t, K, T>, tensor_inline>(activations, extents<int32_t, K, T>());
        auto b = tensor<device int8_t, extents<int32_t, N, K>, tensor_inline>(weights, extents<int32_t, N, K>());
        auto c = tensor<threadgroup int, extents<int32_t, N, T>, tensor_inline>(acc, extents<int32_t, N, T>());
        constexpr auto descriptor = matmul2d_descriptor(T, N, K, false, false, false, matmul2d_descriptor::mode::multiply);
        matmul2d<descriptor, execution_simdgroups<SG>> op;
        op.run(a, b, c);
    }
}

} // namespace fsr411_gpu

// Logical passes 1, 2, 12 and the auxiliary POST CNB. Host supplies the exact
// per-pass integer shifts and independent source/target tensor offsets from
// the generated manifest; POST uses a separately appended scratch region.
// Bindings: 0 scratch, 1 packed 4.1.1 weights+biases, 2 PassParams (212 bytes).
// Dispatch uses the selected entry point's tile and SIMD32 group count.
namespace fsr411_gpu {
template <uint TW, uint TH, uint SG>
__attribute__((always_inline)) inline void cnb(
    device uchar *scratch, device uchar *weights,
    FSR411_PARAMS_ADDRESS fsr411::PassParams &p, threadgroup uint *activation,
    threadgroup int *acc, uint2 group, uint tid) {
    static_assert(SG == 1 || SG == 2 || SG == 4, "CNB needs one, two or four SIMD32 groups");
    constexpr uint T = TW * TH;
    auto a = (threadgroup int8_t *)activation;

    fsr411_gpu::conv3x3<TW, TH, SG>(scratch, weights, p, acc, group);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {
        const uint channel = (tid & 3u) * 4u;
        const int4 bias = fsr411_gpu::load_bias(weights, p.biases[0], channel);
        for (uint i = tid; i < T * 4u; i += (SG * 32u)) {
            const uint pixel = i / 4u;
            activation[i] = fsr411_gpu::quantize_cnb_intermediate<false>(fsr411_gpu::load_acc(acc, pixel * 16u + channel), bias, p.q[0]);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    fsr411_gpu::pointwise<T, 32, 16, false, SG>(a, (device int8_t *)(weights + p.weights[1]), acc, tid);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {
        const uint channel = (tid & 7u) * 4u;
        const int4 bias = fsr411_gpu::load_bias(weights, p.biases[1], channel);
        for (uint i = tid; i < T * 8u; i += (SG * 32u)) {
            const uint pixel = i / 8u;
            activation[i] = fsr411_gpu::quantize_cnb_intermediate<true>(fsr411_gpu::load_acc(acc, pixel * 32u + channel), bias, p.q[1]);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    fsr411_gpu::pointwise<T, 16, 32, false, SG>(a, (device int8_t *)(weights + p.weights[2]), acc, tid);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint channel = (tid & 3u) * 4u;
    const int4 bias = fsr411_gpu::load_bias(weights, p.biases[2], channel);
    for (uint i = tid; i < T * 4u; i += (SG * 32u)) {
        const uint pixel = i / 4u;
        const uint x = group.x * TW + pixel % TW, y = group.y * TH + pixel / TW;
        if (x >= p.output.width || y >= p.output.height) continue;
        int4 residual(0);
        if (x < p.input.width && y < p.input.height)
            residual = int4(as_type<char4>(*(device uint *)(scratch + fsr411::tensor_address(p.input, x, y, channel))));
        const uint value = fsr411_gpu::quantize(fsr411_gpu::load_acc(acc, pixel * 16u + channel),
                                               bias, residual, (channel < 8u ? p.q[2] : p.q[3]));
        *(device uint *)(scratch + fsr411::tensor_address(p.output, x, y, channel)) = value;
    }
}
} // namespace fsr411_gpu

#define FSR411_CNB(NAME, TW, TH, SG) \
kernel void NAME(device uchar *scratch [[buffer(0)]], device uchar *weights [[buffer(1)]], \
                 constant fsr411::PassParams &raw [[buffer(2)]], \
                 uint2 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) { \
    FSR411_LOCAL_PARAMS; \
    threadgroup uint activation[(TW) * (TH) * 32 / 4]; \
    threadgroup int4 accumulator[(TW) * (TH) * 32 / 4]; \
    fsr411_gpu::cnb<TW, TH, SG>(scratch, weights, p, activation, \
        (threadgroup int *)accumulator, group, tid); \
}
// Preserve the original entry name for cached 32x4/128-thread manifests.
FSR411_CNB(fsr411_cnb, 32, 4, 4)
// Smaller tiles and matching SIMD scopes reduce CNB storage and synchronization scope.
FSR411_CNB(fsr411_cnb_32x1_sg1, 32, 1, 1)
#undef FSR411_CNB

namespace fsr411_gpu {
template <uint CIN, uint COUT, uint T>
__attribute__((always_inline)) inline void downsample(device uchar *scratch, device uchar *weights,
                        FSR411_PARAMS_ADDRESS fsr411::PassParams &p, threadgroup int8_t *a,
                        threadgroup int *acc, uint2 group, uint tid) {
    constexpr uint K = 4u * CIN, CH = CIN / 16u;
    for (uint i = tid; i < T * 4u * CH; i += fsr411::kThreads) {
        const uint pixel = i / (4u * CH), tap = (i / CH) % 4u, channel = (i % CH) * 16u;
        const uint x = 2u * (group.x * T + pixel) + (tap & 1u), y = 2u * group.y + (tap >> 1u);
        uint4 value(0);
        if (x < p.input.width && y < p.input.height)
            value = *(device uint4 *)(scratch + fsr411::tensor_address(p.input, x, y, channel));
        *(threadgroup uint4 *)(a + pixel * K + tap * CIN + channel) = value;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    pointwise<T, COUT, K, CIN == 16u && COUT == 32u && T == 64u>(
        a, (device int8_t *)(weights + p.weights[0]), acc, tid);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint channel = (tid % (COUT / 4u)) * 4u;
    const int4 bias = load_bias(weights, p.biases[0], channel);
    for (uint i = tid; i < T * (COUT / 4u); i += fsr411::kThreads) {
        const uint pixel = i / (COUT / 4u), x = group.x * T + pixel, y = group.y;
        if (x >= p.output.width || y >= p.output.height) continue;
        const uint value = quantize(load_acc(acc, pixel * COUT + channel), bias, int4(0), (channel < COUT / 2u ? p.q[4] : p.q[5]));
        *(device uint *)(scratch + fsr411::tensor_address(p.output, x, y, channel)) = value;
    }
}
} // namespace fsr411_gpu

#define FSR411_DOWNSAMPLE(NAME, CIN, COUT, T) \
kernel void NAME(device uchar *scratch [[buffer(0)]], device uchar *weights [[buffer(1)]], \
                 constant fsr411::PassParams &raw [[buffer(2)]], \
                 uint2 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) { \
    FSR411_LOCAL_PARAMS; \
    threadgroup uint4 activation[(T) * 4 * (CIN) / 16]; \
    threadgroup int4 accumulator[(T) * (COUT) / 4]; \
    fsr411_gpu::downsample<CIN, COUT, T>(scratch, weights, p, (threadgroup int8_t *)activation, \
                                        (threadgroup int *)accumulator, group, tid); \
}
FSR411_DOWNSAMPLE(fsr411_ds16, 16, 32, 64)
FSR411_DOWNSAMPLE(fsr411_ds32, 32, 64, 32)
#undef FSR411_DOWNSAMPLE

namespace fsr411_gpu {
__attribute__((always_inline)) inline int4 read_pixel(device uchar *scratch, fsr411::Tensor tensor, uint x, uint y, uint channel) {
    if (x >= tensor.width || y >= tensor.height) return int4(0);
    return int4(as_type<char4>(*(device uint *)(scratch + fsr411::tensor_address(tensor, x, y, channel))));
}

template <uint NF, uint TW, uint TH, bool SPLIT_ROWS, uint SG = 4>
__attribute__((always_inline)) inline void fnb_core(device uchar *scratch, device uchar *weights, FSR411_PARAMS_ADDRESS fsr411::PassParams &p,
                      threadgroup int8_t *a, threadgroup int *acc, uint2 group, uint tid) {
    constexpr uint T = TW * TH;
    for (uint bank = 0; bank < NF / 32u; ++bank) {
        conv3x3<TW, TH, SG>(scratch, weights, p, acc, group, bank);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const uint channel = (tid & 3u) * 4u;
        const int4 bias = load_bias(weights, p.biases[0], bank * 16u + channel);
        for (uint i = tid; i < T * 4u; i += (SG * 32u)) {
            const uint pixel = i / 4u;
            *(threadgroup uint *)(a + pixel * NF + bank * 16u + channel) =
                quantize(load_acc(acc, pixel * 16u + channel), bias, int4(0), p.q[0]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint i = tid; i < T * (NF / 8u); i += (SG * 32u)) {
        const uint pixel = i / (NF / 8u), channel = NF / 2u + (i % (NF / 8u)) * 4u;
        const uint x = group.x * TW + pixel % TW, y = group.y * TH + pixel / TW;
        const int4 value = read_pixel(scratch, p.input, x, y, channel);
        *(threadgroup uint *)(a + pixel * NF + channel) = as_type<uint>(char4(value));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    pointwise<T, 2u * NF, NF, SPLIT_ROWS, SG>(a, (device int8_t *)(weights + p.weights[1]), acc, tid);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint channel = (tid % (NF / 2u)) * 4u;
    const int4 bias = load_bias(weights, p.biases[1], channel);
    for (uint i = tid; i < T * (NF / 2u); i += (SG * 32u)) {
        const uint pixel = i / (NF / 2u);
        *(threadgroup uint *)(a + pixel * 2u * NF + channel) =
            quantize(load_acc(acc, pixel * 2u * NF + channel), bias, int4(0), p.q[1]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    pointwise<T, NF, 2u * NF, SPLIT_ROWS, SG>(a, (device int8_t *)(weights + p.weights[2]), acc, tid);
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

template <uint NF, uint TW, uint TH, bool UPSAMPLE, uint SG = 4>
__attribute__((always_inline)) inline void fnb(device uchar *scratch, device uchar *weights, FSR411_PARAMS_ADDRESS fsr411::PassParams &p,
                 threadgroup int8_t *a, threadgroup int *acc, uint2 group, uint tid) {
    constexpr uint T = TW * TH;
    constexpr bool SPLIT_ROWS = NF == 32u && T == 64u && !UPSAMPLE && SG == 4u;
    fnb_core<NF, TW, TH, SPLIT_ROWS, SG>(scratch, weights, p, a, acc, group, tid);
    const uint channel = (tid % (NF / 4u)) * 4u;
    const int4 bias = load_bias(weights, p.biases[2], channel);
    for (uint i = tid; i < T * (NF / 4u); i += (SG * 32u)) {
        const uint pixel = i / (NF / 4u), x = group.x * TW + pixel % TW, y = group.y * TH + pixel / TW;
        const int4 residual = read_pixel(scratch, p.input, x, y, channel);
        const uint value = quantize(load_acc(acc, pixel * NF + channel), bias, residual, (channel < NF / 2u ? p.q[2] : p.q[3]));
        if (UPSAMPLE) *(threadgroup uint *)(a + pixel * NF + channel) = value;
        else if (x < p.output.width && y < p.output.height)
            *(device uint *)(scratch + fsr411::tensor_address(p.output, x, y, channel)) = value;
    }
    if (!UPSAMPLE) return;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Four taps are adjacent output columns of one larger GEMM. The shared
    // activation buffer can be reused: input scratch is separate from output,
    // and residuals are reread from that immutable input plane.
    constexpr uint OUT = NF / 2u;
    pointwise<T, 4u * OUT, NF, false, SG>(a, (device int8_t *)(weights + p.weights[3]), acc, tid);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < T * OUT; i += (SG * 32u)) {
        const uint pixel = i / OUT, tap = (i % OUT) / (OUT / 4u), c = (i % (OUT / 4u)) * 4u;
        const uint ix = group.x * TW + pixel % TW, iy = group.y * TH + pixel / TW;
        const uint x = ix * 2u + (tap & 1u), y = iy * 2u + (tap >> 1u);
        if (ix >= p.input.width || iy >= p.input.height || x >= p.output.width || y >= p.output.height) continue;
        const int4 b = load_bias(weights, p.biases[3], c);
        const int4 residual = read_pixel(scratch, p.skip, x, y, c);
        const uint value = quantize(load_acc(acc, pixel * 4u * OUT + tap * OUT + c), b, residual, (c < OUT / 2u ? p.q[4] : p.q[5]));
        *(device uint *)(scratch + fsr411::tensor_address(p.output, x, y, c)) = value;
    }
}
} // namespace fsr411_gpu

#define FSR411_FNB(NAME, NF, TW, TH, UPSAMPLE, SG) \
kernel void NAME(device uchar *scratch [[buffer(0)]], device uchar *weights [[buffer(1)]], \
                 constant fsr411::PassParams &raw [[buffer(2)]], \
                 uint2 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) { \
    FSR411_LOCAL_PARAMS; \
    threadgroup uint activation[(TW) * (TH) * 2 * (NF) / 4]; \
    threadgroup int4 accumulator[(TW) * (TH) * 2 * (NF) / 4]; \
    fsr411_gpu::fnb<NF, TW, TH, UPSAMPLE, SG>(scratch, weights, p, (threadgroup int8_t *)activation, \
                                        (threadgroup int *)accumulator, group, tid); \
}
// Preserve the original entry points for manifests built before tile tuning.
FSR411_FNB(fsr411_fnb32, 32, 32, 2, false, 4)
FSR411_FNB(fsr411_fnb64, 64, 32, 1, false, 4)
FSR411_FNB(fsr411_ct32, 32, 32, 2, true, 4)
FSR411_FNB(fsr411_ct64, 64, 32, 1, true, 4)
// These shapes need 10 KiB of threadgroup storage instead of 20 KiB. The
// generator selects them only for the blocks that benefit on Apple10.
FSR411_FNB(fsr411_fnb64_16x1, 64, 16, 1, false, 4)
FSR411_FNB(fsr411_ct64_16x1, 64, 16, 1, true, 4)
FSR411_FNB(fsr411_ct32_32x1, 32, 32, 1, true, 4)
// NF32 blocks use 5 KiB of threadgroup storage and one SIMD32 group.
// Dedicated names preserve every previously cached tile/thread contract.
FSR411_FNB(fsr411_fnb32_16x1_sg1, 32, 16, 1, false, 1)
FSR411_FNB(fsr411_ct32_16x1_sg1, 32, 16, 1, true, 1)
#undef FSR411_FNB

// Register-resident CNB/FNB32/CT32 chains, one SIMD32 group. Each stage's
// cooperative destination feeds the next matrix product's left operand
// without threadgroup storage. The element ownership of cooperative tensors
// is implementation-defined, so every entry first verifies the layouts it
// relies on (simd_all) and otherwise runs the established SG1 kernel. The
// result is byte-identical either way. Native tests can force the fallback.
#ifndef FSR411_CHAIN_FORCE_FALLBACK
#define FSR411_CHAIN_FORCE_FALLBACK 0
#endif

namespace fsr411_chain {
template <uint I, uint COUNT, bool CONV, class Source, class Destination>
__attribute__((always_inline)) inline bool matching_elements(thread Source &src, thread Destination &dst) {
    const auto s = src.get_multidimensional_index(I);
    const auto d = dst.get_multidimensional_index(I);
    bool matches = src.is_valid_element(I) && dst.is_valid_element(I);
    if constexpr (CONV)
        matches = matches && s[0] == d[0] && s[1] + 32 * s[2] == d[1] && s[3] == 0;
    else
        matches = matches && s[0] == d[0] && s[1] == d[1];
    // The vector quantizer loads one bias quartet for four adjacent channels.
    if constexpr ((I & 3u) != 0) {
        const auto first = src.get_multidimensional_index(I & ~3u);
        matches = matches && s[0] == first[0] + int(I & 3u) && s[1] == first[1];
        if constexpr (CONV) matches = matches && s[2] == first[2] && s[3] == first[3];
    } else matches = matches && (s[0] & 3) == 0;
    if constexpr (I + 1u < COUNT)
        matches = matching_elements<I + 1u, COUNT, CONV>(src, dst) && matches;
    return matches;
}

template <uint COUNT, bool CONV, class Source, class Destination>
__attribute__((always_inline)) inline bool matching_layout(thread Source &src, thread Destination &dst) {
    bool matches = !FSR411_CHAIN_FORCE_FALLBACK && src.get_capacity() == COUNT && dst.get_capacity() == COUNT;
    if (matches) matches = matching_elements<0, COUNT, CONV>(src, dst);
    return simd_all(matches);
}

template <uint I, uint COUNT, bool RELU, class Source, class Destination>
__attribute__((always_inline)) inline void quantize_into(thread Source &src, thread Destination &dst,
    device uchar *weights, uint bias_offset, fsr411::Epilogue q) {
    const auto c = src.get_multidimensional_index(I);
    const int4 accumulator(src[I], src[I + 1u], src[I + 2u], src[I + 3u]);
    const int4 bias = fsr411_gpu::load_bias(weights, bias_offset, uint(c[0]));
    const char4 value = as_type<char4>(fsr411_gpu::quantize_cnb_intermediate<RELU>(accumulator, bias, q));
    dst[I] = value.x;
    dst[I + 1u] = value.y;
    dst[I + 2u] = value.z;
    dst[I + 3u] = value.w;
    if constexpr (I + 4u < COUNT)
        quantize_into<I + 4u, COUNT, RELU>(src, dst, weights, bias_offset, q);
}

template <uint I, class Result>
__attribute__((always_inline)) inline void store_scalar(thread Result &result, device uchar *scratch,
    device uchar *weights, FSR411_PARAMS_ADDRESS fsr411::PassParams &p, uint2 group) {
    if (result.is_valid_element(I)) {
        const auto c = result.get_multidimensional_index(I);
        const uint channel = uint(c[0]), x = group.x * 32u + uint(c[1]), y = group.y;
        if (x < p.output.width && y < p.output.height && channel < 16u) {
            const int bias = p.biases[2] == fsr411::kNoOffset ? 0 : *(device int *)(weights + p.biases[2] + channel * 4u);
            int residual = 0;
            if (x < p.input.width && y < p.input.height)
                residual = *(device char *)(scratch + fsr411::tensor_address(p.input, x, y, channel));
            const int value = fsr411::finish(result[I], bias, residual, channel < 8u ? p.q[2] : p.q[3]);
            *(device char *)(scratch + fsr411::tensor_address(p.output, x, y, channel)) = char(value);
        }
    }
}

template <uint I, uint COUNT, class Result>
__attribute__((always_inline)) inline void store_result(thread Result &result, device uchar *scratch,
    device uchar *weights, FSR411_PARAMS_ADDRESS fsr411::PassParams &p, uint2 group) {
    const auto c0 = result.get_multidimensional_index(I);
    const auto c1 = result.get_multidimensional_index(I + 1u);
    const auto c2 = result.get_multidimensional_index(I + 2u);
    const auto c3 = result.get_multidimensional_index(I + 3u);
    const bool quartet = result.is_valid_element(I) && result.is_valid_element(I + 1u)
        && result.is_valid_element(I + 2u) && result.is_valid_element(I + 3u)
        && (c0[0] & 3) == 0 && c1[0] == c0[0] + 1 && c2[0] == c0[0] + 2 && c3[0] == c0[0] + 3
        && c1[1] == c0[1] && c2[1] == c0[1] && c3[1] == c0[1];
    if (quartet) {
        const uint channel = uint(c0[0]), x = group.x * 32u + uint(c0[1]), y = group.y;
        if (x < p.output.width && y < p.output.height) {
            const int4 bias = fsr411_gpu::load_bias(weights, p.biases[2], channel);
            int4 residual(0);
            if (x < p.input.width && y < p.input.height)
                residual = int4(as_type<char4>(*(device uint *)(scratch + fsr411::tensor_address(p.input, x, y, channel))));
            const int4 accumulator(result[I], result[I + 1u], result[I + 2u], result[I + 3u]);
            const uint value = fsr411_gpu::quantize(accumulator, bias, residual, channel < 8u ? p.q[2] : p.q[3]);
            *(device uint *)(scratch + fsr411::tensor_address(p.output, x, y, channel)) = value;
        }
    } else {
        store_scalar<I>(result, scratch, weights, p, group);
        store_scalar<I + 1u>(result, scratch, weights, p, group);
        store_scalar<I + 2u>(result, scratch, weights, p, group);
        store_scalar<I + 3u>(result, scratch, weights, p, group);
    }
    if constexpr (I + 4u < COUNT)
        store_result<I + 4u, COUNT>(result, scratch, weights, p, group);
}

__attribute__((always_inline)) inline bool cnb(device uchar *scratch, device uchar *weights,
    FSR411_PARAMS_ADDRESS fsr411::PassParams &p, uint2 group) {
    constexpr auto expansion_descriptor = matmul2d_descriptor(32, 32, 16, false, false, false, matmul2d_descriptor::mode::multiply);
    constexpr auto projection_descriptor = matmul2d_descriptor(32, 16, 32, false, false, false, matmul2d_descriptor::mode::multiply);
    matmul2d<expansion_descriptor, execution_simdgroup> expansion;
    matmul2d<projection_descriptor, execution_simdgroup> projection;
    auto projection_input = projection.get_left_input_cooperative_tensor<int8_t, int8_t, int>();
    {
        auto expansion_input = expansion.get_left_input_cooperative_tensor<int8_t, int8_t, int>();
        {
            auto input = tensor<device int8_t, dextents<int,4>, tensor_inline>(
                (device int8_t *)(scratch + fsr411::tensor_address(p.input, 0, 0, 0)),
                dextents<int,4>(16, int(p.input.width), int(p.input.height), 1),
                array<int,4>{1, int(p.input.strideX), int(p.input.strideY), int(p.input.strideY * p.input.height)});
            auto weight = tensor<device int8_t, extents<int,16,16,3,3>, tensor_inline>(
                (device int8_t *)(weights + p.weights[0]), extents<int,16,16,3,3>());
            constexpr auto descriptor = convolution2d_descriptor(
                int4(16,32,1,1), int4(16,FSR411_MAXW,FSR411_MAXH,1), int2(3),
                convolution2d_activation_layout::nhwc, convolution2d_weights_layout::hwio,
                int2(1), int2(1), 1, false, convolution2d_descriptor::mode::multiply);
            convolution2d<descriptor, execution_simdgroup> convolution;
            auto result = convolution.get_destination_cooperative_tensor<decltype(input), decltype(weight), int>();
            if (!matching_layout<16, true>(result, expansion_input)) return false;
            convolution.set_offsets(int2(group * uint2(32,1)));
            convolution.run(input, weight, result);
            quantize_into<0, 16, false>(result, expansion_input, weights, p.biases[0], p.q[0]);
        }
        auto weight = tensor<device int8_t, extents<int,32,16>, tensor_inline>(
            (device int8_t *)(weights + p.weights[1]), extents<int,32,16>());
        auto result = expansion.get_destination_cooperative_tensor<decltype(expansion_input), decltype(weight), int>();
        if (!matching_layout<32, false>(result, projection_input)) return false;
        expansion.run(expansion_input, weight, result);
        quantize_into<0, 32, true>(result, projection_input, weights, p.biases[1], p.q[1]);
    }
    auto weight = tensor<device int8_t, extents<int,16,32>, tensor_inline>(
        (device int8_t *)(weights + p.weights[2]), extents<int,16,32>());
    auto result = projection.get_destination_cooperative_tensor<decltype(projection_input), decltype(weight), int>();
    if (!simd_all(result.get_capacity() == 16u)) return false;
    projection.run(projection_input, weight, result);
    store_result<0,16>(result, scratch, weights, p, group);
    return true;
}
} // namespace fsr411_chain

kernel void fsr411_cnb_coop_chain_32x1(device uchar *scratch [[buffer(0)]], device uchar *weights [[buffer(1)]],
    constant fsr411::PassParams &raw [[buffer(2)]], uint2 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    FSR411_LOCAL_PARAMS;
    // Fallback storage is the SG1 kernel's; the chain itself uses none.
    threadgroup uint activation[32u * 32u / 4u];
    threadgroup int4 accumulator[32u * 32u / 4u];
    if (!fsr411_chain::cnb(scratch, weights, p, group))
        fsr411_gpu::cnb<32,1,1>(scratch, weights, p, activation, (threadgroup int *)accumulator, group, tid);
}

// FNB32/CT32 tile 16x1: conv3x3 bank, bypass channels, expansion, projection
// and, for CT, the transpose convolution stay in cooperative tensors.
namespace fsr411_fnb_chain {
template <uint I, class Tensor>
__attribute__((always_inline)) inline bool quartet_element(thread Tensor &value) {
    const auto c = value.get_multidimensional_index(I);
    bool matches = value.is_valid_element(I);
    if constexpr ((I & 3u) == 0u) {
        matches = matches && (c[0] & 3) == 0;
    } else {
        const auto first = value.get_multidimensional_index(I & ~3u);
        matches = matches && c[0] == first[0] + int(I & 3u) && c[1] == first[1];
    }
    return matches;
}

template <uint I, uint COUNT, class Source, class Destination>
__attribute__((always_inline)) inline bool matching_elements(thread Source &src, thread Destination &dst) {
    const auto s = src.get_multidimensional_index(I);
    const auto d = dst.get_multidimensional_index(I);
    bool matches = quartet_element<I>(src) && dst.is_valid_element(I)
        && s[0] == d[0] && s[1] == d[1];
    if constexpr (I + 1u < COUNT)
        matches = matching_elements<I + 1u, COUNT>(src, dst) && matches;
    return matches;
}

template <uint COUNT, class Source, class Destination>
__attribute__((always_inline)) inline bool matching_layout(thread Source &src, thread Destination &dst) {
    bool matches = src.get_capacity() == COUNT && dst.get_capacity() == COUNT;
    if (matches) matches = matching_elements<0, COUNT>(src, dst);
    return simd_all(matches);
}

template <uint I, class Source, class Destination>
__attribute__((always_inline)) inline bool concat_element(thread Source &src, thread Destination &dst) {
    const auto s = src.get_multidimensional_index(I);
    const auto lower = dst.get_multidimensional_index(I);
    const auto upper = dst.get_multidimensional_index(I + 8u);
    bool matches = src.is_valid_element(I) && dst.is_valid_element(I) && dst.is_valid_element(I + 8u)
        && s[0] >= 0 && s[0] < 16 && s[1] >= 0 && s[1] < 16 && s[2] == 0 && s[3] == 0
        && lower[0] == s[0] && lower[1] == s[1]
        && upper[0] == s[0] + 16 && upper[1] == s[1];
    if constexpr ((I & 3u) == 0u) {
        matches = matches && (s[0] & 3) == 0;
    } else {
        const auto first = src.get_multidimensional_index(I & ~3u);
        matches = matches && s[0] == first[0] + int(I & 3u) && s[1] == first[1]
            && s[2] == first[2] && s[3] == first[3];
    }
    if constexpr (I + 1u < 8u) matches = concat_element<I + 1u>(src, dst) && matches;
    return matches;
}

template <class Source, class Destination>
__attribute__((always_inline)) inline bool concat_layout(thread Source &src, thread Destination &dst) {
    bool matches = !FSR411_CHAIN_FORCE_FALLBACK && src.get_capacity() == 8u && dst.get_capacity() == 16u;
    if (matches) matches = concat_element<0>(src, dst);
    return simd_all(matches);
}

template <uint I, uint COUNT, bool RELU, class Source, class Destination>
__attribute__((always_inline)) inline void quantize_into(thread Source &src, thread Destination &dst,
    device uchar *weights, uint bias_offset, fsr411::Epilogue q) {
    const auto c = src.get_multidimensional_index(I);
    const int4 accumulator(src[I], src[I + 1u], src[I + 2u], src[I + 3u]);
    const int4 bias = fsr411_gpu::load_bias(weights, bias_offset, uint(c[0]));
    const char4 value = as_type<char4>(fsr411_gpu::quantize_cnb_intermediate<RELU>(accumulator, bias, q));
    dst[I] = value.x;
    dst[I + 1u] = value.y;
    dst[I + 2u] = value.z;
    dst[I + 3u] = value.w;
    if constexpr (I + 4u < COUNT)
        quantize_into<I + 4u, COUNT, RELU>(src, dst, weights, bias_offset, q);
}

template <uint I, class Destination>
__attribute__((always_inline)) inline void load_bypass(thread Destination &dst, device uchar *scratch,
    FSR411_PARAMS_ADDRESS fsr411::PassParams &p, uint2 group) {
    const auto c = dst.get_multidimensional_index(I);
    const uint channel = uint(c[0]), x = group.x * 16u + uint(c[1]), y = group.y;
    const char4 value = char4(fsr411_gpu::read_pixel(scratch, p.input, x, y, channel));
    dst[I] = value.x;
    dst[I + 1u] = value.y;
    dst[I + 2u] = value.z;
    dst[I + 3u] = value.w;
    if constexpr (I + 4u < 16u) load_bypass<I + 4u>(dst, scratch, p, group);
}

template <uint I, class Source, class Destination>
__attribute__((always_inline)) inline void residual_into(thread Source &src, thread Destination &dst,
    device uchar *scratch, device uchar *weights, FSR411_PARAMS_ADDRESS fsr411::PassParams &p, uint2 group) {
    const auto c = src.get_multidimensional_index(I);
    const uint channel = uint(c[0]), x = group.x * 16u + uint(c[1]), y = group.y;
    const int4 accumulator(src[I], src[I + 1u], src[I + 2u], src[I + 3u]);
    const int4 bias = fsr411_gpu::load_bias(weights, p.biases[2], channel);
    const int4 residual = fsr411_gpu::read_pixel(scratch, p.input, x, y, channel);
    const char4 value = as_type<char4>(fsr411_gpu::quantize(accumulator, bias, residual, channel < 16u ? p.q[2] : p.q[3]));
    dst[I] = value.x;
    dst[I + 1u] = value.y;
    dst[I + 2u] = value.z;
    dst[I + 3u] = value.w;
    if constexpr (I + 4u < 16u) residual_into<I + 4u>(src, dst, scratch, weights, p, group);
}

template <uint I, bool UPSAMPLE, class Result>
__attribute__((always_inline)) inline void store_scalar(thread Result &result, device uchar *scratch,
    device uchar *weights, FSR411_PARAMS_ADDRESS fsr411::PassParams &p, uint2 group) {
    if (result.is_valid_element(I)) {
        const auto c = result.get_multidimensional_index(I);
        const uint raw_channel = uint(c[0]), ix = group.x * 16u + uint(c[1]), iy = group.y;
        const uint channel = UPSAMPLE ? raw_channel % 16u : raw_channel;
        const uint tap = raw_channel / 16u;
        const uint x = UPSAMPLE ? ix * 2u + (tap & 1u) : ix;
        const uint y = UPSAMPLE ? iy * 2u + (tap >> 1u) : iy;
        if (raw_channel >= (UPSAMPLE ? 64u : 32u) || x >= p.output.width || y >= p.output.height) return;
        if (UPSAMPLE && (ix >= p.input.width || iy >= p.input.height)) return;
        const uint bias_offset = UPSAMPLE ? p.biases[3] : p.biases[2];
        const int bias = bias_offset == fsr411::kNoOffset ? 0 : *(device int *)(weights + bias_offset + channel * 4u);
        const fsr411::Tensor residual_tensor = UPSAMPLE ? p.skip : p.input;
        int residual = 0;
        if (x < residual_tensor.width && y < residual_tensor.height)
            residual = *(device char *)(scratch + fsr411::tensor_address(residual_tensor, x, y, channel));
        const fsr411::Epilogue q = UPSAMPLE ? (channel < 8u ? p.q[4] : p.q[5]) : (channel < 16u ? p.q[2] : p.q[3]);
        const int value = fsr411::finish(result[I], bias, residual, q);
        *(device char *)(scratch + fsr411::tensor_address(p.output, x, y, channel)) = char(value);
    }
}

template <uint I, uint COUNT, bool UPSAMPLE, class Result>
__attribute__((always_inline)) inline void store_result(thread Result &result, device uchar *scratch,
    device uchar *weights, FSR411_PARAMS_ADDRESS fsr411::PassParams &p, uint2 group) {
    const auto c0 = result.get_multidimensional_index(I);
    const auto c1 = result.get_multidimensional_index(I + 1u);
    const auto c2 = result.get_multidimensional_index(I + 2u);
    const auto c3 = result.get_multidimensional_index(I + 3u);
    const bool quartet = result.is_valid_element(I) && result.is_valid_element(I + 1u)
        && result.is_valid_element(I + 2u) && result.is_valid_element(I + 3u)
        && (c0[0] & 3) == 0 && c1[0] == c0[0] + 1 && c2[0] == c0[0] + 2 && c3[0] == c0[0] + 3
        && c1[1] == c0[1] && c2[1] == c0[1] && c3[1] == c0[1];
    if (quartet) {
        const uint raw_channel = uint(c0[0]), ix = group.x * 16u + uint(c0[1]), iy = group.y;
        const uint channel = UPSAMPLE ? raw_channel % 16u : raw_channel;
        const uint tap = raw_channel / 16u;
        const uint x = UPSAMPLE ? ix * 2u + (tap & 1u) : ix;
        const uint y = UPSAMPLE ? iy * 2u + (tap >> 1u) : iy;
        if (raw_channel < (UPSAMPLE ? 64u : 32u) && x < p.output.width && y < p.output.height
            && (!UPSAMPLE || (ix < p.input.width && iy < p.input.height))) {
            const int4 accumulator(result[I], result[I + 1u], result[I + 2u], result[I + 3u]);
            const int4 bias = fsr411_gpu::load_bias(weights, UPSAMPLE ? p.biases[3] : p.biases[2], channel);
            const int4 residual = fsr411_gpu::read_pixel(scratch, UPSAMPLE ? p.skip : p.input, x, y, channel);
            const fsr411::Epilogue q = UPSAMPLE ? (channel < 8u ? p.q[4] : p.q[5]) : (channel < 16u ? p.q[2] : p.q[3]);
            const uint value = fsr411_gpu::quantize(accumulator, bias, residual, q);
            *(device uint *)(scratch + fsr411::tensor_address(p.output, x, y, channel)) = value;
        }
    } else {
        store_scalar<I, UPSAMPLE>(result, scratch, weights, p, group);
        store_scalar<I + 1u, UPSAMPLE>(result, scratch, weights, p, group);
        store_scalar<I + 2u, UPSAMPLE>(result, scratch, weights, p, group);
        store_scalar<I + 3u, UPSAMPLE>(result, scratch, weights, p, group);
    }
    if constexpr (I + 4u < COUNT)
        store_result<I + 4u, COUNT, UPSAMPLE>(result, scratch, weights, p, group);
}

template <bool UPSAMPLE>
__attribute__((always_inline)) inline bool fnb(device uchar *scratch, device uchar *weights,
    FSR411_PARAMS_ADDRESS fsr411::PassParams &p, uint2 group) {
    constexpr auto expansion_descriptor = matmul2d_descriptor(16, 64, 32, false, false, false, matmul2d_descriptor::mode::multiply);
    constexpr auto projection_descriptor = matmul2d_descriptor(16, 32, 64, false, false, false, matmul2d_descriptor::mode::multiply);
    matmul2d<expansion_descriptor, execution_simdgroup> expansion;
    matmul2d<projection_descriptor, execution_simdgroup> projection;
    auto projection_input = projection.template get_left_input_cooperative_tensor<int8_t, int8_t, int>();
    {
        auto expansion_input = expansion.template get_left_input_cooperative_tensor<int8_t, int8_t, int>();
        {
            auto input = tensor<device int8_t, dextents<int,4>, tensor_inline>(
                (device int8_t *)(scratch + fsr411::tensor_address(p.input, 0, 0, 0)),
                dextents<int,4>(16, int(p.input.width), int(p.input.height), 1),
                array<int,4>{1, int(p.input.strideX), int(p.input.strideY), int(p.input.strideY * p.input.height)});
            auto weight = tensor<device int8_t, extents<int,16,16,3,3>, tensor_inline>(
                (device int8_t *)(weights + p.weights[0]), extents<int,16,16,3,3>());
            constexpr auto descriptor = convolution2d_descriptor(
                int4(16,16,1,1), int4(16,FSR411_MAXW,FSR411_MAXH,1), int2(3),
                convolution2d_activation_layout::nhwc, convolution2d_weights_layout::hwio,
                int2(1), int2(1), 1, false, convolution2d_descriptor::mode::multiply);
            convolution2d<descriptor, execution_simdgroup> convolution;
            auto result = convolution.template get_destination_cooperative_tensor<decltype(input), decltype(weight), int>();
            if (!concat_layout(result, expansion_input)) return false;
            convolution.set_offsets(int2(group * uint2(16,1)));
            convolution.run(input, weight, result);
            quantize_into<0, 8, false>(result, expansion_input, weights, p.biases[0], p.q[0]);
        }
        load_bypass<8>(expansion_input, scratch, p, group);
        auto weight = tensor<device int8_t, extents<int,64,32>, tensor_inline>(
            (device int8_t *)(weights + p.weights[1]), extents<int,64,32>());
        auto result = expansion.template get_destination_cooperative_tensor<decltype(expansion_input), decltype(weight), int>();
        if (!matching_layout<32>(result, projection_input)) return false;
        expansion.run(expansion_input, weight, result);
        quantize_into<0, 32, true>(result, projection_input, weights, p.biases[1], p.q[1]);
    }
    auto weight = tensor<device int8_t, extents<int,32,64>, tensor_inline>(
        (device int8_t *)(weights + p.weights[2]), extents<int,32,64>());
    auto result = projection.template get_destination_cooperative_tensor<decltype(projection_input), decltype(weight), int>();
    if constexpr (UPSAMPLE) {
        constexpr auto descriptor = matmul2d_descriptor(16, 64, 32, false, false, false, matmul2d_descriptor::mode::multiply);
        matmul2d<descriptor, execution_simdgroup> transpose;
        auto transpose_input = transpose.template get_left_input_cooperative_tensor<int8_t, int8_t, int>();
        if (!matching_layout<16>(result, transpose_input)) return false;
        projection.run(projection_input, weight, result);
        residual_into<0>(result, transpose_input, scratch, weights, p, group);
        auto transpose_weight = tensor<device int8_t, extents<int,64,32>, tensor_inline>(
            (device int8_t *)(weights + p.weights[3]), extents<int,64,32>());
        auto transpose_result = transpose.template get_destination_cooperative_tensor<decltype(transpose_input), decltype(transpose_weight), int>();
        if (!simd_all(transpose_result.get_capacity() == 32u)) return false;
        transpose.run(transpose_input, transpose_weight, transpose_result);
        store_result<0,32,true>(transpose_result, scratch, weights, p, group);
    } else {
        if (!simd_all(result.get_capacity() == 16u)) return false;
        projection.run(projection_input, weight, result);
        store_result<0,16,false>(result, scratch, weights, p, group);
    }
    return true;
}
} // namespace fsr411_fnb_chain

#define FSR411_FNB_CHAIN(NAME, UPSAMPLE) \
kernel void NAME(device uchar *scratch [[buffer(0)]], device uchar *weights [[buffer(1)]], \
    constant fsr411::PassParams &raw [[buffer(2)]], uint2 group [[threadgroup_position_in_grid]], \
    uint tid [[thread_index_in_threadgroup]]) { \
    FSR411_LOCAL_PARAMS; \
    threadgroup uint activation[16u * 64u / 4u]; \
    threadgroup int4 accumulator[16u * 64u / 4u]; \
    if (!fsr411_fnb_chain::fnb<UPSAMPLE>(scratch, weights, p, group)) \
        fsr411_gpu::fnb<32,16,1,UPSAMPLE,1>(scratch, weights, p, (threadgroup int8_t *)activation, \
            (threadgroup int *)accumulator, group, tid); \
}
FSR411_FNB_CHAIN(fsr411_fnb32_coop_chain_16x1, false)
FSR411_FNB_CHAIN(fsr411_ct32_coop_chain_16x1, true)
#undef FSR411_FNB_CHAIN
