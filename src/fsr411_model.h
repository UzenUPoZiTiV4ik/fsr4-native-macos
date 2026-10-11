// SPDX-License-Identifier: MIT
// FSR 4.1.1 integer model ABI and arithmetic shared by Metal and host tests.
// This is deliberately separate from the older v07 float-scale/half-bias ABI.
#ifndef FSR411_MODEL_H
#define FSR411_MODEL_H

#if defined(__METAL_VERSION__)
#include <metal_stdlib>
namespace fsr411 {
using u32 = metal::uint;
using i32 = metal::int32_t;
#else
#include <cstdint>
#include <type_traits>
namespace fsr411 {
using u32 = std::uint32_t;
using i32 = std::int32_t;
#endif

enum : u32 { kAbiVersion = 1, kThreads = 128, kNoOffset = 0xffffffffu };

// Byte offsets refer to the logical (0,0) pixel. Row strides may include the
// physical zero border used by AMD's tensor allocation. Kernels synthesize zero
// for logical out-of-bounds inputs; they do not consume or produce border bytes.
struct Tensor {
    u32 offset;
    u32 strideX;
    u32 strideY;
    u32 width;
    u32 height;
    u32 channels;
    // FSR411 stores channels >=16 in separate spatial banks. Zero selects
    // ordinary contiguous channels; otherwise each 16-channel bank advances
    // by this many bytes, independently of the row/pixel strides.
    u32 bankStride;
};

enum EpilogueFlags : u32 {
    kReluBeforeRound = 1u,
    // Some AMD model tails convert the rounded result to int16 before pack_s8.
    // Preserve the narrowing even though normal model values fit that range.
    kNarrowInt16 = 2u,
    // Explicit lossy profile only: nearest rounding, but half-way values
    // resolve toward positive infinity instead of the even integer.
    kRoundHalfUp = 4u,
};

// n = ((dot + int32_bias) << accumulatorShift) +
//     (residual_int8 << residualShift); out = RNE(n / 2^rightShift).
// The additions and left shifts have 32-bit wrapping semantics, matching the
// shader integer operations. All shift counts must be in [0,30].
struct Epilogue {
    u32 accumulatorShift;
    u32 residualShift;
    u32 rightShift;
    u32 flags;
};

// Blob layouts, all in bytes:
//   weights[0]: 3x3 [group][ky][kx][input=16][output=16], or stride-2
//               [tap][input][output] for a downsample kernel;
//   weights[1/2]: pointwise [input][output];
//   weights[3]: transposed convolution [input][tap][output].
//   biases[0..3]: int32[channel], kNoOffset means no bias.
// Each weights region is 64-byte aligned; bias regions are 16-byte aligned.
// q[0/1]: first convolution / ReLU expansion;
// q[2/3]: residual contraction, lower / upper channel halves;
// q[4/5]: stride-2 or transposed-convolution output, lower / upper halves.
struct PassParams {
    Tensor input;
    Tensor output;
    Tensor skip;
    u32 weights[4];
    u32 biases[4];
    Epilogue q[6];
};

inline u32 tensor_address(Tensor t, u32 x, u32 y, u32 channel) {
    const u32 ch = t.bankStride ? (channel / 16u) * t.bankStride + channel % 16u : channel;
    return t.offset + y * t.strideY + x * t.strideX + ch;
}

inline i32 signed_bits(u32 value) {
#if defined(__METAL_VERSION__)
    return as_type<i32>(value);
#else
    // Do not rely on implementation-defined unsigned-to-signed conversion.
    return value <= 0x7fffffffu ? i32(value) : -1 - i32(0xffffffffu - value);
#endif
}

inline i32 add_wrap(i32 a, i32 b) {
    return signed_bits(u32(a) + u32(b));
}

inline i32 left_wrap(i32 value, u32 bits) {
    return signed_bits(u32(value) << bits);
}

inline i32 arithmetic_right(i32 value, u32 bits) {
    if (bits == 0) return value;
    const u32 sign = value < 0 ? (~u32(0) << (32u - bits)) : 0u;
    return signed_bits((u32(value) >> bits) | sign);
}

// Exact shader sequence, including overflow at the signed 32-bit endpoints.
// For the bounded model accumulators this is ordinary round-to-nearest-even.
inline i32 round_shift_even(i32 value, u32 bits) {
    if (bits == 0) return value;
    const u32 halfMinusOne = (1u << (bits - 1u)) - 1u;
    const u32 odd = (u32(value) >> bits) & 1u;
    return arithmetic_right(signed_bits(u32(value) + halfMinusOne + odd), bits);
}

inline i32 round_shift_half_up(i32 value, u32 bits) {
    if (bits == 0) return value;
    return arithmetic_right(signed_bits(u32(value) + (1u << (bits - 1u))), bits);
}

inline i32 narrow_int16(i32 value) {
    const u32 bits = u32(value) & 65535u;
    return bits < 32768u ? i32(bits) : i32(bits) - 65536;
}

inline i32 clamp_s8(i32 value) {
    return value < -128 ? -128 : (value > 127 ? 127 : value);
}

inline i32 finish(i32 accumulator, i32 bias, i32 residual, Epilogue q) {
    i32 value = add_wrap(left_wrap(add_wrap(accumulator, bias), q.accumulatorShift),
                         left_wrap(residual, q.residualShift));
    if ((q.flags & kReluBeforeRound) && value < 0) value = 0;
    value = (q.flags & kRoundHalfUp) ? round_shift_half_up(value, q.rightShift) : round_shift_even(value, q.rightShift);
    if (q.flags & kNarrowInt16) value = narrow_int16(value);
    return clamp_s8(value);
}

inline bool valid_epilogue(Epilogue q) {
    return q.accumulatorShift <= 30 && q.residualShift <= 30 &&
           q.rightShift <= 30 && (q.flags & ~(kReluBeforeRound | kNarrowInt16 | kRoundHalfUp)) == 0;
}

#if !defined(__METAL_VERSION__)
// Pipeline constants omit the dimensions, which can change for each frame.
// The auxiliary POST tensor is appended to a session-sized allocation, so its
// output offset also stays dynamic. Keep PassParams and on-disk caches intact.
enum : u32 { kPassSpecializationCount = 12 };
struct PassSpecialization {
    u32 values[kPassSpecializationCount][4];
};

inline PassSpecialization specialize_metadata(const PassParams &p, bool dynamicOutputOffset) {
    return {{
        {p.input.offset, p.input.strideX, p.input.strideY, p.input.channels},
        {p.input.bankStride, p.output.offset, p.output.strideX, p.output.strideY},
        {p.output.channels, p.output.bankStride, p.skip.offset, p.skip.strideX},
        {p.skip.strideY, p.skip.channels, p.skip.bankStride, u32(dynamicOutputOffset)},
        {p.weights[0], p.weights[1], p.weights[2], p.weights[3]},
        {p.biases[0], p.biases[1], p.biases[2], p.biases[3]},
        {p.q[0].accumulatorShift, p.q[0].residualShift, p.q[0].rightShift, p.q[0].flags},
        {p.q[1].accumulatorShift, p.q[1].residualShift, p.q[1].rightShift, p.q[1].flags},
        {p.q[2].accumulatorShift, p.q[2].residualShift, p.q[2].rightShift, p.q[2].flags},
        {p.q[3].accumulatorShift, p.q[3].residualShift, p.q[3].rightShift, p.q[3].flags},
        {p.q[4].accumulatorShift, p.q[4].residualShift, p.q[4].rightShift, p.q[4].flags},
        {p.q[5].accumulatorShift, p.q[5].residualShift, p.q[5].rightShift, p.q[5].flags},
    }};
}

static_assert(sizeof(Tensor) == 28, "Metal tensor ABI");
static_assert(sizeof(Epilogue) == 16, "Metal epilogue ABI");
static_assert(sizeof(PassParams) == 212, "Metal pass ABI");
static_assert(std::is_standard_layout<PassParams>::value, "Metal pass ABI layout");
#endif
} // namespace fsr411

#if defined(__METAL_VERSION__)
// Standalone native tests and older callers can keep the generic ABI. The
// runtime enables specialization and supplies these values per logical pass.
// Shared by the Apple10 tensor kernels and the Apple7+ simdgroup kernels.
#ifndef FSR411_SPECIALIZE
#define FSR411_SPECIALIZE 0
#endif
#if FSR411_SPECIALIZE
#define FSR411_PARAMS_ADDRESS thread const
#define FSR411_LOCAL_PARAMS const fsr411::PassParams p = fsr411_specialize(raw)
constant metal::uint4 fsr411Spec0 [[function_constant(0)]];
constant metal::uint4 fsr411Spec1 [[function_constant(1)]];
constant metal::uint4 fsr411Spec2 [[function_constant(2)]];
constant metal::uint4 fsr411Spec3 [[function_constant(3)]];
constant metal::uint4 fsr411Spec4 [[function_constant(4)]];
constant metal::uint4 fsr411Spec5 [[function_constant(5)]];
constant metal::uint4 fsr411Spec6 [[function_constant(6)]];
constant metal::uint4 fsr411Spec7 [[function_constant(7)]];
constant metal::uint4 fsr411Spec8 [[function_constant(8)]];
constant metal::uint4 fsr411Spec9 [[function_constant(9)]];
constant metal::uint4 fsr411Spec10 [[function_constant(10)]];
constant metal::uint4 fsr411Spec11 [[function_constant(11)]];

__attribute__((always_inline)) inline fsr411::PassParams fsr411_specialize(constant fsr411::PassParams &r) {
    // Widths/heights and the session-appended POST output stay dynamic.
    return {
        {fsr411Spec0.x,fsr411Spec0.y,fsr411Spec0.z,r.input.width,r.input.height,fsr411Spec0.w,fsr411Spec1.x},
        {fsr411Spec3.w ? r.output.offset : fsr411Spec1.y,fsr411Spec1.z,fsr411Spec1.w,r.output.width,r.output.height,fsr411Spec2.x,fsr411Spec2.y},
        {fsr411Spec2.z,fsr411Spec2.w,fsr411Spec3.x,r.skip.width,r.skip.height,fsr411Spec3.y,fsr411Spec3.z},
        {fsr411Spec4.x,fsr411Spec4.y,fsr411Spec4.z,fsr411Spec4.w},
        {fsr411Spec5.x,fsr411Spec5.y,fsr411Spec5.z,fsr411Spec5.w},
        {{fsr411Spec6.x,fsr411Spec6.y,fsr411Spec6.z,fsr411Spec6.w},{fsr411Spec7.x,fsr411Spec7.y,fsr411Spec7.z,fsr411Spec7.w},{fsr411Spec8.x,fsr411Spec8.y,fsr411Spec8.z,fsr411Spec8.w},{fsr411Spec9.x,fsr411Spec9.y,fsr411Spec9.z,fsr411Spec9.w},{fsr411Spec10.x,fsr411Spec10.y,fsr411Spec10.z,fsr411Spec10.w},{fsr411Spec11.x,fsr411Spec11.y,fsr411Spec11.z,fsr411Spec11.w}}
    };
}
#else
#define FSR411_PARAMS_ADDRESS constant
#define FSR411_LOCAL_PARAMS constant fsr411::PassParams &p = raw
#endif
#endif
#endif
