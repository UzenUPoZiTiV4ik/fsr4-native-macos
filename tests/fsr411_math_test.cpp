// SPDX-License-Identifier: MIT
#include "../src/fsr411_model.h"
#include <array>
#include <cfenv>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>

namespace {
using fsr411::i32;
using fsr411::u32;
unsigned long checks = 0;

void equal(i32 actual, i32 expected, const char *label) {
    ++checks;
    if (actual != expected) {
        std::fprintf(stderr, "%s: got %d, expected %d\n", label, actual, expected);
        std::exit(1);
    }
}

// Floating point is an independent oracle here: all int32 values and all
// powers-of-two divisions are exactly representable in double.
i32 rational_round(i32 value, u32 shift) {
    return static_cast<i32>(std::nearbyint(std::ldexp(double(value), -int(shift))));
}

i32 rational_half_up(i32 value, u32 shift) {
    return static_cast<i32>(std::floor(std::ldexp(double(value), -int(shift)) + 0.5));
}

i32 reference_tail(i32 acc, i32 bias, i32 residual, double cs, double rs, double os) {
    const double sum = double(acc) + double(bias);
    const i32 rounded = static_cast<i32>(std::nearbyint((sum * cs + residual * rs) * os));
    const i32 wrapped = ((rounded % 65536) + 65536) % 65536;
    return fsr411::clamp_s8(wrapped >= 32768 ? wrapped - 65536 : wrapped);
}
}

int main() {
    std::fesetround(FE_TONEAREST);
    for (u32 shift : {1u, 6u, 7u, 8u, 9u, 15u, 30u}) {
        for (i32 value = -131072; value <= 131072; ++value) {
            equal(fsr411::round_shift_even(value, shift), rational_round(value, shift), "RNE rational");
            equal(fsr411::round_shift_half_up(value, shift), rational_half_up(value, shift), "half-up rational");
        }
        // Every positive and negative tie surrounding the int8 clamp boundaries.
        for (i32 q = -130; q <= 130; ++q) {
            const std::int64_t tie = std::int64_t(q) * (std::int64_t(1) << shift) +
                                     (std::int64_t(1) << (shift - 1));
            for (i32 delta : {-1, 0, 1}) {
                const auto value = tie + delta;
                if (value < -2147483648LL || value > 2147483647LL - (1LL << (shift - 1))) continue;
                equal(fsr411::round_shift_even(i32(value), shift), rational_round(i32(value), shift), "RNE tie");
                equal(fsr411::round_shift_half_up(i32(value), shift), rational_half_up(i32(value), shift), "half-up tie");
            }
        }
    }

    constexpr i32 lo = std::numeric_limits<i32>::min();
    constexpr i32 hi = std::numeric_limits<i32>::max();
    equal(fsr411::add_wrap(hi, 1), lo, "bias add wraps");
    equal(fsr411::add_wrap(lo, -1), hi, "negative bias add wraps");
    equal(fsr411::left_wrap(-128, 24), lo, "negative shift is defined");
    equal(fsr411::arithmetic_right(-1, 30), -1, "sign extension");
    equal(fsr411::round_shift_even(hi, 1), -1073741824, "shader endpoint overflow");
    equal(fsr411::round_shift_even(hi, 8), -8388608, "shader endpoint overflow 8");
    equal(fsr411::round_shift_even(lo, 8), -8388608, "minimum exact multiple");
    equal(fsr411::round_shift_even(hi, 0), hi, "zero shift");
    equal(fsr411::round_shift_half_up(hi, 8), -8388608, "half-up shader endpoint overflow");
    equal(fsr411::round_shift_half_up(hi, 0), hi, "half-up zero shift");

    const fsr411::Epilogue identity{0, 0, 0, 0};
    const fsr411::Epilogue narrow{0, 0, 0, fsr411::kNarrowInt16};
    equal(fsr411::finish(32768, 0, 0, identity), 127, "ordinary saturation");
    equal(fsr411::finish(32768, 0, 0, narrow), -128, "int16 before saturation");
    equal(fsr411::finish(65535, 0, 0, narrow), -1, "int16 sign extension");
    equal(fsr411::finish(65536, 0, 0, narrow), 0, "int16 truncation");
    equal(fsr411::finish(-65537, 0, 0, narrow), -1, "negative int16 truncation");
    const fsr411::Epilogue halfUp{0, 0, 7, fsr411::kRoundHalfUp};
    const fsr411::Epilogue halfEven{0, 0, 7, 0};
    equal(fsr411::finish(64, 0, 0, halfUp), 1, "lossy positive tie differs");
    equal(fsr411::finish(64, 0, 0, halfEven), 0, "exact positive tie preserved");
    equal(fsr411::finish(-192, 0, 0, halfUp), -1, "lossy negative tie toward positive");
    equal(fsr411::finish(-192, 0, 0, halfEven), -2, "exact negative tie preserved");

    // The three actual CNB tails: p1 input/64 + acc/4096 -> x64;
    // p2 same -> x32; p12 input/32 + acc/2048 -> x16.
    const std::array<fsr411::Epilogue, 3> tails{{
        {0, 6, 6, fsr411::kNarrowInt16},
        {0, 6, 7, fsr411::kNarrowInt16},
        {0, 6, 7, fsr411::kNarrowInt16}}};
    const double cs[] = {1.0 / 4096, 1.0 / 4096, 1.0 / 2048};
    const double rs[] = {1.0 / 64, 1.0 / 64, 1.0 / 32};
    const double os[] = {64, 32, 16};
    for (unsigned pass = 0; pass < tails.size(); ++pass)
        for (i32 residual = -128; residual <= 127; ++residual)
            for (i32 acc = -65536; acc <= 65536; acc += 31)
                equal(fsr411::finish(acc, 17, residual, tails[pass]),
                      reference_tail(acc, 17, residual, cs[pass], rs[pass], os[pass]), "CNB tail rational");

    for (u32 shift : {7u, 8u}) {
        const fsr411::Epilogue relu{0, 0, shift, fsr411::kReluBeforeRound};
        const fsr411::Epilogue reluHalfUp{0, 0, shift, fsr411::kReluBeforeRound | fsr411::kRoundHalfUp};
        for (i32 value = -65536; value <= 65536; ++value) {
            const i32 nonnegative = value < 0 ? 0 : value;
            equal(fsr411::finish(value, 0, 0, relu), fsr411::clamp_s8(rational_round(nonnegative, shift)), "ReLU before round");
            equal(fsr411::finish(value, 0, 0, reluHalfUp), fsr411::clamp_s8(rational_half_up(nonnegative, shift)), "ReLU before half-up round");
        }
    }
    equal(fsr411::valid_epilogue(identity), true, "valid epilogue");
    equal(fsr411::valid_epilogue({0, 0, 31, 0}), false, "invalid right shift");
    equal(fsr411::valid_epilogue({31, 0, 0, 0}), false, "invalid left shift");
    equal(fsr411::valid_epilogue(halfUp), true, "lossy flag accepted");
    equal(fsr411::valid_epilogue({0, 0, 0, 8}), false, "unknown flag");
    std::printf("fsr411_math_test: %lu checks passed\n", checks);
}
