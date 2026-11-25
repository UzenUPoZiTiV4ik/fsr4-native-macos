// SPDX-License-Identifier: MIT
// Host-only, lossless F(2x2,3x3) preparation for the pre-M5 SIMD backend.
#pragma once
#include "fsr411_model.h"
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace fsr411 {

// Only CNB/NF16 and the first, convolved 16-channel bank of NF32 qualify.
// NF32's other bank is a bypass, not a 32->32 convolution. weights[1/2]
// distinguish these blocks from stride-2 DS, whose raw shape is different.
// The caller must supply the native [tap=ky*3+kx][ic=16][oc=16] ABI, and
// validate the rest of the pass separately: offsets cannot identify shapes.
//
// One shared buffer has bytes*2 HALF ELEMENTS (bytes*4 physical bytes).
// Layout: [transform=ky*4+kx][ic][oc], at half offset p.weights[0]*2.
// The valid source span must not alias packed or quarterBound storage.
// An empty buffer is zero-initialized only after the gate succeeds; later
// calls require that same size and preserve other regions. Raw first-conv
// regions must be disjoint (or identical at the same offset): each 2304-byte
// region reserves 4608 halves, of which integer T4=4U uses 4096. Arbitrary
// overlapping raw shapes cannot be detected from this stateless API.
//
// False never writes the candidate region or changes packed. Use the bool,
// not the buffer's contents, to select Winograd versus direct SIMD. A failed
// re-prepare can leave an older region, which must not be used for that pass.
// quarterBound is zero for invalid arguments, otherwise the maximum bound
// on scaled-by-four inverse partials (also returned on a lossless-gate reject).
// Allocation uses std::vector's normal exception behavior. INT8 stays read-only.
inline bool prepare_simd_winograd_weights(const uint8_t *weights, size_t bytes,
                                         const PassParams &p,
                                         std::vector<_Float16> &packed,
                                         uint32_t *quarterBound = nullptr) {
    static_assert(sizeof(_Float16) == 2, "Winograd storage requires binary16");
    static_assert(std::numeric_limits<float>::is_iec559 &&
                  std::numeric_limits<float>::radix == 2 &&
                  std::numeric_limits<float>::digits == 24,
                  "Winograd exactness gate requires IEEE binary32");
    constexpr size_t channels = 16, matrix = channels * channels;
    constexpr size_t rawBytes = 9 * matrix, transformedElements = 16 * matrix;
    constexpr uint64_t exactLimit = uint64_t(1) << 24;
    if (quarterBound) *quarterBound = 0;
    if (!weights || (p.input.channels != 16 && p.input.channels != 32) ||
        p.weights[0] == kNoOffset || p.weights[1] == kNoOffset ||
        p.weights[2] == kNoOffset) return false;

    // Check both the element count and physical byte size before multiplying,
    // allocating, or reading. Subtraction also avoids offset+2304 overflow.
    if (bytes > std::numeric_limits<size_t>::max() / (2 * sizeof(_Float16))) return false;
    const size_t elements = bytes * 2;
    if (elements > packed.max_size() || (!packed.empty() && packed.size() != elements) ||
        p.weights[0] > bytes) return false; // Compare before any narrowing to size_t.
    const size_t offset = size_t(p.weights[0]);
    if (rawBytes > bytes - offset) return false;
    const size_t destination = offset * 2;
    if (destination > elements || transformedElements > elements - destination) return false;

    // H = 2G. T4 = H*g*H^T = 4U is integer. Every H row has l1 norm
    // <=3, so |T4| <= 3*128*3 = 1152 < 2048: EVERY T4 is exact in half.
    // Store T4, not U=T4/4; the GPU divides only after the complete inverse.
    constexpr int h[4][3] = {{2, 0, 0}, {1, 1, 1}, {1, -1, 1}, {0, 0, 2}};
    std::array<int16_t, transformedElements> t4;
    const uint8_t *raw = weights + offset;
    for (size_t ic = 0; ic < channels; ++ic) {
        for (size_t oc = 0; oc < channels; ++oc) {
            int g[3][3], hg[4][3] = {};
            for (size_t y = 0; y < 3; ++y)
                for (size_t x = 0; x < 3; ++x) {
                    const int byte = raw[(y * 3 + x) * matrix + ic * channels + oc];
                    g[y][x] = byte < 128 ? byte : byte - 256;
                }
            for (size_t y = 0; y < 4; ++y)
                for (size_t x = 0; x < 3; ++x)
                    for (size_t k = 0; k < 3; ++k) hg[y][x] += h[y][k] * g[k][x];
            for (size_t y = 0; y < 4; ++y)
                for (size_t x = 0; x < 4; ++x) {
                    int value = 0;
                    for (size_t k = 0; k < 3; ++k) value += hg[y][k] * h[x][k];
                    t4[(y * 4 + x) * matrix + ic * channels + oc] = int16_t(value);
                }
        }
    }

    // Each V = B^T*d*B sums four signed INT8 values, so |V|<=512 and
    // V is an exact half integer. A^T = {{1,1,1,0},{0,1,-1,-1}} selects
    // 3x3 transform components per output quarter, with signs of magnitude 1.
    // With T4 storage, dot products and inverse partials are integers scaled
    // by four. 512*sum_ic,selected |T4| bounds every dot partial and every
    // streaming inverse partial, including any association order. STRICTLY
    // <2^24 makes them exact fp32 integers. Every transform component belongs
    // to a quarter, so this covers all dots, not just the final four outputs.
    // The full inverse is 4*directConv: multiplying by 0.25f ONCE is exact,
    // before integer conversion and the unchanged INT32 bias/epilogue.
    uint64_t maximum = 0;
    for (size_t oc = 0; oc < channels; ++oc)
        for (size_t oy = 0; oy < 2; ++oy)
            for (size_t ox = 0; ox < 2; ++ox) {
                uint64_t sum = 0;
                for (size_t y = oy; y < oy + 3; ++y)
                    for (size_t x = ox; x < ox + 3; ++x)
                        for (size_t ic = 0; ic < channels; ++ic) {
                            const int value = t4[(y * 4 + x) * matrix + ic * channels + oc];
                            sum += uint64_t(value < 0 ? -value : value);
                        }
                const uint64_t bound = 512 * sum;
                if (bound > maximum) maximum = bound;
            }
    // Even the conservative global maximum 512*16*9*1152 fits uint32_t.
    if (quarterBound) *quarterBound = uint32_t(maximum);
    if (maximum >= exactLimit) return false;

    // Publish only a fully validated candidate. The extra 512 half elements
    // at the end of its reserved region stay zero on first initialization.
    if (packed.empty()) packed.assign(elements, static_cast<_Float16>(0));
    for (size_t i = 0; i < transformedElements; ++i)
        packed[destination + i] = static_cast<_Float16>(t4[i]);
    return true;
}

} // namespace fsr411
