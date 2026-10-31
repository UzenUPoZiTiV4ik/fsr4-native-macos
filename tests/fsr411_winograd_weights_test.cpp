// SPDX-License-Identifier: MIT
// CPU-only: clang++ -std=c++17 -O2 -Wall -Wextra -Werror
//   -fno-fast-math -ffp-contract=off tests/fsr411_winograd_weights_test.cpp -o /tmp/fsr411_winograd_test
#include "../src/fsr411_simd_weights.h"
#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <limits>

namespace {
constexpr size_t C = 16, M = C * C, RAW = 9 * M, TRANSFORMED = 16 * M;
constexpr uint32_t LIMIT = uint32_t(1) << 24;
using T4 = std::array<int, TRANSFORMED>; // T4=4U, stored as integer half.
using Kernel = std::array<int, 9>;
unsigned long checks = 0;
uint32_t randomState = 0x41120303u;

void require(bool condition, const char *label) {
    ++checks;
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", label);
        std::exit(1);
    }
}
void equal(double actual, double expected, const char *label) {
    ++checks;
    if (actual != expected) {
        std::fprintf(stderr, "FAIL: %s: got %.17g, expected %.17g\n", label, actual, expected);
        std::exit(1);
    }
}
uint32_t random_bits() {
    randomState ^= randomState << 13;
    randomState ^= randomState >> 17;
    randomState ^= randomState << 5;
    return randomState;
}
int signed_byte(uint8_t byte) { return byte < 128 ? int(byte) : int(byte) - 256; }
size_t raw_index(size_t tap, size_t ic, size_t oc) { return tap * M + ic * C + oc; }
fsr411::PassParams pass(uint32_t offset = 0, uint32_t nf = 16) {
    fsr411::PassParams p = {};
    p.input.channels = nf;
    p.output.channels = nf;
    p.weights[0] = offset;
    // Only the non-DS discriminator is relevant to this first-conv helper;
    // no pointwise or bias region is read or rewritten by these tests.
    p.weights[1] = p.weights[2] = 0;
    p.weights[3] = fsr411::kNoOffset;
    return p;
}
void set_kernel(std::vector<uint8_t> &raw, size_t offset, size_t ic, size_t oc, const Kernel &g) {
    for (size_t tap = 0; tap < 9; ++tap)
        raw[offset + raw_index(tap, ic, oc)] = static_cast<uint8_t>(g[tap]);
}

// Independent tensor-product oracle: nine integer terms, not the production
// two-stage transform. T4=4U verifies integer half storage and recovered
// quarter rationals.
T4 reference_t4(const std::vector<uint8_t> &raw, size_t offset) {
    constexpr int H[4][3] = {{2,0,0},{1,1,1},{1,-1,1},{0,0,2}};
    T4 result = {};
    for (size_t ty = 0; ty < 4; ++ty) for (size_t tx = 0; tx < 4; ++tx)
        for (size_t ic = 0; ic < C; ++ic) for (size_t oc = 0; oc < C; ++oc)
            for (size_t y = 0; y < 3; ++y) for (size_t x = 0; x < 3; ++x)
                result[(ty * 4 + tx) * M + ic * C + oc] +=
                    H[ty][y] * signed_byte(raw[offset + raw_index(y * 3 + x, ic, oc)]) * H[tx][x];
    return result;
}
uint32_t reference_bound(const T4 &t4) {
    // Deliberately select by A signs, rather than the helper's index ranges.
    constexpr int A[2][4] = {{1,1,1,0},{0,1,-1,-1}};
    uint64_t maximum = 0;
    for (size_t oc = 0; oc < C; ++oc) for (size_t oy = 0; oy < 2; ++oy)
        for (size_t ox = 0; ox < 2; ++ox) {
            uint64_t sum = 0;
            for (size_t k = 0; k < 16; ++k) if (A[oy][k / 4] && A[ox][k % 4])
                for (size_t ic = 0; ic < C; ++ic)
                    sum += uint64_t(std::abs(t4[k * M + ic * C + oc]));
            maximum = std::max(maximum, sum * 512);
        }
    require(maximum <= std::numeric_limits<uint32_t>::max(), "diagnostic bound fits u32");
    return uint32_t(maximum);
}
void check_pack(const std::vector<_Float16> &packed, size_t offset, const T4 &t4) {
    for (size_t i = 0; i < TRANSFORMED; ++i) {
        require(std::abs(t4[i]) <= 1152, "integer transform range");
        equal(double(packed[offset * 2 + i]), t4[i], "half equals exact integer T4");
        equal(double(float(packed[offset * 2 + i]) * 0.25f), double(t4[i]) * 0.25, "recovered U is an exact quarter rational");
    }
}

void known_transforms_and_signed_bytes() {
    constexpr size_t offset = 13; // No alignment or pointer-punning assumption.
    std::vector<uint8_t> raw(offset + RAW + 37, 0xff);
    std::fill(raw.begin() + offset, raw.begin() + offset + RAW, 0);
    set_kernel(raw, offset, 2, 7, {{1,2,3,4,5,6,7,8,9}});
    set_kernel(raw, offset, 7, 1, {{0,0,0,0,1,0,0,0,0}});
    set_kernel(raw, offset, 13, 4, {{-128,127,-1,0,1,-64,63,-127,126}});
    set_kernel(raw, offset, 15, 15, {{-128,-128,-128,-128,-128,-128,-128,-128,-128}});
    set_kernel(raw, offset, 0, 15, {{127,127,127,127,127,127,127,127,127}});
    const auto original = raw;
    std::vector<_Float16> packed;
    uint32_t bound = 0;
    require(fsr411::prepare_simd_winograd_weights(raw.data(), raw.size(), pass(offset, 32), packed, &bound),
            "NF32 first bank accepted; trailing bytes are not a second conv bank");
    require(raw == original, "native INT8 model unchanged");
    require(packed.size() == raw.size() * 2 && packed.size() * sizeof(_Float16) == raw.size() * 4,
            "element count is twice bytes, physical size is four times bytes");
    const T4 t4 = reference_t4(raw, offset);
    check_pack(packed, offset, t4);
    equal(bound, reference_bound(t4), "maximum bound oracle");
    constexpr int known[16] = {4,12,4,12,24,45,15,36,8,15,5,12,28,48,16,36};
    constexpr int centre[16] = {0,0,0,0,0,1,-1,0,0,-1,1,0,0,0,0,0};
    for (size_t k = 0; k < 16; ++k) {
        equal(double(packed[offset * 2 + k * M + 2 * C + 7]), known[k], "hand-computed 1..9 layout");
        equal(double(packed[offset * 2 + k * M + 7 * C + 1]), centre[k], "centre basis signs");
    }
    equal(double(packed[offset * 2 + 5 * M + 15 * C + 15]), -1152, "signed -128 maximum integer transform");
    equal(double(packed[offset * 2 + 5 * M + 15]), 1143, "signed +127 integer transform endpoint");
    equal(double(float(packed[offset * 2 + 5 * M + 15]) * 0.25f), 285.75, "signed +127 recovered quarter-grid endpoint");
    for (size_t i = 0; i < packed.size(); ++i)
        if (i < offset * 2 || i >= offset * 2 + TRANSFORMED)
            equal(double(packed[i]), 0, "initial prefix, reserved tail and unrelated bytes are zero");

    raw.assign(RAW, 0);
    packed.clear();
    for (int value = -128; value <= 127; ++value) {
        Kernel g;
        g.fill(value);
        set_kernel(raw, 0, 5, 11, g);
        require(fsr411::prepare_simd_winograd_weights(raw.data(), raw.size(), pass(), packed), "all signed byte values accepted sparsely");
        check_pack(packed, 0, reference_t4(raw, 0));
    }
}

void invalid_arguments_and_sizes() {
    std::vector<uint8_t> raw(RAW, 0);
    std::vector<_Float16> packed;
    const auto reject = [&](const uint8_t *data, size_t bytes, const fsr411::PassParams &p) {
        const auto previous = packed;
        uint32_t bound = 123;
        require(!fsr411::prepare_simd_winograd_weights(data, bytes, p, packed, &bound), "invalid arguments rejected");
        require(packed == previous, "invalid arguments do not publish or erase any region");
        equal(bound, 0, "invalid argument diagnostic reset");
    };
    reject(nullptr, RAW, pass());
    reject(nullptr, 0, pass());
    for (size_t bytes : {size_t(0), size_t(1), RAW - 1}) reject(raw.data(), bytes, pass());
    for (uint32_t nf : {0u,8u,15u,17u,31u,33u,48u,64u,0xffffffffu}) reject(raw.data(), RAW, pass(0, nf));
    for (uint32_t nf : {16u,32u}) for (size_t slot = 0; slot < 3; ++slot) {
        auto p = pass(0, nf); p.weights[slot] = fsr411::kNoOffset;
        reject(raw.data(), RAW, p); // Includes DS16/DS32 even though their NF matches.
    }
    reject(raw.data(), RAW, pass(1));
    reject(raw.data(), RAW, pass(uint32_t(RAW)));
    reject(raw.data(), RAW, pass(uint32_t(RAW + 1)));
    reject(raw.data(), RAW, pass(0xfffffffeu)); // offset+2304 would wrap in uint32_t.
    const size_t maximum = std::numeric_limits<size_t>::max();
    reject(raw.data(), maximum, pass());
    reject(raw.data(), maximum / 2 + 1, pass());
    reject(raw.data(), maximum / (2 * sizeof(_Float16)) + 1, pass());
    reject(raw.data(), packed.max_size() / 2 + 1, pass());
    packed.assign(3, static_cast<_Float16>(7));
    reject(raw.data(), RAW, pass()); // An existing buffer cannot silently change model size.
    packed.clear();
    require(fsr411::prepare_simd_winograd_weights(raw.data(), RAW, pass(), packed), "exactly 2304 source bytes accepted");
    std::vector<uint8_t> unaligned(RAW + 1, 0);
    std::vector<_Float16> other;
    require(fsr411::prepare_simd_winograd_weights(unaligned.data(), unaligned.size(), pass(1), other), "exact ending offset and unaligned source accepted");
}

void shared_regions() {
    constexpr size_t first = 64, second = first + RAW;
    std::vector<uint8_t> raw(second + RAW + 64, 0);
    set_kernel(raw, first, 1, 14, {{1,2,-3,4,0,6,-7,8,9}});
    set_kernel(raw, second, 14, 1, {{0,0,0,0,-3,0,0,0,0}});
    const auto original = raw;
    std::vector<_Float16> packed;
    require(fsr411::prepare_simd_winograd_weights(raw.data(), raw.size(), pass(first), packed), "first shared region");
    const auto afterFirst = packed;
    require(fsr411::prepare_simd_winograd_weights(raw.data(), raw.size(), pass(second, 32), packed), "adjacent raw region with independent NF32 metadata");
    for (size_t i = 0; i < TRANSFORMED; ++i)
        equal(double(packed[first * 2 + i]), double(afterFirst[first * 2 + i]), "second call preserves first region");
    check_pack(packed, second, reference_t4(raw, second));
    for (size_t i = 0; i < packed.size(); ++i)
        if (!((i >= first * 2 && i < first * 2 + TRANSFORMED) ||
              (i >= second * 2 && i < second * 2 + TRANSFORMED)))
            equal(double(packed[i]), 0, "shared buffer padding stays zero");
    const auto completed = packed;
    require(fsr411::prepare_simd_winograd_weights(raw.data(), raw.size(), pass(first, 32), packed), "identical raw offset may be reused");
    require(packed == completed, "repeat preparation is idempotent and does not reinitialize");
    require(raw == original, "shared preparation never edits source weights");
    std::fill(raw.begin() + second, raw.begin() + second + RAW, uint8_t(128));
    uint32_t bound = 0;
    require(!fsr411::prepare_simd_winograd_weights(raw.data(), raw.size(), pass(second), packed, &bound), "unsafe re-prepare rejected");
    require(packed == completed && bound >= LIMIT, "rejected candidate leaves all old regions untouched; caller must honor false");
}

void exact_gate() {
    std::vector<uint8_t> raw(RAW, 0);
    std::vector<_Float16> packed;
    uint32_t bound = 123;
    require(fsr411::prepare_simd_winograd_weights(raw.data(), RAW, pass(), packed, &bound), "zero model accepted");
    equal(bound, 0, "zero model bound");
    for (size_t i = 0; i < RAW; ++i) raw[i] = static_cast<uint8_t>(int((i * 17 + 5) % 7) - 3);
    require(fsr411::prepare_simd_winograd_weights(raw.data(), RAW, pass(), packed, &bound), "representative small dense signed model accepted");
    equal(bound, reference_bound(reference_t4(raw, 0)), "small model maximum bound");
    require(bound < LIMIT, "small model safely below exact limit");
    for (uint8_t byte : {uint8_t(128), uint8_t(127)}) {
        raw.assign(RAW, byte);
        packed.clear();
        require(!fsr411::prepare_simd_winograd_weights(raw.data(), RAW, pass(), packed, &bound), "adversarial dense endpoint model rejected");
        equal(bound, reference_bound(reference_t4(raw, 0)), "unsafe model still returns exact diagnostic bound");
        require(bound > LIMIT && packed.empty(), "unsafe first call allocates and publishes nothing");
        packed.assign(RAW * 2, static_cast<_Float16>(0));
        const auto zero = packed;
        require(!fsr411::prepare_simd_winograd_weights(raw.data(), RAW, pass(), packed), "unsafe model rejected without diagnostic pointer");
        require(packed == zero, "unsafe candidate region stays zero");
    }
    // Opposite centre sign defeats the tempting but WRONG assumption that
    // uniform endpoint kernels maximize each quarter: here the per-ic sum is
    // 4862, not 36*128. Seven such channels are unsafe; six remain safe.
    raw.assign(RAW, 0);
    for (size_t tap = 0; tap < 9; ++tap) for (size_t ic = 0; ic < 7; ++ic)
        for (size_t oc = 0; oc < C; ++oc) raw[raw_index(tap, ic, oc)] = uint8_t(tap == 4 ? 127 : 128);
    packed.clear();
    require(!fsr411::prepare_simd_winograd_weights(raw.data(), RAW, pass(), packed, &bound), "adversarial opposite-centre model rejected");
    equal(bound, uint64_t(4862) * 7 * 512, "real component magnitudes, not a uniform-kernel shortcut");
    require(packed.empty(), "opposite-centre candidate never published");
    for (size_t tap = 0; tap < 9; ++tap) for (size_t oc = 0; oc < C; ++oc) raw[raw_index(tap, 6, oc)] = 0;
    require(fsr411::prepare_simd_winograd_weights(raw.data(), RAW, pass(), packed, &bound), "six opposite-centre channels accepted");
    equal(bound, uint64_t(4862) * 6 * 512, "safe opposite-centre bound");
    // A corner-only kernel has quarter l1 factor 16, but each individual dot
    // has a much smaller bound. This detects dot-only gates and <= mistakes.
    // Rotate the corner to make EACH of the four output quarters the maximum,
    // in the last output channel to catch partial channel validation too.
    for (size_t oy = 0; oy < 2; ++oy) for (size_t ox = 0; ox < 2; ++ox) {
        raw.assign(RAW, 0);
        const size_t tap = oy * 6 + ox * 2;
        for (size_t ic = 0; ic < C; ++ic) raw[raw_index(tap, ic, 15)] = 128;
        packed.clear();
        require(!fsr411::prepare_simd_winograd_weights(raw.data(), RAW, pass(), packed, &bound), "strict boundary rejected in every quarter");
        equal(bound, LIMIT, "exactly 2^24 is unsafe");
        require(packed.empty(), "boundary candidate not published");
        raw[raw_index(tap, 0, 15)] = 129; // -127 instead of -128.
        require(fsr411::prepare_simd_winograd_weights(raw.data(), RAW, pass(), packed, &bound), "nearest attainable corner model below limit accepted");
        equal(bound, LIMIT - 8192, "exact bound immediately below corner threshold");
        check_pack(packed, 0, reference_t4(raw, 0));
    }
}

int sample(const std::vector<int> &image, int width, int height, int nf, int x, int y, int ic) {
    return x < 0 || y < 0 || x >= width || y >= height ? 0 : image[(y * width + x) * nf + ic];
}
int direct_conv(const std::vector<uint8_t> &raw, size_t offset, const std::vector<int> &image,
                int width, int height, int nf, int x, int y, size_t oc) {
    int result = 0;
    for (size_t ky = 0; ky < 3; ++ky) for (size_t kx = 0; kx < 3; ++kx)
        for (size_t ic = 0; ic < C; ++ic)
            result += sample(image, width, height, nf, x + int(kx) - 1, y + int(ky) - 1, int(ic)) *
                      signed_byte(raw[offset + raw_index(ky * 3 + kx, ic, oc)]);
    return result; // Worst direct sum is 16*9*128*128, safely inside int32.
}

void compare_image(const std::vector<uint8_t> &raw, const fsr411::PassParams &p,
                   const std::vector<int> &image, int width, int height) {
    std::vector<_Float16> packed;
    uint32_t bound = 0;
    require(fsr411::prepare_simd_winograd_weights(raw.data(), raw.size(), p, packed, &bound), "test convolution model must pass lossless gate");
    const T4 t4 = reference_t4(raw, p.weights[0]);
    equal(bound, reference_bound(t4), "convolution model bound oracle");
    check_pack(packed, p.weights[0], t4);
    constexpr int BT[4][4] = {{1,0,-1,0},{0,1,1,0},{0,-1,1,0},{0,1,0,-1}};
    constexpr int A[2][4] = {{1,1,1,0},{0,1,-1,-1}};
    for (int y0 = 0; y0 < height; y0 += 2) for (int x0 = 0; x0 < width; x0 += 2) {
        _Float16 v[16][16];
        int vi[16][16] = {};
        for (size_t ic = 0; ic < C; ++ic) {
            int d[4][4];
            _Float16 bt[4][4];
            for (size_t y = 0; y < 4; ++y) for (size_t x = 0; x < 4; ++x)
                d[y][x] = sample(image, width, height, int(p.input.channels), x0 + int(x) - 1, y0 + int(y) - 1, int(ic));
            for (size_t x = 0; x < 4; ++x) {
                bt[0][x] = static_cast<_Float16>(d[0][x] - d[2][x]);
                bt[1][x] = static_cast<_Float16>(d[1][x] + d[2][x]);
                bt[2][x] = static_cast<_Float16>(d[2][x] - d[1][x]);
                bt[3][x] = static_cast<_Float16>(d[1][x] - d[3][x]);
            }
            for (size_t y = 0; y < 4; ++y) {
                v[y * 4][ic] = static_cast<_Float16>(float(bt[y][0]) - float(bt[y][2]));
                v[y * 4 + 1][ic] = static_cast<_Float16>(float(bt[y][1]) + float(bt[y][2]));
                v[y * 4 + 2][ic] = static_cast<_Float16>(float(bt[y][2]) - float(bt[y][1]));
                v[y * 4 + 3][ic] = static_cast<_Float16>(float(bt[y][1]) - float(bt[y][3]));
                for (size_t x = 0; x < 4; ++x) {
                    for (size_t dy = 0; dy < 4; ++dy) for (size_t dx = 0; dx < 4; ++dx)
                        vi[y * 4 + x][ic] += BT[y][dy] * d[dy][dx] * BT[x][dx];
                    require(std::abs(vi[y * 4 + x][ic]) <= 512, "four-term input transform bound");
                    equal(double(v[y * 4 + x][ic]), vi[y * 4 + x][ic], "two-stage half input transform exact");
                }
            }
        }
        // Stream each transform's dot into its selected inverse quarters.
        // Two opposite dot/component orders test both association and signs.
        for (size_t order = 0; order < 2; ++order) {
            float out[4][16] = {};
            int64_t inverse4[4][16] = {};
            for (size_t t = 0; t < 16; ++t) {
                const size_t k = order ? 15 - t : t;
                for (size_t oc = 0; oc < C; ++oc) {
                    float dot = 0;
                    int64_t dot4 = 0;
                    for (size_t i = 0; i < C; ++i) {
                        const size_t ic = order ? C - 1 - i : i;
                        dot += float(v[k][ic]) * float(packed[size_t(p.weights[0]) * 2 + k * M + ic * C + oc]);
                        dot4 += int64_t(vi[k][ic]) * t4[k * M + ic * C + oc];
                        equal(double(dot), double(dot4), "every scaled-by-four float dot partial is an exact integer");
                    }
                    for (size_t oy = 0; oy < 2; ++oy) for (size_t ox = 0; ox < 2; ++ox) {
                        const int sign = A[oy][k / 4] * A[ox][k % 4];
                        if (!sign) continue;
                        const size_t q = oy * 2 + ox;
                        out[q][oc] += float(sign) * dot;
                        inverse4[q][oc] += sign * dot4;
                        equal(double(out[q][oc]), double(inverse4[q][oc]), "every scaled-by-four streaming inverse partial exact");
                    }
                }
            }
            for (size_t oy = 0; oy < 2; ++oy) for (size_t ox = 0; ox < 2; ++ox)
                for (size_t oc = 0; oc < C; ++oc) {
                    const size_t q = oy * 2 + ox;
                    const int direct = direct_conv(raw, p.weights[0], image, width, height, int(p.input.channels), x0 + int(ox), y0 + int(oy), oc);
                    equal(double(inverse4[q][oc]), int64_t(direct) * 4, "integer Winograd numerator equals four times direct conv");
                    equal(double(out[q][oc]), int64_t(direct) * 4, "raw float inverse equals four times direct convolution");
                    const float result = out[q][oc] * 0.25f; // Divide only after the full inverse.
                    equal(double(result), direct, "ALL four output quarters exactly equal integer 3x3 convolution");
                    equal(int(result), direct, "integer conversion is lossless, not hiding a fractional error");
                }
        }
    }
}

void convolution_properties() {
    std::vector<uint8_t> raw(RAW, 0);
    // Dense small signed weights plus full-range random signed INT8 inputs.
    // Every geometry exercises zero-padding; odd sizes still check all four
    // inverse quarters, including quarters the GPU would decline to store.
    for (int trial = 0; trial < 64; ++trial) {
        for (auto &byte : raw) byte = static_cast<uint8_t>(int(random_bits() % 17) - 8);
        const int width = int(random_bits() % 7) + 1, height = int(random_bits() % 7) + 1;
        const auto p = pass(0, trial % 2 ? 32 : 16);
        std::vector<int> image(size_t(width * height) * p.input.channels);
        for (auto &value : image) value = int(random_bits() & 255) - 128;
        compare_image(raw, p, image, width, height);
    }
    // Three active channels permit ALL signed INT8 weights: the three
    // selected H row norms sum to 8, so sum_selected |T4| <=8*8*128 per ic.
    // The coarse bound 512*3*64*128 <2^24 guarantees these random models pass.
    for (int trial = 0; trial < 16; ++trial) {
        raw.assign(RAW, 0);
        for (size_t tap = 0; tap < 9; ++tap) for (size_t ic = 0; ic < 3; ++ic)
            for (size_t oc = 0; oc < C; ++oc) raw[raw_index(tap, ic, oc)] = uint8_t(random_bits() & 255);
        std::vector<int> image(5 * 3 * 32);
        for (auto &value : image) value = int(random_bits() & 255) - 128;
        compare_image(raw, pass(0, 32), image, 5, 3);
    }
    // For UNIFORM g only, H row sums are {2,3,1,2}, giving quarter norm
    // 6*6*|weight|. Thus seven uniform endpoint channels are safely accepted.
    for (int weight : {-128,127}) for (int input : {-128,127}) {
        raw.assign(RAW, 0);
        for (size_t tap = 0; tap < 9; ++tap) for (size_t ic = 0; ic < 7; ++ic)
            for (size_t oc = 0; oc < C; ++oc) raw[raw_index(tap, ic, oc)] = static_cast<uint8_t>(weight);
        for (const auto &shape : {std::array<int,2>{{1,1}}, {{1,2}}, {{2,1}}, {{2,2}}, {{5,3}}}) {
            std::vector<int> image(size_t(shape[0] * shape[1]) * 16, input);
            compare_image(raw, pass(), image, shape[0], shape[1]);
        }
    }
    // Near-threshold corner weights, negative/positive endpoints and a
    // checkerboard reach large transform values and cancellation partials.
    for (size_t corner : {size_t(0),size_t(2),size_t(6),size_t(8)}) {
        raw.assign(RAW, 0);
        for (size_t ic = 0; ic < C; ++ic) for (size_t oc = 0; oc < C; ++oc)
            raw[raw_index(corner, ic, oc)] = uint8_t(ic ? 128 : 129);
        std::vector<int> image(7 * 5 * 16);
        for (size_t i = 0; i < image.size(); ++i) image[i] = (i / 16 + i % 16) % 2 ? 127 : -128;
        compare_image(raw, pass(), image, 7, 5);
    }
    // A safe, almost worst-case opposite-centre model, not uniform weights.
    raw.assign(RAW, 0);
    for (size_t tap = 0; tap < 9; ++tap) for (size_t ic = 0; ic < 6; ++ic)
        for (size_t oc = 0; oc < C; ++oc) raw[raw_index(tap, ic, oc)] = uint8_t(tap == 4 ? 127 : 128);
    std::vector<int> image(5 * 5 * 32);
    for (auto &value : image) value = random_bits() % 2 ? -128 : 127;
    compare_image(raw, pass(0, 32), image, 5, 5);
}
} // namespace

int main() {
    known_transforms_and_signed_bytes();
    invalid_arguments_and_sizes();
    shared_regions();
    exact_gate();
    convolution_properties();
    std::printf("fsr411_winograd_weights_test: %lu exact checks passed (CPU only)\n", checks);
}
