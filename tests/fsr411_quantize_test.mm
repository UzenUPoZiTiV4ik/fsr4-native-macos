// SPDX-License-Identifier: MIT
// Compare the production Metal int4 quantizer and both CNB specializations
// with the original scalar CPU finish() contract. Runtime buffers prevent
// constant-folding the test cases.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "../src/fsr411_model.h"
#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

struct alignas(16) QuantizeCase {
    int32_t accumulator[4], bias[4], residual[4];
    fsr411::Epilogue q;
};
static_assert(sizeof(QuantizeCase) == 64 && offsetof(QuantizeCase, q) == 48,
              "Metal int4 quantization probe ABI");

static void require(bool ok, const std::string &why) {
    if (!ok) throw std::runtime_error(why);
}
static std::string metal_error(NSError *error) {
    return error ? error.localizedDescription.UTF8String : "Metal allocation failed";
}

// Pin representative original scalar semantics independently of random input
// generation, including wrap during rounding and narrow-before-saturation.
static void check_scalar_contract() {
    using namespace fsr411;
    struct Check { int32_t acc, bias, residual; Epilogue q; int32_t expected; };
    const Check checks[] = {
        {INT32_MAX, 1, 0, {0, 0, 0, 0}, -128},
        {INT32_MIN, -1, 0, {0, 0, 0, 0}, 127},
        {0x40000000, 0, 0, {1, 0, 0, 0}, -128},
        {INT32_MAX, 0, 1, {0, 0, 0, 0}, -128},
        {5, 0, -1, {0, 2, 0, 0}, 1},
        {3, 0, 0, {0, 0, 1, 0}, 2},
        {5, 0, 0, {0, 0, 1, 0}, 2},
        {-3, 0, 0, {0, 0, 1, 0}, -2},
        {-5, 0, 0, {0, 0, 1, 0}, -2},
        {-3, 0, 0, {0, 0, 1, kRoundHalfUp}, -1},
        {INT32_MAX, 0, 0, {0, 0, 1, 0}, -128},
        {32768, 0, 0, {0, 0, 0, kNarrowInt16}, -128},
        {65535, 0, 0, {0, 0, 0, kNarrowInt16}, -1},
        {65536, 0, 0, {0, 0, 0, kNarrowInt16}, 0},
        {-32769, 0, 0, {0, 0, 0, kNarrowInt16}, 127},
        {-1, 0, 0, {0, 0, 0, kReluBeforeRound}, 0},
        {INT32_MIN, 0, 0, {0, 0, 31, 0}, -1},
        {0x40000000, 0, 0, {0, 0, 31, 0}, 0},
        {0x40000000, 0, 0, {0, 0, 31, kRoundHalfUp}, -1},
    };
    for (size_t i = 0; i < std::size(checks); ++i) {
        const auto &c = checks[i];
        require(finish(c.acc, c.bias, c.residual, c.q) == c.expected,
                "original scalar quantization contract changed at case " + std::to_string(i));
    }
}

static std::vector<QuantizeCase> test_cases() {
    std::vector<QuantizeCase> result;
    result.reserve(260000);
    uint32_t seed = 0x4114242u;
    auto random = [&]() { seed = seed * 1664525u + 1013904223u; return seed; };
    const int32_t endpoints[] = {0, 1, -1, 126, 127, 128, -127, -128, -129,
        32767, 32768, -32768, -32769, 65535, 65536, -65536, INT32_MAX, INT32_MIN};
    const int32_t residuals[] = {-128, -127, -1, 0, 1, 126, 127};

    // Exhaust the Cartesian product of every model-valid left-shift pair,
    // every flag combination and every right shift. Right shift 31 is an
    // additional synthetic arithmetic boundary; valid_epilogue stays <= 30.
    for (uint32_t flags = 0; flags < 8; ++flags)
        for (uint32_t right = 0; right < 32; ++right)
            for (uint32_t accShift = 0; accShift <= 30; ++accShift)
                for (uint32_t residualShift = 0; residualShift <= 30; ++residualShift) {
                    QuantizeCase c = {};
                    c.q = {accShift, residualShift, right, flags};
                    for (uint32_t lane = 0; lane < 4; ++lane) {
                        c.accumulator[lane] = lane & 1u ? fsr411::signed_bits(random())
                            : endpoints[random() % std::size(endpoints)];
                        c.bias[lane] = lane & 2u ? fsr411::signed_bits(random())
                            : endpoints[random() % std::size(endpoints)];
                        c.residual[lane] = lane & 1u ? int32_t(random() % 256u) - 128
                            : residuals[random() % std::size(residuals)];
                    }
                    result.push_back(c);
                }

    // Half-way values and their neighbours, for positive/negative even/odd
    // quotients and int8/int16 saturation boundaries. Unsigned construction
    // deliberately retains the shader's wrapping int32 endpoint semantics.
    const int32_t quotients[] = {-65537, -65536, -32769, -32768, -129, -128,
        -127, -3, -2, -1, 0, 1, 2, 3, 126, 127, 128, 129, 32767, 32768, 65535, 65536};
    for (uint32_t flags = 0; flags < 8; ++flags)
        for (uint32_t right = 0; right < 32; ++right)
            for (int32_t quotient : quotients) {
                QuantizeCase c = {};
                c.q = {0, 0, right, flags};
                const uint32_t base = uint32_t(quotient) << right;
                const uint32_t half = right ? 1u << (right - 1u) : 0u;
                c.accumulator[0] = fsr411::signed_bits(base + half - 1u);
                c.accumulator[1] = fsr411::signed_bits(base + half);
                c.accumulator[2] = fsr411::signed_bits(base + half + 1u);
                c.accumulator[3] = fsr411::signed_bits(0u - base - half);
                result.push_back(c);
            }

    // Keep the dispatch tail partial, and pin cross-lane distinctions that
    // detect accidentally broadcasting any accumulator, bias or residual.
    result.push_back({{INT32_MAX, INT32_MIN, 5, -5}, {1, -1, 0, 0}, {0, 0, -1, 1}, {0, 0, 1, 0}});
    result.push_back({{32767, 32768, 65535, 65536}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, fsr411::kNarrowInt16}});
    result.push_back({{-128, -1, 1, 127}, {0, 0, 0, 0}, {127, 1, -1, -128}, {0, 2, 0, fsr411::kReluBeforeRound}});
    return result;
}

// Count eligibility without duplicating any quantization arithmetic. The
// scalar oracle always uses the supplied flags, even when the template's
// ReLU setting differs and the production helper must take its fallback.
struct CnbCoverage {
    size_t even = 0, halfUp = 0, shiftOnly = 0, flagsOnly = 0, shiftAndFlags = 0;
    uint32_t evenRightMask = 0, halfUpRightMask = 0;
    uint32_t evenResidualMask = 0, halfUpResidualMask = 0;
    uint32_t fallbackFlagMask = 0, fallbackShiftMask = 0;

    void record(fsr411::Epilogue q, bool relu) {
        const uint32_t flags = relu ? fsr411::kReluBeforeRound : 0u;
        const bool matchingFlags = q.flags == flags || q.flags == (flags | fsr411::kRoundHalfUp);
        if (q.accumulatorShift) {
            if (matchingFlags) { ++shiftOnly; fallbackShiftMask |= 1u << q.accumulatorShift; }
            else ++shiftAndFlags;
        } else if (!matchingFlags) {
            ++flagsOnly;
            fallbackFlagMask |= 1u << q.flags;
        } else if (q.flags & fsr411::kRoundHalfUp) {
            ++halfUp;
            halfUpRightMask |= 1u << q.rightShift;
            halfUpResidualMask |= 1u << q.residualShift;
        } else {
            ++even;
            evenRightMask |= 1u << q.rightShift;
            evenResidualMask |= 1u << q.residualShift;
        }
    }

    void check(bool relu, size_t count) const {
        const uint32_t flags = relu ? fsr411::kReluBeforeRound : 0u;
        const uint32_t otherFlagMask = 255u & ~((1u << flags) | (1u << (flags | fsr411::kRoundHalfUp)));
        require(even && halfUp && shiftOnly && flagsOnly && shiftAndFlags
            && even + halfUp + shiftOnly + flagsOnly + shiftAndFlags == count
            && evenRightMask == UINT32_MAX && halfUpRightMask == UINT32_MAX
            && evenResidualMask == 0x7fffffffu && halfUpResidualMask == 0x7fffffffu
            && fallbackFlagMask == otherFlagMask && fallbackShiftMask == 0x7ffffffeu,
            std::string("CNB ") + (relu ? "ReLU" : "linear") + " coverage is incomplete");
        std::printf("CNB <%s> coverage: round-even=%zu half-up=%zu; fallback shift=%zu flags=%zu both=%zu PASS\n",
                    relu ? "true" : "false", even, halfUp, shiftOnly, flagsOnly, shiftAndFlags);
    }
};

static uint32_t scalar_packed(const QuantizeCase &c, bool useResidual) {
    uint32_t packed = 0;
    for (uint32_t lane = 0; lane < 4; ++lane) {
        const int32_t value = fsr411::finish(c.accumulator[lane], c.bias[lane],
                                           useResidual ? c.residual[lane] : 0, c.q);
        packed |= (uint32_t(value) & 255u) << (lane * 8u);
    }
    return packed;
}

static void run_probe(id<MTLDevice> device, id<MTLLibrary> library, id<MTLCommandQueue> queue,
                      id<MTLBuffer> input, const std::vector<QuantizeCase> &cases,
                      const std::vector<uint32_t> &expected, const char *kernel,
                      const char *label, bool useResidual) {
    NSError *error = nil;
    auto function = [library newFunctionWithName:[NSString stringWithUTF8String:kernel]];
    auto pipeline = function ? [device newComputePipelineStateWithFunction:function error:&error] : nil;
    require(pipeline != nil, std::string("cannot load ") + label + ": " + metal_error(error));
    const uint32_t count = uint32_t(cases.size());
    constexpr size_t guardWords = 128;
    constexpr uint32_t poison = 0xa5a5a5a5u;
    auto output = [device newBufferWithLength:(cases.size() + guardWords) * sizeof(uint32_t) options:MTLResourceStorageModeShared];
    require(output != nil, std::string("cannot allocate ") + label + " output");
    std::fill_n((uint32_t *)output.contents, cases.size() + guardWords, poison);
    auto command = [queue commandBuffer];
    auto encoder = [command computeCommandEncoder];
    require(encoder != nil, "cannot create quantization command buffer/encoder");
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:input offset:0 atIndex:0];[encoder setBuffer:output offset:0 atIndex:1];
    [encoder setBytes:&count length:sizeof(count) atIndex:2];
    [encoder dispatchThreadgroups:MTLSizeMake((count + 127u) / 128u, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    [encoder endEncoding];[command commit];[command waitUntilCompleted];
    require(command.status == MTLCommandBufferStatusCompleted,
            std::string(label) + " GPU failure: " + metal_error(command.error));
    const auto actual = (const uint32_t *)output.contents;
    size_t failures = 0;
    for (size_t index = 0; index < cases.size(); ++index) if (actual[index] != expected[index]) {
        if (failures++ < 12u) {
            const auto &c = cases[index];
            std::fprintf(stderr, "%s case %zu flags=%u shifts=%u,%u,%u actual=%08x expected=%08x\n",
                label, index, c.q.flags, c.q.accumulatorShift, c.q.residualShift, c.q.rightShift, actual[index], expected[index]);
            for (unsigned lane = 0; lane < 4; ++lane)
                std::fprintf(stderr, "  lane %u: accumulator=%d bias=%d residual=%d (used=%d)\n", lane,
                             c.accumulator[lane], c.bias[lane], c.residual[lane], useResidual ? c.residual[lane] : 0);
        }
    }
    for (size_t i = cases.size(); i < cases.size() + guardWords; ++i)
        require(actual[i] == poison, std::string(label) + " wrote beyond its logical output");
    require(!failures, std::string(label) + ": " + std::to_string(failures)
            + " packed cases differed from scalar CPU finish()");
    std::printf("FSR 4.1.1 %s: %zu packed cases / %zu lanes; guard PASS\n",
                label, cases.size(), cases.size() * 4u);
}

int main(int argc, char **argv) { @autoreleasepool {
    try {
        const bool cpuOnly = argc == 2 && std::strcmp(argv[1], "--cpu-only") == 0;
        require(argc == 2, "usage: fsr411_quantize_test METALLIB | --cpu-only");
        check_scalar_contract();
        const auto cases = test_cases();
        require(cases.size() < UINT32_MAX && cases.size() % 128u != 0,
                "quantization case count must exercise a partial threadgroup");
        std::vector<uint32_t> expected, expectedNoResidual;
        expected.reserve(cases.size());
        expectedNoResidual.reserve(cases.size());
        std::array<CnbCoverage, 2> cnbCoverage;
        uint32_t flagMask = 0, rightMask = 0, accMask = 0, residualMask = 0;
        size_t abiCases = 0, shift31Cases = 0;
        for (const auto &c : cases) {
            expected.push_back(scalar_packed(c, true));
            expectedNoResidual.push_back(scalar_packed(c, false));
            cnbCoverage[0].record(c.q, false);
            cnbCoverage[1].record(c.q, true);
            flagMask |= 1u << c.q.flags;rightMask |= 1u << c.q.rightShift;
            accMask |= 1u << c.q.accumulatorShift;residualMask |= 1u << c.q.residualShift;
            if (fsr411::valid_epilogue(c.q)) ++abiCases;
            else { require(c.q.rightShift == 31u, "unexpected non-ABI test case");++shift31Cases; }
        }
        require(flagMask == 255u && rightMask == UINT32_MAX && accMask == 0x7fffffffu
            && residualMask == 0x7fffffffu && abiCases && shift31Cases, "quantization coverage is incomplete");
        cnbCoverage[0].check(false, cases.size());
        cnbCoverage[1].check(true, cases.size());
        if (cpuOnly) {
            std::printf("FSR 4.1.1 quantize CPU: %zu packed cases / %zu lanes per probe; "
                        "19 pinned scalar contracts PASS; three GPU probes not run\n",
                         cases.size(), cases.size() * 4u);
            return 0;
        }
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        require(device != nil, "no Metal device");
        NSError *error = nil;
        auto library = [device newLibraryWithURL:[NSURL fileURLWithPath:[NSString stringWithUTF8String:argv[1]]] error:&error];
        require(library != nil, "cannot load production quantization probes: " + metal_error(error));
        auto input = [device newBufferWithBytes:cases.data() length:cases.size() * sizeof(QuantizeCase) options:MTLResourceStorageModeShared];
        auto queue = [device newCommandQueue];
        require(input && queue, "cannot allocate quantization input/queue");
        run_probe(device, library, queue, input, cases, expected,
                  "fsr411_quantize_test", "production vector quantize", true);
        run_probe(device, library, queue, input, cases, expectedNoResidual,
                  "fsr411_quantize_cnb_linear_test", "production CNB <false>", false);
        run_probe(device, library, queue, input, cases, expectedNoResidual,
                  "fsr411_quantize_cnb_relu_test", "production CNB <true>", false);
        std::printf("All three production probes PASS: all 8 flag combinations, left shifts 0..30, right shifts 0..31; "
                    "%zu ABI cases + %zu right-shift-31 stress cases per probe; "
                    "wrap, ties, residual, narrow, specialization, fallback, guard PASS\n", abiCases, shift31Cases);
        return 0;
    } catch (const std::exception &error) {
        std::fprintf(stderr, "FSR 4.1.1 quantize: %s\n", error.what());
        return 1;
    }
} }
