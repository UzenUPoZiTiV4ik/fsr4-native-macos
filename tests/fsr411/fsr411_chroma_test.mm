// SPDX-License-Identifier: MIT
// The shader is compiled from the imported PRE expression by run_fsr411_chroma.py.
// Runtime buffer inputs keep the compiler from folding signed chroma cases.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

struct alignas(16) ChromaPair {
    float cbCurrent, cbHistory, crCurrent, crHistory;
};
static_assert(sizeof(ChromaPair) == 16, "HLSL float4 input stride changed");
struct Case {
    std::string name;
    ChromaPair pair;
    int symmetryGroup = -1;
};

static void require(bool value, const std::string &message) {
    if (!value) throw std::runtime_error(message);
}
static NSString *ns(const std::string &text) {
    return [NSString stringWithUTF8String:text.c_str()];
}
static std::vector<Case> testCases() {
    std::vector<Case> result = {
        {"first_frame_zero", {0.0f, 0.0f, 0.0f, 0.0f}},
        {"first_frame_equal_chroma", {0.1875f, 0.1875f, -0.3125f, -0.3125f}},
        {"equal_negative_chroma", {-0.375f, -0.375f, -0.25f, -0.25f}},
        {"signed_zero", {-0.0f, 0.0f, 0.0f, -0.0f}},
        {"positive_cb_only", {0.125f, -0.125f, 0.0625f, 0.0625f}},
        {"negative_cb_only", {-0.125f, 0.125f, 0.0625f, 0.0625f}},
        {"positive_cr_only", {0.125f, 0.125f, 0.0625f, -0.0625f}},
        {"negative_cr_only", {0.125f, 0.125f, -0.0625f, 0.0625f}},
    };
    auto quadrants = [&](const char *label, ChromaPair base, int group) {
        for (unsigned signs = 0; signs < 4; ++signs) {
            ChromaPair pair = base;
            if (signs & 1) std::swap(pair.cbCurrent, pair.cbHistory);
            if (signs & 2) std::swap(pair.crCurrent, pair.crHistory);
            result.push_back({std::string(label) + ((signs & 1) ? "_cb-" : "_cb+") +
                              ((signs & 2) ? "_cr-" : "_cr+"), pair, group});
        }
    };
    // The first quartet's distance is ~0.27951 for every sign combination.
    // All operands are exact binary fractions within the normal chroma range.
    quadrants("quarter_eighth", {0.125f, -0.125f, 0.0625f, -0.0625f}, 0);
    quadrants("positive_operands", {0.3125f, 0.0625f, 0.1875f, 0.0625f}, 1);
    quadrants("negative_operands", {-0.0625f, -0.3125f, -0.0625f, -0.1875f}, 2);
    quadrants("near_equal", {0.25f + 1.0f / 1024, 0.25f,
                              -0.25f + 1.0f / 2048, -0.25f}, 3);
    quadrants("epsilon_dominated", {1.0f / 65536, 0.0f, 1.0f / 32768, 0.0f}, 4);
    quadrants("chroma_limits", {0.5f, -0.5f, 0.25f, -0.25f}, 5);
    return result;
}

int main(int argc, char **argv) { @autoreleasepool {
    try {
        require(argc == 2, "usage: fsr411_chroma_test PROBE_BUILD_DIRECTORY");
        const std::string directory = argv[1];
        NSError *error = nil;
        NSData *data = [NSData dataWithContentsOfFile:ns(directory + "/binding.json")];
        require(data != nil, "missing probe binding metadata");
        NSDictionary *info = [NSJSONSerialization JSONObjectWithData:data options:0 error:&error];
        require([info isKindOfClass:[NSDictionary class]], "invalid probe binding metadata");
        NSArray *buffers = info[@"buffers"], *group = info[@"wgsize"];
        require(buffers.count == 2 && [info[@"textures"] count] == 0 &&
                [info[@"samplers"] count] == 0, "unexpected probe resources");
        require(group.count == 3 && [group[0] unsignedIntegerValue] == 1 &&
                [group[1] unsignedIntegerValue] == 1 && [group[2] unsignedIntegerValue] == 1,
                "expected one invocation per reflected threadgroup");
        NSUInteger inputIndex = NSNotFound, outputIndex = NSNotFound;
        for (NSArray *binding in buffers) {
            require(binding.count == 2, "invalid reflected buffer binding");
            NSString *name = binding[1];
            if ([name isEqualToString:@"inputPairs"]) inputIndex = [binding[0] unsignedIntegerValue];
            else if ([name isEqualToString:@"outputDistances"]) outputIndex = [binding[0] unsignedIntegerValue];
            else throw std::runtime_error("unknown reflected buffer name");
        }
        require(inputIndex != NSNotFound && outputIndex != NSNotFound && inputIndex != outputIndex,
                "missing or aliased reflected buffer indices");

        auto device = MTLCreateSystemDefaultDevice();
        require(device != nil, "Metal unavailable");
        auto queue = [device newCommandQueue];
        require(queue != nil, "cannot create Metal command queue");
        NSString *libraryPath = [ns(directory) stringByAppendingPathComponent:info[@"library"]];
        auto library = [device newLibraryWithURL:[NSURL fileURLWithPath:libraryPath] error:&error];
        require(library != nil, error ? error.localizedDescription.UTF8String : "cannot load library");
        auto function = [library newFunctionWithName:info[@"entry"]];
        require(function != nil, "missing reflected kernel entry");
        auto pipeline = [device newComputePipelineStateWithFunction:function error:&error];
        require(pipeline != nil, error ? error.localizedDescription.UTF8String : "cannot create pipeline");
        const auto cases = testCases();
        std::vector<ChromaPair> pairs;
        for (const auto &test : cases) pairs.push_back(test.pair);
        auto input = [device newBufferWithBytes:pairs.data() length:pairs.size() * sizeof(ChromaPair)
                                        options:MTLResourceStorageModeShared];
        auto output = [device newBufferWithLength:cases.size() * sizeof(float)
                                         options:MTLResourceStorageModeShared];
        require(input != nil && output != nil, "cannot allocate probe buffers");
        float *values = static_cast<float *>(output.contents);
        std::fill(values, values + cases.size(), std::numeric_limits<float>::quiet_NaN());
        auto command = [queue commandBuffer];
        auto encoder = [command computeCommandEncoder];
        require(command != nil && encoder != nil, "cannot encode probe");
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:input offset:0 atIndex:inputIndex];
        [encoder setBuffer:output offset:0 atIndex:outputIndex];
        [encoder dispatchThreadgroups:MTLSizeMake(cases.size(), 1, 1)
                threadsPerThreadgroup:MTLSizeMake([group[0] unsignedIntegerValue],
                                                  [group[1] unsignedIntegerValue],
                                                  [group[2] unsignedIntegerValue])];
        [encoder endEncoding];
        [command commit];
        [command waitUntilCompleted];
        require(command.status == MTLCommandBufferStatusCompleted,
                command.error ? command.error.localizedDescription.UTF8String : "GPU command failed");

        unsigned failures = 0, quadrants = 0, zeroDeltaCases = 0;
        double maxAbsoluteError = 0;
        float firstInGroup[6] = {};
        bool haveGroup[6] = {};
        NSMutableArray *records = [NSMutableArray array];
        for (size_t i = 0; i < cases.size(); ++i) {
            const Case &test = cases[i];
            const ChromaPair &pair = test.pair;
            const double cb = double(pair.cbCurrent) - double(pair.cbHistory);
            const double cr = double(pair.crCurrent) - double(pair.crHistory);
            // Independent Euclidean norm, including PRE's 1e-6 under sqrt.
            const double expected = std::hypot(cb, cr, 1e-3);
            const double tolerance = std::max(1e-9, expected * 3e-7);
            const float got = values[i];
            const bool finite = std::isfinite(got);
            const double absoluteError = finite ? std::abs(double(got) - expected) : INFINITY;
            bool passed = finite && got >= 0 && absoluteError <= tolerance;
            if (finite) maxAbsoluteError = std::max(maxAbsoluteError, absoluteError);
            if (cb != 0 && cr != 0) quadrants |= 1u << ((cb < 0 ? 1 : 0) | (cr < 0 ? 2 : 0));
            if (cb == 0 && cr == 0) ++zeroDeltaCases;
            bool symmetric = true;
            if (test.symmetryGroup >= 0) {
                const unsigned g = unsigned(test.symmetryGroup);
                if (!haveGroup[g]) {
                    firstInGroup[g] = got;
                    haveGroup[g] = true;
                } else {
                    symmetric = finite && std::isfinite(firstInGroup[g]) &&
                                std::abs(double(got) - double(firstInGroup[g])) <= tolerance;
                    passed = passed && symmetric;
                }
            }
            if (!passed) {
                ++failures;
                std::fprintf(stderr, "%s: cb delta %+.9g cr delta %+.9g got %.9g expected %.12g%s\n",
                             test.name.c_str(), cb, cr, got, expected, symmetric ? "" : " (asymmetric)");
            }
            NSString *classification = finite ? @"finite" : (std::isnan(got) ? @"nan" : @"infinite");
            [records addObject:@{
                @"name": ns(test.name),
                @"input": @[@(pair.cbCurrent), @(pair.cbHistory), @(pair.crCurrent), @(pair.crHistory)],
                @"cb_delta": @(cb), @"cr_delta": @(cr), @"expected": @(expected),
                @"actual": finite ? (id)@(got) : (id)[NSNull null],
                @"classification": classification, @"tolerance": @(tolerance),
                @"finite": @(finite), @"symmetric": @(symmetric), @"passed": @(passed)
            }];
        }
        require(quadrants == 15 && zeroDeltaCases >= 2, "regression input coverage changed");
        NSDictionary *results = @{
            @"mode": info[@"mode"], @"device": device.name,
            @"case_count": @(cases.size()), @"failure_count": @(failures),
            @"quadrants_tested": @4, @"equal_chroma_cases": @(zeroDeltaCases),
            @"max_finite_absolute_error": @(maxAbsoluteError), @"passed": @(failures == 0),
            @"cases": records
        };
        NSData *json = [NSJSONSerialization dataWithJSONObject:results options:NSJSONWritingPrettyPrinted
                                                       error:&error];
        require(json != nil && [json writeToFile:ns(directory + "/results.json") atomically:YES],
                "cannot write probe results");
        std::printf("FSR 4.1.1 chroma (%s): %zu cases, 4 sign quadrants, %u equal-chroma cases; "
                    "%u failures, max finite absolute error %.9g\n",
                    [info[@"mode"] UTF8String], cases.size(), zeroDeltaCases, failures, maxAbsoluteError);
        return failures ? 1 : 0;
    } catch (const std::exception &error) {
        std::fprintf(stderr, "FSR 4.1.1 chroma: %s\n", error.what());
        return 1;
    }
} }
