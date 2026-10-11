// SPDX-License-Identifier: MIT
// Usage: fsr411_native_test <metallib> <native-dir> [--reference|--specialized|--simd|--simd-specialized]
// Correctness only: tiny odd-sized inputs, all bytes including padding checked.
// --simd runs the Apple7+ simdgroup kernels with the half weight copy at index 3.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "../src/fsr411_model.h"
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <vector>

static std::vector<unsigned char> read_file(const char *path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) { std::fprintf(stderr, "Cannot read %s\n", path); std::exit(1); }
    return {std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>()};
}

static int bias(const std::vector<unsigned char> &weights, unsigned offset, unsigned channel) {
    int value;
    std::memcpy(&value, weights.data() + offset + channel * 4, 4);
    return value;
}

static int signed_byte(unsigned char x) { return x < 128 ? x : int(x) - 256; }

static void reference_cnb(std::vector<unsigned char> &scratch,
                          const std::vector<unsigned char> &weights,
                          const fsr411::PassParams &p) {
    const auto input = scratch;
    const auto weight = [&](unsigned layer, unsigned index) { return signed_byte(weights[p.weights[layer] + index]); };
    const auto pixel = [&](int x, int y, unsigned c) {
        return x < 0 || y < 0 || x >= int(p.input.width) || y >= int(p.input.height) ? 0 :
            signed_byte(input[fsr411::tensor_address(p.input, unsigned(x), unsigned(y), c)]);
    };
    for (unsigned y = 0; y < p.output.height; ++y) {
        for (unsigned x = 0; x < p.output.width; ++x) {
            int a[16], b[32];
            for (unsigned oc = 0; oc < 16; ++oc) {
                int acc = 0;
                for (unsigned ky = 0; ky < 3; ++ky)
                    for (unsigned kx = 0; kx < 3; ++kx)
                        for (unsigned ic = 0; ic < 16; ++ic)
                            acc += pixel(int(x + kx) - 1, int(y + ky) - 1, ic) * weight(0, ((ky * 3 + kx) * 16 + ic) * 16 + oc);
                a[oc] = fsr411::finish(acc, bias(weights, p.biases[0], oc), 0, p.q[0]);
            }
            for (unsigned oc = 0; oc < 32; ++oc) {
                int acc = 0;
                for (unsigned ic = 0; ic < 16; ++ic) acc += a[ic] * weight(1, ic * 32 + oc);
                b[oc] = fsr411::finish(acc, bias(weights, p.biases[1], oc), 0, p.q[1]);
            }
            for (unsigned oc = 0; oc < 16; ++oc) {
                int acc = 0;
                for (unsigned ic = 0; ic < 32; ++ic) acc += b[ic] * weight(2, ic * 16 + oc);
                const int value = fsr411::finish(acc, bias(weights, p.biases[2], oc), pixel(x, y, oc), p.q[oc < 8 ? 2 : 3]);
                scratch[fsr411::tensor_address(p.output, x, y, oc)] = static_cast<unsigned char>(value);
            }
        }
    }
}

static void reference_ds(std::vector<unsigned char> &scratch,
                         const std::vector<unsigned char> &weights,
                         const fsr411::PassParams &p) {
    const auto input = scratch;
    for (unsigned y = 0; y < p.output.height; ++y)
        for (unsigned x = 0; x < p.output.width; ++x)
            for (unsigned oc = 0; oc < p.output.channels; ++oc) {
                int acc = 0;
                for (unsigned tap = 0; tap < 4; ++tap)
                    for (unsigned ic = 0; ic < p.input.channels; ++ic) {
                        const unsigned ix = x * 2 + tap % 2, iy = y * 2 + tap / 2;
                        const int v = ix < p.input.width && iy < p.input.height ?
                            signed_byte(input[fsr411::tensor_address(p.input, ix, iy, ic)]) : 0;
                        const int w = signed_byte(weights[p.weights[0] + (tap * p.input.channels + ic) * p.output.channels + oc]);
                        acc += v * w;
                    }
                const int value = fsr411::finish(acc, bias(weights, p.biases[0], oc), 0, p.q[oc < p.output.channels / 2 ? 4 : 5]);
                scratch[fsr411::tensor_address(p.output, x, y, oc)] = static_cast<unsigned char>(value);
            }
}

static void reference_fnb(std::vector<unsigned char> &scratch,
                          const std::vector<unsigned char> &weights,
                          const fsr411::PassParams &p, bool ct) {
    const auto input = scratch;
    const unsigned nf = p.input.channels;
    const auto pixel = [&](fsr411::Tensor tensor, int x, int y, unsigned c) {
        return x < 0 || y < 0 || x >= int(tensor.width) || y >= int(tensor.height) ? 0 :
            signed_byte(input[fsr411::tensor_address(tensor, unsigned(x), unsigned(y), c)]);
    };
    const auto weight = [&](unsigned layer, unsigned index) { return signed_byte(weights[p.weights[layer] + index]); };
    const unsigned width = ct ? p.input.width : p.output.width, height = ct ? p.input.height : p.output.height;
    for (unsigned y = 0; y < height; ++y)
        for (unsigned x = 0; x < width; ++x) {
            int a[64], b[128], mid[64];
            for (unsigned oc = 0; oc < nf / 2; ++oc) {
                int acc = 0;
                for (unsigned tap = 0; tap < 9; ++tap)
                    for (unsigned ic = 0; ic < 16; ++ic)
                        acc += pixel(p.input, int(x + tap % 3) - 1, int(y + tap / 3) - 1, (oc / 16) * 16 + ic) *
                               weight(0, (oc / 16) * 9 * 16 * 16 + (tap * 16 + ic) * 16 + oc % 16);
                a[oc] = fsr411::finish(acc, bias(weights, p.biases[0], oc), 0, p.q[0]);
            }
            for (unsigned c = nf / 2; c < nf; ++c) a[c] = pixel(p.input, x, y, c);
            for (unsigned oc = 0; oc < 2 * nf; ++oc) {
                int acc = 0;
                for (unsigned ic = 0; ic < nf; ++ic) acc += a[ic] * weight(1, ic * 2 * nf + oc);
                b[oc] = fsr411::finish(acc, bias(weights, p.biases[1], oc), 0, p.q[1]);
            }
            for (unsigned oc = 0; oc < nf; ++oc) {
                int acc = 0;
                for (unsigned ic = 0; ic < 2 * nf; ++ic) acc += b[ic] * weight(2, ic * nf + oc);
                mid[oc] = fsr411::finish(acc, bias(weights, p.biases[2], oc), pixel(p.input, x, y, oc), p.q[oc < nf / 2 ? 2 : 3]);
                if (!ct) scratch[fsr411::tensor_address(p.output, x, y, oc)] = static_cast<unsigned char>(mid[oc]);
            }
            if (!ct) continue;
            const unsigned out = nf / 2;
            for (unsigned tap = 0; tap < 4; ++tap) {
                const unsigned ox = x * 2 + tap % 2, oy = y * 2 + tap / 2;
                if (ox >= p.output.width || oy >= p.output.height) continue;
                for (unsigned oc = 0; oc < out; ++oc) {
                    int acc = 0;
                    for (unsigned ic = 0; ic < nf; ++ic) acc += mid[ic] * weight(3, ic * 4 * out + tap * out + oc);
                    const int value = fsr411::finish(acc, bias(weights, p.biases[3], oc), pixel(p.skip, ox, oy, oc), p.q[oc < out / 2 ? 4 : 5]);
                    scratch[fsr411::tensor_address(p.output, ox, oy, oc)] = static_cast<unsigned char>(value);
                }
            }
        }
}

int main(int argc, char **argv) {
    const char *mode = argc == 4 ? argv[3] : "";
    const bool reference = !std::strcmp(mode, "--reference");
    const bool simd = !std::strcmp(mode, "--simd") || !std::strcmp(mode, "--simd-specialized");
    const bool specialized = !std::strcmp(mode, "--specialized") || !std::strcmp(mode, "--simd-specialized");
    if (argc != 3 && !reference && !specialized && !simd) { std::fprintf(stderr, "Usage: %s <metallib> <native-dir> [--reference|--specialized|--simd|--simd-specialized]\n", argv[0]); return 2; }
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device || (!reference && !simd && ![device supportsFamily:MTLGPUFamilyApple10])
            || (simd && ![device supportsFamily:MTLGPUFamilyApple7])) {
            std::fprintf(stderr, "The native backend requires Apple10 and --simd Apple7; --reference runs on earlier Metal GPUs\n");
            return 2;
        }
        NSError *error = nil;
        id<MTLLibrary> library = [device newLibraryWithURL:[NSURL fileURLWithPath:@(argv[1])] error:&error];
        if (!library) { std::fprintf(stderr, "Metal load failed: %s\n", error.description.UTF8String); return 1; }
        const std::string directory = argv[2];
        const auto weight_bytes = read_file((directory + "/fsr411_native_weights.bin").c_str());
        const auto parameter_bytes = read_file((directory + "/fsr411_native_params.bin").c_str());
        NSData *json = [NSData dataWithContentsOfFile:@((directory + "/fsr411_native_manifest.json").c_str())];
        NSDictionary *manifest = [NSJSONSerialization JSONObjectWithData:json options:0 error:&error];
        if (!manifest || [manifest[@"parameter_size"] unsignedIntValue] != sizeof(fsr411::PassParams)) return 1;
        id<MTLBuffer> weights = [device newBufferWithBytes:weight_bytes.data() length:weight_bytes.size() options:MTLResourceStorageModeShared];
        std::vector<__fp16> half_bytes(weight_bytes.size());
        for (size_t i = 0; i < weight_bytes.size(); ++i) half_bytes[i] = __fp16(signed_byte(weight_bytes[i]));
        id<MTLBuffer> half_weights = [device newBufferWithBytes:half_bytes.data() length:half_bytes.size() * 2 options:MTLResourceStorageModeShared];
        id<MTLCommandQueue> queue = [device newCommandQueue];
        const unsigned cases[][3] = {{1, 1, 0}, {3, 5, 3}, {33, 7, 128}, {65, 5, 128}};
        unsigned tests = 0;
        for (NSString *pass in @[@"1", @"2", @"3", @"4", @"5", @"6", @"7", @"8", @"9", @"10", @"11", @"12", @"post_cnb"]) {
            NSDictionary *entry = [pass isEqual:@"post_cnb"] ? manifest[@"auxiliary_passes"][pass] : manifest[@"passes"][pass];
            if (!entry) continue;
            NSDictionary *schedule = reference ? entry[@"reference"] : simd ? entry[@"simd"] : entry;
            if (!schedule) { std::fprintf(stderr, "pass %s: manifest has no schedule for %s\n", pass.UTF8String, mode); return 1; }
            const unsigned offset = [entry[@"parameter_offset"] unsignedIntValue];
            fsr411::PassParams original;
            std::memcpy(&original, parameter_bytes.data() + offset, sizeof(original));
            id<MTLFunction> function = nil;
            if (specialized) {
                // Build once from the unmodified manifest parameters. Every
                // case changes dimensions and POST's appended output offset.
                const auto metadata = fsr411::specialize_metadata(original,
                    [entry[@"output_offset_base"] isEqual:@"scratch_append"]);
                MTLFunctionConstantValues *values = [MTLFunctionConstantValues new];
                [values setConstantValues:metadata.values type:MTLDataTypeUInt4
                                withRange:NSMakeRange(0, fsr411::kPassSpecializationCount)];
                function = [library newFunctionWithName:schedule[@"function"] constantValues:values error:&error];
            } else function = [library newFunctionWithName:schedule[@"function"]];
            id<MTLComputePipelineState> pipeline = function ? [device newComputePipelineStateWithFunction:function error:&error] : nil;
            if (!pipeline) { std::fprintf(stderr, "Metal pipeline failed: %s\n", error.description.UTF8String); return 1; }
            const bool ds = [entry[@"kind"] isEqual:@"ds"];
            const bool ct = [entry[@"kind"] isEqual:@"ct"];
            const bool fnb = [entry[@"kind"] isEqual:@"fnb"];
            for (const auto &test : cases) {
                fsr411::PassParams p = original;
                p.input.width = p.output.width = test[0];
                p.input.height = p.output.height = test[1];
                if (ds) { p.input.width = test[0] * 2 - 1; p.input.height = test[1] * 2 - 1; }
                if (ct) {
                    p.input.width = (test[0] + 1) / 2; p.input.height = (test[1] + 1) / 2;
                    p.skip.width = test[0]; p.skip.height = test[1];
                }
                if ([entry[@"output_offset_base"] isEqual:@"scratch_append"])
                    p.output.offset += (fsr411::tensor_address(p.input, p.input.width + 1, p.input.height + 1, p.input.channels - 1) + 64 + 255) & ~255u;
                size_t size = std::max(fsr411::tensor_address(p.input, p.input.width + 1, p.input.height + 1, p.input.channels - 1),
                                             fsr411::tensor_address(p.output, p.output.width + 1, p.output.height + 1, p.output.channels - 1)) + 64;
                if (ct) size = std::max(size, size_t(fsr411::tensor_address(p.skip, p.skip.width + 1, p.skip.height + 1, p.skip.channels - 1)) + 64);
                // Nonzero padding catches any accidental physical-border read.
                // The runtime's generic border kernels separately keep its scratch
                // border zero for the other imported HLSL stages.
                std::vector<unsigned char> before(size, 0xa5);
                unsigned random = 0x4110000u + [pass intValue];
                const auto fill = [&](fsr411::Tensor tensor) {
                for (unsigned y = 0; y < tensor.height; ++y)
                    for (unsigned x = 0; x < tensor.width; ++x)
                        for (unsigned c = 0; c < tensor.channels; ++c) {
                            random = random * 1664525u + 1013904223u;
                            int value = test[2] ? int(random % (test[2] * 2)) - int(test[2]) : 0;
                            before[fsr411::tensor_address(tensor, x, y, c)] = static_cast<unsigned char>(value);
                        }
                };
                fill(p.input);
                if (ct) fill(p.skip);
                auto expected = before;
                if (ds) reference_ds(expected, weight_bytes, p);
                else if (ct || fnb) reference_fnb(expected, weight_bytes, p, ct);
                else reference_cnb(expected, weight_bytes, p);
                id<MTLBuffer> scratch = [device newBufferWithBytes:before.data() length:before.size() options:MTLResourceStorageModeShared];
                id<MTLCommandBuffer> command = [queue commandBuffer];
                id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
                [encoder setComputePipelineState:pipeline];
                [encoder setBuffer:scratch offset:0 atIndex:0];
                [encoder setBuffer:weights offset:0 atIndex:1];
                if (simd) [encoder setBuffer:half_weights offset:0 atIndex:3];
                [encoder setBytes:&p length:sizeof(p) atIndex:2];
                const unsigned tw = [schedule[@"tile"][0] unsignedIntValue], th = [schedule[@"tile"][1] unsignedIntValue];
                const unsigned gridw = ct ? p.input.width : p.output.width, gridh = ct ? p.input.height : p.output.height;
                [encoder dispatchThreadgroups:MTLSizeMake((gridw + tw - 1) / tw, (gridh + th - 1) / th, 1)
                         threadsPerThreadgroup:MTLSizeMake([schedule[@"threads"] unsignedIntValue], 1, 1)];
                [encoder endEncoding];
                [command commit];
                [command waitUntilCompleted];
                if (command.status == MTLCommandBufferStatusError) {
                    std::fprintf(stderr, "GPU failed: %s\n", command.error.description.UTF8String); return 1;
                }
                const auto actual = static_cast<const unsigned char *>(scratch.contents);
                size_t differences = 0, first = 0;
                for (size_t i = 0; i < size; ++i) if (actual[i] != expected[i]) { if (!differences) first = i; ++differences; }
                if (differences) {
                    std::fprintf(stderr, "pass %s %ux%u: %zu bytes differ; first %zu GPU=%u CPU=%u\n",
                                 pass.UTF8String, test[0], test[1], differences, first, actual[first], expected[first]);
                    return 1;
                }
                ++tests;
                std::printf("pass %s %ux%u: all %zu bytes identical\n", pass.UTF8String, test[0], test[1], size);
            }
        }
        std::printf("fsr411_native_test: %u cases passed\n", tests);
    }
}
