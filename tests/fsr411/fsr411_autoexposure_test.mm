// SPDX-License-Identifier: MIT
// Standalone SPD publication/viewport/exposure-history checks. The Python
// runner supplies three independently compiled strict-arithmetic shaders.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

struct SpdConstants {
    uint32_t mips, groups, offset[2];
    float invSize[2], preExposure, padding;
    uint32_t validSize[2];
};
static_assert(sizeof(SpdConstants) == 40, "SPD ABI changed");

struct Kernel {
    id<MTLComputePipelineState> pipeline;
    NSArray *buffers, *textures;
};
struct Surface {
    id<MTLTexture> color, luma, exposure, atomicTexture;
    id<MTLBuffer> constants, atomic;
    unsigned width, height, mips;
    float previousCurrent = 0.75f;
};

static void require(bool value, const std::string &message) {
    if (!value) throw std::runtime_error(message);
}
static NSString *ns(const std::string &text) {
    return [NSString stringWithUTF8String:text.c_str()];
}
static id<MTLTexture> texture(id<MTLDevice> device, MTLPixelFormat format,
                              unsigned width, unsigned height) {
    auto d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format
                        width:width height:height mipmapped:NO];
    d.storageMode = MTLStorageModeShared;
    d.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
    auto result = [device newTextureWithDescriptor:d];
    require(result != nil, "cannot allocate texture");
    return result;
}
static Kernel load(id<MTLDevice> device, const std::string &root, const char *name) {
    const std::string directory = root + "/" + name;
    NSError *error = nil;
    NSData *data = [NSData dataWithContentsOfFile:ns(directory + "/binding.json")];
    require(data != nil, "missing probe binding metadata");
    NSDictionary *info = [NSJSONSerialization JSONObjectWithData:data options:0 error:&error];
    require(info != nil, "invalid probe binding metadata");
    NSString *libraryPath = [ns(directory) stringByAppendingPathComponent:info[@"library"]];
    auto library = [device newLibraryWithURL:[NSURL fileURLWithPath:libraryPath] error:&error];
    require(library != nil, error ? error.localizedDescription.UTF8String : "cannot load library");
    auto function = [library newFunctionWithName:info[@"entry"]];
    auto pipeline = [device newComputePipelineStateWithFunction:function error:&error];
    require(pipeline != nil, error ? error.localizedDescription.UTF8String : "cannot build pipeline");
    return {pipeline, info[@"buffers"], info[@"textures"]};
}
static Surface surface(id<MTLDevice> device, unsigned width, unsigned height,
                       bool paddedLuma, bool oracle) {
    Surface result;
    result.width = width; result.height = height;
    result.mips = 0;
    for (unsigned n = std::max(width, height); n > 1; n >>= 1) ++result.mips;
    result.mips = std::max(1u, std::min(result.mips, 12u));
    // Color always has a poisoned backing region beyond validSize.
    result.color = texture(device, MTLPixelFormatRGBA32Float, width + 7, height + 5);
    result.luma = texture(device, MTLPixelFormatR32Float, (width + 63) / 64 + (paddedLuma ? 7 : 0),
                            (height + 63) / 64 + (paddedLuma ? 3 : 0));
    result.exposure = texture(device, MTLPixelFormatR32Float, 2, 1);
    float initial[2] = {result.previousCurrent, 0.625f};
    [result.exposure replaceRegion:MTLRegionMake2D(0, 0, 2, 1) mipmapLevel:0
                        withBytes:initial bytesPerRow:sizeof(initial)];
    result.constants = [device newBufferWithLength:256 options:MTLResourceStorageModeShared];
    if (oracle) {
        result.atomic = [device newBufferWithLength:4096 options:MTLResourceStorageModeShared];
        auto d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR32Uint
                            width:1 height:1 mipmapped:NO];
        d.storageMode = MTLStorageModeShared;
        d.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
        const NSUInteger row = std::max<NSUInteger>(16,
                            [device minimumLinearTextureAlignmentForPixelFormat:MTLPixelFormatR32Uint]);
        result.atomicTexture = [result.atomic newTextureWithDescriptor:d offset:0 bytesPerRow:row];
        require(result.atomicTexture != nil, "cannot allocate oracle atomic alias");
    }
    return result;
}
static void fill(Surface &s, unsigned iteration, float preExposure, bool constant) {
    const unsigned pw = unsigned(s.color.width), ph = unsigned(s.color.height);
    std::vector<float> color(size_t(pw) * ph * 4, 37.0f + iteration);
    const float gray = 0.125f * float((iteration % 5) + 1);
    for (unsigned y = 0; y < s.height; ++y) for (unsigned x = 0; x < s.width; ++x) {
        const size_t i = (size_t(y) * pw + x) * 4;
        color[i] = constant ? gray : 0.05f + 0.013f * float((x + iteration * 3) % 31);
        color[i + 1] = constant ? gray : 0.08f + 0.017f * float((y + iteration * 5) % 19);
        color[i + 2] = constant ? gray : 0.11f + 0.009f * float((x + y + iteration * 7) % 43);
        color[i + 3] = 1.0f;
    }
    [s.color replaceRegion:MTLRegionMake2D(0, 0, pw, ph) mipmapLevel:0
                     withBytes:color.data() bytesPerRow:pw * 16];
    std::vector<float> poison(s.luma.width * s.luma.height, 11.0f + iteration * 0.25f);
    [s.luma replaceRegion:MTLRegionMake2D(0, 0, s.luma.width, s.luma.height) mipmapLevel:0
                    withBytes:poison.data() bytesPerRow:s.luma.width * 4];
    if (s.atomic) std::memset(s.atomic.contents, 0, s.atomic.length);
    SpdConstants c = {};
    c.mips = s.mips; c.groups = ((s.width + 63) / 64) * ((s.height + 63) / 64);
    c.invSize[0] = 1.0f / s.width; c.invSize[1] = 1.0f / s.height;
    c.preExposure = preExposure; c.validSize[0] = s.width; c.validSize[1] = s.height;
    std::memcpy(s.constants.contents, &c, sizeof(c));
}
static void bind(id<MTLComputeCommandEncoder> encoder, const Kernel &kernel, Surface &s) {
    [encoder setComputePipelineState:kernel.pipeline];
    for (NSArray *binding in kernel.buffers) {
        NSString *name = binding[1];
        id<MTLBuffer> value = nil;
        if ([name isEqualToString:@"AutoExposureSPDConstants"]) value = s.constants;
        if ([name isEqualToString:@"rw_spd_global_atomic_atomic"]) value = s.atomic;
        require(value != nil, std::string("unknown or absent buffer: ") + name.UTF8String);
        [encoder setBuffer:value offset:0 atIndex:[binding[0] unsignedIntegerValue]];
    }
    for (NSArray *binding in kernel.textures) {
        NSString *name = binding[1];
        id<MTLTexture> value = nil;
        if ([name isEqualToString:@"r_input_color"]) value = s.color;
        if ([name isEqualToString:@"rw_autoexp_mip_5"]) value = s.luma;
        if ([name isEqualToString:@"rw_auto_exposure_texture"]) value = s.exposure;
        if ([name isEqualToString:@"rw_spd_global_atomic"]) value = s.atomicTexture;
        require(value != nil, std::string("unknown or absent texture: ") + name.UTF8String);
        [encoder setTexture:value atIndex:[binding[0] unsignedIntegerValue]];
    }
}
static float run(id<MTLCommandQueue> queue, const Kernel &first, const Kernel *finalize,
                 Surface &s, const char *label) {
    auto cb = [queue commandBuffer];
    auto enc = [cb computeCommandEncoder];
    bind(enc, first, s);
    [enc dispatchThreadgroups:MTLSizeMake((s.width + 63) / 64, (s.height + 63) / 64, 1)
              threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    if (finalize && s.mips > 6) {
        [enc memoryBarrierWithScope:MTLBarrierScopeTextures | MTLBarrierScopeBuffers];
        bind(enc, *finalize, s);
        [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
                  threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    }
    [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
    require(cb.status == MTLCommandBufferStatusCompleted,
                cb.error ? cb.error.localizedDescription.UTF8String : "GPU command failed");
    float values[2] = {};
    [s.exposure getBytes:values bytesPerRow:sizeof(values)
                  fromRegion:MTLRegionMake2D(0, 0, 2, 1) mipmapLevel:0];
    require(std::isfinite(values[0]) && values[0] > 0, std::string(label) + ": invalid current exposure");
    if (values[1] != s.previousCurrent) {
        std::fprintf(stderr, "%s previous got %.9g expected %.9g\n", label, values[1], s.previousCurrent);
        throw std::runtime_error("previous exposure did not preserve its known marker");
    }
    s.previousCurrent = values[0];
    return values[0];
}

int main(int argc, char **argv) { @autoreleasepool {
    try {
        require(argc == 2, "usage: fsr411_autoexposure_test PROBE_BUILD_DIRECTORY");
        auto device = MTLCreateSystemDefaultDevice(); auto queue = [device newCommandQueue];
        require(device != nil && queue != nil, "Metal unavailable");
        Kernel oracle = load(device, argv[1], "ae_oracle");
        Kernel tiles = load(device, argv[1], "ae_tiles");
        Kernel finalize = load(device, argv[1], "ae_finalize");
        const unsigned sizes[][2] = {
            {128,128}, {256,256}, {512,512}, {130,129}, {190,120},
            {257,251}, {128,65}, {960,540}, {1280,720}
        };
        unsigned comparisons = 0;
        for (const auto &size : sizes) {
            Surface a = surface(device, size[0], size[1], false, true);
            Surface b = surface(device, size[0], size[1], false, false);
            Surface c = surface(device, size[0], size[1], true, false);
            for (unsigned frame = 0; frame < 6; ++frame) {
                const float pre = 0.5f + 0.375f * float(frame % 4);
                fill(a, frame, pre, false); fill(b, frame, pre, false); fill(c, frame, pre, false);
                const float av = run(queue, oracle, nullptr, a, "oracle");
                const float bv = run(queue, tiles, &finalize, b, "split");
                const float cv = run(queue, tiles, &finalize, c, "split padded");
                if (av != bv || bv != cv) {
                    std::fprintf(stderr, "AE %ux%u frame%u current oracle %.9g split %.9g padded %.9g\n",
                                 size[0], size[1], frame, av, bv, cv);
                    if (size[0] == 128 && size[1] == 128) {
                        float at[4], bt[4];
                        [a.luma getBytes:at bytesPerRow:8 fromRegion:MTLRegionMake2D(0,0,2,2) mipmapLevel:0];
                        [b.luma getBytes:bt bytesPerRow:8 fromRegion:MTLRegionMake2D(0,0,2,2) mipmapLevel:0];
                        std::fprintf(stderr, "completed luma oracle [%g,%g,%g,%g] split [%g,%g,%g,%g]\n",
                                     at[0],at[1],at[2],at[3],bt[0],bt[1],bt[2],bt[3]);
                    }
                    throw std::runtime_error("split/oracle or padded-view exposure differs");
                }
                ++comparisons;
            }
            std::printf("AE %ux%u mips%u: 6 frames exact current/previous, poisoned luma and padding\n",
                        size[0], size[1], a.mips);
        }
        const unsigned tiny[][2] = {{1,1}, {24,12}, {32,32}, {64,64}, {96,32}, {127,95}};
        unsigned constantChecks = 0;
        for (const auto &size : tiny) {
            Surface s = surface(device, size[0], size[1], true, false);
            for (unsigned frame = 0; frame < 5; ++frame) {
                const float pre = 0.5f + 0.5f * float(frame % 3);
                fill(s, frame, pre, true);
                const float got = run(queue, tiles, &finalize, s, "tiny split");
                const double gray = 0.125 * double((frame % 5) + 1);
                const unsigned side = 1u << s.mips;
                const double fraction = double(std::min(size[0], side)) * std::min(size[1], side) / (double(side) * side);
                const double mean = fraction * std::log(gray / pre) + (1.0 - fraction) * std::log(1e-6);
                const double expected = 1.0 / (1.2 * 8.0 * std::exp(mean));
                const double relative = std::abs(double(got) - expected) / expected;
                if (relative > 3e-5) {
                    std::fprintf(stderr, "tiny AE %ux%u frame%u got %.9g expected %.12g relative %.9g\n",
                                 size[0], size[1], frame, got, expected, relative);
                    throw std::runtime_error("constant-luminance CPU expectation differs");
                }
                ++constantChecks;
            }
            std::printf("AE %ux%u mips%u: 5 frames CPU constant expectation and previous marker\n",
                        size[0], size[1], s.mips);
        }
        std::printf("FSR 4.1.1 AE passed: %u oracle comparisons, %u CPU constant checks; no timings collected\n",
                    comparisons, constantChecks);
        return 0;
    } catch (const std::exception &error) {
        std::fprintf(stderr, "FSR 4.1.1 AE: %s\n", error.what()); return 1;
    }
} }
