// SPDX-License-Identifier: MIT
// Shared Metal/Objective-C plumbing for the standalone GPU A/B and bench
// harnesses in this directory. Fixture math stays in each tool; this header
// owns allocation, readback, submission and reporting boilerplate.
#pragma once
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "../../src/fsr41x/fsr411_model.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

namespace bench {
inline void require(bool ok, const std::string &message) { if (!ok) throw std::runtime_error(message); }
inline NSString *ns(const std::string &s) { return [NSString stringWithUTF8String:s.c_str()]; }
inline std::string metal_error(NSError *e) { return e ? e.localizedDescription.UTF8String : "Metal allocation failed"; }
inline double median(std::vector<double> v) {
    if (v.empty()) return 0;
    std::sort(v.begin(), v.end()); const size_t n = v.size();
    return n % 2 ? v[n/2] : (v[n/2-1] + v[n/2]) * .5;
}
inline double percentile(std::vector<double> v, double p) {
    if (v.empty()) return 0;
    std::sort(v.begin(), v.end()); return v[size_t((v.size()-1) * p)];
}
inline NSArray *numbers(const std::vector<double> &v) {
    auto a = [NSMutableArray array]; for (double x : v) [a addObject:@(x)]; return a;
}
inline id<MTLTexture> texture(id<MTLDevice> dev, MTLPixelFormat format, unsigned w, unsigned h) {
    auto d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format width:w height:h mipmapped:NO];
    d.storageMode = MTLStorageModeShared; d.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
    auto t = [dev newTextureWithDescriptor:d]; require(t != nil, "texture allocation failed"); return t;
}
// Device + classic queue with an optional GPU-family gate (family 0 skips it).
inline id<MTLCommandQueue> device_queue(id<MTLDevice> __strong &dev, MTLGPUFamily family, const char *why) {
    dev = MTLCreateSystemDefaultDevice();
    id<MTLCommandQueue> q = dev ? [dev newCommandQueue] : nil;
    require(dev && q && (!family || [dev supportsFamily:family]), why);
    return q;
}
// Texture readback through a blit encoder. `finite` checks RGBA16Float data
// (bpp 8 only) for the output/history validity contract used by the A/B tools.
inline std::vector<uint8_t> readback(id<MTLCommandQueue> queue, id<MTLTexture> t, unsigned bpp, bool finite = false) {
    require(t != nil, "missing texture during state validation");
    const NSUInteger pitch = (t.width * bpp + 255u) & ~NSUInteger(255);
    auto buffer = [t.device newBufferWithLength:pitch * t.height options:MTLResourceStorageModeShared];
    require(buffer != nil, "readback allocation failed");
    auto cb = [queue commandBuffer]; auto enc = [cb blitCommandEncoder];
    [enc copyFromTexture:t sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0,0,0)
        sourceSize:MTLSizeMake(t.width,t.height,1) toBuffer:buffer destinationOffset:0
        destinationBytesPerRow:pitch destinationBytesPerImage:pitch*t.height];
    [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
    require(cb.status == MTLCommandBufferStatusCompleted, metal_error(cb.error));
    std::vector<uint8_t> out(t.width * t.height * bpp);
    for (NSUInteger y = 0; y < t.height; ++y)
        std::memcpy(out.data() + y*t.width*bpp, (const uint8_t *)buffer.contents + y*pitch, t.width*bpp);
    if (finite && bpp == 8) for (size_t i = 0; i + 1 < out.size(); i += 2) {
        _Float16 v; std::memcpy(&v, out.data() + i, 2); require(std::isfinite(float(v)), "nonfinite output/history");
    }
    return out;
}
// Deterministic LCG colour/depth/motion fixture shared by the full-frame A/B
// tools. `seed` pins each tool's reported fixture identity; exposure optional.
struct Inputs { id<MTLTexture> color, depth, motion, exposure; };
inline Inputs inputs(id<MTLDevice> dev, unsigned rw, unsigned rh, uint32_t seed, bool withExposure) {
    Inputs in{texture(dev,MTLPixelFormatRGBA16Float,rw,rh), texture(dev,MTLPixelFormatR32Float,rw,rh),
              texture(dev,MTLPixelFormatRG16Float,rw,rh), withExposure ? texture(dev,MTLPixelFormatR32Float,1,1) : nil};
    std::vector<_Float16> c(size_t(rw)*rh*4), m(size_t(rw)*rh*2); std::vector<float> d(size_t(rw)*rh);
    uint32_t r = seed;
    for (unsigned y = 0; y < rh; ++y) for (unsigned x = 0; x < rw; ++x) {
        const size_t i = size_t(y)*rw + x; r = r*1664525u + 1013904223u;
        c[i*4]   = (_Float16)(.05f + .8f*float(x)/rw + float(r >> 28)/64.f);
        c[i*4+1] = (_Float16)(.03f + .6f*float(y)/rh + float((r >> 24) & 15)/80.f);
        c[i*4+2] = (_Float16)(float(((x*7 + y*3) % 23) + 1)/30.f); c[i*4+3] = (_Float16)1;
        d[i] = .2f + .6f*float((x/8 + y/8) % 11)/11.f;
        m[i*2] = (_Float16)(float(int(x % 9) - 4)*.0625f); m[i*2+1] = (_Float16)(float(int(y % 5) - 2)*.0625f);
    }
    [in.color replaceRegion:MTLRegionMake2D(0,0,rw,rh) mipmapLevel:0 withBytes:c.data() bytesPerRow:rw*8];
    [in.depth replaceRegion:MTLRegionMake2D(0,0,rw,rh) mipmapLevel:0 withBytes:d.data() bytesPerRow:rw*4];
    [in.motion replaceRegion:MTLRegionMake2D(0,0,rw,rh) mipmapLevel:0 withBytes:m.data() bytesPerRow:rw*4];
    if (withExposure) { const float one = 1.f;
        [in.exposure replaceRegion:MTLRegionMake2D(0,0,1,1) mipmapLevel:0 withBytes:&one bytesPerRow:4]; }
    return in;
}
// The shared eight-frame-cycle jitter used by the temporal A/B fixtures.
inline void jitter(unsigned index, float &jx, float &jy) {
    jx = index & 1u ? -.25f : .125f; jy = index % 3u ? 1.f/6.f : -1.f/6.f;
}
// Commit + wait for a classic command buffer and return its GPU milliseconds.
inline double commit_timed(id<MTLCommandBuffer> cb) {
    [cb commit]; [cb waitUntilCompleted];
    require(cb.status == MTLCommandBufferStatusCompleted, metal_error(cb.error));
    return (cb.GPUEndTime - cb.GPUStartTime) * 1000.;
}
// Metal 4 submission shared by the --mtl4/--metal4 paths. `encode` fills a
// begun MTL4CommandBuffer; a false return abandons it for the caller to report.
struct Mtl4 { id<MTL4CommandQueue> queue = nil; id<MTLSharedEvent> event = nil; uint64_t value = 0; };
inline void mtl4_open(Mtl4 &m, id<MTLDevice> dev, const char *why = "Metal 4 unavailable") {
    if (@available(macOS 26.0, *)) { m.queue = [dev newMTL4CommandQueue]; m.event = [dev newSharedEvent]; }
    require(m.queue && m.event, why);
}
inline bool mtl4_encode(Mtl4 &m, id<MTLDevice> dev, const std::function<bool(id<MTL4CommandBuffer>)> &encode,
                        const char *timeoutWhy = "Metal 4 completion timeout") {
    bool ok = false;
    if (@available(macOS 26.0, *)) {
        auto allocator = [dev newCommandAllocator]; auto cb = [dev newCommandBuffer];
        [cb beginCommandBufferWithAllocator:allocator];
        ok = encode(cb);
        [cb endCommandBuffer];
        if (!ok) return false;
        id<MTL4CommandBuffer> list[] = {cb}; [m.queue commit:list count:1];
        [m.queue signalEvent:m.event value:++m.value];
        require([m.event waitUntilSignaledValue:m.value timeoutMS:20000], timeoutWhy);
    }
    return ok;
}
// fsr411 scratch sizing (+64 pad) shared by the native-pass benches.
inline unsigned scratch_tensor_end(const fsr411::Tensor &t) {
    return t.channels ? fsr411::tensor_address(t, t.width+1, t.height+1, t.channels-1) + 64u : 0u;
}
inline fsr411::PassParams pass_params(NSDictionary *entry, NSData *params) {
    fsr411::PassParams p;
    const size_t offset = [entry[@"parameter_offset"] unsignedIntValue];
    require(offset <= params.length && sizeof(p) <= params.length - offset, "parameter region overflow");
    std::memcpy(&p, (const uint8_t *)params.bytes + offset, sizeof(p));
    return p;
}
// Manifest + weights + params triple used by the isolated-pass benches.
struct NativeModel { NSDictionary *manifest; NSData *weights, *params; };
inline NativeModel native_model(const std::string &dir, const char *why = "missing native model") {
    NSError *e = nil;
    auto data = [NSData dataWithContentsOfFile:ns(dir + "/fsr411_native_manifest.json")];
    NSDictionary *manifest = data ? [NSJSONSerialization JSONObjectWithData:data options:0 error:&e] : nil;
    auto weights = [NSData dataWithContentsOfFile:ns(dir + "/fsr411_native_weights.bin")];
    auto params = [NSData dataWithContentsOfFile:ns(dir + "/fsr411_native_params.bin")];
    require(manifest && weights.length && params.length, why);
    return {manifest, weights, params};
}
// Max logical tensor end over the pass set (including post_cnb), then the
// appended post_cnb region: aligned offset out + total scratch bytes back.
inline unsigned scratch_extent(NSDictionary *passes, NSData *params, unsigned &postOffset,
        const std::function<void(NSString *, const fsr411::PassParams &)> &visit = nullptr) {
    unsigned bytes = 0;
    for (NSString *key in passes) {
        const auto p = pass_params(passes[key], params);
        bytes = std::max({bytes, scratch_tensor_end(p.input), scratch_tensor_end(p.output), scratch_tensor_end(p.skip)});
        if (visit) visit(key, p);
    }
    postOffset = (bytes + 255u) & ~255u;
    return postOffset + [passes[@"post_cnb"][@"scratch_append_bytes"] unsignedIntValue] + 256u;
}
// fsr411_mpp_chain_probe on a capped 32-thread pipeline; needs a library that
// exports the probe (callers wanting optional support check the name first).
inline uint32_t mpp_chain_probe(id<MTLDevice> dev, id<MTLCommandQueue> q, id<MTLLibrary> lib) {
    auto fn = [lib newFunctionWithName:@"fsr411_mpp_chain_probe"]; require(fn != nil, "missing MPP chain probe");
    NSError *e = nil; auto d = [MTLComputePipelineDescriptor new]; d.computeFunction = fn;
    d.maxTotalThreadsPerThreadgroup = 32; d.threadGroupSizeIsMultipleOfThreadExecutionWidth = YES;
    auto p = [dev newComputePipelineStateWithDescriptor:d options:MTLPipelineOptionNone reflection:nil error:&e];
    require(p && p.threadExecutionWidth == 32 && !p.staticThreadgroupMemoryLength, "MPP probe pipeline invalid");
    std::vector<uint8_t> zeros(fsr411::kMppProbeDummyBytes, 0);
    auto b = [dev newBufferWithLength:4 options:MTLResourceStorageModeShared];
    auto dummy = [dev newBufferWithBytes:zeros.data() length:zeros.size() options:MTLResourceStorageModeShared];
    require(b && dummy, "MPP probe allocation failed"); *(uint32_t *)b.contents = ~0u;
    auto cb = [q commandBuffer]; auto enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:p]; [enc setBuffer:b offset:0 atIndex:0]; [enc setBuffer:dummy offset:0 atIndex:1];
    [enc dispatchThreadgroups:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
    [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
    require(cb.status == MTLCommandBufferStatusCompleted, metal_error(cb.error));
    const uint32_t mask = *(const uint32_t *)b.contents;
    require(!(mask & ~fsr411::kMppChainKnownMask), "MPP probe returned unknown features");
    return mask;
}
} // namespace bench
