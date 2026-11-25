// Drive the hooked MetalFX encode with the quality testcase and compare to fsr4mac.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalFX/MetalFX.h>
#include "../../src/live/fsr4_live.h"
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>

static id<MTLTexture> tex(id<MTLDevice> dev, NSUInteger w, NSUInteger h, MTLPixelFormat fmt) {
    MTLTextureDescriptor *d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:fmt width:w height:h mipmapped:NO];
    d.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
    d.storageMode = MTLStorageModeShared;
    return [dev newTextureWithDescriptor:d];
}

static bool read_file(const char *path, std::vector<uint8_t> &out) {
    FILE *f = fopen(path, "rb");
    if (!f) return false;
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (n <= 0) { fclose(f); return false; }
    out.resize((size_t)n);
    bool ok = fread(out.data(), 1, out.size(), f) == out.size();
    fclose(f);
    return ok;
}

static float h2f(uint16_t h) {
    uint32_t x = h;
    uint32_t sign = (x & 0x8000u) << 16;
    uint32_t exp = (x >> 10) & 0x1f;
    uint32_t mant = x & 0x3ff;
    uint32_t f;
    if (exp == 0) f = sign;
    else if (exp == 31) f = sign | 0x7f800000u | (mant << 13);
    else f = sign | ((exp + 112) << 23) | (mant << 13);
    float o;
    memcpy(&o, &f, 4);
    return o;
}

int main(int argc, char **argv) {
        if (argc < 3) {
        fprintf(stderr, "usage: live_test <testcase_dir> <reference_f00_out.bin> [mtl4|classic] [repeats]\n");
        return 2;
    }
    const bool mtl4 = argc > 3 && !strcmp(argv[3], "mtl4");
    const bool directClassic = argc > 3 && !strcmp(argv[3], "classic");
    const int repeats = argc > 4 ? atoi(argv[4]) : 1;
    if (repeats < 1 || repeats > 100) return 2;
    @autoreleasepool {
        const uint32_t rw = 1280, rh = 720, uw = 1920, uh = 1080;
        std::vector<uint8_t> color, depth, mv, ref;
        char path[1024];
        snprintf(path, sizeof path, "%s/f00_color.bin", argv[1]);
        if (!read_file(path, color) || color.size() != (size_t)rw * rh * 8) return 1;
        snprintf(path, sizeof path, "%s/f00_depth.bin", argv[1]);
        if (!read_file(path, depth) || depth.size() != (size_t)rw * rh * 4) return 1;
        snprintf(path, sizeof path, "%s/f00_mv.bin", argv[1]);
        if (!read_file(path, mv) || mv.size() != (size_t)rw * rh * 4) return 1;
        if (!read_file(argv[2], ref) || ref.size() != (size_t)uw * uh * 8) return 1;

        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        id<MTLTexture> c = tex(dev, rw, rh, MTLPixelFormatRGBA16Float);
        id<MTLTexture> d = tex(dev, rw, rh, MTLPixelFormatR32Float);
        id<MTLTexture> m = tex(dev, rw, rh, MTLPixelFormatRG16Float);
        id<MTLTexture> o = tex(dev, uw, uh, MTLPixelFormatRGBA16Float);
        [c replaceRegion:MTLRegionMake2D(0, 0, rw, rh) mipmapLevel:0 withBytes:color.data() bytesPerRow:rw * 8];
        [d replaceRegion:MTLRegionMake2D(0, 0, rw, rh) mipmapLevel:0 withBytes:depth.data() bytesPerRow:rw * 4];
        [m replaceRegion:MTLRegionMake2D(0, 0, rw, rh) mipmapLevel:0 withBytes:mv.data() bytesPerRow:rw * 4];

        MTLFXTemporalScalerDescriptor *desc = [MTLFXTemporalScalerDescriptor new];
        desc.inputWidth = rw;
        desc.inputHeight = rh;
        desc.outputWidth = uw;
        desc.outputHeight = uh;
        desc.colorTextureFormat = MTLPixelFormatRGBA16Float;
        desc.depthTextureFormat = MTLPixelFormatR32Float;
        desc.motionTextureFormat = MTLPixelFormatRG16Float;
        desc.outputTextureFormat = MTLPixelFormatRGBA16Float;
        id<MTLFXTemporalScaler> s = [desc newTemporalScalerWithDevice:dev];
        if (!s) { fprintf(stderr, "no scaler\n"); return 1; }
        s.colorTexture = c;
        s.depthTexture = d;
        s.motionTexture = m;
        s.outputTexture = o;
        s.inputContentWidth = rw;
        s.inputContentHeight = rh;
        s.jitterOffsetX = 0.f;
        s.jitterOffsetY = (float)strtod("-0.166666657", nullptr);
        s.motionVectorScaleX = 1.f;
        s.motionVectorScaleY = 1.f;
        s.preExposure = 1.f;
        s.reset = YES;
        s.depthReversed = NO;
        for (int repeat = 0; repeat < repeats; ++repeat) {
            if (mtl4) {
                if (@available(macOS 26.0, *)) {
                    id<MTL4CommandQueue> q = [dev newMTL4CommandQueue];
                    id<MTL4CommandAllocator> alloc = [dev newCommandAllocator];
                    id<MTL4CommandBuffer> mcb = [dev newCommandBuffer];
                    if (!q || !alloc || !mcb) { fprintf(stderr, "no mtl4 queue\n"); return 1; }
                    [mcb beginCommandBufferWithAllocator:alloc];
                    int rc = fsr4_encode_live((__bridge void *)dev, (__bridge void *)mcb,
                                              (__bridge void *)c, (__bridge void *)d,
                                              (__bridge void *)m, (__bridge void *)o, nullptr,
                                              rw, rh, uw, uh, 0.f, s.jitterOffsetY, 1.f, 1.f, 1.f, 1, 0);
                    if (!rc) { fprintf(stderr, "mtl4 encode returned 0\n"); return 1; }
                    [mcb endCommandBuffer];
                    id<MTL4CommandBuffer> bufs[1] = { mcb };
                    [q commit:bufs count:1];
                    id<MTLSharedEvent> ev = [dev newSharedEvent];
                    [q signalEvent:ev value:1];
                    if (![ev waitUntilSignaledValue:1 timeoutMS:20000]) {
                        fprintf(stderr, "mtl4 wait timed out\n");
                        return 1;
                    }
                } else { fprintf(stderr, "Metal 4 requires macOS 26\n"); return 1; }
            } else {
                id<MTLCommandBuffer> cb = [[dev newCommandQueue] commandBuffer];
                if (directClassic) {
                    if (!fsr4_encode_live((__bridge void *)dev, (__bridge void *)cb,
                                         (__bridge void *)c, (__bridge void *)d,
                                         (__bridge void *)m, (__bridge void *)o, nullptr,
                                         rw, rh, uw, uh, 0.f, s.jitterOffsetY, 1.f, 1.f, 1.f, 1, 0)) {
                        fprintf(stderr, "classic encode returned 0\n"); return 1;
                    }
                } else [s encodeToCommandBuffer:cb];
                [cb commit];
                [cb waitUntilCompleted];
                if (cb.status != MTLCommandBufferStatusCompleted) {
                    fprintf(stderr, "gpu error: %s\n", cb.error.localizedDescription.UTF8String);
                    return 1;
                }
            }
        }
        std::vector<uint8_t> got((size_t)uw * uh * 8);
        [o getBytes:got.data() bytesPerRow:uw * 8 fromRegion:MTLRegionMake2D(0, 0, uw, uh) mipmapLevel:0];
        FILE *dump = fopen("/tmp/fsr4-live-f00.bin", "wb");
        fwrite(got.data(), 1, got.size(), dump);
        fclose(dump);
        double acc = 0;
        float mx = 0;
        size_t n = (size_t)uw * uh;
        const uint16_t *a = (const uint16_t *)got.data();
        const uint16_t *b = (const uint16_t *)ref.data();
        int nonzero = 0;
        int big = 0;
        float chMax[4] = {0, 0, 0, 0};
        for (size_t i = 0; i < n; i++) {
            for (int c = 0; c < 4; c++) {
                float dlt = fabsf(h2f(a[i * 4 + c]) - h2f(b[i * 4 + c]));
                acc += dlt;
                if (dlt > mx) mx = dlt;
                if (dlt > chMax[c]) chMax[c] = dlt;
                if (dlt > 0.01f) big++;
            }
            if (a[i * 4] || a[i * 4 + 1] || a[i * 4 + 2]) nonzero++;
        }
        double mean = acc / (n * 4);
        printf("live vs fsr4mac  mean_abs=%.6f max_abs=%.6f chMax=%.4f %.4f %.4f %.4f big=%d nonzero_px=%d/%zu\n",
               mean, mx, chMax[0], chMax[1], chMax[2], chMax[3], big, nonzero, n);
        if (nonzero == 0 || mean > 0.001) return 1;
        printf("live ok\n");
    }
    return 0;
}
