// SPDX-License-Identifier: MIT
// Shared Metal plumbing for the standalone mpp_bench*/mpp_diag probes:
// device setup, shared buffers, pipeline lookup, repeated-dispatch timing
// and the CPU oracles the probes verify against.
#pragma once
#import <Metal/Metal.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

struct ConvP { uint32_t W, H, C, O, TW, TH; };
// `rep` is consumed by the threadgroup-resident matmul2d probe only; the
// device-memory GEMM kernels simply ignore the trailing word.
struct GemmP { uint32_t M, N, K, rep; };

namespace mpp_probe {
inline id<MTLDevice> dev;
inline id<MTLCommandQueue> q;
inline id<MTLLibrary> lib;

inline id<MTLBuffer> mkbuf(size_t n) {
    return [dev newBufferWithLength:n options:MTLResourceStorageModeShared];
}
// seed 0 continues the current rand() stream, matching `srand(s); fill a; fill w`.
inline void fill_i8(int8_t *p, size_t n, unsigned seed) {
    if (seed) srand(seed);
    for (size_t i = 0; i < n; i++) p[i] = (int8_t)(rand() % 256 - 128);
}
inline void fill_i8(id<MTLBuffer> b, size_t n, unsigned seed) {
    fill_i8(static_cast<int8_t *>(b.contents), n, seed);
}
// Device, queue and metallib in one step; prints the loader error on failure.
inline bool open(const char *path) {
    dev = MTLCreateSystemDefaultDevice();
    q = [dev newCommandQueue];
    NSError *err = nil;
    lib = [dev newLibraryWithURL:[NSURL fileURLWithPath:[NSString stringWithUTF8String:path]] error:&err];
    if (!lib) printf("lib: %s\n", err.localizedDescription.UTF8String);
    return lib != nil;
}
inline id<MTLComputePipelineState> pipeline(const char *name) {
    id<MTLFunction> fn = [lib newFunctionWithName:[NSString stringWithUTF8String:name]];
    if (!fn) { printf("%s: missing\n", name); return nil; }
    NSError *err = nil;
    id<MTLComputePipelineState> p = [dev newComputePipelineStateWithFunction:fn error:&err];
    if (!p) printf("%s: pso error %s\n", name, err.localizedDescription.UTF8String);
    return p;
}
// Best of `iters` command buffers, each encoding `reps` 128-thread dispatches.
inline double time_dispatch(id<MTLComputePipelineState> pso, MTLSize groups,
                            std::vector<id<MTLBuffer>> bufs, const void *cbytes, size_t clen,
                            int iters, int reps = 10) {
    double best = 1e9;
    for (int it = 0; it < iters; it++) {
        id<MTLCommandBuffer> cb = [q commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:pso];
        for (size_t i = 0; i < bufs.size(); i++) [enc setBuffer:bufs[i] offset:0 atIndex:i];
        [enc setBytes:cbytes length:clen atIndex:bufs.size()];
        for (int r = 0; r < reps; r++) [enc dispatchThreadgroups:groups threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        if (cb.status != MTLCommandBufferStatusCompleted) {
            printf("  gpu error: %s\n", cb.error.localizedDescription.UTF8String);
            return -1;
        }
        double ms = (cb.GPUEndTime - cb.GPUStartTime) * 1000.0 / reps;
        if (ms < best) best = ms;
    }
    return best;
}
// conv 3x3 same padding CPU ref, NHWC act (C fastest), HWIO weights (O fastest).
inline void conv_ref(const int8_t *act, const int8_t *w, int32_t *out, int W, int H, int C, int O) {
    for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) for (int o = 0; o < O; o++) {
        int32_t acc = 0;
        for (int ky = 0; ky < 3; ky++) for (int kx = 0; kx < 3; kx++) {
            int yy = y + ky - 1, xx = x + kx - 1;
            if (yy < 0 || xx < 0 || yy >= H || xx >= W) continue;
            for (int c = 0; c < C; c++)
                acc += (int)act[((yy * W + xx) * C) + c] * (int)w[((ky * 3 + kx) * C + c) * O + o];
        }
        out[(y * W + x) * O + o] = acc;
    }
}
// Sampled int8 GEMM check: every `step`-th row of A, all N columns.
inline size_t gemm_verify(const int8_t *a, const int8_t *b, const int32_t *c,
                          int M, int N, int K, int step) {
    size_t bad = 0;
    for (int m = 0; m < M; m += step) for (int n = 0; n < N; n++) {
        int32_t acc = 0;
        for (int k = 0; k < K; k++) acc += (int)a[m * K + k] * (int)b[k * N + n];
        if (acc != c[m * N + n]) bad++;
    }
    return bad;
}
} // namespace mpp_probe
