// Host for tools/bench/mpp_bench.metal: verify vs CPU, then time.
#include "mpp_probe.h"
#include <cmath>
#include <string>

using namespace mpp_probe;

static void bench_conv_i8(const char *name, int TW, int TH, int C, int O, int W, int H, bool verify) {
    id<MTLComputePipelineState> pso = pipeline(name);
    if (!pso) return;
    size_t na = (size_t)W * H * C, nw = 9 * C * O, no = (size_t)W * H * O;
    id<MTLBuffer> a = mkbuf(na), w = mkbuf(nw), o = mkbuf(no * 4);
    fill_i8(a, na, 1); fill_i8(w, nw, 0);
    memset(o.contents, 0, no * 4);
    ConvP p = {(uint32_t)W, (uint32_t)H, (uint32_t)C, (uint32_t)O, (uint32_t)TW, (uint32_t)TH};
    MTLSize groups = MTLSizeMake((W + TW - 1) / TW, (H + TH - 1) / TH, 1);
    double ms = time_dispatch(pso, groups, {a, w, o}, &p, sizeof p, verify ? 2 : 6);
    double gflop = 2.0 * W * H * 9.0 * C * O / 1e9;
    printf("%-26s %4dx%-4d C%-3d O%-3d tile %dx%-3d: %.3f ms  %.0f GOP/s", name, W, H, C, O, TW, TH, ms, gflop / ms * 1e3);
    if (verify) {
        std::vector<int32_t> ref(no);
        conv_ref((const int8_t *)a.contents, (const int8_t *)w.contents, ref.data(), W, H, C, O);
        const int32_t *got = (const int32_t *)o.contents;
        size_t bad = 0, interiorBad = 0; long long maxd = 0;
        for (size_t i = 0; i < no; i++) {
            long long d = llabs((long long)got[i] - ref[i]);
            if (d) {
                bad++;
                size_t px = i / O; int x = px % W, y = px / W;
                if (x > 0 && y > 0 && x < W - 1 && y < H - 1) interiorBad++;
            }
            if (d > maxd) maxd = d;
        }
        printf("  verify: %zu/%zu mismatches (interior %zu), max diff %lld", bad, no, interiorBad, maxd);
    }
    printf("\n");
}

static void bench_conv_f16(const char *name, int TW, int TH, int C, int O, int W, int H, bool verify) {
    id<MTLComputePipelineState> pso = pipeline(name);
    if (!pso) return;
    size_t na = (size_t)W * H * C, nw = 9 * C * O, no = (size_t)W * H * O;
    id<MTLBuffer> a = mkbuf(na * 2), w = mkbuf(nw * 2), o = mkbuf(no * 4);
    __fp16 *ap = (__fp16 *)a.contents, *wp = (__fp16 *)w.contents;
    std::vector<int8_t> ai(na), wi(nw);
    srand(2);
    for (size_t i = 0; i < na; i++) { ai[i] = (int8_t)(rand() % 256 - 128); ap[i] = (__fp16)ai[i]; }
    for (size_t i = 0; i < nw; i++) { wi[i] = (int8_t)(rand() % 256 - 128); wp[i] = (__fp16)wi[i]; }
    memset(o.contents, 0, no * 4);
    ConvP p = {(uint32_t)W, (uint32_t)H, (uint32_t)C, (uint32_t)O, (uint32_t)TW, (uint32_t)TH};
    MTLSize groups = MTLSizeMake((W + TW - 1) / TW, (H + TH - 1) / TH, 1);
    double ms = time_dispatch(pso, groups, {a, w, o}, &p, sizeof p, verify ? 2 : 6);
    double gflop = 2.0 * W * H * 9.0 * C * O / 1e9;
    printf("%-26s %4dx%-4d C%-3d O%-3d tile %dx%-3d: %.3f ms  %.0f GFLOP/s", name, W, H, C, O, TW, TH, ms, gflop / ms * 1e3);
    if (verify) {
        std::vector<int32_t> ref(no);
        conv_ref(ai.data(), wi.data(), ref.data(), W, H, C, O);
        const float *got = (const float *)o.contents;
        size_t bad = 0; double maxd = 0;
        for (size_t i = 0; i < no; i++) {
            double d = fabs((double)got[i] - ref[i]);
            if (d > 0.5) bad++;
            if (d > maxd) maxd = d;
        }
        printf("  verify: %zu/%zu mismatches, max diff %.1f", bad, no, maxd);
    }
    printf("\n");
}

static void bench_gemm_i8(const char *name, int N, int K, int M, bool verify) {
    id<MTLComputePipelineState> pso = pipeline(name);
    if (!pso) return;
    id<MTLBuffer> a = mkbuf((size_t)M * K), b = mkbuf((size_t)K * N), c = mkbuf((size_t)M * N * 4);
    fill_i8(a, (size_t)M * K, 3); fill_i8(b, (size_t)K * N, 0);
    GemmP p = {(uint32_t)M, (uint32_t)N, (uint32_t)K, 0};
    double ms = time_dispatch(pso, MTLSizeMake(M / 32, 1, 1), {a, b, c}, &p, sizeof p, verify ? 2 : 6);
    double gop = 2.0 * M * N * K / 1e9;
    printf("%-26s M%-8d N%-4d K%-4d: %.3f ms  %.0f GOP/s", name, M, N, K, ms, gop / ms * 1e3);
    if (verify) {
        size_t bad = gemm_verify((const int8_t *)a.contents, (const int8_t *)b.contents,
                                 (const int32_t *)c.contents, M, N, K, 97);
        printf("  verify(sampled): %zu mismatches", bad);
    }
    printf("\n");
}


// Search which interpretation the op implements: activation shift (dx,dy),
// weight index permutation, presence of implicit zero padding.
static void diag_conv_i8(const char *name, int TW, int TH, int C, int O, int W, int H) {
    id<MTLComputePipelineState> pso = pipeline(name);
    if (!pso) return;
    size_t na = (size_t)W * H * C, nw = 9 * C * O, no = (size_t)W * H * O;
    id<MTLBuffer> a = mkbuf(na), w = mkbuf(nw), o = mkbuf(no * 4);
    fill_i8(a, na, 1); fill_i8(w, nw, 0);
    memset(o.contents, 0x7f, no * 4);
    ConvP p = {(uint32_t)W, (uint32_t)H, (uint32_t)C, (uint32_t)O, (uint32_t)TW, (uint32_t)TH};
    time_dispatch(pso, MTLSizeMake((W + TW - 1) / TW, (H + TH - 1) / TH, 1), {a, w, o}, &p, sizeof p, 1);
    const int32_t *got = (const int32_t *)o.contents;
    const int8_t *ap = (const int8_t *)a.contents, *wp = (const int8_t *)w.contents;
    // weight index candidates: (ky,kx,c,o) -> linear
    struct WL { const char *n; int (*f)(int,int,int,int,int,int); };
    WL wls[] = {
        {"HWIO o-fast", [](int ky,int kx,int c,int oo,int C,int O){ return ((ky*3+kx)*C+c)*O+oo; }},
        {"HWOI c-fast", [](int ky,int kx,int c,int oo,int C,int O){ return ((ky*3+kx)*O+oo)*C+c; }},
        {"WHIO",        [](int ky,int kx,int c,int oo,int C,int O){ return ((kx*3+ky)*C+c)*O+oo; }},
        {"OHWI",        [](int ky,int kx,int c,int oo,int C,int O){ return ((oo*3+ky)*3+kx)*C+c; }},
        {"OIHW",        [](int ky,int kx,int c,int oo,int C,int O){ return ((oo*C+c)*3+ky)*3+kx; }},
        {"IHWO",        [](int ky,int kx,int c,int oo,int C,int O){ return ((c*3+ky)*3+kx)*O+oo; }},
    };
    for (auto &wl : wls) for (int dy = -2; dy <= 2; dy++) for (int dx = -2; dx <= 2; dx++) for (int flip = 0; flip < 2; flip++) {
        size_t bad = 0, tot = 0;
        for (int y = 3; y < H - 3; y++) for (int x = 3; x < W - 3; x++) for (int oo = 0; oo < O; oo++) {
            int32_t acc = 0;
            for (int ky = 0; ky < 3; ky++) for (int kx = 0; kx < 3; kx++) {
                int fy = flip ? 2 - ky : ky, fx = flip ? 2 - kx : kx;
                int yy = y + ky - 1 + dy, xx = x + kx - 1 + dx;
                for (int c = 0; c < C; c++)
                    acc += (int)ap[((yy * W + xx) * C) + c] * (int)wp[wl.f(fy, fx, c, oo, C, O)];
            }
            tot++;
            if (got[(y * W + x) * O + oo] != acc) bad++;
        }
        if (bad * 20 < tot) printf("  MATCH %s dy=%d dx=%d flip=%d: %zu/%zu bad\n", wl.n, dy, dx, flip, bad, tot);
    }
    // border behaviour: count exact matches at x==0 with zero padding assumption
    printf("  sample got[0..3]=%d %d %d %d\n", got[0], got[1], got[2], got[3]);
}

int main(int argc, char **argv) {
    @autoreleasepool {
        const char *path = argc > 1 ? argv[1] : "/tmp/mpp_bench.metallib";
        if (!open(path)) return 1;
        printf("device: %s\n", dev.name.UTF8String);

        if (argc > 2) { diag_conv_i8("conv_i8_32x4_16_16", 32, 4, 16, 16, 96, 20); return 0; }
        printf("--- correctness (small) ---\n");
        bench_conv_i8("conv_i8_32x4_16_16", 32, 4, 16, 16, 96, 20, true);
        bench_conv_i8("conv_i8_16x8_16_16", 16, 8, 16, 16, 96, 20, true);
        bench_conv_f16("conv_f16_32x4_16_16", 32, 4, 16, 16, 96, 20, true);
        bench_gemm_i8("gemm_i8_n32_k16", 32, 16, 4096, true);
        bench_gemm_i8("gemm_i8_n64_k32", 64, 32, 4096, true);

        printf("--- enc0: 960x540, C16 -> O16, 3x3 (pass1 dw conv) ---\n");
        const char *n16[] = {"conv_i8_32x4_16_16", "conv_i8_16x8_16_16", "conv_i8_64x4_16_16", "conv_i8_32x8_16_16", "conv_i8_16x16_16_16"};
        int t16[][2] = {{32, 4}, {16, 8}, {64, 4}, {32, 8}, {16, 16}};
        for (int i = 0; i < 5; i++) bench_conv_i8(n16[i], t16[i][0], t16[i][1], 16, 16, 960, 540, false);
        bench_conv_f16("conv_f16_32x4_16_16", 32, 4, 16, 16, 960, 540, false);
        bench_conv_f16("conv_f16_16x8_16_16", 16, 8, 16, 16, 960, 540, false);
        bench_conv_f16("conv_f16_32x8_16_16", 32, 8, 16, 16, 960, 540, false);
        printf("--- enc1: 480x270 C16->O16 (FNB32 dw), C32->O32 ---\n");
        bench_conv_i8("conv_i8_32x4_16_16", 32, 4, 16, 16, 480, 270, false);
        bench_conv_i8("conv_i8_32x4_32_32", 32, 4, 32, 32, 480, 270, false);
        bench_conv_i8("conv_i8_16x8_32_32", 16, 8, 32, 32, 480, 270, false);
        printf("--- enc2: 240x135 C16->O32 (FNB64 dw group) ---\n");
        bench_conv_i8("conv_i8_32x4_16_32", 32, 4, 16, 32, 240, 135, false);
        printf("--- pw convs as GEMM over pixels (M = 518400 = 960x540) ---\n");
        bench_gemm_i8("gemm_i8_n32_k16", 32, 16, 518400, false);   // CNB expand 16->32
        bench_gemm_i8("gemm_i8_n16_k32", 16, 32, 518400, false);   // CNB contract 32->16
        bench_gemm_i8("gemm_i8_n64_k32", 64, 32, 129600, false);   // FNB32 expand
        bench_gemm_i8("gemm_i8_n32_k64", 32, 64, 129600, false);   // FNB32 contract
        bench_gemm_i8("gemm_i8_n128_k64", 128, 64, 32416, false);  // FNB64 expand
        bench_gemm_i8("gemm_i8_n64_k128", 64, 128, 32416, false);  // FNB64 contract
    }
    return 0;
}
