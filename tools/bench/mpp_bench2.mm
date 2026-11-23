// Specialized-per-geometry int8/fp16 conv variants and threadgroup-resident
// GEMM probes for mpp_bench2.metal (kernel names embed W x H).
#include "mpp_probe.h"
#include <cmath>

using namespace mpp_probe;

// mode: 0 = device dest, 1 = tg dest (tile-major layout), 2 = fp16 device dest
static void bench(const char *name, int mode, int TW, int TH, int C, int O, int W, int H, bool verify) {
    id<MTLComputePipelineState> pso = pipeline(name);
    if (!pso) return;
    size_t na = (size_t)W * H * C, nw = 9 * C * O;
    int gx = (W + TW - 1) / TW, gy = (H + TH - 1) / TH;
    size_t no = mode == 1 ? (size_t)gx * gy * TW * TH * O : (size_t)W * H * O;
    int es = mode == 2 ? 2 : 1;
    id<MTLBuffer> a = mkbuf(na * es), w = mkbuf(nw * es), o = mkbuf(no * 4);
    std::vector<int8_t> ai(na), wi(nw);
    fill_i8(ai.data(), na, 1); fill_i8(wi.data(), nw, 0);
    if (mode == 2) {
        __fp16 *ap = (__fp16 *)a.contents, *wp = (__fp16 *)w.contents;
        for (size_t i = 0; i < na; i++) ap[i] = (__fp16)ai[i];
        for (size_t i = 0; i < nw; i++) wp[i] = (__fp16)wi[i];
    } else {
        memcpy(a.contents, ai.data(), na); memcpy(w.contents, wi.data(), nw);
    }
    memset(o.contents, 0x7f, no * 4);
    ConvP p = {(uint32_t)W, (uint32_t)H, (uint32_t)C, (uint32_t)O, (uint32_t)TW, (uint32_t)TH};
    double ms = time_dispatch(pso, MTLSizeMake(gx, gy, 1), {a, w, o}, &p, sizeof p, verify ? 2 : 6);
    double gop = 2.0 * W * H * 9.0 * C * O / 1e9;
    printf("%-34s tile %2dx%-2d: %.4f ms  %6.0f GOP/s", name, TW, TH, ms, gop / ms * 1e3);
    if (verify) {
        std::vector<int32_t> ref((size_t)W * H * O);
        conv_ref(ai.data(), wi.data(), ref.data(), W, H, C, O);
        size_t bad = 0, border = 0;
        for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) for (int oo = 0; oo < O; oo++) {
            long long g;
            if (mode == 1) {
                int tx = x / TW, ty = y / TH, lx = x % TW, ly = y % TH;
                g = ((const int32_t *)o.contents)[((size_t)(ty * gx + tx) * TW * TH + ly * TW + lx) * O + oo];
            } else if (mode == 2) g = llround(((const float *)o.contents)[((size_t)y * W + x) * O + oo]);
            else g = ((const int32_t *)o.contents)[((size_t)y * W + x) * O + oo];
            if (g != ref[((size_t)y * W + x) * O + oo]) { bad++; if (x == 0 || y == 0 || x == W - 1 || y == H - 1) border++; }
        }
        printf("  verify: %zu bad (border %zu) of %zu", bad, border, (size_t)W * H * O);
    }
    printf("\n");
}

static void bench_gemm(const char *name, int N, int K, int M, int rep, bool verify) {
    id<MTLComputePipelineState> pso = pipeline(name);
    if (!pso) return;
    id<MTLBuffer> a = mkbuf((size_t)M * K), b = mkbuf((size_t)K * N), c = mkbuf((size_t)M * N * 4);
    int8_t *ap = (int8_t *)a.contents, *bp = (int8_t *)b.contents;
    fill_i8(ap, (size_t)M * K, 3); fill_i8(bp, (size_t)K * N, 0);
    GemmP p = {(uint32_t)M, (uint32_t)N, (uint32_t)K, (uint32_t)rep};
    double ms = time_dispatch(pso, MTLSizeMake(M / 32, 1, 1), {a, b, c}, &p, sizeof p, verify ? 2 : 5);
    double gop = 2.0 * M * N * K * rep / 1e9;
    printf("%-22s M%-7d N%-4d K%-4d rep%-3d: %.4f ms  %6.0f GOP/s", name, M, N, K, rep, ms, gop / ms * 1e3);
    if (verify && rep == 1) {
        const int32_t *got = (const int32_t *)c.contents;
        size_t bad = 0;
        for (int m = 0; m < M; m += 37) for (int n = 0; n < N; n++) {
            int32_t acc = 0;
            for (int k = 0; k < K; k++) acc += (int)ap[m * K + k] * (int)bp[k * N + n];
            if (acc != got[m * N + n]) bad++;
        }
        printf("  verify: %zu bad", bad);
    }
    printf("\n");
}


// extra checks: (a) big source dims, (b) stride-2 2x2, (c) grouped via strided view
static void check_extra() {
    const int W = 96, H = 20;
    {   // (a)
        printf("--- (a) source dims 128x32 > image 96x20 ---\n");
        bench("conv_big_32x4_16_16_128x32", 1, 32, 4, 16, 16, 96, 20, true);
    }
    {   // (b) stride 2
        printf("--- (b) 2x2 stride 2, C16->O32 ---\n");
        id<MTLComputePipelineState> pso = pipeline("conv_s2_32x4_16_32_96x20");
        if (!pso) return;
        const int C = 16, O = 32, OW = W / 2, OH = H / 2, TW = 32, TH = 4;
        int gx = (OW + TW - 1) / TW, gy = (OH + TH - 1) / TH;
        size_t na = (size_t)W * H * C, nw = 4 * C * O, no = (size_t)gx * gy * TW * TH * O;
        id<MTLBuffer> a = mkbuf(na), w = mkbuf(nw), o = mkbuf(no * 4);
        int8_t *ap = (int8_t *)a.contents, *wp = (int8_t *)w.contents;
        fill_i8(ap, na, 5); fill_i8(wp, nw, 0);
        for (int mul = 1; mul <= 2; mul++) {
        ConvP p = {(uint32_t)W, (uint32_t)H, (uint32_t)mul, O, TW, TH};
        time_dispatch(pso, MTLSizeMake(gx, gy, 1), {a, w, o}, &p, sizeof p, 1, 1);
        size_t bad = 0, firstTileBad = 0;
        for (int y = 0; y < OH; y++) for (int x = 0; x < OW; x++) for (int oo = 0; oo < O; oo++) {
            int32_t acc = 0;
            for (int ky = 0; ky < 2; ky++) for (int kx = 0; kx < 2; kx++) {
                int yy = y * 2 + ky, xx = x * 2 + kx;
                for (int c = 0; c < C; c++) acc += (int)ap[((yy * W + xx) * C) + c] * (int)wp[((ky * 2 + kx) * C + c) * O + oo];
            }
            int tx = x / TW, ty = y / TH, lx = x % TW, ly = y % TH;
            int32_t g = ((const int32_t *)o.contents)[((size_t)(ty * gx + tx) * TW * TH + ly * TW + lx) * O + oo];
            if (g != acc) { bad++; if (x < TW && y < TH) firstTileBad++; }
        }
        printf("stride2 offset mul %d: %zu bad of %d (first tile %zu)\n", mul, bad, OW * OH * O, firstTileBad);
        if (mul == 1) for (int sy = -2; sy <= 2; sy++) for (int sx = -2; sx <= 2; sx++) {
            size_t b2 = 0, t2 = 0;
            for (int y = 2; y < OH - 2; y++) for (int x = 2; x < OW - 2; x++) for (int oo = 0; oo < O; oo++) {
                int32_t acc = 0;
                for (int ky = 0; ky < 2; ky++) for (int kx = 0; kx < 2; kx++) {
                    int yy = y * 2 + ky + sy, xx = x * 2 + kx + sx;
                    for (int c = 0; c < C; c++) acc += (int)ap[((yy * W + xx) * C) + c] * (int)wp[((ky * 2 + kx) * C + c) * O + oo];
                }
                int tx = x / TW, ty = y / TH, lx = x % TW, ly = y % TH;
                int32_t g = ((const int32_t *)o.contents)[((size_t)(ty * gx + tx) * TW * TH + ly * TW + lx) * O + oo];
                t2++; if (g != acc) b2++;
            }
            if (b2 * 4 < t2) printf("  stride2 interior matches with source shift (%d,%d): %zu/%zu bad\n", sx, sy, b2, t2);
        }
        }
        id<MTLComputePipelineState> pso2 = pipeline("conv_s2_32x4_16_32_960x540");
        id<MTLBuffer> a2 = mkbuf((size_t)960 * 540 * 16), o2 = mkbuf((size_t)480 * 272 * 32 * 4);
        ConvP p2 = {960, 540, 16, 32, 32, 4};
        double ms = time_dispatch(pso2, MTLSizeMake(15, 68, 1), {a2, w, o2}, &p2, sizeof p2, 5);
        printf("stride2 960x540->480x270 C16->O32: %.4f ms\n", ms);
    }
    {   // (c) grouped
        printf("--- (c) grouped conv via strided channel view (CT=32, group 1) ---\n");
        id<MTLComputePipelineState> pso = pipeline("conv_grp_32x4_16of32_16_96x20");
        if (!pso) return;
        const int C = 16, CT = 32, O = 16, TW = 32, TH = 4;
        int gx = (W + TW - 1) / TW, gy = (H + TH - 1) / TH;
        size_t na = (size_t)W * H * CT, nw = 9 * C * O, no = (size_t)gx * gy * TW * TH * O;
        id<MTLBuffer> a = mkbuf(na), w = mkbuf(nw), o = mkbuf(no * 4);
        int8_t *ap = (int8_t *)a.contents, *wp = (int8_t *)w.contents;
        fill_i8(ap, na, 6); fill_i8(wp, nw, 0);
        for (int g = 0; g < 2; g++) {
            ConvP p = {(uint32_t)W, (uint32_t)H, C, O, TW, (uint32_t)g};
            time_dispatch(pso, MTLSizeMake(gx, gy, 1), {a, w, o}, &p, sizeof p, 1, 1);
            size_t bad = 0;
            for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) for (int oo = 0; oo < O; oo++) {
                int32_t acc = 0;
                for (int ky = 0; ky < 3; ky++) for (int kx = 0; kx < 3; kx++) {
                    int yy = y + ky - 1, xx = x + kx - 1;
                    if (yy < 0 || xx < 0 || yy >= H || xx >= W) continue;
                    for (int c = 0; c < C; c++) acc += (int)ap[((yy * W + xx) * CT) + g * C + c] * (int)wp[((ky * 3 + kx) * C + c) * O + oo];
                }
                int tx = x / TW, ty = y / TH, lx = x % TW, ly = y % TH;
                int32_t gg = ((const int32_t *)o.contents)[((size_t)(ty * gx + tx) * TW * TH + ly * TW + lx) * O + oo];
                if (gg != acc) bad++;
            }
            printf("group %d: %zu bad of %d\n", g, bad, W * H * O);
        }
    }
}

int main(int argc, char **argv) {
    @autoreleasepool {
        if (!open("/tmp/mpp_bench2.metallib")) return 1;
        if (argc > 1) { check_extra(); return 0; }
        printf("--- correctness incl. borders (96x20) ---\n");
        bench("conv_dv_32x4_16_16_96x20", 0, 32, 4, 16, 16, 96, 20, true);
        bench("conv_dv_32x4_16_16_128x32", 0, 32, 4, 16, 16, 96, 20, true);
        bench("conv_tg_32x4_16_16_96x20", 1, 32, 4, 16, 16, 96, 20, true);
        bench("convf_dv_32x4_16_16_96x20", 2, 32, 4, 16, 16, 96, 20, true);
        bench_gemm("gemm_tg_n32_k16", 32, 16, 4096, 1, true);
        bench_gemm("gemm_tg_n64_k128", 64, 128, 4096, 1, true);
        printf("--- enc0 960x540 C16->O16 3x3 int8 ---\n");
        bench("conv_dv_32x4_16_16_960x540", 0, 32, 4, 16, 16, 960, 540, false);
        bench("conv_dv_64x4_16_16_960x540", 0, 64, 4, 16, 16, 960, 540, false);
        bench("conv_dv_32x8_16_16_960x540", 0, 32, 8, 16, 16, 960, 540, false);
        bench("conv_dv_16x8_16_16_960x540", 0, 16, 8, 16, 16, 960, 540, false);
        bench("conv_dv_64x8_16_16_960x540", 0, 64, 8, 16, 16, 960, 540, false);
        bench("conv_tg_32x4_16_16_960x540", 1, 32, 4, 16, 16, 960, 540, false);
        bench("conv_tg_64x4_16_16_960x540", 1, 64, 4, 16, 16, 960, 540, false);
        bench("conv_tg_32x8_16_16_960x540", 1, 32, 8, 16, 16, 960, 540, false);
        bench("conv_tg_64x8_16_16_960x540", 1, 64, 8, 16, 16, 960, 540, false);
        bench("convf_dv_32x4_16_16_960x540", 2, 32, 4, 16, 16, 960, 540, false);
        bench("convf_dv_64x4_16_16_960x540", 2, 64, 4, 16, 16, 960, 540, false);
        printf("--- enc1/enc2 ---\n");
        bench("conv_tg_32x4_16_16_480x270", 1, 32, 4, 16, 16, 480, 270, false);
        bench("conv_tg_64x4_16_16_480x270", 1, 64, 4, 16, 16, 480, 270, false);
        bench("conv_tg_32x4_16_32_240x135", 1, 32, 4, 16, 32, 240, 135, false);
        bench("conv_tg_16x8_16_32_240x135", 1, 16, 8, 16, 32, 240, 135, false);
        printf("--- matmul2d int8, threadgroup-resident A/C (rep=16 => compute-bound) ---\n");
        bench_gemm("gemm_tg_n32_k16", 32, 16, 518400, 1, false);
        bench_gemm("gemm_tg_n32_k16", 32, 16, 518400, 16, false);
        bench_gemm("gemm_tg_n16_k32", 16, 32, 518400, 16, false);
        bench_gemm("gemm_tg_n64_k32", 64, 32, 129600, 16, false);
        bench_gemm("gemm_tg_n32_k64", 32, 64, 129600, 16, false);
        bench_gemm("gemm_tg_n128_k64", 128, 64, 32416, 16, false);
        bench_gemm("gemm_tg_n64_k128", 64, 128, 32416, 16, false);
    }
    return 0;
}
