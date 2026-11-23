#include "mpp_probe.h"

using namespace mpp_probe;

struct DP { uint32_t W, H; int ox, oy, sx, sy, dx, dy; uint32_t useSliceA, useSliceD; };

int main(int argc, char **argv) {
    @autoreleasepool {
        if (!open("/tmp/mpp_diag.metallib")) return 1;
        const int W = 96, H = 20, C = 16, O = 16;
        size_t na = (size_t)W * H * C, nw = 9 * C * O, no = (size_t)W * H * O;
        id<MTLBuffer> a = mkbuf(na), w = mkbuf(nw), o = mkbuf(no * 4);
        fill_i8(a, na, 1); fill_i8(w, nw, 0);
        int8_t *ap = (int8_t *)a.contents, *wp = (int8_t *)w.contents;
        struct K { const char *n; int dw, dh; bool tiled; } ks[] = {
            {"diag_32x4_34x6", 32, 4, true}, {"diag_96x20_96x20", 96, 20, false},
            {"diag_32x4_96x20", 32, 4, true}, {"diag_32x4_32x4", 32, 4, true},
            {"diag_96x20_98x22", 96, 20, false}};
        typedef int (*WF)(int, int, int, int);
        struct WL { const char *n; WF f; } wls[] = {
            {"HWIO", [](int ky, int kx, int c, int oo) { return ((ky * 3 + kx) * 16 + c) * 16 + oo; }},
            {"HWOI", [](int ky, int kx, int c, int oo) { return ((ky * 3 + kx) * 16 + oo) * 16 + c; }},
            {"OIHW", [](int ky, int kx, int c, int oo) { return ((oo * 16 + c) * 3 + ky) * 3 + kx; }},
            {"OHWI", [](int ky, int kx, int c, int oo) { return ((oo * 3 + ky) * 3 + kx) * 16 + c; }},
        };
        for (auto &k : ks) {
            id<MTLFunction> fn = [lib newFunctionWithName:[NSString stringWithUTF8String:k.n]];
            NSError *err = nil;
            id<MTLComputePipelineState> pso = [dev newComputePipelineStateWithFunction:fn error:&err];
            if (!pso) { printf("%s pso: %s\n", k.n, err.localizedDescription.UTF8String); continue; }
            int offs[][2] = {{0, 0}, {-1, -1}, {-1000, -1000}, {1, 1}};   // -1000 => plain 0 without tile add
            int sl[][2] = {{0, 0}, {-1, -1}, {1, 1}};
            for (int oi = 0; oi < 4; oi++) for (int ua = 0; ua < 2; ua++) for (int ud = 0; ud < 2; ud++)
            for (int si = 0; si < (ua ? 3 : 1); si++) for (int di = 0; di < (ud ? 1 : 1); di++) {
                DP p = {};
                p.W = W; p.H = H;
                p.ox = offs[oi][0] == -1000 ? -1 : offs[oi][0];   // -1: no tile add (see kernel)
                p.oy = offs[oi][1] == -1000 ? -1 : offs[oi][1];
                if (offs[oi][0] == -1) { p.ox = -1; p.oy = -1; }
                p.sx = sl[si][0]; p.sy = sl[si][1];
                p.dx = 0; p.dy = 0;
                p.useSliceA = ua; p.useSliceD = ud;
                memset(o.contents, 0x7f, no * 4);
                id<MTLCommandBuffer> cb = [q commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                [enc setComputePipelineState:pso];
                [enc setBuffer:a offset:0 atIndex:0];
                [enc setBuffer:w offset:0 atIndex:1];
                [enc setBuffer:o offset:0 atIndex:2];
                [enc setBytes:&p length:sizeof p atIndex:3];
                MTLSize g = k.tiled ? MTLSizeMake(W / k.dw, H / k.dh, 1) : MTLSizeMake(1, 1, 1);
                [enc dispatchThreadgroups:g threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                [enc endEncoding];
                [cb commit];
                [cb waitUntilCompleted];
                if (cb.status != MTLCommandBufferStatusCompleted) { printf("%s: gpu error\n", k.n); continue; }
                const int32_t *got = (const int32_t *)o.contents;
                size_t untouched = 0, zeros = 0;
                for (size_t i = 0; i < no; i++) { if (got[i] == 0x7f7f7f7f) untouched++; else if (got[i] == 0) zeros++; }
                // search
                for (auto &wl : wls) for (int dy = -2; dy <= 2; dy++) for (int dx = -2; dx <= 2; dx++) {
                    size_t bad = 0, tot = 0;
                    for (int y = 3; y < H - 3; y++) for (int x = 3; x < W - 3; x++) for (int oo = 0; oo < O; oo++) {
                        int32_t acc = 0;
                        for (int ky = 0; ky < 3; ky++) for (int kx = 0; kx < 3; kx++) {
                            int yy = y + ky - 1 + dy, xx = x + kx - 1 + dx;
                            for (int c = 0; c < C; c++)
                                acc += (int)ap[((yy * W + xx) * C) + c] * (int)wp[wl.f(ky, kx, c, oo)];
                        }
                        tot++;
                        if (got[(y * W + x) * O + oo] != acc) bad++;
                    }
                    if (bad * 4 < tot)
                        printf("%s off=%d,%d sliceA=%d(%d,%d) sliceD=%d: %s shift=%d,%d bad %zu/%zu untouched %zu zeros %zu\n",
                               k.n, p.ox, p.oy, ua, p.sx, p.sy, ud, wl.n, dx, dy, bad, tot, untouched, zeros);
                }
                if (oi == 0 && ua == 0 && ud == 0)
                    printf("%s baseline: untouched %zu zeros %zu of %zu, got[0]=%d got[16*W*3+16*3]=%d\n",
                           k.n, untouched, zeros, no, got[0], got[(3 * W + 3) * 16]);
            }
        }
    }
    return 0;
}
