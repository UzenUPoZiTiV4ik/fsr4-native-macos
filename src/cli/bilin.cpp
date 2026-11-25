// bilin - CPU bilinear upscaler for RGBA16F .bin frames (baseline reference).
//   bilin <in.bin> <rw> <rh> <uw> <uh> <out.bin> [out.png]
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include "pngio.h"

static float h2f(uint16_t u) { union { uint16_t u; _Float16 f; } c; c.u = u; return (float)c.f; }
static uint16_t f2h(float v) { union { uint16_t u; _Float16 f; } c; c.f = (_Float16)v; return c.u; }

int main(int argc, char** argv) {
    if (argc < 7) { fprintf(stderr, "usage: bilin <in.bin> <rw> <rh> <uw> <uh> <out.bin> [out.png]\n"); return 1; }
    int rw = atoi(argv[2]), rh = atoi(argv[3]), uw = atoi(argv[4]), uh = atoi(argv[5]);
    FILE* f = fopen(argv[1], "rb");
    if (!f) return 1;
    std::vector<uint16_t> in((size_t)rw * rh * 4);
    if (fread(in.data(), 2, in.size(), f) != in.size()) return 1;
    fclose(f);
    std::vector<uint16_t> out((size_t)uw * uh * 4);
    for (int y = 0; y < uh; y++) {
        // pixel-centre mapping (align_corners off)
        float sy = ((y + 0.5f) * rh / (float)uh) - 0.5f;
        int y0 = (int)floorf(sy), y1 = y0 + 1;
        float fy = sy - y0;
        if (y0 < 0) y0 = 0; if (y1 > rh - 1) y1 = rh - 1;
        for (int x = 0; x < uw; x++) {
            float sx = ((x + 0.5f) * rw / (float)uw) - 0.5f;
            int x0 = (int)floorf(sx), x1 = x0 + 1;
            float fx = sx - x0;
            if (x0 < 0) x0 = 0; if (x1 > rw - 1) x1 = rw - 1;
            for (int c = 0; c < 4; c++) {
                float v00 = h2f(in[((size_t)y0 * rw + x0) * 4 + c]);
                float v01 = h2f(in[((size_t)y0 * rw + x1) * 4 + c]);
                float v10 = h2f(in[((size_t)y1 * rw + x0) * 4 + c]);
                float v11 = h2f(in[((size_t)y1 * rw + x1) * 4 + c]);
                float v = (v00 * (1 - fx) + v01 * fx) * (1 - fy)
                        + (v10 * (1 - fx) + v11 * fx) * fy;
                out[((size_t)y * uw + x) * 4 + c] = f2h(v);
            }
        }
    }
    FILE* o = fopen(argv[6], "wb");
    if (!o) return 1;
    fwrite(out.data(), 2, out.size(), o); fclose(o);
    if (argc > 7) {
        std::vector<uint8_t> png((size_t)uw * uh * 4);
        for (size_t i = 0; i < (size_t)uw * uh * 4; i++) {
            float v = h2f(out[i]);
            v = v <= 0.f ? 0.f : (v <= 0.0031308f ? 12.92f * v
                    : 1.055f * powf(v, 1.f / 2.4f) - 0.055f);
            png[i] = (uint8_t)(fminf(fmaxf(v, 0.f), 1.f) * 255.f + 0.5f);
        }
        png_write(argv[7], png.data(), uw, uh);
    }
    return 0;
}
