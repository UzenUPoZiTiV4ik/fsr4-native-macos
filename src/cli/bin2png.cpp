// bin2png — convert a raw RGBA16F .bin to PNG with the exact same
// linear→sRGB path as fsr4mac's output writer.
//   bin2png <in.bin> <w> <h> <out.png> [colorspace: linear|srgb]
#include "pngio.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 5) {
        fprintf(stderr, "usage: bin2png <in.bin> <w> <h> <out.png> [linear|srgb]\n");
        return 1;
    }
    const char* in = argv[1];
    int w = atoi(argv[2]), h = atoi(argv[3]);
    const char* outp = argv[4];
    bool linear = argc < 6 || strcmp(argv[5], "linear") == 0;
    FILE* f = fopen(in, "rb");
    if (!f) { perror(in); return 1; }
    std::vector<uint8_t> bin((size_t)w * h * 8);
    if (fread(bin.data(), 1, bin.size(), f) != bin.size()) { fprintf(stderr, "short read\n"); return 1; }
    fclose(f);
    std::vector<uint8_t> png((size_t)w * h * 4);
    const uint16_t* hp = (const uint16_t*)bin.data();
    for (size_t i = 0; i < (size_t)w * h * 4; i++) {
        _Float16 hv; memcpy(&hv, hp + i, 2);
        float v = (float)hv;
        if (linear) {
            v = v <= 0.f ? 0.f : (v <= 0.0031308f ? 12.92f * v
                    : 1.055f * powf(v, 1.f / 2.4f) - 0.055f);
        }
        v = fminf(fmaxf(v, 0.f), 1.f);
        png[i] = (uint8_t)(v * 255.f + 0.5f);
    }
    // opaque alpha, matching the reference PNGs
    for (size_t i = 3; i < png.size(); i += 4) png[i] = 255;
    if (!png_write(outp, png.data(), w, h)) { fprintf(stderr, "png write failed\n"); return 1; }
    return 0;
}
