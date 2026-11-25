// compare - diff two RGBA16F .bin frame dumps.
//
//   compare <a.bin> <b.bin> <w> <h> [diff.png]
//   compare <dirA> <dirB> [diff.png_dir]     - per-frame mode, reads meta.json
//
// Prints per-channel max-abs / mean-abs / PSNR; writes an amplified
// visual diff PNG (|a-b| x 16, clamped).

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <sys/stat.h>
#include "pngio.h"

static float h2f(uint16_t u) {
    union { uint16_t u; _Float16 f; } c; c.u = u; return (float)c.f;
}
static std::vector<uint8_t> read_file(const std::string& p) {
    FILE* f = fopen(p.c_str(), "rb");
    if (!f) { fprintf(stderr, "cannot read %s\n", p.c_str()); exit(1); }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    std::vector<uint8_t> v(n);
    if (fread(v.data(), 1, n, f) != (size_t)n) exit(1);
    fclose(f); return v;
}
static bool is_dir(const std::string& p) { struct stat s; return stat(p.c_str(), &s) == 0 && S_ISDIR(s.st_mode); }

// extract w/h from meta.json next to a bin/dir
static void dims_from_meta(const std::string& dir, int& w, int& h) {
    auto t = read_file(dir + "/meta.json");
    t.push_back(0);
    const char* p;
    if ((p = strstr((const char*)t.data(), "upscale_width"))) w = atoi(strchr(p, ':') + 1);
    if ((p = strstr((const char*)t.data(), "upscale_height"))) h = atoi(strchr(p, ':') + 1);
}

struct Stats { double maxAbs = 0, meanAbs = 0, mse = 0; };

static Stats cmp_bin(const std::string& pa, const std::string& pb,
                     int w, int h, std::vector<uint8_t>* diffOut) {
    auto a = read_file(pa), b = read_file(pb);
    size_t need = (size_t)w * h * 8;
    if (a.size() != need || b.size() != need) {
        fprintf(stderr, "size mismatch: %s=%zu %s=%zu want %zu\n",
                pa.c_str(), a.size(), pb.c_str(), b.size(), need);
        exit(1);
    }
    Stats s[4];
    if (diffOut) diffOut->resize((size_t)w * h * 4);
    for (size_t i = 0; i < (size_t)w * h * 4; i++) {
        float va = h2f(((uint16_t*)a.data())[i]);
        float vb = h2f(((uint16_t*)b.data())[i]);
        float d = fabsf(va - vb);
        int ch = (int)(i & 3);
        if (d > s[ch].maxAbs) s[ch].maxAbs = d;
        s[ch].meanAbs += d;
        s[ch].mse += (double)d * d;
        if (diffOut && (i & 3) < 3)
            (*diffOut)[i] = (uint8_t)fminf(d * 16.f * 255.f, 255.f);
        else if (diffOut)
            (*diffOut)[i] = 255;
    }
    double n = (double)w * h;
    Stats tot;
    for (int c = 0; c < 4; c++) {
        s[c].meanAbs /= n; s[c].mse /= n;
        tot.maxAbs = fmax(tot.maxAbs, s[c].maxAbs);
        tot.meanAbs += s[c].meanAbs / 4; tot.mse += s[c].mse / 4;
    }
    for (int c = 0; c < 4; c++) {
        double psnr = s[c].mse > 0 ? 10.0 * log10(1.0 / s[c].mse) : INFINITY;
        printf("  ch%d  max %.6f  mean %.6f  psnr %.2f dB\n",
               c, s[c].maxAbs, s[c].meanAbs, psnr);
    }
    double psnr = tot.mse > 0 ? 10.0 * log10(1.0 / tot.mse) : INFINITY;
    printf("  all  max %.6f  mean %.6f  psnr %.2f dB\n",
           tot.maxAbs, tot.meanAbs, psnr);
    return tot;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: compare <a.bin> <b.bin> <w> <h> [diff.png]\n"
                        "       compare <dirA> <dirB> [diff_png_dir]\n");
        return 1;
    }
    if (is_dir(argv[1])) {
        std::string da = argv[1], db = argv[2];
        int w = 0, h = 0;
        dims_from_meta(da, w, h);
        int frames = 0;
        { auto t = read_file(da + "/meta.json"); t.push_back(0);
          const char* p = strstr((const char*)t.data(), "\"frames\"");
          if (p) frames = atoi(strchr(p, ':') + 1); }
        Stats agg;
        for (int f = 0; f < frames; f++) {
            char pa[512], pb[512], pd[512];
            snprintf(pa, sizeof pa, "%s/f%02d_out.bin", da.c_str(), f);
            snprintf(pb, sizeof pb, "%s/f%02d_out.bin", db.c_str(), f);
            printf("frame %d:\n", f);
            std::vector<uint8_t> diff;
            Stats s = cmp_bin(pa, pb, w, h, argc > 3 ? &diff : nullptr);
            agg.maxAbs = fmax(agg.maxAbs, s.maxAbs);
            agg.meanAbs += s.meanAbs / frames;
            agg.mse += s.mse / frames;
            if (argc > 3) {
                snprintf(pd, sizeof pd, "%s/f%02d_diff.png", argv[3], f);
                png_write(pd, diff.data(), w, h);
            }
        }
        double psnr = agg.mse > 0 ? 10.0 * log10(1.0 / agg.mse) : INFINITY;
        printf("summary: max %.6f  mean %.6f  psnr %.2f dB\n",
               agg.maxAbs, agg.meanAbs, psnr);
        return 0;
    }
    if (argc < 5) { fprintf(stderr, "need w h for bin mode\n"); return 1; }
    int w = atoi(argv[3]), h = atoi(argv[4]);
    std::vector<uint8_t> diff;
    cmp_bin(argv[1], argv[2], w, h, argc > 5 ? &diff : nullptr);
    if (argc > 5) png_write(argv[5], diff.data(), w, h);
    return 0;
}
