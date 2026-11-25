// gen_testcase - deterministic synthetic FSR4 testcase generator.
//
//   gen_testcase <out_dir> [preset=quality rw rh uw uh frames]
//   gen_testcase <out_dir> quality 1280 720 1920 1080 8
//
// Scene: infinite procedural plane (checkerboard + smooth gradient +
// high-frequency stripes), a camera translating sideways at a fixed
// pixel rate, and one square object counter-translating. All values
// computed in plain float math - bit-identical on any platform.
//
// Motion-vector convention (documented in TESTDATA.md):
//   mv.bin stores, per render pixel,  mv = prev_px - cur_px
//   i.e. the position of this pixel's surface in the *previous* frame
//   minus its position in the current frame, in render-pixel units.
//   motionVectorScale (meta mv_scale) is [1,1]: raw mv values are pixels.
//
// Depth: NDC-style in [0,1]. A vertical planar gradient (far at top,
// near at bottom) with the moving square sitting closer.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <sys/stat.h>

static uint16_t f2h(float v) {
    union { uint16_t u; _Float16 f; } c;
    c.f = (_Float16)v;
    return c.u;
}
static void write_file(const std::string& p, const void* d, size_t n) {
    FILE* f = fopen(p.c_str(), "wb");
    if (!f) { fprintf(stderr, "cannot write %s\n", p.c_str()); exit(1); }
    if (fwrite(d, 1, n, f) != n) { fprintf(stderr, "write failed %s\n", p.c_str()); exit(1); }
    fclose(f);
}

// Halton base-b radical inverse (plain float, deterministic)
static float halton(int idx, int base) {
    float f = 1.f, r = 0.f;
    for (int i = idx; i > 0; i /= base) { f /= (float)base; r += f * (float)(i % base); }
    return r;
}

// procedural plane colour in world units
static void plane(float wx, float wy, float& r, float& g, float& b) {
    // checkerboard (8 world units per cell)
    int cx = (int)floorf(wx / 8.f), cy = (int)floorf(wy / 8.f);
    float checker = ((cx + cy) & 1) ? 0.75f : 0.25f;
    // smooth gradient
    float gx = 0.5f + 0.5f * sinf(wx * 0.011f);
    float gy = 0.5f + 0.5f * sinf(wy * 0.013f);
    // high-frequency stripes (~1.5 world units period)
    float stripe = 0.5f + 0.5f * sinf(wx * 4.3f) * sinf(wy * 3.7f);
    r = 0.6f * checker + 0.25f * gx + 0.35f * stripe;
    g = 0.5f * checker + 0.35f * gy + 0.25f * stripe;
    b = 0.45f + 0.3f * gx * gy + 0.25f * checker * stripe;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: gen_testcase <out_dir> [preset rw rh uw uh frames]\n");
        return 1;
    }
    std::string dir = argv[1];
    const char* preset = argc > 2 ? argv[2] : "quality";
    int rw = argc > 3 ? atoi(argv[3]) : 1280;
    int rh = argc > 4 ? atoi(argv[4]) : 720;
    int uw = argc > 5 ? atoi(argv[5]) : 1920;
    int uh = argc > 6 ? atoi(argv[6]) : 1080;
    int frames = argc > 7 ? atoi(argv[7]) : 8;
    {   // mkdir -p
        std::string acc;
        for (char c : dir + "/") {
            acc += c;
            if (c == '/' && acc.size() > 1) { acc.pop_back(); mkdir(acc.c_str(), 0755); acc += '/'; }
        }
    }

    const float camDx = 1.75f;   // render px / frame, camera moves +x
    const float camDy = 0.50f;
    const float objDx = -2.25f;  // object counter-moves
    const float objDy = 1.25f;
    const float wscale = 0.25f;  // world units per render pixel
    const int objHalf = 60;      // half-size of object square in render px
    const float objCx = 640.f, objCy = 360.f; // object centre at frame 0 (approx; drifts)

    std::vector<uint16_t> color((size_t)rw * rh * 4);
    std::vector<uint16_t> mv((size_t)rw * rh * 2);
    std::vector<float>    depth((size_t)rw * rh);
    float exposure = 1.0f;

    std::string meta = "{\n";
    meta += "  \"preset\": \""; meta += preset; meta += "\",\n";
    char lb[512];
    snprintf(lb, sizeof lb,
        "  \"render_width\": %d, \"render_height\": %d,\n"
        "  \"upscale_width\": %d, \"upscale_height\": %d,\n"
        "  \"frames\": %d,\n"
        "  \"depth_inverted\": false, \"low_res_mv\": true, \"jittered_mv\": false,\n"
        "  \"colorspace\": \"linear\", \"auto_exposure\": false, \"pre_exposure\": 1.0,\n"
        "  \"rcas\": false, \"sharpness\": 0.0,\n  \"frame\": [\n",
        rw, rh, uw, uh, frames);
    meta += lb;

    for (int f = 0; f < frames; f++) {
        // Halton(2,3) jitter, 8 phases, sampled as a subpixel offset of the
        // procedural texture position (like a real jittered render).
        float jx = halton((f % 8) + 1, 2) - 0.5f;
        float jy = halton((f % 8) + 1, 3) - 0.5f;

        for (int y = 0; y < rh; y++) {
            for (int x = 0; x < rw; x++) {
                size_t pi = (size_t)y * rw + x;
                // current-frame world position of this pixel's surface
                float wx = (x + 0.5f + jx) * wscale + camDx * f * wscale;
                float wy = (y + 0.5f + jy) * wscale + camDy * f * wscale;
                // object square in *object space*: object centre translates
                // with camera motion + own counter-motion
                float ox = x + camDx * f - (objCx + (camDx + objDx) * f);
                float oy = y + camDy * f - (objCy + (camDy + objDy) * f);
                bool inObj = fabsf(ox) < objHalf && fabsf(oy) < objHalf;
                float r, g, b;
                if (inObj) {
                    // object texture: solid red-ish with fine vertical stripes
                    float st = 0.5f + 0.5f * sinf((x + 0.5f) * 1.9f);
                    r = 0.85f; g = 0.15f + 0.3f * st; b = 0.12f;
                } else {
                    plane(wx, wy, r, g, b);
                }
                color[pi * 4 + 0] = f2h(r);
                color[pi * 4 + 1] = f2h(g);
                color[pi * 4 + 2] = f2h(b);
                color[pi * 4 + 3] = f2h(1.0f);
                // depth: 0.15 (bottom) .. 0.85 (top), object closer at 0.12
                float d = 0.85f - 0.7f * ((float)y / (float)(rh - 1));
                depth[pi] = inObj ? 0.12f : d;
                // mv = prev_px - cur_px (render px). Background point was at
                // (x + camDx, y + camDy) in the previous frame; object also
                // adds its own delta.
                float mvx = camDx, mvy = camDy;
                if (inObj) { mvx = camDx + objDx; mvy = camDy + objDy; }
                mv[pi * 2 + 0] = f2h(mvx);
                mv[pi * 2 + 1] = f2h(mvy);
            }
        }

        char p[512];
        snprintf(p, sizeof p, "%s/f%02d_color.bin", dir.c_str(), f);
        write_file(p, color.data(), color.size() * 2);
        snprintf(p, sizeof p, "%s/f%02d_depth.bin", dir.c_str(), f);
        write_file(p, depth.data(), depth.size() * 4);
        snprintf(p, sizeof p, "%s/f%02d_mv.bin", dir.c_str(), f);
        write_file(p, mv.data(), mv.size() * 2);
        snprintf(p, sizeof p, "%s/f%02d_exposure.bin", dir.c_str(), f);
        write_file(p, &exposure, 4);

        snprintf(lb, sizeof lb,
            "    {\"jitter\": [%.9g, %.9g], \"mv_scale\": [1.0, 1.0], "
            "\"reset\": %s, \"frame_time_ms\": 16.6, \"camera_near\": 0.1, "
            "\"camera_far\": 1000.0, \"fovy\": 1.0}%s\n",
            jx, jy, f == 0 ? "true" : "false", f + 1 == frames ? "" : ",");
        meta += lb;
    }
    meta += "  ]\n}\n";
    write_file(dir + "/meta.json", meta.data(), meta.size());
    printf("wrote %s: %s %dx%d -> %dx%d, %d frames\n",
           dir.c_str(), preset, rw, rh, uw, uh, frames);
    return 0;
}
