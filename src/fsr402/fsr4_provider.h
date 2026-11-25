// Constant-buffer layouts and CPU-side helpers ported from the DX12 provider
// (ffx_provider_fsr4_dx12.cpp, spd/ffx_spd.h, fsr1/ffx_fsr1.h), shared by the
// standalone CLI (fsr4mac.mm) and the live encoder (fsr4_live.mm). The 4.1.x
// bridges keep their own layouts in fsr411_runtime.h.
#pragma once
#import <Metal/Metal.h>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <utility>
#include <vector>
#include "pass0_weights.h"

namespace fsr4 {

struct OptimizedConstants {          // MLSR_Optimized_Constants
    float inv_size[2], scale[2], inv_scale[2], jitter[2], mv_scale[2];
    float tex_size[2], max_renderSize[2], fMotionVectorJitterCancellation[2];
    uint32_t width, height, reset, width_lr;
    uint32_t height_lr;
    float preExposure, previous_preExposure;
    uint32_t rcas_enabled;
    float rcas_sharpness, pad;
};
struct AutoExposureSPDConstants {
    uint32_t mips, numWorkGroups, workGroupOffset[2];
    float invInputSize[2], preExposure, padding;
};
struct RcasConstants {               // cbRCAS
    uint32_t config[4];
    float preExposure;
    uint32_t pad[3];
};
static_assert(sizeof(OptimizedConstants) == 104
    && sizeof(AutoExposureSPDConstants) == 32
    && sizeof(RcasConstants) == 32,
    "provider constant layouts are fixed by the shaders");

// GetScratchSize per resolution tier (1080/2160/4320 output).
inline std::size_t scratch_size(int resIdx) {
    static const std::size_t kSize[3] = {20880256, 83232256, 332352256};
    return kSize[resIdx];
}

inline uint32_t roundUpDiv(uint32_t a, uint32_t b) { return (a + b - 1) / b; }

// FsrRcasCon (fsr1/ffx_fsr1.h CPU path)
inline void FsrRcasCon(uint32_t con[4], float sharpness) {
    sharpness = exp2f(-sharpness);
    union { uint16_t u; _Float16 f; } cvt;
    cvt.f = (_Float16)sharpness;
    con[0] = __builtin_bit_cast(uint32_t, sharpness);
    con[1] = (uint32_t)cvt.u | ((uint32_t)cvt.u << 16); // ffxPackHalf2x16
    con[2] = 0;
    con[3] = 0;
}

// ffxSpdSetup (spd/ffx_spd.h CPU path)
inline void ffxSpdSetup(uint32_t groups[2], uint32_t wgOffset[2],
                        uint32_t numWgMips[2], const uint32_t rect[4]) {
    wgOffset[0] = rect[0] / 64;
    wgOffset[1] = rect[1] / 64;
    const uint32_t endX = (rect[0] + rect[2] - 1) / 64;
    const uint32_t endY = (rect[1] + rect[3] - 1) / 64;
    groups[0] = endX + 1 - wgOffset[0];
    groups[1] = endY + 1 - wgOffset[1];
    numWgMips[0] = groups[0] * groups[1];
    const uint32_t res = rect[2] > rect[3] ? rect[2] : rect[3];
    const float mips = floorf(log2f((float)res));
    numWgMips[1] = (uint32_t)fminf(mips, 12.f);
}

// cbPass_Weights table per preset (pass0_weights.h).
inline const uint32_t *pass0_weights(const char *preset) {
    if (!std::strcmp(preset, "native")) return kPass0Weights_native;
    if (!std::strcmp(preset, "balanced")) return kPass0Weights_balanced;
    if (!std::strcmp(preset, "performance")) return kPass0Weights_performance;
    if (!std::strcmp(preset, "ultraperf")) return kPass0Weights_ultraperf;
    if (!std::strcmp(preset, "drs")) return kPass0Weights_drs;
    return kPass0Weights_quality;
}

// A compiled kernel plus its bindings.tsv resource list.
struct KernelInfo {
    id<MTLComputePipelineState> pso;
    MTLSize wg;
    std::vector<std::pair<NSUInteger, std::string>> buffers;   // idx, name
    std::vector<std::pair<NSUInteger, std::string>> textures;
    std::vector<std::pair<NSUInteger, std::string>> samplers;
};

inline id<MTLTexture> make_tex(id<MTLDevice> dev, MTLPixelFormat fmt,
                               NSUInteger w, NSUInteger h) {
    MTLTextureDescriptor *d = [MTLTextureDescriptor
        texture2DDescriptorWithPixelFormat:fmt width:w height:h mipmapped:NO];
    d.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
    d.storageMode = MTLStorageModeShared;
    return [dev newTextureWithDescriptor:d];
}

inline bool load_file(const std::string &path, std::vector<uint8_t> &out) {
    FILE *f = fopen(path.c_str(), "rb");
    if (!f) return false;
    fseek(f, 0, SEEK_END);
    const long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (n <= 0) { fclose(f); return false; }
    out.resize((size_t)n);
    const bool ok = fread(out.data(), 1, out.size(), f) == out.size();
    fclose(f);
    return ok;
}

// ---------------------------------------------------------------------------
// Shared variant/session plumbing (tools/fsr402/compile_variant.py output layout)
// ---------------------------------------------------------------------------

// bindings.tsv parser: K<tab>name<tab>x<tab>y<tab>z declares a kernel (PSO +
// threadgroup), B/T/S rows bind buffer/texture/sampler slots to provider
// resource names. On error returns false with `err` set; the caller reports.
// missingFn/psoFail are the phrases each caller historically prepends to the
// kernel name ("no MTLFunction "/"PSO failed " in the CLI, "missing kernel
// "/"pso " in the live encoder).
inline bool parse_bindings(id<MTLDevice> dev, id<MTLLibrary> lib, FILE *bf,
                           std::map<std::string, KernelInfo> &kernels,
                           std::string &err,
                           const char *missingFn, const char *psoFail) {
    char line[512];
    while (fgets(line, sizeof line, bf)) {
        char kn[128], rn[128];
        int x, y, z, idx;
        if (sscanf(line, "K\t%127s\t%d\t%d\t%d", kn, &x, &y, &z) == 4) {
            if (x <= 0 || y <= 0 || z <= 0) {
                err = "invalid threadgroup dimensions for " + std::string(kn);
                return false;
            }
            id<MTLFunction> fn = [lib newFunctionWithName:
                [NSString stringWithUTF8String:kn]];
            if (!fn) { err = missingFn + std::string(kn); return false; }
            KernelInfo k;
            NSError *e = nil;
            k.pso = [dev newComputePipelineStateWithFunction:fn error:&e];
            if (!k.pso) { err = psoFail + std::string(kn); return false; }
            k.wg = MTLSizeMake(x, y, z);
            if ((NSUInteger)x * y * z > k.pso.maxTotalThreadsPerThreadgroup ||
                k.pso.staticThreadgroupMemoryLength > dev.maxThreadgroupMemoryLength) {
                err = "GPU threadgroup limits exceeded by " + std::string(kn);
                return false;
            }
            kernels[kn] = k;
        } else if (sscanf(line, "B\t%127s\t%d\t%127s", kn, &idx, rn) == 3)
            kernels[kn].buffers.push_back({(NSUInteger)idx, rn});
        else if (sscanf(line, "T\t%127s\t%d\t%127s", kn, &idx, rn) == 3)
            kernels[kn].textures.push_back({(NSUInteger)idx, rn});
        else if (sscanf(line, "S\t%127s\t%d\t%127s", kn, &idx, rn) == 3)
            kernels[kn].samplers.push_back({(NSUInteger)idx, rn});
    }
    return true;
}

// Model-pass encoder level (INT8 dispatch table): passes 1-2 and 12 tile the
// upscale>>1 canvas, 3-5/10-11 the >>2 level, 6-9 the >>3 level.
inline constexpr int kModelLevel[13] = {0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 1, 1, 0};

// fsr4_model_v07_<f16|i8>_pass<N>
inline void model_pass_name(char *buf, std::size_t size, bool fp16, int pass) {
    snprintf(buf, size, "fsr4_model_v07_%s_pass%d", fp16 ? "f16" : "i8", pass);
}

// <builddir>/fsr4_<preset>_<res>_di<d>_lm<l>_jm<j>_cs<c>_ae<a>_db<d><suffix>:
// the directory compile_variant.py fills with shaders.metallib/bindings.tsv.
inline std::string variant_dir(const std::string &builddir, const char *preset,
                               const char *res, int depthInv, int lowRes,
                               int jittered, int colorspace, int autoExposure,
                               int debugView, const char *suffix) {
    char key[512];
    snprintf(key, sizeof key, "fsr4_%s_%s_di%d_lm%d_jm%d_cs%d_ae%d_db%d%s",
             preset, res, depthInv, lowRes, jittered, colorspace,
             autoExposure, debugView, suffix);
    return builddir + "/" + key;
}

// Runs tools/fsr402/compile_variant.py (the script hashes its inputs and skips
// up-to-date variants). postExtOff >= 0 appends the i8mpp post-ext flag.
inline bool compile_variant(const std::string &root, const std::string &builddir,
                            const char *preset, const char *res, int resIdx,
                            int depthInv, int lowRes, int jittered,
                            int colorspace, int autoExposure, int debugView,
                            int rcas, const char *model, long postExtOff) {
    char cmd[4096], extopt[64] = "";
    if (postExtOff >= 0)
        snprintf(extopt, sizeof extopt, " --post-ext-ct2d %ld", postExtOff);
    snprintf(cmd, sizeof cmd,
        "python3 '%s/tools/fsr402/compile_variant.py' --root '%s' --preset %s "
        "--res %s --res-idx %d --depth-inverted %d --low-res-mv %d "
        "--jittered-mv %d --colorspace %d --auto-exposure %d "
        "--debug-view %d --rcas %d --model %s --builddir '%s'%s",
        root.c_str(), root.c_str(), preset, res, resIdx, depthInv, lowRes,
        jittered, colorspace, autoExposure, debugView, rcas, model,
        builddir.c_str(), extopt);
    return system(cmd) == 0;
}

// ---------------------------------------------------------------------------
// Shared resources
// ---------------------------------------------------------------------------

// Linear-clamp sampler bound at every sampler slot in the provider tables.
inline id<MTLSamplerState> linear_clamp(id<MTLDevice> dev) {
    MTLSamplerDescriptor *d = [MTLSamplerDescriptor new];
    d.minFilter = MTLSamplerMinMagFilterLinear;
    d.magFilter = MTLSamplerMinMagFilterLinear;
    d.mipFilter = MTLSamplerMipFilterLinear;
    d.sAddressMode = MTLSamplerAddressModeClampToEdge;
    d.tAddressMode = MTLSamplerAddressModeClampToEdge;
    d.rAddressMode = MTLSamplerAddressModeClampToEdge;
    d.maxAnisotropy = 1;
    return [dev newSamplerStateWithDescriptor:d];
}

// Zero-fill a shared-storage texture (RGBA8 recurrent / RGBA16F history and
// the small R32F exposure/atomic views all take whole texel rows).
inline void zero_tex(id<MTLTexture> t) {
    const std::size_t bpp = t.pixelFormat == MTLPixelFormatRGBA16Float ? 8 : 4;
    std::vector<uint8_t> z(t.width * t.height * bpp);
    [t replaceRegion:MTLRegionMake2D(0, 0, t.width, t.height)
        mipmapLevel:0 withBytes:z.data() bytesPerRow:t.width * bpp];
}

// The SPD global atomic: a buffer-backed 1x1 R32Uint the shader takes both
// as a texture and as a raw buffer.
inline id<MTLTexture> spd_atomic_texture(id<MTLBuffer> buf) {
    MTLTextureDescriptor *d = [MTLTextureDescriptor
        texture2DDescriptorWithPixelFormat:MTLPixelFormatR32Uint
        width:1 height:1 mipmapped:NO];
    d.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
    d.storageMode = MTLStorageModeShared;
    return [buf newTextureWithDescriptor:d offset:0 bytesPerRow:16];
}

// ---------------------------------------------------------------------------
// Shared UpdateConstants pieces
// ---------------------------------------------------------------------------

// The resolution-derived fields every entry point fills the same way;
// max_renderSize is the largest input rect the session may see.
inline void fill_geometry(OptimizedConstants &oc, uint32_t uw, uint32_t uh,
                          uint32_t rw, uint32_t rh,
                          uint32_t maxRenderW, uint32_t maxRenderH) {
    oc.inv_size[0] = 1.f / uw; oc.inv_size[1] = 1.f / uh;
    oc.scale[0] = 1.f * uw / rw; oc.scale[1] = 1.f * uh / rh;
    oc.inv_scale[0] = 1.f / oc.scale[0]; oc.inv_scale[1] = 1.f / oc.scale[1];
    oc.tex_size[0] = (float)uw; oc.tex_size[1] = (float)uh;
    oc.max_renderSize[0] = (float)maxRenderW;
    oc.max_renderSize[1] = (float)maxRenderH;
    oc.width = uw; oc.height = uh;
    oc.width_lr = rw; oc.height_lr = rh;
}

// SPD dispatch groups plus the AutoExposureSPDConstants the provider
// computes from the render rect.
inline void fill_spd(AutoExposureSPDConstants &sc, uint32_t rw, uint32_t rh,
                     float preExposure, uint32_t groups[2]) {
    uint32_t wgo[2], nwm[2];
    const uint32_t rect[4] = {0, 0, rw, rh};
    ffxSpdSetup(groups, wgo, nwm, rect);
    sc.workGroupOffset[0] = wgo[0]; sc.workGroupOffset[1] = wgo[1];
    sc.numWorkGroups = nwm[0]; sc.mips = nwm[1];
    sc.invInputSize[0] = 1.f / rw; sc.invInputSize[1] = 1.f / rh;
    sc.preExposure = preExposure;
}

} // namespace fsr4
