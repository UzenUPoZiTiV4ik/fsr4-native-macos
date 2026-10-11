// Live FSR4 encoder for the CrossOver / D3DMetal hook.
//
// Faithful to the DX12 provider for the constants (jitter and motion scale go
// straight from the DLSS/NGX parameters — D3DMetal copies them into the MetalFX
// scaler unchanged, and OptiScaler feeds them to FSR the same way), plus the
// things a shared game command stream needs that the CLI does not:
//
//  * per-scaler history. D3DMetal encodes more than one temporal scaler per
//    game frame (Spider-Man: one with the real RG16 motion field and one with a
//    4x4 RGBA8 placeholder). One shared history buffer between them blends the
//    current frame into the history with zero motion every frame — that is a
//    motion trail, and it grows with the upscale ratio because the network
//    trusts history more at low input resolution.
//  * a policy for encodes whose motion texture is a placeholder (FSR4_STUB).
//  * Metal 4 queue-scope barriers around our encoder: Metal 4 does not track
//    hazards between passes, so history/scratch of consecutive encodes would
//    otherwise be free to overlap on the GPU.
//  * SPD auto-exposure when the game asked DLSS for it (descriptor flag).
//  * diagnostics: per-encode trace and a GPU probe (exposure value, |mv| max,
//    centre depth) so a hook.log tells what the game actually handed over.

#include "fsr4_live.h"
#include "pass0_weights.h"
#include "i8mpp_host.h"

#include <Metal/Metal.h>
#include <mach/mach_time.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdarg>
#include <cmath>
#include <map>
#include <mutex>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

struct OptCB {
    float inv_size[2], scale[2], inv_scale[2], jitter[2], mv_scale[2];
    float tex_size[2], max_renderSize[2], fMotionVectorJitterCancellation[2];
    uint32_t width, height, reset, width_lr;
    uint32_t height_lr;
    float preExposure, previous_preExposure;
    uint32_t rcas_enabled;
    float rcas_sharpness, pad;
};

struct SpdCB {                 // AutoExposureSPDConstants, 32 bytes
    uint32_t mips, numWorkGroups, workGroupOffset[2];
    float invInputSize[2], preExposure, padding;
};

struct RcasCB {                // cbRCAS, 32 bytes
    uint32_t config[4];
    float preExposure;
    uint32_t pad[3];
};

static const uint32_t kCbSlot = 256;      // OptCB at +0, SpdCB at +128, probe at +192, RcasCB at +224
static const uint32_t kSpdOff = 128;
static const uint32_t kRcasOff = 224;
static const uint32_t kCbRing = 32;
static const uint32_t kProbeSlots = 8;
static const uint32_t kProbeBytes = 16;

struct Kernel {
    id<MTLComputePipelineState> pso;
    MTLSize wg;
    std::vector<std::pair<NSUInteger, std::string>> buffers;
    std::vector<std::pair<NSUInteger, std::string>> textures;
    std::vector<std::pair<NSUInteger, std::string>> samplers;
};

// Compiled shader variant + its weights. Cached per key so two scalers with
// different flags do not thrash.
struct Session {
    id<MTLDevice> dev;
    std::string key;
    std::string preset;
    int resIdx = 0;
    bool jittered = false;
    bool lowResMv = true;
    bool autoExposure = false;
    id<MTLBuffer> weights, initBuf;
    NSMutableDictionary *argTables;
    std::map<std::string, Kernel> kernels;
    uint64_t lastUse = 0;
    // int8 model on the Metal 4 tensor ops (M5 neural accelerators); the
    // variant's pre/post are the int8 ones, post loads its ConvNextBlock from
    // the pass-13 tensor (FSR4_POST_EXT_CNB).
    bool i8 = false;
    I8Mpp i8mpp;
};

// Temporal state of one MetalFX scaler object.
struct View {
    int index = 0;
    void *owner = nullptr;
    uint32_t uw = 0, uh = 0;
    id<MTLTexture> recurrent, history, reproj;
    bool zeroTargets = false;
    Fsr4InputHistory inputHistory = {};
    bool first = true;
    std::string sessionKey;      // shader variant the last encode ran; a change re-creates the model
    uint64_t encodes = 0;
    uint64_t lastUse = 0;
    id<MTLTexture> lastOut;      // last texture this view produced
};

struct Shared {
    id<MTLDevice> dev;
    fsr4::DeviceCaps caps;
    bool nativeFailed = false;
    bool backendLogged = false;
    bool invalidModelLogged = false;
    fsr4::Backend lastBackend = fsr4::Backend::Auto;
    id<MTLBuffer> scratch;
    int scratchTier = -1;
    bool zeroScratch = false;
    id<MTLBuffer> cbRing[kCbRing] = {};
    uint32_t cbSlot = 0;
    id<MTLSamplerState> samp;
    id<MTLTexture> exposureOne;          // 1x1 R32F = 1.0
    id<MTLTexture> depthCopy;
    uint32_t drw = 0, drh = 0;
    id<MTLTexture> rcasTemp;             // post -> RCAS intermediate (RGBA16F at upscale)
    uint32_t rcasW = 0, rcasH = 0;
    // SPD auto-exposure
    id<MTLTexture> autoExp, spdAtomicTex, lumaMip5;
    id<MTLBuffer> spdAtomicBuf;
    uint32_t lumaW = 0, lumaH = 0;
    // helpers
    id<MTLComputePipelineState> depthPso, clearBufPso, clearTexPso, copyTexPso, probePso;
    id<MTLBuffer> probeBuf;
    NSMutableDictionary *argTables;
    id<MTLResidencySet> residency;
    // views
    std::map<void *, View *> views;
    int nextViewIndex = 0;
    id<MTLTexture> lastRealOut;          // output of the most recent real-motion encode
    uint64_t encodes = 0;
    double lastRealT = 0;                // now_ms() of the last real encode (frame-time EMA)
    float frameMs = 0;
};

static std::mutex g_mu;
static Shared *g_sh = nullptr;
static std::map<std::string, Session *> g_sessions;
static int g_fail_logged = 0;
static uint64_t g_t0 = 0;

static double now_ms() {
    static mach_timebase_info_data_t tb;
    if (!tb.denom) mach_timebase_info(&tb);
    uint64_t t = mach_absolute_time();
    if (!g_t0) g_t0 = t;
    return (double)(t - g_t0) * tb.numer / tb.denom / 1e6;
}

static void flog(const char *msg) {
    fprintf(stderr, "[fsr4] %s\n", msg);
    const char *path = getenv("FSR4_HOOK_LOG");
    if (!path || !path[0]) return;
    FILE *f = fopen(path, "a");
    if (!f) return;
    fprintf(f, "[fsr4] %s\n", msg);
    fclose(f);
}

static void flogf(const char *fmt, ...) {
    char buf[1024];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof buf, fmt, ap);
    va_end(ap);
    flog(buf);
}

static uint32_t divup(uint32_t a, uint32_t b) { return (a + b - 1) / b; }

static const char *root_dir() {
    const char *e = getenv("FSR4_ROOT");
    if (e && e[0]) return e;
    return nullptr;
}

static int env_int(const char *name, int dflt) {
    const char *e = getenv(name);
    if (!e || !e[0]) return dflt;
    return atoi(e);
}

static id<MTLTexture> make_tex(id<MTLDevice> dev, MTLPixelFormat fmt, uint32_t w, uint32_t h) {
    MTLTextureDescriptor *d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:fmt
                                                                                 width:w height:h mipmapped:NO];
    d.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
    d.storageMode = MTLStorageModeShared;
    return [dev newTextureWithDescriptor:d];
}

static void zero_tex(id<MTLTexture> t) {
    size_t bpp = 4;
    if (t.pixelFormat == MTLPixelFormatRGBA16Float) bpp = 8;
    std::vector<uint8_t> z((size_t)t.width * t.height * bpp);
    [t replaceRegion:MTLRegionMake2D(0, 0, t.width, t.height)
         mipmapLevel:0 withBytes:z.data() bytesPerRow:t.width * bpp];
}

static bool load_file(const std::string &path, std::vector<uint8_t> &out) {
    FILE *f = fopen(path.c_str(), "rb");
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

// Same thresholds as ffx_provider_fsr4_dx12.cpp (ratio on the X axis).
static const char *preset_for(uint32_t rw, uint32_t uw) {
    const char *e = getenv("FSR4_PRESET");
    if (e && e[0]) return e;
    float s = uw / (float)rw;
    const char *p = "native";
    if (s >= 1.50f) p = "quality";
    if (s >= 1.69f) p = "balanced";
    if (s >= 1.99f) p = "performance";
    if (s >= 2.99f) p = "ultraperf";
    return p;
}

static const uint32_t *weights_for(const char *preset) {
    if (!strcmp(preset, "native")) return kPass0Weights_native;
    if (!strcmp(preset, "balanced")) return kPass0Weights_balanced;
    if (!strcmp(preset, "performance")) return kPass0Weights_performance;
    if (!strcmp(preset, "ultraperf")) return kPass0Weights_ultraperf;
    if (!strcmp(preset, "drs")) return kPass0Weights_drs;
    return kPass0Weights_quality;
}

// FsrRcasCon (fsr1/ffx_fsr1.h CPU path); the provider passes -2*sharpness + 2
static void rcas_con(uint32_t con[4], float sharpness) {
    sharpness = exp2f(-sharpness);
    union { uint16_t u; _Float16 f; } cvt; cvt.f = (_Float16)sharpness;
    con[0] = __builtin_bit_cast(uint32_t, sharpness);
    con[1] = (uint32_t)cvt.u | ((uint32_t)cvt.u << 16);
    con[2] = 0; con[3] = 0;
}

// FSR sharpness 0..1 (the DLSS/MetalFX path has no sharpness knob, so the
// FSR default is a setting). Defaults: FSR4_SHARPNESS=<0..1> (0.4),
// FSR4_RCAS=0 off. dist/crossover/panel.conf, written by the panel, overrides
// them unless FSR4_SETTINGS=none.
struct Settings {
    bool loaded = false;
    float sharpness = 0.4f;
    bool rcas = true;
    int profile = 1; // 0 Exact, 1 Lossy, 2 Exact-fast, 3 FSR 4.0.2 model
    std::string path;
    time_t mtime = 0;
    long mtimeNs = 0;
    double lastCheck = 0;
};
static Settings g_set;
static std::mutex g_set_mu;
static void settings_read_file_locked();
static Fsr4LiveStatus g_status;
static std::mutex g_status_mu;

// Profile names in panel.conf / FSR4_PROFILE; digits are the older form.
static int profile_from_name(const char *name) {
    if (!strcmp(name, "exact") || !strcmp(name, "0")) return 0;
    if (!strcmp(name, "lossy") || !strcmp(name, "1")) return 1;
    if (!strcmp(name, "exact-fast") || !strcmp(name, "2")) return 2;
    if (!strcmp(name, "fsr4") || !strcmp(name, "3")) return 3;
    return -1;
}
static const char *profile_label(int profile) {
    return profile == 1 ? "lossy" : profile == 2 ? "exact-fast" : profile == 3 ? "fsr4" : "exact";
}

static void settings_load_locked() {
    if (g_set.loaded) return;
    g_set.loaded = true;
    if (const char *e = getenv("FSR4_SHARPNESS")) if (e[0]) g_set.sharpness = (float)atof(e);
    if (const char *e = getenv("FSR4_RCAS")) if (e[0] == '0') g_set.rcas = false;
    if (!(g_set.sharpness > 0.f)) { g_set.sharpness = 0.f; g_set.rcas = false; }
    if (g_set.sharpness > 1.f) g_set.sharpness = 1.f;
    if (const char *e = getenv("FSR4_PROFILE")) {
        const int profile = profile_from_name(e);
        if (profile >= 0) g_set.profile = profile;
        else if (*e) flog("FSR4_PROFILE must be exact, exact-fast, lossy or fsr4; using lossy");
    }
    const char *sp = getenv("FSR4_SETTINGS");
    if (sp && !strcmp(sp, "none")) return;
    if (sp && sp[0]) g_set.path = sp;
    else if (const char *root = root_dir()) g_set.path = std::string(root) + "/dist/crossover/panel.conf";
    settings_read_file_locked();
}

// (re)read panel.conf; the panel app rewrites it when the slider moves
static void settings_read_file_locked() {
    if (g_set.path.empty()) return;
    struct stat st;
    if (stat(g_set.path.c_str(), &st) != 0) return;
    if (st.st_mtime == g_set.mtime && st.st_mtimespec.tv_nsec == g_set.mtimeNs) return;
    FILE *f = fopen(g_set.path.c_str(), "r");
    if (!f) return;
    char line[128];
    while (fgets(line, sizeof line, f)) {
        float v;
        int iv;
        char profile[16];
        if (sscanf(line, "sharpness=%f", &v) == 1 && std::isfinite(v)) g_set.sharpness = v < 0.f ? 0.f : v > 1.f ? 1.f : v;
        else if (sscanf(line, "rcas=%d", &iv) == 1) g_set.rcas = iv != 0;
        else if (sscanf(line, "profile=%15s", profile) == 1) {
            const int value = profile_from_name(profile);
            if (value >= 0) g_set.profile = value;
        }
    }
    const bool ok = !ferror(f);
    fclose(f);
    if (ok) { g_set.mtime = st.st_mtime; g_set.mtimeNs = st.st_mtimespec.tv_nsec; }
}

static void settings_refresh_locked() {
    settings_load_locked();
    const double t = now_ms();
    if (t - g_set.lastCheck > 500.0) { g_set.lastCheck = t; settings_read_file_locked(); }
}

static void settings_write_file_locked() {
    if (g_set.path.empty()) return;
    const std::string temporary = g_set.path + ".tmp." + std::to_string(getpid());
    FILE *f = fopen(temporary.c_str(), "w");
    if (!f) { flog("cannot write panel settings"); return; }
    bool ok = fprintf(f, "sharpness=%.3f\nrcas=%d\nprofile=%s\n", g_set.sharpness,
                      g_set.rcas ? 1 : 0, profile_label(g_set.profile)) > 0;
    if (fclose(f) != 0) ok = false;
    if (!ok || rename(temporary.c_str(), g_set.path.c_str()) != 0) {
        unlink(temporary.c_str()); flog("cannot save panel settings"); return;
    }
    struct stat st;
    if (stat(g_set.path.c_str(), &st) == 0) {
        g_set.mtime = st.st_mtime; g_set.mtimeNs = st.st_mtimespec.tv_nsec;
    }
}

static float sharpness_setting() {
    std::lock_guard<std::mutex> lk(g_set_mu);
    settings_refresh_locked();
    return g_set.rcas ? g_set.sharpness : 0.f;
}

extern "C" void fsr4_live_get_sharpness(float *sharpness, int *rcas) {
    std::lock_guard<std::mutex> lk(g_set_mu);
    settings_refresh_locked();
    if (sharpness) *sharpness = g_set.sharpness;
    if (rcas) *rcas = g_set.rcas ? 1 : 0;
}

extern "C" void fsr4_live_set_sharpness(float sharpness, int rcas) {
    std::lock_guard<std::mutex> lk(g_set_mu);
    settings_load_locked();
    settings_read_file_locked();
    if (!std::isfinite(sharpness)) return;
    g_set.sharpness = sharpness < 0.f ? 0.f : sharpness > 1.f ? 1.f : sharpness;
    g_set.rcas = rcas != 0;
    settings_write_file_locked();
}

extern "C" void fsr4_live_get_profile(int *profile) {
    std::lock_guard<std::mutex> lk(g_set_mu);
    settings_refresh_locked();
    if (profile) *profile = g_set.profile;
}

// FSR4_VERSION=4.0.2 forces the old model. Otherwise (4.1.1 or unset) the
// panel's "fsr4" profile selects it, and every other profile selects 4.1.1.
extern "C" const char *fsr4_live_model_version(void) {
    const char *version = getenv("FSR4_VERSION");
    if (version && *version && strcmp(version, "4.1.1")) return version;
    int profile = 1;
    fsr4_live_get_profile(&profile);
    return profile == 3 ? "4.0.2" : "4.1.1";
}

extern "C" void fsr4_live_set_profile(int profile) {
    if (profile < 0 || profile > 3) return;
    std::lock_guard<std::mutex> lk(g_set_mu);
    settings_load_locked();
    settings_read_file_locked();
    g_set.profile = profile;
    settings_write_file_locked();
}

extern "C" void fsr4_live_get_status(Fsr4LiveStatus *out) {
    if (!out) return;
    std::lock_guard<std::mutex> lk(g_status_mu);
    *out = g_status;
}

extern "C" void fsr4_live_set_status(const Fsr4LiveStatus *status) {
    if (!status) return;
    {
        std::lock_guard<std::mutex> lk(g_status_mu);
        g_status = *status;
        g_status.reason[sizeof(g_status.reason) - 1] = 0;
        g_status.preset[sizeof(g_status.preset) - 1] = 0;
        g_status.model[sizeof(g_status.model) - 1] = 0;
        g_status.variant[sizeof(g_status.variant) - 1] = 0;
    }
    fsr4_panel_publish();
}

extern "C" void fsr4_live_set_inactive(const char *reason) {
    {
        std::lock_guard<std::mutex> lk(g_status_mu);
        g_status.active = 0;
        snprintf(g_status.reason, sizeof g_status.reason, "%s", reason ? reason : "");
        g_status.encodes++;
    }
    fsr4_panel_publish();
}

// encoder exit with a reason for the panel
static int fail_reason(const char *reason) {
    fsr4_live_set_inactive(reason);
    return 0;
}

// ffxSpdSetup (spd/ffx_spd.h CPU path)
static void spd_setup(uint32_t groups[2], uint32_t wgOffset[2],
                      uint32_t numWgMips[2], const uint32_t rect[4]) {
    wgOffset[0] = rect[0] / 64;
    wgOffset[1] = rect[1] / 64;
    uint32_t endX = (rect[0] + rect[2] - 1) / 64;
    uint32_t endY = (rect[1] + rect[3] - 1) / 64;
    groups[0] = endX + 1 - wgOffset[0];
    groups[1] = endY + 1 - wgOffset[1];
    numWgMips[0] = groups[0] * groups[1];
    uint32_t res = rect[2] > rect[3] ? rect[2] : rect[3];
    float m = floorf(log2f((float)res));
    numWgMips[1] = (uint32_t)fminf(m, 12.f);
}

static const char *kHelperSrc =
    "#include <metal_stdlib>\nusing namespace metal;\n"
    "kernel void depth_to_r32(depth2d<float, access::read> src [[texture(0)]],"
    "texture2d<float, access::write> dst [[texture(1)]], uint2 gid [[thread_position_in_grid]]) {"
    "if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) return;"
    "uint2 p = min(gid, uint2(src.get_width()-1, src.get_height()-1));"
    "dst.write(float4(src.read(p)), gid); }\n"
    "kernel void fsr4_clear_buffer(device uint4 *b [[buffer(0)]], constant uint &n4 [[buffer(1)]],"
    "uint gid [[thread_position_in_grid]]) { if (gid < n4) b[gid] = uint4(0); }\n"
    "kernel void fsr4_clear_tex(texture2d<half, access::write> history [[texture(0)]],"
    "texture2d<half, access::write> reproj [[texture(1)]],"
    "texture2d<float, access::write> recurrent [[texture(2)]],"
    "uint2 gid [[thread_position_in_grid]]) {"
    "if (gid.x >= history.get_width() || gid.y >= history.get_height()) return;"
    "history.write(half4(0), gid); reproj.write(half4(0), gid); recurrent.write(float4(0), gid); }\n"
    "kernel void fsr4_copy_tex(texture2d<float, access::read> src [[texture(0)]],"
    "texture2d<float, access::write> dst [[texture(1)]], uint2 gid [[thread_position_in_grid]]) {"
    "if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) return;"
    "uint2 p = min(gid, uint2(src.get_width()-1, src.get_height()-1));"
    "dst.write(src.read(p), gid); }\n"
    // slot: [0] exposure bits, [1] max|mv|*256, [2] centre depth bits, [3] nonzero samples
    "kernel void fsr4_probe(texture2d<float, access::read> expo [[texture(0)]],"
    "texture2d<float, access::read> mv [[texture(1)]],"
    "texture2d<float, access::read> dep [[texture(2)]],"
    "device atomic_uint *out [[buffer(0)]], constant uint2 &content [[buffer(1)]],"
    "uint tid [[thread_position_in_grid]]) {"
    "if (tid >= 256) return;"
    "uint gx = tid % 16, gy = tid / 16;"
    "uint2 p = uint2((gx*2+1) * content.x / 32, (gy*2+1) * content.y / 32);"
    "p = min(p, uint2(mv.get_width()-1, mv.get_height()-1));"
    "float2 v = mv.read(p).xy;"
    "float a = max(abs(v.x), abs(v.y));"
    "if (!isfinite(a)) a = 1e6f;"
    "atomic_fetch_max_explicit(out+1, uint(min(a, 1e6f) * 256.0f), memory_order_relaxed);"
    "if (a > 1e-3f) atomic_fetch_add_explicit(out+3, 1u, memory_order_relaxed);"
    "if (tid == 0) {"
    "atomic_store_explicit(out+0, as_type<uint>(expo.read(uint2(0)).x), memory_order_relaxed);"
    "uint2 c = min(uint2(content.x/2, content.y/2), uint2(dep.get_width()-1, dep.get_height()-1));"
    "atomic_store_explicit(out+2, as_type<uint>(dep.read(c).x), memory_order_relaxed); } }\n";

static Shared *ensure_shared(id<MTLDevice> dev) {
    if (g_sh && g_sh->dev == dev) return g_sh;
    Shared *sh = new Shared();
    sh->dev = dev;
    sh->caps = fsr4::device_caps(dev);
    MTLSamplerDescriptor *sd = [MTLSamplerDescriptor new];
    sd.minFilter = MTLSamplerMinMagFilterLinear;
    sd.magFilter = MTLSamplerMinMagFilterLinear;
    sd.sAddressMode = MTLSamplerAddressModeClampToEdge;
    sd.tAddressMode = MTLSamplerAddressModeClampToEdge;
    sh->samp = [dev newSamplerStateWithDescriptor:sd];
    sh->argTables = [NSMutableDictionary new];
    for (uint32_t i = 0; i < kCbRing; i++)
        sh->cbRing[i] = [dev newBufferWithLength:kCbSlot options:MTLResourceStorageModeShared];
    sh->exposureOne = make_tex(dev, MTLPixelFormatR32Float, 1, 1);
    float one = 1.f;
    [sh->exposureOne replaceRegion:MTLRegionMake2D(0, 0, 1, 1) mipmapLevel:0 withBytes:&one bytesPerRow:4];
    sh->probeBuf = [dev newBufferWithLength:kProbeSlots * kProbeBytes options:MTLResourceStorageModeShared];
    memset(sh->probeBuf.contents, 0, kProbeSlots * kProbeBytes);
    NSError *err = nil;
    id<MTLLibrary> lib = [dev newLibraryWithSource:[NSString stringWithUTF8String:kHelperSrc] options:nil error:&err];
    if (lib) {
        sh->depthPso = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"depth_to_r32"] error:&err];
        sh->clearBufPso = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"fsr4_clear_buffer"] error:&err];
        sh->clearTexPso = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"fsr4_clear_tex"] error:&err];
        sh->copyTexPso = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"fsr4_copy_tex"] error:&err];
        sh->probePso = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"fsr4_probe"] error:&err];
    } else {
        flogf("helper library: %s", err.localizedDescription.UTF8String);
    }
    if (!sh->depthPso || !sh->clearBufPso || !sh->clearTexPso || !sh->copyTexPso || !sh->probePso)
        flogf("helper kernels incomplete: %s", err ? err.localizedDescription.UTF8String : "?");
    // SPD atomic counter: buffer-backed 1x1 R32Uint (the shader takes it both ways)
    sh->spdAtomicBuf = [dev newBufferWithLength:4096 options:MTLResourceStorageModeShared];
    memset(sh->spdAtomicBuf.contents, 0, 4096);
    MTLTextureDescriptor *ad = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR32Uint
                                                                                  width:1 height:1 mipmapped:NO];
    ad.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
    ad.storageMode = MTLStorageModeShared;
    sh->spdAtomicTex = [sh->spdAtomicBuf newTextureWithDescriptor:ad offset:0 bytesPerRow:16];
    sh->autoExp = make_tex(dev, MTLPixelFormatR32Float, 2, 1);
    zero_tex(sh->autoExp);
    delete g_sh;
    for (auto &kv : g_sessions) delete kv.second;
    g_sessions.clear();
    g_sh = sh;
    return sh;
}

static bool ensure_scratch(Shared &sh, int resIdx) {
    static const size_t kScratch[3] = {20880256, 83232256, 332352256};
    if (sh.scratch && sh.scratchTier >= resIdx) return true;
    // fp16 path: 2x the int8 scratch; i8mpp: tensors + the pass-13 parameter
    // tensor (I8Mpp::scratch_bytes, always below 2x). Allocate the larger.
    size_t bytes = kScratch[resIdx] * 2;
    if (I8Mpp::scratch_bytes(resIdx) > bytes) bytes = I8Mpp::scratch_bytes(resIdx);
    sh.scratch = [sh.dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    if (!sh.scratch) return false;
    sh.scratchTier = resIdx;
    sh.zeroScratch = true;
    return true;
}

static Session *load_session(id<MTLDevice> dev, const char *preset, int resIdx,
                               const char *resStr, int depthInv, int lowRes, int jittered,
                               int cs, int ae, bool i8) {
    const char *root = root_dir();
    if (!root) {
        if (!g_fail_logged++) flog("FSR4_ROOT is not set");
        return nullptr;
    }
    char key[256];
    snprintf(key, sizeof key, "%s_%s_di%d_lm%d_jm%d_cs%d_ae%d_%s", preset, resStr, depthInv, lowRes, jittered, cs, ae,
             i8 ? "i8mpp" : "f16");
    auto it = g_sessions.find(key);
    if (it != g_sessions.end()) { it->second->lastUse = ++g_sh->encodes; return it->second; }

    // i8mpp: manifest + weights for the tensor-ops passes (also gives the
    // post ext-CNB scratch offset the post variant is compiled with)
    long postExtOff = -1;
    char gdir[1024] = "";
    if (i8) {
        snprintf(gdir, sizeof gdir, "%s/build/gen_i8mpp_%s_%s", root, preset, resStr);
        std::string mf = std::string(gdir) + "/i8mpp_manifest.json";
        struct stat sm, sg;
        bool fresh = stat(mf.c_str(), &sm) == 0
                  && (stat((std::string(root) + "/tools/gen_i8mpp.py").c_str(), &sg) != 0 || sm.st_mtime >= sg.st_mtime);
        if (!fresh) {
            char cmd[2048];
            snprintf(cmd, sizeof cmd, "python3 '%s/tools/gen_i8mpp.py' --root '%s' --preset %s --res %s --outdir '%s'",
                     root, root, preset, resStr, gdir);
            flogf("generating %s", gdir);
            if (system(cmd) != 0) { flog("gen_i8mpp failed"); return nullptr; }
        }
        std::vector<uint8_t> mj;
        if (!load_file(mf, mj)) { flog("no i8mpp manifest"); return nullptr; }
        std::string mjs(mj.begin(), mj.end());
        size_t k = mjs.find("\"post_ext_ct2d_off\"");
        if (k == std::string::npos) { flog("manifest lacks post_ext_ct2d_off (stale build/gen_i8mpp_*)"); return nullptr; }
        postExtOff = strtol(mjs.c_str() + mjs.find(':', k) + 1, nullptr, 10);
    }

    char vdir[1024];
    snprintf(vdir, sizeof vdir, "%s/build/fsr4_%s_%s_di%d_lm%d_jm%d_cs%d_ae%d_db0%s",
             root, preset, resStr, depthInv, lowRes, jittered, cs, ae, i8 ? "_pt" : "_f16");
    std::string metallib = std::string(vdir) + "/shaders.metallib";
    std::string tsv = std::string(vdir) + "/bindings.tsv";
    char idir[1024];
    if (i8)
        snprintf(idir, sizeof idir, "%s/third_party/fsr4/internal/shaders/fsr4_model_v07_i8_%s/initializers.bin", root, preset);
    else
        snprintf(idir, sizeof idir, "%s/build/gen_f16_%s_%s/initializers_f16.bin", root, preset, resStr);
    // Validate the content hash once per uncached session, including after
    // updates. File existence alone could retain old FP16 boundary bugs.
    {
        char cmd[2048], extopt[64] = "";
        if (i8) snprintf(extopt, sizeof extopt, " --post-ext-ct2d %ld", postExtOff);
        snprintf(cmd, sizeof cmd,
                 "python3 '%s/tools/compile_variant.py' --root '%s' --preset %s "
                 "--res %s --res-idx %d --depth-inverted %d --low-res-mv %d "
                 "--jittered-mv %d --colorspace %d --auto-exposure %d "
                 "--debug-view 0 --rcas 0 --model %s --builddir '%s/build'%s",
                 root, root, preset, resStr, resIdx, depthInv, lowRes, jittered, cs, ae,
                 i8 ? "i8" : "f16", root, extopt);
        flogf("checking shaders %s", key);
        if (system(cmd) != 0 || access(metallib.c_str(), R_OK) != 0) {
            flog("shader compile failed");
            return nullptr;
        }
    }

    NSError *err = nil;
    id<MTLLibrary> lib = [dev newLibraryWithURL:[NSURL fileURLWithPath:[NSString stringWithUTF8String:metallib.c_str()]]
                                          error:&err];
    if (!lib) {
        flogf("metallib: %s", err.localizedDescription.UTF8String);
        return nullptr;
    }
    FILE *bf = fopen(tsv.c_str(), "r");
    if (!bf) { flog("no bindings.tsv"); return nullptr; }

    Session *ns = new Session();
    ns->dev = dev;
    ns->key = key;
    ns->preset = preset;
    ns->resIdx = resIdx;
    ns->i8 = i8;
    ns->jittered = jittered;
    ns->lowResMv = lowRes;
    ns->autoExposure = ae;
    char line[512];
    std::string cur;
    while (fgets(line, sizeof line, bf)) {
        char kn[128], rn[128];
        int x, y, z, idx;
        if (sscanf(line, "K\t%127s\t%d\t%d\t%d", kn, &x, &y, &z) == 4) {
            if (x <= 0 || y <= 0 || z <= 0) {
                fclose(bf); delete ns; flogf("invalid threadgroup dimensions for %s", kn); return nullptr;
            }
            cur = kn;
            id<MTLFunction> fn = [lib newFunctionWithName:[NSString stringWithUTF8String:kn]];
            if (!fn) { fclose(bf); delete ns; flogf("missing kernel %s", kn); return nullptr; }
            Kernel k;
            k.pso = [dev newComputePipelineStateWithFunction:fn error:&err];
            if (!k.pso) { fclose(bf); delete ns; flogf("pso %s", kn); return nullptr; }
            k.wg = MTLSizeMake(x, y, z);
            if ((NSUInteger)x * y * z > k.pso.maxTotalThreadsPerThreadgroup ||
                k.pso.staticThreadgroupMemoryLength > dev.maxThreadgroupMemoryLength) {
                fclose(bf); delete ns; flogf("GPU threadgroup limits exceeded by %s", kn); return nullptr;
            }
            ns->kernels[cur] = k;
        } else if (sscanf(line, "B\t%127s\t%d\t%127s", kn, &idx, rn) == 3)
            ns->kernels[kn].buffers.push_back({(NSUInteger)idx, rn});
        else if (sscanf(line, "T\t%127s\t%d\t%127s", kn, &idx, rn) == 3)
            ns->kernels[kn].textures.push_back({(NSUInteger)idx, rn});
        else if (sscanf(line, "S\t%127s\t%d\t%127s", kn, &idx, rn) == 3)
            ns->kernels[kn].samplers.push_back({(NSUInteger)idx, rn});
    }
    fclose(bf);

    std::vector<uint8_t> init;
    if (!load_file(idir, init)) { delete ns; flog("no initializers"); return nullptr; }
    ns->initBuf = [dev newBufferWithBytes:init.data() length:init.size() options:MTLResourceStorageModeShared];
    if (i8) {
        if (!ns->i8mpp.load(dev, gdir, std::string(root) + "/src/i8mpp.metal")) {
            flogf("i8mpp unavailable (%s), falling back to fp16", ns->i8mpp.error.c_str());
            delete ns;
            return nullptr;
        }
    }
    ns->weights = [dev newBufferWithBytes:weights_for(preset) length:1024 options:MTLResourceStorageModeShared];
    ns->argTables = [NSMutableDictionary new];

    // keep at most 3 variants resident
    while (g_sessions.size() >= 3) {
        auto victim = g_sessions.begin();
        for (auto i = g_sessions.begin(); i != g_sessions.end(); ++i)
            if (i->second->lastUse < victim->second->lastUse) victim = i;
        delete victim->second;
        g_sessions.erase(victim);
    }
    ns->lastUse = ++g_sh->encodes;
    g_sessions[key] = ns;
    flogf("ready %s", key);
    return ns;
}

static Session *ensure_session(id<MTLDevice> dev, const char *preset, int resIdx,
                               const char *resStr, int depthInv, int lowRes, int jittered,
                               int cs, int ae) {
    Shared &sh = *g_sh;
    fsr4::Backend request = fsr4::Backend::Auto;
    const char *model = getenv("FSR4_MODEL");
    if (!fsr4::parse_backend(model, request)) {
        if (!sh.invalidModelLogged) flogf("unknown FSR4_MODEL=%s; using auto", model);
        sh.invalidModelLogged = true;
    }
    // Preserve the live encoder's historic int8 alias. The CLI's --int8
    // remains the separate generated reference path.
    if (request == fsr4::Backend::Int8) request = fsr4::Backend::I8Mpp;
    const auto choice = fsr4::choose_backend(sh.caps, request, sh.nativeFailed);
    if (!sh.backendLogged || sh.lastBackend != choice.backend) {
        flogf("%s: backend %s (%s)", dev.name.UTF8String,
              fsr4::backend_name(choice.backend), choice.reason);
        sh.lastBackend = choice.backend;
        sh.backendLogged = true;
    }
    const bool native = choice.backend == fsr4::Backend::I8Mpp;
    Session *session = load_session(dev, preset, resIdx, resStr, depthInv, lowRes, jittered, cs, ae, native);
    if (!session && native) {
        // Retry this same encode on FP16, even with FSR4_MODEL=i8mpp.
        // The failure state belongs to this device and prevents repeated
        // compilation attempts on every frame. A new device gets new state.
        sh.nativeFailed = true;
        flog("native INT8 initialization failed; retrying this frame with FP16");
        session = load_session(dev, preset, resIdx, resStr, depthInv, lowRes, jittered, cs, ae, false);
    }
    return session;
}

static View *ensure_view(Shared &sh, void *owner) {
    auto it = sh.views.find(owner);
    if (it != sh.views.end()) { it->second->lastUse = sh.encodes; return it->second; }
    while (sh.views.size() >= 4) {
        auto victim = sh.views.begin();
        for (auto i = sh.views.begin(); i != sh.views.end(); ++i)
            if (i->second->lastUse < victim->second->lastUse) victim = i;
        delete victim->second;
        sh.views.erase(victim);
    }
    View *v = new View();
    v->owner = owner;
    v->index = sh.nextViewIndex++;
    v->lastUse = sh.encodes;
    sh.views[owner] = v;
    flogf("view %d for scaler %p", v->index, owner);
    return v;
}

static bool fit_targets(Shared &sh, View &v, uint32_t uw, uint32_t uh) {
    if (v.uw == uw && v.uh == uh && v.history) return true;
    v.history = make_tex(sh.dev, MTLPixelFormatRGBA16Float, uw, uh);
    v.reproj = make_tex(sh.dev, MTLPixelFormatRGBA16Float, uw, uh);
    v.recurrent = make_tex(sh.dev, MTLPixelFormatRGBA8Unorm, uw, uh);
    if (!v.history || !v.reproj || !v.recurrent) return false;
    v.zeroTargets = true;
    v.uw = uw;
    v.uh = uh;
    v.first = true;
    return true;
}

struct Bind {
    id<MTLTexture> color, depth, mv, output, exposure;
    View *view;
    id<MTLTexture> rcasTemp;     // nil = RCAS off: post writes the output directly
};

static id bind_tex(Shared &sh, const std::string &n, const Bind &b) {
    if (n == "r_input_color") return b.color;
    if (n == "r_velocity") return b.mv;
    if (n == "r_depth") return b.depth;
    if (n == "r_input_exposure") return b.exposure;
    if (n == "r_auto_exposure_texture" || n == "rw_auto_exposure_texture") return sh.autoExp;
    if (n == "rw_spd_global_atomic") return sh.spdAtomicTex;
    if (n == "rw_autoexp_mip_5") return sh.lumaMip5;
    if (n == "r_recurrent_0" || n == "rw_recurrent_0") return b.view->recurrent;
    if (n == "r_history_color" || n == "rw_history_color") return b.view->history;
    if (n == "r_reprojected_color" || n == "rw_reprojected_color") return b.view->reproj;
    if (n == "rw_mlsr_output_color") return b.rcasTemp ? b.rcasTemp : b.output;
    if (n == "r_rcas_input" || n == "rw_output_color_for_rcas") return b.rcasTemp;
    if (n == "rw_rcas_output" || n == "rw_final_output_color" || n == "r_result_color")
        return b.output;
    return nil;
}

static id arg_table(Shared &sh, NSMutableDictionary *tables, const char *name) {
    NSString *key = [NSString stringWithUTF8String:name];
    id existing = tables[key];
    if (existing) return existing;
    MTL4ArgumentTableDescriptor *d = [MTL4ArgumentTableDescriptor new];
    d.maxBufferBindCount = 8;
    d.maxTextureBindCount = 16;
    d.maxSamplerStateBindCount = 4;
    d.initializeBindings = YES;
    NSError *err = nil;
    id table = [sh.dev newArgumentTableWithDescriptor:d error:&err];
    if (!table) {
        flogf("argument table: %s", err.localizedDescription.UTF8String);
        return nil;
    }
    tables[key] = table;
    return table;
}

static void residency_add(Shared &sh, id res) {
    if (!res) return;
    if (!sh.residency) {
        NSError *err = nil;
        sh.residency = [sh.dev newResidencySetWithDescriptor:[MTLResidencySetDescriptor new] error:&err];
        if (!sh.residency) {
            flogf("residency set: %s", err.localizedDescription.UTF8String);
            return;
        }
    }
    if ([sh.residency containsAllocation:(id<MTLAllocation>)res]) return;
    [sh.residency addAllocation:(id<MTLAllocation>)res];
}

static void m4_barrier(id enc) {
    [enc barrierAfterEncoderStages:MTLStageDispatch | MTLStageBlit
               beforeEncoderStages:MTLStageDispatch | MTLStageBlit
                 visibilityOptions:MTL4VisibilityOptionDevice];
}

static bool dispatch_m4(Shared &sh, Session &s, id enc, const char *name,
                        uint32_t gx, uint32_t gy, MTLGPUAddress cbAddr, const Bind &b) {
    auto it = s.kernels.find(name);
    if (it == s.kernels.end()) return false;
    Kernel &k = it->second;
    id table = arg_table(sh, s.argTables, name);
    if (!table) return false;
    for (auto &bb : k.buffers) {
        MTLGPUAddress addr = 0;
        if (bb.second == "ScratchBuffer") addr = sh.scratch.gpuAddress;
        else if (bb.second == "InitializerBuffer") addr = s.initBuf.gpuAddress;
        else if (bb.second == "cbPass_Weights") addr = s.weights.gpuAddress;
        else if (bb.second == "MLSR_Optimized_Constants") addr = cbAddr;
        else if (bb.second == "AutoExposureSPDConstants") addr = cbAddr + kSpdOff;
        else if (bb.second == "cbRCAS") addr = cbAddr + kRcasOff;
        else if (bb.second == "rw_spd_global_atomic_atomic") addr = sh.spdAtomicBuf.gpuAddress;
        else { flogf("unknown buffer binding %s", bb.second.c_str()); return false; }
        [(id<MTL4ArgumentTable>)table setAddress:addr atIndex:bb.first];
    }
    for (auto &t : k.textures) {
        id<MTLTexture> tx = bind_tex(sh, t.second, b);
        if (!tx) { flogf("unknown texture binding %s", t.second.c_str()); return false; }
        [(id<MTL4ArgumentTable>)table setTexture:tx.gpuResourceID atIndex:t.first];
    }
    for (auto &sm : k.samplers)
        [(id<MTL4ArgumentTable>)table setSamplerState:sh.samp.gpuResourceID atIndex:sm.first];
    [enc setArgumentTable:table];
    [enc setComputePipelineState:k.pso];
    [enc dispatchThreadgroups:MTLSizeMake(gx, gy, 1) threadsPerThreadgroup:k.wg];
    m4_barrier(enc);
    return true;
}

static bool dispatch_k(Shared &sh, Session &s, id<MTLComputeCommandEncoder> enc, const char *name,
                       uint32_t gx, uint32_t gy, const OptCB &oc, const SpdCB &sc, const RcasCB &rc,
                       const Bind &b) {
    auto it = s.kernels.find(name);
    if (it == s.kernels.end()) return false;
    Kernel &k = it->second;
    [enc setComputePipelineState:k.pso];
    for (auto &bb : k.buffers) {
        if (bb.second == "ScratchBuffer")
            [enc setBuffer:sh.scratch offset:0 atIndex:bb.first];
        else if (bb.second == "InitializerBuffer")
            [enc setBuffer:s.initBuf offset:0 atIndex:bb.first];
        else if (bb.second == "cbPass_Weights")
            [enc setBuffer:s.weights offset:0 atIndex:bb.first];
        else if (bb.second == "MLSR_Optimized_Constants")
            [enc setBytes:&oc length:sizeof oc atIndex:bb.first];
        else if (bb.second == "AutoExposureSPDConstants")
            [enc setBytes:&sc length:sizeof sc atIndex:bb.first];
        else if (bb.second == "cbRCAS")
            [enc setBytes:&rc length:sizeof rc atIndex:bb.first];
        else if (bb.second == "rw_spd_global_atomic_atomic")
            [enc setBuffer:sh.spdAtomicBuf offset:0 atIndex:bb.first];
    }
    for (auto &t : k.textures) {
        id<MTLTexture> tx = bind_tex(sh, t.second, b);
        if (tx) [enc setTexture:tx atIndex:t.first];
    }
    for (auto &sm : k.samplers)
        [enc setSamplerState:sh.samp atIndex:sm.first];
    [enc dispatchThreadgroups:MTLSizeMake(gx, gy, 1) threadsPerThreadgroup:k.wg];
    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers | MTLBarrierScopeTextures];
    return true;
}

// Helper dispatch (own tiny kernels): up to 3 textures + 2 buffers.
static bool helper_m4(Shared &sh, id enc, const char *name, id<MTLComputePipelineState> pso,
                      id<MTLTexture> t0, id<MTLTexture> t1, id<MTLTexture> t2,
                      id<MTLBuffer> b0, id<MTLBuffer> b1, MTLSize grid, MTLSize tg) {
    id table = arg_table(sh, sh.argTables, name);
    if (!table) return false;
    if (t0) [(id<MTL4ArgumentTable>)table setTexture:t0.gpuResourceID atIndex:0];
    if (t1) [(id<MTL4ArgumentTable>)table setTexture:t1.gpuResourceID atIndex:1];
    if (t2) [(id<MTL4ArgumentTable>)table setTexture:t2.gpuResourceID atIndex:2];
    if (b0) [(id<MTL4ArgumentTable>)table setAddress:b0.gpuAddress atIndex:0];
    if (b1) [(id<MTL4ArgumentTable>)table setAddress:b1.gpuAddress atIndex:1];
    [enc setArgumentTable:table];
    [enc setComputePipelineState:pso];
    [enc dispatchThreads:grid threadsPerThreadgroup:tg];
    m4_barrier(enc);
    return true;
}

static void helper_classic(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pso,
                           id<MTLTexture> t0, id<MTLTexture> t1, id<MTLTexture> t2,
                           id<MTLBuffer> b0, id<MTLBuffer> b1, MTLSize grid, MTLSize tg) {
    [enc setComputePipelineState:pso];
    if (t0) [enc setTexture:t0 atIndex:0];
    if (t1) [enc setTexture:t1 atIndex:1];
    if (t2) [enc setTexture:t2 atIndex:2];
    if (b0) [enc setBuffer:b0 offset:0 atIndex:0];
    if (b1) [enc setBuffer:b1 offset:0 atIndex:1];
    [enc dispatchThreads:grid threadsPerThreadgroup:tg];
    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers | MTLBarrierScopeTextures];
}

static void encode_initial_clear(Shared &sh, View &v, id enc, bool mtl4, uint32_t uw, uint32_t uh) {
    if (sh.zeroScratch) {
        const bool mtl4Fill = mtl4 && [enc respondsToSelector:@selector(fillBuffer:range:value:)];
        if (mtl4Fill) {
            [enc fillBuffer:sh.scratch range:NSMakeRange(0, sh.scratch.length) value:0];
            m4_barrier(enc);
        } else if (!mtl4 && sh.clearBufPso) {
            uint32_t n4 = (uint32_t)(sh.scratch.length / 16);
            id<MTLComputeCommandEncoder> classic = enc;
            [classic setComputePipelineState:sh.clearBufPso];
            [classic setBuffer:sh.scratch offset:0 atIndex:0];
            [classic setBytes:&n4 length:sizeof n4 atIndex:1];
            [classic dispatchThreadgroups:MTLSizeMake((n4 + 255) / 256, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [classic memoryBarrierWithScope:MTLBarrierScopeBuffers];
        } else {
            memset(sh.scratch.contents, 0, sh.scratch.length);
        }
        sh.zeroScratch = false;
    }
    if (!v.zeroTargets) return;
    if (sh.clearTexPso) {
        if (mtl4) {
            if (!helper_m4(sh, enc, "fsr4_clear_tex", sh.clearTexPso, v.history, v.reproj, v.recurrent,
                           nil, nil, MTLSizeMake(uw, uh, 1), MTLSizeMake(8, 8, 1))) {
                zero_tex(v.history); zero_tex(v.reproj); zero_tex(v.recurrent);
            }
        } else {
            helper_classic(enc, sh.clearTexPso, v.history, v.reproj, v.recurrent, nil, nil,
                           MTLSizeMake(uw, uh, 1), MTLSizeMake(8, 8, 1));
        }
    } else {
        zero_tex(v.history); zero_tex(v.reproj); zero_tex(v.recurrent);
    }
    v.zeroTargets = false;
}

static const char *fmt_name(MTLPixelFormat f) {
    switch (f) {
        case MTLPixelFormatRGBA8Unorm: return "RGBA8";
        case MTLPixelFormatBGRA8Unorm: return "BGRA8";
        case MTLPixelFormatRG16Float: return "RG16F";
        case MTLPixelFormatRG32Float: return "RG32F";
        case MTLPixelFormatRGBA16Float: return "RGBA16F";
        case MTLPixelFormatRGBA32Float: return "RGBA32F";
        case MTLPixelFormatRG16Snorm: return "RG16S";
        case MTLPixelFormatRG16Unorm: return "RG16U";
        case MTLPixelFormatR32Float: return "R32F";
        case MTLPixelFormatR16Float: return "R16F";
        case MTLPixelFormatDepth32Float: return "D32F";
        case MTLPixelFormatDepth32Float_Stencil8: return "D32FS8";
        case MTLPixelFormatDepth16Unorm: return "D16";
        case MTLPixelFormatRG11B10Float: return "RG11B10F";
        case MTLPixelFormatRGB10A2Unorm: return "RGB10A2";
        default: return "?";
    }
}

enum StubPolicy { STUB_SKIP, STUB_METALFX, STUB_RUN, STUB_RESET };

static StubPolicy stub_policy() {
    const char *e = getenv("FSR4_STUB");
    if (!e || !e[0] || !strcmp(e, "skip")) return STUB_SKIP;
    if (!strcmp(e, "metalfx")) return STUB_METALFX;
    if (!strcmp(e, "run")) return STUB_RUN;
    if (!strcmp(e, "reset")) return STUB_RESET;
    return STUB_SKIP;
}

// Copy the last real output into `out` (stub encode whose output differs).
static bool copy_output(Shared &sh, id enc, bool mtl4, id<MTLTexture> src, id<MTLTexture> dst) {
    if (!src || !dst || src == dst) return true;
    if (src.width != dst.width || src.height != dst.height) return false;
    if (mtl4 && src.pixelFormat == dst.pixelFormat
        && [enc respondsToSelector:@selector(copyFromTexture:toTexture:)]) {
        [enc copyFromTexture:src toTexture:dst];
        m4_barrier(enc);
        return true;
    }
    if (!sh.copyTexPso) return false;
    if (mtl4)
        return helper_m4(sh, enc, "fsr4_copy_tex", sh.copyTexPso, src, dst, nil, nil, nil,
                         MTLSizeMake(dst.width, dst.height, 1), MTLSizeMake(8, 8, 1));
    helper_classic(enc, sh.copyTexPso, src, dst, nil, nil, nil,
                   MTLSizeMake(dst.width, dst.height, 1), MTLSizeMake(8, 8, 1));
    return true;
}

static int encode_legacy_frame(const Fsr4LiveFrame *f, const Fsr4InputGeometry &geometry,
                               int inputContentProperties) {
    if (!f || !f->device || !f->commandBuffer || !f->color || !f->depth || !f->motion || !f->output)
        return fail_reason("no textures from the scaler");
    if (!f->rw || !f->rh || !f->uw || !f->uh) return fail_reason("zero size");
    // Provider: upscaleWidth/Height = AlignUp(upscaleSize, 8); the internal
    // textures, constants and dispatch sizes use the aligned size, the app's
    // output texture keeps its own size (writes past its edge are dropped).
    const uint32_t rw = f->rw, rh = f->rh, uw = (f->uw + 7) & ~7u, uh = (f->uh + 7) & ~7u;
    id<MTLDevice> dev = (__bridge id<MTLDevice>)f->device;
    id cb = (__bridge id)f->commandBuffer;
    id<MTLTexture> col = (__bridge id<MTLTexture>)f->color;
    id<MTLTexture> dep = (__bridge id<MTLTexture>)f->depth;
    id<MTLTexture> mv = (__bridge id<MTLTexture>)f->motion;
    id<MTLTexture> out = (__bridge id<MTLTexture>)f->output;
    id<MTLTexture> exp = (__bridge id<MTLTexture>)f->exposure;

    std::lock_guard<std::mutex> lock(g_mu);
    Shared &sh = *ensure_shared(dev);
    sh.encodes++;
    const uint64_t n = sh.encodes;
    View &v = *ensure_view(sh, f->owner);
    v.encodes++;
    struct InputHistoryGuard {
        View &view;
        Shared &shared;
        bool committed = false;
        ~InputHistoryGuard() {
            if (!committed) {
                fsr4_input_invalidate(&view.inputHistory);
                view.first = true;
                if (view.lastOut && shared.lastRealOut == view.lastOut) shared.lastRealOut = nil;
            }
        }
    } inputHistoryGuard{v, sh};

    // D3DMetal's known unbound-resource placeholder. A valid output-res MV
    // field may be smaller than render content when downscaling; dimensions
    // alone must not classify it as a placeholder or infer its convention.
    const uint32_t expectedMvW = f->outputResMV > 0 ? f->uw : rw;
    const uint32_t expectedMvH = f->outputResMV > 0 ? f->uh : rh;
    const bool stub = mv.width == 4 && mv.height == 4 && mv.pixelFormat == MTLPixelFormatRGBA8Unorm
                   && (mv.width < expectedMvW || mv.height < expectedMvH);
    StubPolicy policy = stub_policy();

    const bool trace = n <= 60 || (n % 600) == 0;
    if (trace)
        flogf("enc#%llu t=%.1fms scaler=%p view=%d cb=%p color=%p %lux%lu %s depth=%p %lux%lu %s "
              "mv=%p %lux%lu %s%s out=%p %lux%lu %s exp=%p content %ux%u->%ux%u jit %.3f,%.3f "
              "mvScale %.3f,%.3f preExp %.3f reset %d depthRev %d flags jit%d outres%d ae%d off %u,%u",
              (unsigned long long)n, now_ms(), f->owner, v.index, f->commandBuffer,
              f->color, (unsigned long)col.width, (unsigned long)col.height, fmt_name(col.pixelFormat),
              f->depth, (unsigned long)dep.width, (unsigned long)dep.height, fmt_name(dep.pixelFormat),
              f->motion, (unsigned long)mv.width, (unsigned long)mv.height, fmt_name(mv.pixelFormat),
              stub ? " STUB" : "",
              f->output, (unsigned long)out.width, (unsigned long)out.height, fmt_name(out.pixelFormat),
              f->exposure, rw, rh, uw, uh, f->jitterX, f->jitterY, f->mvScaleX, f->mvScaleY,
              f->preExposure, f->reset, f->depthReversed, f->jitteredMV, f->outputResMV, f->autoExposure,
              f->mvOffX, f->mvOffY);

    if (stub && policy == STUB_METALFX) return fail_reason("stub motion, handed to MetalFX");

    // Variant flags: descriptor first, environment overrides.
    int lowRes = f->outputResMV > 0 ? 0 : 1;
    if (const char *e = getenv("FSR4_LOW_RES_MV")) if (e[0]) lowRes = e[0] != '0';
    int jittered = f->jitteredMV > 0 ? 1 : 0;
    if (const char *e = getenv("FSR4_JITTERED_MV")) if (e[0]) jittered = e[0] == '1';
    int ae = f->autoExposure > 0 ? 1 : 0;
    if (const char *e = getenv("FSR4_AUTOEXPOSURE")) if (e[0]) ae = e[0] == '1';
    int cs = 0;
    if (const char *e = getenv("FSR4_COLORSPACE")) {
        if (!strcmp(e, "srgb")) cs = 2;
        else if (!strcmp(e, "pq")) cs = 3;
    }
    Fsr4InputDesc input = fsr4_live_input_desc(f, &geometry, inputContentProperties);
    if (const char *e = getenv("FSR4_LOW_RES_MV")) if (e[0]) input.outputResMV = !lowRes;
    if (const char *e = getenv("FSR4_JITTERED_MV")) if (e[0]) input.jitteredMV = jittered;
    if (const char *e = getenv("FSR4_AUTOEXPOSURE")) if (e[0]) input.autoExposure = ae;
    Fsr4InputPrepared prepared = {};
    // A placeholder encode can copy the previous real result without reading
    // its invalid motion field. Every actual network execution must satisfy
    // the full input contract, even under the historic stub run/reset modes.
    if (stub && policy == STUB_SKIP) {
        if (!fsr4_input_region_fits(geometry.output, f->uw, f->uh)
            || fsr4_input_surface_needs_copy(geometry.output, f->uw, f->uh))
            return fail_reason("legacy stub output region requires FSR 4.1.1");
    } else {
        const auto inputStatus = fsr4_input_prepare(&input, &v.inputHistory, 0, &prepared);
        if (!fsr4_input_status_ok(inputStatus)) return fail_reason(fsr4_input_status_reason(inputStatus));
        if (prepared.copyMask) return fail_reason("input region copies require FSR 4.1.1");
    }
    const char *preset = preset_for(rw, uw);
    int resIdx = 0;
    const char *resStr = "1080";
    if (uw > 1920 || uh > 1080) { resIdx = 1; resStr = "2160"; }
    if (uw > 3840 || uh > 2160) { resIdx = 2; resStr = "4320"; }

    // Open the encoder first: a stub encode may only need a copy.
    // Spider-Man's DLSS path hands us an MTL4 command buffer. Classic
    // setBuffer: calls on that encoder throw and the frame stays black.
    id enc = [cb computeCommandEncoder];
    if (!enc) {
        if (!g_fail_logged++) flog("command buffer already encoding");
        return fail_reason("command buffer busy");
    }
    const bool mtl4 = ![enc respondsToSelector:@selector(setBuffer:offset:atIndex:)];
    if (mtl4 && ![enc respondsToSelector:@selector(setArgumentTable:)]) {
        [enc endEncoding];
        flog("encoder is not Metal or Metal 4, MetalFX fallback");
        return fail_reason("unknown encoder type");
    }
    const bool qbar = mtl4 && env_int("FSR4_QBARRIER", 1) != 0
                      && [enc respondsToSelector:@selector(barrierAfterQueueStages:beforeStages:visibilityOptions:)];
    if (qbar)
        [enc barrierAfterQueueStages:MTLStageDispatch | MTLStageBlit | MTLStageFragment | MTLStageVertex
                        beforeStages:MTLStageDispatch | MTLStageBlit
                   visibilityOptions:MTL4VisibilityOptionDevice];

    if (mtl4) {
        residency_add(sh, sh.lastRealOut);
        residency_add(sh, out);
    }

    if (stub && policy == STUB_SKIP) {
        { std::lock_guard<std::mutex> lk(g_status_mu); g_status.stubs++; g_status.encodes++; }
        bool copied = copy_output(sh, enc, mtl4, sh.lastRealOut, out);
        if (qbar)
            [enc barrierAfterStages:MTLStageDispatch | MTLStageBlit beforeQueueStages:MTLStageAll
                  visibilityOptions:MTL4VisibilityOptionDevice];
        [enc endEncoding];
        if (mtl4 && sh.residency) {
            [sh.residency commit];
            if ([cb respondsToSelector:@selector(useResidencySet:)]) [cb useResidencySet:sh.residency];
        }
        static int onceStub = 0;
        if (!onceStub++)
            flogf("stub motion %lux%lu %s: skipped FSR4, %s", (unsigned long)mv.width, (unsigned long)mv.height,
                  fmt_name(mv.pixelFormat), copied ? "copied last real output" : "left output untouched");
        return copied ? 1 : fail_reason("stub motion has no compatible previous output");
    }

    Session *sp = ensure_session(dev, preset, resIdx, resStr, f->depthReversed ? 1 : 0, lowRes, jittered, cs, ae);
    if (!sp || !ensure_scratch(sh, resIdx) || !fit_targets(sh, v, uw, uh)) {
        [enc endEncoding];
        return fail_reason(!sp ? "shaders failed to build (see hook.log)" : "memory allocation failed");
    }
    Session &s = *sp;
    const float sharpness = sharpness_setting();
    const bool rcas = sharpness > 0.f;
    if (!s.kernels.count("fsr4_pre") || !s.kernels.count("fsr4_post") || (ae && !s.kernels.count("fsr4_spd"))
        || (rcas && !s.kernels.count("fsr4_rcas"))) {
        [enc endEncoding]; return fail_reason("metallib is missing kernels");
    }
    if (rcas && (sh.rcasW != uw || sh.rcasH != uh || !sh.rcasTemp)) {
        sh.rcasTemp = make_tex(dev, MTLPixelFormatRGBA16Float, uw, uh);
        sh.rcasW = uw; sh.rcasH = uh;
        if (!sh.rcasTemp) { [enc endEncoding]; return fail_reason("RCAS texture allocation failed"); }
    }
    for (int i = 1; i <= 13 && !s.i8; i++) {
        if (i == 13) break;
        char kn[64];
        snprintf(kn, sizeof kn, "fsr4_model_v07_f16_pass%d", i);
        if (!s.kernels.count(kn)) { [enc endEncoding]; return fail_reason("missing fp16 model pass"); }
    }
    if (s.i8) {
        for (int i = 1; i <= 13; i++)
            if (!s.i8mpp.has(i)) { [enc endEncoding]; flogf("i8mpp: pass %d missing", i); return fail_reason("missing i8mpp pass"); }
        s.i8mpp.set_region(uw, uh);
    }

    // Provider: reset is forced on the dispatch that (re)creates the model —
    // the first frame of a context and every preset/colourspace change.
    const bool modelChanged = v.first || v.sessionKey != s.key;
    const bool forceReset = (stub && policy == STUB_RESET) || modelChanged;
    const auto inputStatus = fsr4_input_prepare(&input, &v.inputHistory, forceReset, &prepared);
    if (!fsr4_input_status_ok(inputStatus)) {
        [enc endEncoding]; return fail_reason(fsr4_input_status_reason(inputStatus));
    }
    const float pe = prepared.preExposure;
    id<MTLTexture> expTex = prepared.exposureMode == FSR4_INPUT_EXPOSURE_TEXTURE ? exp : sh.exposureOne;

    OptCB oc = {};
    oc.inv_size[0] = 1.f / uw; oc.inv_size[1] = 1.f / uh;
    oc.scale[0] = (float)uw / rw; oc.scale[1] = (float)uh / rh;
    oc.inv_scale[0] = 1.f / oc.scale[0]; oc.inv_scale[1] = 1.f / oc.scale[1];
    oc.tex_size[0] = (float)uw; oc.tex_size[1] = (float)uh;
    oc.max_renderSize[0] = (float)prepared.maxInputWidth; oc.max_renderSize[1] = (float)prepared.maxInputHeight;
    oc.width = uw; oc.height = uh;
    oc.width_lr = rw; oc.height_lr = rh;
    oc.jitter[0] = prepared.jitter[0]; oc.jitter[1] = prepared.jitter[1];
    oc.mv_scale[0] = prepared.mvScale[0]; oc.mv_scale[1] = prepared.mvScale[1];
    oc.fMotionVectorJitterCancellation[0] = prepared.jitterCancellation[0];
    oc.fMotionVectorJitterCancellation[1] = prepared.jitterCancellation[1];
    oc.reset = prepared.reset;
    oc.rcas_enabled = rcas ? 1u : 0u;
    oc.rcas_sharpness = sharpness;
    oc.previous_preExposure = prepared.previousPreExposure;
    oc.preExposure = pe;

    SpdCB sc = {};
    uint32_t spdGroups[2] = {0, 0};
    if (ae) {
        uint32_t wgo[2], nwm[2];
        uint32_t rect[4] = {0, 0, rw, rh};
        spd_setup(spdGroups, wgo, nwm, rect);
        sc.workGroupOffset[0] = wgo[0]; sc.workGroupOffset[1] = wgo[1];
        sc.numWorkGroups = nwm[0]; sc.mips = nwm[1];
        sc.invInputSize[0] = 1.f / rw; sc.invInputSize[1] = 1.f / rh;
        sc.preExposure = pe;
        uint32_t lw = divup(rw, 16), lh = divup(rh, 16);
        if (!sh.lumaMip5 || sh.lumaW != lw || sh.lumaH != lh) {
            sh.lumaMip5 = make_tex(dev, MTLPixelFormatR32Float, lw, lh);
            sh.lumaW = lw; sh.lumaH = lh;
        }
    }

    RcasCB rc = {};
    if (rcas) {
        rc.preExposure = pe;
        rcas_con(rc.config, -2.0f * sharpness + 2.0f);
    }

    MTLGPUAddress cbAddr = 0;
    id<MTLBuffer> slot = sh.cbRing[sh.cbSlot++ % kCbRing];
    memcpy(slot.contents, &oc, sizeof oc);
    memcpy((uint8_t *)slot.contents + kSpdOff, &sc, sizeof sc);
    memcpy((uint8_t *)slot.contents + kRcasOff, &rc, sizeof rc);
    cbAddr = slot.gpuAddress;

    // depth: a depth format is copied into R32F (the pre pass indexes a Texture2D<float>)
    const bool depthTex = dep.pixelFormat == MTLPixelFormatDepth16Unorm
                       || dep.pixelFormat == MTLPixelFormatDepth32Float
                       || dep.pixelFormat == MTLPixelFormatDepth32Float_Stencil8;
    if (depthTex && (sh.drw != rw || sh.drh != rh || !sh.depthCopy)) {
        sh.depthCopy = make_tex(dev, MTLPixelFormatR32Float, rw, rh);
        sh.drw = rw; sh.drh = rh;
    }

    if (mtl4) {
        residency_add(sh, sh.scratch);
        residency_add(sh, s.weights);
        residency_add(sh, s.initBuf);
        if (s.i8) { residency_add(sh, s.i8mpp.weights); residency_add(sh, s.i8mpp.params); }
        for (uint32_t i = 0; i < kCbRing; i++) residency_add(sh, sh.cbRing[i]);
        residency_add(sh, v.history);
        residency_add(sh, v.reproj);
        residency_add(sh, v.recurrent);
        residency_add(sh, sh.exposureOne);
        residency_add(sh, sh.autoExp);
        residency_add(sh, sh.spdAtomicBuf);
        residency_add(sh, sh.spdAtomicTex);
        residency_add(sh, sh.lumaMip5);
        residency_add(sh, sh.depthCopy);
        residency_add(sh, sh.rcasTemp);
        residency_add(sh, sh.probeBuf);
        residency_add(sh, col);
        residency_add(sh, dep);
        residency_add(sh, mv);
        residency_add(sh, expTex);
        if (sh.residency) {
            [sh.residency commit];
            if ([cb respondsToSelector:@selector(useResidencySet:)]) [cb useResidencySet:sh.residency];
        }
    }

    @try {
    if (depthTex) {
        if (!sh.depthPso) { [enc endEncoding]; return fail_reason("missing depth copy kernel"); }
        if (mtl4) {
            if (!helper_m4(sh, enc, "depth", sh.depthPso, dep, sh.depthCopy, nil, nil, nil,
                           MTLSizeMake(rw, rh, 1), MTLSizeMake(8, 8, 1))) { [enc endEncoding]; return fail_reason("depth copy failed to encode"); }
        } else {
            helper_classic(enc, sh.depthPso, dep, sh.depthCopy, nil, nil, nil,
                           MTLSizeMake(rw, rh, 1), MTLSizeMake(8, 8, 1));
        }
        dep = sh.depthCopy;
    }
    encode_initial_clear(sh, v, enc, mtl4, uw, uh);

    Bind b = {col, dep, mv, out, expTex, &v, rcas ? sh.rcasTemp : nil};
    uint32_t e0 = uw >> 1, e1 = uh >> 1;
    uint32_t f0 = uw >> 2, f1 = uh >> 2;
    uint32_t g0 = uw >> 3, g1 = uh >> 3;
    struct Pass { const char *n; uint32_t x, y; int i8pass; };
    std::vector<Pass> passes;
    if (ae) passes.push_back({"fsr4_spd", spdGroups[0], spdGroups[1], 0});
    passes.push_back({"fsr4_pre", divup(uw, 16), divup(uh, 16), 0});
    if (s.i8) {
        for (int i = 1; i <= 13; i++) passes.push_back({"i8mpp", 0, 0, i});
    } else {
        const Pass model[12] = {
            {"fsr4_model_v07_f16_pass1", divup(e0, s.kernels.at("fsr4_model_v07_f16_pass1").wg.width), e1, 0},
            {"fsr4_model_v07_f16_pass2", divup(e0, s.kernels.at("fsr4_model_v07_f16_pass2").wg.width), e1, 0},
            {"fsr4_model_v07_f16_pass3", divup(f0, s.kernels.at("fsr4_model_v07_f16_pass3").wg.width), f1, 0},
            {"fsr4_model_v07_f16_pass4", divup(f0, s.kernels.at("fsr4_model_v07_f16_pass4").wg.width), f1, 0},
            {"fsr4_model_v07_f16_pass5", divup(f0, s.kernels.at("fsr4_model_v07_f16_pass5").wg.width), f1, 0},
            {"fsr4_model_v07_f16_pass6", divup(g0, s.kernels.at("fsr4_model_v07_f16_pass6").wg.width), g1, 0},
            {"fsr4_model_v07_f16_pass7", divup(g0, s.kernels.at("fsr4_model_v07_f16_pass7").wg.width), g1, 0},
            {"fsr4_model_v07_f16_pass8", divup(g0, s.kernels.at("fsr4_model_v07_f16_pass8").wg.width), g1, 0},
            {"fsr4_model_v07_f16_pass9", divup(g0, 8), divup(g1, 8), 0},
            {"fsr4_model_v07_f16_pass10", divup(f0, s.kernels.at("fsr4_model_v07_f16_pass10").wg.width), f1, 0},
            {"fsr4_model_v07_f16_pass11", divup(f0, s.kernels.at("fsr4_model_v07_f16_pass11").wg.width), f1, 0},
            {"fsr4_model_v07_f16_pass12", divup(e0, s.kernels.at("fsr4_model_v07_f16_pass12").wg.width), e1, 0},
        };
        for (auto &p : model) passes.push_back(p);
    }
    passes.push_back({"fsr4_post", divup(uw, 16), divup(uh, 16), 0});
    if (rcas) passes.push_back({"fsr4_rcas", divup(uw, 16), divup(uh, 16), 0});
    for (auto &p : passes) {
        bool ok;
        if (p.i8pass) {
            // pass 13 writes this view's recurrent texture
            id<MTLTexture> tex = p.i8pass == 13 ? v.recurrent : nil;
            if (mtl4) {
                char tn[32];
                snprintf(tn, sizeof tn, "i8mpp_%d", p.i8pass);
                id table = arg_table(sh, s.argTables, tn);
                ok = table != nil;
                if (ok) s.i8mpp.dispatch_m4(p.i8pass, enc, table, sh.scratch, tex);
            } else {
                s.i8mpp.dispatch(p.i8pass, enc, sh.scratch, MTLBarrierScopeBuffers | MTLBarrierScopeTextures, tex);
                ok = true;
            }
        } else {
            ok = mtl4 ? dispatch_m4(sh, s, enc, p.n, p.x, p.y, cbAddr, b)
                      : dispatch_k(sh, s, enc, p.n, p.x, p.y, oc, sc, rc, b);
        }
        if (!ok) {
            [enc endEncoding];
            flogf("dispatch failed: %s", p.n);
            return fail_reason("dispatch failed");
        }
    }

    // GPU probe: exposure value, |mv| max over a 16x16 grid, centre depth.
    if (sh.probePso && sh.probeBuf) {
        uint32_t ps = (uint32_t)(n % kProbeSlots);
        uint32_t *pslot = (uint32_t *)((uint8_t *)sh.probeBuf.contents + ps * kProbeBytes);
        if (n > kProbeSlots && (n % 120) == 0) {
            float ev, dv;
            memcpy(&ev, &pslot[0], 4);
            memcpy(&dv, &pslot[2], 4);
            flogf("probe(enc#%llu) exposure=%.5f mvmax=%.2fpx nonzero=%u/256 depth_centre=%.5f",
                  (unsigned long long)(n - kProbeSlots), ev, pslot[1] / 256.f, pslot[3], dv);
        }
        memset(pslot, 0, kProbeBytes);
        uint32_t content[2] = {rw, rh};
        id<MTLBuffer> cbuf = slot;   // reuse the cb slot tail for the probe constants
        memcpy((uint8_t *)slot.contents + kSpdOff + 64, content, sizeof content);
        id<MTLTexture> probeExp = ae ? sh.autoExp : expTex;
        if (mtl4) {
            id table = arg_table(sh, sh.argTables, "fsr4_probe");
            if (table) {
                [(id<MTL4ArgumentTable>)table setTexture:probeExp.gpuResourceID atIndex:0];
                [(id<MTL4ArgumentTable>)table setTexture:mv.gpuResourceID atIndex:1];
                [(id<MTL4ArgumentTable>)table setTexture:dep.gpuResourceID atIndex:2];
                [(id<MTL4ArgumentTable>)table setAddress:sh.probeBuf.gpuAddress + ps * kProbeBytes atIndex:0];
                [(id<MTL4ArgumentTable>)table setAddress:cbuf.gpuAddress + kSpdOff + 64 atIndex:1];
                [enc setArgumentTable:table];
                [enc setComputePipelineState:sh.probePso];
                [enc dispatchThreads:MTLSizeMake(256, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            }
        } else {
            id<MTLComputeCommandEncoder> classic = enc;
            [classic setComputePipelineState:sh.probePso];
            [classic setTexture:probeExp atIndex:0];
            [classic setTexture:mv atIndex:1];
            [classic setTexture:dep atIndex:2];
            [classic setBuffer:sh.probeBuf offset:ps * kProbeBytes atIndex:0];
            [classic setBytes:content length:sizeof content atIndex:1];
            [classic dispatchThreads:MTLSizeMake(256, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        }
    }
    if (qbar)
        [enc barrierAfterStages:MTLStageDispatch | MTLStageBlit beforeQueueStages:MTLStageAll
              visibilityOptions:MTL4VisibilityOptionDevice];
    [enc endEncoding];
    } @catch (NSException *e) {
        flogf("encode exception: %s", e.reason.UTF8String);
        if ([enc respondsToSelector:@selector(endEncoding)]) [enc endEncoding];
        return fail_reason("exception while encoding (see hook.log)");
    }
    fsr4_input_commit(&v.inputHistory, &prepared);
    inputHistoryGuard.committed = true;
    v.sessionKey = s.key;
    v.first = false;
    v.lastOut = out;
    if (!stub) {
        sh.lastRealOut = out;
        const double t = now_ms();
        if (sh.lastRealT > 0) {
            const float dt = (float)(t - sh.lastRealT);
            if (dt > 0 && dt < 1000) sh.frameMs = sh.frameMs > 0 ? sh.frameMs * 0.9f + dt * 0.1f : dt;
        }
        sh.lastRealT = t;
    }
    {
        std::lock_guard<std::mutex> lk(g_status_mu);
        g_status.active = 1;
        g_status.reason[0] = 0;
        snprintf(g_status.preset, sizeof g_status.preset, "%s", preset);
        snprintf(g_status.model, sizeof g_status.model, "%s", s.i8 ? "i8mpp" : "f16");
        snprintf(g_status.variant, sizeof g_status.variant, "%s", s.key.c_str());
        g_status.rw = rw; g_status.rh = rh; g_status.uw = f->uw; g_status.uh = f->uh;
        g_status.uwA = uw; g_status.uhA = uh;
        g_status.frameMs = sh.frameMs;
        g_status.encodes++;
        if (!stub) g_status.replaced++;
        g_status.sharpness = sharpness;
    }
    fsr4_panel_publish();
    static int once = 0;
    if (!once++ || trace)
        flogf("encode %s %ux%u -> %ux%u (aligned %ux%u) preset=%s variant=%s view=%d mv_scale %.6f,%.6f reset=%u sharpness=%.2f%s qbarrier=%d%s",
              mtl4 ? "mtl4" : "classic", rw, rh, f->uw, f->uh, uw, uh, preset, s.key.c_str(), v.index,
              oc.mv_scale[0], oc.mv_scale[1], oc.reset, sharpness, rcas ? " (RCAS)" : " (RCAS off)",
              qbar ? 1 : 0, stub ? " (stub motion, ran anyway)" : "");
    return 1;
}

// Optional while linking a legacy-only dylib. An explicit 4.1.1 request must
// never silently execute the old model under the new version's name.
extern "C" int fsr411_encode_live_frame(const Fsr4LiveFrameV2 *) __attribute__((weak_import));
extern "C" void fsr411_invalidate_owner(void *, void *) __attribute__((weak_import));
extern "C" void fsr411_release_owner(void *, void *) __attribute__((weak_import));

static void invalidate_legacy_input(const Fsr4LiveFrame *frame) {
    if (!frame) return;
    std::lock_guard<std::mutex> lock(g_mu);
    if (!g_sh || (__bridge void *)g_sh->dev != frame->device) return;
    const auto it = g_sh->views.find(frame->owner);
    if (it == g_sh->views.end()) return;
    View &view = *it->second;
    fsr4_input_invalidate(&view.inputHistory);
    view.first = true;
    if (view.lastOut && g_sh->lastRealOut == view.lastOut) g_sh->lastRealOut = nil;
}

void fsr4_invalidate_live_owner(void *device, void *owner) {
    Fsr4LiveFrame frame = {};
    frame.device = device; frame.owner = owner;
    invalidate_legacy_input(&frame);
    if (fsr411_invalidate_owner) fsr411_invalidate_owner(device, owner);
}

void fsr4_release_live_owner(void *device, void *owner) {
    {
        std::lock_guard<std::mutex> lock(g_mu);
        if (g_sh && (__bridge void *)g_sh->dev == device) {
            const auto it = g_sh->views.find(owner);
            if (it != g_sh->views.end()) {
                if (it->second->lastOut && g_sh->lastRealOut == it->second->lastOut)
                    g_sh->lastRealOut = nil;
                delete it->second;
                g_sh->views.erase(it);
            }
        }
    }
    if (fsr411_release_owner) fsr411_release_owner(device, owner);
}

static int fail_input_frame(const Fsr4LiveFrame *frame, const char *reason) {
    if (frame) fsr4_invalidate_live_owner(frame->device, frame->owner);
    return fail_reason(reason);
}

static bool surface_matches_texture(Fsr4InputSurface surface, id<MTLTexture> texture, id<MTLDevice> device) {
    return texture && texture.device == device && texture.width == surface.width && texture.height == surface.height;
}

int fsr4_encode_live_frame_v2(const Fsr4LiveFrameV2 *v2) {
    if (!fsr4_live_frame_v2_valid(v2)) return fail_reason("unsupported live frame ABI or truncated frame");
    const Fsr4LiveFrame *f = &v2->frame;
    if (!f->device || !f->commandBuffer || !f->color || !f->depth || !f->motion || !f->output)
        return fail_input_frame(f, "missing scaler textures or command buffer");
    // Both model versions share the panel; request it before routing to either encoder.
    fsr4_panel_request();
    id<MTLDevice> dev = (__bridge id<MTLDevice>)f->device;
    if (!surface_matches_texture(v2->geometry.color, (__bridge id<MTLTexture>)f->color, dev)
        || !surface_matches_texture(v2->geometry.depth, (__bridge id<MTLTexture>)f->depth, dev)
        || !surface_matches_texture(v2->geometry.motion, (__bridge id<MTLTexture>)f->motion, dev)
        || !surface_matches_texture(v2->geometry.output, (__bridge id<MTLTexture>)f->output, dev))
        return fail_input_frame(f, "input geometry/device disagrees with actual textures");
    if (f->exposure && [(__bridge id<MTLTexture>)f->exposure device] != dev)
        return fail_input_frame(f, "exposure texture belongs to a different device");
    const char *version = fsr4_live_model_version();
    if (!strcmp(version, "4.1.1")) {
        invalidate_legacy_input(f);
        if (!fsr411_encode_live_frame) return fail_reason("FSR 4.1.1 encoder is not linked");
        Fsr4LiveStatus before = {};
        fsr4_live_get_status(&before);
        if (fsr411_encode_live_frame(v2)) return 1;
        Fsr4LiveStatus after = {};
        fsr4_live_get_status(&after);
        // Preserve the new bridge's precise reason and avoid counting the
        // same rejected encode twice. Still fail explicitly if it published
        // no failure status of its own.
        if (!after.active && after.reason[0] && after.encodes != before.encodes) return 0;
        return fail_reason("FSR 4.1.1 rejected frame (see hook.log)");
    }
    if (strcmp(version, "4.0.2")) return fail_input_frame(f, "FSR4_VERSION must be 4.0.2 or 4.1.1");
    // Switching models drops the other model's session and history.
    if (fsr411_invalidate_owner) fsr411_invalidate_owner(f->device, f->owner);
    return encode_legacy_frame(f, v2->geometry, v2->inputContentProperties);
}

int fsr4_encode_live_frame(const Fsr4LiveFrame *frame) {
    if (!frame || !frame->color || !frame->depth || !frame->motion || !frame->output)
        return fail_input_frame(frame, "missing legacy frame textures");
    Fsr4LiveFrameV2 v2 = {};
    v2.structSize = sizeof(v2); v2.abiVersion = FSR4_LIVE_FRAME_ABI_V2;
    v2.frame = *frame;
    v2.inputContentProperties = -1;
    const auto capture = [](void *texture, uint32_t x, uint32_t y, Fsr4InputSurface &surface) {
        id<MTLTexture> tx = (__bridge id<MTLTexture>)texture;
        if (tx.width > UINT32_MAX || tx.height > UINT32_MAX) return false;
        surface = {(uint32_t)tx.width, (uint32_t)tx.height, x, y};
        return true;
    };
    if (!capture(frame->color, 0, 0, v2.geometry.color)
        || !capture(frame->depth, 0, 0, v2.geometry.depth)
        || !capture(frame->motion, frame->mvOffX, frame->mvOffY, v2.geometry.motion)
        || !capture(frame->output, 0, 0, v2.geometry.output))
        return fail_input_frame(frame, "legacy texture size exceeds input ABI");
    return fsr4_encode_live_frame_v2(&v2);
}

int fsr4_encode_live(void *device, void *commandBuffer,
                     void *color, void *depth, void *motion, void *output,
                     void *exposure,
                     uint32_t rw, uint32_t rh, uint32_t uw, uint32_t uh,
                     float jitterX, float jitterY,
                     float mvScaleX, float mvScaleY,
                     float preExposure, int reset, int depthReversed) {
    Fsr4LiveFrame f = {};
    f.device = device; f.commandBuffer = commandBuffer; f.owner = nullptr;
    f.color = color; f.depth = depth; f.motion = motion; f.output = output; f.exposure = exposure;
    f.rw = rw; f.rh = rh; f.uw = uw; f.uh = uh;
    f.jitterX = jitterX; f.jitterY = jitterY;
    f.mvScaleX = mvScaleX; f.mvScaleY = mvScaleY;
    f.preExposure = preExposure; f.reset = reset; f.depthReversed = depthReversed;
    f.jitteredMV = -1; f.outputResMV = -1; f.autoExposure = -1;
    return fsr4_encode_live_frame(&f);
}
