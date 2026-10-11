// fsr4mac - standalone native Metal port of the AMD FSR4 INT8 upscaler.
//
// Faithful port of third_party/fsr4 .../dx12/ffx_provider_fsr4_dx12.cpp:
// same resource set, constant buffer layout, dispatch order and sizes.
// Reads a testcase dir (see tools/gen_testcase + TESTDATA.md), runs N frames
// with persistent recurrent/history state, writes fNN_out.bin/.png.
//
//   fsr4mac <testcase_dir> <out_dir> [--dump-passes] [--builddir <dir>]

#include <Metal/Metal.h>
#include <Foundation/Foundation.h>
#include <mach-o/dyld.h>
#include <sys/stat.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <map>
#include "pngio.h"
#include "pass0_weights.h"
#include "i8mpp_host.h"

// ---------------------------------------------------------------------------
// constants (ported from ffx_provider_fsr4_dx12.cpp)
// ---------------------------------------------------------------------------
struct OptimizedConstants {          // MLSR_Optimized_Constants, 104 bytes
    float inv_size[2], scale[2], inv_scale[2], jitter[2], mv_scale[2];
    float tex_size[2], max_renderSize[2], fMotionVectorJitterCancellation[2];
    uint32_t width, height, reset, width_lr;
    uint32_t height_lr; float preExposure, previous_preExposure; uint32_t rcas_enabled;
    float rcas_sharpness, _pad;
};
struct AutoExposureSPDConstants {    // 32 bytes
    uint32_t mips, numWorkGroups, workGroupOffset[2];
    float invInputSize[2], preExposure, padding;
};
struct RcasConstants {               // cbRCAS, 32 bytes
    uint32_t config[4];
    float preExposure; uint32_t _pad[3];
};

static const size_t kScratchSize[3] = {20880256, 83232256, 332352256}; // GetScratchSize

// FsrRcasCon (fsr1/ffx_fsr1.h CPU path)
static void FsrRcasCon(uint32_t con[4], float sharpness) {
    sharpness = exp2f(-sharpness);
    union { uint16_t u; _Float16 f; } cvt; cvt.f = (_Float16)sharpness;
    uint32_t h = (uint32_t)cvt.u | ((uint32_t)cvt.u << 16); // ffxPackHalf2x16
    con[0] = __builtin_bit_cast(uint32_t, sharpness);
    con[1] = h;
    con[2] = 0; con[3] = 0;
}

// ffxSpdSetup (spd/ffx_spd.h CPU path)
static void ffxSpdSetup(uint32_t groups[2], uint32_t wgOffset[2],
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

// ---------------------------------------------------------------------------
// minimal JSON reader (contract-shaped meta.json only)
// ---------------------------------------------------------------------------
struct JScan {
    const char* s;
    JScan(const std::string& t) : s(t.c_str()) {}
    bool find(const char* key) {  // positions s at value start
        std::string pat = std::string("\"") + key + "\"";
        const char* p = strstr(s, pat.c_str());
        if (!p) return false;
        s = p + pat.size();
        while (*s == ' ' || *s == ':' || *s == '\t' || *s == '\n') s++;
        return true;
    }
    double num(const char* key, double dflt) {
        if (!find(key)) return dflt;
        return strtod(s, nullptr);
    }
    std::string str(const char* key) {
        if (!find(key) || *s != '"') return "";
        const char* e = strchr(++s, '"');
        return std::string(s, e - s);
    }
    bool boolean(const char* key, bool dflt) {
        if (!find(key)) return dflt;
        return !strncmp(s, "true", 4);
    }
};

struct FrameDesc {
    float jitter[2] = {0, 0};
    float mv_scale[2] = {1, 1};
    bool reset = false;
    float frame_time_ms = 16.6f;
    float camera_near = 0.1f, camera_far = 1000.f, fovy = 1.f;
};
struct Meta {
    std::string preset = "quality";
    uint32_t rw = 1280, rh = 720, uw = 1920, uh = 1080;
    int frames = 8;
    bool depth_inverted = false, low_res_mv = true, jittered_mv = false;
    std::string colorspace = "linear";
    bool auto_exposure = false;
    float pre_exposure = 1.f;
    bool rcas = false;
    float sharpness = 0.f;
    std::vector<FrameDesc> frame;
};

static Meta load_meta(const std::string& dir) {
    std::string path = dir + "/meta.json";
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path.c_str()); exit(1); }
    std::string txt; char buf[65536]; size_t n;
    while ((n = fread(buf, 1, sizeof buf, f)) > 0) txt.append(buf, n);
    fclose(f);
    JScan j(txt);
    Meta m;
    m.preset = j.str("preset");
    m.rw = (uint32_t)j.num("render_width", 1280);
    m.rh = (uint32_t)j.num("render_height", 720);
    m.uw = (uint32_t)j.num("upscale_width", 1920);
    m.uh = (uint32_t)j.num("upscale_height", 1080);
    m.frames = (int)j.num("frames", 8);
    m.depth_inverted = j.boolean("depth_inverted", false);
    m.low_res_mv = j.boolean("low_res_mv", true);
    m.jittered_mv = j.boolean("jittered_mv", false);
    m.colorspace = j.str("colorspace");
    m.auto_exposure = j.boolean("auto_exposure", false);
    m.pre_exposure = (float)j.num("pre_exposure", 1.0);
    m.rcas = j.boolean("rcas", false);
    m.sharpness = (float)j.num("sharpness", 0.0);
    // frame array
    const char* fa = strstr(txt.c_str(), "\"frame\"");
    if (fa) {
        const char* p = strchr(fa, '[');
        while (p && (p = strchr(p, '{')) != nullptr) {
            std::string obj(p, strchr(p, '}') - p + 1);
            JScan fj(obj);
            FrameDesc d;
            if (fj.find("jitter")) { d.jitter[0] = strtod(fj.s + 1, nullptr);
                const char* c = strchr(fj.s, ','); if (c) d.jitter[1] = strtod(c + 1, nullptr); }
            JScan fj2(obj);
            if (fj2.find("mv_scale")) { d.mv_scale[0] = strtod(fj2.s + 1, nullptr);
                const char* c = strchr(fj2.s, ','); if (c) d.mv_scale[1] = strtod(c + 1, nullptr); }
            JScan fj3(obj);
            d.reset = fj3.boolean("reset", false);
            JScan fj4(obj); d.frame_time_ms = (float)fj4.num("frame_time_ms", 16.6);
            JScan fj5(obj); d.camera_near = (float)fj5.num("camera_near", 0.1);
            JScan fj6(obj); d.camera_far = (float)fj6.num("camera_far", 1000.0);
            JScan fj7(obj); d.fovy = (float)fj7.num("fovy", 1.0);
            m.frame.push_back(d);
            p = strchr(p, '}');
        }
    }
    while ((int)m.frame.size() < m.frames) m.frame.push_back(FrameDesc{});
    return m;
}

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------
static void die(const char* msg) { fprintf(stderr, "fsr4mac: %s\n", msg); exit(1); }
static void die(const char* msg, const std::string& s) {
    fprintf(stderr, "fsr4mac: %s%s\n", msg, s.c_str()); exit(1);
}
static bool file_exists(const std::string& p) { struct stat st; return stat(p.c_str(), &st) == 0; }
static std::vector<uint8_t> read_file(const std::string& p) {
    FILE* f = fopen(p.c_str(), "rb");
    if (!f) die("cannot read ", p);
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    std::vector<uint8_t> v(n);
    if (fread(v.data(), 1, n, f) != (size_t)n) die("short read ", p);
    fclose(f); return v;
}
static void write_file(const std::string& p, const void* d, size_t n) {
    FILE* f = fopen(p.c_str(), "wb");
    if (!f) die("cannot write ", p);
    if (fwrite(d, 1, n, f) != n) die("write failed ", p);
    fclose(f);
}
static std::string exe_dir() {
    // resolve the repo root relative to this binary (tools/bin/fsr4mac → ../..)
    char buf[4096]; uint32_t sz = sizeof buf;
    if (_NSGetExecutablePath(buf, &sz) != 0) die("no exe path");
    std::string p = realpath(buf, nullptr);
    // tools/bin/fsr4mac → tools/bin → tools → root
    for (int i = 0; i < 3; i++) p = p.substr(0, p.rfind('/'));
    return p;
}

// ---------------------------------------------------------------------------
// Metal host
// ---------------------------------------------------------------------------
struct KernelInfo {
    id<MTLComputePipelineState> pso;
    MTLSize wg;
    std::vector<std::pair<NSUInteger, std::string>> buffers;   // idx, name
    std::vector<std::pair<NSUInteger, std::string>> textures;
    std::vector<std::pair<NSUInteger, std::string>> samplers;
};

// mirrors FnbParams in src/fnb_mpp.metal
struct FnbParams {
    uint32_t inV[4], outV[4], dims[4], w0s[4], w0d[4];
    uint32_t offs0[4], offs1[4], gemm[4];
    float scales[4];
    uint32_t misc[4];
};

// mirrors CnbParams in src/fnb_mpp.metal
struct CnbParams {
    uint32_t inV[4], outV[4], dims[4], w0s[4], w0d[4], w1s[4], w2s[4];
    uint32_t offs0[4], offs1[4];
    float scales[4];
};

struct Host {
    id<MTLDevice> dev;
    id<MTLCommandQueue> queue;
    std::map<std::string, KernelInfo> kernels;
    // native mpp::tensor_ops FNB kernels (fp16 only): pass idx -> params
    id<MTLComputePipelineState> mppPso[3] = {nil, nil, nil}; // 0,1=FNB32/64, 2=CNB
    id<MTLBuffer> mppWgt = nil;
    std::map<int, FnbParams> mpp;
    std::map<int, CnbParams> cnb;
    I8Mpp i8;              // native int8 tensor-ops passes (--i8mpp)
    bool useI8 = false;

    // persistent upscaler state
    id<MTLTexture> recurrent, history, historyReproj, rcasTemp;
    id<MTLTexture> autoExposure, defaultExposure, spdAtomicTex, lumaMip5, debugInfo;
    id<MTLBuffer> spdAtomicBuf;
    id<MTLBuffer> scratch, initializer;
    bool scratchPrivate = false;
    id<MTLBuffer> cbMain, cbSpd, cbRcas, cbPass0;
    id<MTLSamplerState> linearClamp;

    // per-frame inputs
    id<MTLTexture> inColor, inDepth, inMV, inExposure, output;

    float prevJitter[2] = {0, 0};
    bool prevPrevJitterValid = false; // previousJitter semantics below
    float prevPreExposure = 0.f;
    bool firstFrame = true;
};

static id<MTLTexture> make_tex(id<MTLDevice> dev, MTLPixelFormat fmt,
                               NSUInteger w, NSUInteger h) {
    MTLTextureDescriptor* d = [MTLTextureDescriptor
        texture2DDescriptorWithPixelFormat:fmt width:w height:h mipmapped:NO];
    d.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
    d.storageMode = MTLStorageModeShared;
    id<MTLTexture> t = [dev newTextureWithDescriptor:d];
    if (!t) die("texture alloc failed");
    return t;
}
static void tex_write(id<MTLTexture> t, const void* data, size_t rowBytes) {
    [t replaceRegion:MTLRegionMake2D(0, 0, t.width, t.height)
        mipmapLevel:0 withBytes:data bytesPerRow:rowBytes];
}
static void tex_read(id<MTLTexture> t, void* data, size_t rowBytes) {
    [t getBytes:data bytesPerRow:rowBytes
        fromRegion:MTLRegionMake2D(0, 0, t.width, t.height) mipmapLevel:0];
}

// name → resource resolution (port of the DX12 binding tables)
static id<MTLTexture> tex_for(Host& h, const std::string& n, bool rcas,
                              id<MTLTexture> exposureTex) {
    if (n == "r_input_color")            return h.inColor;
    if (n == "r_rcas_input")             return h.rcasTemp;
    if (n == "r_velocity")               return h.inMV;
    if (n == "r_depth")                  return h.inDepth;
    if (n == "r_input_exposure")         return exposureTex;
    if (n == "r_auto_exposure_texture")  return h.autoExposure;
    if (n == "r_recurrent_0")            return h.recurrent;
    if (n == "r_history_color")          return h.history;
    if (n == "r_reprojected_color")      return h.historyReproj;
    if (n == "r_debug_visualization")    return h.debugInfo;
    if (n == "r_result_color")           return h.output;
    if (n == "rw_output_color_for_rcas") return h.rcasTemp;
    if (n == "rw_mlsr_output_color")     return rcas ? h.rcasTemp : h.output;
    if (n == "rw_final_output_color")    return h.output;
    if (n == "rw_rcas_output")           return h.output;
    if (n == "rw_recurrent_0")           return h.recurrent;
    if (n == "rw_history_color")         return h.history;
    if (n == "rw_reprojected_color")     return h.historyReproj;
    if (n == "rw_auto_exposure_texture") return h.autoExposure;
    if (n == "rw_spd_global_atomic")     return h.spdAtomicTex;
    if (n == "rw_autoexp_mip_5")         return h.lumaMip5;
    if (n == "rw_debug_visualization")   return h.debugInfo;
    die("unknown texture binding ", n);
    return nil;
}
static id<MTLBuffer> buf_for(Host& h, const std::string& n) {
    if (n == "ScratchBuffer")                  return h.scratch;
    if (n == "InitializerBuffer")              return h.initializer;
    if (n == "MLSR_Optimized_Constants")       return h.cbMain;
    if (n == "AutoExposureSPDConstants")       return h.cbSpd;
    if (n == "cbRCAS")                         return h.cbRcas;
    if (n == "cbPass_Weights")                 return h.cbPass0;
    if (n == "rw_spd_global_atomic_atomic")    return h.spdAtomicBuf;
    die("unknown buffer binding ", n);
    return nil;
}

// Inter-pass sync strategy. D3D12 inserts UAV barriers between every pass;
// the Metal equivalents have different costs.
//   SYNC_BARRIER: memoryBarrierWithScope(Buffers) after every dispatch —
//      model passes only touch the scratch buffer; the texture scope is only
//      needed around pre/post which also write textures.
//   SYNC_ENCODER: end the compute encoder between passes and open a new one in
//      the same command buffer — default hazard tracking inserts only the
//      fences the resource dependencies actually need.
//   SYNC_NONE:   no sync (wrong results; upper-bound measurement only).
enum SyncMode { SYNC_BARRIER = 0, SYNC_ENCODER = 1, SYNC_NONE = 2 };

static void dispatch(Host& h, id<MTLComputeCommandEncoder> enc,
                     const char* name, uint32_t gx, uint32_t gy, uint32_t gz,
                     bool rcas, id<MTLTexture> exposureTex,
                     MTLBarrierScope postScope = 0) {
    auto it = h.kernels.find(name);
    if (it == h.kernels.end()) die("kernel missing ", name);
    KernelInfo& k = it->second;
    [enc setComputePipelineState:k.pso];
    for (auto& b : k.buffers)
        [enc setBuffer:buf_for(h, b.second) offset:0 atIndex:b.first];
    for (auto& t : k.textures)
        [enc setTexture:tex_for(h, t.second, rcas, exposureTex) atIndex:t.first];
    for (auto& s : k.samplers)
        [enc setSamplerState:h.linearClamp atIndex:s.first];
    [enc dispatchThreadgroups:MTLSizeMake(gx, gy, gz)
        threadsPerThreadgroup:k.wg];
    if (postScope)
        [enc memoryBarrierWithScope:postScope];
}

static uint32_t roundUpDiv(uint32_t a, uint32_t b) { return (a + b - 1) / b; }

// ---------------------------------------------------------------------------
static bool print_backend_info(const char *name) {
    @autoreleasepool {
        fsr4::Backend request;
        if (!fsr4::parse_backend(name, request)) {
            fprintf(stderr, "unknown backend: %s\n", name);
            return false;
        }
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        if (!dev) { fprintf(stderr, "no Metal device\n"); return false; }
        const auto caps = fsr4::device_caps(dev);
        const auto choice = fsr4::choose_backend(caps, request);
        NSDictionary *info = @{
            @"device": dev.name,
            @"apple_gpu_family": @(caps.appleFamily),
            @"metal4": @(caps.metal4),
            @"gpu_tensor_acceleration": @(caps.appleFamily >= 10),
            @"max_threads": @(caps.maxThreads),
            @"threadgroup_bytes": @(caps.threadgroupBytes),
            @"requested": [NSString stringWithUTF8String:fsr4::backend_name(request)],
            @"backend": [NSString stringWithUTF8String:fsr4::backend_name(choice.backend)],
            @"reason": [NSString stringWithUTF8String:choice.reason]
        };
        NSData *data = [NSJSONSerialization dataWithJSONObject:info options:0 error:nil];
        fwrite(data.bytes, 1, data.length, stdout);
        putchar('\n');
        return true;
    }
}

static void print_caps() {
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        if (!dev) { printf("no Metal device\n"); return; }
        printf("device: %s\n", [dev.name UTF8String]);
        printf("os: %s\n", [[NSProcessInfo processInfo].operatingSystemVersionString UTF8String]);
        printf("registryID: %llu\n", (unsigned long long)dev.registryID);
        printf("maxWorkingSet: %llu MB\n",
               (unsigned long long)(dev.recommendedMaxWorkingSetSize >> 20));
        const auto caps = fsr4::device_caps(dev);
        const auto choice = fsr4::choose_backend(caps);
        printf("automatic backend: %s (%s)\n", fsr4::backend_name(choice.backend), choice.reason);
        printf("GPU tensor acceleration: %s\n", caps.appleFamily >= 10 ? "yes" : "no (FP16 preferred)");
        for (int f = MTLGPUFamilyApple7; f <= MTLGPUFamilyApple10; f++)
            printf("gpuFamilyApple%d: %d\n", (int)(f - MTLGPUFamilyApple7 + 7),
                   (int)[dev supportsFamily:(MTLGPUFamily)f]);
        // Metal 4 tensor ops (mpp::tensor_ops / MTLTensor) need macOS 26+ and
        // hardware support; the paravirtual GPU does not expose them.
        printf("MTLTensorDescriptor class: %s\n",
               NSClassFromString(@"MTLTensorDescriptor") ? "present" : "absent");
        if (@available(macOS 26.0, *))
            printf("Metal 4 runtime: yes\n");
        else
            printf("Metal 4 runtime: no\n");

        // Functional probe: actually compile a minimal mpp::tensor_ops::matmul2d
        // kernel on this device. API presence alone does not imply the hardware
        // path exists (paravirtual GPUs compile it but may emulate).
        if (@available(macOS 26.0, *)) {
            NSString *probe = @R"MSL(
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;
kernel void probe(device int8_t *A [[buffer(0)]],
                  device int8_t *B [[buffer(1)]],
                  device int32_t *C [[buffer(2)]])
{
    auto a = tensor<device int8_t, extents<int32_t,16,32>, tensor_inline>(A, extents<int32_t,16,32>());
    auto b = tensor<device int8_t, extents<int32_t,16,16>, tensor_inline>(B, extents<int32_t,16,16>());
    auto c = tensor<device int32_t, extents<int32_t,16,32>, tensor_inline>(C, extents<int32_t,16,32>());
    constexpr auto d = matmul2d_descriptor(32, 16, 16, false, false, false,
                                           matmul2d_descriptor::mode::multiply);
    matmul2d<d, execution_simdgroups<4>> op;
    op.run(a, b, c);
}
)MSL";
            MTLCompileOptions *opt = [MTLCompileOptions new];
            opt.languageVersion = MTLLanguageVersion4_0;
            NSError *err = nil;
            id<MTLLibrary> lib = [dev newLibraryWithSource:probe options:opt error:&err];
            printf("mpp::tensor_ops matmul2d compile: %s\n",
                   lib ? "ok" : "FAILED");
            if (!lib && err)
                printf("  error: %s\n", [[err localizedDescription] UTF8String]);
            if (lib) {
                id<MTLFunction> fn = [lib newFunctionWithName:@"probe"];
                id<MTLComputePipelineState> ps =
                    [dev newComputePipelineStateWithFunction:fn error:&err];
                printf("matmul2d pipeline create: %s\n", ps ? "ok" : "FAILED");
                if (!ps && err)
                    printf("  error: %s\n", [[err localizedDescription] UTF8String]);
                if (ps && ps.maxTotalThreadsPerThreadgroup >= 128 &&
                    ps.staticThreadgroupMemoryLength <= dev.maxThreadgroupMemoryLength) {
                    id<MTLBuffer> a = [dev newBufferWithLength:32 * 16 options:MTLResourceStorageModeShared];
                    id<MTLBuffer> b = [dev newBufferWithLength:16 * 16 options:MTLResourceStorageModeShared];
                    id<MTLBuffer> c = [dev newBufferWithLength:32 * 16 * sizeof(int32_t) options:MTLResourceStorageModeShared];
                    if (a && b && c) {
                        memset(a.contents, 1, a.length); memset(b.contents, 2, b.length);
                        memset(c.contents, 0, c.length);
                        id<MTLCommandBuffer> cb = [[dev newCommandQueue] commandBuffer];
                        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                        [enc setComputePipelineState:ps];
                        [enc setBuffer:a offset:0 atIndex:0]; [enc setBuffer:b offset:0 atIndex:1];
                        [enc setBuffer:c offset:0 atIndex:2];
                        [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                        [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
                        bool exact = cb.status == MTLCommandBufferStatusCompleted;
                        for (int i = 0; i < 32 * 16; ++i)
                            exact = exact && ((const int32_t *)c.contents)[i] == 32;
                        printf("INT8 matmul dispatch + exact result: %s\n", exact ? "ok" : "FAILED");
                    } else printf("INT8 probe allocation FAILED\n");
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// --bench-tensorops: dispatch + time a matmul2d GEMM vs a simdgroup_matrix
// GEMM (and a scalar fp16 baseline) on the actual device. This is the decision
// point for which MMA path the conv operators should use.
// C[M,N] = A[M,K] * B[K,N], all half. mpp path: 32x32 tiles / K-chunk 64 per
// mtlgemm conventions. simdgroup path: 64x64 block per threadgroup, 4
// simdgroups at 32x32 each, K-chunk 16 staged in threadgroup memory.
// ---------------------------------------------------------------------------
static NSString *kGemmProbeSrc = @R"MSL(
#include <metal_stdlib>
#include <metal_tensor>
#include <metal_simdgroup_matrix>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;
struct P { uint M; uint N; uint K; };

kernel void gemm_mpp(device half *A [[buffer(0)]],
                     device half *B [[buffer(1)]],
                     device half *C [[buffer(2)]],
                     constant P  &p [[buffer(3)]],
                     uint2 tgid [[threadgroup_position_in_grid]])
{
    const uint K = p.K, N = p.N, M = p.M;
    auto tA = tensor<device half, dextents<int32_t,2>, tensor_inline>(A, dextents<int32_t,2>((int)K, (int)M));
    auto tB = tensor<device half, dextents<int32_t,2>, tensor_inline>(B, dextents<int32_t,2>((int)N, (int)K));
    auto tC = tensor<device half, dextents<int32_t,2>, tensor_inline>(C, dextents<int32_t,2>((int)N, (int)M));
    constexpr auto d = matmul2d_descriptor(32, 32, 64, false, false, false,
                                           matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<d, execution_simdgroups<4>> op;
    const uint n0 = tgid.x*32, m0 = tgid.y*32;
    for (uint k0 = 0; k0 < K; k0 += 64) {
        auto mA = tA.slice<64,32>((int)k0, (int)m0);
        auto mB = tB.slice<32,64>((int)n0, (int)k0);
        auto mC = tC.slice<32,32>((int)n0, (int)m0);
        op.run(mA, mB, mC);
    }
}

// 64x64 block per threadgroup, 4 simdgroups (2x2) at 32x32 each, K-chunk 16.
kernel void gemm_sg(device half *A [[buffer(0)]],
                    device half *B [[buffer(1)]],
                    device half *C [[buffer(2)]],
                    constant P  &p [[buffer(3)]],
                    uint2 tgid [[threadgroup_position_in_grid]],
                    uint  tid  [[thread_index_in_threadgroup]],
                    uint  sgid [[simdgroup_index_in_threadgroup]])
{
    const uint M = p.M, N = p.N, K = p.K;
    const uint bm = tgid.y * 64, bn = tgid.x * 64;
    const uint sm = (sgid / 2) * 32, sn = (sgid % 2) * 32;

    threadgroup half As[64*16];
    threadgroup half Bs[16*64];

    simdgroup_half8x8 acc[4][4];
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            acc[i][j] = simdgroup_half8x8(0);

    for (uint k0 = 0; k0 < K; k0 += 16) {
        for (uint i = tid; i < 64*16; i += 128) {
            uint r = i / 16, c = i % 16;
            As[i] = A[(bm + r) * K + k0 + c];
        }
        for (uint i = tid; i < 16*64; i += 128) {
            uint r = i / 64, c = i % 64;
            Bs[i] = B[(k0 + r) * N + bn + c];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint kc = 0; kc < 16; kc += 8) {
            simdgroup_half8x8 fa[4], fb[4];
            for (int r = 0; r < 4; r++)
                simdgroup_load(fa[r], &As[(sm + r*8) * 16 + kc], 16);
            for (int c = 0; c < 4; c++)
                simdgroup_load(fb[c], &Bs[kc * 64 + sn + c*8], 64);
            for (int r = 0; r < 4; r++)
                for (int c = 0; c < 4; c++)
                    simdgroup_multiply_accumulate(acc[r][c], fa[r], fb[c], acc[r][c]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (int r = 0; r < 4; r++)
        for (int c = 0; c < 4; c++)
            simdgroup_store(acc[r][c], &C[(bm + sm + r*8) * N + bn + sn + c*8], N);
}

// scalar fp16 baseline: one thread per output element
kernel void gemm_scalar(device half *A [[buffer(0)]],
                        device half *B [[buffer(1)]],
                        device half *C [[buffer(2)]],
                        constant P  &p [[buffer(3)]],
                        uint2 t [[thread_position_in_grid]])
{
    if (t.x >= p.N || t.y >= p.M) return;
    float acc = 0;
    for (uint k = 0; k < p.K; k++)
        acc += float(A[t.y * p.K + k]) * float(B[k * p.N + t.x]);
    C[t.y * p.N + t.x] = (half)acc;
}
)MSL";

static void bench_tensorops() {
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        if (!dev) { printf("no Metal device\n"); return; }
        printf("device: %s\n", [dev.name UTF8String]);

        // 1) correctness check on a small GEMM vs CPU reference
        const uint32_t MV = 64, NV = 64, KV = 64;
        std::vector<uint16_t> Ah(MV*KV), Bh(KV*NV);
        auto f2h = [](float v) { __fp16 h = (__fp16)v; uint16_t u; memcpy(&u,&h,sizeof(u)); return u; };
        auto h2f = [](uint16_t u) { __fp16 h; memcpy(&h,&u,sizeof(h)); return (float)h; };
        for (uint32_t i = 0; i < MV*KV; i++) Ah[i] = f2h(((i*2654435761u)>>16 & 255)/255.0f * 0.5f);
        for (uint32_t i = 0; i < KV*NV; i++) Bh[i] = f2h(((i*2246822519u)>>16 & 255)/255.0f * 0.5f);
        std::vector<float> ref(MV*NV, 0.f);
        for (uint32_t m = 0; m < MV; m++)
            for (uint32_t n = 0; n < NV; n++)
                for (uint32_t k = 0; k < KV; k++)
                    ref[m*NV+n] += h2f(Ah[m*KV+k]) * h2f(Bh[k*NV+n]);

        MTLCompileOptions *opt = [MTLCompileOptions new];
        if (@available(macOS 26.0, *)) opt.languageVersion = MTLLanguageVersion4_0;
        NSError *err = nil;
        id<MTLLibrary> lib = [dev newLibraryWithSource:kGemmProbeSrc options:opt error:&err];
        if (!lib) {
            printf("probe compile FAILED: %s\n", err ? [[err localizedDescription] UTF8String] : "?");
            return;
        }
        id<MTLComputePipelineState> psMpp = nil, psSg = nil, psSc = nil;
        psMpp = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"gemm_mpp"] error:&err];
        if (!psMpp) printf("gemm_mpp pipeline FAILED: %s\n", err ? [[err localizedDescription] UTF8String] : "?");
        psSg = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"gemm_sg"] error:&err];
        if (!psSg) printf("gemm_sg pipeline FAILED: %s\n", err ? [[err localizedDescription] UTF8String] : "?");
        psSc = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"gemm_scalar"] error:&err];
        id<MTLCommandQueue> q = [dev newCommandQueue];

        struct Dim { uint32_t M, N, K; };
        auto runGemm = ^(id<MTLComputePipelineState> ps, bool mpp,
                         id<MTLBuffer> A, id<MTLBuffer> B, id<MTLBuffer> C,
                         uint32_t M, uint32_t N, uint32_t K, double *gpuMs) {
            Dim d{M,N,K};
            id<MTLBuffer> pb = [dev newBufferWithBytes:&d length:sizeof(d) options:MTLResourceStorageModeShared];
            id<MTLCommandBuffer> cb = [q commandBuffer];
            id<MTLComputeCommandEncoder> ce = [cb computeCommandEncoder];
            [ce setComputePipelineState:ps];
            [ce setBuffer:A offset:0 atIndex:0];
            [ce setBuffer:B offset:0 atIndex:1];
            [ce setBuffer:C offset:0 atIndex:2];
            [ce setBuffer:pb offset:0 atIndex:3];
            [ce dispatchThreadgroups:MTLSizeMake(N/(mpp?32:64), M/(mpp?32:64), 1)
               threadsPerThreadgroup:MTLSizeMake(128,1,1)];
            [ce endEncoding];
            [cb commit]; [cb waitUntilCompleted];
            *gpuMs = ([cb GPUEndTime] - [cb GPUStartTime]) * 1000.0;
        };

        // verify: run each kernel on the small case, compare vs CPU ref
        {
            id<MTLBuffer> A = [dev newBufferWithBytes:Ah.data() length:Ah.size()*2 options:MTLResourceStorageModeShared];
            id<MTLBuffer> B = [dev newBufferWithBytes:Bh.data() length:Bh.size()*2 options:MTLResourceStorageModeShared];
            id<MTLBuffer> C = [dev newBufferWithLength:MV*NV*2 options:MTLResourceStorageModeShared];
            memset(C.contents, 0, MV*NV*2);
            const char* names[3] = {"matmul2d", "simdgroup_matrix", "scalar"};
            id<MTLComputePipelineState> pss[3] = {psMpp, psSg, psSc};
            for (int t = 0; t < 3; t++) {
                if (!pss[t]) { printf("verify %-17s : skipped (no pipeline)\n", names[t]); continue; }
                double ms;
                if (t == 2) {
                    // scalar: dispatch per-element grid
                    Dim d{MV,NV,KV};
                    id<MTLBuffer> pb = [dev newBufferWithBytes:&d length:sizeof(d) options:MTLResourceStorageModeShared];
                    id<MTLCommandBuffer> cb = [q commandBuffer];
                    id<MTLComputeCommandEncoder> ce = [cb computeCommandEncoder];
                    [ce setComputePipelineState:psSc];
                    [ce setBuffer:A offset:0 atIndex:0];
                    [ce setBuffer:B offset:0 atIndex:1];
                    [ce setBuffer:C offset:0 atIndex:2];
                    [ce setBuffer:pb offset:0 atIndex:3];
                    [ce dispatchThreads:MTLSizeMake(NV, MV, 1)
                    threadsPerThreadgroup:MTLSizeMake(32,4,1)];
                    [ce endEncoding]; [cb commit]; [cb waitUntilCompleted];
                } else {
                    runGemm(pss[t], t == 0, A, B, C, MV, NV, KV, &ms);
                }
                float maxErr = 0;
                uint16_t *cp = (uint16_t*)C.contents;
                for (uint32_t i = 0; i < MV*NV; i++)
                    maxErr = fmaxf(maxErr, fabsf(h2f(cp[i]) - ref[i]));
                printf("verify %-17s : max abs err %.5f %s\n", names[t], maxErr,
                       maxErr < 0.05f ? "OK" : "MISMATCH");
                memset(C.contents, 0, MV*NV*2);
            }
        }

        // perf: 1024x1024x4096, 20 iterations per kernel
        const uint32_t M = 1024, N = 1024, K = 4096;
        const int iters = 20;
        id<MTLBuffer> A = [dev newBufferWithLength:(size_t)M*K*2 options:MTLResourceStorageModeShared];
        id<MTLBuffer> B = [dev newBufferWithLength:(size_t)K*N*2 options:MTLResourceStorageModeShared];
        id<MTLBuffer> C = [dev newBufferWithLength:(size_t)M*N*2 options:MTLResourceStorageModeShared];
        { // fill with small pattern
            __fp16 *ap = (__fp16*)A.contents, *bp = (__fp16*)B.contents;
            for (size_t i = 0; i < (size_t)M*K; i++) ap[i] = (__fp16)(((i*2654435761u)>>16 & 255)/2048.0f);
            for (size_t i = 0; i < (size_t)K*N; i++) bp[i] = (__fp16)(((i*2246822519u)>>16 & 255)/2048.0f);
        }
        const double flops = 2.0 * M * N * K;
        struct { const char *name; id<MTLComputePipelineState> ps; bool mpp; } cases[2] = {
            {"matmul2d", psMpp, true}, {"simdgroup_matrix", psSg, false}};
        for (int t = 0; t < 2; t++) {
            if (!cases[t].ps) { printf("perf %-17s : skipped\n", cases[t].name); continue; }
            // warmup
            double ms;
            runGemm(cases[t].ps, cases[t].mpp, A, B, C, M, N, K, &ms);
            double totalMs = 0;
            id<MTLCommandBuffer> cb = [q commandBuffer];
            for (int it = 0; it < iters; it++) {
                Dim d{M,N,K};
                id<MTLBuffer> pb = [dev newBufferWithBytes:&d length:sizeof(d) options:MTLResourceStorageModeShared];
                id<MTLComputeCommandEncoder> ce = [cb computeCommandEncoder];
                [ce setComputePipelineState:cases[t].ps];
                [ce setBuffer:A offset:0 atIndex:0];
                [ce setBuffer:B offset:0 atIndex:1];
                [ce setBuffer:C offset:0 atIndex:2];
                [ce setBuffer:pb offset:0 atIndex:3];
                [ce dispatchThreadgroups:MTLSizeMake(N/(cases[t].mpp?32:64), M/(cases[t].mpp?32:64), 1)
                   threadsPerThreadgroup:MTLSizeMake(128,1,1)];
                [ce endEncoding];
            }
            [cb commit]; [cb waitUntilCompleted];
            totalMs = ([cb GPUEndTime] - [cb GPUStartTime]) * 1000.0 / iters;
            printf("perf %-17s : %8.3f ms/GEMM  %8.2f GFLOP/s\n", cases[t].name,
                   totalMs, flops / (totalMs * 1e6));
        }
        printf("note: matmul2d numbers >> simdgroup => neural accelerator path; "
               "parity/similar => likely emulated on shader cores.\n");
    }
}

int fsr411_main(int argc, char **argv);

int main(int argc, char** argv) {
    // Select the model before parsing legacy backend options. A version
    // request can never silently run a different network under its name.
    const char *versionEnv = getenv("FSR4_VERSION");
    std::string version = versionEnv && *versionEnv ? versionEnv : "4.1.1";
    std::vector<char *> versionArgs = {argv[0]};
    bool versionSeen = false;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--fsr-version")) {
            if (versionSeen || ++i >= argc) {
                fprintf(stderr, "--fsr-version requires one version: 4.0.2 or 4.1.1\n"); return 2;
            }
            version = argv[i]; versionSeen = true;
        } else versionArgs.push_back(argv[i]);
    }
    argc = (int)versionArgs.size(); versionArgs.push_back(nullptr); argv = versionArgs.data();
    if (version != "4.0.2" && version != "4.1.1") {
        fprintf(stderr, "--fsr-version / FSR4_VERSION must be 4.0.2 or 4.1.1\n"); return 2;
    }
    // Hardware probes are independent of the selected model version.
    if (argc >= 2 && !strcmp(argv[1], "--caps")) { print_caps(); return 0; }
    if (argc >= 2 && !strcmp(argv[1], "--bench-tensorops")) { bench_tensorops(); return 0; }
    if (version == "4.1.1") return fsr411_main(argc, argv);
    if (argc >= 2 && !strcmp(argv[1], "--backend-info"))
        return print_backend_info(argc > 2 ? argv[2] : "auto") ? 0 : 1;
    if (argc < 3) {
        fprintf(stderr, "usage: fsr4mac --fsr-version 4.0.2 <testcase_dir> <out_dir> [--auto|--int8|--fp16|--i8mpp] [--mpp] [--dump-passes] [--bench N] [--reps R] [--sync barrier|encoder|none] [--builddir dir]\n"
                        "       fsr4mac --caps\n"
                        "       fsr4mac --fsr-version 4.0.2 --backend-info [auto|f16|int8|i8mpp]\n"
                        "       fsr4mac --bench-tensorops\n");
        return 1;
    }
    std::string tcdir = argv[1], outdir = argv[2];
    bool dumpPasses = false;
    fsr4::Backend requestedBackend = fsr4::Backend::Auto;
    bool useMpp = false;   // matmul2d FNB (measured slower on M5)
    bool useCnb = false;   // lane-parallel CNB16 (measured slower on M5)
    int bench = 0;
    int benchReps = 1;
    SyncMode syncMode = SYNC_BARRIER;
    std::string builddir = "build";
    for (int i = 3; i < argc; i++) {
        if (!strcmp(argv[i], "--dump-passes")) dumpPasses = true;
        else if (!strcmp(argv[i], "--auto")) requestedBackend = fsr4::Backend::Auto;
        else if (!strcmp(argv[i], "--fp16")) requestedBackend = fsr4::Backend::Fp16;
        else if (!strcmp(argv[i], "--int8")) requestedBackend = fsr4::Backend::Int8;
        else if (!strcmp(argv[i], "--bench") && i + 1 < argc) bench = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--reps") && i + 1 < argc) benchReps = atoi(argv[++i]) > 0 ? atoi(argv[i]) : 1;
        else if (!strcmp(argv[i], "--mpp")) { useMpp = true; requestedBackend = fsr4::Backend::Fp16; }
        else if (!strcmp(argv[i], "--cnb")) { useCnb = true; requestedBackend = fsr4::Backend::Fp16; }
        else if (!strcmp(argv[i], "--i8mpp")) requestedBackend = fsr4::Backend::I8Mpp;
        else if (!strcmp(argv[i], "--sync") && i + 1 < argc) {
            const char* s = argv[++i];
            if (!strcmp(s, "barrier")) syncMode = SYNC_BARRIER;
            else if (!strcmp(s, "encoder")) syncMode = SYNC_ENCODER;
            else if (!strcmp(s, "none")) syncMode = SYNC_NONE;
            else die("unknown --sync mode ", s);
        }
        else if (!strcmp(argv[i], "--builddir") && i + 1 < argc) builddir = argv[++i];
        else die("unknown arg ", argv[i]);
    }
    std::string root = exe_dir();
    @autoreleasepool {
        Host h;
        h.dev = MTLCreateSystemDefaultDevice();
        if (!h.dev) die("no Metal device");
        h.queue = [h.dev newCommandQueue];
        const auto choice = fsr4::choose_backend(fsr4::device_caps(h.dev), requestedBackend);
        bool fp16 = choice.backend == fsr4::Backend::Fp16;
        bool useI8mpp = choice.backend == fsr4::Backend::I8Mpp;
        fprintf(stderr, "device: %s\nbackend: %s (%s)\n", h.dev.name.UTF8String,
                fsr4::backend_name(choice.backend), choice.reason);
        Meta m = load_meta(tcdir);
        // provider: upscaleWidth/Height = AlignUp(upscaleSize, 8) for the internal
        // textures, constants and dispatch sizes; the output texture keeps its size
        const uint32_t uwA = (m.uw + 7) & ~7u, uhA = (m.uh + 7) & ~7u;
        fprintf(stderr, "fsr4mac: preset=%s render=%ux%u upscale=%ux%u frames=%d ae=%d rcas=%d\n",
                m.preset.c_str(), m.rw, m.rh, m.uw, m.uh, m.frames,
                (int)m.auto_exposure, (int)m.rcas);

        // --- colorspace / resolution mapping (shader_selector.cpp) ---
        int colorspace = 0;                              // linear
        if (m.colorspace == "srgb") colorspace = 2;
        else if (m.colorspace == "pq") colorspace = 3;
        else if (m.colorspace != "linear") die("bad colorspace ", m.colorspace);

        int resIdx = 0;                                  // MaxResolution
        if (m.uw > 1920 || m.uh > 1080) resIdx = 1;      // 2160
        if (m.uw > 3840 || m.uh > 2160) resIdx = 2;      // 4320
        const char* resStr[3] = {"1080", "2160", "4320"};

        bool dbg = false;                                // debug view off by default

        // --- i8mpp: manifest first (the post variant needs the ext-CNB offset) ---
        long postExtOff = -1;
        std::string i8gdir;
        auto prepareI8 = [&]() -> bool {
            i8gdir = builddir + "/gen_i8mpp_" + m.preset + "_" + resStr[resIdx];
            auto mtime = [](const std::string& f) { struct stat st; return stat(f.c_str(), &st) == 0 ? st.st_mtime : (time_t)0; };
            if (!file_exists(i8gdir + "/i8mpp_manifest.json")
                || mtime(i8gdir + "/i8mpp_manifest.json") < mtime(root + "/tools/gen_i8mpp.py")) {
                char cmd[2048];
                snprintf(cmd, sizeof cmd,
                    "python3 '%s/tools/gen_i8mpp.py' --root '%s' --preset %s --res %s --outdir '%s'",
                    root.c_str(), root.c_str(), m.preset.c_str(), resStr[resIdx], i8gdir.c_str());
                if (system(cmd) != 0) { h.i8.error = "gen_i8mpp failed"; return false; }
            }
            std::vector<uint8_t> mjv;
            if (!I8Mpp::read_all(i8gdir + "/i8mpp_manifest.json", mjv)) {
                h.i8.error = "cannot read i8mpp manifest"; return false;
            }
            std::string mj(mjv.begin(), mjv.end());
            size_t k = mj.find("\"post_ext_ct2d_off\"");
            if (k == std::string::npos) { h.i8.error = "manifest lacks post_ext_ct2d_off"; return false; }
            postExtOff = strtol(mj.c_str() + mj.find(':', k) + 1, nullptr, 10);
            return h.i8.load(h.dev, i8gdir, root + "/src/i8mpp.metal");
        };
        if (useI8mpp && !prepareI8()) {
            fprintf(stderr, "native INT8 unavailable (%s), falling back to FP16\n", h.i8.error.c_str());
            useI8mpp = false;
            fp16 = true;
            postExtOff = -1;
        }
        // --- compile shader variant (cached) ---
        {
            char cmd[4096];
            char extopt[64] = "";
            if (postExtOff >= 0) snprintf(extopt, sizeof extopt, " --post-ext-ct2d %ld", postExtOff);
            snprintf(cmd, sizeof cmd,
                "python3 '%s/tools/compile_variant.py' --root '%s' --preset %s "
                "--res %s --res-idx %d --depth-inverted %d --low-res-mv %d "
                "--jittered-mv %d --colorspace %d --auto-exposure %d "
                "--debug-view %d --rcas %d --model %s --builddir '%s'%s",
                root.c_str(), root.c_str(), m.preset.c_str(), resStr[resIdx], resIdx,
                (int)m.depth_inverted, (int)m.low_res_mv, (int)m.jittered_mv,
                colorspace, (int)m.auto_exposure, (int)dbg, (int)m.rcas,
                fp16 ? "f16" : "i8", builddir.c_str(), extopt);
            int rc = system(cmd);
            if (rc != 0) die("shader compile failed");
        }
        char key[512];
        snprintf(key, sizeof key, "fsr4_%s_%s_di%d_lm%d_jm%d_cs%d_ae%d_db%d%s%s",
                 m.preset.c_str(), resStr[resIdx], (int)m.depth_inverted,
                 (int)m.low_res_mv, (int)m.jittered_mv, colorspace,
                 (int)m.auto_exposure, (int)dbg, fp16 ? "_f16" : "", postExtOff >= 0 ? "_pt" : "");
        std::string vdir = builddir + "/" + key;
        if (!file_exists(vdir + "/shaders.metallib"))
            die("metallib missing at ", vdir);

        NSError* err = nil;
        NSString* libPath = [NSString stringWithUTF8String:(vdir + "/shaders.metallib").c_str()];
        id<MTLLibrary> lib = [h.dev newLibraryWithURL:[NSURL fileURLWithPath:libPath] error:&err];
        if (!lib) die([[err localizedDescription] UTF8String]);

        // parse bindings.tsv
        {
            FILE* bf = fopen((vdir + "/bindings.tsv").c_str(), "r");
            if (!bf) die("no bindings.tsv in ", vdir);
            char line[512];
            std::map<std::string, MTLSize> wg;
            std::string cur;
            while (fgets(line, sizeof line, bf)) {
                char kn[128]; int x, y, z, idx; char rn[128];
                if (sscanf(line, "K\t%127s\t%d\t%d\t%d", kn, &x, &y, &z) == 4) {
                    if (x <= 0 || y <= 0 || z <= 0) die("invalid threadgroup dimensions for ", kn);
                    cur = kn;
                    KernelInfo ki;
                    id<MTLFunction> fn = [lib newFunctionWithName:
                        [NSString stringWithUTF8String:kn]];
                    if (!fn) die("no MTLFunction ", kn);
                    ki.pso = [h.dev newComputePipelineStateWithFunction:fn error:&err];
                    if (!ki.pso) die("PSO failed ", kn);
                    ki.wg = MTLSizeMake(x, y, z);
                    if ((NSUInteger)x * y * z > ki.pso.maxTotalThreadsPerThreadgroup ||
                        ki.pso.staticThreadgroupMemoryLength > h.dev.maxThreadgroupMemoryLength)
                        die("GPU threadgroup limits exceeded by ", kn);
                    h.kernels[cur] = ki;
                } else if (sscanf(line, "B\t%127s\t%d\t%127s", kn, &idx, rn) == 3)
                    h.kernels[kn].buffers.push_back({(NSUInteger)idx, rn});
                else if (sscanf(line, "T\t%127s\t%d\t%127s", kn, &idx, rn) == 3)
                    h.kernels[kn].textures.push_back({(NSUInteger)idx, rn});
                else if (sscanf(line, "S\t%127s\t%d\t%127s", kn, &idx, rn) == 3)
                    h.kernels[kn].samplers.push_back({(NSUInteger)idx, rn});
            }
            fclose(bf);
        }

        // --- persistent resources (getResourceDescriptions) ---
        // zero-initialised: shared-memory contents are undefined otherwise
        // and reset-frame reads of padding regions must be deterministic.
        h.recurrent      = make_tex(h.dev, MTLPixelFormatRGBA8Unorm, uwA, uhA);
        h.history        = make_tex(h.dev, MTLPixelFormatRGBA16Float, uwA, uhA);
        h.historyReproj  = make_tex(h.dev, MTLPixelFormatRGBA16Float, uwA, uhA);
        {
            std::vector<uint8_t> zeros((size_t)uwA * uhA * 8, 0);
            tex_write(h.recurrent, zeros.data(), uwA * 4);
            tex_write(h.history, zeros.data(), uwA * 8);
            tex_write(h.historyReproj, zeros.data(), uwA * 8);
        }
        if (m.rcas)
            h.rcasTemp   = make_tex(h.dev, MTLPixelFormatRGBA16Float, uwA, uhA);
        else             // dummy so bindings never nil
            h.rcasTemp   = make_tex(h.dev, MTLPixelFormatRGBA16Float, 4, 4);
        if (m.auto_exposure) {
            h.autoExposure = make_tex(h.dev, MTLPixelFormatR32Float, 2, 1);
            // Match the live encoder: SPD can read the previous exposure on
            // its first dispatch, so a fresh texture must have defined state.
            const float zeroExposure[2] = {0.f, 0.f};
            tex_write(h.autoExposure, zeroExposure, sizeof(zeroExposure));
            h.lumaMip5     = make_tex(h.dev, MTLPixelFormatR32Float,
                                      roundUpDiv(m.rw, 16), roundUpDiv(m.rh, 16));
            // spdAtomicCounter: buffer-backed 1x1 R32Uint (globallycoherent atomic)
            h.spdAtomicBuf = [h.dev newBufferWithLength:4096
                                options:MTLResourceStorageModeShared];
            memset(h.spdAtomicBuf.contents, 0, 4);
            MTLTextureDescriptor* sd = [MTLTextureDescriptor
                texture2DDescriptorWithPixelFormat:MTLPixelFormatR32Uint
                width:1 height:1 mipmapped:NO];
            sd.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
            sd.storageMode = MTLStorageModeShared;
            h.spdAtomicTex = [h.spdAtomicBuf newTextureWithDescriptor:sd
                                offset:0 bytesPerRow:16];
            if (!h.spdAtomicTex) die("spd atomic tex failed");
        }
        {
            float de[2] = {1.f, 1.f};
            h.defaultExposure = make_tex(h.dev, MTLPixelFormatRG32Float, 1, 1);
            tex_write(h.defaultExposure, de, 8);
        }
        if (dbg) h.debugInfo = make_tex(h.dev, MTLPixelFormatR16Unorm, uwA, uhA);
        else     h.debugInfo = make_tex(h.dev, MTLPixelFormatR16Unorm, 4, 4);

        const char* scratchMode = getenv("FSR4_SCRATCH_PRIVATE");
        h.scratchPrivate = scratchMode && *scratchMode == '1';
        // i8mpp appends the pass-13 model-parameter tensor after the tensors
        size_t scratchBytes = kScratchSize[resIdx] * (fp16 ? 2 : 1);
        if (useI8mpp) scratchBytes = I8Mpp::scratch_bytes(resIdx);
        if (h.scratchPrivate) {
            h.scratch = [h.dev newBufferWithLength:scratchBytes
                            options:MTLResourceStorageModePrivate];
            // zero via a staging blit
            id<MTLBuffer> z = [h.dev newBufferWithLength:scratchBytes
                                options:MTLResourceStorageModeShared];
            memset(z.contents, 0, z.length);
            id<MTLCommandBuffer> zb = [h.queue commandBuffer];
            id<MTLBlitCommandEncoder> be = [zb blitCommandEncoder];
            [be copyFromBuffer:z sourceOffset:0 toBuffer:h.scratch
                destinationOffset:0 size:z.length];
            [be endEncoding]; [zb commit]; [zb waitUntilCompleted];
        } else {
            h.scratch = [h.dev newBufferWithLength:scratchBytes
                            options:MTLResourceStorageModeShared];
            memset(h.scratch.contents, 0, scratchBytes);
        }

        // fp16 weights live in the generated dir (gen_f16.py); int8 reads
        // the original initializers.bin.
        std::string initPath = fp16
            ? builddir + "/gen_f16_" + m.preset + "_" + resStr[resIdx]
                         + "/initializers_f16.bin"
            : root + "/third_party/fsr4/internal/shaders/fsr4_model_v07_i8_"
                     + m.preset + "/initializers.bin";
        auto initData = read_file(initPath);
        h.initializer = [h.dev newBufferWithBytes:initData.data()
                            length:initData.size()
                            options:MTLResourceStorageModeShared];

        // --- native mpp::tensor_ops FNB kernels (fp16 path, Metal 4) ---
        // src/fnb_mpp.metal + gen_f16.py's mpp sidecar (transposed pw weights
        // + per-pass tensor geometry). --no-mpp disables.
        if (fp16 && (useMpp || useCnb)) {
            std::string gdir = builddir + "/gen_f16_" + m.preset + "_"
                             + resStr[resIdx];
            std::string mlib = builddir + "/fnb_mpp.metallib";
            if (!file_exists(mlib)) {
                char cmd[2048];
                snprintf(cmd, sizeof cmd,
                    "$(xcrun -f metal) -std=metal4.0 -O3 -c "
                    "'%s/src/fnb_mpp.metal' -o '%s/fnb_mpp.air' && "
                    "$(xcrun -f metallib) '%s/fnb_mpp.air' -o '%s'",
                    root.c_str(), builddir.c_str(), builddir.c_str(),
                    mlib.c_str());
                if (system(cmd) != 0)
                    fprintf(stderr, "mpp: fnb_mpp.metal compile failed\n");
            }
            if (file_exists(mlib) &&
                file_exists(gdir + "/mpp_manifest.json") &&
                file_exists(gdir + "/mpp_weights.bin")) {
                NSError* merr = nil;
                id<MTLLibrary> ml = [h.dev newLibraryWithURL:
                    [NSURL fileURLWithPath:
                        [NSString stringWithUTF8String:mlib.c_str()]]
                    error:&merr];
                if (ml) {
                    const char* names[3] = {"fnb32_mpp", "fnb64_mpp",
                                            "cnb_mpp"};
                    for (int s = 0; s < 3; s++) {
                        id<MTLFunction> fn = [ml newFunctionWithName:
                            [NSString stringWithUTF8String:names[s]]];
                        if (fn)
                            h.mppPso[s] = [h.dev
                                newComputePipelineStateWithFunction:fn
                                error:&merr];
                    }
                }
                if (h.mppPso[0] && h.mppPso[1] && h.mppPso[2]) {
                    if (!useMpp && !useCnb) { /* unreachable */ }
                    auto wb = read_file(gdir + "/mpp_weights.bin");
                    h.mppWgt = [h.dev newBufferWithBytes:wb.data()
                                length:wb.size()
                                options:MTLResourceStorageModeShared];
                    NSData* jd = [NSData dataWithContentsOfFile:
                        [NSString stringWithUTF8String:
                            (gdir + "/mpp_manifest.json").c_str()]];
                    NSDictionary* mj = [NSJSONSerialization
                        JSONObjectWithData:jd options:0 error:&merr];
                    NSDictionary* ps = mj[@"passes"];
                    for (NSString* k in ps) {
                        NSDictionary* e = ps[k];
                        auto u4 = [](NSArray* a, uint32_t* d) {
                            for (int i = 0; i < 4 && i < (int)a.count; i++)
                                d[i] = [a[i] unsignedIntValue];
                        };
                        if ([e[@"kind"] isEqualToString:@"cnb"]) {
                            if (!useCnb) continue;
                            CnbParams c = {};
                            u4(e[@"inT"][@"strides"], c.inV);
                            c.inV[3] = [e[@"inT"][@"off"] unsignedIntValue];
                            u4(e[@"outT"][@"strides"], c.outV);
                            c.outV[3] = [e[@"outT"][@"off"] unsignedIntValue];
                            u4(e[@"inT"][@"dims"], c.dims);
                            u4(e[@"w0s"], c.w0s);
                            u4(e[@"w0d"], c.w0d);
                            u4(e[@"w1s"], c.w1s);
                            u4(e[@"w2s"], c.w2s);
                            c.offs0[0] = [e[@"w0"] unsignedIntValue];
                            c.offs0[1] = [e[@"b0"] unsignedIntValue];
                            c.offs0[2] = [e[@"w1"] unsignedIntValue];
                            c.offs0[3] = [e[@"b1"] unsignedIntValue];
                            c.offs1[0] = [e[@"w2"] unsignedIntValue];
                            c.offs1[1] = [e[@"b2"] unsignedIntValue];
                            for (int i = 0; i < 3; i++)
                                c.scales[i] = [e[@"scales"][i] floatValue];
                            h.cnb[k.intValue] = c;
                            continue;
                        }
                        if (!useMpp) continue;
                        FnbParams pr = {};
                        u4(e[@"inT"][@"strides"], pr.inV);
                        pr.inV[3] = [e[@"inT"][@"off"] unsignedIntValue];
                        u4(e[@"outT"][@"strides"], pr.outV);
                        pr.outV[3] = [e[@"outT"][@"off"] unsignedIntValue];
                        u4(e[@"inT"][@"dims"], pr.dims);
                        u4(e[@"w0s"], pr.w0s);
                        u4(e[@"w0d"], pr.w0d);
                        pr.offs0[0] = [e[@"w0"] unsignedIntValue];
                        pr.offs0[1] = [e[@"b0"] unsignedIntValue];
                        pr.offs0[2] = [e[@"w1"] unsignedIntValue];
                        pr.offs0[3] = [e[@"b1"] unsignedIntValue];
                        pr.offs1[0] = [e[@"w2"] unsignedIntValue];
                        pr.offs1[1] = [e[@"b2"] unsignedIntValue];
                        pr.gemm[0] = [e[@"K1"] unsignedIntValue];
                        pr.gemm[1] = [e[@"N1"] unsignedIntValue];
                        pr.gemm[2] = [e[@"K2"] unsignedIntValue];
                        pr.gemm[3] = [e[@"N2"] unsignedIntValue];
                        for (int i = 0; i < 4; i++)
                            pr.scales[i] = [e[@"scales"][i] floatValue];
                        pr.misc[0] = [e[@"ciPG"] unsignedIntValue];
                        pr.misc[1] = [e[@"fPG"] unsignedIntValue];
                        pr.misc[2] = [e[@"ng"] unsignedIntValue];
                        h.mpp[k.intValue] = pr;
                    }
                    fprintf(stderr, "native kernels: %zu fnb_mpp + %zu cnb passes\n",
                            h.mpp.size(), h.cnb.size());
                } else {
                    fprintf(stderr, "native kernels unavailable (Metal 4 "
                            "required), using generated fp16 passes\n");
                }
            }
        }

        if (useI8mpp) {
            h.i8.set_region(uwA, uhA);
            h.useI8 = true;
            fprintf(stderr, "native int8 tensor-ops kernels: %zu passes\n", h.i8.passes.size());
        }
        h.cbMain  = [h.dev newBufferWithLength:256 options:MTLResourceStorageModeShared];
        h.cbSpd   = [h.dev newBufferWithLength:256 options:MTLResourceStorageModeShared];
        h.cbRcas  = [h.dev newBufferWithLength:256 options:MTLResourceStorageModeShared];
        {
            const uint32_t* w = kPass0Weights_quality;
            if (m.preset == "native")      w = kPass0Weights_native;
            else if (m.preset == "balanced")    w = kPass0Weights_balanced;
            else if (m.preset == "performance") w = kPass0Weights_performance;
            else if (m.preset == "ultraperf")   w = kPass0Weights_ultraperf;
            else if (m.preset == "drs")         w = kPass0Weights_drs;
            h.cbPass0 = [h.dev newBufferWithBytes:w length:1024
                            options:MTLResourceStorageModeShared];
        }

        MTLSamplerDescriptor* smd = [MTLSamplerDescriptor new];
        smd.minFilter = MTLSamplerMinMagFilterLinear;
        smd.magFilter = MTLSamplerMinMagFilterLinear;
        smd.mipFilter = MTLSamplerMipFilterLinear;
        smd.sAddressMode = MTLSamplerAddressModeClampToEdge;
        smd.tAddressMode = MTLSamplerAddressModeClampToEdge;
        smd.rAddressMode = MTLSamplerAddressModeClampToEdge;
        smd.maxAnisotropy = 1;
        h.linearClamp = [h.dev newSamplerStateWithDescriptor:smd];

        // --- per-frame input textures ---
        h.inColor = make_tex(h.dev, MTLPixelFormatRGBA16Float, m.rw, m.rh);
        h.inDepth = make_tex(h.dev, MTLPixelFormatR32Float, m.rw, m.rh);
        h.inMV    = make_tex(h.dev, MTLPixelFormatRG16Float, m.rw, m.rh);
        h.inExposure = make_tex(h.dev, MTLPixelFormatR32Float, 1, 1);
        h.output  = make_tex(h.dev, MTLPixelFormatRGBA16Float, m.uw, m.uh);

        // --- dispatch sizes (INT8 path of provider) ---
        uint32_t e[3][2] = {{uwA >> 1, uhA >> 1}, {uwA >> 2, uhA >> 2}, {uwA >> 3, uhA >> 3}};
        uint32_t passDim[13][2];
        // INT8 dispatch table: 1,2→enc0 3,4,5→enc1 6,7,8→enc2
        // 9→enc2(/8) 10,11→enc1 12→enc0; encI = upscale >> (I+1)
        static const int level[13] = {0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 1, 1, 0};
        for (int pass = 1; pass <= 12; ++pass) {
            char name[64];
            snprintf(name, sizeof name, "fsr4_model_v07_%s_pass%d", fp16 ? "f16" : "i8", pass);
            const MTLSize wg = h.kernels.at(name).wg;
            passDim[pass][0] = roundUpDiv(e[level[pass]][0], (uint32_t)wg.width);
            passDim[pass][1] = roundUpDiv(e[level[pass]][1], (uint32_t)wg.height);
        }

        OptimizedConstants& oc = *(OptimizedConstants*)h.cbMain.contents;
        AutoExposureSPDConstants& sc = *(AutoExposureSPDConstants*)h.cbSpd.contents;
        RcasConstants& rc = *(RcasConstants*)h.cbRcas.contents;

        // static fields
        oc.inv_size[0] = 1.f / uwA; oc.inv_size[1] = 1.f / uhA;
        oc.scale[0] = 1.f * uwA / m.rw; oc.scale[1] = 1.f * uhA / m.rh;
        oc.inv_scale[0] = 1.f / oc.scale[0]; oc.inv_scale[1] = 1.f / oc.scale[1];
        oc.tex_size[0] = (float)uwA; oc.tex_size[1] = (float)uhA;
        oc.max_renderSize[0] = (float)m.rw; oc.max_renderSize[1] = (float)m.rh;
        oc.width = uwA; oc.height = uhA;
        oc.width_lr = m.rw; oc.height_lr = m.rh;
        oc.rcas_enabled = m.rcas ? 1 : 0;
        oc.rcas_sharpness = m.sharpness;
        oc.fMotionVectorJitterCancellation[0] = oc.fMotionVectorJitterCancellation[1] = 0;
        const uint32_t mvW = m.low_res_mv ? m.rw : m.uw;
        const uint32_t mvH = m.low_res_mv ? m.rh : m.uh;

        if (m.auto_exposure) {
            uint32_t groups[2], wgo[2], nwm[2];
            uint32_t rect[4] = {0, 0, m.rw, m.rh};
            ffxSpdSetup(groups, wgo, nwm, rect);
            sc.workGroupOffset[0] = wgo[0]; sc.workGroupOffset[1] = wgo[1];
            sc.numWorkGroups = nwm[0]; sc.mips = nwm[1];
            sc.invInputSize[0] = 1.f / m.rw; sc.invInputSize[1] = 1.f / m.rh;
            sc.preExposure = m.pre_exposure;
            passDim[0][0] = groups[0]; passDim[0][1] = groups[1]; // reuse slot for spd groups
        }
        if (m.rcas) {
            rc.preExposure = m.pre_exposure;
            FsrRcasCon(rc.config, -2.0f * m.sharpness + 2.0f);
        }

        const int iters = bench > 0 ? bench : 1;
        std::vector<float> gpuMs(m.frames * iters, 0.f);
        std::map<std::string, double> stageMs;
        std::map<std::string, int> stageN;
        [[NSFileManager defaultManager] createDirectoryAtPath:
            [NSString stringWithUTF8String:outdir.c_str()]
            withIntermediateDirectories:YES attributes:nil error:nil];

        // bench mode runs two phases: per-stage (each stage in its own command
        // buffer for clean GPU timestamps) then fused (all stages in one
        // command buffer with the --sync mode's inter-pass sync) so the
        // per-stage-sum vs fused-frame gap is measured in the same run.
        for (int phase = 0; phase < (bench ? 2 : 1); phase++) {
        const bool fusedStage = (phase == 1);
        for (int it = 0; it < iters; it++)
        for (int fi = 0; fi < m.frames; fi++) {
            const int gi = it * m.frames + fi;
            FrameDesc& fd = m.frame[fi];
            char base[256];
            snprintf(base, sizeof base, "%s/f%02d_", tcdir.c_str(), fi);

            // upload inputs
            {
                auto col = read_file(std::string(base) + "color.bin");
                if (col.size() != (size_t)m.rw * m.rh * 8) die("bad color.bin size");
                tex_write(h.inColor, col.data(), m.rw * 8);
                auto dep = read_file(std::string(base) + "depth.bin");
                if (dep.size() != (size_t)m.rw * m.rh * 4) die("bad depth.bin size");
                tex_write(h.inDepth, dep.data(), m.rw * 4);
                auto mv = read_file(std::string(base) + "mv.bin");
                if (mv.size() != (size_t)m.rw * m.rh * 4) die("bad mv.bin size");
                tex_write(h.inMV, mv.data(), m.rw * 4);
            }
            // mirror the reference harness: always bind an explicit 1x1
            // R32_FLOAT exposure = fNN_exposure.bin, else meta.pre_exposure
            std::string exf = std::string(base) + "exposure.bin";
            if (file_exists(exf)) {
                auto ex = read_file(exf);
                tex_write(h.inExposure, ex.data(), 4);
            } else {
                float pe = m.pre_exposure;
                tex_write(h.inExposure, &pe, 4);
            }
            id<MTLTexture> exposureTex = h.inExposure;

            // dynamic constants
            oc.jitter[0] = fd.jitter[0]; oc.jitter[1] = fd.jitter[1];
            oc.mv_scale[0] = fd.mv_scale[0] / mvW; oc.mv_scale[1] = fd.mv_scale[1] / mvH;
            if (m.jittered_mv) {
                oc.fMotionVectorJitterCancellation[0] = (h.prevJitter[0] - oc.jitter[0]) / mvW;
                oc.fMotionVectorJitterCancellation[1] = (h.prevJitter[1] - oc.jitter[1]) / mvH;
            }
            h.prevJitter[0] = oc.jitter[0]; h.prevJitter[1] = oc.jitter[1];
            // provider forces reset on the dispatch that creates the model (frame 0)
            oc.reset = (fd.reset || h.firstFrame) ? 1u : 0u;
            oc.previous_preExposure = h.firstFrame ? oc.preExposure : h.prevPreExposure;
            // provider: previous_preExposure = old main.preExposure before update.
            // first dispatch of a context: main.preExposure is zero-initialised.
            if (h.firstFrame) oc.previous_preExposure = 0.f;
            oc.preExposure = m.pre_exposure;
            h.prevPreExposure = oc.preExposure;
            h.firstFrame = false;

            // In dump mode each pass runs in its own command buffer so the
            // scratch buffer can be snapshotted after every model pass.
            id<MTLCommandBuffer> cb = nil;
            id<MTLComputeCommandEncoder> enc = nil;
            auto flush = [&]() {
                if (!enc) return;
                [enc endEncoding];
                [cb commit];
                [cb waitUntilCompleted];
                if (cb.status == MTLCommandBufferStatusError)
                    die("GPU error: ", [[cb.error localizedDescription] UTF8String]);
                enc = nil;
            };
            auto begin = [&]() {
                cb = [h.queue commandBuffer];
                enc = [cb computeCommandEncoder];
            };
            begin();
            // snapshot scratch after pass pp (00=pre .. 13=post, matching the
            // reference passdumps naming)
            auto dumpScratch = [&](int pp) {
                flush(); begin();
                char p[512];
                snprintf(p, sizeof p, "%s/f%02d_pass%02d_scratch.bin", outdir.c_str(), fi, pp);
                if (!h.scratchPrivate) {
                    write_file(p, h.scratch.contents, h.scratch.length);
                } else {
                    flush();
                    id<MTLCommandBuffer> db = [h.queue commandBuffer];
                    id<MTLBlitCommandEncoder> be = [db blitCommandEncoder];
                    id<MTLBuffer> stag = [h.dev newBufferWithLength:h.scratch.length options:MTLResourceStorageModeShared];
                    [be copyFromBuffer:h.scratch sourceOffset:0 toBuffer:stag destinationOffset:0 size:h.scratch.length];
                    [be endEncoding]; [db commit]; [db waitUntilCompleted];
                    write_file(p, stag.contents, stag.length);
                    begin();
                }
            };
            // in bench mode each stage runs in its own command buffer so
            // GPUStartTime/GPUEndTime gives a per-stage timing (serialized,
            // so bench totals overstate a fused submission — the breakdown
            // is the point).
            // model passes only read/write the scratch buffer; pre/post also
            // touch textures (reprojected history, recurrent, output), so the
            // texture scope is kept on those.
            auto cnbDispatch = [&](const CnbParams& p,
                                   id<MTLComputeCommandEncoder> ce,
                                   uint32_t gx, uint32_t gy,
                                   MTLBarrierScope scope) {
                [ce setComputePipelineState:h.mppPso[2]];
                [ce setBuffer:h.scratch offset:0 atIndex:0];
                [ce setBuffer:h.mppWgt offset:0 atIndex:1];
                [ce setBytes:&p length:sizeof(CnbParams) atIndex:2];
                [ce dispatchThreadgroups:MTLSizeMake(gx, gy, 1)
                    threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                if (scope) [ce memoryBarrierWithScope:scope];
            };
            auto mppDispatch = [&](const FnbParams& p,
                                   id<MTLComputeCommandEncoder> ce,
                                   uint32_t gx, uint32_t gy,
                                   MTLBarrierScope scope) {
                [ce setComputePipelineState:h.mppPso[p.gemm[3] == 64 ? 1 : 0]];
                [ce setBuffer:h.scratch offset:0 atIndex:0];
                [ce setBuffer:h.mppWgt offset:0 atIndex:1];
                [ce setBytes:&p length:sizeof(FnbParams) atIndex:2];
                [ce dispatchThreadgroups:MTLSizeMake(gx, gy, 1)
                    threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                if (scope) [ce memoryBarrierWithScope:scope];
            };
            auto timed = [&](const char* kn, uint32_t gx, uint32_t gy,
                             bool touchesTextures,
                             const FnbParams* mp = nullptr,
                             const CnbParams* cn = nullptr,
                             int i8pass = 0) {
                const MTLBarrierScope scope =
                    touchesTextures ? (MTLBarrierScopeBuffers | MTLBarrierScopeTextures)
                                    : MTLBarrierScopeBuffers;
                std::string sn =
                    (mp || cn) ? std::string(kn) + "_mpp" : (i8pass ? std::string(kn) + "_i8mpp" : kn);
                const MTLBarrierScope sc =
                    syncMode == SYNC_BARRIER ? scope : (MTLBarrierScope)0;
                // pass 13 writes the recurrent texture itself
                id<MTLTexture> i8tex = i8pass == 13 ? h.recurrent : nil;
                if (!bench || fusedStage) {
                    if (i8pass) h.i8.dispatch(i8pass, enc, h.scratch, sc, i8tex);
                    else if (mp) mppDispatch(*mp, enc, gx, gy, sc);
                    else if (cn) cnbDispatch(*cn, enc, gx, gy, sc);
                    else dispatch(h, enc, kn, gx, gy, 1, m.rcas, exposureTex, sc);
                    if (syncMode == SYNC_ENCODER) {
                        [enc endEncoding];
                        enc = [cb computeCommandEncoder];
                    }
                    return;
                }
                flush();
                id<MTLCommandBuffer> pb = [h.queue commandBuffer];
                id<MTLComputeCommandEncoder> pe = [pb computeCommandEncoder];
                // --reps R: the stage is encoded R times back to back (every
                // stage is idempotent on its inputs) so the GPU clock is warm
                // and launch cost amortised; the mean per dispatch is reported.
                for (int r = 0; r < benchReps; r++) {
                    if (i8pass) h.i8.dispatch(i8pass, pe, h.scratch, scope, i8tex);
                    else if (mp) mppDispatch(*mp, pe, gx, gy, scope);
                    else if (cn) cnbDispatch(*cn, pe, gx, gy, scope);
                    else dispatch(h, pe, kn, gx, gy, 1, m.rcas, exposureTex, scope);
                }
                [pe endEncoding]; [pb commit]; [pb waitUntilCompleted];
                stageMs[sn] += (pb.GPUEndTime - pb.GPUStartTime) * 1000.0 / benchReps;
                stageN[sn]++;
                begin();
            };
            if (m.auto_exposure)
                timed("fsr4_spd", passDim[0][0], passDim[0][1], true);
            timed("fsr4_pre", roundUpDiv(uwA, 16), roundUpDiv(uhA, 16), true);
            if (dumpPasses && !bench) dumpScratch(0);
            for (int i = 1; i <= 12; i++) {
                char pname[96];
                snprintf(pname, sizeof pname, "fsr4_model_v07_%s_pass%d",
                         fp16 ? "f16" : "i8", i);
                auto mit = h.mpp.find(i);
                auto cit = h.cnb.find(i);
                if (h.useI8 && h.i8.has(i)) {
                    timed(pname, 0, 0, false, nullptr, nullptr, i);
                } else if (mit != h.mpp.end()) {
                    // tile width per threadgroup: fnb32=128px, fnb64=64px
                    const uint32_t T = mit->second.gemm[3] == 64 ? 64 : 128;
                    timed(pname, roundUpDiv(mit->second.dims[0], T),
                          mit->second.dims[1], false, &mit->second);
                } else if (cit != h.cnb.end())
                    timed(pname, roundUpDiv(cit->second.dims[0], 4),
                          cit->second.dims[1], false, nullptr, &cit->second);
                else
                    timed(pname, passDim[i][0], passDim[i][1], false);
                if (dumpPasses && !bench) dumpScratch(i);
            }
            if (h.useI8 && h.i8.has(13))
                timed("fsr4_post_cnb_ct2d", 0, 0, true, nullptr, nullptr, 13);
            timed("fsr4_post", roundUpDiv(uwA, 16), roundUpDiv(uhA, 16), true);
            if (dumpPasses && !bench) dumpScratch(13);
            if (m.rcas)
                timed("fsr4_rcas", roundUpDiv(uwA, 16), roundUpDiv(uhA, 16), true);
            if (dbg)
                timed("fsr4_dbg", roundUpDiv(uwA, 8), roundUpDiv(uhA, 8), true);
            flush();
            if (cb.status == MTLCommandBufferStatusError)
                die("GPU error: ", [[cb.error localizedDescription] UTF8String]);
            gpuMs[gi] = (float)((cb.GPUEndTime - cb.GPUStartTime) * 1000.0);

            // --- dumps ---
            if (dumpPasses) {
                // NOTE: scratch dumps post-run are frame-end state; per-pass
                // dumping would need per-dispatch command buffers (slow).
                char p[512];
                snprintf(p, sizeof p, "%s/f%02d_scratch.bin", outdir.c_str(), fi);
                if (!h.scratchPrivate)
                    write_file(p, h.scratch.contents, h.scratch.length);
                else {
                    id<MTLCommandBuffer> db = [h.queue commandBuffer];
                    id<MTLBlitCommandEncoder> be = [db blitCommandEncoder];
                    id<MTLBuffer> stag = [h.dev newBufferWithLength:h.scratch.length options:MTLResourceStorageModeShared];
                    [be copyFromBuffer:h.scratch sourceOffset:0 toBuffer:stag destinationOffset:0 size:h.scratch.length];
                    [be endEncoding]; [db commit]; [db waitUntilCompleted];
                    write_file(p, stag.contents, stag.length);
                }
                snprintf(p, sizeof p, "%s/f%02d_recurrent.bin", outdir.c_str(), fi);
                { std::vector<uint8_t> t((size_t)uwA * uhA * 4);
                  tex_read(h.recurrent, t.data(), uwA * 4); write_file(p, t.data(), t.size()); }
                snprintf(p, sizeof p, "%s/f%02d_history.bin", outdir.c_str(), fi);
                { std::vector<uint8_t> t((size_t)uwA * uhA * 8);
                  tex_read(h.history, t.data(), uwA * 8); write_file(p, t.data(), t.size()); }
                snprintf(p, sizeof p, "%s/f%02d_reprojected.bin", outdir.c_str(), fi);
                { std::vector<uint8_t> t((size_t)uwA * uhA * 8);
                  tex_read(h.historyReproj, t.data(), uwA * 8); write_file(p, t.data(), t.size()); }
                snprintf(p, sizeof p, "%s/f%02d_constants.bin", outdir.c_str(), fi);
                write_file(p, &oc, sizeof(oc));
            }

            if (bench) continue;   // bench = timing only, no writes

            // --- output readback ---
            std::vector<uint8_t> out((size_t)m.uw * m.uh * 8);
            tex_read(h.output, out.data(), m.uw * 8);
            char op[512];
            snprintf(op, sizeof op, "%s/f%02d_out.bin", outdir.c_str(), fi);
            write_file(op, out.data(), out.size());

            // PNG (linear→sRGB when input colorspace is linear)
            std::vector<uint8_t> png(m.uw * m.uh * 4);
            const uint16_t* hp = (const uint16_t*)out.data();
            for (size_t i = 0; i < (size_t)m.uw * m.uh * 4; i++) {
                _Float16 hv; memcpy(&hv, hp + i, 2);
                float v = (float)hv;
                if (colorspace == 0) {
                    v = v <= 0.f ? 0.f : (v <= 0.0031308f ? 12.92f * v
                            : 1.055f * powf(v, 1.f / 2.4f) - 0.055f);
                }
                v = fminf(fmaxf(v, 0.f), 1.f);
                png[i] = (uint8_t)(v * 255.f + 0.5f);
            }
            // opaque alpha in the preview PNG, matching the reference harness
            for (size_t i = 3; i < png.size(); i += 4) png[i] = 255;
            snprintf(op, sizeof op, "%s/f%02d_out.png", outdir.c_str(), fi);
            png_write(op, png.data(), m.uw, m.uh);
            fprintf(stderr, "frame %d done (gpu %.2f ms)\n", fi, gpuMs[gi]);
        }
        } // phase

        if (bench) {
            double tot = 0;
            printf("per-stage gpu_ms (mean over %d frame dispatches):\n",
                   iters * m.frames);
            // machine-parseable: one CSV line per stage
            printf("bench_csv,model,stage,gpu_ms\n");
            for (auto& kv : stageMs) {
                printf("  %-28s %7.3f\n", kv.first.c_str(),
                       kv.second / stageN[kv.first]);
                printf("bench_csv,%s,%s,%.4f\n", fp16 ? "fp16" : "int8",
                       kv.first.c_str(), kv.second / stageN[kv.first]);
                tot += kv.second / stageN[kv.first];
            }
            printf("  %-28s %7.3f\n", "TOTAL(per-pass sum)", tot);
            printf("bench_csv,%s,TOTAL,%.4f\n", fp16 ? "fp16" : "int8", tot);
            // fused-phase whole-CB times over every iteration x frame (gpuMs
            // was overwritten by phase 1): mean, and the min as the
            // warm-clock figure (the GPU clock ramps over the first frames)
            float fsum = 0, fmin = 1e9f;
            for (int i = 0; i < m.frames * iters; i++) { fsum += gpuMs[i]; if (gpuMs[i] < fmin) fmin = gpuMs[i]; }
            const char* sn = syncMode == SYNC_BARRIER ? "barrier"
                         : syncMode == SYNC_ENCODER ? "encoder" : "none";
            printf("fused frame gpu_ms (sync=%s): mean %.2f  min %.2f  (%d frames)\n", sn,
                   fsum / (m.frames * iters), fmin, m.frames * iters);
            printf("bench_csv,%s,FUSED_FRAME_%s,%.4f\n",
                   fp16 ? "fp16" : "int8", sn, fsum / (m.frames * iters));
            printf("bench_csv,%s,FUSED_FRAME_MIN_%s,%.4f\n",
                   fp16 ? "fp16" : "int8", sn, fmin);
        } else {
            float sum = 0;
            for (int fi = 0; fi < m.frames; fi++) sum += gpuMs[fi];
            printf("gpu_time_ms:");
            for (int fi = 0; fi < m.frames; fi++) printf(" %.2f", gpuMs[fi]);
            printf("  mean %.2f\n", sum / m.frames);
        }
    }
    return 0;
}
