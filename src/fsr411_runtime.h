// SPDX-License-Identifier: MIT
// Shared implementation for the standalone runner and MetalFX live bridge.
// Compile Objective-C++ users with ARC. encode() never commits the caller's
// command buffer. One View is one ordered temporal stream / MetalFX scaler.
#pragma once
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include "fsr4_input.h"
#include "fsr4_log.h"
#include "fsr411_model.h"

namespace fsr411 {

// Simd: Apple7+ (M1–M4) simdgroup half matrices with exact float accumulation;
// byte-identical to Scalar. Auto: Native on Apple10, else Simd, else Scalar.
enum class Backend { Auto, Reference, Native, Scalar, Simd };
// Integer network implementation chosen for one session.
enum class Kernels { Reference, Scalar, Simd, Mpp };
inline const char *kernels_label(Kernels k) {
    return k==Kernels::Mpp?"int8-mpp":k==Kernels::Simd?"int8-simd":k==Kernels::Scalar?"int8-scalar":"int8-reference";
}
// Exact: original coefficients, IEEE PRE/POST math, inference every frame.
// ExactFast: Exact's network with reciprocal division and fast exp/log in
// PRE/POST. Lossy: folded coefficients, half-up rounding and inference reuse.
enum class Profile { Exact, Lossy, ExactFast };
inline const char *profile_name(Profile p) {
    return p==Profile::Lossy?"lossy":p==Profile::ExactFast?"exact-fast":"exact";
}
inline bool parse_profile(const std::string &name, Profile &p) {
    if(name=="exact") p=Profile::Exact;
    else if(name=="exact-fast") p=Profile::ExactFast;
    else if(name=="lossy") p=Profile::Lossy;
    else return false;
    return true;
}
// Live settings encoding (panel.conf, fsr4_live_get_profile): 0/1 keep their
// original meaning so older settings files still load.
inline Profile profile_from_int(int v) { return v==1?Profile::Lossy:v==2?Profile::ExactFast:Profile::Exact; }
inline int profile_to_int(Profile p) { return p==Profile::Lossy?1:p==Profile::ExactFast?2:0; }
// Synchronous, opt-in CLI instrumentation. A profiler may replace the classic
// compute encoder to attach stage-boundary counters. Ordinary/live encodes do
// not install one, retain their single encoder, and record no counter samples.
struct StageProfiler {
    virtual ~StageProfiler() = default;
    virtual void begin_stage(id __strong &encoder, const std::string &name) = 0;
    virtual void end_stage(id encoder) = 0;
};
struct FrameSnapshot {
    Fsr4InputDesc input = {};
    Fsr4InputPrepared prepared = {};
    // Logical input/output regions, with source format values preserved.
    // A fresh snapshot belongs to one encode; its completion token never resets.
    id<MTLTexture> color = nil, depth = nil, motion = nil, output = nil;
    id<MTLBuffer> completion = nil;
    bool ready() const {
        return completion && __atomic_load_n((uint32_t *)completion.contents, __ATOMIC_ACQUIRE) == 1u;
    }
};
struct Options {
    Backend backend = Backend::Auto;
    Profile profile = Profile::Lossy;
    int tier = 0, colorspace = 0, optimize = 1;
    bool rcas = false, allowCompile = true, ultraPerformance = false;
    // Tests only: Exact Fast/Lossy PRE/POST with ordered arithmetic, so their
    // optimized graph can be compared bit for bit with the imported HLSL.
    bool strictMath = false;
    float sharpness = 0;
    // Diagnostic snapshots are GPU copies at a precise graph boundary.
    // They are not enabled by the live bridge or ordinary benchmarking.
    std::vector<std::string> captureStages;
    // Optional immutable CPU-readable frame. No allocation or dispatch when null.
    std::shared_ptr<FrameSnapshot> snapshot;
    StageProfiler *stageProfiler = nullptr; // caller owns it until encode returns
};
struct Frame {
    id<MTLTexture> color = nil, depth = nil, motion = nil, output = nil, exposure = nil;
    Fsr4InputDesc input = {};
};

struct FrameConstants {
    float inv_size[2], scale[2], inv_scale[2], jitter[2], mv_scale[2];
    float tex_size[2], max_renderSize[2], jitterCancellation[2];
    uint32_t width, height, reset, width_lr;
    uint32_t height_lr; float preExposure, previousPreExposure; uint32_t rcas;
    float sharpness, pad;
};
struct SpdConstants {
    uint32_t mips, groups, offset[2];
    float invInputSize[2], preExposure, padding;
    uint32_t validSize[2]; // 4.1.1 reimplementation's viewport extension
};
struct RcasConstants { uint32_t config[4]; float preExposure; uint32_t pad[3]; };
struct LossyConstants { uint32_t skip, blockWidth, blockHeight, reserved; };
struct RegionConstants { uint32_t source[4], destination[4]; };
struct TensorSizes { uint32_t v[17][4]; };
static_assert(sizeof(FrameConstants) == 104 && sizeof(SpdConstants) == 40,
              "FSR 4.1.1 frame constant ABI");
static_assert(sizeof(TensorSizes) == 272 && sizeof(RcasConstants) == 32,
              "FSR 4.1.1 tensor/RCAS ABI");

inline uint32_t ceil_div(uint32_t x, uint32_t y) { return (x + y - 1u) / y; }
inline NSString *ns(const std::string &x) { return [NSString stringWithUTF8String:x.c_str()]; }
inline std::string utf8(NSString *x) { return x ? std::string(x.UTF8String) : std::string(); }
inline std::string metal_error(NSError *error) { return error ? utf8(error.localizedDescription) : "Metal allocation failed"; }
inline TensorSizes tensor_sizes(uint32_t w, uint32_t h) {
    // Provider tensor constants are indexed by graph edges, not by passes.
    constexpr unsigned shifts[17] = {1,0,1,1,2,2,2,3,3,3,2,2,1,1,0,0,0};
    TensorSizes result = {};
    for (int i = 0; i < 17; ++i) { result.v[i][0] = w >> shifts[i]; result.v[i][1] = h >> shifts[i]; }
    return result;
}
inline int required_tier(uint32_t w, uint32_t h) {
    if (w <= 1920 && h <= 1080) return 1080;
    if (w <= 3840 && h <= 2160) return 2160;
    if (w <= 7680 && h <= 4320) return 4320;
    return 0;
}
inline int pass_shift(int pass) {
    if (pass <= 2 || pass == 12) return 1;
    if (pass >= 6 && pass <= 9) return 3;
    return 2;
}
inline int pad_shift(int pad) {
    if (pad <= 2 || pad >= 11) return 1;
    if (pad >= 6 && pad <= 8) return 3;
    return 2;
}
inline uint32_t padding_groups(int tier, int pad, uint32_t uw, uint32_t uh) {
    const int shift = pad_shift(pad);
    const uint32_t w = uw >> shift, h = uh >> shift;
    const uint32_t maxW = uint32_t(tier * 16 / 9) >> shift, maxH = uint32_t(tier) >> shift;
    const uint32_t c = std::min(maxW + 1u - w, 5u), d = std::min(maxH + 1u - h, 1u);
    const uint32_t f = w + 1u + c;
    return ceil_div(f + h + h * c + f * d, 32);
}

struct Kernel {
    id<MTLComputePipelineState> pso = nil;
    id<MTLBuffer> literalTables = nil;
    MTLSize threads = MTLSizeMake(1, 1, 1);
    std::vector<std::pair<NSUInteger, std::string>> buffers, textures, samplers;
};
struct NativePass {
    id<MTLComputePipelineState> pso = nil;
    PassParams parameters = {};
    unsigned inputSlot = 0, outputSlot = 0, skipSlot = 0;
    unsigned tileX = 1, tileY = 1, threads = 128, offset = 0;
    bool inputDispatch = false, hasSkip = false;
};
struct Session {
    std::string key;
    int tier = 0;
    uint32_t postPixelsPerGroup = 32;
    bool splitAutoExposure = false;
    size_t scratchBytes = 0;
    bool ultraPerformance = false, externalPostCnb = false;
    Profile profile = Profile::Exact;
    // halfWeights: Simd only, nativeWeights widened element-wise to half.
    id<MTLBuffer> initializer = nil, nativeWeights = nil, halfWeights = nil;
    std::map<std::string, Kernel> kernels;
    std::map<int, NativePass> native;
    std::vector<std::string> order;
};

struct FrameSlot {
    static constexpr size_t kSize = 32768, kDone = kSize - 4;
    id<MTLBuffer> constants = nil;
    NSMutableDictionary *tables = nil;
    NSMutableArray *resources = nil;
    id<MTLResidencySet> residency = nil;
    std::shared_ptr<FrameSnapshot> snapshot;
    StageProfiler *stageProfiler = nullptr;
    bool busy = false;
    bool retired() const {
        return !busy || (constants && __atomic_load_n(
            (uint32_t *)((uint8_t *)constants.contents + kDone), __ATOMIC_ACQUIRE) == 1);
    }
};

struct View {
    Fsr4InputHistory inputHistory = {};
    Fsr4InputPrepared prepared = {};
    std::string variantKey, backendLabel;
    id<MTLTexture> history = nil, reprojected = nil, recurrent = nil, autoExposure = nil;
    id<MTLBuffer> scratch = nil, spdAtomic = nil;
    id<MTLBuffer> lossyMotion = nil, lossyPlane = nil;
    id<MTLTexture> spdAtomicTexture = nil, luma = nil;
    id<MTLTexture> colorCopy = nil, depthCopy = nil, motionCopy = nil, outputCopy = nil, rcasTemp = nil;
    std::map<std::string, id<MTLBuffer>> captures;
    std::vector<std::unique_ptr<FrameSlot>> slots;
    uint32_t width = 0, height = 0;
    uint64_t encodes = 0;
    uint64_t neuralRuns = 0, reusedFrames = 0;
    bool cacheValid = false, skippedInference = false;
    bool zeroScratch = true, zeroTargets = true;
    void invalidate() { fsr4_input_invalidate(&inputHistory);cacheValid=false; }
};

class Runtime {
public:
    Runtime(id<MTLDevice> dev, const std::string &root, const std::string &buildDir)
        : device_(dev), root_(root), buildDir_(buildDir) {}

    bool copy_output(id cb, View &lifetime, id<MTLTexture> source, Fsr4InputRect sourceRect,
                     id<MTLTexture> output, Fsr4InputRect outputRect, std::string &error) {
        if (!source || !output || sourceRect.width != outputRect.width || sourceRect.height != outputRect.height
            || sourceRect.x > source.width || sourceRect.y > source.height
            || sourceRect.width > source.width-sourceRect.x || sourceRect.height > source.height-sourceRect.y
            || outputRect.x > output.width || outputRect.y > output.height
            || outputRect.width > output.width-outputRect.x || outputRect.height > output.height-outputRect.y) {
            error="placeholder encode has no compatible completed FSR output"; return false;
        }
        if (source == output && sourceRect.x==outputRect.x && sourceRect.y==outputRect.y) return true;
        if (source == output) { error="overlapping placeholder output copy requires a separate texture";return false; }
        if (!ensure_helpers(error)) return false;
        auto slot=acquire_slot(lifetime,error); if(!slot) return false;
        [slot->resources addObjectsFromArray:@[slot->constants,source,output]];
        id enc=[cb computeCommandEncoder]; if(!enc) { error="caller command buffer already encoding"; return false; }
        bool m4=![enc respondsToSelector:@selector(setBuffer:offset:atIndex:)];
        if (m4 && (!table(*slot,"copyOnly",error) || !table(*slot,"retire",error) || !bind_residency(*slot,cb,error))) {
            [enc endEncoding]; return false;
        }
        slot->busy=true;
        __atomic_store_n((uint32_t *)((uint8_t *)slot->constants.contents+FrameSlot::kDone),0u,__ATOMIC_RELEASE);
        @try {
            if(m4) queue_before(enc);
            copy(enc,m4,*slot,"copyOnly",source,output,sourceRect,outputRect.x,outputRect.y,8192,false);
            helper(enc,m4,*slot,"retire","fsr411_retire",{{slot->constants,FrameSlot::kDone}},{},MTLSizeMake(1,1,1));
            if(m4) queue_after(enc);
            [enc endEncoding]; return true;
        } @catch(NSException *exception) {
            @try { [enc endEncoding]; } @catch(NSException *) {}
            error=utf8(exception.reason); return false;
        }
    }

    bool encode(id commandBuffer, View &view, const Frame &frame, const Options &options, std::string &error) {
        error.clear();
        auto fail = [&](const std::string &why) { error = why; view.invalidate(); return false; };
        if (!device_ || !commandBuffer) return fail("missing Metal device or command buffer");
        if (!validate_textures(frame, error)) return fail(error);
        if (options.colorspace < 0 || options.colorspace > 3 || !std::isfinite(options.sharpness)
            || options.sharpness < 0 || options.sharpness > 1) return fail("invalid colour space or sharpness");
        Fsr4InputPrepared p = {};
        auto status = fsr4_input_prepare(&frame.input, &view.inputHistory, 0, &p);
        if (!fsr4_input_status_ok(status)) return fail(fsr4_input_status_reason(status));
        int tier = options.tier ? options.tier : required_tier(p.alignedUW, p.alignedUH);
        if ((tier != 1080 && tier != 2160 && tier != 4320) || !required_tier(p.alignedUW, p.alignedUH)
            || p.alignedUW > uint32_t(tier * 16 / 9) || p.alignedUH > uint32_t(tier))
            return fail("output content exceeds FSR 4.1.1 tensor tier");
        const bool native = (options.backend==Backend::Auto || options.backend==Backend::Native) && native_capable();
        if (options.backend == Backend::Native && !native) return fail("native FSR 4.1.1 requires Apple GPU family 10 and macOS 26");
        if (options.backend == Backend::Simd && !simd_family()) return fail("simd FSR 4.1.1 requires Apple GPU family 7 (M1) or newer");
        Kernels kind = native ? Kernels::Mpp : options.backend==Backend::Reference ? Kernels::Reference
            : options.backend==Backend::Scalar ? Kernels::Scalar
            : (options.backend==Backend::Simd || (simd_family() && !simdFailed_)) ? Kernels::Simd : Kernels::Scalar;
        if (!ensure_helpers(error)) return fail(error);
        Session *session = ensure_session(tier, p, options, kind, error);
        if (!session && kind==Kernels::Simd && options.backend==Backend::Auto) {
            // Never retry a failed simd setup on this device; scalar is exact too.
            fsr4_logf("FSR 4.1.1 simd backend unavailable (%s); using scalar", error.c_str());
            simdFailed_ = true; kind = Kernels::Scalar; error.clear();
            session = ensure_session(tier, p, options, kind, error);
        }
        if (!session) return fail(error);
        const bool native_or_portable = kind != Kernels::Reference;
        status = fsr4_input_prepare(&frame.input, &view.inputHistory, view.variantKey != session->key, &p);
        if (!fsr4_input_status_ok(status)) return fail(fsr4_input_status_reason(status));
        if (!ensure_targets(view, *session, p, frame, options, error)) return fail(error);
        if (options.snapshot && !prepare_snapshot(*options.snapshot, frame, p, error)) return fail(error);
        FrameSlot *slot = acquire_slot(view, error);
        if (!slot) return fail(error);
        slot->snapshot = options.snapshot;
        slot->stageProfiler = options.stageProfiler;
        const bool rcas = options.rcas && options.sharpness > 0;
        const bool lossy = options.profile==Profile::Lossy;
        // The supplied vectors describe one frame of motion. Never apply
        // them directly to a cache older than one frame (unusual jitter can
        // have consecutive negatives). Refresh after every reused frame.
        const bool skip = lossy && view.cacheValid && !view.skippedInference && !p.reset && p.jitter[0]<0.f
            && uint64_t(p.rw)*2u>=p.alignedUW && !options.ultraPerformance;
        TensorSizes sizes = tensor_sizes(p.alignedUW, p.alignedUH);
        fill_constants(*slot, *session, p, sizes, rcas, options.sharpness);
        const LossyConstants lc={skip?1u:0u,p.alignedUW/4u,p.alignedUH/4u,0};
        put(*slot,kLossy,lc);
        view.captures.clear();
        for (const auto &name : options.captureStages) {
            if (std::find(session->order.begin(),session->order.end(),name)==session->order.end()) return fail("unknown capture stage " + name);
            if(skip && name!="pre" && name!="post") continue;
            auto buffer = [device_ newBufferWithLength:session->scratchBytes options:MTLResourceStorageModeShared];
            if (!buffer) return fail("cannot allocate diagnostic scratch snapshot");
            view.captures.emplace(name, buffer);
        }
        id enc = [commandBuffer computeCommandEncoder];
        if (!enc) return fail("caller command buffer already has an active encoder");
        const bool m4 = ![enc respondsToSelector:@selector(setBuffer:offset:atIndex:)];
        if (m4 && options.stageProfiler) {
            [enc endEncoding]; return fail("CLI stage counters require a classic Metal command buffer");
        }
        if (m4 && ![enc respondsToSelector:@selector(setArgumentTable:)]) {
            [enc endEncoding]; return fail("unsupported Metal compute encoder");
        }
        // Reserve all argument tables before recording the first GPU write.
        if (m4 && !prepare_tables(*slot, *session, view, error)) {
            [enc endEncoding]; return fail(error);
        }
        retain_resources(*slot, *session, view, frame);
        if (m4 && !bind_residency(*slot, commandBuffer, error)) {
            [enc endEncoding]; return fail(error);
        }
        slot->busy = true;
        __atomic_store_n((uint32_t *)((uint8_t *)slot->constants.contents + FrameSlot::kDone), 0u, __ATOMIC_RELEASE);
        Bindings b{frame.color, frame.depth, frame.motion, frame.output,
                   p.exposureMode == FSR4_INPUT_EXPOSURE_TEXTURE ? frame.exposure : exposureOne_, &view, rcas};
        @try {
            if (m4) queue_before(enc);
            if (view.zeroScratch) clear_buffer(enc, m4, *slot, "clearScratch", view.scratch, 8192);
            if (skip) clear_buffer(enc,m4,*slot,"clearLossyPlane",view.lossyPlane,10240);
            if (view.zeroTargets || p.reset) {
                clear_texture(enc, m4, *slot, "clearHistory", view.history);
                clear_texture(enc, m4, *slot, "clearReprojected", view.reprojected);
                clear_texture(enc, m4, *slot, "clearRecurrent", view.recurrent);
                helper(enc, m4, *slot, "initExposure", "fsr411_init_exposure", {}, {view.autoExposure}, MTLSizeMake(1,1,1));
            }
            if (p.copyMask & FSR4_INPUT_COPY_COLOR) {
                copy(enc, m4, *slot, "copyColor", frame.color, view.colorCopy, p.colorRegion, 0, 0, 8448, false);
                b.color = view.colorCopy;
            }
            if ((p.copyMask & FSR4_INPUT_COPY_DEPTH) || depth_format(frame.depth.pixelFormat)) {
                copy(enc, m4, *slot, "copyDepth", frame.depth, view.depthCopy, p.depthRegion, 0, 0, 8704, depth_format(frame.depth.pixelFormat));
                b.depth = view.depthCopy;
            }
            if (p.copyMask & FSR4_INPUT_COPY_MOTION) {
                copy(enc, m4, *slot, "copyMotion", frame.motion, view.motionCopy, p.motionRegion, 0, 0, 8960, false);
                b.motion = view.motionCopy;
            }
            if (slot->snapshot) {
                const Fsr4InputRect renderRegion{0,0,p.rw,p.rh}, motionRegion{0,0,p.mvW,p.mvH};
                copy(enc,m4,*slot,"snapshotColor",b.color,slot->snapshot->color,renderRegion,0,0,kSnapshot,false);
                copy(enc,m4,*slot,"snapshotDepth",b.depth,slot->snapshot->depth,renderRegion,0,0,kSnapshot+256,false);
                copy(enc,m4,*slot,"snapshotMotion",b.motion,slot->snapshot->motion,motionRegion,0,0,kSnapshot+512,false);
            }
            if (p.copyMask & FSR4_INPUT_COPY_OUTPUT) b.output = view.outputCopy;
            if (p.autoExposure) {
                if(!session->splitAutoExposure) {
                    clear_buffer(enc, m4, *slot, "clearAtomic", view.spdAtomic, 9216);
                    clear_texture(enc, m4, *slot, "clearLuma", view.luma);
                }
                dispatch(enc, m4, *slot, *session, "ae", MTLSizeMake(ceil_div(p.rw,64),ceil_div(p.rh,64),1), b);
                // dispatch() inserts a texture/buffer barrier on both encoder
                // APIs. The final group sees every tile without relying on
                // DirectX globallycoherent texture/atomic publication rules.
                if(session->splitAutoExposure && std::max(p.rw,p.rh)>=128)
                    dispatch(enc,m4,*slot,*session,"ae_finalize",MTLSizeMake(1,1,1),b);
            } else {
                const uint32_t reset = p.reset;
                put(*slot, 9472, reset);
                helper(enc, m4, *slot, "updateExposure", "fsr411_exposure",
                       {{slot->constants,9472}}, {b.exposure,view.autoExposure}, MTLSizeMake(1,1,1));
            }
            for (const auto &name : session->order) {
                if (skip && name!="pre" && name!="post") continue;
                if (name=="post" && session->externalPostCnb && !skip)
                    dispatch_native(enc,m4,*slot,*session,13,view,sizes);
                const int pass = name.rfind("pass",0) == 0 ? std::atoi(name.c_str()+4) : 0;
                if (native_or_portable && pass && session->native.count(pass)) {
                    dispatch_native(enc, m4, *slot, *session, pass, view, sizes);
                } else {
                    MTLSize groups;
                    if (name == "pre") groups = MTLSizeMake(ceil_div(p.alignedUW,16),ceil_div(p.alignedUH,16),1);
                    else if (name == "post") groups = MTLSizeMake(ceil_div(p.alignedUW,session->postPixelsPerGroup),ceil_div(p.alignedUH,session->postPixelsPerGroup),1);
                    else if (name.rfind("pad",0) == 0) groups = MTLSizeMake(padding_groups(tier,std::atoi(name.c_str()+3),p.alignedUW,p.alignedUH),1,1);
                    else { const int shift = pass_shift(pass); groups = MTLSizeMake(ceil_div(p.alignedUW >> shift,64),p.alignedUH >> shift,1); }
                    dispatch(enc, m4, *slot, *session, name, groups, b);
                }
                const auto capture = view.captures.find(name);
                if (capture != view.captures.end()) {
                    const uint32_t count = uint32_t(view.scratch.length / 16);
                    put(*slot,9728,count);
                    helper(enc,m4,*slot,"capture_"+name,"fsr411_snapshot",
                           {{view.scratch,0},{capture->second,0},{slot->constants,9728}}, {}, MTLSizeMake(count,1,1));
                }
            }
            if (rcas) dispatch(enc,m4,*slot,*session,"rcas",MTLSizeMake(ceil_div(p.alignedUW,16),ceil_div(p.alignedUH,16),1),b);
            if (p.copyMask & FSR4_INPUT_COPY_OUTPUT) {
                const Fsr4InputRect region{0,0,p.uw,p.uh};
                copy(enc,m4,*slot,"copyOutput",view.outputCopy,frame.output,region,p.outputRegion.x,p.outputRegion.y,9984,false);
            }
            if (slot->snapshot) {
                copy(enc,m4,*slot,"snapshotOutput",frame.output,slot->snapshot->output,p.outputRegion,0,0,kSnapshot+768,false);
                // Every snapshot texture write is visible before the standalone
                // token changes to one. Later frames never write these resources.
                helper(enc,m4,*slot,"snapshotDone","fsr411_retire",{{slot->snapshot->completion,0}},{},MTLSizeMake(1,1,1));
            }
            helper(enc,m4,*slot,"retire","fsr411_retire",{{slot->constants,FrameSlot::kDone}}, {}, MTLSizeMake(1,1,1));
            if (m4) queue_after(enc);
            [enc endEncoding];
        } @catch (NSException *exception) {
            // Partial GPU writes invalidate history. Keep the slot retained if
            // recording the retirement token itself failed; never recycle it.
            @try { [enc endEncoding]; } @catch (NSException *) {}
            return fail("FSR 4.1.1 encode: " + utf8(exception.reason));
        }
        view.zeroScratch = view.zeroTargets = false;
        fsr4_input_commit(&view.inputHistory,&p);
        view.prepared = p;
        view.variantKey = session->key;
        view.backendLabel = kernels_label(kind);
        view.cacheValid=session->externalPostCnb;
        view.skippedInference=skip;
        if(skip) ++view.reusedFrames;else ++view.neuralRuns;
        ++view.encodes;
        return true;
    }

private:
    struct BufferBinding { id<MTLBuffer> buffer; NSUInteger offset; };
    struct Bindings {
        id<MTLTexture> color, depth, motion, output, exposure;
        View *view; bool rcas;
    };
    static constexpr NSUInteger kTensor = 256, kSpd = 768, kRcas = 1024, kLossy = 1280, kNative = 2048;
    static constexpr NSUInteger kSnapshot = 12288;
    static_assert(kSnapshot + 768 + sizeof(RegionConstants) < FrameSlot::kDone, "snapshot constant region overflow");
    id<MTLDevice> device_;
    std::string root_, buildDir_;
    std::map<std::string,std::unique_ptr<Session>> sessions_;
    std::map<std::string,id<MTLComputePipelineState>> helpers_;
    id<MTLSamplerState> sampler_ = nil;
    id<MTLTexture> exposureOne_ = nil;
    // Simd probe result per Runtime (one device): -1 unknown, 0 failed, 1 ok.
    int simdVerified_ = -1;
    bool simdFailed_ = false;

    bool native_capable() const {
        if (@available(macOS 26.0,*)) return [device_ supportsFamily:static_cast<MTLGPUFamily>(1010)]
            && device_.maxThreadgroupMemoryLength >= 20480 && device_.maxThreadsPerThreadgroup.width >= 128;
        return false;
    }
    bool simd_family() const { return [device_ supportsFamily:MTLGPUFamilyApple7]; }
    // Confirms the lane ownership the simd epilogue relies on and that half
    // products of INT8 extremes accumulate exactly in float on this GPU.
    bool verify_simd(id<MTLLibrary> lib, std::string &error) {
        if (simdVerified_ >= 0) { if (!simdVerified_) error = "simdgroup probe failed on this GPU"; return simdVerified_ == 1; }
        simdVerified_ = 0;
        NSError *e = nil;
        id<MTLFunction> fn = [lib newFunctionWithName:@"fsr411_simd_probe"];
        id<MTLComputePipelineState> pso = fn ? [device_ newComputePipelineStateWithFunction:fn error:&e] : nil;
        if (!pso || pso.maxTotalThreadsPerThreadgroup < 32 || pso.threadExecutionWidth != 32) {
            error = "simdgroup probe pipeline: " + (e ? metal_error(e) : std::string("needs SIMD32")); return false;
        }
        _Float16 a[64], b[64]; int ai[64], bi[64]; uint32_t r = 0x4117u;
        for (int i = 0; i < 64; ++i) {
            r = r * 1664525u + 1013904223u;
            ai[i] = i % 3 == 0 ? -128 : i % 3 == 1 ? 127 : int(r >> 24) - 128;
            bi[i] = i % 5 == 0 ? 127 : -128;
            a[i] = (_Float16)ai[i]; b[i] = (_Float16)bi[i];
        }
        id<MTLBuffer> A = [device_ newBufferWithBytes:a length:sizeof a options:MTLResourceStorageModeShared];
        id<MTLBuffer> B = [device_ newBufferWithBytes:b length:sizeof b options:MTLResourceStorageModeShared];
        id<MTLBuffer> out = [device_ newBufferWithLength:96 * sizeof(int) options:MTLResourceStorageModeShared];
        id<MTLCommandQueue> queue = [device_ newCommandQueue];
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:pso]; [enc setBuffer:A offset:0 atIndex:0];
        [enc setBuffer:B offset:0 atIndex:1]; [enc setBuffer:out offset:0 atIndex:2];
        [enc dispatchThreadgroups:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
        [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
        if (cb.status != MTLCommandBufferStatusCompleted) { error = "simdgroup probe did not complete"; return false; }
        const int *v = (const int *)out.contents;
        for (int i = 0; i < 8; ++i) for (int j = 0; j < 8; ++j) {
            int s = 0; for (int k = 0; k < 8; ++k) s += ai[i*8+k] * bi[k*8+j];
            if (v[i*8+j] != 2 * s) { error = "simdgroup half products do not accumulate exactly in float"; return false; }
        }
        for (int lane = 0; lane < 32; ++lane)
            if (v[64+lane] != 1) { error = "simdgroup element layout differs from the verified Apple7+ layout"; return false; }
        simdVerified_ = 1; return true;
    }
    static bool depth_format(MTLPixelFormat format) {
        return format == MTLPixelFormatDepth16Unorm || format == MTLPixelFormatDepth32Float
            || format == MTLPixelFormatDepth32Float_Stencil8;
    }
    bool validate_textures(const Frame &f, std::string &error) const {
        const id<MTLTexture> textures[] = {f.color,f.depth,f.motion,f.output};
        const Fsr4InputSurface surfaces[] = {f.input.geometry.color,f.input.geometry.depth,f.input.geometry.motion,f.input.geometry.output};
        for (int i=0;i<4;++i) {
            if (!textures[i] || textures[i].device != device_ || textures[i].textureType != MTLTextureType2D
                || textures[i].sampleCount != 1 || textures[i].width != surfaces[i].width || textures[i].height != surfaces[i].height) {
                error = "frame geometry does not match single-sample Metal textures"; return false;
            }
        }
        if (f.input.hasExposureTexture != (f.exposure != nil)) { error="exposure descriptor/resource mismatch"; return false; }
        if (f.exposure && (f.exposure.device != device_ || f.exposure.textureType != MTLTextureType2D
            || !f.exposure.width || !f.exposure.height || f.exposure.sampleCount != 1)) { error="invalid exposure texture"; return false; }
        return true;
    }
    id<MTLTexture> texture(MTLPixelFormat format, uint32_t w, uint32_t h, MTLStorageMode storage = MTLStorageModePrivate) {
        MTLTextureDescriptor *d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format width:w height:h mipmapped:NO];
        d.storageMode = storage;
        d.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
        return [device_ newTextureWithDescriptor:d];
    }
    bool prepare_snapshot(FrameSnapshot &snapshot, const Frame &frame, const Fsr4InputPrepared &p, std::string &error) {
        if (snapshot.completion || snapshot.color || snapshot.depth || snapshot.motion || snapshot.output) {
            error = "a frame snapshot can only be used for one encode"; return false;
        }
        auto color=texture(MTLPixelFormatRGBA32Float,p.rw,p.rh,MTLStorageModeShared);
        auto depth=texture(MTLPixelFormatR32Float,p.rw,p.rh,MTLStorageModeShared);
        auto motion=texture(MTLPixelFormatRG32Float,p.mvW,p.mvH,MTLStorageModeShared);
        auto output=texture(MTLPixelFormatRGBA32Float,p.uw,p.uh,MTLStorageModeShared);
        auto completion=[device_ newBufferWithLength:sizeof(uint32_t) options:MTLResourceStorageModeShared];
        if (!color || !depth || !motion || !output || !completion) {
            error = "cannot allocate CPU-readable frame snapshot"; return false;
        }
        __atomic_store_n((uint32_t *)completion.contents,0u,__ATOMIC_RELEASE);
        snapshot.input=frame.input; snapshot.prepared=p;
        snapshot.color=color; snapshot.depth=depth; snapshot.motion=motion; snapshot.output=output;
        snapshot.completion=completion;
        return true;
    }
    bool fit(id<MTLTexture> __strong &target, MTLPixelFormat format, uint32_t w, uint32_t h) {
        if (!target || target.width != w || target.height != h || target.pixelFormat != format) target = texture(format,w,h);
        return target != nil;
    }
    bool ensure_targets(View &v, const Session &s, const Fsr4InputPrepared &p, const Frame &f, const Options &o, std::string &error) {
        if (!v.scratch || v.scratch.length != s.scratchBytes) {
            v.scratch = [device_ newBufferWithLength:s.scratchBytes options:MTLResourceStorageModePrivate];
            v.zeroScratch = true;
        }
        if (v.width != p.alignedUW || v.height != p.alignedUH || !v.history || !v.reprojected || !v.recurrent) {
            v.history = texture(MTLPixelFormatRGBA16Float,p.alignedUW,p.alignedUH);
            v.reprojected = texture(MTLPixelFormatRGBA16Float,p.alignedUW,p.alignedUH);
            v.recurrent = texture(MTLPixelFormatRGBA8Unorm,p.alignedUW,p.alignedUH);
            v.width=p.alignedUW; v.height=p.alignedUH; v.zeroTargets=true;
        }
        if (!v.autoExposure) v.autoExposure=texture(MTLPixelFormatR32Float,2,1);
        bool ok = v.scratch && v.history && v.reprojected && v.recurrent && v.autoExposure;
        if(o.profile==Profile::Lossy) {
            const size_t blocks=size_t(p.alignedUW/4u)*(p.alignedUH/4u);
            if(!v.lossyMotion || v.lossyMotion.length!=blocks*16u)
                v.lossyMotion=[device_ newBufferWithLength:blocks*16u options:MTLResourceStorageModePrivate];
            if(!v.lossyPlane || v.lossyPlane.length!=blocks*4u)
                v.lossyPlane=[device_ newBufferWithLength:blocks*4u options:MTLResourceStorageModePrivate];
            ok &= v.lossyMotion && v.lossyPlane;
        }
        // Float32 canonical copies preserve the values returned by the source
        // texture, including sRGB decode and packed R11G11B10 input precision.
        if (p.copyMask & FSR4_INPUT_COPY_COLOR) ok &= fit(v.colorCopy,MTLPixelFormatRGBA32Float,p.rw,p.rh);
        if ((p.copyMask & FSR4_INPUT_COPY_DEPTH) || depth_format(f.depth.pixelFormat)) ok &= fit(v.depthCopy,MTLPixelFormatR32Float,p.rw,p.rh);
        if (p.copyMask & FSR4_INPUT_COPY_MOTION) ok &= fit(v.motionCopy,MTLPixelFormatRG32Float,p.mvW,p.mvH);
        if (p.copyMask & FSR4_INPUT_COPY_OUTPUT) ok &= fit(v.outputCopy,MTLPixelFormatRGBA32Float,p.uw,p.uh);
        if (o.rcas && o.sharpness > 0) ok &= fit(v.rcasTemp,MTLPixelFormatRGBA16Float,p.alignedUW,p.alignedUH);
        if (p.autoExposure) {
            ok &= fit(v.luma,MTLPixelFormatR32Float,std::max(ceil_div(p.rw,64),1u),std::max(ceil_div(p.rh,64),1u));
            if (!s.splitAutoExposure && !v.spdAtomicTexture) {
                if (!v.spdAtomic) v.spdAtomic=[device_ newBufferWithLength:4096 options:MTLResourceStorageModeShared];
                const NSUInteger row=std::max<NSUInteger>(16,[device_ minimumLinearTextureAlignmentForPixelFormat:MTLPixelFormatR32Uint]);
                auto d=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR32Uint width:1 height:1 mipmapped:NO];
                d.storageMode=MTLStorageModeShared; d.usage=MTLTextureUsageShaderRead|MTLTextureUsageShaderWrite;
                v.spdAtomicTexture=[v.spdAtomic newTextureWithDescriptor:d offset:0 bytesPerRow:row];
            }
            if(!s.splitAutoExposure) ok &= v.spdAtomic && v.spdAtomicTexture;
        }
        if (!ok) error="cannot allocate FSR 4.1.1 textures or scratch";
        return ok;
    }
    bool ensure_helpers(std::string &error) {
        if (!helpers_.empty()) return true;
        NSError *e=nil;
        NSString *source=[NSString stringWithContentsOfFile:ns(root_+"/src/fsr411_helpers.metal") encoding:NSUTF8StringEncoding error:&e];
        if (!source) { error=metal_error(e); return false; }
        MTLCompileOptions *co=[MTLCompileOptions new]; co.languageVersion=MTLLanguageVersion2_3;
        id<MTLLibrary> lib=[device_ newLibraryWithSource:source options:co error:&e];
        if (!lib) { error=metal_error(e); return false; }
        const char *names[]={"fsr411_copy","fsr411_copy_depth","fsr411_clear_texture","fsr411_clear_buffer",
                             "fsr411_snapshot","fsr411_exposure","fsr411_init_exposure","fsr411_retire"};
        std::map<std::string,id<MTLComputePipelineState>> next;
        for (auto name:names) {
            id<MTLFunction> fn=[lib newFunctionWithName:ns(name)];
            auto pso=[device_ newComputePipelineStateWithFunction:fn error:&e];
            if (!pso) { error=metal_error(e); return false; }
            next.emplace(name,pso);
        }
        MTLSamplerDescriptor *sd=[MTLSamplerDescriptor new];
        sd.minFilter=sd.magFilter=MTLSamplerMinMagFilterLinear;
        sd.sAddressMode=sd.tAddressMode=MTLSamplerAddressModeClampToEdge;
        sd.supportArgumentBuffers=YES;
        sampler_=[device_ newSamplerStateWithDescriptor:sd];
        auto d=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR32Float width:1 height:1 mipmapped:NO];
        d.storageMode=MTLStorageModeShared; d.usage=MTLTextureUsageShaderRead;
        exposureOne_=[device_ newTextureWithDescriptor:d];
        float one=1;
        [exposureOne_ replaceRegion:MTLRegionMake2D(0,0,1,1) mipmapLevel:0 withBytes:&one bytesPerRow:4];
        if (!sampler_ || !exposureOne_) { error="cannot allocate common FSR 4.1.1 resources"; return false; }
        helpers_=std::move(next);
        return true;
    }
    bool run(NSArray<NSString *> *args, std::string &error) {
        // Keep stderr in a file: waiting with unread pipes can deadlock a cold
        // shader build. The executable and every argument are passed directly.
        NSString *log=[NSTemporaryDirectory() stringByAppendingPathComponent:[[NSUUID UUID].UUIDString stringByAppendingString:@"-fsr411.log"]];
        [[NSFileManager defaultManager] createFileAtPath:log contents:nil attributes:nil];
        NSFileHandle *handle=[NSFileHandle fileHandleForWritingAtPath:log];
        NSTask *task=[NSTask new]; task.executableURL=[NSURL fileURLWithPath:@"/usr/bin/env"];
        task.arguments=args; task.currentDirectoryURL=[NSURL fileURLWithPath:ns(root_)];
        task.standardOutput=handle; task.standardError=handle;
        NSError *e=nil;
        const double started=[NSDate timeIntervalSinceReferenceDate];
        fsr4_logf("4.1.1 prepare begin: %s",utf8([args componentsJoinedByString:@" "]).c_str());
        if (![task launchAndReturnError:&e]) { [handle closeFile]; error=metal_error(e); return false; }
        [task waitUntilExit]; [handle closeFile];
        if (task.terminationStatus) {
            NSString *text=[NSString stringWithContentsOfFile:log encoding:NSUTF8StringEncoding error:nil];
            error="FSR 4.1.1 shader build failed: "+utf8(text);
            fsr4_logf("%s (build log: %s)",error.c_str(),log.UTF8String);
            return false;
        }
        fsr4_logf("4.1.1 prepare complete cpuMs=%.3f",([NSDate timeIntervalSinceReferenceDate]-started)*1000);
        [[NSFileManager defaultManager] removeItemAtPath:log error:nil];
        return true;
    }
    NSDictionary *json(const std::string &path, std::string &error) {
        NSError *e=nil; NSData *data=[NSData dataWithContentsOfFile:ns(path) options:0 error:&e];
        id result=data?[NSJSONSerialization JSONObjectWithData:data options:0 error:&e]:nil;
        if (![result isKindOfClass:[NSDictionary class]]) { error="invalid FSR 4.1.1 manifest: "+path+": "+metal_error(e); return nil; }
        return result;
    }
    bool load_native(Session &s, bool compile, Kernels kind, std::string &error) {
        const bool useMpp=kind==Kernels::Mpp, useSimd=kind==Kernels::Simd;
        const std::string dir=buildDir_+"/fsr411_native_"+std::to_string(s.tier)+(s.ultraPerformance?"_ultraperformance":"")
            +(s.profile==Profile::Lossy?"_lossy":"");
        // ExactFast differs only in PRE/POST math and shares Exact's network.
        if (compile && !run(@[@"python3",ns(root_+"/tools/gen_fsr411_native.py"),@"--tier",ns(std::to_string(s.tier)),
            @"--preset",s.ultraPerformance?@"ultraperformance":@"normal",@"--profile",s.profile==Profile::Lossy?@"lossy":@"exact",
            @"--outdir",ns(dir)],error)) return false;
        auto manifest=json(dir+"/fsr411_native_manifest.json",error);
        if (!manifest || [manifest[@"abi_version"] unsignedIntValue]!=kAbiVersion
            || [manifest[@"parameter_size"] unsignedIntValue]!=sizeof(PassParams)) { if(error.empty()) error="native 4.1.1 ABI mismatch"; return false; }
        NSData *weights=[NSData dataWithContentsOfFile:ns(dir+"/fsr411_native_weights.bin")];
        NSData *params=[NSData dataWithContentsOfFile:ns(dir+"/fsr411_native_params.bin")];
        if (!weights.length || !params.length) { error="missing native 4.1.1 weights/parameters"; return false; }
        if(weights.length != [manifest[@"weights_bytes"] unsignedLongLongValue]) {
            error="native 4.1.1 weight cache is incomplete; regenerate this model";return false;
        }
        if(manifest[@"params_bytes"] && params.length != [manifest[@"params_bytes"] unsignedLongLongValue]) {
            error="native 4.1.1 parameter cache is incomplete; regenerate this model";return false;
        }
        s.nativeWeights=[device_ newBufferWithBytes:weights.bytes length:weights.length options:MTLResourceStorageModeShared];
        if (useSimd) {
            // Same element indices as the INT8 blob; bias words become unused
            // halves (biases are read as int32 from nativeWeights).
            std::vector<_Float16> widened(weights.length);
            const int8_t *bytes=(const int8_t *)weights.bytes;
            for (size_t i=0;i<widened.size();++i) widened[i]=(_Float16)bytes[i];
            s.halfWeights=[device_ newBufferWithBytes:widened.data() length:widened.size()*sizeof(_Float16) options:MTLResourceStorageModeShared];
            if (!s.halfWeights) { error="cannot allocate simd half weights"; return false; }
        }
        // Native sources include only the ABI header. Compile the saved source
        // as a real Metal library; no half-network substitution is possible.
        const char *file=useMpp?"/src/fsr411_mpp.metal":useSimd?"/src/fsr411_simd.metal":"/src/fsr411_reference.metal";
        NSString *source=[NSString stringWithContentsOfFile:ns(root_+file) encoding:NSUTF8StringEncoding error:nil];
        NSString *header=[NSString stringWithContentsOfFile:ns(root_+"/src/fsr411_model.h") encoding:NSUTF8StringEncoding error:nil];
        if (!source || !header) { error="missing native 4.1.1 Metal source"; return false; }
        source=[source stringByReplacingOccurrencesOfString:@"#include \"fsr411_model.h\"" withString:header];
        NSError *e=nil; MTLCompileOptions *co=[MTLCompileOptions new];
        co.languageVersion=MTLLanguageVersion3_0;
        if(useMpp) if (@available(macOS 26.0,*)) co.languageVersion=MTLLanguageVersion4_0;
        const bool specialize=useMpp||useSimd;
        co.preprocessorMacros=@{@"FSR411_MAXW":@(s.tier*8/9),@"FSR411_MAXH":@(s.tier/2),
                               @"FSR411_SPECIALIZE":@(specialize)};
        id<MTLLibrary> lib=[device_ newLibraryWithSource:source options:co error:&e];
        if (!lib) { error=metal_error(e); return false; }
        if (useSimd && !verify_simd(lib,error)) return false;
        std::map<std::string,id<MTLComputePipelineState>> states;
        NSMutableDictionary *passes=[manifest[@"passes"] mutableCopy];
        const size_t appendBase=(s.scratchBytes+255)&~size_t(255);
        if(s.externalPostCnb) {
            NSDictionary *post=manifest[@"auxiliary_passes"][@"post_cnb"];
            if(!post || ![post[@"output_offset_base"] isEqual:@"scratch_append"]) { error="missing native 4.1.1 POST CNB";return false; }
            passes[@"13"]=post;
            s.scratchBytes=appendBase+[post[@"scratch_append_bytes"] unsignedLongLongValue];
        }
        for (NSString *key in passes) {
            NSDictionary *entry=passes[key]; NativePass np;
            NSDictionary *dispatch=useMpp?entry:useSimd?entry[@"simd"]:entry[@"reference"];
            if (!dispatch) { error="native 4.1.1 manifest predates the simd kernels; regenerate this model"; return false; }
            const std::string name=utf8(useMpp?entry[@"kernel"]:dispatch[@"function"]);
            np.offset=[entry[@"parameter_offset"] unsignedIntValue];
            if (np.offset+sizeof(PassParams)>params.length || kNative+np.offset+sizeof(PassParams)>8192) { error="native parameter region overflow"; return false; }
            memcpy(&np.parameters,(const uint8_t *)params.bytes+np.offset,sizeof(PassParams));
            // Shared entry names have different weights, strides and epilogues
            // for each logical pass. Dimensions remain per-frame parameters.
            const std::string pipelineKey=name+(specialize?"/"+utf8(key):"");
            if (!states.count(pipelineKey)) {
                id<MTLFunction> fn=nil;
                if (specialize) {
                    const auto metadata=specialize_metadata(np.parameters,[key intValue]==13);
                    MTLFunctionConstantValues *values=[MTLFunctionConstantValues new];
                    [values setConstantValues:metadata.values type:MTLDataTypeUInt4
                                    withRange:NSMakeRange(0,kPassSpecializationCount)];
                    fn=[lib newFunctionWithName:ns(name) constantValues:values error:&e];
                } else fn=[lib newFunctionWithName:ns(name)];
                auto pso=fn?[device_ newComputePipelineStateWithFunction:fn error:&e]:nil;
                if (!pso) { error=metal_error(e); return false; }
                if (pso.maxTotalThreadsPerThreadgroup < [dispatch[@"threads"] unsignedIntValue]) {
                    error="pipeline "+name+" supports fewer threads than its tile needs"; return false;
                }
                states.emplace(pipelineKey,pso);
            }
            np.pso=states.at(pipelineKey);
            if ([key intValue]==13) np.parameters.output.offset+=uint32_t(appendBase);
            np.inputSlot=[entry[@"input_slot"] unsignedIntValue]; np.outputSlot=[entry[@"output_slot"] unsignedIntValue];
            np.hasSkip=entry[@"skip_slot"]!=nil; np.skipSlot=[entry[@"skip_slot"] unsignedIntValue];
            np.tileX=[dispatch[@"tile"][0] unsignedIntValue]; np.tileY=[dispatch[@"tile"][1] unsignedIntValue];
            np.threads=[dispatch[@"threads"] unsignedIntValue]; np.inputDispatch=[entry[@"dispatch_space"] isEqual:@"input"];
            if (np.inputSlot>=17 || np.outputSlot>=17 || np.skipSlot>=17 || !np.tileX || !np.tileY || !np.threads) { error="invalid native dispatch metadata"; return false; }
            s.native.emplace([key intValue],np);
        }
        return s.nativeWeights && !s.native.empty();
    }
    Session *ensure_session(int tier, const Fsr4InputPrepared &p, const Options &o, Kernels kind, std::string &error) {
        const bool portable=kind!=Kernels::Reference;
        const bool externalPost=portable || o.profile==Profile::Lossy;
        const std::string key="fsr411_"+std::to_string(tier)+"_di"+std::to_string(p.depthReversed)
            +"_lm"+std::to_string(p.lowResMV)+"_jm"+std::to_string(p.jitteredMV)+"_cs"+std::to_string(o.colorspace)
            +"_ae"+std::to_string(p.autoExposure)+"_o"+std::to_string(o.optimize)
            +(o.ultraPerformance?"_up1":"")+(o.profile==Profile::Lossy?"_lossy":o.profile==Profile::ExactFast?"_fast":"")
            +(externalPost?"_pc1":"")+(o.strictMath&&o.profile!=Profile::Exact?"_sm1":"");
        const std::string cacheKey=key+(kind==Kernels::Mpp?"_mpp":kind==Kernels::Simd?"_simd":kind==Kernels::Scalar?"_scalar":"_ref");
        auto existing=sessions_.find(cacheKey); if(existing!=sessions_.end()) return existing->second.get();
        if (o.allowCompile && !run(@[@"python3",ns(root_+"/tools/compile_fsr411.py"),@"--root",ns(root_),@"--builddir",ns(buildDir_),
            @"--res",ns(std::to_string(tier)),@"--depth-inverted",ns(std::to_string(p.depthReversed)),
            @"--low-res-mv",ns(std::to_string(p.lowResMV)),@"--jittered-mv",ns(std::to_string(p.jitteredMV)),
            @"--colorspace",ns(std::to_string(o.colorspace)),@"--auto-exposure",ns(std::to_string(p.autoExposure)),
            @"--optimize",ns(std::to_string(o.optimize)),@"--ultra-performance",o.ultraPerformance?@"1":@"0",
            @"--profile",@(profile_name(o.profile)),@"--external-post-cnb",externalPost?@"1":@"0",
            @"--strict-math",o.strictMath&&o.profile!=Profile::Exact?@"1":@"0"],error)) return nullptr;
        auto manifest=json(buildDir_+"/"+key+"/bindings.json",error);
        if (!manifest || ![manifest[@"version"] isEqual:@"4.1.1"] || [manifest[@"order"] count]!=27) return nullptr;
        auto s=std::make_unique<Session>(); s->key=cacheKey; s->tier=tier;
        s->postPixelsPerGroup=manifest[@"post_pixels_per_group"]?[manifest[@"post_pixels_per_group"] unsignedIntValue]:32;
        if(s->postPixelsPerGroup!=16 && s->postPixelsPerGroup!=32) {error="invalid POST dispatch geometry";return nullptr;}
        s->splitAutoExposure=[manifest[@"auto_exposure_dispatches"] unsignedIntValue]==2;
        s->ultraPerformance=o.ultraPerformance;s->externalPostCnb=externalPost;s->profile=o.profile;
        s->scratchBytes=[manifest[@"scratch_bytes"] unsignedLongLongValue];
        NSData *init=[NSData dataWithContentsOfFile:ns(root_+"/third_party/fsr411/"+
            (o.ultraPerformance?"initializers-ultraperformance.bin":"initializers.bin"))];
        if (init.length!=131072 || !s->scratchBytes) { error="missing real FSR 4.1.1 initializer"; return nullptr; }
        s->initializer=[device_ newBufferWithBytes:init.bytes length:init.length options:MTLResourceStorageModeShared];
        if (!s->initializer) { error="cannot allocate FSR 4.1.1 initializer"; return nullptr; }
        for (NSString *stage in manifest[@"order"]) s->order.push_back(utf8(stage));
        if (externalPost && !load_native(*s,o.allowCompile,kind,error)) return nullptr;
        NSDictionary *entries=manifest[@"kernels"];
        for (NSString *name in entries) {
            NSDictionary *entry=entries[name]; NSError *e=nil;
            // Native stages never execute the generated scalar shader. Avoid
            // compiling unused PSOs (and their large constant-folding graphs).
            const std::string stage=utf8(name);
            const int logical=stage.rfind("pass",0)==0?std::atoi(stage.c_str()+4):0;
            if(portable && logical && s->native.count(logical)) {s->kernels.emplace(stage,Kernel{});continue;}
            auto url=[NSURL fileURLWithPath:ns(buildDir_+"/"+key+"/"+utf8(entry[@"library"]))];
            auto lib=[device_ newLibraryWithURL:url error:&e];
            if (!lib) { error=metal_error(e); return nullptr; }
            auto fn=[lib newFunctionWithName:entry[@"entry"]];
            auto pso=fn?[device_ newComputePipelineStateWithFunction:fn error:&e]:nil;
            if (!pso) { error="pipeline "+utf8(name)+": "+metal_error(e); return nullptr; }
            Kernel k; k.pso=pso; NSArray *wg=entry[@"wgsize"];
            if(NSString *tables=entry[@"literal_tables"]) {
                NSData *bytes=[NSData dataWithContentsOfFile:ns(buildDir_+"/"+key+"/"+utf8(tables))];
                if(!bytes.length) {error="missing shader literal tables for "+stage;return nullptr;}
                k.literalTables=[device_ newBufferWithBytes:bytes.bytes length:bytes.length options:MTLResourceStorageModeShared];
                if(!k.literalTables) {error="cannot allocate shader literal tables";return nullptr;}
            }
            k.threads=MTLSizeMake([wg[0] unsignedIntValue],[wg[1] unsignedIntValue],[wg[2] unsignedIntValue]);
            for (NSArray *a in entry[@"buffers"]) k.buffers.emplace_back([a[0] unsignedIntegerValue],utf8(a[1]));
            for (NSArray *a in entry[@"textures"]) k.textures.emplace_back([a[0] unsignedIntegerValue],utf8(a[1]));
            for (NSArray *a in entry[@"samplers"]) k.samplers.emplace_back([a[0] unsignedIntegerValue],utf8(a[1]));
            s->kernels.emplace(utf8(name),std::move(k));
        }
        for (const auto &name:s->order) if(!s->kernels.count(name)) { error="incomplete 4.1.1 graph: "+name; return nullptr; }
        if (!s->kernels.count("ae") || !s->kernels.count("rcas")) { error="missing 4.1.1 AE/RCAS kernels"; return nullptr; }
        if(s->splitAutoExposure && !s->kernels.count("ae_finalize")) {error="missing 4.1.1 AE final reduction";return nullptr;}
        auto result=s.get(); sessions_.emplace(cacheKey,std::move(s)); return result;
    }
    FrameSlot *acquire_slot(View &v, std::string &error) {
        for (auto &slot:v.slots) if(slot->retired()) {
            slot->busy=false; [slot->resources removeAllObjects];
            slot->snapshot.reset();
            slot->stageProfiler=nullptr;
            if (slot->residency) { [slot->residency removeAllAllocations]; [slot->residency commit]; }
            return slot.get();
        }
        if (v.slots.size()>=32) { error="32 FSR frames are still in flight; wait for their command buffers"; return nullptr; }
        auto slot=std::make_unique<FrameSlot>();
        slot->constants=[device_ newBufferWithLength:FrameSlot::kSize options:MTLResourceStorageModeShared];
        slot->tables=[NSMutableDictionary new]; slot->resources=[NSMutableArray new];
        if (!slot->constants) { error="cannot allocate per-frame FSR constants"; return nullptr; }
        memset(slot->constants.contents,0,FrameSlot::kSize);
        auto result=slot.get(); v.slots.push_back(std::move(slot)); return result;
    }
    template <typename T> static void put(FrameSlot &slot, NSUInteger offset, const T &value) {
        memcpy((uint8_t *)slot.constants.contents+offset,&value,sizeof(T));
    }
    void fill_constants(FrameSlot &slot, const Session &s, const Fsr4InputPrepared &p, const TensorSizes &sizes, bool rcas, float sharpness) {
        FrameConstants c={};
        c.inv_size[0]=1.f/p.alignedUW; c.inv_size[1]=1.f/p.alignedUH;
        c.scale[0]=float(p.alignedUW)/p.rw; c.scale[1]=float(p.alignedUH)/p.rh;
        c.inv_scale[0]=float(p.rw)/p.alignedUW; c.inv_scale[1]=float(p.rh)/p.alignedUH;
        c.tex_size[0]=p.alignedUW; c.tex_size[1]=p.alignedUH;
        c.max_renderSize[0]=p.maxInputWidth; c.max_renderSize[1]=p.maxInputHeight;
        memcpy(c.jitter,p.jitter,sizeof c.jitter); memcpy(c.mv_scale,p.mvScale,sizeof c.mv_scale);
        memcpy(c.jitterCancellation,p.jitterCancellation,sizeof c.jitterCancellation);
        c.width=p.alignedUW; c.height=p.alignedUH; c.width_lr=p.rw; c.height_lr=p.rh;
        c.reset=p.reset; c.preExposure=p.preExposure; c.previousPreExposure=p.previousPreExposure;
        c.rcas=rcas; c.sharpness=sharpness; put(slot,0,c); put(slot,kTensor,sizes);
        SpdConstants ae={}; ae.mips=std::clamp(uint32_t(std::floor(std::log2(std::max(p.rw,p.rh)))),1u,12u);
        ae.groups=ceil_div(p.rw,64)*ceil_div(p.rh,64); ae.invInputSize[0]=1.f/p.rw; ae.invInputSize[1]=1.f/p.rh;
        ae.preExposure=p.preExposure; ae.validSize[0]=p.rw; ae.validSize[1]=p.rh; put(slot,kSpd,ae);
        RcasConstants rc={}; const float value=std::exp2(2.f*sharpness-2.f);
        memcpy(&rc.config[0],&value,sizeof value); const _Float16 half=(_Float16)value; uint16_t h;
        memcpy(&h,&half,sizeof h); rc.config[1]=uint32_t(h)|(uint32_t(h)<<16); rc.preExposure=p.preExposure; put(slot,kRcas,rc);
        for (const auto &item:s.native) {
            const auto &n=item.second; auto param=n.parameters;
            param.input.width=sizes.v[n.inputSlot][0]; param.input.height=sizes.v[n.inputSlot][1];
            param.output.width=sizes.v[n.outputSlot][0]; param.output.height=sizes.v[n.outputSlot][1];
            if (n.hasSkip) { param.skip.width=sizes.v[n.skipSlot][0]; param.skip.height=sizes.v[n.skipSlot][1]; }
            put(slot,kNative+n.offset,param);
        }
    }
    bool table(FrameSlot &slot, const std::string &name, std::string &error) {
        if(slot.tables[ns(name)]) return true;
        if (@available(macOS 26.0,*)) {
            MTL4ArgumentTableDescriptor *d=[MTL4ArgumentTableDescriptor new];
            d.maxBufferBindCount=8; d.maxTextureBindCount=16; d.maxSamplerStateBindCount=4; d.initializeBindings=YES;
            NSError *e=nil; auto t=[device_ newArgumentTableWithDescriptor:d error:&e];
            if (!t) { error=metal_error(e); return false; }
            slot.tables[ns(name)]=t; return true;
        }
        error="Metal 4 argument tables unavailable"; return false;
    }
    bool prepare_tables(FrameSlot &slot, const Session &s, const View &v, std::string &error) {
        for (const auto &k:s.kernels) if(!table(slot,k.first,error)) return false;
        for (const auto &n:s.native) if(!table(slot,"native"+std::to_string(n.first),error)) return false;
        const char *names[]={"clearScratch","clearHistory","clearReprojected","clearRecurrent","initExposure","copyColor",
            "copyDepth","copyMotion","copyOutput","clearAtomic","clearLuma","clearLossyPlane","updateExposure","retire"};
        for (auto n:names) if(!table(slot,n,error)) return false;
        if (slot.snapshot) {
            const char *snapshotNames[]={"snapshotColor","snapshotDepth","snapshotMotion","snapshotOutput","snapshotDone"};
            for (auto n:snapshotNames) if(!table(slot,n,error)) return false;
        }
        for (const auto &c:v.captures) if(!table(slot,"capture_"+c.first,error)) return false;
        return true;
    }
    void retain_resources(FrameSlot &slot,const Session &s,const View &v,const Frame &f) {
        // The owner may resize on the next CPU encode while this frame is on
        // the GPU. Keep the old resources alive until its retirement token.
        const id resources[]={slot.constants,s.initializer,s.nativeWeights,s.halfWeights,v.history,v.reprojected,v.recurrent,v.autoExposure,
            v.scratch,v.spdAtomic,v.spdAtomicTexture,v.luma,v.colorCopy,v.depthCopy,v.motionCopy,v.outputCopy,v.rcasTemp,v.lossyMotion,v.lossyPlane,
            f.color,f.depth,f.motion,f.output,f.exposure,exposureOne_};
        for (id resource:resources) if(resource && ![slot.resources containsObject:resource]) [slot.resources addObject:resource];
        for(const auto &k:s.kernels) if(k.second.literalTables) [slot.resources addObject:k.second.literalTables];
        for (const auto &c:v.captures) [slot.resources addObject:c.second];
        if (slot.snapshot) {
            const auto &snapshot=*slot.snapshot;
            [slot.resources addObjectsFromArray:@[snapshot.color,snapshot.depth,snapshot.motion,snapshot.output,snapshot.completion]];
        }
    }
    bool bind_residency(FrameSlot &slot,id cb,std::string &error) {
        if (@available(macOS 26.0,*)) {
            if (!slot.residency) {
                NSError *e=nil; slot.residency=[device_ newResidencySetWithDescriptor:[MTLResidencySetDescriptor new] error:&e];
                if (!slot.residency) { error=metal_error(e); return false; }
            }
            for (id resource in slot.resources) [slot.residency addAllocation:(id<MTLAllocation>)resource];
            [slot.residency commit];
            if (![cb respondsToSelector:@selector(useResidencySet:)]) { error="Metal 4 command buffer lacks residency API"; return false; }
            [cb useResidencySet:slot.residency]; return true;
        }
        error="Metal 4 residency unavailable"; return false;
    }
    static void barrier(id enc,bool m4) {
        if (m4) [enc barrierAfterEncoderStages:MTLStageDispatch|MTLStageBlit beforeEncoderStages:MTLStageDispatch|MTLStageBlit visibilityOptions:MTL4VisibilityOptionDevice];
        else [(id<MTLComputeCommandEncoder>)enc memoryBarrierWithScope:MTLBarrierScopeBuffers|MTLBarrierScopeTextures];
    }
    static void queue_before(id enc) {
        [enc barrierAfterQueueStages:MTLStageAll beforeStages:MTLStageDispatch|MTLStageBlit visibilityOptions:MTL4VisibilityOptionDevice];
    }
    static void queue_after(id enc) {
        [enc barrierAfterStages:MTLStageDispatch|MTLStageBlit beforeQueueStages:MTLStageAll visibilityOptions:MTL4VisibilityOptionDevice];
    }
    void helper(id __strong &enc,bool m4,FrameSlot &slot,const std::string &tag,const std::string &name,
                std::initializer_list<BufferBinding> buffers,std::initializer_list<id<MTLTexture>> textures,MTLSize grid) {
        if(slot.stageProfiler) slot.stageProfiler->begin_stage(enc,tag);
        auto pso=helpers_.at(name); NSUInteger index=0;
        if (m4) {
            id<MTL4ArgumentTable> t=slot.tables[ns(tag)];
            for(auto b:buffers) [t setAddress:b.buffer.gpuAddress+b.offset atIndex:index++];
            index=0; for(auto tx:textures) [t setTexture:tx.gpuResourceID atIndex:index++];
            [enc setArgumentTable:t];
        } else {
            auto ce=(id<MTLComputeCommandEncoder>)enc;
            for(auto b:buffers) [ce setBuffer:b.buffer offset:b.offset atIndex:index++];
            index=0; for(auto tx:textures) [ce setTexture:tx atIndex:index++];
        }
        [enc setComputePipelineState:pso];
        MTLSize threads=grid.height>1?MTLSizeMake(8,8,1):MTLSizeMake(std::min<NSUInteger>(256,grid.width),1,1);
        [enc dispatchThreads:grid threadsPerThreadgroup:threads]; barrier(enc,m4);
        if(slot.stageProfiler) slot.stageProfiler->end_stage(enc);
    }
    void clear_texture(id __strong &enc,bool m4,FrameSlot &s,const std::string &tag,id<MTLTexture> t) {
        helper(enc,m4,s,tag,"fsr411_clear_texture",{},{t},MTLSizeMake(t.width,t.height,1));
    }
    void clear_buffer(id __strong &enc,bool m4,FrameSlot &s,const std::string &tag,id<MTLBuffer> b,NSUInteger off) {
        uint32_t count=uint32_t(b.length/16); put(s,off,count);
        helper(enc,m4,s,tag,"fsr411_clear_buffer",{{b,0},{s.constants,off}},{},MTLSizeMake(count,1,1));
    }
    void copy(id __strong &enc,bool m4,FrameSlot &s,const std::string &tag,id<MTLTexture> from,id<MTLTexture> to,
              Fsr4InputRect rect,uint32_t x,uint32_t y,NSUInteger off,bool depth) {
        RegionConstants r={{rect.x,rect.y,rect.width,rect.height},{x,y,0,0}}; put(s,off,r);
        helper(enc,m4,s,tag,depth?"fsr411_copy_depth":"fsr411_copy",{{s.constants,off}},{from,to},MTLSizeMake(rect.width,rect.height,1));
    }
    id<MTLTexture> bind_texture(const std::string &name,const Bindings &b) {
        if(name=="r_input_color") return b.color;
        if(name=="r_depth") return b.depth;
        if(name=="r_velocity") return b.motion;
        if(name=="r_input_exposure") return b.exposure;
        if(name=="r_auto_exposure_texture" || name=="rw_auto_exposure_texture") return b.view->autoExposure;
        if(name=="rw_spd_global_atomic") return b.view->spdAtomicTexture;
        if(name=="rw_autoexp_mip_5") return b.view->luma;
        if(name=="r_recurrent_0" || name=="rw_recurrent_0") return b.view->recurrent;
        if(name=="r_history_color" || name=="rw_history_color") return b.view->history;
        if(name=="r_reprojected_color" || name=="rw_reprojected_color") return b.view->reprojected;
        if(name=="rw_mlsr_output_color") return b.rcas?b.view->rcasTemp:b.output;
        if(name=="r_rcas_input") return b.view->rcasTemp;
        if(name=="rw_rcas_output") return b.output;
        [NSException raise:@"FSR411Binding" format:@"Unknown texture %s",name.c_str()]; return nil;
    }
    BufferBinding bind_buffer(const std::string &stage,const std::string &name,FrameSlot &slot,const Session &s,const Bindings &b) {
        if(name=="ScratchBuffer") return {b.view->scratch,0};
        if(name=="Fsr411Tables") return {s.kernels.at(stage).literalTables,0};
        if(name=="Fsr411LossyConstants") return {slot.constants,kLossy};
        if(name=="Fsr411LossyMotion") return {b.view->lossyMotion,0};
        if(name=="Fsr411LossyPlane") return {b.view->lossyPlane,0};
        if(name=="InitializerBuffer" || name=="t18_initializer") return {s.initializer,0};
        if(name=="MLSR_Optimized_Constants") return {slot.constants,(stage=="pre" || stage=="post")?0:kTensor};
        if(name=="CsTensorSizes") return {slot.constants,kTensor};
        if(name=="AutoExposureSPDConstants") return {slot.constants,kSpd};
        if(name=="cbRCAS") return {slot.constants,kRcas};
        if(name=="rw_spd_global_atomic_atomic") return {b.view->spdAtomic,0};
        [NSException raise:@"FSR411Binding" format:@"Unknown buffer %s",name.c_str()]; return {nil,0};
    }
    void dispatch(id __strong &enc,bool m4,FrameSlot &slot,const Session &s,const std::string &name,MTLSize groups,const Bindings &b) {
        if(slot.stageProfiler) slot.stageProfiler->begin_stage(enc,name);
        const auto &k=s.kernels.at(name);
        if(m4) {
            id<MTL4ArgumentTable> t=slot.tables[ns(name)];
            for(const auto &bb:k.buffers) {auto binding=bind_buffer(name,bb.second,slot,s,b); [t setAddress:binding.buffer.gpuAddress+binding.offset atIndex:bb.first];}
            for(const auto &tx:k.textures) [t setTexture:bind_texture(tx.second,b).gpuResourceID atIndex:tx.first];
            for(const auto &sm:k.samplers) [t setSamplerState:sampler_.gpuResourceID atIndex:sm.first];
            [enc setArgumentTable:t];
        } else {
            auto ce=(id<MTLComputeCommandEncoder>)enc;
            for(const auto &bb:k.buffers) {auto binding=bind_buffer(name,bb.second,slot,s,b); [ce setBuffer:binding.buffer offset:binding.offset atIndex:bb.first];}
            for(const auto &tx:k.textures) [ce setTexture:bind_texture(tx.second,b) atIndex:tx.first];
            for(const auto &sm:k.samplers) [ce setSamplerState:sampler_ atIndex:sm.first];
        }
        [enc setComputePipelineState:k.pso]; [enc dispatchThreadgroups:groups threadsPerThreadgroup:k.threads]; barrier(enc,m4);
        if(slot.stageProfiler) slot.stageProfiler->end_stage(enc);
    }
    void dispatch_native(id __strong &enc,bool m4,FrameSlot &slot,const Session &s,int pass,const View &v,const TensorSizes &sizes) {
        if(slot.stageProfiler) slot.stageProfiler->begin_stage(enc,pass==13?"post_cnb":"pass"+std::to_string(pass));
        const auto &n=s.native.at(pass); const unsigned shape=n.inputDispatch?n.inputSlot:n.outputSlot;
        if(m4) {
            id<MTL4ArgumentTable> t=slot.tables[ns("native"+std::to_string(pass))];
            [t setAddress:v.scratch.gpuAddress atIndex:0]; [t setAddress:s.nativeWeights.gpuAddress atIndex:1];
            [t setAddress:slot.constants.gpuAddress+kNative+n.offset atIndex:2];
            if (s.halfWeights) [t setAddress:s.halfWeights.gpuAddress atIndex:3];
            [enc setArgumentTable:t];
        } else {
            auto ce=(id<MTLComputeCommandEncoder>)enc;
            [ce setBuffer:v.scratch offset:0 atIndex:0]; [ce setBuffer:s.nativeWeights offset:0 atIndex:1];
            [ce setBuffer:slot.constants offset:kNative+n.offset atIndex:2];
            if (s.halfWeights) [ce setBuffer:s.halfWeights offset:0 atIndex:3];
        }
        [enc setComputePipelineState:n.pso];
        [enc dispatchThreadgroups:MTLSizeMake(ceil_div(sizes.v[shape][0],n.tileX),ceil_div(sizes.v[shape][1],n.tileY),1)
            threadsPerThreadgroup:MTLSizeMake(n.threads,1,1)]; barrier(enc,m4);
        if(slot.stageProfiler) slot.stageProfiler->end_stage(enc);
    }
};
} // namespace fsr411
