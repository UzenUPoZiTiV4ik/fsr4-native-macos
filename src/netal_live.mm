// SPDX-License-Identifier: MIT
// Optional DLSS SR bridge. The caller owns submission and presentation.
#include "netal_live.h"
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

static thread_local std::string g_profile, g_render;
extern "C" const char *fsr4_netal_geometry_profile(void) { return g_profile.c_str(); }
extern "C" const char *fsr4_netal_geometry_render(void) { return g_render.c_str(); }

#ifndef FSR4_WITH_NETAL
int fsr4_netal_encode_live_frame(const Fsr4LiveFrameV2 *, void *) {
    fsr4_live_set_inactive("Netal SR is not linked; rebuild with --netal-source");
    return 0;
}
void fsr4_netal_invalidate_owner(void *, void *) {}
void fsr4_netal_release_owner(void *, void *) {}
#else
#import "dlss_sr_session.h"
#import "dlss_sr_texture_adapter.h"

namespace {
struct GeometryScope {
    std::string oldProfile = g_profile, oldRender = g_render;
    GeometryScope(const Fsr4LiveFrame &f) {
        g_profile = std::to_string(f.uw) + "x" + std::to_string(f.uh);
        g_render = std::to_string(f.rw) + "x" + std::to_string(f.rh);
    }
    ~GeometryScope() { g_profile = oldProfile; g_render = oldRender; }
};
struct View {
    id<MTLDevice> device = nil;
    Fsr4NetalSRSession *session = nil;
    Fsr4NetalTextureAdapter *adapter = nil;
    id<MTLBuffer> completion = nil;
    id<MTLComputePipelineState> retirePipeline = nil;
    id<MTLResidencySet> residency = nil;
    id<MTL4ArgumentTable> retireTable = nil;
    NSArray *resources = nil;
    bool busy = false, needsReset = true, failed = false;
    uint32_t rw = 0, rh = 0, uw = 0, uh = 0;
    std::string bundle;
    bool finished() const {
        return !busy || (completion && __atomic_load_n(
            (uint32_t *)completion.contents, __ATOMIC_ACQUIRE) != 0u);
    }
};
using Key = std::pair<uintptr_t, uintptr_t>;
std::mutex mutex;
std::map<Key, std::shared_ptr<View>> views;
std::vector<std::shared_ptr<View>> retired;
struct Failure {
    std::string identity, reason;
    std::chrono::steady_clock::time_point retry;
};
std::map<Key, Failure> failures;
Fsr4LiveStatus status = {};
std::chrono::steady_clock::time_point lastEncode;

Key key(void *device, void *owner) { return {(uintptr_t)device, (uintptr_t)owner}; }
std::string bundle_path() {
    const char *configured = std::getenv("FSR4_NETAL_RUNTIME");
    if (configured && *configured) return configured;
    const char *root = std::getenv("FSR4_ROOT");
    return root && *root ? std::string(root) + "/build/netal_runtime" : "";
}
void collect() {
    for (auto it = retired.begin(); it != retired.end();) {
        if ((*it)->finished()) it = retired.erase(it);
        else ++it;
    }
}
void retire(std::shared_ptr<View> view) {
    if (view && !view->finished()) retired.push_back(std::move(view));
}
int reject(const char *reason) { fsr4_live_set_inactive(reason); return 0; }
int reject_error(NSError *error, const char *fallback) {
    const char *message = error.localizedDescription.UTF8String;
    char why[256];
    snprintf(why, sizeof why, "Netal: %s", message && *message ? message : fallback);
    fprintf(stderr, "[fsr4] %s\n", why);
    const char *path = std::getenv("FSR4_HOOK_LOG");
    if (path && *path) if (FILE *log = fopen(path, "a")) {
        fprintf(log, "[fsr4] %s\n", why); fclose(log);
    }
    return reject(why);
}
std::shared_ptr<View> create(const Fsr4LiveFrame &f, const std::string &path, NSError **error) {
    auto view = std::make_shared<View>();
    view->device = (__bridge id<MTLDevice>)f.device;
    view->bundle = path;
    view->rw = f.rw; view->rh = f.rh; view->uw = f.uw; view->uh = f.uh;
    GeometryScope scope(f);
    NSURL *url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:path.c_str()] isDirectory:YES];
    NSData *manifestData = [NSData dataWithContentsOfURL:[url URLByAppendingPathComponent:@"manifest.json"]];
    id manifest = manifestData ? [NSJSONSerialization JSONObjectWithData:manifestData options:0 error:error] : nil;
    if (![manifest isKindOfClass:NSDictionary.class]
        || ![manifest[@"schema"] isEqual:@"netal.sr.runtime.v2"]
        || ![manifest[@"model"] isEqual:@"DLSS_SR_Feature1_Quality_K"]) {
        if (error) *error = [NSError errorWithDomain:@"fsr4.netal" code:1
            userInfo:@{NSLocalizedDescriptionKey:@"complete netal.sr.runtime.v2 Preset K bundle required"}];
        return {};
    }
    // The upstream loader verifies the full v2 inventory, model identities,
    // scalar tables and pointer sanitization before loading any GPU kernels.
    view->session = [[Fsr4NetalSRSession alloc] initWithDevice:view->device runtimeBundleURL:url error:error];
    if (!view->session) return {};
    if (view->session.inputWidth != f.rw || view->session.inputHeight != f.rh
        || view->session.outputWidth != f.uw || view->session.outputHeight != f.uh) return {};
    view->adapter = [[Fsr4NetalTextureAdapter alloc] initWithDevice:view->device session:view->session error:error];
    if (!view->adapter) return {};
    // A completion word gives both Metal APIs an exact resource-lifetime
    // boundary. No wall-time guess and no hidden queue/commit/completion hook.
    NSString *source = @"#include <metal_stdlib>\nusing namespace metal;\n"
        "kernel void fsr4_netal_retire(device atomic_uint *done [[buffer(0)]]) {"
        "atomic_store_explicit(done,1u,memory_order_relaxed);}\n";
    MTLCompileOptions *options = [MTLCompileOptions new];
    options.languageVersion = MTLLanguageVersion3_0;
    id<MTLLibrary> library = [view->device newLibraryWithSource:source options:options error:error];
    id<MTLFunction> function = [library newFunctionWithName:@"fsr4_netal_retire"];
    if (!function) return {};
    view->retirePipeline = [view->device newComputePipelineStateWithFunction:function error:error];
    view->completion = [view->device newBufferWithLength:256 options:MTLResourceStorageModeShared];
    if (!view->retirePipeline || !view->completion) return {};
    *static_cast<uint32_t *>(view->completion.contents) = 1;
    if (@available(macOS 26.0, *)) {
        MTLResidencySetDescriptor *rd = [MTLResidencySetDescriptor new];
        rd.initialCapacity = 8;
        view->residency = [view->device newResidencySetWithDescriptor:rd error:error];
        MTL4ArgumentTableDescriptor *td = [MTL4ArgumentTableDescriptor new];
        td.maxBufferBindCount = 1;
        view->retireTable = [view->device newArgumentTableWithDescriptor:td error:error];
        if (!view->residency || !view->retireTable) return {};
        [view->retireTable setAddress:view->completion.gpuAddress atIndex:0];
    }
    return view;
}
bool compatible(const View &v, const Fsr4LiveFrame &f, const std::string &path) {
    return v.rw == f.rw && v.rh == f.rh && v.uw == f.uw && v.uh == f.uh && v.bundle == path;
}
bool encode_retirement(View &view, id command, bool metal4) {
    if (metal4) {
        id<MTL4CommandBuffer> cb = command;
        [cb useResidencySet:view.residency];
        id<MTL4ComputeCommandEncoder> enc = [cb computeCommandEncoder];
        if (!enc) return false;
        [enc barrierAfterQueueStages:MTLStageAll beforeStages:MTLStageDispatch visibilityOptions:MTL4VisibilityOptionDevice];
        [enc setComputePipelineState:view.retirePipeline];
        [enc setArgumentTable:view.retireTable];
        [enc dispatchThreads:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)];
        [enc endEncoding];
    } else {
        id<MTLCommandBuffer> cb = command;
        // Metal 3 has a completion callback after all encoded work. Retain
        // the adapter/session even with an unretained command buffer.
        id<MTLBuffer> done = view.completion;
        NSArray *resources = view.resources;
        Fsr4NetalTextureAdapter *adapter = view.adapter;
        [cb addCompletedHandler:^(id<MTLCommandBuffer> completed) {
            (void)resources; (void)adapter; // strong block captures retain these until completion
            __atomic_store_n((uint32_t *)done.contents,
                completed.status == MTLCommandBufferStatusCompleted ? 1u : 2u, __ATOMIC_RELEASE);
        }];
    }
    return true;
}
} // namespace

int fsr4_netal_encode_live_frame(const Fsr4LiveFrameV2 *v2, void *fence) { @autoreleasepool {
    const char *reason = fsr4_netal_input_reason(v2);
    if (reason) return reject(reason);
    const Fsr4LiveFrame &f = v2->frame;
    std::lock_guard<std::mutex> lock(mutex);
    collect();
    if (retired.size() >= 8) return reject("Netal: pending retired resources; keeping MetalFX");
    id command = (__bridge id)f.commandBuffer;
    id<MTLDevice> device = (__bridge id<MTLDevice>)f.device;
    if (![command respondsToSelector:@selector(device)] || [command device] != device)
        return reject("Netal: command buffer device mismatch");
    const bool metal4 = [command respondsToSelector:@selector(useResidencySet:)]
                     && ![command respondsToSelector:@selector(addCompletedHandler:)];
    if (!metal4 && ![command respondsToSelector:@selector(addCompletedHandler:)])
        return reject("Netal: unsupported command buffer API");
    const Fsr4InputSurface surfaces[] = {v2->geometry.color, v2->geometry.depth, v2->geometry.motion, v2->geometry.output};
    void *textures[] = {f.color, f.depth, f.motion, f.output};
    for (size_t i = 0; i < 4; ++i) {
        id<MTLTexture> texture = (__bridge id<MTLTexture>)textures[i];
        if (texture.device != device || texture.width != surfaces[i].width || texture.height != surfaces[i].height)
            return reject("Netal: actual texture geometry/device mismatch");
    }
    if (f.exposure && [(__bridge id<MTLTexture>)f.exposure device] != device)
        return reject("Netal: exposure device mismatch");
    if (fence && [(__bridge id<MTLFence>)fence device] != device)
        return reject("Netal: scaler fence device mismatch");
    const std::string path = bundle_path();
    if (path.empty()) return reject("Netal: set FSR4_NETAL_RUNTIME to a complete SR bundle");
    const Key ownerKey = key(f.device, f.owner);
    const std::string identity = path + ":" + std::to_string(f.rw) + "x" + std::to_string(f.rh)
                              + ":" + std::to_string(f.uw) + "x" + std::to_string(f.uh);
    auto cached = failures.find(ownerKey);
    if (cached != failures.end() && cached->second.identity == identity
        && std::chrono::steady_clock::now() < cached->second.retry) return reject(cached->second.reason.c_str());
    failures.erase(ownerKey);
    auto &view = views[ownerKey];
    if (view && !view->finished()) {
        view->needsReset = true;
        return reject("Netal: previous frame is still on the GPU; keeping MetalFX");
    }
    if (view && view->busy && __atomic_load_n((uint32_t *)view->completion.contents, __ATOMIC_ACQUIRE) == 2u)
        return reject("Netal: GPU command buffer failed; restart the game before retrying DLSS");
    if (view && (!compatible(*view, f, path) || view->failed)) {
        retire(view); view.reset();
    }
    NSError *error = nil;
    if (!view) {
        @try { view = create(f, path, &error); }
        @catch (NSException *exception) {
            error = [NSError errorWithDomain:@"fsr4.netal" code:1
                userInfo:@{NSLocalizedDescriptionKey:exception.reason ?: @"session exception"}];
        }
        if (!view) {
            const char *why = error.localizedDescription.UTF8String;
            failures[ownerKey] = {identity, std::string("Netal: ") + (why && *why ? why : "session or geometry initialization failed"),
                                 std::chrono::steady_clock::now() + std::chrono::seconds(2)};
            return reject_error(error, "session or geometry initialization failed");
        }
    }
    // One in-flight encode per scaler in this initial integration. This also
    // makes upstream CPU history resets and its six-slot Metal 4 ring safe.
    view->busy = false;
    view->resources = @[(__bridge id)f.color, (__bridge id)f.depth,
                        (__bridge id)f.motion, (__bridge id)f.output,
                        view->adapter.diagnosticModelOutputBuffer,
                        view->adapter.diagnosticColorBuffer, view->adapter.diagnosticMotionBuffer,
                        view->adapter.diagnosticDepthBuffer, view->adapter.diagnosticExposureBuffer,
                        view->completion];
    if (f.exposure) view->resources = [view->resources arrayByAddingObject:(__bridge id)f.exposure];
    if (metal4) {
        [view->residency removeAllAllocations];
        for (id resource in view->resources) [view->residency addAllocation:resource];
        [view->residency commit];
        [(__bridge id<MTL4CommandBuffer>)f.commandBuffer useResidencySet:view->residency];
    }
    __atomic_store_n((uint32_t *)view->completion.contents, 0u, __ATOMIC_RELEASE);
    view->busy = true;
    view->adapter.pendingFence = (__bridge id<MTLFence>)fence;
    BOOL encoded = NO;
    @try {
        if (metal4) encoded = [view->adapter encodeTemporalFrameToMTL4CommandBuffer:command
            colorTexture:(__bridge id<MTLTexture>)f.color motionTexture:(__bridge id<MTLTexture>)f.motion
            depthTexture:(__bridge id<MTLTexture>)f.depth exposureTexture:(__bridge id<MTLTexture>)f.exposure
            outputTexture:(__bridge id<MTLTexture>)f.output inputContentWidth:f.rw inputContentHeight:f.rh
            jitterOffsetX:f.jitterX jitterOffsetY:f.jitterY motionVectorScaleX:f.mvScaleX motionVectorScaleY:f.mvScaleY
            preExposure:f.preExposure reset:f.reset || view->needsReset depthReversed:f.depthReversed error:&error];
        else encoded = [view->adapter encodeTemporalFrameToCommandBuffer:command
            colorTexture:(__bridge id<MTLTexture>)f.color motionTexture:(__bridge id<MTLTexture>)f.motion
            depthTexture:(__bridge id<MTLTexture>)f.depth exposureTexture:(__bridge id<MTLTexture>)f.exposure
            outputTexture:(__bridge id<MTLTexture>)f.output inputContentWidth:f.rw inputContentHeight:f.rh
            jitterOffsetX:f.jitterX jitterOffsetY:f.jitterY motionVectorScaleX:f.mvScaleX motionVectorScaleY:f.mvScaleY
            preExposure:f.preExposure reset:f.reset || view->needsReset depthReversed:f.depthReversed error:&error];
    } @catch (NSException *exception) {
        error = [NSError errorWithDomain:@"fsr4.netal" code:1 userInfo:@{NSLocalizedDescriptionKey:exception.reason ?: @"encode exception"}];
    }
    if (!encode_retirement(*view, command, metal4)) {
        // Unknown completion means it is unsafe to destroy or reuse this view.
        view->failed = true; view->needsReset = true;
        return reject("Netal: retirement encoder unavailable; resources held, keeping MetalFX");
    }
    view->needsReset = !encoded;
    if (!encoded) return reject_error(error, "unsupported frame or encode failed");
    Fsr4LiveStatus prior = {};
    fsr4_live_get_status(&prior);
    status.encodes = prior.encodes; status.replaced = prior.replaced; status.stubs = prior.stubs;
    status.active = 1;
    status.reason[0] = 0;
    snprintf(status.preset, sizeof status.preset, "preset-k");
    snprintf(status.model, sizeof status.model, "dlss");
    snprintf(status.variant, sizeof status.variant, "netal_sr_%ux%u_%s", f.uw, f.uh, metal4 ? "metal4" : "metal3");
    status.rw = f.rw; status.rh = f.rh; status.uw = status.uwA = f.uw; status.uh = status.uhA = f.uh;
    status.sharpness = 0;
    const auto now = std::chrono::steady_clock::now();
    if (lastEncode.time_since_epoch().count()) {
        const float elapsed = std::chrono::duration<float, std::milli>(now - lastEncode).count();
        status.frameMs = status.frameMs > 0 ? .9f * status.frameMs + .1f * elapsed : elapsed;
    }
    lastEncode = now;
    ++status.encodes; ++status.replaced;
    fsr4_live_set_status(&status);
    return 1;
} }
void fsr4_netal_invalidate_owner(void *device, void *owner) {
    std::lock_guard<std::mutex> lock(mutex);
    const auto found = views.find(key(device, owner));
    if (found != views.end() && found->second) found->second->needsReset = true;
}
void fsr4_netal_release_owner(void *device, void *owner) {
    std::lock_guard<std::mutex> lock(mutex);
    const auto found = views.find(key(device, owner));
    if (found != views.end()) { retire(found->second); views.erase(found); }
    failures.erase(key(device, owner));
    collect();
}
#endif
