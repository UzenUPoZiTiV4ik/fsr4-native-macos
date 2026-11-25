// SPDX-License-Identifier: MIT
// ANR owns the FG method only while selected (ANR 2x or ANR 3x). MetalFX restores
// its original IMP: no per-frame mode polling, ANR mutex, copies or kernels on native FG.
// When ANR 3x is selected, a lightweight presentation hook inserts the t=2/3 interpolated
// frame immediately after OutputInterpolated (t=1/3) on the caller's exact Metal 3 or Metal 4 queue.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalFX/MetalFX.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <dispatch/dispatch.h>
#include "../src/anr/anr_live.h"
#include "../src/common/fsr4_fg_settings.h"
#include <atomic>
#include <map>
#include <mutex>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static IMP originalNew = NULL, originalNew4 = NULL;
static const void *lifetimeKey = &lifetimeKey;
static const void *distortionKey = &distortionKey;
static std::atomic<int> selectedMode{0}; // 0=MetalFX, 1=ANR 2x, 2=ANR 3x
static std::mutex methodMutex, settingsMutex, presentMutex;
static NSHashTable *owners = nil;
static dispatch_source_t settingsTimer;
static void encode(id, SEL, id);
static void resetNativeOnce(id, SEL, id);
static thread_local bool g_inExtraPresent = false;

struct ScopedExtraPresentGuard {
    bool prev;
    ScopedExtraPresentGuard() : prev(g_inExtraPresent) { g_inExtraPresent = true; }
    ~ScopedExtraPresentGuard() { g_inExtraPresent = prev; }
};

struct AtomicClass {
    std::atomic<void *> value{nullptr};
    Class load(std::memory_order order = std::memory_order_seq_cst) const { return (__bridge Class)value.load(order); }
    void store(Class cls, std::memory_order order = std::memory_order_seq_cst) { value.store((__bridge void *)cls, order); }
};
struct AnrEncodeHook {
    AtomicClass cls;
    std::atomic<IMP> original{NULL};
};
static AnrEncodeHook slots[8];

@interface AnrOwnerLifetime : NSObject {
@public
    std::atomic<bool> resetPending, originalDirty;
    IMP original;
}
@property(nonatomic, strong) id<MTLDevice> device;
@property(nonatomic, assign) void *owner;
@property(nonatomic, assign) Class hookClass;
@end

@implementation AnrOwnerLifetime
- (id)init {
    self = [super init];
    if (self) {
        resetPending.store(false);
        originalDirty.store(false);
    }
    return self;
}
- (void)dealloc {
    fsr4_anr_release_owner((__bridge void *)_device, _owner);
}
@end

static void logHook(const char *message, Class cls) {
    fprintf(stderr, "[anr-fg] %s %s\n", message, cls ? class_getName(cls) : "");
    const char *path = getenv("FSR4_HOOK_LOG");
    if (path && *path) {
        if (FILE *file = fopen(path, "a")) {
            fprintf(file, "[anr-fg] %s %s\n", message, cls ? class_getName(cls) : "");
            fclose(file);
        }
    }
}

static IMP findOriginal(Class cls) {
    for (Class current = cls; current; current = class_getSuperclass(current)) {
        for (auto &slot : slots) {
            if (slot.cls.load(std::memory_order_acquire) == current) return slot.original.load();
        }
    }
    return NULL;
}

static bool pendingReset(Class cls) {
    for (AnrOwnerLifetime *life in owners.allObjects) {
        if (life->resetPending.load(std::memory_order_acquire) && life.hookClass == cls) return true;
    }
    return false;
}

static void setEncode(Class cls, IMP original, IMP replacement) {
    SEL sel = @selector(encodeToCommandBuffer:);
    Method m = class_getInstanceMethod(cls, sel);
    if (!m) return;
    IMP current = method_getImplementation(m);
    if (current != original && current != (IMP)encode && current != (IMP)resetNativeOnce) {
        logHook("foreign FG encode retained; no competing swizzle", cls);
        return;
    }
    if (current == replacement) return;
    class_addMethod(cls, sel, original, method_getTypeEncoding(m));
    method_setImplementation(class_getInstanceMethod(cls, sel), replacement);
}

// ---------- 3x Extra-Frame Presentation Blit & Queue Hooks ----------

struct Metal4PresentSlot {
    id<MTL4CommandAllocator> allocator = nil;
    id<MTL4CommandBuffer> command = nil;
    id<MTL4ArgumentTable> argTable = nil;
    id<MTLResidencySet> residency = nil;
    id<MTLTexture> retainedSrc = nil;
    id<MTLTexture> retainedDst = nil;
    uint64_t signalValue = 0;
    uint64_t ticket = 0;
    bool inFlight = false;
};

struct PresentResources {
    id<MTLLibrary> helperLib = nil;
    std::map<MTLPixelFormat, id<MTLRenderPipelineState>> renderPSOs;
    id<MTLSharedEvent> event = nil;
    uint64_t nextSignal = 1;
    Metal4PresentSlot m4Slots[4];
    unsigned nextM4Slot = 0;
};

static std::map<uintptr_t, PresentResources> g_presentRes;
// Metal 4: drawable ptr -> queue4, and device ptr -> last active queue4
static std::map<uintptr_t, __weak id<MTL4CommandQueue>> g_drawableQueue4;
static std::map<uintptr_t, __weak id<MTL4CommandQueue>> g_deviceQueue4;
// Metal 3: commandBuffer ptr -> pending extra frame to commit right after originalCommit
struct Metal3PendingCommit {
    id drawable1 = nil;
    id<MTLTexture> extraTex = nil;
    double stepSec = 0.0;
    uint64_t ticket = 0;
};
static std::map<uintptr_t, Metal3PendingCommit> g_m3PendingCommit;

static std::atomic<bool> g_drawableHooked{false};
static IMP g_origDrawablePresent = NULL;
static IMP g_origDrawablePresentAfter = NULL;

static std::map<Class, IMP> g_origQueue4SignalDrawable;
static std::map<Class, IMP> g_origCB3Present;
static std::map<Class, IMP> g_origCB3PresentAfter;
static std::map<Class, IMP> g_origCB3Commit;

static id<MTLRenderPipelineState> getPresentPSO(id<MTLDevice> device, MTLPixelFormat fmt, PresentResources &res) {
    auto it = res.renderPSOs.find(fmt);
    if (it != res.renderPSOs.end()) return it->second;
    if (!res.helperLib) {
        const char *root = getenv("FSR4_ROOT");
        if (!root || !*root) return nil;
        NSString *path = [[NSString stringWithUTF8String:root] stringByAppendingPathComponent:@"build/anr_helpers_ui_v2.metallib"];
        NSError *err = nil;
        res.helperLib = [device newLibraryWithURL:[NSURL fileURLWithPath:path] error:&err];
        if (!res.helperLib) return nil;
    }
    id<MTLFunction> vs = [res.helperLib newFunctionWithName:@"anr_present_vs"];
    id<MTLFunction> fs = [res.helperLib newFunctionWithName:@"anr_present_fs"];
    if (!vs || !fs) return nil;
    MTLRenderPipelineDescriptor *rpd = [MTLRenderPipelineDescriptor new];
    rpd.vertexFunction = vs;
    rpd.fragmentFunction = fs;
    rpd.colorAttachments[0].pixelFormat = fmt;
    NSError *err = nil;
    id<MTLRenderPipelineState> pso = [device newRenderPipelineStateWithDescriptor:rpd error:&err];
    if (pso) res.renderPSOs[fmt] = pso;
    return pso;
}

static bool initMetal4PresentResources(id<MTLDevice> device, PresentResources &res) {
    if (!res.event) {
        res.event = [device newSharedEvent];
        if (!res.event) return false;
    }
    for (auto &s : res.m4Slots) {
        if (s.allocator) continue;
        NSError *err = nil;
        s.allocator = [device newCommandAllocator];
        s.command = [device newCommandBuffer];
        MTL4ArgumentTableDescriptor *td = [MTL4ArgumentTableDescriptor new];
        td.maxTextureBindCount = 1;
        s.argTable = [device newArgumentTableWithDescriptor:td error:&err];
        MTLResidencySetDescriptor *rd = [MTLResidencySetDescriptor new];
        rd.initialCapacity = 4;
        s.residency = [device newResidencySetWithDescriptor:rd error:&err];
        if (!s.allocator || !s.command || !s.argTable || !s.residency) return false;
    }
    return true;
}

static void retireCompletedM4SlotsLocked(PresentResources &res, uint64_t *retiredTickets, unsigned &retiredCount) {
    if (!res.event) return;
    const uint64_t signaled = res.event.signaledValue;
    for (auto &s : res.m4Slots) {
        if (s.inFlight && signaled >= s.signalValue) {
            if (retiredCount < 4) retiredTickets[retiredCount++] = s.ticket;
            s.inFlight = false;
            s.ticket = 0;
            s.retainedSrc = nil;
            s.retainedDst = nil;
        }
    }
}

static void retireCompletedM4SlotsForDevice(id<MTLDevice> device) {
    if (!device) return;
    uint64_t retired[4] = {};
    unsigned retiredCount = 0;
    {
        std::lock_guard<std::mutex> lock(presentMutex);
        auto it = g_presentRes.find((uintptr_t)(__bridge void *)device);
        if (it != g_presentRes.end()) retireCompletedM4SlotsLocked(it->second, retired, retiredCount);
    }
    for (unsigned i = 0; i < retiredCount; ++i) fsr4_anr_release_extra_ticket(retired[i], 1);
}

static id acquireExtraDrawable(id drawable1, id<MTLDevice> device) {
    SEL layerSel = @selector(layer);
    if (!drawable1 || ![drawable1 respondsToSelector:layerSel]) return nil;
    id layer = ((id(*)(id, SEL))objc_msgSend)(drawable1, layerSel);
    if (!layer) return nil;
    SEL maxCountSel = @selector(maximumDrawableCount);
    SEL setMaxCountSel = @selector(setMaximumDrawableCount:);
    if ([layer respondsToSelector:maxCountSel] && [layer respondsToSelector:setMaxCountSel]) {
        NSUInteger cur = ((NSUInteger(*)(id, SEL))objc_msgSend)(layer, maxCountSel);
        if (cur < 3) {
            @try {
                ((void(*)(id, SEL, NSUInteger))objc_msgSend)(layer, setMaxCountSel, (NSUInteger)3);
            } @catch (NSException *) {}
        }
    }
    SEL nextSel = @selector(nextDrawable);
    if (![layer respondsToSelector:nextSel]) return nil;
    id d2 = ((id(*)(id, SEL))objc_msgSend)(layer, nextSel);
    if (!d2) return nil;
    SEL texSel = @selector(texture);
    id<MTLTexture> dst = [d2 respondsToSelector:texSel] ? ((id<MTLTexture>(*)(id, SEL))objc_msgSend)(d2, texSel) : nil;
    if (!dst || dst.device != device || dst.width < 64 || dst.height < 64) return nil;
    return d2;
}

static bool layerFramebufferOnly(id drawable) {
    SEL layerSel = @selector(layer);
    if (!drawable || ![drawable respondsToSelector:layerSel]) return true;
    id layer = ((id(*)(id, SEL))objc_msgSend)(drawable, layerSel);
    SEL fbSel = @selector(framebufferOnly);
    if (layer && [layer respondsToSelector:fbSel]) {
        return ((BOOL(*)(id, SEL))objc_msgSend)(layer, fbSel) != NO;
    }
    return true;
}

static void submitExtraMetal4(id<MTLDevice> device, id<MTL4CommandQueue> queue4, id drawable1,
                              id<MTLTexture> extraTex, double stepSec, uint64_t ticket) {
    if (!device || !queue4 || !drawable1 || !extraTex) {
        fsr4_anr_release_extra_ticket(ticket, 0);
        return;
    }
    ScopedExtraPresentGuard guard;
    id d2 = acquireExtraDrawable(drawable1, device);
    if (!d2) {
        fsr4_anr_release_extra_ticket(ticket, 0);
        return;
    }
    id<MTLTexture> dst = ((id<MTLTexture>(*)(id, SEL))objc_msgSend)(d2, @selector(texture));

    uint64_t retired[4] = {};
    unsigned retiredCount = 0;
    id<MTL4CommandBuffer> cmd = nil;
    id<MTLSharedEvent> ev = nil;
    uint64_t sigVal = 0;
    bool ready = false;
    {
        std::lock_guard<std::mutex> lock(presentMutex);
        auto &res = g_presentRes[(uintptr_t)(__bridge void *)device];
        if (initMetal4PresentResources(device, res)) {
            retireCompletedM4SlotsLocked(res, retired, retiredCount);

            Metal4PresentSlot *slot = nullptr;
            for (unsigned i = 0; i < 4; ++i) {
                auto &cand = res.m4Slots[(res.nextM4Slot + i) % 4];
                if (!cand.inFlight) {
                    slot = &cand;
                    res.nextM4Slot = (res.nextM4Slot + i + 1) % 4;
                    break;
                }
            }
            if (slot) {
                const bool fbOnly = layerFramebufferOnly(d2);
                const bool exactCopy = !fbOnly && extraTex.pixelFormat == dst.pixelFormat
                    && extraTex.width == dst.width && extraTex.height == dst.height;

                id<MTLRenderPipelineState> pso = exactCopy ? nil : getPresentPSO(device, dst.pixelFormat, res);
                if (exactCopy || pso) {
                    [slot->residency removeAllAllocations];
                    [slot->residency addAllocation:extraTex];
                    [slot->residency addAllocation:dst];
                    [slot->residency commit];

                    [slot->allocator reset];
                    [slot->command beginCommandBufferWithAllocator:slot->allocator];
                    [slot->command useResidencySet:slot->residency];

                    bool encoded = false;
                    if (exactCopy) {
                        id<MTL4ComputeCommandEncoder> cenc = [slot->command computeCommandEncoder];
                        if (cenc) {
                            [cenc barrierAfterQueueStages:MTLStageAll beforeStages:MTLStageBlit visibilityOptions:MTL4VisibilityOptionDevice];
                            [cenc copyFromTexture:extraTex toTexture:dst];
                            [cenc endEncoding];
                            encoded = true;
                        }
                    } else {
                        MTL4RenderPassDescriptor *rp = [MTL4RenderPassDescriptor new];
                        rp.colorAttachments[0].texture = dst;
                        rp.colorAttachments[0].loadAction = MTLLoadActionDontCare;
                        rp.colorAttachments[0].storeAction = MTLStoreActionStore;
                        id<MTL4RenderCommandEncoder> renc = [slot->command renderCommandEncoderWithDescriptor:rp];
                        if (renc) {
                            [renc barrierAfterQueueStages:MTLStageAll beforeStages:MTLStageFragment visibilityOptions:MTL4VisibilityOptionDevice];
                            [renc setRenderPipelineState:pso];
                            [slot->argTable setTexture:extraTex.gpuResourceID atIndex:0];
                            [renc setArgumentTable:slot->argTable atStages:MTLRenderStageFragment];
                            [renc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
                            [renc endEncoding];
                            encoded = true;
                        }
                    }
                    [slot->command endCommandBuffer];
                    if (encoded) {
                        sigVal = res.nextSignal++;
                        slot->retainedSrc = extraTex;
                        slot->retainedDst = dst;
                        slot->signalValue = sigVal;
                        slot->ticket = ticket;
                        slot->inFlight = true;
                        cmd = slot->command;
                        ev = res.event;
                        ready = true;
                    }
                }
            }
        }
    }
    // Release completed tickets and submit to Metal 4 queue AFTER unlocking presentMutex
    // (signalDrawable: re-enters hookedQueue4SignalDrawable which locks presentMutex).
    for (unsigned i = 0; i < retiredCount; ++i) fsr4_anr_release_extra_ticket(retired[i], 1);
    if (!ready) {
        fsr4_anr_release_extra_ticket(ticket, 0);
        return;
    }

    [queue4 waitForDrawable:d2];
    id<MTL4CommandBuffer> cbs[] = {cmd};
    [queue4 commit:cbs count:1];
    [queue4 signalEvent:ev value:sigVal];
    [queue4 signalDrawable:d2];
    if (stepSec > 0.0 && [d2 respondsToSelector:@selector(presentAfterMinimumDuration:)]) {
        ((void(*)(id, SEL, CFTimeInterval))objc_msgSend)(d2, @selector(presentAfterMinimumDuration:), stepSec);
    } else {
        ((void(*)(id, SEL))objc_msgSend)(d2, @selector(present));
    }
}

static void submitExtraMetal3(id<MTLCommandQueue> queue3, id drawable1,
                              id<MTLTexture> extraTex, double stepSec, uint64_t ticket) {
    if (!queue3 || !drawable1 || !extraTex) {
        fsr4_anr_release_extra_ticket(ticket, 0);
        return;
    }
    ScopedExtraPresentGuard guard;
    id<MTLDevice> device = queue3.device;
    id d2 = acquireExtraDrawable(drawable1, device);
    if (!d2) {
        fsr4_anr_release_extra_ticket(ticket, 0);
        return;
    }
    id<MTLTexture> dst = ((id<MTLTexture>(*)(id, SEL))objc_msgSend)(d2, @selector(texture));
    const bool fbOnly = layerFramebufferOnly(d2);
    const bool exactCopy = !fbOnly && extraTex.pixelFormat == dst.pixelFormat
        && extraTex.width == dst.width && extraTex.height == dst.height;

    id<MTLRenderPipelineState> pso = nil;
    if (!exactCopy) {
        std::lock_guard<std::mutex> lock(presentMutex);
        auto &res = g_presentRes[(uintptr_t)(__bridge void *)device];
        pso = getPresentPSO(device, dst.pixelFormat, res);
        if (!pso) {
            fsr4_anr_release_extra_ticket(ticket, 0);
            return;
        }
    }

    id<MTLCommandBuffer> cb2 = [queue3 commandBuffer];
    if (!cb2) {
        fsr4_anr_release_extra_ticket(ticket, 0);
        return;
    }
    bool encoded = false;
    if (exactCopy) {
        id<MTLBlitCommandEncoder> benc = [cb2 blitCommandEncoder];
        if (benc) {
            [benc copyFromTexture:extraTex toTexture:dst];
            [benc endEncoding];
            encoded = true;
        }
    } else {
        MTLRenderPassDescriptor *rp = [MTLRenderPassDescriptor renderPassDescriptor];
        rp.colorAttachments[0].texture = dst;
        rp.colorAttachments[0].loadAction = MTLLoadActionDontCare;
        rp.colorAttachments[0].storeAction = MTLStoreActionStore;
        id<MTLRenderCommandEncoder> renc = [cb2 renderCommandEncoderWithDescriptor:rp];
        if (renc) {
            [renc setRenderPipelineState:pso];
            [renc setFragmentTexture:extraTex atIndex:0];
            [renc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
            [renc endEncoding];
            encoded = true;
        }
    }
    if (!encoded) {
        fsr4_anr_release_extra_ticket(ticket, 0);
        return;
    }

    id<MTLTexture> keepSrc = extraTex;
    [cb2 addCompletedHandler:^(id<MTLCommandBuffer> done) {
        (void)keepSrc;
        fsr4_anr_release_extra_ticket(ticket, done.status == MTLCommandBufferStatusCompleted ? 1 : 0);
    }];
    if (stepSec > 0.0) {
        [cb2 presentDrawable:d2 afterMinimumDuration:stepSec];
    } else {
        [cb2 presentDrawable:d2];
    }
    [cb2 commit];
}

// Metal 4 hooks: queue4 signalDrawable: + CAMetalDrawable present / presentAfterMinimumDuration:
static void hookedQueue4SignalDrawable(id self, SEL sel, id drawable) {
    IMP orig = NULL;
    {
        std::lock_guard<std::mutex> lock(presentMutex);
        for (Class c = object_getClass(self); c && !orig; c = class_getSuperclass(c)) {
            auto it = g_origQueue4SignalDrawable.find(c);
            if (it != g_origQueue4SignalDrawable.end()) orig = it->second;
        }
        if (!g_inExtraPresent && selectedMode.load(std::memory_order_relaxed) == 2 && drawable) {
            id<MTL4CommandQueue> q4 = self;
            g_drawableQueue4[(uintptr_t)(__bridge void *)drawable] = q4;
            if (q4.device) {
                g_deviceQueue4[(uintptr_t)(__bridge void *)q4.device] = q4;
            }
        }
    }
    if (orig) ((void(*)(id, SEL, id))orig)(self, sel, drawable);
}

static void handleDrawablePresent(id self, SEL, bool hasDuration, CFTimeInterval duration) {
    auto callOriginal = [&](bool useDur, CFTimeInterval dur) {
        if (useDur && g_origDrawablePresentAfter) {
            ((void(*)(id, SEL, CFTimeInterval))g_origDrawablePresentAfter)(self, @selector(presentAfterMinimumDuration:), dur);
        } else if (g_origDrawablePresent) {
            ((void(*)(id, SEL))g_origDrawablePresent)(self, @selector(present));
        }
    };

    if (g_inExtraPresent || selectedMode.load(std::memory_order_relaxed) != 2) {
        callOriginal(hasDuration, duration);
        return;
    }

    id<MTLTexture> tex = [self respondsToSelector:@selector(texture)]
        ? ((id<MTLTexture>(*)(id, SEL))objc_msgSend)(self, @selector(texture)) : nil;
    id<MTLDevice> dev = tex ? tex.device : nil;
    if (!dev || tex.width < 64 || tex.height < 64) {
        callOriginal(hasDuration, duration);
        return;
    }

    id<MTL4CommandQueue> q4 = nil;
    {
        std::lock_guard<std::mutex> lock(presentMutex);
        auto it = g_drawableQueue4.find((uintptr_t)(__bridge void *)self);
        if (it != g_drawableQueue4.end()) {
            q4 = it->second;
            g_drawableQueue4.erase(it);
        } else {
            auto dit = g_deviceQueue4.find((uintptr_t)(__bridge void *)dev);
            if (dit != g_deviceQueue4.end()) q4 = dit->second;
        }
    }
    // If no Metal 4 queue signaled this drawable, Metal 3 handles it via MTLCommandBuffer hooks.
    if (!q4) {
        callOriginal(hasDuration, duration);
        return;
    }

    void *extraRaw = nullptr;
    double stepSec = 0.0;
    uint64_t ticket = 0;
    const int event = fsr4_anr_on_present_event((__bridge void *)dev, &extraRaw, &stepSec, &ticket);
    if (event == 1 && extraRaw) {
        id<MTLTexture> extraTex = (__bridge_transfer id<MTLTexture>)extraRaw;
        // Present I_{1/3} with 1/3 pacing, then immediately submit & present I_{2/3}.
        const double pace = stepSec > 0.0 ? stepSec : duration;
        callOriginal(pace > 0.0, pace);
        submitExtraMetal4(dev, q4, self, extraTex, stepSec, ticket);
        return;
    }
    if (event == 2 && stepSec > 0.0) {
        // OutputReal (R_n) following inserted I_{2/3}: pace with stepSec.
        callOriginal(true, stepSec);
        return;
    }
    callOriginal(hasDuration, duration);
}

static void hookedDrawablePresent(id self, SEL sel) {
    handleDrawablePresent(self, sel, false, 0.0);
}

static void hookedDrawablePresentAfter(id self, SEL sel, CFTimeInterval duration) {
    handleDrawablePresent(self, sel, true, duration);
}

// Metal 3 hooks: MTLCommandBuffer presentDrawable* + commit
static IMP findCB3Orig(const std::map<Class, IMP> &table, id obj) {
    for (Class c = object_getClass(obj); c; c = class_getSuperclass(c)) {
        auto it = table.find(c);
        if (it != table.end()) return it->second;
    }
    return NULL;
}

static void handleCB3Present(id self, SEL, id drawable, bool hasDuration, CFTimeInterval duration) {
    IMP origPlain = NULL, origAfter = NULL;
    {
        std::lock_guard<std::mutex> lock(presentMutex);
        origPlain = findCB3Orig(g_origCB3Present, self);
        origAfter = findCB3Orig(g_origCB3PresentAfter, self);
    }
    auto callOriginal = [&](bool useDur, CFTimeInterval dur) {
        if (useDur && origAfter) {
            ((void(*)(id, SEL, id, CFTimeInterval))origAfter)(self, @selector(presentDrawable:afterMinimumDuration:), drawable, dur);
        } else if (origPlain) {
            ((void(*)(id, SEL, id))origPlain)(self, @selector(presentDrawable:), drawable);
        }
    };

    if (g_inExtraPresent || selectedMode.load(std::memory_order_relaxed) != 2 || !drawable) {
        callOriginal(hasDuration, duration);
        return;
    }

    id<MTLCommandBuffer> cb = self;
    id<MTLDevice> dev = cb.device;
    void *extraRaw = nullptr;
    double stepSec = 0.0;
    uint64_t ticket = 0;
    const int event = fsr4_anr_on_present_event((__bridge void *)dev, &extraRaw, &stepSec, &ticket);
    if (event == 1 && extraRaw) {
        id<MTLTexture> extraTex = (__bridge_transfer id<MTLTexture>)extraRaw;
        const double pace = stepSec > 0.0 ? stepSec : duration;
        callOriginal(pace > 0.0, pace);
        std::lock_guard<std::mutex> lock(presentMutex);
        g_m3PendingCommit[(uintptr_t)(__bridge void *)self] = {drawable, extraTex, stepSec, ticket};
        return;
    }
    if (event == 2 && stepSec > 0.0) {
        callOriginal(true, stepSec);
        return;
    }
    callOriginal(hasDuration, duration);
}

static void hookedCB3Present(id self, SEL sel, id drawable) {
    handleCB3Present(self, sel, drawable, false, 0.0);
}

static void hookedCB3PresentAfter(id self, SEL sel, id drawable, CFTimeInterval duration) {
    handleCB3Present(self, sel, drawable, true, duration);
}

static void hookedCB3Commit(id self, SEL sel) {
    IMP orig = NULL;
    Metal3PendingCommit pending = {};
    bool hasPending = false;
    {
        std::lock_guard<std::mutex> lock(presentMutex);
        orig = findCB3Orig(g_origCB3Commit, self);
        auto it = g_m3PendingCommit.find((uintptr_t)(__bridge void *)self);
        if (it != g_m3PendingCommit.end()) {
            pending = it->second;
            hasPending = true;
            g_m3PendingCommit.erase(it);
        }
    }
    if (orig) ((void(*)(id, SEL))orig)(self, sel);
    if (hasPending) {
        id<MTLCommandBuffer> cb = self;
        submitExtraMetal3(cb.commandQueue, pending.drawable1, pending.extraTex, pending.stepSec, pending.ticket);
    }
}

static void ensurePresentHooks(id<MTLDevice> device, id commandBuffer) {
    if (selectedMode.load(std::memory_order_acquire) != 2 || !device) return;
    std::lock_guard<std::mutex> lock(presentMutex);

    if (!g_drawableHooked.load(std::memory_order_relaxed)) {
        Class dCls = objc_getClass("CAMetalDrawable");
        if (dCls) {
            Method m1 = class_getInstanceMethod(dCls, @selector(present));
            Method m2 = class_getInstanceMethod(dCls, @selector(presentAfterMinimumDuration:));
            if (m1 && method_getImplementation(m1) != (IMP)hookedDrawablePresent) {
                g_origDrawablePresent = method_getImplementation(m1);
                method_setImplementation(m1, (IMP)hookedDrawablePresent);
            }
            if (m2 && method_getImplementation(m2) != (IMP)hookedDrawablePresentAfter) {
                g_origDrawablePresentAfter = method_getImplementation(m2);
                method_setImplementation(m2, (IMP)hookedDrawablePresentAfter);
            }
            g_drawableHooked.store(true, std::memory_order_release);
            logHook("3x presentation hook ready on", dCls);
        }
    }

    const bool isM4 = commandBuffer && [commandBuffer respondsToSelector:@selector(useResidencySet:)]
        && ![commandBuffer respondsToSelector:@selector(addCompletedHandler:)];
    if (isM4) {
        if ([device respondsToSelector:@selector(newMTL4CommandQueue)]) {
            id<MTL4CommandQueue> probeQ4 = [device newMTL4CommandQueue];
            if (probeQ4) {
                Class q4Cls = object_getClass(probeQ4);
                if (q4Cls && !g_origQueue4SignalDrawable.count(q4Cls)) {
                    SEL sigSel = @selector(signalDrawable:);
                    Method ms = class_getInstanceMethod(q4Cls, sigSel);
                    if (ms && method_getImplementation(ms) != (IMP)hookedQueue4SignalDrawable) {
                        IMP orig = method_getImplementation(ms);
                        g_origQueue4SignalDrawable[q4Cls] = orig;
                        class_addMethod(q4Cls, sigSel, orig, method_getTypeEncoding(ms));
                        method_setImplementation(class_getInstanceMethod(q4Cls, sigSel), (IMP)hookedQueue4SignalDrawable);
                        logHook("3x Metal 4 queue hook ready on", q4Cls);
                    }
                }
            }
        }
    } else if (commandBuffer) {
        Class cbCls = object_getClass(commandBuffer);
        if (cbCls && !g_origCB3Commit.count(cbCls)) {
            auto hookM = [&](SEL s, IMP repl, std::map<Class, IMP> &store) {
                Method m = class_getInstanceMethod(cbCls, s);
                if (m && method_getImplementation(m) != repl) {
                    IMP orig = method_getImplementation(m);
                    store[cbCls] = orig;
                    class_addMethod(cbCls, s, orig, method_getTypeEncoding(m));
                    method_setImplementation(class_getInstanceMethod(cbCls, s), repl);
                }
            };
            hookM(@selector(presentDrawable:), (IMP)hookedCB3Present, g_origCB3Present);
            hookM(@selector(presentDrawable:afterMinimumDuration:), (IMP)hookedCB3PresentAfter, g_origCB3PresentAfter);
            hookM(@selector(commit), (IMP)hookedCB3Commit, g_origCB3Commit);
            logHook("3x Metal 3 commandBuffer hook ready on", cbCls);
        }
    }
}

// ---------- Mode & Settings Control ----------

extern "C" int fsr4_anr_is_selected(void) {
    return selectedMode.load(std::memory_order_acquire) != 0 ? 1 : 0;
}

extern "C" int fsr4_anr_selected_mode(void) {
    return selectedMode.load(std::memory_order_acquire);
}

extern "C" int fsr4_anr_multiplier(void) {
    const int m = selectedMode.load(std::memory_order_acquire);
    return m == 2 ? 3 : (m == 1 ? 2 : 1);
}

extern "C" void fsr4_anr_refresh_settings(void) {
    @autoreleasepool {
        std::lock_guard<std::mutex> refresh(settingsMutex);
        fsr4_fg::Mode mode = fsr4_fg::Mode::MetalFX;
        if (const char *env = getenv("FSR4_FG")) fsr4_fg::parse_mode(env, mode);
        auto path = fsr4_fg::settings_path(getenv("FSR4_ROOT"), getenv("FSR4_FG_SETTINGS"));
        if (!path.empty()) {
            const auto saved = fsr4_fg::read_settings(path);
            if (saved.valid) mode = saved.mode;
        }
        const int want = (int)mode;
        const int before = selectedMode.load(std::memory_order_acquire);
        if (before != want) {
            if (want == 0) {
                std::lock_guard<std::mutex> lock(methodMutex);
                for (AnrOwnerLifetime *life in owners.allObjects) {
                    if (life->originalDirty.load()) life->resetPending.store(true, std::memory_order_release);
                }
                for (auto &slot : slots) {
                    Class cls = slot.cls.load();
                    if (cls) setEncode(cls, slot.original.load(), (IMP)resetNativeOnce);
                }
                selectedMode.store(0, std::memory_order_release);
            }
            fsr4_anr_mode_changed(want);
            if (want != 0) {
                std::lock_guard<std::mutex> lock(methodMutex);
                for (auto &slot : slots) {
                    Class cls = slot.cls.load();
                    if (cls) setEncode(cls, slot.original.load(), (IMP)encode);
                }
                selectedMode.store(want, std::memory_order_release);
            }
            logHook(want == 2 ? "ANR 3x selected; FG method replaced"
                              : want == 1 ? "ANR 2x selected; FG method replaced"
                                          : "MetalFX selected; restoring native FG methods", Nil);
        }
        if (want == 0) {
            {
                std::lock_guard<std::mutex> lock(methodMutex);
                for (auto &slot : slots) {
                    Class cls = slot.cls.load();
                    if (!cls) continue;
                    if (method_getImplementation(class_getInstanceMethod(cls, @selector(encodeToCommandBuffer:))) == (IMP)resetNativeOnce
                        && !pendingReset(cls)) {
                        setEncode(cls, slot.original.load(), slot.original.load());
                    }
                }
            }
            fsr4_anr_publish_passthrough();
        }
    }
}

static NSUInteger optionalExtent(id object, const char *name) {
    SEL sel = sel_registerName(name);
    if (![object respondsToSelector:sel]) return 0;
    return ((NSUInteger(*)(id, SEL))objc_msgSend)(object, sel);
}

static int replace(id object, id command) {
    id<MTLFXFrameInterpolatorBase> fg = object;
    if ([objc_getAssociatedObject(object, distortionKey) boolValue]) {
        fsr4_anr_set_inactive("ANR: distortion texture unsupported");
        return 0;
    }
    for (const char *name : {"depthContentOffsetX", "depthContentOffsetY", "motionContentOffsetX",
                             "motionContentOffsetY", "outputOffsetX", "outputOffsetY"}) {
        if (optionalExtent(object, name)) {
            fsr4_anr_set_inactive("ANR: FG subrect offsets unsupported");
            return 0;
        }
    }
    if (selectedMode.load(std::memory_order_acquire) == 2 && fg.colorTexture.device) {
        ensurePresentHooks(fg.colorTexture.device, command);
        retireCompletedM4SlotsForDevice(fg.colorTexture.device);
    }
    Fsr4AnrFrame f = {};
    f.structSize = sizeof(f);
    f.abiVersion = FSR4_ANR_FG_ABI;
    f.owner = (__bridge void *)object;
    f.commandBuffer = (__bridge void *)command;
    f.color = (__bridge void *)fg.colorTexture;
    f.previousColor = (__bridge void *)fg.prevColorTexture;
    f.depth = (__bridge void *)fg.depthTexture;
    f.motion = (__bridge void *)fg.motionTexture;
    f.output = (__bridge void *)fg.outputTexture;
    f.ui = (__bridge void *)fg.uiTexture;
    f.uiComposited = fg.isUITextureComposited;
    f.fence = (__bridge void *)fg.fence;
    f.device = (__bridge void *)fg.colorTexture.device;
    NSUInteger rw = optionalExtent(object, "contentWidth"), rh = optionalExtent(object, "contentHeight");
    if (!rw && !rh) { rw = fg.inputWidth; rh = fg.inputHeight; }
    if (rw > UINT32_MAX || rh > UINT32_MAX || fg.outputWidth > UINT32_MAX || fg.outputHeight > UINT32_MAX) return 0;
    f.rw = (uint32_t)rw;
    f.rh = (uint32_t)rh;
    f.uw = (uint32_t)fg.outputWidth;
    f.uh = (uint32_t)fg.outputHeight;
    f.motionScaleX = fg.motionVectorScaleX;
    f.motionScaleY = fg.motionVectorScaleY;
    f.jitterX = fg.jitterOffsetX;
    f.jitterY = fg.jitterOffsetY;
    f.nearPlane = fg.nearPlane;
    f.farPlane = fg.farPlane;
    f.fieldOfView = fg.fieldOfView;
    f.aspectRatio = fg.aspectRatio;
    f.deltaTime = fg.deltaTime;
    f.reset = fg.shouldResetHistory;
    f.depthReversed = fg.isDepthReversed;
    if (fg.colorTexture.pixelFormat != fg.outputTexture.pixelFormat) {
        fsr4_anr_set_inactive("ANR: input/output color formats differ");
        return 0;
    }
    MTLPixelFormat format = fg.colorTexture.pixelFormat;
    if (format == MTLPixelFormatRGB10A2Unorm) f.colorEncoding = 2;
    else if (format == MTLPixelFormatBGRA8Unorm || format == MTLPixelFormatRGBA8Unorm) f.colorEncoding = 1;
    else if (format == MTLPixelFormatRGBA16Float || format == MTLPixelFormatRGB9E5Float
             || format == MTLPixelFormatRG11B10Float || format == MTLPixelFormatBGRA8Unorm_sRGB
             || format == MTLPixelFormatRGBA8Unorm_sRGB) f.colorEncoding = 0;
    else {
        fsr4_anr_set_inactive("ANR: color format unsupported");
        return 0;
    }
    if (const char *encoding = getenv("FSR4_ANR_COLOR")) {
        if (*encoding) {
            if (!strcmp(encoding, "linear")) f.colorEncoding = 0;
            else if (!strcmp(encoding, "srgb")) f.colorEncoding = 1;
            else if (!strcmp(encoding, "pq10")) f.colorEncoding = 2;
            else {
                fsr4_anr_set_inactive("ANR: FSR4_ANR_COLOR must be linear, srgb or pq10");
                return 0;
            }
        }
    }
    return fsr4_anr_encode_frame(&f);
}

static void resetNativeOnce(id self, SEL selector, id command) {
    @autoreleasepool {
        AnrOwnerLifetime *life = objc_getAssociatedObject(self, lifetimeKey);
        IMP original = life && life->original ? life->original : findOriginal(object_getClass(self));
        if (!original) return;
        if (life && !life->resetPending.load(std::memory_order_acquire)
            && !life->originalDirty.load(std::memory_order_acquire)) {
            ((void(*)(id, SEL, id))original)(self, selector, command);
            return;
        }
        id<MTLFXFrameInterpolatorBase> fg = self;
        const BOOL previous = fg.shouldResetHistory;
        bool reset = false;
        if (life) {
            reset = life->resetPending.exchange(false, std::memory_order_acq_rel);
            reset |= life->originalDirty.exchange(false, std::memory_order_acq_rel);
        }
        if (reset) fg.shouldResetHistory = YES;
        ((void(*)(id, SEL, id))original)(self, selector, command);
        fg.shouldResetHistory = previous;
        if (selectedMode.load(std::memory_order_acquire) == 0) {
            std::lock_guard<std::mutex> lock(methodMutex);
            Class cls = life.hookClass ?: object_getClass(self);
            if (selectedMode.load() == 0 && !pendingReset(cls)) setEncode(cls, original, original);
        }
    }
}

static void encode(id self, SEL selector, id command) {
    @autoreleasepool {
        IMP original = findOriginal(object_getClass(self));
        if (!original) {
            logHook("original FG encode unavailable", object_getClass(self));
            return;
        }
        if (selectedMode.load(std::memory_order_acquire) == 0) {
            resetNativeOnce(self, selector, command);
            return;
        }
        int result = replace(self, command);
        AnrOwnerLifetime *life = objc_getAssociatedObject(self, lifetimeKey);
        if (result == 1) {
            if (life) life->originalDirty.store(true, std::memory_order_release);
            return;
        }
        if (result != 2 && life) fsr4_anr_invalidate_owner((__bridge void *)life.device, (__bridge void *)self);
        id<MTLFXFrameInterpolatorBase> fg = self;
        const BOOL previous = fg.shouldResetHistory;
        bool reset = result == 2;
        if (life) {
            reset |= life->resetPending.exchange(false, std::memory_order_acq_rel);
            reset |= life->originalDirty.exchange(false, std::memory_order_acq_rel);
        }
        if (reset) fg.shouldResetHistory = YES;
        ((void(*)(id, SEL, id))original)(self, selector, command);
        fg.shouldResetHistory = previous;
    }
}

static void remember(id object, id descriptor, id<MTLDevice> device) {
    if (!object) return;
    AnrOwnerLifetime *life = [AnrOwnerLifetime new];
    life.device = device;
    life.owner = (__bridge void *)object;
    objc_setAssociatedObject(object, lifetimeKey, life, OBJC_ASSOCIATION_RETAIN_NONATOMIC);
    BOOL distortion = NO;
    SEL sel = sel_registerName("isDistortionTextureEnabled");
    if ([descriptor respondsToSelector:sel]) distortion = ((BOOL(*)(id, SEL))objc_msgSend)(descriptor, sel);
    objc_setAssociatedObject(object, distortionKey, @(distortion), OBJC_ASSOCIATION_RETAIN_NONATOMIC);
    std::lock_guard<std::mutex> lock(methodMutex);
    if (!owners) owners = [NSHashTable weakObjectsHashTable];
    [owners addObject:life];
    Class cls = object_getClass(object);
    for (auto &slot : slots) {
        if (slot.cls.load() == cls) {
            life.hookClass = cls;
            life->original = slot.original.load();
            return;
        }
    }
    Method method = class_getInstanceMethod(cls, @selector(encodeToCommandBuffer:));
    if (!method) return;
    IMP current = method_getImplementation(method);
    if (current == (IMP)encode || current == (IMP)resetNativeOnce) {
        for (Class parent = class_getSuperclass(cls); parent; parent = class_getSuperclass(parent)) {
            for (auto &slot : slots) {
                if (slot.cls.load() == parent) {
                    life.hookClass = parent;
                    life->original = slot.original.load();
                    return;
                }
            }
        }
        return;
    }
    for (auto &slot : slots) {
        if (!slot.cls.load()) {
            life.hookClass = cls;
            life->original = current;
            slot.original.store(current);
            slot.cls.store(cls, std::memory_order_release);
            if (selectedMode.load() != 0) setEncode(cls, current, (IMP)encode);
            logHook(selectedMode.load() != 0 ? "hooked encode" : "native FG encode retained", cls);
            return;
        }
    }
    logHook("FG class table full; original retained", cls);
}

static id newFG(id self, SEL selector, id device) {
    id object = ((id(*)(id, SEL, id))originalNew)(self, selector, device);
    fsr4_anr_refresh_settings();
    remember(object, self, device);
    return object;
}

static id newFG4(id self, SEL selector, id device, id compiler) {
    id object = ((id(*)(id, SEL, id, id))originalNew4)(self, selector, device, compiler);
    fsr4_anr_refresh_settings();
    remember(object, self, device);
    return object;
}

__attribute__((constructor)) static void initializeFG() {
    @autoreleasepool {
        Class cls = objc_getClass("MTLFXFrameInterpolatorDescriptor");
        if (!cls) {
            logHook("FG descriptor unavailable", Nil);
            return;
        }
        Method method = class_getInstanceMethod(cls, @selector(newFrameInterpolatorWithDevice:));
        if (method) {
            originalNew = method_getImplementation(method);
            method_setImplementation(method, (IMP)newFG);
        }
        method = class_getInstanceMethod(cls, @selector(newFrameInterpolatorWithDevice:compiler:));
        if (method) {
            originalNew4 = method_getImplementation(method);
            method_setImplementation(method, (IMP)newFG4);
        }
        fsr4_anr_refresh_settings();
        auto queue = dispatch_queue_create("org.fsr4mac.fg-control", DISPATCH_QUEUE_SERIAL);
        settingsTimer = dispatch_source_create(DISPATCH_SOURCE_TYPE_TIMER, 0, 0, queue);
        dispatch_source_set_timer(settingsTimer, dispatch_time(DISPATCH_TIME_NOW, 500 * NSEC_PER_MSEC),
                                  500 * NSEC_PER_MSEC, 50 * NSEC_PER_MSEC);
        dispatch_source_set_event_handler(settingsTimer, ^{ fsr4_anr_refresh_settings(); });
        dispatch_resume(settingsTimer);
        logHook("descriptor hooks ready; native MetalFX has no frame hook", cls);
    }
}
