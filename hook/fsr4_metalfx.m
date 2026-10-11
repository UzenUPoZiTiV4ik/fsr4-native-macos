// Loaded by the copied D3DMetal (LC_LOAD_DYLIB). Swizzles only the temporal
// scaler D3DMetal creates for DLSS. Spatial, frame generation and denoise
// stay on MetalFX. FSR4_REPLACE=1 is the switch; until the encoder returns
// success the original encode still runs, so a bad build does not black the frame.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalFX/MetalFX.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <stdio.h>
#import <stdarg.h>
#import <string.h>
#import <unistd.h>
#import "fsr4_live.h"

// A newer hook may be loaded beside an older encoder. Never read a V2-sized
// frame through the V1 entry point or silently discard an active subrect.
extern int fsr4_encode_live_frame_v2(const Fsr4LiveFrameV2 *) __attribute__((weak_import));
extern void fsr4_invalidate_live_owner(void *, void *) __attribute__((weak_import));
extern void fsr4_release_live_owner(void *, void *) __attribute__((weak_import));

@interface Fsr4OwnerLifetime : NSObject
@property(nonatomic, strong) id<MTLDevice> device;
@property(nonatomic, assign) void *owner;
@end

@implementation Fsr4OwnerLifetime
- (void)dealloc {
    if (fsr4_release_live_owner)
        fsr4_release_live_owner((__bridge void *)_device, _owner);
}
@end

static IMP g_orig_new = NULL;
static IMP g_orig_new4 = NULL;
static const void *kFlagsAssoc = &kFlagsAssoc;
static const void *kLifetimeAssoc = &kLifetimeAssoc;

typedef struct { Class cls; IMP imp; } EncodeSlot;
static EncodeSlot g_encode[4];

static IMP orig_encode(Class cls) {
    for (int i = 0; i < 4; i++)
        if (g_encode[i].cls == cls) return g_encode[i].imp;
    return NULL;
}

static void hook_log(const char *fmt, ...) {
    char buf[1024];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof buf, fmt, ap);
    va_end(ap);
    fprintf(stderr, "[fsr4] %s\n", buf);
    const char *path = getenv("FSR4_HOOK_LOG");
    if (!path || !path[0]) return;
    FILE *f = fopen(path, "a");
    if (!f) return;
    fprintf(f, "[fsr4] %s\n", buf);
    fclose(f);
}

// Descriptor flags D3DMetal set before creating the scaler. Read on the
// descriptor itself (self of newTemporalScalerWithDevice:), then attached to
// the scaler object: jittered MVs, output-resolution MVs, auto exposure.
static int desc_flag(id desc, SEL sel) {
    if (![desc respondsToSelector:sel]) return -1;
    BOOL (*get)(id, SEL) = (BOOL (*)(id, SEL))objc_msgSend;
    return get(desc, sel) ? 1 : 0;
}

static void remember_flags(id scaler, id desc, id<MTLDevice> device) {
    if (!scaler) return;
    Fsr4OwnerLifetime *lifetime = [Fsr4OwnerLifetime new];
    lifetime.device = device;
    lifetime.owner = (__bridge void *)scaler;
    objc_setAssociatedObject(scaler, kLifetimeAssoc, lifetime, OBJC_ASSOCIATION_RETAIN_NONATOMIC);
    int jit = desc_flag(desc, @selector(isJitteredMotionVectorsEnabled));
    int outres = desc_flag(desc, @selector(isOutputResolutionMotionVectorsEnabled));
    int ae = desc_flag(desc, @selector(isAutoExposureEnabled));
    int props = desc_flag(desc, @selector(isInputContentPropertiesEnabled));
    NSArray *flags = @[@(jit), @(outres), @(ae), @(props)];
    objc_setAssociatedObject(scaler, kFlagsAssoc, flags, OBJC_ASSOCIATION_RETAIN_NONATOMIC);
    MTLFXTemporalScalerDescriptor *d = desc;
    hook_log("scaler %p created: desc in %lux%lu out %lux%lu color %lu depth %lu motion %lu out %lu "
             "jitteredMV=%d outputResMV=%d autoExposure=%d inputContentProps=%d minScale %.3f maxScale %.3f",
             scaler, (unsigned long)d.inputWidth, (unsigned long)d.inputHeight,
             (unsigned long)d.outputWidth, (unsigned long)d.outputHeight,
             (unsigned long)d.colorTextureFormat, (unsigned long)d.depthTextureFormat,
             (unsigned long)d.motionTextureFormat, (unsigned long)d.outputTextureFormat,
             jit, outres, ae, props, d.inputContentMinScale, d.inputContentMaxScale);
}

static NSUInteger optional_offset(id scaler, const char *name) {
    SEL sel = sel_registerName(name);
    if (![scaler respondsToSelector:sel]) return 0;
    NSUInteger (*get)(id, SEL) = (NSUInteger (*)(id, SEL))objc_msgSend;
    return get(scaler, sel);
}

static int capture_surface(Fsr4InputSurface *surface, id<MTLTexture> texture,
                           NSUInteger x, NSUInteger y) {
    if (texture.width > UINT32_MAX || texture.height > UINT32_MAX || x > UINT32_MAX || y > UINT32_MAX)
        return 0;
    surface->width = (uint32_t)texture.width;
    surface->height = (uint32_t)texture.height;
    surface->offsetX = (uint32_t)x; surface->offsetY = (uint32_t)y;
    return 1;
}

// Returns 1 if this encode was consumed by FSR4.
static int fsr4_replace_encode(id scaler, id commandBuffer) {
    id<MTLFXTemporalScaler> s = scaler;
    id<MTLTexture> color = s.colorTexture;
    id<MTLTexture> depth = s.depthTexture;
    id<MTLTexture> motion = s.motionTexture;
    id<MTLTexture> output = s.outputTexture;
    if (!color || !depth || !motion || !output) { fsr4_live_set_inactive("scaler has no textures"); return 0; }
    NSUInteger rw = s.inputContentWidth, rh = s.inputContentHeight;
    // An unset content extent uses the descriptor's input size, never an
    // oversized backing texture. A half-specified extent is rejected by the
    // shared contract instead of independently guessing each dimension.
    if (!rw && !rh) { rw = s.inputWidth; rh = s.inputHeight; }
    const NSUInteger uw = s.outputWidth, uh = s.outputHeight;
    if (rw > UINT32_MAX || rh > UINT32_MAX || uw > UINT32_MAX || uh > UINT32_MAX
        || s.inputWidth > UINT32_MAX || s.inputHeight > UINT32_MAX) {
        fsr4_live_set_inactive("scaler size does not fit the input ABI"); return 0;
    }
    NSArray *flags = objc_getAssociatedObject(scaler, kFlagsAssoc);
    Fsr4LiveFrameV2 v2;
    memset(&v2, 0, sizeof v2);
    v2.structSize = sizeof v2;
    v2.abiVersion = FSR4_LIVE_FRAME_ABI_V2;
    Fsr4LiveFrame *f = &v2.frame;
    f->device = (__bridge void *)color.device;
    f->commandBuffer = (__bridge void *)commandBuffer;
    f->owner = (__bridge void *)scaler;
    f->color = (__bridge void *)color;
    f->depth = (__bridge void *)depth;
    f->motion = (__bridge void *)motion;
    f->output = (__bridge void *)output;
    f->exposure = (__bridge void *)s.exposureTexture;
    f->rw = (uint32_t)rw; f->rh = (uint32_t)rh;
    f->uw = (uint32_t)uw; f->uh = (uint32_t)uh;
    // D3DMetal already performed the NGX -> MetalFX conversion. The public
    // MetalFX properties use the same pixel sampling / previous-current
    // conventions as FSR; no additional sign flip or exposure multiplication.
    f->jitterX = s.jitterOffsetX; f->jitterY = s.jitterOffsetY;
    f->mvScaleX = s.motionVectorScaleX; f->mvScaleY = s.motionVectorScaleY;
    f->preExposure = s.preExposure;
    f->reset = s.reset ? 1 : 0;
    f->depthReversed = s.depthReversed ? 1 : 0;
    f->jitteredMV = flags.count > 0 ? [flags[0] intValue] : -1;
    f->outputResMV = flags.count > 1 ? [flags[1] intValue] : -1;
    f->autoExposure = flags.count > 2 ? [flags[2] intValue] : -1;
    v2.inputContentProperties = flags.count > 3 ? [flags[3] intValue] : -1;
    v2.geometry.maxInputWidth = (uint32_t)s.inputWidth;
    v2.geometry.maxInputHeight = (uint32_t)s.inputHeight;
    if (!capture_surface(&v2.geometry.color, color, optional_offset(scaler, "colorContentOffsetX"), optional_offset(scaler, "colorContentOffsetY"))
        || !capture_surface(&v2.geometry.depth, depth, optional_offset(scaler, "depthContentOffsetX"), optional_offset(scaler, "depthContentOffsetY"))
        || !capture_surface(&v2.geometry.motion, motion, optional_offset(scaler, "motionContentOffsetX"), optional_offset(scaler, "motionContentOffsetY"))
        || !capture_surface(&v2.geometry.output, output, optional_offset(scaler, "outputOffsetX"), optional_offset(scaler, "outputOffsetY"))) {
        fsr4_live_set_inactive("texture size or offset does not fit the input ABI"); return 0;
    }
    f->mvOffX = v2.geometry.motion.offsetX; f->mvOffY = v2.geometry.motion.offsetY;
    if (fsr4_encode_live_frame_v2) return fsr4_encode_live_frame_v2(&v2);

    // Compatibility with a V1 encoder is limited to its actual contract.
    // More involved input must reach the original MetalFX implementation.
    Fsr4InputDesc input = fsr4_live_input_desc(f, &v2.geometry, v2.inputContentProperties);
    Fsr4InputPrepared prepared;
    Fsr4InputStatus status = fsr4_input_prepare(&input, NULL, 0, &prepared);
    if (!fsr4_input_status_ok(status)) {
        fsr4_live_set_inactive(fsr4_input_status_reason(status)); return 0;
    }
    if (prepared.copyMask || (!prepared.lowResMV && (prepared.uw != prepared.alignedUW || prepared.uh != prepared.alignedUH))) {
        fsr4_live_set_inactive("this input region needs live ABI v2"); return 0;
    }
    return fsr4_encode_live_frame(f);
}

static void hooked_encode(id self, SEL _cmd, id commandBuffer) {
    IMP orig = orig_encode(object_getClass(self));
    const char *rep = getenv("FSR4_REPLACE");
    if (rep && rep[0] == '0') fsr4_live_set_inactive("FSR4_REPLACE=0");
    else if (fsr4_replace_encode(self, commandBuffer))
        return;
    Fsr4OwnerLifetime *lifetime = objc_getAssociatedObject(self, kLifetimeAssoc);
    if (lifetime && fsr4_invalidate_live_owner)
        fsr4_invalidate_live_owner((__bridge void *)lifetime.device, (__bridge void *)self);
    if (!orig) {
        hook_log("encode: no original IMP on %s", class_getName(object_getClass(self)));
        return;
    }
    ((void (*)(id, SEL, id))orig)(self, _cmd, commandBuffer);
}

static void install_encode(Class cls) {
    if (!cls || orig_encode(cls)) return;
    SEL sel = @selector(encodeToCommandBuffer:);
    Method m = class_getInstanceMethod(cls, sel);
    if (!m) {
        hook_log("no encodeToCommandBuffer: on %s", class_getName(cls));
        return;
    }
    int slot = -1;
    for (int i = 0; i < 4; i++)
        if (!g_encode[i].cls) { slot = i; break; }
    if (slot < 0) {
        hook_log("encode swizzle table full");
        return;
    }
    g_encode[slot].cls = cls;
    g_encode[slot].imp = method_getImplementation(m);
    method_setImplementation(m, (IMP)hooked_encode);
    hook_log("swizzled %s encodeToCommandBuffer:", class_getName(cls));
}

static id hooked_new(id self, SEL _cmd, id device) {
    id scaler = ((id (*)(id, SEL, id))g_orig_new)(self, _cmd, device);
    if (scaler) {
        remember_flags(scaler, self, device);
        install_encode(object_getClass(scaler));
    }
    return scaler;
}

static id hooked_new4(id self, SEL _cmd, id device, id compiler) {
    id scaler = ((id (*)(id, SEL, id, id))g_orig_new4)(self, _cmd, device, compiler);
    if (scaler) {
        remember_flags(scaler, self, device);
        install_encode(object_getClass(scaler));
    }
    return scaler;
}

static void swizzle_new(Class cls, SEL sel, IMP repl, IMP *slot) {
    Method m = class_getInstanceMethod(cls, sel);
    if (!m) return;
    *slot = method_getImplementation(m);
    method_setImplementation(m, repl);
    hook_log("swizzled %s %s", class_getName(cls), sel_getName(sel));
}

__attribute__((constructor))
static void fsr4_hook_init(void) {
    @autoreleasepool {
    hook_log("loaded pid=%d", getpid());
    Class desc = objc_getClass("MTLFXTemporalScalerDescriptor");
    if (!desc) {
        hook_log("MTLFXTemporalScalerDescriptor missing");
        return;
    }
    swizzle_new(desc, @selector(newTemporalScalerWithDevice:), (IMP)hooked_new, &g_orig_new);
    swizzle_new(desc, @selector(newTemporalScalerWithDevice:compiler:), (IMP)hooked_new4, &g_orig_new4);
    }
}
