// SPDX-License-Identifier: MIT
// ANR owns the FG method only while selected. MetalFX restores its original
// IMP: no per-frame mode polling, ANR mutex, copies or kernels on native FG.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalFX/MetalFX.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <dispatch/dispatch.h>
#include "anr_live.h"
#include "fsr4_fg_settings.h"
#include <atomic>
#include <mutex>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static IMP originalNew=NULL,originalNew4=NULL;
static const void *lifetimeKey=&lifetimeKey;
static const void *distortionKey=&distortionKey;
static std::atomic<int> selected{0};
static std::mutex methodMutex,settingsMutex;
static NSHashTable *owners=nil;
static dispatch_source_t settingsTimer;
static void encode(id,SEL,id);
static void resetNativeOnce(id,SEL,id);
struct AtomicClass {
    std::atomic<void*> value{nullptr};
    Class load(std::memory_order order=std::memory_order_seq_cst)const{return (__bridge Class)value.load(order);}
    void store(Class cls,std::memory_order order=std::memory_order_seq_cst){value.store((__bridge void*)cls,order);}
};
struct AnrEncodeHook {AtomicClass cls;std::atomic<IMP> original{NULL};};
static AnrEncodeHook slots[8];
@interface AnrOwnerLifetime : NSObject {
@public std::atomic<bool> resetPending,originalDirty; IMP original;
}
@property(nonatomic,strong) id<MTLDevice> device;
@property(nonatomic,assign) void *owner;
@property(nonatomic,assign) Class hookClass;
@end
@implementation AnrOwnerLifetime
- (id)init {self=[super init];if(self){resetPending.store(false);originalDirty.store(false);}return self;}
- (void)dealloc {fsr4_anr_release_owner((__bridge void*)_device,_owner);}
@end
static void logHook(const char *message,Class cls){
    fprintf(stderr,"[anr-fg] %s %s\n",message,cls?class_getName(cls):"");
    const char *path=getenv("FSR4_HOOK_LOG");if(path&&*path)if(FILE *file=fopen(path,"a")){
        fprintf(file,"[anr-fg] %s %s\n",message,cls?class_getName(cls):"");fclose(file);
    }
}
static IMP findOriginal(Class cls){
    for(Class current=cls;current;current=class_getSuperclass(current))
        for(auto &slot:slots)if(slot.cls.load(std::memory_order_acquire)==current)return slot.original.load();
    return NULL;
}
static bool pendingReset(Class cls){
    for(AnrOwnerLifetime *life in owners.allObjects)
        if(life->resetPending.load(std::memory_order_acquire)&&life.hookClass==cls)return true;
    return false;
}
static void setEncode(Class cls,IMP original,IMP replacement){
    SEL sel=@selector(encodeToCommandBuffer:);Method m=class_getInstanceMethod(cls,sel);
    if(!m)return;
    IMP current=method_getImplementation(m);
    if(current!=original&&current!=(IMP)encode&&current!=(IMP)resetNativeOnce){
        logHook("foreign FG encode retained; no competing swizzle",cls);return;
    }
    if(current==replacement)return;
    // Do not mutate a superclass's inherited implementation.
    class_addMethod(cls,sel,original,method_getTypeEncoding(m));
    method_setImplementation(class_getInstanceMethod(cls,sel),replacement);
}
extern "C" int fsr4_anr_is_selected(void){return selected.load(std::memory_order_acquire);}
extern "C" void fsr4_anr_refresh_settings(void){@autoreleasepool{
    std::lock_guard<std::mutex> refresh(settingsMutex);
    fsr4_fg::Mode mode=fsr4_fg::Mode::MetalFX;
    if(const char *env=getenv("FSR4_FG"))fsr4_fg::parse_mode(env,mode);
    auto path=fsr4_fg::settings_path(getenv("FSR4_ROOT"),getenv("FSR4_FG_SETTINGS"));
    if(!path.empty()){
        const auto saved=fsr4_fg::read_settings(path);
        if(saved.valid)mode=saved.mode;
    }
    const int want=mode==fsr4_fg::Mode::ANR;
    const int before=selected.load(std::memory_order_acquire);
    if(before!=want){
        // Stop new ANR entries before retiring resources. A frame already
        // being encoded completes normally; the mode mutex is off that path.
        if(!want){
            std::lock_guard<std::mutex> lock(methodMutex);
            for(AnrOwnerLifetime *life in owners.allObjects)if(life->originalDirty.load())life->resetPending.store(true,std::memory_order_release);
            for(auto &slot:slots){Class cls=slot.cls.load();if(cls)setEncode(cls,slot.original.load(),(IMP)resetNativeOnce);}
            selected.store(0,std::memory_order_release);
        }
        fsr4_anr_mode_changed(want);
        if(want){
            std::lock_guard<std::mutex> lock(methodMutex);
            for(auto &slot:slots){Class cls=slot.cls.load();if(cls)setEncode(cls,slot.original.load(),(IMP)encode);}
            selected.store(1,std::memory_order_release);
        }
        logHook(want?"ANR selected; FG method replaced":"MetalFX selected; restoring native FG methods",Nil);
    }
    if(!want){
        {std::lock_guard<std::mutex> lock(methodMutex);
         for(auto &slot:slots){Class cls=slot.cls.load();if(!cls)continue;
             if(method_getImplementation(class_getInstanceMethod(cls,@selector(encodeToCommandBuffer:)))==(IMP)resetNativeOnce&&!pendingReset(cls))
                 setEncode(cls,slot.original.load(),slot.original.load());
         }}
        fsr4_anr_publish_passthrough();
    }
}}
static NSUInteger optionalExtent(id object,const char *name){
    SEL sel=sel_registerName(name);if(![object respondsToSelector:sel])return 0;
    return ((NSUInteger(*)(id,SEL))objc_msgSend)(object,sel);
}
static int replace(id object,id command){
    id<MTLFXFrameInterpolatorBase> fg=object;
    if([objc_getAssociatedObject(object,distortionKey) boolValue]){fsr4_anr_set_inactive("ANR: distortion texture unsupported");return 0;}
    for(const char *name:{"depthContentOffsetX","depthContentOffsetY","motionContentOffsetX","motionContentOffsetY","outputOffsetX","outputOffsetY"})
        if(optionalExtent(object,name)){fsr4_anr_set_inactive("ANR: FG subrect offsets unsupported");return 0;}
    Fsr4AnrFrame f={};f.structSize=sizeof(f);f.abiVersion=FSR4_ANR_FG_ABI;
    f.owner=(__bridge void*)object;f.commandBuffer=(__bridge void*)command;
    f.color=(__bridge void*)fg.colorTexture;f.previousColor=(__bridge void*)fg.prevColorTexture;
    f.depth=(__bridge void*)fg.depthTexture;f.motion=(__bridge void*)fg.motionTexture;f.output=(__bridge void*)fg.outputTexture;
    f.ui=(__bridge void*)fg.uiTexture;f.uiComposited=fg.isUITextureComposited;
    f.fence=(__bridge void*)fg.fence;f.device=(__bridge void*)fg.colorTexture.device;
    NSUInteger rw=optionalExtent(object,"contentWidth"),rh=optionalExtent(object,"contentHeight");
    if(!rw&&!rh){rw=fg.inputWidth;rh=fg.inputHeight;}
    if(rw>UINT32_MAX||rh>UINT32_MAX||fg.outputWidth>UINT32_MAX||fg.outputHeight>UINT32_MAX)return 0;
    f.rw=(uint32_t)rw;f.rh=(uint32_t)rh;f.uw=(uint32_t)fg.outputWidth;f.uh=(uint32_t)fg.outputHeight;
    f.motionScaleX=fg.motionVectorScaleX;f.motionScaleY=fg.motionVectorScaleY;
    f.jitterX=fg.jitterOffsetX;f.jitterY=fg.jitterOffsetY;
    f.nearPlane=fg.nearPlane;f.farPlane=fg.farPlane;f.fieldOfView=fg.fieldOfView;f.aspectRatio=fg.aspectRatio;f.deltaTime=fg.deltaTime;
    f.reset=fg.shouldResetHistory;f.depthReversed=fg.isDepthReversed;
    if(fg.colorTexture.pixelFormat!=fg.outputTexture.pixelFormat){fsr4_anr_set_inactive("ANR: input/output color formats differ");return 0;}
    MTLPixelFormat format=fg.colorTexture.pixelFormat;
    if(format==MTLPixelFormatRGB10A2Unorm)f.colorEncoding=2;
    else if(format==MTLPixelFormatBGRA8Unorm||format==MTLPixelFormatRGBA8Unorm)f.colorEncoding=1;
    else if(format==MTLPixelFormatRGBA16Float||format==MTLPixelFormatRGB9E5Float||format==MTLPixelFormatRG11B10Float
         ||format==MTLPixelFormatBGRA8Unorm_sRGB||format==MTLPixelFormatRGBA8Unorm_sRGB)f.colorEncoding=0;
    else {fsr4_anr_set_inactive("ANR: color format unsupported");return 0;}
    if(const char *encoding=getenv("FSR4_ANR_COLOR"))if(*encoding){
        if(!strcmp(encoding,"linear"))f.colorEncoding=0;
        else if(!strcmp(encoding,"srgb"))f.colorEncoding=1;
        else if(!strcmp(encoding,"pq10"))f.colorEncoding=2;
        else {fsr4_anr_set_inactive("ANR: FSR4_ANR_COLOR must be linear, srgb or pq10");return 0;}
    }
    return fsr4_anr_encode_frame(&f);
}
static void resetNativeOnce(id self,SEL selector,id command){@autoreleasepool{
    AnrOwnerLifetime *life=objc_getAssociatedObject(self,lifetimeKey);
    IMP original=life&&life->original?life->original:findOriginal(object_getClass(self));if(!original)return;
    // Cached but inactive interpolators may still owe their first reset.
    // Already-reset instances do not scan that registry on every frame.
    if(life&&!life->resetPending.load(std::memory_order_acquire)&&!life->originalDirty.load(std::memory_order_acquire)){
        ((void(*)(id,SEL,id))original)(self,selector,command);return;
    }
    id<MTLFXFrameInterpolatorBase> fg=self;
    const BOOL previous=fg.shouldResetHistory;
    bool reset=false;
    if(life){reset=life->resetPending.exchange(false,std::memory_order_acq_rel);reset|=life->originalDirty.exchange(false,std::memory_order_acq_rel);}
    if(reset)fg.shouldResetHistory=YES;
    ((void(*)(id,SEL,id))original)(self,selector,command);
    fg.shouldResetHistory=previous;
    // One transition frame per live interpolator. Thereafter the class's
    // method is exactly the original IMP, including Objective-C dispatch.
    if(!selected.load(std::memory_order_acquire)){
        std::lock_guard<std::mutex> lock(methodMutex);
        Class cls=life.hookClass?:object_getClass(self);
        if(!selected.load()&&!pendingReset(cls))setEncode(cls,original,original);
    }
}}
static void encode(id self,SEL selector,id command){@autoreleasepool{
    IMP original=findOriginal(object_getClass(self));if(!original){logHook("original FG encode unavailable",object_getClass(self));return;}
    if(!selected.load(std::memory_order_acquire)){resetNativeOnce(self,selector,command);return;}
    int result=replace(self,command);
    AnrOwnerLifetime *life=objc_getAssociatedObject(self,lifetimeKey);
    if(result==1){if(life)life->originalDirty.store(true,std::memory_order_release);return;}
    if(result!=2&&life)fsr4_anr_invalidate_owner((__bridge void*)life.device,(__bridge void*)self);
    id<MTLFXFrameInterpolatorBase> fg=self;const BOOL previous=fg.shouldResetHistory;
    bool reset=result==2;
    if(life){reset|=life->resetPending.exchange(false,std::memory_order_acq_rel);reset|=life->originalDirty.exchange(false,std::memory_order_acq_rel);}
    if(reset)fg.shouldResetHistory=YES;
    ((void(*)(id,SEL,id))original)(self,selector,command);fg.shouldResetHistory=previous;
}}
static void remember(id object,id descriptor,id<MTLDevice> device){
    if(!object)return;
    AnrOwnerLifetime *life=[AnrOwnerLifetime new];life.device=device;life.owner=(__bridge void*)object;
    objc_setAssociatedObject(object,lifetimeKey,life,OBJC_ASSOCIATION_RETAIN_NONATOMIC);
    BOOL distortion=NO;SEL sel=sel_registerName("isDistortionTextureEnabled");
    if([descriptor respondsToSelector:sel])distortion=((BOOL(*)(id,SEL))objc_msgSend)(descriptor,sel);
    objc_setAssociatedObject(object,distortionKey,@(distortion),OBJC_ASSOCIATION_RETAIN_NONATOMIC);
    std::lock_guard<std::mutex> lock(methodMutex);
    if(!owners)owners=[NSHashTable weakObjectsHashTable];[owners addObject:life];
    Class cls=object_getClass(object);
    for(auto &slot:slots)if(slot.cls.load()==cls){life.hookClass=cls;life->original=slot.original.load();return;}
    Method method=class_getInstanceMethod(cls,@selector(encodeToCommandBuffer:));if(!method)return;
    IMP current=method_getImplementation(method);
    if(current==(IMP)encode||current==(IMP)resetNativeOnce){
        for(Class parent=class_getSuperclass(cls);parent;parent=class_getSuperclass(parent))
            for(auto &slot:slots)if(slot.cls.load()==parent){life.hookClass=parent;life->original=slot.original.load();return;}
        return;
    }
    for(auto &slot:slots)if(!slot.cls.load()){
        life.hookClass=cls;life->original=current;slot.original.store(current);slot.cls.store(cls,std::memory_order_release);
        if(selected.load())setEncode(cls,current,(IMP)encode);
        logHook(selected.load()?"hooked encode":"native FG encode retained",cls);return;
    }
    logHook("FG class table full; original retained",cls);
}
static id newFG(id self,SEL selector,id device){id object=((id(*)(id,SEL,id))originalNew)(self,selector,device);fsr4_anr_refresh_settings();remember(object,self,device);return object;}
static id newFG4(id self,SEL selector,id device,id compiler){id object=((id(*)(id,SEL,id,id))originalNew4)(self,selector,device,compiler);fsr4_anr_refresh_settings();remember(object,self,device);return object;}
__attribute__((constructor))static void initializeFG(){@autoreleasepool{
    Class cls=objc_getClass("MTLFXFrameInterpolatorDescriptor");if(!cls){logHook("FG descriptor unavailable",Nil);return;}
    Method method=class_getInstanceMethod(cls,@selector(newFrameInterpolatorWithDevice:));
    if(method){originalNew=method_getImplementation(method);method_setImplementation(method,(IMP)newFG);}
    method=class_getInstanceMethod(cls,@selector(newFrameInterpolatorWithDevice:compiler:));
    if(method){originalNew4=method_getImplementation(method);method_setImplementation(method,(IMP)newFG4);}
    fsr4_anr_refresh_settings();
    auto queue=dispatch_queue_create("org.fsr4mac.fg-control",DISPATCH_QUEUE_SERIAL);
    settingsTimer=dispatch_source_create(DISPATCH_SOURCE_TYPE_TIMER,0,0,queue);
    dispatch_source_set_timer(settingsTimer,dispatch_time(DISPATCH_TIME_NOW,500*NSEC_PER_MSEC),500*NSEC_PER_MSEC,50*NSEC_PER_MSEC);
    dispatch_source_set_event_handler(settingsTimer,^{fsr4_anr_refresh_settings();});dispatch_resume(settingsTimer);
    logHook("descriptor hooks ready; native MetalFX has no frame hook",cls);
}}
