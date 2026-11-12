// SPDX-License-Identifier: MIT
// Independent, opt-in MetalFX FG replacement. The SR hook is not modified.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalFX/MetalFX.h>
#import <objc/runtime.h>
#import <objc/message.h>
#include "anr_live.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static IMP originalNew=NULL,originalNew4=NULL;
static const void *lifetimeKey=&lifetimeKey;
static const void *distortionKey=&distortionKey;
typedef struct {Class cls;IMP original;} AnrEncodeHook;
static AnrEncodeHook slots[8];
@interface AnrOwnerLifetime : NSObject
@property(nonatomic,strong) id<MTLDevice> device;
@property(nonatomic,assign) void *owner;
@end
@implementation AnrOwnerLifetime
- (void)dealloc { fsr4_anr_release_owner((__bridge void*)_device,_owner); }
@end
static void logHook(const char *message,Class cls){
    fprintf(stderr,"[anr-fg] %s %s\n",message,cls?class_getName(cls):"");
    const char *path=getenv("FSR4_HOOK_LOG");if(path&&*path)if(FILE *file=fopen(path,"a")){
        fprintf(file,"[anr-fg] %s %s\n",message,cls?class_getName(cls):"");fclose(file);
    }
}
static IMP findOriginal(Class cls){
    for(Class current=cls;current;current=class_getSuperclass(current))
        for(unsigned i=0;i<8;++i)if(slots[i].cls==current)return slots[i].original;
    return NULL;
}
static NSUInteger optionalExtent(id object,const char *name){
    SEL sel=sel_registerName(name);if(![object respondsToSelector:sel])return 0;
    return ((NSUInteger(*)(id,SEL))objc_msgSend)(object,sel);
}
static int replace(id object,id command){
    const char *mode=getenv("FSR4_FG");if(!mode||strcmp(mode,"anr")){fsr4_anr_set_inactive("MetalFX FG selected");return 0;}
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
    // Never guess an output colorspace different from the input. SRGB textures
    // linearize reads/encode writes in Metal, so they use the linear route.
    if(fg.colorTexture.pixelFormat!=fg.outputTexture.pixelFormat){fsr4_anr_set_inactive("ANR: input/output color formats differ");return 0;}
    MTLPixelFormat format=fg.colorTexture.pixelFormat;
    if(format==MTLPixelFormatRGB10A2Unorm)f.colorEncoding=2;
    else if(format==MTLPixelFormatBGRA8Unorm||format==MTLPixelFormatRGBA8Unorm)f.colorEncoding=1;
    else if(format==MTLPixelFormatRGBA16Float||format==MTLPixelFormatRGB9E5Float||format==MTLPixelFormatRG11B10Float
         ||format==MTLPixelFormatBGRA8Unorm_sRGB||format==MTLPixelFormatRGBA8Unorm_sRGB)f.colorEncoding=0;
    else {fsr4_anr_set_inactive("ANR: color format unsupported");return 0;}
    const char *encoding=getenv("FSR4_ANR_COLOR");
    if(encoding&&*encoding){
        if(!strcmp(encoding,"linear"))f.colorEncoding=0;
        else if(!strcmp(encoding,"srgb"))f.colorEncoding=1;
        else if(!strcmp(encoding,"pq10"))f.colorEncoding=2;
        else {fsr4_anr_set_inactive("ANR: FSR4_ANR_COLOR must be linear, srgb or pq10");return 0;}
    }
    return fsr4_anr_encode_frame(&f);
}
static void encode(id self,SEL selector,id command){@autoreleasepool{
    IMP original=findOriginal(object_getClass(self));if(!original){logHook("original FG encode unavailable",object_getClass(self));return;}
    int result=replace(self,command);
    if(result==1)return;
    // The first/reset frame establishes ANR history before original MetalFX.
    // Do not invalidate that successful priming when its original encode runs.
    if(result!=2){AnrOwnerLifetime *lifetime=objc_getAssociatedObject(self,lifetimeKey);
        if(lifetime)fsr4_anr_invalidate_owner((__bridge void*)lifetime.device,(__bridge void*)self);}
    ((void(*)(id,SEL,id))original)(self,selector,command);
}}
static void remember(id object,id descriptor,id<MTLDevice> device){
    if(!object)return;
    AnrOwnerLifetime *lifetime=[AnrOwnerLifetime new];lifetime.device=device;lifetime.owner=(__bridge void*)object;
    objc_setAssociatedObject(object,lifetimeKey,lifetime,OBJC_ASSOCIATION_RETAIN_NONATOMIC);
    BOOL distortion=NO;SEL sel=sel_registerName("isDistortionTextureEnabled");
    if([descriptor respondsToSelector:sel])distortion=((BOOL(*)(id,SEL))objc_msgSend)(descriptor,sel);
    objc_setAssociatedObject(object,distortionKey,@(distortion),OBJC_ASSOCIATION_RETAIN_NONATOMIC);
    Class cls=object_getClass(object);@synchronized([MTLFXFrameInterpolatorDescriptor class]){
        for(unsigned i=0;i<8;++i)if(slots[i].cls==cls)return;
        Method method=class_getInstanceMethod(cls,@selector(encodeToCommandBuffer:));if(!method)return;
        IMP current=method_getImplementation(method);
        // A subclass inheriting our already-swizzled encode uses its parent's
        // recorded original. Never record our replacement as the original.
        if(current==(IMP)encode)return;
        for(unsigned i=0;i<8;++i)if(!slots[i].cls){slots[i]={cls,current};
            class_addMethod(cls,@selector(encodeToCommandBuffer:),current,method_getTypeEncoding(method));
            method=class_getInstanceMethod(cls,@selector(encodeToCommandBuffer:));method_setImplementation(method,(IMP)encode);
            logHook("hooked encode",cls);return;}
        logHook("FG class table full; original retained",cls);
    }
}
static id newFG(id self,SEL selector,id device){id object=((id(*)(id,SEL,id))originalNew)(self,selector,device);remember(object,self,device);return object;}
static id newFG4(id self,SEL selector,id device,id compiler){id object=((id(*)(id,SEL,id,id))originalNew4)(self,selector,device,compiler);remember(object,self,device);return object;}
__attribute__((constructor))static void initializeFG(){@autoreleasepool{
    Class cls=objc_getClass("MTLFXFrameInterpolatorDescriptor");if(!cls){logHook("FG descriptor unavailable",NULL);return;}
    Method method=class_getInstanceMethod(cls,@selector(newFrameInterpolatorWithDevice:));
    if(method){originalNew=method_getImplementation(method);method_setImplementation(method,(IMP)newFG);}
    method=class_getInstanceMethod(cls,@selector(newFrameInterpolatorWithDevice:compiler:));
    if(method){originalNew4=method_getImplementation(method);method_setImplementation(method,(IMP)newFG4);}
    logHook("descriptor hooks ready",cls);
}}
