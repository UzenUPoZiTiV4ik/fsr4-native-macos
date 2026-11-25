// SPDX-License-Identifier: MIT
// Create the actual MetalFX FG seam and assert replacement, priming and opt-out.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalFX/MetalFX.h>
#import <QuartzCore/QuartzCore.h>
#import <objc/runtime.h>
#include "../../src/anr/anr_live.h"
#include "../../src/common/fsr4_fg_settings.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>
#include <thread>
#include <chrono>
static void need(bool value,const char *why){if(!value)throw std::runtime_error(why);}
static id<MTLTexture> texture(id<MTLDevice> device,unsigned w,unsigned h,MTLPixelFormat format,bool privateMode=false){
    auto d=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format width:w height:h mipmapped:NO];
    d.storageMode=privateMode?MTLStorageModePrivate:MTLStorageModeShared;d.usage=MTLTextureUsageShaderRead|MTLTextureUsageShaderWrite|MTLTextureUsageRenderTarget;
    return [device newTextureWithDescriptor:d];
}
int main(int argc,char **argv){@autoreleasepool{try{
    need(argc>=3,"usage: anr_live_test <root> <runtime> [metal4|classic]");
    setenv("FSR4_ROOT",argv[1],1);setenv("FSR4_ANR_RUNTIME",argv[2],1);setenv("FSR4_FG","anr",1);
    setenv("FSR4_ANR_STATUS_DISABLED","1",1);setenv("FSR4_FG_SETTINGS","none",1);
    const bool metal4=argc>3&&!strcmp(argv[3],"metal4");
    setenv("FSR4_ANR_COLOR","pq10",1);
    id<MTLDevice> device=MTLCreateSystemDefaultDevice();need(device!=nil,"no device");
    NSData *metadata=[NSData dataWithContentsOfFile:[[NSString stringWithUTF8String:argv[2]] stringByAppendingPathComponent:@"manifest.json"]];
    NSDictionary *manifest=metadata?[NSJSONSerialization JSONObjectWithData:metadata options:0 error:nil]:nil;
    const bool isV2=[manifest[@"schema"] isEqual:@"fsr4_anr_runtime_v2"];
    unsigned w=isV2?(getenv("ANR_TEST_WIDTH")?(unsigned)atoi(getenv("ANR_TEST_WIDTH")):1920u):[manifest[@"width"] unsignedIntValue];
    unsigned h=isV2?(getenv("ANR_TEST_HEIGHT")?(unsigned)atoi(getenv("ANR_TEST_HEIGHT")):1080u):[manifest[@"height"] unsignedIntValue];
    need(w&&h,"runtime geometry unavailable");
    MTLFXFrameInterpolatorDescriptor *d=[MTLFXFrameInterpolatorDescriptor new];d.inputWidth=w;d.inputHeight=h;d.outputWidth=w;d.outputHeight=h;
    const bool testHdr=getenv("ANR_TEST_HDR")!=nullptr;
    const bool testSdr=!testHdr&&getenv("ANR_TEST_SDR")!=nullptr;
    MTLPixelFormat colorFormat=testHdr?MTLPixelFormatRGBA16Float:testSdr?MTLPixelFormatRGBA8Unorm:MTLPixelFormatRGB10A2Unorm;
    setenv("FSR4_ANR_COLOR",testHdr?"linear":testSdr?"srgb":"pq10",1);
    d.colorTextureFormat=d.outputTextureFormat=d.uiTextureFormat=colorFormat;d.depthTextureFormat=MTLPixelFormatR32Float;d.motionTextureFormat=MTLPixelFormatRG16Float;
    setenv("FSR4_FG","metalfx",1);fsr4_anr_refresh_settings();
    NSError *error=nil;id fg=nil;
    if(metal4){id<MTL4Compiler> compiler=[device newCompilerWithDescriptor:[MTL4CompilerDescriptor new] error:&error];
        fg=[d newFrameInterpolatorWithDevice:device compiler:compiler];}
    else fg=[d newFrameInterpolatorWithDevice:device];
    need(fg!=nil,"no MetalFX frame interpolator");
    const IMP nativeIMP=method_getImplementation(class_getInstanceMethod(object_getClass(fg),@selector(encodeToCommandBuffer:)));
    setenv("FSR4_FG","anr",1);fsr4_anr_refresh_settings();
    id<MTLFXFrameInterpolatorBase> base=fg;
    auto c=texture(device,w,h,colorFormat),prev=texture(device,w,h,colorFormat);
    auto z=texture(device,w,h,MTLPixelFormatR32Float),mv=texture(device,w,h,MTLPixelFormatRG16Float),out=texture(device,w,h,colorFormat,true);
    need(c&&prev&&z&&mv&&out,"textures unavailable");
    std::vector<uint32_t> color(size_t(w)*h);std::vector<float> depth(size_t(w)*h,.5f);std::vector<uint16_t> motion(size_t(w)*h*2,0);
    for(unsigned y=0;y<h;++y)for(unsigned x=0;x<w;++x){
        if(testSdr)color[size_t(y)*w+x]=0xff000000u|((x%256)<<8)|(y%256)|(((x/2)%256)<<16);
        else color[size_t(y)*w+x]=0xc0000000u|((x%1024)<<10)|(y%1024)|((x/2%1024)<<20);
    }
    std::vector<uint16_t> hdr;
    const void *inputBytes=color.data();size_t bpp=4;
    if(colorFormat==MTLPixelFormatRGBA16Float){hdr.resize(color.size()*4);bpp=8;
        for(size_t i=0;i<color.size();++i){_Float16 values[4]={(_Float16)((color[i]&1023)/1023.f),
            (_Float16)(((color[i]>>10)&1023)/1023.f),(_Float16)(((color[i]>>20)&1023)/1023.f),(_Float16)1};
            memcpy(hdr.data()+i*4,values,8);}inputBytes=hdr.data();}
    [c replaceRegion:MTLRegionMake2D(0,0,w,h) mipmapLevel:0 withBytes:inputBytes bytesPerRow:w*bpp];
    [prev replaceRegion:MTLRegionMake2D(0,0,w,h) mipmapLevel:0 withBytes:inputBytes bytesPerRow:w*bpp];
    [z replaceRegion:MTLRegionMake2D(0,0,w,h) mipmapLevel:0 withBytes:depth.data() bytesPerRow:w*4];
    [mv replaceRegion:MTLRegionMake2D(0,0,w,h) mipmapLevel:0 withBytes:motion.data() bytesPerRow:w*4];
    base.colorTexture=c;base.prevColorTexture=prev;base.depthTexture=z;base.motionTexture=mv;base.outputTexture=out;
    base.motionVectorScaleX=base.motionVectorScaleY=1;base.nearPlane=.1f;base.farPlane=1000;base.fieldOfView=60;base.aspectRatio=float(w)/h;
    base.deltaTime=1.f/60;base.depthReversed=NO;base.shouldResetHistory=YES;
    id<MTL4CommandQueue> q4=[device newMTL4CommandQueue];id<MTL4CommandAllocator> allocator=[device newCommandAllocator];
    id<MTLCommandQueue> q=[device newCommandQueue];id<MTLSharedEvent> event=[device newSharedEvent];uint64_t token=0;
    auto encode=[&]{
        if(metal4){id<MTL4CommandBuffer> cb=[device newCommandBuffer];[allocator reset];[cb beginCommandBufferWithAllocator:allocator];
            [(id<MTL4FXFrameInterpolator>)fg encodeToCommandBuffer:cb];[cb endCommandBuffer];id<MTL4CommandBuffer> buffers[]={cb};[q4 commit:buffers count:1];
            [q4 signalEvent:event value:++token];need([event waitUntilSignaledValue:token timeoutMS:20000],"GPU timeout");}
        else{auto cb=[q commandBuffer];[(id<MTLFXFrameInterpolator>)fg encodeToCommandBuffer:cb];[cb commit];[cb waitUntilCompleted];
            need(cb.status==MTLCommandBufferStatusCompleted,"Metal3 GPU failure");}
    };
    encode();Fsr4AnrStatus s={};fsr4_anr_get_status(&s);need(s.primed==1,"ANR did not prime actual FG hook");
    base.shouldResetHistory=NO;encode();fsr4_anr_get_status(&s);need(s.generated==1&&s.active,"ANR did not replace actual FG hook");
    const uint64_t beforeNative=s.encodes;
    char preferenceDirectory[]="/tmp/anr-live-switch-XXXXXX";need(mkdtemp(preferenceDirectory)!=nullptr,"FG preference directory failed");
    const std::string preference=std::string(preferenceDirectory)+"/fg-settings.conf";
    need(fsr4_fg::write_settings(preference,fsr4_fg::Mode::MetalFX),"native preference write failed");
    setenv("FSR4_FG_SETTINGS",preference.c_str(),1);
    std::this_thread::sleep_for(std::chrono::milliseconds(650));
    need(!fsr4_anr_is_selected(),"background controller did not apply MetalFX preference");encode();
    need(method_getImplementation(class_getInstanceMethod(object_getClass(fg),@selector(encodeToCommandBuffer:)))==nativeIMP,"MetalFX did not restore original IMP");
    encode();fsr4_anr_get_status(&s);need(s.generated==1&&s.encodes==beforeNative,"native mode entered ANR per-frame path");
    setenv("FSR4_FG","metalfx",1);
    need(fsr4_fg::write_settings(preference,fsr4_fg::Mode::ANR),"ANR preference write failed");
    std::this_thread::sleep_for(std::chrono::milliseconds(650));
    need(fsr4_anr_is_selected(),"saved ANR preference did not override native environment");
    encode();fsr4_anr_get_status(&s);need(s.primed==2,"fallback did not reset ANR history");
    encode();fsr4_anr_get_status(&s);need(s.generated==2,"ANR did not resume after fallback");
    // Queue three frames without waiting. Their CPU descriptor/constants
    // must not overwrite one another while the GPU is still working.
    const bool pipelined=getenv("ANR_TEST_SEQUENTIAL")==nullptr;
    if(!pipelined){for(unsigned i=0;i<3;++i)encode();}
    else if(metal4){
        id<MTL4CommandBuffer> pending[3]={};id<MTL4CommandAllocator> allocators[3]={};
        for(unsigned i=0;i<3;++i){pending[i]=[device newCommandBuffer];allocators[i]=[device newCommandAllocator];
            [pending[i] beginCommandBufferWithAllocator:allocators[i]];
            [(id<MTL4FXFrameInterpolator>)fg encodeToCommandBuffer:pending[i]];[pending[i] endCommandBuffer];}
        [q4 commit:pending count:3];[q4 signalEvent:event value:++token];
        need([event waitUntilSignaledValue:token timeoutMS:20000],"pipelined GPU timeout");
    }else{
        id<MTLCommandBuffer> pending[3]={};
        for(unsigned i=0;i<3;++i){pending[i]=[q commandBuffer];[(id<MTLFXFrameInterpolator>)fg encodeToCommandBuffer:pending[i]];[pending[i] commit];}
        [pending[2] waitUntilCompleted];for(auto cb:pending)need(cb.status==MTLCommandBufferStatusCompleted,"pipelined GPU failure");
    }
    fsr4_anr_get_status(&s);need(s.generated==5,"pipelined frames fell back or overwrote a slot");
    auto readback=[device newBufferWithLength:size_t(w)*h*bpp options:MTLResourceStorageModeShared];
    auto readOutput=[&]{
    if(metal4){auto cb=[device newCommandBuffer];[allocator reset];[cb beginCommandBufferWithAllocator:allocator];
        MTLResidencySetDescriptor *rd=[MTLResidencySetDescriptor new];rd.initialCapacity=2;
        auto residency=[device newResidencySetWithDescriptor:rd error:&error];need(residency!=nil,"readback residency missing");
        [residency addAllocation:out];[residency addAllocation:readback];[residency commit];[cb useResidencySet:residency];
        auto enc=[cb computeCommandEncoder];[enc barrierAfterQueueStages:MTLStageAll beforeStages:MTLStageBlit visibilityOptions:MTL4VisibilityOptionDevice];
        [enc copyFromTexture:out sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0,0,0) sourceSize:MTLSizeMake(w,h,1)
            toBuffer:readback destinationOffset:0 destinationBytesPerRow:w*bpp destinationBytesPerImage:size_t(w)*h*bpp];
        [enc endEncoding];[cb endCommandBuffer];id<MTL4CommandBuffer> buffers[]={cb};[q4 commit:buffers count:1];
        [q4 signalEvent:event value:++token];need([event waitUntilSignaledValue:token timeoutMS:20000],"readback timeout");
    }else{auto cb=[q commandBuffer];auto enc=[cb blitCommandEncoder];
        [enc copyFromTexture:out sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0,0,0) sourceSize:MTLSizeMake(w,h,1)
            toBuffer:readback destinationOffset:0 destinationBytesPerRow:w*bpp destinationBytesPerImage:size_t(w)*h*bpp];
        [enc endEncoding];[cb commit];[cb waitUntilCompleted];need(cb.status==MTLCommandBufferStatusCompleted,"readback failed");}
    };
    readOutput();
    const auto *actual=(const uint32_t*)readback.contents;double errorSum=0;unsigned maxDiff=0;
    if(colorFormat==MTLPixelFormatRGBA16Float){const auto *a=(const _Float16*)readback.contents,*b=(const _Float16*)hdr.data();
        for(size_t i=0;i<color.size();++i)for(unsigned channel=0;channel<3;++channel){float diff=fabsf(float(a[i*4+channel])-float(b[i*4+channel]));
            need(std::isfinite(diff),"nonfinite HDR output");errorSum+=diff*1023;maxDiff=std::max(maxDiff,(unsigned)(diff*1023));}}
    else if(testSdr){for(size_t i=0;i<color.size();++i)for(unsigned shift:{0u,8u,16u}){
        unsigned diff=(unsigned)abs(int((actual[i]>>shift)&255)-int((color[i]>>shift)&255));errorSum+=diff*4;maxDiff=std::max(maxDiff,diff*4);}}
    else for(size_t i=0;i<color.size();++i)for(unsigned shift:{0u,10u,20u}){
        unsigned diff=(unsigned)abs(int((actual[i]>>shift)&1023)-int((color[i]>>shift)&1023));errorSum+=diff;maxDiff=std::max(maxDiff,diff);}
    const double meanDiff=errorSum/(color.size()*3);
    // Opt-out must keep original behavior without reading a model or changing SR.
    printf("static-frame mean code error %.6f max %u (first=%08x source=%08x)\n",meanDiff,maxDiff,actual[0],color[0]);
    need(meanDiff<2.5,"stationary source-frame quality regressed");
    // A flag without an optional UI texture is not a precomposited input.
    base.uiTextureComposited=YES;encode();fsr4_anr_get_status(&s);
    need(s.generated==6&&s.active&&s.uiMode==FSR4_ANR_UI_NONE,"UI flag without texture rejected ANR");
    auto ui=texture(device,w,h,colorFormat);need(ui!=nil,"UI texture allocation failed");
    std::vector<uint8_t> composite(color.size()*bpp);memcpy(composite.data(),inputBytes,composite.size());
    auto inHud=[](unsigned x,unsigned y){return x>=31&&x<159&&y>=29&&y<93;};
    for(unsigned y=0;y<h;++y)for(unsigned x=0;x<w;++x)if(inHud(x,y)){
        auto *pixel=composite.data()+(size_t(y)*w+x)*bpp;
        if(colorFormat==MTLPixelFormatRGBA16Float){_Float16 value[4]={(_Float16).9f,(_Float16).03f,(_Float16).63f,(_Float16)0};memcpy(pixel,value,8);}
        else if(testSdr){uint32_t value=230u|(8u<<8)|(161u<<16);memcpy(pixel,&value,4);}
        else{uint32_t value=921u|(31u<<10)|(644u<<20);memcpy(pixel,&value,4);}
    }
    [ui replaceRegion:MTLRegionMake2D(0,0,w,h) mipmapLevel:0 withBytes:composite.data() bytesPerRow:w*bpp];
    base.uiTexture=ui;encode();fsr4_anr_get_status(&s);
    need(s.generated==7&&s.active&&s.uiComposited==1&&s.uiMode==FSR4_ANR_UI_PRECOMPOSITED,"precomposited UI did not replace MetalFX");
    readOutput();unsigned hudPixels=0;
    const uint32_t rgbMask=testSdr?0x00ffffffu:0x3fffffffu;
    for(unsigned y=0;y<h;++y)for(unsigned x=0;x<w;++x)if(inHud(x,y)){
        const size_t offset=(size_t(y)*w+x)*bpp;
        if(bpp==8)need(!memcmp((const uint8_t*)readback.contents+offset,composite.data()+offset,6),"HDR HUD pixels changed");
        if(bpp==4){uint32_t a,b;memcpy(&a,(const uint8_t*)readback.contents+offset,4);memcpy(&b,composite.data()+offset,4);
            need((a&rgbMask)==(b&rgbMask),"HUD pixels changed");}
        ++hudPixels;
    }
    // A separate premultiplied overlay uses alpha; here it is intentionally
    // opaque at the same screen-space HUD pixels and transparent elsewhere.
    std::vector<uint8_t> overlay(composite.size(),0);
    for(unsigned y=0;y<h;++y)for(unsigned x=0;x<w;++x)if(inHud(x,y)){
        auto *pixel=overlay.data()+(size_t(y)*w+x)*bpp;
        if(bpp==8){_Float16 value[4]={(_Float16).9f,(_Float16).03f,(_Float16).63f,(_Float16)1};memcpy(pixel,value,8);}
        else if(testSdr){uint32_t value=0xff000000u|230u|(8u<<8)|(161u<<16);memcpy(pixel,&value,4);}
        else{uint32_t value=0xc0000000u|921u|(31u<<10)|(644u<<20);memcpy(pixel,&value,4);}
    }
    auto separate=texture(device,w,h,colorFormat);
    [separate replaceRegion:MTLRegionMake2D(0,0,w,h) mipmapLevel:0 withBytes:overlay.data() bytesPerRow:w*bpp];
    base.uiTexture=separate;base.uiTextureComposited=NO;encode();fsr4_anr_get_status(&s);
    need(s.generated==8&&s.active&&s.uiOverlay==1&&s.uiMode==FSR4_ANR_UI_OVERLAY,"premultiplied UI did not replace MetalFX");
    readOutput();
    for(unsigned y=0;y<h;++y)for(unsigned x=0;x<w;++x)if(inHud(x,y)){
        const size_t offset=(size_t(y)*w+x)*bpp;
        if(bpp==8)need(!memcmp((const uint8_t*)readback.contents+offset,overlay.data()+offset,6),"overlay HDR HUD pixels changed");
        else{uint32_t a,b;memcpy(&a,(const uint8_t*)readback.contents+offset,4);memcpy(&b,overlay.data()+offset,4);
            need((a&rgbMask)==(b&rgbMask),"overlay HUD pixels changed");}
    }
    // Queue changing UI modes. Constants, arguments and retained UI resources
    // belong to their frame slot, not mutable state of the next CPU encode.
    auto configureUI=[&](unsigned i){base.uiTexture=i==1?separate:ui;base.uiTextureComposited=i!=1;};
    if(metal4){
        id<MTL4CommandBuffer> pending[3]={};id<MTL4CommandAllocator> allocators[3]={};
        for(unsigned i=0;i<3;++i){configureUI(i);pending[i]=[device newCommandBuffer];allocators[i]=[device newCommandAllocator];
            [pending[i] beginCommandBufferWithAllocator:allocators[i]];[(id<MTL4FXFrameInterpolator>)fg encodeToCommandBuffer:pending[i]];[pending[i] endCommandBuffer];}
        [q4 commit:pending count:3];[q4 signalEvent:event value:++token];need([event waitUntilSignaledValue:token timeoutMS:20000],"queued UI timeout");
    }else{
        id<MTLCommandBuffer> pending[3]={};for(unsigned i=0;i<3;++i){configureUI(i);pending[i]=[q commandBuffer];[(id<MTLFXFrameInterpolator>)fg encodeToCommandBuffer:pending[i]];[pending[i] commit];}
        [pending[2] waitUntilCompleted];for(auto cb:pending)need(cb.status==MTLCommandBufferStatusCompleted,"queued UI GPU failure");
    }
    fsr4_anr_get_status(&s);need(s.generated==11&&s.uiComposited==3&&s.uiOverlay==2,"queued UI lost a frame/mode");
    need(s.uiSamples>0&&s.uiOpaque<=s.uiChanged&&s.uiOpacitySum<=4096u*s.uiSamples,"completed UI statistics invalid");
    readOutput();
    for(unsigned y=0;y<h;++y)for(unsigned x=0;x<w;++x)if(inHud(x,y)){
        const size_t offset=(size_t(y)*w+x)*bpp;
        if(bpp==8)need(!memcmp((const uint8_t*)readback.contents+offset,composite.data()+offset,6),"queued HDR HUD pixels changed");
        else{uint32_t a,b;memcpy(&a,(const uint8_t*)readback.contents+offset,4);memcpy(&b,composite.data()+offset,4);
            need((a&rgbMask)==(b&rgbMask),"queued HUD pixels changed");}
    }
    // End-to-end 3x live switch & CAMetalLayer presentation insertion on V.B20!
    uint64_t extra3xCount=0;
    if(isV2){
        need(fsr4_fg::write_settings(preference,fsr4_fg::Mode::ANR3x),"ANR 3x preference write failed");
        std::this_thread::sleep_for(std::chrono::milliseconds(650));
        need(fsr4_anr_selected_mode()==2&&fsr4_anr_multiplier()==3,"background controller did not apply ANR 3x mode");
        base.uiTexture=nil;base.uiTextureComposited=NO;
        // The 650 ms timer sleep exceeds the 500 ms temporal-gap threshold, so the
        // first frame after pausing safely primes history before 3x generation begins.
        encode();
        CAMetalLayer *layer=[CAMetalLayer layer];
        layer.device=device;layer.pixelFormat=MTLPixelFormatBGRA8Unorm;
        layer.drawableSize=CGSizeMake(w,h);layer.maximumDrawableCount=3;
        layer.framebufferOnly=YES;
        for(unsigned step=0;step<2;++step){
            encode();
            fsr4_anr_get_status(&s);
            need(s.active&&s.multiplier==3,"ANR 3x did not activate on V.B20");
            id<CAMetalDrawable> d1=[layer nextDrawable];
            need(d1!=nil,"3x test layer nextDrawable returned nil");
            if(metal4){
                [q4 signalDrawable:d1];
                [d1 present];
                // Present OutputReal to complete the 3-frame cycle, then wait for GPU queue drain.
                id<CAMetalDrawable> dReal=[layer nextDrawable];
                if(dReal){[q4 signalDrawable:dReal];[dReal present];}
                [q4 signalEvent:event value:++token];
                need([event waitUntilSignaledValue:token timeoutMS:20000],"3x Metal4 present drain timeout");
            }else{
                auto cbP=[q commandBuffer];[cbP presentDrawable:d1];[cbP commit];[cbP waitUntilCompleted];
                auto cbR=[q commandBuffer];
                id<CAMetalDrawable> dReal=[layer nextDrawable];
                if(dReal)[cbR presentDrawable:dReal];
                [cbR commit];[cbR waitUntilCompleted];
            }
        }
        // One more encode triggers retirement check of the previous Metal 4 present slots.
        encode();
        fsr4_anr_get_status(&s);
        // Trigger retirement on Metal 4 via another signal or direct check.
        if(metal4){
            id<CAMetalDrawable> d1=[layer nextDrawable];
            if(d1){[q4 signalDrawable:d1];[d1 present];}
            [q4 signalEvent:event value:++token];
            need([event waitUntilSignaledValue:token timeoutMS:20000],"3x Metal4 final drain timeout");
        }
        fsr4_anr_get_status(&s);
        extra3xCount=s.generatedExtra3x;
        need(extra3xCount>=1,"ANR 3x presentation hook did not present extra t=2/3 frame");
        need(fsr4_fg::write_settings(preference,fsr4_fg::Mode::ANR),"restore ANR 2x failed");
        std::this_thread::sleep_for(std::chrono::milliseconds(650));
        need(fsr4_anr_selected_mode()==1&&fsr4_anr_multiplier()==2,"switch back from 3x to 2x failed");
    }
    // Invalid UI inputs must fail before any GPU encode. Call the bridge
    // directly so the original interpolator never sees these invalid inputs.
    Fsr4AnrFrame check={};check.structSize=sizeof(check);check.abiVersion=FSR4_ANR_FG_ABI;
    auto validationCommand=[q commandBuffer];check.commandBuffer=(__bridge void*)validationCommand;
    check.device=(__bridge void*)device;check.owner=(__bridge void*)fg;check.color=(__bridge void*)c;
    check.depth=(__bridge void*)z;check.motion=(__bridge void*)mv;check.output=(__bridge void*)out;
    check.rw=check.uw=w;check.rh=check.uh=h;check.nearPlane=.1f;check.farPlane=1000;
    check.motionScaleX=check.motionScaleY=1;check.colorEncoding=testHdr?0:testSdr?1:2;check.uiComposited=1;
    auto shortUI=texture(device,w/2,h,colorFormat);check.ui=(__bridge void*)shortUI;
    need(fsr4_anr_encode_frame(&check)==0,"UI extent mismatch was accepted");
    fsr4_anr_get_status(&s);need(strstr(s.reason,"UI texture extent/device mismatch"),"UI extent diagnostic missing");
    check.ui=(__bridge void*)out;need(fsr4_anr_encode_frame(&check)==0,"read/write UI alias was accepted");
    fsr4_anr_get_status(&s);need(strstr(s.reason,"UI aliases FG output"),"UI alias diagnostic missing");
    auto wrongUI=texture(device,w,h,testSdr?MTLPixelFormatRGB10A2Unorm:MTLPixelFormatBGRA8Unorm);check.ui=(__bridge void*)wrongUI;
    need(fsr4_anr_encode_frame(&check)==0,"UI color format mismatch was accepted");
    fsr4_anr_get_status(&s);need(strstr(s.reason,"UI and scene color formats differ"),"UI format diagnostic missing");
    fprintf(stdout,"ANR live MetalFX %s hook passed (%ux%u, generated=%llu primed=%llu extra3x=%llu, %u exact HUD pixels, 3 changing UI modes in flight, unsafe UI rejected)\n",
            metal4?"Metal4":"Metal3",w,h,(unsigned long long)s.generated,(unsigned long long)s.primed,(unsigned long long)extra3xCount,hudPixels);
    unlink(preference.c_str());rmdir(preferenceDirectory);
    return 0;
}catch(const std::exception &e){fprintf(stderr,"ANR live test: %s\n",e.what());return 1;}}}
