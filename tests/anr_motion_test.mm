// SPDX-License-Identifier: MIT
// End-to-end moving-frame regression. No game/model data is recorded here.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalFX/MetalFX.h>
#include "anr_live.h"
#include "fsr4_fg_settings.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>
static void need(bool v,const char *why){if(!v)throw std::runtime_error(why);}
static id<MTLTexture> texture(id<MTLDevice> device,unsigned w,unsigned h,MTLPixelFormat f,bool privateMode=false){
    auto d=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:f width:w height:h mipmapped:NO];
    d.usage=MTLTextureUsageShaderRead|MTLTextureUsageShaderWrite|MTLTextureUsageRenderTarget;
    d.storageMode=privateMode?MTLStorageModePrivate:MTLStorageModeShared;
    auto t=[device newTextureWithDescriptor:d];need(t!=nil,"texture allocation failed");return t;
}
static std::array<float,3> pattern(float x,float y){
    return {.48f+.12f*sinf(.11f*x+.017f*y)+.06f*cosf(.027f*x-.041f*y),
            .51f+.14f*sinf(.039f*x-.079f*y)+.04f*cosf(.093f*x+.011f*y),
            .43f+.12f*cosf(.067f*x+.049f*y)+.05f*sinf(.121f*x-.037f*y)};
}
static uint32_t pixel(float x,float y){auto p=pattern(x,y);uint32_t v=0xc0000000u;for(unsigned c=0;c<3;++c)v|=(uint32_t)lroundf(p[c]*1023)<<(c*10);return v;}
int main(int argc,char **argv){@autoreleasepool{try{
    need(argc>=3,"usage: anr_motion_test <root> <runtime> [classic|metal4]");
    setenv("FSR4_ROOT",argv[1],1);setenv("FSR4_ANR_RUNTIME",argv[2],1);
    setenv("FSR4_ANR_COLOR","pq10",1);setenv("FSR4_ANR_STATUS_DISABLED","1",1);setenv("FSR4_FG_SETTINGS","none",1);
    const bool m4=argc>3&&!strcmp(argv[3],"metal4");
    auto device=MTLCreateSystemDefaultDevice();need(device!=nil,"no Metal device");
    auto metadata=[NSData dataWithContentsOfFile:[[NSString stringWithUTF8String:argv[2]] stringByAppendingPathComponent:@"manifest.json"]];
    NSDictionary *manifest=[NSJSONSerialization JSONObjectWithData:metadata options:0 error:nil];
    unsigned w=[manifest[@"width"] unsignedIntValue],h=[manifest[@"height"] unsignedIntValue];need(w&&h,"bad runtime dimensions");
    auto q=[device newCommandQueue];auto q4=[device newMTL4CommandQueue];auto allocator=[device newCommandAllocator];
    auto event=[device newSharedEvent];uint64_t token=0;NSError *error=nil;
    auto compiler=[device newCompilerWithDescriptor:[MTL4CompilerDescriptor new] error:&error];
    for(bool low:{false,true})for(const char *mode:{"metalfx","anr"})for(float dx:{0.f,24.f}){
        const unsigned rw=low?w*2/3:w,rh=low?h*2/3:h;
        setenv("FSR4_FG",mode,1);
        auto d=[MTLFXFrameInterpolatorDescriptor new];d.inputWidth=rw;d.inputHeight=rh;d.outputWidth=w;d.outputHeight=h;
        d.colorTextureFormat=d.outputTextureFormat=d.uiTextureFormat=MTLPixelFormatRGB10A2Unorm;
        d.depthTextureFormat=MTLPixelFormatR32Float;d.motionTextureFormat=MTLPixelFormatRG16Float;
        id fg=nil;
        if(m4)fg=[d newFrameInterpolatorWithDevice:device compiler:compiler];
        else fg=[d newFrameInterpolatorWithDevice:device];
        need(fg!=nil,"no MetalFX interpolator");id<MTLFXFrameInterpolatorBase> base=fg;
        auto c=texture(device,w,h,MTLPixelFormatRGB10A2Unorm),prev=texture(device,w,h,MTLPixelFormatRGB10A2Unorm);
        auto z=texture(device,rw,rh,MTLPixelFormatR32Float),mv=texture(device,rw,rh,MTLPixelFormatRG16Float),out=texture(device,w,h,MTLPixelFormatRGB10A2Unorm,true);
        std::vector<float> depth(size_t(rw)*rh,.5f);std::vector<_Float16> motion(size_t(rw)*rh*2,0);
        for(size_t i=0;i<size_t(rw)*rh;++i)motion[i*2]=(_Float16)(-dx*rw/w);
        [z replaceRegion:MTLRegionMake2D(0,0,rw,rh) mipmapLevel:0 withBytes:depth.data() bytesPerRow:rw*4];
        [mv replaceRegion:MTLRegionMake2D(0,0,rw,rh) mipmapLevel:0 withBytes:motion.data() bytesPerRow:rw*4];
        base.colorTexture=c;base.prevColorTexture=prev;base.depthTexture=z;base.motionTexture=mv;base.outputTexture=out;
        base.motionVectorScaleX=base.motionVectorScaleY=1;base.nearPlane=.1f;base.farPlane=1000;
        base.fieldOfView=60;base.aspectRatio=float(w)/h;base.depthReversed=NO;base.deltaTime=1.f/60;
        auto readback=[device newBufferWithLength:size_t(w)*h*4 options:MTLResourceStorageModeShared];
        std::vector<uint32_t> source(size_t(w)*h),previous(source.size());
        const float jitters[][2]={{.125f,.2777778f},{-.125f,-.2777778f},{.375f,.0555556f},{-.4375f,.388889f},{.0625f,-.462963f},{-.1875f,-.12963f}};
        for(unsigned n=0;n<6;++n){
            for(unsigned y=0;y<h;++y)for(unsigned x=0;x<w;++x){source[size_t(y)*w+x]=pixel(x-dx*n,y);previous[size_t(y)*w+x]=pixel(x-dx*(n?n-1:0),y);}
            [c replaceRegion:MTLRegionMake2D(0,0,w,h) mipmapLevel:0 withBytes:source.data() bytesPerRow:w*4];
            [prev replaceRegion:MTLRegionMake2D(0,0,w,h) mipmapLevel:0 withBytes:previous.data() bytesPerRow:w*4];
            base.shouldResetHistory=n==0;base.jitterOffsetX=jitters[n][0];base.jitterOffsetY=jitters[n][1];
            if(m4){
                auto cb=[device newCommandBuffer];[allocator reset];[cb beginCommandBufferWithAllocator:allocator];
                // MetalFX does not promise to make borrowed inputs resident.
                // The ANR bridge does; the standalone original path must too.
                auto rd=[MTLResidencySetDescriptor new];rd.initialCapacity=6;auto residency=[device newResidencySetWithDescriptor:rd error:&error];
                for(id r in @[c,prev,z,mv,out,readback])[residency addAllocation:r];[residency commit];[cb useResidencySet:residency];
                [(id<MTL4FXFrameInterpolator>)fg encodeToCommandBuffer:cb];
                auto enc=[cb computeCommandEncoder];[enc barrierAfterQueueStages:MTLStageAll beforeStages:MTLStageBlit visibilityOptions:MTL4VisibilityOptionDevice];
                [enc copyFromTexture:out sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0,0,0) sourceSize:MTLSizeMake(w,h,1) toBuffer:readback destinationOffset:0 destinationBytesPerRow:w*4 destinationBytesPerImage:size_t(w)*h*4];
                [enc endEncoding];[cb endCommandBuffer];id<MTL4CommandBuffer> buffers[]={cb};[q4 commit:buffers count:1];[q4 signalEvent:event value:++token];
                need([event waitUntilSignaledValue:token timeoutMS:20000],"Metal4 timeout");
            }else{
                auto cb=[q commandBuffer];[(id<MTLFXFrameInterpolator>)fg encodeToCommandBuffer:cb];auto enc=[cb blitCommandEncoder];
                [enc copyFromTexture:out sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0,0,0) sourceSize:MTLSizeMake(w,h,1) toBuffer:readback destinationOffset:0 destinationBytesPerRow:w*4 destinationBytesPerImage:size_t(w)*h*4];
                [enc endEncoding];[cb commit];[cb waitUntilCompleted];need(cb.status==MTLCommandBufferStatusCompleted,"Metal3 failure");
            }
            if(!n)continue;
            const auto *actual=(const uint32_t*)readback.contents;
            const float expected=dx*(n-.5f);double best=1.e99;float bestPosition=0;
            for(float candidate=expected-dx;candidate<=expected+dx+.001f;candidate+=dx?dx/8:1){
                double squared=0;unsigned count=0;
                for(unsigned y=64;y<h-64;y+=13)for(unsigned x=64;x<w-64;x+=13){
                    uint32_t b=pixel(x-candidate,y),a=actual[size_t(y)*w+x];
                    for(unsigned c=0;c<3;++c){double difference=int((a>>(10*c))&1023)-int((b>>(10*c))&1023);squared+=difference*difference;++count;}}
                double rmse=sqrt(squared/count);if(rmse<best){best=rmse;bestPosition=candidate;}
            }
            printf("%s %s render=%ux%u dx=%.0f frame=%u expected=%.2f best=%.2f rmse=%.3f\n",m4?"Metal4":"Metal3",mode,rw,rh,dx,n,expected,bestPosition,best);
            if(!strcmp(mode,"anr")){
                Fsr4AnrStatus s={};fsr4_anr_get_status(&s);need(s.active,"ANR fell back in moving-frame test");
                need(fabsf(bestPosition-expected)<.25f,"ANR temporal position is not the midpoint");
                need(best<2.0,"ANR synthetic moving-frame quality regressed");
            }
        }
        base=nil;fg=nil;
    }
    return 0;
}catch(const std::exception &e){fprintf(stderr,"ANR motion test: %s\n",e.what());return 1;}}}
