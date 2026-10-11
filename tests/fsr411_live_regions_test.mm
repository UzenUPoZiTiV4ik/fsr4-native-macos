// SPDX-License-Identifier: MIT
// Compare the V2 live bridge's cropped/padded resources against exact-size
// inputs. The generated content is independent of any Windows reference.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "fsr4_live.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace {
constexpr uint32_t rw=37, rh=23, uw=65, uh=41;
constexpr uint16_t sentinel=0xc880; // half(-9)

[[noreturn]] void fail(const char *message) {throw std::runtime_error(message);}
uint16_t half(float value) {_Float16 h=(_Float16)value; uint16_t bits; std::memcpy(&bits,&h,2); return bits;}
float single(uint16_t bits) {_Float16 h; std::memcpy(&h,&bits,2); return float(h);}

id<MTLTexture> texture(id<MTLDevice> device, Fsr4InputSurface surface, MTLPixelFormat format) {
    auto d=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format
        width:surface.width height:surface.height mipmapped:NO];
    d.storageMode=MTLStorageModeShared; d.usage=MTLTextureUsageShaderRead|MTLTextureUsageShaderWrite;
    auto result=[device newTextureWithDescriptor:d];
    if(!result) fail("texture allocation failed"); return result;
}

struct Inputs {
    Fsr4LiveFrameV2 descriptor={};
    id<MTLTexture> color,depth,motion,output,exposure;
};

Inputs inputs(id<MTLDevice> device, bool padded, bool outputMotion, void *owner) {
    Inputs result;
    auto &v=result.descriptor; auto &f=v.frame; auto &g=v.geometry;
    v.structSize=sizeof v; v.abiVersion=FSR4_LIVE_FRAME_ABI_V2; v.inputContentProperties=1;
    const uint32_t mw=outputMotion?uw:rw, mh=outputMotion?uh:rh;
    if(padded) {
        g.color={rw+3+5,rh+2+7,3,2}; g.depth={rw+5+4,rh+1+8,5,1};
        g.motion={mw+7+6,mh+4+3,7,4}; g.output={uw+9+5,uh+6+3,9,6};
    } else {
        g.color={rw,rh,0,0}; g.depth=g.color; g.motion={mw,mh,0,0}; g.output={uw,uh,0,0};
    }
    g.maxInputWidth=rw; g.maxInputHeight=rh;
    result.color=texture(device,g.color,MTLPixelFormatRGBA16Float);
    result.depth=texture(device,g.depth,MTLPixelFormatR32Float);
    result.motion=texture(device,g.motion,MTLPixelFormatRG16Float);
    result.output=texture(device,g.output,MTLPixelFormatRGBA16Float);
    result.exposure=texture(device,{1,1,0,0},MTLPixelFormatR32Float);
    const float exposure=.625f;
    [result.exposure replaceRegion:MTLRegionMake2D(0,0,1,1) mipmapLevel:0 withBytes:&exposure bytesPerRow:4];
    f.device=(__bridge void *)device; f.owner=owner;
    f.color=(__bridge void *)result.color; f.depth=(__bridge void *)result.depth;
    f.motion=(__bridge void *)result.motion; f.output=(__bridge void *)result.output;
    f.exposure=(__bridge void *)result.exposure;
    f.rw=rw; f.rh=rh; f.uw=uw; f.uh=uh;
    f.mvScaleX=1.5f; f.mvScaleY=-2.f; f.preExposure=1.25f;
    f.outputResMV=outputMotion; f.jitteredMV=outputMotion; f.depthReversed=outputMotion;
    f.autoExposure=0; f.mvOffX=g.motion.offsetX; f.mvOffY=g.motion.offsetY;
    return result;
}

void upload(Inputs &in, unsigned frame) {
    const auto &g=in.descriptor.geometry;
    std::vector<uint16_t> color(size_t(g.color.width)*g.color.height*4,half(7.f));
    std::vector<float> depth(size_t(g.depth.width)*g.depth.height,17.f);
    for(uint32_t y=0;y<rh;++y) for(uint32_t x=0;x<rw;++x) {
        size_t i=(size_t(y+g.color.offsetY)*g.color.width+x+g.color.offsetX)*4;
        color[i]=half(float(16+((x+frame)%29)*5)/256.f);
        color[i+1]=half(float(24+((y+frame)%19)*7)/256.f);
        color[i+2]=half(float(32+((x+2*y+frame)%31)*4)/256.f); color[i+3]=half(1);
        depth[size_t(y+g.depth.offsetY)*g.depth.width+x+g.depth.offsetX]=.15f+float((x+y)%17)/24.f;
    }
    const uint32_t mw=in.descriptor.frame.outputResMV?uw:rw, mh=in.descriptor.frame.outputResMV?uh:rh;
    std::vector<uint16_t> motion(size_t(g.motion.width)*g.motion.height*2,half(11.f));
    for(uint32_t y=0;y<mh;++y) for(uint32_t x=0;x<mw;++x) {
        size_t i=(size_t(y+g.motion.offsetY)*g.motion.width+x+g.motion.offsetX)*2;
        motion[i]=half(-.25f+float((x+frame)%3)/32.f);
        motion[i+1]=half(.1875f+float((y+frame)%5)/64.f);
    }
    std::vector<uint16_t> out(size_t(g.output.width)*g.output.height*4,sentinel);
    [in.color replaceRegion:MTLRegionMake2D(0,0,g.color.width,g.color.height) mipmapLevel:0
        withBytes:color.data() bytesPerRow:g.color.width*8];
    [in.depth replaceRegion:MTLRegionMake2D(0,0,g.depth.width,g.depth.height) mipmapLevel:0
        withBytes:depth.data() bytesPerRow:g.depth.width*4];
    [in.motion replaceRegion:MTLRegionMake2D(0,0,g.motion.width,g.motion.height) mipmapLevel:0
        withBytes:motion.data() bytesPerRow:g.motion.width*4];
    [in.output replaceRegion:MTLRegionMake2D(0,0,g.output.width,g.output.height) mipmapLevel:0
        withBytes:out.data() bytesPerRow:g.output.width*8];
    in.descriptor.frame.jitterX=float(int(frame%4)-2)/8.f;
    in.descriptor.frame.jitterY=float(int((frame*3)%7)-3)/12.f;
    in.descriptor.frame.reset=frame==0;
}

void encode(id<MTLDevice> device, id<MTLCommandQueue> queue, Inputs &in, bool metal4, bool checkProfile) {
    if(metal4) {
        if(@available(macOS 26.0,*)) {
            auto q=[device newMTL4CommandQueue]; auto allocator=[device newCommandAllocator];
            auto cb=[device newCommandBuffer];
            if(!q || !allocator || !cb) fail("Metal 4 queue allocation failed");
            [cb beginCommandBufferWithAllocator:allocator];
            in.descriptor.frame.commandBuffer=(__bridge void *)cb;
            if(!fsr4_encode_live_frame_v2(&in.descriptor)) fail("Metal 4 V2 encode was rejected");
            [cb endCommandBuffer]; id<MTL4CommandBuffer> list[]={cb}; [q commit:list count:1];
            auto event=[device newSharedEvent]; [q signalEvent:event value:1];
            if(!event || ![event waitUntilSignaledValue:1 timeoutMS:20000]) fail("Metal 4 frame timed out");
        } else fail("Metal 4 requires macOS 26");
    } else {
        auto cb=[queue commandBuffer]; in.descriptor.frame.commandBuffer=(__bridge void *)cb;
        if(!fsr4_encode_live_frame_v2(&in.descriptor)) fail("classic V2 encode was rejected");
        [cb commit]; [cb waitUntilCompleted];
        if(cb.status!=MTLCommandBufferStatusCompleted) fail("GPU execution failed");
    }
    Fsr4LiveStatus status={}; fsr4_live_get_status(&status);
    if(!status.active || std::strncmp(status.variant,"fsr411_",7)) fail("frame did not run the FSR 4.1.1 live bridge");
    if(checkProfile) {
        int requested=0;fsr4_live_get_profile(&requested);
        const bool lossy=std::strstr(status.variant,"_lossy")!=nullptr;
        if(lossy!=(requested==1)) fail("frame did not run the selected profile");
    }
}

void compare(const Inputs &canonical, const Inputs &padded) {
    const auto &g=padded.descriptor.geometry.output;
    std::vector<uint16_t> a(size_t(uw)*uh*4),b(size_t(g.width)*g.height*4);
    [canonical.output getBytes:a.data() bytesPerRow:uw*8 fromRegion:MTLRegionMake2D(0,0,uw,uh) mipmapLevel:0];
    [padded.output getBytes:b.data() bytesPerRow:g.width*8 fromRegion:MTLRegionMake2D(0,0,g.width,g.height) mipmapLevel:0];
    size_t differences=0,nonzero=0; float maxError=0;
    for(uint32_t y=0;y<g.height;++y) for(uint32_t x=0;x<g.width;++x) for(uint32_t channel=0;channel<4;++channel) {
        const auto actual=b[(size_t(y)*g.width+x)*4+channel];
        if(x<g.offsetX || y<g.offsetY || x>=g.offsetX+uw || y>=g.offsetY+uh) {
            if(actual!=sentinel) fail("FSR overwrote pixels outside the output content rectangle");
        } else {
            const auto expected=a[(size_t(y-g.offsetY)*uw+x-g.offsetX)*4+channel];
            if(!std::isfinite(single(actual)) || !std::isfinite(single(expected))) fail("nonfinite FSR output");
            differences+=actual!=expected; nonzero+=channel<3 && expected!=0;
            maxError=std::max(maxError,std::fabs(single(actual)-single(expected)));
        }
    }
    if(!nonzero) fail("FSR output has no nonzero colour samples");
    if(differences) {
        std::fprintf(stderr,"cropped output: %zu different half values, max abs %.9f\n",differences,maxError);
        fail("cropped and canonical outputs differ");
    }
}
} // namespace

int main(int argc,char **argv) {
    if(argc<2 || argc>5 || (argc==5 && std::strcmp(argv[4],"switch"))) {
        std::fprintf(stderr,"usage: fsr411_live_regions_test <repo-root> [reference|scalar|i8mpp] [classic|mtl4] [switch]\n"); return 2;
    }
    setenv("FSR4_ROOT",argv[1],1);
    unsetenv("FSR4_VERSION"); // Exercise the default 4.1.1 live router.
    setenv("FSR4_PROFILE","exact",1);
    setenv("FSR4_MODEL",argc>2?argv[2]:"reference",1);
    setenv("FSR4_SHARPNESS","0",1); setenv("FSR4_SETTINGS","none",1);
    const bool metal4=argc>3 && !std::strcmp(argv[3],"mtl4");
    const bool switchProfiles=argc>4;
    const unsigned frameCount=switchProfiles?10:6;
    @autoreleasepool {try {
        auto device=MTLCreateSystemDefaultDevice(); if(!device) fail("no Metal device");
        auto queue=[device newCommandQueue]; if(!queue) fail("no Metal command queue");
        int owners[3]={};
        for(bool outputMotion:{false,true}) {
            fsr4_live_set_profile(0);
            auto canonical=inputs(device,false,outputMotion,&owners[0]);
            auto padded=inputs(device,true,outputMotion,&owners[1]);
            for(unsigned frame=0;frame<frameCount;++frame) {
                if(frame==3) {
                    fsr4_invalidate_live_owner((__bridge void *)device,canonical.descriptor.frame.owner);
                    fsr4_release_live_owner((__bridge void *)device,padded.descriptor.frame.owner);
                }
                if(frame==4) {
                    auto bad=canonical.descriptor; bad.geometry.color.width++;
                    auto cb=[queue commandBuffer]; bad.frame.commandBuffer=(__bridge void *)cb;
                    if(fsr4_encode_live_frame_v2(&bad)) fail("mismatched V2 geometry was accepted");
                    fsr4_release_live_owner((__bridge void *)device,padded.descriptor.frame.owner);
                }
                if(frame==5) {
                    fsr4_release_live_owner((__bridge void *)device,canonical.descriptor.frame.owner);
                    fsr4_release_live_owner((__bridge void *)device,padded.descriptor.frame.owner);
                    padded.descriptor.frame.owner=&owners[2];
                }
                if(frame==6)fsr4_live_set_profile(1);
                if(frame==8)fsr4_live_set_profile(0);
                upload(canonical,frame); upload(padded,frame);
                if(frame==6 || frame==7) {
                    canonical.descriptor.frame.jitterX=-.375f+float(frame-6)*.125f;
                    canonical.descriptor.frame.jitterY=-.25f-float(frame-6)/12.f;
                    padded.descriptor.frame.jitterX=canonical.descriptor.frame.jitterX;
                    padded.descriptor.frame.jitterY=canonical.descriptor.frame.jitterY;
                }
                // A profile change must act like an explicit reset. Retain
                // both streams' previous jitter: recreating only one owner
                // would change jitter cancellation even on a reset frame.
                if(frame==6 || frame==8)padded.descriptor.frame.reset=1;
                encode(device,queue,canonical,metal4,switchProfiles);
                encode(device,queue,padded,metal4,switchProfiles);
                compare(canonical,padded);
            }
            fsr4_release_live_owner((__bridge void *)device,canonical.descriptor.frame.owner);
            fsr4_release_live_owner((__bridge void *)device,padded.descriptor.frame.owner);
            std::printf("V2 regions: %s motion, %s, %u frames byte-identical; sentinel and lifecycle passed%s\n",
                outputMotion?"output-resolution":"render-resolution",metal4?"Metal 4":"classic Metal",frameCount,
                switchProfiles?"; Exact/Lossy switches matched explicit resets":"");
        }
        return 0;
    } catch(const std::exception &error) {
        Fsr4LiveStatus status={}; fsr4_live_get_status(&status);
        std::fprintf(stderr,"live V2 test: %s (%s)\n",error.what(),status.reason); return 1;
    }}
}
