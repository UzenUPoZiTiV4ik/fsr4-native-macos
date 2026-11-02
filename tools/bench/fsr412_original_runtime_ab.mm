// SPDX-License-Identifier: MIT
// Production Runtime::encode (int8-lite-lossy-original) versus the standalone
// original-shader oracle: same inputs, every frame compares output, history,
// recurrent, reprojected and the full logical scratch. No DLL/game execution.
#include "fsr412_upstream_replay.h"
#include "../../src/fsr412_runtime.h"

using namespace fsr412_original;
namespace {
std::vector<uint8_t> blit(id<MTLCommandQueue> queue,id<MTLTexture> t,unsigned bpp) {
    const NSUInteger pitch=(t.width*bpp+255u)&~NSUInteger(255);
    auto buffer=[t.device newBufferWithLength:pitch*t.height options:MTLResourceStorageModeShared];
    auto cb=[queue commandBuffer];auto enc=[cb blitCommandEncoder];
    [enc copyFromTexture:t sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0,0,0) sourceSize:MTLSizeMake(t.width,t.height,1)
        toBuffer:buffer destinationOffset:0 destinationBytesPerRow:pitch destinationBytesPerImage:pitch*t.height];
    [enc endEncoding];[cb commit];[cb waitUntilCompleted];check(cb.status==MTLCommandBufferStatusCompleted,"readback failed");
    std::vector<uint8_t> out(t.width*t.height*bpp);
    for(NSUInteger y=0;y<t.height;++y)std::memcpy(out.data()+y*t.width*bpp,(const uint8_t *)buffer.contents+y*pitch,t.width*bpp);
    return out;
}
std::vector<uint8_t> shared(id<MTLTexture> t,unsigned bpp) {
    std::vector<uint8_t> out(t.width*t.height*bpp);
    [t getBytes:out.data() bytesPerRow:t.width*bpp fromRegion:MTLRegionMake2D(0,0,t.width,t.height) mipmapLevel:0];return out;
}
void same(const std::vector<uint8_t> &a,const std::vector<uint8_t> &b,const std::string &what,unsigned frame) {
    if(a==b) return;
    size_t i=0;while(i<a.size() && i<b.size() && a[i]==b[i])++i;
    throw std::runtime_error(what+" differs at frame "+std::to_string(frame)+", byte "+std::to_string(i));
}
void fill(Replay &r,unsigned frame) {
    std::vector<float> c(r.renderWidth*r.renderHeight*4),d(r.renderWidth*r.renderHeight),m(r.renderWidth*r.renderHeight*2);
    for(unsigned y=0;y<r.renderHeight;++y)for(unsigned x=0;x<r.renderWidth;++x){size_t i=size_t(y)*r.renderWidth+x;
        const float a=.05f+.5f*float(x)/r.renderWidth+.1f*float(y)/r.renderHeight;
        c[i*4]=a;c[i*4+1]=a*.7f;c[i*4+2]=a*.4f;c[i*4+3]=1;
        if((x+frame)%29==0)c[i*4]=c[i*4+1]=c[i*4+2]=.9f;
        const bool foreground=x>r.renderWidth/2;d[i]=foreground?.8f:.2f;m[i*2]=foreground?1.f/r.width:0;m[i*2+1]=0;
    }
    r.upload(r.textures["input"],c.data());r.upload(r.textures["depth"],d.data());r.upload(r.textures["velocity"],m.data());
}
}
int main(int argc,char **argv) {@autoreleasepool {try {
    check(argc>=2,"usage: fsr412_original_runtime_ab ROOT [--simd|--scalar] [--frames N] [--size RW RH UW UH]");
    std::string root=argv[1];unsigned rw=384,rh=224,uw=640,uh=384,frames=8;auto backend=fsr411::Backend::Native;bool metal4=false;
    for(int i=2;i<argc;++i){std::string a=argv[i];
        if(a=="--simd")backend=fsr411::Backend::Simd;else if(a=="--scalar")backend=fsr411::Backend::Scalar;
        else if(a=="--mtl4")metal4=true;
        else if(a=="--frames" && i+1<argc)frames=std::stoul(argv[++i]);
        else if(a=="--size" && i+4<argc){rw=std::stoul(argv[++i]);rh=std::stoul(argv[++i]);uw=std::stoul(argv[++i]);uh=std::stoul(argv[++i]);}
        else throw std::runtime_error("unknown option "+a);}
    // Pinned six: tier 2160, normal, di1 lm1 jm0 cs0, AMD auto-exposure enabled.
    Replay oracle(root+"/build/fsr412_upstream",uw,uh,rw,rh,"isolated","u8");
    oracle.load_pass1_border();
    fsr411::Runtime runtime(oracle.device,root,root+"/build");fsr411::View view;
    fsr411::Options o;o.modelLite=true;o.profile=fsr411::Profile::Int8LiteLossy;o.originalLossy=true;o.tier=2160;o.backend=backend;
    auto output=oracle.texture(MTLPixelFormatRGBA16Float,uw,uh);unsigned skipped=0;
    for(unsigned f=0;f<frames;++f) { @autoreleasepool {
        fill(oracle,f);
        const float jx=f&1u?-.25f:.125f,jy=f%3?1.f/6.f:-1.f/6.f;
        fsr411::Frame frame;frame.color=oracle.textures["input"];frame.depth=oracle.textures["depth"];frame.motion=oracle.textures["velocity"];frame.output=output;
        auto &p=frame.input;p.rw=rw;p.rh=rh;p.uw=uw;p.uh=uh;p.preExposure=1;p.reset=f==0;p.depthReversed=1;
        p.geometry.color=p.geometry.depth=p.geometry.motion={rw,rh,0,0};p.geometry.output={uw,uh,0,0};
        p.mvScaleX=float(rw);p.mvScaleY=float(rh);p.jitterX=jx;p.jitterY=jy;
        p.jitteredMV=0;p.outputResMV=0;p.autoExposure=1;p.inputContentProperties=0;p.hasExposureTexture=0;
        std::string error;
        if(metal4) {
            if(@available(macOS 26.0,*)) {
                static id<MTL4CommandQueue> queue4=[oracle.device newMTL4CommandQueue];static id<MTLSharedEvent> event=[oracle.device newSharedEvent];static uint64_t value=0;
                check(queue4 && event,"Metal 4 unavailable");
                auto allocator=[oracle.device newCommandAllocator];auto cb=[oracle.device newCommandBuffer];[cb beginCommandBufferWithAllocator:allocator];
                check(runtime.encode(cb,view,frame,o,error),"runtime: "+error);[cb endCommandBuffer];
                id<MTL4CommandBuffer> list[]={cb};[queue4 commit:list count:1];[queue4 signalEvent:event value:++value];
                check([event waitUntilSignaledValue:value timeoutMS:20000],"Metal 4 completion timeout");
            } else throw std::runtime_error("Metal 4 requires macOS 26");
        } else {
            auto cb=[oracle.queue commandBuffer];
            check(runtime.encode(cb,view,frame,o,error),"runtime: "+error);
            [cb commit];[cb waitUntilCompleted];check(cb.status==MTLCommandBufferStatusCompleted,"runtime GPU failure");
        }
        o.allowCompile=false;
        // Same exposure input for the oracle's PRE/POST: the runtime's AE result.
        {auto copy=[oracle.queue commandBuffer];auto blitter=[copy blitCommandEncoder];
         [blitter copyFromTexture:view.autoExposure toTexture:oracle.textures["exposure"]];[blitter endEncoding];[copy commit];[copy waitUntilCompleted];}
        oracle.set_frame(jx,jy,f==0,1.f,f==0?0.f:1.f);
        oracle.dispatch("pre");
        const bool gpuSkip=*(const uint32_t *)((const uint8_t *)oracle.scratch.contents+33273792)==1511506142u;
        check(gpuSkip==view.skippedInference,"runtime skip prediction differs from original PRE marker");
        skipped+=gpuSkip;
        oracle.dispatch("pass1");oracle.dispatch("pass2");oracle.dispatch("border");oracle.dispatch("pass3");oracle.dispatch("post");
        same(blit(oracle.queue,output,8),shared(oracle.textures["output"],8),"output",f);
        same(blit(oracle.queue,view.history,8),shared(oracle.textures["history_next"],8),"history",f);
        same(blit(oracle.queue,view.recurrent,4),shared(oracle.textures["recurrent_next"],4),"recurrent",f);
        same(blit(oracle.queue,view.reprojected,8),shared(oracle.textures["reprojected"],8),"reprojected",f);
        {auto copy=[oracle.queue commandBuffer];auto blitter=[copy blitCommandEncoder];
         auto host=[oracle.device newBufferWithLength:oracle.logicalScratchBytes options:MTLResourceStorageModeShared];
         [blitter copyFromBuffer:view.scratch sourceOffset:0 toBuffer:host destinationOffset:0 size:oracle.logicalScratchBytes];
         [blitter endEncoding];[copy commit];[copy waitUntilCompleted];
         std::vector<uint8_t> a((const uint8_t *)host.contents,(const uint8_t *)host.contents+oracle.logicalScratchBytes);
         std::vector<uint8_t> b((const uint8_t *)oracle.scratch.contents,(const uint8_t *)oracle.scratch.contents+oracle.logicalScratchBytes);
         same(a,b,"logical scratch",f);}
        oracle.advance_history();
    }}
    NSData *final=oracle.read_texture(oracle.textures["output"]);
    std::printf("Runtime int8-lite-lossy-original (%s%s) vs original oracle:",view.backendLabel.c_str(),metal4?", Metal 4":"");
    std::printf("  %u frames (%u reused), output/history/recurrent/reprojected/scratch identical; output %s\n",
        frames,skipped,hash(final.bytes,final.length).c_str());
    return 0;
} catch(const std::exception &e){std::fprintf(stderr,"Original runtime A/B: %s\n",e.what());return 1;} }}
