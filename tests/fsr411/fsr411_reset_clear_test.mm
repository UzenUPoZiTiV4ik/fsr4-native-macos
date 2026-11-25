// SPDX-License-Identifier: MIT
// New/dirty temporal textures must be overwritten before reset can read them.
// This executable intentionally runs the real GPU graph; it is not a benchmark.
#include "../../src/fsr41x/fsr411_runtime.h"
#include <cstdio>
#include <stdexcept>
#include <vector>

namespace {
void require(bool ok,const std::string &message) { if(!ok) throw std::runtime_error(message); }

id<MTLTexture> texture(id<MTLDevice> device,MTLPixelFormat format,unsigned width,unsigned height) {
    auto d=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format width:width height:height mipmapped:NO];
    d.storageMode=MTLStorageModeShared; d.usage=MTLTextureUsageShaderRead|MTLTextureUsageShaderWrite;
    auto result=[device newTextureWithDescriptor:d];require(result!=nil,"texture allocation failed");return result;
}

void poison_targets(fsr411::View &view,bool dirty) {
    const size_t count=size_t(view.width)*view.height*4;
    std::vector<uint16_t> half(count,0);
    // NaNs, both infinities, negative zero, a large finite and normal values.
    const uint16_t patterns[]={0x7e01,0xfc00,0x7c00,0x8000,0x7bff,0xbc00,0x3555,0x0400};
    if(dirty) for(size_t i=0;i<count;++i) half[i]=patterns[i%8];
    for(auto t:{view.history,view.reprojected})
        [t replaceRegion:MTLRegionMake2D(0,0,view.width,view.height) mipmapLevel:0 withBytes:half.data() bytesPerRow:view.width*8];
    std::vector<uint8_t> recurrent(count,0);
    if(dirty) for(size_t i=0;i<count;++i) recurrent[i]=uint8_t(37*i+19);
    [view.recurrent replaceRegion:MTLRegionMake2D(0,0,view.width,view.height) mipmapLevel:0
        withBytes:recurrent.data() bytesPerRow:view.width*4];
}

void allocate_targets(id<MTLDevice> device,fsr411::View &view,unsigned width,unsigned height,bool dirty) {
    view.width=(width+7u)&~7u;view.height=(height+7u)&~7u;
    view.history=texture(device,MTLPixelFormatRGBA16Float,view.width,view.height);
    view.reprojected=texture(device,MTLPixelFormatRGBA16Float,view.width,view.height);
    view.recurrent=texture(device,MTLPixelFormatRGBA8Unorm,view.width,view.height);
    view.zeroTargets=true;poison_targets(view,dirty);
}

void equal_texture(id<MTLTexture> a,id<MTLTexture> b,size_t bytesPerPixel,const char *name) {
    require(a.width==b.width && a.height==b.height,"texture shape changed");
    std::vector<uint8_t> x(a.width*a.height*bytesPerPixel),y(x.size());
    for(auto item:{std::make_pair(a,x.data()),std::make_pair(b,y.data())})
        [item.first getBytes:item.second bytesPerRow:a.width*bytesPerPixel
            fromRegion:MTLRegionMake2D(0,0,a.width,a.height) mipmapLevel:0];
    require(x==y,std::string(name)+" differs after dirty reset");
}

struct Trace final : fsr411::StageProfiler {
    std::vector<std::string> stages;
    void begin_stage(id __strong &,const std::string &name) override {stages.push_back(name);}
    void end_stage(id) override {}
    unsigned count(const char *name) const {return unsigned(std::count(stages.begin(),stages.end(),name));}
};

struct Queue {
    id<MTLDevice> device;id<MTLCommandQueue> classic=nil;id<MTL4CommandQueue> metal4=nil;
    id<MTLSharedEvent> completed=nil;uint64_t sequence=0;bool useMetal4;
    Queue(id<MTLDevice> d,bool use):device(d),useMetal4(use) {
        if(useMetal4) {
            if(@available(macOS 26.0,*)) {
                metal4=[device newMTL4CommandQueue];completed=[device newSharedEvent];
                require(metal4 && completed,"Metal 4 queue allocation failed");
            } else throw std::runtime_error("Metal 4 requires macOS 26");
        } else {classic=[device newCommandQueue];require(classic!=nil,"queue allocation failed");}
    }
    void run(fsr411::Runtime &runtime,fsr411::View &view,const fsr411::Frame &frame,fsr411::Options options,
             bool reset,bool expectClears) {
        id command=nil;id<MTL4CommandAllocator> allocator=nil;
        if(useMetal4) {
            if(@available(macOS 26.0,*)) {
                allocator=[device newCommandAllocator];command=[device newCommandBuffer];
                require(allocator && command,"Metal 4 command allocation failed");
                [command beginCommandBufferWithAllocator:allocator];
            }
        } else command=[classic commandBuffer];
        Trace trace;if(!useMetal4) options.stageProfiler=&trace;
        std::string error;bool encoded=runtime.encode(command,view,frame,options,error);
        require(encoded,"encode failed: "+error);require(bool(view.prepared.reset)==reset,"wrong reset decision");
        if(!useMetal4) {
            for(const char *name:{"clearHistory","clearReprojected","clearRecurrent"})
                require(trace.count(name)==unsigned(expectClears),std::string("wrong dispatch count: ")+name);
            require(trace.count("initExposure")==unsigned(reset||expectClears),"exposure initialization changed");
        }
        if(useMetal4) {
            [command endCommandBuffer];id<MTL4CommandBuffer> list[]={command};[metal4 commit:list count:1];
            [metal4 signalEvent:completed value:++sequence];
            require([completed waitUntilSignaledValue:sequence timeoutMS:20000],"Metal 4 execution timed out");
        } else {
            auto cb=(id<MTLCommandBuffer>)command;[cb commit];[cb waitUntilCompleted];
            require(cb.status==MTLCommandBufferStatusCompleted,"GPU failed: "+fsr411::metal_error(cb.error));
        }
    }
};

void run_case(id<MTLDevice> device,fsr411::Runtime &runtime,Queue &queue,fsr411::Options options,unsigned uw,unsigned uh) {
    const unsigned rw=(uw+7u)/2u,rh=(uh+7u)/2u;
    fsr411::Frame a;a.color=texture(device,MTLPixelFormatRGBA32Float,rw,rh);
    a.depth=texture(device,MTLPixelFormatR32Float,rw,rh);a.motion=texture(device,MTLPixelFormatRG32Float,rw,rh);
    a.output=texture(device,MTLPixelFormatRGBA32Float,uw,uh);
    a.input.geometry={{rw,rh,0,0},{rw,rh,0,0},{rw,rh,0,0},{uw,uh,0,0},rw,rh};
    a.input.rw=rw;a.input.rh=rh;a.input.uw=uw;a.input.uh=uh;
    a.input.preExposure=a.input.mvScaleX=a.input.mvScaleY=1;a.input.depthReversed=1;a.input.autoExposure=1;
    auto b=a;b.output=texture(device,MTLPixelFormatRGBA32Float,uw,uh);
    fsr411::View clean,dirty;allocate_targets(device,clean,uw,uh,false);allocate_targets(device,dirty,uw,uh,true);
    std::vector<float> depth(size_t(rw)*rh,.5f),motion(size_t(rw)*rh*2,0.f),color(size_t(rw)*rh*4);
    [a.depth replaceRegion:MTLRegionMake2D(0,0,rw,rh) mipmapLevel:0 withBytes:depth.data() bytesPerRow:rw*4];
    [a.motion replaceRegion:MTLRegionMake2D(0,0,rw,rh) mipmapLevel:0 withBytes:motion.data() bytesPerRow:rw*8];
    for(unsigned frame=0;frame<7;++frame) {
        const bool reset=frame==0||frame==3,extraClear=frame==6;
        if(frame==3||extraClear) {poison_targets(clean,false);poison_targets(dirty,true);}
        if(extraClear) clean.zeroTargets=dirty.zeroTargets=true;
        for(size_t i=0;i<color.size();++i) color[i]=i%4==3?1.f:float(1+(i*13+frame*7)%67)/80.f;
        [a.color replaceRegion:MTLRegionMake2D(0,0,rw,rh) mipmapLevel:0 withBytes:color.data() bytesPerRow:rw*16];
        a.input.reset=reset;a.input.jitterX=frame%3==1?-.125f:.25f;a.input.jitterY=.125f;b.input=a.input;
        queue.run(runtime,clean,a,options,reset,extraClear);queue.run(runtime,dirty,b,options,reset,extraClear);
        equal_texture(a.output,b.output,16,"output");
        // Compare the complete aligned region, including pixels outside the
        // logical destination and partial 16/32-pixel dispatch tiles.
        equal_texture(clean.history,dirty.history,8,"history");
        equal_texture(clean.reprojected,dirty.reprojected,8,"reprojected");
        equal_texture(clean.recurrent,dirty.recurrent,4,"recurrent");
        std::vector<float> output(size_t(uw)*uh*4);
        [a.output getBytes:output.data() bytesPerRow:uw*16 fromRegion:MTLRegionMake2D(0,0,uw,uh) mipmapLevel:0];
        for(size_t i=0;i<output.size();++i) if(i%4!=3) require(std::isfinite(output[i]),"nonfinite output");
        std::printf("reset clear %s %ux%u frame%u reset=%u fallback=%u: byte-identical\n",
            fsr411::profile_name(options.profile),uw,uh,frame,reset,extraClear);
    }
}
}

int main(int argc,char **argv) { @autoreleasepool { try {
    if(argc<3) {std::fprintf(stderr,"usage: fsr411_reset_clear_test <root> <cache> [--compile] [--mtl4] [--profile lossy|exact-fast]\n");return 2;}
    bool metal4=false;fsr411::Options options;options.backend=fsr411::Backend::Native;options.allowCompile=false;
    options.tier=2160;options.profile=fsr411::Profile::Lossy;options.rcas=true;options.sharpness=.401f;
    for(int i=3;i<argc;++i) {
        if(!std::strcmp(argv[i],"--mtl4")) metal4=true;
        else if(!std::strcmp(argv[i],"--compile")) options.allowCompile=true;
        else if(!std::strcmp(argv[i],"--profile")&&i+1<argc) require(fsr411::parse_profile(argv[++i],options.profile),"unknown profile");
        else throw std::runtime_error("unknown option");
    }
    auto device=MTLCreateSystemDefaultDevice();require(device!=nil,"Metal device unavailable");
    Queue queue(device,metal4);fsr411::Runtime runtime(device,argv[1],argv[2]);
    for(auto size:{std::make_pair(72u,48u),std::make_pair(97u,65u)}) run_case(device,runtime,queue,options,size.first,size.second);
    std::puts("FSR 4.1.1 dirty-reset checks passed");return 0;
} catch(const std::exception &e) {std::fprintf(stderr,"%s\n",e.what());return 1;} } }
