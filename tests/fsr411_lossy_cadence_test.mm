// SPDX-License-Identifier: MIT
// Exercise temporal policy through the real native graph. The separate
// fsr411_lossy_test checks motion math; these tests check host scheduling,
// cache lifetime and GPU buffer ownership across complete frames.
#include "../src/fsr411_runtime.h"
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {
size_t assertions=0;
constexpr float sentinel=-123.f;

void require(bool condition,const std::string &message) {
    ++assertions;
    if(!condition) throw std::runtime_error(message);
}

id<MTLTexture> texture(id<MTLDevice> device,MTLPixelFormat format,unsigned w,unsigned h) {
    auto descriptor=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format width:w height:h mipmapped:NO];
    descriptor.storageMode=MTLStorageModeShared;
    descriptor.usage=MTLTextureUsageShaderRead|MTLTextureUsageShaderWrite;
    auto result=[device newTextureWithDescriptor:descriptor];
    require(result!=nil,"texture allocation failed");
    return result;
}

fsr411::Frame frame(id<MTLDevice> device,unsigned rw=64,unsigned rh=32,unsigned uw=128,unsigned uh=64) {
    fsr411::Frame f;
    f.color=texture(device,MTLPixelFormatRGBA32Float,rw,rh);
    f.depth=texture(device,MTLPixelFormatR32Float,rw,rh);
    f.motion=texture(device,MTLPixelFormatRG32Float,rw,rh);
    f.output=texture(device,MTLPixelFormatRGBA32Float,uw,uh);
    f.input.geometry={{rw,rh,0,0},{rw,rh,0,0},{rw,rh,0,0},{uw,uh,0,0},rw,rh};
    f.input.rw=rw;f.input.rh=rh;f.input.uw=uw;f.input.uh=uh;
    f.input.preExposure=f.input.mvScaleX=f.input.mvScaleY=1;
    return f;
}

void upload(fsr411::Frame &f,unsigned index) {
    const auto &g=f.input.geometry;
    std::vector<float> color(size_t(g.color.width)*g.color.height*4,3.f);
    std::vector<float> depth(size_t(g.depth.width)*g.depth.height,.5f);
    std::vector<float> motion(size_t(g.motion.width)*g.motion.height*2,0.f);
    for(unsigned y=0;y<f.input.rh;++y) for(unsigned x=0;x<f.input.rw;++x) {
        const size_t pixel=(size_t(y+g.color.offsetY)*g.color.width+x+g.color.offsetX)*4;
        color[pixel]=float(17+(x+3*index)%37)/64.f;
        color[pixel+1]=float(9+(y+index)%23)/48.f;
        color[pixel+2]=float(5+(x*3+y*5+index)%19)/32.f;
        color[pixel+3]=1;
    }
    [f.color replaceRegion:MTLRegionMake2D(0,0,g.color.width,g.color.height) mipmapLevel:0
        withBytes:color.data() bytesPerRow:g.color.width*16];
    [f.depth replaceRegion:MTLRegionMake2D(0,0,g.depth.width,g.depth.height) mipmapLevel:0
        withBytes:depth.data() bytesPerRow:g.depth.width*4];
    [f.motion replaceRegion:MTLRegionMake2D(0,0,g.motion.width,g.motion.height) mipmapLevel:0
        withBytes:motion.data() bytesPerRow:g.motion.width*8];
    std::vector<float> output(size_t(g.output.width)*g.output.height*4,sentinel);
    [f.output replaceRegion:MTLRegionMake2D(0,0,g.output.width,g.output.height) mipmapLevel:0
        withBytes:output.data() bytesPerRow:g.output.width*16];
}

void check_output(const fsr411::Frame &f) {
    const auto &g=f.input.geometry.output;
    std::vector<float> output(size_t(g.width)*g.height*4);
    [f.output getBytes:output.data() bytesPerRow:g.width*16
        fromRegion:MTLRegionMake2D(0,0,g.width,g.height) mipmapLevel:0];
    float lo=1e30f,hi=-1e30f;
    for(unsigned y=0;y<g.height;++y) for(unsigned x=0;x<g.width;++x) {
        const bool inside=x>=g.offsetX && y>=g.offsetY && x-g.offsetX<f.input.uw && y-g.offsetY<f.input.uh;
        for(unsigned channel=0;channel<4;++channel) {
            float value=output[(size_t(y)*g.width+x)*4+channel];
            if(!inside) require(value==sentinel,"frame wrote outside the logical output rectangle");
            else if(channel<3) {
                require(std::isfinite(value) && value!=sentinel,"frame did not produce finite RGB output");
                lo=std::min(lo,value);hi=std::max(hi,value);
            }
        }
    }
    require(hi>lo && hi>0,"frame did not produce nonconstant colour");
}

struct Harness {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    id<MTL4CommandQueue> queue4=nil;
    id<MTLSharedEvent> completed=nil;
    fsr411::Runtime runtime;
    fsr411::View view;
    fsr411::Options options;
    std::vector<uint8_t> previousScratch;
    unsigned count=0,skips=0;
    bool metal4=false;

    Harness(id<MTLDevice> d,const std::string &root,bool compile,int optimize,bool useMetal4)
        :device(d),queue([d newCommandQueue]),runtime(d,root,root+"/build"),metal4(useMetal4) {
        require(queue!=nil,"command queue allocation failed");
        if(metal4) {
            if(@available(macOS 26.0,*)) {
                queue4=[device newMTL4CommandQueue];completed=[device newSharedEvent];
                require(queue4!=nil && completed!=nil,"Metal 4 queue/event allocation failed");
            } else throw std::runtime_error("Metal 4 requires macOS 26");
        }
        options.backend=fsr411::Backend::Native;
        options.profile=fsr411::Profile::Lossy;
        options.allowCompile=compile;options.optimize=optimize;
    }

    void run(const char *name,fsr411::Frame &f,float jitter,bool skip,unsigned resetReasons=0) { @autoreleasepool {
        upload(f,count);
        f.input.jitterX=jitter;f.input.jitterY=.125f;
        const auto beforeFull=view.neuralRuns,beforeReuse=view.reusedFrames,beforeEncode=view.encodes;
        id cb=nil;
        id<MTL4CommandAllocator> allocator=nil;
        if(metal4) {
            if(@available(macOS 26.0,*)) {
                allocator=[device newCommandAllocator];cb=[device newCommandBuffer];
                require(allocator!=nil && cb!=nil,"Metal 4 command allocation failed");
                [cb beginCommandBufferWithAllocator:allocator];
            }
        } else cb=[queue commandBuffer];
        require(cb!=nil,"command buffer allocation failed");
        // Each reused frame must clear ownership immediately before PRE.
        // Maximum unsigned keys cannot be repaired by atomic-max alone.
        if(skip) {
            require(view.lossyPlane!=nil && view.lossyMotion!=nil,"skip has no motion/owner allocation");
            id blit=nil;
            if(metal4) blit=[cb computeCommandEncoder];else blit=[cb blitCommandEncoder];
            require(blit!=nil,"poison encoder allocation failed");
            if(metal4) [blit barrierAfterQueueStages:MTLStageAll beforeStages:MTLStageBlit visibilityOptions:MTL4VisibilityOptionDevice];
            [blit fillBuffer:view.lossyPlane range:NSMakeRange(0,view.lossyPlane.length) value:0xff];
            [blit fillBuffer:view.lossyMotion range:NSMakeRange(0,view.lossyMotion.length) value:0xff];
            if(metal4) [blit barrierAfterStages:MTLStageBlit beforeQueueStages:MTLStageAll visibilityOptions:MTL4VisibilityOptionDevice];
            [blit endEncoding];
        }
        std::string error;
        const bool encoded=runtime.encode(cb,view,f,options,error);
        require(encoded,std::string(name)+": "+error);
        require(view.skippedInference==skip,std::string(name)+": wrong inference scheduling");
        require(view.neuralRuns==beforeFull+!skip && view.reusedFrames==beforeReuse+skip,
                std::string(name)+": wrong inference counters");
        require(view.encodes==beforeEncode+1,std::string(name)+": encode counter did not advance once");
        require(view.cacheValid,std::string(name)+": native POST CNB cache unavailable");
        require((view.prepared.resetReasons&resetReasons)==resetReasons,std::string(name)+": expected reset reason absent");
        if(skip) require(!view.prepared.reset,std::string(name)+": reused a reset frame");
        id<MTLBuffer> scratch=[device newBufferWithLength:view.scratch.length options:MTLResourceStorageModeShared];
        id<MTLBuffer> motion=nil,plane=nil;
        require(scratch!=nil,"scratch readback allocation failed");
        if(skip) {
            motion=[device newBufferWithLength:view.lossyMotion.length options:MTLResourceStorageModeShared];
            plane=[device newBufferWithLength:view.lossyPlane.length options:MTLResourceStorageModeShared];
            require(motion!=nil && plane!=nil,"policy readback allocation failed");
        }
        id<MTLResidencySet> readbackResidency=nil;
        if(metal4) {
            NSError *e=nil;
            readbackResidency=[device newResidencySetWithDescriptor:[MTLResidencySetDescriptor new] error:&e];
            require(readbackResidency!=nil,"readback residency allocation failed: "+fsr411::metal_error(e));
            [readbackResidency addAllocation:scratch];
            [readbackResidency addAllocation:view.scratch];
            if(skip) {
                [readbackResidency addAllocation:motion];[readbackResidency addAllocation:plane];
                [readbackResidency addAllocation:view.lossyMotion];[readbackResidency addAllocation:view.lossyPlane];
            }
            [readbackResidency commit];[cb useResidencySet:readbackResidency];
        }
        id blit=nil;
        if(metal4) blit=[cb computeCommandEncoder];else blit=[cb blitCommandEncoder];
        require(blit!=nil,"readback encoder allocation failed");
        if(metal4) [blit barrierAfterQueueStages:MTLStageAll beforeStages:MTLStageBlit visibilityOptions:MTL4VisibilityOptionDevice];
        [blit copyFromBuffer:view.scratch sourceOffset:0 toBuffer:scratch destinationOffset:0 size:scratch.length];
        if(skip) {
            [blit copyFromBuffer:view.lossyMotion sourceOffset:0 toBuffer:motion destinationOffset:0 size:motion.length];
            [blit copyFromBuffer:view.lossyPlane sourceOffset:0 toBuffer:plane destinationOffset:0 size:plane.length];
        }
        if(metal4) [blit barrierAfterStages:MTLStageBlit beforeQueueStages:MTLStageAll visibilityOptions:MTL4VisibilityOptionDevice];
        [blit endEncoding];
        if(metal4) {
            if(@available(macOS 26.0,*)) {
                [cb endCommandBuffer];id<MTL4CommandBuffer> list[]={cb};
                [queue4 commit:list count:1];[queue4 signalEvent:completed value:count+1];
                require([completed waitUntilSignaledValue:count+1 timeoutMS:20000],std::string(name)+": Metal 4 execution timed out");
            }
        } else {
            auto classic=(id<MTLCommandBuffer>)cb;
            [classic commit];[classic waitUntilCompleted];
            require(classic.status==MTLCommandBufferStatusCompleted,std::string(name)+": "+fsr411::metal_error(classic.error));
        }
        check_output(f);
        if(skip) {
            require(previousScratch.size()==scratch.length,std::string(name)+": cached tensor allocation changed");
            require(!std::memcmp(previousScratch.data(),scratch.contents,scratch.length),
                    std::string(name)+": skip overwrote neural tensors or POST CNB cache");
            const auto *motionWords=static_cast<const uint32_t *>(motion.contents);
            const auto *owners=static_cast<const uint32_t *>(plane.contents);
            size_t covered=0;
            for(size_t block=0;block<plane.length/4;++block) {
                require(owners[block]!=UINT32_MAX,std::string(name)+": ownership plane was not cleared before PRE");
                require(motionWords[block*4+3]<=1,std::string(name)+": PRE did not rewrite motion metadata");
                covered+=owners[block]!=0;
            }
            require(covered>0,std::string(name)+": skipped PRE recorded no valid ownership");
            ++skips;
        }
        const auto *bytes=static_cast<const uint8_t *>(scratch.contents);
        previousScratch.assign(bytes,bytes+scratch.length);
        ++count;
        std::printf("cadence %-32s %s\n",name,skip?"reuse":"full");std::fflush(stdout);
    } }

    void reject(fsr411::Frame &f) {
        const auto before=view.encodes;
        auto invalid=f;invalid.input.jitterX=std::numeric_limits<float>::quiet_NaN();
        auto cb=[queue commandBuffer];std::string error;
        require(!runtime.encode(cb,view,invalid,options,error),"nonfinite input was accepted");
        require(!error.empty(),"rejected input has no error");
        require(view.encodes==before,"rejected input advanced frame bookkeeping");
        require(!view.cacheValid && !view.inputHistory.valid,"rejected input did not invalidate temporal cache");
    }
};

struct PendingFrame {
    fsr411::Frame frame;
    fsr411::Options options;
    id command=nil;
    id<MTL4CommandAllocator> allocator=nil;
    bool reused=false;
};

struct OrderedQueue {
    id<MTLDevice> device;
    id<MTLCommandQueue> classic;
    id<MTL4CommandQueue> metal4=nil;
    id<MTLSharedEvent> completed=nil;
    uint64_t token=0;
    bool useMetal4;

    OrderedQueue(id<MTLDevice> d,bool use):device(d),classic([d newCommandQueue]),useMetal4(use) {
        require(classic!=nil,"inflight queue allocation failed");
        if(useMetal4) {
            if(@available(macOS 26.0,*)) {
                metal4=[device newMTL4CommandQueue];completed=[device newSharedEvent];
                require(metal4!=nil && completed!=nil,"inflight Metal 4 queue/event allocation failed");
            } else throw std::runtime_error("Metal 4 requires macOS 26");
        }
    }

    bool encode(fsr411::Runtime &runtime,fsr411::View &view,PendingFrame &item,std::string &error) {
        if(useMetal4) {
            if(@available(macOS 26.0,*)) {
                item.allocator=[device newCommandAllocator];item.command=[device newCommandBuffer];
                require(item.allocator!=nil && item.command!=nil,"inflight Metal 4 command allocation failed");
                [item.command beginCommandBufferWithAllocator:item.allocator];
            }
        } else item.command=[classic commandBuffer];
        require(item.command!=nil,"inflight command allocation failed");
        const bool encoded=runtime.encode(item.command,view,item.frame,item.options,error);
        item.reused=view.skippedInference;
        if(useMetal4) [item.command endCommandBuffer];
        return encoded;
    }

    void commit(std::vector<PendingFrame> &items) {
        require(!items.empty(),"empty inflight commit");
        if(useMetal4) {
            if(@available(macOS 26.0,*)) {
                id<MTL4CommandBuffer> list[32];
                require(items.size()<=32,"inflight commit exceeds test capacity");
                for(size_t i=0;i<items.size();++i) list[i]=items[i].command;
                [metal4 commit:list count:items.size()];[metal4 signalEvent:completed value:++token];
                require([completed waitUntilSignaledValue:token timeoutMS:20000],"inflight Metal 4 execution timed out");
            }
        } else {
            for(auto &item:items) [(id<MTLCommandBuffer>)item.command commit];
            for(auto &item:items) {
                auto cb=(id<MTLCommandBuffer>)item.command;
                [cb waitUntilCompleted];
                require(cb.status==MTLCommandBufferStatusCompleted,"inflight GPU failure: "+fsr411::metal_error(cb.error));
            }
        }
    }
};

PendingFrame inflight_input(id<MTLDevice> device,unsigned index,const fsr411::Options &base,bool withoutUp) {
    PendingFrame item;item.options=base;
    const unsigned outputWidth=index%12>=8 && index%12<=10?125:128;
    const unsigned inputHeight=index%9==7?31:32;
    item.frame=frame(device,64,inputHeight,outputWidth,64);
    upload(item.frame,index);
    constexpr float jitter[6]={-.25f,-.125f,-.375f,0.f,-0.f,.25f};
    item.frame.input.jitterX=jitter[index%6];item.frame.input.jitterY=float(int(index%5)-2)/8.f;
    item.frame.input.preExposure=.75f+float(index%4)*.125f;
    item.frame.input.reset=index==13;
    item.frame.exposure=texture(device,MTLPixelFormatR32Float,1,1);
    item.frame.input.hasExposureTexture=1;
    const float exposure=.875f+float(index%5)*.125f;
    [item.frame.exposure replaceRegion:MTLRegionMake2D(0,0,1,1) mipmapLevel:0 withBytes:&exposure bytesPerRow:4];
    item.options.profile=index>=15 && index<18?fsr411::Profile::Exact:fsr411::Profile::Lossy;
    item.options.ultraPerformance=!withoutUp && index>=26 && index<30;
    item.options.sharpness=float(index%3)*.25f;item.options.rcas=item.options.sharpness>0;
    return item;
}

std::vector<uint8_t> output_bytes(const fsr411::Frame &f) {
    std::vector<uint8_t> bytes(size_t(f.output.width)*f.output.height*16);
    [f.output getBytes:bytes.data() bytesPerRow:f.output.width*16
        fromRegion:MTLRegionMake2D(0,0,f.output.width,f.output.height) mipmapLevel:0];
    return bytes;
}

void run_inflight(id<MTLDevice> device,const std::string &root,bool compile,int optimize,bool metal4,bool withoutUp) {
    fsr411::Runtime runtime(device,root,root+"/build");
    fsr411::Options options;options.backend=fsr411::Backend::Native;options.profile=fsr411::Profile::Lossy;
    options.allowCompile=compile;options.optimize=optimize;
    OrderedQueue queue(device,metal4);
    fsr411::View sequential,batched;
    std::vector<std::vector<uint8_t>> expected;
    std::vector<bool> expectedReuse;
    std::string error;
    for(unsigned index=0;index<36;++index) { @autoreleasepool {
        std::vector<PendingFrame> items;
        items.push_back(inflight_input(device,index,options,withoutUp));
        const bool encoded=queue.encode(runtime,sequential,items[0],error);
        require(encoded,"sequential inflight reference encode: "+error);
        queue.commit(items);check_output(items[0].frame);
        expected.push_back(output_bytes(items[0].frame));expectedReuse.push_back(items[0].reused);
    } }
    for(unsigned batch=0;batch<2;++batch) { @autoreleasepool {
        std::vector<PendingFrame> items;items.reserve(18);
        for(unsigned offset=0;offset<18;++offset) {
            const unsigned index=batch*18+offset;
            items.push_back(inflight_input(device,index,options,withoutUp));
            const bool encoded=queue.encode(runtime,batched,items.back(),error);
            require(encoded,"batched inflight encode: "+error);
            require(items.back().reused==expectedReuse[index],"inflight scheduling differs from sequential scheduling");
        }
        // No command buffer in this batch has been committed yet. Every
        // pending frame must still own a distinct immutable constant slot.
        require(batched.slots.size()==18,"inflight encodes did not retain 18 constant slots");
        for(const auto &slot:batched.slots) require(slot->busy && !slot->retired(),"uncommitted frame slot was retired");
        queue.commit(items);
        for(unsigned offset=0;offset<18;++offset) {
            check_output(items[offset].frame);
            require(output_bytes(items[offset].frame)==expected[batch*18+offset],
                    "inflight output differs at frame "+std::to_string(batch*18+offset));
        }
        std::printf("inflight batch %u: 18 queued frames match sequential output byte-for-byte\n",batch+1);std::fflush(stdout);
    } }
    require(batched.neuralRuns==sequential.neuralRuns && batched.reusedFrames==sequential.reusedFrames,
            "inflight model/reuse counters differ");

    // Reaching the bounded ring must reject the 33rd uncommitted frame,
    // preserve all 32 pending uniforms and recover after they retire.
    fsr411::View limited;
    std::vector<PendingFrame> pending;pending.reserve(32);
    for(unsigned index=0;index<32;++index) {
        pending.push_back(inflight_input(device,index,options,withoutUp));
        const bool encoded=queue.encode(runtime,limited,pending.back(),error);
        require(encoded,"slot-limit setup: "+error);
    }
    require(limited.slots.size()==32,"slot-limit setup did not fill the bounded ring");
    std::vector<std::vector<uint8_t>> uniforms;
    for(const auto &slot:limited.slots) {
        const auto *bytes=static_cast<const uint8_t *>(slot->constants.contents);
        uniforms.emplace_back(bytes,bytes+slot->constants.length);
    }
    auto overflow=inflight_input(device,32,options,withoutUp);
    require(!queue.encode(runtime,limited,overflow,error),"33rd uncommitted frame was accepted");
    require(error.find("32 FSR frames")!=std::string::npos,"slot-limit rejection did not identify pending frames");
    require(limited.encodes==32 && limited.slots.size()==32,"rejected encode changed the bounded ring");
    require(!limited.cacheValid && !limited.inputHistory.valid,"rejected frame did not invalidate temporal bookkeeping");
    for(size_t index=0;index<uniforms.size();++index)
        require(!std::memcmp(uniforms[index].data(),limited.slots[index]->constants.contents,uniforms[index].size()),
                "overflow encode overwrote pending frame uniforms");
    queue.commit(pending);
    for(unsigned index=0;index<pending.size();++index)
        require(output_bytes(pending[index].frame)==expected[index],"slot-limit setup corrupted an earlier frame");
    std::vector<PendingFrame> retry;
    retry.push_back(inflight_input(device,32,options,withoutUp));
    const bool encoded=queue.encode(runtime,limited,retry[0],error);
    require(encoded,"slot-limit recovery failed: "+error);
    require(!retry[0].reused && limited.prepared.reset,"slot-limit recovery reused stale cache");
    queue.commit(retry);check_output(retry[0].frame);
    require(limited.encodes==33 && limited.slots.size()==32,"slot-limit recovery did not reuse a retired slot");
    std::printf("FSR 4.1.1 inflight (%s): 36 frames in 18-frame batches match sequential output; 32-slot rejection/recovery passed, %zu assertions\n",
                metal4?"Metal 4":"classic Metal",assertions);
}
} // namespace

int main(int argc,char **argv) { @autoreleasepool { try {
    if(argc<2) {
        std::fprintf(stderr,"usage: fsr411_lossy_cadence_test <repository> [--compile] [--optimize0] [--without-up] [--mtl4] [--inflight]\n");return 2;
    }
    bool compile=false,withoutUp=false,metal4=false,inflight=false;int optimize=1;
    for(int i=2;i<argc;++i) {
        if(!std::strcmp(argv[i],"--compile")) compile=true;
        else if(!std::strcmp(argv[i],"--optimize0")) optimize=0;
        else if(!std::strcmp(argv[i],"--without-up")) withoutUp=true;
        else if(!std::strcmp(argv[i],"--mtl4")) metal4=true;
        else if(!std::strcmp(argv[i],"--inflight")) inflight=true;
        else throw std::runtime_error(std::string("unknown argument ")+argv[i]);
    }
    auto device=MTLCreateSystemDefaultDevice();require(device!=nil,"no Metal device");
    if(inflight) {run_inflight(device,argv[1],compile,optimize,metal4,withoutUp);return 0;}
    Harness h(device,argv[1],compile,optimize,metal4);
    auto f=frame(device);
    h.run("first negative",f,-.25f,false,FSR4_INPUT_RESET_FIRST_FRAME);
    h.run("positive jitter",f,.25f,false);
    h.run("negative zero",f,-0.f,false);
    require(std::signbit(h.view.prepared.jitter[0]),"test lost the negative-zero jitter bit");
    h.run("positive zero",f,0.f,false);
    h.run("eligible negative",f,-.25f,true);
    h.run("consecutive negative",f,-.125f,false);
    h.run("negative after refresh",f,-.375f,true);
    h.run("refresh before reset",f,.25f,false);
    f.input.reset=1;
    h.run("explicit reset",f,-.25f,false,FSR4_INPUT_RESET_REQUESTED);
    f.input.reset=0;
    h.run("after reset",f,-.25f,true);
    h.run("refresh before invalidation",f,.25f,false);
    h.view.invalidate();
    require(!h.view.cacheValid && !h.view.inputHistory.valid,"explicit invalidation retained the cache");
    h.run("owner invalidation",f,-.25f,false,FSR4_INPUT_RESET_FIRST_FRAME);
    h.run("after invalidation",f,-.25f,true);
    h.run("refresh before input failure",f,.25f,false);
    h.reject(f);
    h.run("after rejected input",f,-.25f,false,FSR4_INPUT_RESET_FIRST_FRAME);
    h.run("after failure refresh",f,-.25f,true);

    h.run("refresh before narrow input",f,.25f,false);
    auto narrow=f;narrow.input.rw=63;
    h.run("under half width",narrow,-.25f,false);
    h.run("under half width repeated",narrow,-.25f,false);
    auto shortInput=f;shortInput.input.rh=31;
    h.run("width-only eligibility",shortInput,-.25f,true);
    h.run("width-only refresh",shortInput,-.25f,false);
    h.run("width-only reuse",shortInput,-.25f,true);

    h.run("refresh before profile change",f,.25f,false);
    h.options.profile=fsr411::Profile::Exact;
    h.run("switch to Exact",f,-.25f,false,FSR4_INPUT_RESET_MODEL);
    h.run("Exact never skips",f,-.25f,false);
    h.options.profile=fsr411::Profile::Lossy;
    h.run("switch to Lossy",f,-.25f,false,FSR4_INPUT_RESET_MODEL);
    h.run("Lossy cache refreshed",f,-.25f,true);
    h.run("refresh before logical resize",f,.25f,false);
    auto cropped=f;cropped.input.uw=125;
    h.run("same allocation logical resize",cropped,-.25f,false,FSR4_INPUT_RESET_OUTPUT_SIZE);
    h.run("after logical resize",cropped,-.25f,true);
    auto wider=frame(device,64,32,129,64);
    h.run("aligned allocation resize",wider,-.25f,false,FSR4_INPUT_RESET_OUTPUT_SIZE);
    h.run("aligned width threshold",wider,-.25f,false);

    if(!withoutUp) {
        h.options.ultraPerformance=true;
        h.run("Ultra Performance change",f,-.25f,false,FSR4_INPUT_RESET_MODEL);
        h.run("Ultra Performance never skips",f,-.25f,false);
        h.options.ultraPerformance=false;
        h.run("return from Ultra Performance",f,-.25f,false,FSR4_INPUT_RESET_MODEL);
        h.run("normal cache refreshed",f,-.25f,true);
    }
    std::printf("FSR 4.1.1 Lossy runtime cadence (%s): %u complete frames, %u reuses, %zu assertions passed%s\n",
        metal4?"Metal 4":"classic Metal",h.count,h.skips,assertions,withoutUp?" (Ultra Performance omitted)":"");
    return 0;
} catch(const std::exception &error) {
    std::fprintf(stderr,"Lossy cadence QA: %s\n",error.what());return 1;
} } }
