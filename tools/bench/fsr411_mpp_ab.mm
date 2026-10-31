// SPDX-License-Identifier: MIT
// Full-frame M5 INT8 A/B, one process, serial alternating pairs. The baseline
// pins FSR4_MPP_OPTIMIZE=0 before loading its sessions; the candidate uses the
// probed register-only kernels and real threadgroup pipeline limits. No
// compilation, counters or diagnostic copies occur in timed command buffers.
// Metal 4 is validation-only (no unsupported classic timestamp substitution).
// usage: fsr411_mpp_ab ROOT [--before-root ROOT] [--rounds N] [--size RW RH UW UH]
//        [--models lite,exact,lossy2] [--verify-every-frame] [--mtl4] [--report FILE]
#include "../../src/fsr412_runtime.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <vector>

namespace {
void require(bool ok,const std::string &message) {if(!ok)throw std::runtime_error(message);}
id<MTLTexture> texture(id<MTLDevice> dev,MTLPixelFormat format,unsigned w,unsigned h) {
    auto d=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format width:w height:h mipmapped:NO];
    d.storageMode=MTLStorageModeShared;d.usage=MTLTextureUsageShaderRead|MTLTextureUsageShaderWrite;
    auto t=[dev newTextureWithDescriptor:d];require(t!=nil,"texture allocation failed");return t;
}
struct Inputs {id<MTLTexture> color,depth,motion,exposure;};
Inputs inputs(id<MTLDevice> dev,unsigned rw,unsigned rh) {
    Inputs in{texture(dev,MTLPixelFormatRGBA16Float,rw,rh),texture(dev,MTLPixelFormatR32Float,rw,rh),
              texture(dev,MTLPixelFormatRG16Float,rw,rh),texture(dev,MTLPixelFormatR32Float,1,1)};
    std::vector<_Float16> c(size_t(rw)*rh*4),m(size_t(rw)*rh*2);std::vector<float>d(size_t(rw)*rh);
    uint32_t r=0x4128u;
    for(unsigned y=0;y<rh;++y)for(unsigned x=0;x<rw;++x) {
        const size_t i=size_t(y)*rw+x;r=r*1664525u+1013904223u;
        c[i*4]=(_Float16)(.05f+.8f*float(x)/rw+float(r>>28)/64.f);
        c[i*4+1]=(_Float16)(.03f+.6f*float(y)/rh+float((r>>24)&15)/80.f);
        c[i*4+2]=(_Float16)(float(((x*7+y*3)%23)+1)/30.f);c[i*4+3]=(_Float16)1;
        d[i]=.2f+.6f*float((x/8+y/8)%11)/11.f;
        m[i*2]=(_Float16)(float(int(x%9)-4)*.0625f);m[i*2+1]=(_Float16)(float(int(y%5)-2)*.0625f);
    }
    [in.color replaceRegion:MTLRegionMake2D(0,0,rw,rh) mipmapLevel:0 withBytes:c.data() bytesPerRow:rw*8];
    [in.depth replaceRegion:MTLRegionMake2D(0,0,rw,rh) mipmapLevel:0 withBytes:d.data() bytesPerRow:rw*4];
    [in.motion replaceRegion:MTLRegionMake2D(0,0,rw,rh) mipmapLevel:0 withBytes:m.data() bytesPerRow:rw*4];
    const float one=1;[in.exposure replaceRegion:MTLRegionMake2D(0,0,1,1) mipmapLevel:0 withBytes:&one bytesPerRow:4];return in;
}
fsr411::Frame frame(const Inputs &in,id<MTLTexture> output,unsigned rw,unsigned rh,unsigned uw,unsigned uh,unsigned index) {
    fsr411::Frame f;f.color=in.color;f.depth=in.depth;f.motion=in.motion;f.exposure=in.exposure;f.output=output;
    auto &p=f.input;p.rw=rw;p.rh=rh;p.uw=uw;p.uh=uh;
    p.geometry.color={rw,rh,0,0};p.geometry.depth=p.geometry.motion=p.geometry.color;p.geometry.output={uw,uh,0,0};
    // Eight-frame cycles include reset, full non-reset inference and reuse.
    // This avoids the reset+reuse-only classification of the two-frame replay.
    p.reset=index==0;p.jitterX=index%2?-.25f:.125f;p.jitterY=index%3?1.f/6.f:-1.f/6.f;
    p.mvScaleX=p.mvScaleY=1;p.depthReversed=1;p.preExposure=1;p.hasExposureTexture=1;return f;
}
std::vector<uint8_t> read_texture(id<MTLCommandQueue> queue,id<MTLTexture> t,unsigned bytes) {
    require(t!=nil,"missing texture during state validation");
    const NSUInteger pitch=(t.width*bytes+255u)&~NSUInteger(255);
    auto b=[t.device newBufferWithLength:pitch*t.height options:MTLResourceStorageModeShared];require(b!=nil,"readback allocation failed");
    auto cb=[queue commandBuffer];auto enc=[cb blitCommandEncoder];
    [enc copyFromTexture:t sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0,0,0) sourceSize:MTLSizeMake(t.width,t.height,1)
        toBuffer:b destinationOffset:0 destinationBytesPerRow:pitch destinationBytesPerImage:pitch*t.height];
    [enc endEncoding];[cb commit];[cb waitUntilCompleted];require(cb.status==MTLCommandBufferStatusCompleted,fsr411::metal_error(cb.error));
    std::vector<uint8_t> result(t.width*t.height*bytes);
    for(NSUInteger y=0;y<t.height;++y)std::memcpy(result.data()+y*t.width*bytes,(const uint8_t *)b.contents+y*pitch,t.width*bytes);
    if(bytes==8)for(size_t i=0;i<result.size();i+=2) {
        _Float16 value;std::memcpy(&value,result.data()+i,2);require(std::isfinite(float(value)),"nonfinite output/history");
    }
    return result;
}
double median(std::vector<double> v) {
    if(v.empty())return 0;std::sort(v.begin(),v.end());const size_t n=v.size();return n%2?v[n/2]:(v[n/2-1]+v[n/2])*.5;
}
NSArray *array(const std::vector<double> &v) {auto a=[NSMutableArray new];for(double x:v)[a addObject:@(x)];return a;}
struct State {
    fsr411::View view[2];id<MTLTexture> output[2];fsr411::Options options;
    std::vector<double> neural[2],full[2],reset[2],reused[2],ratios,fullRatios,reuseRatios;
    unsigned neuralCount=0,reuseCount=0,verified=0;
};
bool identical(id<MTLCommandQueue> queue,const State &s) {
    return read_texture(queue,s.output[0],8)==read_texture(queue,s.output[1],8)
        && read_texture(queue,s.view[0].history,8)==read_texture(queue,s.view[1].history,8)
        && read_texture(queue,s.view[0].recurrent,4)==read_texture(queue,s.view[1].recurrent,4)
        && read_texture(queue,s.view[0].autoExposure,4)==read_texture(queue,s.view[1].autoExposure,4);
}
}

int main(int argc,char **argv) {@autoreleasepool {try {
    require(argc>=2,"usage: fsr411_mpp_ab ROOT [--before-root ROOT] [--rounds N] [--size RW RH UW UH] [--models lite,exact,lossy2] [--verify-every-frame] [--mtl4] [--report FILE]");
    const std::string root=argv[1];std::string beforeRoot=root,models="lite,exact,lossy2",reportPath="build/fsr411_mpp_ab.json";
    unsigned rounds=30,rw=1000,rh=648,uw=1712,uh=1112;bool verifyEvery=false,metal4=false;
    for(int i=2;i<argc;++i) {
        const std::string a=argv[i];
        if(a=="--rounds" && i+1<argc)rounds=std::stoul(argv[++i]);
        else if(a=="--before-root" && i+1<argc)beforeRoot=argv[++i];
        else if(a=="--models" && i+1<argc)models=argv[++i];
        else if(a=="--report" && i+1<argc)reportPath=argv[++i];
        else if(a=="--size" && i+4<argc){rw=std::stoul(argv[++i]);rh=std::stoul(argv[++i]);uw=std::stoul(argv[++i]);uh=std::stoul(argv[++i]);}
        else if(a=="--verify-every-frame")verifyEvery=true;
        else if(a=="--mtl4")metal4=true;
        else throw std::runtime_error("unknown or incomplete option: "+a);
    }
    require(rounds>=2 && rounds<=1000,"rounds must be 2..1000");
    require(rw && rh && uw && uh && rw<=8192 && rh<=4320 && uw<=7680 && uh<=4320,"invalid image dimensions");
    auto dev=MTLCreateSystemDefaultDevice();auto queue=[dev newCommandQueue];require(dev && queue && [dev supportsFamily:MTLGPUFamilyApple10],"Apple10/M5 required");
    const auto thermalStart=NSProcessInfo.processInfo.thermalState;const auto in=inputs(dev,rw,rh);
    id<MTL4CommandQueue> queue4=nil;id<MTLSharedEvent> event=nil;uint64_t eventValue=0;
    if(metal4){if(@available(macOS 26.0,*)){queue4=[dev newMTL4CommandQueue];event=[dev newSharedEvent];}require(queue4 && event,"Metal 4 unavailable");}
    struct Model {std::string name;bool lite;fsr411::Profile profile;};std::vector<Model> list;
    for(const auto &m:std::vector<Model>{{"lite",true,fsr411::Profile::Int8Lite},{"exact",false,fsr411::Profile::Exact},
        {"lossy2",false,fsr411::Profile::Lossy2},{"exact-fast",false,fsr411::Profile::ExactFast}})
        if((","+models+",").find(","+m.name+",")!=std::string::npos)list.push_back(m);
    require(!list.empty(),"no models selected");std::vector<State> states(list.size());
    auto encode=[&](fsr411::Runtime &rt,State &s,unsigned variant,unsigned index) {@autoreleasepool {
        std::string error;const auto f=frame(in,s.output[variant],rw,rh,uw,uh,index);
        double ms=0;
        if(metal4) {
            if(@available(macOS 26.0,*)) {
                auto alloc=[dev newCommandAllocator];auto cb=[dev newCommandBuffer];[cb beginCommandBufferWithAllocator:alloc];
                require(rt.encode(cb,s.view[variant],f,s.options,error),error);[cb endCommandBuffer];
                id<MTL4CommandBuffer> list[]={cb};[queue4 commit:list count:1];[queue4 signalEvent:event value:++eventValue];
                require([event waitUntilSignaledValue:eventValue timeoutMS:20000],"Metal 4 GPU timeout");
            }
        } else {
            auto cb=[queue commandBuffer];require(rt.encode(cb,s.view[variant],f,s.options,error),error);
            [cb commit];[cb waitUntilCompleted];require(cb.status==MTLCommandBufferStatusCompleted,fsr411::metal_error(cb.error));
            ms=(cb.GPUEndTime-cb.GPUStartTime)*1000.;require(std::isfinite(ms) && ms>0,"invalid GPU timestamp");
        }
        return std::make_pair(ms,s.view[variant].skippedInference);
    }};
    // Preserve a caller's environment; both Runtime objects pin the setting
    // before the measured loop. Build/import caches separately before this tool.
    const char *oldSetting=std::getenv("FSR4_MPP_OPTIMIZE");const bool hadSetting=oldSetting!=nullptr;const std::string saved=oldSetting?oldSetting:"";
    setenv("FSR4_MPP_OPTIMIZE","0",1);fsr411::Runtime before(dev,beforeRoot,root+"/build");
    for(size_t i=0;i<list.size();++i) {
        auto &s=states[i];if(list[i].lite)s.options=fsr412::Options();else s.options.profile=list[i].profile;
        s.options.backend=fsr411::Backend::Native;s.options.allowCompile=false;s.options.optimize=1;
        for(unsigned v=0;v<2;++v)s.output[v]=texture(dev,MTLPixelFormatRGBA16Float,uw,uh);
        encode(before,s,0,0);require(s.view[0].backendLabel=="int8-mpp" && !s.view[0].mppFeatures && !s.view[0].mppFastPasses,"baseline MPP opt-out not selected");
    }
    setenv("FSR4_MPP_OPTIMIZE","1",1);fsr411::Runtime after(dev,root,root+"/build");
    for(auto &s:states){encode(after,s,1,0);require(s.view[1].mppFeatures==fsr411::kMppChainKnownMask && s.view[1].mppFastPasses>0,"candidate register-only kernels not selected");}
    if(hadSetting)setenv("FSR4_MPP_OPTIMIZE",saved.c_str(),1);else unsetenv("FSR4_MPP_OPTIMIZE");
    fsr411::Runtime *runtime[]={&before,&after};constexpr unsigned period=8,warmups=2;
    for(unsigned round=0;round<rounds+warmups;++round)for(size_t i=0;i<list.size();++i)for(unsigned index=0;index<period;++index) {
        auto &s=states[i];double t[2];bool reused[2];
        for(unsigned k=0;k<2;++k){const unsigned v=(round+index)%2?1-k:k;const auto r=encode(*runtime[v],s,v,index);t[v]=r.first;reused[v]=r.second;}
        require(reused[0]==reused[1],"inference reuse decisions differ");
        if(verifyEvery){require(identical(queue,s),list[i].name+": output/history/recurrent/exposure differ");++s.verified;}
        if(round<warmups)continue;
        if(reused[0])++s.reuseCount;else ++s.neuralCount;
        if(!metal4) {
            for(unsigned v=0;v<2;++v) {
                (reused[v]?s.reused[v]:s.neural[v]).push_back(t[v]);
                if(!reused[v])(index==0?s.reset[v]:s.full[v]).push_back(t[v]);
            }
            if(reused[0])s.reuseRatios.push_back(t[1]/t[0]);
            else{s.ratios.push_back(t[1]/t[0]);if(index)s.fullRatios.push_back(t[1]/t[0]);}
        }
    }
    auto rows=[NSMutableArray new];
    for(size_t i=0;i<list.size();++i) {
        auto &s=states[i];require(identical(queue,s),list[i].name+": final output/history/recurrent/exposure differ");++s.verified;
        if(metal4)std::printf("%-10s Metal 4: neural pairs %u, reused %u, byte-identical checks %u\n",list[i].name.c_str(),s.neuralCount,s.reuseCount,s.verified);
        else std::printf("%-10s neural median: %.3f -> %.3f ms; neural pairs %u, reused %u, byte-identical checks %u\n",list[i].name.c_str(),median(s.neural[0]),median(s.neural[1]),s.neuralCount,s.reuseCount,s.verified);
        [rows addObject:@{@"model":fsr411::ns(list[i].name),@"byte_identical":@YES,@"verified_frame_pairs":@(s.verified),
            @"neural_pairs":@(s.neuralCount),@"reused_pairs":@(s.reuseCount),@"mpp_features":@(s.view[1].mppFeatures),@"register_only_passes":@(s.view[1].mppFastPasses),
            @"neural_before_ms":array(s.neural[0]),@"neural_after_ms":array(s.neural[1]),@"full_before_ms":array(s.full[0]),@"full_after_ms":array(s.full[1]),
            @"reset_before_ms":array(s.reset[0]),@"reset_after_ms":array(s.reset[1]),@"reused_before_ms":array(s.reused[0]),@"reused_after_ms":array(s.reused[1]),
            @"neural_before_median_ms":metal4?[NSNull null]:@(median(s.neural[0])),@"neural_after_median_ms":metal4?[NSNull null]:@(median(s.neural[1])),
            @"full_before_median_ms":metal4?[NSNull null]:@(median(s.full[0])),@"full_after_median_ms":metal4?[NSNull null]:@(median(s.full[1])),
            @"neural_paired_change_percent":s.ratios.empty()?[NSNull null]:@((median(s.ratios)-1)*100),@"full_paired_change_percent":s.fullRatios.empty()?[NSNull null]:@((median(s.fullRatios)-1)*100),
            @"reuse_paired_change_percent":s.reuseRatios.empty()?[NSNull null]:@((median(s.reuseRatios)-1)*100)}];
    }
    NSDictionary *report=@{@"device":dev.name,@"scope":metal4?@"Metal 4 byte-identity validation only (no timing)":@"full frame command buffer, M5 INT8, serial interleaved A/B in one process",
        @"root":fsr411::ns(root),@"before_root":fsr411::ns(beforeRoot),@"render":@[@(rw),@(rh)],@"output":@[@(uw),@(uh)],@"rounds":@(rounds),@"warmup_cycles":@(warmups),@"frames_per_cycle":@(period),
        @"verify_every_frame":@(verifyEvery),@"production_timing":@(!metal4 && !verifyEvery),@"low_power_mode":@(NSProcessInfo.processInfo.lowPowerModeEnabled),
        @"thermal_state_start":@(thermalStart),@"thermal_state_end":@(NSProcessInfo.processInfo.thermalState),@"models":rows};
    NSError *e=nil;auto data=[NSJSONSerialization dataWithJSONObject:report options:NSJSONWritingPrettyPrinted error:&e];require(data!=nil,fsr411::metal_error(e));
    require([data writeToFile:fsr411::ns(reportPath) options:NSDataWritingAtomic error:&e],fsr411::metal_error(e));std::printf("report: %s\n",reportPath.c_str());return 0;
} catch(const std::exception &e){std::fprintf(stderr,"fsr411_mpp_ab: %s\n",e.what());return 1;}}}
