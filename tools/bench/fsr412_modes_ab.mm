// SPDX-License-Identifier: MIT
// Real GPU parity/cadence tests and interleaved mode timing. No game/DLL execution.
#include "../../src/fsr412_runtime.h"
#include <numeric>
#include <stdexcept>

namespace {
void require(bool value,const std::string &reason) {if(!value)throw std::runtime_error(reason);}
id<MTLTexture> texture(id<MTLDevice> device,MTLPixelFormat format,unsigned w,unsigned h) {
    auto d=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format width:w height:h mipmapped:NO];
    d.storageMode=MTLStorageModeShared;d.usage=MTLTextureUsageShaderRead|MTLTextureUsageShaderWrite;
    auto t=[device newTextureWithDescriptor:d];require(t!=nil,"texture allocation failed");return t;
}
std::vector<uint8_t> bytes(id<MTLCommandQueue> queue,id<MTLTexture> t,unsigned bpp) {
    const NSUInteger pitch=(t.width*bpp+255u)&~NSUInteger(255);
    auto buffer=[t.device newBufferWithLength:pitch*t.height options:MTLResourceStorageModeShared];
    auto cb=[queue commandBuffer];auto enc=[cb blitCommandEncoder];
    [enc copyFromTexture:t sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0,0,0)
        sourceSize:MTLSizeMake(t.width,t.height,1) toBuffer:buffer destinationOffset:0
        destinationBytesPerRow:pitch destinationBytesPerImage:pitch*t.height];
    [enc endEncoding];[cb commit];[cb waitUntilCompleted];
    require(cb.status==MTLCommandBufferStatusCompleted,fsr411::metal_error(cb.error));
    std::vector<uint8_t> result(t.width*t.height*bpp);
    for(NSUInteger y=0;y<t.height;++y)std::memcpy(result.data()+y*t.width*bpp,(const uint8_t *)buffer.contents+y*pitch,t.width*bpp);
    if(bpp==8)for(size_t i=0;i<result.size();i+=2) {
        _Float16 v;std::memcpy(&v,result.data()+i,2);require(std::isfinite(float(v)),"nonfinite output/history");
    }
    return result;
}
struct Mode {
    std::string name;
    fsr411::Options options;
    fsr411::View view;
    id<MTLTexture> output=nil;
    std::vector<double> times,full,reused,partial;
};
double percentile(std::vector<double> values,double p) {
    if(values.empty())return 0;
    std::sort(values.begin(),values.end());return values[size_t((values.size()-1)*p)];
}
NSArray *numbers(const std::vector<double> &values) {auto a=[NSMutableArray array];for(double value:values)[a addObject:@(value)];return a;}
}
int main(int argc,char **argv) {@autoreleasepool {try {
    require(argc>=2,"usage: fsr412_modes_ab ROOT [--bench N] [--size RW RH UW UH] [--mtl4] [--legacy] [--ultra] [--report FILE]");
    std::string root=argv[1],report="build/lite-lossy-review/modes-ab.json";
    unsigned rounds=2,rw=128,rh=80,uw=256,uh=160;bool benchmark=false,metal4=false,legacy=false,ultra=false,original=false,autoExposure=false;
    for(int i=2;i<argc;++i) {
        const std::string a=argv[i];
        if(a=="--bench" && i+1<argc) {benchmark=true;rounds=std::stoul(argv[++i]);}
        else if(a=="--size" && i+4<argc) {rw=std::stoul(argv[++i]);rh=std::stoul(argv[++i]);uw=std::stoul(argv[++i]);uh=std::stoul(argv[++i]);}
        else if(a=="--report" && i+1<argc) report=argv[++i];
        else if(a=="--mtl4")metal4=true;
        else if(a=="--legacy")legacy=true;
        else if(a=="--ultra")ultra=true;
        else if(a=="--original")original=true;
        else if(a=="--auto-exposure")autoExposure=true;
        else throw std::runtime_error("unknown option: "+a);
    }
    require(rounds && rounds<=1000 && rw && rh && uw>=4 && uh>=4 && !(uw&3u) && !(uh&3u)
        && rw<=3840 && rh<=2160 && uw<=3840 && uh<=2160,"invalid rounds or dimensions");
    require(!metal4 || !benchmark,"Metal 4 checks have no classic command-buffer GPU timestamps");
    auto device=MTLCreateSystemDefaultDevice();auto queue=[device newCommandQueue];
    require(device && queue,"no Metal device");
    fsr411::Runtime runtime(device,root,root+"/build");
    id<MTL4CommandQueue> queue4=nil;id<MTLSharedEvent> event=nil;uint64_t eventValue=0;
    if(metal4) {if(@available(macOS 26.0,*)){queue4=[device newMTL4CommandQueue];event=[device newSharedEvent];}
        require(queue4 && event,"Metal 4 unavailable");}
    std::vector<std::unique_ptr<Mode>> modes;
    auto add=[&](const std::string &name,fsr411::Profile profile,bool lite,unsigned tile=0,int quality=0,fsr411::Backend backend=fsr411::Backend::Auto) {
        auto m=std::make_unique<Mode>();m->name=name;m->options.profile=profile;m->options.modelLite=lite;
        m->options.backend=backend;m->options.neuralTileSize=tile;m->options.temporalQuality=quality;
        m->options.ultraPerformance=ultra;
        m->output=texture(device,MTLPixelFormatRGBA16Float,uw,uh);modes.push_back(std::move(m));
    };
    if(benchmark) {
        add("lite",fsr411::Profile::Int8Lite,true);
        add("lite-lossy",fsr411::Profile::Int8LiteLossy,true);
        add("lite-lossy-tiled",fsr411::Profile::Int8LiteLossy,true,512);
        if(original) {add("lite-lossy-original",fsr411::Profile::Int8LiteLossy,true);modes.back()->options.originalLossy=true;}
        if(legacy) {
            add("lossy2",fsr411::Profile::Lossy2,false);
            add("lossy2-quality",fsr411::Profile::Lossy2,false,0,1);
            add("lossy2-quality-fast",fsr411::Profile::Lossy2,false,0,2);
        }
    } else {
        for(unsigned tile:{0u,128u})for(auto backend:{fsr411::Backend::Auto,fsr411::Backend::Scalar,fsr411::Backend::Simd})
            add(std::string(tile?"tiled-":"whole-")+std::to_string(int(backend)),fsr411::Profile::Int8LiteLossy,true,tile,0,backend);
        if(legacy)for(auto backend:{fsr411::Backend::Auto,fsr411::Backend::Scalar})
            add("legacy-quality-"+std::to_string(int(backend)),fsr411::Profile::Lossy2,false,0,2,backend);
    }
    auto color=texture(device,MTLPixelFormatRGBA16Float,rw,rh);
    auto depth=texture(device,MTLPixelFormatR32Float,rw,rh);
    auto motion=texture(device,MTLPixelFormatRG16Float,rw,rh);
    std::vector<_Float16> colors(size_t(rw)*rh*4),vectors(size_t(rw)*rh*2);
    std::vector<float> depths(size_t(rw)*rh,.6f);
    for(unsigned y=0;y<rh;++y)for(unsigned x=0;x<rw;++x) {
        size_t i=size_t(y)*rw+x;float value=.15f+.65f*float(x+2*y)/float(rw+2*rh);
        colors[i*4]=(_Float16)value;colors[i*4+1]=(_Float16)(value*.7f);colors[i*4+2]=(_Float16)(value*.4f);colors[i*4+3]=1;
        vectors[i*2]=(_Float16)(x<rw/2?.25f:-.25f);vectors[i*2+1]=0;
    }
    [color replaceRegion:MTLRegionMake2D(0,0,rw,rh) mipmapLevel:0 withBytes:colors.data() bytesPerRow:rw*8];
    [depth replaceRegion:MTLRegionMake2D(0,0,rw,rh) mipmapLevel:0 withBytes:depths.data() bytesPerRow:rw*4];
    [motion replaceRegion:MTLRegionMake2D(0,0,rw,rh) mipmapLevel:0 withBytes:vectors.data() bytesPerRow:rw*4];
    unsigned verified=0;constexpr unsigned frames=8,warmups=2;
    for(unsigned round=0;round<rounds+(benchmark?warmups:0);++round)for(unsigned index=0;index<frames;++index) {
        for(size_t order=0;order<modes.size();++order) { @autoreleasepool {
            const size_t position=(round+index)&1u?modes.size()-order-1:order;auto &m=*modes[position];
            fsr411::Frame f;f.color=color;f.depth=depth;f.motion=motion;f.output=m.output;
            auto &p=f.input;p.rw=rw;p.rh=rh;p.uw=uw;p.uh=uh;p.preExposure=index<4?1.f:1.4f;
            p.geometry.color=p.geometry.depth=p.geometry.motion={rw,rh,0,0};p.geometry.output={uw,uh,0,0};
            p.reset=index==0 || (!benchmark && index==3);p.depthReversed=1;p.mvScaleX=p.mvScaleY=1;
            p.autoExposure=autoExposure?1:0;
            p.jitterX=index&1u?-.25f:.125f;p.jitterY=index%3?1.f/6.f:-1.f/6.f;
            std::string error;double milliseconds=0;
            if(metal4) {
                if(@available(macOS 26.0,*)) {
                    auto allocator=[device newCommandAllocator];auto cb=[device newCommandBuffer];[cb beginCommandBufferWithAllocator:allocator];
                    bool encoded=runtime.encode(cb,m.view,f,m.options,error);
                    require(encoded,m.name+": "+error);[cb endCommandBuffer];
                    id<MTL4CommandBuffer> list[]={cb};[queue4 commit:list count:1];[queue4 signalEvent:event value:++eventValue];
                    require([event waitUntilSignaledValue:eventValue timeoutMS:20000],"Metal 4 completion timeout");
                }
            } else {
                auto cb=[queue commandBuffer];bool encoded=runtime.encode(cb,m.view,f,m.options,error);
                require(encoded,m.name+": "+error);
                [cb commit];[cb waitUntilCompleted];require(cb.status==MTLCommandBufferStatusCompleted,fsr411::metal_error(cb.error));
                milliseconds=(cb.GPUEndTime-cb.GPUStartTime)*1000.;
            }
            m.options.allowCompile=false;
            require(m.view.neuralRuns+m.view.reusedFrames+m.view.partialFrames==m.view.encodes,"frame accounting mismatch");
            if(index==0 || index==4 || (!benchmark && index==3))require(!m.view.skippedInference && !m.view.partialInference,"reset/exposure did not refresh inference");
            if(m.options.modelLite && m.options.profile==fsr411::Profile::Int8LiteLossy)
                require(m.options.neuralTileSize ? m.view.litePrevious!=m.view.liteCurrent : m.view.litePrevious==m.view.liteCurrent,"raw cache alias contract violated");
            if(benchmark && round>=warmups) {
                m.times.push_back(milliseconds);(m.view.partialInference?m.partial:m.view.skippedInference?m.reused:m.full).push_back(milliseconds);
            }
        }}
        if(!benchmark) {
            for(size_t group=0;group+2<modes.size() && group<6;group+=3) {
                const auto output=bytes(queue,modes[group]->output,8),history=bytes(queue,modes[group]->view.history,8),recurrent=bytes(queue,modes[group]->view.recurrent,4);
                for(size_t k=1;k<3;++k) {
                    require(output==bytes(queue,modes[group+k]->output,8),"native/scalar/SIMD output mismatch: "+modes[group]->name);
                    require(history==bytes(queue,modes[group+k]->view.history,8),"native/scalar/SIMD history mismatch");
                    require(recurrent==bytes(queue,modes[group+k]->view.recurrent,4),"native/scalar/SIMD recurrent mismatch");
                    require(modes[group]->view.skippedInference==modes[group+k]->view.skippedInference && modes[group]->view.partialInference==modes[group+k]->view.partialInference,"backend cadence mismatch");
                }
                ++verified;
            }
            if(legacy) {
                require(bytes(queue,modes[6]->output,8)==bytes(queue,modes[7]->output,8),"legacy quality native/scalar mismatch");
                ++verified;
            }
        }
    }
    if(benchmark) {
        auto rows=[NSMutableArray array];
        for(const auto &mode:modes) {
            const auto &m=*mode;double mean=std::accumulate(m.times.begin(),m.times.end(),0.)/m.times.size();
            [rows addObject:@{@"mode":fsr411::ns(m.name),@"mean_ms":@(mean),@"median_ms":@(percentile(m.times,.5)),
                @"p99_ms":@(percentile(m.times,.99)),@"max_ms":@(percentile(m.times,1)),@"full":numbers(m.full),
                @"reused":numbers(m.reused),@"partial":numbers(m.partial),@"times":numbers(m.times)}];
            std::printf("%s: median %.3f ms, mean %.3f ms, P99 %.3f ms\n",m.name.c_str(),percentile(m.times,.5),mean,percentile(m.times,.99));
        }
        auto data=[NSJSONSerialization dataWithJSONObject:@{@"device":device.name,@"rounds":@(rounds),@"warmup_cycles":@(warmups),
            @"render":@[@(rw),@(rh)],@"output":@[@(uw),@(uh)],@"interleaved":@YES,@"diagnostic_copies":@NO,@"rows":rows} options:NSJSONWritingPrettyPrinted error:nil];
        require([data writeToFile:fsr411::ns(report) atomically:YES],"cannot write benchmark report");
    } else std::printf("Lite Lossy %s: %u backend/state parity groups, resets and cache ownership passed\n",metal4?"Metal 4":"classic",verified);
    return 0;
} catch(const std::exception &error) {std::fprintf(stderr,"Lite mode validation: %s\n",error.what());return 1;} }}
