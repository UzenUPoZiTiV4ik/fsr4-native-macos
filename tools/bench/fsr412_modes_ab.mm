// SPDX-License-Identifier: MIT
// Real GPU parity/cadence tests and interleaved mode timing. No game/DLL execution.
#include "../../src/fsr412_runtime.h"
#include "bench_gpu.h"
#include <numeric>
#include <stdexcept>

namespace {
using namespace bench;
struct Mode {
    std::string name;
    fsr411::Options options;
    fsr411::View view;
    id<MTLTexture> output=nil;
    std::vector<double> times,full,reused,partial;
};
}
int main(int argc,char **argv) {@autoreleasepool {try {
    require(argc>=2,"usage: fsr412_modes_ab ROOT [--bench N] [--size RW RH UW UH] [--mtl4] [--legacy] [--ultra] [--report FILE]");
    std::string root=argv[1],report="build/lite-lossy-review/modes-ab.json";
    unsigned rounds=2,rw=128,rh=80,uw=256,uh=160;bool benchmark=false,metal4=false,legacy=false,ultra=false,original=false,autoExposure=false,newLook=false;
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
        else if(a=="--new-look")newLook=true;
        else throw std::runtime_error("unknown option: "+a);
    }
    require(rounds && rounds<=1000 && rw && rh && uw>=4 && uh>=4 && !(uw&3u) && !(uh&3u)
        && rw<=3840 && rh<=2160 && uw<=3840 && uh<=2160,"invalid rounds or dimensions");
    require(!metal4 || !benchmark,"Metal 4 checks have no classic command-buffer GPU timestamps");
    id<MTLDevice> device;auto queue=device_queue(device,MTLGPUFamily(0),"no Metal device");
    fsr411::Runtime runtime(device,root,root+"/build");
    Mtl4 m4;if(metal4)mtl4_open(m4,device);
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
        if(newLook) {add("new-look",fsr411::Profile::Int8LiteLossy,true);modes.back()->options.newLook=true;}
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
        if(newLook)for(auto backend:{fsr411::Backend::Auto,fsr411::Backend::Scalar,fsr411::Backend::Simd}) {
            add("new-look-"+std::to_string(int(backend)),fsr411::Profile::Int8LiteLossy,true,0,0,backend);
            modes.back()->options.newLook=true;
        }
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
            const bool lookMode=m.name.rfind("new-look",0)==0;
            if(!benchmark && lookMode) {m.options.newLook=index!=5;m.options.allowCompile=true;}
            fsr411::Frame f;f.color=color;f.depth=depth;f.motion=motion;f.output=m.output;
            auto &p=f.input;p.rw=rw;p.rh=rh;p.uw=uw;p.uh=uh;p.preExposure=index<4?1.f:1.4f;
            p.geometry.color=p.geometry.depth=p.geometry.motion={rw,rh,0,0};p.geometry.output={uw,uh,0,0};
            p.reset=index==0 || (!benchmark && index==3);p.depthReversed=1;p.mvScaleX=p.mvScaleY=1;
            p.autoExposure=autoExposure?1:0;
            jitter(index,p.jitterX,p.jitterY);
            std::string error;double milliseconds=0;
            if(metal4) {
                require(mtl4_encode(m4,device,[&](id<MTL4CommandBuffer> cb){return runtime.encode(cb,m.view,f,m.options,error);}),m.name+": "+error);
            } else {
                auto cb=[queue commandBuffer];
                require(runtime.encode(cb,m.view,f,m.options,error),m.name+": "+error);
                milliseconds=commit_timed(cb);
            }
            m.options.allowCompile=false;
            require(m.view.neuralRuns+m.view.reusedFrames+m.view.partialFrames==m.view.encodes,"frame accounting mismatch");
            if(index==0 || index==4 || (!benchmark && (index==3 || (lookMode && (index==5 || index==6)))))
                require(!m.view.skippedInference && !m.view.partialInference,"reset/exposure/new-look switch did not refresh inference");
            if(lookMode)require((m.view.variantKey.find("_nl"+std::to_string(fsr411::kLiteNewLookRevision))!=std::string::npos)==m.options.newLook,"new-look cache identity mismatch");
            if(m.options.modelLite && m.options.profile==fsr411::Profile::Int8LiteLossy)
                require(m.options.neuralTileSize ? m.view.litePrevious!=m.view.liteCurrent : m.view.litePrevious==m.view.liteCurrent,"raw cache alias contract violated");
            if(benchmark && round>=warmups) {
                m.times.push_back(milliseconds);(m.view.partialInference?m.partial:m.view.skippedInference?m.reused:m.full).push_back(milliseconds);
            }
        }}
        if(!benchmark) {
            const size_t groupedModes=newLook?9u:6u;
            for(size_t group=0;group+2<modes.size() && group<groupedModes;group+=3) {
                const auto output=readback(queue,modes[group]->output,8,true),history=readback(queue,modes[group]->view.history,8,true),recurrent=readback(queue,modes[group]->view.recurrent,4,true);
                for(size_t k=1;k<3;++k) {
                    require(output==readback(queue,modes[group+k]->output,8,true),"native/scalar/SIMD output mismatch: "+modes[group]->name);
                    require(history==readback(queue,modes[group+k]->view.history,8,true),"native/scalar/SIMD history mismatch");
                    require(recurrent==readback(queue,modes[group+k]->view.recurrent,4,true),"native/scalar/SIMD recurrent mismatch");
                    require(modes[group]->view.skippedInference==modes[group+k]->view.skippedInference && modes[group]->view.partialInference==modes[group+k]->view.partialInference,"backend cadence mismatch");
                }
                ++verified;
            }
            if(legacy) {
                require(readback(queue,modes[groupedModes]->output,8,true)==readback(queue,modes[groupedModes+1]->output,8,true),"legacy quality native/scalar mismatch");
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
