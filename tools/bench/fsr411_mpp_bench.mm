// SPDX-License-Identifier: MIT
// Apple10 INT8 MPP A/B on identical weights and independent scratch buffers.
// Isolated neural passes, NOT full-frame or in-game latency. A uses the saved
// runtime's dispatch/pipeline contract; B can test an exact-safe new kernel.
// usage: fsr411_mpp_bench ROOT NATIVE_DIR [--compare-root ROOT] [--rounds N]
//        [--size UW UH] [--kernel KIND NAME TW TH THREADS] [--uncapped] [--legacy-kernels]
// KIND: cnb, fnb32, fnb64, ct32, ct64, ds16, ds32.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "../../src/fsr411_model.h"
#include "../../src/fsr411_mpp_policy.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void require(bool ok, const std::string &message) { if (!ok) throw std::runtime_error(message); }
NSString *ns(const std::string &s) { return [NSString stringWithUTF8String:s.c_str()]; }
std::string metal_error(NSError *e) { return e ? e.localizedDescription.UTF8String : "Metal allocation failed"; }
double median(std::vector<double> v) {
    require(!v.empty(), "empty timing sample"); std::sort(v.begin(),v.end());
    const size_t n=v.size(); return n%2 ? v[n/2] : (v[n/2-1]+v[n/2])*.5;
}
NSArray *numbers(const std::vector<double> &v) { auto a=[NSMutableArray new];for(double x:v)[a addObject:@(x)];return a; }
unsigned tensor_end(const fsr411::Tensor &t) {
    return t.channels ? fsr411::tensor_address(t,t.width+1,t.height+1,t.channels-1)+64u : 0u;
}
id<MTLLibrary> library(id<MTLDevice> dev,const std::string &root,unsigned tier) {
    NSError *e=nil;
    auto source=[NSString stringWithContentsOfFile:ns(root+"/src/fsr411_mpp.metal") encoding:NSUTF8StringEncoding error:&e];
    auto header=[NSString stringWithContentsOfFile:ns(root+"/src/fsr411_model.h") encoding:NSUTF8StringEncoding error:&e];
    require(source && header,"missing MPP source/header: "+root);
    source=[source stringByReplacingOccurrencesOfString:@"#include \"fsr411_model.h\"" withString:header];
    auto o=[MTLCompileOptions new];o.languageVersion=MTLLanguageVersion4_0;
    o.preprocessorMacros=@{@"FSR411_SPECIALIZE":@1,@"FSR411_MAXW":@(tier*8u/9u),@"FSR411_MAXH":@(tier/2u)};
    auto lib=[dev newLibraryWithSource:source options:o error:&e];require(lib!=nil,metal_error(e));return lib;
}
struct Shape { std::string name;unsigned x,y,threads; };
id<MTLComputePipelineState> pipeline(id<MTLDevice> dev,id<MTLLibrary> lib,const Shape &s,
                                     MTLFunctionConstantValues *values,bool capped) {
    NSError *e=nil;auto fn=[lib newFunctionWithName:ns(s.name) constantValues:values error:&e];
    require(fn!=nil,metal_error(e));id<MTLComputePipelineState> p=nil;
    if(capped) {
        auto d=[MTLComputePipelineDescriptor new];d.computeFunction=fn;
        d.maxTotalThreadsPerThreadgroup=s.threads;d.threadGroupSizeIsMultipleOfThreadExecutionWidth=s.threads%32u==0;
        p=[dev newComputePipelineStateWithDescriptor:d options:MTLPipelineOptionNone reflection:nil error:&e];
    } else p=[dev newComputePipelineStateWithFunction:fn error:&e];
    require(p!=nil,metal_error(e));
    require(p.threadExecutionWidth==32 && p.maxTotalThreadsPerThreadgroup>=s.threads,"invalid SIMD32 pipeline");
    require(p.staticThreadgroupMemoryLength<=dev.maxThreadgroupMemoryLength,"threadgroup storage exceeds device limit");return p;
}
uint32_t probe(id<MTLDevice> dev,id<MTLLibrary> lib,id<MTLCommandQueue> queue) {
    auto fn=[lib newFunctionWithName:@"fsr411_mpp_chain_probe"];if(!fn)return 0;
    NSError *e=nil;auto d=[MTLComputePipelineDescriptor new];d.computeFunction=fn;
    d.maxTotalThreadsPerThreadgroup=32;d.threadGroupSizeIsMultipleOfThreadExecutionWidth=YES;
    auto p=[dev newComputePipelineStateWithDescriptor:d options:MTLPipelineOptionNone reflection:nil error:&e];
    require(p && p.threadExecutionWidth==32 && !p.staticThreadgroupMemoryLength,"MPP probe pipeline invalid");
    std::vector<uint8_t> zeros(fsr411::kMppProbeDummyBytes,0);const uint32_t sentinel=~0u;
    auto b=[dev newBufferWithBytes:&sentinel length:sizeof(sentinel) options:MTLResourceStorageModeShared];
    auto dummy=[dev newBufferWithBytes:zeros.data() length:zeros.size() options:MTLResourceStorageModeShared];
    require(b && dummy,"MPP probe allocation failed");auto cb=[queue commandBuffer];auto enc=[cb computeCommandEncoder];
    [enc setComputePipelineState:p];[enc setBuffer:b offset:0 atIndex:0];[enc setBuffer:dummy offset:0 atIndex:1];
    [enc dispatchThreadgroups:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
    [enc endEncoding];[cb commit];[cb waitUntilCompleted];require(cb.status==MTLCommandBufferStatusCompleted,metal_error(cb.error));
    const uint32_t mask=*(const uint32_t *)b.contents;require(!(mask & ~fsr411::kMppChainKnownMask),"MPP probe returned unknown features");return mask;
}
id<MTLBuffer> output_copy(id<MTLDevice> dev,id<MTLCommandQueue> q,id<MTLBuffer> scratch,const fsr411::Tensor &t) {
    auto result=[dev newBufferWithLength:size_t(t.width)*t.height*t.channels options:MTLResourceStorageModeShared];
    require(result!=nil,"output readback allocation failed");auto cb=[q commandBuffer];auto b=[cb blitCommandEncoder];
    const unsigned banks=t.bankStride ? t.channels/16u : 1u;
    const unsigned rowBytes=t.width*t.strideX;size_t offset=0;
    require(size_t(rowBytes)*t.height*banks==result.length,"unsupported output layout");
    for(unsigned bank=0;bank<banks;++bank)for(unsigned y=0;y<t.height;++y) {
        [b copyFromBuffer:scratch sourceOffset:fsr411::tensor_address(t,0,y,bank*16u)
            toBuffer:result destinationOffset:offset size:rowBytes];offset+=rowBytes;
    }
    [b endEncoding];[cb commit];[cb waitUntilCompleted];
    require(cb.status==MTLCommandBufferStatusCompleted,metal_error(cb.error));return result;
}
}
int main(int argc,char **argv) { @autoreleasepool { try {
    require(argc>=3,"usage: fsr411_mpp_bench ROOT NATIVE_DIR [--compare-root ROOT] [--rounds N] [--size UW UH] [--kernel KIND NAME TW TH THREADS] [--uncapped]");
    const std::string root=argv[1],dir=argv[2];std::string beforeRoot=root;
    unsigned rounds=20,uw=0,uh=0;bool capped=true,legacyKernels=false;std::map<std::string,Shape> overrides;
    for(int i=3;i<argc;++i) {
        const std::string a=argv[i];
        if(a=="--uncapped") capped=false;
        else if(a=="--legacy-kernels") legacyKernels=true;
        else if(a=="--compare-root" && i+1<argc) beforeRoot=argv[++i];
        else if(a=="--rounds" && i+1<argc) rounds=std::stoul(argv[++i]);
        else if(a=="--size" && i+2<argc) {uw=std::stoul(argv[++i]);uh=std::stoul(argv[++i]);}
        else if(a=="--kernel" && i+5<argc) {
            const std::string kind=argv[++i],name=argv[++i];
            const unsigned x=std::stoul(argv[++i]),y=std::stoul(argv[++i]),threads=std::stoul(argv[++i]);
            require(x && y && threads && threads<=1024 && threads%32==0,"invalid kernel shape");overrides[kind]={name,x,y,threads};
        } else throw std::runtime_error("unknown or incomplete option: "+a);
    }
    require(rounds>=2 && rounds<=1000,"rounds must be 2..1000");
    auto dev=MTLCreateSystemDefaultDevice();auto q=[dev newCommandQueue];
    require(dev && q && [dev supportsFamily:MTLGPUFamilyApple10],"Apple10/M5 and Metal 4 required");
    const auto thermalStart=NSProcessInfo.processInfo.thermalState;NSError *error=nil;
    auto data=[NSData dataWithContentsOfFile:ns(dir+"/fsr411_native_manifest.json")];
    NSDictionary *manifest=data?[NSJSONSerialization JSONObjectWithData:data options:0 error:&error]:nil;
    auto weights=[NSData dataWithContentsOfFile:ns(dir+"/fsr411_native_weights.bin")];
    auto params=[NSData dataWithContentsOfFile:ns(dir+"/fsr411_native_params.bin")];
    require(manifest && weights.length && params.length,"missing native model files");
    require([manifest[@"parameter_size"] unsignedIntValue]==sizeof(fsr411::PassParams),"native parameter ABI mismatch");
    require(weights.length==[manifest[@"weights_bytes"] unsignedLongLongValue],"incomplete weights");
    const unsigned tier=[manifest[@"tier"] unsignedIntValue];const bool lite=[manifest[@"fsr_version"] isEqual:@"4.1.2"];
    if(uw || uh)require(uw>=32 && uh>=8 && uw<=tier*16u/9u && uh<=tier,"active dimensions exceed model tier");
    auto wb=[dev newBufferWithBytes:weights.bytes length:weights.length options:MTLResourceStorageModeShared];
    auto currentLib=library(dev,root,tier),beforeLib=beforeRoot==root?currentLib:library(dev,beforeRoot,tier);
    const uint32_t features=legacyKernels?0:probe(dev,currentLib,q);
    NSMutableDictionary *passes=[manifest[@"passes"] mutableCopy];passes[@"post_cnb"]=manifest[@"auxiliary_passes"][@"post_cnb"];
    unsigned scratchBytes=0;
    for(NSString *key in passes) {
        fsr411::PassParams p;const size_t offset=[passes[key][@"parameter_offset"] unsignedIntValue];
        require(offset<=params.length && sizeof(p)<=params.length-offset,"parameter region overflow");
        memcpy(&p,(const uint8_t *)params.bytes+offset,sizeof(p));
        scratchBytes=std::max({scratchBytes,tensor_end(p.input),tensor_end(p.output),tensor_end(p.skip)});
    }
    const unsigned postOffset=(scratchBytes+255u)&~255u;
    scratchBytes=postOffset+[passes[@"post_cnb"][@"scratch_append_bytes"] unsignedIntValue]+256u;
    id<MTLBuffer> scratch[2]={[dev newBufferWithLength:scratchBytes options:MTLResourceStorageModePrivate],
                             [dev newBufferWithLength:scratchBytes options:MTLResourceStorageModePrivate]};
    require(wb && scratch[0] && scratch[1],"scratch allocation failed");
    auto clear=[q commandBuffer];auto blit=[clear blitCommandEncoder];
    for(auto b:scratch)[blit fillBuffer:b range:NSMakeRange(0,scratchBytes) value:0x31];
    [blit endEncoding];[clear commit];[clear waitUntilCompleted];require(clear.status==MTLCommandBufferStatusCompleted,"scratch initialization failed");
    auto results=[NSMutableArray new];constexpr unsigned batch=8,warmups=4;
    for(NSString *key in @[@"1",@"2",@"3",@"4",@"5",@"6",@"7",@"8",@"9",@"10",@"11",@"12",@"post_cnb"]) {
        NSDictionary *entry=passes[key];if(!entry)continue;fsr411::PassParams p;
        memcpy(&p,(const uint8_t *)params.bytes+[entry[@"parameter_offset"] unsignedIntValue],sizeof(p));
        const bool post=[key isEqual:@"post_cnb"];
        const auto metadata=fsr411::specialize_metadata(p,post);auto values=[MTLFunctionConstantValues new];
        [values setConstantValues:metadata.values type:MTLDataTypeUInt4 withRange:NSMakeRange(0,fsr411::kPassSpecializationCount)];
        Shape shape[2]={{[entry[@"kernel"] UTF8String],[entry[@"tile"][0] unsignedIntValue],[entry[@"tile"][1] unsignedIntValue],[entry[@"threads"] unsignedIntValue]}, {}};
        shape[1]=shape[0];
        const auto tuned=fsr411::mpp_fast_dispatch(shape[0].name.c_str(),shape[0].x,shape[0].y,shape[0].threads,features);
        if(tuned.feature)shape[1]={tuned.function,tuned.tileX,tuned.tileY,tuned.threads};
        const std::string kind=std::string([entry[@"kind"] UTF8String])+(p.input.channels==16 && [entry[@"kind"] isEqual:@"cnb"]?"":std::to_string(p.input.channels));
        if(overrides.count(kind))shape[1]=overrides.at(kind);
        // Match production A: Lite was capped, full Exact/Lossy were not.
        id<MTLComputePipelineState> state[2]={pipeline(dev,beforeLib,shape[0],values,lite),pipeline(dev,currentLib,shape[1],values,capped)};
        if(post)p.output.offset+=postOffset;
        if(uw) {
            constexpr unsigned fullShifts[17]={1,0,1,1,2,2,2,3,3,3,2,2,1,1,0,0,0},liteShifts[9]={1,0,1,2,2,1,0,0,0};
            auto size=[&](fsr411::Tensor &t,NSString *slot) { if(!slot)return;const unsigned i=slot.intValue;
                require(i<(lite?9u:17u),"invalid tensor slot");const unsigned shift=lite?liteShifts[i]:fullShifts[i];t.width=uw>>shift;t.height=uh>>shift; };
            size(p.input,entry[@"input_slot"]);size(p.output,entry[@"output_slot"]);size(p.skip,entry[@"skip_slot"]);
        }
        const auto &space=[entry[@"dispatch_space"] isEqual:@"input"]?p.input:p.output;
        std::vector<double> times[2];
        for(unsigned round=0;round<rounds+warmups;++round)for(unsigned j=0;j<2;++j) {
            const unsigned v=(round&1u)^j;auto cb=[q commandBuffer];auto enc=[cb computeCommandEncoder];
            [enc setComputePipelineState:state[v]];[enc setBuffer:scratch[v] offset:0 atIndex:0];[enc setBuffer:wb offset:0 atIndex:1];[enc setBytes:&p length:sizeof(p) atIndex:2];
            for(unsigned k=0;k<batch;++k)[enc dispatchThreadgroups:MTLSizeMake((space.width+shape[v].x-1)/shape[v].x,(space.height+shape[v].y-1)/shape[v].y,1)
                threadsPerThreadgroup:MTLSizeMake(shape[v].threads,1,1)];
            [enc endEncoding];[cb commit];[cb waitUntilCompleted];require(cb.status==MTLCommandBufferStatusCompleted,metal_error(cb.error));
            const double t=(cb.GPUEndTime-cb.GPUStartTime)*1000./batch;require(std::isfinite(t) && t>0,"invalid GPU timestamp");
            if(round>=warmups)times[v].push_back(t);
        }
        const auto a=output_copy(dev,q,scratch[0],p.output),b=output_copy(dev,q,scratch[1],p.output);
        const bool identical=a.length==b.length && !memcmp(a.contents,b.contents,a.length);
        require(identical,"A/B output differs at native pass "+std::string(key.UTF8String));
        std::vector<double> ratios;for(size_t i=0;i<times[0].size();++i)ratios.push_back(times[1][i]/times[0][i]);
        [results addObject:@{@"pass":key,@"kind":ns(kind),@"before_kernel":ns(shape[0].name),@"kernel":ns(shape[1].name),
            @"before_ms":@(median(times[0])),@"after_ms":@(median(times[1])),@"paired_change_percent":@((median(ratios)-1)*100),
            @"before_samples_ms":numbers(times[0]),@"after_samples_ms":numbers(times[1]),@"byte_identical":@(identical),
            @"before_threadgroup_bytes":@(state[0].staticThreadgroupMemoryLength),@"threadgroup_bytes":@(state[1].staticThreadgroupMemoryLength),
            @"before_pipeline_max_threads":@(state[0].maxTotalThreadsPerThreadgroup),@"pipeline_max_threads":@(state[1].maxTotalThreadsPerThreadgroup),
            @"tile":@[@(shape[1].x),@(shape[1].y)],@"threads":@(shape[1].threads)}];
    }
    NSDictionary *report=@{@"device":dev.name,@"scope":@"isolated native INT8 MPP passes; 8 dispatches; interleaved A/B; not full-frame latency",
        @"root":ns(root),@"compare_root":ns(beforeRoot),@"model":manifest[@"fsr_version"]?:@"4.1.1",@"profile":manifest[@"profile"],@"tier":@(tier),
        @"weights_sha256":manifest[@"weights_sha256"]?:[NSNull null],@"params_sha256":manifest[@"params_sha256"]?:[NSNull null],
        @"rounds":@(rounds),@"warmup_pairs":@(warmups),@"mpp_probe_features":@(features),@"legacy_candidate":@(legacyKernels),@"low_power_mode":@(NSProcessInfo.processInfo.lowPowerModeEnabled),
        @"thermal_state_start":@(thermalStart),@"thermal_state_end":@(NSProcessInfo.processInfo.thermalState),@"passes":results};
    auto json=[NSJSONSerialization dataWithJSONObject:report options:NSJSONWritingPrettyPrinted error:&error];require(json!=nil,metal_error(error));
    fwrite(json.bytes,1,json.length,stdout);puts("");return 0;
} catch(const std::exception &e) {fprintf(stderr,"MPP benchmark: %s\n",e.what());return 1;} } }
