// SPDX-License-Identifier: MIT
// Standalone exact DS16 candidate/oracle and interleaved M5 A/B bench.
// Default mode is CPU-only. GPU dispatch requires explicit --correctness or
// --timing-lock. Neither an isolated pass nor p1->DS->p2 is full-frame latency.
// Build: xcrun clang++ -std=c++17 -O2 -fobjc-arc -framework Foundation
//        -framework Metal tools/bench/fsr412_mpp_tuning.mm -o /tmp/fsr412_mpp_tuning
// Run: fsr412_mpp_tuning ROOT NATIVE_DIR INITIALIZER [--cpu-only|--compile-only|
//      --correctness|--timing-lock] [--size W H] [--rounds N] [--batch N]
//      [--scope isolated|p1-ds-p2|cnb-ds] [--candidate NAME] [--compare-root ROOT]
//      [--order-phase 0|1] (reverse the initial AB/BA order for fused timing)
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <CommonCrypto/CommonDigest.h>
#include "fsr412_mpp_oracle.h"
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>

namespace {
using fsr411::PassParams;
using fsr412_tuning_cpu::tensor_end;
void need(bool value,const std::string &message) {if(!value)throw std::runtime_error(message);}
NSString *ns(const std::string &s) {return [NSString stringWithUTF8String:s.c_str()];}
std::string error(NSError *e) {return e?e.localizedDescription.UTF8String:"Metal allocation failed";}
std::string contents(const std::string &p) {
    std::ifstream f(p,std::ios::binary);need(bool(f),"cannot read "+p);
    std::ostringstream s;s<<f.rdbuf();return s.str();
}
std::vector<uint8_t> bytes(const std::string &p) {auto s=contents(p);return {s.begin(),s.end()};}
std::string sha(const std::vector<uint8_t> &data) {
    unsigned char hash[CC_SHA256_DIGEST_LENGTH];
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
    CC_SHA256(data.data(),CC_LONG(data.size()),hash);
#pragma clang diagnostic pop
    std::string s;const char *hex="0123456789abcdef";
    for(auto x:hash) {s+=hex[x>>4];s+=hex[x&15];}return s;
}
PassParams params(const std::vector<uint8_t> &data,size_t offset) {
    need(offset<=data.size() && sizeof(PassParams)<=data.size()-offset,"parameter overflow");
    PassParams p;std::memcpy(&p,data.data()+offset,sizeof(p));return p;
}
struct Shape {const char *name;unsigned x,y,threads,bit;};
const Shape baseline={"fsr411_ds16",64,1,128,0};
const Shape shapes[]={
    {"fsr412_ds16_coop_8x1",8,1,32,0},{"fsr412_ds16_coop_16x1",16,1,32,1},
    {"fsr412_ds16_coop_32x1",32,1,32,2},{"fsr412_ds16_coop_64x1",64,1,32,3},
    {"fsr412_ds16_coop_8x2",8,2,32,4},{"fsr412_ds16_coop_16x2",16,2,32,5},
    {"fsr412_ds16_coop_64x1_sg4",64,1,128,6}};
const Shape fusedShapes[]={{"fsr412_cnb_ds_coop_32x2",16,1,32,0},{"fsr412_cnb_ds_coop_64x2",32,1,32,1}};
id<MTLLibrary> library(id<MTLDevice> dev,const std::string &root,bool candidate,bool specialized,unsigned tier) {
    std::string s=contents(root+"/src/fsr411_mpp.metal");
    const std::string include="#include \"fsr411_model.h\"";
    auto at=s.find(include);need(at!=s.npos,"model include missing");
    s.replace(at,include.size(),contents(root+"/src/fsr411_model.h"));
    if(candidate) {
        auto c=contents(root+"/src/fsr412_mpp.metal");const std::string common="#include \"fsr411_mpp.metal\"";
        auto pos=c.find(common);need(pos!=c.npos,"common MPP include missing");c.replace(pos,common.size(),s);s=c;
    }
    auto o=[MTLCompileOptions new];o.languageVersion=MTLLanguageVersion4_0;
    o.preprocessorMacros=@{@"FSR411_SPECIALIZE":@(specialized),@"FSR411_MAXW":@(tier*8u/9u),@"FSR411_MAXH":@(tier/2u)};
    NSError *e=nil;auto lib=[dev newLibraryWithSource:ns(s) options:o error:&e];need(lib!=nil,error(e));return lib;
}
id<MTLComputePipelineState> pipeline(id<MTLDevice> dev,id<MTLLibrary> lib,const Shape &s,
    const PassParams *p=nullptr,const PassParams *second=nullptr) {
    NSError *e=nil;id<MTLFunction> f=nil;
    if(p) {
        auto v=[MTLFunctionConstantValues new];auto m=fsr411::specialize_metadata(*p,false);
        [v setConstantValues:m.values type:MTLDataTypeUInt4 withRange:NSMakeRange(0,fsr411::kPassSpecializationCount)];
        if(second) {
            const auto ds=fsr411::specialize_metadata(*second,false);
            [v setConstantValues:ds.values type:MTLDataTypeUInt4 withRange:NSMakeRange(12,6)];
            [v setConstantValues:ds.values[10] type:MTLDataTypeUInt4 withRange:NSMakeRange(22,2)];
        }
        f=[lib newFunctionWithName:ns(s.name) constantValues:v error:&e];
    } else f=[lib newFunctionWithName:ns(s.name)];
    need(f!=nil,error(e));auto d=[MTLComputePipelineDescriptor new];d.computeFunction=f;
    d.maxTotalThreadsPerThreadgroup=s.threads;d.threadGroupSizeIsMultipleOfThreadExecutionWidth=YES;
    auto state=[dev newComputePipelineStateWithDescriptor:d options:MTLPipelineOptionNone reflection:nil error:&e];
    need(state && state.threadExecutionWidth==32 && state.maxTotalThreadsPerThreadgroup>=s.threads,error(e));return state;
}
void done(id<MTLCommandBuffer> cb) {
    [cb commit];[cb waitUntilCompleted];need(cb.status==MTLCommandBufferStatusCompleted,error(cb.error));
}
void dispatch(id<MTLComputeCommandEncoder> e,id<MTLComputePipelineState> state,const Shape &s,
    id<MTLBuffer> scratch,id<MTLBuffer> weights,const PassParams &p) {
    [e setComputePipelineState:state];[e setBuffer:scratch offset:0 atIndex:0];[e setBuffer:weights offset:0 atIndex:1];
    [e setBytes:&p length:sizeof(p) atIndex:2];
    [e dispatchThreadgroups:MTLSizeMake((p.output.width+s.x-1)/s.x,(p.output.height+s.y-1)/s.y,1)
        threadsPerThreadgroup:MTLSizeMake(s.threads,1,1)];
}
uint32_t probe(id<MTLDevice> dev,id<MTLCommandQueue> q,id<MTLLibrary> lib,
    const char *name="fsr412_mpp_tuning_probe",unsigned known=127u) {
    Shape s={name,1,1,32,0};auto state=pipeline(dev,lib,s);
    auto b=[dev newBufferWithLength:4 options:MTLResourceStorageModeShared];need(b!=nil,"probe buffer");
    *(uint32_t *)b.contents=~0u;auto cb=[q commandBuffer];auto e=[cb computeCommandEncoder];
    [e setComputePipelineState:state];[e setBuffer:b offset:0 atIndex:0];
    [e dispatchThreadgroups:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];[e endEncoding];done(cb);
    auto mask=*(uint32_t *)b.contents;need(!(mask&~known),"invalid probe result");return mask;
}
uint32_t common_probe(id<MTLDevice> dev,id<MTLCommandQueue> q,id<MTLLibrary> lib) {
    Shape s={"fsr411_mpp_chain_probe",1,1,32,0};auto state=pipeline(dev,lib,s);
    auto b=[dev newBufferWithLength:4 options:MTLResourceStorageModeShared];
    std::vector<uint8_t> zeros(fsr411::kMppProbeDummyBytes,0);
    auto dummy=[dev newBufferWithBytes:zeros.data() length:zeros.size() options:MTLResourceStorageModeShared];
    need(b && dummy,"common probe buffers");*(uint32_t *)b.contents=~0u;
    auto cb=[q commandBuffer];auto e=[cb computeCommandEncoder];[e setComputePipelineState:state];
    [e setBuffer:b offset:0 atIndex:0];[e setBuffer:dummy offset:0 atIndex:1];
    [e dispatchThreadgroups:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];[e endEncoding];done(cb);
    const unsigned mask=*(uint32_t *)b.contents;need(!(mask&~fsr411::kMppChainKnownMask),"invalid common probe");return mask;
}
size_t correctness(id<MTLDevice> dev,id<MTLCommandQueue> q,id<MTLLibrary> before,id<MTLLibrary> after,
    PassParams p,const std::vector<uint8_t> &weights,const std::vector<uint8_t> &init,unsigned mask) {
    std::vector<Shape> selected={baseline};for(auto s:shapes)if(mask&(1u<<s.bit))selected.push_back(s);
    NSMutableArray *states=[NSMutableArray new];for(size_t i=0;i<selected.size();++i)
        [states addObject:pipeline(dev,i?after:before,selected[i])];
    auto wb=[dev newBufferWithBytes:weights.data() length:weights.size() options:MTLResourceStorageModeShared];
    need(wb!=nil,"correctness weights allocation");size_t count=0;
    for(auto wh : {std::pair<unsigned,unsigned>{1,1},{3,5},{17,9},{129,3},{130,4}})
        for(bool banked : {true,false})for(unsigned pattern=0;pattern<5;++pattern) {
            auto a=fsr412_tuning_cpu::shape(p,wh.first,wh.second,banked);
            const auto input=fsr412_tuning_cpu::input(a,pattern);
            const auto oracle=fsr412_tuning_cpu::ds16(a,input,weights,init,true);auto expected=input;
            for(unsigned y=0;y<a.output.height;++y)for(unsigned x=0;x<a.output.width;++x)for(unsigned c=0;c<32;++c)
                expected.at(fsr411::tensor_address(a.output,x,y,c))=oracle[(size_t(y)*a.output.width+x)*32u+c];
            NSMutableArray *buffers=[NSMutableArray new];auto cb=[q commandBuffer];auto e=[cb computeCommandEncoder];
            for(size_t i=0;i<selected.size();++i) {
                auto b=[dev newBufferWithBytes:input.data() length:input.size() options:MTLResourceStorageModeShared];
                need(b!=nil,"correctness scratch allocation");[buffers addObject:b];dispatch(e,states[i],selected[i],b,wb,a);
            }
            [e endEncoding];done(cb);
            for(size_t i=0;i<selected.size();++i) {
                id<MTLBuffer> b=buffers[i];const auto *got=(const uint8_t *)b.contents;
                auto mismatch=std::mismatch(expected.begin(),expected.end(),got);
                need(mismatch.first==expected.end(),std::string(selected[i].name)+" CPU oracle/guard mismatch, input "
                    +std::to_string(wh.first)+"x"+std::to_string(wh.second)+", byte "+std::to_string(mismatch.first-expected.begin()));
                ++count;
            }
        }
    return count;
}
double median(std::vector<double> v) {
    need(!v.empty(),"no timing samples");std::sort(v.begin(),v.end());size_t n=v.size();
    return n%2?v[n/2]:(v[n/2-1]+v[n/2])/2;
}
NSArray *numbers(const std::vector<double> &v) {auto a=[NSMutableArray new];for(auto x:v)[a addObject:@(x)];return a;}
void cpu_check_output(id<MTLBuffer> b,const PassParams &p,const std::vector<uint8_t> &weights,
    const std::vector<uint8_t> &initializer) {
    const auto *data=(const uint8_t *)b.contents;
    const std::vector<uint8_t> snapshot(data,data+b.length);
    const auto oracle=fsr412_tuning_cpu::ds16(p,snapshot,weights,initializer,true);
    for(unsigned y=0;y<p.output.height;++y)for(unsigned x=0;x<p.output.width;++x)for(unsigned c=0;c<32;++c)
        need(data[fsr411::tensor_address(p.output,x,y,c)]==oracle[(size_t(y)*p.output.width+x)*32u+c],
            "timed-size GPU output differs from the raw initializer CPU oracle");
}
void equal_buffers(id<MTLBuffer> a,id<MTLBuffer> b) {
    need(a.length==b.length && !std::memcmp(a.contents,b.contents,a.length),"A/B scratch differs (including guards)");
}
void fused_dispatch(id<MTLComputeCommandEncoder> e,id<MTLComputePipelineState> state,const Shape &s,
    id<MTLBuffer> scratch,id<MTLBuffer> weights,fsr412_tuning_cpu::CnbDsParams p) {
    need(fsr412_tuning_cpu::valid_fused_params(p),"invalid/aliased fused CNB+DS geometry");
    [e setComputePipelineState:state];[e setBuffer:scratch offset:0 atIndex:0];[e setBuffer:weights offset:0 atIndex:1];
    [e setBytes:&p.cnb length:sizeof(p.cnb) atIndex:2];[e setBytes:&p.ds length:sizeof(p.ds) atIndex:3];
    // Cover ALL CNB/B pixels even when DS output is floor(input/2).
    [e dispatchThreadgroups:MTLSizeMake((p.cnb.output.width+2*s.x-1)/(2*s.x),(p.cnb.output.height+1)/2,1)
        threadsPerThreadgroup:MTLSizeMake(s.threads,1,1)];
}
size_t fused_correctness(id<MTLDevice> dev,id<MTLCommandQueue> q,id<MTLLibrary> before,id<MTLLibrary> after,
    PassParams cnb,PassParams ds,const std::vector<uint8_t> &weights,const std::vector<uint8_t> &init,unsigned mask,bool specialized=false) {
    const Shape baseCnb={"fsr411_cnb_coop_chain_32x1",32,1,32,0};
    id<MTLComputePipelineState> cp=nil,dp=nil;
    NSMutableArray *states=[NSMutableArray new];std::vector<Shape> selected;
    for(auto s:fusedShapes)if(mask&(1u<<s.bit))selected.push_back(s);
    if(!specialized) {
        cp=pipeline(dev,before,baseCnb);dp=pipeline(dev,before,baseline);
        for(auto s:selected)[states addObject:pipeline(dev,after,s)];
    }
    auto wb=[dev newBufferWithBytes:weights.data() length:weights.size() options:MTLResourceStorageModeShared];
    need(wb!=nil,"fused correctness weights");size_t count=0;
    for(auto wh : {std::pair<unsigned,unsigned>{1,1},{3,5},{17,9},{65,3},{129,5},{130,4}})
        for(bool banked : {true,false})for(bool floorOutput : {true,false})for(unsigned pattern=0;pattern<5;++pattern) {
            const auto p=fsr412_tuning_cpu::fused_shape(cnb,ds,wh.first,wh.second,banked,floorOutput);
            if(specialized) {
                cp=pipeline(dev,before,baseCnb,&p.cnb);dp=pipeline(dev,before,baseline,&p.ds);[states removeAllObjects];
                for(auto s:selected)[states addObject:pipeline(dev,after,s,&p.cnb,&p.ds)];
            }
            const auto input=fsr412_tuning_cpu::fused_input(p,pattern);
            const auto expected=fsr412_tuning_cpu::fused_oracle(p,input,weights,init,true);
            NSMutableArray *buffers=[NSMutableArray new];auto cb=[q commandBuffer];auto e=[cb computeCommandEncoder];
            for(size_t i=0;i<=selected.size();++i) {
                auto b=[dev newBufferWithBytes:input.data() length:input.size() options:MTLResourceStorageModeShared];
                need(b!=nil,"fused correctness scratch");[buffers addObject:b];
                if(i)fused_dispatch(e,states[i-1],selected[i-1],b,wb,p);
                else {dispatch(e,cp,baseCnb,b,wb,p.cnb);if(p.ds.output.width && p.ds.output.height)dispatch(e,dp,baseline,b,wb,p.ds);}
            }
            [e endEncoding];done(cb);
            for(size_t i=0;i<buffers.count;++i) {
                id<MTLBuffer> b=buffers[i];const auto *got=(const uint8_t *)b.contents;
                const auto mismatch=std::mismatch(expected.begin(),expected.end(),got);
                need(mismatch.first==expected.end(),std::string(i?selected[i-1].name:"CNB+DS baseline")
                    +" fused CPU oracle/B skip/guard mismatch, "+std::to_string(wh.first)+"x"+std::to_string(wh.second)
                    +", byte "+std::to_string(mismatch.first-expected.begin()));
                ++count;
            }
        }
    return count;
}
void fused_benchmark(id<MTLDevice> dev,id<MTLLibrary> before,id<MTLLibrary> after,
    id<MTLLibrary> beforeSpec,id<MTLLibrary> afterSpec,PassParams cnb,PassParams ds,
    const std::vector<uint8_t> &weights,const std::vector<uint8_t> &init,const std::string &mode,
    const std::string &only,unsigned width,unsigned height,unsigned rounds,unsigned batch,unsigned orderPhase,NSMutableDictionary *report) {
    auto p=fsr412_tuning_cpu::fused_shape(cnb,ds,width,height);auto compiled=[NSMutableArray new];
    for(auto s:fusedShapes) {
        auto g=pipeline(dev,after,s),sp=pipeline(dev,afterSpec,s,&p.cnb,&p.ds);
        [compiled addObject:@{@"kernel":ns(s.name),@"generic_threadgroup_bytes":@(g.staticThreadgroupMemoryLength),
            @"specialized_threadgroup_bytes":@(sp.staticThreadgroupMemoryLength),@"max_threads":@(sp.maxTotalThreadsPerThreadgroup),
            @"cnb_tile":@[@(s.x*2),@2],@"ds_tile":@[@(s.x),@1]}];
    }
    report[@"compiled"]=compiled;report[@"fusion_contract"]=@"B retained globally for p3 skip; full CNB output dispatch including odd floor-DS boundaries; one SIMD device-memory barrier; no cross-threadgroup B reads; separate PassParams buffers2/3; no ABI replacement";
    if(mode=="compile")return;
    auto q=[dev newCommandQueue];need(q!=nil,"fused queue");const unsigned mask=probe(dev,q,after,"fsr412_cnb_ds_tuning_probe",3u);
    report[@"fused_probe_mask"]=@(mask);
    report[@"fused_gpu_oracle_cases"]=@(fused_correctness(dev,q,before,after,cnb,ds,weights,init,mask));
    report[@"fused_gpu_specialized_oracle_cases"]=@(fused_correctness(dev,q,beforeSpec,afterSpec,cnb,ds,weights,init,mask,true));
    if(mode!="timing")return;
    need(mask!=0,"no supported fused layouts: do not dispatch these candidates");
    const unsigned commonMask=common_probe(dev,q,before)&common_probe(dev,q,after);
    const Shape baseCnb={(commonMask&fsr411::kMppChainCnb)?"fsr411_cnb_coop_fast_32x1":"fsr411_cnb_coop_chain_32x1",32,1,32,0};
    auto cp=pipeline(dev,beforeSpec,baseCnb,&p.cnb),dp=pipeline(dev,beforeSpec,baseline,&p.ds);
    const auto initial=fsr412_tuning_cpu::fused_input(p,0);
    const auto expected=fsr412_tuning_cpu::fused_oracle(p,initial,weights,init,true);
    auto wb=[dev newBufferWithBytes:weights.data() length:weights.size() options:MTLResourceStorageModeShared];need(wb!=nil,"fused weights");
    auto results=[NSMutableArray new],completed=[NSMutableArray new];const unsigned warmups=4;
    const auto thermal=NSProcessInfo.processInfo.thermalState;
    for(auto s:fusedShapes) {
        if(!(mask&(1u<<s.bit)) || (!only.empty() && only!=s.name))continue;
        auto candidate=pipeline(dev,afterSpec,s,&p.cnb,&p.ds);
        id<MTLBuffer> scratch[2]={[dev newBufferWithBytes:initial.data() length:initial.size() options:MTLResourceStorageModeShared],
            [dev newBufferWithBytes:initial.data() length:initial.size() options:MTLResourceStorageModeShared]};
        need(scratch[0] && scratch[1],"fused scratch");std::vector<double> times[2];
        for(unsigned r=0;r<rounds+warmups;++r)for(unsigned j=0;j<2;++j) {
            const unsigned v=((r+orderPhase)&1u)^j;auto cb=[q commandBuffer];auto e=[cb computeCommandEncoder];
            for(unsigned k=0;k<batch;++k) {
                if(v)fused_dispatch(e,candidate,s,scratch[v],wb,p);
                else {dispatch(e,cp,baseCnb,scratch[v],wb,p.cnb);dispatch(e,dp,baseline,scratch[v],wb,p.ds);}
            }
            [e endEncoding];done(cb);const double ms=(cb.GPUEndTime-cb.GPUStartTime)*1000./batch;
            need(std::isfinite(ms) && ms>0,"invalid fused timestamp");
            need(scratch[v].length==expected.size() && !memcmp(scratch[v].contents,expected.data(),expected.size()),
                std::string(v?s.name:"tuned CNB+DS baseline")+" measured-batch raw CPU oracle/B skip/guard mismatch");
            if(r>=warmups)times[v].push_back(ms);
        }
        equal_buffers(scratch[0],scratch[1]);[completed addObject:scratch[1]];
        std::vector<double> ratios;for(unsigned r=0;r<rounds;++r)ratios.push_back(times[1][r]/times[0][r]);
        [results addObject:@{@"baseline":@[ns(baseCnb.name),ns(baseline.name)],@"candidate":ns(s.name),
            @"baseline_ms":@(median(times[0])),@"candidate_ms":@(median(times[1])),@"paired_change_percent":@((median(ratios)-1)*100),
            @"baseline_samples_ms":numbers(times[0]),@"candidate_samples_ms":numbers(times[1]),@"byte_identical":@YES,
            @"candidate_threadgroup_bytes":@(candidate.staticThreadgroupMemoryLength),@"cnb_tile":@[@(s.x*2),@2],@"threads":@(s.threads),
            @"batch_cpu_oracle_checks":@((rounds+warmups)*2),@"order_phase":@(orderPhase)}];
    }
    need(results.count>0,"no supported fused candidates selected");
    // The expected raw-initializer result is computed once before warmups.
    // Every measured batch is compared outside its timed command buffer.
    for(id<MTLBuffer> b in completed)need(b.length==expected.size() && !memcmp(b.contents,expected.data(),expected.size()),"fused timed-size raw CPU oracle/B skip mismatch");
    report[@"timings"]=results;report[@"timed_size_full_scratch_cpu_oracle"]=@YES;
    report[@"size"]=@[@(width),@(height)];report[@"rounds"]=@(rounds);report[@"batch"]=@(batch);report[@"warmup_pairs"]=@(warmups);
    report[@"thermal_start"]=@(thermal);report[@"thermal_end"]=@(NSProcessInfo.processInfo.thermalState);
    report[@"low_power_mode"]=@(NSProcessInfo.processInfo.lowPowerModeEnabled);
    report[@"order_phase"]=@(orderPhase);report[@"per_batch_oracle_outside_timing"]=@YES;
    report[@"timing_contract"]=@"same fixed A/weights and retained B/C; independent scratch; alternating AB/BA pairs with explicit initial-order phase; specialized/capped pipelines; exact full-size integer oracle before warmup, full scratch oracle after each batch outside timestamps; no timed copies; not full-frame";
}
}
int main(int argc,char **argv) { @autoreleasepool {try {
    need(argc>=4,"usage: fsr412_mpp_tuning ROOT NATIVE_DIR INITIALIZER [--cpu-only|--compile-only|--correctness|--timing-lock]");
    std::string root=argv[1],dir=argv[2],beforeRoot=root,mode="cpu",scope="isolated",only;
    unsigned width=960,height=540,rounds=20,batch=16,orderPhase=0;
    for(int i=4;i<argc;++i) {
        std::string a=argv[i];
        if(a=="--cpu-only")mode="cpu";else if(a=="--compile-only")mode="compile";
        else if(a=="--correctness")mode="correctness";else if(a=="--timing-lock")mode="timing";
        else if(a=="--compare-root" && i+1<argc)beforeRoot=argv[++i];
        else if(a=="--candidate" && i+1<argc)only=argv[++i];
        else if(a=="--scope" && i+1<argc)scope=argv[++i];
        else if(a=="--rounds" && i+1<argc)rounds=std::stoul(argv[++i]);
        else if(a=="--batch" && i+1<argc)batch=std::stoul(argv[++i]);
        else if(a=="--order-phase" && i+1<argc)orderPhase=std::stoul(argv[++i]);
        else if(a=="--size" && i+2<argc) {width=std::stoul(argv[++i]);height=std::stoul(argv[++i]);}
        else throw std::runtime_error("unknown/incomplete option "+a);
    }
    need(width && height && width<=7680 && height<=4320 && rounds>=2 && rounds<=1000 && batch>=1 && batch<=256,"invalid size/rounds/batch");
    need(orderPhase<=1,"order-phase must be 0 or 1");
    need(scope=="isolated" || scope=="p1-ds-p2" || scope=="cnb-ds","invalid scope");
    auto packed=bytes(dir+"/fsr411_native_weights.bin"),pb=bytes(dir+"/fsr411_native_params.bin"),init=bytes(argv[3]);
    NSError *e=nil;auto md=[NSData dataWithContentsOfFile:ns(dir+"/fsr411_native_manifest.json")];
    NSDictionary *manifest=md?[NSJSONSerialization JSONObjectWithData:md options:0 error:&e]:nil;
    need(manifest && [manifest[@"fsr_version"] isEqual:@"4.1.2"],"missing/non-Lite manifest");
    need(sha(init)==[manifest[@"initializer_sha256"] UTF8String],"initializer identity mismatch");
    need(sha(packed)==[manifest[@"weights_sha256"] UTF8String] && sha(pb)==[manifest[@"params_sha256"] UTF8String],"native blob identity mismatch");
    auto p=params(pb,[manifest[@"passes"][@"4"][@"parameter_offset"] unsignedIntValue]);
    const auto p1=params(pb,[manifest[@"passes"][@"1"][@"parameter_offset"] unsignedIntValue]);
    fsr412_tuning_cpu::source_epilogues(p1,p,[manifest[@"model"] isEqual:@"ultraperformance"]);
    const size_t cpuCases=fsr412_tuning_cpu::cpu_parity(p,packed,init);
    const size_t fusedCpuCases=scope=="cnb-ds"?fsr412_tuning_cpu::fused_cpu_parity(p1,p,packed,init):0;
    auto report=[NSMutableDictionary dictionaryWithDictionary:@{@"mode":ns(mode),@"initializer_sha256":ns(sha(init)),
        @"weights_sha256":ns(sha(packed)),@"profile":manifest[@"profile"],@"model":manifest[@"model"],
        @"root":ns(root),@"compare_root":ns(beforeRoot),
        @"candidate_source_sha256":ns(sha(bytes(root+"/src/fsr412_mpp.metal"))),
        @"candidate_common_source_sha256":ns(sha(bytes(root+"/src/fsr411_mpp.metal"))),
        @"candidate_header_sha256":ns(sha(bytes(root+"/src/fsr411_model.h"))),
        @"baseline_source_sha256":ns(sha(bytes(beforeRoot+"/src/fsr411_mpp.metal"))),
        @"baseline_header_sha256":ns(sha(bytes(beforeRoot+"/src/fsr411_model.h"))),
        @"cpu_oracle_cases":@(cpuCases),@"fused_cpu_oracle_cases":@(fusedCpuCases),@"scope":ns(scope),@"full_frame_measured":@NO,
        @"full_frame_suitability":@"Requires original-shader/temporal parity and full-frame measurements in production; this bench selects no runtime kernel."}];
    if(mode!="cpu") {
        auto dev=MTLCreateSystemDefaultDevice();need(dev && [dev supportsFamily:MTLGPUFamilyApple10],"M5/Apple10 required");
        unsigned tier=[manifest[@"tier"] unsignedIntValue];
        auto before=library(dev,beforeRoot,false,false,tier),after=library(dev,root,true,false,tier);
        auto beforeSpec=library(dev,beforeRoot,false,true,tier),afterSpec=library(dev,root,true,true,tier);
        auto a=fsr412_tuning_cpu::shape(p,width,height);
        NSMutableArray *compiled=[NSMutableArray new];
        for(auto s:shapes) {
            auto g=pipeline(dev,after,s),sp=pipeline(dev,afterSpec,s,&a);
            [compiled addObject:@{@"kernel":ns(s.name),@"generic_threadgroup_bytes":@(g.staticThreadgroupMemoryLength),
                @"specialized_threadgroup_bytes":@(sp.staticThreadgroupMemoryLength),@"max_threads":@(sp.maxTotalThreadsPerThreadgroup)}];
        }
        pipeline(dev,before,baseline);pipeline(dev,beforeSpec,baseline,&a);
        report[@"device"]=dev.name;report[@"compiled"]=compiled;
        if(scope=="cnb-ds") {
            fused_benchmark(dev,before,after,beforeSpec,afterSpec,p1,p,packed,init,mode,only,width,height,rounds,batch,orderPhase,report);
        } else if(mode!="compile") {
            auto q=[dev newCommandQueue];need(q!=nil,"command queue unavailable");const unsigned mask=probe(dev,q,after);
            report[@"probe_mask"]=@(mask);report[@"gpu_oracle_cases"]=@(correctness(dev,q,before,after,p,packed,init,mask));
            if(mode=="timing") {
                auto thermal=NSProcessInfo.processInfo.thermalState;
                auto wb=[dev newBufferWithBytes:packed.data() length:packed.size() options:MTLResourceStorageModeShared];need(wb!=nil,"weights buffer");
                const unsigned commonMask=common_probe(dev,q,before)&common_probe(dev,q,after);
                const Shape cnb={(commonMask&fsr411::kMppChainCnb)?"fsr411_cnb_coop_fast_32x1":"fsr411_cnb_coop_chain_32x1",32,1,32,0};
                const Shape fnb={(commonMask&fsr411::kMppChainFnb32)?"fsr411_fnb32_coop_fast_16x1":"fsr411_fnb32_coop_chain_16x1",16,1,32,0};
                report[@"neighbor_probe_features"]=@(commonMask);
                report[@"neighbor_kernels"]=@[ns(cnb.name),ns(fnb.name)];
                PassParams p1=params(pb,[manifest[@"passes"][@"1"][@"parameter_offset"] unsignedIntValue]);
                PassParams p2=params(pb,[manifest[@"passes"][@"2"][@"parameter_offset"] unsignedIntValue]);
                auto initial=fsr412_tuning_cpu::input(a,0);
                id<MTLComputePipelineState> pre[2]={nil,nil},post[2]={nil,nil};
                if(scope=="p1-ds-p2") {
                    // Preserve each production tensor's strides/banks, but
                    // relocate p1 input and p2 output to avoid scratch aliasing
                    // across repeated dispatches. Both variants use this layout.
                    p1.input=a.input;p1.input.offset=unsigned((initial.size()+255u)&~255u);
                    p1.output=a.input;p2.input=a.output;p2.output=a.output;
                    p2.output.offset=unsigned((tensor_end(p1.input)+511u)&~255u);
                    initial.resize(tensor_end(p2.output)+512u,0xa7);
                    for(unsigned y=0;y<height;++y)for(unsigned x=0;x<width;++x)for(unsigned c=0;c<16;++c)
                        initial.at(fsr411::tensor_address(p1.input,x,y,c))=initial.at(fsr411::tensor_address(a.input,x,y,c));
                    pre[0]=pipeline(dev,beforeSpec,cnb,&p1);pre[1]=pipeline(dev,afterSpec,cnb,&p1);
                    post[0]=pipeline(dev,beforeSpec,fnb,&p2);post[1]=pipeline(dev,afterSpec,fnb,&p2);
                }
                auto results=[NSMutableArray new];const unsigned warmups=4;
                for(auto s:shapes) {
                    if(!(mask&(1u<<s.bit)) || (!only.empty() && only!=s.name))continue;
                    auto base=pipeline(dev,beforeSpec,baseline,&a),candidate=pipeline(dev,afterSpec,s,&a);
                    id<MTLBuffer> scratch[2]={[dev newBufferWithBytes:initial.data() length:initial.size() options:MTLResourceStorageModeShared],
                        [dev newBufferWithBytes:initial.data() length:initial.size() options:MTLResourceStorageModeShared]};
                    need(scratch[0] && scratch[1],"timing scratch allocation");std::vector<double> times[2];
                    for(unsigned r=0;r<rounds+warmups;++r)for(unsigned j=0;j<2;++j) {
                        const unsigned v=(r&1u)^j;auto cb=[q commandBuffer];auto enc=[cb computeCommandEncoder];
                        for(unsigned k=0;k<batch;++k) {
                            if(pre[v])dispatch(enc,pre[v],cnb,scratch[v],wb,p1);
                            dispatch(enc,v?candidate:base,v?s:baseline,scratch[v],wb,a);
                            if(post[v])dispatch(enc,post[v],fnb,scratch[v],wb,p2);
                        }
                        [enc endEncoding];done(cb);double ms=(cb.GPUEndTime-cb.GPUStartTime)*1000./batch;
                        need(std::isfinite(ms) && ms>0,"invalid timestamp");if(r>=warmups)times[v].push_back(ms);
                    }
                    equal_buffers(scratch[0],scratch[1]);
                    cpu_check_output(scratch[1],a,packed,init);
                    std::vector<double> ratios;for(unsigned r=0;r<rounds;++r)ratios.push_back(times[1][r]/times[0][r]);
                    [results addObject:@{@"baseline":ns(baseline.name),@"candidate":ns(s.name),@"baseline_ms":@(median(times[0])),
                        @"candidate_ms":@(median(times[1])),@"paired_change_percent":@((median(ratios)-1)*100),
                        @"baseline_samples_ms":numbers(times[0]),@"candidate_samples_ms":numbers(times[1]),@"byte_identical":@YES,@"timed_size_cpu_oracle":@YES,
                        @"baseline_threadgroup_bytes":@(base.staticThreadgroupMemoryLength),@"candidate_threadgroup_bytes":@(candidate.staticThreadgroupMemoryLength),
                        @"tile":@[@(s.x),@(s.y)],@"threads":@(s.threads)}];
                }
                need(results.count>0,"no probed candidates selected");report[@"timings"]=results;
                report[@"size"]=@[@(width),@(height)];report[@"rounds"]=@(rounds);report[@"batch"]=@(batch);report[@"warmup_pairs"]=@(warmups);
                report[@"thermal_start"]=@(thermal);report[@"thermal_end"]=@(NSProcessInfo.processInfo.thermalState);
                report[@"low_power_mode"]=@(NSProcessInfo.processInfo.lowPowerModeEnabled);
                report[@"timing_contract"]=@"same fixed input/weights; independent scratch; alternating AB/BA pairs; capped specialized pipelines; CPU oracle before timing";
            }
        }
    }
    auto json=[NSJSONSerialization dataWithJSONObject:report options:NSJSONWritingPrettyPrinted error:&e];need(json!=nil,error(e));
    fwrite(json.bytes,1,json.length,stdout);puts("");return 0;
} catch(const std::exception &e) {fprintf(stderr,"FSR412 MPP tuning: %s\n",e.what());return 1;} } }
