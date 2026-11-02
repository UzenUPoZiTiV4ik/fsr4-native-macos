// SPDX-License-Identifier: MIT
// Supplemental SIMD layouts: exact GPU/CPU oracle and interleaved timing.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wreturn-type"
#define main fsr411_unused_native_test_main
#include "../../tests/fsr411_native_test.mm"
#undef main
#pragma clang diagnostic pop
#include <numeric>

namespace {
struct Candidate {const char *name;unsigned x,y,threads;};
unsigned tensor_end(fsr411::Tensor t) {return t.channels?fsr411::tensor_address(t,t.width+1u,t.height+1u,t.channels-1u)+64u:0u;}
void ensure(bool ok,const std::string &reason) {if(!ok)throw std::runtime_error(reason);}
double percentile(std::vector<double> v,double p) {std::sort(v.begin(),v.end());return v.empty()?0:v[size_t((v.size()-1)*p)];}
NSArray *numbers(const std::vector<double> &v) {auto a=[NSMutableArray array];for(double x:v)[a addObject:@(x)];return a;}
std::vector<Candidate> candidates(const std::string &kind) {
    const std::string prefix=kind=="cnb"?"fsr412_simd_cnb_":kind=="fnb"?"fsr412_simd_fnb32_":kind=="ct"?"fsr412_simd_ct32_":"fsr412_simd_ds16_";
    static const char *names[4][6]={
        {"fsr412_simd_cnb_16x2","fsr412_simd_cnb_32x1","fsr412_simd_cnb_32x2","fsr412_simd_cnb_64x4",nullptr,nullptr},
        {"fsr412_simd_fnb32_16x2","fsr412_simd_fnb32_32x1","fsr412_simd_fnb32_32x2","fsr412_simd_fnb32_64x4",nullptr,nullptr},
        {"fsr412_simd_ct32_16x2","fsr412_simd_ct32_32x1","fsr412_simd_ct32_32x2","fsr412_simd_ct32_64x4",nullptr,nullptr},
        {"fsr412_simd_ds16_16x2","fsr412_simd_ds16_32x1","fsr412_simd_ds16_32x2","fsr412_simd_ds16_64x4","fsr412_simd_ds16_32x1_s8","fsr412_simd_ds16_32x2_s8"}};
    unsigned row=kind=="cnb"?0:kind=="fnb"?1:kind=="ct"?2:3;
    const unsigned dims[][3]={{16,2,64},{32,1,32},{32,2,64},{64,4,128},{32,1,32},{32,2,64}};
    std::vector<Candidate> result;for(unsigned i=0;i<6;++i)if(names[row][i])result.push_back({names[row][i],dims[i][0],dims[i][1],dims[i][2]});
    return result;
}
}
int main(int argc,char **argv) {@autoreleasepool {try {
    ensure(argc>=3,"usage: fsr412_simd_tuning ROOT NATIVE_DIR [--size HALF_W HALF_H] [--timing-lock --rounds N] [--report FILE]");
    std::string root=argv[1],dir=argv[2],report="build/lite-lossy-optimization/simd-tuning.json";
    bool timing=false;unsigned width=38,height=12,rounds=20;
    for(int i=3;i<argc;++i) {
        const std::string a=argv[i];
        if(a=="--timing-lock")timing=true;
        else if(a=="--size" && i+2<argc){width=std::stoul(argv[++i]);height=std::stoul(argv[++i]);}
        else if(a=="--rounds" && i+1<argc)rounds=std::stoul(argv[++i]);
        else if(a=="--report" && i+1<argc)report=argv[++i];
        else throw std::runtime_error("unknown option: "+a);
    }
    ensure(width>=4 && height>=4 && rounds>=2 && rounds<=200,"invalid size/rounds");
    auto dev=MTLCreateSystemDefaultDevice();auto queue=[dev newCommandQueue];ensure(dev && queue,"missing GPU");
    NSError *error=nil;
    auto load=[&](const std::string &path){return [NSData dataWithContentsOfFile:[NSString stringWithUTF8String:path.c_str()]];};
    NSData *wdata=load(dir+"/fsr411_native_weights.bin"),*pdata=load(dir+"/fsr411_native_params.bin"),*mdata=load(dir+"/fsr411_native_manifest.json");
    ensure(wdata && pdata && mdata,"missing model cache");
    NSDictionary *manifest=[NSJSONSerialization JSONObjectWithData:mdata options:0 error:&error];ensure(manifest!=nil,"invalid manifest");
    NSString *source=[NSString stringWithContentsOfFile:[NSString stringWithUTF8String:(root+"/src/fsr411_simd.metal").c_str()] encoding:NSUTF8StringEncoding error:&error];
    NSString *extra=[NSString stringWithContentsOfFile:[NSString stringWithUTF8String:(root+"/src/fsr412_simd.metal").c_str()] encoding:NSUTF8StringEncoding error:&error];
    NSString *header=[NSString stringWithContentsOfFile:[NSString stringWithUTF8String:(root+"/src/fsr411_model.h").c_str()] encoding:NSUTF8StringEncoding error:&error];
    ensure(source && extra && header,"missing shader sources");
    source=[[source stringByReplacingOccurrencesOfString:@"#include \"fsr411_model.h\"" withString:header] stringByAppendingString:extra];
    auto options=[MTLCompileOptions new];options.languageVersion=MTLLanguageVersion3_0;options.preprocessorMacros=@{@"FSR411_SPECIALIZE":@0};
    auto lib=[dev newLibraryWithSource:source options:options error:&error];ensure(lib!=nil,error?error.description.UTF8String:"library failed");
    std::vector<unsigned char> weights(wdata.length);std::memcpy(weights.data(),wdata.bytes,wdata.length);
    auto wb=[dev newBufferWithBytes:wdata.bytes length:wdata.length options:MTLResourceStorageModeShared];
    std::vector<_Float16> half(weights.size());for(size_t i=0;i<weights.size();++i)half[i]=(_Float16)signed_byte(weights[i]);
    auto wh=[dev newBufferWithBytes:half.data() length:half.size()*2 options:MTLResourceStorageModeShared];
    NSMutableDictionary *passes=[manifest[@"passes"] mutableCopy];passes[@"post_cnb"]=manifest[@"auxiliary_passes"][@"post_cnb"];
    auto rows=[NSMutableArray array];unsigned verified=0;
    for(NSString *key in @[@"1",@"4",@"2",@"3",@"post_cnb"]) {
        NSDictionary *entry=passes[key];if(!entry)continue;
        fsr411::PassParams p;std::memcpy(&p,(const uint8_t *)pdata.bytes+[entry[@"parameter_offset"] unsignedIntValue],sizeof p);
        auto shape=[&](unsigned slot){return slot==3 || slot==4?std::pair<unsigned,unsigned>(width/2,height/2):std::pair<unsigned,unsigned>(width,height);};
        const auto in=shape([entry[@"input_slot"] unsignedIntValue]),out=shape([entry[@"output_slot"] unsignedIntValue]);
        p.input.width=in.first;p.input.height=in.second;p.output.width=out.first;p.output.height=out.second;
        if(entry[@"skip_slot"]){p.skip.width=width;p.skip.height=height;}
        unsigned end=std::max({tensor_end(p.input),tensor_end(p.output),tensor_end(p.skip)})+512;
        if([key isEqual:@"post_cnb"]){p.output.offset=(end+511u)&~255u;end=p.output.offset+p.output.strideY*p.output.height+512;}
        std::vector<unsigned char> initial(end,0x93);uint32_t seed=0x1badf00d;
        for(unsigned y=0;y<p.input.height;++y)for(unsigned x=0;x<p.input.width;++x)for(unsigned c=0;c<p.input.channels;++c){seed^=seed<<13;seed^=seed>>17;seed^=seed<<5;initial[fsr411::tensor_address(p.input,x,y,c)]=uint8_t(seed);}
        auto expected=initial;const std::string kind=[entry[@"kind"] UTF8String];
        if(kind=="cnb")reference_cnb(expected,weights,p);else if(kind=="ds")reference_ds(expected,weights,p);else reference_fnb(expected,weights,p,kind=="ct");
        std::vector<_Float16> transformed;const bool winograd=kind!="ds" && kind!="ct" && fsr411::prepare_simd_winograd_weights(weights.data(),weights.size(),p,transformed);
        auto wg=transformed.empty()?nil:[dev newBufferWithBytes:transformed.data() length:transformed.size()*2 options:MTLResourceStorageModeShared];
        NSString *baseline=entry[@"simd"][@"function"];if(winograd)baseline=[baseline stringByAppendingString:@"_winograd"];
        auto specs=candidates(kind);specs.insert(specs.begin(),{baseline.UTF8String,[entry[@"simd"][@"tile"][0] unsignedIntValue],[entry[@"simd"][@"tile"][1] unsignedIntValue],128});
        std::vector<id<MTLComputePipelineState>> pipelines;std::vector<id<MTLBuffer>> scratch;
        for(const auto &s:specs) {
            auto fn=[lib newFunctionWithName:[NSString stringWithUTF8String:s.name]];ensure(fn!=nil,std::string("missing kernel ")+s.name);
            auto d=[MTLComputePipelineDescriptor new];d.computeFunction=fn;d.maxTotalThreadsPerThreadgroup=s.threads;d.threadGroupSizeIsMultipleOfThreadExecutionWidth=YES;
            auto ps=[dev newComputePipelineStateWithDescriptor:d options:MTLPipelineOptionNone reflection:nil error:&error];ensure(ps!=nil,error?error.description.UTF8String:"pipeline failed");
            pipelines.push_back(ps);scratch.push_back([dev newBufferWithBytes:initial.data() length:initial.size() options:MTLResourceStorageModeShared]);
        }
        auto dispatch=[&](size_t which,unsigned count){
            auto cb=[queue commandBuffer];auto e=[cb computeCommandEncoder];[e setComputePipelineState:pipelines[which]];
            [e setBuffer:scratch[which] offset:0 atIndex:0];[e setBuffer:wb offset:0 atIndex:1];[e setBytes:&p length:sizeof p atIndex:2];[e setBuffer:wh offset:0 atIndex:3];if(wg)[e setBuffer:wg offset:0 atIndex:4];
            const unsigned w=kind=="ct"?p.input.width:p.output.width,h=kind=="ct"?p.input.height:p.output.height;
            for(unsigned k=0;k<count;++k){[e dispatchThreadgroups:MTLSizeMake((w+specs[which].x-1)/specs[which].x,(h+specs[which].y-1)/specs[which].y,1) threadsPerThreadgroup:MTLSizeMake(specs[which].threads,1,1)];if(count>1)[e memoryBarrierWithScope:MTLBarrierScopeBuffers];}
            [e endEncoding];[cb commit];[cb waitUntilCompleted];ensure(cb.status==MTLCommandBufferStatusCompleted,cb.error?cb.error.description.UTF8String:"GPU failure");return (cb.GPUEndTime-cb.GPUStartTime)*1000./count;
        };
        for(size_t v=0;v<specs.size();++v){
            dispatch(v,1);
            if(std::memcmp(scratch[v].contents,expected.data(),expected.size())!=0) {
                const auto actual=static_cast<const unsigned char *>(scratch[v].contents);unsigned printed=0;
                for(unsigned yy=0;yy<p.output.height && printed<12;++yy)for(unsigned xx=0;xx<p.output.width && printed<12;++xx)for(unsigned ch=0;ch<p.output.channels && printed<12;++ch) {
                    const auto offset=fsr411::tensor_address(p.output,xx,yy,ch);
                    if(actual[offset]!=expected[offset]) {std::fprintf(stderr,"mismatch %s pass %s (%u,%u,%u): %d != %d, first-bias %u flags %u\n",specs[v].name,key.UTF8String,xx,yy,ch,signed_byte(actual[offset]),signed_byte(expected[offset]),fsr411::first_tap_bias_offset(p,xx,yy),p.q[0].flags);++printed;}
                }
                throw std::runtime_error(std::string("CPU oracle/guard mismatch ")+specs[v].name+" pass "+key.UTF8String);
            }
            ++verified;
        }
        if(timing)for(size_t v=1;v<specs.size();++v){
            std::vector<double> old,current,ratios;
            for(unsigned r=0;r<rounds+4;++r){double t[2];for(unsigned order=0;order<2;++order){unsigned a=(r&1u)^order;t[a]=dispatch(a?v:0,8);}if(r>=4){old.push_back(t[0]);current.push_back(t[1]);ratios.push_back(t[1]/t[0]);}}
            [rows addObject:@{@"pass":key,@"candidate":[NSString stringWithUTF8String:specs[v].name],@"baseline":baseline,
                @"median_ratio":@(percentile(ratios,.5)),@"before_ms":numbers(old),@"candidate_ms":numbers(current),@"threadgroup_bytes":@(pipelines[v].staticThreadgroupMemoryLength)}];
        }
    }
    auto result=@{@"device":dev.name,@"profile":manifest[@"profile"],@"verified_cases":@(verified),@"timing_enabled":@(timing),
        @"physical_pre_m5_measured":@NO,@"size":@[@(width),@(height)],@"rows":rows};
    auto json=[NSJSONSerialization dataWithJSONObject:result options:NSJSONWritingPrettyPrinted error:&error];ensure(json!=nil,"JSON failed");
    ensure([json writeToFile:[NSString stringWithUTF8String:report.c_str()] atomically:YES],"report write failed");
    std::printf("SIMD layouts: %u CPU/GPU full-scratch parity cases passed; %s\n",verified,report.c_str());return 0;
} catch(const std::exception &e){std::fprintf(stderr,"SIMD tuning: %s\n",e.what());return 1;} }}
