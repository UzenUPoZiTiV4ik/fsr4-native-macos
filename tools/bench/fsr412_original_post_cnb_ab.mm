// SPDX-License-Identifier: MIT
// Controlled original POST versus precomputed native POST-CNB. No src changes.
#include "fsr412_upstream_replay.h"
#include "../../src/fsr412_runtime.h"
#include <chrono>
using namespace fsr412_original;
namespace {
constexpr size_t precomputedBase=133094720, precomputedStride=30720, allocatedBytes=166272320;
struct NativeCnb {
    fsr411::Runtime runtime;
    fsr411::Session session;
    fsr411::NativePass op;
    NativeCnb(Replay &r,const std::string &root,fsr411::Kernels kind):runtime(r.device,root,root+"/build") {
        session.modelLite=true;session.profile=fsr411::Profile::Int8LiteLossy;session.tier=2160;session.externalPostCnb=true;
        check(fsr411::post_cnb_scratch_layout(2160,false,session.scratchLayout),"native layout");
        NSData *init=[NSData dataWithContentsOfFile:ns(root+"/build/fsr412_lossy/initializers-lite-lossy.bin")];
        check(init.length==27392,"missing trained native initializer");
        session.initializer=[r.device newBufferWithBytes:init.bytes length:init.length options:MTLResourceStorageModeShared];
        std::string error;check(runtime.prepare_integer_operators(session,false,kind,error),error);
        op=session.native.at(13);
        check((op.parameters.q[0].flags & fsr411::kFirstTapBias) && op.parameters.biases[3]!=fsr411::kNoOffset,"conditional original POST biases missing");
    }
    fsr411::PassParams params(Replay &r) const {
        auto p=op.parameters;
        p.input.offset=66547328+30752+16;p.input.width=r.width/2;p.input.height=r.height/2;
        p.input.strideX=16;p.input.strideY=30752;p.input.channels=16;p.input.bankStride=0;
        p.output.offset=precomputedBase;p.output.width=r.width/2;p.output.height=r.height/2;
        p.output.strideX=16;p.output.strideY=precomputedStride;p.output.channels=16;p.output.bankStride=0;
        return p;
    }
    double dispatch(Replay &r) {
        auto p=params(r);auto cb=[r.queue commandBuffer];auto e=[cb computeCommandEncoder];
        [e setComputePipelineState:op.pso];[e setBuffer:r.scratch offset:0 atIndex:0];
        [e setBuffer:session.nativeWeights offset:0 atIndex:1];[e setBytes:&p length:sizeof(p) atIndex:2];
        if(session.halfWeights)[e setBuffer:session.halfWeights offset:0 atIndex:3];
        if(op.winograd)[e setBuffer:session.winogradWeights offset:0 atIndex:4];
        [e dispatchThreadgroups:MTLSizeMake(ceil_div(p.output.width,op.tileX),ceil_div(p.output.height,op.tileY),1)
            threadsPerThreadgroup:MTLSizeMake(op.threads,1,1)];
        [e endEncoding];[cb commit];[cb waitUntilCompleted];
        check(cb.status==MTLCommandBufferStatusCompleted,"native POST CNB failed: "+utf8(cb.error.localizedDescription));
        return (cb.GPUEndTime-cb.GPUStartTime)*1000;
    }
};
void inputs(Replay &r,unsigned f) {
    std::vector<float> c(size_t(r.renderWidth)*r.renderHeight*4),d(size_t(r.renderWidth)*r.renderHeight),m(size_t(r.renderWidth)*r.renderHeight*2);
    for(unsigned y=0;y<r.renderHeight;++y)for(unsigned x=0;x<r.renderWidth;++x){size_t i=size_t(y)*r.renderWidth+x;
        float a=.05f+.5f*float(x)/r.renderWidth+.1f*float(y)/r.renderHeight;
        c[4*i]=a;c[4*i+1]=a*.7f;c[4*i+2]=a*.4f;c[4*i+3]=1;
        if((x+f)%29==0)c[4*i]=c[4*i+1]=c[4*i+2]=.9f;
        bool fg=x>r.renderWidth/2;d[i]=fg?.8f:.2f;
        m[2*i]=fg?(f%3?3.f:-3.f)/r.width:0.f;
        m[2*i+1]=fg?(f%4?1.f:-1.f)/r.height:0.f;
    }
    r.upload(r.textures["input"],c.data());r.upload(r.textures["depth"],d.data());r.upload(r.textures["velocity"],m.data());
    float exposure[2]={1,1};r.upload(r.textures["exposure"],exposure);
}
void stress_cache(Replay &r,bool injectMisses,bool wideMisses=false) {
    // Controlled branch coverage, identical on both sides: odd signed motion
    // activates X/Y/XY lookups; zero encoded words force primary/side misses.
    auto p=static_cast<uint8_t *>(r.scratch.contents);
    for(unsigned y=0;y<r.height/2;y+=2)for(unsigned x=0;x<r.width/2;x+=2){
        unsigned row=y>>2,offset=((y>>1)&1)?16384:192;
        offset+=(x>>1)*8;unsigned hole=row*32+16;
        if(offset>=hole)offset+=32;if(offset>=hole+15392)offset+=32;
        int16_t motion[2]={3,-3};std::memcpy(p+33304416+size_t(row)*30752+offset,motion,4);
    }
    for(unsigned y=0;injectMisses && y<(wideMisses?r.height/2:16u) && y<r.height/2;y+=wideMisses?8u:1u)for(unsigned x=0;x<(wideMisses?r.width/2:16u) && x<r.width/2;x+=wideMisses?8u:1u){
        uint32_t zero=0;std::memcpy(p+49941264+size_t(y)*30752+x*16,&zero,4);
    }
}
double post(Replay &r,int rowOrder=0) {
    auto bytes=static_cast<uint8_t *>(r.scratch.contents);
    std::memcpy(bytes+r.cacheRelocatedByteOffset,bytes+49941264,r.cacheCopyBytes);
    const auto groups=r.groups("post");double gpuMs=0;
    auto encode=[&](MTLSize grid,unsigned row){
    auto &k=r.kernels.at("post");auto cb=[r.queue commandBuffer];auto e=[cb computeCommandEncoder];[e setComputePipelineState:k.pipeline];
    for(NSDictionary *binding in k.receipt[@"bindings"]){auto reg=utf8(binding[@"d3d_register"]),kind=utf8(binding[@"metal_kind"]);unsigned slot=[binding[@"metal_slot"] unsignedIntValue];
        if(kind=="buffer") {[e setBuffer:reg=="u11"?r.scratch:reg=="b1"?r.b1:reg=="b0"?r.b0:r.initializer offset:0 atIndex:slot];}
        else if(kind=="texture")[e setTexture:r.d3d_texture(reg) atIndex:slot];
        else if(kind=="sampler")[e setSamplerState:r.sampler atIndex:slot];
        else throw std::runtime_error("unexpected POST binding");}
    if(rowOrder)[e setBytes:&row length:sizeof(row) atIndex:4];
    [e dispatchThreadgroups:grid threadsPerThreadgroup:k.threads];[e endEncoding];[cb commit];[cb waitUntilCompleted];
    check(cb.status==MTLCommandBufferStatusCompleted,"POST failed: "+utf8(cb.error.localizedDescription));
    gpuMs+=(cb.GPUEndTime-cb.GPUStartTime)*1000;
    };
    if(rowOrder){for(unsigned i=0;i<groups.height;++i)encode(MTLSizeMake(groups.width,1,1),rowOrder>0?i:unsigned(groups.height-1-i));}
    else encode(groups,0);
    std::memcpy(bytes+49941264,bytes+r.cacheRelocatedByteOffset,r.cacheCopyBytes);
    return gpuMs;
}
void same(Replay &a,Replay &b,unsigned f) {
    bool anyMismatch=false;
    for(const auto &name:{"output","history_next","recurrent_next","reprojected"}) {
        NSData *x=a.read_texture(a.textures.at(name)),*y=b.read_texture(b.textures.at(name));
        if(x.length!=y.length || std::memcmp(x.bytes,y.bytes,x.length)) {
            auto xx=static_cast<const uint8_t *>(x.bytes),yy=static_cast<const uint8_t *>(y.bytes);size_t first=0,count=0;
            for(size_t i=0;i<x.length;++i)if(xx[i]!=yy[i]){if(!count)first=i;++count;}
            size_t firstPixel=first/Replay::bytes_per_pixel(a.textures.at(name));
            std::fprintf(stderr,"DIFF %s frame=%u bytes=%zu firstByte=%zu firstPixel=(%zu,%zu)\n",name,f,count,first,firstPixel%a.width,firstPixel/a.width);
            anyMismatch=true;

        }
    }
    auto x=static_cast<const uint8_t *>(a.scratch.contents),y=static_cast<const uint8_t *>(b.scratch.contents);
    if(std::memcmp(x,y,a.logicalScratchBytes)) {
        size_t first=0,count=0;for(size_t i=0;i<a.logicalScratchBytes;++i)if(x[i]!=y[i]){if(!count)first=i;++count;}
        std::fprintf(stderr,"DIFF logicalScratch frame=%u bytes=%zu firstByte=%zu\n",f,count,first);anyMismatch=true;

    }
    if(std::memcmp(x+a.cacheRelocatedByteOffset,y+b.cacheRelocatedByteOffset,a.cacheCopyBytes)){
        size_t count=0,first=0;for(size_t i=0;i<a.cacheCopyBytes;++i)if(x[a.cacheRelocatedByteOffset+i]!=y[b.cacheRelocatedByteOffset+i]){if(!count)first=i;++count;}
        std::fprintf(stderr,"DIFF isolatedCache frame=%u bytes=%zu firstOffset=%zu\n",f,count,first);anyMismatch=true;
    }
    check(!anyMismatch,"bitwise POST comparison failed");

}
void install(Replay &r,const std::string &path) {
    NSDictionary *receipt=json(path+".json");NSData *source=[NSData dataWithContentsOfFile:ns(path+".metal")];
    check(source && hash(source.bytes,source.length)==utf8(receipt[@"metal_sha256"]),"rewritten POST identity mismatch");
    check([receipt[@"precomputed_base"] unsignedLongLongValue]==precomputedBase && [receipt[@"physical_scratch_bytes"] unsignedLongLongValue]==allocatedBytes,"precomputed layout mismatch");
    auto expanded=[r.device newBufferWithLength:allocatedBytes options:MTLResourceStorageModeShared];check(expanded!=nil,"expanded scratch allocation");
    std::memset(expanded.contents,0,expanded.length);std::memcpy(expanded.contents,r.scratch.contents,r.scratch.length);r.scratch=expanded;
    NSError *error=nil;auto lib=[r.device newLibraryWithURL:[NSURL fileURLWithPath:ns(path+".metallib")] error:&error];check(lib!=nil,utf8(error.localizedDescription));
    r.kernels["post"].pipeline=[r.device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"main0"] error:&error];check(r.kernels["post"].pipeline!=nil,utf8(error.localizedDescription));
}
}
int main(int argc,char **argv){@autoreleasepool{try {
    check(argc>=2,"usage: original_post_cnb_ab ROOT [--simd|--scalar] [--frames N] [--size RW RH UW UH] [--time N] [--stress-cache|--stress-wide|--stress-motion-only] [--all-frames|--original-pair|--ordered-original|--reverse-original]");
    std::string root=argv[1];unsigned rw=384,rh=224,w=640,h=384,frames=8,timing=0;auto kind=fsr411::Kernels::Mpp;bool stress=false,originalPair=false,stressMisses=true,allFrames=false,wideMisses=false;int rowOrder=0;
    for(int i=2;i<argc;++i){std::string a=argv[i];auto next=[&]{check(i+1<argc,"missing arg");return unsigned(std::stoul(argv[++i]));};
        if(a=="--simd")kind=fsr411::Kernels::Simd;else if(a=="--scalar")kind=fsr411::Kernels::Scalar;else if(a=="--stress-cache")stress=true;else if(a=="--original-pair")originalPair=true;else if(a=="--stress-motion-only"){stress=true;stressMisses=false;}else if(a=="--all-frames")allFrames=true;else if(a=="--ordered-original"){originalPair=true;rowOrder=1;}else if(a=="--reverse-original"){originalPair=true;rowOrder=-1;}else if(a=="--stress-wide"){stress=true;wideMisses=true;}
        else if(a=="--frames")frames=next();else if(a=="--time")timing=next();else if(a=="--size"){rw=next();rh=next();w=next();h=next();}else throw std::runtime_error("unknown arg "+a);}
    check(frames>=8,"at least8 frames required");
    Replay a(root+"/build/fsr412_upstream",w,h,rw,rh,"isolated","u8"),b(root+"/build/fsr412_upstream",w,h,rw,rh,"isolated","u8");
    auto originalCandidatePipeline=b.kernels["post"].pipeline;
    install(b,root+"/build/fsr412_upstream/a69f41a0bb37376178001a7f73087d4726e21c0d531e7635653277053ac4b268.precomputed");
    auto rewrittenPipeline=b.kernels["post"].pipeline;
    if(originalPair)b.kernels["post"].pipeline=originalCandidatePipeline;
    if(rowOrder){NSError *error=nil;auto lib=[b.device newLibraryWithURL:[NSURL fileURLWithPath:ns(root+"/build/fsr412_upstream/a69f41a0bb37376178001a7f73087d4726e21c0d531e7635653277053ac4b268.isolated-row.metallib")] error:&error];check(lib!=nil,utf8(error.localizedDescription));b.kernels["post"].pipeline=[b.device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"main0"] error:&error];check(b.kernels["post"].pipeline!=nil,utf8(error.localizedDescription));}
    NativeCnb native(b,root,kind);unsigned skipped=0;double originalMs=0,rewriteMs=0,cnbMs=0;
    for(unsigned f=0;f<frames;++f){inputs(a,f);inputs(b,f);float jx=f&1?-.25f:.125f,jy=f%3?1.f/6.f:-1.f/6.f;
        a.set_frame(jx,jy,f==0);b.set_frame(jx,jy,f==0);
        for(const std::string &stage:{"pre","pass1","pass2","border","pass3"}){a.dispatch(stage);b.dispatch(stage);}
        auto marker=*(const uint32_t *)((const uint8_t *)b.scratch.contents+33273792);bool skip=marker==1511506142;skipped+=skip;
        if(stress && skip){stress_cache(a,stressMisses,wideMisses);stress_cache(b,stressMisses,wideMisses);}
        std::fprintf(stderr,"POST-CNB frame=%u skip=%d cacheStress=%d size=%ux%u\n",f,skip,stress&&skip,w,h);
        // Skip POST can read/write the same cache cell in different threads.
        // Timing of four non-atomic encoded-word stores is observable under
        // controlled cache-miss stress. Keep literal isolated POST on skip.
        bool accelerated=!skip || allFrames;
        if(!originalPair)b.kernels["post"].pipeline=accelerated?rewrittenPipeline:originalCandidatePipeline;
        if(accelerated)cnbMs+=native.dispatch(b);
        if(f&1){rewriteMs+=post(b,rowOrder);originalMs+=post(a);}else{originalMs+=post(a);rewriteMs+=post(b,rowOrder);}

        same(a,b,f);a.advance_history();b.advance_history();}
    check(skipped>0,"fixture did not reach skip frame");
    NSData *final=b.read_texture(b.textures["output"]);
    const char *comparison=rowOrder>0?"original-row-forward":rowOrder<0?"original-row-reverse":originalPair?"original-pair":allFrames?"precomputed-allframes-diagnostic":"precomputed-run-original-skip";
    std::printf("POST-CNB %s mode=%s %ux%u: frames=%u skipped=%u bitwise output/history/recurrent/reprojected/logical-scratch/cache OK hash=%s; GPU means original=%.4f rewritten=%.4f cnb=%.4f sum=%.4f ms\n",fsr411::kernels_label(kind),comparison,w,h,frames,skipped,hash(final.bytes,final.length).c_str(),originalMs/frames,rewriteMs/frames,cnbMs/frames,(rewriteMs+cnbMs)/frames);
    // Controlled interleaved timing at identical run-frame pre-POST state.
    if(timing){
        check(!originalPair && !rowOrder,"timing only compares precomputed run POST");
        inputs(a,frames);inputs(b,frames);a.set_frame(.125f,1.f/6.f,false);b.set_frame(.125f,1.f/6.f,false);
        for(const std::string &stage:{"pre","pass1","pass2","border","pass3"}){a.dispatch(stage);b.dispatch(stage);}
        if(!originalPair)b.kernels["post"].pipeline=rewrittenPipeline;
        std::vector<uint8_t> baseline(a.logicalScratchBytes);std::memcpy(baseline.data(),a.scratch.contents,baseline.size());
        std::vector<double> old,changed;
        for(unsigned i=0;i<timing+2;++i){std::memcpy(a.scratch.contents,baseline.data(),baseline.size());std::memcpy(b.scratch.contents,baseline.data(),baseline.size());
            double o,c;if(i&1){c=native.dispatch(b)+post(b);o=post(a);}else{o=post(a);c=native.dispatch(b)+post(b);}
            same(a,b,frames);if(i>=2){old.push_back(o);changed.push_back(c);}}
        auto median=[](std::vector<double> v){std::sort(v.begin(),v.end());return v[v.size()/2];};
        std::printf("INTERLEAVED run POST samples=%u originalMedian=%.4f candidateMedian=%.4f ms speedup=%.3fx\n",timing,median(old),median(changed),median(old)/median(changed));}
    return 0;
}catch(const std::exception &e){std::fprintf(stderr,"POST-CNB AB: %s\n",e.what());return 1;}}}
