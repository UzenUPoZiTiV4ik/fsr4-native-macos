// SPDX-License-Identifier: MIT
// Original, stable DXIL oracle versus exact Metal integer neural operators.
#include "fsr412_upstream_replay.h"
#include "../../src/fsr412_runtime.h"
#include <numeric>

using namespace fsr412_original;
namespace {
struct Native {
    fsr411::Runtime runtime;
    fsr411::Session session;
    fsr411::Kernels kind;
    Native(Replay &r,const std::string &root,fsr411::Kernels backend):runtime(r.device,root,root+"/build"),kind(backend) {
        session.modelLite=true;session.profile=fsr411::Profile::Int8LiteLossy;session.tier=2160;session.externalPostCnb=false;
        NSData *init=[NSData dataWithContentsOfFile:ns(root+"/build/fsr412_lossy/initializers-lite-lossy.bin")];
        check(init.length==27392,"missing native trained initializer");
        session.initializer=[r.device newBufferWithBytes:init.bytes length:init.length options:MTLResourceStorageModeShared];
        std::string error;bool prepared=runtime.prepare_integer_operators(session,true,kind,error);check(prepared,error);
        r.load_pass1_border();
    }
    void dispatch(Replay &r,int pass) {
        const auto &n=session.native.at(pass);auto p=n.parameters;
        auto size=[&](unsigned slot){const unsigned shift=slot==3 || slot==4?2u:1u;return std::pair<unsigned,unsigned>(r.width>>shift,r.height>>shift);};
        const auto in=size(n.inputSlot),out=size(n.outputSlot);p.input.width=in.first;p.input.height=in.second;p.output.width=out.first;p.output.height=out.second;
        if(n.hasSkip){auto skip=size(n.skipSlot);p.skip.width=skip.first;p.skip.height=skip.second;}
        if(pass==1 || pass==4) {
            const unsigned halfW=std::min(1920u,ceil_div(r.width/2u,64u)*64u);
            const unsigned halfH=std::min(1080u,ceil_div(r.height/2u,4u)*4u);
            p.output.width=pass==1?halfW:halfW/2u;p.output.height=pass==1?halfH:halfH/2u;
        }
        unsigned w=n.inputDispatch?p.input.width:p.output.width,h=n.inputDispatch?p.input.height:p.output.height;
        p.input.width=pass==1 || pass==4?1920u:960u;
        p.input.height=pass==1 || pass==4?1080u:540u; // original class-sized raw neural halo
        auto cb=[r.queue commandBuffer];auto e=[cb computeCommandEncoder];[e setComputePipelineState:n.pso];
        [e setBuffer:r.scratch offset:0 atIndex:0];[e setBuffer:session.nativeWeights offset:0 atIndex:1];[e setBytes:&p length:sizeof p atIndex:2];
        if(session.halfWeights)[e setBuffer:session.halfWeights offset:0 atIndex:3];if(n.winograd)[e setBuffer:session.winogradWeights offset:0 atIndex:4];
        [e dispatchThreadgroups:MTLSizeMake((w+n.tileX-1)/n.tileX,(h+n.tileY-1)/n.tileY,1) threadsPerThreadgroup:MTLSizeMake(n.threads,1,1)];
        [e endEncoding];[cb commit];[cb waitUntilCompleted];check(cb.status==MTLCommandBufferStatusCompleted,"native operator failed: "+utf8(cb.error.localizedDescription));
        if(pass==4) r.dispatch("pass1_border");
    }
};
void input(Replay &r,unsigned frame) {
    std::vector<float> c(r.renderWidth*r.renderHeight*4),d(r.renderWidth*r.renderHeight),m(r.renderWidth*r.renderHeight*2);
    for(unsigned y=0;y<r.renderHeight;++y)for(unsigned x=0;x<r.renderWidth;++x){size_t i=size_t(y)*r.renderWidth+x;
        const float a=.05f+.5f*float(x)/r.renderWidth+.1f*float(y)/r.renderHeight;
        c[i*4]=a;c[i*4+1]=a*.7f;c[i*4+2]=a*.4f;c[i*4+3]=1;
        if((x+frame)%29==0)c[i*4]=c[i*4+1]=c[i*4+2]=.9f;
        const bool foreground=x>r.renderWidth/2;d[i]=foreground?.8f:.2f;m[i*2]=foreground?1.f/r.width:0;m[i*2+1]=0;
    }
    r.upload(r.textures["input"],c.data());r.upload(r.textures["depth"],d.data());r.upload(r.textures["velocity"],m.data());float exp[2]={1,1};r.upload(r.textures["exposure"],exp);
}
bool identical(NSData *a,NSData *b){return a.length==b.length && std::memcmp(a.bytes,b.bytes,a.length)==0;}
void tensor(Replay &a,Replay &b,const std::string &name,size_t base,size_t pitch,unsigned w,unsigned h) {
    NSData *x=a.compact_tensor(base,pitch,w,h),*y=b.compact_tensor(base,pitch,w,h);
    if(!identical(x,y)) {
        const auto aa=static_cast<const uint8_t *>(x.bytes),bb=static_cast<const uint8_t *>(y.bytes);unsigned count=0;
        for(size_t i=0;i<x.length && count<8;++i)if(aa[i]!=bb[i]){std::fprintf(stderr,"%s (%zu,%zu,%zu): %d != %d\n",name.c_str(),i/16%w,i/16/w,i%16,int(int8_t(aa[i])),int(int8_t(bb[i])));++count;}
        throw std::runtime_error(name+" differs from original neural shader");
    }
}
}
int main(int argc,char **argv) {@autoreleasepool {try {
    check(argc>=2,"usage: fsr412_upstream_native_ab ROOT [--simd|--scalar] [--frames N] [--width W --height H --render-width RW --render-height RH]");
    std::string root=argv[1];unsigned w=640,h=384,rw=384,rh=224,frames=6;auto kind=fsr411::Kernels::Mpp;bool p2O0=false,p3O0=false;
    for(int i=2;i<argc;++i){std::string a=argv[i];auto next=[&]{check(i+1<argc,"missing argument");return unsigned(std::stoul(argv[++i]));};
        if(a=="--simd")kind=fsr411::Kernels::Simd;else if(a=="--scalar")kind=fsr411::Kernels::Scalar;else if(a=="--original-p2-o0")p2O0=true;else if(a=="--original-p3-o0")p3O0=true;else if(a=="--frames")frames=next();
        else if(a=="--width")w=next();else if(a=="--height")h=next();else if(a=="--render-width")rw=next();else if(a=="--render-height")rh=next();else throw std::runtime_error("unknown option "+a);}
    Replay original(root+"/build/fsr412_upstream",w,h,rw,rh,"isolated","u8"),candidate(root+"/build/fsr412_upstream",w,h,rw,rh,"isolated","u8");
    if(p2O0) {
        NSError *error=nil;auto lib=[original.device newLibraryWithURL:[NSURL fileURLWithPath:ns(root+"/build/fsr412_upstream/p2-O0.metallib")] error:&error];
        check(lib!=nil,utf8(error.localizedDescription));auto fn=[lib newFunctionWithName:@"main0"];
        original.kernels["pass2"].pipeline=[original.device newComputePipelineStateWithFunction:fn error:&error];
        check(original.kernels["pass2"].pipeline!=nil,utf8(error.localizedDescription));
    }
    if(p3O0) {
        NSError *error=nil;auto lib=[original.device newLibraryWithURL:[NSURL fileURLWithPath:ns(root+"/build/fsr412_upstream/p3-O0.metallib")] error:&error];
        check(lib!=nil,utf8(error.localizedDescription));auto fn=[lib newFunctionWithName:@"main0"];
        original.kernels["pass3"].pipeline=[original.device newComputePipelineStateWithFunction:fn error:&error];
        check(original.kernels["pass3"].pipeline!=nil,utf8(error.localizedDescription));
    }
    Native native(candidate,root,kind);unsigned matches=0;
    for(unsigned f=0;f<frames;++f){input(original,f);input(candidate,f);float jx=f&1u?-.25f:.125f,jy=f%3?1.f/6.f:-1.f/6.f;
        original.set_frame(jx,jy,f==0);candidate.set_frame(jx,jy,f==0);
        std::fprintf(stderr,"upstream/native frame=%u\n",f);
        original.dispatch("pre");candidate.dispatch("pre");
        const uint32_t markerValue=*(const uint32_t *)((const uint8_t *)candidate.scratch.contents+33273792);
        const bool skip=markerValue==1511506142u;
        std::fprintf(stderr,"marker=%u skip=%d\n",markerValue,skip);
        original.dispatch("pass1");if(!skip){native.dispatch(candidate,1);native.dispatch(candidate,4);}
        tensor(original,candidate,"CNB",original.planeBytes+original.rowBytes+16,original.rowBytes,w/2,h/2);
        for(unsigned bank=0;bank<2;++bank) {
            tensor(original,candidate,"DS",2*original.planeBytes+bank*original.quarterPlaneBytes+original.quarterRowBytes+16,original.quarterRowBytes,w/4,h/4);
            tensor(original,candidate,"DS-with-halo",2*original.planeBytes+bank*original.quarterPlaneBytes,original.quarterRowBytes,w/4+2,h/4+2);
        }
        original.dispatch("pass2");if(!skip)native.dispatch(candidate,2);
        for(unsigned bank=0;bank<2;++bank)tensor(original,candidate,"FNB",bank*original.quarterPlaneBytes+original.quarterRowBytes+16,original.quarterRowBytes,w/4,h/4);
        original.dispatch("border");candidate.dispatch("border");
        for(unsigned bank=0;bank<2;++bank)tensor(original,candidate,"FNB-with-halo",bank*original.quarterPlaneBytes,original.quarterRowBytes,w/4+2,h/4+2);
        original.dispatch("pass3");if(!skip)native.dispatch(candidate,3);
        tensor(original,candidate,"CT",2*original.planeBytes+original.rowBytes+16,original.rowBytes,w/2,h/2);
        original.dispatch("post");candidate.dispatch("post");
        for(const std::string &name:{"output","history_next","recurrent_next","reprojected"})check(identical(original.read_texture(original.textures[name]),candidate.read_texture(candidate.textures[name])),name+" differs from original shader graph");
        original.advance_history();candidate.advance_history();++matches;
    }
    NSData *final=candidate.read_texture(candidate.textures["output"]);
    std::printf("Original Lite Lossy vs %s: %u frames, all neural interiors/output/history/recurrent identical; output %s\n",fsr411::kernels_label(kind),matches,hash(final.bytes,final.length).c_str());return 0;
} catch(const std::exception &e){std::fprintf(stderr,"Original/native parity: %s\n",e.what());return 1;} }}
