// SPDX-License-Identifier: MIT
// Actual complete-graph Metal checks, with synthetic DLSS/MetalFX-style input.
#include "../src/fsr411_runtime.h"
#include <cstdio>
#include <cstdlib>
#include <vector>

static id<MTLTexture> make_texture(id<MTLDevice> dev,MTLPixelFormat format,unsigned w,unsigned h) {
    auto d=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format width:w height:h mipmapped:NO];
    d.storageMode=MTLStorageModeShared; d.usage=MTLTextureUsageShaderRead|MTLTextureUsageShaderWrite;
    auto t=[dev newTextureWithDescriptor:d];
    if(!t) { std::fprintf(stderr,"texture allocation failed\n"); std::exit(1); }
    return t;
}
static fsr411::Frame make_frame(id<MTLDevice> dev,unsigned rw,unsigned rh,unsigned uw,unsigned uh,bool cropped,bool highMotion=false) {
    fsr411::Frame f;
    const unsigned off=cropped?3:0, tail=cropped?11:0;
    f.color=make_texture(dev,MTLPixelFormatRGBA32Float,rw+tail,rh+tail);
    f.depth=make_texture(dev,MTLPixelFormatR32Float,rw+tail,rh+tail);
    const unsigned mw=highMotion?uw:rw,mh=highMotion?uh:rh;
    f.motion=make_texture(dev,MTLPixelFormatRG32Float,mw+tail,mh+tail);
    f.output=make_texture(dev,MTLPixelFormatRGBA32Float,uw+tail,uh+tail);
    f.exposure=make_texture(dev,MTLPixelFormatR32Float,1,1);
    f.input.geometry={{rw+tail,rh+tail,off,off},{rw+tail,rh+tail,off,off},
                      {mw+tail,mh+tail,off,off},{uw+tail,uh+tail,off,off},rw+tail,rh+tail};
    f.input.rw=rw; f.input.rh=rh; f.input.uw=uw; f.input.uh=uh;
    f.input.mvScaleX=f.input.mvScaleY=1; f.input.preExposure=1; f.input.hasExposureTexture=1;
    f.input.outputResMV=highMotion;
    std::vector<float> color((rw+tail)*(rh+tail)*4,9.f),depth((rw+tail)*(rh+tail),0.9f),motion((mw+tail)*(mh+tail)*2,14.f);
    for(unsigned y=0;y<rh;++y) for(unsigned x=0;x<rw;++x) {
        const size_t pixel=(y+off)*(rw+tail)+x+off;
        color[pixel*4]=0.05f+0.8f*float(x)/rw;
        color[pixel*4+1]=0.02f+0.7f*float(y)/rh;
        color[pixel*4+2]=float(((x*7+y*3)%23)+1)/30;
        color[pixel*4+3]=1;
        depth[pixel]=0.2f+0.6f*float((x+y)%11)/11;
    }
    for(unsigned y=0;y<mh;++y) for(unsigned x=0;x<mw;++x) {
        const size_t pixel=(y+off)*(mw+tail)+x+off;
        motion[pixel*2]=float(int(x%5)-2)*0.125f; motion[pixel*2+1]=float(int(y%3)-1)*0.125f;
    }
    [f.color replaceRegion:MTLRegionMake2D(0,0,rw+tail,rh+tail) mipmapLevel:0 withBytes:color.data() bytesPerRow:(rw+tail)*16];
    [f.depth replaceRegion:MTLRegionMake2D(0,0,rw+tail,rh+tail) mipmapLevel:0 withBytes:depth.data() bytesPerRow:(rw+tail)*4];
    [f.motion replaceRegion:MTLRegionMake2D(0,0,mw+tail,mh+tail) mipmapLevel:0 withBytes:motion.data() bytesPerRow:(mw+tail)*8];
    std::vector<float> sentinel((uw+tail)*(uh+tail)*4,-123.f);
    [f.output replaceRegion:MTLRegionMake2D(0,0,uw+tail,uh+tail) mipmapLevel:0 withBytes:sentinel.data() bytesPerRow:(uw+tail)*16];
    float exposure=1.15f; [f.exposure replaceRegion:MTLRegionMake2D(0,0,1,1) mipmapLevel:0 withBytes:&exposure bytesPerRow:4];
    return f;
}
static std::vector<float> read_output(const fsr411::Frame &f) {
    std::vector<float> data(size_t(f.input.uw)*f.input.uh*4);
    auto r=f.input.geometry.output;
    [f.output getBytes:data.data() bytesPerRow:f.input.uw*16 fromRegion:MTLRegionMake2D(r.offsetX,r.offsetY,f.input.uw,f.input.uh) mipmapLevel:0];
    return data;
}
static bool run_frame(fsr411::Runtime &runtime,fsr411::View &view,fsr411::Frame &frame,
                       const fsr411::Options &options,id<MTLCommandQueue> queue,int index) {
    frame.input.reset=index==0;
    frame.input.jitterX=(index%2)?0.25f:-0.125f; frame.input.jitterY=(index%2)?-0.125f:0.25f;
    frame.input.preExposure=1.f+0.25f*index;
    float exposure=1.15f+0.5f*index;
    [frame.exposure replaceRegion:MTLRegionMake2D(0,0,1,1) mipmapLevel:0 withBytes:&exposure bytesPerRow:4];
    auto cb=[queue commandBuffer]; std::string error;
    if(!runtime.encode(cb,view,frame,options,error)) { std::fprintf(stderr,"encode failed: %s\n",error.c_str()); return false; }
    [cb commit]; [cb waitUntilCompleted];
    if(cb.status!=MTLCommandBufferStatusCompleted) { std::fprintf(stderr,"GPU error: %s\n",cb.error.localizedDescription.UTF8String); return false; }
    const auto output=read_output(frame);
    float lo=1e30f,hi=-1e30f;
    for(size_t i=0;i<output.size();++i) if(i%4!=3) {
        if(!std::isfinite(output[i])) { std::fprintf(stderr,"nonfinite output[%zu]\n",i); return false; }
        lo=std::min(lo,output[i]); hi=std::max(hi,output[i]);
    }
    std::printf("%s %ux%u -> %ux%u frame%d %.3fms [%g,%g]\n",view.backendLabel.c_str(),frame.input.rw,frame.input.rh,
                frame.input.uw,frame.input.uh,index,(cb.GPUEndTime-cb.GPUStartTime)*1000,lo,hi);
    std::fflush(stdout); return hi>lo && lo>=0;
}

// Mean |Δ| per RGB value, relaxed vs ordered PRE/POST (linear units). The
// noise fixture is a worst case: one PRE byte moves pixels by up to ~0.2, and
// these cases measured 0.0036-0.0051. Game captures stay >= 63 dB.
static constexpr double kRelaxedMeanTolerance=1e-2;

int main(int argc,char **argv) { @autoreleasepool {
    if(argc<2) { std::fprintf(stderr,"usage: fsr411_runtime_test <repository> [--compile] [--optimize] [--lossy] [--scalar|--simd] [--ultra] [--tier N] [--colorspace N] [--inverted] [--high-mv] [--jittered] [--auto-exposure] [--rcas] [--odd] [--size RW RH UW UH] [--capture-pre] [--strict-math] [--relaxed-tolerance]\n"); return 2; }
    const std::string root=argv[1]; bool smoke=false,compile=false,optimize=false,nativeSmoke=false,lossy=false,fast=false,scalar=false,simd=false,ultra=false,capture=false;
    bool highMotion=false,inverted=false,jittered=false,autoExposure=false,rcas=false,capturePre=false;
    // --strict-math: Exact Fast/Lossy with ordered PRE/POST, compared bit for bit.
    // --relaxed-tolerance: the production relaxed build against that strict reference.
    bool strictMath=false,relaxedTolerance=false;
    unsigned rw=64,rh=36,uw=128,uh=72;int tier=0,colorspace=0,frames=3;
    for(int i=2;i<argc;++i) {if(!std::strcmp(argv[i],"--smoke"))smoke=true; if(!std::strcmp(argv[i],"--compile"))compile=true; if(!std::strcmp(argv[i],"--optimize"))optimize=true;
        if(!std::strcmp(argv[i],"--native-smoke"))smoke=nativeSmoke=true;
        if(!std::strcmp(argv[i],"--lossy"))lossy=true;if(!std::strcmp(argv[i],"--fast"))fast=true;if(!std::strcmp(argv[i],"--scalar"))scalar=true;if(!std::strcmp(argv[i],"--ultra"))ultra=true;
        if(!std::strcmp(argv[i],"--capture"))capture=true;
        if(!std::strcmp(argv[i],"--simd"))simd=true;
        if(!std::strcmp(argv[i],"--strict-math"))strictMath=true;
        if(!std::strcmp(argv[i],"--relaxed-tolerance"))relaxedTolerance=true;
        if(!std::strcmp(argv[i],"--capture-pre"))capturePre=true;
        if(!std::strcmp(argv[i],"--high-mv"))highMotion=true;
        if(!std::strcmp(argv[i],"--inverted"))inverted=true;
        if(!std::strcmp(argv[i],"--jittered"))jittered=true;
        if(!std::strcmp(argv[i],"--auto-exposure"))autoExposure=true;
        if(!std::strcmp(argv[i],"--rcas"))rcas=true;
        if(!std::strcmp(argv[i],"--odd")){rw=37;rh=23;uw=65;uh=41;}
        if(!std::strcmp(argv[i],"--tier") && i+1<argc)tier=std::atoi(argv[++i]);
        if(!std::strcmp(argv[i],"--colorspace") && i+1<argc)colorspace=std::atoi(argv[++i]);
        if(!std::strcmp(argv[i],"--frames") && i+1<argc)frames=std::atoi(argv[++i]);
        if(!std::strcmp(argv[i],"--size") && i+4<argc){rw=std::atoi(argv[++i]);rh=std::atoi(argv[++i]);uw=std::atoi(argv[++i]);uh=std::atoi(argv[++i]);}}
    auto dev=MTLCreateSystemDefaultDevice(); auto queue=[dev newCommandQueue];
    if(!dev || !queue) return 2;
    fsr411::Runtime runtime(dev,root,root+"/build");
    fsr411::Options o; o.backend=fsr411::Backend::Reference; o.optimize=optimize; o.allowCompile=compile;
    o.profile=lossy?fsr411::Profile::Lossy:fast?fsr411::Profile::ExactFast:fsr411::Profile::Exact;o.ultraPerformance=ultra;
    o.tier=tier;o.colorspace=colorspace;o.rcas=rcas;o.sharpness=0.35f;
    if(capturePre)o.captureStages={"pre"};
    if(capture)o.captureStages={"pre","pass1","pass2","pass3","pass4","pass5","pass6","pass7","pass8","pass9","pass10","pass11","pass12"};
    fsr411::View reference,native,cropped;
    auto a=make_frame(dev,rw,rh,uw,uh,false,highMotion);
    auto b=make_frame(dev,rw,rh,uw,uh,false,highMotion);
    auto c=make_frame(dev,rw,rh,uw,uh,true,highMotion);
    for(auto f:{&a,&b,&c}) {f->input.depthReversed=inverted;f->input.jitteredMV=jittered;f->input.autoExposure=autoExposure;}
    for(int i=0;i<(smoke?1:frames);++i) {
        o.backend=nativeSmoke?fsr411::Backend::Native:fsr411::Backend::Reference;
        o.optimize=nativeSmoke?int(optimize):0;o.strictMath=strictMath||relaxedTolerance;
        if(!run_frame(runtime,reference,a,o,queue,i)) return 1;
        if(smoke) break;
        o.backend=simd?fsr411::Backend::Simd:scalar?fsr411::Backend::Scalar:fsr411::Backend::Native;o.optimize=optimize;o.strictMath=strictMath;
        if(!run_frame(runtime,native,b,o,queue,i)) return 1;
        if(simd && native.backendLabel!="int8-simd") {std::fprintf(stderr,"expected int8-simd, ran %s\n",native.backendLabel.c_str());return 1;}
        if(!run_frame(runtime,cropped,c,o,queue,i)) return 1;
        const auto ar=read_output(a),br=read_output(b),cr=read_output(c);
        size_t differences=0,cropDifferences=0,nonFinite=0; float maxDifference=0; double sumDifference=0;
        for(size_t j=0;j<ar.size();++j) if(j%4!=3) {
            differences+=ar[j]!=br[j]; cropDifferences+=br[j]!=cr[j]; nonFinite+=!std::isfinite(br[j]);
            maxDifference=std::max(maxDifference,std::abs(ar[j]-br[j])); sumDifference+=std::abs(ar[j]-br[j]);
        }
        const double meanDifference=sumDifference/double(ar.size()/4*3);
        std::printf("parity frame%d native differences=%zu max=%g mean=%g, crop differences=%zu\n",i,differences,maxDifference,meanDifference,cropDifferences);
        // Relaxed PRE/POST reorders additions, so a few INT8 PRE values and
        // their downstream pixels move. The crop must still match exactly.
        if(relaxedTolerance) {
            if(nonFinite || cropDifferences || meanDifference>kRelaxedMeanTolerance) {
                std::fprintf(stderr,"relaxed math outside tolerance: nonfinite=%zu mean=%g\n",nonFinite,meanDifference);return 1;}
        } else if(differences || cropDifferences) {
            if(capture || capturePre) {
                for(const auto &snapshot:reference.captures) {
                    const auto bn=native.captures.find(snapshot.first);if(bn==native.captures.end())continue;
                    auto ap=(const uint8_t *)snapshot.second.contents,bp=(const uint8_t *)bn->second.contents;
                    size_t n=0,first=0;
                    for(size_t k=0;k<snapshot.second.length;++k)if(ap[k]!=bp[k]){if(!n)first=k;++n;}
                    std::printf("scratch %s differences=%zu first=%zu %u/%u\n",snapshot.first.c_str(),n,first,ap[first],bp[first]);
                }
            }
            return 1;
        }
        if((capture || capturePre) && !native.skippedInference && !relaxedTolerance) {
            const auto ap=(const uint8_t *)reference.captures.at("pre").contents;
            const auto bp=(const uint8_t *)native.captures.at("pre").contents;
            const unsigned activeW=native.prepared.alignedUW/2,activeH=native.prepared.alignedUH/2;
            const unsigned stride=((tier?tier:fsr411::required_tier(native.prepared.alignedUW,native.prepared.alignedUH))*8/9+2)*16;
            for(unsigned y=0;y<activeH;++y) for(unsigned x=0;x<activeW*16;++x)
                if(ap[(y+1)*stride+16+x]!=bp[(y+1)*stride+16+x]) {std::fprintf(stderr,"PRE INT8 mismatch at row%u byte%u\n",y,x);return 1;}
        }
        auto ea=make_texture(dev,MTLPixelFormatR32Float,2,1),eb=make_texture(dev,MTLPixelFormatR32Float,2,1),ec=make_texture(dev,MTLPixelFormatR32Float,2,1);
        auto readback=[queue commandBuffer];auto blit=[readback blitCommandEncoder];
        [blit copyFromTexture:reference.autoExposure toTexture:ea];
        [blit copyFromTexture:native.autoExposure toTexture:eb];
        [blit copyFromTexture:cropped.autoExposure toTexture:ec];
        [blit endEncoding];[readback commit];[readback waitUntilCompleted];
        float ae[2],be[2],ce[2];
        [ea getBytes:ae bytesPerRow:8 fromRegion:MTLRegionMake2D(0,0,2,1) mipmapLevel:0];
        [eb getBytes:be bytesPerRow:8 fromRegion:MTLRegionMake2D(0,0,2,1) mipmapLevel:0];
        [ec getBytes:ce bytesPerRow:8 fromRegion:MTLRegionMake2D(0,0,2,1) mipmapLevel:0];
        for(int k=0;k<2;++k) if(!std::isfinite(ae[k]) || ae[k]<=0 || ae[k]!=be[k] || be[k]!=ce[k]) {std::fprintf(stderr,"exposure mismatch or invalid: %g %g %g\n",ae[k],be[k],ce[k]);return 1;}
        if(!autoExposure) {
            const float expectedCurrent=1.15f+0.5f*i,expectedPrevious=i?1.15f+0.5f*(i-1):expectedCurrent;
            if(ae[0]!=expectedCurrent || ae[1]!=expectedPrevious) {std::fprintf(stderr,"exposure pair is not current/previous: %g/%g\n",ae[0],ae[1]);return 1;}
        }
        std::vector<float> entire(c.output.width*c.output.height*4);
        [c.output getBytes:entire.data() bytesPerRow:c.output.width*16 fromRegion:MTLRegionMake2D(0,0,c.output.width,c.output.height) mipmapLevel:0];
        for(unsigned y=0;y<c.output.height;++y) for(unsigned x=0;x<c.output.width;++x)
            if(x<3 || y<3 || x>=3+c.input.uw || y>=3+c.input.uh)
                for(unsigned k=0;k<4;++k) if(entire[(y*c.output.width+x)*4+k]!=-123.f) { std::fprintf(stderr,"output subrect clobber\n"); return 1; }
    }
    std::puts("FSR 4.1.1 full graph checks passed"); return 0;
} }
