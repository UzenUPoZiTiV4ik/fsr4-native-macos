// SPDX-License-Identifier: MIT
// The shared float32 color cache must preserve the model across changing
// exposure, input/output sizes, subrects, resets and RCAS on both encoder APIs.
// Native optimize=0 must also match AMD's Lite shaders translated to Metal
// (Backend::Reference) bit for bit, including the Ultra Performance model.
#include "../src/fsr412_runtime.h"
#include <cstdio>
#include <stdexcept>

namespace {
void require(bool value,const std::string &message) {if(!value) throw std::runtime_error(message);}
id<MTLTexture> texture(id<MTLDevice> device,MTLPixelFormat format,unsigned w,unsigned h) {
    auto d=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format width:w height:h mipmapped:NO];
    d.storageMode=MTLStorageModeShared; d.usage=MTLTextureUsageShaderRead|MTLTextureUsageShaderWrite;
    auto result=[device newTextureWithDescriptor:d]; require(result!=nil,"texture allocation failed"); return result;
}
fsr412::Frame frame(id<MTLDevice> device,unsigned index,bool ae) {
    fsr412::Frame f; auto &in=f.input;
    in.rw=(ae?137u:61u)+(index%3)*2u; in.rh=ae?83u:39u;
    in.uw=(ae?257u:127u)+(index>=4?2u:0u); in.uh=ae?161u:79u;
    in.geometry.color={in.rw+5,in.rh+3,3,2};in.geometry.depth=in.geometry.color;
    const unsigned mw=ae?in.uw:in.rw,mh=ae?in.uh:in.rh;
    in.geometry.motion={mw+5,mh+3,3,2};in.geometry.output={in.uw+5,in.uh+3,3,2};
    in.jitterX=float(int(index%4)-2)/8.f;in.jitterY=float(int(index%3)-1)/6.f;
    in.mvScaleX=-1;in.mvScaleY=1;in.depthReversed=ae;in.jitteredMV=ae;in.outputResMV=ae;
    in.autoExposure=ae;in.hasExposureTexture=1;in.reset=index==0 || index==3;
    in.preExposure=index>=3?.625f:1.25f;
    auto g=in.geometry;
    f.color=texture(device,MTLPixelFormatRGBA16Float,g.color.width,g.color.height);
    f.depth=texture(device,MTLPixelFormatR32Float,g.depth.width,g.depth.height);
    f.motion=texture(device,MTLPixelFormatRG16Float,g.motion.width,g.motion.height);
    f.output=texture(device,MTLPixelFormatRGBA16Float,g.output.width,g.output.height);
    f.exposure=texture(device,MTLPixelFormatR32Float,1,1);
    std::vector<_Float16> c(size_t(g.color.width)*g.color.height*4,(_Float16)7.f);
    std::vector<float> d(size_t(g.depth.width)*g.depth.height,1.f);
    for(unsigned y=0;y<in.rh;++y) for(unsigned x=0;x<in.rw;++x) {
        const auto i=(size_t(y+2)*g.color.width+x+3)*4;
        c[i]=(_Float16)(in.preExposure*(.015f+float((x+index)%23)/17.f));
        c[i+1]=(_Float16)(in.preExposure*(.025f+float((y+index)%19)/7.f));
        c[i+2]=(_Float16)(in.preExposure*(.012f+float((x+y+index)%17)/13.f));c[i+3]=1;
        d[size_t(y+2)*g.depth.width+x+3]=.1f+float((x+y)%17)/21.f;
    }
    std::vector<_Float16> m(size_t(g.motion.width)*g.motion.height*2,(_Float16)0);
    for(unsigned y=0;y<mh;++y) for(unsigned x=0;x<mw;++x) {
        const auto i=(size_t(y+2)*g.motion.width+x+3)*2;
        m[i]=(_Float16)(-.25f+float((x+index)%7)/32.f);m[i+1]=(_Float16).125f;
    }
    [f.color replaceRegion:MTLRegionMake2D(0,0,g.color.width,g.color.height) mipmapLevel:0 withBytes:c.data() bytesPerRow:g.color.width*8];
    [f.depth replaceRegion:MTLRegionMake2D(0,0,g.depth.width,g.depth.height) mipmapLevel:0 withBytes:d.data() bytesPerRow:g.depth.width*4];
    [f.motion replaceRegion:MTLRegionMake2D(0,0,g.motion.width,g.motion.height) mipmapLevel:0 withBytes:m.data() bytesPerRow:g.motion.width*4];
    float exposure=index>=2?.375f:.75f;
    [f.exposure replaceRegion:MTLRegionMake2D(0,0,1,1) mipmapLevel:0 withBytes:&exposure bytesPerRow:4];
    return f;
}
void compare(id<MTLTexture> a,id<MTLTexture> b,unsigned bytes,const char *name,bool exact=false) {
    require(a.width==b.width && a.height==b.height,"texture dimensions differ");
    const auto pitch=(a.width*bytes+255u)&~255u;
    auto first=[a.device newBufferWithLength:pitch*a.height options:MTLResourceStorageModeShared];
    auto second=[a.device newBufferWithLength:pitch*a.height options:MTLResourceStorageModeShared];
    auto cb=[[a.device newCommandQueue] commandBuffer];auto blit=[cb blitCommandEncoder];
    for(auto pair:{std::make_pair(a,first),std::make_pair(b,second)})
        [blit copyFromTexture:pair.first sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0,0,0)
            sourceSize:MTLSizeMake(a.width,a.height,1) toBuffer:pair.second destinationOffset:0
            destinationBytesPerRow:pitch destinationBytesPerImage:pitch*a.height];
    [blit endEncoding];[cb commit];[cb waitUntilCompleted];require(cb.status==MTLCommandBufferStatusCompleted,"readback failed");
    std::vector<uint8_t> x(a.width*a.height*bytes),y(x.size());
    for(size_t row=0;row<a.height;++row) {
        std::memcpy(x.data()+row*a.width*bytes,(const uint8_t *)first.contents+row*pitch,a.width*bytes);
        std::memcpy(y.data()+row*a.width*bytes,(const uint8_t *)second.contents+row*pitch,a.width*bytes);
    }
    if(bytes==8) {
        double sum=0;float maximum=0;size_t count=x.size()/2;
        for(size_t i=0;i<count;++i) {
            _Float16 a,b;std::memcpy(&a,x.data()+2*i,2);std::memcpy(&b,y.data()+2*i,2);
            require(std::isfinite(float(a)) && std::isfinite(float(b)),"nonfinite output");
            const float error=std::abs(float(a)-float(b));sum+=error;maximum=std::max(maximum,error);
        }
        if(x!=y)std::printf("  %s: max=%g mean=%g\n",name,maximum,sum/count);
        require(exact ? x==y : maximum<=.005f && sum/count<=1.e-5,std::string(name)+" differs from its FSR 4.1.2 reference");
    } else require(x==y,std::string(name)+" differs from its FSR 4.1.2 reference");
}
void encode(fsr412::Runtime &runtime,id<MTLDevice> dev,id<MTLCommandQueue> queue,
            fsr412::View &view,const fsr412::Frame &f,const fsr412::Options &o,bool metal4) {
    std::string error;
    if(metal4) {
        if(@available(macOS 26.0,*)) {
            auto q=[dev newMTL4CommandQueue];auto alloc=[dev newCommandAllocator];auto cb=[dev newCommandBuffer];
            [cb beginCommandBufferWithAllocator:alloc];
            require(runtime.encode(cb,view,f,o,error),error);
            [cb endCommandBuffer];id<MTL4CommandBuffer> list[]={cb};[q commit:list count:1];
            auto ev=[dev newSharedEvent];[q signalEvent:ev value:1];
            require([ev waitUntilSignaledValue:1 timeoutMS:20000],"Metal 4 GPU timeout");
        }
    } else {
        auto cb=[queue commandBuffer];require(runtime.encode(cb,view,f,o,error),error);
        [cb commit];[cb waitUntilCompleted];require(cb.status==MTLCommandBufferStatusCompleted,fsr412::metal_error(cb.error));
    }
}
}
#ifndef FSR412_TEST_HELPERS_ONLY
int main(int argc,char **argv) {
    @autoreleasepool {try {
        require(argc>=2,"usage: fsr412_optimization_test <root> [mtl4] [--simd|--scalar]");
        bool metal4=false; auto backend=fsr412::Backend::Native;
        for(int i=2;i<argc;++i) {
            if(!std::strcmp(argv[i],"mtl4")) metal4=true;
            else if(!std::strcmp(argv[i],"--simd")) backend=fsr412::Backend::Simd;
            else if(!std::strcmp(argv[i],"--scalar")) backend=fsr412::Backend::Scalar;
            else throw std::runtime_error(std::string("unknown option: ")+argv[i]);
        }
        auto dev=MTLCreateSystemDefaultDevice();auto queue=[dev newCommandQueue];require(dev && queue,"Metal unavailable");
        if(metal4) {if(@available(macOS 26.0,*)) {} else throw std::runtime_error("Metal 4 requires macOS 26+");}
        fsr412::Runtime runtime(dev,argv[1],std::string(argv[1])+"/build");
        for(unsigned mode=0;mode<3;++mode) {
            fsr412::View dll,reference,optimized;
            // Preallocate dirty scratch to verify that every path initializes
            // reused buffers correctly before the first frame.
            dll.scratch=[dev newBufferWithLength:25027392 options:MTLResourceStorageModePrivate];
            reference.scratch=[dev newBufferWithLength:25027392 options:MTLResourceStorageModePrivate];
            optimized.scratch=[dev newBufferWithLength:25027392 options:MTLResourceStorageModePrivate];
            auto cb=[queue commandBuffer];auto blit=[cb blitCommandEncoder];
            [blit fillBuffer:dll.scratch range:NSMakeRange(0,25027392) value:0x3c];
            [blit fillBuffer:reference.scratch range:NSMakeRange(0,25027392) value:0x59];
            [blit fillBuffer:optimized.scratch range:NSMakeRange(0,25027392) value:0xa7];
            [blit endEncoding];[cb commit];[cb waitUntilCompleted];
            fsr412::Options options;options.backend=backend;options.ultraPerformance=mode==2;
            for(unsigned index=0;index<6;++index) {
                auto a=frame(dev,index,mode==1),b=a,s=a;
                b.output=texture(dev,MTLPixelFormatRGBA16Float,a.output.width,a.output.height);
                s.output=texture(dev,MTLPixelFormatRGBA16Float,a.output.width,a.output.height);
                std::vector<_Float16> sentinel(a.output.width*a.output.height*4,(_Float16)-9);
                for(auto t:{a.output,b.output,s.output})[t replaceRegion:MTLRegionMake2D(0,0,t.width,t.height) mipmapLevel:0 withBytes:sentinel.data() bytesPerRow:t.width*8];
                options.rcas=index%2;options.sharpness=.498f;options.optimize=0;
                options.backend=fsr412::Backend::Reference;encode(runtime,dev,queue,dll,s,options,metal4);
                options.backend=backend;
                encode(runtime,dev,queue,reference,a,options,metal4);
                options.optimize=1;encode(runtime,dev,queue,optimized,b,options,metal4);
                std::printf("checking mode=%u frame=%u\n",mode,index);std::fflush(stdout);
                compare(s.output,a.output,8,"DLL shader output",true);compare(dll.history,reference.history,8,"DLL shader history",true);
                compare(dll.recurrent,reference.recurrent,4,"DLL shader recurrent",true);compare(dll.autoExposure,reference.autoExposure,4,"DLL shader exposure",true);
                compare(a.output,b.output,8,"output");compare(reference.history,optimized.history,8,"history");
                compare(reference.recurrent,optimized.recurrent,4,"recurrent");compare(reference.autoExposure,optimized.autoExposure,4,"exposure");
                if(backend==fsr412::Backend::Simd)
                    require(reference.backendLabel=="int8-simd" && optimized.backendLabel=="int8-simd","requested SIMD backend was not used");
                require(optimized.preparedColor && !reference.preparedColor,"color cache mode not selected");
                require(!optimized.skippedInference && optimized.neuralRuns==index+1,"Lite inference was reused");
            }
            std::printf("FSR 4.1.2 %s: mode %u, 6 frames match DLL shaders and the color cache\n",metal4?"Metal 4":"classic",mode);
        }
        return 0;
    } catch(const std::exception &e) {std::fprintf(stderr,"FSR 4.1.2 optimization: %s\n",e.what());return 1;}}
}
#endif
