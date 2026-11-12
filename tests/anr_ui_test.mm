// SPDX-License-Identifier: MIT
// Shader-only UI conformance: generated world differs from both real inputs.
// No model or game frames are loaded; copying the whole UI frame cannot pass.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "anr_live.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>
using Pixel=std::array<float,4>;
struct Convert {uint32_t w,h,rw,rh,encoding,reversed;float whiteNits;uint32_t uiMode;float uiNoiseFloor;};
static_assert(sizeof(Convert)==36,"helper constants ABI");
static float noiseFloor(MTLPixelFormat f){return f==MTLPixelFormatRGB10A2Unorm?2.5f/1023.f:f==MTLPixelFormatRGBA16Float?1.e-4f:2.5f/255.f;}
static void need(bool ok,const char *why){if(!ok)throw std::runtime_error(why);}
static float srgb(float x){return x<=.0031308f?12.92f*x:1.055f*std::pow(x,1.f/2.4f)-.055f;}
static float linear(float x){return x<=.04045f?x/12.92f:std::pow((x+.055f)/1.055f,2.4f);}
static float pq(float x){
    float p=std::pow(std::max(x,0.f)*203.f/10000.f,2610.f/16384.f);
    return std::pow((3424.f/4096.f+(2413.f/128.f)*p)/(1.f+(2392.f/128.f)*p),2523.f/32.f);
}
static size_t bpp(MTLPixelFormat f){return f==MTLPixelFormatRGBA16Float?8:f==MTLPixelFormatRGBA32Float?16:4;}
static unsigned code(float x,unsigned max){return (unsigned)std::lround(std::clamp(x,0.f,1.f)*max);}
static void store(uint8_t *out,MTLPixelFormat format,Pixel p){
    if(format==MTLPixelFormatRGBA16Float){_Float16 h[4];for(unsigned c=0;c<4;++c)h[c]=(_Float16)p[c];memcpy(out,h,8);}
    else if(format==MTLPixelFormatRGBA32Float)memcpy(out,p.data(),16);
    else if(format==MTLPixelFormatRGB10A2Unorm){uint32_t word=code(p[0],1023)|(code(p[1],1023)<<10)|(code(p[2],1023)<<20)|(code(p[3],3)<<30);memcpy(out,&word,4);}
    else{for(unsigned c=0;c<4;++c)out[c]=(uint8_t)code(c<3&&format==MTLPixelFormatRGBA8Unorm_sRGB?srgb(p[c]):p[c],255);}
}
static Pixel load(const uint8_t *in,MTLPixelFormat format){
    Pixel p{};
    if(format==MTLPixelFormatRGBA16Float){_Float16 h[4];memcpy(h,in,8);for(unsigned c=0;c<4;++c)p[c]=(float)h[c];}
    else if(format==MTLPixelFormatRGB10A2Unorm){uint32_t word;memcpy(&word,in,4);for(unsigned c=0;c<3;++c)p[c]=float((word>>(10*c))&1023)/1023;p[3]=float(word>>30)/3;}
    else{for(unsigned c=0;c<4;++c)p[c]=c<3&&format==MTLPixelFormatRGBA8Unorm_sRGB?linear(float(in[c])/255):float(in[c])/255;}
    return p;
}
static id<MTLTexture> texture(id<MTLDevice> device,unsigned w,unsigned h,MTLPixelFormat f){
    auto d=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:f width:w height:h mipmapped:NO];
    d.storageMode=MTLStorageModeShared;d.usage=MTLTextureUsageShaderRead|MTLTextureUsageShaderWrite;
    auto t=[device newTextureWithDescriptor:d];need(t!=nil,"texture allocation failed");return t;
}
int main(int argc,char **argv){@autoreleasepool{try{
    need(argc>=2,"usage: anr_ui_test <root> [classic|metal4]");
    const bool metal4=argc>2&&!strcmp(argv[2],"metal4");
    auto device=MTLCreateSystemDefaultDevice();need(device!=nil,"no Metal device");
    NSError *error=nil;
    auto path=[[NSString stringWithUTF8String:argv[1]] stringByAppendingPathComponent:@"build/anr_helpers_ui_v2.metallib"];
    auto library=[device newLibraryWithURL:[NSURL fileURLWithPath:path] error:&error];need(library!=nil,"UI helper library missing");
    auto pso=[device newComputePipelineStateWithFunction:[library newFunctionWithName:@"anr_export"] error:&error];need(pso!=nil,"UI export pipeline missing");
    auto queue=[device newCommandQueue];auto q4=[device newMTL4CommandQueue];auto allocator=[device newCommandAllocator];
    auto event=[device newSharedEvent];uint64_t token=0;unsigned cases=0;double maxError=0;
    constexpr unsigned w=13,h=9,ow=16,oh=12;
    for(auto format:{MTLPixelFormatRGB10A2Unorm,MTLPixelFormatRGBA16Float,MTLPixelFormatRGBA8Unorm_sRGB,MTLPixelFormatRGBA8Unorm}){
        const size_t stride=bpp(format);
        for(unsigned variant=0;variant<4;++variant){
            const auto mode=variant==0?FSR4_ANR_UI_NONE:variant==3?FSR4_ANR_UI_OVERLAY:FSR4_ANR_UI_PRECOMPOSITED;
            const bool noisy=variant==2;
            auto clean=texture(device,w,h,format),ui=texture(device,w,h,format),out=texture(device,ow,oh,format);
            auto generated=texture(device,w,h,MTLPixelFormatRGBA32Float);
            std::vector<uint8_t> cleanBytes(w*h*stride),uiBytes(cleanBytes.size()),initial(ow*oh*stride),pqBytes(w*h*16);
            std::vector<Pixel> expected(w*h);
            for(unsigned y=0;y<h;++y)for(unsigned x=0;x<w;++x){
                const size_t i=size_t(y)*w+x;
                Pixel c={.13f+x*.005f,.31f+y*.004f,.68f,1},g={.43f,.17f+x*.003f,.48f,1};
                if(format==MTLPixelFormatRGBA8Unorm&&x==0&&y==0)c={1.f/255,1.f/255,1.f/255,1};
                Pixel u=c;
                const bool hud=x>=3&&x<=7&&y>=3&&y<=6;
                if(mode==FSR4_ANR_UI_OVERLAY){
                    // Poison RGB at zero opacity must not leak into the output.
                    float alpha=hud?((x+y)%2?.75f:.25f):0.f;
                    u=hud?Pixel{.91f*alpha,.03f*alpha,.63f*alpha,alpha}:Pixel{.8f,.7f,.6f,0};
                }else if(hud)u={.91f,.03f,.63f,0}; // Composite alpha is deliberately zero.
                else if(mode==FSR4_ANR_UI_PRECOMPOSITED&&noisy){
                    // Small full-screen differences are not a HUD. The 8-bit
                    // near-black control removes the last code bit: relative
                    // contrast alone must not turn this noise into opaque UI.
                    float amount=format==MTLPixelFormatRGB10A2Unorm?1.f/1023.f:
                        format==MTLPixelFormatRGBA16Float?.0005f:format==MTLPixelFormatRGBA8Unorm?1.f/255.f:.0025f;
                    u[0]+=((x+y)&1u)?amount:-amount;
                }
                if(format==MTLPixelFormatRGBA16Float){
                    if(x==1&&y==1)u[0]=NAN; // Invalid RGB: ignore the overlay, preserve ANR.
                    if(hud&&x==3&&y==3)u[3]=NAN; // Composite alpha is not an opacity mask.
                }
                store(cleanBytes.data()+i*stride,format,c);store(uiBytes.data()+i*stride,format,u);
                auto storedUI=load(uiBytes.data()+i*stride,format);
                Pixel want=g;
                if(mode==FSR4_ANR_UI_PRECOMPOSITED){
                    auto scene=load(cleanBytes.data()+i*stride,format);
                    if(std::isfinite(storedUI[0])&&std::isfinite(storedUI[1])&&std::isfinite(storedUI[2])){
                        float alpha=0,delta=0,magnitude=1;
                        for(unsigned ch=0;ch<3;++ch){
                            float difference=storedUI[ch]-scene[ch];delta=std::max(delta,fabsf(difference));
                            magnitude=std::max(magnitude,std::max(fabsf(scene[ch]),fabsf(storedUI[ch])));
                            float room=difference>=0?1.f-scene[ch]:scene[ch];
                            alpha=std::max(alpha,fabsf(difference)/std::max(room,1.e-7f));
                        }
                        float floor=noiseFloor(format)*magnitude;
                        float t=std::clamp((delta-floor)/(7.f*floor),0.f,1.f);
                        alpha=std::clamp(alpha*1.2f,0.f,1.f)*t*t*(3.f-2.f*t);
                        for(unsigned ch=0;ch<3;++ch)want[ch]=storedUI[ch]+(1-alpha)*(g[ch]-scene[ch]);
                    }
                }
                else if(mode==FSR4_ANR_UI_OVERLAY&&hud){
                    if(std::isfinite(storedUI[3]))for(unsigned ch=0;ch<3;++ch)want[ch]=storedUI[ch]+(1.f-storedUI[3])*g[ch];
                }
                expected[i]=want;
                Pixel input=g;
                if(format!=MTLPixelFormatRGB10A2Unorm)for(unsigned ch=0;ch<3;++ch)input[ch]=pq(format==MTLPixelFormatRGBA8Unorm?linear(input[ch]):input[ch]);
                store(pqBytes.data()+i*16,MTLPixelFormatRGBA32Float,input);
            }
            for(size_t i=0;i<ow*oh;++i)store(initial.data()+i*stride,format,{.61f,.62f,.63f,1});
            [clean replaceRegion:MTLRegionMake2D(0,0,w,h) mipmapLevel:0 withBytes:cleanBytes.data() bytesPerRow:w*stride];
            [ui replaceRegion:MTLRegionMake2D(0,0,w,h) mipmapLevel:0 withBytes:uiBytes.data() bytesPerRow:w*stride];
            [generated replaceRegion:MTLRegionMake2D(0,0,w,h) mipmapLevel:0 withBytes:pqBytes.data() bytesPerRow:w*16];
            [out replaceRegion:MTLRegionMake2D(0,0,ow,oh) mipmapLevel:0 withBytes:initial.data() bytesPerRow:ow*stride];
            uint32_t zeros[5]={};auto stats=[device newBufferWithBytes:zeros length:sizeof zeros options:MTLResourceStorageModeShared];
            Convert constants={w,h,w,h,format==MTLPixelFormatRGB10A2Unorm?2u:format==MTLPixelFormatRGBA8Unorm?1u:0u,0,203,
                (uint32_t)mode|(mode==FSR4_ANR_UI_PRECOMPOSITED?256u:0u),noiseFloor(format)};
            auto cbConstants=[device newBufferWithBytes:&constants length:sizeof(constants) options:MTLResourceStorageModeShared];
            if(metal4){
                auto cb=[device newCommandBuffer];[allocator reset];[cb beginCommandBufferWithAllocator:allocator];
                auto rd=[MTLResidencySetDescriptor new];rd.initialCapacity=6;
                auto residency=[device newResidencySetWithDescriptor:rd error:&error];need(residency!=nil,"residency allocation failed");
                for(id r in @[clean,ui,out,generated,cbConstants,stats])[residency addAllocation:r];[residency commit];[cb useResidencySet:residency];
                auto td=[MTL4ArgumentTableDescriptor new];td.maxBufferBindCount=2;td.maxTextureBindCount=4;
                auto args=[device newArgumentTableWithDescriptor:td error:&error];need(args!=nil,"argument table allocation failed");
                [args setAddress:cbConstants.gpuAddress atIndex:0];[args setAddress:stats.gpuAddress atIndex:1];id<MTLTexture> textures[]={generated,out,clean,ui};
                for(unsigned i=0;i<4;++i)[args setTexture:textures[i].gpuResourceID atIndex:i];
                auto enc=[cb computeCommandEncoder];[enc setComputePipelineState:pso];[enc setArgumentTable:args];
                [enc dispatchThreads:MTLSizeMake(32,16,1) threadsPerThreadgroup:MTLSizeMake(16,8,1)];[enc endEncoding];[cb endCommandBuffer];
                id<MTL4CommandBuffer> buffers[]={cb};[q4 commit:buffers count:1];[q4 signalEvent:event value:++token];
                need([event waitUntilSignaledValue:token timeoutMS:10000],"Metal4 UI timeout");
            }else{
                auto cb=[queue commandBuffer];auto enc=[cb computeCommandEncoder];[enc setComputePipelineState:pso];[enc setBuffer:cbConstants offset:0 atIndex:0];[enc setBuffer:stats offset:0 atIndex:1];
                id<MTLTexture> textures[]={generated,out,clean,ui};for(unsigned i=0;i<4;++i)[enc setTexture:textures[i] atIndex:i];
                [enc dispatchThreads:MTLSizeMake(32,16,1) threadsPerThreadgroup:MTLSizeMake(16,8,1)];[enc endEncoding];[cb commit];[cb waitUntilCompleted];
                need(cb.status==MTLCommandBufferStatusCompleted,"Metal3 UI GPU failure");
            }
            std::vector<uint8_t> result(initial.size()),unchanged(cleanBytes.size());
            [out getBytes:result.data() bytesPerRow:ow*stride fromRegion:MTLRegionMake2D(0,0,ow,oh) mipmapLevel:0];
            const double tolerance=format==MTLPixelFormatRGBA8Unorm_sRGB?.009:format==MTLPixelFormatRGBA16Float?.002:.002;
            for(unsigned y=0;y<oh;++y)for(unsigned x=0;x<ow;++x){
                const size_t i=size_t(y)*ow+x;
                if(x>=w||y>=h){need(!memcmp(result.data()+i*stride,initial.data()+i*stride,stride),"UI export overwrote a canary margin");continue;}
                auto actual=load(result.data()+i*stride,format);const auto &want=expected[size_t(y)*w+x];
                for(unsigned ch=0;ch<3;++ch){double diff=fabs(double(actual[ch])-want[ch]);maxError=std::max(maxError,diff);
                    if(!std::isfinite(diff)||diff>tolerance){fprintf(stderr,"format=%lu mode=%u (%u,%u) ch=%u got=%f want=%f\n",(unsigned long)format,(unsigned)mode,x,y,ch,actual[ch],want[ch]);need(false,"UI/world recomposition mismatch");}}
            }
            [clean getBytes:unchanged.data() bytesPerRow:w*stride fromRegion:MTLRegionMake2D(0,0,w,h) mipmapLevel:0];need(unchanged==cleanBytes,"clean scene was modified");
            [ui getBytes:unchanged.data() bytesPerRow:w*stride fromRegion:MTLRegionMake2D(0,0,w,h) mipmapLevel:0];need(unchanged==uiBytes,"UI input was modified");
            const auto *observed=(const uint32_t*)stats.contents;
            if(mode==FSR4_ANR_UI_PRECOMPOSITED){need(observed[0]==1,"UI sampling did not run");need(observed[3]==0,"non-HUD world sample marked opaque");}
            ++cases;
        }
    }
    printf("ANR UI %s conformance passed: %u cases, PQ10/HDR/sRGB/SDR, grain/near-black phase controls, sampled stats, canaries (max error %.6f)\n",metal4?"Metal4":"Metal3",cases,maxError);
    return 0;
}catch(const std::exception &e){fprintf(stderr,"ANR UI test: %s\n",e.what());return 1;}}}
