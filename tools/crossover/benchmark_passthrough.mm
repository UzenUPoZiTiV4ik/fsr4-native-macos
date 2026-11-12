// SPDX-License-Identifier: MIT
// Compare the original MetalFX FG with an optionally loaded combined hook.
// No game, SR network, ANR model or presentation FPS is measured here.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalFX/MetalFX.h>
#import <objc/runtime.h>
#import <CommonCrypto/CommonDigest.h>
#include "anr_live.h"
#include "fsr4_fg_settings.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>
#include <mach/mach_time.h>
#include <stdexcept>
#include <vector>
static void need(bool ok,const char *why){if(!ok)throw std::runtime_error(why);}
static id<MTLTexture> texture(id<MTLDevice> device,unsigned w,unsigned h,MTLPixelFormat format,bool privateMode=false){
    auto d=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format width:w height:h mipmapped:NO];
    d.usage=MTLTextureUsageShaderRead|MTLTextureUsageShaderWrite|MTLTextureUsageRenderTarget;
    d.storageMode=privateMode?MTLStorageModePrivate:MTLStorageModeShared;auto t=[device newTextureWithDescriptor:d];need(t!=nil,"texture allocation failed");return t;
}
static double percentile(std::vector<double> values,double p){std::sort(values.begin(),values.end());return values[std::min(values.size()-1,size_t(p*(values.size()-1)))];}
int main(int argc,char **argv){@autoreleasepool{try{
    need(argc>=3,"usage: benchmark_passthrough <classic|metal4> <out.json> [hook.dylib]");
    const bool m4=!strcmp(argv[1],"metal4"),loaded=argc>3;
    setenv("FSR4_FG","metalfx",1);setenv("FSR4_REPLACE","0",1);setenv("FSR4_PANEL","0",1);setenv("FSR4_ANR_STATUS_DISABLED","0",1);
    char directory[]="/tmp/fsr4-passthrough-XXXXXX";need(mkdtemp(directory)!=nullptr,"temp directory failed");
    setenv("FSR4_ROOT",directory,1);
    need(!mkdir((std::string(directory)+"/dist").c_str(),0700)&&!mkdir((std::string(directory)+"/dist/crossover").c_str(),0700),"native status fixture failed");
    const std::string settings=std::string(directory)+"/fg-settings.conf";
    need(fsr4_fg::write_settings(settings,fsr4_fg::Mode::MetalFX),"cannot write native-mode fixture");setenv("FSR4_FG_SETTINGS",settings.c_str(),1);
    auto device=MTLCreateSystemDefaultDevice();need(device!=nil,"no Metal device");NSError *error=nil;
    auto compiler=[device newCompilerWithDescriptor:[MTL4CompilerDescriptor new] error:&error];
    constexpr unsigned w=1920,h=1080;
    auto d=[MTLFXFrameInterpolatorDescriptor new];d.inputWidth=d.outputWidth=w;d.inputHeight=d.outputHeight=h;
    d.colorTextureFormat=d.outputTextureFormat=d.uiTextureFormat=MTLPixelFormatRGB10A2Unorm;
    d.depthTextureFormat=MTLPixelFormatR32Float;d.motionTextureFormat=MTLPixelFormatRG16Float;
    id baseline=nil;if(m4)baseline=[d newFrameInterpolatorWithDevice:device compiler:compiler];else baseline=[d newFrameInterpolatorWithDevice:device];need(baseline!=nil,"no baseline MetalFX FG");
    const IMP nativeFG=method_getImplementation(class_getInstanceMethod(object_getClass(baseline),@selector(encodeToCommandBuffer:)));
    const IMP nativeSR=method_getImplementation(class_getInstanceMethod(MTLFXTemporalScalerDescriptor.class,@selector(newTemporalScalerWithDevice:)));
    void *library=nullptr;
    if(loaded){library=dlopen(argv[3],RTLD_NOW|RTLD_LOCAL);if(!library){fprintf(stderr,"%s\n",dlerror());need(false,"hook load failed");}}
    id fg=nil;if(m4)fg=[d newFrameInterpolatorWithDevice:device compiler:compiler];else fg=[d newFrameInterpolatorWithDevice:device];need(fg!=nil,"no tested MetalFX FG");
    need(method_getImplementation(class_getInstanceMethod(object_getClass(fg),@selector(encodeToCommandBuffer:)))==nativeFG,"native FG IMP replaced");
    need(method_getImplementation(class_getInstanceMethod(MTLFXTemporalScalerDescriptor.class,@selector(newTemporalScalerWithDevice:)))==nativeSR,"disabled SR factory swizzled");
    auto c=texture(device,w,h,MTLPixelFormatRGB10A2Unorm),prev=texture(device,w,h,MTLPixelFormatRGB10A2Unorm);
    auto z=texture(device,w,h,MTLPixelFormatR32Float),mv=texture(device,w,h,MTLPixelFormatRG16Float),out=texture(device,w,h,MTLPixelFormatRGB10A2Unorm,true);
    std::vector<uint32_t> pixels(size_t(w)*h);std::vector<float> depth(pixels.size(),.5f);std::vector<uint32_t> motion(pixels.size(),0);
    for(unsigned y=0;y<h;++y)for(unsigned x=0;x<w;++x)pixels[size_t(y)*w+x]=0xc0000000u|((x%1024)<<10)|(y%1024)|((x/2%1024)<<20);
    for(auto t:{c,prev})[t replaceRegion:MTLRegionMake2D(0,0,w,h) mipmapLevel:0 withBytes:pixels.data() bytesPerRow:w*4];
    [z replaceRegion:MTLRegionMake2D(0,0,w,h) mipmapLevel:0 withBytes:depth.data() bytesPerRow:w*4];
    [mv replaceRegion:MTLRegionMake2D(0,0,w,h) mipmapLevel:0 withBytes:motion.data() bytesPerRow:w*4];
    id<MTLFXFrameInterpolatorBase> base=fg;base.colorTexture=c;base.prevColorTexture=prev;base.depthTexture=z;base.motionTexture=mv;base.outputTexture=out;
    base.motionVectorScaleX=base.motionVectorScaleY=1;base.nearPlane=.1f;base.farPlane=1000;base.fieldOfView=60;base.aspectRatio=float(w)/h;base.deltaTime=1.f/60;
    auto q=[device newCommandQueue];auto q4=[device newMTL4CommandQueue];auto allocator=[device newCommandAllocator];auto event=[device newSharedEvent];uint64_t token=0;
    auto rd=[MTLResidencySetDescriptor new];rd.initialCapacity=5;auto residency=[device newResidencySetWithDescriptor:rd error:&error];
    for(id value in @[c,prev,z,mv,out])[residency addAllocation:value];[residency commit];
    MTL4CounterHeapDescriptor *cd=[MTL4CounterHeapDescriptor new];cd.type=MTL4CounterHeapTypeTimestamp;cd.count=2;
    auto counters=[device newCounterHeapWithDescriptor:cd error:&error];need(!m4||counters!=nil,"GPU counters unavailable");
    std::vector<double> cpu,gpu;MTLTimestamp cpu0=0,gpu0=0,cpu1=0,gpu1=0;[device sampleTimestamps:&cpu0 gpuTimestamp:&gpu0];
    constexpr int warmups=32,samples=512;
    for(int i=-warmups;i<samples;++i){
        base.shouldResetHistory=i==-warmups;
        if(m4){
            auto cb=[device newCommandBuffer];[allocator reset];[cb beginCommandBufferWithAllocator:allocator];[cb useResidencySet:residency];
            [cb writeTimestampIntoHeap:counters atIndex:0];auto start=std::chrono::steady_clock::now();[(id<MTL4FXFrameInterpolator>)fg encodeToCommandBuffer:cb];auto end=std::chrono::steady_clock::now();
            [cb writeTimestampIntoHeap:counters atIndex:1];[cb endCommandBuffer];id<MTL4CommandBuffer> list[]={cb};[q4 commit:list count:1];[q4 signalEvent:event value:++token];need([event waitUntilSignaledValue:token timeoutMS:20000],"Metal4 GPU timeout");
            NSData *resolved=[counters resolveCounterRange:NSMakeRange(0,2)];need(resolved.length==2*sizeof(MTL4TimestampHeapEntry),"bad GPU counter result");
            auto stamps=(const MTL4TimestampHeapEntry*)resolved.bytes;need(stamps[1].timestamp>stamps[0].timestamp,"unordered GPU stamps");
            if(i>=0){cpu.push_back(std::chrono::duration<double,std::micro>(end-start).count());gpu.push_back(double(stamps[1].timestamp-stamps[0].timestamp));}
        }else{
            auto cb=[q commandBuffer];auto start=std::chrono::steady_clock::now();[(id<MTLFXFrameInterpolator>)fg encodeToCommandBuffer:cb];auto end=std::chrono::steady_clock::now();
            [cb commit];[cb waitUntilCompleted];need(cb.status==MTLCommandBufferStatusCompleted,"Metal3 GPU failure");
            if(i>=0){cpu.push_back(std::chrono::duration<double,std::micro>(end-start).count());gpu.push_back((cb.GPUEndTime-cb.GPUStartTime)*1000);}
        }
    }
    [device sampleTimestamps:&cpu1 gpuTimestamp:&gpu1];
    if(m4){mach_timebase_info_data_t tb={};mach_timebase_info(&tb);double scale=double(cpu1-cpu0)*tb.numer/tb.denom/double(gpu1-gpu0)/1.e6;for(auto &t:gpu)t*=scale;}
    uint64_t encodes=0,generated=0;
    if(loaded){using Get=void(*)(Fsr4AnrStatus*);auto get=(Get)dlsym(library,"fsr4_anr_get_status");need(get!=nullptr,"no ANR status symbol");Fsr4AnrStatus s={};get(&s);encodes=s.encodes;generated=s.generated;need(!encodes&&!generated&&!s.primed,"native mode ran ANR work");}
    auto readback=[device newBufferWithLength:pixels.size()*4 options:MTLResourceStorageModeShared];
    if(m4){auto cb=[device newCommandBuffer];[allocator reset];[cb beginCommandBufferWithAllocator:allocator];[residency addAllocation:readback];[residency commit];[cb useResidencySet:residency];
        auto enc=[cb computeCommandEncoder];[enc barrierAfterQueueStages:MTLStageAll beforeStages:MTLStageBlit visibilityOptions:MTL4VisibilityOptionDevice];
        [enc copyFromTexture:out sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0,0,0) sourceSize:MTLSizeMake(w,h,1) toBuffer:readback destinationOffset:0 destinationBytesPerRow:w*4 destinationBytesPerImage:pixels.size()*4];
        [enc endEncoding];[cb endCommandBuffer];id<MTL4CommandBuffer> list[]={cb};[q4 commit:list count:1];[q4 signalEvent:event value:++token];need([event waitUntilSignaledValue:token timeoutMS:20000],"readback timeout");
    }else{auto cb=[q commandBuffer];auto enc=[cb blitCommandEncoder];
        [enc copyFromTexture:out sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0,0,0) sourceSize:MTLSizeMake(w,h,1) toBuffer:readback destinationOffset:0 destinationBytesPerRow:w*4 destinationBytesPerImage:pixels.size()*4];
        [enc endEncoding];[cb commit];[cb waitUntilCompleted];need(cb.status==MTLCommandBufferStatusCompleted,"readback failure");}
    unsigned char digest[32];CC_SHA256(readback.contents,(CC_LONG)readback.length,digest);NSMutableString *hash=[NSMutableString string];for(auto b:digest)[hash appendFormat:@"%02x",b];
    NSDictionary *report=@{@"schema":@"fsr4_metalfx_passthrough_bench_v1",@"api":m4?@"Metal4":@"Metal3",@"host_arch":@"x86_64-Rosetta",@"device":device.name,@"hook_loaded":@(loaded),@"controller_status_io_enabled":@YES,@"warmups":@(warmups),@"samples":@(samples),
        @"cpu_encode_median_us":@(percentile(cpu,.5)),@"cpu_encode_p95_us":@(percentile(cpu,.95)),@"gpu_median_ms":@(percentile(gpu,.5)),@"gpu_p95_ms":@(percentile(gpu,.95)),
        @"original_fg_imp_retained":@YES,@"disabled_sr_factory_retained":@YES,@"anr_encodes":@(encodes),@"anr_generated":@(generated),@"output_sha256":hash,@"scope":@"Standalone fixed-frame FG, no game FPS or SR workload"};
    NSData *json=[NSJSONSerialization dataWithJSONObject:report options:NSJSONWritingPrettyPrinted error:nil];need([json writeToFile:[NSString stringWithUTF8String:argv[2]] atomically:YES],"cannot save report");fwrite(json.bytes,1,json.length,stdout);puts("");
    unlink(settings.c_str());unlink((std::string(directory)+"/dist/crossover/fg-status.conf").c_str());
    rmdir((std::string(directory)+"/dist/crossover").c_str());rmdir((std::string(directory)+"/dist").c_str());rmdir(directory);return 0;
}catch(const std::exception &e){fprintf(stderr,"passthrough benchmark: %s\n",e.what());return 1;}}}
