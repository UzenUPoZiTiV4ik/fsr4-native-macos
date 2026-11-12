// SPDX-License-Identifier: MIT
// Independent ANR replay: never loads/runs the ANR test-kit executable.
#include "anr_runtime.h"
#import <CommonCrypto/CommonDigest.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>
#include <mach/mach_time.h>

static NSData *read(const std::string &path) {
    NSData *data=[NSData dataWithContentsOfFile:[NSString stringWithUTF8String:path.c_str()]];
    if(!data)throw std::runtime_error("cannot read "+path);return data;
}
static id<MTLTexture> texture(id<MTLDevice> device,unsigned w,unsigned h,MTLPixelFormat format) {
    MTLTextureDescriptor *descriptor=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format width:w height:h mipmapped:NO];
    descriptor.usage=MTLTextureUsageShaderRead|MTLTextureUsageShaderWrite;descriptor.storageMode=MTLStorageModeShared;
    auto result=[device newTextureWithDescriptor:descriptor];if(!result)throw std::runtime_error("texture allocation failed");return result;
}
static void upload(id<MTLTexture> texture,NSData *data,unsigned bpp) {
    if(data.length!=texture.width*texture.height*bpp)throw std::runtime_error("input texture size mismatch");
    [texture replaceRegion:MTLRegionMake2D(0,0,texture.width,texture.height) mipmapLevel:0 withBytes:data.bytes bytesPerRow:texture.width*bpp];
}
static void require(bool condition,const std::string &why) {if(!condition)throw std::runtime_error(why);}
static double median(std::vector<double> values) {
    if(values.empty())return 0;std::sort(values.begin(),values.end());return values[values.size()/2];
}
static NSString *hash(NSData *data) {
    unsigned char digest[32];CC_SHA256(data.bytes,(CC_LONG)data.length,digest);
    NSMutableString *result=[NSMutableString string];for(unsigned char value:digest)[result appendFormat:@"%02x",value];return result;
}
int main(int argc,char **argv) { @autoreleasepool { try {
    if(argc<4){fprintf(stderr,"usage: anr_replay <runtime> <inputs> <new-out-dir> [frames=8] [phase=0.5] [reference] [prime=1]\n");return 2;}
    const std::string root=argv[1],inputs=argv[2],out=argv[3];
    const int frames=argc>4?std::stoi(argv[4]):8;double phase=argc>5?std::stod(argv[5]):.5;
    const int prime=argc>7?std::stoi(argv[7]):1;
    require(frames>=1&&frames<=512,"frame count outside 1..512");
    require(!std::filesystem::exists(out),"output exists; choose a new directory");
    id<MTLDevice> device=MTLCreateSystemDefaultDevice();require(device!=nil,"no Metal device");
    anr::Runtime runtime(device,root);std::string error;require(runtime.initialize(error),error);
    const unsigned w=runtime.width(),h=runtime.height();
    id<MTLTexture> color=texture(device,w,h,MTLPixelFormatRGB10A2Unorm);
    id<MTLTexture> depth=texture(device,w,h,MTLPixelFormatR32Float);
    id<MTLTexture> motion=texture(device,w,h,MTLPixelFormatRG16Float);
    id<MTLTexture> output=texture(device,w,h,MTLPixelFormatRGB10A2Unorm);
    NSData *frame0=read(inputs+"/frame0.bin"),*frame1=read(inputs+"/frame.bin");
    require(frame0.length==204&&frame1.length==204,"frame input must be 204 bytes");
    id<MTL4CommandQueue> queue=[device newMTL4CommandQueue];id<MTL4CommandAllocator> allocator=[device newCommandAllocator];
    id<MTL4CommandBuffer> command=[device newCommandBuffer];id<MTLSharedEvent> done=[device newSharedEvent];uint64_t token=0;
    require(queue&&allocator&&command&&done,"Metal 4 command allocation failed");
    NSError *counterError=nil;MTL4CounterHeapDescriptor *counterDesc=[MTL4CounterHeapDescriptor new];
    counterDesc.type=MTL4CounterHeapTypeTimestamp;counterDesc.count=3;
    id<MTL4CounterHeap> counters=[device newCounterHeapWithDescriptor:counterDesc error:&counterError];
    require(counters!=nil,"GPU timestamp counter unavailable");
    auto begin=[&]{[allocator reset];[command beginCommandBufferWithAllocator:allocator];};
    auto finish=[&]{[command endCommandBuffer];id<MTL4CommandBuffer> commands[]={command};[queue commit:commands count:1];
        [queue signalEvent:done value:++token];require([done waitUntilSignaledValue:token timeoutMS:20000],"GPU wait timeout");};
    const char *coalesced=getenv("ANR_COALESCED");
    runtime.set_coalesced(!coalesced || strcmp(coalesced,"0"));
    auto seedPrevious=[&]{runtime.swap_history();};
    if(prime) {
        upload(color,read(inputs+"/color0.bin"),4);upload(depth,read(inputs+"/depth0.bin"),4);upload(motion,read(inputs+"/motion0.bin"),4);
        begin();require(runtime.encode_pair(command,color,depth,motion,frame0.bytes,frame0.length,error),error);finish();seedPrevious();
    }
    upload(color,read(inputs+"/color1.bin"),4);upload(depth,read(inputs+"/depth1.bin"),4);upload(motion,read(inputs+"/motion10.bin"),4);
    std::vector<double> times,gpuTimes,pairTimes,phaseTimes;
    MTLTimestamp cpuStart=0,gpuStart=0,cpuEnd=0,gpuEnd=0;
    [device sampleTimestamps:&cpuStart gpuTimestamp:&gpuStart];
    for(int i=-8;i<frames;++i) {
        begin();const auto start=std::chrono::steady_clock::now();
        [command writeTimestampIntoHeap:counters atIndex:0];
        require(runtime.encode_pair(command,color,depth,motion,frame1.bytes,frame1.length,error),error);
        [command writeTimestampIntoHeap:counters atIndex:1];
        require(runtime.encode_interpolation(command,phase,output,error),error);
        [command writeTimestampIntoHeap:counters atIndex:2];finish();
        const auto end=std::chrono::steady_clock::now();
        NSData *resolved=[counters resolveCounterRange:NSMakeRange(0,3)];
        require(resolved.length==3*sizeof(MTL4TimestampHeapEntry),"invalid GPU timestamp result");
        const auto *stamp=(const MTL4TimestampHeapEntry*)resolved.bytes;
        require(stamp[0].timestamp<stamp[1].timestamp && stamp[1].timestamp<stamp[2].timestamp,"unordered GPU timestamps");
        if(i>=0){
            times.push_back(std::chrono::duration<double,std::milli>(end-start).count());
            gpuTimes.push_back(double(stamp[2].timestamp-stamp[0].timestamp));
            pairTimes.push_back(double(stamp[1].timestamp-stamp[0].timestamp));
            phaseTimes.push_back(double(stamp[2].timestamp-stamp[1].timestamp));
        }
        // This is a repeated fixed-pair replay. History stays the frame0 state
        // to compare the same input pair, not an invented moving sequence.
    }
    [device sampleTimestamps:&cpuEnd gpuTimestamp:&gpuEnd];
    mach_timebase_info_data_t timebase={};mach_timebase_info(&timebase);
    require(cpuEnd>cpuStart&&gpuEnd>gpuStart,"timestamp calibration failed");
    const double gpuTickMs=double(cpuEnd-cpuStart)*timebase.numer/timebase.denom/double(gpuEnd-gpuStart)/1e6;
    for(auto values:{&gpuTimes,&pairTimes,&phaseTimes})for(double &value:*values)value*=gpuTickMs;
    NSMutableData *generated=[NSMutableData dataWithLength:size_t(w)*h*4];
    [output getBytes:generated.mutableBytes bytesPerRow:w*4 fromRegion:MTLRegionMake2D(0,0,w,h) mipmapLevel:0];
    NSMutableDictionary *report=[@{@"schema":@"anr_independent_replay_v1",@"device":device.name,@"width":@(w),@"height":@(h),
        @"frames":@(frames),@"phase":@(phase),@"primed":@(prime),@"host_wall_median_ms":@(median(times)),
        @"output_sha256":hash(generated),@"testkit_executable_used":@NO,@"game_integration_verified":@NO,
        @"gpu_median_ms":@(median(gpuTimes)),@"pair_gpu_median_ms":@(median(pairTimes)),
        @"phase_gpu_median_ms":@(median(phaseTimes)),@"warmups":@8,@"gpu_tick_ns":@(gpuTickMs*1e6),
        @"coalesced":@(!coalesced || strcmp(coalesced,"0"))} mutableCopy];
    if(argc>6) {
        NSData *reference=read(argv[6]);require(reference.length==generated.length,"reference size mismatch");
        const auto *a=(const uint32_t*)generated.bytes,*b=(const uint32_t*)reference.bytes;
        double squared=0;uint64_t pixels=0,components=0;unsigned maxDifference=0;
        for(size_t i=0;i<size_t(w)*h;++i){if(a[i]!=b[i])++pixels;for(unsigned shift:{0u,10u,20u}){
            const int diff=int((a[i]>>shift)&1023)-int((b[i]>>shift)&1023);
            squared+=double(diff)*diff;if(diff)++components;maxDifference=std::max(maxDifference,unsigned(std::abs(diff)));}}
        const double rmse=std::sqrt(squared/(double(w)*h*3));
        report[@"differing_pixels"]=@(pixels);report[@"differing_components"]=@(components);report[@"max_code_difference"]=@(maxDifference);
        report[@"psnr_rgb10_db"]=rmse>0?@(20*std::log10(1023/rmse)):@"infinity";
    }
    require(std::filesystem::create_directories(out),"cannot create output");
    if(getenv("ANR_DUMP_BUFFERS")) for(unsigned index:{12u,13u,14u,15u,17u,18u,29u,30u,31u,33u,34u,45u,46u,47u,51u,55u,56u,57u,58u,59u,61u,62u,63u,64u,65u,67u,68u,69u,73u,79u,80u}) {
        auto buffer=runtime.diagnostic_buffer(index);
        NSData *dump=[NSData dataWithBytes:buffer.contents length:buffer.length];
        const std::string name=out+"/b"+std::to_string(index)+".bin";
        require([dump writeToFile:[NSString stringWithUTF8String:name.c_str()] atomically:YES],"cannot save diagnostic buffer");
    }
    NSString *outputPath=[NSString stringWithUTF8String:out.c_str()];
    require([generated writeToFile:[outputPath stringByAppendingPathComponent:@"out.bin"] atomically:YES],"cannot save output");
    NSData *json=[NSJSONSerialization dataWithJSONObject:report options:NSJSONWritingPrettyPrinted error:nil];
    require([json writeToFile:[outputPath stringByAppendingPathComponent:@"run.json"] atomically:YES],"cannot save report");
    fwrite(json.bytes,1,json.length,stdout);puts("");return 0;
} catch(const std::exception &e) {fprintf(stderr,"ANR replay: %s\n",e.what());return 1;} } }
