// SPDX-License-Identifier: MIT
// Isolated new-look scalar-rule conformance runner, no model or game inputs.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cstdio>
#include <cstring>
#include <stdexcept>

static void require(bool ok,const char *message) {if(!ok)throw std::runtime_error(message);}
int main(int argc,char **argv) { @autoreleasepool {try {
    require(argc==5,"usage: new_look_gpu LIB INPUT OUTPUT classic|metal4");
    const bool metal4=!std::strcmp(argv[4],"metal4");
    require(metal4 || !std::strcmp(argv[4],"classic"),"unknown command API");
    NSData *data=[NSData dataWithContentsOfFile:[NSString stringWithUTF8String:argv[2]]];
    constexpr NSUInteger inputStride=400, outputStride=16;
    require(data.length && data.length%inputStride==0,"invalid input stride");
    const NSUInteger count=data.length/inputStride;
    id<MTLDevice> device=MTLCreateSystemDefaultDevice();require(device!=nil,"no Metal device");
    NSError *error=nil;
    id<MTLLibrary> library=[device newLibraryWithURL:[NSURL fileURLWithPath:[NSString stringWithUTF8String:argv[1]]] error:&error];
    id<MTLFunction> function=[library newFunctionWithName:@"new_look_test"];
    id<MTLComputePipelineState> pipeline=function?[device newComputePipelineStateWithFunction:function error:&error]:nil;
    if(!pipeline)throw std::runtime_error(error.localizedDescription.UTF8String ?: "missing pipeline");
    id<MTLBuffer> input=[device newBufferWithBytes:data.bytes length:data.length options:MTLResourceStorageModeShared];
    id<MTLBuffer> output=[device newBufferWithLength:(count+2)*outputStride options:MTLResourceStorageModeShared];
    require(input && output,"cannot allocate buffers");
    std::memset(output.contents,0xa5,output.length);
    if(metal4) {
        if(@available(macOS 26.0,*)) {
            id<MTL4CommandQueue> queue=[device newMTL4CommandQueue];
            auto allocator=[device newCommandAllocator];
            id<MTL4CommandBuffer> cb=[device newCommandBuffer];
            MTLResidencySetDescriptor *rd=[MTLResidencySetDescriptor new];rd.initialCapacity=2;
            id<MTLResidencySet> residency=[device newResidencySetWithDescriptor:rd error:&error];
            require(queue && allocator && cb && residency,"cannot allocate Metal 4 objects");
            [residency addAllocation:input];[residency addAllocation:output];[residency commit];[residency requestResidency];
            [cb beginCommandBufferWithAllocator:allocator];[cb useResidencySet:residency];
            id<MTL4ComputeCommandEncoder> encoder=[cb computeCommandEncoder];
            MTL4ArgumentTableDescriptor *td=[MTL4ArgumentTableDescriptor new];td.maxBufferBindCount=2;
            id<MTL4ArgumentTable> table=[device newArgumentTableWithDescriptor:td error:&error];
            require(encoder && table,"cannot allocate Metal 4 encoder/table");
            [table setAddress:input.gpuAddress atIndex:0];[table setAddress:output.gpuAddress+outputStride atIndex:1];
            [encoder setArgumentTable:table];[encoder setComputePipelineState:pipeline];
            [encoder dispatchThreads:MTLSizeMake(count,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
            [encoder barrierAfterStages:MTLStageDispatch beforeQueueStages:MTLStageAll visibilityOptions:MTL4VisibilityOptionDevice];
            [encoder endEncoding];[cb endCommandBuffer];
            id<MTL4CommandBuffer> buffers[]={cb};[queue commit:buffers count:1];
            id<MTLSharedEvent> event=[device newSharedEvent];[queue signalEvent:event value:1];
            require([event waitUntilSignaledValue:1 timeoutMS:20000],"Metal 4 completion timeout");
            [residency endResidency];
        } else throw std::runtime_error("Metal 4 requires macOS 26");
    } else {
        id<MTLCommandQueue> queue=[device newCommandQueue];id<MTLCommandBuffer> cb=[queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder=[cb computeCommandEncoder];
        require(encoder!=nil,"cannot allocate encoder");
        [encoder setComputePipelineState:pipeline];[encoder setBuffer:input offset:0 atIndex:0];
        [encoder setBuffer:output offset:outputStride atIndex:1];
        [encoder dispatchThreads:MTLSizeMake(count,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
        [encoder endEncoding];[cb commit];[cb waitUntilCompleted];
        require(cb.status==MTLCommandBufferStatusCompleted,"GPU command failed");
    }
    const auto *bytes=static_cast<const unsigned char *>(output.contents);
    for(NSUInteger i=0;i<outputStride;++i)
        require(bytes[i]==0xa5 && bytes[(count+1)*outputStride+i]==0xa5,"output canary overwritten");
    NSData *result=[NSData dataWithBytes:bytes+outputStride length:count*outputStride];
    require([result writeToFile:[NSString stringWithUTF8String:argv[3]] atomically:YES],"cannot save output");
    std::printf("new-look %s: %lu scalar cases, output canaries passed on %s\n",argv[4],(unsigned long)count,device.name.UTF8String);
    return 0;
} catch(const std::exception &e) {std::fprintf(stderr,"new-look GPU: %s\n",e.what());return 1;} } }
