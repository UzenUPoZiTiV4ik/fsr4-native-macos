// SPDX-License-Identifier: MIT
#include "anr_runtime.h"
#import <CommonCrypto/CommonDigest.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <map>
#include <mutex>
#include <vector>

namespace anr {
namespace {
std::string utf8(NSString *s) { return s ? std::string(s.UTF8String) : ""; }
NSString *ns(const std::string &s) { return [NSString stringWithUTF8String:s.c_str()]; }
NSData *load(NSString *root,NSString *relative) { return [NSData dataWithContentsOfFile:[root stringByAppendingPathComponent:relative]]; }
std::string hash(NSData *data) {
    CC_SHA256_CTX context;CC_SHA256_Init(&context);
    const auto *ptr=(const unsigned char*)data.bytes;size_t left=data.length;
    while(left){const auto n=(CC_LONG)std::min(left,size_t(1<<30));CC_SHA256_Update(&context,ptr,n);ptr+=n;left-=n;}
    unsigned char bytes[32];CC_SHA256_Final(bytes,&context);
    static const char hex[]="0123456789abcdef";std::string result;
    for(unsigned char byte:bytes){result+=hex[byte>>4];result+=hex[byte&15];}return result;
}
bool safe_path(NSString *path) {
    if(![path isKindOfClass:NSString.class] || !path.length || path.isAbsolutePath)return false;
    for(NSString *part in [path componentsSeparatedByString:@"/"])
        if(!part.length || [part isEqual:@"."] || [part isEqual:@".."])return false;
    return [path rangeOfCharacterFromSet:[NSCharacterSet characterSetWithCharactersInString:@"\\\0"]].location==NSNotFound;
}
std::mutex pipelineMutex;
std::map<std::pair<uintptr_t,std::string>,id<MTLComputePipelineState>> pipelineCache;
struct Binding {std::string name;unsigned index=0;bool uniform=false,sampler=false;};
struct Kernel {id<MTLComputePipelineState> pipeline=nil;bool native=false;unsigned resources=0,samplers=0;std::vector<Binding> bindings;};
struct Stage {
    std::string kernel,label;MTLSize groups={},threads={};id<MTLBuffer> constants=nil,resources=nil,samplers=nil,root=nil;
    id<MTL4ArgumentTable> arguments=nil;
    uint64_t bufferRevision=0;
    std::map<std::string,unsigned> buffers;
    std::map<std::string,std::string> images;
};
struct Descriptor {uint64_t address=0,resource=0,metadata=0;};
static_assert(sizeof(Descriptor)==24,"Apple descriptor ABI");
MTLSize extent(NSArray *v) { return MTLSizeMake([v[0] unsignedLongLongValue],[v[1] unsignedLongLongValue],[v[2] unsignedLongLongValue]); }
}

struct Runtime::Impl {
    id<MTLDevice> device=nil;NSString *directory=nil;NSDictionary *manifest=nil;
    unsigned width=0,height=0;bool initialized=false,coalesced=true;
    uint64_t bufferRevision=1;unsigned frameSlot=0;
    id<MTLBuffer> frameBuffers[4]={};
    id<MTLResidencySet> borrowedSets[4]={};
    id<MTLResidencySet> residency=nil;
    id<MTLBuffer> empty=nil;id<MTLSamplerState> sampler=nil;
    std::vector<id<MTLBuffer>> buffers;
    std::map<std::string,Kernel> kernels;
    std::map<std::string,std::vector<Stage>> stages[4];
    NSArray *swaps=nil;
    id<MTLTexture> inputColor=nil,inputDepth=nil,inputMotion=nil,output=nil;
    // All CPU descriptor writes are made before the group encoder starts.
    // Caller serialization protects the next group/frame from in-flight reuse.
    bool fail(std::string &error,const std::string &why){error=why;return false;}
    bool add(id<MTLAllocation> resource,std::string &error) {
        if(!resource)return fail(error,"ANR resource allocation failed");
        [residency addAllocation:resource];return true;
    }
    id<MTLBuffer> buffer(size_t bytes) {return [device newBufferWithLength:std::max(bytes,size_t(256)) options:MTLResourceStorageModeShared];}
    bool prepare_stage(Stage &stage,std::string &error) {
        auto found=kernels.find(stage.kernel);if(found==kernels.end())return fail(error,"ANR kernel missing: "+stage.kernel);
        Kernel &kernel=found->second;
        // Both imported native kernels and converted DXIL use a root pointer
        // to a dense, 24-byte-per-resource table at Metal buffer bind point 2.
        // nativeMetal.index is the original source argument index, NOT a Metal
        // buffer slot. Table order is nativeMetal.bindings order (NRTable).
        {
            auto *entries=(Descriptor*)stage.resources.contents;
            auto *samplerEntries=(Descriptor*)stage.samplers.contents;
            for(const Binding &binding:kernel.bindings) {
                if(binding.sampler){samplerEntries[binding.index]={sampler.gpuResourceID._impl,0,0};continue;}
                Descriptor entry;
                auto image=stage.images.find(binding.name);
                if(image!=stage.images.end()) {
                    id<MTLTexture> texture=nil;
                    if(image->second=="color")texture=inputColor;
                    else if(image->second=="depth")texture=inputDepth;
                    else if(image->second=="motion")texture=inputMotion;
                    else if(image->second=="output")texture=output;
                    if(!texture)return fail(error,"ANR stage requires texture "+image->second);
                    entry.resource=texture.gpuResourceID._impl;
                } else {
                    if(stage.bufferRevision==bufferRevision)continue;
                    id<MTLBuffer> target=empty;
                    if(binding.uniform)target=stage.constants;
                    else {auto it=stage.buffers.find(binding.name);if(it!=stage.buffers.end())target=buffers.at(it->second);}
                    entry.address=target.gpuAddress;entry.metadata=target.length;
                }
                entries[binding.index]=entry;
            }
            if(!stage.bufferRevision)[stage.arguments setAddress:stage.root.gpuAddress atIndex:2];
            stage.bufferRevision=bufferRevision;
        }
        return true;
    }
    bool encode_group(id command,bool metal4,std::vector<Stage> &group,std::string &error) {
        for(Stage &stage:group)if(!prepare_stage(stage,error))return false;
        if(metal4){[(id<MTL4CommandBuffer>)command useResidencySet:residency];
            [(id<MTL4CommandBuffer>)command useResidencySet:borrowedSets[frameSlot]];}
        NSMutableArray *used=[NSMutableArray array];
        if(!metal4){
            for(auto buffer:buffers)[used addObject:buffer];[used addObject:empty];
            for(Stage &stage:group){[used addObjectsFromArray:@[stage.constants,stage.resources,stage.samplers,stage.root]];}
            for(id texture in @[inputColor,inputDepth,inputMotion]) [used addObject:texture];
            if(output)[used addObject:output];
        }
        id encoder=nil;bool first=true;
        for(Stage &stage:group){
            if(!encoder){
                if(metal4)encoder=[(id<MTL4CommandBuffer>)command computeCommandEncoder];
                else encoder=[(id<MTLCommandBuffer>)command computeCommandEncoder];
                if(!encoder)return fail(error,"ANR compute encoder allocation failed");
                if(metal4)[(id<MTL4ComputeCommandEncoder>)encoder barrierAfterQueueStages:MTLStageAll
                    beforeStages:MTLStageDispatch visibilityOptions:MTL4VisibilityOptionDevice];
                else {
                    std::vector<id<MTLResource>> resources;for(id value in used)resources.push_back(value);
                    [(id<MTLComputeCommandEncoder>)encoder useResources:resources.data() count:resources.size()
                        usage:MTLResourceUsageRead|MTLResourceUsageWrite];
                }
                first=true;
            }
            Kernel &kernel=kernels.at(stage.kernel);
            if(metal4){
                auto enc=(id<MTL4ComputeCommandEncoder>)encoder;
                if(!first)[enc barrierAfterEncoderStages:MTLStageDispatch beforeEncoderStages:MTLStageDispatch
                    visibilityOptions:MTL4VisibilityOptionDevice];
                [enc setComputePipelineState:kernel.pipeline];[enc setArgumentTable:stage.arguments];
                [enc dispatchThreadgroups:stage.groups threadsPerThreadgroup:stage.threads];
            }else{
                auto enc=(id<MTLComputeCommandEncoder>)encoder;
                if(!first)[enc memoryBarrierWithScope:MTLBarrierScopeBuffers|MTLBarrierScopeTextures];
                [enc setComputePipelineState:kernel.pipeline];[enc setBuffer:stage.root offset:0 atIndex:2];
                [enc dispatchThreadgroups:stage.groups threadsPerThreadgroup:stage.threads];
            }
            first=false;
            if(!coalesced){[encoder endEncoding];encoder=nil;}
        }
        if(encoder)[encoder endEncoding];
        return true;
    }
};
Runtime::Runtime(id<MTLDevice> device,const std::string &directory):impl(std::make_unique<Impl>()) {
    impl->device=device;impl->directory=ns(directory);
}
Runtime::~Runtime()=default;
unsigned Runtime::width() const {return impl->width;}
unsigned Runtime::height() const {return impl->height;}
id<MTLBuffer> Runtime::diagnostic_buffer(unsigned index) const {return index<impl->buffers.size()?impl->buffers[index]:nil;}

bool Runtime::initialize(std::string &error,bool twoXOnly) { @autoreleasepool {
    auto &r=*impl;if(r.initialized)return true;
    NSData *md=load(r.directory,@"manifest.json");
    if(!md||md.length>32u*1024u*1024u)return r.fail(error,"ANR missing/oversized manifest");
    r.manifest=md ? [NSJSONSerialization JSONObjectWithData:md options:0 error:nil] : nil;
    if(![r.manifest isKindOfClass:NSDictionary.class] || ![r.manifest[@"schema"] isEqual:@"fsr4_anr_runtime_v1"]
       || ![r.manifest[@"checkpoint_id"] isEqual:@"e09deb7012c96c172e49e72eb0b747b4611a8b6e3f14e5f5563189c580772ac1"])
        return r.fail(error,"ANR runtime manifest/checkpoint mismatch");
    NSDictionary *files=r.manifest[@"files"];
    if(![files isKindOfClass:NSDictionary.class] || !files.count || files.count>1000)return r.fail(error,"ANR runtime inventory missing");
    for(NSString *name in files) {
        if(!safe_path(name))return r.fail(error,"ANR unsafe asset path");
        NSString *path=[r.directory stringByAppendingPathComponent:name];
        NSDictionary *attributes=[NSFileManager.defaultManager attributesOfItemAtPath:path error:nil];
        if(![attributes[NSFileType] isEqual:NSFileTypeRegular])return r.fail(error,"ANR asset missing/nonregular: "+utf8(name));
        NSDictionary *row=files[name];
        if(![row isKindOfClass:NSDictionary.class]||![row[@"bytes"] isKindOfClass:NSNumber.class]
           ||[row[@"bytes"] unsignedLongLongValue]>32u*1024u*1024u||![row[@"sha256"] isKindOfClass:NSString.class])
            return r.fail(error,"ANR malformed inventory entry");
        NSData *data=load(r.directory,name);if(!data)return r.fail(error,"ANR unreadable asset");
        if(data.length!=[row[@"bytes"] unsignedLongLongValue] || hash(data)!=utf8(row[@"sha256"]))
            return r.fail(error,"ANR asset identity mismatch: "+utf8(name));
    }
    r.width=[r.manifest[@"width"] unsignedIntValue];r.height=[r.manifest[@"height"] unsignedIntValue];
    if(!((r.width==1920&&r.height==1080)||(r.width==2560&&r.height==1440)||(r.width==3840&&r.height==2160)))
        return r.fail(error,"ANR unsupported geometry");
    MTLResidencySetDescriptor *rd=[MTLResidencySetDescriptor new];rd.initialCapacity=1600;
    NSError *metalError=nil;r.residency=[r.device newResidencySetWithDescriptor:rd error:&metalError];
    if(!r.residency)return r.fail(error,"ANR residency: "+utf8(metalError.localizedDescription));
    NSData *weights=load(r.directory,@"weights.bin");
    if(!weights || weights.length!=43520 || hash(weights)!="af6c2f200ddb7baeaf4d82b5515af6f077f67896b2a9d26163ef6ea23975acb5")
        return r.fail(error,"ANR unsupported model weights");
    if(![r.manifest[@"buffers"] isKindOfClass:NSArray.class]||[r.manifest[@"buffers"] count]<80||[r.manifest[@"buffers"] count]>128)
        return r.fail(error,"ANR invalid buffer inventory");
    r.buffers.resize([r.manifest[@"buffers"] count]);
    for(NSDictionary *record in r.manifest[@"buffers"]) {
        unsigned index=[record[@"index"] unsignedIntValue];size_t bytes=[record[@"size"] unsignedLongLongValue];
        if(index>=r.buffers.size() || r.buffers[index] || !bytes || bytes>(size_t(1)<<30))return r.fail(error,"ANR buffer shape invalid");
        r.buffers[index]=r.buffer(bytes);if(!r.add(r.buffers[index],error))return false;
        std::memset(r.buffers[index].contents,0,r.buffers[index].length);
        if([record[@"kind"] isEqual:@"weight"]) {
            size_t offset=[record[@"data_offset"] unsignedLongLongValue];
            if(offset>weights.length || bytes>weights.length-offset)return r.fail(error,"ANR weight span invalid");
            std::memcpy(r.buffers[index].contents,(const uint8_t*)weights.bytes+offset,bytes);
        }
    }
    for(unsigned slot=0;slot<4;++slot){
        r.frameBuffers[slot]=r.buffer(204);if(!r.add(r.frameBuffers[slot],error))return false;
        MTLResidencySetDescriptor *bd=[MTLResidencySetDescriptor new];bd.initialCapacity=4;
        r.borrowedSets[slot]=[r.device newResidencySetWithDescriptor:bd error:&metalError];
        if(!r.borrowedSets[slot])return r.fail(error,"ANR frame residency allocation failed");
    }
    r.empty=r.buffer(256);if(!r.add(r.empty,error))return false;std::memset(r.empty.contents,0,r.empty.length);
    MTLSamplerDescriptor *sd=[MTLSamplerDescriptor new];sd.supportArgumentBuffers=YES;
    sd.minFilter=sd.magFilter=MTLSamplerMinMagFilterLinear;r.sampler=[r.device newSamplerStateWithDescriptor:sd];
    NSDictionary *kernelRecords=r.manifest[@"kernels"];
    NSMutableSet *neededKernels=[NSMutableSet set];
    NSDictionary *stageRecords=r.manifest[@"stages"];
    for(NSString *key in stageRecords){
        if(twoXOnly&&![key isEqual:@"pair"]&&![key isEqual:@"phase:0.5:image"])continue;
        for(NSDictionary *stage in stageRecords[key])[neededKernels addObject:stage[@"kernel"]];
    }
    for(NSString *key in kernelRecords) {
        if(![neededKernels containsObject:key])continue;
        NSDictionary *record=kernelRecords[key];Kernel kernel;kernel.native=[record[@"route"] isEqual:@"native"];
        NSString *assetPath=record[@"path"];
        if(!safe_path(assetPath)||!files[assetPath])return r.fail(error,"ANR unlisted kernel asset");
        // Pipeline cache is keyed by exact shader bytes and device, not name.
        // Neural state and scratch remain independent per runtime owner.
        const auto cacheKey=std::make_pair((uintptr_t)(__bridge void*)r.device,
            utf8(files[assetPath][@"sha256"])+":"+utf8(record[@"entry"]));
        {std::lock_guard<std::mutex> lock(pipelineMutex);
            auto cached=pipelineCache.find(cacheKey);
            if(cached!=pipelineCache.end())kernel.pipeline=cached->second;
            else{
                NSURL *url=[NSURL fileURLWithPath:[r.directory stringByAppendingPathComponent:assetPath]];
                id<MTLLibrary> library=[r.device newLibraryWithURL:url error:&metalError];
                id<MTLFunction> function=[library newFunctionWithName:record[@"entry"]];
                kernel.pipeline=function ? [r.device newComputePipelineStateWithFunction:function error:&metalError] : nil;
                if(kernel.pipeline)pipelineCache.emplace(cacheKey,kernel.pipeline);
            }
        }
        if(!kernel.pipeline)return r.fail(error,"ANR pipeline "+utf8(key)+": "+utf8(metalError.localizedDescription));
        kernel.resources=[record[@"resource_count"] unsignedIntValue];kernel.samplers=[record[@"sampler_count"] unsignedIntValue];
        if(!kernel.resources||kernel.resources>256||kernel.samplers>16)return r.fail(error,"ANR descriptor table size invalid");
        unsigned tableSlot=0;
        for(NSDictionary *b in record[@"bindings"])kernel.bindings.push_back({utf8(b[@"name"]),
            kernel.native ? tableSlot++ : [b[@"slot"] unsignedIntValue],
            (bool)[b[@"uniform"] boolValue],(bool)[b[@"sampler"] boolValue]});
        for(const auto &binding:kernel.bindings)if(binding.index>=(binding.sampler?kernel.samplers:kernel.resources))
            return r.fail(error,"ANR descriptor slot out of bounds");
        r.kernels.emplace(utf8(key),std::move(kernel));
    }
    for(unsigned slot=0;slot<4;++slot)for(NSString *key in stageRecords) {
        if(twoXOnly&&![key isEqual:@"pair"]&&![key isEqual:@"phase:0.5:image"])continue;
        auto &stages=r.stages[slot][utf8(key)];
        for(NSDictionary *record in stageRecords[key]) {
            Stage stage;stage.kernel=utf8(record[@"kernel"]);stage.label=utf8(record[@"label"]);
            stage.groups=extent(record[@"groups"]);stage.threads=extent(record[@"threads"]);
            auto it=r.kernels.find(stage.kernel);if(it==r.kernels.end())return r.fail(error,"ANR unknown stage kernel");
            Kernel &kernel=it->second;
            if(stage.threads.width*stage.threads.height*stage.threads.depth>kernel.pipeline.maxTotalThreadsPerThreadgroup)
                return r.fail(error,"ANR threadgroup exceeds pipeline limit");
            if(!safe_path(record[@"uniforms"])||!files[record[@"uniforms"]])return r.fail(error,"ANR unlisted uniform asset");
            NSData *data=load(r.directory,record[@"uniforms"]);if(!data||data.length>4096)return r.fail(error,"ANR uniform size invalid");
            stage.constants=r.buffer(data.length);
            if(!r.add(stage.constants,error))return false;
            std::memset(stage.constants.contents,0,stage.constants.length);std::memcpy(stage.constants.contents,data.bytes,data.length);
            for(NSString *name in record[@"buffers"]) {
                unsigned index=[record[@"buffers"][name] unsignedIntValue];if(index>=r.buffers.size())return r.fail(error,"ANR buffer reference invalid");
                stage.buffers.emplace(utf8(name),index);
            }
            for(NSString *name in record[@"images"])stage.images.emplace(utf8(name),utf8(record[@"images"][name]));
            MTL4ArgumentTableDescriptor *td=[MTL4ArgumentTableDescriptor new];
            td.maxBufferBindCount=3;
            stage.arguments=[r.device newArgumentTableWithDescriptor:td error:&metalError];
            if(!stage.arguments)return r.fail(error,"ANR argument table: "+utf8(metalError.localizedDescription));
            {
                stage.resources=r.buffer(kernel.resources*sizeof(Descriptor));stage.samplers=r.buffer(kernel.samplers*sizeof(Descriptor));stage.root=r.buffer(16);
                if(!r.add(stage.resources,error)||!r.add(stage.samplers,error)||!r.add(stage.root,error))return false;
                std::memset(stage.resources.contents,0,stage.resources.length);std::memset(stage.samplers.contents,0,stage.samplers.length);
                uint64_t pointers[]={stage.resources.gpuAddress,stage.samplers.gpuAddress};std::memcpy(stage.root.contents,pointers,sizeof(pointers));
            }
            stages.push_back(std::move(stage));
        }
    }
    r.swaps=r.manifest[@"swaps"];
    [r.residency commit];r.initialized=true;return true;
} }

bool Runtime::encode_pair(id<MTL4CommandBuffer> command,id<MTLTexture> color,id<MTLTexture> depth,id<MTLTexture> motion,
                          const void *frame,size_t frameBytes,std::string &error) {
    return pair(command,true,color,depth,motion,frame,frameBytes,error);
}
bool Runtime::encode_pair_classic(id<MTLCommandBuffer> command,id<MTLTexture> color,id<MTLTexture> depth,id<MTLTexture> motion,
                                  const void *frame,size_t frameBytes,std::string &error) {
    return pair(command,false,color,depth,motion,frame,frameBytes,error);
}
bool Runtime::pair(id command,bool metal4,id<MTLTexture> color,id<MTLTexture> depth,id<MTLTexture> motion,
                   const void *frame,size_t frameBytes,std::string &error) {
    auto &r=*impl;if(!r.initialized)return r.fail(error,"ANR runtime not initialized");
    if(!command || [command device]!=r.device || !color || !depth || !motion || !frame || frameBytes!=204)
        return r.fail(error,"ANR pair contract mismatch");
    // Reusing this slot is legal only after caller-observed completion. GPU
    // neural/history buffers stay shared and dispatch barriers preserve order.
    auto borrowed=r.borrowedSets[r.frameSlot];[borrowed removeAllAllocations];
    for(id texture in @[color,depth,motion])[borrowed addAllocation:texture];[borrowed commit];
    r.inputColor=color;r.inputDepth=depth;r.inputMotion=motion;r.output=nil;
    r.buffers.at([r.manifest[@"inputs"][@"frame"] unsignedIntValue])=r.frameBuffers[r.frameSlot];
    ++r.bufferRevision;
    std::memcpy(r.frameBuffers[r.frameSlot].contents,frame,frameBytes);
    return r.encode_group(command,metal4,r.stages[r.frameSlot].at("pair"),error);
}
bool Runtime::encode_interpolation(id<MTL4CommandBuffer> command,double phase,id<MTLTexture> output,std::string &error) {
    return interpolate(command,true,phase,output,error);
}
bool Runtime::encode_interpolation_classic(id<MTLCommandBuffer> command,double phase,id<MTLTexture> output,std::string &error) {
    return interpolate(command,false,phase,output,error);
}
void Runtime::set_coalesced(bool enabled){impl->coalesced=enabled;}
void Runtime::set_frame_slot(unsigned slot){impl->frameSlot=slot%4;}
bool Runtime::reset_history(id command,bool metal4,std::string &error){
    auto &r=*impl;if(!r.initialized||!command||[command device]!=r.device)return r.fail(error,"ANR reset command mismatch");
    id encoder=nil;
    if(metal4){[(id<MTL4CommandBuffer>)command useResidencySet:r.residency];
        encoder=[(id<MTL4CommandBuffer>)command computeCommandEncoder];
        if(encoder)[(id<MTL4ComputeCommandEncoder>)encoder barrierAfterQueueStages:MTLStageAll beforeStages:MTLStageBlit visibilityOptions:MTL4VisibilityOptionDevice];}
    else encoder=[(id<MTLCommandBuffer>)command blitCommandEncoder];
    if(!encoder)return r.fail(error,"ANR reset encoder allocation failed");
    for(NSDictionary *record in r.manifest[@"buffers"]){
        if(![record[@"kind"] isEqual:@"persistent"])continue;
        auto buffer=r.buffers.at([record[@"index"] unsignedIntValue]);
        if(metal4)[(id<MTL4ComputeCommandEncoder>)encoder fillBuffer:buffer range:NSMakeRange(0,buffer.length) value:0];
        else [(id<MTLBlitCommandEncoder>)encoder fillBuffer:buffer range:NSMakeRange(0,buffer.length) value:0];
    }
    [encoder endEncoding];return true;
}
bool Runtime::interpolate(id command,bool metal4,double phase,id<MTLTexture> output,std::string &error) {
    auto &r=*impl;if(!r.initialized || !command || [command device]!=r.device || !output || output.device!=r.device)return r.fail(error,"ANR interpolation contract mismatch");
    char name[64];snprintf(name,sizeof name,"phase:%.6g:image",phase);
    auto found=r.stages[r.frameSlot].find(name);if(found==r.stages[r.frameSlot].end())return r.fail(error,"ANR unsupported phase");
    r.output=output;[r.borrowedSets[r.frameSlot] addAllocation:output];[r.borrowedSets[r.frameSlot] commit];
    return r.encode_group(command,metal4,found->second,error);
}
void Runtime::swap_history() {
    auto &r=*impl;
    // These two swaps are implicit in the upstream host's input contract,
    // in addition to the explicitly listed model-history swaps.
    std::swap(r.buffers.at([r.manifest[@"inputs"][@"color0"] unsignedIntValue]),r.buffers.at([r.manifest[@"inputs"][@"color1"] unsignedIntValue]));
    std::swap(r.buffers.at([r.manifest[@"inputs"][@"depth0"] unsignedIntValue]),r.buffers.at([r.manifest[@"inputs"][@"depth1"] unsignedIntValue]));
    for(NSArray *pair in r.swaps)std::swap(r.buffers.at([pair[0] unsignedIntValue]),r.buffers.at([pair[1] unsignedIntValue]));
    ++r.bufferRevision;
}
}
