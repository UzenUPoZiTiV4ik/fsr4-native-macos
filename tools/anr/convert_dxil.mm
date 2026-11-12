// SPDX-License-Identifier: MIT
// Compile one explicitly supplied DXIL kernel with the supplied Apple converter.
// No target program is run and the source files/library are never modified.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <metal_irconverter/metal_irconverter.h>
#include <dlfcn.h>
#include <cstdio>
#include <string>
#include <vector>

#define LOAD_IR(name) auto name = reinterpret_cast<decltype(&::name)>(dlsym(library, #name)); \
    if (!name) { fprintf(stderr,"missing Apple converter API: %s\n",#name); return 1; }

int main(int argc,char **argv) { @autoreleasepool {
    if(argc!=5) {fprintf(stderr,"usage: convert_dxil <dxil> <reflection.json> <libmetalirconverter.dylib> <new-output-directory>\n");return 2;}
    NSData *dxil=[NSData dataWithContentsOfFile:[NSString stringWithUTF8String:argv[1]]];
    NSData *rd=[NSData dataWithContentsOfFile:[NSString stringWithUTF8String:argv[2]]];
    id reflection=rd ? [NSJSONSerialization JSONObjectWithData:rd options:0 error:nil] : nil;
    if(!dxil.length || ![reflection isKindOfClass:NSDictionary.class]) {fprintf(stderr,"missing DXIL or reflection\n");return 1;}
    NSString *out=[NSString stringWithUTF8String:argv[4]];
    if([NSFileManager.defaultManager fileExistsAtPath:out]) {fprintf(stderr,"output exists\n");return 1;}
    void *library=dlopen(argv[3],RTLD_NOW|RTLD_LOCAL);
    if(!library) {fprintf(stderr,"converter: %s\n",dlerror());return 1;}
    LOAD_IR(IRCompilerCreate);LOAD_IR(IRCompilerDestroy);LOAD_IR(IRCompilerSetGlobalRootSignature);
    LOAD_IR(IRCompilerSetMinimumDeploymentTarget);LOAD_IR(IRCompilerSetMinimumGPUFamily);
    LOAD_IR(IRRootSignatureCreateFromDescriptor);LOAD_IR(IRRootSignatureDestroy);
    LOAD_IR(IRObjectCreateFromDXIL);LOAD_IR(IRObjectDestroy);LOAD_IR(IRCompilerAllocCompileAndLink);
    LOAD_IR(IRErrorGetCode);LOAD_IR(IRErrorDestroy);
    LOAD_IR(IRMetalLibBinaryCreate);LOAD_IR(IRMetalLibBinaryDestroy);
    LOAD_IR(IRObjectGetMetalLibBinary);LOAD_IR(IRMetalLibGetBytecodeSize);LOAD_IR(IRMetalLibGetBytecode);
    LOAD_IR(IRShaderReflectionCreate);LOAD_IR(IRShaderReflectionDestroy);LOAD_IR(IRObjectGetReflection);
    LOAD_IR(IRShaderReflectionGetEntryPointFunctionName);
    LOAD_IR(IRShaderReflectionAllocStringAndSerialize);LOAD_IR(IRShaderReflectionFreeString);
    std::vector<IRDescriptorRange1> resources,samplers;
    NSMutableArray *bindings=[NSMutableArray array];
    auto add=[&](IRDescriptorRangeType kind,unsigned index,NSString *name,bool uniform) {
        auto &ranges=kind==IRDescriptorRangeTypeSampler ? samplers : resources;
        IRDescriptorRange1 range={};range.RangeType=kind;range.NumDescriptors=1;
        range.BaseShaderRegister=index;range.RegisterSpace=0;range.OffsetInDescriptorsFromTableStart=(unsigned)ranges.size();
        ranges.push_back(range);
        [bindings addObject:@{@"name":name,@"register":@(index),@"slot":@(range.OffsetInDescriptorsFromTableStart),
            @"kind":@(kind),@"uniform":@(uniform),@"sampler":@(kind==IRDescriptorRangeTypeSampler)}];
    };
    // Slang's global uniform block is cb0. Resources stay in their reflected
    // register classes, including holes; table slots are dense and explicit.
    bool uniforms=false;
    for(NSDictionary *parameter in reflection[@"parameters"])
        if([parameter[@"binding"][@"kind"] isEqual:@"uniform"]) uniforms=true;
    if(uniforms) add(IRDescriptorRangeTypeCBV,0,@"globalParams",true);
    for(NSDictionary *parameter in reflection[@"parameters"]) {
        NSString *kind=parameter[@"binding"][@"kind"], *name=parameter[@"name"];
        unsigned index=[parameter[@"binding"][@"index"] unsignedIntValue];
        if([kind isEqual:@"uniform"]) continue;
        if([kind isEqual:@"constantBuffer"]) add(IRDescriptorRangeTypeCBV,index,name,false);
        else if([kind isEqual:@"shaderResource"]) add(IRDescriptorRangeTypeSRV,index,name,false);
        else if([kind isEqual:@"unorderedAccess"]) add(IRDescriptorRangeTypeUAV,index,name,false);
        else if([kind isEqual:@"samplerState"]) add(IRDescriptorRangeTypeSampler,index,name,false);
        else {fprintf(stderr,"unsupported reflection resource: %s\n",kind.UTF8String);return 1;}
    }
    IRRootParameter1 parameters[2]={};unsigned count=0;
    if(!resources.empty()) {
        parameters[count].ParameterType=IRRootParameterTypeDescriptorTable;
        parameters[count].DescriptorTable={(unsigned)resources.size(),resources.data()};
        parameters[count++].ShaderVisibility=IRShaderVisibilityAll;
    }
    if(!samplers.empty()) {
        parameters[count].ParameterType=IRRootParameterTypeDescriptorTable;
        parameters[count].DescriptorTable={(unsigned)samplers.size(),samplers.data()};
        parameters[count++].ShaderVisibility=IRShaderVisibilityAll;
    }
    IRVersionedRootSignatureDescriptor descriptor={};descriptor.version=IRRootSignatureVersion_1_1;
    descriptor.desc_1_1.NumParameters=count;descriptor.desc_1_1.pParameters=parameters;
    IRError *error=nullptr;
    IRRootSignature *root=IRRootSignatureCreateFromDescriptor(&descriptor,&error);
    if(!root) {fprintf(stderr,"root signature failed: %u\n",error?IRErrorGetCode(error):0);if(error)IRErrorDestroy(error);return 1;}
    IRCompiler *compiler=IRCompilerCreate();IRCompilerSetGlobalRootSignature(compiler,root);
    IRCompilerSetMinimumDeploymentTarget(compiler,IROperatingSystem_macOS,"26.0.0");
    IRCompilerSetMinimumGPUFamily(compiler,IRGPUFamilyApple7);
    IRObject *input=IRObjectCreateFromDXIL((const uint8_t*)dxil.bytes,dxil.length,IRBytecodeOwnershipNone);
    // The stripped DXIL retains its real entry name internally; use nullptr
    // instead of the obfuscated display name in Slang's JSON reflection.
    IRObject *compiled=IRCompilerAllocCompileAndLink(compiler,nullptr,input,&error);
    if(!compiled) {fprintf(stderr,"DXIL conversion failed: %u\n",error?IRErrorGetCode(error):0);if(error)IRErrorDestroy(error);return 1;}
    IRMetalLibBinary *binary=IRMetalLibBinaryCreate();
    if(!IRObjectGetMetalLibBinary(compiled,IRShaderStageCompute,binary)) {fprintf(stderr,"no compute metallib\n");return 1;}
    IRShaderReflection *convertedReflection=IRShaderReflectionCreate();
    if(!IRObjectGetReflection(compiled,IRShaderStageCompute,convertedReflection)) {fprintf(stderr,"no converted reflection\n");return 1;}
    const char *entry=IRShaderReflectionGetEntryPointFunctionName(convertedReflection);
    if(!entry || !*entry) {fprintf(stderr,"no entry name\n");return 1;}
    const char *json=IRShaderReflectionAllocStringAndSerialize(convertedReflection);
    NSDictionary *result=@{@"schema":@"anr_converted_kernel_v1",@"entry":[NSString stringWithUTF8String:entry],
        @"buffer_bind_point":@2,@"resource_count":@(resources.size()),@"sampler_count":@(samplers.size()),@"bindings":bindings};
    NSMutableData *bytecode=[NSMutableData dataWithLength:IRMetalLibGetBytecodeSize(binary)];
    IRMetalLibGetBytecode(binary,(uint8_t*)bytecode.mutableBytes);
    NSError *writeError=nil;
    if(![NSFileManager.defaultManager createDirectoryAtPath:out withIntermediateDirectories:YES attributes:nil error:&writeError]) {
        fprintf(stderr,"cannot create output: %s\n",writeError.localizedDescription.UTF8String);return 1;
    }
    NSData *bindingData=[NSJSONSerialization dataWithJSONObject:result options:NSJSONWritingPrettyPrinted error:&writeError];
    if(![bytecode writeToFile:[out stringByAppendingPathComponent:@"kernel.metallib"] options:NSDataWritingAtomic error:&writeError]
       || ![bindingData writeToFile:[out stringByAppendingPathComponent:@"bindings.json"] options:NSDataWritingAtomic error:&writeError]) {
        fprintf(stderr,"cannot save output: %s\n",writeError.localizedDescription.UTF8String);return 1;
    }
    if(json) [[NSString stringWithUTF8String:json] writeToFile:[out stringByAppendingPathComponent:@"converter-reflection.json"] atomically:YES encoding:NSUTF8StringEncoding error:nil];
    printf("converted %s (%lu bytes)\n",entry,(unsigned long)bytecode.length);
    if(json)IRShaderReflectionFreeString(json);
    IRShaderReflectionDestroy(convertedReflection);IRMetalLibBinaryDestroy(binary);
    IRObjectDestroy(compiled);IRObjectDestroy(input);IRCompilerDestroy(compiler);IRRootSignatureDestroy(root);
    if(error)IRErrorDestroy(error);dlclose(library);
    return 0;
} }
