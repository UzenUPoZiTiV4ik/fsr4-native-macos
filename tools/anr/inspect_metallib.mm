// SPDX-License-Identifier: MIT
// Read the actual Metal ABI from an explicit local compiled shader.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cstdio>
int main(int argc,char **argv){@autoreleasepool{
    if(argc!=2)return 2;
    id<MTLDevice> device=MTLCreateSystemDefaultDevice();NSError *error=nil;
    id<MTLLibrary> library=[device newLibraryWithURL:[NSURL fileURLWithPath:[NSString stringWithUTF8String:argv[1]]] error:&error];
    if(!library){fprintf(stderr,"%s\n",error.localizedDescription.UTF8String);return 1;}
    NSMutableArray *functions=[NSMutableArray array];
    for(NSString *name in library.functionNames){
        id<MTLFunction> function=[library newFunctionWithName:name];MTLComputePipelineReflection *reflection=nil;
        id<MTLComputePipelineState> pipeline=[device newComputePipelineStateWithFunction:function
            options:MTLPipelineOptionArgumentInfo|MTLPipelineOptionBufferTypeInfo reflection:&reflection error:&error];
        if(!pipeline){fprintf(stderr,"%s\n",error.localizedDescription.UTF8String);return 1;}
        NSMutableArray *arguments=[NSMutableArray array];
        for(MTLArgument *argument in reflection.arguments){
            NSMutableDictionary *record=[@{@"name":argument.name,@"index":@(argument.index),@"type":@(argument.type),@"active":@(argument.active)} mutableCopy];
            if(argument.type==MTLArgumentTypeBuffer){
                record[@"bytes"]=@(argument.bufferDataSize);record[@"alignment"]=@(argument.bufferAlignment);
                NSMutableArray *members=[NSMutableArray array];
                for(MTLStructMember *member in argument.bufferStructType.members)
                    [members addObject:@{@"name":member.name,@"offset":@(member.offset),@"type":@(member.dataType)}];
                record[@"members"]=members;
            }
            [arguments addObject:record];
        }
        [functions addObject:@{@"function":name,@"arguments":arguments,@"static_threadgroup_bytes":@(pipeline.staticThreadgroupMemoryLength)}];
    }
    NSData *json=[NSJSONSerialization dataWithJSONObject:functions options:NSJSONWritingPrettyPrinted error:nil];
    fwrite(json.bytes,1,json.length,stdout);puts("");return 0;
}}
