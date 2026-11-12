// SPDX-License-Identifier: MIT
// Model initialization only. No synthetic FSR frame is claimed as DLSS proof.
#include "netal_live.h"
#import "dlss_sr_session.h"
#import "dlss_sr_texture_adapter.h"
#include <cstdio>

extern "C" void fsr4_live_set_inactive(const char *reason) { fprintf(stderr,"%s\n",reason); }
extern "C" void fsr4_live_set_status(const Fsr4LiveStatus *) {}
extern "C" void fsr4_live_get_status(Fsr4LiveStatus *out) { *out = {}; }

int main(int argc,char **argv) { @autoreleasepool {
    if(argc!=2) return 2;
    id<MTLDevice> device=MTLCreateSystemDefaultDevice();
    if(!device) {fprintf(stderr,"no Metal device\n");return 1;}
    NSError *error=nil;
    NSURL *url=[NSURL fileURLWithPath:[NSString stringWithUTF8String:argv[1]] isDirectory:YES];
    Fsr4NetalSRSession *session=[[Fsr4NetalSRSession alloc] initWithDevice:device runtimeBundleURL:url error:&error];
    if(!session) {fprintf(stderr,"Netal session: %s\n",error.localizedDescription.UTF8String);return 1;}
    Fsr4NetalTextureAdapter *adapter=[[Fsr4NetalTextureAdapter alloc] initWithDevice:device session:session error:&error];
    if(!adapter) {fprintf(stderr,"Netal texture adapter: %s\n",error.localizedDescription.UTF8String);return 1;}
    NSDictionary *result=@{@"schema":@"fsr4_netal_probe_v1",@"device":device.name,
        @"session_initialized":@YES,@"texture_adapter_initialized":@YES,@"inference_executed":@NO,
        @"rw":@(session.inputWidth),@"rh":@(session.inputHeight),@"uw":@(session.outputWidth),@"uh":@(session.outputHeight),
        @"pipeline_layers_present":@(session.pipelineLayersPresent),@"pipeline_layers_total":@(session.pipelineLayersTotal)};
    NSData *data=[NSJSONSerialization dataWithJSONObject:result options:NSJSONWritingPrettyPrinted error:nil];
    fwrite(data.bytes,1,data.length,stdout);puts("");
    return 0;
} }
