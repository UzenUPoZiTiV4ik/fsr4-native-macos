// SPDX-License-Identifier: MIT
// Replay only original P2 against independently evaluated DXIL integer SSA.
#include "fsr412_upstream_replay.h"
#include <iostream>
using namespace fsr412_original;
int main(int argc,char **argv) {@autoreleasepool {try {
    check(argc>=3,"usage: p2_probe ASSETS CAPTURE [o0|dot4|probe|literal]");
    std::string assets=argv[1],capture=argv[2],mode=argc>3?argv[3]:"o0";
    NSDictionary *receipt=json(capture+"/receipt.json");NSArray *size=receipt[@"output_size"],*render=receipt[@"render_size"];
    Replay replay(assets,[size[0] unsignedIntValue],[size[1] unsignedIntValue],[render[0] unsignedIntValue],[render[1] unsignedIntValue],"isolated","u8");
    NSData *b0=[NSData dataWithContentsOfFile:ns(capture+"/"+utf8(receipt[@"tensor_sizes"][@"path"]))];check(b0.length==144,"invalid captured b0");std::memcpy(replay.b0.contents,b0.bytes,b0.length);
    NSDictionary *stage=nil;for(NSDictionary *s in receipt[@"frames"][0][@"stages"])if([s[@"stage"] isEqual:@"pass1"])stage=s;check(stage!=nil,"missing captured DS");
    for(NSString *name in @[@"ds0",@"ds1"]) {
        NSDictionary *desc=stage[@"tensors"][name];NSData *data=[NSData dataWithContentsOfFile:ns(capture+"/"+utf8(desc[@"path"]))];
        size_t row=[desc[@"width_cells"] unsignedLongLongValue]*16,h=[desc[@"height_cells"] unsignedLongLongValue],pitch=[desc[@"scratch_row_bytes"] unsignedLongLongValue],base=[desc[@"scratch_byte_offset"] unsignedLongLongValue];
        check(data.length==row*h,"invalid captured DS shape");for(size_t y=0;y<h;++y)std::memcpy(static_cast<uint8_t *>(replay.scratch.contents)+base+pitch*y,static_cast<const uint8_t *>(data.bytes)+row*y,row);
    }
    Kernel &kernel=replay.kernels["pass2"];std::string identity=utf8(kernel.receipt[@"dxil_sha256"]),stem=identity+(mode=="literal"?"":"."+mode);
    NSError *error=nil;id<MTLLibrary> library=[replay.device newLibraryWithURL:[NSURL fileURLWithPath:ns(assets+"/"+stem+".metallib")] error:&error];check(library!=nil,"probe library: "+utf8(error.localizedDescription));
    kernel.pipeline=[replay.device newComputePipelineStateWithFunction:[library newFunctionWithName:@"main0"] error:&error];check(kernel.pipeline!=nil,"probe pipeline: "+utf8(error.localizedDescription));
    id<MTLBuffer> trace=[replay.device newBufferWithLength:408 options:MTLResourceStorageModeShared];std::memset(trace.contents,0,408);
    id<MTLCommandBuffer> cb=[replay.queue commandBuffer];id<MTLComputeCommandEncoder> e=[cb computeCommandEncoder];[e setComputePipelineState:kernel.pipeline];
    for(NSDictionary *binding in kernel.receipt[@"bindings"]) {std::string reg=utf8(binding[@"d3d_register"]);id<MTLBuffer> buffer=reg=="t18"?replay.initializer:reg=="u11"?replay.scratch:reg=="b0"?replay.b0:nil;check(buffer!=nil,"unexpected P2 resource");[e setBuffer:buffer offset:0 atIndex:[binding[@"metal_slot"] unsignedIntValue]];}
    if(mode=="probe")[e setBuffer:trace offset:0 atIndex:5];[e dispatchThreadgroups:replay.groups("pass2") threadsPerThreadgroup:kernel.threads];[e endEncoding];[cb commit];[cb waitUntilCompleted];check(cb.status==MTLCommandBufferStatusCompleted,"P2 GPU failure");
    NSMutableArray *pixels=[NSMutableArray array];for(const auto &xy:{std::pair<unsigned,unsigned>{0,0},{1,0},{16,4},{80,48}}) {
        NSMutableArray *values=[NSMutableArray array];size_t offset=(xy.second+1)*replay.quarterRowBytes+(xy.first+1)*16;
        for(unsigned bank=0;bank<2;++bank){const int8_t *p=static_cast<const int8_t *>(replay.scratch.contents)+offset+bank*replay.quarterPlaneBytes;for(unsigned c=0;c<16;++c)[values addObject:@(p[c])];}
        [pixels addObject:@{@"pixel":@[@(xy.first),@(xy.second)],@"output":values}];
    }
    NSMutableDictionary *result=[@{@"mode":ns(mode),@"pixels":pixels} mutableCopy];if(mode=="probe"){NSMutableArray *values=[NSMutableArray array];auto p=static_cast<const uint32_t *>(trace.contents);for(unsigned i=0;i<102;++i)[values addObject:@(p[i])];result[@"trace_words"]=values;}
    NSData *data=[NSJSONSerialization dataWithJSONObject:result options:NSJSONWritingPrettyPrinted error:nil];std::cout.write(static_cast<const char *>(data.bytes),data.length);std::cout<<"\n";return 0;
} catch(const std::exception &e){std::cerr<<e.what()<<"\n";return 1;}}}
