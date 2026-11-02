// SPDX-License-Identifier: MIT
// Independent, data-only Metal loader for external original-DXIL replay assets.
// No shader policy, weights, GPL shader source or production runtime is embedded.
#pragma once
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <CommonCrypto/CommonDigest.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace fsr412_original {
inline NSString *ns(const std::string &s) { return [NSString stringWithUTF8String:s.c_str()]; }
inline std::string utf8(NSString *s) { return s ? std::string(s.UTF8String) : std::string(); }
inline void check(bool ok, const std::string &message) { if (!ok) throw std::runtime_error(message); }
inline std::string hash(const void *data, size_t size) {
    unsigned char digest[CC_SHA256_DIGEST_LENGTH]; CC_SHA256(data, CC_LONG(size), digest);
    const char *digits="0123456789abcdef"; std::string result;
    for (unsigned char c: digest) { result += digits[c>>4]; result += digits[c&15]; }
    return result;
}
inline NSDictionary *json(const std::string &path) {
    NSData *data=[NSData dataWithContentsOfFile:ns(path)]; NSError *error=nil;
    id value=data?[NSJSONSerialization JSONObjectWithData:data options:0 error:&error]:nil;
    check([value isKindOfClass:[NSDictionary class]], "cannot read JSON: "+path+" "+utf8(error.localizedDescription));
    return value;
}
inline unsigned ceil_div(unsigned n,unsigned d) { return (n+d-1)/d; }
struct FrameConstants {
    float invSize[2], scale[2], invScale[2], jitter[2], mvScale[2], texSize[2], maxRenderSize[2], mvJitterCancellation[2];
    uint32_t width,height,reset,widthLR,heightLR; float preExposure,previousPreExposure;
    uint32_t rcasEnabled; float rcasSharpness,pad;
};
static_assert(sizeof(FrameConstants)==104,"original b1 ABI");
struct Kernel { id<MTLComputePipelineState> pipeline=nil; NSDictionary *receipt=nil; MTLSize threads; };

// Public original resources permit A/B substitution of neural stages only.
class Replay {
public:
    id<MTLDevice> device=nil; id<MTLCommandQueue> queue=nil;
    id<MTLBuffer> scratch=nil,initializer=nil,b0=nil,b1=nil;
    id<MTLSamplerState> sampler=nil;
    std::map<std::string,id<MTLTexture>> textures;
    std::map<std::string,Kernel> kernels;
    std::string assets,postMode,recurrentFormat,neuralProfile;
    size_t cacheRelocatedByteOffset=99821056,cacheCopyBytes=33273664;
    unsigned width=0,height=0,renderWidth=0,renderHeight=0;
    size_t logicalScratchBytes=99820992, rowBytes=30752, planeBytes=33273664, quarterRowBytes=15392, quarterPlaneBytes=8342464;
    bool guardPaddingOnly=false;

    Replay(const std::string &directory,unsigned w,unsigned h,unsigned rw,unsigned rh,const std::string &mode="literal",const std::string &recurrent="u8",const std::string &profile="correctness"):assets(directory),postMode(mode),recurrentFormat(recurrent),neuralProfile(profile),width(w),height(h),renderWidth(rw),renderHeight(rh) {
        check(profile=="correctness" || profile=="legacy","unknown original neural compiler profile");
        check(recurrent=="u8" || recurrent=="f16","unknown recurrent fixture format");
        check(mode=="literal" || mode=="ordered" || mode=="isolated","unknown controlled POST mode");
        check(w>0 && h>0 && w<=3840 && h<=2160 && w%4==0 && h%4==0 && rw>0 && rh>0,"this loader is the independently selected 2160 permutation; dimensions must fit and be divisible by four");
        NSDictionary *contract=json(assets+"/contract.json");
        check([contract[@"source_dll_sha256"] isEqual:@"d9b0a1f5781c2d633a327272f6f1c8607057df7cafd0233cf38c7395c8f7dbc5"],"wrong source DLL contract");
        device=MTLCreateSystemDefaultDevice(); check(device!=nil,"Metal device unavailable"); queue=[device newCommandQueue];
        // Independently guarded per-word resource IO preserves D3D zero/discard
        // behavior. Allocate exactly the original logical resource extent.
        scratch=[device newBufferWithLength:(postMode=="isolated"?cacheRelocatedByteOffset+cacheCopyBytes:logicalScratchBytes) options:MTLResourceStorageModeShared];
        b0=[device newBufferWithLength:144 options:MTLResourceStorageModeShared];
        b1=[device newBufferWithLength:112 options:MTLResourceStorageModeShared];
        check(scratch && b0 && b1,"buffer allocation failed"); std::memset(scratch.contents,0,scratch.length); std::memset(b0.contents,0,b0.length);
        NSData *weights=[NSData dataWithContentsOfFile:ns(assets+"/initializers-normal.bin")];
        check(weights.length==131072 && hash(weights.bytes,weights.length)=="621a99e22c4259c3200097513947d072633f67634eaf95365fb4123b4c56ac10","full original Lite t18 initializer mismatch");
        initializer=[device newBufferWithBytes:weights.bytes length:weights.length options:MTLResourceStorageModeShared];
        // Original Lite graph-edge constants, not pass numbers. Only slot 4 is
        // read by this selected p2/p3; retain all nine declared uint4 slots.
        const unsigned shifts[9]={1,0,1,2,2,1,0,0,0}; auto sizes=static_cast<uint32_t *>(b0.contents);
        for(unsigned i=0;i<9;++i) { sizes[4*i]=w>>shifts[i]; sizes[4*i+1]=h>>shifts[i]; }
        MTLSamplerDescriptor *sd=[MTLSamplerDescriptor new]; sd.minFilter=MTLSamplerMinMagFilterLinear; sd.magFilter=MTLSamplerMinMagFilterLinear;
        sd.mipFilter=MTLSamplerMipFilterNotMipmapped; sd.sAddressMode=MTLSamplerAddressModeClampToEdge; sd.tAddressMode=MTLSamplerAddressModeClampToEdge;
        sd.normalizedCoordinates=YES; sampler=[device newSamplerStateWithDescriptor:sd];
        textures["input"]=texture(MTLPixelFormatRGBA32Float,rw,rh);
        textures["velocity"]=texture(MTLPixelFormatRG32Float,rw,rh);
        textures["depth"]=texture(MTLPixelFormatR32Float,rw,rh);
        textures["exposure"]=texture(MTLPixelFormatR32Float,2,1);
        for(const auto &name:{"history_prev","history_next","reprojected","output"}) textures[name]=texture(MTLPixelFormatRGBA16Float,w,h);
        for(const auto &name:{"recurrent_prev","recurrent_next"}) textures[name]=texture(recurrentFormat=="u8"?MTLPixelFormatRGBA8Unorm:MTLPixelFormatRGBA16Float,w,h);
        for(const auto &entry: textures) clear(entry.second);
        for(NSDictionary *stage in contract[@"active_2160_normal_di1_lm1_jm0_cs0_ae0"]) {
            std::string name=utf8(stage[@"stage"]),identity=utf8(stage[@"sha256"]);
            std::string stem=identity+(name=="post" && postMode!="literal"?"."+postMode:"");
            if(neuralProfile=="correctness" && (name=="pass2" || name=="pass3")) stem=identity+".o0";
            NSDictionary *receipt=json(assets+"/"+stem+".translation.json");
            check([receipt[@"dxil_sha256"] isEqual:stage[@"sha256"]] && receipt[@"io_guard"]!=nil,"shader receipt/IO guard mismatch");
            NSData *msl=[NSData dataWithContentsOfFile:ns(assets+"/"+stem+".metal")];
            check(msl && hash(msl.bytes,msl.length)==utf8(receipt[@"metal_sha256"]),"translated source identity mismatch");
            NSError *error=nil; id<MTLLibrary> library=[device newLibraryWithURL:[NSURL fileURLWithPath:ns(assets+"/"+stem+".metallib")] error:&error];
            check(library!=nil,"library "+name+": "+utf8(error.localizedDescription)); id<MTLFunction> function=[library newFunctionWithName:receipt[@"entry"]];
            check(function!=nil,"missing original entry"); id<MTLComputePipelineState> pipeline=[device newComputePipelineStateWithFunction:function error:&error];
            check(pipeline!=nil,"pipeline "+name+": "+utf8(error.localizedDescription)); NSArray *wg=stage[@"workgroup"];
            kernels[name]={pipeline,receipt,MTLSizeMake([wg[0] unsignedIntValue],[wg[1] unsignedIntValue],[wg[2] unsignedIntValue])};
        }
    }
    id<MTLTexture> texture(MTLPixelFormat format,unsigned w,unsigned h) {
        MTLTextureDescriptor *td=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format width:w height:h mipmapped:NO];
        td.storageMode=MTLStorageModeShared; td.usage=MTLTextureUsageShaderRead|MTLTextureUsageShaderWrite;
        id<MTLTexture> t=[device newTextureWithDescriptor:td]; check(t!=nil,"texture allocation failed"); return t;
    }
    static unsigned bytes_per_pixel(id<MTLTexture> t) {
        switch(t.pixelFormat) {case MTLPixelFormatRGBA8Unorm:return 4;case MTLPixelFormatRGBA16Float: return 8;case MTLPixelFormatRGBA32Float:return 16;case MTLPixelFormatRG32Float:return 8;case MTLPixelFormatR32Float:return 4;default:throw std::runtime_error("unsupported fixture texture format");}
    }
    void clear(id<MTLTexture> t) { std::vector<uint8_t> zero(t.width*t.height*bytes_per_pixel(t)); upload(t,zero.data()); }
    void upload(id<MTLTexture> t,const void *bytes) { [t replaceRegion:MTLRegionMake2D(0,0,t.width,t.height) mipmapLevel:0 withBytes:bytes bytesPerRow:t.width*bytes_per_pixel(t)]; }
    NSData *read_texture(id<MTLTexture> t) { NSMutableData *data=[NSMutableData dataWithLength:t.width*t.height*bytes_per_pixel(t)]; [t getBytes:data.mutableBytes bytesPerRow:t.width*bytes_per_pixel(t) fromRegion:MTLRegionMake2D(0,0,t.width,t.height) mipmapLevel:0]; return data; }
    id<MTLTexture> d3d_texture(const std::string &reg) {
        const std::map<std::string,std::string> names={{"t0","history_prev"},{"t1","velocity"},{"t2","depth"},{"t3","input"},{"t4","recurrent_prev"},{"t9","reprojected"},{"t17","exposure"},{"u1","history_next"},{"u2","output"},{"u3","reprojected"},{"u6","recurrent_next"}};
        auto it=names.find(reg); check(it!=names.end(),"unknown texture register "+reg); return textures.at(it->second);
    }
    void set_frame(float jitterX,float jitterY,bool reset,float preExposure=1,float previousPreExposure=1) {
        FrameConstants c={}; c.width=width;c.height=height;c.widthLR=renderWidth;c.heightLR=renderHeight;c.reset=reset;
        c.invSize[0]=1.0f/width;c.invSize[1]=1.0f/height;c.scale[0]=float(width)/renderWidth;c.scale[1]=float(height)/renderHeight;
        c.invScale[0]=float(renderWidth)/width;c.invScale[1]=float(renderHeight)/height;c.jitter[0]=jitterX;c.jitter[1]=jitterY;
        c.mvScale[0]=c.mvScale[1]=1.0f;c.texSize[0]=float(width);c.texSize[1]=float(height);
        c.maxRenderSize[0]=float(renderWidth);c.maxRenderSize[1]=float(renderHeight);c.preExposure=preExposure;c.previousPreExposure=previousPreExposure;
        std::memset(b1.contents,0,b1.length);std::memcpy(b1.contents,&c,sizeof(c));
    }
    void load_pass1_border() {
        const std::string identity=utf8(kernels.at("pass1").receipt[@"dxil_sha256"]);
        NSDictionary *receipt=json(assets+"/"+identity+".border-only.translation.json");NSError *error=nil;
        NSData *source=[NSData dataWithContentsOfFile:ns(assets+"/"+identity+".border-only.metal")];
        check(source && hash(source.bytes,source.length)==utf8(receipt[@"metal_sha256"]),"pass1 border source identity mismatch");
        id<MTLLibrary> library=[device newLibraryWithURL:[NSURL fileURLWithPath:ns(assets+"/"+utf8(receipt[@"library"]))] error:&error];
        check(library!=nil,"pass1 border library: "+utf8(error.localizedDescription));
        id<MTLComputePipelineState> pipeline=[device newComputePipelineStateWithFunction:[library newFunctionWithName:receipt[@"entry"]] error:&error];
        check(pipeline!=nil,"pass1 border pipeline: "+utf8(error.localizedDescription));
        kernels["pass1_border"]={pipeline,receipt,MTLSizeMake(64,2,1)};
    }
    MTLSize groups(const std::string &stage) const {
        if(stage=="pre") return MTLSizeMake(ceil_div(width,16),ceil_div(height,16),1);
        if(stage=="post") return MTLSizeMake(ceil_div(width,32),ceil_div(height,32),1);
        if(stage=="border" || stage=="pass1_border") return MTLSizeMake(1,1,1);
        if(stage=="pass1") return MTLSizeMake(ceil_div(width/2,64),height/2,1);
        return MTLSizeMake(ceil_div(width/4,64),height/4,1);
    }
    void dispatch(const std::string &name) {
        Kernel &k=kernels.at(name);
        const bool orderedPost=name=="post" && postMode=="ordered",isolatedPost=name=="post" && postMode=="isolated";
        auto bytes=static_cast<uint8_t *>(scratch.contents);
        if(isolatedPost) std::memcpy(bytes+cacheRelocatedByteOffset,bytes+49941264,cacheCopyBytes);
        auto run=[&](MTLSize groupCount,unsigned row) {
            id<MTLCommandBuffer> command=[queue commandBuffer]; id<MTLComputeCommandEncoder> encoder=[command computeCommandEncoder];
            [encoder setComputePipelineState:k.pipeline];
            for(NSDictionary *binding in k.receipt[@"bindings"]) {
                std::string reg=utf8(binding[@"d3d_register"]),kind=utf8(binding[@"metal_kind"]); unsigned slot=[binding[@"metal_slot"] unsignedIntValue];
                if(kind=="buffer") { id<MTLBuffer> buffer=reg=="t18"?initializer:reg=="u11"?scratch:reg=="b0"?b0:reg=="b1"?b1:nil; check(buffer!=nil,"unknown buffer register "+reg);[encoder setBuffer:buffer offset:0 atIndex:slot]; }
                else if(kind=="texture") [encoder setTexture:d3d_texture(reg) atIndex:slot];
                else if(kind=="sampler") {check(reg=="s0","unknown sampler register");[encoder setSamplerState:sampler atIndex:slot];}
                else throw std::runtime_error("unknown binding kind");
            }
            if(orderedPost) [encoder setBytes:&row length:sizeof(row) atIndex:4];
            [encoder dispatchThreadgroups:groupCount threadsPerThreadgroup:k.threads];[encoder endEncoding];[command commit];[command waitUntilCompleted];
            check(command.status==MTLCommandBufferStatusCompleted,"original stage "+name+" failed: "+utf8(command.error.localizedDescription));
        };
        MTLSize groupCount=groups(name);
        if(orderedPost) { for(unsigned y=0;y<groupCount.height;++y) run(MTLSizeMake(groupCount.width,1,1),y); }
        else run(groupCount,0);
        if(isolatedPost) std::memcpy(bytes+49941264,bytes+cacheRelocatedByteOffset,cacheCopyBytes);
    }
    void advance_history() { std::swap(textures["history_prev"],textures["history_next"]);std::swap(textures["recurrent_prev"],textures["recurrent_next"]); }
    NSData *compact_tensor(size_t base,size_t stride,unsigned w,unsigned h) {
        const size_t row=size_t(w)*16;check(base+stride*(h-1)+row<=logicalScratchBytes,"snapshot range outside logical scratch");
        NSMutableData *data=[NSMutableData dataWithLength:row*h]; auto src=static_cast<const uint8_t *>(scratch.contents)+base;auto dst=static_cast<uint8_t *>(data.mutableBytes);
        for(unsigned y=0;y<h;++y) std::memcpy(dst+row*y,src+stride*y,row);return data;
    }
};
} // namespace fsr412_original
