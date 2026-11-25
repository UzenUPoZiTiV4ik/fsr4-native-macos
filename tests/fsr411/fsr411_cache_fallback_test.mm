// SPDX-License-Identifier: MIT
// --no-compile must tolerate an interrupted alias build without changing a
// loaded stream's layout. Host checks need no Metal device or compiled cache.
#include "../../src/fsr41x/fsr411_runtime.h"
#include <cstdio>
#include <filesystem>
#include <functional>
#include <stdexcept>

namespace fs=std::filesystem;
static void check(bool condition,const std::string &message) {
    if(!condition) throw std::runtime_error(message);
}
struct TemporaryDirectory {
    fs::path path;
    TemporaryDirectory() {
        path=fsr411::utf8([NSTemporaryDirectory() stringByAppendingPathComponent:
            [@"fsr411-cache-" stringByAppendingString:[NSUUID UUID].UUIDString]]);
        fs::create_directories(path);
    }
    ~TemporaryDirectory() {std::error_code error;fs::remove_all(path,error);}
};
static void save_manifest(const fs::path &dir,NSDictionary *manifest) {
    NSData *bytes=[NSJSONSerialization dataWithJSONObject:manifest options:0 error:nil];
    check(bytes && [bytes writeToFile:fsr411::ns((dir/"bindings.json").string()) atomically:YES],"write manifest");
}
static NSMutableDictionary *read_manifest(const fs::path &dir) {
    NSData *bytes=[NSData dataWithContentsOfFile:fsr411::ns((dir/"bindings.json").string())];
    id result=bytes?[NSJSONSerialization JSONObjectWithData:bytes options:NSJSONReadingMutableContainers error:nil]:nil;
    check([result isKindOfClass:[NSMutableDictionary class]],"read fixture manifest");
    return result;
}
static void save_bytes(const fs::path &path) {
    check([[@"fixture" dataUsingEncoding:NSUTF8StringEncoding] writeToFile:fsr411::ns(path.string()) atomically:YES],"write fixture file");
}
static NSMutableDictionary *host_manifest() {
    NSMutableDictionary *entries=[NSMutableDictionary new];
    auto add=[&](NSString *name) {entries[name]=[@{@"library":@"kernel.metallib",@"entry":@"main",
        @"wgsize":@[@32,@1,@1],@"buffers":@[],@"textures":@[],@"samplers":@[]} mutableCopy];};
    for(unsigned i=0;i<27;++i) add(fsr411::ns(i==0?"pre":i==26?"post":
        std::string(i&1?"pad":"pass")+std::to_string(i/2)));
    for(NSString *name in @[@"ae",@"rcas",@"ae_finalize",@"pre_reuse"]) add(name);
    entries[@"post"][@"buffers"]=@[@[@7,@"Fsr411Tables"]];
    entries[@"post"][@"literal_tables"]=@"post.tables.bin";
    return [@{@"kernels":entries,@"auto_exposure_dispatches":@2,@"separate_reuse_pre":@YES} mutableCopy];
}
static void host_checks() {
    TemporaryDirectory temp;
    unsigned cases=0;
    auto run=[&](const char *name,const std::function<void(NSMutableDictionary *)> &edit,bool expected) {
        save_bytes(temp.path/"kernel.metallib");save_bytes(temp.path/"post.tables.bin");
        NSMutableDictionary *manifest=host_manifest();edit(manifest);save_manifest(temp.path,manifest);
        check(fsr411::shader_bundle_files_ready(temp.path.string())==expected,name);
        ++cases;
    };
    run("complete bundle",[](auto){},true);
    run("partial manifest",[](auto m){m[@"kernels"]=[@{@"pre":m[@"kernels"][@"pre"]} mutableCopy];},false);
    for(NSString *name in @[@"post",@"pass12",@"ae",@"rcas",@"ae_finalize",@"pre_reuse"])
        run(("missing entry "+fsr411::utf8(name)).c_str(),[&](auto m){[m[@"kernels"] removeObjectForKey:name];},false);
    run("missing POST library",[](auto m){m[@"kernels"][@"post"][@"library"]=@"absent.metallib";},false);
    run("missing POST table",[&](auto){fs::remove(temp.path/"post.tables.bin");},false);
    run("missing table metadata",[](auto m){[m[@"kernels"][@"post"] removeObjectForKey:@"literal_tables"];},false);
    run("invalid workgroup",[](auto m){m[@"kernels"][@"post"][@"wgsize"]=@[@0,@1,@1];},false);
    run("invalid bindings",[](auto m){m[@"kernels"][@"post"][@"buffers"]=@[@[@7]];},false);
    run("broken optional entry",[](auto m){m[@"kernels"][@"extra"]=@{};},false);
    run("empty library",[&](auto){check([[NSData data] writeToFile:fsr411::ns((temp.path/"kernel.metallib").string()) atomically:YES],"empty fixture");},false);
    run("directory is not a library",[&](auto m){m[@"kernels"][@"post"][@"library"]=@"subdir";fs::create_directory(temp.path/"subdir");},false);
    fs::create_symlink(temp.path/"kernel.metallib",temp.path/"linked.metallib");
    run("symlink library",[](auto m){m[@"kernels"][@"post"][@"library"]=@"linked.metallib";},true);
    run("dangling symlink",[&](auto m){m[@"kernels"][@"post"][@"library"]=@"linked.metallib";fs::remove(temp.path/"kernel.metallib");},false);
    run("layout validation belongs to loader",[](auto m){m[@"post_cnb_layout"]=@{@"offset":@1};m[@"post_cnb_alias"]=@NO;m[@"order"]=@[];},true);
    save_bytes(temp.path/"bindings.json");
    check(!fsr411::shader_bundle_files_ready(temp.path.string()),"invalid JSON");
    fs::remove(temp.path/"bindings.json");
    check(!fsr411::shader_bundle_files_ready(temp.path.string()),"missing manifest");
    std::printf("FSR 4.1.1 cache completeness host checks passed (%u cases)\n",cases+2);
}

static const std::string kLegacy="fsr411_1080_di0_lm1_jm0_cs0_ae0_o1_lossy_pc1";
static void install_bundle(const fs::path &source,const fs::path &destination) {
    check(fsr411::shader_bundle_files_ready(source.string()),"source cache incomplete: "+source.string());
    fs::remove_all(destination);fs::create_directories(destination);
    fs::copy_file(source/"bindings.json",destination/"bindings.json");
    NSDictionary *entries=read_manifest(source)[@"kernels"];
    for(NSString *name in entries) for(NSString *field in @[@"library",@"literal_tables"])
        if(NSString *file=entries[name][field]) {
            const auto relative=fsr411::utf8(file);
            if(!fs::exists(destination/relative)) fs::create_symlink(fs::absolute(source/relative),destination/relative);
        }
}
static id<MTLTexture> texture(id<MTLDevice> device,MTLPixelFormat format,unsigned width,unsigned height) {
    auto descriptor=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format width:width height:height mipmapped:NO];
    descriptor.storageMode=MTLStorageModeShared;descriptor.usage=MTLTextureUsageShaderRead|MTLTextureUsageShaderWrite;
    auto result=[device newTextureWithDescriptor:descriptor];check(result!=nil,"texture allocation");return result;
}
static fsr411::Frame make_frame(id<MTLDevice> device) {
    constexpr unsigned rw=32,rh=24,uw=64,uh=48;
    fsr411::Frame f;
    f.color=texture(device,MTLPixelFormatRGBA32Float,rw,rh);f.depth=texture(device,MTLPixelFormatR32Float,rw,rh);
    f.motion=texture(device,MTLPixelFormatRG32Float,rw,rh);f.output=texture(device,MTLPixelFormatRGBA32Float,uw,uh);
    f.exposure=texture(device,MTLPixelFormatR32Float,1,1);
    f.input.geometry={{rw,rh,0,0},{rw,rh,0,0},{rw,rh,0,0},{uw,uh,0,0},rw,rh};
    f.input.rw=rw;f.input.rh=rh;f.input.uw=uw;f.input.uh=uh;
    f.input.mvScaleX=f.input.mvScaleY=1;f.input.preExposure=1;f.input.hasExposureTexture=1;
    std::vector<float> color(rw*rh*4),depth(rw*rh,0.5f),motion(rw*rh*2,0.f);
    for(unsigned y=0;y<rh;++y) for(unsigned x=0;x<rw;++x) {
        const unsigned p=(y*rw+x)*4;color[p]=0.1f+0.7f*x/rw;color[p+1]=0.1f+0.6f*y/rh;
        color[p+2]=0.2f+float((x*7+y*3)%23)/30.f;color[p+3]=1.f;
    }
    [f.color replaceRegion:MTLRegionMake2D(0,0,rw,rh) mipmapLevel:0 withBytes:color.data() bytesPerRow:rw*16];
    [f.depth replaceRegion:MTLRegionMake2D(0,0,rw,rh) mipmapLevel:0 withBytes:depth.data() bytesPerRow:rw*4];
    [f.motion replaceRegion:MTLRegionMake2D(0,0,rw,rh) mipmapLevel:0 withBytes:motion.data() bytesPerRow:rw*8];
    float exposure=1.f;[f.exposure replaceRegion:MTLRegionMake2D(0,0,1,1) mipmapLevel:0 withBytes:&exposure bytesPerRow:4];
    return f;
}
static void gpu_checks(const std::string &root,const fs::path &source) {
    auto device=MTLCreateSystemDefaultDevice();auto queue=[device newCommandQueue];check(device && queue,"Metal device/queue");
    TemporaryDirectory temp;
    const auto alias=temp.path/(kLegacy+"_ca1");
    install_bundle(source/kLegacy,temp.path/kLegacy);
    fs::create_directory_symlink(fs::absolute(source/"fsr411_native_1080_lossy"),temp.path/"fsr411_native_1080_lossy");
    fsr411::Options options;options.allowCompile=false;options.optimize=1;options.profile=fsr411::Profile::Lossy;
    std::vector<std::vector<float>> expected;
    auto sequence=[&](const char *name,bool useAlias,const std::function<void()> &afterFirst={}) {
        fsr411::Runtime runtime(device,root,temp.path.string());fsr411::View view;auto frame=make_frame(device);
        for(unsigned i=0;i<3;++i) {
            frame.input.reset=i==0;frame.input.jitterX=i==1?-0.125f:0.25f;frame.input.jitterY=0.125f;
            auto command=[queue commandBuffer];std::string error;
            const bool encoded=runtime.encode(command,view,frame,options,error);
            check(encoded,std::string(name)+": "+error);
            [command commit];[command waitUntilCompleted];
            check(command.status==MTLCommandBufferStatusCompleted,std::string(name)+": GPU completion");
            check(view.postCnbAlias==useAlias,std::string(name)+": selected wrong cache");
            check(view.skippedInference==(i==1),std::string(name)+": full/reuse cadence");
            std::vector<float> output(64*48*4);
            [frame.output getBytes:output.data() bytesPerRow:64*16 fromRegion:MTLRegionMake2D(0,0,64,48) mipmapLevel:0];
            for(float value:output) check(std::isfinite(value),std::string(name)+": nonfinite output");
            if(expected.size()<3) expected.push_back(output);
            else check(!std::memcmp(output.data(),expected[i].data(),output.size()*sizeof(float)),
                std::string(name)+": differs from legacy frame "+std::to_string(i));
            if(!i && afterFirst) afterFirst();
        }
        std::printf("cache GPU PASS: %s (reset/full, reuse, full; byte-identical output)\n",name);std::fflush(stdout);
    };
    sequence("complete legacy",false);
    install_bundle(source/(kLegacy+"_ca1"),alias);
    sequence("complete alias and loaded session reuse",true,[&]{fs::remove(alias/"bindings.json");});
    install_bundle(source/(kLegacy+"_ca1"),alias);
    auto fullManifest=read_manifest(alias);
    auto partial=read_manifest(alias);partial[@"kernels"]=[@{@"pre":partial[@"kernels"][@"pre"]} mutableCopy];save_manifest(alias,partial);
    sequence("partial manifest and pinned legacy session",false,[&]{save_manifest(alias,fullManifest);});
    for(NSString *field in @[@"entry",@"library",@"literal_tables"]) {
        install_bundle(source/(kLegacy+"_ca1"),alias);auto manifest=read_manifest(alias);
        if([field isEqual:@"entry"]) [manifest[@"kernels"] removeObjectForKey:@"post"];
        else fs::remove(alias/fsr411::utf8(manifest[@"kernels"][@"post"][field]));
        save_manifest(alias,manifest);
        sequence(("missing POST "+fsr411::utf8(field)).c_str(),false);
    }
    install_bundle(source/(kLegacy+"_ca1"),alias);auto malformed=read_manifest(alias);
    malformed[@"post_cnb_layout"][@"offset"]=@1;save_manifest(alias,malformed);
    fsr411::Runtime runtime(device,root,temp.path.string());fsr411::View view;auto frame=make_frame(device);
    auto command=[queue commandBuffer];std::string error;
    const bool encoded=runtime.encode(command,view,frame,options,error);
    check(!encoded && error.find("layout metadata mismatch")!=std::string::npos,
        "complete malformed alias must fail strict validation: "+error);
    std::puts("cache GPU PASS: complete malformed layout rejected");
}
int main(int argc,char **argv) { @autoreleasepool {
    try {
        if(argc==1) host_checks();
        else if(argc==4 && std::string(argv[1])=="--gpu") gpu_checks(argv[2],argv[3]);
        else {std::fprintf(stderr,"usage: fsr411_cache_fallback_test [--gpu repository builddir]\n");return 2;}
    } catch(const std::exception &error) {std::fprintf(stderr,"FSR 4.1.1 cache regression failed: %s\n",error.what());return 1;}
    return 0;
} }
