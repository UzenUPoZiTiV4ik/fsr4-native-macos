// SPDX-License-Identifier: MIT
// Standalone controlled Metal replay of the shipped original Lite Lossy DXIL.
// Compile with clang++ -std=c++17 -fobjc-arc -framework Foundation -framework Metal.
#include "fsr412_upstream_replay.h"
#include <filesystem>
#include <iostream>

using namespace fsr412_original;
namespace {
NSDictionary *save_data(const std::string &directory,const std::string &name,NSData *data) {
    check([data writeToFile:ns(directory+"/"+name) atomically:YES],"snapshot write failed");
    return @{@"path":ns(name),@"bytes":@(data.length),@"sha256":ns(hash(data.bytes,data.length))};
}
void save_json(const std::string &path,id value) {
    NSError *error=nil;NSData *data=[NSJSONSerialization dataWithJSONObject:value options:NSJSONWritingPrettyPrinted error:&error];
    check(data && [data writeToFile:ns(path) atomically:YES],"JSON write failed: "+utf8(error.localizedDescription));
}
void fixture(Replay &r,unsigned frame,bool motion,bool exposureRamp) {
    std::vector<float> color(r.renderWidth*r.renderHeight*4),depth(r.renderWidth*r.renderHeight),velocity(r.renderWidth*r.renderHeight*2);
    const float exposure=exposureRamp?std::pow(1.1f,float(frame)):1.0f;
    for(unsigned y=0;y<r.renderHeight;++y) for(unsigned x=0;x<r.renderWidth;++x) {
        size_t i=size_t(y)*r.renderWidth+x;float fx=(float(x)+0.5f)/r.renderWidth,fy=(float(y)+0.5f)/r.renderHeight;
        unsigned movingX=x+(motion?frame:0);float stripe=((movingX/3+y/5)&1)?0.8f:0.03f;
        color[4*i]=0.03f+0.3f*fx+stripe*0.4f;color[4*i+1]=0.02f+0.5f*fy;color[4*i+2]=0.03f+0.2f*stripe;color[4*i+3]=1;
        // Inverted-depth permutation. Add a moving foreground boundary and
        // thin bright details; these are fixture choices, not upstream policy.
        bool foreground=int(x)>int(r.renderWidth/2)-(motion?int(frame):0);
        depth[i]=foreground?0.9f:0.2f;
        if(movingX%31==0 && y%19==0) color[4*i]=color[4*i+1]=color[4*i+2]=2.5f;
        velocity[2*i]=(motion && foreground)?1.0f/r.width:0;velocity[2*i+1]=0;
    }
    r.upload(r.textures["input"],color.data());r.upload(r.textures["velocity"],velocity.data());r.upload(r.textures["depth"],depth.data());
    float exposureValues[2]={exposure,exposureRamp && frame?std::pow(1.1f,float(frame-1)):1.0f};r.upload(r.textures["exposure"],exposureValues);
}
}
int main(int argc,char **argv) {
    @autoreleasepool { try {
        std::string assets="build/fsr412_upstream",out="build/fsr412_upstream-oracle/run",postMode="literal",recurrentFormat="u8",neuralProfile="correctness";unsigned width=640,height=384,rw=384,rh=224,frames=4;
        bool motion=true,exposureRamp=false;
        for(int i=1;i<argc;++i) {
            std::string arg=argv[i];auto next=[&] {check(i+1<argc,"missing option value");return std::string(argv[++i]);};
            if(arg=="--neural-profile") neuralProfile=next();else if(arg=="--recurrent-format") recurrentFormat=next();else if(arg=="--post-mode") postMode=next();else if(arg=="--assets") assets=next();else if(arg=="--out") out=next();else if(arg=="--width") width=std::stoul(next());else if(arg=="--height") height=std::stoul(next());
            else if(arg=="--render-width") rw=std::stoul(next());else if(arg=="--render-height") rh=std::stoul(next());else if(arg=="--frames") frames=std::stoul(next());
            else if(arg=="--still") motion=false;else if(arg=="--exposure-ramp") exposureRamp=true;else throw std::runtime_error("unknown option "+arg);
        }
        check(frames>0 && frames<=32,"frames out of range");
        auto canonical=std::filesystem::weakly_canonical(out);auto build=std::filesystem::weakly_canonical("build/fsr412_upstream-oracle");
        check(canonical.string().find(build.string()+"/")==0,"state outputs must remain inside build/fsr412_upstream-oracle");
        std::filesystem::create_directories(canonical);out=canonical.string();Replay replay(assets,width,height,rw,rh,postMode,recurrentFormat,neuralProfile);
        NSMutableArray *frameRecords=[NSMutableArray array];
        const std::vector<std::string> order={"pre","pass1","pass2","border","pass3","post"};
        NSDictionary *initialConstants=save_data(out,"tensor-sizes.bin",[NSData dataWithBytes:replay.b0.contents length:replay.b0.length]);
        for(unsigned frame=0;frame<frames;++frame) {
            fixture(replay,frame,motion,exposureRamp);float jx=frame&1?-0.25f:0.25f,jy=frame&1?0.0f:-0.25f;
            replay.set_frame(jx,jy,frame==0);NSMutableArray *stages=[NSMutableArray array];NSMutableDictionary *input=[NSMutableDictionary dictionary];
            std::string prefix="f"+std::to_string(frame);
            input[@"b1"]=save_data(out,prefix+"-b1.bin",[NSData dataWithBytes:replay.b1.contents length:replay.b1.length]);
            for(const auto &name:{"input","velocity","depth","exposure","history_prev","recurrent_prev"}) input[ns(name)]=save_data(out,prefix+"-input-"+name+".bin",replay.read_texture(replay.textures[name]));
            for(const std::string &stage:order) {
                replay.dispatch(stage);NSMutableDictionary *tensors=[NSMutableDictionary dictionary];
                auto tensor=[&](const std::string &name,size_t base,size_t pitch,unsigned w,unsigned h) {
                    NSMutableDictionary *desc=[save_data(out,prefix+"-"+stage+"-"+name+".bin",replay.compact_tensor(base,pitch,w,h)) mutableCopy];
                    desc[@"scratch_byte_offset"]=@(base);desc[@"scratch_row_bytes"]=@(pitch);desc[@"width_cells"]=@(w);desc[@"height_cells"]=@(h);desc[@"cell_bytes"]=@16;tensors[ns(name)]=desc;
                };
                unsigned hw=width/2,hh=height/2,qw=width/4,qh=height/4;
                if(stage=="pre") tensor("pre",0,replay.rowBytes,hw+2,hh+2);
                if(stage=="pass1") {tensor("cnb",replay.planeBytes,replay.rowBytes,hw+2,hh+2);tensor("ds0",2*replay.planeBytes,replay.quarterRowBytes,qw+2,qh+2);tensor("ds1",2*replay.planeBytes+replay.quarterPlaneBytes,replay.quarterRowBytes,qw+2,qh+2);}
                if(stage=="pass2" || stage=="border") {tensor("fnb0",0,replay.quarterRowBytes,qw+2,qh+2);tensor("fnb1",replay.quarterPlaneBytes,replay.quarterRowBytes,qw+2,qh+2);}
                if(stage=="pass3") tensor("upsample",2*replay.planeBytes,replay.rowBytes,hw+2,hh+2);
                if(stage=="post") tensor("saved-cnb",3*replay.planeBytes/2+replay.rowBytes+16,replay.rowBytes,hw,hh);
                const auto marker=static_cast<const uint32_t *>(replay.scratch.contents)+33273792/4;
                MTLSize groups=replay.groups(stage);NSMutableDictionary *record=[@{@"stage":ns(stage),@"dxil_sha256":replay.kernels.at(stage).receipt[@"dxil_sha256"],
                    @"dispatch_groups":@[@(groups.width),@(groups.height),@(groups.depth)],@"scratch_logical_sha256":ns(hash(replay.scratch.contents,replay.logicalScratchBytes)),
                    @"skip_marker":@[@(marker[0]),@(marker[1]),@(marker[2]),@(marker[3])],@"tensors":tensors} mutableCopy];
                record[@"marker_bytes"]=save_data(out,prefix+"-"+stage+"-marker.bin",[NSData dataWithBytes:marker length:16]);
                if(stage=="pre" || stage=="post") {
                    NSMutableDictionary *images=[NSMutableDictionary dictionary];
                    for(const auto &name:(stage=="pre"?std::vector<std::string>{"reprojected"}:std::vector<std::string>{"output","history_next","recurrent_next"})) images[ns(name)]=save_data(out,prefix+"-"+stage+"-"+name+".bin",replay.read_texture(replay.textures[name]));
                    record[@"textures"]=images;
                }
                [stages addObject:record];
            }
            [frameRecords addObject:@{@"frame":@(frame),@"reset":@(frame==0),@"jitter":@[@(jx),@(jy)],@"inputs":input,@"stages":stages}];replay.advance_history();
        }
        NSData *output=replay.read_texture(replay.textures["output"]);const _Float16 *pixels=static_cast<const _Float16 *>(output.bytes);size_t finite=0,nonzero=0;
        for(size_t i=0;i<output.length/2;++i) {finite+=std::isfinite(float(pixels[i]));nonzero+=pixels[i]!=0;}
        NSDictionary *receipt=@{@"schema":@"fsr412_original_metal_replay_v1",@"authority":@"controlled Metal replay of original DXIL translation, NOT Windows runtime",
            @"neural_compiler_profile":ns(neuralProfile),
            @"recurrent_format":ns(recurrentFormat=="u8"?"RGBA8Unorm replay fixture":"RGBA16Float replay fixture"),
            @"post_mode":ns(postMode),@"cache_copyback":@(postMode=="isolated"),
            @"device":replay.device.name,@"assets":ns(assets),@"output_size":@[@(width),@(height)],@"render_size":@[@(rw),@(rh)],
            @"permutation":@{@"tier":@2160,@"preset":@"normal",@"di":@1,@"lm":@1,@"jm":@0,@"cs":@0,@"ae":@0},
            @"tensor_sizes":initialConstants,@"logical_scratch_bytes":@(replay.logicalScratchBytes),@"physical_scratch_bytes":@(replay.scratch.length),
            @"guard_padding_only":@NO,@"resource_io_guards":@YES,@"initializers_sha256":ns(hash(replay.initializer.contents,replay.initializer.length)),
            @"frames":frameRecords,@"last_output_sha256":ns(hash(output.bytes,output.length)),@"finite_half_values":@(finite),@"nonzero_half_values":@(nonzero),
            @"limits":@[@"Sampler/texture formats are explicit replay fixture choices, not observed Windows allocation state.",@"Per-word SSBO bounds guards and integer texture IO guards enforce zero/discard; valid accesses and shader math remain unchanged.",@"Ordered/isolated POST are explicitly controlled dispatch/cache-IO adaptations; literal mode preserves the original alias.",@"Cross-workgroup scratch alias ordering and translation float parity require independent A/B tests.",@"Original PRE/POST, scratch addresses, fine motion, persistence alpha and cadence are unmodified shader instructions."]};
        save_json(out+"/receipt.json",receipt);std::cout<<"original Metal replay: "<<frames<<" frames, output "<<hash(output.bytes,output.length)<<", finite "<<finite<<", nonzero "<<nonzero<<", receipt "<<out<<"/receipt.json\n";
        check(finite==output.length/2 && nonzero>0,"invalid/empty original replay output");return 0;
    } catch(const std::exception &error) {std::cerr<<"original replay: "<<error.what()<<"\n";return 1;} }
}
