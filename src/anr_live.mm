// SPDX-License-Identifier: MIT
// Replacement for the exact MetalFX FG encode created by D3DMetal.
#include "anr_live.h"
#include "anr_runtime.h"
#include "fsr4_fg_settings.h"
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>
#include <exception>
#include <unistd.h>

namespace {
struct Convert {uint32_t width,height,rw,rh,encoding,reversed;float whiteNits;uint32_t uiMode;float uiNoiseFloor;};
static_assert(sizeof(Convert)==36,"ANR helper constants ABI");
struct FrameSlot {
    id<MTLBuffer> constants=nil,completion=nil;
    id<MTL4ArgumentTable> importArgs=nil,exportArgs=nil,retireArgs=nil;
    id<MTLResidencySet> residency=nil;NSArray *resources=nil;id<MTLFence> fence=nil;
    bool busy=false;uint64_t uiSequence=0;
    bool finished()const{return !busy||__atomic_load_n((uint32_t*)completion.contents,__ATOMIC_ACQUIRE)!=0;}
};
struct View {
    id<MTLDevice> device=nil;
    std::unique_ptr<anr::Runtime> runtime;
    id<MTLTexture> color=nil,depth=nil,motion=nil,generated=nil;
    id<MTLComputePipelineState> importPSO=nil,exportPSO=nil,retirePSO=nil;
    FrameSlot slots[4];unsigned activeSlot=0,nextSlot=0;
    uint32_t width=0,height=0,rw=0,rh=0;int encoding=0;
    bool history=false,swapPending=false,failed=false;
    std::chrono::steady_clock::time_point lastFrame,nextUIStats;
    uint64_t uiSequence=0,uiStatsRead=0;
    std::string directory;
    bool finished()const{for(const auto &slot:slots)if(!slot.finished())return false;return true;}
    bool acquire(){
        for(unsigned i=0;i<4;++i){unsigned index=(nextSlot+i)%4;if(slots[index].finished()){activeSlot=index;nextSlot=(index+1)%4;return true;}}
        return false;
    }
    bool gpu_failed()const{for(const auto &slot:slots)if(slot.busy&&__atomic_load_n((uint32_t*)slot.completion.contents,__ATOMIC_ACQUIRE)==2)return true;return false;}
};
using Key=std::pair<uintptr_t,uintptr_t>;
std::mutex mutex;
std::map<Key,std::shared_ptr<View>> views;
std::vector<std::shared_ptr<View>> retired;
struct Failure {std::string directory,reason;std::chrono::steady_clock::time_point retry;};
std::map<Key,Failure> failures;
Fsr4AnrStatus status={};
std::chrono::steady_clock::time_point lastPublish;
Key key(void *device,void *owner){return {(uintptr_t)device,(uintptr_t)owner};}
void publish(){
    // Synthetic GPU checks must not overwrite the running game's receipt.
    if(const char *disabled=getenv("FSR4_ANR_STATUS_DISABLED"))if(!strcmp(disabled,"1"))return;
    const char *root=getenv("FSR4_ROOT");if(!root||!*root)return;
    const auto now=std::chrono::steady_clock::now();
    if(lastPublish.time_since_epoch().count()&&now-lastPublish<std::chrono::milliseconds(500))return;
    lastPublish=now;const std::string path=std::string(root)+"/dist/crossover/fg-status.conf",temporary=path+".tmp";
    FILE *f=fopen(temporary.c_str(),"w");if(!f)return;
    const char *uiMode=status.uiMode==FSR4_ANR_UI_PRECOMPOSITED?"precomposited-adaptive":
        status.uiMode==FSR4_ANR_UI_OVERLAY?"premultiplied-overlay":"none";
    fprintf(f,"t=%ld\npid=%d\nactive=%d\nreason=%s\nwidth=%u\nheight=%u\nencodes=%llu\ngenerated=%llu\nprimed=%llu\nfallbacks=%llu\nui_mode=%s\nui_composited=%llu\nui_overlay=%llu\ncolor_format=%u\nui_format=%u\n",
        (long)time(nullptr),(int)getpid(),status.active,status.reason,status.width,status.height,
        (unsigned long long)status.encodes,(unsigned long long)status.generated,(unsigned long long)status.primed,(unsigned long long)status.fallbacks,
        uiMode,(unsigned long long)status.uiComposited,(unsigned long long)status.uiOverlay,status.colorFormat,status.uiFormat);
    fprintf(f,"ui_policy=adaptive-residual-v2\nui_samples=%u\nui_raw_changed=%u\nui_soft=%u\nui_opaque=%u\nui_opacity_mean=%.6f\nmotion_scale_x=%.6f\nmotion_scale_y=%.6f\nanr_motion_scale_x=%.6f\nanr_motion_scale_y=%.6f\n",
        status.uiSamples,status.uiChanged,status.uiSoft,status.uiOpaque,
        status.uiSamples?double(status.uiOpacitySum)/(4096.0*status.uiSamples):0.0,
        status.motionScaleX,status.motionScaleY,status.anrMotionScaleX,status.anrMotionScaleY);
    fprintf(f,"selected_mode=%s\nnative_passthrough=%d\nanr_configured=%d\n",
        status.selectedAnr?"anr":"metalfx",status.selectedAnr?0:1,
        getenv("FSR4_ANR_RUNTIME")&&*getenv("FSR4_ANR_RUNTIME")?1:0);
    fclose(f);rename(temporary.c_str(),path.c_str());
}
int reject(const char *why){
    const bool changed=strncmp(status.reason,why,sizeof status.reason)!=0||status.active;
    if(changed){
        fprintf(stderr,"[anr-fg] %s\n",why);
        const char *path=getenv("FSR4_HOOK_LOG");if(path&&*path)if(FILE *file=fopen(path,"a")){
            fprintf(file,"[anr-fg] %s\n",why);fclose(file);}
    }
    status.active=0;snprintf(status.reason,sizeof status.reason,"%s",why);++status.fallbacks;publish();return 0;
}
std::string runtime_path(uint32_t width,uint32_t height){
    const char *base=getenv("FSR4_ANR_RUNTIME");if(!base||!*base)return "";
    NSString *root=[NSString stringWithUTF8String:base];
    NSString *profile=[root stringByAppendingPathComponent:[NSString stringWithFormat:@"%ux%u",width,height]];
    if([NSFileManager.defaultManager fileExistsAtPath:[profile stringByAppendingPathComponent:@"manifest.json"]])return profile.UTF8String;
    if([NSFileManager.defaultManager fileExistsAtPath:[root stringByAppendingPathComponent:@"manifest.json"]])return base;
    return "";
}
// BGRA/RGBA twins have identical read-space/precision. Do not infer a transfer
// function for a UI texture in a different format from the clean scene.
bool ui_format_matches(MTLPixelFormat a,MTLPixelFormat b){
    return a==b || ((a==MTLPixelFormatBGRA8Unorm||a==MTLPixelFormatRGBA8Unorm)
        &&(b==MTLPixelFormatBGRA8Unorm||b==MTLPixelFormatRGBA8Unorm))
        || ((a==MTLPixelFormatBGRA8Unorm_sRGB||a==MTLPixelFormatRGBA8Unorm_sRGB)
        &&(b==MTLPixelFormatBGRA8Unorm_sRGB||b==MTLPixelFormatRGBA8Unorm_sRGB));
}
id<MTLTexture> tex(id<MTLDevice> device,MTLPixelFormat format,uint32_t width,uint32_t height){
    auto descriptor=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format width:width height:height mipmapped:NO];
    descriptor.usage=MTLTextureUsageShaderRead|MTLTextureUsageShaderWrite;descriptor.storageMode=MTLStorageModePrivate;
    return [device newTextureWithDescriptor:descriptor];
}
std::shared_ptr<View> create(const Fsr4AnrFrame &f,const std::string &directory,std::string &error){
    auto v=std::make_shared<View>();v->device=(__bridge id<MTLDevice>)f.device;v->directory=directory;
    v->width=f.uw;v->height=f.uh;v->rw=f.rw;v->rh=f.rh;v->encoding=f.colorEncoding;
    v->runtime=std::make_unique<anr::Runtime>(v->device,directory);
    if(!v->runtime->initialize(error,true))return {};
    if(v->runtime->width()!=f.uw||v->runtime->height()!=f.uh){error="ANR runtime resolution does not match game FG";return {};}
    // GPU timestamp A/B passed the quality gate and favored coalescing.
    // Keep the separate-encoder route for regression and hardware comparison.
    const char *coalesced=getenv("FSR4_ANR_COALESCED");v->runtime->set_coalesced(!coalesced||*coalesced!='0');
    const char *root=getenv("FSR4_ROOT");if(!root||!*root){error="FSR4_ROOT required for ANR helper shaders";return {};}
    NSString *libraryPath=[[NSString stringWithUTF8String:root] stringByAppendingPathComponent:@"build/anr_helpers_ui_v2.metallib"];
    NSError *metalError=nil;id<MTLLibrary> library=[v->device newLibraryWithURL:[NSURL fileURLWithPath:libraryPath] error:&metalError];
    auto pipeline=[&](NSString *name){return [v->device newComputePipelineStateWithFunction:[library newFunctionWithName:name] error:&metalError];};
    if(!library){error="ANR helpers: "+std::string(metalError.localizedDescription.UTF8String);return {};}
    v->importPSO=pipeline(@"anr_import");v->exportPSO=pipeline(@"anr_export");v->retirePSO=pipeline(@"anr_retire");
    v->color=tex(v->device,MTLPixelFormatRGB10A2Unorm,f.uw,f.uh);v->depth=tex(v->device,MTLPixelFormatR32Float,f.uw,f.uh);
    v->motion=tex(v->device,MTLPixelFormatRG16Float,f.uw,f.uh);v->generated=tex(v->device,MTLPixelFormatRGB10A2Unorm,f.uw,f.uh);
    if(!v->importPSO||!v->exportPSO||!v->retirePSO||!v->color||!v->depth||!v->motion||!v->generated){error="ANR view allocation failed";return {};}
    for(auto &slot:v->slots){
        slot.constants=[v->device newBufferWithLength:256 options:MTLResourceStorageModeShared];
        slot.completion=[v->device newBufferWithLength:256 options:MTLResourceStorageModeShared];
        if(!slot.constants||!slot.completion){error="ANR frame slot allocation failed";return {};}
        *static_cast<uint32_t*>(slot.completion.contents)=1;
        MTLResidencySetDescriptor *rd=[MTLResidencySetDescriptor new];rd.initialCapacity=16;
        slot.residency=[v->device newResidencySetWithDescriptor:rd error:&metalError];
        MTL4ArgumentTableDescriptor *td=[MTL4ArgumentTableDescriptor new];td.maxBufferBindCount=2;td.maxTextureBindCount=6;
        slot.importArgs=[v->device newArgumentTableWithDescriptor:td error:&metalError];
        slot.exportArgs=[v->device newArgumentTableWithDescriptor:td error:&metalError];
        slot.retireArgs=[v->device newArgumentTableWithDescriptor:td error:&metalError];
        if(!slot.residency||!slot.importArgs||!slot.exportArgs||!slot.retireArgs){error="ANR argument/residency allocation failed";return {};}
    }
    return v;
}
void collect(){for(auto it=retired.begin();it!=retired.end();)if((*it)->finished())it=retired.erase(it);else ++it;}
void read_ui_stats(View &v){
    for(const auto &slot:v.slots)if(slot.uiSequence>v.uiStatsRead&&slot.finished()){
        uint32_t values[5];memcpy(values,(const uint8_t*)slot.completion.contents+16,sizeof values);
        status.uiSamples=values[0];status.uiChanged=values[1];status.uiSoft=values[2];
        status.uiOpaque=values[3];status.uiOpacitySum=values[4];v.uiStatsRead=slot.uiSequence;
    }
}
float ui_noise_floor(MTLPixelFormat format){
    if(format==MTLPixelFormatRGBA8Unorm||format==MTLPixelFormatBGRA8Unorm
       ||format==MTLPixelFormatRGBA8Unorm_sRGB||format==MTLPixelFormatBGRA8Unorm_sRGB)return 2.5f/255.f;
    if(format==MTLPixelFormatRGB10A2Unorm)return 2.5f/1023.f;
    return 1.e-4f;
}
bool helper(View &v,id command,bool metal4,int which,id<MTLTexture> color,id<MTLTexture> depth,id<MTLTexture> motion,
            id<MTLTexture> output,id<MTLFence> fence,id<MTLTexture> ui=nil){
    auto &slot=v.slots[v.activeSlot];
    auto pso=which==0?v.importPSO:which==1?v.exportPSO:v.retirePSO;
    const MTLSize threads=which==2?MTLSizeMake(1,1,1):MTLSizeMake(v.width,v.height,1),group=which==2?MTLSizeMake(1,1,1):MTLSizeMake(16,16,1);
    if(metal4){
        auto enc=[(id<MTL4CommandBuffer>)command computeCommandEncoder];if(!enc)return false;
        [enc barrierAfterQueueStages:MTLStageAll beforeStages:MTLStageDispatch visibilityOptions:MTL4VisibilityOptionDevice];
        if(which==0&&fence)[enc waitForFence:fence beforeEncoderStages:MTLStageDispatch];
        auto args=which==0?slot.importArgs:which==1?slot.exportArgs:slot.retireArgs;
        [args setAddress:(which==2?slot.completion:slot.constants).gpuAddress atIndex:0];
        if(which==1)[args setAddress:slot.completion.gpuAddress+16 atIndex:1];
        if(which==0){id<MTLTexture> textures[]={color,depth,motion,v.color,v.depth,v.motion};for(unsigned i=0;i<6;++i)[args setTexture:textures[i].gpuResourceID atIndex:i];}
        if(which==1){id<MTLTexture> textures[]={v.generated,output,color,ui?:color};for(unsigned i=0;i<4;++i)[args setTexture:textures[i].gpuResourceID atIndex:i];}
        [enc setComputePipelineState:pso];[enc setArgumentTable:args];[enc dispatchThreads:threads threadsPerThreadgroup:group];
        if(which==1&&fence)[enc updateFence:fence afterEncoderStages:MTLStageDispatch];
        [enc endEncoding];
    }else{
        auto enc=[(id<MTLCommandBuffer>)command computeCommandEncoder];if(!enc)return false;
        if(which==0&&fence)[enc waitForFence:fence];
        [enc setComputePipelineState:pso];[enc setBuffer:which==2?slot.completion:slot.constants offset:0 atIndex:0];
        if(which==1)[enc setBuffer:slot.completion offset:16 atIndex:1];
        if(which==0){id<MTLTexture> textures[]={color,depth,motion,v.color,v.depth,v.motion};for(unsigned i=0;i<6;++i)[enc setTexture:textures[i] atIndex:i];}
        if(which==1){id<MTLTexture> textures[]={v.generated,output,color,ui?:color};for(unsigned i=0;i<4;++i)[enc setTexture:textures[i] atIndex:i];}
        [enc dispatchThreads:threads threadsPerThreadgroup:group];if(which==1&&fence)[enc updateFence:fence];[enc endEncoding];
    }
    return true;
}
}
int fsr4_anr_encode_frame(const Fsr4AnrFrame *frame){@autoreleasepool{
    std::lock_guard<std::mutex> lock(mutex);++status.encodes;collect();
    status.selectedAnr=fsr4_anr_is_selected();
    if(!status.selectedAnr)return reject("MetalFX FG selected");
    const char *reason=fsr4_anr_frame_reason(frame);if(reason)return reject(reason);const auto &f=*frame;
    status.width=f.uw;status.height=f.uh;status.uiMode=fsr4_anr_ui_mode(&f);
    status.motionScaleX=f.motionScaleX;status.motionScaleY=f.motionScaleY;
    status.anrMotionScaleX=f.motionScaleX*float(f.uw)/f.rw;status.anrMotionScaleY=f.motionScaleY*float(f.uh)/f.rh;
    id command=(__bridge id)f.commandBuffer;id<MTLDevice> device=(__bridge id<MTLDevice>)f.device;
    const bool metal4=[command respondsToSelector:@selector(useResidencySet:)]&&![command respondsToSelector:@selector(addCompletedHandler:)];
    if(![command respondsToSelector:@selector(device)]||[command device]!=device)return reject("ANR: command buffer device mismatch");
    if(!metal4&&![command respondsToSelector:@selector(addCompletedHandler:)])return reject("ANR: unsupported FG command buffer API");
    id<MTLTexture> color=(__bridge id<MTLTexture>)f.color,depth=(__bridge id<MTLTexture>)f.depth,motion=(__bridge id<MTLTexture>)f.motion,output=(__bridge id<MTLTexture>)f.output;
    id<MTLTexture> ui=(__bridge id<MTLTexture>)f.ui;
    status.colorFormat=(uint32_t)color.pixelFormat;status.uiFormat=(uint32_t)ui.pixelFormat;
    for(auto texture:{color,depth,motion,output})if(texture.device!=device||texture.textureType!=MTLTextureType2D)return reject("ANR: incompatible FG texture/device");
    if(ui){
        if(ui.device!=device||ui.textureType!=MTLTextureType2D||ui.width!=f.uw||ui.height!=f.uh)
            return reject("ANR: UI texture extent/device mismatch; keeping MetalFX");
        if(!(ui.usage&MTLTextureUsageShaderRead))return reject("ANR: UI texture must be shader-readable; keeping MetalFX");
        if(!ui_format_matches(ui.pixelFormat,color.pixelFormat))return reject("ANR: UI and scene color formats differ; keeping MetalFX");
        if(ui==output)return reject("ANR: UI aliases FG output; keeping MetalFX");
        if(color==output)return reject("ANR: scene aliases FG output during UI recomposition; keeping MetalFX");
        if(!f.uiComposited&&(ui.pixelFormat==MTLPixelFormatRGB9E5Float||ui.pixelFormat==MTLPixelFormatRG11B10Float))
            return reject("ANR: separate UI overlay needs an alpha channel; keeping MetalFX");
    }
    if(color.width!=f.uw||color.height!=f.uh||output.width!=f.uw||output.height!=f.uh||depth.width<f.rw||depth.height<f.rh||motion.width<f.rw||motion.height<f.rh)
        return reject("ANR: unsupported FG texture extent");
    if(depth.pixelFormat!=MTLPixelFormatR32Float&&depth.pixelFormat!=MTLPixelFormatDepth32Float&&depth.pixelFormat!=MTLPixelFormatDepth32Float_Stencil8)
        return reject("ANR: depth format unsupported");
    if(motion.pixelFormat!=MTLPixelFormatRG16Float&&motion.pixelFormat!=MTLPixelFormatRG32Float)return reject("ANR: motion format unsupported");
    if(!(output.usage&MTLTextureUsageShaderWrite)||!(color.usage&MTLTextureUsageShaderRead)
       ||!(depth.usage&MTLTextureUsageShaderRead)||!(motion.usage&MTLTextureUsageShaderRead))
        return reject("ANR: FG texture shader usage flags unavailable");
    const std::string directory=runtime_path(f.uw,f.uh);
    if(directory.empty()){
        if(!getenv("FSR4_ANR_RUNTIME")||!*getenv("FSR4_ANR_RUNTIME"))return reject("ANR: FSR4_ANR_RUNTIME not configured");
        char why[160];snprintf(why,sizeof why,"ANR: no prepared graph for %ux%u; use 1920x1080, 2560x1440 or 3840x2160",f.uw,f.uh);return reject(why);
    }
    if(retired.size()>=8)return reject("ANR: retired GPU resources still pending");
    const Key ownerKey=key(f.device,f.owner);
    auto cached=failures.find(ownerKey);
    if(cached!=failures.end()&&cached->second.directory==directory&&std::chrono::steady_clock::now()<cached->second.retry)
        return reject(cached->second.reason.c_str());
    failures.erase(ownerKey);
    auto &view=views[ownerKey];
    if(view&&view->gpu_failed())return reject("ANR: GPU failure; restart game");
    if(view&&(view->directory!=directory||view->width!=f.uw||view->height!=f.uh||view->encoding!=f.colorEncoding||view->rw!=f.rw||view->rh!=f.rh||view->failed)){
        if(!view->finished())retired.push_back(view);view.reset();
    }
    std::string error;
    if(!view){
        try {
            @try {view=create(f,directory,error);}
            @catch(NSException *exception){error=exception.reason?exception.reason.UTF8String:"ANR runtime initialization exception";}
        } catch(const std::exception &exception){error=exception.what();}
    }
    if(!view){failures[ownerKey]={directory,error,std::chrono::steady_clock::now()+std::chrono::seconds(2)};return reject(error.c_str());}
    View &v=*view;read_ui_stats(v);
    if(!v.acquire()){v.history=false;return reject("ANR: all four frame slots busy; keeping MetalFX");}
    auto &slot=v.slots[v.activeSlot];v.runtime->set_frame_slot(v.activeSlot);
    const auto now=std::chrono::steady_clock::now();
    const bool temporalGap=v.lastFrame.time_since_epoch().count()&&now-v.lastFrame>std::chrono::milliseconds(500);
    const bool priming=!v.history||f.reset||temporalGap;v.lastFrame=now;
    if(v.swapPending){v.runtime->swap_history();v.swapPending=false;}
    const bool uiStats=!priming&&status.uiMode==FSR4_ANR_UI_PRECOMPOSITED&&now>=v.nextUIStats;
    if(uiStats)v.nextUIStats=now+std::chrono::milliseconds(500);
    slot.uiSequence=uiStats?++v.uiSequence:0;
    memset((uint8_t*)slot.completion.contents+16,0,5*sizeof(uint32_t));
    Convert constants={f.uw,f.uh,f.rw,f.rh,(uint32_t)f.colorEncoding,(uint32_t)f.depthReversed,203.f,
        (uint32_t)status.uiMode|(uiStats?256u:0u),ui_noise_floor(color.pixelFormat)};
    const char *white=getenv("FSR4_ANR_WHITE_NITS");if(white){float value=strtof(white,nullptr);if(std::isfinite(value)&&value>0)constants.whiteNits=value;}
    memcpy(slot.constants.contents,&constants,sizeof(constants));
    slot.resources=@[color,depth,motion,output,v.color,v.depth,v.motion,v.generated,slot.constants,slot.completion];
    if(ui)slot.resources=[slot.resources arrayByAddingObject:ui];
    [slot.residency removeAllAllocations];for(id value in slot.resources)[slot.residency addAllocation:value];[slot.residency commit];
    if(metal4)[(id<MTL4CommandBuffer>)command useResidencySet:slot.residency];
    id<MTLFence> fence=(__bridge id<MTLFence>)f.fence;slot.fence=fence;
    if(fence&&fence.device!=device)return reject("ANR: fence belongs to another device");
    __atomic_store_n((uint32_t*)slot.completion.contents,0u,__ATOMIC_RELEASE);slot.busy=true;
    uint8_t hardFrame[204];fsr4_anr_frame_constants(&f,hardFrame,!priming);
    // UI must be recomposited after neural work. Never write the interpolated
    // frame directly to the output and then read/write it in one UI dispatch.
    const bool direct=!ui&&f.colorEncoding==2&&color.pixelFormat==MTLPixelFormatRGB10A2Unorm
        &&output.pixelFormat==MTLPixelFormatRGB10A2Unorm&&depth.pixelFormat==MTLPixelFormatR32Float
        &&motion.pixelFormat==MTLPixelFormatRG16Float&&f.rw==f.uw&&f.rh==f.uh;
    bool ok=!priming||v.runtime->reset_history(command,metal4,error);
    if(ok&&!direct)ok=helper(v,command,metal4,0,color,depth,motion,output,fence);
    else if(ok&&fence){
        if(metal4){auto enc=[(id<MTL4CommandBuffer>)command computeCommandEncoder];ok=enc!=nil;
            if(enc){[enc waitForFence:fence beforeEncoderStages:MTLStageDispatch];[enc endEncoding];}}
        else{auto enc=[(id<MTLCommandBuffer>)command computeCommandEncoder];ok=enc!=nil;
            if(enc){[enc waitForFence:fence];[enc endEncoding];}}
    }
    auto pairColor=direct?color:v.color,pairDepth=direct?depth:v.depth,pairMotion=direct?motion:v.motion;
    auto target=direct?output:v.generated;
    if(ok)ok=metal4?v.runtime->encode_pair(command,pairColor,pairDepth,pairMotion,hardFrame,sizeof hardFrame,error)
                   :v.runtime->encode_pair_classic(command,pairColor,pairDepth,pairMotion,hardFrame,sizeof hardFrame,error);
    if(ok&&!priming)ok=metal4?v.runtime->encode_interpolation(command,.5,target,error)
                           :v.runtime->encode_interpolation_classic(command,.5,target,error);
    if(ok&&!priming&&!direct)ok=helper(v,command,metal4,1,color,depth,motion,output,fence,ui);
    if(ok&&direct&&fence){
        if(metal4){auto enc=[(id<MTL4CommandBuffer>)command computeCommandEncoder];ok=enc!=nil;
            if(enc){[enc barrierAfterQueueStages:MTLStageAll beforeStages:MTLStageDispatch visibilityOptions:MTL4VisibilityOptionDevice];
                [enc updateFence:fence afterEncoderStages:MTLStageDispatch];[enc endEncoding];}}
        else{auto enc=[(id<MTLCommandBuffer>)command computeCommandEncoder];ok=enc!=nil;
            if(enc){[enc updateFence:fence];[enc endEncoding];}}
    }
    if(metal4){if(!helper(v,command,true,2,nil,nil,nil,nil,nil)){v.failed=true;return reject("ANR: completion encoder unavailable; resources retained");}}
    else{
        const auto retained=view;const unsigned index=v.activeSlot;
        [(id<MTLCommandBuffer>)command addCompletedHandler:^(id<MTLCommandBuffer> done){
            __atomic_store_n((uint32_t*)retained->slots[index].completion.contents,done.status==MTLCommandBufferStatusCompleted?1u:2u,__ATOMIC_RELEASE);
        }];
    }
    if(!ok){v.history=false;return reject(error.empty()?"ANR: FG encode failed; keeping MetalFX":error.c_str());}
    v.history=true;v.swapPending=true;status.width=f.uw;status.height=f.uh;
    if(priming){++status.primed;status.active=0;snprintf(status.reason,sizeof status.reason,"ANR: history primed; MetalFX presents first/reset frame");publish();return 2;}
    status.active=1;status.reason[0]=0;++status.generated;
    if(status.uiMode==FSR4_ANR_UI_PRECOMPOSITED)++status.uiComposited;
    else if(status.uiMode==FSR4_ANR_UI_OVERLAY)++status.uiOverlay;
    publish();return 1;
}}
void fsr4_anr_invalidate_owner(void *device,void *owner){std::lock_guard<std::mutex> lock(mutex);auto it=views.find(key(device,owner));if(it!=views.end()&&it->second)it->second->history=false;}
void fsr4_anr_release_owner(void *device,void *owner){std::lock_guard<std::mutex> lock(mutex);auto it=views.find(key(device,owner));if(it!=views.end()){if(it->second&&!it->second->finished())retired.push_back(it->second);views.erase(it);}failures.erase(key(device,owner));collect();}
extern "C" void fsr4_anr_mode_changed(int anr){
    std::lock_guard<std::mutex> lock(mutex);
    status.selectedAnr=anr;status.active=0;
    for(auto &entry:views)if(entry.second&&!entry.second->finished())retired.push_back(entry.second);
    views.clear();failures.clear();collect();
    snprintf(status.reason,sizeof status.reason,"%s",anr?"ANR selected; waiting for an accepted frame":"MetalFX selected; original native encode");
    publish();
}
void fsr4_anr_publish_passthrough(void){
    std::lock_guard<std::mutex> lock(mutex);collect();
    status.selectedAnr=0;status.active=0;
    snprintf(status.reason,sizeof status.reason,"MetalFX selected; original native encode");publish();
}
void fsr4_anr_set_inactive(const char *why){std::lock_guard<std::mutex> lock(mutex);reject(why?why:"ANR inactive");}
void fsr4_anr_get_status(Fsr4AnrStatus *out){if(!out)return;std::lock_guard<std::mutex> lock(mutex);*out=status;}
