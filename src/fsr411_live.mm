// SPDX-License-Identifier: MIT
// The hook receives MetalFX inputs AFTER D3DMetal translated DLSS/NGX.
// All coordinate/unit conversion is centralized in fsr4_input.h.
#include "fsr4_live.h"
#include "fsr4_log.h"
#include "fsr411_runtime.h"
#include "fsr411_capture_io.h"
#include <chrono>
#include <mutex>

namespace {
struct OwnerTrace {
    uint64_t calls=0;
    double lastLog=0;
    fsr411::Profile profile=fsr411::Profile::Lossy;
};
struct DeviceContext {
    fsr411::Runtime runtime;
    fsr411_capture::Recorder recorder;
    std::map<void *,std::unique_ptr<fsr411::View>> owners;
    std::vector<std::unique_ptr<fsr411::View>> retired;
    std::map<void *,OwnerTrace> traces;
    id<MTLTexture> lastOutput=nil;
    Fsr4InputRect lastOutputRect={};
    void *lastOwner=nullptr;
    double lastTime=0;
    double lastOutputTime=0;
    DeviceContext(id<MTLDevice> dev,const std::string &root,const std::string &build)
        :runtime(dev,root,build),recorder(root){}
    void collect() {
        retired.erase(std::remove_if(retired.begin(),retired.end(),[](const auto &view){
            return std::all_of(view->slots.begin(),view->slots.end(),[](const auto &slot){return slot->retired();});
        }),retired.end());
    }
};
std::mutex contextMutex;
std::map<void *,std::unique_ptr<DeviceContext>> contexts;

const char *env(const char *key) { const char *value=std::getenv(key); return value&&*value?value:nullptr; }
bool parse_flag(const char *key,int &value,std::string &error) {
    if(const char *v=env(key)) {
        if(std::strcmp(v,"0") && std::strcmp(v,"1")) {error=std::string(key)+" must be 0 or 1";return false;}
        value=v[0]=='1';
    }
    return true;
}
double now_ms() {
    using namespace std::chrono;
    return duration<double,std::milli>(steady_clock::now().time_since_epoch()).count();
}
const char *preset(const Fsr4LiveFrameV2 &f) {
    if(auto value=env("FSR4_PRESET")) return value;
    if(f.inputContentProperties==1) return "drs";
    const float ratio=f.frame.rw?float(f.frame.uw)/f.frame.rw:1;
    if(ratio>=2.99f) return "ultraperf";
    if(ratio>=1.99f) return "performance";
    if(ratio>=1.69f) return "balanced";
    if(ratio>=1.50f) return "quality";
    return "native";
}
bool options_for(const Fsr4LiveFrameV2 &f,fsr411::Frame &frame,fsr411::Options &o,std::string &error) {
    int profile=1;fsr4_live_get_profile(&profile);
    o.profile=fsr411::profile_from_int(profile);
    if(auto model=env("FSR4_MODEL")) {
        if(!std::strcmp(model,"auto")) o.backend=fsr411::Backend::Auto;
        else if(!std::strcmp(model,"reference") || !std::strcmp(model,"int8")) o.backend=fsr411::Backend::Reference;
        else if(!std::strcmp(model,"native") || !std::strcmp(model,"i8mpp")) o.backend=fsr411::Backend::Native;
        else if(!std::strcmp(model,"scalar")) o.backend=fsr411::Backend::Scalar;
        else {error="FSR 4.1.1 model must be auto, reference/int8, scalar or native/i8mpp";return false;}
    }
    const std::string p=preset(f);
    o.ultraPerformance=p=="ultraperf" || p=="ultraperformance" || p=="ultra_performance";
    if(auto cs=env("FSR4_COLORSPACE")) {
        if(!std::strcmp(cs,"linear") || !std::strcmp(cs,"0")) o.colorspace=0;
        else if(!std::strcmp(cs,"nonlinear") || !std::strcmp(cs,"non_linear") || !std::strcmp(cs,"1")) o.colorspace=1;
        else if(!std::strcmp(cs,"srgb") || !std::strcmp(cs,"2")) o.colorspace=2;
        else if(!std::strcmp(cs,"pq") || !std::strcmp(cs,"3")) o.colorspace=3;
        else {error="invalid FSR4_COLORSPACE";return false;}
    }
    int lowRes=frame.input.outputResMV!=1;
    if(!parse_flag("FSR4_LOW_RES_MV",lowRes,error))return false;
    if(env("FSR4_LOW_RES_MV"))frame.input.outputResMV=!lowRes;
    if(!parse_flag("FSR4_JITTERED_MV",frame.input.jitteredMV,error)
        || !parse_flag("FSR4_AUTOEXPOSURE",frame.input.autoExposure,error)
        || !parse_flag("FSR4_OPTIMIZE",o.optimize,error)) return false;
    int compile=1;
    if(!parse_flag("FSR4_COMPILE",compile,error))return false;
    o.allowCompile=compile;
    int rcas=0; fsr4_live_get_sharpness(&o.sharpness,&rcas); o.rcas=rcas;
    return true;
}
int failure(Fsr4LiveStatus &status,const std::string &reason) {
    status.active=0; ++status.encodes;
    std::snprintf(status.reason,sizeof(status.reason),"%s",reason.c_str());
    fsr4_live_set_status(&status);
    fsr4_logf("4.1.1 encode failed: %s",reason.c_str());
    return 0;
}

void trace_input(const Fsr4LiveFrameV2 &v2, const fsr411::Options &options,
                 uint64_t call, bool stub) {
    const auto &f=v2.frame;
    const auto &g=v2.geometry;
    fsr4_logf("4.1.1 input call=%llu owner=%p cb=%p profile=%s%s "
              "content=%ux%u->%ux%u max=%ux%u "
              "color=%p:%ux%u+%u,%u depth=%p:%ux%u+%u,%u "
              "motion=%p:%ux%u+%u,%u output=%p:%ux%u+%u,%u "
              "jitter=%.9g,%.9g mvScale=%.9g,%.9g preExposure=%.9g "
              "reset=%d depthReversed=%d jitteredMV=%d outputResMV=%d autoExposure=%d drs=%d",
              (unsigned long long)call,f.owner,f.commandBuffer,
              fsr411::profile_name(options.profile),stub?" STUB":"",
              f.rw,f.rh,f.uw,f.uh,g.maxInputWidth,g.maxInputHeight,
              f.color,g.color.width,g.color.height,g.color.offsetX,g.color.offsetY,
              f.depth,g.depth.width,g.depth.height,g.depth.offsetX,g.depth.offsetY,
              f.motion,g.motion.width,g.motion.height,g.motion.offsetX,g.motion.offsetY,
              f.output,g.output.width,g.output.height,g.output.offsetX,g.output.offsetY,
              f.jitterX,f.jitterY,f.mvScaleX,f.mvScaleY,f.preExposure,
              f.reset,f.depthReversed,f.jitteredMV,f.outputResMV,f.autoExposure,v2.inputContentProperties);
}
}

extern "C" int fsr411_encode_live_frame(const Fsr4LiveFrameV2 *input) {
    @autoreleasepool {
        std::lock_guard<std::mutex> lock(contextMutex);
        Fsr4LiveStatus status={}; fsr4_live_get_status(&status);
        if(!fsr4_live_frame_v2_valid(input))return failure(status,"invalid live frame v2 ABI");
        const auto &f=input->frame;
        if(!f.device || !f.commandBuffer || !f.color || !f.depth || !f.motion || !f.output)
            return failure(status,"missing FSR 4.1.1 Metal input");
        const char *root=env("FSR4_ROOT");
        if(!root)return failure(status,"FSR4_ROOT is required for FSR 4.1.1");
        id<MTLDevice> dev=(__bridge id<MTLDevice>)f.device;
        auto &context=contexts[f.device];
        if(!context) {
            const std::string build=env("FSR4_BUILD_DIR")?env("FSR4_BUILD_DIR"):std::string(root)+"/build";
            context=std::make_unique<DeviceContext>(dev,root,build);
        }
        context->collect();
        auto &owner=context->owners[f.owner]; if(!owner)owner=std::make_unique<fsr411::View>();
        fsr411::Frame frame;
        frame.color=(__bridge id<MTLTexture>)f.color; frame.depth=(__bridge id<MTLTexture>)f.depth;
        frame.motion=(__bridge id<MTLTexture>)f.motion; frame.output=(__bridge id<MTLTexture>)f.output;
        frame.exposure=(__bridge id<MTLTexture>)f.exposure;
        frame.input=fsr4_live_input_desc(&f,&input->geometry,input->inputContentProperties);
        fsr411::Options options; std::string error;
        auto failed=[&](const std::string &why){
            owner->invalidate();
            context->recorder.failed(f.owner,why);
            if(context->lastOwner==f.owner) {context->lastOutput=nil;context->lastOwner=nullptr;}
            return failure(status,why);
        };
        if(!options_for(*input,frame,options,error))return failed(error);
        auto &trace=context->traces[f.owner];
        const double start=now_ms();
        ++trace.calls;
        const bool log=trace.calls<=16 || start-trace.lastLog>=5000 || trace.profile!=options.profile;
        if(log) {trace.lastLog=start;trace.profile=options.profile;}
        const uint32_t mvW=frame.input.outputResMV==1?f.uw:f.rw, mvH=frame.input.outputResMV==1?f.uh:f.rh;
        const bool stub=frame.motion.width==4 && frame.motion.height==4 && frame.motion.pixelFormat==MTLPixelFormatRGBA8Unorm
            && (mvW>4 || mvH>4);
        if(log)trace_input(*input,options,trace.calls,stub);
        if(stub) {
            owner->invalidate();
            if(auto policy=env("FSR4_STUB")) if(std::strcmp(policy,"skip"))
                return failed("FSR 4.1.1 cannot run a network on placeholder motion");
            Fsr4InputRect output{input->geometry.output.offsetX,input->geometry.output.offsetY,f.uw,f.uh};
            if(!context->runtime.copy_output((__bridge id)f.commandBuffer,*owner,context->lastOutput,context->lastOutputRect,
                                              frame.output,output,error)) return failed(error);
            if(log)fsr4_logf("4.1.1 placeholder owner=%p sourceOwner=%p source=%p destination=%p ageMs=%.3f; history not advanced",
                            f.owner,context->lastOwner,(__bridge void *)context->lastOutput,f.output,
                            context->lastOutputTime?start-context->lastOutputTime:0);
            ++status.encodes; ++status.stubs; status.active=1;
            std::snprintf(status.reason,sizeof(status.reason),"placeholder: copied previous FSR 4.1.1 output");
            fsr4_live_set_status(&status); return 1;
        }
        options.snapshot=context->recorder.request(f.owner,frame);
        if(!context->runtime.encode((__bridge id)f.commandBuffer,*owner,frame,options,error))return failed(error);
        context->recorder.record(f.owner,*owner,options,options.snapshot);
        context->lastOutput=frame.output;
        context->lastOutputRect={input->geometry.output.offsetX,input->geometry.output.offsetY,f.uw,f.uh};
        context->lastOwner=f.owner;
        const double time=now_ms();
        context->lastOutputTime=time;
        if(context->lastTime) {float delta=float(time-context->lastTime);status.frameMs=status.frameMs?0.9f*status.frameMs+0.1f*delta:delta;}
        context->lastTime=time;
        status.active=1; status.reason[0]=0; ++status.encodes; ++status.replaced;
        status.rw=f.rw;status.rh=f.rh;status.uw=f.uw;status.uh=f.uh;
        status.uwA=owner->prepared.alignedUW;status.uhA=owner->prepared.alignedUH;status.sharpness=options.rcas?options.sharpness:0;
        std::snprintf(status.model,sizeof(status.model),"%s",owner->backendLabel=="int8-mpp"?"i8mpp":owner->backendLabel=="int8-scalar"?"scalar":"i8-ref");
        std::snprintf(status.preset,sizeof(status.preset),"%s",preset(*input));
        std::snprintf(status.variant,sizeof(status.variant),"%s",owner->variantKey.c_str());
        fsr4_live_set_status(&status);
        if(log || time-start>=100) {
            const auto &p=owner->prepared;
            fsr4_logf("4.1.1 encoded owner=%p call=%llu backend=%s variant=%s "
                      "mvExtent=%ux%u normalizedMvScale=%.9g,%.9g jitterCancellation=%.9g,%.9g "
                      "effectiveReset=%u resetReasons=0x%x copyMask=0x%x "
                      "history=%p recurrent=%p inference=%s neural=%llu reused=%llu cpuMs=%.3f",
                      f.owner,(unsigned long long)trace.calls,owner->backendLabel.c_str(),owner->variantKey.c_str(),
                      p.mvW,p.mvH,p.mvScale[0],p.mvScale[1],p.jitterCancellation[0],p.jitterCancellation[1],
                      p.reset,p.resetReasons,p.copyMask,(__bridge void *)owner->history,(__bridge void *)owner->recurrent,
                      owner->skippedInference?"reuse":"full",(unsigned long long)owner->neuralRuns,
                      (unsigned long long)owner->reusedFrames,time-start);
        }
        return 1;
    }
}

extern "C" void fsr411_invalidate_owner(void *device,void *owner) {
    std::lock_guard<std::mutex> lock(contextMutex);
    auto c=contexts.find(device);if(c==contexts.end())return;
    auto v=c->second->owners.find(owner);if(v!=c->second->owners.end())v->second->invalidate();
    if(c->second->lastOwner==owner){c->second->lastOutput=nil;c->second->lastOwner=nullptr;}
}
extern "C" void fsr411_release_owner(void *device,void *owner) {
    std::lock_guard<std::mutex> lock(contextMutex);
    auto c=contexts.find(device);if(c==contexts.end())return;
    c->second->recorder.release_owner(owner);
    auto v=c->second->owners.find(owner);
    if(v!=c->second->owners.end()){
        c->second->retired.push_back(std::move(v->second));c->second->owners.erase(v);
    }
    if(c->second->lastOwner==owner){c->second->lastOutput=nil;c->second->lastOwner=nullptr;}
    c->second->traces.erase(owner);
    c->second->collect();
}
