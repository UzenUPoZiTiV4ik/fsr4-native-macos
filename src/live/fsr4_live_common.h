// SPDX-License-Identifier: MIT
// Shared per-owner live bridge for the FSR 4.1.x runtimes. The traits supply
// the Options type, version labels, the initial profile for log throttling,
// the report profile name, and the per-version option parser; device contexts,
// owner lifecycle, placeholder copies, tracing and status are shared.
#pragma once
#include "fsr4_live.h"
#include "fsr4_log.h"
#include "fsr411_runtime.h"
#include "fsr411_capture_io.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <vector>

namespace fsr4_live_bridge {

template<typename T>
struct DeviceContext {
    fsr411::Runtime runtime;
    fsr411_capture::Recorder recorder;
    std::map<void *,std::unique_ptr<fsr411::View>> owners;
    std::vector<std::unique_ptr<fsr411::View>> retired;
    struct Trace {
        uint64_t calls=0;
        double lastLog=0;
        fsr411::Profile profile=T::trace_profile;
    };
    std::map<void *,Trace> traces;
    id<MTLTexture> lastOutput=nil;
    Fsr4InputRect lastOutputRect={};
    void *lastOwner=nullptr;
    double lastTime=0;
    double lastOutputTime=0;
    DeviceContext(id<MTLDevice> dev,const std::string &root,const std::string &build)
        :runtime(dev,root,build),recorder(root,T::version){}
    void collect() {
        retired.erase(std::remove_if(retired.begin(),retired.end(),[](const auto &view){
            return std::all_of(view->slots.begin(),view->slots.end(),[](const auto &slot){return slot->retired();});
        }),retired.end());
    }
};

// One context map and mutex per bridge, like the previous per-TU statics.
template<typename T>
std::map<void *,std::unique_ptr<DeviceContext<T>>> &contexts() {
    static std::map<void *,std::unique_ptr<DeviceContext<T>>> value;
    return value;
}
template<typename T>
std::mutex &context_mutex() {
    static std::mutex value;
    return value;
}

inline const char *env(const char *key) { const char *value=std::getenv(key); return value&&*value?value:nullptr; }
inline bool parse_flag(const char *key,int &value,std::string &error) {
    if(const char *v=env(key)) {
        if(std::strcmp(v,"0") && std::strcmp(v,"1")) {error=std::string(key)+" must be 0 or 1";return false;}
        value=v[0]=='1';
    }
    return true;
}
inline double now_ms() {
    using namespace std::chrono;
    return duration<double,std::milli>(steady_clock::now().time_since_epoch()).count();
}
inline const char *preset(const Fsr4LiveFrameV2 &f) {
    if(auto value=env("FSR4_PRESET")) return value;
    if(f.inputContentProperties==1) return "drs";
    const float ratio=f.frame.rw?float(f.frame.uw)/f.frame.rw:1;
    if(ratio>=2.99f) return "ultraperf";
    if(ratio>=1.99f) return "performance";
    if(ratio>=1.69f) return "balanced";
    if(ratio>=1.50f) return "quality";
    return "native";
}

// FSR4_MODEL/colorspace/motion-flag env handling shared by both bridges.
// `model` is the "FSR 4.1.x" label used in the rejection message.
inline bool env_options(const Fsr4LiveFrameV2 &f,fsr411::Frame &frame,fsr411::Options &o,
                        const char *model,std::string &error) {
    if(auto m=env("FSR4_MODEL")) {
        if(!std::strcmp(m,"auto")) o.backend=fsr411::Backend::Auto;
        else if(!std::strcmp(m,"reference") || !std::strcmp(m,"int8")) o.backend=fsr411::Backend::Reference;
        else if(!std::strcmp(m,"native") || !std::strcmp(m,"i8mpp")) o.backend=fsr411::Backend::Native;
        else if(!std::strcmp(m,"scalar")) o.backend=fsr411::Backend::Scalar;
        else if(!std::strcmp(m,"simd") || !std::strcmp(m,"fp16") || !std::strcmp(m,"f16")) o.backend=fsr411::Backend::Simd;
        else {error=std::string(model)+" model must be auto, reference/int8, scalar, simd/fp16 or native/i8mpp";return false;}
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
    return parse_flag("FSR4_JITTERED_MV",frame.input.jitteredMV,error)
        && parse_flag("FSR4_AUTOEXPOSURE",frame.input.autoExposure,error)
        && parse_flag("FSR4_OPTIMIZE",o.optimize,error);
}
inline bool tail_options(fsr411::Options &o,std::string &error) {
    int compile=1;
    if(!parse_flag("FSR4_COMPILE",compile,error))return false;
    o.allowCompile=compile;
    int rcas=0; fsr4_live_get_sharpness(&o.sharpness,&rcas); o.rcas=rcas;
    return true;
}

inline int failure(Fsr4LiveStatus &status,const char *version,const std::string &reason) {
    status.active=0; ++status.encodes;
    std::snprintf(status.reason,sizeof(status.reason),"%s",reason.c_str());
    fsr4_live_set_status(&status);
    fsr4_logf("%s encode failed: %s",version,reason.c_str());
    return 0;
}

template<typename T>
void trace_input(const Fsr4LiveFrameV2 &v2, const typename T::Options &options,
                 uint64_t call, bool stub) {
    const auto &f=v2.frame;
    const auto &g=v2.geometry;
    fsr4_logf("%s input call=%llu owner=%p cb=%p profile=%s%s "
              "content=%ux%u->%ux%u max=%ux%u "
              "color=%p:%ux%u+%u,%u depth=%p:%ux%u+%u,%u "
              "motion=%p:%ux%u+%u,%u output=%p:%ux%u+%u,%u "
              "jitter=%.9g,%.9g mvScale=%.9g,%.9g preExposure=%.9g "
              "reset=%d depthReversed=%d jitteredMV=%d outputResMV=%d autoExposure=%d drs=%d",
              T::version,(unsigned long long)call,f.owner,f.commandBuffer,
              T::profile_name(options),stub?" STUB":"",
              f.rw,f.rh,f.uw,f.uh,g.maxInputWidth,g.maxInputHeight,
              f.color,g.color.width,g.color.height,g.color.offsetX,g.color.offsetY,
              f.depth,g.depth.width,g.depth.height,g.depth.offsetX,g.depth.offsetY,
              f.motion,g.motion.width,g.motion.height,g.motion.offsetX,g.motion.offsetY,
              f.output,g.output.width,g.output.height,g.output.offsetX,g.output.offsetY,
              f.jitterX,f.jitterY,f.mvScaleX,f.mvScaleY,f.preExposure,
              f.reset,f.depthReversed,f.jitteredMV,f.outputResMV,f.autoExposure,v2.inputContentProperties);
}

template<typename T>
int encode_live_frame(const Fsr4LiveFrameV2 *input) {
    @autoreleasepool {
        std::lock_guard<std::mutex> lock(context_mutex<T>());
        Fsr4LiveStatus status={}; fsr4_live_get_status(&status);
        if(!fsr4_live_frame_v2_valid(input))return failure(status,T::version,"invalid live frame v2 ABI");
        const auto &f=input->frame;
        if(!f.device || !f.commandBuffer || !f.color || !f.depth || !f.motion || !f.output)
            return failure(status,T::version,std::string("missing ")+T::model+" Metal input");
        const char *root=env("FSR4_ROOT");
        if(!root)return failure(status,T::version,std::string("FSR4_ROOT is required for ")+T::model);
        id<MTLDevice> dev=(__bridge id<MTLDevice>)f.device;
        auto &context=contexts<T>()[f.device];
        if(!context) {
            const char *buildOverride=env("FSR4_BUILD_DIR");
            const std::string build=buildOverride?buildOverride:std::string(root)+"/build";
            context=std::make_unique<DeviceContext<T>>(dev,root,build);
        }
        context->collect();
        auto &owner=context->owners[f.owner]; if(!owner)owner=std::make_unique<fsr411::View>();
        fsr411::Frame frame;
        frame.color=(__bridge id<MTLTexture>)f.color; frame.depth=(__bridge id<MTLTexture>)f.depth;
        frame.motion=(__bridge id<MTLTexture>)f.motion; frame.output=(__bridge id<MTLTexture>)f.output;
        frame.exposure=(__bridge id<MTLTexture>)f.exposure;
        frame.input=fsr4_live_input_desc(&f,&input->geometry,input->inputContentProperties);
        typename T::Options options; std::string error;
        auto failed=[&](const std::string &why){
            owner->invalidate();
            context->recorder.failed(f.owner,why);
            if(context->lastOwner==f.owner) {context->lastOutput=nil;context->lastOwner=nullptr;}
            return failure(status,T::version,why);
        };
        if(!T::parse_options(*input,frame,options,error))return failed(error);
        auto &trace=context->traces[f.owner];
        const double start=now_ms();
        ++trace.calls;
        const bool log=trace.calls<=16 || start-trace.lastLog>=5000 || trace.profile!=options.profile;
        if(log) {trace.lastLog=start;trace.profile=options.profile;}
        const uint32_t mvW=frame.input.outputResMV==1?f.uw:f.rw, mvH=frame.input.outputResMV==1?f.uh:f.rh;
        const bool stub=frame.motion.width==4 && frame.motion.height==4 && frame.motion.pixelFormat==MTLPixelFormatRGBA8Unorm
            && (mvW>4 || mvH>4);
        if(log)trace_input<T>(*input,options,trace.calls,stub);
        if(stub) {
            owner->invalidate();
            if(auto policy=env("FSR4_STUB")) if(std::strcmp(policy,"skip"))
                return failed(std::string(T::model)+" cannot run a network on placeholder motion");
            Fsr4InputRect output{input->geometry.output.offsetX,input->geometry.output.offsetY,f.uw,f.uh};
            if(!context->runtime.copy_output((__bridge id)f.commandBuffer,*owner,context->lastOutput,context->lastOutputRect,
                                              frame.output,output,error)) return failed(error);
            if(log)fsr4_logf("%s placeholder owner=%p sourceOwner=%p source=%p destination=%p ageMs=%.3f; history not advanced",
                            T::version,f.owner,context->lastOwner,(__bridge void *)context->lastOutput,f.output,
                            context->lastOutputTime?start-context->lastOutputTime:0);
            ++status.encodes; ++status.stubs; status.active=1;
            std::snprintf(status.reason,sizeof(status.reason),"placeholder: copied previous %s output",T::model);
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
        std::snprintf(status.model,sizeof(status.model),"%s",owner->backendLabel=="int8-mpp"?"i8mpp":owner->backendLabel=="int8-simd"?"simd"
            :owner->backendLabel=="int8-scalar"?"scalar":"i8-ref");
        std::snprintf(status.preset,sizeof(status.preset),"%s",preset(*input));
        std::snprintf(status.variant,sizeof(status.variant),"%s",owner->variantKey.c_str());
        fsr4_live_set_status(&status);
        if(log || time-start>=100) {
            const auto &p=owner->prepared;
            fsr4_logf("%s encoded owner=%p call=%llu backend=%s variant=%s "
                      "mvExtent=%ux%u normalizedMvScale=%.9g,%.9g jitterCancellation=%.9g,%.9g "
                      "effectiveReset=%u resetReasons=0x%x copyMask=0x%x "
                      "history=%p recurrent=%p inference=%s neural=%llu reused=%llu cpuMs=%.3f",
                      T::version,f.owner,(unsigned long long)trace.calls,owner->backendLabel.c_str(),owner->variantKey.c_str(),
                      p.mvW,p.mvH,p.mvScale[0],p.mvScale[1],p.jitterCancellation[0],p.jitterCancellation[1],
                      p.reset,p.resetReasons,p.copyMask,(__bridge void *)owner->history,(__bridge void *)owner->recurrent,
                      owner->skippedInference?"reuse":"full",(unsigned long long)owner->neuralRuns,
                      (unsigned long long)owner->reusedFrames,time-start);
        }
        return 1;
    }
}

template<typename T>
void invalidate_owner(void *device,void *owner) {
    std::lock_guard<std::mutex> lock(context_mutex<T>());
    auto &all=contexts<T>();
    auto c=all.find(device);if(c==all.end())return;
    auto v=c->second->owners.find(owner);if(v!=c->second->owners.end())v->second->invalidate();
    if(c->second->lastOwner==owner){c->second->lastOutput=nil;c->second->lastOwner=nullptr;}
}
template<typename T>
void release_owner(void *device,void *owner) {
    std::lock_guard<std::mutex> lock(context_mutex<T>());
    auto &all=contexts<T>();
    auto c=all.find(device);if(c==all.end())return;
    c->second->recorder.release_owner(owner);
    auto v=c->second->owners.find(owner);
    if(v!=c->second->owners.end()){
        c->second->retired.push_back(std::move(v->second));c->second->owners.erase(v);
    }
    if(c->second->lastOwner==owner){c->second->lastOutput=nil;c->second->lastOwner=nullptr;}
    c->second->traces.erase(owner);
    c->second->collect();
}

} // namespace fsr4_live_bridge
