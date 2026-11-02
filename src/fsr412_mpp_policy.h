// SPDX-License-Identifier: MIT
// Host-only conservative policy for the standalone FSR 4.1.2 M5 fusion.
// This does NOT enable any runtime path or alter the common/on-disk ABI.
#pragma once
#include "fsr411_model.h"
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>

namespace fsr412 {
namespace mpp_tuning {
using fsr411::PassParams;
using fsr411::Tensor;
using fsr411::u32;

// Separate capability namespace from fsr411::kMppChain*; never OR these into
// the existing runtime feature mask. Probe writes one uint32, reads no scratch
// or model bytes, and uses one SIMD32 group. Keep masks tied to shader bits.
enum : u32 { kFusedCnb32x2 = 1u, kFusedCnb64x2 = 2u, kFusedKnownMask = 3u };
struct ProbeDescriptor {
    const char *source;
    const char *function;
    u32 threads, outputBytes, scratchBytes, weightsBytes;
};
inline constexpr ProbeDescriptor fusion_probe() {
    return {"src/fsr412_mpp.metal","fsr412_cnb_ds_tuning_probe",32,4,0,0};
}

struct FunctionConstantRange { u32 first,count; };
inline constexpr FunctionConstantRange fusion_constant_ranges[]={{0,12},{12,6},{22,2}};
// 12 uint4 CNB records at indices0..11, DS records logically at12..23.
// The shader declares DS12..17 and22..23 only. DO NOT set unused18..21 via
// Metal's function-constant API. Use fusion_constant_ranges for the actual
// three calls to setConstantValues: UInt4, range:first/count.
struct FusionSpecialization { u32 values[24][4]; };
inline FusionSpecialization fusion_specialization(const PassParams &cnb,const PassParams &ds) {
    FusionSpecialization result{};
    const auto a=fsr411::specialize_metadata(cnb,false),b=fsr411::specialize_metadata(ds,false);
    std::memcpy(result.values,a.values,sizeof(a.values));
    std::memcpy(result.values+12,b.values,sizeof(b.values));
    return result;
}

struct FusionRequest {
    // Caller must verify the original-model route, trained normal Lite Lossy
    // artifact identity, Metal4/Apple10 device and explicit user opt-in.
    // A generic "optimize" flag alone is deliberately insufficient.
    bool originalModelRoute=false;
    bool explicitFusionOptIn=false;
    bool verifiedNormalLiteLossy=false;
    bool apple10Metal4=false;
    u32 probedFeatures=0;
    std::uint64_t scratchBytes=0,weightsBytes=0,parameterBytes=0;
    std::uint64_t cnbParameterOffset=0,dsParameterOffset=0;
};
enum class Reject {
    None, NotOriginalOptIn, UnverifiedModel, UnsupportedDevice, UnsupportedProbe,
    UnmeasuredActiveSize, WrongTensorGeometry, AliasedScratch, InvalidWeightRegions,
    ChangedQuantizers, InvalidParameterBindings
};
struct FusionPlan {
    bool enabled=false;
    Reject reject=Reject::NotOriginalOptIn;
    const char *function=nullptr;
    u32 tileInputX=0,tileInputY=0,groupsX=0,groupsY=0,threads=0;
    u32 cnbParameterBinding=2,dsParameterBinding=3,parameterBytesEach=sizeof(PassParams);
    std::uint64_t cnbParameterOffset=0,dsParameterOffset=0;
};
inline FusionPlan rejected(Reject r) {FusionPlan p;p.reject=r;return p;}
inline bool same_tensor(const Tensor &a,const Tensor &b) {
    return a.offset==b.offset && a.strideX==b.strideX && a.strideY==b.strideY
        && a.width==b.width && a.height==b.height && a.channels==b.channels && a.bankStride==b.bankStride;
}
inline std::uint64_t end64(Tensor t) {
    if(!t.width || !t.height || !t.channels)return t.offset;
    const std::uint64_t last=t.bankStride?std::uint64_t((t.channels-1)/16)*t.bankStride+(t.channels-1)%16:t.channels-1;
    return std::uint64_t(t.offset)+std::uint64_t(t.height-1)*t.strideY+std::uint64_t(t.width-1)*t.strideX+last+1;
}
inline bool valid_tensor(Tensor t,std::uint64_t bytes) {
    if(!t.width || !t.height || (t.channels!=16 && t.channels!=32))return false;
    const u32 pixel=t.bankStride?16:t.channels;
    if(t.offset%16 || t.strideX%16 || t.strideY%16 || t.strideX<pixel
        || std::uint64_t(t.strideY)<std::uint64_t(t.width)*t.strideX
        || (t.bankStride && (t.bankStride%16 || std::uint64_t(t.bankStride)<std::uint64_t(t.height)*t.strideY)))return false;
    const auto end=end64(t);
    // Shader tensor_address() wraps uint32: reject rather than permit wrap.
    return end<=bytes && end<=std::numeric_limits<u32>::max();
}
inline bool overlap(Tensor a,Tensor b) {
    return std::uint64_t(a.offset)<end64(b) && std::uint64_t(b.offset)<end64(a);
}
inline bool region(u32 offset,std::uint64_t length,std::uint64_t bytes,u32 alignment) {
    return offset!=fsr411::kNoOffset && offset%alignment==0 && std::uint64_t(offset)+length<=bytes;
}
inline bool q_equal(fsr411::Epilogue a,fsr411::Epilogue b) {
    return a.accumulatorShift==b.accumulatorShift && a.residualShift==b.residualShift
        && a.rightShift==b.rightShift && a.flags==b.flags;
}
inline FusionPlan fusion_plan(FusionRequest r,const PassParams &cnb,const PassParams &ds) {
    if(!r.originalModelRoute || !r.explicitFusionOptIn)return rejected(Reject::NotOriginalOptIn);
    if(!r.verifiedNormalLiteLossy)return rejected(Reject::UnverifiedModel);
    if(!r.apple10Metal4)return rejected(Reject::UnsupportedDevice);
    if((r.probedFeatures&~kFusedKnownMask) || !(r.probedFeatures&kFusedCnb32x2))return rejected(Reject::UnsupportedProbe);
    // ONLY measured active size. Neither tier2160 nor >=1920x1080 implies a
    // speed win. In particular a856x556 viewport in tier2160 is not eligible.
    if(cnb.input.width!=1920 || cnb.input.height!=1080
        || cnb.output.width!=1920 || cnb.output.height!=1080
        || ds.output.width!=960 || ds.output.height!=540)return rejected(Reject::UnmeasuredActiveSize);
    if(cnb.input.channels!=16 || cnb.output.channels!=16 || ds.input.channels!=16 || ds.output.channels!=32
        || !same_tensor(cnb.output,ds.input)
        || !valid_tensor(cnb.input,r.scratchBytes) || !valid_tensor(cnb.output,r.scratchBytes)
        || !valid_tensor(ds.output,r.scratchBytes))return rejected(Reject::WrongTensorGeometry);
    if(overlap(cnb.input,cnb.output) || overlap(cnb.input,ds.output) || overlap(cnb.output,ds.output))
        return rejected(Reject::AliasedScratch);
    if(!region(cnb.weights[0],2304,r.weightsBytes,64) || !region(cnb.weights[1],512,r.weightsBytes,64)
        || !region(cnb.weights[2],512,r.weightsBytes,64) || !region(ds.weights[0],2048,r.weightsBytes,64)
        || !region(cnb.biases[0],64,r.weightsBytes,16) || !region(cnb.biases[1],128,r.weightsBytes,16)
        || !region(cnb.biases[2],64,r.weightsBytes,16) || !region(ds.biases[0],128,r.weightsBytes,16))
        return rejected(Reject::InvalidWeightRegions);
    constexpr fsr411::Epilogue cq[]={{0,0,7,0},{0,0,9,1},{0,5,6,0},{0,5,6,0}};
    constexpr fsr411::Epilogue dq[]={{0,0,6,0},{0,0,7,0}};
    for(u32 i=0;i<4;++i)if(!q_equal(cnb.q[i],cq[i]))return rejected(Reject::ChangedQuantizers);
    for(u32 i=0;i<2;++i)if(!q_equal(ds.q[i+4],dq[i]))return rejected(Reject::ChangedQuantizers);
    const auto a=r.cnbParameterOffset,b=r.dsParameterOffset,n=std::uint64_t(sizeof(PassParams));
    if(a%256 || b%256 || a>r.parameterBytes || b>r.parameterBytes
        || n>r.parameterBytes-a || n>r.parameterBytes-b || (a<b+n && b<a+n))
        return rejected(Reject::InvalidParameterBindings);
    FusionPlan p;p.enabled=true;p.reject=Reject::None;p.function="fsr412_cnb_ds_coop_32x2";
    p.tileInputX=32;p.tileInputY=2;p.groupsX=60;p.groupsY=540;p.threads=32;
    p.cnbParameterOffset=a;p.dsParameterOffset=b;return p;
}
} // namespace mpp_tuning
} // namespace fsr412
