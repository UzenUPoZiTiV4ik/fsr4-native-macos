// SPDX-License-Identifier: MIT
// FSR 4.1.2 INT8 Lite adapter. The model has its own imported shaders,
// initializer, graph, native parameters and cache; Metal lifetime/input code
// and exact integer primitives are shared with the 4.1.1 implementation.
#pragma once
#include "fsr411_runtime.h"
namespace fsr412 {
using fsr411::Backend;
using fsr411::Profile;
using fsr411::StageProfiler;
using fsr411::FrameSnapshot;
using fsr411::Frame;
using fsr411::View;
using fsr411::Runtime;
using fsr411::TensorSizes;
using fsr411::FrameConstants;
using fsr411::ns;
using fsr411::utf8;
using fsr411::metal_error;
using fsr411::ceil_div;
using fsr411::required_tier;
struct Options : fsr411::Options {
    Options() {modelLite=true;profile=Profile::Int8Lite;}
};
inline const char *profile_name(Profile) {return "int8-lite";}
// int8-lite-pixfix is the same model with Options::pixelFix (POST filter widening).
inline bool pixel_fix_profile(const std::string &name) {return name=="int8-lite-pixfix" || name=="lite-pixfix";}
// Pixelization fix strength 0..1 (panel slider): 0 leaves the filter as AMD
// predicts it, 0.5 is the default, 1 widens it most (floor 0.2, about 2.2x).
constexpr float kPixelFixStrength=0.5f;
inline float pixel_fix_floor(float strength) {
    strength=std::isfinite(strength)?std::clamp(strength,0.f,1.f):kPixelFixStrength;
    return 1.f-0.8f*strength;
}
// Strength 0 selects plain Lite: the patched POST is mathematically the same
// there but not bit-identical under fast math.
inline void apply_pixel_fix(Options &o,bool enabled,float strength) {
    o.pixelFixFloor=pixel_fix_floor(strength);
    o.pixelFix=enabled && o.pixelFixFloor<1.f;
}
inline bool parse_profile(const std::string &name,Profile &p) {
    if(name!="int8-lite" && name!="lite" && !pixel_fix_profile(name))return false;
    p=Profile::Int8Lite;return true;
}
inline Profile profile_from_int(int) {return Profile::Int8Lite;}
inline TensorSizes tensor_sizes(uint32_t w,uint32_t h) {
    TensorSizes sizes={};constexpr unsigned shifts[9]={1,0,1,2,2,1,0,0,0};
    for(unsigned i=0;i<9;++i) {sizes.v[i][0]=w>>shifts[i];sizes.v[i][1]=h>>shifts[i];}
    return sizes;
}
}
