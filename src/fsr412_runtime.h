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
inline const char *profile_name(Profile p) {return p==Profile::Int8LiteLossy?"int8-lite-lossy":"int8-lite";}
inline const char *profile_name(const Options &o) {
    if(o.profile==Profile::Int8LiteLossy && o.originalLossy) return "int8-lite-lossy-original";
    return o.profile==Profile::Int8LiteLossy && o.neuralTileSize
        ?"int8-lite-lossy-tiled":fsr412::profile_name(o.profile);
}
// int8-lite-pixfix is the same model with Options::pixelFix (POST filter widening).
inline bool pixel_fix_profile(const std::string &name) {return name=="int8-lite-pixfix" || name=="lite-pixfix" || name=="int8-lite-hair" || name=="lite-hair";}
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
inline bool tiled_profile(const std::string &name) {return name=="int8-lite-lossy-tiled" || name=="lite-lossy-tiled";}
// Opt-in: the release's original PRE/BORDER/POST shaders with exact native
// integer neural stages. Needs the external original bundle (build/).
inline bool original_profile(const std::string &name) {return name=="int8-lite-lossy-original" || name=="lite-lossy-original";}
inline bool parse_profile(const std::string &name,Profile &p) {
    if(name=="int8-lite-lossy" || name=="lite-lossy" || tiled_profile(name) || original_profile(name)) {
        p=Profile::Int8LiteLossy;return true;
    }
    if(name!="int8-lite" && name!="lite" && !pixel_fix_profile(name))return false;
    p=Profile::Int8Lite;return true;
}
inline Profile profile_from_int(int value) {return value==7 || value==8?Profile::Int8LiteLossy:Profile::Int8Lite;}
inline bool parse_neural_tile_size(const std::string &text,unsigned &size) {
    for(unsigned value : {0u,128u,256u,512u,1024u,2048u})
        if(text==std::to_string(value)) {size=value;return true;}
    return false;
}
inline bool validate_options(const Options &o,std::string &error) {
    if(o.temporalQuality!=0) {
        error="--temporal-quality / FSR4_TEMPORAL_QUALITY is only supported by FSR 4.1.1 Lossy/lossy2; set it to 0 for FSR 4.1.2";
        return false;
    }
    if(o.neuralTileSize && o.profile!=Profile::Int8LiteLossy) {
        error="neural tile size is supported only by FSR 4.1.2 Lite Lossy";return false;
    }
    unsigned parsed=0;
    if(!parse_neural_tile_size(std::to_string(o.neuralTileSize),parsed)) {
        error="neural tile size must be 0, 128, 256, 512, 1024, or 2048";return false;
    }
    if(o.profile==Profile::Int8LiteLossy && o.pixelFix) {
        error="the pixelization fix patches the ordinary Lite POST; it is not available for Lite Lossy";return false;
    }
    if(o.originalLossy && (o.profile!=Profile::Int8LiteLossy || o.neuralTileSize || o.pixelFix)) {
        error="int8-lite-lossy-original runs the release graph unchanged: no neural tiles or pixelization fix";
        return false;
    }
    if(o.profile==Profile::Int8LiteLossy && o.backend==Backend::Reference) {
        error="FSR 4.1.2 Lite Lossy does not support the AMD reference backend; use auto, native/i8mpp, scalar, or simd";
        return false;
    }
    return true;
}
inline TensorSizes tensor_sizes(uint32_t w,uint32_t h) {
    TensorSizes sizes={};constexpr unsigned shifts[9]={1,0,1,2,2,1,0,0,0};
    for(unsigned i=0;i<9;++i) {sizes.v[i][0]=w>>shifts[i];sizes.v[i][1]=h>>shifts[i];}
    return sizes;
}
}
