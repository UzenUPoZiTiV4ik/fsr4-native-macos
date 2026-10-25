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
inline bool parse_profile(const std::string &name,Profile &p) {
    if(name!="int8-lite" && name!="lite")return false;
    p=Profile::Int8Lite;return true;
}
inline Profile profile_from_int(int) {return Profile::Int8Lite;}
inline TensorSizes tensor_sizes(uint32_t w,uint32_t h) {
    TensorSizes sizes={};constexpr unsigned shifts[9]={1,0,1,2,2,1,0,0,0};
    for(unsigned i=0;i<9;++i) {sizes.v[i][0]=w>>shifts[i];sizes.v[i][1]=h>>shifts[i];}
    return sizes;
}
}
