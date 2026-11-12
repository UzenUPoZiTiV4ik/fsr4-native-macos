// SPDX-License-Identifier: MIT
#include "netal_live.h"
#include <cassert>
#include <cstring>
#include <limits>
#include <initializer_list>
#include <cstdio>

static Fsr4LiveFrameV2 fixture() {
    Fsr4LiveFrameV2 v={};v.structSize=sizeof(v);v.abiVersion=FSR4_LIVE_FRAME_ABI_V2;
    v.frame.device=v.frame.commandBuffer=v.frame.owner=v.frame.color=v.frame.depth=v.frame.motion=v.frame.output=(void*)1;
    v.frame.rw=1280;v.frame.rh=720;v.frame.uw=1920;v.frame.uh=1080;
    v.frame.depthReversed=1;v.frame.preExposure=1;v.frame.mvScaleX=1280;v.frame.mvScaleY=720;
    v.geometry.color=v.geometry.depth=v.geometry.motion={1280,720,0,0};v.geometry.output={1920,1080,0,0};
    v.geometry.maxInputWidth=1280;v.geometry.maxInputHeight=720;return v;
}
int main() {
    auto v=fixture();assert(!fsr4_netal_input_reason(&v));
    auto bad=v;bad.structSize--;assert(fsr4_netal_input_reason(&bad));
    bad=v;bad.frame.owner=nullptr;assert(fsr4_netal_input_reason(&bad));
    bad=v;bad.geometry.color.offsetX=1;assert(fsr4_netal_input_reason(&bad));
    bad=v;bad.geometry.motion.width=4;assert(fsr4_netal_input_reason(&bad));
    bad=v;bad.geometry.maxInputHeight=719;assert(fsr4_netal_input_reason(&bad));
    bad=v;bad.frame.depthReversed=0;assert(fsr4_netal_input_reason(&bad));
    bad=v;bad.frame.preExposure=2;assert(fsr4_netal_input_reason(&bad));
    for(int value : {-1,1}) {
        bad=v;bad.frame.jitteredMV=value;assert(fsr4_netal_input_reason(&bad));
        bad=v;bad.frame.outputResMV=value;assert(fsr4_netal_input_reason(&bad));
        bad=v;bad.frame.autoExposure=value;assert(fsr4_netal_input_reason(&bad));
    }
    bad=v;bad.frame.jitterX=std::numeric_limits<float>::quiet_NaN();assert(fsr4_netal_input_reason(&bad));
    bad=v;bad.frame.jitterY=.6f;assert(fsr4_netal_input_reason(&bad));
    bad=v;bad.frame.mvScaleX=0;assert(fsr4_netal_input_reason(&bad));
    bad=v;bad.frame.mvScaleY=std::numeric_limits<float>::infinity();assert(fsr4_netal_input_reason(&bad));
    puts("Netal input contract passed");
}
