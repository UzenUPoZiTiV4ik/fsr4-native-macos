// SPDX-License-Identifier: MIT
#include "anr_live.h"
#include <cassert>
#include <cstdio>
#include <cstring>
#include <limits>

static Fsr4AnrFrame fixture(){
    Fsr4AnrFrame f={};f.structSize=sizeof(f);f.abiVersion=FSR4_ANR_FG_ABI;
    f.device=f.commandBuffer=f.owner=f.color=f.depth=f.motion=f.output=(void*)1;
    f.rw=1280;f.rh=720;f.uw=1920;f.uh=1080;
    f.motionScaleX=1280;f.motionScaleY=720;f.nearPlane=.1f;f.farPlane=1000;f.deltaTime=1.f/60;
    f.depthReversed=1;return f;
}
int main(){
    auto f=fixture();assert(!fsr4_anr_frame_reason(&f));
    auto bad=f;bad.abiVersion=99;assert(fsr4_anr_frame_reason(&bad));
    bad=f;bad.structSize--;assert(fsr4_anr_frame_reason(&bad));
    bad=f;bad.owner=nullptr;assert(fsr4_anr_frame_reason(&bad));
    bad=f;bad.rw=0;assert(fsr4_anr_frame_reason(&bad));
    bad=f;bad.rw=bad.uw+1;assert(fsr4_anr_frame_reason(&bad));
    bad=f;bad.nearPlane=0;assert(fsr4_anr_frame_reason(&bad));
    bad=f;bad.farPlane=bad.nearPlane;assert(fsr4_anr_frame_reason(&bad));
    bad=f;bad.motionScaleX=0;assert(fsr4_anr_frame_reason(&bad));
    bad=f;bad.motionScaleY=std::numeric_limits<float>::infinity();assert(fsr4_anr_frame_reason(&bad));
    bad=f;bad.jitterX=std::numeric_limits<float>::quiet_NaN();assert(fsr4_anr_frame_reason(&bad));
    // The flag describes ui, not color. No UI resource means no compositing.
    bad=f;bad.uiComposited=1;assert(!fsr4_anr_frame_reason(&bad));
    assert(fsr4_anr_ui_mode(&bad)==FSR4_ANR_UI_NONE);
    bad.ui=(void*)2;assert(!fsr4_anr_frame_reason(&bad));
    assert(fsr4_anr_ui_mode(&bad)==FSR4_ANR_UI_PRECOMPOSITED);
    bad.uiComposited=0;assert(!fsr4_anr_frame_reason(&bad));
    assert(fsr4_anr_ui_mode(&bad)==FSR4_ANR_UI_OVERLAY);
    assert(fsr4_anr_ui_mode(nullptr)==FSR4_ANR_UI_NONE);
    bad=f;bad.colorEncoding=3;assert(fsr4_anr_frame_reason(&bad));
    uint8_t buffer[208];memset(buffer,0xaa,sizeof buffer);fsr4_anr_frame_constants(&f,buffer,1);
    float floats[8];memcpy(floats,buffer,32);assert(floats[4]==1920&&floats[5]==1080);
    uint32_t flags;memcpy(&flags,buffer+32,4);assert(flags==(128u|1u|256u));
    uint32_t dimensions[4];memcpy(dimensions,buffer+36,16);assert(dimensions[0]==1920&&dimensions[1]==1080);
    for(unsigned i=204;i<208;++i)assert(buffer[i]==0xaa);
    fsr4_anr_frame_constants(&f,buffer,0);memcpy(&flags,buffer+32,4);assert(!(flags&256));
    puts("ANR frame ABI/input checks passed");
}
