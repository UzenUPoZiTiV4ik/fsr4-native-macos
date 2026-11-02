// SPDX-License-Identifier: MIT
// Uniform group selection for the experimental Lite Lossy spatial cadence.
// Coordinates and halos are in OUTPUT pixels, even for quarter-scale tensors.
#pragma once
#include "fsr411_model.h"
namespace fsr412_tiling {
using fsr411::u32;
struct Control {
    u32 tileSize, phase, halo, outputScale;
    u32 groupWidth, groupHeight, outputWidth, outputHeight;
};
static_assert(sizeof(Control)==32, "Lite neural tile constant ABI");
inline u32 minimum(u32 a,u32 b) {return a<b?a:b;}
inline bool valid_size(u32 tile) {return tile>=128u && tile<=2048u && !(tile&(tile-1u));}
inline bool fresh_pixel(u32 tile,u32 phase,u32 width,u32 height,u32 x,u32 y) {
    if(!valid_size(tile) || width<4u || height<4u) return true;
    return x<2u || y<2u || x>=width-2u || y>=height-2u ||
        (((x/tile)+(y/tile))&1u)==(phase&1u);
}
// This predicate is identical for every thread of a group. A rejected group
// returns BEFORE any MPP/SIMD operation or threadgroup barrier. Invalid
// controls conservatively run the original full graph, never omit work.
inline bool active(Control c,u32 gx,u32 gy) {
    if(!c.tileSize) return true;
    if(!valid_size(c.tileSize) || c.phase>1u || c.halo>64u ||
       !(c.outputScale==1u || c.outputScale==2u || c.outputScale==4u) ||
       !c.groupWidth || !c.groupHeight || c.groupWidth>128u || c.groupHeight>8u ||
       !c.outputWidth || !c.outputHeight || c.outputWidth>7680u || c.outputHeight>4320u ||
       gx>7680u || gy>4320u) return true;
    const u32 sx=c.groupWidth*c.outputScale,sy=c.groupHeight*c.outputScale;
    const u32 x=gx*sx,y=gy*sy;
    if(x>=c.outputWidth || y>=c.outputHeight) return true;
    const u32 right=minimum(x+sx-1u,c.outputWidth-1u);
    const u32 bottom=minimum(y+sy-1u,c.outputHeight-1u);
    // The POST fresh mask always refreshes the outer two pixels. Include
    // their input halo too; never leave a stale border in an intermediate.
    if(c.outputWidth<4u || c.outputHeight<4u || x<=c.halo+1u || y<=c.halo+1u ||
       right+c.halo>=c.outputWidth-2u || bottom+c.halo>=c.outputHeight-2u) return true;
    const u32 left=x>c.halo?x-c.halo:0u,top=y>c.halo?y-c.halo:0u;
    const u32 lastX=minimum(right+c.halo,c.outputWidth-1u);
    const u32 lastY=minimum(bottom+c.halo,c.outputHeight-1u);
    for(u32 ty=top/c.tileSize;ty<=lastY/c.tileSize;++ty)
        for(u32 tx=left/c.tileSize;tx<=lastX/c.tileSize;++tx)
            if(((tx+ty)&1u)==c.phase) return true;
    return false;
}
}
