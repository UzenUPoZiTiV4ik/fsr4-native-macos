// SPDX-License-Identifier: MIT
// Host-only physical layout contract; the original 212-byte model ABI stays intact.
#pragma once
#include <cstdint>
#include "fsr411_model.h"

namespace fsr411 {
struct PostCnbScratchLayout {
    uint64_t arenaBytes=0,sourcePlaneBytes=0,cacheOffset=0,cacheBytes=0,allocationBytes=0;
    uint32_t width=0,height=0,rowStride=0;
};

inline bool post_cnb_scratch_layout(uint32_t tier,bool alias,PostCnbScratchLayout &layout) {
    uint64_t arena=0;
    if(tier==1080) arena=20880256;
    else if(tier==2160) arena=83232256;
    else if(tier==4320) arena=332352256;
    else return false;
    const uint32_t width=tier*8u/9u,height=tier/2u;
    const uint64_t source=uint64_t(width+2u)*(height+2u)*16u;
    const uint64_t bytes=uint64_t(width)*height*16u;
    const uint64_t offset=((alias?source:arena)+255u)&~uint64_t(255u);
    const uint64_t allocated=alias?arena:offset+bytes;
    if(offset<source || offset+bytes>allocated) return false;
    layout={arena,source,offset,bytes,allocated,width,height,width*16u};
    return true;
}

inline bool post_cnb_tensors_match_layout(const PassParams &p,const PostCnbScratchLayout &layout) {
    const uint32_t row=(layout.width+2u)*16u;
    // The last CNB reads only legacy tensor A, including its residual. Keep
    // a complete physical A plane protected, not merely the active viewport.
    return p.input.offset==row+16u && p.input.strideX==16u && p.input.strideY==row &&
        p.input.width==layout.width && p.input.height==layout.height && p.input.channels==16u && !p.input.bankStride &&
        p.output.offset==0u && p.output.strideX==16u && p.output.strideY==layout.rowStride &&
        p.output.width==layout.width && p.output.height==layout.height && p.output.channels==16u && !p.output.bankStride;
}
}
