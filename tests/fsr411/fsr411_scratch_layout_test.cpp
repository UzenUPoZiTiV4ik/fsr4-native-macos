// SPDX-License-Identifier: MIT
#include "../../src/fsr41x/fsr411_scratch_layout.h"
#include <cassert>
#include <cstdio>
#include <initializer_list>

int main() {
    struct Expected { uint32_t tier; uint64_t arena, appendBase, sourceEnd, aliasBase, cacheEnd; };
    const Expected cases[]={{1080,20880256,20880384,8342464,8342528,16636928},
        {2160,83232256,83232256,33273664,33273856,66451456},
        {4320,332352256,332352256,132902464,132902656,265613056}};
    for(const auto &c:cases) for(bool alias:{false,true}) {
        fsr411::PostCnbScratchLayout l;
        assert(fsr411::post_cnb_scratch_layout(c.tier,alias,l));
        assert(l.arenaBytes==c.arena && l.sourcePlaneBytes==c.sourceEnd);
        assert(l.cacheOffset==(alias?c.aliasBase:c.appendBase));
        assert(l.cacheOffset%256==0 && l.sourcePlaneBytes<=l.cacheOffset);
        assert(l.cacheBytes==c.cacheEnd-c.aliasBase);
        assert(l.cacheOffset+l.cacheBytes<=l.allocationBytes);
        assert(l.allocationBytes==(alias?c.arena:c.appendBase+l.cacheBytes));

        fsr411::PassParams p={};
        p.input={(l.width+3u)*16u,16u,(l.width+2u)*16u,l.width,l.height,16u,0u};
        p.output={0u,16u,l.rowStride,l.width,l.height,16u,0u};
        assert(fsr411::post_cnb_tensors_match_layout(p,l));
        // Both spatial extremes and every channel remain in the protected A
        // plane. The padded physical plane is larger than all legal CNB reads.
        for(uint32_t y:{0u,l.height-1}) for(uint32_t x:{0u,l.width-1})
            for(uint32_t channel=0;channel<16;++channel)
                assert(fsr411::tensor_address(p.input,x,y,channel)<l.sourcePlaneBytes);
        auto output=p.output;output.offset=uint32_t(l.cacheOffset);
        assert(fsr411::tensor_address(output,l.width-1,l.height-1,15)+1u==l.cacheOffset+l.cacheBytes);
        for(uint32_t fsr411::Tensor::*field:{&fsr411::Tensor::offset,&fsr411::Tensor::strideX,
                &fsr411::Tensor::strideY,&fsr411::Tensor::width,&fsr411::Tensor::height,
                &fsr411::Tensor::channels,&fsr411::Tensor::bankStride}) {
            auto bad=p; ++(bad.input.*field);assert(!fsr411::post_cnb_tensors_match_layout(bad,l));
            bad=p;++(bad.output.*field);assert(!fsr411::post_cnb_tensors_match_layout(bad,l));
        }
        // Existing function constants explicitly leave pass13 output dynamic.
        auto specialization=fsr411::specialize_metadata(p,true);
        assert(specialization.values[3][3]==1u);
    }
    fsr411::PostCnbScratchLayout invalid;
    assert(!fsr411::post_cnb_scratch_layout(0,true,invalid));
    assert(!fsr411::post_cnb_scratch_layout(1440,true,invalid));
    std::puts("PASS: POST-CNB scratch layouts, all tiers and both policies");
}
