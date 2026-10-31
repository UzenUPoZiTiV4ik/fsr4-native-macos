// SPDX-License-Identifier: MIT
#include "../src/fsr411_mpp_policy.h"
#include <cassert>
#include <cstdio>
#include <cstring>

int main() {
    using namespace fsr411;
    struct Case { const char *before,*after; u32 x,y,threads,afterX,feature; };
    constexpr Case cases[] = {
        {"fsr411_cnb_coop_chain_32x1","fsr411_cnb_coop_fast_32x1",32,1,32,32,kMppChainCnb},
        {"fsr411_fnb32_coop_chain_16x1","fsr411_fnb32_coop_fast_16x1",16,1,32,16,kMppChainFnb32},
        {"fsr411_ct32_coop_chain_16x1","fsr411_ct32_coop_fast_16x1",16,1,32,16,kMppChainCt32},
        {"fsr411_fnb64_16x1","fsr411_fnb64_coop_fast_16x1",16,1,128,16,kMppChainFnb64},
        {"fsr411_ct64_16x1","fsr411_ct64_coop_fast_16x1",16,1,128,16,kMppChainCt64},
        {"fsr411_ds32","fsr411_ds32_coop_fast_16x1",32,1,128,16,kMppChainDs32},
    };
    unsigned checks=0;
    for (const auto &c : cases) for (u32 mask=0;mask<=kMppChainKnownMask;++mask) {
        const auto d=mpp_fast_dispatch(c.before,c.x,c.y,c.threads,mask);
        const bool enabled=(mask & c.feature)!=0;
        assert(!std::strcmp(d.function,enabled?c.after:c.before));
        assert(d.tileX==(enabled?c.afterX:c.x) && d.tileY==c.y);
        assert(d.threads==(enabled?32u:c.threads));
        assert(d.feature==(enabled?c.feature:0));
        for (unsigned axis=0;axis<3;++axis) {
            const u32 x=c.x+(axis==0),y=c.y+(axis==1),t=c.threads+(axis==2);
            const auto legacy=mpp_fast_dispatch(c.before,x,y,t,mask);
            assert(!std::strcmp(legacy.function,c.before) && legacy.tileX==x && legacy.tileY==y && legacy.threads==t && !legacy.feature);
            ++checks;
        }
        ++checks;
    }
    const char *legacyNames[]={"fsr411_ds16","fsr411_fnb64","fsr411_cnb","fsr411_simd_fnb32","unknown"};
    for (const char *name : legacyNames) {
        const auto d=mpp_fast_dispatch(name,32,1,128,kMppChainKnownMask);
        assert(d.function==name && d.tileX==32 && d.tileY==1 && d.threads==128 && !d.feature);
        ++checks;
    }
    assert(!mpp_fast_dispatch(nullptr,0,0,0,kMppChainKnownMask).feature);
    std::printf("MPP dispatch policy: %u checks passed; cache contracts retained\n",checks+1);
}
