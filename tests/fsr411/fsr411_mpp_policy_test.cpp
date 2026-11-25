// SPDX-License-Identifier: MIT
#include "../../src/fsr41x/fsr411_mpp_policy.h"
#include <cassert>
#include <cstdio>
#include <cstring>

int main() {
    using namespace fsr411;
    struct Case {
        const char *before;
        u32 x, y, threads;
        const char *primary;
        u32 primaryX, primaryY, primaryFeature;
        const char *secondary;
        u32 secondaryX, secondaryY, secondaryFeature;
    };
    constexpr Case cases[] = {
        {"fsr411_cnb_coop_chain_32x1", 32, 1, 32,
            "fsr411_cnb_coop_fast_32x2", 32, 2, kMppChainCnb32x2,
            "fsr411_cnb_coop_fast_32x1", 32, 1, kMppChainCnb},
        {"fsr411_fnb32_coop_chain_16x1", 16, 1, 32,
            "fsr411_fnb32_coop_fast_32x1", 32, 1, kMppChainFnb32Wide,
            "fsr411_fnb32_coop_fast_16x1", 16, 1, kMppChainFnb32},
        {"fsr411_ct32_coop_chain_16x1", 16, 1, 32,
            "fsr411_ct32_coop_fast_16x1", 16, 1, kMppChainCt32,
            nullptr, 0, 0, 0},
        {"fsr411_fnb64_16x1", 16, 1, 128,
            "fsr411_fnb64_coop_fast_16x1", 16, 1, kMppChainFnb64,
            nullptr, 0, 0, 0},
        {"fsr411_ct64_16x1", 16, 1, 128,
            "fsr411_ct64_coop_fast_16x1", 16, 1, kMppChainCt64,
            nullptr, 0, 0, 0},
        {"fsr411_ds32", 32, 1, 128,
            "fsr411_ds32_coop_fast_16x1", 16, 1, kMppChainDs32,
            nullptr, 0, 0, 0},
    };
    unsigned checks=0;
    for (const auto &c : cases) for (u32 mask=0;mask<=kMppChainKnownMask;++mask) {
        const auto d=mpp_fast_dispatch(c.before,c.x,c.y,c.threads,mask);
        const bool usePrimary = (mask & c.primaryFeature) != 0;
        const bool useSecondary = !usePrimary && c.secondary && (mask & c.secondaryFeature) != 0;
        const char *expectedFn = usePrimary ? c.primary : useSecondary ? c.secondary : c.before;
        const u32 expectedX = usePrimary ? c.primaryX : useSecondary ? c.secondaryX : c.x;
        const u32 expectedY = usePrimary ? c.primaryY : useSecondary ? c.secondaryY : c.y;
        const u32 expectedThreads = (usePrimary || useSecondary) ? 32u : c.threads;
        const u32 expectedFeature = usePrimary ? c.primaryFeature : useSecondary ? c.secondaryFeature : 0u;
        assert(!std::strcmp(d.function, expectedFn));
        assert(d.tileX == expectedX && d.tileY == expectedY);
        assert(d.threads == expectedThreads);
        assert(d.feature == expectedFeature);
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
