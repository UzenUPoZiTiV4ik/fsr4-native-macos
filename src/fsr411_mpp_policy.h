// SPDX-License-Identifier: MIT
// Exact-safe M5 kernel upgrades for established cache dispatch contracts.
#pragma once
#include "fsr411_model.h"
#include <cstring>

namespace fsr411 {
struct MppDispatch {
    const char *function;
    u32 tileX, tileY, threads;
    u32 feature; // zero means keep the original kernel and schedule
};

inline MppDispatch mpp_fast_dispatch(const char *function, u32 x, u32 y, u32 threads, u32 features) {
    const MppDispatch original{function,x,y,threads,0};
    if (!function || !features) return original;
    struct Variant { const char *original; u32 x,y,threads; MppDispatch fast; };
    constexpr Variant variants[] = {
        {"fsr411_cnb_coop_chain_32x1",32,1,32,
            {"fsr411_cnb_coop_fast_32x1",32,1,32,kMppChainCnb}},
        {"fsr411_fnb32_coop_chain_16x1",16,1,32,
            {"fsr411_fnb32_coop_fast_16x1",16,1,32,kMppChainFnb32}},
        {"fsr411_ct32_coop_chain_16x1",16,1,32,
            {"fsr411_ct32_coop_fast_16x1",16,1,32,kMppChainCt32}},
        {"fsr411_fnb64_16x1",16,1,128,
            {"fsr411_fnb64_coop_fast_16x1",16,1,32,kMppChainFnb64}},
        {"fsr411_ct64_16x1",16,1,128,
            {"fsr411_ct64_coop_fast_16x1",16,1,32,kMppChainCt64}},
        {"fsr411_ds32",32,1,128,
            {"fsr411_ds32_coop_fast_16x1",16,1,32,kMppChainDs32}},
    };
    for (const auto &v : variants)
        if ((features & v.fast.feature) && x==v.x && y==v.y && threads==v.threads
            && !std::strcmp(function,v.original)) return v.fast;
    // Never reinterpret a legacy, custom or malformed cache's tile/threads.
    // DS16 deliberately stays on its measured faster shared-memory kernel.
    return original;
}
} // namespace fsr411
