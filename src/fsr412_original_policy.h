// SPDX-License-Identifier: MIT
#pragma once
#include <cstdint>
#include <cstring>
#include "fsr411_model.h"
#include "fsr4_input.h"

namespace fsr412_original_policy {

inline bool should_skip(int tier, const Fsr4InputPrepared &p) {
    if ((tier != 1080 && tier != 2160) || p.reset) return false;

    // Mirror the pinned PRE's bit comparisons, including negative zero and
    // the exact cutoff (two float32 steps below -1/6).
    uint32_t x = 0, y = 0;
    std::memcpy(&x, &p.jitter[0], sizeof(x));
    std::memcpy(&y, &p.jitter[1], sizeof(y));
    const bool negativeX = (x >> 31) && (x & 0x7fffffffu);
    const bool belowCutoff = (y >> 31) && (y & 0x7fffffffu) > 1042983597u;
    return negativeX && (uint64_t(p.rw) * 2 >= p.alignedUW || !belowCutoff);
}

// The pinned dll-2026-11-01 graph uses round-to-nearest-even for the
// integer epilogues replaced by native kernels.
inline bool native_rounding_matches_pin(const fsr411::PassParams &p) {
    for (const auto &q : p.q)
        if (q.flags & fsr411::kRoundHalfUp) return false;
    return true;
}

} // namespace fsr412_original_policy
