// SPDX-License-Identifier: MIT
#include "../src/fsr412_original_policy.h"
#include <cassert>
#include <cmath>
#include <initializer_list>

using fsr412_original_policy::should_skip;

static void check_skip() {
    Fsr4InputPrepared p{};
    p.alignedUW = 640;
    p.rw = 320;
    p.jitter[0] = -0.25f;
    p.jitter[1] = -0.25f;
    assert(should_skip(1080, p));
    assert(should_skip(2160, p));
    assert(!should_skip(4320, p));

    p.reset = 1;
    assert(!should_skip(2160, p));
    p.reset = 0;
    for (const float x : {0.125f, 0.0f, -0.0f}) {
        p.jitter[0] = x;
        assert(!should_skip(2160, p));
    }

    p.jitter[0] = -0.25f;
    p.rw = 319;
    constexpr float cutoff = -0x1.55555ap-3f;
    p.jitter[1] = std::nextafter(cutoff, -1.0f);
    assert(!should_skip(2160, p));
    p.jitter[1] = cutoff;
    assert(should_skip(2160, p));
    p.jitter[1] = std::nextafter(cutoff, 0.0f);
    assert(should_skip(2160, p));
    p.jitter[1] = -1.f / 6.f;
    assert(should_skip(2160, p));
}

static void check_rounding() {
    fsr411::PassParams pass{};
    pass.q[0] = {0, 0, 8, fsr411::kFirstTapBias};
    pass.q[1] = {0, 0, 8, fsr411::kReluBeforeRound};
    pass.q[2] = {0, 5, 6, 0};
    pass.q[3] = {0, 6, 7, 0};
    pass.q[4] = {0, 7, 8, 0};
    pass.q[5] = {0, 7, 8, 0};
    assert(fsr412_original_policy::native_rounding_matches_pin(pass));
    for (auto &q : pass.q) {
        q.flags |= fsr411::kRoundHalfUp;
        assert(!fsr412_original_policy::native_rounding_matches_pin(pass));
        q.flags &= ~fsr411::kRoundHalfUp;
    }
}

int main() {
    check_skip();
    check_rounding();
}
