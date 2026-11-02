// SPDX-License-Identifier: MIT
// CPU only: clang++ -std=c++17 -O2 tests/fsr412_original_geometry_test.cpp
//   -o /tmp/fsr412_original_geometry_test && /tmp/fsr412_original_geometry_test
#include "../src/fsr412_original_geometry.h"
#include <cassert>
#include <iostream>
#include <vector>
using namespace fsr412_original_geometry;

static void check_coverage(Extent viewport, Extent classHalf = {1920, 1080}) {
    const auto g = make(viewport, classHalf);
    assert(g.valid);
    std::vector<unsigned char> cnb(cells(g.p1CnbWrites));
    std::vector<unsigned char> ds(2 * cells(g.p1DsWrites) * 4);
    unsigned borderGroups = 0;
    for (uint32_t gy = 0; gy < g.p1Groups.height; ++gy) {
        for (uint32_t gx = 0; gx < g.p1Groups.width; ++gx) {
            if (gy & 2) { borderGroups += gx == 0 && gy == 2; continue; }
            for (uint32_t ty = 0; ty < 2; ++ty) {
                for (uint32_t tx = 0; tx < 64; ++tx) {
                    // Literal integer lane mapping, separate from rectangle derivation.
                    const uint32_t x = tx % 2 + ((2 * gx | (gy & 1)) * 32) + (tx / 4) * 2;
                    const uint32_t y = 2 * ty + (gy / 4) * 4 + (tx % 4) / 2;
                    // Correct provider P1 grid never reaches class-invalid stores,
                    // even for4320 where the sentinel address is in allocation.
                    assert(x < classHalf.width && y < classHalf.height);
                    assert(x < g.p1CnbWrites.width && y < g.p1CnbWrites.height);
                    ++cnb[y * g.p1CnbWrites.width + x];
                    // P1 mask3 stores two uint words; lane0/1 plane0, lane2/3 plane1.
                    const uint32_t plane = (tx % 4) / 2, word = (tx % 2) * 2;
                    const auto index = ((plane * g.p1DsWrites.height + y / 2) * g.p1DsWrites.width + x / 2) * 4 + word;
                    ++ds[index]; ++ds[index + 1];
                }
            }
        }
    }
    assert(borderGroups == 1);
    for (auto count : cnb) assert(count == 1);
    for (auto count : ds) assert(count == 1);

    // Enumerate quarter-grid P3 overdispatch. Original dynamic b0[4] gate is
    // before2x2 upsample stores, unlike P2's dynamic-invalid sentinel stores.
    std::vector<unsigned char> p3(cells(g.p3Writes));
    for (uint32_t y = 0; y < g.p3Groups.height; ++y) {
        for (uint32_t x = 0; x < 64 * g.p3Groups.width; ++x) {
            if (x >= g.classQuarter.width || y >= g.classQuarter.height) continue;
            if (x >= g.activeQuarter.width || y >= g.activeQuarter.height) continue;
            for (uint32_t dy = 0; dy < 2; ++dy)
                for (uint32_t dx = 0; dx < 2; ++dx) {
                    assert(2 * x + dx < g.p3Writes.width && 2 * y + dy < g.p3Writes.height);
                    ++p3[(2 * y + dy) * g.p3Writes.width + 2 * x + dx];
                }
        }
    }
    for (auto count : p3) assert(count == 1);
}

int main() {
    constexpr auto a = make({640, 384}, {1920, 1080});
    static_assert(a.valid && a.p1Groups == Extent{5, 192});
    static_assert(a.p1CnbWrites == Extent{320, 192} && a.p1DsWrites == Extent{160, 96});
    static_assert(a.p1InputInBounds == Extent{321, 193});
    static_assert(a.p2ProviderInvocations == Extent{320, 192});
    static_assert(a.p2FirstRawReads == Extent{322, 194} && a.p2SecondRawReads == Extent{320, 192});
    static_assert(a.p2ValidFirstRawReads == Extent{162, 98});
    static_assert(a.p2ProviderDynamicInvalid == 46080 && a.p2QuarterGridDynamicInvalid == 3072);
    constexpr auto b = make({644, 388}, {1920, 1080});
    static_assert(b.p1CnbWrites == Extent{384, 196} && b.p1DsWrites == Extent{192, 98});
    static_assert(b.p1CnbWrites.width - b.activeHalf.width == 62);
    static_assert(b.p1CnbWrites.height - b.activeHalf.height == 2);
    static_assert(b.p1DsWrites.width - b.activeQuarter.width == 31);
    static_assert(b.p1DsWrites.height - b.activeQuarter.height == 1);
    static_assert(b.p1InputInBounds == Extent{385, 197});
    static_assert(b.p2FirstRawReads == Extent{386, 196} && b.p2SecondRawReads == Extent{384, 194});
    static_assert(b.p2Writes == Extent{161, 97} && b.p3Writes == Extent{322, 194});
    static_assert(b.p2ProviderDynamicInvalid == 58879 && b.p2QuarterGridDynamicInvalid == 3007);
    constexpr auto full = make({3840, 2160}, {1920, 1080});
    static_assert(full.p1CnbWrites == Extent{1920, 1080} && full.p1DsWrites == Extent{960, 540});
    static_assert(full.p1InputInBounds == Extent{1920, 1080});
    static_assert(full.p2FirstRawReads == Extent{962, 542} && full.p2ProviderDynamicInvalid == 0);
    static_assert(full.p3Writes == Extent{1920, 1080});
    static_assert(!make({643, 388}, {1920, 1080}).valid);
    static_assert(!make({644, 387}, {1920, 1080}).valid);
    static_assert(!make({3844, 2160}, {1920, 1080}).valid);
    static_assert(!make({0, 0}, {1920, 1080}).valid);

    for (auto v : {Extent{640, 384}, Extent{644, 388}, Extent{640, 388}, Extent{644, 384}, Extent{3840, 2160}})
        check_coverage(v);
    check_coverage({644, 388}, {3840, 2160});
    check_coverage({7680, 4320}, {3840, 2160});
    // Bottom extent194 needs P1 rows194/195, even when X is perfectly aligned.
    assert((make({640, 388}, {1920, 1080}).p1CnbWrites == Extent{320, 196}));
    assert((make({640, 388}, {1920, 1080}).p1DsWrites == Extent{160, 98}));

    // Pure address geometry of the4320 invalid P2 uint4 sentinel.
    constexpr uint64_t sentinel = 134217728, plane = 132902464, stride = 61472;
    static_assert(sentinel + 16 <= 398707392 && sentinel >= plane);
    static_assert((sentinel - plane) / stride - 1 == 20);
    static_assert((sentinel - plane) % stride / 16 - 1 == 1521);
    constexpr auto full8k = make({7680, 4320}, {3840, 2160});
    static_assert(full8k.p2ProviderDynamicInvalid == 0);
    constexpr auto aligned = make({256, 384}, {3840, 2160});
    static_assert(aligned.p2QuarterGridDynamicInvalid == 0 && aligned.p2ProviderDynamicInvalid > 0);
    constexpr auto l = make_geometry(2160, 644, 388);
    static_assert(l.valid && l.classHalf == Extent{1920, 1080} && l.classQuarter == Extent{960, 540});
    static_assert(l.CNBwritten == Extent{384, 196} && l.DSwritten == Extent{192, 98});
    static_assert(l.actualHalf == Extent{322, 194} && l.actualQuarter == Extent{161, 97});
    static_assert(l.originalProviderP2grid == Extent{6, 194});
    static_assert(l.P == 33273664 && l.Q == 8342464 && l.allocationBytes == 99820992);
    static_assert(l.hasSkipMarker && l.skipMarkerByteOffset == 33273792);
    static_assert(l.hasSavedCNB && l.savedCNBByteOffset == 49941264);
    static_assert(!l.P2sentinelInAllocationNeedsClear);
    constexpr auto l1080 = make_geometry(1080, 640, 384);
    static_assert(l1080.P == 8342464 && l1080.allocationBytes == 25027392);
    static_assert(l1080.skipMarkerByteOffset == 8342592 && l1080.savedCNBByteOffset == 12529104);
    constexpr auto l4320 = make_geometry(4320, 644, 388);
    static_assert(l4320.P == 132902464 && l4320.Q == 33273664 && l4320.allocationBytes == 398707392);
    static_assert(l4320.P2sentinelInAllocationNeedsClear && !l4320.hasSkipMarker && !l4320.hasSavedCNB);
    static_assert(!make_geometry(4320, 7680, 4320).P2sentinelInAllocationNeedsClear);
    static_assert(!make_geometry(1440, 640, 384).valid);
    std::cout << "original geometry CPU tests: OK\n";
}
