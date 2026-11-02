// SPDX-License-Identifier: MIT
#pragma once
#include <cstdint>

// Independently derived geometry only: no shader math, memory clears, cadence,
// resources, or native ABI changes. Evidence: docs/FSR412_ORIGINAL_GEOMETRY.md.
// Original P1 2638456d... .ll:70-101 remap, 3020-3029 guards, 3395-3411 stores.
// Original P2 6a5d98fc... .ll:89-92 class gate, 7432-7459 dynamic store gate.
// Original P3 4a8443a4... .ll:1341-1351 dynamic early return, 1393-1397 2x2.
namespace fsr412_original_geometry {

struct Extent {
    uint32_t width = 0, height = 0;
};
constexpr bool operator==(Extent a, Extent b) {
    return a.width == b.width && a.height == b.height;
}
constexpr bool operator!=(Extent a, Extent b) { return !(a == b); }
constexpr uint32_t ceil_div(uint32_t n, uint32_t d) { return n / d + (n % d != 0); }
constexpr uint32_t minimum(uint32_t a, uint32_t b) { return a < b ? a : b; }
constexpr Extent clip(Extent a, Extent b) {
    return {minimum(a.width, b.width), minimum(a.height, b.height)};
}
constexpr uint64_t cells(Extent e) { return uint64_t(e.width) * e.height; }
constexpr Extent half(Extent e) { return {e.width / 2, e.height / 2}; }
constexpr Extent halo_raw(Extent e) { return {e.width + 2, e.height + 2}; }

struct Geometry {
    bool valid = false;
    Extent activeHalf, activeQuarter, classHalf, classQuarter;
    Extent p1Groups, p1CnbWrites, p1DsWrites, p1InputInBounds;
    Extent p2ProviderGroups, p2ProviderInvocations, p2Writes;
    Extent p2FirstRawReads, p2SecondRawReads;
    Extent p2ValidFirstRawReads, p2ValidSecondRawReads;
    Extent p3Groups, p3Invocations, p3Writes;
    uint64_t p2ProviderDynamicInvalid = 0, p2QuarterGridDynamicInvalid = 0;
};

// viewport = full output W,H. classHalf = baked half-resolution class extent,
// e.g. {1920,1080} for tier2160. Contract here is W,H positive multiples of4,
// within the class. Odd/incomplete quarter dimensions are intentionally rejected.
constexpr Geometry make(Extent viewport, Extent classHalf) {
    Geometry g{};
    if (!viewport.width || !viewport.height || viewport.width % 4 || viewport.height % 4 ||
        !classHalf.width || !classHalf.height || classHalf.width % 64 || classHalf.height % 4 ||
        uint64_t(viewport.width) > uint64_t(classHalf.width) * 2 ||
        uint64_t(viewport.height) > uint64_t(classHalf.height) * 2)
        return g;
    g.valid = true;
    g.activeHalf = half(viewport);
    g.activeQuarter = half(g.activeHalf);
    g.classHalf = classHalf;
    g.classQuarter = half(classHalf);
    g.p1Groups = {ceil_div(g.activeHalf.width, 64), g.activeHalf.height};
    // P1 active group pairs cover64 columns and4 rows. GroupY&2 groups
    // do not contribute neural writes; only(0,2) has the separate border branch.
    g.p1CnbWrites = clip({g.p1Groups.width * 64, ceil_div(g.activeHalf.height, 4) * 4}, classHalf);
    g.p1DsWrites = half(g.p1CnbWrites);
    // In-bounds part of logical CNB-input taps [-1,cnbW] x [-1,cnbH].
    g.p1InputInBounds = clip({g.p1CnbWrites.width + 1, g.p1CnbWrites.height + 1}, classHalf);
    g.p2ProviderGroups = g.p1Groups;
    g.p2ProviderInvocations = clip({g.p2ProviderGroups.width * 64, g.p2ProviderGroups.height}, g.classQuarter);
    g.p2Writes = g.activeQuarter;
    // Raw padded first-plane coordinates start(0,0), second-plane start(1,1).
    // Only first DS plane gets a3x3 halo; second DS plane is center-only.
    g.p2FirstRawReads = halo_raw(g.p2ProviderInvocations);
    g.p2SecondRawReads = g.p2ProviderInvocations;
    g.p2ValidFirstRawReads = halo_raw(g.p2Writes);
    g.p2ValidSecondRawReads = g.p2Writes;
    g.p2ProviderDynamicInvalid = cells(g.p2ProviderInvocations) - cells(g.p2Writes);
    g.p3Groups = {ceil_div(g.activeQuarter.width, 64), g.activeQuarter.height};
    g.p3Invocations = clip({g.p3Groups.width * 64, g.p3Groups.height}, g.classQuarter);
    g.p2QuarterGridDynamicInvalid = cells(g.p3Invocations) - cells(g.p2Writes);
    // P3 dynamic-invalid threads return before stores; each valid invocation
    // writes2x2 half-resolution cells. No P1-like rounded extra write rectangle.
    g.p3Writes = g.activeHalf;
    return g;
}

// Provider-facing layout adapter. Offsets are facts of the original scratch
// layout, not a request to clear memory. A missing optional region has offset0.
struct Layout {
    bool valid = false;
    uint32_t tier = 0;
    Extent classHalf, classQuarter, CNBwritten, DSwritten, actualHalf, actualQuarter;
    Extent originalProviderP2grid;
    bool P2sentinelInAllocationNeedsClear = false;
    bool hasSkipMarker = false, hasSavedCNB = false;
    uint64_t halfRowBytes = 0, quarterRowBytes = 0, P = 0, Q = 0, allocationBytes = 0;
    uint64_t skipMarkerByteOffset = 0, savedJitterByteOffset = 0, savedCNBByteOffset = 0;
    Geometry neural;
};

constexpr Layout make_geometry(uint32_t tier, uint32_t width, uint32_t height) {
    Layout l{};
    if (tier != 1080 && tier != 2160 && tier != 4320) return l;
    const Extent classHalf{tier * 8 / 9, tier / 2};
    const auto g = make({width, height}, classHalf);
    if (!g.valid) return l;
    l.valid = true;
    l.tier = tier;
    l.neural = g;
    l.classHalf = g.classHalf;
    l.classQuarter = g.classQuarter;
    l.CNBwritten = g.p1CnbWrites;
    l.DSwritten = g.p1DsWrites;
    l.actualHalf = g.activeHalf;
    l.actualQuarter = g.activeQuarter;
    l.originalProviderP2grid = g.p2ProviderGroups; // groups, not threads; Z=1.
    l.halfRowBytes = uint64_t(classHalf.width + 2) * 16;
    l.quarterRowBytes = uint64_t(g.classQuarter.width + 2) * 16;
    l.P = l.halfRowBytes * (classHalf.height + 2);
    l.Q = l.quarterRowBytes * (g.classQuarter.height + 2);
    l.allocationBytes = 3 * l.P;
    l.P2sentinelInAllocationNeedsClear = 134217728ull + 16 <= l.allocationBytes &&
                                        g.p2ProviderDynamicInvalid != 0;
    // Original 1080/2160 PRE/P1/P2/P3 cadence marker and POST saved-CNB cache.
    // For4320 those regions/roles do not exist. Cache is16bytes per half cell,
    // starts at logical(0,0) below, and aliases live neural input atY=classH/2.
    l.hasSkipMarker = l.hasSavedCNB = tier != 4320;
    if (l.hasSkipMarker) {
        l.skipMarkerByteOffset = l.P + 128;
        l.savedJitterByteOffset = l.P + 136;
        l.savedCNBByteOffset = 3 * l.P / 2 + l.halfRowBytes + 16;
    }
    return l;
}

} // namespace fsr412_original_geometry
