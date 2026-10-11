#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Generate Metal-portable PRE/POST for the pinned October 7 lossy policy.

Fresh HLSL helpers operate on the MIT import. No GPL shader rewriter is
included or executed. These transformations require the native POST-CNB
cache; they preserve the exact imported files and a separate exact profile.

Host contract: b2 is four uints (skip, block width, block height, reserved).
u12 is a dense uint4 per 4x4 output block (signed motion x/y, signed-depth
bits, depth-valid flag). u13 is one uint per block and must be cleared before
every skipped PRE, with a buffer memory barrier between PRE and POST.
The extents use the aligned output viewport. No scratch aliasing is used for
motion or the ownership plane. The existing appended POST-CNB cache survives
skipped frames; the host must force a full frame until that cache is valid.
"""

import re


SCRATCH_BYTES = {1080: 20880256, 2160: 83232256, 4320: 332352256}
CONSTANT_BYTES = 16
MOTION_BYTES_PER_BLOCK = 16
PLANE_BYTES_PER_BLOCK = 4


def _tier(tier):
    if tier not in SCRATCH_BYTES:
        raise ValueError("unknown FSR 4.1.1 tensor tier")
    return (1080, 2160, 4320).index(tier)


def _replace(source, before, after):
    if source.count(before) != 1:
        raise ValueError("lossy shader transform expected one expression: " + before[:100])
    return source.replace(before, after)


def _declarations(tier):
    index = _tier(tier)
    # The original release quantizes nearness according to the maximum size
    # class, even if a frame occupies a smaller viewport. Keep its precision
    # and tie ordering while using independent dense buffers on Metal.
    xbits = 9 + index
    idbits = 18 + 2 * index
    mantissa = 9 - 2 * index
    return """
// Independent Metal storage for the dll-2026-10-07 lossy temporal policy.
cbuffer Fsr411LossyConstants : register(b2)
{
    uint lossySkip;
    uint lossyBlockWidth;
    uint lossyBlockHeight;
    uint lossyReserved;
};
RWStructuredBuffer<uint4> Fsr411LossyMotion : register(u12);
RWByteAddressBuffer Fsr411LossyPlane : register(u13);
static const uint FSR411_LOSSY_X_BITS = %du;
static const uint FSR411_LOSSY_ID_BITS = %du;
static const uint FSR411_LOSSY_DEPTH_SHIFT = %du;
static const uint FSR411_LOSSY_DEPTH_BIAS = %du;
static const uint FSR411_LOSSY_DEPTH_MAX = %du;
""" % (xbits, idbits, 23 - mantissa, (127 - 31) << mantissa, (1 << (32 - idbits)) - 1)


PRE_HELPERS = r"""
void Fsr411LossyRecord(uint2 pixel, float2 velocity, int2 selectedDepth, bool invalidMotion)
{
    if (lossySkip == 0u || ((pixel.x | pixel.y) & 3u) != 0u)
        return;
    uint2 block = pixel >> 2u;
    if (block.x >= lossyBlockWidth || block.y >= lossyBlockHeight)
        return;
    int2 displacement = invalidMotion ? int2(0, 0) :
        int2(clamp(round(velocity * float2(width, height)), -254.0f, 254.0f));
    float rawDepth = 0.0f;
    float signedDepth = 0.0f;
    uint validDepth = 0u;
#if FFX_MLSR_LOW_RES_MV
    int2 safeDepth = clamp(selectedDepth, int2(0, 0), int2(width_lr, height_lr) - 1);
    rawDepth = InValidRegion(selectedDepth) ? r_depth[safeDepth] : 0.0f;
#if FFX_MLSR_DEPTH_INVERTED
    signedDepth = -rawDepth;
#else
    signedDepth = rawDepth;
#endif
    validDepth = (!invalidMotion && isfinite(rawDepth) && asuint(signedDepth) != 0u) ? 1u : 0u;
#endif
    Fsr411LossyMotion[block.y * lossyBlockWidth + block.x] =
        uint4(asuint(displacement), asuint(signedDepth), validDepth);
    // The published policy still scatters normal depth zero. Its winner
    // masks farther owners, then POST rejects its zero depth bits. Keep
    // that behavior at sky/geometry boundaries; -0 for inverted depth is
    // a present depth because the upstream sentinel test uses raw bits.
#if FFX_MLSR_LOW_RES_MV
    if (invalidMotion || !isfinite(rawDepth))
        return;
#else
    return;
#endif

#if FFX_MLSR_DEPTH_INVERTED
    float nearness = rawDepth;
#else
    float nearness = 1.0f - rawDepth;
#endif
    int compressed = int(asuint(nearness) >> FSR411_LOSSY_DEPTH_SHIFT) - int(FSR411_LOSSY_DEPTH_BIAS);
    uint depthKey = uint(clamp(compressed, 1, int(FSR411_LOSSY_DEPTH_MAX)));
    uint owner = (block.y << FSR411_LOSSY_X_BITS) | block.x;
    uint candidate = (depthKey << FSR411_LOSSY_ID_BITS) | owner;
    int2 previousOrigin = int2(pixel) + displacement;
    int2 first = previousOrigin >> 2;
    int2 last = (previousOrigin + 3) >> 2;
    // A translated 4x4 footprint intersects at most four 4x4 blocks.
    [loop] for (int row = first.y; row <= last.y; ++row)
        [loop] for (int column = first.x; column <= last.x; ++column)
            if (column >= 0 && row >= 0 && column < int(lossyBlockWidth) && row < int(lossyBlockHeight))
            {
                uint ignored;
                Fsr411LossyPlane.InterlockedMax((uint(row) * lossyBlockWidth + uint(column)) * 4u,
                                                candidate, ignored);
            }
}
"""


POST_HELPERS = r"""
uint4 Fsr411LossyReadMotion(int2 cell)
{
    int2 clampedCell = clamp(cell, int2(0, 0), int2(width / 2u, height / 2u) - 1);
    uint2 block = uint2(clampedCell) >> 1u;
    return Fsr411LossyMotion[block.y * lossyBlockWidth + block.x];
}

bool Fsr411LossyDifferentMotion(int2 a, int2 b)
{
    int2 difference = abs(a - b);
    return max(difference.x, difference.y) > 1;
}

void Fsr411LossyClassify(int2 cell, uint4 own, out bool atRest, out bool uncovered)
{
    int2 displacement = asint(own.xy);
    bool otherMotion = Fsr411LossyDifferentMotion(displacement,
        asint(Fsr411LossyReadMotion(cell + (displacement >> 1)).xy));
    // The release probes in units of 2x2 output-pixel cells, not blocks.
    static const int2 offsets[8] = {
        int2(3, 0), int2(-3, 0), int2(0, 3), int2(0, -3),
        int2(8, 0), int2(-8, 0), int2(0, 8), int2(0, -8)
    };
    [loop] for (uint probe = 0u; probe < 8u; ++probe)
        otherMotion = otherMotion || Fsr411LossyDifferentMotion(displacement,
            asint(Fsr411LossyReadMotion(cell + offsets[probe]).xy));
    atRest = all(displacement == int2(0, 0)) && !otherMotion;
    uncovered = false;

    int2 previousBlock = ((cell * 2) + displacement) >> 2;
    if (own.w != 0u && all(previousBlock >= int2(0, 0)) &&
        previousBlock.x < int(lossyBlockWidth) && previousBlock.y < int(lossyBlockHeight))
    {
        uint key = Fsr411LossyPlane.Load((uint(previousBlock.y) * lossyBlockWidth + uint(previousBlock.x)) * 4u);
        uint idMask = (1u << FSR411_LOSSY_ID_BITS) - 1u;
        uint owner = key & idMask;
        uint2 previousOwner = uint2(owner & ((1u << FSR411_LOSSY_X_BITS) - 1u),
                                    owner >> FSR411_LOSSY_X_BITS);
        if (key != 0u && previousOwner.x < lossyBlockWidth && previousOwner.y < lossyBlockHeight)
        {
            uint4 surface = Fsr411LossyMotion[previousOwner.y * lossyBlockWidth + previousOwner.x];
            uncovered = surface.w != 0u && asfloat(surface.z) < asfloat(own.z) &&
                Fsr411LossyDifferentMotion(asint(surface.xy), displacement);
        }
    }
}
"""


def apply_lossy_pre(source, tier):
    """Call after optional quad routing and before portability conversion."""
    _tier(tier)
    marker = "// fsr411-generated-lossy-pre: tier=%d\n" % tier
    if source.startswith(marker):
        return source
    if "fsr411-generated-lossy-pre:" in source:
        raise ValueError("PRE already transformed for a different tier")
    source = _replace(source, "[numthreads(GROUP_SIZE_*GROUP_SIZE_, 1, 1)]",
                      _declarations(tier) + PRE_HELPERS + "\n[numthreads(GROUP_SIZE_*GROUP_SIZE_, 1, 1)]")
    deadzone = "    velocity.y = abs(velocity.y) <= inv_size.y * 0.03125f ? 0.0f : velocity.y;"
    source = _replace(source, deadzone, deadzone +
                      "\n    Fsr411LossyRecord(dtid, velocity, mv_dialated_pos, disocclusion);")
    history_store = "    rw_reprojected_color[dtid.xy] = DATA_TYPE3(history);"
    source = _replace(source, history_store, history_store + "\n\n"
                      "    // The cached POST-CNB owns skipped frames; leave every neural tensor intact.\n"
                      "    if (lossySkip != 0u) return;")
    return marker + source


def apply_lossy_post(source, tier):
    """Consume original or externally split POST; read its native CNB cache.

    Per destination pixel p, select CNB cell floor((p + D) / 2) and the CT2D
    coefficients for parity(p + D). Clamping applies to the cell, preserving
    parity for odd motion at the viewport edge as in the reference policy.
    """
    _tier(tier)
    marker = "// fsr411-generated-lossy-post: tier=%d\n" % tier
    if source.startswith(marker):
        return source
    if "fsr411-generated-lossy-post:" in source:
        raise ValueError("POST already transformed for a different tier")
    if "    int acc[16];" in source:
        start = source.index("    int acc[16];")
    elif "    uint4 externalCnb = ScratchBuffer.Load4(" in source:
        start = source.index("    uint4 externalCnb = ScratchBuffer.Load4(")
    else:
        raise ValueError("POST is missing its original or external CNB body")
    end = source.index("#if FFX_MLSR_COLORSPACE == FFX_MLSR_COLORSPACE_LINEAR", start)
    offset = (SCRATCH_BYTES[tier] + 255) & ~255
    stride = tier * 8 // 9 * 16
    initial = """    // Native CNB computes this on a full frame; skipped frames preserve it.
    uint4 externalCnb = ScratchBuffer.Load4(%du + uint(iy) * %du + uint(ix) * 16u);
    uint packedVss[4] = {externalCnb.x, externalCnb.y, externalCnb.z, externalCnb.w};
    uint4 lossyOwn = uint4(0u, 0u, 0u, 0u);
    bool lossyAtRest = false, lossyUncovered = false;
    if (lossySkip != 0u)
    {
        lossyOwn = Fsr411LossyReadMotion(int2(ix, iy));
        Fsr411LossyClassify(int2(ix, iy), lossyOwn, lossyAtRest, lossyUncovered);
    }

""" % (offset, stride)
    source = source[:start] + initial + source[end:]
    source = _replace(source, "[numthreads(256, 1, 1)]",
                      _declarations(tier) + POST_HELPERS + "\n[numthreads(256, 1, 1)]")
    coords = "        uint2 tid = uint2((uint)(ix * 2) + (uint)off.x, (uint)(iy * 2) + (uint)off.y);"
    displaced = """
        uint4 pixelCnb = uint4(packedVss[0], packedVss[1], packedVss[2], packedVss[3]);
        uint cnbPixel = pk;
        if (lossySkip != 0u)
        {
            int2 sourcePixel = int2(tid) + asint(lossyOwn.xy);
            int2 sourceCell = clamp(sourcePixel >> 1, int2(0, 0), int2(w2, h2) - 1);
            pixelCnb = ScratchBuffer.Load4(%du + uint(sourceCell.y) * %du + uint(sourceCell.x) * 16u);
            cnbPixel = (asuint(sourcePixel.x) & 1u) | ((asuint(sourcePixel.y) & 1u) << 1u);
        }
""" % (offset, stride)
    source = _replace(source, coords, coords + "\n" + displaced)
    source = _replace(source, "ct[ch] = CT2D_S[pk * 8 + ch];", "ct[ch] = CT2D_S[cnbPixel * 8 + ch];")
    source = _replace(source, "int b = (pk * 32 + ch2 * 4);", "int b = (cnbPixel * 32 + ch2 * 4);")
    for word in range(4):
        old = "packedVss[%d], ct[ch2]);" % word
        source = _replace(source, old, "pixelCnb[%d], ct[ch2]);" % word)
    source = _replace(source, "        float3 up = 0.0f;", "        float3 up = 0.0f;\n"
                      "        float3 lossyMinimum = 3.402823466e+38f, lossyMaximum = -3.402823466e+38f;")
    source = _replace(source, "                up += ApplyMuLaw411(r_input_color[pc], exposure) * g;", """                float3 currentSample = ApplyMuLaw411(r_input_color[pc], exposure);
                up += currentSample * g;
                if (lossySkip != 0u)
                {
                    lossyMinimum = min(lossyMinimum, currentSample);
                    lossyMaximum = max(lossyMaximum, currentSample);
                }""")
    mix = "        float3 res = up * (1.0f - bf) + reproj * bf;"
    source = _replace(source, mix, mix + "\n" + r"""        if (lossySkip != 0u)
        {
            // The release's SB_CONST=0 makes the jitter damping term zero:
            // its current-frame weight is exactly (1 - historyWeight)^2.
            float currentWeight = (1.0f - bf) * (1.0f - bf);
            currentWeight = min(lossyAtRest ? 0.0f : currentWeight, 1.0f);
            float historyWeight = 1.0f - currentWeight;
            res = up * currentWeight + reproj * historyWeight;
            float3 margin = (lossyMaximum - lossyMinimum) * 0.5f;
            float3 limitedHistory = max(min(reproj, lossyMaximum + margin), lossyMinimum - margin);
            res += (limitedHistory - reproj) * historyWeight;
            bool darkEdge = all(reproj <= max(up * 0.1f, 0.0f));
            if (darkEdge || lossyUncovered)
                res = up;
        }
""")
    return marker + source


CELL_HELPERS = r"""
// One classification per 2x2 output cell, shared by its four output pixels.
// The inputs are the cell's own motion and the probed cells only, so every
// pixel of the cell computed identical integer results before this sharing.
groupshared uint2 Fsr411LossyCellMotion[64];
groupshared uint Fsr411LossyCellFlags[64];

void Fsr411LossyClassifyCells(uint lane, uint2 groupId)
{
    // Lanes 192..255 fill at most one color-cache entry each, so they take
    // the 8x8 cells of this 16x16 output tile before the shared barrier.
    if (lossySkip == 0u || lane < 192u)
        return;
    uint local = lane - 192u;
    int2 cell = int2(groupId * 8u + uint2(local & 7u, local >> 3u));
    uint4 own = Fsr411LossyReadMotion(cell);
    bool atRest, uncovered;
    Fsr411LossyClassify(cell, own, atRest, uncovered);
    Fsr411LossyCellMotion[local] = own.xy;
    Fsr411LossyCellFlags[local] = (atRest ? 1u : 0u) | (uncovered ? 2u : 0u);
}
"""


def apply_lossy_post_cells(source):
    """Share per-cell skip classification across a cell's four pixels.

    Call after apply_lossy_post and apply_post_optimization. Classification
    runs in the cache helper before its existing group barrier; only the
    unchanged integer motion/classification values are shared.
    """
    marker = "// fsr411-generated-lossy-post-cells\n"
    if source.startswith(marker):
        return source
    if "fsr411-generated-lossy-post:" not in source or "fsr411-generated-post-color-cache:" not in source:
        raise ValueError("cell classification requires the lossy color-cached POST")
    # The cache helper calls the cell classifier, so declare it first.
    source = _replace(source, "#define FSR411_POST_CACHE_SIDE 20\n", CELL_HELPERS + "#define FSR411_POST_CACHE_SIDE 20\n")
    source = _replace(source,
                      "    // All lanes, including those outside a partial output tile, reach this.\n"
                      "    GroupMemoryBarrierWithGroupSync();",
                      "    Fsr411LossyClassifyCells(lane, outputOrigin >> 4u);\n"
                      "    // All lanes, including those outside a partial output tile, reach this.\n"
                      "    GroupMemoryBarrierWithGroupSync();")
    source = _replace(source,
                      "        lossyOwn = Fsr411LossyReadMotion(int2(ix, iy));\n"
                      "        Fsr411LossyClassify(int2(ix, iy), lossyOwn, lossyAtRest, lossyUncovered);",
                      "        uint cellIndex = ((outputPixel.y >> 1u) & 7u) * 8u + ((outputPixel.x >> 1u) & 7u);\n"
                      "        lossyOwn.xy = Fsr411LossyCellMotion[cellIndex];\n"
                      "        lossyAtRest = (Fsr411LossyCellFlags[cellIndex] & 1u) != 0u;\n"
                      "        lossyUncovered = (Fsr411LossyCellFlags[cellIndex] & 2u) != 0u;")
    return marker + source
