#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Cache POST's float32 input transform and filter one output pixel per lane.

Apply after external_post_cnb / apply_lossy_post and before the portability
conversion. The native CNB cache is required. The imported model sources stay
unchanged. CT2D uses exact float32 integer products and sums; their absolute
partial-sum bound is below 2^24 for every pinned normal/UP table. The original
filter arithmetic and tap order stay intact.

Host dispatch is ceil(outputWidth / 16), ceil(outputHeight / 16), 1 with the
same 256x1x1 threadgroup. Record post_pixels_per_group=16 in variant metadata
and include this module in the build cache identity when enabling it.
"""

POST_PIXELS_PER_GROUP = 16
POST_THREADS_PER_GROUP = 256
POST_CACHE_SIDE = 20
MARKER = "// fsr411-generated-post-color-cache: output-tile=16 cache-side=20 active-load=1 exact-float-ct2d=1\n"


def _replace(source, before, after):
    if source.count(before) != 1:
        raise ValueError("POST color-cache transform expected one expression: " + before[:120])
    return source.replace(before, after)


CT2D_HELPER = r"""
// Each signed INT8 product is an exact float32 integer. For every channel
// in the pinned normal/UP tables, abs(bias) + 128 * sum(abs(weights)) is at
// most 110151/87727. This bounds every partial sum, including any dot-product
// reduction order, below 2^24; the original int32 result is preserved exactly.
// Keep float32 here: half products/accumulation do not have this property.
float Fsr411PostDot4Exact(uint a, uint b, float acc)
{
    return dot(float4(unpack_s8s32(a)), float4(unpack_s8s32(b))) + acc;
}
"""


def _float32_ct2d(source):
    # Accept only the two pinned forms: exact uses its own phase and packed
    # CNB; lossy may reproject the CNB and select another CT2D phase. Retain
    # all indexing and the four dot products in their original order.
    phase, values = ("cnbPixel", "pixelCnb") if "ct[ch] = CT2D_S[cnbPixel * 8 + ch];" in source else ("pk", "packedVss")
    block = """        int ct[8];
        [unroll] for (uint ch = 0; ch < 8; ch++) ct[ch] = CT2D_S[PHASE * 8 + ch];
        [unroll] for (uint ch2 = 0; ch2 < 8; ch2++)
        {
            int b = (PHASE * 32 + ch2 * 4);
            ct[ch2] = dot4add_i8packed(CT2D_W[b], VALUES[0], ct[ch2]);
            ct[ch2] = dot4add_i8packed(CT2D_W[b + 1], VALUES[1], ct[ch2]);
            ct[ch2] = dot4add_i8packed(CT2D_W[b + 2], VALUES[2], ct[ch2]);
            ct[ch2] = dot4add_i8packed(CT2D_W[b + 3], VALUES[3], ct[ch2]);
        }""".replace("PHASE", phase).replace("VALUES", values)
    replacement = block.replace("int ct[8];", "float ct[8];").replace("dot4add_i8packed(", "Fsr411PostDot4Exact(")
    return _replace(source, block, replacement)


CACHE_HELPERS = r"""
// A 16x16 output tile shares overlapping 3x3 input footprints. Store the
// original transform in float32; introducing half here changes the model.
#define FSR411_POST_CACHE_SIDE 20
groupshared float3 Fsr411PostColor[FSR411_POST_CACHE_SIDE * FSR411_POST_CACHE_SIDE];
groupshared int2 Fsr411PostColorOrigin;

void Fsr411CachePostColor(uint lane, uint2 outputOrigin, float exposure)
{
    // Keep this conservative origin separate from lr_f's half-pixel
    // expression: sharing that expression can change FMA contraction.
    const int2 origin = int2(floor(float2(outputOrigin) * inv_scale + jitter)) - 2;
    if (lane == 0u)
        Fsr411PostColorOrigin = origin;
    if (all(inv_scale <= 1.0f))
    {
        [loop] for (uint index = lane; index < FSR411_POST_CACHE_SIDE * FSR411_POST_CACHE_SIDE; index += 256u)
        {
            int2 p = origin + int2(index % FSR411_POST_CACHE_SIDE, index / FSR411_POST_CACHE_SIDE);
            int2 pc = clamp(p, int2(0, 0), int2((int)width_lr - 1, (int)height_lr - 1));
            // The allocation keeps the native-scale footprint, but upscaling
            // uses a smaller rectangle: e.g. 13x13 instead of 20x20 near 2x.
            // The four-texel margin covers rounding and jitter at tile edges.
            if (all(uint2(index % FSR411_POST_CACHE_SIDE, index / FSR411_POST_CACHE_SIDE) <
                        uint2(ceil(inv_scale * 16.0f)) + 4u))
                Fsr411PostColor[index] = ApplyMuLaw411(r_input_color[pc], exposure);
        }
    }
    // All lanes, including those outside a partial output tile, reach this.
    GroupMemoryBarrierWithGroupSync();
}
"""


EXPOSURE = """#if FFX_MLSR_COLORSPACE == FFX_MLSR_COLORSPACE_LINEAR
    float exposure = r_auto_exposure_texture[int2(0, 0)].r / preExposure;
#else
    float exposure = 1.0f;
#endif
"""


COORDINATES = """    uint gx = GID.x, gy = GID.y;
    uint lane = GTID.x;
    uint x = (gx << 4) | (lane & 1u) | (((lane >> 1) & 14u));
    uint y = (gy << 4) + ((((lane >> 5) << 1) | ((lane >> 1) & 1u)));
    int ix = (int)x, iy = (int)y;
    int w2 = (int)width / 2, h2 = (int)height / 2;
    if (ix >= w2 || iy >= h2)
        return;
"""


def apply_post_optimization(source):
    """Return generated exact/lossy POST with 16x16 output dispatch geometry.

    Source shapes must match the pinned common or Ultra Performance import.
    The original three taps per axis, MuLaw expression, CT2D phase selection,
    float32 normalization, blend, recurrence and output conversion remain.
    """
    if source.startswith(MARKER):
        return source
    if "fsr411-generated-post-color-cache:" in source:
        raise ValueError("POST was already transformed with another cache policy")
    if "    uint4 externalCnb = ScratchBuffer.Load4(" not in source or "    int acc[16];" in source:
        raise ValueError("POST color caching requires the native external CNB cache")
    source = _float32_ct2d(source)

    # Move the unchanged exposure load above the first possible return so
    # every lane participates in loading and synchronizing the shared tile.
    source = _replace(source, EXPOSURE, "")
    coordinates = (
        EXPOSURE +
        "    uint2 outputPixel = GID.xy * 16u + uint2(GTID.x & 15u, GTID.x >> 4u);\n"
        "    Fsr411CachePostColor(GTID.x, GID.xy * 16u, exposure);\n"
        "    if (outputPixel.x >= width || outputPixel.y >= height)\n"
        "        return;\n"
        "    int ix = int(outputPixel.x >> 1u), iy = int(outputPixel.y >> 1u);\n"
        "    int w2 = (int)width / 2, h2 = (int)height / 2;\n"
        "    uint pk = (outputPixel.x & 1u) | ((outputPixel.y & 1u) << 1u);\n"
    )
    source = _replace(source, COORDINATES, coordinates)
    source = _replace(source, "[numthreads(256, 1, 1)]", CT2D_HELPER + CACHE_HELPERS + "\n[numthreads(256, 1, 1)]")
    source = source.replace("    // ---- CT2D 2x2 + epilogue per output px ----\n", "")
    source = _replace(source, "    [unroll] for (uint pk = 0; pk < 4; pk++)\n",
                      "    // ---- CT2D for this output pixel's phase + original epilogue ----\n")

    # Cache coordinates use the original unclamped footprint. Only loading
    # the shared value clamps to the actual render viewport, exactly as each
    # direct tap did. The Gaussian still uses its original unclamped p.
    source = _replace(source, "        int2 lr_i = (int2)round(lr_f);",
                      "        int2 lr_i = (int2)round(lr_f);\n"
                      "        const int2 cachedOrigin = lr_i - 1 - Fsr411PostColorOrigin;\n"
                      "        const bool cachedFilter = all(inv_scale <= 1.0f) && all(cachedOrigin >= 0)\n"
                      "            && all(cachedOrigin < min(int2(ceil(inv_scale * 16.0f)) + 4, int2(20,20)) - 2);\n")
    cache_sample = """                float3 currentSample;
                if (cachedFilter)
                    currentSample = Fsr411PostColor[(cachedOrigin.y + dy) * FSR411_POST_CACHE_SIDE + cachedOrigin.x + dx];
                else
                    currentSample = ApplyMuLaw411(r_input_color[pc], exposure);"""
    direct = "                up += ApplyMuLaw411(r_input_color[pc], exposure) * g;"
    lossy = "                float3 currentSample = ApplyMuLaw411(r_input_color[pc], exposure);"
    if direct in source:
        source = _replace(source, direct, cache_sample + "\n                up += currentSample * g;")
    else:
        source = _replace(source, lossy, cache_sample)
    return MARKER + source
