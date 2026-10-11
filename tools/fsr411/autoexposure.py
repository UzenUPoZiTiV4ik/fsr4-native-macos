#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Split the imported SPD autoexposure into two portable Metal dispatches.

The tile pass retains the first six SPD reductions. After a host texture
barrier, one 256-thread group executes the original remaining SPD calls.
This removes reliance on a cross-threadgroup globallycoherent texture and
relaxed atomic publication, while preserving the existing float32 tile
store/load boundary and reduction arithmetic.

ABI stays AutoExposureSPDConstants at b0, with its 40-byte validSize
extension. The host supplies mips in [1, 12], dispatches ae_tiles over
ceil(validSize/64), and ae_finalize as a single group when mips > 6.
The original ae_ours.hlsl remains available as an independent oracle.
"""


THREADS_PER_GROUP = 256
TILE_SIDE = 64
MARKER = "// fsr411-generated-autoexposure: "


def _replace(source, before, after):
    if source.count(before) != 1:
        raise ValueError("autoexposure transform expected one expression: " + before[:120])
    return source.replace(before, after)


ORIGINAL_LOAD = """FfxFloat32x4 SpdLoad(FfxInt32x2 tex, FfxUInt32 slice)
{
    return FfxFloat32x4(rw_autoexp_mip_5[tex], 0, 0, 0);
}"""


BOUNDED_LOAD = """FfxFloat32x4 SpdLoad(FfxInt32x2 tex, FfxUInt32 slice)
{
    // Reproduce D3D's zero outside the logical tile image, including when
    // its backing allocation is larger than the current render viewport.
    FfxInt32x2 tileSize = FfxInt32x2((validSize + FfxUInt32x2(63u, 63u)) / 64u);
    FfxInt32x2 safeTex = clamp(tex, FfxInt32x2(0, 0), tileSize - 1);
    float loaded = rw_autoexp_mip_5[safeTex];
    bool inRect = all(tex >= 0) && all(tex < tileSize);
    return FfxFloat32x4(inRect ? loaded : 0.0f, 0, 0, 0);
}"""


ORIGINAL_MAIN = """[numthreads(256, 1, 1)]
void main(uint3 WorkGroupId : SV_GroupID, uint LocalThreadIndex : SV_GroupIndex)
{
SpdDownsample(
    FfxUInt32x2(WorkGroupId.xy),
    FfxUInt32(LocalThreadIndex),
    FfxUInt32(MipCount()),
    FfxUInt32(NumWorkGroups()),
    FfxUInt32(WorkGroupId.z),
    FfxUInt32x2(WorkGroupOffset()));
}
"""


TILES_MAIN = """[numthreads(256, 1, 1)]
void main(uint3 WorkGroupId : SV_GroupID, uint LocalThreadIndex : SV_GroupIndex)
{
    // mips remains the original b0 value for SpdStore's final-level test.
    // Call precisely the tile portion of SpdDownsample so no unreachable
    // atomic tail or its buffer/texture bindings survive compilation.
    FfxUInt32x2 sub_xy = ffxRemapForWaveReduction(FfxUInt32(LocalThreadIndex) % 64u);
    FfxUInt32 x = sub_xy.x + 8u * ((FfxUInt32(LocalThreadIndex) >> 6u) % 2u);
    FfxUInt32 y = sub_xy.y + 8u * (FfxUInt32(LocalThreadIndex) >> 7u);
    FfxUInt32x2 tile = FfxUInt32x2(WorkGroupId.xy) + WorkGroupOffset();
    SpdDownsampleMips_0_1(x, y, tile, FfxUInt32(LocalThreadIndex),
                         min(MipCount(), 6u), FfxUInt32(WorkGroupId.z));
    SpdDownsampleNextFour(x, y, tile, FfxUInt32(LocalThreadIndex),
                         2u, min(MipCount(), 6u), FfxUInt32(WorkGroupId.z));
}
"""


FINALIZE_MAIN = """[numthreads(256, 1, 1)]
void main(uint3 WorkGroupId : SV_GroupID, uint LocalThreadIndex : SV_GroupIndex)
{
    if (MipCount() <= 6u)
        return;
    // Identical remap and tail calls to the imported SpdDownsample. Every
    // lane in this single group follows the same mip-count branch.
    FfxUInt32x2 sub_xy = ffxRemapForWaveReduction(FfxUInt32(LocalThreadIndex) % 64u);
    FfxUInt32 x = sub_xy.x + 8u * ((FfxUInt32(LocalThreadIndex) >> 6u) % 2u);
    FfxUInt32 y = sub_xy.y + 8u * (FfxUInt32(LocalThreadIndex) >> 7u);
    SpdDownsampleMips_6_7(x, y, MipCount(), FfxUInt32(WorkGroupId.z));
    SpdDownsampleNextFour(x, y, FfxUInt32x2(0, 0), FfxUInt32(LocalThreadIndex),
                         8u, MipCount(), FfxUInt32(WorkGroupId.z));
}
"""


def split_autoexposure(source):
    """Return {ae_tiles.hlsl, ae_finalize.hlsl} without editing the import.

    The mip-six store also updates exposure: the original else-if silently
    skipped that update for render maxima 64..127. Every other supported mip
    retains the original store conditions and all arithmetic expressions.
    """
    if MARKER in source:
        raise ValueError("autoexposure source has already been split")
    main_header = "[numthreads(256, 1, 1)]\nvoid main(uint3 WorkGroupId : SV_GroupID, uint LocalThreadIndex : SV_GroupIndex)\n"
    if source.count(main_header) != 1:
        raise ValueError("autoexposure source is missing its pinned entry point")
    start = source.index(main_header)
    main = source[start:]
    if "".join(main.split()) != "".join(ORIGINAL_MAIN.split()):
        raise ValueError("autoexposure source has an unrecognized SPD entry body")
    source = source[:start]
    source = _replace(source, ORIGINAL_LOAD, BOUNDED_LOAD)
    source = _replace(source,
                      "    else if (index == MipCount() - 1) { //accumulate on 1x1 level",
                      "    // Mip six must publish its tile and update exposure.\n"
                      "    if (index == MipCount() - 1) { //accumulate on 1x1 level")
    source = _replace(source,
                      "\t\t\trw_auto_exposure_texture[int2(0,0)] = GetExposure(outValue.x);\n"
                      "            rw_auto_exposure_texture[int2(1,0)] = prev_exposre;",
                      "            // Consume the old texel before overwriting it. SPIRV-Cross\n"
                      "            // otherwise sinks the read into the later previous-slot store.\n"
                      "            rw_auto_exposure_texture[int2(1,0)] = prev_exposre;\n"
                      "            rw_auto_exposure_texture[int2(0,0)] = GetExposure(outValue.x);")
    return {
        "ae_tiles.hlsl": MARKER + "tiles\n" + source + TILES_MAIN,
        "ae_finalize.hlsl": MARKER + "finalize\n" + source + FINALIZE_MAIN,
    }
