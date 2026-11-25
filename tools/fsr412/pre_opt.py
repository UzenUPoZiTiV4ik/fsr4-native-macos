# SPDX-License-Identifier: MIT
"""Specializations of the imported, prepared-color Lite PRE and PRE_REUSE."""
import re

from fsr412.optimization import closing


PRE_TILE_SIDE = 16
_FRAME = "MLSR_Optimized_Constants"
_CURRENT = "r_auto_exposure_texture.read(uint2(uint2(0u)), 0u).x"
_PREVIOUS = "r_auto_exposure_texture.read(uint2(uint2(1u, 0u)), 0u).x"
_INPUT = "r_input_exposure.read(uint2(uint2(0u)), 0u).x"
_COLOR = re.compile(
    r"r_prepared_color\.read\(uint2\(uint2\(clamp\(int2\(uint2\((_\d+), (_\d+)\)\), int2\(0\), (_\d+)\)\)\), 0u\)\.xyz")
_DEPTH = re.compile(r"r_depth\.read\(uint2\(uint2\((_\d+)\)\), 0u\)")


def pre_exposure(text, skip_roundtrip):
    """Load each exposure once; only approximate the round-trip in lossy modes."""
    current = re.search(r"^        float (_\d+) = [^;\n]+ / MLSR_Optimized_Constants\.preExposure;", text, re.M)
    if not current or text.count(_PREVIOUS) != 1:
        raise ValueError("Lite PRE exposure layout changed")
    if text.count(_INPUT) == 2:
        start = text.index("        if (" + _INPUT)
        text = (text[:start] + f"        const float fsr412InputExposure = {_INPUT};\n" +
                text[start:].replace(_INPUT, "fsr412InputExposure"))
    elif text.count(_CURRENT) == 1 and _INPUT not in text:
        # Gather the red component at the seam of the 2x1 R32Float texture.
        # With edge clamp, the lower and upper rows are identical: x=current,
        # y=previous. Unlike a linear sample this has no rounding or blending.
        load = ("        constexpr sampler fsr412ExposureSampler(coord::normalized, address::clamp_to_edge);\n"
                "        const float2 fsr412Exposures = r_auto_exposure_texture.gather(\n"
                "            fsr412ExposureSampler, float2(0.5)).xy;\n")
        text = text[:current.start()] + load + text[current.start():].replace(_CURRENT, "fsr412Exposures.x").replace(_PREVIOUS, "fsr412Exposures.y")
    else:
        raise ValueError("Lite PRE current exposure changed")
    if not skip_roundtrip:
        return text

    start = text.index("precise::max(log(")
    end = closing(text, text.index("(", start)) + 1
    expr = text[start:end]
    history = re.search(r"exp\((precise::max\(_\d+, float3\(0\.0\)\)) \* 8\.51788043975830078125\)", expr)
    previous = (_PREVIOUS if _PREVIOUS in expr else "fsr412Exposures.y") + f" / {_FRAME}.previous_preExposure"
    if not history or previous not in expr or f" * {current[1]}) * 150.0" not in expr:
        raise ValueError("Lite PRE history MuLaw changed")
    line = text.rindex("\n", 0, start) + 1
    slow = expr.replace(previous, "fsr412PreviousExposure")
    return (text[:line] + f"            const float fsr412PreviousExposure = {previous};\n"
            f"            const bool fsr412Unchanged = fsr412PreviousExposure == {current[1]} && "
            "fsr412PreviousExposure > 0.0 && isfinite(fsr412PreviousExposure);\n" + text[line:start] +
            f"(fsr412Unchanged ? {history[1]} : {slow})" + text[end:])


def pre_tile(text):
    """One texel per lane, no divergent barriers, and no per-tap bounds tests.

    With T the output group origin, v the render/output ratio, start the tile
    at floor(T*v)-2. Relative to floor(T*v), the four color taps are
    floor(frac(T*v) + (o+0.5)*v - 0.5 + jitter) + {0,1}, o in [0,15].
    Depth rounds frac(T*v) + (o+0.5)*v - 0.5, o in [-1,16], to nearest.
    For |jitter|<=1 and v<=(16-3)/18 both ranges are in [-2,13] with room for
    fp32 error even at the largest Lite tier. Thus all indices are [0,15].
    Keep depth's *unclamped* uint2 texture read, including at negative coords.
    The optional extra clamped LossyRecord load is not a dilation tap.
    """
    taps = list(_COLOR.finditer(text))
    if len(taps) != 4 or len({tap[3] for tap in taps}) != 1:
        raise ValueError("Lite PRE color taps changed")
    header = re.search(r"^kernel void [^\n]*\bgl_LocalInvocationID \[\[thread_position_in_threadgroup\]\][^\n]*\)\n\{\n", text, re.M)
    if not header or "gl_WorkGroupID.x * 16u, gl_WorkGroupID.y * 16u" not in text:
        raise ValueError("Lite PRE group layout changed")
    side = PRE_TILE_SIDE
    fill = f"""    threadgroup packed_float3 fsr412ColorTile[{side * side}];
    const uint fsr412LaneIndex = gl_LocalInvocationID.x;
    const int2 fsr412Origin = int2(floor(float2(gl_WorkGroupID.xy * 16u) * {_FRAME}.inv_scale)) - 2;
    const int2 fsr412Coord = fsr412Origin + int2(fsr412LaneIndex % {side}u, fsr412LaneIndex / {side}u);
    const int2 fsr412Max = int2({_FRAME}.width_lr, {_FRAME}.height_lr) - 1;
    fsr412ColorTile[fsr412LaneIndex] = r_prepared_color.read(uint2(clamp(fsr412Coord, int2(0), fsr412Max)), 0u).xyz;
"""
    if "texture2d<float> r_depth" in text:
        reads = _DEPTH.findall(text)
        if len(set(reads)) != 9:
            raise ValueError("Lite PRE depth dilation changed")
        fill += f"""    threadgroup float fsr412DepthTile[{side * side}];
    fsr412DepthTile[fsr412LaneIndex] = r_depth.read(uint2(fsr412Coord), 0u).x;
"""
        text = _DEPTH.sub(lambda tap: f"float4(fsr412DepthTile[fsr412PreIndex({tap[1]}, fsr412Origin)])", text)
    fill += "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    text = _COLOR.sub(lambda tap: f"float3(fsr412ColorTile[fsr412PreIndex(int2({tap[1]}, {tap[2]}), fsr412Origin)])", text)
    helper = ("static inline uint fsr412PreIndex(int2 coord, int2 origin) {\n"
              f"    const int2 cell = coord - origin;\n    return uint(cell.y * {side} + cell.x);\n}}\n\n")
    return text[:header.start()] + helper + text[header.start():header.end()] + fill + text[header.end():]


def register_pre_tile(kernels, out, stage, run):
    info = kernels[stage]
    if info["wgsize"] != [256, 1, 1]:
        raise ValueError("Lite PRE requires 256 threads")
    name = stage + "_tile"
    base = out / name
    source = base.with_suffix(".metal")
    source.write_text(pre_tile((out / stage).with_suffix(".metal").read_text()))
    air = base.with_suffix(".air")
    lib = base.with_suffix(".metallib")
    run(["xcrun", "metal", "-std=macos-metal2.3", "-O3", "-c", source, "-o", air])
    run(["xcrun", "metallib", air, "-o", lib])
    kernels[name] = dict(info, stage=name, library=lib.name, tile_side=PRE_TILE_SIDE)
