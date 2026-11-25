"""Structural Metal optimizations for the imported AMD INT8 Lite graph.

The neural weights and temporal/filter expressions remain those of the
imported model. Transforms reject unknown source shapes instead of guessing.
"""
from __future__ import annotations
import re
try:
    from common.hlsl_parse import balanced_end
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from common.hlsl_parse import balanced_end

REVISION = 3

_CT2D_PRODUCT = re.compile(
    r"reduce_add\(int4\(as_type<char4>\((_\d+)\)\) \* int4\(as_type<char4>\((_\d+)\[([^\]]+)\]\)\)\)")


def ct2d_float_dot(text: str) -> str:
    """Evaluate POST's INT8 CT2D products as float dots against a float table.

    Each product is four int8 x int8 terms, at most 4 * 128 * 128 = 2^16 in
    magnitude, so every float32 product and partial sum is an exact integer
    and int(dot) equals reduce_add for any summation order. The constant
    weight table is widened once at compile time instead of per lane.
    """
    tables = sorted({match[2] for match in _CT2D_PRODUCT.finditer(text)})
    if len(tables) != 1:
        raise ValueError(f"Lite POST CT2D table count changed: {tables}")
    name = tables[0]
    declaration = re.search(r"constant spvUnsafeArray<uint, (\d+)> %s = spvUnsafeArray<uint, \d+>\(\{ ([^}]*) \}\);"
                            % re.escape(name), text)
    if not declaration:
        raise ValueError("Lite POST CT2D table declaration not found")

    def signed(byte):
        return byte - 256 if byte > 127 else byte

    words = [int(word.rstrip("u")) for word in declaration[2].split(", ")]
    if len(words) != int(declaration[1]):
        raise ValueError("Lite POST CT2D table size changed")
    values = ", ".join("float4(%d, %d, %d, %d)" % tuple(signed((word >> (8 * i)) & 255) for i in range(4))
                       for word in words)
    widened = "\nconstant spvUnsafeArray<float4, %d> %sF = spvUnsafeArray<float4, %d>({ %s });" % (
        len(words), name, len(words), values)
    text = text[:declaration.end()] + widened + text[declaration.end():]
    text, count = _CT2D_PRODUCT.subn(
        lambda match: "int(dot(float4(as_type<char4>(%s)), %sF[%s]))" % (match[1], match[2], match[3]), text)
    if count != 32:
        raise ValueError(f"Lite POST CT2D product count changed: {count} != 32")
    return text


POST_TILE_SIDE = 28
_TAP_READ = re.compile(
    r"r_prepared_color\.read\(uint2\(uint2\(clamp\(int2\(uint2\((_\d+), (_\d+)\)\), int2\(0\), (_\d+)\)\)\), 0u\)\.xyz")
_CELL = re.compile(r"^( +)uint2 (_\d+) = uint2\(\(\(_\d+ % 8u\) << 1u\) \| \(gl_LocalInvocationID\.x & 1u\), "
                   r"\(\(_\d+ / 8u\) << 1u\) \| \(\(gl_LocalInvocationID\.x & 3u\) >> 1u\)\) \+ "
                   r"\(gl_WorkGroupID\.xy \* uint2\(16u\)\);\n", re.M)


def post_color_tile(text: str) -> str:
    """Specialize POST to read its nine prepared-colour taps from a tile.

    Returns a second POST kernel ("post_tile"); the host selects it only
    when ceil(32 * inv_scale) + 6 <= POST_TILE_SIDE on both axes, i.e. for
    render/output ratios up to 0.6875 (Quality and below). A group covers
    16x16 cells = 32x32 output pixels. Every lane copies part of the clamped
    prepared-colour footprint into threadgroup memory before the first early
    exit; the taps then read the tile unconditionally.

    Coverage: with T the group's first output pixel, the tile starts at
    floor(T * inv + jitter) - 3. A tap is rint(p) - 1 .. rint(p) + 1 with
    p = (o + 0.5) * inv + jitter - 0.5, T <= o <= T + 31, so its cell lies
    in [1, 31.5 * inv + 5] and below ceil(32 * inv) + 6. The tile holds the
    texel at clamp(origin + cell), the same clamp the texture path applies,
    so every tap reads identical float3 values.

    The original statements stay branch-free: a per-tap or per-statement
    branch between tile and texture let the compiler contract the weighted
    sum differently, which changed output bits. The cell index is clamped to
    the tile for memory safety only; the host check makes it a no-op.
    """
    reads = list(_TAP_READ.finditer(text))
    if len(reads) != 9 or len({match[3] for match in reads}) != 1:
        raise ValueError("Lite POST colour taps changed")
    cell = _CELL.search(text)
    header = re.search(
        r"kernel\s+void\s+\w+\([^\n]*uint3\s+gl_LocalInvocationID\s*\[\[thread_position_in_threadgroup\]\][^\n]*\)\n\{\n",
        text,
    )
    if (not cell or not header or "fsr412PostTile" in text or
            not re.search(rf"^{re.escape(cell[1])}if \(any\({re.escape(cell[2])} >= _\d+\)\)\n", text[cell.end():], re.M)):
        raise ValueError("Lite POST group mapping changed")
    indent = cell[1]
    side = POST_TILE_SIDE
    fill = (f"{indent}// Clamped prepared colour of this group's 32x32-pixel footprint.\n"
            f"{indent}const int2 fsr412TileOrigin = int2(floor(float2(gl_WorkGroupID.xy * uint2(32u)) * "
            f"MLSR_Optimized_Constants.inv_scale + MLSR_Optimized_Constants.jitter)) - int2(3);\n"
            f"{indent}const int2 fsr412TileExtent = min(int2(ceil(MLSR_Optimized_Constants.inv_scale * 32.0)) + int2(6), int2({side}));\n"
            f"{indent}const int2 fsr412TileMax = int2(int(MLSR_Optimized_Constants.width_lr - 1u), "
            f"int(MLSR_Optimized_Constants.height_lr - 1u));\n"
            f"{indent}for (uint fsr412Index = gl_LocalInvocationID.x; fsr412Index < {side * side}u; fsr412Index += 256u)\n"
            f"{indent}{{\n"
            f"{indent}    const int2 fsr412Cell = int2(int(fsr412Index % {side}u), int(fsr412Index / {side}u));\n"
            f"{indent}    if (all(fsr412Cell < fsr412TileExtent))\n"
            f"{indent}        fsr412PostTile[fsr412Index] = r_prepared_color.read(uint2(clamp(fsr412TileOrigin + fsr412Cell, "
            f"int2(0), fsr412TileMax)), 0u).xyz;\n"
            f"{indent}}}\n"
            f"{indent}threadgroup_barrier(mem_flags::mem_threadgroup);\n")
    text = text[:cell.end()] + fill + text[cell.end():]
    header = re.search(
        r"kernel\s+void\s+\w+\([^\n]*uint3\s+gl_LocalInvocationID\s*\[\[thread_position_in_threadgroup\]\][^\n]*\)\n\{\n",
        text,
    )
    text = text[:header.end()] + f"    threadgroup float3 fsr412PostTile[{side * side}];\n" + text[header.end():]
    helper = ("static inline uint fsr412_tile_index(int2 origin, uint x, uint y)\n{\n"
              f"    const int2 cell = clamp(int2(uint2(x, y)) - origin, int2(0), int2({side - 1}));\n"
              f"    return uint(cell.y) * {side}u + uint(cell.x);\n}}\n")
    kernel = text.index("kernel void ")
    text = text[:kernel] + helper + "\n" + text[kernel:]
    text, count = _TAP_READ.subn(
        lambda tap: f"fsr412PostTile[fsr412_tile_index(fsr412TileOrigin, {tap[1]}, {tap[2]})]", text)
    if count != 9:
        raise ValueError("Lite POST colour tap rewrite failed")
    return text


def closing(text, start, left="(", right=")"):
    assert text[start] == left
    return balanced_end(text, start, left, right,
                        "unbalanced imported Metal expression")


def input_color_cache(text: str, expected: int) -> tuple[str, str]:
    """Reuse the same float32 MuLaw result in PRE and POST within one frame."""
    replacements = []
    formula = None
    for match in re.finditer(r"precise::max\(log\(", text):
        end = closing(text, text.index("(", match.start())) + 1
        expression = text[match.start():end]
        if "r_input_color.read(" not in expression:
            continue
        read = expression.index("r_input_color.read(")
        rstart = expression.index("(", read)
        rend = closing(expression, rstart)
        arguments = expression[rstart + 1:rend]
        if not arguments.endswith(", 0u"):
            raise ValueError("Lite color load mip changed")
        position = arguments[:-4]
        sample = expression[:read] + "r_input_color.read(pixel, 0u)" + expression[rend + 1:]
        # The only nonliteral value outside the source texel is the exposure.
        sample = re.sub(r"(?<=\* )_\d+(?=\) \* 150.0)", "exposure", sample)
        if formula is not None and formula != sample:
            raise ValueError("Lite input color transforms differ")
        formula = sample
        replacements.append((match.start(), end, f"r_prepared_color.read({position}, 0u).xyz"))
    if len(replacements) != expected:
        raise ValueError(f"Lite color transform count changed: {len(replacements)} != {expected}")
    for start, end, replacement in reversed(replacements):
        text = text[:start] + replacement + text[end:]
    group = "uint3 gl_WorkGroupID [[threadgroup_position_in_grid]]"
    if text.count(group) != 1 or "[[texture(8)]]" in text:
        raise ValueError("Lite color cache binding/dispatch signature changed")
    text = text.replace(group, "texture2d<float> r_prepared_color [[texture(8)]], " + group, 1)
    return text, formula


def color_cache_source(text: str, formula: str) -> str:
    constants = re.search(r"struct type_MLSR_Optimized_Constants\n\{.*?\n\};", text, re.S)
    if not constants:
        raise ValueError("Lite frame constants not found")
    return """#include <metal_stdlib>
using namespace metal;
""" + constants[0] + """
kernel void fsr412_prepare_color(constant type_MLSR_Optimized_Constants& MLSR_Optimized_Constants [[buffer(0)]],
    texture2d<float> r_input_color [[texture(0)]], texture2d<float> r_auto_exposure_texture [[texture(1)]],
    texture2d<float, access::write> rw_prepared_color [[texture(2)]], uint2 pixel [[thread_position_in_grid]]) {
    if (any(pixel >= uint2(MLSR_Optimized_Constants.width_lr, MLSR_Optimized_Constants.height_lr))) return;
    float exposure = r_auto_exposure_texture.read(uint2(0u), 0u).x / MLSR_Optimized_Constants.preExposure;
    float3 value = """ + formula + ";\n    rw_prepared_color.write(float4(value, 0.0), pixel);\n}\n"


def pixelization_fix(text: str) -> str:
    """Widen POST's current-frame filter where the network rejects history.

    Lite predicts a narrower Gaussian than the 4.1 network. Where history is
    discarded (moving hair, disocclusion) that kernel nearly reproduces the
    half-resolution input, which shows 2x2 blocks. Scaling both precision terms
    by mix(floor, 1, history) leaves pixels that keep history unchanged.
    The host passes floor in the otherwise unused constant `pad` each frame
    (Options::pixelFixFloor), so the panel can change it without a rebuild.
    """
    sx = re.search(r"^( +)float (_\d+) = 2\.0 / \(1\.0 \+ exp\(-(_\d+)\.y\)\);\n", text, re.M)
    sy = sx and re.search(r"^ +float (_\d+) = 2\.0 / \(1\.0 \+ exp\(-%s\.z\)\);\n" % sx[3], text, re.M)
    history = sx and re.search(r"= 1\.0 / \(1\.0 \+ exp\(-%s\.w\)\);" % sx[3], text)
    if not sx or not sy or not history or sy.start() != sx.end():
        raise ValueError("Lite POST filter parameters not found")
    indent, params = sx[1], sx[3]
    code = (f"{indent}// Pixelization fix: wider current-frame filter where history is rejected.\n"
            f"{indent}const float fsr412Widen = sqrt(mix(MLSR_Optimized_Constants.pad, 1.0, 1.0 / (1.0 + exp(-{params}.w))));\n"
            f"{indent}float {sx[2]} = (2.0 / (1.0 + exp(-{params}.y))) * fsr412Widen;\n"
            f"{indent}float {sy[1]} = (2.0 / (1.0 + exp(-{params}.z))) * fsr412Widen;\n")
    return text[:sx.start()] + code + text[sy.end():]


_PRE_QUAD_TAIL = re.compile(
    r"^( +)uint2 (_\d+) = (_\d+) % uint2\(2u\);\n"
    r"\1uint (_\d+) = \(16u \* \2\.x\) \+ \(512u \* \2\.y\);\n"
    r"(.*?)"
    r"\1if \(all\(\2 == uint2\(0u\)\)\)\n"
    r"\1\{\n"
    r"(.*?)"
    r"\1    int2 (_\d+) = \(int2\(gl_WorkGroupID\.xy\) \* int2\(8\)\) \+ int2\(\3 / uint2\(2u\)\);\n"
    r"\1    uint4 (_\d+) = uint4\(uint\(\7\.x\), uint\(\7\.y\), 0u, 0u\) \+ uint4\(1u, 1u, 0u, 0u\);\n"
    r"\1    uint (_\d+) = \(\(\(\(16u \* \8\.x\) \+ \((\d+)u \* \8\.y\)\) \+ \8\.z\) \+ \(\d+u \* \8\.w\)\) >> 2u;\n"
    r"\1    ScratchBuffer\._m0\[\9\] = [^\n]+;\n"
    r"\1    ScratchBuffer\._m0\[\9 \+ 1u\] = [^\n]+;\n"
    r"\1    ScratchBuffer\._m0\[\9 \+ 2u\] = [^\n]+;\n"
    r"\1    ScratchBuffer\._m0\[\9 \+ 3u\] = [^\n]+;\n"
    r"\1\}",
    re.M | re.S,
)


_PRE_QUAD_HELPER = """static inline float fsr412_tap_dot(device const uint *init, uint wordOffset, uint4 inPacked)
{
    const uint4 w = *(device const uint4 *)(init + wordOffset);
    const float2 p0 = float2(as_type<half2>(w.x) * as_type<half2>(inPacked.x));
    const float2 p1 = float2(as_type<half2>(w.y) * as_type<half2>(inPacked.y));
    const float2 p2 = float2(as_type<half2>(w.z) * as_type<half2>(inPacked.z));
    const float2 p3 = float2(as_type<half2>(w.w) * as_type<half2>(inPacked.w));
    // Match LLVM's flattened fast-math chain inside each tap so AGX emits the
    // exact same FMA sequence as the imported SPIR-V shader.
    return ((((((p0.x + p0.y) + p1.y) + p1.x) + p2.y) + p2.x) + p3.y) + p3.x;
}

static inline float fsr412_quad_reduce(float t0, float t1, float t2, float t3)
{
    #pragma clang fp reassociate(off)
    const float s01 = t0 + t1;
    const float s23 = t2 + t3;
    return s01 + s23;
}

static inline uint fsr412_quad_group(device const uint *init, uint group, uint4 in0, uint4 in1, uint4 in2, uint4 in3)
{
    const uint c0 = group * 4u;
    const float s0 = fsr412_quad_reduce(
        fsr412_tap_dot(init, c0 * 8u, in0),
        fsr412_tap_dot(init, c0 * 8u + 4u, in1),
        fsr412_tap_dot(init, c0 * 8u + 128u, in2),
        fsr412_tap_dot(init, c0 * 8u + 132u, in3));
    const float s1 = fsr412_quad_reduce(
        fsr412_tap_dot(init, (c0 + 1u) * 8u, in0),
        fsr412_tap_dot(init, (c0 + 1u) * 8u + 4u, in1),
        fsr412_tap_dot(init, (c0 + 1u) * 8u + 128u, in2),
        fsr412_tap_dot(init, (c0 + 1u) * 8u + 132u, in3));
    const float s2 = fsr412_quad_reduce(
        fsr412_tap_dot(init, (c0 + 2u) * 8u, in0),
        fsr412_tap_dot(init, (c0 + 2u) * 8u + 4u, in1),
        fsr412_tap_dot(init, (c0 + 2u) * 8u + 128u, in2),
        fsr412_tap_dot(init, (c0 + 2u) * 8u + 132u, in3));
    const float s3 = fsr412_quad_reduce(
        fsr412_tap_dot(init, (c0 + 3u) * 8u, in0),
        fsr412_tap_dot(init, (c0 + 3u) * 8u + 4u, in1),
        fsr412_tap_dot(init, (c0 + 3u) * 8u + 128u, in2),
        fsr412_tap_dot(init, (c0 + 3u) * 8u + 132u, in3));
    const float4 bias = as_type<float4>(*(device const uint4 *)(init + 256u + group * 4u));
    const int4 quantized = int4(rint((float4(s0, s1, s2, s3) + bias) * 64.0));
    return as_type<uint>(char4(clamp(quantized, int4(-128), int4(127))));
}
"""


def pre_quad_routing(text: str) -> str:
    """Route the 2x2 quad's half inputs, then compute 4 output channels per lane.

    The imported SPIR-V PRE tail has every lane in a 2x2 quad evaluate all 16
    downscale channels (64 scalar initializer reads and 32 quad swaps per lane),
    then discards three lanes' work. Exchanging the 8 packed half inputs once
    via three uint4 quad shuffles lets each lane compute 4 channels with four
    aligned uint4 weight loads and one coalesced ScratchBuffer store.

    Keeping the four routed inputs in registers (no private array) and
    preserving the (t0+t1)+(t2+t3) reduction order yields bit-exact ScratchBuffer
    output on both even and odd resolutions.
    """
    if "fsr412_quad_group" in text:
        raise ValueError("Lite PRE quad routing already applied")
    matches = list(_PRE_QUAD_TAIL.finditer(text))
    if len(matches) != 1:
        raise ValueError(f"Lite PRE neural tail match count changed: {len(matches)}")
    match = matches[0]
    indent = match[1]
    local_coord = match[3]
    offset_var = match[4]
    conv_body = match[5]
    store_body = match[6]
    row_stride = int(match[10])

    if conv_body.count("spvQuadSwap(") != 32:
        raise ValueError("Lite PRE quad swap count changed")
    in_xy = re.search(r"half2 _\d+ = (_\d+)\.xy;", conv_body)
    in_zw = in_xy and re.search(r"half2 _\d+ = %s\.zw;" % re.escape(in_xy[1]), conv_body)
    in2_xy = in_xy and re.search(r"half2 _\d+ = (?!%s\b)(_\d+)\.xy;" % re.escape(in_xy[1]), conv_body)
    in2_zw = in2_xy and re.search(r"half2 _\d+ = %s\.zw;" % re.escape(in2_xy[1]), conv_body)
    if not (in_xy and in_zw and in2_xy and in2_zw):
        raise ValueError("Lite PRE quad input vectors changed")
    in0, in1 = in_xy[1], in2_xy[1]
    for ch in range(1, 16):
        if f"({offset_var} + {ch * 32}u) >> 2u;" not in conv_body:
            raise ValueError(f"Lite PRE channel {ch} weight offset changed")
    for bias_word in range(256, 272):
        if f"InitializerBuffer._m0[{bias_word}u]" not in store_body:
            raise ValueError(f"Lite PRE bias word {bias_word} changed")

    code = f"""{indent}const uint fsr412Lane = gl_LocalInvocationID.x & 3u;
{indent}const uint4 fsr412Local = uint4(as_type<uint>({in0}.xy), as_type<uint>({in0}.zw), as_type<uint>({in1}.xy), as_type<uint>({in1}.zw));
{indent}const uint4 fsr412AcrossX = quad_shuffle_xor(fsr412Local, 1u);
{indent}const uint4 fsr412AcrossY = quad_shuffle_xor(fsr412Local, 2u);
{indent}const uint4 fsr412AcrossD = quad_shuffle_xor(fsr412Local, 3u);
{indent}const uint4 fsr412In0 = fsr412Lane == 0u ? fsr412Local : fsr412Lane == 1u ? fsr412AcrossX : fsr412Lane == 2u ? fsr412AcrossY : fsr412AcrossD;
{indent}const uint4 fsr412In1 = fsr412Lane == 1u ? fsr412Local : fsr412Lane == 0u ? fsr412AcrossX : fsr412Lane == 3u ? fsr412AcrossY : fsr412AcrossD;
{indent}const uint4 fsr412In2 = fsr412Lane == 2u ? fsr412Local : fsr412Lane == 3u ? fsr412AcrossX : fsr412Lane == 0u ? fsr412AcrossY : fsr412AcrossD;
{indent}const uint4 fsr412In3 = fsr412Lane == 3u ? fsr412Local : fsr412Lane == 2u ? fsr412AcrossX : fsr412Lane == 1u ? fsr412AcrossY : fsr412AcrossD;
{indent}const uint2 fsr412QuadBase = (gl_WorkGroupID.xy * uint2(16u)) + ({local_coord} & ~uint2(1u));
{indent}const bool fsr412QuadRight = fsr412QuadBase.x + 1u < MLSR_Optimized_Constants.width;
{indent}const bool fsr412QuadBottom = fsr412QuadBase.y + 1u < MLSR_Optimized_Constants.height;
{indent}const uint2 fsr412Cell = (gl_WorkGroupID.xy * uint2(8u)) + ({local_coord} >> 1u) + uint2(1u);
{indent}const uint fsr412WordBase = ((16u * fsr412Cell.x) + ({row_stride}u * fsr412Cell.y)) >> 2u;
{indent}if (fsr412QuadRight && fsr412QuadBottom)
{indent}{{
{indent}    ScratchBuffer._m0[fsr412WordBase + fsr412Lane] = fsr412_quad_group(
{indent}        InitializerBuffer._m0, fsr412Lane, fsr412In0, fsr412In1, fsr412In2, fsr412In3);
{indent}}}
{indent}else if (fsr412Lane == 0u)
{indent}{{
{indent}    const uint4 fsr412Edge1 = fsr412QuadRight ? fsr412AcrossX : uint4(0u);
{indent}    const uint4 fsr412Edge2 = fsr412QuadBottom ? fsr412AcrossY : uint4(0u);
{indent}    ScratchBuffer._m0[fsr412WordBase] = fsr412_quad_group(InitializerBuffer._m0, 0u, fsr412Local, fsr412Edge1, fsr412Edge2, uint4(0u));
{indent}    ScratchBuffer._m0[fsr412WordBase + 1u] = fsr412_quad_group(InitializerBuffer._m0, 1u, fsr412Local, fsr412Edge1, fsr412Edge2, uint4(0u));
{indent}    ScratchBuffer._m0[fsr412WordBase + 2u] = fsr412_quad_group(InitializerBuffer._m0, 2u, fsr412Local, fsr412Edge1, fsr412Edge2, uint4(0u));
{indent}    ScratchBuffer._m0[fsr412WordBase + 3u] = fsr412_quad_group(InitializerBuffer._m0, 3u, fsr412Local, fsr412Edge1, fsr412Edge2, uint4(0u));
{indent}}}"""
    text = text[:match.start()] + code + text[match.end():]
    kernel = text.index("kernel void ")
    return text[:kernel] + _PRE_QUAD_HELPER + "\n" + text[kernel:]
