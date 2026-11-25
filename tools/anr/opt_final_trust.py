"""Specialize the three pinned V.B20 32x8 final image kernels after SPIRV-Cross.

Keep the generated descriptor table, low-band features, color/gain filtering,
output conversion and diagnostic counters. Only replace the hot uniform/tile
operations. SSA names below refer to the pinned package, not the kernel index.
"""
import hashlib
import re
import warnings


_PREFIX = "fg_lite_lowband.slang:final_trust_image_fill_"
# Only rewrite the SSA we inspected. Another SPIRV-Cross version may rename
# IDs without changing their types; a compilable partial rewrite isn't safe.
_KERNEL_HASHES = {
    _PREFIX + "low_r32": "8498522b6220d1733958b385ce0ac79b2c33874eadfc7f8d446d245e5eb2b229",
    _PREFIX + "tex_srgb8_low_r32": "1785eac01150ee011c51855906e78d0de62c22575d448b96ad2b6d182693a4b4",
    _PREFIX + "tex_pq10_low_r32": "c7fd7aab6cfa7defb1d57062fd356c506eae4902d57b1eac57f8899ac2a88d35",
}
_BARRIER = "    threadgroup_barrier(mem_flags::mem_threadgroup);"


_HELPERS = """
struct MotionTap {
    float4 flow;
    float2 trust;
    uint valid;
};

inline MotionTap motionTap(constant spvDescriptorSetBuffer0& args, uint index,
                           bool packed, float scale) {
    MotionTap tap;
    if (packed) {
        tap.flow = float4(as_type<short4>(args.PACKED_FLOW->_m0[index])) * scale;
        tap.trust = float2(args.PACKED_TRUST->_m0[index]);
    } else {
        tap.flow = args.FLOAT_FLOW->_m0[index];
        tap.trust = args.FLOAT_TRUST->_m0[index];
    }
    // Bit tests, not isfinite(): the compiler is in fast-math mode. Even a
    // zero-weight NaN/Inf tap must invalidate the original interpolation.
    bool finite = all((as_type<uint4>(tap.flow) & 0x7f800000u) != 0x7f800000u) &&
                  all((as_type<uint2>(tap.trust) & 0x7f800000u) != 0x7f800000u);
    tap.valid = finite ? ~0u : 0u;
    return tap;
}

// Each axis contributes 1 inside the image and a linear ramp on its 1px
// border. This is the sum of the four in-bounds bilinear tap weights.
inline float warpCoverage(float2 position, float2 extent) {
    float2 weight = saturate(min(position + 1.0f, extent - position));
    return weight.x * weight.y;
}
"""

_STAGE = """
    constant auto& params = *spvDescriptorSet0.PARAMS;
    const uint2 fullSize(params._m21, params._m20);
    const uint2 flowSize(params._m23, params._m22);
    const bool packed = params._m48 != 0u;
    // In the original fmax(NaN, 64) selects 64. Do not depend on a fast
    // intrinsic's NaN behavior. Leave infinities for the tap validity test.
    const uint rawScale = packed ? spvDescriptorSet0.SCALE->_m0[0] : 0u;
    const float bound = (rawScale & 0x7fffffffu) > 0x7f800000u ? 0.0f : as_type<float>(rawScale);
    const float scale = fast::max(2.0f * bound, 64.0f) * (1.0f / 32767.0f);
    const uint linear = gl_LocalInvocationID.y * 32u + gl_LocalInvocationID.x;
    // floor((gid+.5)/2-.5), plus the adjacent bilinear texel. Even interior
    // groups need both halos: [16*gx-1,16*gx+16] x [4*gy-1,4*gy+4].
    const int2 tileBase = int2(gl_WorkGroupID.xy * uint2(16, 4)) - 1;
    const bool staged = all(flowSize != fullSize) &&
                        all(flowSize >= fullSize / 2u) && all(flowSize <= (fullSize + 1u) / 2u);
    threadgroup MotionTap tile[18 * 6];
    if (staged && linear < 18u * 6u) {
        uint2 src = uint2(clamp(tileBase + int2(linear % 18u, linear / 18u),
                                int2(0), int2(flowSize) - 1));
        tile[linear] = motionTap(spvDescriptorSet0, src.y * flowSize.x + src.x, packed, scale);
    }
    // Publish mapping checks with the *existing* low-band staging barrier.
    // The choice of the slow barrier path must be uniform for the whole tile.
    threadgroup uchar rowLayout[8];
    if (simdWidth == 32u) {
        bool sameRow = simd_all(gl_LocalInvocationID.x == lane && gl_LocalInvocationID.y == subgroup);
        if (lane == 0u) rowLayout[subgroup] = sameRow;
    }
"""

_ROW_MASK = """
    uint rowMask = 0;
    bool rows = simdWidth == 32u;
    if (rows) {
        for (uint i = 0; i < 8; ++i) rows &= bool(rowLayout[i]);
    }
    if (rows) {
        rowMask = uint((simd_vote::vote_t)simd_ballot(HOLE));
    } else {
        threadgroup uchar holes[256];
        holes[linear] = HOLE;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (gl_LocalInvocationID.x == 0u) {
            for (uint x = 0; x < 32u; ++x) rowMask |= uint(holes[linear + x]) << x;
        }
    }
"""


def _span(source, start, end, replacement):
    first = source.index(start)
    last = source.index(end, first + len(start))
    return source[:first] + replacement + source[last:]


def optimize_final_trust_msl(entry_key, msl_code):
    if entry_key not in _KERNEL_HASHES:
        return msl_code
    hdr = entry_key == _PREFIX + "low_r32"
    head, kernel = msl_code.split("kernel void ", 1)
    signature_and_body = kernel.partition("(")[2]
    if hashlib.sha256(signature_and_body.encode()).hexdigest() != _KERNEL_HASHES[entry_key]:
        warnings.warn(f"{entry_key}: unfamiliar SPIRV-Cross output; keeping original MSL", stacklevel=2)
        return msl_code
    # Bindings 1..11 contain the read-only buffers/textures/sampler in all
    # three variants. Do not change field names, order, IDs or writable slots.
    head = re.sub(r"(?m)^(    )device (.*\[\[id\((?:[1-9]|10|11)\)\]\];)$", r"\1const device \2", head)
    fields = re.findall(r"(m_\d+) \[\[id\(\d+\)\]\]", head)
    params, _, flow, scale, packed_flow, packed_trust, trust = fields[:7]
    helper = (_HELPERS.replace("PACKED_FLOW", packed_flow).replace("PACKED_TRUST", packed_trust)
              .replace("FLOAT_FLOW", flow).replace("FLOAT_TRUST", trust))
    stage = _STAGE.replace("PARAMS", params).replace("SCALE", scale)
    # SPIRV-Cross uses different SSA IDs for scRGB, whose color taps are in
    # buffers. Both texture variants have identical IDs (only UNORM differs).
    row, lane_zero, hole = (9, 234, 2011) if hdr else (7, 184, 1205)
    kernel = kernel.replace("[[thread_position_in_threadgroup]])", "[[thread_position_in_threadgroup]], "
                            "uint simdWidth [[threads_per_simdgroup]], uint lane [[thread_index_in_simdgroup]], "
                            "uint subgroup [[simdgroup_index_in_threadgroup]])")
    kernel = kernel.replace(f"    threadgroup spvUnsafeArray<uint, 8> _{row};", stage)
    kernel = _span(kernel, f"    if (_{lane_zero})", "    int _", "")

    # Keep the emitter's coordinate clamp and all four (ordered) FMAs. Replace
    # only the four decodes/finite checks and their validity-mask reduction.
    y0, y1, x0, x1 = (448, 449, 471, 472) if hdr else (310, 311, 333, 334)
    index = (475, 476, 478, 479) if hdr else (337, 338, 340, 341)
    trusts = (537, 616, 697, 778) if hdr else (399, 478, 559, 640)
    flows = (538, 617, 698, 779) if hdr else (400, 479, 560, 641)
    fetch = "        MotionTap taps[4];\n        if (staged) {\n"
    fetch += f"            int2 lo = int2(_{x0}, _{y0}) - tileBase;\n"
    fetch += f"            int2 hi = int2(_{x1}, _{y1}) - tileBase;\n"
    for i, (x, y) in enumerate((("lo", "lo"), ("hi", "lo"), ("lo", "hi"), ("hi", "hi"))):
        fetch += f"            taps[{i}] = tile[{y}.y * 18 + {x}.x];\n"
    fetch += "        } else {\n"
    for i, idx in enumerate(index):
        fetch += f"            taps[{i}] = motionTap(spvDescriptorSet0, _{idx}, packed, scale);\n"
    fetch += "        }\n"
    for i in range(4):
        fetch += f"        float2 _{trusts[i]} = taps[{i}].trust;\n        float4 _{flows[i]} = taps[{i}].flow;\n"
    kernel = _span(kernel, f"        float2 _{trusts[0]};", f"        float2 _{810 if hdr else 672} =", fetch)
    kernel = re.sub(r"(uint _(?:886|745) = ).*?( & _(?:12|10)\[_\d+\];)",
                    r"\1taps[0].valid & taps[1].valid & taps[2].valid & taps[3].valid\2", kernel, count=1)

    if hdr:
        # Manual scRGB taps still need their weights, just not the interior sum.
        for result in (1130, 1259):
            kernel = re.sub(fr"_{result} = \(\(.*?;", f"_{result} = 1.0f;", kernel)
    else:
        # Keep the normalized, clamped _1075 hardware gain samples. A 6x3
        # RGBA16F cache covers the footprint, but was slower on M5 and its
        # FP32 mix does not reproduce the sampler's rounding.
        for result, x, y, next_decl in ((949, 861, 864, "float _951"), (1038, 951, 953, "float2 _1043")):
            kernel = _span(kernel, f"        float _{result};", f"        {next_decl}",
                           f"        float _{result} = warpCoverage(float2(_{x}, _{y}), float2(fullSize));\n")
    kernel = kernel.replace("precise::tanh", "fast::tanh").replace("precise::max", "fast::max")
    start = kernel.index(_BARRIER, kernel.index(f"_{hole} = false;"))
    end = kernel.index(_BARRIER, start + len(_BARRIER)) + len(_BARRIER)
    kernel = kernel[:start] + _ROW_MASK.replace("HOLE", f"_{hole}") + kernel[end:]
    kernel = kernel.replace(f"_{row}[gl_LocalInvocationID.y]", "rowMask")
    return head + helper + "kernel void " + kernel
