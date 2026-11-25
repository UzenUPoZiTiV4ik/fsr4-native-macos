#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Independent Metal temporal policy around genuine imported Lite CT weights.

Apply PRE after signed_square/input_color_cache, POST after external_post_cnb
and input_color_cache (and before post_color_tile). No GPL tools/code or
replacement neural weights are used. This is NOT upstream lossy parity; partial
mode is EXPERIMENTAL.

ABI, reflected names (bindings chosen among free indices 0..7):
  Fsr412LossyConstants: 16-byte {uint flags,tileSize,phase,reserved}.
  flags bit0 wholeReuse, bit1 partial, bit2 previousValid. Bits0+1 together,
  unknown bits, nonzero reserved, reset, odd/empty extent invalidate reuse.
  partial requires power-of-two tileSize in [128,2048] and phase=0/1. Default
  tileSize=512; default full flags=0.
  Fsr412LossyMotion: dense uint4 per 2x2 output cell, 16B/cell, width/2 stride.
    xy signed round-even output motion as uint; z signed selected depth bits;
    w bit0 valid motion/history, bit1 valid depth. PRE writes, POST reads.
  Fsr412LossyPrevious/Current: dense Fsr412LossyRaw per OUTPUT pixel, width
    stride. Record = {half4 recurrent; half4 params}, 16B, storing bounded
    post-activation values:
      recurrent = half4(sigmoid(raw_recurrent / 1024)) in (0, 1)
      params    = half4(2*tanh(fx), sx, sy, historyShare) in [-2, 2]
    Raw int32 CT accumulators (normal/UP bounds 53084/73070) are NEVER packed
    into int16; fresh pixels keep exact float32 activations, while reuse frames
    load half4 directly and skip all 9 per-pixel exp() + 6 divisions.

Whole: use one cache aliased Previous==Current if desired. Specialized
`post_full` runs fresh CT + coalesced 2x2 16B cache writes with zero reuse
overhead; specialized `post_reuse` loads 16B cache + threadgroup-tiled motion
and reprojection windows with zero CT/ScratchBuffer/Current traffic.
`pre_reuse` strips the dead `r_recurrent_0` sample and unreachable neural tail.
Partial: ALWAYS distinct Previous/Current using unified `post`. Fresh mask is
checker tile parity ((x/tileSize+y/tileSize)&1)==phase OR x<2/y<2/x>=width-2/
y>=height-2. Every 2x2 cell shares the same fresh predicate; every pixel writes
Current coalesced so its age stays one frame.
"""
from __future__ import annotations

import hashlib
import math
import re

try:
    from common.hlsl_parse import balanced_end, argument_list
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from common.hlsl_parse import balanced_end, argument_list

REVISION = 2
CACHE_BYTES_PER_PIXEL = 16
MOTION_BYTES_PER_CELL = 16
CONSTANT_BYTES = 16
PARTIAL_TILE_SIZES = (128, 256, 512, 1024, 2048)
TIERS = (1080, 2160, 4320)
POST_MODES = ("unified", "full", "reuse")
_NAME = r"_\d+"


def _require(ok, message):
    if not ok:
        raise ValueError("Lite lossy: " + message)


def _one(pattern, text, description):
    found = list(re.finditer(pattern, text, re.M))
    _require(len(found) == 1, description + " shape changed")
    return found[0]


def _close(text, start, left="(", right=")"):
    _require(text[start] == left, "opening boundary")
    return balanced_end(text, start, left, right, "Lite lossy: unbalanced expression")


def _factor(text, end):
    """One unchanged working-color factor before an imported Gaussian weight."""
    while text[end - 1].isspace():
        end -= 1
    stop = end
    if text[end - 1] == ")":
        depth = 1
        start = end - 2
        while start >= 0 and depth:
            depth += (text[start] == ")") - (text[start] == "(")
            start -= 1
        _require(depth == 0, "unbalanced sample factor")
        start += 1
        while start > 0 and (text[start - 1].isalnum() or text[start - 1] in "_.:"):
            start -= 1
        return text[start:stop]
    start = end - 1
    while start > 0 and (text[start - 1].isalnum() or text[start - 1] in "_.:"):
        start -= 1
    if text[start:stop] in (".xyz", ".rgb"):
        return _factor(text, start) + text[start:stop]
    _require(bool(re.fullmatch(_NAME, text[start:stop])), "unrecognized sample factor")
    return text[start:stop]


def _tail(text):
    loop = _one(r"^        for \(int (" + _NAME + r") = 0; \1 < 4; \1\+\+\)$", text, "four-pixel loop")
    begin = text.index("{", loop.end())
    end = _close(text, begin, "{", "}")
    body = text[begin + 1:end]
    recurrent = _one(
        r"^            (" + _NAME + r")\[(" + _NAME + r")\] = float4\(1\.0\) / \(float4\(1\.0\) \+ exp\(-\(float4\((" + _NAME + r")\) \* 0\.0009765625\)\)\);$",
        body,
        "raw recurrent sigmoid",
    )
    _require(recurrent[2] == loop[1], "recurrent phase")
    filt = _one(
        r"^            float4 (" + _NAME + r") = float4\((" + _NAME + r")\) \* 0\.0009765625;$",
        body,
        "raw filter scale",
    )
    _require(recurrent.end() < filt.start(), "CT tail order")
    declarations = list(re.finditer(r"^            int4 " + _NAME + r";$", body[:recurrent.start()], re.M))
    _require(bool(declarations), "raw recurrent declaration")
    start = declarations[0].start()
    raw = body[start:filt.start()]
    biases = re.findall(r"^            " + _NAME + r" = int4\((-?\d+), (-?\d+), (-?\d+), (-?\d+)\);$", raw, re.M)
    _require(len(biases) == 2, "CT biases")
    table = _one(
        r"^constant spvUnsafeArray<uint, 128> (" + _NAME + r") = spvUnsafeArray<uint, 128>\(\{(.*?)\}\);$",
        text,
        "CT weight table",
    )
    words = [int(v) for v in re.findall(r"(\d+)u", table[2])]
    _require(len(words) == 128 and all(0 <= x <= 0xFFFFFFFF for x in words), "CT weight literals")
    _require(len(re.findall(re.escape(table[1]) + r"\[", raw)) == 32, "CT column operations")
    phase = _one(
        r"^            uint (" + _NAME + r") = \(128u \* uint\((" + _NAME + r")\[(" + _NAME + r")\]\.x\)\) \+ \(256u \* uint\(\2\[\3\]\.y\)\);$",
        body,
        "CT phase offsets",
    )
    _require(phase[3] == loop[1], "CT phase loop")
    _require(
        re.search(r"= " + re.escape(phase[1]) + r" \+ 64u;", raw)
        and re.search(r"= " + re.escape(phase[1]) + r";", raw),
        "CT half offsets",
    )
    weights = [((word >> shift) & 255) for word in words for shift in (0, 8, 16, 24)]
    weights = [x if x < 128 else x - 256 for x in weights]
    bias = [tuple(map(int, b)) for b in biases]
    bounds = []
    for p in range(4):
        # Table is filter first, recurrent second; execution is the reverse.
        for half in range(2):
            for ch in range(4):
                col = weights[p * 128 + half * 64 + ch * 16:p * 128 + half * 64 + ch * 16 + 16]
                bounds.append(128 * sum(abs(x) for x in col) + abs(bias[1 - half][ch]))
    _require(max(bounds) <= 2147483647, "CT int32 bound overflow")
    return dict(
        loop=loop,
        begin=begin,
        end=end,
        body=body,
        rec=recurrent,
        filt=filt,
        start=start,
        raw=raw,
        phase=phase,
        table=table,
        bounds=tuple(bounds),
    )


def _filter_params(body, filt_var):
    """Extract the post-activation filter expressions (tanh2, sx, sy, historyShare)."""
    fn = re.escape(filt_var)
    filt_x = _one(r"^            float (" + _NAME + r") = " + fn + r"\.x;$", body, "filter x alias")
    exp_pos = _one(r"^            float (" + _NAME + r") = exp\(" + re.escape(filt_x[1]) + r"\);$", body, "filter exp(+x)")
    exp_neg = _one(r"^            float (" + _NAME + r") = exp\(-" + re.escape(filt_x[1]) + r"\);$", body, "filter exp(-x)")
    sx = _one(r"^            float (" + _NAME + r") = 2\.0 / \(1\.0 \+ exp\(-" + fn + r"\.y\)\);$", body, "filter sx")
    sy = _one(r"^            float (" + _NAME + r") = 2\.0 / \(1\.0 \+ exp\(-" + fn + r"\.z\)\);$", body, "filter sy")
    tanh2 = _one(
        r"^            float (" + _NAME + r") = 2\.0 \* \(\(" + re.escape(exp_pos[1]) + r" - " + re.escape(exp_neg[1]) + r"\) / \(" + re.escape(exp_pos[1]) + r" \+ " + re.escape(exp_neg[1]) + r"\)\);$",
        body,
        "filter tanh2",
    )
    history = _one(r"^            float (" + _NAME + r") = 1\.0 / \(1\.0 \+ exp\(-" + fn + r"\.w\)\);$", body, "history share")
    _require(
        filt_x.end() < exp_pos.start()
        and exp_pos.end() < exp_neg.start()
        and exp_neg.end() < sx.start()
        and sx.end() < sy.start()
        and sy.end() < tanh2.start()
        and tanh2.end() < history.start(),
        "filter activation order",
    )
    return dict(
        filt_x=filt_x,
        exp_pos=exp_pos,
        exp_neg=exp_neg,
        sx=sx,
        sy=sy,
        tanh2=tanh2,
        history=history,
    )


def _dedup_gaussian_samples(body, weights, samples):
    """Load the 9 prepared/working color taps once into fsr412Samples[9].

    The imported SPIR-V splits the 9-tap Gaussian sum across two statements
    (_3160 for taps 0..4, then _3184 = _3001 + 1u, then _3260 for taps 5..8).
    Populating fsr412Samples[0..4] right before the first statement and
    fsr412Samples[5..8] right before the second keeps every coordinate variable
    in scope, preserves exact summation parenthesization, and leaves exactly 9
    r_prepared_color reads so post_color_tile rewrites them cleanly.
    """
    uses = []
    for idx, weight in enumerate(weights):
        match = _one(r"\* " + re.escape(weight[1]) + r"\b", body, "Gaussian sample weight")
        factor = samples[idx]
        factor_end = match.start()
        while body[factor_end - 1].isspace():
            factor_end -= 1
        factor_start = factor_end - len(factor)
        _require(body[factor_start:factor_end] == factor, "Gaussian sample span")
        line_start = body.rfind("\n", 0, factor_start) + 1
        uses.append((line_start, factor_start, factor_end, idx, factor))

    line_starts = sorted({u[0] for u in uses})
    _require(len(line_starts) == 2, "Gaussian weighted sum split")

    # Apply replacements from back to front so string indices remain valid.
    by_line = {ls: [] for ls in line_starts}
    for u in uses:
        by_line[u[0]].append(u)

    result = body
    for line_idx in reversed(range(2)):
        ls = line_starts[line_idx]
        entries = sorted(by_line[ls], key=lambda item: item[1], reverse=True)
        for _, fstart, fend, tap_idx, _ in entries:
            result = result[:fstart] + f"fsr412Samples[{tap_idx}]" + result[fend:]
        ordered = sorted(by_line[ls], key=lambda item: item[3])
        decl = "            float3 fsr412Samples[9];\n" if line_idx == 0 else ""
        loads = "".join(f"            fsr412Samples[{tap_idx}] = {factor};\n" for _, _, _, tap_idx, factor in ordered)
        result = result[:ls] + decl + loads + result[ls:]
    return result


def cache_layout(text):
    """Actual column bounds and post-activation half4 cache layout."""
    tail = _tail(text)
    return {
        "cache_bytes_per_pixel": CACHE_BYTES_PER_PIXEL,
        "motion_bytes_per_cell": MOTION_BYTES_PER_CELL,
        "max_accumulator_bound": max(tail["bounds"]),
        "column_bounds": tail["bounds"],
        "int16_supported": max(tail["bounds"]) <= 32767,
        "schema": "post_activation_half4_recurrent_params_v2",
    }


def cache_buffer_bytes(width, height):
    _require(
        isinstance(width, int)
        and isinstance(height, int)
        and not isinstance(width, bool)
        and not isinstance(height, bool)
        and 2 <= width <= 7680
        and 2 <= height <= 4320
        and (width | height) & 1 == 0,
        "cache requires an even positive aligned output viewport",
    )
    return width * height * CACHE_BYTES_PER_PIXEL


def fresh_pixel(pixel, size, flags=0, tile_size=512, phase=0, reset=False):
    """CPU host/native-gate specification; outer two pixels are always fresh in partial."""
    valid = (
        flags & ~7 == 0
        and flags & 4
        and not reset
        and min(size) >= 2
        and (size[0] | size[1]) & 1 == 0
    )
    mode = (flags & 3) if valid else 0
    if mode == 1:
        return False
    if mode != 2 or tile_size not in PARTIAL_TILE_SIZES or phase not in (0, 1):
        return True
    x, y = pixel
    width, height = size
    return x < 2 or y < 2 or x >= width - 2 or y >= height - 2 or ((x // tile_size + y // tile_size) & 1) == phase


def sanitized_motion(velocity, size, invalid=False, reset=False):
    """CPU round-even/safety specification; no float->int overflow/NaN conversion."""
    if invalid or reset or len(velocity) != 2 or not all(math.isfinite(v) for v in velocity):
        return (0, 0), False
    pixels = tuple(v * s for v, s in zip(velocity, size))
    if not all(math.isfinite(v) and abs(v) <= 254 for v in pixels):
        return (0, 0), False
    pixels = tuple(0 if abs(v) <= 0.03125 else v for v in pixels)
    return tuple(round(v) for v in pixels), True


COMMON_STRUCTS = """
// Independent MIT Lite lossy ABI, not upstream release parity.
struct Fsr412LossyConstants { uint flags; uint tileSize; uint phase; uint reserved; };
struct Fsr412LossyRaw { half4 recurrent; half4 params; };
"""

COMMON_MODE = """static inline uint Fsr412LossyMode(constant Fsr412LossyConstants& control,
    constant type_MLSR_Optimized_Constants& frame)
{
    if (control.reserved != 0u || (control.flags & ~7u) != 0u ||
        (control.flags & 4u) == 0u || frame.reset != 0u ||
        frame.width < 2u || frame.height < 2u || ((frame.width | frame.height) & 1u) != 0u)
        return 0u;
    uint mode = control.flags & 3u;
    if (mode == 2u && (control.tileSize < 128u || control.tileSize > 2048u ||
        (control.tileSize & (control.tileSize - 1u)) != 0u || control.phase > 1u)) return 0u;
    return mode == 1u || mode == 2u ? mode : 0u;
}
static inline bool Fsr412LossyFresh(uint2 pixel, constant Fsr412LossyConstants& control,
    constant type_MLSR_Optimized_Constants& frame)
{
    uint mode = Fsr412LossyMode(control, frame);
    if (mode == 1u) return false;
    if (mode != 2u) return true;
    return pixel.x < 2u || pixel.y < 2u || pixel.x >= frame.width - 2u || pixel.y >= frame.height - 2u ||
        (((pixel.x / control.tileSize + pixel.y / control.tileSize) & 1u) == control.phase);
}
"""

PRE = """static inline void Fsr412LossyRecord(uint2 pixel, float2 velocity, bool invalid,
    float depth, device uint4* records, constant type_MLSR_Optimized_Constants& frame)
{
    if (((pixel.x | pixel.y) & 1u) != 0u || (frame.width | frame.height) % 2u != 0u) return;
    float2 pixels = velocity * float2(frame.width, frame.height);
    bool valid = !invalid && frame.reset == 0u && all(isfinite(pixels)) && all(abs(pixels) <= float2(254.0));
    pixels = select(pixels, float2(0.0), abs(pixels) <= float2(0.03125));
    int2 displacement = valid ? int2(rint(clamp(pixels, float2(-254.0), float2(254.0)))) : int2(0);
    bool depthValid = valid && isfinite(depth) && as_type<uint>(depth) != 0u;
    records[(pixel.y / 2u) * (frame.width / 2u) + pixel.x / 2u] =
        uint4(as_type<uint2>(displacement), depthValid ? as_type<uint>(depth) : 0u,
              (valid ? 1u : 0u) | (depthValid ? 2u : 0u));
}
"""

POST_MOTION = """static inline uint4 Fsr412LossyReadMotion(int2 cell, const device uint4* records,
    constant type_MLSR_Optimized_Constants& frame)
{
    cell = clamp(cell, int2(0), int2(frame.width / 2u, frame.height / 2u) - int2(1));
    return records[uint(cell.y) * (frame.width / 2u) + uint(cell.x)];
}
static inline bool Fsr412LossyDifferent(int2 a, int2 b) { return any(abs(a - b) > int2(1)); }
"""

POST_PLAIN_REUSE = """static inline void Fsr412LossyCellMotion(int2 cell, const device uint4* records,
    constant type_MLSR_Optimized_Constants& frame,
    thread uint4& own, thread int2& motion, thread bool& neighborDiff)
{
    own = Fsr412LossyReadMotion(cell, records, frame);
    motion = as_type<int2>(own.xy);
    const int2 probes[12] = {int2(1,0),int2(-1,0),int2(0,1),int2(0,-1),
        int2(3,0),int2(-3,0),int2(0,3),int2(0,-3),int2(8,0),int2(-8,0),int2(0,8),int2(0,-8)};
    bool diff = false;
    for (uint i = 0u; i < 12u; ++i)
        diff = diff || Fsr412LossyDifferent(motion, as_type<int2>(Fsr412LossyReadMotion(cell + probes[i], records, frame).xy));
    neighborDiff = diff;
}
static inline float Fsr412LossyLuma(float3 color) { return dot(color, float3(.2126f, .7152f, .0722f)); }
static inline float3 Fsr412LossyResolve(float3 current, float3 history, float historyShare,
    thread const float3* samples, float lo, float hi, bool lumaFinite,
    bool atRest, bool otherMotion, bool uncovered)
{
    float3 minimum = samples[0], maximum = samples[0];
    for (uint i = 1u; i < 9u; ++i) { minimum = min(minimum, samples[i]); maximum = max(maximum, samples[i]); }
    float3 span = maximum - minimum;
    float3 difference = abs(current - history);
    bool changed = any(history < minimum) || any(history > maximum) || any(difference > span * .25f + .05f);
    for (uint ch = 0u; ch < 3u; ++ch)
        changed = changed || (abs(minimum[ch] + maximum[ch] - 2.0f * history[ch]) > span[ch] * .85f &&
                              difference[ch] > span[ch] * .05f + .01f);
    float freshShare = 1.0f - historyShare;
    float weight = freshShare * freshShare;
    if (atRest && !changed) weight *= saturate((freshShare - .95f) / .045f);
    float3 safeHistory = otherMotion ? clamp(history, minimum, maximum) : history;
    float3 result = current * weight + safeHistory * (1.0f - weight);
    if (!lumaFinite) return uncovered ? current : result;
    // A sample must have support on BOTH axes. Lone pixels and thin rain do not win.
    float support = min(Fsr412LossyLuma(samples[4]),
        min(max(Fsr412LossyLuma(samples[3]), Fsr412LossyLuma(samples[5])),
            max(Fsr412LossyLuma(samples[1]), Fsr412LossyLuma(samples[7]))));
    float threshold = clamp(.08f + (hi - lo) * .5f, .08f, .25f);
    float spark = isfinite(support) ? saturate((support - hi - threshold) / threshold) : 0.0f;
    result = mix(result, current, spark);
    bool darkEdge = !atRest && all(history <= max(current * .1f, float3(0.0)));
    return uncovered || darkEdge ? current : result;
}
"""


def _prepare(text, tier, stage, extra="", new_look=False, post_mode="unified"):
    _require(tier in TIERS, "unknown tensor tier")
    _require(post_mode in POST_MODES, "unknown post mode")
    version_tag = 3 if new_look else 2
    marker = "// FSR412_LOSSY_V%d_%s tier=%d %s" % (version_tag, stage.upper(), tier, extra)
    if text.startswith(marker + "\n"):
        # Integrity is checked on repeated application, including generated body.
        receipt = re.match(re.escape(marker) + r"\n// payload-sha256=([0-9a-f]{64})\n", text)
        _require(receipt is not None, "missing existing transform receipt")
        _require(hashlib.sha256(text[receipt.end():].encode()).hexdigest() == receipt[1], "modified existing transform")
        return None, marker
    _require("FSR412_LOSSY_" not in text and "Fsr412Lossy" not in text, "existing/colliding lossy transform")
    kernel = _one(r"kernel void (\w*fsr4_model_v07_fp8_no_scale_" + stage + r"pass)\(([^\n]*)\)\n\{\n", text, "kernel")
    _require(len(re.findall(r"\bkernel\s+void\b", text)) == 1, "multiple kernels")
    used = set(map(int, re.findall(r"\[\[buffer\((\d+)\)\]\]", kernel[2])))
    free = [x for x in range(8) if x not in used]

    if stage == "pre":
        need = 2
        _require(len(free) >= need, "not enough buffer indices <=7")
        bindings = [
            "constant Fsr412LossyConstants& Fsr412LossyConstants [[buffer(%d)]]" % free[0],
            "device uint4* Fsr412LossyMotion [[buffer(%d)]]" % free[1],
        ]
        helpers = COMMON_STRUCTS + PRE
    elif post_mode == "full":
        need = 2
        _require(len(free) >= need, "not enough buffer indices <=7")
        bindings = [
            "constant Fsr412LossyConstants& Fsr412LossyConstants [[buffer(%d)]]" % free[0],
            "device Fsr412LossyRaw* Fsr412LossyCurrent [[buffer(%d)]]" % free[1],
        ]
        helpers = COMMON_STRUCTS
        if new_look:
            from fsr412.new_look import COLOR_RULES
            helpers += COLOR_RULES
    elif post_mode == "reuse":
        need = 3
        _require(len(free) >= need, "not enough buffer indices <=7")
        bindings = [
            "constant Fsr412LossyConstants& Fsr412LossyConstants [[buffer(%d)]]" % free[0],
            "const device uint4* Fsr412LossyMotion [[buffer(%d)]]" % free[1],
            "const device Fsr412LossyRaw* Fsr412LossyPrevious [[buffer(%d)]]" % free[2],
        ]
        helpers = COMMON_STRUCTS + POST_MOTION
        if new_look:
            from fsr412.new_look import COLOR_RULES, MOTION_RULES
            helpers += COLOR_RULES + MOTION_RULES
        else:
            helpers += POST_PLAIN_REUSE
    else:
        need = 4
        _require(len(free) >= need, "not enough buffer indices <=7")
        bindings = [
            "constant Fsr412LossyConstants& Fsr412LossyConstants [[buffer(%d)]]" % free[0],
            "const device uint4* Fsr412LossyMotion [[buffer(%d)]]" % free[1],
            "const device Fsr412LossyRaw* Fsr412LossyPrevious [[buffer(%d)]]" % free[2],
            "device Fsr412LossyRaw* Fsr412LossyCurrent [[buffer(%d)]]" % free[3],
        ]
        helpers = COMMON_STRUCTS + COMMON_MODE + POST_MOTION
        if new_look:
            from fsr412.new_look import COLOR_RULES, MOTION_RULES
            helpers += COLOR_RULES + MOTION_RULES
        else:
            helpers += POST_PLAIN_REUSE

    signature_end = kernel.end() - len(")\n{\n")
    text = text[:signature_end] + ", " + ", ".join(bindings) + text[signature_end:]
    text = text[:kernel.start()] + helpers + text[kernel.start():]
    return text, marker


def _receipt(text, marker):
    return marker + "\n// payload-sha256=" + hashlib.sha256(text.encode()).hexdigest() + "\n" + text


def apply_lossy_pre(text, tier, low_res_mv=True, depth_inverted=False, reuse_only=False, new_look=False):
    original = text
    extra = "lm=%d di=%d reuse-only=%d" % (bool(low_res_mv), bool(depth_inverted), bool(reuse_only))
    text, marker = _prepare(text, tier, "pre", extra, new_look)
    if text is None:
        return original
    velocity = _one(r"^        float2 (" + _NAME + r");\n        bool (" + _NAME + r");\n        do\n        \{", text, "sanitized velocity pair")
    write = _one(r"^        rw_reprojected_color\.write\([^\n;]+, uint2\((" + _NAME + r")\)\);$", text, "history reprojection store")
    _require(velocity.start() < write.start(), "motion before reprojection")
    _require(
        text.count(velocity[1] + " = float2(0.0);") == 1
        and text.count(velocity[2] + " = true;") == 1
        and text.count(velocity[2] + " = false;") == 1,
        "velocity sanitization",
    )
    depth = "0.0f"
    if low_res_mv:
        _require("texture2d<float> r_depth" in text, "low-resolution motion requires depth")
        depth_tests = re.findall(r"bool " + _NAME + r" = " + _NAME + r" ([<>]) " + _NAME + r";", text[:velocity.start()])
        _require(
            len(depth_tests) == 7 and all(op == (">" if depth_inverted else "<") for op in depth_tests),
            "depth inversion variant mismatch",
        )
        sample = _one(r"r_velocity\.read\(", text, "velocity load")
        closing = _close(text, sample.end() - 1)
        args = argument_list(text[sample.end():closing])
        _require(len(args) == 2 and args[1] == "0u", "velocity sample mip")
        # Dilation reads unclamped coordinates. Only its interior winner can
        # be returned verbatim; the original record clamps an exterior winner.
        final = _one(r"(r_depth\.read\(uint2\(uint2\(_\d+\)\), 0u\)\.x) ([<>]) "
                     r"(\(_\d+ \? _\d+ : _\d+\))", args[0], "final dilation comparison")
        _require(final[2] == (">" if depth_inverted else "<"), "final depth direction")
        coord = args[0].replace(final[0], "fsr412LastWins")
        declarations = (
            f"        const float fsr412LastDepth = {final[1]};\n"
            f"        const float fsr412PriorDepth = {final[3]};\n"
            f"        const bool fsr412LastWins = fsr412LastDepth {final[2]} fsr412PriorDepth;\n"
            f"        const uint2 fsr412Winner = {coord};\n"
        )
        line = text.rindex("\n", 0, sample.start()) + 1
        text = text[:line] + declarations + text[line:sample.end()] + "fsr412Winner, 0u" + text[closing:]
        depth = ("(all(fsr412Winner < uint2(MLSR_Optimized_Constants.width_lr, MLSR_Optimized_Constants.height_lr)) "
                 "? (fsr412LastWins ? fsr412LastDepth : fsr412PriorDepth) "
                 ": r_depth.read(uint2(clamp(int2(fsr412Winner), int2(0), "
                 "int2(MLSR_Optimized_Constants.width_lr, MLSR_Optimized_Constants.height_lr)-int2(1))), 0u).x)")
        write = _one(r"^        rw_reprojected_color\.write\([^\n;]+, uint2\((" + _NAME + r")\)\);$", text, "reprojection store")
        if depth_inverted:
            depth = "-(" + depth + ")"
    else:
        _require("r_depth.read(" not in text, "full-resolution motion variant mismatch")

    injected = (
        "\n        Fsr412LossyRecord(" + write[1] + ", " + velocity[1] + ", " + velocity[2] + ", " + depth
        + ", Fsr412LossyMotion, MLSR_Optimized_Constants);\n"
    )
    text = text[:write.end()] + injected + text[write.end():]

    if new_look:
        # Preserve the central Catmull-Rom tap's note in the unused alpha.
        # The variable is neutral on invalid/reset history and out of bounds.
        taps = list(re.finditer(r"r_history_color\.sample\(", text))
        _require(len(taps) == 9, "new-look history tap count")
        tap = taps[4]
        close = _close(text, tap.end() - 1)
        sampled_note = text[tap.start():close + 1] + ".w"
        statement_end = text.index(";", close) + 1
        text = text[:statement_end] + (
            "\n                fsr412LookPreviousNote = "
            "(Fsr412LossyConstants.flags & 4u) != 0u ? float(" + sampled_note + ") : 0.0f;\n"
            "                if (!isfinite(fsr412LookPreviousNote)) fsr412LookPreviousNote = 0.0f;"
        ) + text[statement_end:]
        velocity = _one(r"^        float2 (" + _NAME + r");\n        bool (" + _NAME + r");\n        do\n        \{", text, "new-look velocity scope")
        text = text[:velocity.start()] + "        float fsr412LookPreviousNote = 0.0f;\n" + text[velocity.start():]
        store = _one(r"rw_reprojected_color\.write\(half3\((" + _NAME + r")\)\.(?:xyzx|xyzz), uint2\((" + _NAME + r")\)\);", text, "new-look reprojection note store")
        text = text[:store.start()] + (
            "rw_reprojected_color.write(half4(half3(" + store[1] + "), "
            "half(fsr412LookPreviousNote)), uint2(" + store[2] + "));"
        ) + text[store.end():]

    if reuse_only:
        # Whole-frame reuse PRE is only dispatched when neural passes are skipped:
        # drop the dead r_recurrent_0 sample and unreachable neural input tail.
        rec_sample = _one(r"^(\s+)(_\d+) = float4\(r_recurrent_0\.sample\([^\n]+\)\);$", text, "pre recurrent sample")
        text = text[:rec_sample.start()] + f"{rec_sample[1]}{rec_sample[2]} = float4(0.0);" + text[rec_sample.end():]
        record_end = text.index("\n", text.index("Fsr412LossyRecord(" + write[1])) + 1
        tail_end = _one(r"        break;\n    \} while\(false\);\n\}\n?$", text, "pre kernel epilogue")
        _require(record_end < tail_end.start(), "pre reuse tail order")
        text = text[:record_end] + "        break;\n    } while(false);\n}\n"
        _require("ScratchBuffer._m0[" not in text and "r_recurrent_0.sample(" not in text, "pre reuse dead tail left behind")
    else:
        _require(text.find("ScratchBuffer._m0[") > text.find("Fsr412LossyRecord(" + write[1]), "tensor store precedes reuse return")
    return _receipt(text, marker)


def _rewrite_new_look_history_stores(text, body, loop_var, blend_var):
    store = _one(r"^            (" + _NAME + r")\[" + re.escape(loop_var) + r"\] = " + re.escape(blend_var) + r";$", body, "new-look history array")
    pattern = r"rw_history_color\.write\(" + re.escape(store[1]) + r"\[([0-3])\]\.(?:xyzx|xyzz), ([^\n]+)\);"
    text, count = re.subn(
        pattern,
        lambda m: "rw_history_color.write(float4(" + store[1] + "[" + m[1] + "], fsr412LookNotes[" + m[1] + "]), " + m[2] + ");",
        text,
    )
    _require(count == 4, "new-look four history note stores")
    return text


def _coalesced_cache_write(origin_var, offset_arr, guard=False):
    lines = [
        f"            const uint fsr412OutW = MLSR_Optimized_Constants.width;",
    ]
    for i in range(4):
        lines.append(
            f"            Fsr412LossyCurrent[({origin_var}.y + uint({offset_arr}[{i}].y)) * fsr412OutW + ({origin_var}.x + uint({offset_arr}[{i}].x))] = fsr412OutCache[{i}];"
        )
    inner = "\n".join(lines) + "\n"
    if guard:
        return (
            "        if (Fsr412LossyMode(Fsr412LossyConstants, MLSR_Optimized_Constants) != 1u)\n"
            "        {\n"
            + inner
            + "        }\n"
        )
    return "        {\n" + inner + "        }\n"


def apply_lossy_post(text, tier, new_look=False, mode="unified"):
    original = text
    extra = "" if mode == "unified" else f"mode={mode}"
    text, marker = _prepare(text, tier, "post", extra=extra, new_look=new_look, post_mode=mode)
    if text is None:
        return original

    tail = _tail(text)
    body = tail["body"]
    rec = tail["rec"]
    filt = tail["filt"]
    loop_var = tail["loop"][1]
    params = _filter_params(body, filt[1])

    output = _one(
        r"^            uint2 (" + _NAME + r") = \((" + _NAME + r") \* uint2\(2u\)\) \+ uint2\((" + _NAME + r")\[(" + _NAME + r")\]\);$",
        body,
        "output pixel",
    )
    _require(output[4] == loop_var, "output phase")
    pixel_var, cell_var, offset_arr = output[1], output[2], output[3]

    cache = _one(r"^        uint fsr412CacheWord = \((" + _NAME + r")\.y \* (\d+)u \+ \1\.x \* 16u\) >> 2u;$", text, "external CNB cache")
    _require(cache[1] == cell_var and int(cache[2]) == tier * 8 // 9 * 16, "CNB tier/coordinate")
    loads = list(re.finditer(r"^        uint (" + _NAME + r") = ScratchBuffer\._m0\[fsr412CacheWord \+ (\d)u\];$", text, re.M))
    _require(len(loads) == 4 and [int(x[2]) for x in loads] == list(range(4)), "CNB word loads")
    _require(text.count("ScratchBuffer._m0[") == 4, "unexpected CNB/scratch reads")

    weights = list(re.finditer(r"^            float (" + _NAME + r") = exp\(\(-2\.2634675502777099609375\) \* [^\n]+;$", body, re.M))
    _require(len(weights) == 9, "Gaussian tap count")
    samples = []
    for weight in weights:
        use = _one(r"\* " + re.escape(weight[1]) + r"\b", body, "Gaussian sample weight")
        samples.append(_factor(body, use.start()))

    history_var = params["history"][1]
    blend = _one(
        r"^            float3 (" + _NAME + r") = [^\n]+ \+ \(float3\(r_reprojected_color\.read\(uint2\(" + re.escape(pixel_var) + r"\), 0u\)\.xyz\) \* " + re.escape(history_var) + r"\);$",
        body,
        "original blend",
    )
    current_weight = _one(r"\* \(1\.0 - " + re.escape(history_var) + r"\)", blend[0], "current share")
    current_var = _factor(blend[0], current_weight.start())

    origin_stmt = _one(r"^        uint2 (" + _NAME + r") = " + re.escape(cell_var) + r" \* uint2\(2u\);$", text, "output origin")
    origin_var = origin_stmt[1]

    rec_arr = rec[1]
    tanh2_var = params["tanh2"][1]
    sx_var = params["sx"][1]
    sy_var = params["sy"][1]
    blend_var = blend[1]

    # --- MODE 1: FULL (whole-frame fresh inference, zero reuse code) ---
    if mode == "full":
        new_body = body
        if new_look:
            reproj_read = f"float3(r_reprojected_color.read(uint2({pixel_var}), 0u).xyz)"
            new_body = new_body.replace(
                blend[0],
                f"            float4 fsr412LookReproj = float4(r_reprojected_color.read(uint2({pixel_var}), 0u));\n"
                + blend[0].replace(reproj_read, "fsr412LookReproj.xyz", 1)
                + f"\n            bool fsr412LookValid = (Fsr412LossyConstants.flags & 4u) != 0u && MLSR_Optimized_Constants.reset == 0u;\n"
                + f"            float fsr412LookPreviousNote = fsr412LookValid ? fsr412LookReproj.w : 0.0f;\n"
                + f"            if (!isfinite(fsr412LookPreviousNote)) fsr412LookPreviousNote = 0.0f;\n"
                + f"            {blend_var} = Fsr412LookFinish({blend_var}, {current_var}, fsr412LookReproj.xyz, fsr412LookPreviousNote, fsr412LookValid, fsr412LookNotes[{loop_var}]);\n"
                + f"            fsr412OutCache[{loop_var}] = {{half4({rec_arr}[{loop_var}]), half4({tanh2_var}, {sx_var}, {sy_var}, {history_var})}};",
                1,
            )
        else:
            new_body = new_body.replace(
                blend[0],
                blend[0]
                + f"\n            fsr412OutCache[{loop_var}] = {{half4({rec_arr}[{loop_var}]), half4({tanh2_var}, {sx_var}, {sy_var}, {history_var})}};",
                1,
            )

        text = text[:tail["begin"] + 1] + new_body + text[tail["end"]:]
        pre_loop_decl = "        Fsr412LossyRaw fsr412OutCache[4];\n"
        if new_look:
            pre_loop_decl += "        float fsr412LookNotes[4] = {0.0f, 0.0f, 0.0f, 0.0f};\n"
            text = _rewrite_new_look_history_stores(text, body, loop_var, blend_var)
        text = text.replace(tail["loop"][0], pre_loop_decl + tail["loop"][0], 1)
        text = text.replace(origin_stmt[0], origin_stmt[0] + "\n" + _coalesced_cache_write(origin_var, offset_arr, guard=False), 1)
        return _receipt(text, marker)

    # --- MODE 2: REUSE (whole-frame reuse, TG-tiled motion + reprojection, zero CT) ---
    if mode == "reuse":
        # Strip the unused CT weight table so the kernel stays lean.
        table_match = tail["table"]
        # Build band sample expressions before deduplicating samples in body.
        band_lines = ""
        if new_look:
            center = _one(r"int2\(uint2\((" + _NAME + r"), (" + _NAME + r")\)\)", samples[4], "new-look center sample")
            band = [
                samples[4][:center.start()] + "(" + center[0] + " + int2(%d,%d))" % offset + samples[4][center.end():]
                for offset in ((-8, 0), (8, 0), (0, -8), (0, 8))
            ]
            band_lines = "\n".join(f"                fsr412BandSamples[{i}] = {expr};" for i, expr in enumerate(band))

        # Rewrite body: load fsr412Samples[9] once, remove CT + exp/tanh2/history, inject reuse resolve.
        new_body = _dedup_gaussian_samples(body, weights, samples)
        new_body = new_body.replace(params["tanh2"][0] + "\n", "", 1)
        new_body = new_body.replace(params["history"][0] + "\n", "", 1)

        blend_after = _one(
            r"^            float3 (" + _NAME + r") = [^\n]+ \+ \(float3\(r_reprojected_color\.read\(uint2\(" + re.escape(pixel_var) + r"\), 0u\)\.xyz\) \* " + re.escape(history_var) + r"\);$",
            new_body,
            "reuse blend",
        )
        current_weight_after = _one(r"\* \(1\.0 - " + re.escape(history_var) + r"\)", blend_after[0], "reuse current share")
        current_after = _factor(blend_after[0], current_weight_after.start())

        ct_and_act_start = new_body.index(body[tail["start"]:tail["start"] + 40])
        ct_and_act_end = new_body.index(params["sy"][0]) + len(params["sy"][0]) + 1
        cache_load = f"""            int2 fsr412PreviousPixel = int2({pixel_var}) + fsr412CellMotion;
            uint2 fsr412SafePrevious = uint2(clamp(fsr412PreviousPixel, int2(0), fsr412OutMax));
            Fsr412LossyRaw fsr412Raw = Fsr412LossyPrevious[fsr412SafePrevious.y * MLSR_Optimized_Constants.width + fsr412SafePrevious.x];
            {rec_arr}[{loop_var}] = float4(fsr412Raw.recurrent);
            float {tanh2_var} = float(fsr412Raw.params.x);
            float {sx_var} = float(fsr412Raw.params.y);
            float {sy_var} = float(fsr412Raw.params.z);
            float {history_var} = float(fsr412Raw.params.w);
"""
        new_body = new_body[:ct_and_act_start] + cache_load + new_body[ct_and_act_end:]
        _require(filt[1] not in new_body, "unexpected filter variable use in reuse body")

        if new_look:
            resolve_block = f"""            float3 fsr412Current = {current_after};
            int fsr412SubX = int({pixel_var}.x & 1u);
            int fsr412SubY = int({pixel_var}.y & 1u);
            int fsr412RepX = fsr412LocCell.x * 2 + fsr412SubX;
            int fsr412RepY = fsr412LocCell.y * 2 + fsr412SubY;
            float4 fsr412LookReproj = float4(fsr412RepTile[uint(fsr412RepY + 1) * 34u + uint(fsr412RepX + 1)]);
            float3 fsr412LookHistory = fsr412LookReproj.xyz;
            bool fsr412LookValid = (Fsr412LossyConstants.flags & 4u) != 0u && MLSR_Optimized_Constants.reset == 0u;
            float fsr412LookPreviousNote = fsr412LookValid ? fsr412LookReproj.w : 0.0f;
            if (!isfinite(fsr412LookPreviousNote)) fsr412LookPreviousNote = 0.0f;
            float3 fsr412HistorySamples[9];
            float3 fsr412HistoryMaximum = float3(0.0f);
            for (int ry = 0; ry < 3; ++ry)
            {{
                for (int rx = 0; rx < 3; ++rx)
                {{
                    float3 value = float3(fsr412RepTile[uint(fsr412RepY + ry) * 34u + uint(fsr412RepX + rx)].xyz);
                    fsr412HistorySamples[ry * 3 + rx] = value;
                    fsr412HistoryMaximum = max(fsr412HistoryMaximum, value);
                }}
            }}
            float3 fsr412BandSamples[4] = {{float3(0.0f), float3(0.0f), float3(0.0f), float3(0.0f)}};
            if (Fsr412LookArrival(fsr412Samples, fsr412HistoryMaximum) > 0.0f)
            {{
{band_lines}
            }}
            bool fsr412Uncovered = (fsr412CellOwn.w & 1u) == 0u || any(fsr412PreviousPixel < int2(0)) ||
                fsr412PreviousPixel.x >= int(MLSR_Optimized_Constants.width) || fsr412PreviousPixel.y >= int(MLSR_Optimized_Constants.height);
            bool fsr412SurfDiff = false;
            if (!fsr412ZeroMotion)
            {{
                int2 fsr412PrevCell = fsr412PreviousPixel / 2;
                fsr412SurfDiff = Fsr412LossyDifferent(fsr412CellMotion, Fsr412LookMotion(fsr412PrevCell, Fsr412LossyMotion, MLSR_Optimized_Constants));
                uint4 fsr412Surface = Fsr412LookDepthRecord(fsr412PrevCell, Fsr412LossyMotion, MLSR_Optimized_Constants);
                fsr412Uncovered = fsr412Uncovered || ((fsr412CellDepth.w & fsr412Surface.w & 2u) != 0u &&
                    as_type<float>(fsr412Surface.z) < as_type<float>(fsr412CellDepth.z) && fsr412SurfDiff);
            }}
            bool fsr412OtherMotion = fsr412SurfDiff || fsr412NeighborDiff;
            bool fsr412AtRest = fsr412Still && !fsr412OtherMotion;
            float3 {blend_var} = Fsr412LookResolve(fsr412Current, fsr412LookHistory, {history_var}, fsr412Samples,
                fsr412HistorySamples, fsr412BandSamples, fsr412AtRest, fsr412OtherMotion,
                fsr412Uncovered, fsr412CellDepthEdge, fsr412LookPreviousNote);
            if (fsr412Still && fsr412OtherMotion)
                {rec_arr}[{loop_var}].z = max({rec_arr}[{loop_var}].z, 0.45f);
            {blend_var} = Fsr412LookFinish({blend_var}, fsr412Current, fsr412LookHistory, fsr412LookPreviousNote,
                fsr412LookValid, fsr412LookNotes[{loop_var}]);"""
        else:
            resolve_block = f"""            float3 fsr412Current = {current_after};
            bool fsr412Uncovered = (fsr412CellOwn.w & 1u) == 0u || any(fsr412PreviousPixel < int2(0)) ||
                fsr412PreviousPixel.x >= int(MLSR_Optimized_Constants.width) || fsr412PreviousPixel.y >= int(MLSR_Optimized_Constants.height);
            bool fsr412SurfDiff = false;
            if (!fsr412ZeroMotion)
            {{
                uint4 fsr412Surface = Fsr412LossyReadMotion(fsr412PreviousPixel / 2, Fsr412LossyMotion, MLSR_Optimized_Constants);
                fsr412SurfDiff = Fsr412LossyDifferent(fsr412CellMotion, as_type<int2>(fsr412Surface.xy));
                fsr412Uncovered = fsr412Uncovered || ((fsr412CellOwn.w & fsr412Surface.w & 2u) != 0u &&
                    as_type<float>(fsr412Surface.z) < as_type<float>(fsr412CellOwn.z) && fsr412SurfDiff);
            }}
            bool fsr412OtherMotion = fsr412SurfDiff || fsr412NeighborDiff;
            bool fsr412AtRest = (fsr412CellOwn.w & 1u) != 0u && fsr412ZeroMotion && !fsr412OtherMotion;
            int fsr412SubX = int({pixel_var}.x & 1u);
            int fsr412SubY = int({pixel_var}.y & 1u);
            float fsr412Lo = 3.402823466e+38f;
            float fsr412Hi = -3.402823466e+38f;
            bool fsr412LumaFinite = true;
            for (int ry = 0; ry < 3; ++ry)
            {{
                for (int rx = 0; rx < 3; ++rx)
                {{
                    float v = fsr412RepLuma[(fsr412SubY + ry) * 4 + (fsr412SubX + rx)];
                    fsr412LumaFinite = fsr412LumaFinite && isfinite(v);
                    fsr412Lo = min(fsr412Lo, v);
                    fsr412Hi = max(fsr412Hi, v);
                }}
            }}
            float3 {blend_var} = Fsr412LossyResolve(fsr412Current, float3(r_reprojected_color.read(uint2({pixel_var}), 0u).xyz), {history_var}, fsr412Samples,
                fsr412Lo, fsr412Hi, fsr412LumaFinite, fsr412AtRest, fsr412OtherMotion, fsr412Uncovered);"""

        _require(new_body.count(blend_after[0]) == 1, "reuse blend replacement")
        new_body = new_body.replace(blend_after[0], resolve_block, 1)

        text = text[:tail["begin"] + 1] + new_body + text[tail["end"]:]
        # Remove the dead ScratchBuffer CNB block.
        cnb_block = re.search(
            r"        // Native POST-CNB cache[^\n]*\n        uint fsr412CacheWord = [^\n]+\n(?:        uint _\d+ = ScratchBuffer\._m0\[fsr412CacheWord \+ \du\];\n){4}",
            text,
        )
        _require(cnb_block is not None, "reuse CNB removal boundary")
        text = text[:cnb_block.start()] + text[cnb_block.end():]
        # Remove unused CT weight table declaration.
        text = text.replace(table_match[0] + "\n", "", 1)

        if new_look:
            text = _rewrite_new_look_history_stores(text, body, loop_var, blend_var)

        # Insert threadgroup memory tiles at kernel start and cooperative loads before early-out.
        # Plain reuse needs only neighborhood luma. Compute it once per
        # staged texel, and read each output pixel's RGB center in the loop.
        rep_type = "half4" if new_look else "float"
        rep_tile = "fsr412RepTile" if new_look else "fsr412RepLumaTile"
        rep_read_expr = "r_reprojected_color.read(uint2(p), 0u)"
        if not new_look:
            rep_read_expr = f"Fsr412LossyLuma(float3({rep_read_expr}.xyz))"
        mot_read_expr = (
            "short2(Fsr412LookMotion(c, Fsr412LossyMotion, MLSR_Optimized_Constants))"
            if new_look
            else "short2(as_type<int2>(Fsr412LossyReadMotion(c, Fsr412LossyMotion, MLSR_Optimized_Constants).xy))"
        )
        kernel_open = _one(r"kernel void \w+\([^\n]*\)\n\{\n", text, "reuse kernel open")
        tg_decls = (
            f"    threadgroup {rep_type} {rep_tile}[34 * 34];\n"
            f"    threadgroup short2 fsr412MotTile[32 * 32];\n"
        )
        text = text[:kernel_open.end()] + tg_decls + text[kernel_open.end():]

        early_exit = _one(r"^        if \(any\(" + re.escape(cell_var) + r" >= _\d+\)\)\n", text, "reuse cell early exit")
        tg_fill = f"""        {{
            const int2 fsr412RepOrigin = int2(gl_WorkGroupID.xy * uint2(32u)) - int2(1);
            const int2 fsr412TileOutMax = int2(MLSR_Optimized_Constants.width, MLSR_Optimized_Constants.height) - int2(1);
            for (uint i = gl_LocalInvocationID.x; i < 34u * 34u; i += 256u)
            {{
                const int2 p = clamp(fsr412RepOrigin + int2(int(i % 34u), int(i / 34u)), int2(0), fsr412TileOutMax);
                {rep_tile}[i] = {rep_read_expr};
            }}
            const int2 fsr412MotOrigin = int2(gl_WorkGroupID.xy * uint2(16u)) - int2(8);
            const int2 fsr412MotMax = int2(MLSR_Optimized_Constants.width / 2u, MLSR_Optimized_Constants.height / 2u) - int2(1);
            for (uint i = gl_LocalInvocationID.x; i < 32u * 32u; i += 256u)
            {{
                const int2 c = clamp(fsr412MotOrigin + int2(int(i % 32u), int(i / 32u)), int2(0), fsr412MotMax);
                fsr412MotTile[i] = {mot_read_expr};
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }}
"""
        text = text[:early_exit.start()] + tg_fill + text[early_exit.start():]

        if new_look:
            pre_loop = f"""        int2 fsr412Cell = int2({cell_var});
        int2 fsr412LocCell = int2({cell_var} - gl_WorkGroupID.xy * uint2(16u));
        int2 fsr412OutMax = int2(MLSR_Optimized_Constants.width, MLSR_Optimized_Constants.height) - int2(1);
        uint4 fsr412CellOwn = Fsr412LossyReadMotion(fsr412Cell, Fsr412LossyMotion, MLSR_Optimized_Constants);
        int2 fsr412CellMotion = int2(fsr412MotTile[uint(fsr412LocCell.y + 8) * 32u + uint(fsr412LocCell.x + 8)]);
        bool fsr412ZeroMotion = all(fsr412CellMotion == int2(0));
        uint4 fsr412CellDepth = Fsr412LookDepthRecord(fsr412Cell, Fsr412LossyMotion, MLSR_Optimized_Constants);
        bool fsr412NeighborDiff = false;
        {{
            const int2 probes[8] = {{int2(3,0),int2(-3,0),int2(0,3),int2(0,-3),int2(8,0),int2(-8,0),int2(0,8),int2(0,-8)}};
            const int2 loc = fsr412LocCell + int2(8);
            for (uint i = 0u; i < 8u; ++i)
            {{
                const int2 q = loc + probes[i];
                fsr412NeighborDiff = fsr412NeighborDiff || Fsr412LossyDifferent(fsr412CellMotion, int2(fsr412MotTile[uint(q.y) * 32u + uint(q.x)]));
            }}
        }}
        bool fsr412Still = (fsr412CellOwn.w & 1u) != 0u && fsr412ZeroMotion;
        bool fsr412CellDepthEdge = !(fsr412Still && !fsr412NeighborDiff) && Fsr412LookDepthEdge(fsr412Cell, Fsr412LossyMotion, MLSR_Optimized_Constants);
        float fsr412LookNotes[4] = {{0.0f, 0.0f, 0.0f, 0.0f}};
"""
        else:
            pre_loop = f"""        int2 fsr412Cell = int2({cell_var});
        int2 fsr412LocCell = int2({cell_var} - gl_WorkGroupID.xy * uint2(16u));
        int2 fsr412OutMax = int2(MLSR_Optimized_Constants.width, MLSR_Optimized_Constants.height) - int2(1);
        uint4 fsr412CellOwn = Fsr412LossyReadMotion(fsr412Cell, Fsr412LossyMotion, MLSR_Optimized_Constants);
        int2 fsr412CellMotion = as_type<int2>(fsr412CellOwn.xy);
        bool fsr412ZeroMotion = all(fsr412CellMotion == int2(0));
        bool fsr412NeighborDiff = false;
        {{
            const int2 probes[12] = {{int2(1,0),int2(-1,0),int2(0,1),int2(0,-1),
                int2(3,0),int2(-3,0),int2(0,3),int2(0,-3),int2(8,0),int2(-8,0),int2(0,8),int2(0,-8)}};
            const int2 loc = fsr412LocCell + int2(8);
            for (uint i = 0u; i < 12u; ++i)
            {{
                const int2 q = loc + probes[i];
                fsr412NeighborDiff = fsr412NeighborDiff || Fsr412LossyDifferent(fsr412CellMotion, int2(fsr412MotTile[uint(q.y) * 32u + uint(q.x)]));
            }}
        }}
        float fsr412RepLuma[16];
        {{
            const int2 repLoc = fsr412LocCell * int2(2);
            for (int ry = 0; ry < 4; ++ry)
            {{
                for (int rx = 0; rx < 4; ++rx)
                {{
                    fsr412RepLuma[ry * 4 + rx] = fsr412RepLumaTile[uint(repLoc.y + ry) * 34u + uint(repLoc.x + rx)];
                }}
            }}
        }}
"""
        text = text.replace(tail["loop"][0], pre_loop + tail["loop"][0], 1)
        return _receipt(text, marker)

    # --- MODE 3: UNIFIED (supports full, whole-reuse, and partial/tiled mode) ---
    band_lines = ""
    if new_look:
        center = _one(r"int2\(uint2\((" + _NAME + r"), (" + _NAME + r")\)\)", samples[4], "new-look center sample")
        band = [
            samples[4][:center.start()] + "(" + center[0] + " + int2(%d,%d))" % offset + samples[4][center.end():]
            for offset in ((-8, 0), (8, 0), (0, -8), (0, 8))
        ]
        band_lines = "\n".join(f"                    fsr412BandSamples[{i}] = {expr};" for i, expr in enumerate(band))

    new_body = _dedup_gaussian_samples(body, weights, samples)
    new_body = new_body.replace(params["tanh2"][0] + "\n", "", 1)
    new_body = new_body.replace(params["history"][0] + "\n", "", 1)

    blend_after = _one(
        r"^            float3 (" + _NAME + r") = [^\n]+ \+ \(float3\(r_reprojected_color\.read\(uint2\(" + re.escape(pixel_var) + r"\), 0u\)\.xyz\) \* " + re.escape(history_var) + r"\);$",
        new_body,
        "unified blend",
    )
    current_weight_after = _one(r"\* \(1\.0 - " + re.escape(history_var) + r"\)", blend_after[0], "unified current share")
    current_after = _factor(blend_after[0], current_weight_after.start())

    ct_and_act_start = new_body.index(body[tail["start"]:tail["start"] + 40])
    ct_and_act_end = new_body.index(params["sy"][0]) + len(params["sy"][0]) + 1
    raw_ct_only = tail["raw"].replace(rec[0] + "\n", "")
    # Indent raw CT lines by +4 spaces inside `if (!fsr412CellReuse)`.
    indented_raw = "".join("    " + ln if ln.strip() else ln for ln in raw_ct_only.splitlines(True))

    unified_ct_or_cache = f"""            Fsr412LossyRaw fsr412Raw;
            float {tanh2_var};
            float {sx_var};
            float {sy_var};
            float {history_var};
            int2 fsr412PreviousPixel = int2(0);
            if (!fsr412CellReuse)
            {{
{indented_raw}                {rec_arr}[{loop_var}] = float4(1.0) / (float4(1.0) + exp(-(float4({rec[3]}) * 0.0009765625)));
                float4 {filt[1]} = float4({filt[2]}) * 0.0009765625;
                float {params["filt_x"][1]} = {filt[1]}.x;
                float {params["exp_pos"][1]} = exp({params["filt_x"][1]});
                float {params["exp_neg"][1]} = exp(-{params["filt_x"][1]});
                {tanh2_var} = 2.0 * (({params["exp_pos"][1]} - {params["exp_neg"][1]}) / ({params["exp_pos"][1]} + {params["exp_neg"][1]}));
                {sx_var} = 2.0 / (1.0 + exp(-{filt[1]}.y));
                {sy_var} = 2.0 / (1.0 + exp(-{filt[1]}.z));
                {history_var} = 1.0 / (1.0 + exp(-{filt[1]}.w));
                fsr412Raw.recurrent = half4({rec_arr}[{loop_var}]);
                fsr412Raw.params = half4({tanh2_var}, {sx_var}, {sy_var}, {history_var});
            }}
            else
            {{
                fsr412PreviousPixel = int2({pixel_var}) + fsr412CellMotion;
                uint2 fsr412SafePrevious = uint2(clamp(fsr412PreviousPixel, int2(0), fsr412OutMax));
                fsr412Raw = Fsr412LossyPrevious[fsr412SafePrevious.y * MLSR_Optimized_Constants.width + fsr412SafePrevious.x];
                {rec_arr}[{loop_var}] = float4(fsr412Raw.recurrent);
                {tanh2_var} = float(fsr412Raw.params.x);
                {sx_var} = float(fsr412Raw.params.y);
                {sy_var} = float(fsr412Raw.params.z);
                {history_var} = float(fsr412Raw.params.w);
            }}
            fsr412OutCache[{loop_var}] = fsr412Raw;
"""
    new_body = new_body[:ct_and_act_start] + unified_ct_or_cache + new_body[ct_and_act_end:]

    if new_look:
        policy = f"""
            float3 fsr412Current = {current_after};
            float4 fsr412LookReproj = float4(r_reprojected_color.read(uint2({pixel_var}), 0u));
            float3 fsr412LookHistory = fsr412LookReproj.xyz;
            bool fsr412LookValid = (Fsr412LossyConstants.flags & 4u) != 0u && MLSR_Optimized_Constants.reset == 0u;
            float fsr412LookPreviousNote = fsr412LookValid ? fsr412LookReproj.w : 0.0f;
            if (!isfinite(fsr412LookPreviousNote)) fsr412LookPreviousNote = 0.0f;
            if (fsr412CellReuse)
            {{
                int fsr412SubX = int({pixel_var}.x & 1u);
                int fsr412SubY = int({pixel_var}.y & 1u);
                float3 fsr412HistorySamples[9];
                float3 fsr412HistoryMaximum = float3(0.0f);
                for (int ry = 0; ry < 3; ++ry)
                {{
                    for (int rx = 0; rx < 3; ++rx)
                    {{
                        float3 value = fsr412RepWin[(fsr412SubY + ry) * 4 + (fsr412SubX + rx)];
                        fsr412HistorySamples[ry * 3 + rx] = value;
                        fsr412HistoryMaximum = max(fsr412HistoryMaximum, value);
                    }}
                }}
                float3 fsr412BandSamples[4] = {{float3(0.0f), float3(0.0f), float3(0.0f), float3(0.0f)}};
                if (Fsr412LookArrival(fsr412Samples, fsr412HistoryMaximum) > 0.0f)
                {{
{band_lines}
                }}
                bool fsr412Uncovered = (fsr412CellOwn.w & 1u) == 0u || any(fsr412PreviousPixel < int2(0)) ||
                    fsr412PreviousPixel.x >= int(MLSR_Optimized_Constants.width) || fsr412PreviousPixel.y >= int(MLSR_Optimized_Constants.height);
                bool fsr412SurfDiff = false;
                if (!fsr412ZeroMotion)
                {{
                    int2 fsr412PrevCell = fsr412PreviousPixel / 2;
                    fsr412SurfDiff = Fsr412LossyDifferent(fsr412CellMotion, Fsr412LookMotion(fsr412PrevCell, Fsr412LossyMotion, MLSR_Optimized_Constants));
                    uint4 fsr412Surface = Fsr412LookDepthRecord(fsr412PrevCell, Fsr412LossyMotion, MLSR_Optimized_Constants);
                    fsr412Uncovered = fsr412Uncovered || ((fsr412CellDepth.w & fsr412Surface.w & 2u) != 0u &&
                        as_type<float>(fsr412Surface.z) < as_type<float>(fsr412CellDepth.z) && fsr412SurfDiff);
                }}
                bool fsr412OtherMotion = fsr412SurfDiff || fsr412NeighborDiff;
                bool fsr412AtRest = fsr412Still && !fsr412OtherMotion;
                {blend_var} = Fsr412LookResolve(fsr412Current, fsr412LookHistory, {history_var}, fsr412Samples,
                    fsr412HistorySamples, fsr412BandSamples, fsr412AtRest, fsr412OtherMotion,
                    fsr412Uncovered, fsr412CellDepthEdge, fsr412LookPreviousNote);
                if (fsr412Still && fsr412OtherMotion)
                    {rec_arr}[{loop_var}].z = max({rec_arr}[{loop_var}].z, 0.45f);
            }}
            {blend_var} = Fsr412LookFinish({blend_var}, fsr412Current, fsr412LookHistory, fsr412LookPreviousNote,
                fsr412LookValid, fsr412LookNotes[{loop_var}]);"""
    else:
        policy = f"""
            if (fsr412CellReuse)
            {{
                bool fsr412Uncovered = (fsr412CellOwn.w & 1u) == 0u || any(fsr412PreviousPixel < int2(0)) ||
                    fsr412PreviousPixel.x >= int(MLSR_Optimized_Constants.width) || fsr412PreviousPixel.y >= int(MLSR_Optimized_Constants.height);
                bool fsr412SurfDiff = false;
                if (!fsr412ZeroMotion)
                {{
                    uint4 fsr412Surface = Fsr412LossyReadMotion(fsr412PreviousPixel / 2, Fsr412LossyMotion, MLSR_Optimized_Constants);
                    fsr412SurfDiff = Fsr412LossyDifferent(fsr412CellMotion, as_type<int2>(fsr412Surface.xy));
                    fsr412Uncovered = fsr412Uncovered || ((fsr412CellOwn.w & fsr412Surface.w & 2u) != 0u &&
                        as_type<float>(fsr412Surface.z) < as_type<float>(fsr412CellOwn.z) && fsr412SurfDiff);
                }}
                bool fsr412OtherMotion = fsr412SurfDiff || fsr412NeighborDiff;
                bool fsr412AtRest = (fsr412CellOwn.w & 1u) != 0u && fsr412ZeroMotion && !fsr412OtherMotion;
                int fsr412SubX = int({pixel_var}.x & 1u);
                int fsr412SubY = int({pixel_var}.y & 1u);
                float fsr412Lo = 3.402823466e+38f;
                float fsr412Hi = -3.402823466e+38f;
                bool fsr412LumaFinite = true;
                for (int ry = 0; ry < 3; ++ry)
                {{
                    for (int rx = 0; rx < 3; ++rx)
                    {{
                        float v = fsr412RepLuma[(fsr412SubY + ry) * 4 + (fsr412SubX + rx)];
                        fsr412LumaFinite = fsr412LumaFinite && isfinite(v);
                        fsr412Lo = min(fsr412Lo, v);
                        fsr412Hi = max(fsr412Hi, v);
                    }}
                }}
                {blend_var} = Fsr412LossyResolve({current_after}, fsr412RepCenter[fsr412SubY * 2 + fsr412SubX], {history_var}, fsr412Samples,
                    fsr412Lo, fsr412Hi, fsr412LumaFinite, fsr412AtRest, fsr412OtherMotion, fsr412Uncovered);
            }}"""

    _require(new_body.count(blend_after[0]) == 1, "blend replacement boundary")
    new_body = new_body.replace(blend_after[0], blend_after[0] + policy, 1)

    text = text[:tail["begin"] + 1] + new_body + text[tail["end"]:]

    # Guard CNB loads with cell-level predicate computed once per 2x2 cell.
    guarded = f"        bool fsr412CellReuse = !Fsr412LossyFresh({cell_var} * uint2(2u), Fsr412LossyConstants, MLSR_Optimized_Constants);\n"
    guarded += "".join("        uint " + m[1] + " = 0u;\n" for m in loads)
    guarded += "        if (!fsr412CellReuse)\n        {\n"
    guarded += "".join("            " + m[1] + " = ScratchBuffer._m0[fsr412CacheWord + " + m[2] + "u];\n" for m in loads)
    guarded += "        }"
    text = text[:loads[0].start()] + guarded + text[loads[-1].end():]

    # Hoist cell motion probes and 4x4 reprojection window before the 4-pixel loop.
    if new_look:
        pre_loop = f"""        Fsr412LossyRaw fsr412OutCache[4];
        float fsr412LookNotes[4] = {{0.0f, 0.0f, 0.0f, 0.0f}};
        int2 fsr412OutMax = int2(MLSR_Optimized_Constants.width, MLSR_Optimized_Constants.height) - int2(1);
        uint4 fsr412CellOwn = uint4(0u);
        uint4 fsr412CellDepth = uint4(0u);
        int2 fsr412CellMotion = int2(0);
        bool fsr412ZeroMotion = false;
        bool fsr412NeighborDiff = false;
        bool fsr412Still = false;
        bool fsr412CellDepthEdge = false;
        float3 fsr412RepWin[16];
        if (fsr412CellReuse)
        {{
            int2 fsr412Cell = int2({cell_var});
            fsr412CellOwn = Fsr412LossyReadMotion(fsr412Cell, Fsr412LossyMotion, MLSR_Optimized_Constants);
            fsr412CellMotion = Fsr412LookMotion(fsr412Cell, Fsr412LossyMotion, MLSR_Optimized_Constants);
            fsr412ZeroMotion = all(fsr412CellMotion == int2(0));
            fsr412CellDepth = Fsr412LookDepthRecord(fsr412Cell, Fsr412LossyMotion, MLSR_Optimized_Constants);
            const int2 probes[8] = {{int2(3,0),int2(-3,0),int2(0,3),int2(0,-3),int2(8,0),int2(-8,0),int2(0,8),int2(0,-8)}};
            for (uint i = 0u; i < 8u; ++i)
                fsr412NeighborDiff = fsr412NeighborDiff || Fsr412LossyDifferent(fsr412CellMotion, Fsr412LookMotion(fsr412Cell + probes[i], Fsr412LossyMotion, MLSR_Optimized_Constants));
            fsr412Still = (fsr412CellOwn.w & 1u) != 0u && fsr412ZeroMotion;
            fsr412CellDepthEdge = !(fsr412Still && !fsr412NeighborDiff) && Fsr412LookDepthEdge(fsr412Cell, Fsr412LossyMotion, MLSR_Optimized_Constants);
            int2 fsr412BaseOut = int2({cell_var} * uint2(2u));
            for (int ry = 0; ry < 4; ++ry)
            {{
                for (int rx = 0; rx < 4; ++rx)
                {{
                    int2 coord = clamp(fsr412BaseOut + int2(rx - 1, ry - 1), int2(0), fsr412OutMax);
                    fsr412RepWin[ry * 4 + rx] = float3(r_reprojected_color.read(uint2(coord), 0u).xyz);
                }}
            }}
        }}
"""
        text = _rewrite_new_look_history_stores(text, body, loop_var, blend_var)
    else:
        pre_loop = f"""        Fsr412LossyRaw fsr412OutCache[4];
        int2 fsr412OutMax = int2(MLSR_Optimized_Constants.width, MLSR_Optimized_Constants.height) - int2(1);
        uint4 fsr412CellOwn = uint4(0u);
        int2 fsr412CellMotion = int2(0);
        bool fsr412ZeroMotion = false;
        bool fsr412NeighborDiff = false;
        float fsr412RepLuma[16];
        float3 fsr412RepCenter[4];
        if (fsr412CellReuse)
        {{
            Fsr412LossyCellMotion(int2({cell_var}), Fsr412LossyMotion, MLSR_Optimized_Constants,
                fsr412CellOwn, fsr412CellMotion, fsr412NeighborDiff);
            fsr412ZeroMotion = all(fsr412CellMotion == int2(0));
            int2 fsr412BaseOut = int2({cell_var} * uint2(2u));
            for (int ry = 0; ry < 4; ++ry)
            {{
                for (int rx = 0; rx < 4; ++rx)
                {{
                    int2 coord = clamp(fsr412BaseOut + int2(rx - 1, ry - 1), int2(0), fsr412OutMax);
                    float3 c = float3(r_reprojected_color.read(uint2(coord), 0u).xyz);
                    fsr412RepLuma[ry * 4 + rx] = Fsr412LossyLuma(c);
                    if (rx >= 1 && rx <= 2 && ry >= 1 && ry <= 2)
                        fsr412RepCenter[(ry - 1) * 2 + (rx - 1)] = c;
                }}
            }}
        }}
"""
    text = text.replace(tail["loop"][0], pre_loop + tail["loop"][0], 1)
    text = text.replace(origin_stmt[0], origin_stmt[0] + "\n" + _coalesced_cache_write(origin_var, offset_arr, guard=True), 1)
    return _receipt(text, marker)
