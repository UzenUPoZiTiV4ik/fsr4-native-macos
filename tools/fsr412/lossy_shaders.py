#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Independent Metal temporal policy around genuine imported Lite CT weights.

Apply PRE after signed_square/input_color_cache, POST after external_post_cnb,
input_color_cache/hair fix/post_tiling. No GPL tools/code or replacement neural
weights are used. This is NOT upstream lossy parity; partial mode is EXPERIMENTAL.

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
    stride. Raw record = {int4 recurrent; int4 filter}, 32B, before the exact
    original 1/1024 scale/sigmoid. Actual normal/UP bounds 53084/73070 exclude
    int16 packing. No narrowing, floating storage, or spare persistence bits.

Whole: use one cache aliased Previous==Current if desired. Full neural frame
writes Current only; reuse reads Previous only, NEVER writes Current or reads
CNB. Host permits at most one consecutive skip, with valid neural cache.
PRE reuse_only specializes a separate pipeline, but returns ONLY when valid
whole reuse is requested, AFTER original history reprojection/motion recording.
Partial: ALWAYS distinct Previous/Current. Fresh mask is checker tile parity
((x/tileSize+y/tileSize)&1)==phase OR x<2/y<2/x>=width-2/y>=height-2. Native producer
must use exactly this mask, with 32-output-pixel halo at intermediate passes
and zero CNB halo. Fresh cells load CNB and compute genuine CT; other pixels
read previous raw parameters at pixel+motion. EVERY pixel writes Current, so
its age is one frame. Cached final pixel params already include CT parity.

Update: PRE -> buffer barrier -> native fresh inference/CNB -> POST -> buffer
barrier/completion -> swap partial planes. Serialize this chain or fence all
in-flight frames. Invalidate previousValid on reset/viewport/preset/tier/color
space/exposure discontinuity, mode/tile-size switch, disabled inference chain,
or failed dispatch;
force a full neural frame. No cache clears needed after a complete fresh frame.
OOB/invalid motion uses CLAMPED previous raw params plus current-only resolve,
not a CNB read on a frame/cell that has no inference. Fresh pixels retain the
original filter/blend/output math. Host must reject odd logical extents or use
an even aligned viewport consistently with the original truncating dispatch.

No two-frame color signs or slow-fade repair in this initial Lite profile:
raw8 consumes the whole 32B record. Legacy enhanced temporal mode has separate
sign history. Correlated noise, 2x2 motion, depth-owner approximation, genuine
single-pixel sparks being rejected, and tile boundaries remain quality risks.
Additional reuse-only sample/history probes and very large caches cost memory
and bandwidth; no performance/quality claim without game-scene GPU tests.
"""
from __future__ import annotations

import hashlib
import math
import re

REVISION = 1
CACHE_BYTES_PER_PIXEL = 32
MOTION_BYTES_PER_CELL = 16
CONSTANT_BYTES = 16
PARTIAL_TILE_SIZE = 512
PARTIAL_TILE_SIZES = (128, 256, 512, 1024, 2048)
TIERS = (1080, 2160, 4320)
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
    depth = 0
    for i in range(start, len(text)):
        if text[i] == left:
            depth += 1
        elif text[i] == right:
            depth -= 1
            if depth == 0:
                return i
    raise ValueError("Lite lossy: unbalanced expression")


def _args(text):
    depth, begin, parts = 0, 0, []
    for i, c in enumerate(text):
        if c in "([":
            depth += 1
        elif c in ")]":
            depth -= 1
        elif c == "," and depth == 0:
            parts.append(text[begin:i].strip()); begin = i + 1
    parts.append(text[begin:].strip())
    return parts


def _factor(text, end):
    """One unchanged working-color factor before an imported Gaussian weight."""
    while text[end - 1].isspace():
        end -= 1
    stop = end
    if text[end - 1] == ")":
        depth, start = 1, end - 2
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
    loop = _one(r"^        for \(int ("+_NAME+r") = 0; \1 < 4; \1\+\+\)$", text, "four-pixel loop")
    begin = text.index("{", loop.end())
    end = _close(text, begin, "{", "}")
    body = text[begin + 1:end]
    recurrent = _one(r"^            ("+_NAME+r")\[("+_NAME+r")\] = float4\(1\.0\) / \(float4\(1\.0\) \+ exp\(-\(float4\(("+_NAME+r")\) \* 0\.0009765625\)\)\);$", body, "raw recurrent sigmoid")
    _require(recurrent[2] == loop[1], "recurrent phase")
    filt = _one(r"^            float4 ("+_NAME+r") = float4\(("+_NAME+r")\) \* 0\.0009765625;$", body, "raw filter scale")
    _require(recurrent.end() < filt.start(), "CT tail order")
    declarations = list(re.finditer(r"^            int4 "+_NAME+r";$", body[:recurrent.start()], re.M))
    _require(bool(declarations), "raw recurrent declaration")
    start = declarations[0].start()
    raw = body[start:filt.start()]
    biases = re.findall(r"^            "+_NAME+r" = int4\((-?\d+), (-?\d+), (-?\d+), (-?\d+)\);$", raw, re.M)
    _require(len(biases) == 2, "CT biases")
    table = _one(r"^constant spvUnsafeArray<uint, 128> ("+_NAME+r") = spvUnsafeArray<uint, 128>\(\{(.*?)\}\);$", text, "CT weight table")
    words = [int(v) for v in re.findall(r"(\d+)u", table[2])]
    _require(len(words) == 128 and all(0 <= x <= 0xffffffff for x in words), "CT weight literals")
    _require(len(re.findall(re.escape(table[1])+r"\[", raw)) == 32, "CT column operations")
    phase = _one(r"^            uint ("+_NAME+r") = \(128u \* uint\(("+_NAME+r")\[("+_NAME+r")\]\.x\)\) \+ \(256u \* uint\(\2\[\3\]\.y\)\);$", body, "CT phase offsets")
    _require(phase[3] == loop[1], "CT phase loop")
    _require(re.search(r"= "+re.escape(phase[1])+r" \+ 64u;", raw) and
             re.search(r"= "+re.escape(phase[1])+r";", raw), "CT half offsets")
    weights = [((word >> shift) & 255) for word in words for shift in (0,8,16,24)]
    weights = [x if x < 128 else x - 256 for x in weights]
    bias = [tuple(map(int,b)) for b in biases]
    bounds = []
    for p in range(4):
        # Table is filter first, recurrent second; execution is the reverse.
        for half in range(2):
            for ch in range(4):
                col = weights[p*128+half*64+ch*16:p*128+half*64+ch*16+16]
                bounds.append(128*sum(abs(x) for x in col)+abs(bias[1-half][ch]))
    _require(max(bounds) <= 2147483647, "CT int32 bound overflow")
    return dict(loop=loop, begin=begin, end=end, body=body, rec=recurrent,
                filt=filt, start=start, raw=raw, phase=phase, bounds=tuple(bounds))


def cache_layout(text):
    """Actual column bounds, not an assumed int16 format; works on original POST."""
    tail = _tail(text)
    return {"cache_bytes_per_pixel": CACHE_BYTES_PER_PIXEL,
            "motion_bytes_per_cell": MOTION_BYTES_PER_CELL,
            "max_accumulator_bound": max(tail["bounds"]),
            "column_bounds": tail["bounds"], "int16_supported": max(tail["bounds"]) <= 32767,
            "schema": "raw_recurrent_int4_filter_int4_v1"}


def cache_buffer_bytes(width, height):
    _require(isinstance(width,int) and isinstance(height,int) and
             not isinstance(width,bool) and not isinstance(height,bool) and
             2 <= width <= 7680 and 2 <= height <= 4320 and (width | height) & 1 == 0,
             "cache requires an even positive aligned output viewport")
    return width * height * CACHE_BYTES_PER_PIXEL


def fresh_pixel(pixel, size, flags=0, tile_size=512, phase=0, reset=False):
    """CPU host/native-gate specification; outer two pixels are always fresh in partial."""
    valid = (flags & ~7 == 0 and flags & 4 and not reset and
             min(size) >= 2 and (size[0] | size[1]) & 1 == 0)
    mode = (flags & 3) if valid else 0
    if mode == 1:
        return False
    if mode != 2 or tile_size not in PARTIAL_TILE_SIZES or phase not in (0,1):
        return True
    x,y = pixel; width,height = size
    return x < 2 or y < 2 or x >= width-2 or y >= height-2 or ((x//tile_size+y//tile_size)&1) == phase


def sanitized_motion(velocity, size, invalid=False, reset=False):
    """CPU round-even/safety specification; no float->int overflow/NaN conversion."""
    if invalid or reset or len(velocity) != 2 or not all(math.isfinite(v) for v in velocity):
        return (0,0), False
    pixels = tuple(v*s for v,s in zip(velocity,size))
    if not all(math.isfinite(v) and abs(v) <= 254 for v in pixels):
        return (0,0), False
    pixels = tuple(0 if abs(v) <= .03125 else v for v in pixels)
    return tuple(round(v) for v in pixels), True


COMMON = """
// Independent MIT Lite lossy ABI, not upstream release parity.
struct Fsr412LossyConstants { uint flags; uint tileSize; uint phase; uint reserved; };
struct Fsr412LossyRaw { int4 recurrent; int4 filter; };
static inline uint Fsr412LossyMode(constant Fsr412LossyConstants& control,
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

PRE = """
static inline void Fsr412LossyRecord(uint2 pixel, float2 velocity, bool invalid,
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

POST = """
static inline uint4 Fsr412LossyReadMotion(int2 cell, const device uint4* records,
    constant type_MLSR_Optimized_Constants& frame)
{
    cell = clamp(cell, int2(0), int2(frame.width / 2u, frame.height / 2u) - int2(1));
    return records[uint(cell.y) * (frame.width / 2u) + uint(cell.x)];
}
static inline bool Fsr412LossyDifferent(int2 a, int2 b) { return any(abs(a-b) > int2(1)); }
static inline void Fsr412LossyClassify(uint2 pixel, const device uint4* records,
    constant type_MLSR_Optimized_Constants& frame, thread bool& atRest,
    thread bool& otherMotion, thread bool& uncovered)
{
    int2 cell = int2(pixel / 2u);
    uint4 own = Fsr412LossyReadMotion(cell, records, frame);
    int2 motion = as_type<int2>(own.xy);
    int2 previous = int2(pixel) + motion;
    uncovered = (own.w & 1u) == 0u || any(previous < int2(0)) ||
        previous.x >= int(frame.width) || previous.y >= int(frame.height);
    uint4 surface = Fsr412LossyReadMotion(previous / 2, records, frame);
    otherMotion = Fsr412LossyDifferent(motion, as_type<int2>(surface.xy));
    // Approximation only: current motion/depth plane is not a previous-owner scatter plane.
    uncovered = uncovered || ((own.w & surface.w & 2u) != 0u &&
        as_type<float>(surface.z) < as_type<float>(own.z) && otherMotion);
    const int2 probes[12] = {int2(1,0),int2(-1,0),int2(0,1),int2(0,-1),
        int2(3,0),int2(-3,0),int2(0,3),int2(0,-3),int2(8,0),int2(-8,0),int2(0,8),int2(0,-8)};
    for (uint i=0u;i<12u;++i)
        otherMotion = otherMotion || Fsr412LossyDifferent(motion,
            as_type<int2>(Fsr412LossyReadMotion(cell+probes[i],records,frame).xy));
    atRest = (own.w & 1u) != 0u && all(motion == int2(0)) && !otherMotion;
}
static inline float Fsr412LossyLuma(float3 color) { return dot(color,float3(.2126,.7152,.0722)); }
static inline float3 Fsr412LossyResolve(float3 current, float3 history, float historyShare,
    thread const float3* samples, uint2 pixel, texture2d<half> reprojection,
    constant type_MLSR_Optimized_Constants& frame, bool atRest, bool otherMotion, bool uncovered)
{
    float3 minimum = samples[0], maximum = samples[0];
    for (uint i=1u;i<9u;++i) { minimum=min(minimum,samples[i]); maximum=max(maximum,samples[i]); }
    float3 span = maximum-minimum;
    float3 difference = abs(current-history);
    bool changed = any(history < minimum) || any(history > maximum) || any(difference > span*.25f+.05f);
    for (uint ch=0u;ch<3u;++ch)
        changed = changed || (abs(minimum[ch]+maximum[ch]-2.0f*history[ch]) > span[ch]*.85f &&
                              difference[ch] > span[ch]*.05f+.01f);
    float freshShare = 1.0f-historyShare;
    float weight = freshShare*freshShare;
    if (atRest && !changed) weight *= saturate((freshShare-.95f)/.045f);
    float3 safeHistory = otherMotion ? clamp(history,minimum,maximum) : history;
    float3 result = current*weight+safeHistory*(1.0f-weight);
    // A sample must have support on BOTH axes. Lone pixels and thin rain do not win.
    float support = min(Fsr412LossyLuma(samples[4]),
        min(max(Fsr412LossyLuma(samples[3]),Fsr412LossyLuma(samples[5])),
            max(Fsr412LossyLuma(samples[1]),Fsr412LossyLuma(samples[7]))));
    float lo=3.402823466e+38f, hi=-3.402823466e+38f;
    for (int y=-1;y<=1;++y) for (int x=-1;x<=1;++x) {
        int2 h=clamp(int2(pixel)+int2(x,y),int2(0),int2(frame.width,frame.height)-int2(1));
        float value=Fsr412LossyLuma(float3(reprojection.read(uint2(h),0u).xyz));
        if (!isfinite(value)) return uncovered ? current : result;
        lo=min(lo,value); hi=max(hi,value);
    }
    float threshold=clamp(.08f+(hi-lo)*.5f,.08f,.25f);
    float spark=isfinite(support) ? saturate((support-hi-threshold)/threshold) : 0.0f;
    result=mix(result,current,spark);
    bool darkEdge=!atRest && all(history <= max(current*.1f,float3(0.0)));
    return uncovered || darkEdge ? current : result;
}
"""


def _prepare(text, tier, stage, extra=""):
    _require(tier in TIERS, "unknown tensor tier")
    marker = "// FSR412_LOSSY_V1_%s tier=%d %s" % (stage.upper(),tier,extra)
    if text.startswith(marker+"\n"):
        # Integrity is checked on repeated application, including generated body.
        receipt = re.match(re.escape(marker)+r"\n// payload-sha256=([0-9a-f]{64})\n",text)
        _require(receipt is not None, "missing existing transform receipt")
        _require(hashlib.sha256(text[receipt.end():].encode()).hexdigest() == receipt[1], "modified existing transform")
        return None, marker
    _require("FSR412_LOSSY_" not in text and "Fsr412Lossy" not in text, "existing/colliding lossy transform")
    kernel = _one(r"kernel void (\w*fsr4_model_v07_fp8_no_scale_"+stage+r"pass)\(([^\n]*)\)\n\{\n",text,"kernel")
    _require(len(re.findall(r"\bkernel\s+void\b",text)) == 1, "multiple kernels")
    used = set(map(int,re.findall(r"\[\[buffer\((\d+)\)\]\]",kernel[2])))
    free = [x for x in range(8) if x not in used]
    need = 2 if stage == "pre" else 4
    _require(len(free) >= need, "not enough buffer indices <=7")
    bindings = ["constant Fsr412LossyConstants& Fsr412LossyConstants [[buffer(%d)]]" % free[0],
                "device uint4* Fsr412LossyMotion [[buffer(%d)]]" % free[1]]
    if stage == "post":
        bindings[1] = "const " + bindings[1]
        bindings += ["const device Fsr412LossyRaw* Fsr412LossyPrevious [[buffer(%d)]]" % free[2],
                     "device Fsr412LossyRaw* Fsr412LossyCurrent [[buffer(%d)]]" % free[3]]
    signature_end = kernel.end()-len(")\n{\n")
    text = text[:signature_end]+", "+", ".join(bindings)+text[signature_end:]
    text = text[:kernel.start()]+COMMON+(PRE if stage == "pre" else POST)+text[kernel.start():]
    return text, marker


def _receipt(text, marker):
    return marker+"\n// payload-sha256="+hashlib.sha256(text.encode()).hexdigest()+"\n"+text


def apply_lossy_pre(text, tier, low_res_mv=True, depth_inverted=False, reuse_only=False):
    original = text
    extra="lm=%d di=%d reuse-only=%d" % (bool(low_res_mv),bool(depth_inverted),bool(reuse_only))
    text, marker = _prepare(text,tier,"pre",extra)
    if text is None:
        return original
    velocity = _one(r"^        float2 ("+_NAME+r");\n        bool ("+_NAME+r");\n        do\n        \{",text,"sanitized velocity pair")
    write = _one(r"^        rw_reprojected_color\.write\([^\n;]+, uint2\(("+_NAME+r")\)\);$",text,"history reprojection store")
    _require(velocity.start() < write.start(), "motion before reprojection")
    _require(text.count(velocity[1]+" = float2(0.0);") == 1 and
             text.count(velocity[2]+" = true;") == 1 and text.count(velocity[2]+" = false;") == 1,
             "velocity sanitization")
    depth = "0.0f"
    if low_res_mv:
        _require("texture2d<float> r_depth" in text, "low-resolution motion requires depth")
        depth_tests = re.findall(r"bool "+_NAME+r" = "+_NAME+r" ([<>]) "+_NAME+r";",text[:velocity.start()])
        _require(len(depth_tests) == 7 and all(op == (">" if depth_inverted else "<") for op in depth_tests),
                 "depth inversion variant mismatch")
        sample = _one(r"r_velocity\.read\(",text,"velocity load")
        closing = _close(text,sample.end()-1)
        args = _args(text[sample.end():closing])
        _require(len(args) == 2 and args[1] == "0u", "velocity sample mip")
        depth = ("r_depth.read(uint2(clamp(int2("+args[0]+"), int2(0), "
                 "int2(MLSR_Optimized_Constants.width_lr, MLSR_Optimized_Constants.height_lr)-int2(1))),0u).x")
        if depth_inverted:
            depth = "-("+depth+")"
    else:
        _require("r_depth.read(" not in text, "full-resolution motion variant mismatch")
    injected = ("\n        Fsr412LossyRecord("+write[1]+", "+velocity[1]+", "+velocity[2]+", "+depth+
                ", Fsr412LossyMotion, MLSR_Optimized_Constants);\n")
    if reuse_only:
        injected += "        if (Fsr412LossyMode(Fsr412LossyConstants, MLSR_Optimized_Constants) == 1u) return;\n"
    text = text[:write.end()]+injected+text[write.end():]
    _require(text.find("ScratchBuffer._m0[") > text.find("Fsr412LossyRecord("+write[1]), "tensor store precedes reuse return")
    return _receipt(text,marker)


def apply_lossy_post(text,tier):
    original = text
    text, marker = _prepare(text,tier,"post")
    if text is None:
        return original
    tail = _tail(text)
    body=tail["body"]; rec=tail["rec"]; filt=tail["filt"]
    output = _one(r"^            uint2 ("+_NAME+r") = \(("+_NAME+r") \* uint2\(2u\)\) \+ uint2\(("+_NAME+r")\[("+_NAME+r")\]\);$",body,"output pixel")
    _require(output[4] == tail["loop"][1], "output phase")
    cache = _one(r"^        uint fsr412CacheWord = \(("+_NAME+r")\.y \* (\d+)u \+ \1\.x \* 16u\) >> 2u;$",text,"external CNB cache")
    _require(cache[1] == output[2] and int(cache[2]) == tier*8//9*16, "CNB tier/coordinate")
    loads = list(re.finditer(r"^        uint ("+_NAME+r") = ScratchBuffer\._m0\[fsr412CacheWord \+ (\d)u\];$",text,re.M))
    _require(len(loads) == 4 and [int(x[2]) for x in loads] == list(range(4)), "CNB word loads")
    _require(text.count("ScratchBuffer._m0[") == 4, "unexpected CNB/scratch reads")
    guarded = "".join("        uint "+m[1]+" = 0u;\n" for m in loads)
    guarded += "        if (Fsr412LossyFresh("+output[2]+" * uint2(2u), Fsr412LossyConstants, MLSR_Optimized_Constants))\n        {\n"
    guarded += "".join("            "+m[1]+" = ScratchBuffer._m0[fsr412CacheWord + "+m[2]+"u];\n" for m in loads)+"        }"
    text=text[:loads[0].start()]+guarded+text[loads[-1].end():]
    # Original nine-tap arithmetic/transfer/exposure remains byte-identical.
    weights=list(re.finditer(r"^            float ("+_NAME+r") = exp\(\(-2\.2634675502777099609375\) \* [^\n]+;$",body,re.M))
    _require(len(weights)==9,"Gaussian tap count")
    samples=[]
    for weight in weights:
        use=_one(r"\* "+re.escape(weight[1])+r"\b",body,"Gaussian sample weight")
        samples.append(_factor(body,use.start()))
    history = _one(r"^            float ("+_NAME+r") = 1\.0 / \(1\.0 \+ exp\(-"+re.escape(filt[1])+r"\.w\)\);$",body,"history share")
    blend = _one(r"^            float3 ("+_NAME+r") = [^\n]+ \+ \(float3\(r_reprojected_color\.read\(uint2\("+re.escape(output[1])+r"\), 0u\)\.xyz\) \* "+re.escape(history[1])+r"\);$",body,"original blend")
    current_weight = _one(r"\* \(1\.0 - "+re.escape(history[1])+r"\)",blend[0],"current share")
    current = _factor(blend[0],current_weight.start())
    raw=tail["raw"].replace(rec[0]+"\n", "")
    # Strip ONLY the raw CT block, not the filter or its four output writes.
    code = """            bool fsr412PixelReuse = !Fsr412LossyFresh(@pixel@, Fsr412LossyConstants, MLSR_Optimized_Constants);
            Fsr412LossyRaw fsr412Raw;
            if (!fsr412PixelReuse)
            {
@raw@                fsr412Raw.recurrent = @rec@;
                fsr412Raw.filter = @filter@;
            }
            else
            {
                uint4 fsr412Own = Fsr412LossyReadMotion(int2(@pixel@ / 2u), Fsr412LossyMotion, MLSR_Optimized_Constants);
                int2 fsr412PreviousPixel = int2(@pixel@) + as_type<int2>(fsr412Own.xy);
                uint2 fsr412SafePrevious = uint2(clamp(fsr412PreviousPixel, int2(0),
                    int2(MLSR_Optimized_Constants.width, MLSR_Optimized_Constants.height)-int2(1)));
                fsr412Raw = Fsr412LossyPrevious[fsr412SafePrevious.y * MLSR_Optimized_Constants.width + fsr412SafePrevious.x];
            }
            if (Fsr412LossyMode(Fsr412LossyConstants, MLSR_Optimized_Constants) != 1u)
                Fsr412LossyCurrent[@pixel@.y * MLSR_Optimized_Constants.width + @pixel@.x] = fsr412Raw;
@sigmoid@
@scale@
"""
    replacements={"pixel":output[1],"raw":raw,"rec":rec[3],"filter":filt[2],
                  "sigmoid":rec[0].replace("float4("+rec[3]+")","float4(fsr412Raw.recurrent)"),
                  "scale":filt[0].replace("float4("+filt[2]+")","float4(fsr412Raw.filter)")}
    for key,value in replacements.items(): code=code.replace("@"+key+"@",value)
    old=body[tail["start"]:filt.end()]+"\n"
    _require(text.count(old)==1,"raw CT replacement boundary")
    text=text.replace(old,code,1)
    policy = "\n            if (fsr412PixelReuse)\n            {\n                float3 fsr412Samples[9] = {"+", ".join(samples)+"};\n"
    policy += "                bool fsr412AtRest, fsr412OtherMotion, fsr412Uncovered;\n"
    policy += "                Fsr412LossyClassify("+output[1]+", Fsr412LossyMotion, MLSR_Optimized_Constants, fsr412AtRest, fsr412OtherMotion, fsr412Uncovered);\n"
    policy += "                "+blend[1]+" = Fsr412LossyResolve("+current+", float3(r_reprojected_color.read(uint2("+output[1]+"),0u).xyz), "+history[1]+", fsr412Samples, "+output[1]+", r_reprojected_color, MLSR_Optimized_Constants, fsr412AtRest, fsr412OtherMotion, fsr412Uncovered);\n            }"
    _require(text.count(blend[0])==1,"blend replacement boundary")
    text=text.replace(blend[0],blend[0]+policy,1)
    return _receipt(text,marker)
