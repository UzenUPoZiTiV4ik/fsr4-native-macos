#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Optional, independently written temporal repairs for legacy Lossy/lossy2.

Not upstream release parity. Only prose in docs/lossy-rules.md and
research/frame-skip/README.md was consulted; no GPL rewriter is imported.
Apply PRE after apply_lossy_pre (before or after reuse_only_pre). Apply POST
LAST, after apply_lossy_post and any color-cache/cell-classification transforms,
but before HLSL portability conversion. Unoptimized POST is supported too.
Generate mode 2 explicitly with apply_temporal_quality_post(..., fast_notes=True).
The default fast_notes=False leaves the conservative generated shader unchanged;
flags=7 alone on that shader safely falls back to reference notes.
Do not call legacy source transforms again after this transform.

Host ABI (all dimensions are the EVEN, aligned b1 output viewport):
* b3, 16 bytes: uint flags, historyValid, cellWidth, cellHeight, in that order.
  flags bit 0 enables fine motion/persistence/fade; bit 1 enables the spark
  check too; bit 2 (4) opts into fast notes on cached one-pixel/lane POST.
  Default flags=3 preserves conservative reference notes; flags=7 enables
  fast notes plus sparks; flags=0 disables quality. Other bits must be zero.
  historyValid=0 on reset/first frame, 1 after one completed
  enabled frame. cellWidth=width/2; cellHeight=height/2, NOT the tensor tier.
* u14 previous, u15 current: DIFFERENT dense RWStructuredBuffer<uint4>,
  16 bytes per cell, row-major with cellWidth stride; no scratch aliases.
  xy: signed round-to-even output-pixel motion, clamped to +/-254, as uint.
  w: bit 0 valid motion/history; bit 1 motion fits +/-31 before clamping.
  z: four 8-bit pixel notes, phase=xParity+2*yParity. In each note, bits
  0..1=R, 2..3=G, 4..5=B, 6..7=luminance. Each sign is 0=neutral,
  1=positive, 2=negative. 3 is reserved and never matches. RGB needs the
  noise floor; luminance uses the smaller relative fade floor.

Update sequence, EVERY enabled frame, full AND skipped:
1. Bind last completed buffer at u14 and the other buffer at u15 in both
   PRE/POST. PRE writes one full uint4 per cell from its top-left pixel.
2. Buffer memory barrier PRE -> POST. Legacy u13 still needs its usual
   clear/barrier on skipped frames, unchanged.
3. POST reads current.xy/w as SCALARS only (immutable after PRE), and reads
   previous.z/w only after historyValid and motion/bounds checks. Its sole
   even/even pixel owner writes current.z ONCE, with four independently
   resampled float32 differences in conservative mode. Fast notes reuse each
   pixel's already-computed Gaussian up/history, gather the four 8-bit notes
   with Wave32 shuffles, then let only the even/even owner store current.z.
   All VALID pixel lanes participate BEFORE the owner predicate, including
   reset/invalid-motion lanes (neutral notes). Even aligned dimensions keep
   all four cell members active at partial tile edges. Unknown wave sizes or
   lane/group mapping fall back to the original reference record function.
   Uncached four-output-pixels/lane POST always uses the reference path.
   Neither path reads current.z, modifies xy/w, or writes a full uint4. No
   atomic or extra group barrier.
   These are disjoint scalar addresses even when adjacent cells are read.
   Preserve scalar component stores in backend lowering, not vector RMW.
4. Buffer barrier/completion before next PRE; swap buffer roles ONLY after
   successful enabled PRE+POST. Serialize frames using these buffers, or
   supply separate buffers/fences for every in-flight history chain.

Reset, flag/preset/colorspace/tier/viewport switches, exposure discontinuity,
failed dispatch, and any disabled frame invalidate historyValid. Force a full
legacy frame until its CNB cache is valid too. No new-buffer clear is needed
when historyValid=0: PRE/POST overwrite the entire active dense viewport.
Do not set historyValid=1 if only part of the viewport was dispatched.
When opted out at generation (enabled=False), return the original source
byte-for-byte. Runtime flags=0 causes no new-buffer access and preserves the
legacy output; b3 and legal resource bindings still belong to that pipeline.
Full-frame output remains legacy, even when enabled; only notes are recorded.

Risks: 2x2 top-left motion still misses subpixel edges; CNB remains 4x4.
Fast notes are an explicitly DIFFERENT policy: model Gaussian up instead of
an independent bilinear reference. Reference agreement in Resolve remains
unchanged, but previous signs may differ; invalidate history on switching.
The fast recording path removes up to 16 current + 4 history texture loads
per cell (5 per output pixel), not the separate Resolve/spark probes. It adds
four wave shuffles per pixel and scalar motion-validity reads. This is an
estimated access reduction, not a measured speedup or GPU quality guarantee.
Bilinear reference notes intentionally differ from the network's anisotropic
filter and require agreement with its current result before a rule fires.
Correlated noise can repeat signs; bounded floors and fade band limit, not
eliminate, that risk. Spark checks deliberately reject one-pixel streaks,
and may lose genuine tiny sparks. Extra resamples/history probes cost GPU
bandwidth. No quality/performance claim without real-scene GPU validation.
"""

from dataclasses import dataclass
import hashlib
import math
import re

CONSTANT_BYTES = 16
RECORD_BYTES_PER_CELL = 16
DEFAULT_FLAGS = 3
ENABLE_FLAG = 1
SPARK_FLAG = 2
FAST_NOTES_FLAG = 4
FAST_FLAGS = DEFAULT_FLAGS | FAST_NOTES_FLAG
FAST_NOTES_WAVE_SIZE = 32
TIERS = (1080, 2160, 4320)


@dataclass(frozen=True)
class QualitySettings:
    """Compile-time bounds in POST's working color domain, not display RGB."""
    noise_floor: float = 0.02
    fade_floor: float = 0.0001
    fade_lower: float = 0.0015
    fade_upper: float = 0.01
    spark_floor: float = 0.08
    spark_span: float = 0.5
    spark_cap: float = 0.25

    def __post_init__(self):
        bounds = ((self.noise_floor, 0.005, 0.05),
                  (self.fade_floor, 0.0001, 0.002),
                  (self.fade_lower, 0.001, 0.005),
                  (self.fade_upper, 0.005, 0.02),
                  (self.spark_floor, 0.04, 0.2),
                  (self.spark_span, 0.0, 1.0),
                  (self.spark_cap, 0.08, 0.5))
        if any(not math.isfinite(v) or not lo <= v <= hi for v, lo, hi in bounds):
            raise ValueError("temporal quality threshold outside safe bounds")
        if self.fade_lower >= self.fade_upper or self.spark_floor > self.spark_cap:
            raise ValueError("temporal quality threshold bands are reversed")


DEFAULT_SETTINGS = QualitySettings()


def record_buffer_bytes(width, height):
    """Bytes for ONE plane; host allocates two, plus the unchanged legacy ABI."""
    if (not isinstance(width, int) or not isinstance(height, int) or
            isinstance(width, bool) or isinstance(height, bool) or
            width < 2 or height < 2 or (width | height) & 1):
        raise ValueError("temporal quality needs a positive even output viewport")
    if width > 7680 or height > 4320:
        raise ValueError("temporal quality viewport exceeds the largest tensor tier")
    return (width // 2) * (height // 2) * RECORD_BYTES_PER_CELL


def luminance(rgb):
    return sum(a * b for a, b in zip(rgb, (0.2126, 0.7152, 0.0722)))


def _sign(difference, floor):
    if not math.isfinite(difference):
        return 0
    return 1 if difference > floor else 2 if difference < -floor else 0


def pixel_signs(current, history, settings=DEFAULT_SETTINGS):
    """CPU specification for the unsigned 8-bit GPU note."""
    if len(current) != 3 or len(history) != 3:
        raise ValueError("a temporal note needs RGB triples")
    if not all(math.isfinite(v) for v in (*current, *history)):
        return 0
    note = sum(_sign(c - h, settings.noise_floor) << (2 * ch)
               for ch, (c, h) in enumerate(zip(current, history)))
    y = luminance(history)
    return note | (_sign(luminance(current) - y,
                         settings.fade_floor + abs(y) * settings.fade_lower) << 6)


def pack_signs(current, history, settings=DEFAULT_SETTINGS):
    if len(current) != 4 or len(history) != 4:
        raise ValueError("a 2x2 temporal record needs four RGB pairs")
    return sum(pixel_signs(c, h, settings) << (8 * phase)
               for phase, (c, h) in enumerate(zip(current, history)))


def matching_color_sign(a, b):
    return any(((a >> shift) & 3) in (1, 2) and
               ((a >> shift) & 3) == ((b >> shift) & 3) for shift in (0, 2, 4))


def slow_fade(current, history, previous_note, settings=DEFAULT_SETTINGS):
    note = pixel_signs(current, history, settings)
    sign = (note >> 6) & 3
    drift = abs(luminance(current) - luminance(history))
    return (sign in (1, 2) and sign == ((previous_note >> 6) & 3) and
            drift <= settings.fade_floor + abs(luminance(history)) * settings.fade_upper)


def spark_response(center, left, right, above, below, history_min, history_max,
                   settings=DEFAULT_SETTINGS):
    """Scalar luminance check: isolated/one-axis samples cannot win."""
    if not all(math.isfinite(v) for v in
               (center, left, right, above, below, history_min, history_max)):
        return 0.0
    if history_max < history_min:
        return 0.0
    support = min(center, max(left, right), max(above, below))
    threshold = min(settings.spark_cap, max(settings.spark_floor,
                    settings.spark_floor + (history_max - history_min) * settings.spark_span))
    return max(0.0, min(1.0, (support - history_max - threshold) / threshold))


def _replace(source, before, after):
    if source.count(before) != 1:
        raise ValueError("temporal quality expected one expression: " + before[:100])
    return source.replace(before, after)


def _prepare(source, tier, stage, enabled, settings, fast_notes=False):
    if not enabled:
        return source, None
    if tier not in TIERS:
        raise ValueError("unknown temporal quality tensor tier")
    if not isinstance(settings, QualitySettings):
        raise ValueError("temporal quality settings must be QualitySettings")
    digest = hashlib.sha256(repr(settings).encode()).hexdigest()[:16]
    suffix = " fast-notes=1" if fast_notes else ""
    marker = "// fsr411-generated-temporal-quality-%s: tier=%d settings=%s%s\n" % (stage, tier, digest, suffix)
    if marker in source:
        if source.count(marker) != 1:
            raise ValueError("duplicated temporal quality marker")
        return source, None
    if "fsr411-generated-temporal-quality-" in source:
        raise ValueError("temporal quality already has another policy/stage")
    if not re.search(r"// fsr411-generated-lossy-%s: tier=%d(?:\s|$)" % (stage, tier), source):
        raise ValueError("temporal quality requires matching legacy lossy " + stage.upper())
    # Fail before introducing collisions into either resource or symbol ABI.
    if re.search(r"register\(\s*(?:[uU]1[45]|[bB]3)\s*\)", source) or "Fsr411Quality" in source:
        raise ValueError("temporal quality ABI resources/symbols are already occupied")
    return source, marker


DECLARATIONS = r"""
// Independent MIT temporal-quality ABI. PRE -> POST buffer barrier required.
cbuffer Fsr411QualityConstants : register(b3)
{
    uint qualityFlags;
    uint qualityHistoryValid;
    uint qualityCellWidth;
    uint qualityCellHeight;
};
RWStructuredBuffer<uint4> Fsr411QualityPrevious : register(u14);
RWStructuredBuffer<uint4> Fsr411QualityCurrent : register(u15);

bool Fsr411QualityEnabled()
{
    // Refuse an inconsistent host extent without touching a new buffer.
    return (qualityFlags & 1u) != 0u && (width | height) % 2u == 0u &&
           qualityCellWidth == width / 2u && qualityCellHeight == height / 2u &&
           qualityCellWidth != 0u && qualityCellHeight != 0u;
}
"""

PRE_HELPERS = r"""
void Fsr411QualityRecordMotion(uint2 pixel, float2 velocity, bool invalidMotion)
{
    if (!Fsr411QualityEnabled() || ((pixel.x | pixel.y) & 1u) != 0u)
        return;
    uint2 cell = pixel >> 1u;
    if (cell.x >= qualityCellWidth || cell.y >= qualityCellHeight)
        return;
    bool valid = !invalidMotion && reset == 0u && all(isfinite(velocity));
    // Clamp BEFORE float-to-int; nonfinite/huge motion cannot overflow.
    float2 pixels = velocity * float2(width, height);
    valid = valid && all(isfinite(pixels)) && all(abs(pixels) <= 254.0f);
    int2 motion = valid ? int2(round(clamp(pixels, -254.0f, 254.0f))) : int2(0, 0);
    uint flags = (valid ? 1u : 0u) | (valid && all(abs(motion) <= 31) ? 2u : 0u);
    Fsr411QualityCurrent[cell.y * qualityCellWidth + cell.x] = uint4(asuint(motion), 0u, flags);
}
"""

POST_HELPERS = r"""
static const float FSR411_QUALITY_NOISE = @noise_floor@f;
static const float FSR411_QUALITY_FADE_FLOOR = @fade_floor@f;
static const float FSR411_QUALITY_FADE_LOWER = @fade_lower@f;
static const float FSR411_QUALITY_FADE_UPPER = @fade_upper@f;
static const float FSR411_QUALITY_SPARK_FLOOR = @spark_floor@f;
static const float FSR411_QUALITY_SPARK_SPAN = @spark_span@f;
static const float FSR411_QUALITY_SPARK_CAP = @spark_cap@f;

uint Fsr411QualityIndex(int2 cell)
{
    return uint(cell.y) * qualityCellWidth + uint(cell.x);
}

int2 Fsr411QualityMotion(int2 cell)
{
    int2 safe = clamp(cell, int2(0, 0), int2(qualityCellWidth, qualityCellHeight) - 1);
    uint index = Fsr411QualityIndex(safe);
    // Scalar reads only: POST's sole writer changes z, NEVER xy/w.
    uint flags = Fsr411QualityCurrent[index].w;
    if ((flags & 1u) == 0u)
        return int2(0, 0);
    if ((flags & 2u) == 0u && lossySkip != 0u)
        return asint(Fsr411LossyReadMotion(safe).xy);
    return int2(asint(Fsr411QualityCurrent[index].x), asint(Fsr411QualityCurrent[index].y));
}

bool Fsr411QualityMotionDiffers(int2 a, int2 b)
{
    return any(abs(a - b) > 1);
}

void Fsr411QualityClassify(int2 cell, uint4 coarse, out bool atRest,
                          out bool otherMotion, out bool uncovered)
{
    int2 motion = Fsr411QualityMotion(cell);
    otherMotion = Fsr411QualityMotionDiffers(motion, Fsr411QualityMotion(cell + (motion >> 1)));
    // Immediate probes matter for a thin fixed object over a moving background.
    static const int2 probes[12] = {
        int2(1,0), int2(-1,0), int2(0,1), int2(0,-1),
        int2(3,0), int2(-3,0), int2(0,3), int2(0,-3),
        int2(8,0), int2(-8,0), int2(0,8), int2(0,-8)
    };
    [unroll] for (uint p = 0u; p < 12u; ++p)
        otherMotion = otherMotion || Fsr411QualityMotionDiffers(motion, Fsr411QualityMotion(cell + probes[p]));
    uint ownFlags = Fsr411QualityCurrent[Fsr411QualityIndex(cell)].w;
    atRest = (ownFlags & 1u) != 0u && all(motion == int2(0, 0)) && !otherMotion;
    // Invalid/offscreen motion must never resurrect missing history.
    uncovered = (ownFlags & 1u) == 0u;
    int2 previousPixel = cell * 2 + motion;
    uncovered = uncovered || any(previousPixel < 0) ||
                previousPixel.x >= int(width) || previousPixel.y >= int(height);
    int2 block = previousPixel >> 2;
    if (!uncovered && coarse.w != 0u && all(block >= 0) &&
        block.x < int(lossyBlockWidth) && block.y < int(lossyBlockHeight))
    {
        uint key = Fsr411LossyPlane.Load((uint(block.y) * lossyBlockWidth + uint(block.x)) * 4u);
        uint owner = key & ((1u << FSR411_LOSSY_ID_BITS) - 1u);
        uint2 ownerBlock = uint2(owner & ((1u << FSR411_LOSSY_X_BITS) - 1u), owner >> FSR411_LOSSY_X_BITS);
        if (key != 0u && ownerBlock.x < lossyBlockWidth && ownerBlock.y < lossyBlockHeight)
        {
            uint4 surface = Fsr411LossyMotion[ownerBlock.y * lossyBlockWidth + ownerBlock.x];
            uncovered = surface.w != 0u && asfloat(surface.z) < asfloat(coarse.z) &&
                Fsr411QualityMotionDiffers(motion, Fsr411QualityMotion(int2(ownerBlock * 2u)));
        }
    }
}

float Fsr411QualityLuma(float3 color)
{
    return dot(color, float3(0.2126f, 0.7152f, 0.0722f));
}

uint Fsr411QualitySign(float difference, float floorValue)
{
    return !isfinite(difference) ? 0u : difference > floorValue ? 1u : difference < -floorValue ? 2u : 0u;
}

uint Fsr411QualitySigns(float3 current, float3 history)
{
    if (!all(isfinite(current)) || !all(isfinite(history)))
        return 0u;
    float3 difference = current - history;
    float y = Fsr411QualityLuma(history);
    return Fsr411QualitySign(difference.r, FSR411_QUALITY_NOISE) |
           (Fsr411QualitySign(difference.g, FSR411_QUALITY_NOISE) << 2u) |
           (Fsr411QualitySign(difference.b, FSR411_QUALITY_NOISE) << 4u) |
           (Fsr411QualitySign(Fsr411QualityLuma(current) - y,
                FSR411_QUALITY_FADE_FLOOR + abs(y) * FSR411_QUALITY_FADE_LOWER) << 6u);
}

float3 Fsr411QualitySample(int2 p, float exposure)
{
    p = clamp(p, int2(0, 0), int2(width_lr, height_lr) - 1);
    return ApplyMuLaw411(r_input_color[p], exposure);
}

// An independent, fixed float32 reference, unaffected by cached CT2D phase.
// Transform the four inputs before interpolation, in POST's working domain.
float3 Fsr411QualityReference(uint2 pixel, float exposure)
{
    float2 position = inv_scale * (float2(pixel) + 0.5f) + jitter - 0.5f;
    int2 first = int2(floor(position));
    float2 fraction = position - float2(first);
    float3 a = lerp(Fsr411QualitySample(first, exposure),
                    Fsr411QualitySample(first + int2(1, 0), exposure), fraction.x);
    float3 b = lerp(Fsr411QualitySample(first + int2(0, 1), exposure),
                    Fsr411QualitySample(first + int2(1, 1), exposure), fraction.x);
    return lerp(a, b, fraction.y);
}

void Fsr411QualityRecordSigns(uint2 pixel, float exposure)
{
    if (!Fsr411QualityEnabled() || ((pixel.x | pixel.y) & 1u) != 0u)
        return;
    uint index = Fsr411QualityIndex(int2(pixel >> 1u));
    uint packed = 0u;
    if (reset == 0u && (Fsr411QualityCurrent[index].w & 1u) != 0u)
    {
        [unroll] for (uint phase = 0u; phase < 4u; ++phase)
        {
            uint2 p = pixel + uint2(phase & 1u, phase >> 1u);
            packed |= Fsr411QualitySigns(Fsr411QualityReference(p, exposure),
                                        float3(r_reprojected_color[p])) << (phase * 8u);
        }
    }
    // EXACTLY ONE scalar store per cell, including full frames and reset.
    // Never vector-read/modify/write this record; neighbours read xy/w.
    Fsr411QualityCurrent[index].z = packed;
}

uint Fsr411QualityPreviousNote(uint2 pixel)
{
    int2 cell = int2(pixel >> 1u);
    if (qualityHistoryValid == 0u || reset != 0u ||
        (Fsr411QualityCurrent[Fsr411QualityIndex(cell)].w & 1u) == 0u)
        return 0u;
    int2 p = int2(pixel) + Fsr411QualityMotion(cell);
    if (any(p < 0) || p.x >= int(width) || p.y >= int(height))
        return 0u;
    uint index = Fsr411QualityIndex(p >> 1);
    if ((Fsr411QualityPrevious[index].w & 1u) == 0u)
        return 0u;
    uint phase = (uint(p.x) & 1u) | ((uint(p.y) & 1u) << 1u);
    return (Fsr411QualityPrevious[index].z >> (phase * 8u)) & 255u;
}

bool Fsr411QualitySameSign(uint a, uint b)
{
    return (a == 1u || a == 2u) && a == b;
}

float Fsr411QualitySpark(uint2 pixel, float exposure)
{
    if ((qualityFlags & 2u) == 0u)
        return 0.0f;
    int2 p = int2(round(inv_scale * (float2(pixel) + 0.5f) + jitter - 0.5f));
    // Clamped edge duplicates must not count as two independent neighbours.
    if (p.x <= 0 || p.y <= 0 || p.x >= int(width_lr) - 1 || p.y >= int(height_lr) - 1)
        return 0.0f;
    float center = Fsr411QualityLuma(Fsr411QualitySample(p, exposure));
    float horizontal = max(Fsr411QualityLuma(Fsr411QualitySample(p + int2(-1,0), exposure)),
                           Fsr411QualityLuma(Fsr411QualitySample(p + int2(1,0), exposure)));
    float vertical = max(Fsr411QualityLuma(Fsr411QualitySample(p + int2(0,-1), exposure)),
                         Fsr411QualityLuma(Fsr411QualitySample(p + int2(0,1), exposure)));
    float supported = min(center, min(horizontal, vertical));
    if (!isfinite(supported))
        return 0.0f;
    float minimum = 3.402823466e+38f, maximum = -3.402823466e+38f;
    [unroll] for (int y = -1; y <= 1; ++y)
        [unroll] for (int x = -1; x <= 1; ++x)
        {
            int2 h = clamp(int2(pixel) + int2(x,y), int2(0,0), int2(width,height) - 1);
            float value = Fsr411QualityLuma(float3(r_reprojected_color[h]));
            if (!isfinite(value))
                return 0.0f;
            minimum = min(minimum, value);
            maximum = max(maximum, value);
        }
    float threshold = clamp(FSR411_QUALITY_SPARK_FLOOR + (maximum - minimum) * FSR411_QUALITY_SPARK_SPAN,
                            FSR411_QUALITY_SPARK_FLOOR, FSR411_QUALITY_SPARK_CAP);
    return saturate((supported - maximum - threshold) / threshold);
}

float3 Fsr411QualityResolve(uint2 pixel, float exposure, float3 current, float3 history,
                           float3 minimum, float3 maximum, float networkWeight,
                           bool atRest, bool otherMotion, bool uncovered)
{
    uint previous = Fsr411QualityPreviousNote(pixel);
    float3 referenceColor = Fsr411QualityReference(pixel, exposure);
    uint reference = Fsr411QualitySigns(referenceColor, history);
    uint actual = Fsr411QualitySigns(current, history);
    bool persistent = false;
    [unroll] for (uint shift = 0u; shift < 6u; shift += 2u)
    {
        uint sign = (actual >> shift) & 3u;
        persistent = persistent || (Fsr411QualitySameSign(sign, (previous >> shift) & 3u) &&
                                    Fsr411QualitySameSign(sign, (reference >> shift) & 3u));
    }
    float3 result = Fsr411LossyResolve(current, history, minimum, maximum, networkWeight,
                                     atRest && !persistent, otherMotion, uncovered);
    uint driftSign = (actual >> 6u) & 3u;
    float ceiling = FSR411_QUALITY_FADE_FLOOR + abs(Fsr411QualityLuma(history)) * FSR411_QUALITY_FADE_UPPER;
    bool fade = atRest && Fsr411QualitySameSign(driftSign, previous >> 6u) &&
                Fsr411QualitySameSign(driftSign, reference >> 6u) &&
                abs(Fsr411QualityLuma(current - history)) <= ceiling &&
                abs(Fsr411QualityLuma(referenceColor - history)) <= ceiling;
    if (fade)
        result = lerp(history, current, saturate(networkWeight));
    // A bounded spark response cannot increase the new frame's share past 1.
    result = lerp(result, current, Fsr411QualitySpark(pixel, exposure));
    bool darkEdge = !atRest && all(history <= max(current * 0.1f, 0.0f));
    return (uncovered || darkEdge) ? current : result;
}
"""


FAST_NOTE_HELPERS = r"""
// fsr411-quality-fast-notes: cached 16x16 row-major pixels, Wave32 only.
void Fsr411QualityRecordSignsFast(uint2 pixel, uint groupLane, float3 current,
                                 float3 history, float exposure)
{
    if (!Fsr411QualityEnabled())
        return;
    // Uniform frame/wave conditions. Mode 1 keeps the original function/data.
    if ((qualityFlags & 4u) == 0u || WaveGetLaneCount() != 32u)
    {
        Fsr411QualityRecordSigns(pixel, exposure);
        return;
    }
    uint lane = WaveGetLaneIndex();
    // Wave32 alone does not guarantee HLSL thread-to-lane assignment. Refuse
    // permutations or waves mixing different logical 32-thread row pairs.
    bool rowMajor = WaveActiveAllTrue(lane == (groupLane & 31u));
    bool sameRows = WaveActiveAllEqual(groupLane >> 5u);
    if (!rowMajor || !sameRows)
    {
        Fsr411QualityRecordSigns(pixel, exposure);
        return;
    }
    uint index = Fsr411QualityIndex(int2(pixel >> 1u));
    uint note = 0u;
    if (reset == 0u && (Fsr411QualityCurrent[index].w & 1u) != 0u)
        note = Fsr411QualitySigns(current, history);
    // ALL valid pixel lanes execute all shuffles, even non-owners/neutral
    // notes. A 2x2 cell fits in this wave's two 16-wide rows. Never read a
    // lane belonging to another wave, or an inactive lane at an even edge.
    uint base = lane & ~17u;
    uint a = WaveReadLaneAt(note, base);
    uint b = WaveReadLaneAt(note, base + 1u);
    uint c = WaveReadLaneAt(note, base + 16u);
    uint d = WaveReadLaneAt(note, base + 17u);
    uint packed = a | (b << 8u) | (c << 16u) | (d << 24u);
    if (((pixel.x | pixel.y) & 1u) == 0u)
        Fsr411QualityCurrent[index].z = packed;
}
"""


def _cached_note_geometry(source):
    """Only the pinned color-cache mapping proves four active lanes per cell."""
    if "fsr411-generated-post-color-cache:" not in source:
        if "uint2 outputPixel =" in source:
            raise ValueError("fast notes cannot infer an unknown POST pixel mapping")
        return False
    coordinate = "    uint2 outputPixel = GID.xy * 16u + uint2(GTID.x & 15u, GTID.x >> 4u);"
    guard = "    if (outputPixel.x >= width || outputPixel.y >= height)\n        return;"
    for expression in (coordinate, guard, "    uint pk = (outputPixel.x & 1u) | ((outputPixel.y & 1u) << 1u);"):
        if source.count(expression) != 1:
            raise ValueError("fast notes require the pinned cached 16x16 POST layout")
    if "for (uint pk = 0; pk < 4; pk++)" in source:
        raise ValueError("fast notes found a mixed per-cell/per-pixel POST")
    return True


def apply_temporal_quality_pre(source, tier, enabled=True, settings=DEFAULT_SETTINGS):
    source, marker = _prepare(source, tier, "pre", enabled, settings)
    if marker is None:
        return source
    source = _replace(source, "[numthreads(GROUP_SIZE_*GROUP_SIZE_, 1, 1)]",
                      DECLARATIONS + PRE_HELPERS + "\n[numthreads(GROUP_SIZE_*GROUP_SIZE_, 1, 1)]")
    call = "    Fsr411LossyRecord(dtid, velocity, mv_dialated_pos, disocclusion);"
    source = _replace(source, call, call + "\n    Fsr411QualityRecordMotion(dtid, velocity, disocclusion);")
    return marker + source


def apply_temporal_quality_post(source, tier, enabled=True, settings=DEFAULT_SETTINGS, fast_notes=False):
    """Generate conservative POST by default; mode 2 opts into Wave32 fast notes.

    Keep fast_notes=False for temporal-quality mode 1, so its generated shader
    and reference data algorithm remain unchanged. Mode 2 requires both this
    generation switch and b3 flags bit 2; flag-disabled/unsupported waves still
    use reference notes. Switching generation policies requires legacy input.
    """
    source, marker = _prepare(source, tier, "post", enabled, settings, fast_notes=fast_notes)
    if marker is None:
        return source
    cached_notes = bool(fast_notes) and _cached_note_geometry(source)
    helpers = POST_HELPERS + (FAST_NOTE_HELPERS if cached_notes else "")
    for name, value in settings.__dict__.items():
        literal = format(value, ".9g")
        if "." not in literal and "e" not in literal:
            literal += ".0"
        helpers = helpers.replace("@" + name + "@", literal)
    source = _replace(source, "[numthreads(256, 1, 1)]",
                      DECLARATIONS + helpers + "\n[numthreads(256, 1, 1)]")
    # Both POST geometries keep this statement after classification. Always
    # reload coarse depth here: the shared variant has only populated xy.
    anchor = "    bool convert = (rcas_enabled == 0);"
    source = _replace(source, anchor, """    if (lossySkip != 0u && Fsr411QualityEnabled())
        Fsr411QualityClassify(int2(ix, iy), Fsr411LossyReadMotion(int2(ix, iy)),
                             lossyAtRest, lossyOtherMotion, lossyUncovered);
""" + anchor)
    old = """            res = Fsr411LossyResolve(up, reproj, lossyMinimum, lossyMaximum, 1.0f - bf,
                                    lossyAtRest, lossyOtherMotion, lossyUncovered);"""
    new = """            if (Fsr411QualityEnabled())
                res = Fsr411QualityResolve(tid, exposure, up, reproj, lossyMinimum, lossyMaximum, 1.0f - bf,
                                          lossyAtRest, lossyOtherMotion, lossyUncovered);
            else
                res = Fsr411LossyResolve(up, reproj, lossyMinimum, lossyMaximum, 1.0f - bf,
                                        lossyAtRest, lossyOtherMotion, lossyUncovered);"""
    source = _replace(source, old, new)
    note_call = ("        Fsr411QualityRecordSignsFast(tid, GTID.x, up, reproj, exposure);" if cached_notes else
                 "        Fsr411QualityRecordSigns(tid, exposure);")
    source = _replace(source, "        rw_history_color[tid] = res;",
                      note_call + "\n        rw_history_color[tid] = res;")
    return marker + source
