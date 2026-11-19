#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Independent new-look rules for Lite Lossy, from cyboman's public explanation.

Reference: https://github.com/UzenUPoZiTiV4ik/fsr4-native-macos/issues/20#issuecomment-6839502725
Only the explanation was consulted; no upstream shader-generator code is used.
Constants are in Lite's log working space. The feature is opt-in, linear-input,
whole-frame Lite Lossy only. It is not a claim of bit-exact upstream parity:
our current-owner motion/depth plane is still an approximation to its scatter.
The plain Lite Lossy transform must remain byte-for-byte unchanged when off.
"""
from __future__ import annotations

import math

REVISION = 2
SOURCE_URL = "https://github.com/UzenUPoZiTiV4ik/fsr4-native-macos/issues/20#issuecomment-6839502725"


def _clamp(value, low=0.0, high=1.0):
    return min(max(value, low), high)


def note(current, history):
    difference = [c - h for c, h in zip(current, history)]
    threshold = 0.0015 * history[1] + 0.0001
    return (int(any(d > 0.02 for d in difference)) -
            int(any(d < -0.02 for d in difference)) +
            (4 if difference[1] > threshold else -4 if difference[1] < -threshold else 0))


def finish(output, current, history, previous_note=0.0, history_valid=True):
    flag = note(current, history)
    fading = ((flag > 2 and previous_note > 2) or
              (flag < -2 and previous_note < -2))
    keep = history_valid and not fading and all(
        abs(o - h) < h * (2.0 / 1024.0) for o, h in zip(output, history))
    return (tuple(history) if keep else tuple(output)), flag


def resolve(current, history, history_share, samples, history_samples, band_samples,
            *, at_rest=False, other_motion=False, uncovered=False, depth_edge=False,
            previous_note=0.0, history_valid=True):
    """CPU specification for focused regression tests, not a DLSS/FSR oracle."""
    if len(samples) != 9 or len(history_samples) != 9 or len(band_samples) != 4:
        raise ValueError("new-look requires 9 current, 9 history and 4 band samples")
    values = [*current, *history, history_share, previous_note,
              *(v for group in (samples, history_samples, band_samples) for row in group for v in row)]
    if not all(math.isfinite(v) for v in values):
        raise ValueError("nonfinite new-look input")
    low = [min(s[c] for s in samples) for c in range(3)]
    high = [max(s[c] for s in samples) for c in range(3)]
    span = [hi - lo for lo, hi in zip(low, high)]
    hlow = [min(s[c] for s in history_samples) for c in range(3)]
    hhigh = [max(s[c] for s in history_samples) for c in range(3)]
    hspan = [hi - lo for lo, hi in zip(hlow, hhigh)]
    difference = [c - h for c, h in zip(current, history)]
    rgb_note = previous_note - (4 if previous_note > 2 else -4 if previous_note < -2 else 0)
    changed = any(
        h < lo - 0.25 * r or h > hi + 0.25 * r or
        abs(d) > 0.25 * r + 0.05 or
        (abs(lo + hi - 2 * h) > 0.85 * r and abs(d) > 0.05 * r + 0.01) or
        (d > 0.02 and rgb_note > 0) or (d < -0.02 and rgb_note < 0)
        for h, lo, hi, r, d in zip(history, low, high, span, difference))
    blob = [min(samples[4][c], max(samples[3][c], samples[5][c]),
                max(samples[1][c], samples[7][c])) for c in range(3)]
    spark = max(_clamp((max(blob[c] - hhigh[c], hlow[c] - samples[4][c]) -
                        (2 * hspan[c] + 0.05)) / (hspan[c] + 0.05)) for c in range(3))
    lift = [lo - hi for lo, hi in zip(low, hhigh)]
    arrival = max(_clamp((v - 0.005) / 0.005) for v in lift)
    band_threshold = max(0.75 * max(lift), 0.005)
    band = any(min(band_samples[a][c], band_samples[a + 1][c]) - hhigh[c] > band_threshold
               for a in (0, 2) for c in range(3))
    if band:
        arrival = 0.0
    spark_share = max(spark, arrival)
    weight = 1.0 - history_share
    verdict = weight * max(
        _clamp((weight - 0.2) / 0.3) * (1 - _clamp((weight - 0.6) / 0.1)),
        _clamp((weight - 0.95) / 0.045))
    lower, upper = 0.0015 * history[1] + 0.0001, 0.01 * history[1]
    slow = ((lower < difference[1] < upper and previous_note > 2) or
            (lower < -difference[1] < upper and previous_note < -2))
    base = weight if slow else verdict if at_rest and not changed and spark_share <= 0 else weight * weight
    new_share = min(1.0, max(base, spark_share))
    safe_history = [_clamp(h, lo, hi) if other_motion and not at_rest else h
                    for h, lo, hi in zip(history, low, high)]
    output = [(1 - new_share) * h + new_share * c for h, c in zip(safe_history, current)]
    brightness = [max(h) for h in history_samples]
    line = any(min(brightness[a], brightness[b]) >= 0.5 * brightness[4]
               for a, b in ((3, 5), (1, 7), (0, 8), (2, 6)))
    if not line:
        output = [min(o, hi + 0.05) for o, hi in zip(output, high)]
    mean = [sum(s[c] for s in samples) / 9.0 for c in range(3)]
    if not at_rest and depth_edge:
        neighbors = samples[:4] + samples[5:]
        for c, h in enumerate(history):
            floor, ceiling = 0.9 * h, 1.5 * h
            if mean[c] >= 0.7 * h and (not other_motion or (
                    samples[4][c] < floor and all(s[c] >= floor for s in neighbors))):
                output[c] = max(output[c], floor)
            if mean[c] <= 1.7 * h and (not other_motion or (
                    samples[4][c] > ceiling and all(s[c] <= ceiling for s in neighbors))):
                output[c] = min(output[c], ceiling)
    dark_band = not at_rest and all(h <= max(0.1 * c, 0.0) for h, c in zip(history, current))
    if uncovered or dark_band:
        output = list(current)
    output, flag = finish(output, current, history, previous_note, history_valid)
    return {"output": output, "note": flag, "new_share": new_share,
            "spark": spark, "arrival": arrival, "band": band, "line": line, "slow": slow}


COLOR_RULES = r"""
// new-look: independent implementation of the public issue #20 formulas.
static inline float Fsr412LookNote(float3 current, float3 history) {
    float3 d = current - history;
    float t = 0.0015f * history.y + 0.0001f;
    return float(any(d > float3(0.02f))) - float(any(d < float3(-0.02f)))
        + (d.y > t ? 4.0f : d.y < -t ? -4.0f : 0.0f);
}
static inline float3 Fsr412LookFinish(float3 out, float3 current, float3 history,
    float previousNote, bool historyValid, thread float& note) {
    note = Fsr412LookNote(current, history);
    bool fading = (note > 2.0f && previousNote > 2.0f) || (note < -2.0f && previousNote < -2.0f);
    return historyValid && !fading && all(abs(out - history) < history * (2.0f / 1024.0f)) ? history : out;
}
static inline float Fsr412LookArrival(thread const float3* samples, float3 historyMaximum) {
    float3 minimum = samples[0];
    for (uint i = 1u; i < 9u; ++i) minimum = min(minimum, samples[i]);
    float3 g = saturate((minimum - historyMaximum - 0.005f) / 0.005f);
    return max(g.x, max(g.y, g.z));
}
static inline float3 Fsr412LookResolve(float3 current, float3 history, float historyShare,
    thread const float3* samples, thread const float3* historySamples, thread const float3* bandSamples,
    bool atRest, bool otherMotion, bool uncovered, bool depthEdge, float previousNote) {
    float3 minimum = samples[0], maximum = samples[0];
    float3 hmin = historySamples[0], hmax = historySamples[0];
    for (uint i = 1u; i < 9u; ++i) {
        minimum = min(minimum, samples[i]); maximum = max(maximum, samples[i]);
        hmin = min(hmin, historySamples[i]); hmax = max(hmax, historySamples[i]);
    }
    float3 range = maximum - minimum, hr = hmax - hmin, d = current - history;
    float rgbNote = previousNote - (previousNote > 2.0f ? 4.0f : previousNote < -2.0f ? -4.0f : 0.0f);
    bool changed = any(history < minimum - 0.25f * range) || any(history > maximum + 0.25f * range)
        || any(abs(d) > 0.25f * range + 0.05f)
        || any((abs(minimum + maximum - 2.0f * history) > 0.85f * range) && (abs(d) > 0.05f * range + 0.01f))
        || (rgbNote > 0.0f && any(d > float3(0.02f))) || (rgbNote < 0.0f && any(d < float3(-0.02f)));
    float3 blob = min(samples[4], min(max(samples[3], samples[5]), max(samples[1], samples[7])));
    float3 s = saturate((max(blob - hmax, hmin - samples[4]) - (2.0f * hr + 0.05f)) / (hr + 0.05f));
    float spark = max(s.x, max(s.y, s.z));
    float3 lift = minimum - hmax;
    float arrival = Fsr412LookArrival(samples, hmax);
    float threshold = max(0.75f * max(lift.x, max(lift.y, lift.z)), 0.005f);
    bool band = any(min(bandSamples[0], bandSamples[1]) - hmax > threshold)
        || any(min(bandSamples[2], bandSamples[3]) - hmax > threshold);
    if (band) arrival = 0.0f;
    float sparkShare = max(spark, arrival), w = 1.0f - historyShare;
    float verdict = w * max(saturate((w - 0.2f) / 0.3f) * (1.0f - saturate((w - 0.6f) / 0.1f)),
                            saturate((w - 0.95f) / 0.045f));
    float lower = 0.0015f * history.y + 0.0001f, upper = 0.01f * history.y;
    bool slow = (d.y > lower && d.y < upper && previousNote > 2.0f)
        || (-d.y > lower && -d.y < upper && previousNote < -2.0f);
    float base = slow ? w : atRest && !changed && sparkShare <= 0.0f ? verdict : w * w;
    float share = min(1.0f, max(base, sparkShare));
    float3 old = otherMotion && !atRest ? clamp(history, minimum, maximum) : history;
    float3 result = (1.0f - share) * old + share * current;
    float b[9];
    for (uint i = 0u; i < 9u; ++i) b[i] = max(historySamples[i].x, max(historySamples[i].y, historySamples[i].z));
    bool line = min(b[3], b[5]) >= 0.5f * b[4] || min(b[1], b[7]) >= 0.5f * b[4]
        || min(b[0], b[8]) >= 0.5f * b[4] || min(b[2], b[6]) >= 0.5f * b[4];
    if (!line) result = min(result, maximum + 0.05f);
    float3 mean = float3(0.0f);
    for (uint i = 0u; i < 9u; ++i) mean += samples[i];
    mean *= 0.111111111f;
    if (!atRest && depthEdge) {
        float3 floor = 0.9f * history, ceiling = 1.5f * history;
        bool3 applyFloor = mean >= 0.7f * history, applyCeiling = mean <= 1.7f * history;
        // At a motion boundary, do not clamp the new surface back to stale history.
        if (otherMotion) {
            float3 neighborMin = samples[0], neighborMax = samples[0];
            for (uint i = 1u; i < 9u; ++i) {
                if (i == 4u) continue;
                neighborMin = min(neighborMin, samples[i]);
                neighborMax = max(neighborMax, samples[i]);
            }
            applyFloor = applyFloor && (samples[4] < floor) && (neighborMin >= floor);
            applyCeiling = applyCeiling && (samples[4] > ceiling) && (neighborMax <= ceiling);
        }
        result = select(result, max(result, floor), applyFloor);
        result = select(result, min(result, ceiling), applyCeiling);
    }
    bool darkBand = !atRest && all(history <= max(0.1f * current, float3(0.0f)));
    return uncovered || darkBand ? current : result;
}
"""

MOTION_RULES = r"""
static inline uint4 Fsr412LookDepthRecord(int2 cell, const device uint4* records,
    constant type_MLSR_Optimized_Constants& frame) {
    cell = clamp(cell, int2(0), int2(frame.width / 2u, frame.height / 2u) - int2(1));
    // Upstream depth is one word per 4x4 block, not per output pixel.
    return Fsr412LossyReadMotion(cell & ~int2(1), records, frame);
}
static inline int2 Fsr412LookMotion(int2 cell, const device uint4* records,
    constant type_MLSR_Optimized_Constants& frame) {
    int2 fine = as_type<int2>(Fsr412LossyReadMotion(cell, records, frame).xy);
    return any(abs(fine) > int2(31)) ? as_type<int2>(Fsr412LookDepthRecord(cell, records, frame).xy) : fine;
}
static inline bool Fsr412LookDepthEdge(int2 cell, const device uint4* records,
    constant type_MLSR_Optimized_Constants& frame) {
    uint4 own = Fsr412LookDepthRecord(cell, records, frame);
    if ((own.w & 2u) == 0u) return true;
    float z = as_type<float>(own.z);
    const int2 directions[4] = {int2(1,0), int2(-1,0), int2(0,1), int2(0,-1)};
    bool edge = false;
    for (int radius = 1; radius <= 3; ++radius) for (uint i = 0u; i < 4u; ++i) {
        uint4 n = Fsr412LookDepthRecord(cell + radius * directions[i], records, frame);
        float nz = as_type<float>(n.z);
        edge = edge || ((n.w & 2u) != 0u && abs(nz - z) > 0.02f * max(abs(nz), abs(z)));
    }
    return edge;
}
static inline void Fsr412LookClassify(uint2 pixel, const device uint4* records,
    constant type_MLSR_Optimized_Constants& frame, thread bool& still, thread bool& atRest,
    thread bool& otherMotion, thread bool& uncovered) {
    int2 cell = int2(pixel / 2u), motion = Fsr412LookMotion(cell, records, frame);
    uint4 own = Fsr412LossyReadMotion(cell, records, frame);
    int2 previous = int2(pixel) + motion;
    uncovered = (own.w & 1u) == 0u || any(previous < int2(0))
        || previous.x >= int(frame.width) || previous.y >= int(frame.height);
    otherMotion = Fsr412LossyDifferent(motion, Fsr412LookMotion(previous / 2, records, frame));
    uint4 z = Fsr412LookDepthRecord(cell, records, frame);
    uint4 surface = Fsr412LookDepthRecord(previous / 2, records, frame);
    // Current-owner approximation retained; no claim of upstream scatter parity.
    uncovered = uncovered || ((z.w & surface.w & 2u) != 0u
        && as_type<float>(surface.z) < as_type<float>(z.z) && otherMotion);
    const int2 probes[8] = {int2(3,0), int2(-3,0), int2(0,3), int2(0,-3),
                           int2(8,0), int2(-8,0), int2(0,8), int2(0,-8)};
    for (uint i = 0u; i < 8u; ++i)
        otherMotion = otherMotion || Fsr412LossyDifferent(motion, Fsr412LookMotion(cell + probes[i], records, frame));
    still = (own.w & 1u) != 0u && all(motion == int2(0));
    atRest = still && !otherMotion;
}
"""
