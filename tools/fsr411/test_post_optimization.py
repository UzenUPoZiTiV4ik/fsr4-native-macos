"""Dispatch, footprint and arithmetic contracts for the shared POST filter."""

import math
from pathlib import Path
import re
import struct
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    from compile_fsr411 import external_post_cnb, math_flags
    from fsr411.lossy_shaders import apply_lossy_post, apply_lossy_post_cells
    from fsr411.post_optimization import apply_post_optimization, POST_CACHE_SIDE, POST_PIXELS_PER_GROUP
finally:
    sys.path.pop(0)


ROOT = Path(__file__).resolve().parents[2]


def f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def mad(a, b, c, fused):
    return f32(a * b + c) if fused else f32(f32(a * b) + c)


def function(source, name):
    start = source.index("float3 " + name + "(")
    # The two colorspace helpers have no nested statement scopes.
    return source[start:source.index("\n}", start) + 2]


def variants():
    for tier, folder in ((1080, "reimpl411"), (2160, "reimpl2160"), (4320, "reimpl4320")):
        for preset in ("shaders", "ultraperformance"):
            path = next((ROOT / "third_party/fsr411" / preset / folder).glob("post_rev_411*.hlsl"))
            source = external_post_cnb(path.read_text(), tier)
            for profile in ("exact", "lossy"):
                yield tier, preset, profile, source if profile == "exact" else apply_lossy_post(source, tier)


def ct2d_tables():
    # Mirror only the include search order, so each tier/preset proves the
    # exact table that compile_fsr411 installs into its generated include tree.
    for tier, folder in ((1080, "reimpl411"), (2160, "reimpl2160"), (4320, "reimpl4320")):
        for preset in ("shaders", "ultraperformance"):
            candidates = [ROOT / "third_party/fsr411/shaders" / directory / "post_rev_411_tables.hlsli"
                          for directory in dict.fromkeys(("reimpl411", folder))]
            if preset == "ultraperformance":
                candidates += [ROOT / "third_party/fsr411/ultraperformance" / directory / "post_rev_411_tables.hlsli"
                               for directory in dict.fromkeys(("reimpl411", folder))]
            table = [path for path in candidates if path.exists()][-1]
            source = table.read_text()
            arrays = []
            for name, count in (("CT2D_W", 128), ("CT2D_S", 32)):
                matches = re.findall(r"static const int " + name + r"\[" + str(count) + r"\] = \{(.*?)\};", source, re.S)
                if len(matches) != 1 or re.sub(r"-?\d+|[\s,]", "", matches[0]):
                    raise ValueError("unrecognized pinned CT2D table: " + str(table))
                values = list(map(int, re.findall(r"-?\d+", matches[0])))
                if len(values) != count or any(value < -(1 << 31) or value >= (1 << 31) for value in values):
                    raise ValueError("invalid signed CT2D table: " + str(table))
                arrays.append(values)
            yield tier, preset, arrays[0], arrays[1]


def unpack_signed_bytes(word):
    values = [(word >> shift) & 255 for shift in (0, 8, 16, 24)]
    return [value - 256 if value >= 128 else value for value in values]


class PostOptimizationTests(unittest.TestCase):
    def test_ct2d_float32_dot_keeps_all_phase_and_cnb_indices(self):
        for tier, preset, profile, source in variants():
            with self.subTest(tier=tier, preset=preset, profile=profile):
                result = apply_post_optimization(source)
                phase, values = ("pk", "packedVss") if profile == "exact" else ("cnbPixel", "pixelCnb")
                self.assertIn("float ct[8];", result)
                self.assertNotIn("int ct[8];", result)
                self.assertIn("ct[ch] = CT2D_S[" + phase + " * 8 + ch];", result)
                self.assertIn("int b = (" + phase + " * 32 + ch2 * 4);", result)
                for word in range(4):
                    index = "b" if word == 0 else "b + " + str(word)
                    self.assertIn("ct[ch2] = Fsr411PostDot4Exact(CT2D_W[" + index + "], " + values +
                                  "[" + str(word) + "], ct[ch2]);", result)
                self.assertEqual(result.count("ct[ch2] = Fsr411PostDot4Exact("), 4)
                self.assertIn("return dot(float4(unpack_s8s32(a)), float4(unpack_s8s32(b))) + acc;", result)

    def test_all_tier_tables_bound_every_float32_partial_sum(self):
        for tier, preset, packed, biases in ct2d_tables():
            with self.subTest(tier=tier, preset=preset):
                maxima = []
                for channel, bias in enumerate(biases):
                    weights = [weight for word in packed[channel * 4:channel * 4 + 4]
                               for weight in unpack_signed_bytes(word)]
                    self.assertEqual(len(weights), 16)
                    # This is an absolute-sum bound, not an estimate of the
                    # final (possibly cancelling) sum. Every product, every
                    # subset/tree reduction inside dot4, and every running
                    # accumulator is within this interval for ALL signed
                    # INT8 inputs and ALL association orders.
                    product_bounds = [128 * abs(weight) for weight in weights]
                    self.assertLessEqual(max(product_bounds), 1 << 14)
                    absolute_bound = abs(bias) + sum(product_bounds)
                    self.assertLess(absolute_bound, 1 << 24)
                    maxima.append(absolute_bound)
                    # Also exercise each intermediate float32 operation on
                    # extremal and mixed inputs, including alternate dot4
                    # associations. No final-only comparison can hide an
                    # inexact intermediate here.
                    inputs = [[-128] * 16, [127] * 16,
                              [127 if weight >= 0 else -128 for weight in weights],
                              [-128 if weight >= 0 else 127 for weight in weights]]
                    inputs += [[((channel * 37 + i * 43 + case * 71) % 256) - 128
                                for i in range(16)] for case in range(8)]
                    for values in inputs:
                        for order in ((0, 1, 2, 3), (3, 2, 1, 0), (2, 0, 3, 1)):
                            actual, expected = f32(bias), bias
                            for word in range(4):
                                dot, integer_dot = 0.0, 0
                                for lane in order:
                                    index = word * 4 + lane
                                    product = weights[index] * values[index]
                                    float_product = f32(float(weights[index]) * float(values[index]))
                                    self.assertEqual(float_product, product)
                                    dot = f32(dot + float_product)
                                    integer_dot += product
                                    self.assertEqual(dot, integer_dot)
                                actual = f32(actual + dot)
                                expected += integer_dot
                                self.assertEqual(actual, expected)
                self.assertEqual(max(maxima), 110151 if preset == "shaders" else 87727)

    def test_all_profiles_keep_transfer_and_filter_arithmetic(self):
        for tier, preset, profile, source in variants():
            with self.subTest(tier=tier, preset=preset, profile=profile):
                result = apply_post_optimization(source)
                self.assertEqual(apply_post_optimization(result), result)
                for name in ("ApplyMuLaw411", "RemoveMuLaw411"):
                    self.assertEqual(function(result, name), function(source, name))
                # Keep precisely the original association and rounding from
                # model parameters through sample positions and each weight.
                for pattern in (r"        float corr =.*?int2 lr_i = \(int2\)round\(lr_f\);",
                                r"                float xd =.*?tw \+= g;",
                                r"        up /= tw;.*?rw_mlsr_output_color\[tid\] = final_color;"):
                    self.assertEqual(re.search(pattern, result, re.S)[0], re.search(pattern, source, re.S)[0])
                self.assertIn("up += currentSample * g;", result)
                if profile == "lossy":
                    for expression in ("lossyMinimum = min(lossyMinimum, currentSample);",
                                       "lossyMaximum = max(lossyMaximum, currentSample);",
                                       "cnbPixel = (asuint(sourcePixel.x) & 1u) | ((asuint(sourcePixel.y) & 1u) << 1u);"):
                        self.assertIn(expression, result)

    def test_every_lane_loads_before_any_partial_tile_return(self):
        for _, _, _, source in variants():
            result = apply_post_optimization(source)
            main = result[result.index("void fsr4_model_v07_fp8_no_scale_postpass("):]
            self.assertLess(main.index("Fsr411CachePostColor("), main.index("return;"))
            self.assertIn("GroupMemoryBarrierWithGroupSync();", result[:result.index("[numthreads(")])
            self.assertIn("groupshared float3 Fsr411PostColor[", result)
            self.assertNotIn("groupshared half", result)
            self.assertNotIn("for (uint pk", main)

    def test_lossy_cells_classify_once_before_the_shared_barrier(self):
        for _, _, profile, source in variants():
            optimized = apply_post_optimization(source)
            if profile == "exact":
                with self.assertRaises(ValueError):
                    apply_lossy_post_cells(optimized)
                continue
            result = apply_lossy_post_cells(optimized)
            self.assertEqual(apply_lossy_post_cells(result), result)
            helper = result[result.index("void Fsr411CachePostColor("):result.index("[numthreads(")]
            self.assertLess(helper.index("Fsr411LossyClassifyCells(lane"), helper.index("GroupMemoryBarrierWithGroupSync();"))
            self.assertLess(result.index("void Fsr411LossyClassifyCells("), result.index("void Fsr411CachePostColor("))
            main = result[result.index("void fsr4_model_v07_fp8_no_scale_postpass("):]
            self.assertEqual(main.count("Fsr411LossyClassify("), 0)
            self.assertIn("lossyOwn.xy = Fsr411LossyCellMotion[cellIndex];", main)

    def test_lossy_cell_lanes_cover_each_tile_cell_once(self):
        # Classifier lane 192+i owns local cell (i & 7, i >> 3); each output
        # pixel reads the cell that contains it, from its own 16x16 tile.
        for lane in range(256):
            px, py = lane & 15, lane >> 4
            index = ((py >> 1) & 7) * 8 + ((px >> 1) & 7)
            local = 192 + index
            self.assertEqual(((local - 192) & 7, (local - 192) >> 3), (px >> 1, py >> 1))
        self.assertEqual(len({((l - 192) & 7, (l - 192) >> 3) for l in range(192, 256)}), 64)

    def test_fast_profiles_use_relaxed_pre_post(self):
        for profile in ("exact-fast", "lossy"):
            for name in ("pre", "post"):
                flags = math_flags(name, profile)
                self.assertEqual(flags, ["-fmetal-math-mode=relaxed", "-fmetal-math-fp32-functions=fast"])
                # Relaxed keeps Inf/NaN: PRE's isnan/isinf colour checks must survive.
                self.assertFalse(any(flag in ("-ffast-math", "-fmetal-math-mode=fast") for flag in flags))
        for profile in ("exact", "exact-fast", "lossy"):
            for name in ("ae", "ae_finalize"):
                self.assertEqual(math_flags(name, profile), ["-fno-fast-math", "-ffp-contract=off"])
            self.assertEqual(math_flags("pass1", profile), [])

    def test_exact_keeps_ieee_pre_post(self):
        for name in ("pre", "post"):
            self.assertEqual(math_flags(name, "exact"), ["-fno-fast-math", "-ffp-contract=off"])

    def test_dispatch_writes_each_pixel_once_in_partial_tiles(self):
        # Runtime output viewports are aligned to eight; they need not be
        # multiples of sixteen. No edge lane may prevent the cache barrier.
        for width, height in ((8, 8), (16, 24), (40, 24), (120, 72)):
            seen = set()
            for gy in range((height + POST_PIXELS_PER_GROUP - 1) // POST_PIXELS_PER_GROUP):
                for gx in range((width + POST_PIXELS_PER_GROUP - 1) // POST_PIXELS_PER_GROUP):
                    for lane in range(256):
                        px, py = gx * 16 + (lane & 15), gy * 16 + (lane >> 4)
                        if px >= width or py >= height:
                            continue
                        cell, phase = (px >> 1, py >> 1), (px & 1) | ((py & 1) << 1)
                        reconstructed = (cell[0] * 2 + (phase & 1), cell[1] * 2 + (phase >> 1))
                        self.assertEqual(reconstructed, (px, py))
                        self.assertNotIn(reconstructed, seen)
                        seen.add(reconstructed)
            self.assertEqual(len(seen), width * height)

    def test_cache_covers_rounded_footprints_and_viewport_clamps(self):
        # Test each possible independent FMA contraction of the conservative
        # cache origin and the unchanged half-pixel sample coordinate. The
        # two axes are independent, so one-dimensional exhaustive edges
        # cover every three-by-three footprint without huge image fixtures.
        for output, render in ((120, 30), (120, 79), (1920, 1280), (1920, 1920),
                               (3840, 2560), (7680, 2560), (7680, 7680)):
            scale = f32(render / output)
            active_side = min(POST_CACHE_SIDE, math.ceil(f32(scale * 16.0)) + 4)
            for tile in (0, 16, (output // 32) * 16, ((output - 1) // 16) * 16):
                for jitter in (-0.5, -0.499999, -0.25, 0, 0.25, 0.499999, 0.5):
                    jitter = f32(jitter)
                    for origin_fused in (False, True):
                        origin = math.floor(mad(float(tile), scale, jitter, origin_fused)) - 2
                        for coordinate_fused in (False, True):
                            for pixel in range(tile, min(tile + 16, output)):
                                position = mad(scale, f32(pixel + 0.5), f32(jitter - 0.5), coordinate_fused)
                                center = round(position)
                                cache_start = center - 1 - origin
                                self.assertGreaterEqual(cache_start, 0)
                                self.assertLess(cache_start, active_side - 2)
                                for tap in range(3):
                                    direct = max(0, min(render - 1, center - 1 + tap))
                                    cached = max(0, min(render - 1, origin + cache_start + tap))
                                    self.assertEqual(cached, direct)

    def test_active_cache_loads_match_the_guarded_filter_footprint(self):
        # The fixed allocation/stride remain valid for native resolution.
        # At the recorded game scale, only 169 lanes perform RGB MuLaw.
        for rw, rh, uw, uh, expected in ((864, 560, 1712, 1112, 169),
                                         (960, 540, 1920, 1080, 144),
                                         (1280, 720, 1920, 1080, 225),
                                         (1920, 1080, 1920, 1080, 400)):
            extent = (math.ceil(f32(f32(rw / uw) * 16.0)) + 4,
                      math.ceil(f32(f32(rh / uh) * 16.0)) + 4)
            loaded = set()
            for lane in range(256):
                for index in range(lane, POST_CACHE_SIDE * POST_CACHE_SIDE, 256):
                    x, y = index % POST_CACHE_SIDE, index // POST_CACHE_SIDE
                    if x < extent[0] and y < extent[1]:
                        self.assertNotIn((x, y), loaded)
                        loaded.add((x, y))
            self.assertEqual(len(loaded), expected)
            # Every footprint accepted by the shader's active-extent guard
            # must have all nine values initialized before the barrier.
            for y in range(min(extent[1], POST_CACHE_SIDE) - 2):
                for x in range(min(extent[0], POST_CACHE_SIDE) - 2):
                    for dy in range(3):
                        for dx in range(3):
                            self.assertIn((x + dx, y + dy), loaded)

    def test_scalar_and_unrecognized_sources_are_rejected(self):
        source = (ROOT / "third_party/fsr411/shaders/reimpl411/post_rev_411.hlsl").read_text()
        with self.assertRaises(ValueError):
            apply_post_optimization(source)
        source = external_post_cnb(source, 1080)
        for broken in (source.replace("uint gx = GID.x", "uint gx = DTID.x"),
                       source.replace("int2 lr_i = (int2)round(lr_f)", "int2 lr_i = (int2)floor(lr_f)"),
                       source.replace("up += ApplyMuLaw411(r_input_color[pc], exposure) * g", "up += 0.0f"),
                       source.replace("int ct[8];", "half ct[8];"),
                       source.replace("CT2D_W[b + 3], packedVss[3]", "CT2D_W[b + 3], packedVss[2]"),
                       source.replace("CT2D_S[pk * 8 + ch]", "CT2D_S[ch]")):
            with self.assertRaises(ValueError):
                apply_post_optimization(broken)
        lossy = apply_lossy_post(source, 1080)
        with self.assertRaises(ValueError):
            apply_post_optimization(lossy.replace("int b = (cnbPixel * 32 + ch2 * 4);",
                                                 "int b = (pk * 32 + ch2 * 4);"))


if __name__ == "__main__":
    unittest.main()
