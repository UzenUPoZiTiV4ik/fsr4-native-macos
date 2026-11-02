# SPDX-License-Identifier: MIT
"""CPU specifications/source contracts only. No GPU or heavy build required."""
from dataclasses import replace
from pathlib import Path
import math
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools/fsr411"))
try:
    import lossy_shaders as lossy
    import post_optimization as post_cache
    import temporal_quality as quality
finally:
    sys.path.pop(0)


SOURCES = (
    (1080, "shaders/reimpl411", "pre_rev_411.hlsl", "post_rev_411.hlsl"),
    (2160, "shaders/reimpl2160", "pre_rev_411_v15_2160.hlsl", "post_rev_411_2160.hlsl"),
    (4320, "shaders/reimpl4320", "pre_rev_411_v15_4320.hlsl", "post_rev_411_4320.hlsl"),
    (1080, "ultraperformance/reimpl411", None, "post_rev_411.hlsl"),
    (2160, "ultraperformance/reimpl2160", None, "post_rev_411_2160.hlsl"),
    (4320, "ultraperformance/reimpl4320", None, "post_rev_411_4320.hlsl"),
)


def shader(folder, name):
    return (ROOT / "third_party/fsr411" / folder / name).read_text()


def body(source, name):
    start = source.index("{", source.index(name))
    depth = 1
    for end in range(start + 1, len(source)):
        depth += (source[end] == "{") - (source[end] == "}")
        if depth == 0:
            return source[start + 1:end]
    raise AssertionError("unclosed shader body")


class ThresholdTests(unittest.TestCase):
    def test_default_noise_and_sign_encoding(self):
        history = (0.5, 0.5, 0.5)
        self.assertEqual(quality.pixel_signs(history, history), 0)
        for delta in (-0.019, 0.019):
            self.assertEqual(quality.pixel_signs((0.5 + delta, 0.5, 0.5), history) & 3, 0)
        self.assertEqual(quality.pixel_signs((0.53, 0.5, 0.5), history) & 3, 1)
        self.assertEqual(quality.pixel_signs((0.47, 0.5, 0.5), history) & 3, 2)
        self.assertFalse(quality.matching_color_sign(0, 0))
        self.assertFalse(quality.matching_color_sign(3, 3))

    def test_repeated_changes_without_vectors_pass_only_matching_channel(self):
        history = (0.5, 0.5, 0.5)
        dark_leaf = quality.pixel_signs((0.475, 0.475, 0.475), history)
        next_leaf = quality.pixel_signs((0.47, 0.47, 0.47), history)
        bright_noise = quality.pixel_signs((0.53, 0.53, 0.53), history)
        self.assertTrue(quality.matching_color_sign(dark_leaf, next_leaf))
        self.assertFalse(quality.matching_color_sign(dark_leaf, bright_noise))
        self.assertFalse(quality.matching_color_sign(dark_leaf, 0))
        self.assertFalse(quality.matching_color_sign(1, 1 << 2))
        self.assertTrue(quality.matching_color_sign(2 << 4, 2 << 4))

    def test_slow_bright_and_dark_fades_bypass_squaring_not_noise(self):
        history = (0.5, 0.5, 0.5)
        for current in ((0.502,)*3, (0.498,)*3):
            note = quality.pixel_signs(current, history)
            self.assertEqual(note & 63, 0)  # Below RGB floor, above drift floor.
            self.assertTrue(quality.slow_fade(current, history, note))
            self.assertFalse(quality.slow_fade(current, history, 0))
            opposite = quality.pixel_signs(tuple(1 - c for c in current), history)
            self.assertFalse(quality.slow_fade(current, history, opposite))
        for current in ((0.50005,)*3, (0.52,)*3, (0.48,)*3):
            self.assertFalse(quality.slow_fade(current, history, quality.pixel_signs(current, history)))
        # Absolute floor keeps the dark endpoint from accepting arbitrarily tiny changes.
        self.assertFalse(quality.slow_fade((0.00001,)*3, (0.0,)*3, 1 << 6))
        self.assertFalse(quality.slow_fade((0.00011,)*3, (0.0,)*3, 1 << 6))

    def test_per_pixel_pack_retains_all_four_notes_and_parities(self):
        history = [(0.5,)*3] * 4
        current = [(0.53,0.5,0.5), (0.5,0.47,0.5), (0.5,0.5,0.53), (0.5,)*3]
        packed = quality.pack_signs(current, history)
        self.assertLessEqual(packed, 0xFFFFFFFF)
        for phase in range(4):
            self.assertEqual((packed >> (phase * 8)) & 255,
                             quality.pixel_signs(current[phase], history[phase]))
        for values in ([], [history[0]]*3, [history[0]]*5):
            with self.assertRaises(ValueError):
                quality.pack_signs(values, history)

    def test_nonfinite_differences_are_neutral(self):
        for value in (math.nan, math.inf, -math.inf):
            self.assertEqual(quality.pixel_signs((value,0.5,0.5), (0.5,)*3), 0)
            self.assertEqual(quality.pixel_signs((0.5,)*3, (0.5,value,0.5)), 0)
            self.assertFalse(quality.slow_fade((value,)*3, (0.5,)*3, 1 << 6))

    def test_isolated_and_one_axis_sparks_do_not_pass(self):
        self.assertEqual(quality.spark_response(1, .1, .1, .1, .1, .1, .1), 0)
        self.assertEqual(quality.spark_response(1, 1, 1, .1, .1, .1, .1), 0)
        self.assertEqual(quality.spark_response(1, .1, .1, 1, 1, .1, .1), 0)
        self.assertEqual(quality.spark_response(.17, .17, .17, .17, .17, .1, .1), 0)
        self.assertGreater(quality.spark_response(.24, .24, .1, .24, .1, .1, .1), 0)
        self.assertEqual(quality.spark_response(10, 10, 10, 10, 10, .1, .1), 1)
        # Existing bright texture in the neighbourhood cannot be called a new spark.
        self.assertEqual(quality.spark_response(.9, .9, .9, .9, .9, .1, 1), 0)
        self.assertEqual(quality.spark_response(math.nan, 1, 1, 1, 1, .1, .1), 0)

    def test_bounds_reject_unbounded_or_reversed_settings(self):
        for key, value in (("noise_floor", 0), ("noise_floor", 1),
                           ("fade_floor", 0), ("fade_upper", 1),
                           ("spark_floor", 0), ("spark_cap", 100),
                           ("spark_span", math.nan), ("fade_lower", math.inf)):
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                replace(quality.DEFAULT_SETTINGS, **{key:value})
        with self.assertRaises(ValueError):
            replace(quality.DEFAULT_SETTINGS, fade_lower=.005, fade_upper=.005)
        with self.assertRaises(ValueError):
            replace(quality.DEFAULT_SETTINGS, spark_floor=.2, spark_cap=.1)


class HlslCompileTests(unittest.TestCase):
    @unittest.skipUnless((ROOT / "tools/bin/dxc").is_file() and
                         (ROOT / "tools/bin/spirv-cross").is_file(), "native shader tools not installed")
    def test_standalone_helpers_compile_and_msl_keeps_scalar_sign_store(self):
        # CPU-only shader validation, no Metal compiler/library or GPU dispatch.
        common = """
cbuffer TestFrame : register(b1) {
    uint width; uint height; uint reset; uint width_lr;
    uint height_lr; float2 inv_scale; float2 jitter;
};
Texture2D<float3> r_input_color : register(t3);
Texture2D<float3> r_reprojected_color : register(t9);
RWStructuredBuffer<float3> TestResult : register(u0);
float3 ApplyMuLaw411(float3 color, float exposure) { return color * exposure; }
""" + lossy._declarations(1080)
        pre = common + quality.DECLARATIONS + quality.PRE_HELPERS + """
[numthreads(1,1,1)]
void main(uint3 id : SV_DispatchThreadID) {
    Fsr411QualityRecordMotion(id.xy, r_input_color[id.xy].xy, false);
}
"""
        post = common + lossy.POST_HELPERS + quality.DECLARATIONS + quality.POST_HELPERS
        for key, value in quality.DEFAULT_SETTINGS.__dict__.items():
            post = post.replace("@"+key+"@", repr(float(value)))
        post += """
[numthreads(1,1,1)]
void main(uint3 id : SV_DispatchThreadID) {
    if (!Fsr411QualityEnabled()) return;
    bool atRest, otherMotion, uncovered;
    Fsr411QualityClassify(int2(id.xy >> 1u), Fsr411LossyReadMotion(int2(id.xy >> 1u)),
                         atRest, otherMotion, uncovered);
    Fsr411QualityRecordSigns(id.xy, 1.0f);
    float3 current = r_input_color[id.xy], history = r_reprojected_color[id.xy];
    TestResult[id.x] = Fsr411QualityResolve(id.xy, 1.0f, current, history, current - 0.1f,
                                         current + 0.1f, 0.5f, atRest, otherMotion, uncovered);
}
"""
        with tempfile.TemporaryDirectory(prefix="fsr411-quality-") as directory:
            path = Path(directory)
            for stage, source in (("pre",pre), ("post",post)):
                hlsl = path / (stage + ".hlsl")
                spv = path / (stage + ".spv")
                hlsl.write_text(source)
                built = subprocess.run([str(ROOT / "tools/bin/dxc"), "-spirv", "-T", "cs_6_6",
                    "-E", "main", "-HV", "2021", "-O3", "-fspv-target-env=vulkan1.3",
                    "-fvk-t-shift", "32", "0", "-fvk-u-shift", "64", "0", "-Fo",str(spv),str(hlsl)],
                    capture_output=True, text=True)
                self.assertEqual(built.returncode,0,built.stderr)
                msl = subprocess.run([str(ROOT / "tools/bin/spirv-cross"),str(spv),
                                      "--msl","--msl-version","20300"], capture_output=True,text=True)
                self.assertEqual(msl.returncode,0,msl.stderr)
                if stage == "post":
                    # Prevent a backend vector read/modify/write race with xy/w readers.
                    writes = [line.strip() for line in msl.stdout.splitlines()
                              if "Fsr411QualityCurrent." in line and re.search(r"\]\s*=(?!=)",line)]
                    self.assertEqual(len(writes),1,writes)
                    # DXC lowers uint4's scalar z into a component pointer;
                    # SPIRV-Cross emits (&record[index])[2], not a uint4 RMW.
                    self.assertRegex(writes[0],r"\]\)\[2\]\s*=")
                    previous_writes = [line for line in msl.stdout.splitlines()
                                       if "Fsr411QualityPrevious." in line and re.search(r"\]\s*=(?!=)",line)]
                    self.assertEqual(previous_writes,[])


    @unittest.skipUnless((ROOT / "tools/bin/dxc").is_file() and
                         (ROOT / "tools/bin/spirv-cross").is_file(), "native shader tools not installed")
    def test_fast_notes_wave_gather_compiles_to_msl_without_group_barrier(self):
        # Shader compilation only: no library loading, pipeline or GPU dispatch.
        source="""
cbuffer TestFrame : register(b1) {
    uint width; uint height; uint reset; uint width_lr;
    uint height_lr; float2 inv_scale; float2 jitter;
};
Texture2D<float3> r_input_color : register(t3);
Texture2D<float3> r_reprojected_color : register(t9);
float3 ApplyMuLaw411(float3 color, float exposure) { return color * exposure; }
"""+lossy._declarations(1080)+lossy.POST_HELPERS+quality.DECLARATIONS+quality.POST_HELPERS+quality.FAST_NOTE_HELPERS
        for key,value in quality.DEFAULT_SETTINGS.__dict__.items():source=source.replace("@"+key+"@",repr(float(value)))
        source+="""
[numthreads(256,1,1)]
void main(uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID) {
    uint2 pixel=group.xy*16u+uint2(local.x&15u,local.x>>4u);
    if (pixel.x>=width || pixel.y>=height) return;
    float3 current=r_input_color[pixel], history=r_reprojected_color[pixel];
    Fsr411QualityRecordSignsFast(pixel,local.x,current,history,1.0f);
}
"""
        with tempfile.TemporaryDirectory(prefix="fsr411-fast-notes-") as directory:
            path=Path(directory);hlsl=path/"fast.hlsl";spv=path/"fast.spv";hlsl.write_text(source)
            built=subprocess.run([str(ROOT/"tools/bin/dxc"),"-spirv","-T","cs_6_6","-E","main",
                "-HV","2021","-O3","-fspv-target-env=vulkan1.3","-fvk-t-shift","32","0",
                "-fvk-u-shift","64","0","-Fo",str(spv),str(hlsl)],capture_output=True,text=True)
            self.assertEqual(built.returncode,0,built.stderr)
            converted=subprocess.run([str(ROOT/"tools/bin/spirv-cross"),str(spv),"--msl","--msl-version","20300"],
                                     capture_output=True,text=True)
            self.assertEqual(converted.returncode,0,converted.stderr)
            msl=converted.stdout
            # SPIRV-Cross wraps simd_shuffle in overloaded helpers. Count the
            # four actual kernel calls, not those template definitions.
            calls=list(re.finditer(r"uint _\d+ = spvSubgroupShuffle\([^;]+\);",msl))
            self.assertEqual(len(calls),4)
            self.assertRegex(msl,r"gl_SubgroupSize \[\[(?:thread_execution_width|threads_per_simdgroup)\]\]")
            self.assertIn("thread_index_in_simdgroup",msl)
            self.assertRegex(msl[calls[-1].end():],r"if \(\(\([^\n]+ & 1u\) == 0u\)")
            self.assertNotIn("threadgroup_barrier",msl)
            writes=[line.strip() for line in msl.splitlines() if "Fsr411QualityCurrent." in line and re.search(r"\]\s*=(?!=)",line)]
            self.assertGreaterEqual(len(writes),1)
            for write in writes:self.assertRegex(write,r"\]\)\[2\]\s*=")
            if shutil.which("xcrun"):
                metal=path/"fast.metal";metal.write_text(msl)
                compiled=subprocess.run(["xcrun","metal","-std=macos-metal2.3","-O0","-c",str(metal),
                                         "-o",str(path/"fast.air")],capture_output=True,text=True)
                self.assertEqual(compiled.returncode,0,compiled.stdout+compiled.stderr)


class FastNoteTests(unittest.TestCase):
    def test_flags_and_policy_are_explicit_and_noise_guard_is_shared(self):
        self.assertEqual(quality.DEFAULT_FLAGS,3)
        self.assertEqual(quality.FAST_NOTES_FLAG,4)
        self.assertEqual(quality.FAST_FLAGS,7)
        history=[(.5,)*3]*4
        reference=[(.505,)*3]*4
        gaussian=[(.525,)*3]*4
        self.assertNotEqual(quality.pack_signs(reference,history),quality.pack_signs(gaussian,history))
        self.assertEqual(quality.pack_signs(history,history),0)
        self.assertEqual(quality.pixel_signs((math.nan,.5,.5),(.5,)*3),0)
        helpers=quality.FAST_NOTE_HELPERS
        self.assertIn("note = Fsr411QualitySigns(current, history);",helpers)
        self.assertNotIn("Fsr411QualityReference",helpers)
        self.assertNotIn("r_input_color",helpers)
        self.assertNotIn("r_reprojected_color",helpers)
        self.assertNotIn("GroupMemoryBarrier",helpers)
        self.assertNotIn("Interlocked",helpers)
        self.assertIn("WaveGetLaneCount() != 32u",helpers)
        self.assertIn("WaveActiveAllTrue(lane == (groupLane & 31u))",helpers)
        self.assertIn("WaveActiveAllEqual(groupLane >> 5u)",helpers)
        self.assertEqual(helpers.count("Fsr411QualityCurrent[index].z = packed;"),1)
        self.assertLess(helpers.index("WaveReadLaneAt(note, base + 17u)"),
                        helpers.index("if (((pixel.x | pixel.y) & 1u) == 0u)"))
        self.assertLess(helpers.index("if ((qualityFlags & 4u) == 0u"),helpers.index("WaveGetLaneIndex()"))
        # Fallback calls precede unconditional returns; no fallback can also write fast z.
        self.assertEqual(helpers.count("Fsr411QualityRecordSigns(pixel, exposure);\n        return;"),2)

    def test_wave32_lane_gather_covers_every_even_cell_once_in_partial_tiles(self):
        # Includes aligned odd logical sizes and tiny/partial output tiles.
        logical_sizes=((1,1),(3,15),(17,65),(97,41),(2,2),(6,18),(18,2),(104,72))
        for logical_width,logical_height in logical_sizes:
            width=(logical_width+7)&~7;height=(logical_height+7)&~7
            expected={(x,y) for y in range(height//2) for x in range(width//2)}
            writes=[]
            for gy in range((height+15)//16):
                for gx in range((width+15)//16):
                    active={lane:(gx*16+(lane&15),gy*16+(lane>>4)) for lane in range(256)
                            if gx*16+(lane&15)<width and gy*16+(lane>>4)<height}
                    notes={lane:((active[lane][0]*13+active[lane][1]*7)&255) for lane in active}
                    for group_lane,pixel in active.items():
                        wave_start=group_lane&~31;wave_lane=group_lane&31;base=wave_lane&~17
                        sources=[wave_start+base+d for d in (0,1,16,17)]
                        self.assertTrue(all(s in active for s in sources),(width,height,gx,gy,group_lane))
                        self.assertTrue(all((s>>5)==(group_lane>>5) for s in sources))
                        origin=(pixel[0]&~1,pixel[1]&~1)
                        wanted=[(origin[0]+dx,origin[1]+dy) for dx,dy in ((0,0),(1,0),(0,1),(1,1))]
                        self.assertEqual([active[s] for s in sources],wanted)
                        packed=sum(notes[s]<<(phase*8) for phase,s in enumerate(sources))
                        wanted_pack=sum(((x*13+y*7)&255)<<(phase*8) for phase,(x,y) in enumerate(wanted))
                        self.assertEqual(packed,wanted_pack)
                        if (pixel[0]|pixel[1])&1==0:
                            writes.append((pixel[0]//2,pixel[1]//2))
            self.assertEqual(set(writes),expected)
            self.assertEqual(len(writes),len(expected))

    def test_unknown_wave_size_or_lane_mapping_falls_back(self):
        def supported(wave_size,ids):
            return wave_size==32 and all(lane==(thread&31) for lane,thread in ids) and len({thread>>5 for _,thread in ids})==1
        self.assertTrue(supported(32,[(i,i) for i in range(32)]))
        self.assertTrue(supported(32,[(i,i+64) for i in (0,1,16,17)]))
        for wave in (16,64):self.assertFalse(supported(wave,[(i,i) for i in range(wave)]))
        self.assertFalse(supported(32,[(i,i^1) for i in range(32)]))
        self.assertFalse(supported(32,[(0,0),(1,33),(16,16),(17,49)]))

    def test_only_pinned_cached_geometry_emits_fast_helper(self):
        legacy=lossy.apply_lossy_post(shader("shaders/reimpl411","post_rev_411.hlsl"),1080)
        cached=post_cache.apply_post_optimization(legacy)
        reference=quality.apply_temporal_quality_post(cached,1080)
        self.assertNotIn("fast-notes=1",reference)
        self.assertNotIn("Wave",reference)
        result=quality.apply_temporal_quality_post(cached,1080,fast_notes=True)
        self.assertIn("fast-notes=1",result)
        self.assertIn(quality.FAST_NOTE_HELPERS,result)
        self.assertNotIn(quality.FAST_NOTE_HELPERS,quality.apply_temporal_quality_post(legacy,1080,fast_notes=True))
        self.assertEqual(quality.apply_temporal_quality_post(result,1080,fast_notes=True),result)
        with self.assertRaises(ValueError):quality.apply_temporal_quality_post(reference,1080,fast_notes=True)
        with self.assertRaises(ValueError):quality.apply_temporal_quality_post(result,1080)
        # The complete conservative recording algorithm remains byte-identical.
        self.assertIn(body(quality.POST_HELPERS,"void Fsr411QualityRecordSigns("),result)
        for bad in (cached.replace("GID.xy * 16u + uint2(GTID.x & 15u, GTID.x >> 4u)",
                                   "GID.xy * 16u + uint2(GTID.x >> 4u, GTID.x & 15u)"),
                    cached.replace("if (outputPixel.x >= width || outputPixel.y >= height)","if (outputPixel.x >= width)"),
                    cached+"\nfor (uint pk = 0; pk < 4; pk++) {}\n"):
            with self.assertRaises(ValueError):quality.apply_temporal_quality_post(bad,1080,fast_notes=True)


class SourceContractTests(unittest.TestCase):
    def test_disabled_is_byte_exact_and_needs_no_prerequisites(self):
        for transform in (quality.apply_temporal_quality_pre, quality.apply_temporal_quality_post):
            for source in ("arbitrary untouched", "", shader("shaders/reimpl411", "post_rev_411.hlsl")):
                self.assertEqual(transform(source, 7, enabled=False), source)

    def test_all_pre_tiers_and_reuse_orders(self):
        for tier, folder, pre, _ in SOURCES[:3]:
            original = shader(folder, pre)
            legacy = lossy.apply_lossy_pre(original, tier)
            for reuse in (False, True):
                source = lossy.reuse_only_pre(legacy) if reuse else legacy
                result = quality.apply_temporal_quality_pre(source, tier)
                self.assertEqual(quality.apply_temporal_quality_pre(result, tier), result)
                main = body(result, "void fsr4_model_v07_fp8_no_scale_prepass(")
                self.assertLess(main.index("Fsr411QualityRecordMotion(dtid"), main.index("if (lossySkip != 0u) return;"))
                self.assertEqual(main.count("Fsr411QualityRecordMotion(dtid"), 1)
                self.assertEqual(result.count("register(b3)"), 1)
                self.assertEqual(result.count("register(u14)"), 1)
                self.assertEqual(result.count("register(u15)"), 1)
            result = quality.apply_temporal_quality_pre(legacy, tier)
            self.assertIn("Fsr411QualityRecordMotion(dtid", lossy.reuse_only_pre(result))
            self.assertEqual(original, shader(folder, pre))

    def test_all_post_tiers_presets_speck_and_cache_geometries(self):
        for tier, folder, _, post in SOURCES:
            for speck in (False, True):
                legacy = lossy.apply_lossy_post(shader(folder, post), tier, speck=speck)
                cached = post_cache.apply_post_optimization(legacy)
                variants = (legacy, cached, lossy.apply_lossy_post_cells(cached))
                for source in variants:
                    with self.subTest(tier=tier, folder=folder, speck=speck, cached=source.startswith(post_cache.MARKER)):
                        result = quality.apply_temporal_quality_post(source, tier)
                        self.assertEqual(quality.apply_temporal_quality_post(result, tier), result)
                        main = body(result, "void fsr4_model_v07_fp8_no_scale_postpass(")
                        self.assertEqual(main.count("Fsr411QualityRecordSigns(tid, exposure);"), 1)
                        self.assertNotIn("Fsr411QualityRecordSignsFast", result)
                        fast_result=quality.apply_temporal_quality_post(source,tier,fast_notes=True)
                        fast_main=body(fast_result,"void fsr4_model_v07_fp8_no_scale_postpass(")
                        if "fsr411-generated-post-color-cache:" in source:
                            self.assertEqual(fast_main.count("Fsr411QualityRecordSignsFast(tid, GTID.x, up, reproj, exposure);"),1)
                            self.assertNotIn("Fsr411QualityRecordSigns(tid, exposure);",fast_main)
                        else:
                            self.assertNotIn("Fsr411QualityRecordSignsFast",fast_result)
                        self.assertEqual(quality.apply_temporal_quality_post(fast_result,tier,fast_notes=True),fast_result)
                        self.assertEqual(main.count("Fsr411QualityResolve(tid, exposure"), 1)
                        self.assertEqual(main.count("Fsr411LossyResolve(up, reproj"), 1)
                        self.assertEqual(main.count("Fsr411LossySpeck(res"), 1)
                        self.assertLess(main.index("Fsr411QualityClassify(int2(ix, iy)"), main.index("bool convert"))
                        self.assertLess(main.index("Fsr411QualityResolve(tid"), main.index("Fsr411LossySpeck(res"))
                        # Neural-cache addressing is deliberately unchanged.
                        self.assertEqual(re.findall(r"ScratchBuffer.Load4\([^;]+", result),
                                         re.findall(r"ScratchBuffer.Load4\([^;]+", source))
                        self.assertEqual(main.count("rw_history_color[tid] = res;"), 1)
                        self.assertNotIn("GroupMemoryBarrier", quality.POST_HELPERS)
                        self.assertNotIn("Wave", quality.POST_HELPERS)

    def test_sign_plane_is_written_once_and_never_read_in_current_frame(self):
        helpers = quality.POST_HELPERS
        self.assertEqual(re.findall(r"Fsr411QualityCurrent\[[^\]]+\]\.[xyzw]\s*=", helpers),
                         ["Fsr411QualityCurrent[index].z ="])
        self.assertNotRegex(helpers, r"Fsr411QualityCurrent\[[^\]]+\]\s*=")
        self.assertEqual(helpers.count("Fsr411QualityCurrent[index].z"), 1)
        self.assertNotRegex(helpers, r"Fsr411QualityCurrent\[[^\]]+\](?!\.)")
        self.assertNotRegex(helpers, r"Fsr411QualityPrevious\[[^\]]+\](?:\.[xyzw])?\s*=")
        record = body(helpers, "void Fsr411QualityRecordSigns(")
        self.assertIn("((pixel.x | pixel.y) & 1u) != 0u", record)
        self.assertIn("for (uint phase = 0u; phase < 4u;", record)
        note = body(helpers, "uint Fsr411QualityPreviousNote(")
        self.assertLess(note.index("qualityHistoryValid == 0u"), note.index("Fsr411QualityPrevious[index].z"))
        self.assertLess(note.index("any(p < 0)"), note.index("Fsr411QualityPrevious[index].w"))
        self.assertNotIn("rw_history_color", helpers)
        self.assertNotIn("half", helpers)
        self.assertIn("round(clamp(pixels, -254.0f, 254.0f))", quality.PRE_HELPERS)
        self.assertIn("all(isfinite(pixels))", quality.PRE_HELPERS)

    def test_motion_rules_are_fine_but_network_transport_stays_coarse(self):
        classify = body(quality.POST_HELPERS, "void Fsr411QualityClassify(")
        self.assertIn("int2(1,0), int2(-1,0), int2(0,1), int2(0,-1)", classify)
        motion = body(quality.POST_HELPERS, "int2 Fsr411QualityMotion(")
        self.assertIn("(flags & 2u) == 0u && lossySkip != 0u", motion)
        self.assertIn("Fsr411LossyReadMotion(safe).xy", motion)
        source = lossy.apply_lossy_post(shader("shaders/reimpl411", "post_rev_411.hlsl"), 1080)
        result = quality.apply_temporal_quality_post(source, 1080)
        self.assertIn("int2 sourcePixel = int2(tid) + asint(lossyOwn.xy);", result)
        self.assertNotIn("lossyOwn.xy = asuint(Fsr411QualityMotion", result)
        # Thin stationary foreground/background within a coarse block: old
        # replicated motion cannot see the difference, immediate fine probes can.
        coarse = [(8,0), (8,0), (8,0)]
        fine = [(8,0), (0,0), (8,0)]
        differs = lambda a,b:any(abs(x-y)>1 for x,y in zip(a,b))
        self.assertFalse(differs(coarse[1], coarse[0]))
        self.assertTrue(differs(fine[1], fine[0]))

    def test_host_sizes_and_owner_coverage_for_both_post_dispatches(self):
        self.assertEqual(quality.record_buffer_bytes(1920,1080), 8294400)
        self.assertEqual(quality.record_buffer_bytes(3840,2160), 33177600)
        self.assertEqual(quality.CONSTANT_BYTES, 16)
        for w,h in ((1,2), (2,1), (0,0), (3,2), (2,3), (7682,4320), (2.0,2), (True,2)):
            with self.assertRaises(ValueError):
                quality.record_buffer_bytes(w,h)
        for width,height in ((8,8),(24,40),(72,48),(104,72)):
            expected = {(x,y) for x in range(width//2) for y in range(height//2)}
            owners = []
            for gy in range((height+15)//16):
                for gx in range((width+15)//16):
                    for lane in range(256):
                        px=gx*16+(lane&15); py=gy*16+(lane>>4)
                        if px<width and py<height and (px|py)&1==0:
                            owners.append((px//2,py//2))
            self.assertEqual(set(owners), expected)
            self.assertEqual(len(owners), len(expected))
            owners=[]
            for gy in range((height+31)//32):
                for gx in range((width+31)//32):
                    for lane in range(256):
                        ix=(gx<<4)|(lane&1)|((lane>>1)&14)
                        iy=(gy<<4)+(((lane>>5)<<1)|((lane>>1)&1))
                        if ix<width//2 and iy<height//2:
                            owners.append((ix,iy))
            self.assertEqual(set(owners), expected)
            self.assertEqual(len(owners), len(expected))

    def test_policies_markers_and_unknown_source_shapes_fail_closed(self):
        raw = shader("shaders/reimpl411", "post_rev_411.hlsl")
        source = lossy.apply_lossy_post(raw, 1080)
        result = quality.apply_temporal_quality_post(source, 1080)
        for bad,tier in ((raw,1080), ("",1080), (source,7), (source,2160),
                         (source.replace("rw_history_color[tid] = res;", "other_store = res;"),1080),
                         (source.replace("[numthreads(256, 1, 1)]", "[numthreads(128, 1, 1)]"),1080),
                         (source+"\nRWStructuredBuffer<uint4> extra : register(u14);",1080),
                         (source+"\ncbuffer extra : register(b3) {};",1080)):
            with self.assertRaises(ValueError):
                quality.apply_temporal_quality_post(bad,tier)
        with self.assertRaises(ValueError):
            quality.apply_temporal_quality_post(result,1080,settings=replace(quality.DEFAULT_SETTINGS,noise_floor=.03))
        with self.assertRaises(ValueError):
            quality.apply_temporal_quality_post(result,2160)
        with self.assertRaises(ValueError):
            quality.apply_temporal_quality_post(result+result,1080)
        for span in (0,1):
            generated=quality.apply_temporal_quality_post(source,1080,settings=replace(quality.DEFAULT_SETTINGS,spark_span=span))
            self.assertIn("FSR411_QUALITY_SPARK_SPAN = %d.0f;" % span, generated)
            self.assertNotIn("@spark", generated)


if __name__ == "__main__":
    unittest.main()
