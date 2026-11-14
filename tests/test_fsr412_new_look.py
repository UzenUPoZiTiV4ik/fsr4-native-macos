# SPDX-License-Identifier: MIT
"""new-look CPU/source contracts and an opt-in real Metal scalar-rule check."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from fsr412 import new_look as look
from fsr412 import lossy_shaders as lossy
from fsr412 import compile_fsr412 as compiler
from fsr412 import optimization
sys.path.pop(0)
CATALOG = json.loads((ROOT / "tools/fsr412/shader_catalog.json").read_text())["shaders"]
ROWS = [r for r in CATALOG if r["stage"] in ("pre", "post") and r["tier"] <= 2160 and r.get("colorspace") == 0]
AVAILABLE = all((ROOT / "build/fsr412_model/lite-shaders" / (r["sha256"] + ".metal")).is_file() for r in ROWS)


def rgb(value):
    return (value, value, value)


def case(**overrides):
    result = dict(current=rgb(.5), history=rgb(.5), history_share=.6,
                  samples=[rgb(.45), rgb(.55)] + [rgb(.5)] * 7,
                  history_samples=[rgb(.5)] * 9, band_samples=[rgb(.45)] * 4,
                  at_rest=True, other_motion=False, uncovered=False, depth_edge=False,
                  previous_note=0.0, history_valid=True)
    result.update(overrides)
    return result


class RuleTests(unittest.TestCase):
    def test_mid_weight_verdict_is_not_the_old_squared_high_weight_gate(self):
        result = look.resolve(**case())
        self.assertAlmostEqual(result["new_share"], .4 * (2 / 3))
        self.assertEqual(result["spark"], 0)
        self.assertEqual(result["output"], rgb(.5))

    def test_compact_bright_arrival_updates_but_wide_bands_do_not(self):
        c = case(current=rgb(.206), history=rgb(.2), history_share=.9,
                 samples=[rgb(.206)] * 9, history_samples=[rgb(.2)] * 9,
                 band_samples=[rgb(.2)] * 4)
        compact = look.resolve(**c)
        c["band_samples"] = [rgb(.206)] * 4
        band = look.resolve(**c)
        self.assertAlmostEqual(compact["arrival"], .2)
        self.assertAlmostEqual(band["arrival"], 0)
        self.assertGreater(compact["output"][0], band["output"][0])
        self.assertTrue(band["band"])

    def test_arrival_is_bright_only_and_uses_neighborhood_maximum(self):
        c = case(current=rgb(.8), history=rgb(.7), samples=[rgb(.8)] * 9,
                 history_samples=[rgb(.85)] + [rgb(.7)] * 8)
        self.assertEqual(look.resolve(**c)["arrival"], 0)
        c.update(current=rgb(.2), history=rgb(.7), samples=[rgb(.2)] * 9,
                 history_samples=[rgb(.7)] * 9)
        self.assertEqual(look.resolve(**c)["arrival"], 0)

    def test_spark_is_per_channel_and_requires_support_on_both_axes(self):
        samples = [(1.0, .01, .01)] * 9
        c = case(current=(1.0, .01, .01), history=rgb(.01), samples=samples,
                 history_samples=[rgb(.01)] * 9, band_samples=[rgb(.01)] * 4)
        self.assertEqual(look.resolve(**c)["spark"], 1)
        c["samples"] = [rgb(.01)] * 9
        c["samples"][3] = c["samples"][4] = c["samples"][5] = (1.0, .01, .01)
        self.assertEqual(look.resolve(**c)["spark"], 0)

    def test_speck_cap_preserves_a_line_through_history_center(self):
        neighborhood = [rgb(0)] * 9
        neighborhood[4] = rgb(1)
        c = case(current=rgb(.2), history=rgb(1), history_share=.99,
                 samples=[rgb(.2)] * 9, history_samples=neighborhood, at_rest=False)
        isolated = look.resolve(**c)
        neighborhood[3] = neighborhood[5] = rgb(.6)
        line = look.resolve(**c)
        self.assertFalse(isolated["line"])
        self.assertAlmostEqual(isolated["output"][0], .25)
        self.assertTrue(line["line"])
        self.assertGreater(line["output"][0], .99)

    def test_history_clamp_only_at_motion_boundaries(self):
        c = case(current=rgb(.4), history=rgb(1), history_share=.8,
                 samples=[rgb(.3), rgb(.5)] + [rgb(.4)] * 7,
                 history_samples=[rgb(.6)] * 4 + [rgb(1)] * 5, at_rest=False, history_valid=False)
        pan = look.resolve(**c)
        boundary = look.resolve(**{**c, "other_motion": True})
        self.assertGreater(pan["output"][0], .9)
        self.assertLessEqual(boundary["output"][0], .5)

    def test_dark_fleck_floor_needs_depth_edge_and_isolation(self):
        c = case(current=rgb(.05), history=rgb(1), history_share=.5,
                 samples=[rgb(.05)] + [rgb(.81)] * 8, history_samples=[rgb(1)] * 9,
                 at_rest=False, other_motion=True, depth_edge=True, history_valid=False)
        edge = look.resolve(**c)
        flat = look.resolve(**{**c, "depth_edge": False})
        self.assertAlmostEqual(edge["output"][0], .9)
        self.assertLess(flat["output"][0], .7)
        big_object = look.resolve(**{**c, "current": rgb(.2), "samples": [rgb(.2)] * 9})
        self.assertLess(big_object["output"][0], .7)

    def test_white_speck_ceiling_does_not_cap_a_large_arriving_object(self):
        c = case(current=rgb(8), history=rgb(1), history_share=.5,
                 samples=[rgb(8)] + [rgb(.8)] * 8, history_samples=[rgb(1)] * 9,
                 at_rest=False, depth_edge=True, history_valid=False)
        self.assertAlmostEqual(look.resolve(**c)["output"][0], 1.5)
        self.assertGreater(look.resolve(**{**c, "depth_edge": False})["output"][0], 1.5)
        c["samples"] = [rgb(10)] * 9
        self.assertGreater(look.resolve(**c)["output"][0], 1.5)

    def test_uncovered_guard_runs_after_fleck_rules(self):
        c = case(current=rgb(.05), history=rgb(1), history_share=.5,
                 samples=[rgb(.05)] + [rgb(.81)] * 8, history_samples=[rgb(1)] * 9,
                 at_rest=False, depth_edge=True, uncovered=True)
        self.assertEqual(look.resolve(**c)["output"], rgb(.05))

    def test_green_fade_uses_network_weight_not_its_square(self):
        c = case(current=rgb(.502), history=rgb(.5), history_share=.5,
                 samples=[rgb(.502)] * 9, history_samples=[rgb(.5)] * 9,
                 band_samples=[rgb(.5)] * 4, previous_note=4)
        self.assertTrue(look.resolve(**c)["slow"])
        self.assertEqual(look.resolve(**c)["new_share"], .5)
        self.assertFalse(look.resolve(**{**c, "previous_note": 0})["slow"])

    def test_dead_band_releases_a_persistent_fade_and_reset(self):
        self.assertEqual(look.finish(rgb(1.0005), rgb(1.0001), rgb(1))[0], rgb(1))
        self.assertEqual(look.finish(rgb(1.0005), rgb(1.003), rgb(1), 4)[0], rgb(1.0005))
        self.assertEqual(look.finish(rgb(1.0005), rgb(1.0001), rgb(1), history_valid=False)[0], rgb(1.0005))
        self.assertEqual(look.note((1.03, .9, 1), rgb(1)), -4)

    def test_nonfinite_inputs_are_not_accepted_by_cpu_spec(self):
        with self.assertRaises(ValueError):
            look.resolve(**case(current=rgb(math.nan)))


@unittest.skipUnless(AVAILABLE, "imported Lite shader artifacts not present")
class SourceTests(unittest.TestCase):
    def test_all_linear_pre_and_post_variants_are_idempotent_and_keep_bindings(self):
        for row in ROWS:
            source = compiler.source_for(row).read_text()
            with self.subTest(stage=row["stage"], tier=row["tier"], sha=row["sha256"]):
                if row["stage"] == "pre":
                    for reuse in (False, True):
                        text = lossy.apply_lossy_pre(source, row["tier"], bool(row.get("low_res_mv")),
                            bool(row.get("depth_inverted")), reuse, new_look=True)
                        self.assertEqual(text, lossy.apply_lossy_pre(text, row["tier"], bool(row.get("low_res_mv")),
                            bool(row.get("depth_inverted")), reuse, new_look=True))
                        self.assertIn("half(fsr412LookPreviousNote)", text)
                        self.assertIn("float fsr412LookPreviousNote = 0.0f;", text)
                else:
                    source = compiler.external_post_cnb(source, row["tier"])
                    for prepared in (False, True):
                        src = optimization.input_color_cache(source, 9)[0] if prepared else source
                        text = lossy.apply_lossy_post(src, row["tier"], new_look=True)
                        self.assertEqual(text, lossy.apply_lossy_post(text, row["tier"], new_look=True))
                        main = text[text.index("kernel void"):]
                        self.assertEqual(main.count("rw_history_color.write("), 4)
                        self.assertEqual(main.count("rw_recurrent_0.write("), 4)
                        self.assertEqual(main.count("Fsr412LossyCurrent["), 1)
                        self.assertEqual(main.count("Fsr412LossyPrevious["), 1)
                        self.assertEqual(main.count("ScratchBuffer._m0["), 4)
                        self.assertIn("fsr412BandSamples[3] =", main)
                        self.assertIn("Fsr412LookFinish(", main)
                indices = re.findall(r"\[\[buffer\((\d+)\)\]\]", text[text.index("kernel void"):].split("\n")[0])
                self.assertEqual(len(indices), len(set(indices)))
                self.assertLessEqual(max(map(int, indices)), 7)

    def test_opt_out_is_byte_identical_to_v31_transform(self):
        try:
            old = subprocess.check_output(["git", "show", "2a81163:tools/fsr412/lossy_shaders.py"], cwd=ROOT, stderr=subprocess.DEVNULL).decode()
        except (OSError, subprocess.CalledProcessError):
            self.skipTest("v3.1 source history unavailable")
        namespace = {"__name__": "fsr412_lossy_v31"}
        exec(compile(old, "v3.1/lossy_shaders.py", "exec"), namespace)
        for row in ROWS:
            source = compiler.source_for(row).read_text()
            if row["stage"] == "pre":
                for reuse in (False, True):
                    args = (source, row["tier"], bool(row.get("low_res_mv")), bool(row.get("depth_inverted")), reuse)
                    self.assertEqual(lossy.apply_lossy_pre(*args), namespace["apply_lossy_pre"](*args))
            else:
                source = compiler.external_post_cnb(source, row["tier"])
                for prepared in (False, True):
                    src = optimization.input_color_cache(source, 9)[0] if prepared else source
                    self.assertEqual(lossy.apply_lossy_post(src, row["tier"]), namespace["apply_lossy_post"](src, row["tier"]))

    @unittest.skipUnless(shutil.which("xcrun"), "Apple Metal compiler unavailable")
    def test_real_pre_and_post_compile_with_fast_math(self):
        pre = next(r for r in ROWS if r["stage"] == "pre" and r["tier"] == 1080 and r.get("low_res_mv") and r.get("depth_inverted"))
        post = next(r for r in ROWS if r["stage"] == "post" and r["tier"] == 1080 and r["preset"] == "normal")
        pre_text = lossy.apply_lossy_pre(compiler.signed_square(compiler.source_for(pre).read_text()),
            1080, True, True, True, new_look=True)
        base = compiler.external_post_cnb(compiler.source_for(post).read_text(), 1080)
        sources = {"pre": pre_text, "post": lossy.apply_lossy_post(base, 1080, new_look=True),
                   "prepared": lossy.apply_lossy_post(optimization.input_color_cache(base, 9)[0], 1080, new_look=True)}
        with tempfile.TemporaryDirectory(prefix="fsr412-look-compile-") as tmp:
            for name, source in sources.items():
                path = Path(tmp) / (name + ".metal")
                path.write_text(source)
                built = subprocess.run(["xcrun", "metal", "-std=macos-metal2.3", "-O3", "-c", str(path), "-o", str(path.with_suffix(".air"))], capture_output=True, text=True)
                self.assertEqual(built.returncode, 0, built.stdout + built.stderr)


GPU_SHADER = r"""
#include <metal_stdlib>
using namespace metal;
""" + look.COLOR_RULES + r"""
struct LookCase {
    float4 current, history, control;
    float4 samples[9], old[9], bands[4];
};
kernel void new_look_test(const device LookCase* cases [[buffer(0)]],
    device float4* outputs [[buffer(1)]], uint i [[thread_position_in_grid]]) {
    LookCase c = cases[i];
    float3 samples[9], old[9], bands[4];
    for (uint j=0u;j<9u;++j) {samples[j]=c.samples[j].xyz;old[j]=c.old[j].xyz;}
    for (uint j=0u;j<4u;++j) bands[j]=c.bands[j].xyz;
    uint flags=uint(c.control.z);
    float3 value=Fsr412LookResolve(c.current.xyz,c.history.xyz,c.control.x,samples,old,bands,
        (flags&1u)!=0u,(flags&2u)!=0u,(flags&4u)!=0u,(flags&8u)!=0u,c.control.y);
    float note;
    value=Fsr412LookFinish(value,c.current.xyz,c.history.xyz,c.control.y,(flags&16u)!=0u,note);
    outputs[i]=float4(value,note);
}
"""


@unittest.skipUnless(os.environ.get("FSR412_NEW_LOOK_GPU_TEST") == "1", "run make test-fsr412-new-look for GPU conformance")
class GPUTests(unittest.TestCase):
    def test_cpu_formulas_match_real_metal_with_canaries(self):
        rng = random.Random(41220)
        cases = [case(), case(current=rgb(.206), history=rgb(.2), history_share=.9,
                             samples=[rgb(.206)]*9, history_samples=[rgb(.2)]*9, band_samples=[rgb(.2)]*4)]
        for _ in range(1000):
            h = tuple(rng.uniform(.01, 1.5) for _ in range(3))
            c = tuple(rng.uniform(0, 2) for _ in range(3))
            samples = [tuple(rng.uniform(0, 2) for _ in range(3)) for _ in range(9)]
            old = [tuple(max(0, v+rng.uniform(-.2,.2)) for v in h) for _ in range(9)]
            old[4] = h
            cases.append(case(current=c, history=h, history_share=rng.random(), samples=samples,
                history_samples=old, band_samples=[tuple(rng.uniform(0,2) for _ in range(3)) for _ in range(4)],
                at_rest=rng.choice((False,True)), other_motion=rng.choice((False,True)),
                uncovered=rng.choice((False,True)), depth_edge=rng.choice((False,True)),
                previous_note=rng.choice((-5,-4,-3,-1,0,1,3,4,5)), history_valid=rng.choice((False,True))))
        payload=bytearray(); expected=[]
        f32=lambda x:struct.unpack("<f",struct.pack("<f",x))[0]
        for raw in cases:
            c={key:([tuple(f32(v) for v in row) for row in value] if key in ("samples","history_samples","band_samples")
                    else tuple(f32(v) for v in value) if key in ("current","history")
                    else f32(value) if key in ("history_share","previous_note") else value) for key,value in raw.items()}
            flags=sum(int(c[key])*bit for key,bit in (("at_rest",1),("other_motion",2),("uncovered",4),("depth_edge",8),("history_valid",16)))
            values=[*c["current"],0,*c["history"],0,c["history_share"],c["previous_note"],flags,0]
            for key in ("samples","history_samples","band_samples"):
                for row in c[key]:values.extend((*row,0))
            payload.extend(struct.pack("<100f",*values))
            result=look.resolve(**c);expected.append((*result["output"],result["note"]))
        helper=ROOT/"build/tests/fsr412_new_look_gpu"
        self.assertTrue(helper.is_file(),"build the GPU runner first")
        with tempfile.TemporaryDirectory(prefix="fsr412-look-gpu-") as tmp:
            folder=Path(tmp);source=folder/"rules.metal";source.write_text(GPU_SHADER)
            air=folder/"rules.air";library=folder/"rules.metallib"
            for cmd in (["xcrun","metal","-std=macos-metal2.3","-O3","-c",str(source),"-o",str(air)],
                        ["xcrun","metallib",str(air),"-o",str(library)]):
                built=subprocess.run(cmd,capture_output=True,text=True)
                self.assertEqual(built.returncode,0,built.stdout+built.stderr)
            inputs=folder/"inputs.bin";inputs.write_bytes(payload)
            outputs=[]
            for api in ("classic","metal4"):
                output=folder/(api+".bin")
                ran=subprocess.run([str(helper),str(library),str(inputs),str(output),api],capture_output=True,text=True)
                self.assertEqual(ran.returncode,0,ran.stdout+ran.stderr)
                data=output.read_bytes();self.assertEqual(len(data),len(cases)*16)
                for index,(actual,wanted) in enumerate(zip(struct.iter_unpack("<4f",data),expected)):
                    for a,b in zip(actual,wanted):
                        self.assertTrue(math.isfinite(a),f"nonfinite GPU value at case {index}")
                        self.assertAlmostEqual(a,b,delta=3e-5,msg=f"{api}, case {index}, {actual} != {wanted}")
                outputs.append(data)
            self.assertEqual(outputs[0],outputs[1],"Metal 3/4 scalar rules differ")


if __name__ == "__main__":
    unittest.main()
