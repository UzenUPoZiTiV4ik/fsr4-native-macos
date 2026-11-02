"""Standalone M5 DS16 candidates: real integer oracle, no automatic GPU runs.

Normal, Ultra Performance and trained Lite Lossy are tested with their actual
hash-verified external weights. Missing artifacts are skipped, never replaced
by synthetic coefficients. GPU correctness is opt-in and timing is never run
by this suite; coordinate a GPU window before setting the environment flag.
"""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import random
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
SPEC = importlib.util.spec_from_file_location("fsr412_native_tuning", ROOT / "tools/fsr412/native_model.py")
native = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(native)


def signed32(value):
    value &= 0xffffffff
    return value if value < 0x80000000 else value - 0x100000000


def quantize(acc, bias, residual, q):
    ash, rsh, shift, flags = q
    value = signed32(((acc + bias) & 0xffffffff) << ash)
    value = signed32(value + ((residual & 0xffffffff) << rsh))
    if flags & 1:
        value = max(value, 0)
    if shift:
        floor = value // (1 << shift)
        increment = (1 << (shift - 1)) if flags & 4 else (1 << (shift - 1)) - 1 + (floor & 1)
        value = signed32(value + increment) // (1 << shift)
    if flags & 2:
        value = (value + 32768) % 65536 - 32768
    return max(-128, min(127, value))


def actual_models():
    from fsr412.model_data import initializer_for
    for tier, preset, profile in [
        (1080, "normal", "int8-lite"),
        (1080, "ultraperformance", "int8-lite"),
        (1080, "normal", "int8-lite-lossy"),
        (2160, "normal", "int8-lite-lossy"),
        (1080, "ultraperformance", "int8-lite-lossy"),
        (4320, "normal", "int8-lite-lossy"),
    ]:
        try:
            path, data, _ = initializer_for(ROOT, tier, preset, profile)
        except (OSError, ValueError):
            continue
        yield tier, preset, profile, path, data


class IntegerOracleTests(unittest.TestCase):
    def test_ties_even_and_int32_overflow_are_not_float_rounding(self):
        for shift in (1, 6, 7, 8, 9):
            for result in range(-120, 120):
                halfway = result * (1 << shift) + (1 << (shift - 1))
                expected = result if result % 2 == 0 else result + 1
                self.assertEqual(quantize(halfway, 0, 0, (0, 0, shift, 0)), expected)
        self.assertEqual(quantize(0x7fffffff, 1, 0, (0, 0, 0, 0)), -128)
        self.assertEqual(quantize(0x7fffffff, 0, 0, (0, 0, 1, 0)), -128)
        self.assertEqual(quantize(65535, 0, 0, (0, 0, 0, 2)), -1)
        self.assertEqual(quantize(-3, 0, 0, (0, 0, 1, 0)), -2)
        self.assertEqual(quantize(-3, 0, 0, (0, 0, 1, 4)), -1)

    def test_actual_raw_ds16_coefficients_and_biases(self):
        seen = set()
        models = list(actual_models())
        if not models:
            self.skipTest("verified external Lite initializers unavailable")
        for tier, preset, profile, _, data in models:
            with self.subTest(tier=tier, preset=preset, profile=profile):
                passes, _ = native.model(tier, preset, data, profile)
                p = passes[4]
                ultra = preset == "ultraperformance"
                self.assertEqual(p["q"][4:], [[0, 0, 7 if ultra else 6, 0], [0, 0, 8 if ultra else 7, 0]])
                self.assertEqual(passes[1]["q"][:4], [[0, 0, 8 if ultra else 7, 0], [0, 0, 8 if ultra else 9, 1],
                                                       [0, 5, 6, 0], [0, 5, 6, 0]])
                matrix, = p["matrices"]
                bias, = p["bias_values"]
                self.assertEqual(bias, list(struct.unpack_from("<32i", data, 6912)))
                for tap in range(4):
                    for ic in range(16):
                        for oc in range(32):
                            self.assertEqual(matrix[(tap * 16 + ic) * 32 + oc],
                                             native.signed(data[4864 + tap * 512 + oc * 16 + ic]))
                # Input basis vectors exercise every tap/channel and both
                # quantizer halves; extrema cannot be hidden by all-zero data.
                for tap in range(4):
                    for ic in range(16):
                        for value in (-128, -1, 1, 127):
                            for oc in range(32):
                                raw = value * native.signed(data[4864 + tap * 512 + oc * 16 + ic])
                                packed = value * matrix[(tap * 16 + ic) * 32 + oc]
                                q = p["q"][4 if oc < 16 else 5]
                                self.assertEqual(quantize(raw, bias[oc], 0, q), quantize(packed, bias[oc], 0, q))
                seen.add(native.digest(data))
        # The trained model must remain a distinct coefficient set whenever
        # all three external initializers are installed.
        if (ROOT / "build/fsr412_lossy/initializers-lite-lossy.bin").exists() and len(models) == 6:
            self.assertEqual(len(seen), 3)

    def test_source_has_separate_probe_and_no_common_abi_changes(self):
        source = (ROOT / "src/fsr412_mpp.metal").read_text()
        for name in ("8x1", "16x1", "32x1", "64x1", "8x2", "16x2"):
            self.assertIn("FSR412_DS16(fsr412_ds16_coop_" + name, source)
        self.assertIn("fsr412_ds16_coop_64x1_sg4", source)
        self.assertIn("fsr412_mpp_tuning_probe", source)
        for width in (32, 64):
            self.assertIn(f"FSR412_CNB_DS(fsr412_cnb_ds_coop_{width}x2", source)
            self.assertIn(f"cnb_layout<{width},2>()", source)
        self.assertIn("fsr412_cnb_ds_tuning_probe", source)
        self.assertIn("[[buffer(3)]]", source)
        self.assertIn("simdgroup_barrier(mem_flags::mem_device)", source)
        self.assertNotIn("threadgroup_barrier", source)
        self.assertNotIn("threadgroup int", source)
        self.assertNotIn("half(", source)
        bench = (ROOT / "tools/bench/fsr412_mpp_tuning.mm").read_text()
        self.assertIn('mode="cpu"', bench)
        self.assertIn('mode=="timing"', bench)
        self.assertIn('full_frame_measured', bench)
        self.assertIn('--order-phase', bench)
        self.assertIn('measured-batch raw CPU oracle/B skip/guard mismatch', bench)
        self.assertIn('per_batch_oracle_outside_timing', bench)


@unittest.skipUnless(sys.platform == "darwin" and shutil.which("xcrun"), "Apple toolchain required")
class StandaloneToolchainTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix="fsr412-mpp-test-")
        cls.directory = Path(cls.tmp.name)
        cls.bench = cls.directory / "bench"
        subprocess.run(["xcrun", "clang++", "-std=c++17", "-O2", "-fobjc-arc",
                        "-framework", "Foundation", "-framework", "Metal",
                        str(ROOT / "tools/bench/fsr412_mpp_tuning.mm"), "-o", str(cls.bench)],
                       check=True, capture_output=True, text=True)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_metal_generic_and_specialized_compile_without_dispatch(self):
        for specialized in (0, 1):
            with self.subTest(specialized=specialized):
                subprocess.run(["xcrun", "metal", "-std=metal4.0", "-O2",
                                f"-DFSR411_SPECIALIZE={specialized}", "-I", str(ROOT / "src"),
                                "-c", str(ROOT / "src/fsr412_mpp.metal"), "-o",
                                str(self.directory / f"candidate-{specialized}.air")],
                               check=True, capture_output=True, text=True)

    def test_real_models_cpu_integer_parity_and_odd_boundaries(self):
        models = list(actual_models())
        if not models:
            self.skipTest("verified external Lite initializers unavailable")
        for tier, preset, profile, path, data in models:
            with self.subTest(tier=tier, preset=preset, profile=profile), tempfile.TemporaryDirectory(dir=self.directory) as tmp:
                native.pack(tier, preset, data, Path(tmp), profile)
                run = subprocess.run([str(self.bench), str(ROOT), tmp, str(path), "--cpu-only"],
                                     check=True, capture_output=True, text=True)
                result = json.loads(run.stdout)
                self.assertEqual(result["cpu_oracle_cases"], 50)
                self.assertEqual(result["initializer_sha256"], native.digest(data))
                self.assertFalse(result["full_frame_measured"])
                self.assertNotIn("gpu_oracle_cases", result)

    def test_real_models_cpu_cnb_ds_fusion_and_retained_skip(self):
        models = list(actual_models())
        if not models:
            self.skipTest("verified external Lite initializers unavailable")
        for tier, preset, profile, path, data in models:
            with self.subTest(tier=tier, preset=preset, profile=profile), tempfile.TemporaryDirectory(dir=self.directory) as tmp:
                native.pack(tier, preset, data, Path(tmp), profile)
                run = subprocess.run([str(self.bench), str(ROOT), tmp, str(path), "--cpu-only", "--scope", "cnb-ds"],
                                     check=True, capture_output=True, text=True)
                result = json.loads(run.stdout)
                # Includes floor/ceil DS sizes, even when DS is zero-sized.
                # Full expected scratch retains every valid odd-edge B pixel.
                self.assertEqual(result["fused_cpu_oracle_cases"], 120)
                self.assertFalse(result["full_frame_measured"])
                self.assertNotIn("fused_gpu_oracle_cases", result)

    def test_original_opt_in_fusion_policy_cpu(self):
        exe = self.directory / "policy"
        subprocess.run(["xcrun", "clang++", "-std=c++17", "-O2", "-Wall", "-Wextra",
                        str(ROOT / "tests/fsr412_mpp_policy_test.cpp"), "-o", str(exe)],
                       check=True, capture_output=True, text=True)
        run = subprocess.run([str(exe)], check=True, capture_output=True, text=True)
        self.assertIn("147 checks pass", run.stdout)
        policy = (ROOT / "src/fsr412_mpp_policy.h").read_text()
        source = (ROOT / "src/fsr412_mpp.metal").read_text()
        for name in ("fsr412_cnb_ds_coop_32x2", "fsr412_cnb_ds_tuning_probe"):
            self.assertIn(name, policy)
            self.assertIn(name, source)
        for constant in (12, 13, 14, 15, 16, 17, 22, 23):
            self.assertIn(f"function_constant({constant})", source)

    def test_fusion_rejects_alias_and_wrong_b_binding(self):
        code = self.directory / "fusion-contract.cpp"
        exe = self.directory / "fusion-contract"
        code.write_text('#include "tools/bench/fsr412_mpp_oracle.h"\n#include <cassert>\n'
                        'int main(){using namespace fsr412_tuning_cpu;'
                        'for(bool floor:{true,false})for(bool bank:{true,false})for(unsigned w:{1u,3u,33u,65u,129u}){'
                        'auto p=fused_shape({}, {}, w, 5, bank, floor);assert(valid_fused_params(p));'
                        'auto bad=p;bad.ds.input.offset+=16;assert(!valid_fused_params(bad));'
                        'bad=p;bad.ds.output.width=w+1;assert(!valid_fused_params(bad));'
                        'bad=p;bad.cnb.input=bad.cnb.output;assert(!valid_fused_params(bad));}}\n')
        subprocess.run(["xcrun", "clang++", "-std=c++17", "-O2", "-I", str(ROOT), str(code), "-o", str(exe)],
                       check=True, capture_output=True, text=True)
        subprocess.run([str(exe)], check=True, capture_output=True, text=True)

    def test_fused_metal_pipeline_compile_without_dispatch(self):
        models = list(actual_models())
        if not models:
            self.skipTest("verified external Lite initializers unavailable")
        tier, preset, profile, path, data = models[0]
        with tempfile.TemporaryDirectory(dir=self.directory) as tmp:
            native.pack(tier, preset, data, Path(tmp), profile)
            run = subprocess.run([str(self.bench), str(ROOT), tmp, str(path), "--compile-only", "--scope", "cnb-ds"],
                                 check=True, capture_output=True, text=True)
            result = json.loads(run.stdout)
            self.assertEqual(len(result["compiled"]), 2)
            self.assertNotIn("fused_probe_mask", result)
            self.assertNotIn("fused_gpu_oracle_cases", result)
            for pipeline in result["compiled"]:
                self.assertEqual(pipeline["generic_threadgroup_bytes"], 0)
                self.assertEqual(pipeline["specialized_threadgroup_bytes"], 0)

    @unittest.skipUnless(os.environ.get("FSR412_MPP_GPU_FUSION_CORRECTNESS") == "1", "Fusion GPU correctness requires an approved window")
    def test_opt_in_fused_gpu_integer_oracle(self):
        models = list(actual_models())
        if not models:
            self.skipTest("verified external Lite initializers unavailable")
        for tier, preset, profile, path, data in models:
            with self.subTest(tier=tier, preset=preset, profile=profile), tempfile.TemporaryDirectory(dir=self.directory) as tmp:
                native.pack(tier, preset, data, Path(tmp), profile)
                run = subprocess.run([str(self.bench), str(ROOT), tmp, str(path), "--correctness", "--scope", "cnb-ds"],
                                     check=True, capture_output=True, text=True)
                result = json.loads(run.stdout)
                mask = result["fused_probe_mask"]
                self.assertEqual(mask & ~3, 0)
                self.assertEqual(result["fused_gpu_oracle_cases"], 120 * (1 + mask.bit_count()))
                self.assertEqual(result["fused_gpu_specialized_oracle_cases"], result["fused_gpu_oracle_cases"])
                self.assertNotIn("timings", result)

    def test_independent_cpp_quantizer_matches_python_extremes(self):
        code = self.directory / "quantizer.cpp"
        exe = self.directory / "quantizer"
        code.write_text('#include "tools/bench/fsr412_mpp_oracle.h"\n#include <iostream>\n'
                        'int main(){int32_t a,b,r;fsr411::Epilogue q;'
                        'while(std::cin>>a>>b>>r>>q.accumulatorShift>>q.residualShift>>q.rightShift>>q.flags)'
                        'std::cout<<int(fsr412_tuning_cpu::quantize(a,b,r,q))<<"\\n";}\n')
        subprocess.run(["xcrun", "clang++", "-std=c++17", "-O2", "-I", str(ROOT), str(code), "-o", str(exe)],
                       check=True, capture_output=True, text=True)
        rng = random.Random(412)
        cases = []
        extremes = (-2**31, -2**31+1, -65536, -32769, -128, -1, 0, 1, 127, 32768, 65535, 2**31-2, 2**31-1)
        for shift in (0, 1, 6, 7, 8, 9, 16, 30):
            for flags in range(8):
                for a in extremes:
                    cases.append((a, 0, 0, (0, 0, shift, flags)))
        for _ in range(4000):
            cases.append((rng.randrange(-2**31, 2**31), rng.randrange(-2**31, 2**31), rng.randrange(-128, 128),
                          (rng.randrange(31), rng.randrange(31), rng.randrange(31), rng.randrange(8))))
        lines = "".join(" ".join(map(str, (a, b, r, *q))) + "\n" for a, b, r, q in cases)
        run = subprocess.run([str(exe)], input=lines, check=True, capture_output=True, text=True)
        self.assertEqual(list(map(int, run.stdout.split())), [quantize(a, b, r, q) for a, b, r, q in cases])

    @unittest.skipUnless(os.environ.get("FSR412_MPP_GPU_CORRECTNESS") == "1", "GPU correctness requires an approved window")
    def test_opt_in_gpu_integer_oracle(self):
        models = list(actual_models())
        if not models:
            self.skipTest("verified external Lite initializers unavailable")
        for tier, preset, profile, path, data in models:
            with self.subTest(tier=tier, preset=preset, profile=profile), tempfile.TemporaryDirectory(dir=self.directory) as tmp:
                native.pack(tier, preset, data, Path(tmp), profile)
                run = subprocess.run([str(self.bench), str(ROOT), tmp, str(path), "--correctness"],
                                     check=True, capture_output=True, text=True)
                result = json.loads(run.stdout)
                mask = result["probe_mask"]
                self.assertEqual(mask & ~127, 0)
                self.assertTrue(mask & (1 << 1), "M5 must support the 16x1 DS16 layout")
                self.assertEqual(result["gpu_oracle_cases"], 50 * (1 + mask.bit_count()))
                self.assertNotIn("timings", result)
                for pipeline in result["compiled"]:
                    self.assertEqual(pipeline["generic_threadgroup_bytes"], 0)
                    self.assertEqual(pipeline["specialized_threadgroup_bytes"], 0)


if __name__ == "__main__":
    unittest.main()
