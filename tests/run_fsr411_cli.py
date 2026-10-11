#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""GPU regression for CLI defaults, Lossy reuse, snapshots and benchmark accounting."""
import argparse
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("auto", "i8mpp", "scalar"), default="auto")
    parser.add_argument("--binary", type=Path, default=ROOT / "tools/bin/fsr4mac")
    parser.add_argument("--stage-profiling", action="store_true",
                        help="also validate optional GPU timestamp sampling and output equivalence")
    args = parser.parse_args()
    env = os.environ.copy()
    for name in ("FSR4_VERSION", "FSR4_PROFILE", "FSR4_MODEL", "FSR4_OPTIMIZE"):
        env.pop(name, None)
    env["FSR4_ROOT"] = str(ROOT)

    def cli(*flags, success=True):
        result = subprocess.run([str(args.binary.resolve()), *map(str, flags)],
                                cwd=ROOT, env=env, capture_output=True, text=True)
        if success and result.returncode:
            raise RuntimeError(result.stdout + result.stderr)
        if not success:
            assert result.returncode, "invalid CLI request succeeded"
        return result

    info = json.loads(cli("--backend-info").stdout)
    assert info["fsr_version"] == "4.1.1" and info["profile"] == "lossy"
    assert info["selection_only"] is True
    legacy = json.loads(cli("--fsr-version", "4.0.2", "--backend-info").stdout)
    assert legacy["backend"] in ("f16", "i8mpp")
    # --fp16 now selects the Apple7+ simdgroup backend of the same INT8 model.
    simd = json.loads(cli("--backend-info", "fp16").stdout)
    assert simd["backend"] == "int8-simd", simd
    assert "backend must be" in cli("--backend-info", "half", success=False).stderr

    with tempfile.TemporaryDirectory(prefix="fsr411-cli-") as folder:
        base = Path(folder)
        testcase, output, benchmark = (base / name for name in ("input", "output", "benchmark"))
        testcase.mkdir()
        rw, rh, uw, uh = 43, 25, 65, 41
        frames = [{"jitter": [value, .125], "reset": index == 0}
                  for index, value in enumerate((0, -.25, -.375))]
        meta = {"preset": "quality", "render_width": rw, "render_height": rh,
                "upscale_width": uw, "upscale_height": uh, "frames": len(frames),
                "low_res_mv": True, "depth_inverted": False, "auto_exposure": False,
                "pre_exposure": 1., "colorspace": "linear", "frame": frames}
        (testcase / "meta.json").write_text(json.dumps(meta))
        for index in range(len(frames)):
            color = bytearray()
            depth = bytearray()
            for y in range(rh):
                for x in range(rw):
                    color.extend(struct.pack("<4e", (16 + (x + index) % 29) / 64,
                                             (24 + y % 19) / 64, (32 + (x + y) % 23) / 64, 1.))
                    depth.extend(struct.pack("<f", .2 + ((x + y) % 13) / 24))
            prefix = testcase / f"f{index:02d}"
            Path(str(prefix) + "_color.bin").write_bytes(color)
            Path(str(prefix) + "_depth.bin").write_bytes(depth)
            Path(str(prefix) + "_mv.bin").write_bytes(struct.pack("<2e", 0., 0.) * rw * rh)

        def report(path):
            data = json.loads((path / "run.json").read_text())
            assert data["fsr_version"] == "4.1.1"
            assert data["measured_frames"] == len(data["measurements"])
            assert data["neural_runs"] + data["reused_frames"] == data["measured_frames"]
            assert data["reused_frames"] == sum(frame["skipped_inference"] for frame in data["measurements"])
            for frame in data["measurements"]:
                assert frame["sequence_neural_runs"] + frame["sequence_reused_frames"] == frame["frame"] + 1
                assert math.isfinite(frame["gpu_ms"]) and frame["gpu_ms"] > 0
            return data

        cli(testcase, output, "--" + args.backend, "--profile", "exact", "--capture-stages", "pass4")
        exact = report(output)
        assert exact["profile"] == "exact" and (exact["neural_runs"], exact["reused_frames"]) == (3, 0)
        assert (output / "f01_pass04_scratch.bin").is_file()

        # Reuse the folder to catch stale neural snapshots on a skipped frame.
        cli(testcase, output, "--" + args.backend, "--capture-stages", "pass4")
        lossy = report(output)
        assert lossy["profile"] == "lossy" and (lossy["neural_runs"], lossy["reused_frames"]) == (2, 1)
        assert [frame["skipped_inference"] for frame in lossy["measurements"]] == [False, True, False]
        assert not (output / "f01_pass04_scratch.bin").exists()
        for index, frame in enumerate(lossy["measurements"]):
            assert frame["captured_stages"] == ([] if index == 1 else ["pass4"])
            pixels = (output / f"f{index:02d}_out.bin").read_bytes()
            assert len(pixels) == uw * uh * 8
            assert all(math.isfinite(value[0]) for i, value in enumerate(struct.iter_unpack("<e", pixels)) if i % 4 != 3)
            png = (output / f"f{index:02d}_out.png").read_bytes()
            assert png[:8] == b"\x89PNG\r\n\x1a\n" and struct.unpack(">II", png[16:24]) == (uw, uh)

        cli(testcase, benchmark, "--" + args.backend, "--bench", 2)
        measured = report(benchmark)
        assert measured["warmup_sequences"] == 1 and measured["benchmark_iterations"] == 2
        assert (measured["measured_frames"], measured["neural_runs"], measured["reused_frames"]) == (6, 4, 2)
        assert [frame["iteration"] for frame in measured["measurements"]] == [0, 0, 0, 1, 1, 1]
        assert not measured["diagnostic_gpu_copies"]
        assert not measured["diagnostic_gpu_counters"]
        assert "stage_profiling" not in measured
        assert all("stage_profile" not in frame for frame in measured["measurements"])
        assert [path.name for path in benchmark.iterdir()] == ["run.json"]
        if args.stage_profiling:
            profiled = base / "profiled"
            cli(testcase, profiled, "--" + args.backend, "--profile-stages", "--no-compile")
            timing = report(profiled)
            assert timing["diagnostic_gpu_counters"]
            assert timing["stage_profiling"]["production_timing"] is False
            assert timing["stage_profiling"]["method"] in (
                "dispatch_boundary", "separate_encoder_boundaries")
            assert [frame["skipped_inference"] for frame in timing["measurements"]] == [False, True, False]
            for index, frame in enumerate(timing["measurements"]):
                assert (profiled / f"f{index:02d}_out.bin").read_bytes() == (output / f"f{index:02d}_out.bin").read_bytes()
                stages = frame["stage_profile"]["stages"]
                names = [stage["stage"] for stage in stages]
                assert len(names) == len(set(names))
                assert "pre" in names and "post" in names and "retire" in names
                assert ("pass1" in names) == (index != 1)
                assert ("post_cnb" in names) == (index != 1)
                assert ("pad0" in names) == (index != 1)
                assert all(math.isfinite(stage["gpu_ms"]) and stage["gpu_ms"] >= 0 for stage in stages)
                assert math.isclose(sum(stage["gpu_ms"] for stage in stages),
                                    frame["stage_profile"]["stage_sum_ms"], rel_tol=1e-9)
                assert frame["stage_profile"]["sampled_span_ms"] <= frame["gpu_ms"] + max(.1, frame["gpu_ms"] * .1)
                assert frame["stage_profile"]["nonoverlapping_stage_ms"] <= frame["stage_profile"]["sampled_span_ms"] + 1e-9
            print("FSR 4.1.1 stage profiling: sampled timings, unchanged pixels and Lossy stage accounting passed")
    print("FSR 4.1.1 CLI: default Lossy, legacy selection, finite outputs, skipped snapshots and warmup accounting passed")


if __name__ == "__main__":
    main()
