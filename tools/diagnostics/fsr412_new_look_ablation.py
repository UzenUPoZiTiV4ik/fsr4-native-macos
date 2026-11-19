#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Measure the new-look floor and ceiling on the analytic moving-object scene."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile

from fsr411_quality_scenes import FRAMES, compare_runs, generate

ROOT = Path(__file__).resolve().parents[2]
SCENE = "moving_object"
GATE = "if (!atRest && depthEdge) {"


def without_depth_clamps(source):
    if source.count(GATE) != 1:
        raise ValueError("expected one new-look floor/ceiling gate in generated POST")
    return source.replace(GATE, "if (false) {")


def ablate(output, depth):
    case = output / "input"
    generate(case, (SCENE,), "linear", moving_object_depth=depth)
    runs = {}
    with tempfile.TemporaryDirectory(prefix="shader-cache-", dir=output) as tmp:
        cache = Path(tmp)
        cmd = [ROOT / "tools/bin/fsr4mac", "--fsr-version", "4.1.2", case / SCENE,
               "--profile", "int8-lite-lossy", "--simd", "--optimize", "1", "--tier", "1080",
               "--neural-tile-size", "0", "--root", ROOT, "--builddir", cache]
        for name, flags in (("lite_lossy", []), ("new_look", ["--new-look"]),
                            ("no_floor_ceiling", ["--new-look", "--no-compile"])):
            if name == "no_floor_ceiling":
                # Only replace the POST library in this disposable cache. Use
                # --no-compile to replay it with exactly the same other shaders.
                (post,) = cache.glob("fsr412_*_nl1*/post.metal")
                patched = post.with_name("ablation.metal")
                patched.write_text(without_depth_clamps(post.read_text()))
                air = patched.with_suffix(".air")
                subprocess.run(["xcrun", "metal", "-std=macos-metal2.3", "-O3", "-c",
                                patched, "-o", air], check=True)
                subprocess.run(["xcrun", "metallib", air, "-o", post.with_suffix(".metallib")],
                               check=True)
            runs[name] = output / name
            subprocess.run([*cmd, runs[name] / SCENE, *flags], cwd=ROOT, check=True)

    reports = [json.loads((base / SCENE / "run.json").read_text()) for base in runs.values()]
    cadences = [[m["skipped_inference"] for m in report["measurements"]] for report in reports]
    if (len(cadences[0]) != FRAMES or not any(cadences[0][8:])
            or cadences[0] != cadences[1] or cadences[1] != cadences[2]):
        raise ValueError("expected identical frame-skip cadence with measured skipped frames")
    if len({report["backend"] for report in reports}) != 1:
        raise ValueError("backend changed between runs")
    result = compare_runs(case, runs, (SCENE,))
    result.update({"moving_object_depth": depth, "device": reports[0]["device"],
                   "backend": reports[0]["backend"]})
    report = output / "comparison.json"
    report.write_text(json.dumps(result, indent=2) + "\n")
    print(f"\n{SCENE} ({depth} depth):")
    for name, metrics in result["scenes"][SCENE].items():
        print(f"{name:17s} PSNR {metrics['psnr_db']:.2f} dB, "
              f"skip PSNR {metrics['reuse_psnr_db']:.2f} dB, "
              f"temporal error {metrics['temporal_error']:.6g}")
    print(f"Report: {report}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="new or empty output directory")
    parser.add_argument("--depth", choices=("step", "flat"), default="step")
    args = parser.parse_args()
    if sys.platform != "darwin":
        parser.error("this diagnostic requires Apple Silicon macOS with the Metal toolchain")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        parser.error(f"output directory is not empty: {output}")
    ablate(output, args.depth)


if __name__ == "__main__":
    main()
