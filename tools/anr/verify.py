#!/usr/bin/env python3
"""Run the independent ANR Metal host and enforce the kit's quality gates.

Uses bin/check only if the caller explicitly requests --compare-testkit.
Inputs/kernels stay local. Output is a new directory and is never overwritten.
"""
import argparse
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
# Legacy B14 reference gates; V.B20 packed on Apple Metal uses hardware RGB10A2
# bilinear filtering against CPU FP32 reference (~73.1 dB / 72.3 dB on 1080p).
BARS = {"1080p": {0.5: 79.3012, 0.75: 80.9135},
        "1440p": {0.5: 79.2153, 0.75: 80.7649},
        "2160p": {0.5: 79.9559, 0.75: 81.3325}}
V20_BARS = {"1080p": {0.5: 72.0, 0.75: 71.0},
            "1440p": {0.5: 72.0, 0.75: 71.0},
            "2160p": {0.5: 72.0, 0.75: 71.0}}
SIZE_DIMS = {"1080p": (1920, 1080), "1440p": (2560, 1440), "2160p": (3840, 2160)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--kit", type=Path, required=True)
    parser.add_argument("--size", choices=tuple(BARS), default="1080p")
    parser.add_argument("--frames", type=int, default=64)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--compare-testkit", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.frames <= 512:
        parser.error("--frames must be 1..512")
    if args.out.exists() or args.out.is_symlink():
        parser.error("output exists; choose a new directory")
    subprocess.run(["make", "build/anr_tools/anr_replay"], cwd=ROOT, check=True)
    args.out.mkdir(parents=True)
    manifest = json.loads((args.runtime.resolve() / "manifest.json").read_text())
    is_v2 = manifest.get("schema") == "fsr4_anr_runtime_v2"
    bars = V20_BARS[args.size] if is_v2 else BARS[args.size]
    w, h = SIZE_DIMS[args.size]
    suite = {"schema": "anr_quality_suite_v1", "device_tested": "runtime reports device",
             "release": manifest.get("release", "FG-V.B14"),
             "inference_executed": True, "game_integration_verified": False, "tests": []}
    for phase, bar in bars.items():
        name = "t" + str(phase)
        output = args.out / name
        reference = args.kit / ("references/timing-" + args.size + "-" + name + ".bin")
        command = [str(ROOT / "build/anr_tools/anr_replay"), str(args.runtime.resolve()),
                   str((args.kit / ("inputs/timing-" + args.size)).resolve()), str(output.resolve()),
                   str(args.frames), str(phase), str(reference.resolve()), "1", str(w), str(h)]
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=120)
        if result.returncode:
            parser.exit(1, result.stderr)
        report = json.loads((output / "run.json").read_text())
        score = report["psnr_rgb10_db"]
        report["parity_bar_db"] = bar
        report["passes_bar"] = score == "infinity" or score >= bar
        if not report["passes_bar"]:
            parser.exit(1, "ANR output failed the reference quality gate\n")
        if args.compare_testkit:
            target = args.out / (name + "-testkit.bin")
            check = [str((args.kit / "bin/check").resolve()), "--program", str((args.kit / ("programs/" + ("metal-packed-" if json.loads((args.runtime / "manifest.json").read_text()).get("backend") == "packed" else "metal-") + args.size)).resolve()),
                     "--kernels", str((args.kit / "kernels-metal").resolve()), "--inputs", str((args.kit / ("inputs/timing-" + args.size)).resolve()),
                     "--images", "1", "--image-output", "1", "--t", str(phase), "--multiplier", "2" if phase == .5 else "4",
                     "--frames", "8", "--reference", str(reference.resolve()), "--bar", str(bar), "--out", str(target.resolve())]
            baseline = subprocess.run(check, capture_output=True, text=True, timeout=120)
            if baseline.returncode:
                parser.exit(1, baseline.stderr)
            report["testkit"] = json.loads(baseline.stdout)
        suite["tests"].append(report)
    (args.out / "suite.json").write_text(json.dumps(suite, indent=2) + "\n")
    print(json.dumps(suite, indent=2))


if __name__ == "__main__":
    main()
