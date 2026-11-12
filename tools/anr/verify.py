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
BARS = {"1080p": {0.5: 79.3012, 0.75: 80.9135},
        "1440p": {0.5: 79.2153, 0.75: 80.7649},
        "2160p": {0.5: 79.9559, 0.75: 81.3325}}


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
    suite = {"schema": "anr_quality_suite_v1", "device_tested": "runtime reports device",
             "inference_executed": True, "game_integration_verified": False, "tests": []}
    for phase, bar in BARS[args.size].items():
        name = "t" + str(phase)
        output = args.out / name
        reference = args.kit / ("references/timing-" + args.size + "-" + name + ".bin")
        command = [str(ROOT / "build/anr_tools/anr_replay"), str(args.runtime.resolve()),
                   str((args.kit / ("inputs/timing-" + args.size)).resolve()), str(output.resolve()),
                   str(args.frames), str(phase), str(reference.resolve())]
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
