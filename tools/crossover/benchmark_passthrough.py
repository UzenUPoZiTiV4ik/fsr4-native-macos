#!/usr/bin/env python3
"""Measure native MetalFX FG with/without an explicitly supplied project hook.

Runs original Metal3/4 APIs in fresh processes, alternating A/B order. No
CrossOver/bottle/game setting, system resolution or model asset is changed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess

ROOT = Path(__file__).resolve().parents[2]


def benchmark(hook, output, repeats=5):
    hook = Path(hook).expanduser().resolve(strict=True)
    output = Path(output).expanduser().absolute()
    if not hook.is_file() or hook.suffix != ".dylib":
        raise ValueError("supply a local compiled project dylib")
    if output.exists() or output.is_symlink():
        raise ValueError("output exists; choose a new directory")
    if not 2 <= repeats <= 20:
        raise ValueError("repeats must be 2..20")
    subprocess.run(["make", "build/anr_tools/benchmark_passthrough"], cwd=ROOT, check=True)
    output.mkdir(parents=True)
    binary = ROOT / "build/anr_tools/benchmark_passthrough"
    suite = {"schema": "fsr4_metalfx_passthrough_suite_v1", "hook_sha256": hashlib.sha256(hook.read_bytes()).hexdigest(),
             "scope": "Standalone fixed-frame FG; no game FPS, SR workload or presentation timing", "tests": {}}
    for api in ("classic", "metal4"):
        runs = {"bare": [], "hook": []}
        for repeat in range(repeats):
            order = ("bare", "hook") if repeat % 2 == 0 else ("hook", "bare")
            for kind in order:
                path = output / f"{api}-{repeat}-{kind}.json"
                command = ["arch", "-x86_64", str(binary), api, str(path)] + ([str(hook)] if kind == "hook" else [])
                completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=120)
                if completed.returncode:
                    raise RuntimeError(completed.stderr or completed.stdout)
                runs[kind].append(json.loads(path.read_text()))
        hashes = {run["output_sha256"] for group in runs.values() for run in group}
        if len(hashes) != 1:
            raise RuntimeError("native output changed across the A/B runs")
        gpu = {kind: statistics.median(run["gpu_median_ms"] for run in group) for kind, group in runs.items()}
        cpu = {kind: statistics.median(run["cpu_encode_median_us"] for run in group) for kind, group in runs.items()}
        suite["tests"][api] = {"runs": runs, "repeats": repeats, "gpu_median_ms": gpu, "cpu_encode_median_us": cpu,
            "gpu_delta_percent": 100 * (gpu["hook"] / gpu["bare"] - 1),
            "cpu_delta_us": cpu["hook"] - cpu["bare"], "output_identical": True,
            "original_fg_method_retained": all(run["original_fg_imp_retained"] for run in runs["hook"]),
            "disabled_sr_methods_retained": all(run["disabled_sr_factory_retained"] for run in runs["hook"]),
            "anr_encodes": sum(run["anr_encodes"] for run in runs["hook"])}
        print(api, {key: value for key, value in suite["tests"][api].items() if key != "runs"})
    (output / "suite.json").write_text(json.dumps(suite, indent=2) + "\n")
    return suite


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hook", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    try:
        benchmark(args.hook, args.out, args.repeats)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        parser.exit(1, f"MetalFX passthrough benchmark: {error}\n")


if __name__ == "__main__":
    main()
