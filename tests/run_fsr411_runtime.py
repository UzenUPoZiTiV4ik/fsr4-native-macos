#!/usr/bin/env python3
"""Run the FSR 4.1.1 full-graph GPU matrix sequentially on an Apple family 10 GPU.

Each case compares direct imported HLSL against optimized native kernels,
then checks independently offset input/output rectangles. Every RGB value
must match; the PRE cases also check each active quantized tensor byte.
No Windows 4.0 capture or performance assertion is used.
"""
import argparse
from pathlib import Path
import subprocess

# Exact Fast/Lossy ship relaxed PRE/POST math (additions may be reordered),
# so bit-exact cases build them with ordered math (--strict-math, `_sm1`).
# The *_relaxed cases check the production build against that reference
# within a mean tolerance, with exact crop parity and finite output.
STRICT = ["--strict-math"]
CASES = [
    ("lossy_pre", ["--lossy", "--capture-pre"] + STRICT),
    ("exact", ["--capture-pre"]),
    ("exact_fast_pre", ["--fast", "--capture-pre"] + STRICT),
    ("exact_fast_rcas", ["--fast", "--rcas", "--odd"] + STRICT),
    ("exact_ultra", ["--ultra"]),
    ("lossy_ultra", ["--lossy", "--ultra"] + STRICT),
    ("scalar_lossy", ["--lossy", "--scalar"] + STRICT),
    ("exact_odd", ["--odd", "--high-mv", "--jittered", "--inverted"]),
    ("lossy_odd", ["--lossy", "--odd", "--high-mv", "--jittered", "--inverted"] + STRICT),
    ("lossy_2160", ["--tier", "2160", "--lossy", "--odd"] + STRICT),
    ("exact_4320", ["--tier", "4320", "--odd"]),
    ("lossy_4320", ["--tier", "4320", "--lossy", "--odd"] + STRICT),
    ("lossy_cs1", ["--lossy", "--colorspace", "1", "--odd"] + STRICT),
    ("lossy_cs2", ["--lossy", "--colorspace", "2", "--odd"] + STRICT),
    ("lossy_cs3", ["--lossy", "--colorspace", "3", "--odd"] + STRICT),
    ("exact_rcas", ["--rcas", "--odd"]),
    ("ae_mip6", ["--lossy", "--auto-exposure"] + STRICT),
    ("ae_odd", ["--auto-exposure", "--size", "131", "75", "264", "152"]),
    ("ae_mip5", ["--lossy", "--auto-exposure", "--odd"] + STRICT),
    ("exact_fast_relaxed", ["--fast", "--rcas", "--odd", "--relaxed-tolerance"]),
    ("lossy_relaxed", ["--lossy", "--odd", "--high-mv", "--relaxed-tolerance"]),
    ("lossy_relaxed_ae", ["--lossy", "--auto-exposure", "--relaxed-tolerance"]),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-compile", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    logs = root / "build/validation411"
    logs.mkdir(parents=True, exist_ok=True)
    for name, flags in CASES:
        command = [str(root / "build/tests/fsr411_runtime_test"), str(root), "--optimize"]
        if not args.no_compile:
            command.append("--compile")
        with (logs / (name + ".log")).open("w") as log:
            result = subprocess.run(command + flags, cwd=root, stdout=log, stderr=subprocess.STDOUT)
        print(f"{name}: {'PASS' if result.returncode == 0 else 'FAIL'}", flush=True)
        if result.returncode:
            print((logs / (name + ".log")).read_text()[-8000:])
            return result.returncode
    print(f"FSR 4.1.1 full graph: {len(CASES)} cases, {len(CASES) * 3} frame comparisons passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
