#!/usr/bin/env python3
"""Serial, heat-aware 1080p/1440p comparison of all native model profiles.

Build build/tests/fsr_modes_thermal and prime its caches first. No games or
Windows DLLs are started. Defaults: 1080p, 1440p, 1440p, 1080p, with every
model interleaved inside each process; 120 s initial and 30 s repeat warmups.
Raw timings/telemetry stay in --out. --publish-report writes portable numeric
results, without machine paths, source DLLs, weights or translated shaders.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
SIZES = {"1080p": (1280, 720, 1920, 1080), "1440p": (1706, 960, 2560, 1440)}
NAMES = {"exact-fast", "lossy2", "int8-lite", "int8-lite-pixfix", "int8-lite-lossy",
         "int8-lite-lossy-tiled", "int8-lite-lossy-original", "exact", "lossy",
         "lossy2-temporal1", "lossy2-temporal2"}


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def percentile(values, fraction):
    if not values:
        raise ValueError("empty timing group")
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or x <= 0 for x in values):
        raise ValueError("timings must be finite positive numbers")
    ordered = sorted(values)
    return ordered[max(0, min(len(ordered)-1, math.ceil(fraction*len(ordered))-1))]


def power():
    p = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True, check=True)
    source = re.search(r"drawing from '([^']+)'", p.stdout)
    charge = re.search(r"(\d+)%;", p.stdout)
    return {"source": source.group(1) if source else "unknown", "battery_percent": int(charge.group(1)) if charge else None}


def machine_info():
    p = subprocess.run(["sysctl", "-n", "hw.model", "hw.memsize"], capture_output=True, text=True, check=True)
    model, memory = p.stdout.splitlines()
    return {"identifier": model, "memory_gib": int(memory)/2**30,
            "name": "MacBook Air M5" if model == "Mac17,4" else "Apple Silicon Mac",
            "cooling": "fanless" if model == "Mac17,4" else "not recorded"}


def validate_run(run, size, rounds, frames, legacy):
    if run.get("schema") != "fsr_modes_thermal_v1" or run.get("output") != list(SIZES[size][2:]):
        raise ValueError("wrong report schema/output dimensions")
    if run.get("render") != list(SIZES[size][:2]) or run.get("rounds") != rounds or run.get("frames_per_round") != frames:
        raise ValueError("wrong render dimensions/measurement counts")
    if run.get("backend") != "native INT8 MPP" or run.get("metal") != "classic command buffers":
        raise ValueError("unexpected backend/Metal timing scope")
    if run.get("low_power_mode") or run.get("rcas") or not run.get("auto_exposure"):
        raise ValueError("unexpected power/RCAS/exposure options")
    rows = run.get("rows", [])
    expected = NAMES | ({"fsr4-legacy"} if legacy else set())
    if {row["name"] for row in rows} != expected or len(rows) != len(expected):
        raise ValueError("missing or duplicated model profiles")
    for row in rows:
        blocks = row["frames_by_round_ms"]
        if len(blocks) != rounds or any(len(block) != frames for block in blocks):
            raise ValueError(row["name"] + ": incomplete frame blocks")
        for block, reported in zip(blocks, row["round_means_ms"]):
            percentile(block, .99)
            if not math.isclose(statistics.mean(block), reported, abs_tol=1e-9):
                raise ValueError(row["name"] + ": incorrect round mean")
        if len(row["round_means_ms"]) != rounds:
            raise ValueError("incomplete round means")
        counts = [row[group+"_count"] for group in ("full", "reuse", "partial")]
        if sum(counts) != rounds*frames:
            raise ValueError("incorrect frame classification")
        for group in ("full", "reuse", "partial"):
            if row[group+"_count"] != len(row[group+"_ms"]):
                raise ValueError("incomplete classified timings")
        if row["name"] == "int8-lite-lossy-tiled":
            if counts != [0, 0, rounds*frames]:
                raise ValueError("tiled did not use partial inference on every measured frame")
        elif row["name"] in {"lossy", "lossy2", "lossy2-temporal1", "lossy2-temporal2", "int8-lite-lossy", "int8-lite-lossy-original"}:
            if counts != [rounds*frames//2, rounds*frames//2, 0]:
                raise ValueError("eligible whole-frame model did not use 50:50 full/reuse cadence")
        elif counts != [rounds*frames, 0, 0]:
            raise ValueError("full-inference model unexpectedly skipped work")


def summarize(runs):
    result = {}
    for size in SIZES:
        matching = [item for item in runs if item["size"] == size]
        if not matching:
            raise ValueError("missing size " + size)
        collected = {}
        for item in matching:
            for row in item["data"]["rows"]:
                target = collected.setdefault(row["name"], {"label": row["label"], "means": [], "all": [], "full": [], "reuse": [], "partial": [], "sessions": []})
                target["means"].extend(row["round_means_ms"])
                target["all"].extend(x for block in row["frames_by_round_ms"] for x in block)
                for group in ("full", "reuse", "partial"):
                    target[group].extend(row[group+"_ms"])
                target["sessions"].append({"pass": item["pass"], "mean_ms": row["median_round_mean_ms"],
                    "late_vs_early_percent": row["late_vs_early_percent"]})
        table = {}
        for name, row in collected.items():
            table[name] = {"label": row["label"], "gpu_ms": statistics.median(row["means"]),
                "round_means_ms": row["means"], "round_range_ms": [min(row["means"]), max(row["means"])],
                "measured_frames": len(row["all"]), "frame_p99_ms": percentile(row["all"], .99), "frame_max_ms": max(row["all"]),
                "sessions": row["sessions"]}
            for group in ("full", "reuse", "partial"):
                table[name][group+"_count"] = len(row[group])
                table[name][group+"_mean_ms"] = statistics.mean(row[group]) if row[group] else None
        result[size] = {"render": list(SIZES[size][:2]), "output": list(SIZES[size][2:]), "models": table}
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=ROOT/"build/v3-thermal-benchmark")
    p.add_argument("--rounds", type=int, default=6)
    p.add_argument("--frames-per-round", type=int, default=64)
    p.add_argument("--warmup-seconds", type=float, default=120)
    p.add_argument("--repeat-warmup-seconds", type=float, default=30)
    p.add_argument("--legacy", action="store_true", help="also measure the optional 4.0.2 model")
    p.add_argument("--publish-report", type=Path)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    if not 2 <= args.rounds <= 100 or not 8 <= args.frames_per_round <= 1024 or args.frames_per_round % 8:
        p.error("rounds must be 2..100; frames-per-round must be 8..1024 and divisible by eight")
    if any(not math.isfinite(x) or not 0 <= x <= 900 for x in (args.warmup_seconds, args.repeat_warmup_seconds)):
        p.error("warmups must be 0..900 seconds")
    binary = ROOT/"build/tests/fsr_modes_thermal"
    out = args.out.resolve()
    if not args.dry_run and out.exists() and any(out.iterdir()):
        raise ValueError("output directory must be empty: " + str(out))
    seen, runs = set(), []
    for index, size in enumerate(("1080p", "1440p", "1440p", "1080p"), 1):
        warmup = args.repeat_warmup_seconds if size in seen else args.warmup_seconds
        seen.add(size)
        target = out/f"pass{index}-{size}.json"
        command = ["caffeinate", "-i", str(binary), str(ROOT), "--size", *map(str, SIZES[size]), "--no-compile",
                   "--rounds", str(args.rounds), "--frames-per-round", str(args.frames_per_round),
                   "--warmup-seconds", str(warmup), "--report", str(target)]
        if args.legacy:
            command.append("--legacy")
        if index > 2:
            command.append("--reverse")
        if args.dry_run:
            print(json.dumps({"pass": index, "size": size, "command": command}))
            continue
        out.mkdir(parents=True, exist_ok=True)
        before = power(); started = time.time()
        log = out/f"pass{index}-{size}.log"
        print(f"pass {index}/4 {size}: {warmup:.0f} s warmup; see {log}", flush=True)
        with log.open("w") as handle:
            completed = subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
        if completed.returncode:
            raise RuntimeError(f"benchmark failed ({completed.returncode}); see {log}")
        data = json.loads(target.read_text())
        validate_run(data, size, args.rounds, args.frames_per_round, args.legacy)
        run = {"pass": index, "size": size, "power_before": before, "power_after": power(),
               "wall_seconds": time.time()-started, "data": data}
        runs.append(run)
        write_json(out/"runs.json", runs)
        print(f"pass {index}: thermal states {sorted({sample['state'] for sample in data['thermal_samples']})}", flush=True)
    if args.dry_run:
        return
    thermal = [{"pass": item["pass"], "size": item["size"], "power_before": item["power_before"], "power_after": item["power_after"],
                "warmup_actual_s": item["data"]["warmup_actual_s"], "measured_wall_s": item["data"]["measured_wall_s"],
                "samples": item["data"]["thermal_samples"], "rounds": item["data"]["round_thermal"]} for item in runs]
    states = {sample["state"] for item in thermal for sample in item["samples"]}
    report = {"schema": "fsr_modes_v3_thermal_summary_v1", "date_utc": runs[0]["data"]["date_utc"],
        "device": runs[0]["data"]["device"], "os": runs[0]["data"]["os"], "machine": machine_info(),
        "backend": "native INT8 MPP", "auto_exposure": True, "rcas": False,
        "scope": "whole production frame GPU milliseconds, not game frame time or FPS",
        "fixture": runs[0]["data"]["fixture"], "cadence": runs[0]["data"]["cadence"],
        "protocol": "1080p/1440p/1440p/1080p; all models serially interleaved, first model rotated and direction reversed per frame; no measured resets; median of all round means, never best/cold run",
        "rounds_per_resolution": 2*args.rounds, "frames_per_round_per_model": args.frames_per_round,
        "initial_warmup_s": args.warmup_seconds, "repeat_warmup_s": args.repeat_warmup_seconds,
        "thermal_note": "NSProcessInfo thermal-pressure states; no privileged temperature/frequency sensor measurements. Sustained prewarming and order reversal reduce heat/order bias, but nominal does not prove absence of clock throttling.",
        "serious_or_critical_seen": bool(states & {2, 3}), "thermal_states_seen": sorted(states), "thermal": thermal,
        "legacy_note": "Optional 4.0.2 is measured through its production live encoder, including its small built-in GPU probe; source weights are not in release payloads.",
        "binary_sha256": digest(binary), "benchmark_source_sha256": digest(ROOT/"tools/bench/fsr_modes_thermal.mm"),
        "results": summarize(runs)}
    write_json(out/"summary.json", report)
    if args.publish_report:
        write_json(args.publish_report.resolve(), report)
    for size, result in report["results"].items():
        print(size)
        for name, model in result["models"].items():
            print(f"  {name:27s} {model['gpu_ms']:.3f} ms ({model['round_range_ms'][0]:.3f}..{model['round_range_ms'][1]:.3f})")
    print(out/"summary.json")


if __name__ == "__main__":
    main()
