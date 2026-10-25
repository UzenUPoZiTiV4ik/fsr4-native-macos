#!/usr/bin/env python3
"""Compare FSR 4.1.2 reference (direct DLL shaders) and optimized dumps.

The reference run executes AMD's Lite SPIR-V shaders translated to Metal one
to one (--reference --optimize 0). The optimized run is the shipping Native
INT8 path (--i8mpp --optimize 1). Both share the Metal frame setup and the
auto exposure/RCAS helpers, so this is a numerical shader comparison, not a
proof of equality with AMD's Windows runtime.

  tools/bin/fsr4mac --fsr-version 4.1.2 CASE REF --reference --optimize 0 --dump-state
  tools/bin/fsr4mac --fsr-version 4.1.2 CASE OPT --i8mpp --optimize 1 --dump-state
  python3 tools/diagnostics/fsr412_dll_parity.py --case name=REF:OPT ... --report out.json

Surfaces use the formats written by the runtime in fNN_state.json. Output,
history and reprojected are RGBA16Float, recurrent is RGBA8Unorm and auto
exposure is R32Float. Scratch arenas are not compared: the optimized path
aliases neural intermediates after pass3, so the raw arenas differ by design.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

# MTLPixelFormat values used by the 4.1.2 runtime dumps.
FORMATS = {115: (np.float16, 4, "RGBA16Float"), 70: (np.uint8, 4, "RGBA8Unorm"),
           55: (np.float32, 1, "R32Float")}
SURFACES = ("history", "reprojected", "recurrent", "auto_exposure")


def load(path: Path, dtype, channels: int, width: int, height: int) -> np.ndarray:
    data = np.fromfile(path, dtype=dtype)
    expected = width*height*channels
    if data.size != expected:
        raise SystemExit(f"{path}: {data.size} values, expected {expected}")
    return data.reshape(height, width, channels)


def metrics(ref: np.ndarray, opt: np.ndarray, peak: float, label: str) -> dict:
    a, b = ref.astype(np.float64), opt.astype(np.float64)
    error = np.abs(a-b)
    mse = float(np.mean(error**2))
    exact = ref.view(np.uint8).reshape(*ref.shape[:2], -1) == opt.view(np.uint8).reshape(*opt.shape[:2], -1)
    return {
        "changed_components": float(np.mean(ref != opt)),
        "changed_pixels": float(np.mean(np.any(ref != opt, axis=-1))),
        "bitwise_equal": bool(exact.all()),
        "max_abs": float(error.max()),
        "mean_abs": float(error.mean()),
        "rmse": math.sqrt(mse),
        "psnr_db": None if mse == 0 else float(10*math.log10(peak*peak/mse)),
        "psnr_range": label,
        "nonfinite_reference": int(np.count_nonzero(~np.isfinite(a))),
        "nonfinite_optimized": int(np.count_nonzero(~np.isfinite(b))),
    }


def compare_frame(ref_dir: Path, opt_dir: Path, frame: int, out_size: tuple[int, int]) -> dict:
    tag = f"f{frame:02d}"
    ref_state = json.loads((ref_dir/f"{tag}_state.json").read_text())
    opt_state = json.loads((opt_dir/f"{tag}_state.json").read_text())
    row: dict = {"frame": frame}
    width, height = out_size
    ref_out = load(ref_dir/f"{tag}_out.bin", np.float16, 4, width, height)
    opt_out = load(opt_dir/f"{tag}_out.bin", np.float16, 4, width, height)
    rgb_ref, rgb_opt = ref_out[..., :3], opt_out[..., :3]
    peak = float(np.max(rgb_ref.astype(np.float64)))
    row["output_rgb"] = metrics(rgb_ref, rgb_opt, peak, f"reference RGB peak {peak:.6g}")
    row["output_rgb_unit"] = metrics(np.clip(rgb_ref, 0, 1), np.clip(rgb_opt, 0, 1), 1.0,
                                     "RGB clipped to [0, 1], peak 1")
    row["output_alpha"] = metrics(ref_out[..., 3:], opt_out[..., 3:], 1.0, "alpha, peak 1")
    eight_ref = np.clip(np.rint(np.clip(rgb_ref.astype(np.float32), 0, 1)*255), 0, 255).astype(np.uint8)
    eight_opt = np.clip(np.rint(np.clip(rgb_opt.astype(np.float32), 0, 1)*255), 0, 255).astype(np.uint8)
    row["output_rgb_8bit"] = metrics(eight_ref, eight_opt, 255.0, "8-bit quantized [0, 1], peak 255")
    for name in SURFACES:
        info, other = ref_state[name], opt_state[name]
        if (info["metal_pixel_format"], info["width"], info["height"]) != \
                (other["metal_pixel_format"], other["width"], other["height"]):
            raise SystemExit(f"{tag} {name}: layout differs between runs")
        dtype, channels, label = FORMATS[info["metal_pixel_format"]]
        a = load(ref_dir/f"{tag}_{name}.bin", dtype, channels, info["width"], info["height"])
        b = load(opt_dir/f"{tag}_{name}.bin", dtype, channels, info["width"], info["height"])
        peak = 255.0 if dtype == np.uint8 else max(float(np.max(np.abs(a.astype(np.float64)))), 1e-30)
        result = metrics(a, b, peak, f"{label}, peak {peak:.6g}")
        result["format"] = label
        row[name] = result
    return row


def output_size(run: dict, ref_dir: Path) -> tuple[int, int]:
    first = run["measurements"][0]
    width, height = first["upscale_width"], first["upscale_height"]
    if (ref_dir/"f00_out.bin").stat().st_size != width*height*8:
        raise SystemExit(f"{ref_dir}: output is not {width}x{height} RGBA16Float")
    return width, height


def summarize(frames: list[dict]) -> dict:
    keys = ("output_rgb", "output_rgb_unit", "output_rgb_8bit", "output_alpha") + SURFACES
    summary = {}
    for key in keys:
        values = [frame[key] for frame in frames]
        psnrs = [v["psnr_db"] for v in values if v["psnr_db"] is not None]
        summary[key] = {
            "bitwise_equal_frames": sum(v["bitwise_equal"] for v in values),
            "changed_components_max": max(v["changed_components"] for v in values),
            "changed_pixels_max": max(v["changed_pixels"] for v in values),
            "max_abs": max(v["max_abs"] for v in values),
            "mean_abs_mean": float(np.mean([v["mean_abs"] for v in values])),
            "rmse_max": max(v["rmse"] for v in values),
            "psnr_db_min": min(psnrs) if psnrs else None,
            "nonfinite": sum(v["nonfinite_reference"]+v["nonfinite_optimized"] for v in values),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case", action="append", required=True, help="name=REFERENCE_DIR:OPTIMIZED_DIR")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = {"schema_version": 1, "cases": {}}
    for spec in args.case:
        name, dirs = spec.split("=", 1)
        ref_dir, opt_dir = (Path(p) for p in dirs.split(":", 1))
        ref_run = json.loads((ref_dir/"run.json").read_text())
        opt_run = json.loads((opt_dir/"run.json").read_text())
        if ref_run["measured_frames"] != opt_run["measured_frames"]:
            raise SystemExit(f"{name}: frame counts differ")
        size = output_size(ref_run, ref_dir)
        frames = [compare_frame(ref_dir, opt_dir, i, size) for i in range(ref_run["measured_frames"])]
        report["cases"][name] = {
            "testcase": ref_run["testcase"], "preset": ref_run["preset"],
            "reference": {"backend": ref_run["backend"], "optimize": ref_run["optimize"],
                          "variant": ref_run["variant"], "gpu_mean_ms": ref_run["gpu_mean_ms"]},
            "optimized": {"backend": opt_run["backend"], "optimize": opt_run["optimize"],
                          "variant": opt_run["variant"], "gpu_mean_ms": opt_run["gpu_mean_ms"]},
            "output_size": list(size), "frames": len(frames),
            "summary": summarize(frames), "per_frame": frames,
        }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2)+"\n")
    for name, case in report["cases"].items():
        s = case["summary"]
        print(f"{name}: {case['frames']} frames {case['output_size'][0]}x{case['output_size'][1]}")
        for key, value in s.items():
            psnr = "inf" if value["psnr_db_min"] is None else f"{value['psnr_db_min']:.2f}"
            print(f"  {key:16s} equal {value['bitwise_equal_frames']}/{case['frames']} "
                  f"changed px<={value['changed_pixels_max']:.4%} comp<={value['changed_components_max']:.4%} "
                  f"max {value['max_abs']:.6g} mean {value['mean_abs_mean']:.3g} "
                  f"rmse<={value['rmse_max']:.3g} psnr>={psnr} nonfinite {value['nonfinite']}")


if __name__ == "__main__":
    main()
