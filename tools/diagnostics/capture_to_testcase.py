#!/usr/bin/env python3
"""Convert a CrossOver diagnostics capture into a fsr4mac testcase.

A capture records GPU readbacks as little-endian float32 (see
tools/crossover/panel_app.mm). The offline CLI reads half-float color/motion
and float32 depth from a testcase, so this rewrites the four recorded planes
and writes the meta.json the CLI expects.

Usage:
  python3 tools/diagnostics/capture_to_testcase.py CAPTURE_DIR TESTCASE_DIR
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def write_half(path, array):
    np.ascontiguousarray(array, dtype="<f2").tofile(path)


def write_float(path, array):
    np.ascontiguousarray(array, dtype="<f4").tofile(path)


def convert(capture: Path, out: Path) -> dict:
    manifest = json.loads((capture / "capture.json").read_text())
    if manifest.get("schema_version") != 1:
        raise SystemExit("expected capture.json schema_version=1")
    if manifest.get("format") != "little-endian float32, C row-major":
        raise SystemExit(f"unsupported capture format {manifest.get('format')!r}")
    frames = manifest.get("frames") or []
    if not frames:
        raise SystemExit("capture has no frames")
    out.mkdir(parents=True, exist_ok=True)
    first = frames[0]
    meta = {
        "preset": "quality",
        "render_width": first["rw"],
        "render_height": first["rh"],
        "upscale_width": first["uw"],
        "upscale_height": first["uh"],
        "frames": len(frames),
        "depth_inverted": bool(first["depthReversed"]),
        "low_res_mv": not bool(first["outputResMV"]),
        "jittered_mv": bool(first["jitteredMV"]),
        "colorspace": "linear",
        "auto_exposure": bool(first["autoExposure"]),
        "pre_exposure": float(first.get("preExposure", 1.0)),
        "rcas": bool(first.get("rcas", False)),
        "sharpness": float(first.get("sharpness", 0.0)),
        "frame": [],
    }
    for index, frame in enumerate(frames):
        if (frame["rw"], frame["rh"], frame["uw"], frame["uh"]) != (
                first["rw"], first["rh"], first["uw"], first["uh"]):
            raise SystemExit("capture frames change size; not supported by one testcase")
        color = np.fromfile(capture / frame["files"]["color"], dtype="<f4").reshape(
            frame["rh"], frame["rw"], 4)
        depth = np.fromfile(capture / frame["files"]["depth"], dtype="<f4").reshape(
            frame["rh"], frame["rw"])
        motion = np.fromfile(capture / frame["files"]["motion"], dtype="<f4").reshape(
            frame["mvH"], frame["mvW"], 2)
        prefix = out / f"f{index:02d}_"
        write_half(f"{prefix}color.bin", color)
        write_float(f"{prefix}depth.bin", depth)
        write_half(f"{prefix}mv.bin", motion)
        meta["frame"].append({
            "jitter": [float(frame["jitter"][0]), float(frame["jitter"][1])],
            "mv_scale": [float(frame["mvScale"][0]), float(frame["mvScale"][1])],
            "reset": bool(frame["reset"]) or index == 0,
            "frame_time_ms": 16.6,
            "camera_near": 0.1,
            "camera_far": 1000.0,
            "fovy": 1.0,
        })
    (out / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    return meta


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("capture", type=Path)
    parser.add_argument("testcase", type=Path)
    args = parser.parse_args()
    meta = convert(args.capture, args.testcase)
    print(f"{meta['frames']} frames {meta['render_width']}x{meta['render_height']} -> "
          f"{meta['upscale_width']}x{meta['upscale_height']} written to {args.testcase}")


if __name__ == "__main__":
    main()
