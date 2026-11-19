#!/usr/bin/env python3
"""Generate analytic FSR 4.1 fixtures and compare recorded CLI outputs.

Requires NumPy. The point-sampled 192x108 input has the CLI's jitter sign:
world = (render_pixel + .5 - jitter) * 2. Motion points to the previous
frame in render pixels. References integrate 8x8 samples per output pixel.
No network output is used to construct the reference. This small SDR suite
is a regression diagnostic, not a substitute for game/HDR quality testing.

  python3 tools/diagnostics/fsr411_quality_scenes.py generate build/quality-scenes
  tools/bin/fsr4mac build/quality-scenes/particles build/quality-after/particles --profile lossy
  python3 tools/diagnostics/fsr411_quality_scenes.py compare build/quality-scenes build/quality-before build/quality-after --report build/quality.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

SCENES = ("wire_static", "wire_pan", "particles", "dark_particles", "static",
          "grain6", "grain15", "screen_bar", "moving_object")
JITTER = ((.25, -1/6), (-.25, 1/6), (.125, -7/18), (-.375, -1/18),
          (.375, 5/18), (-.125, -5/18), (.0625, 1/18), (-.4375, 7/18))
WIDTH, HEIGHT, FRAMES = 384, 216, 32


def scene(name, x, y, frame):
    world_x = x + (2*frame if name == "wire_pan" else 8*frame if name == "screen_bar" else 0)
    background = .18 + .035*np.sin(world_x*.035)*np.cos(y*.041)
    if name.startswith("wire"):
        line = np.minimum(np.abs((world_x+.18) % 29-14.5), np.abs((y-.37) % 37-18.5)) < .43
        return np.where(line, .64, background)
    if name in ("particles", "dark_particles"):
        speed = 5 if name == "particles" else 3
        cx, cy = 45+(frame*speed) % 285, 108+16*np.sin(frame*.13)
        strength = np.exp(-((x-cx)**2+(y-cy)**2)/(2*3.4**2))
        return background+.4*strength if name == "particles" else background*(1-.9*strength)
    if name == "screen_bar":
        background += .12*np.sin(world_x*.2)*np.cos(y*.09)
        return np.where(np.abs(x-172.35) < .75, .7, background)
    if name == "moving_object":
        return np.where((x-(60+4*frame))**2+(y-108)**2 < 18**2, .65, background)
    if name in ("static", "grain6", "grain15"):
        return np.clip(background+.13*np.sin(world_x*.21)*np.cos(y*.13), .01, .9)
    raise ValueError("unknown scene " + name)


def generate(output, names, colorspace="nonlinear", *, moving_object_depth="step"):
    rw, rh = WIDTH//2, HEIGHT//2
    y, x = np.mgrid[:rh, :rw].astype(np.float32)
    gy, gx = np.mgrid[:HEIGHT, :WIDTH].astype(np.float32)
    for name in names:
        folder = output/name
        folder.mkdir(parents=True, exist_ok=True)
        meta = {"preset": "performance", "render_width": rw, "render_height": rh,
                "upscale_width": WIDTH, "upscale_height": HEIGHT, "frames": FRAMES,
                "colorspace": colorspace, "auto_exposure": False, "low_res_mv": True,
                "depth_inverted": False, "rcas": False, "frame": []}
        for frame in range(FRAMES):
            jx, jy = JITTER[frame % len(JITTER)]
            px, py = (x+.5-jx)*2, (y+.5-jy)*2
            value = scene(name, px, py, frame)
            if name.startswith("grain"):
                rng = np.random.default_rng(411000+frame)
                value = np.clip(value+rng.normal(0, int(name[5:])/255, value.shape), 0, 1)
            color = np.ones((rh, rw, 4), dtype="<f2")
            color[:, :, :3] = value[:, :, None]
            motion = np.zeros((rh, rw, 2), dtype="<f2")
            depth = np.full((rh, rw), .7, dtype="<f4")
            if name == "wire_pan":
                motion[:, :, 0] = 1
            if name == "screen_bar":
                foreground = np.abs(px-172.35) < .75
                motion[:, :, 0] = np.where(foreground, 0, 4)
                depth[foreground] = .2
            if name == "moving_object":
                foreground = (px-(60+4*frame))**2+(py-108)**2 < 18**2
                motion[:, :, 0] = np.where(foreground, -2, 0)
                if moving_object_depth == "step":
                    depth[foreground] = .2
            prefix = folder/f"f{frame:02d}"
            color.tofile(str(prefix)+"_color.bin")
            motion.tofile(str(prefix)+"_mv.bin")
            depth.tofile(str(prefix)+"_depth.bin")
            truth = np.zeros((HEIGHT, WIDTH), dtype="<f4")
            for sy in range(8):
                for sx in range(8):
                    truth += scene(name, gx+(sx+.5)/8, gy+(sy+.5)/8, frame)/64
            truth.tofile(str(prefix)+"_truth.bin")
            meta["frame"].append({"jitter": [jx, jy], "mv_scale": [1, 1],
                                  "pre_exposure": 1, "reset": frame == 0})
        (folder/"meta.json").write_text(json.dumps(meta, indent=2)+"\n")
        print("Generated", name, flush=True)


def read(path, shape, dtype):
    values = np.fromfile(path, dtype=dtype)
    if values.size != int(np.prod(shape)) or not np.isfinite(values).all():
        raise ValueError(f"invalid image: {path}")
    return values.reshape(shape).astype(np.float32)


def compare(inputs, before, after, names):
    return compare_runs(inputs, {"before": before, "after": after}, names)


def compare_runs(inputs, runs, names):
    results = {}
    # Ignore the eight-frame warmup and twelve output pixels at each border.
    for name in names:
        meta = json.loads((inputs/name/"meta.json").read_text())
        w, h, count = meta["upscale_width"], meta["upscale_height"], meta["frames"]
        truth = np.stack([read(inputs/name/f"f{i:02d}_truth.bin", (h, w), "<f4")[12:-12, 12:-12]
                          for i in range(8, count)])[:, :, :, None]
        row = {}
        for label, base in runs.items():
            run = json.loads((base/name/"run.json").read_text())
            if run["diagnostic_gpu_copies"] or run["benchmark_iterations"]:
                raise ValueError("compare needs ordinary CLI image outputs")
            values = np.stack([read(base/name/f"f{i:02d}_out.bin", (h, w, 4), "<f2")[12:-12, 12:-12, :3]
                               for i in range(8, count)])
            error = values-truth
            skipped = [m["skipped_inference"] for m in run["measurements"][8:]]
            mse = np.mean(error**2, dtype=np.float64)
            row[label] = {"psnr_db": float(-10*np.log10(mse)),
                          "mae": float(np.abs(error).mean()),
                          "reuse_psnr_db": float(-10*np.log10(np.mean(error[skipped]**2, dtype=np.float64))),
                          "temporal_error": float(np.abs(np.diff(values, axis=0)-np.diff(truth, axis=0)).mean()),
                          "neural_runs": run["neural_runs"], "reused_frames": run["reused_frames"]}
        results[name] = row
    return {"reference": "8x8 analytic integration; unit SDR peak; all RGB channels",
            "measured_frames_per_scene": FRAMES-8, "border_excluded": 12, "scenes": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    generate_parser = commands.add_parser("generate")
    generate_parser.add_argument("output", type=Path)
    generate_parser.add_argument("--colorspace", choices=("linear", "nonlinear"), default="nonlinear")
    generate_parser.add_argument("--moving-object-depth", choices=("step", "flat"), default="step")
    compare_parser = commands.add_parser("compare")
    for arg in ("inputs", "before", "after"):
        compare_parser.add_argument(arg, type=Path)
    compare_parser.add_argument("--report", type=Path, required=True)
    for command in (generate_parser, compare_parser):
        command.add_argument("--scenes", choices=SCENES, nargs="+", default=SCENES)
    args = parser.parse_args()
    if args.command == "generate":
        generate(args.output, args.scenes, args.colorspace, moving_object_depth=args.moving_object_depth)
    else:
        result = compare(args.inputs, args.before, args.after, args.scenes)
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, indent=2)+"\n")
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
