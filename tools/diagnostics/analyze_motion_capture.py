#!/usr/bin/env python3
"""Diagnose motion conventions from a real FSR 4.1.1 input capture (CPU only).

Usage:
  python3 tools/diagnostics/analyze_motion_capture.py CAPTURE_DIR --json report.json
  python3 tools/diagnostics/analyze_motion_capture.py --self-test

Reads capture.json with schema_version=1 and frames in temporal order. Each
frame has index, rw/rh/uw/uh, mvW/mvH, jitter[2], mvScale[2], jitteredMV,
outputResMV, preExposure, reset, owner, ownerFrame, and files containing color,
depth, motion and output paths relative to capture.json. Binary files are
little-endian float32 in C row order, with 4/1/2/4 channels respectively.
Output is validated for completeness but is not used to judge input motion.

The declared mapping uses raw * signed mvScale exactly once. For a current
color pixel p, previous color is sampled at
  ((p + .5 - jitterCurrent) / renderCurrent + motionUV) * renderPrevious
      - .5 + jitterPrevious,
where motionUV = (raw * signedScale - jitterCancellationPixels) / mvExtent.
CancellationPixels is jitterPrevious - jitterCurrent only for jittered MV.
The hypotheses vary raw-motion sign, magnitude, and normalization extent.

This is a photometric diagnostic, not a game-parameter correction. It samples
textured surface interiors, excludes depth boundaries, and evaluates all
hypotheses on the same valid points. HDR is compared after removing supplied
preExposure and applying log1p with a per-pair luminance scale. The largest
20% of residuals are trimmed to reduce disocclusion/reactive-material bias.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys
import tempfile

try:
    import numpy as np
except ImportError:
    raise SystemExit("NumPy is required; use a Python environment with numpy installed.")


@dataclass
class Frame:
    meta: dict
    color: np.ndarray
    depth: np.ndarray
    motion: np.ndarray

    @property
    def render(self):
        return np.array([self.meta["rw"], self.meta["rh"]], dtype=np.float64)

    @property
    def output(self):
        return np.array([self.meta["uw"], self.meta["uh"]], dtype=np.float64)

    @property
    def jitter(self):
        return np.asarray(self.meta["jitter"], dtype=np.float64)

    @property
    def scale(self):
        return np.asarray(self.meta["mvScale"], dtype=np.float64)


@dataclass(frozen=True)
class Hypothesis:
    normalization: str = "declared"
    sign: int = 1
    multiplier: float = 1.0

    @property
    def label(self):
        if self.normalization == "zero":
            return "no-motion"
        return f"{self.normalization}/sign{self.sign:+d}/x{self.multiplier:g}"

    def affine(self, previous, current):
        """Return raw-MV coefficient and offset, in previous input pixels."""
        if self.normalization == "zero":
            return np.zeros(2), np.zeros(2)
        mode = self.normalization
        if mode == "declared":
            mode = "output" if current.meta["outputResMV"] else "input"
        extent = current.output if mode == "output" else current.render
        coefficient = current.scale * (self.sign * self.multiplier) * previous.render / extent
        cancellation = previous.jitter - current.jitter if current.meta["jitteredMV"] else np.zeros(2)
        return coefficient, -cancellation * previous.render / extent


def read_capture(path):
    path = Path(path)
    if path.is_dir():
        path /= "capture.json"
    data = json.loads(path.read_text())
    if data.get("schema_version") != 1:
        raise ValueError("Expected capture.json schema_version=1")
    frames, warnings = [], []
    if data.get("status") != "complete":
        warnings.append(f"Capture status is {data.get('status', 'unspecified')!r}; only recorded frames are analyzed.")
    for meta in data.get("frames", []):
        index = meta.get("index", "?")
        for name in ("rw", "rh", "uw", "uh", "mvW", "mvH"):
            if not isinstance(meta.get(name), int) or meta[name] <= 0:
                raise ValueError(f"Frame {index}: invalid {name}")
        for name in ("jitter", "mvScale"):
            value = np.asarray(meta.get(name), dtype=np.float64)
            if value.shape != (2,) or not np.isfinite(value).all():
                raise ValueError(f"Frame {index}: {name} must have two finite values")
        for name in ("jitteredMV", "outputResMV", "reset"):
            if meta.get(name) not in (0, 1, False, True):
                raise ValueError(f"Frame {index}: invalid {name}")
        exposure = meta.get("preExposure")
        if not isinstance(exposure, (int, float)) or not math.isfinite(exposure) or exposure <= 0:
            raise ValueError(f"Frame {index}: invalid preExposure")
        expected_mv = (meta["uw"], meta["uh"]) if meta["outputResMV"] else (meta["rw"], meta["rh"])
        if (meta["mvW"], meta["mvH"]) != expected_mv:
            warnings.append(f"Frame {index}: actual MV dimensions disagree with outputResMV; verify capture canonicalization.")
        shapes = {"color": (meta["rh"], meta["rw"], 4),
                  "depth": (meta["rh"], meta["rw"]),
                  "motion": (meta["mvH"], meta["mvW"], 2),
                  "output": (meta["uh"], meta["uw"], 4)}
        arrays = {}
        for name, shape in shapes.items():
            resource = path.parent / meta["files"][name]
            expected_bytes = math.prod(shape) * 4
            if resource.stat().st_size != expected_bytes:
                raise ValueError(f"Frame {index}: {resource.name} has {resource.stat().st_size} bytes, expected {expected_bytes}")
            if name != "output":
                arrays[name] = np.memmap(resource, dtype="<f4", mode="r", shape=shape)
        frames.append(Frame(meta, **arrays))
    if len(frames) < 2:
        raise ValueError("At least two complete consecutive real frames are required")
    return path, data, frames, warnings


def sample(array, positions, nearest=False):
    """Clamped computation plus a separate validity mask; never trust OOB data."""
    height, width = array.shape[:2]
    valid = np.isfinite(positions).all(axis=1)
    valid &= (positions[:, 0] >= 0) & (positions[:, 0] <= width - 1)
    valid &= (positions[:, 1] >= 0) & (positions[:, 1] <= height - 1)
    safe = np.where(np.isfinite(positions), positions, 0)
    safe = np.clip(safe, [0, 0], [width - 1, height - 1])
    if nearest:
        xy = np.rint(safe).astype(np.int64)
        result = np.asarray(array[xy[:, 1], xy[:, 0]], dtype=np.float64)
    else:
        xy = np.floor(safe).astype(np.int64)
        high = np.minimum(xy + 1, [width - 1, height - 1])
        fraction = safe - xy
        if array.ndim == 3:
            fraction = fraction[:, :, None]
        a = array[xy[:, 1], xy[:, 0]] * (1 - fraction[:, 0]) + array[xy[:, 1], high[:, 0]] * fraction[:, 0]
        b = array[high[:, 1], xy[:, 0]] * (1 - fraction[:, 0]) + array[high[:, 1], high[:, 0]] * fraction[:, 0]
        result = a * (1 - fraction[:, 1]) + b * fraction[:, 1]
    finite = np.isfinite(result)
    valid &= finite.all(axis=1) if result.ndim == 2 else finite
    return result, valid


def tone(rgb, exposure, luminance_scale):
    with np.errstate(invalid="ignore", over="ignore", divide="ignore"):
        return np.log1p(np.maximum(rgb[..., :3] / exposure, 0) / luminance_scale)


def stats(values, trim=0.2):
    values = np.asarray(values)
    count = max(1, int(len(values) * (1 - trim)))
    return {"trimmed_mean": float(np.partition(values, count - 1)[:count].mean()),
            "median": float(np.median(values)), "p90": float(np.percentile(values, 90))}


def valid_pairs(frames):
    pairs, skipped = [], []
    for previous, current in zip(frames, frames[1:]):
        reason = None
        if current.meta["reset"]:
            reason = "current frame resets temporal history"
        elif previous.meta.get("owner") != current.meta.get("owner"):
            reason = "temporal owner changed"
        elif "ownerFrame" in previous.meta and "ownerFrame" in current.meta:
            if current.meta["ownerFrame"] != previous.meta["ownerFrame"] + 1:
                reason = "ownerFrame sequence is not consecutive"
        if reason:
            skipped.append({"previous": previous.meta["index"], "current": current.meta["index"], "reason": reason})
        else:
            pairs.append((previous, current))
    return pairs, skipped


def group_hypotheses(pairs):
    hypotheses = [Hypothesis(mode, sign, multiplier)
                  for mode in ("declared", "input", "output")
                  for sign in (1, -1) for multiplier in (1.0, 0.5, 2.0)]
    hypotheses.append(Hypothesis("zero"))
    groups = {}
    for hypothesis in hypotheses:
        # Equal raw coefficients AND jitter offsets are truly indistinguishable.
        key = tuple(round(float(x), 12) for previous, current in pairs
                    for vector in hypothesis.affine(previous, current) for x in vector)
        groups.setdefault(key, []).append(hypothesis)
    return list(groups.values())


def analyze_pair(previous, current, groups, max_samples, min_samples, motion_sampling):
    result = {"previous": previous.meta["index"], "current": current.meta["index"]}
    stride = max(1, math.ceil(math.sqrt(np.prod(current.render) / (max_samples * 3))))
    xs, ys = np.arange(2, current.meta["rw"] - 2, stride), np.arange(2, current.meta["rh"] - 2, stride)
    positions = np.stack(np.meshgrid(xs, ys), axis=-1).reshape(-1, 2).astype(np.float64)
    result["candidate_samples"] = len(positions)
    if len(positions) < min_samples:
        return {**result, "skipped": "too few interior input pixels"}
    raw_color, valid = sample(current.color, positions)
    linear = np.maximum(raw_color[:, :3] / current.meta["preExposure"], 0)
    luminance = linear[:, 0] * 0.2126 + linear[:, 1] * 0.7152 + linear[:, 2] * 0.0722
    positive = luminance[valid & (luminance > 0)]
    scale = max(float(np.median(positive)), 1e-8) if len(positive) else 1.0
    reference = tone(raw_color, current.meta["preExposure"], scale)
    neighbors = [tone(sample(current.color, positions + shift)[0], current.meta["preExposure"], scale)
                 for shift in ((-1, 0), (1, 0), (0, -1), (0, 1))]
    gradient = np.mean(np.abs(neighbors[0] - neighbors[1]) + np.abs(neighbors[2] - neighbors[3]), axis=1) / 2
    valid &= np.isfinite(gradient)
    if not valid.any():
        return {**result, "skipped": "no finite color samples"}
    textured = gradient > max(float(np.percentile(gradient[valid], 45)), 1e-5)

    unjittered_uv = (positions + 0.5 - current.jitter) / current.render
    motion_positions = unjittered_uv * np.array([current.meta["mvW"], current.meta["mvH"]]) - 0.5
    raw_motion, motion_valid = sample(current.motion, motion_positions, nearest=motion_sampling == "nearest")
    valid &= motion_valid
    depth_position = positions - current.jitter
    depths = []
    for shift in ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)):
        depth, inside = sample(current.depth, depth_position + shift, nearest=True)
        valid &= inside
        depths.append(depth)
    depths = np.stack(depths, axis=1)
    depth_smooth = np.ptp(depths, axis=1) <= np.maximum(np.abs(depths[:, 0]) * 0.02, 1e-5)
    selected = np.flatnonzero(valid & textured & depth_smooth)
    result.update(luminance_scale=scale, texture_retained=int(np.count_nonzero(valid & textured)),
                  depth_interior_retained=int(len(selected)))
    if len(selected) > max_samples:
        selected = np.sort(np.random.default_rng(411 + int(current.meta["index"])).choice(selected, max_samples, replace=False))
    if len(selected) < min_samples:
        return {**result, "skipped": "too few textured pixels away from depth boundaries"}
    reference, raw_motion = reference[selected], raw_motion[selected]
    base = unjittered_uv[selected] * previous.render - 0.5 + previous.jitter
    residuals, common_valid = [], np.ones(len(selected), dtype=bool)
    for group in groups:
        coefficient, offset = group[0].affine(previous, current)
        previous_position = base + raw_motion * coefficient + offset
        sampled, inside = sample(previous.color, previous_position)
        previous_tone = tone(sampled, previous.meta["preExposure"], scale)
        error = np.mean(np.abs(previous_tone - reference), axis=1)
        common_valid &= inside & np.isfinite(error)
        residuals.append(error)
    result["common_valid_samples"] = int(common_valid.sum())
    if common_valid.sum() < min_samples:
        return {**result, "skipped": "too little common in-bounds support across hypotheses"}
    coefficient, offset = Hypothesis().affine(previous, current)
    displacement = raw_motion * coefficient + offset
    magnitude = np.linalg.norm(displacement, axis=1)
    moving = common_valid & (magnitude >= 0.25)
    # Static backgrounds must not drown out motion on a smaller moving object.
    mask = moving if moving.sum() >= min_samples else common_valid
    result.update(sample_count=int(mask.sum()), moving_samples=int(moving.sum()),
                  domain="moving_textured_interiors" if moving.sum() >= min_samples else "all_textured_interiors",
                  declared_motion_input_pixels={"median": float(np.median(magnitude[common_valid])),
                                                "p90": float(np.percentile(magnitude[common_valid], 90))},
                  scores=[stats(error[mask]) for error in residuals])
    return result


def analyze(frames, max_samples=16000, min_samples=128, motion_sampling="nearest"):
    pairs, skipped = valid_pairs(frames)
    if not pairs:
        raise ValueError("No consecutive pair without reset or owner change")
    groups = group_hypotheses(pairs)
    results = [analyze_pair(previous, current, groups, max_samples, min_samples, motion_sampling)
               for previous, current in pairs]
    usable = [pair for pair in results if "scores" in pair]
    if not usable:
        return {"rankings": [], "pairs": results, "skipped_transitions": skipped,
                "interpretation": "Insufficient textured common support; no motion conclusion is possible."}
    zero_index = next(i for i, group in enumerate(groups) if any(h.normalization == "zero" for h in group))
    rankings = []
    for i, group in enumerate(groups):
        relative = []
        for pair in usable:
            score = pair["scores"][i]["trimmed_mean"]
            zero = pair["scores"][zero_index]["trimmed_mean"]
            relative.append(1.0 if zero < 1e-8 and score < 1e-8 else score / max(zero, 1e-8))
        weights = [pair["sample_count"] for pair in usable]
        scores = [pair["scores"][i]["trimmed_mean"] for pair in usable]
        wins = sum(scores_i["trimmed_mean"] <= min(score["trimmed_mean"] for score in pair["scores"]) + 1e-10
                   for pair, scores_i in ((pair, pair["scores"][i]) for pair in usable))
        rankings.append({"hypothesis_index": i, "name": group[0].label, "aliases": [h.label for h in group[1:]],
                         "score_vs_no_motion": float(np.mean(relative)), "median_vs_no_motion": float(np.median(relative)),
                         "trimmed_mean": float(np.average(scores, weights=weights)), "pair_wins": wins})
    rankings.sort(key=lambda row: (row["score_vs_no_motion"], row["trimmed_mean"]))
    best = rankings[0]
    runner = rankings[1] if len(rankings) > 1 else best
    gap = (runner["score_vs_no_motion"] - best["score_vs_no_motion"]) / max(runner["score_vs_no_motion"], 1e-8)
    signal_pairs = sum(pair["moving_samples"] >= min_samples and pair["scores"][zero_index]["trimmed_mean"] > 1e-5 for pair in usable)
    declared_label = Hypothesis().label
    declared = next(row for row in rankings if declared_label in [row["name"], *row["aliases"]])
    if not signal_pairs or gap < 0.02:
        interpretation = "Weak or ambiguous separation. Capture a deliberate camera pan before inferring a convention change."
    elif "no-motion" in [best["name"], *best["aliases"]]:
        interpretation = "No-motion fits best. Check repeated/paused frames, temporal pairing and MV validity before changing signs or units."
    elif best is declared:
        interpretation = "The declared signed scale and normalization fit best among the tested mappings; no input-motion convention error is indicated."
    else:
        interpretation = "A different mapping fits better than the declared contract. This is evidence to investigate, not authorization to change game parameters."
    return {"rankings": rankings, "pairs": results, "skipped_transitions": skipped,
            "usable_pairs": len(usable), "pairs_with_motion_signal": signal_pairs,
            "runner_up_gap_fraction": gap, "declared_rank": rankings.index(declared) + 1,
            "interpretation": interpretation,
            "limits": ["Photometric matching can be biased by disocclusion, particles, reflections, animation and incorrect capture frame pairing.",
                       "Equivalent normalization/multiplier hypotheses are aliases; image evidence cannot distinguish them.",
                       "Unit hypotheses cannot restore missing motion pixels if an incorrect descriptor already caused the capture to crop the wrong region.",
                       "This samples raw MV inside surfaces. It does not reproduce depth dilation, FSR inference, temporal color history or GPU synchronization."]}


def self_test():
    def sequence(raw_factor=1.0, vector_sign=1, output_mv=False, jittered=False, actual_motion=True):
        frames, width, height = [], 128, 96
        shifts = np.array([2.1, -1.3]) if actual_motion else np.zeros(2)
        jitters = [np.array(x) for x in ((0, -1 / 6), (-.25, 1 / 6), (.25, -7 / 18), (-.375, -1 / 18),
                                                (.125, 5 / 18), (-.125, -5 / 18), (.375, 1 / 18), (-.4375, 7 / 18))]
        yy, xx = np.mgrid[:height, :width]
        for index, jitter in enumerate(jitters):
            x, y = xx - jitter[0] - index * shifts[0], yy - jitter[1] - index * shifts[1]
            red = .4 + .16 * np.sin(.19 * x + .09 * y) + .11 * np.cos(.31 * x - .11 * y)
            green = .5 + .21 * np.sin(.13 * x - .23 * y) + .08 * np.cos(.39 * y)
            blue = .3 + .17 * np.cos(.27 * x + .17 * y)
            rgb = np.stack([red, green, blue], axis=-1)
            rgb += (12 * np.exp(-((x - 72) ** 2 + (y - 52) ** 2) / 350))[..., None]
            exposure = 0.8 + index * .13
            color = np.concatenate([rgb * exposure, np.ones((*xx.shape, 1))], axis=-1).astype(np.float32)
            extent_scale = 2 if output_mv else 1
            motion = -np.array([2.1, -1.3]) * extent_scale
            if jittered and index:
                motion += jitters[index - 1] - jitter
            raw = motion / np.array([-1., -1.]) * raw_factor * vector_sign
            metadata = dict(index=index, rw=width, rh=height, uw=width * 2, uh=height * 2,
                            mvW=width * extent_scale, mvH=height * extent_scale, jitter=jitter.tolist(),
                            mvScale=[-1., -1.], jitteredMV=jittered, outputResMV=output_mv,
                            preExposure=exposure, reset=index == 0, owner="test", ownerFrame=index)
            frames.append(Frame(metadata, color, np.full((height, width), .5, dtype=np.float32),
                                np.broadcast_to(raw.astype(np.float32), (height * extent_scale, width * extent_scale, 2))))
        return frames

    cases = [("signed scale and jitter", {}, "declared/sign+1/x1"),
             ("inverse sign", {"vector_sign": -1}, "declared/sign-1/x1"),
             ("half multiplier", {"raw_factor": 2.0}, "declared/sign+1/x0.5"),
             ("double multiplier", {"raw_factor": .5}, "declared/sign+1/x2"),
             ("output-resolution units", {"output_mv": True}, "declared/sign+1/x1"),
             ("jittered motion", {"jittered": True}, "declared/sign+1/x1"),
             ("no-motion control", {"actual_motion": False}, "no-motion")]
    for name, arguments, expected in cases:
        result = analyze(sequence(**arguments), max_samples=1200)
        best = result["rankings"][0]
        assert expected in [best["name"], *best["aliases"]], (name, expected, best)
        assert result["usable_pairs"] == 7
        print(f"PASS {name}: {best['name']} ({best['score_vs_no_motion']:.4f} of no-motion)")
    zero_scale = sequence()
    for frame in zero_scale:
        frame.meta["mvScale"] = [0.0, 0.0]
    result = analyze(zero_scale, max_samples=1200)
    assert len(result["rankings"]) == 1 and result["pairs_with_motion_signal"] == 0
    assert "ambiguous" in result["interpretation"]
    print("PASS zero signed scales: ambiguous mapping, no false direction conclusion")
    reset_frames = sequence()
    reset_frames[4].meta["reset"] = True
    result = analyze(reset_frames, max_samples=1200)
    assert result["usable_pairs"] == 6 and len(result["skipped_transitions"]) == 1
    print("PASS reset frame: crossing pair excluded")
    with tempfile.TemporaryDirectory(prefix="fsr411-motion-selftest-") as temporary:
        folder, metadata = Path(temporary), []
        for frame in sequence()[:3]:
            entry = {**frame.meta, "files": {}}
            arrays = {"color": frame.color, "depth": frame.depth, "motion": frame.motion,
                      "output": np.zeros((entry["uh"], entry["uw"], 4), dtype="<f4")}
            for name, array in arrays.items():
                filename = f"f{entry['index']:03d}_{name}.bin"
                entry["files"][name] = filename
                np.asarray(array, dtype="<f4").tofile(folder / filename)
            metadata.append(entry)
        (folder / "capture.json").write_text(json.dumps({"schema_version": 1, "status": "complete", "frames": metadata}))
        _, _, restored, warnings = read_capture(folder)
        result = analyze(restored, max_samples=1200)
        assert not warnings and result["declared_rank"] == 1 and result["usable_pairs"] == 2
        (folder / metadata[-1]["files"]["output"]).write_bytes(b"\0" * 4)
        try:
            read_capture(folder)
        except ValueError as error:
            assert "bytes, expected" in str(error)
        else:
            raise AssertionError("truncated output was accepted")
    print("PASS capture schema/binary roundtrip and truncated-file rejection")
    print(f"{len(cases) + 3} synthetic/schema tests passed; no GPU work performed.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("capture", nargs="?", help="capture.json or its containing directory")
    parser.add_argument("--json", type=Path, help="write detailed analysis JSON here")
    parser.add_argument("--samples", type=int, default=16000, help="maximum samples per frame pair (default 16000)")
    parser.add_argument("--min-samples", type=int, default=128)
    parser.add_argument("--motion-sampling", choices=("nearest", "bilinear"), default="nearest")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if not args.capture:
        parser.error("provide a capture directory or --self-test")
    if not 1 <= args.min_samples <= args.samples:
        parser.error("require 1 <= --min-samples <= --samples")
    path, capture, frames, warnings = read_capture(args.capture)
    report = analyze(frames, args.samples, args.min_samples, args.motion_sampling)
    report.update(capture=str(path.resolve()), capture_status=capture.get("status"), frame_count=len(frames),
                  motion_sampling=args.motion_sampling, warnings=warnings,
                  metric="mean RGB absolute log1p HDR residual, largest 20% trimmed; normalized per pair to no-motion",
                  signed_motion_scales=[frame.meta["mvScale"] for frame in frames],
                  jitter_accounted_for=True)
    print(f"Capture: {path}\nFrames: {len(frames)}; usable pairs: {report.get('usable_pairs', 0)}")
    print("Lower score is better; no-motion = 1.000. Equal mappings are grouped.")
    print(f"{'Rank':>4}  {'Mapping':<29} {'Score':>9} {'Wins':>5}")
    for rank, row in enumerate(report["rankings"], 1):
        print(f"{rank:>4}  {row['name']:<29} {row['score_vs_no_motion']:>9.4f} {row['pair_wins']:>5}")
        if row["aliases"]:
            print("      aliases: " + ", ".join(row["aliases"]))
    print(report["interpretation"])
    for pair in report["pairs"]:
        if "skipped" in pair:
            print(f"Pair {pair['previous']} -> {pair['current']}: skipped ({pair['skipped']})")
        else:
            motion = pair["declared_motion_input_pixels"]
            print(f"Pair {pair['previous']} -> {pair['current']}: {pair['sample_count']} points, "
                  f"declared displacement median/p90={motion['median']:.3f}/{motion['p90']:.3f} input pixels")
    for warning in warnings:
        print("Warning: " + warning)
    if args.json:
        args.json.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(f"Detailed report: {args.json}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError) as error:
        raise SystemExit(f"Capture analysis failed: {error}")
