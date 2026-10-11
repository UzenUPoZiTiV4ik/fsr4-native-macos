#!/usr/bin/env python3
"""Precompile legacy FSR 4.0.2 shaders for this Mac or a chosen backend."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PRESETS = ("native", "quality", "balanced", "performance", "ultraperf", "drs")
DEFAULT_PRESETS = ("quality", "balanced", "performance", "ultraperf")


def selected_models(request, info):
    if request == "all":
        return ("f16", "i8mpp")
    if request == "auto":
        backend = info["backend"]
        if backend not in ("f16", "i8mpp"):
            raise ValueError(f"unexpected automatic backend: {backend}")
        return (backend,)
    return (request,)


def compile_command(root, model, preset, resolution, auto_exposure, offset=None):
    args = [sys.executable, str(root / "tools/compile_variant.py"),
            "--root", str(root), "--preset", preset, "--res", resolution,
            "--res-idx", str(("1080", "2160", "4320").index(resolution)),
            "--depth-inverted", "1", "--low-res-mv", "1", "--jittered-mv", "0",
            "--colorspace", "0", "--auto-exposure", str(auto_exposure),
            "--debug-view", "0", "--rcas", "0", "--model", "f16" if model == "f16" else "i8",
            "--builddir", str(root / "build")]
    if model == "i8mpp":
        if offset is None:
            raise ValueError("i8mpp needs its generated post parameter offset")
        args += ["--post-ext-ct2d", str(offset)]
    return args


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("auto", "f16", "i8mpp", "all"), default="auto")
    parser.add_argument("--presets", nargs="+", choices=PRESETS, default=DEFAULT_PRESETS)
    parser.add_argument("--res", nargs="+", choices=("1080", "2160", "4320"), default=("1080", "2160"))
    args = parser.parse_args(argv)
    info = {}
    if args.model == "auto":
        # Query the legacy policy explicitly, including
        # hardware and OS checks. Do not infer capabilities from CPU names.
        info = json.loads(subprocess.check_output(
            [str(ROOT / "tools/bin/fsr4mac"), "--fsr-version", "4.0.2", "--backend-info"], text=True))
        print(f"{info['device']}: {info['backend']} ({info['reason']})", flush=True)
    for model in selected_models(args.model, info):
        for preset in args.presets:
            for resolution in args.res:
                offset = None
                if model == "i8mpp":
                    gdir = ROOT / "build" / f"gen_i8mpp_{preset}_{resolution}"
                    generator = ROOT / "tools/gen_i8mpp.py"
                    manifest = gdir / "i8mpp_manifest.json"
                    if not manifest.exists() or manifest.stat().st_mtime < generator.stat().st_mtime:
                        subprocess.run([sys.executable, str(generator), "--root", str(ROOT),
                                        "--preset", preset, "--res", resolution,
                                        "--outdir", str(gdir)], check=True)
                    offset = json.loads(manifest.read_text())["post_ext_ct2d_off"]
                for auto_exposure in (0, 1):
                    subprocess.run(compile_command(ROOT, model, preset, resolution,
                                                   auto_exposure, offset), check=True)


if __name__ == "__main__":
    main()
