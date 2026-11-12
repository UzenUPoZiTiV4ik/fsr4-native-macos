#!/usr/bin/env python3
"""Build or initialize the optional Netal SR integration without changing a bottle."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from netal_integration import ROOT, prepare_sources, validate_runtime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, help="complete local v2 bundle, not source-only assets")
    parser.add_argument("--arch", choices=("arm64", "x86_64"), default="arm64")
    parser.add_argument("--build-only", action="store_true", help="compile without models; no inference claim")
    args = parser.parse_args()
    if not args.build_only and args.runtime is None:
        parser.error("--runtime is required unless --build-only is set")
    try:
        bundle = validate_runtime(args.runtime) if args.runtime else None
        generated = prepare_sources(args.source)
        output = ROOT / "build/tests" / ("netal_probe_" + args.arch)
        output.parent.mkdir(parents=True, exist_ok=True)
        command = ["xcrun", "--sdk", "macosx", "clang++", "-std=c++20", "-O2", "-Wall", "-Wextra", "-Werror",
                   "-fobjc-arc", "-mmacosx-version-min=26.0", "-arch", args.arch,
                   "-I", str(ROOT / "src"), "-I", str(generated), "-DFSR4_WITH_NETAL=1",
                   str(ROOT / "tests/netal_probe.mm"), str(ROOT / "src/netal_live.mm"),
                   str(generated / "dlss_sr_session.mm"), str(generated / "dlss_sr_texture_adapter.mm"),
                   "-framework", "Metal", "-framework", "Foundation", "-o", str(output)]
        subprocess.run(command, check=True)
        if not args.build_only:
            subprocess.run(["arch", "-" + args.arch, str(output), str(bundle)], check=True)
        else:
            print(json.dumps({"compiled": True, "arch": args.arch, "models_loaded": False, "inference_executed": False}))
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as error:
        parser.exit(1, "Netal probe: " + str(error) + "\n")


if __name__ == "__main__":
    main()
