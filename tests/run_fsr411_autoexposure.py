#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Compile and compare split SPD exposure with the original single dispatch.

The oracle receives the exposure-store ordering fix and an explicit device
barrier before atomic publication to preserve HLSL globallycoherent semantics
on Metal. --raw-oracle retains the original synchronization for diagnosis.
Both oracle modes retain the original atomic path, mip-six behavior and
physical tile reads. All kernels use the same strict Metal arithmetic options.
--compile-only prepares artifacts without using the GPU.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from compile_fsr411 import portability_source, run
from compile_variant import parse_signature, patch_spv, wgsize
from fsr411.autoexposure import split_autoexposure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--builddir", type=Path, default=ROOT / "build/fsr411-ae-probe")
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--raw-oracle", action="store_true",
                        help="diagnose the original globallycoherent translation without an explicit publication fence")
    args = parser.parse_args()
    output = args.builddir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    source_path = ROOT / "third_party/fsr411/shaders/ae_411_ours/ae_ours.hlsl"
    source = source_path.read_text()
    previous_first = (
        "            rw_auto_exposure_texture[int2(1,0)] = prev_exposre;\n"
        "            rw_auto_exposure_texture[int2(0,0)] = GetExposure(outValue.x);"
    )
    oracle, count = re.subn(
        r"\t\t\trw_auto_exposure_texture\[int2\(0,0\)\] = GetExposure\(outValue.x\);\n"
        r"            rw_auto_exposure_texture\[int2\(1,0\)\] = prev_exposre;",
        previous_first, source,
    )
    if count != 1:
        raise ValueError("oracle exposure-store shape changed")
    if not args.raw_oracle:
        old_counter = ("void SpdIncreaseAtomicCounter(FfxUInt32 slice)\n"
                       "{\n    InterlockedAdd(rw_spd_global_atomic[FfxInt32x2(0, 0)], 1, spdCounter);")
        if oracle.count(old_counter) != 1:
            raise ValueError("oracle atomic callback shape changed")
        oracle = oracle.replace(old_counter,
                                "void SpdIncreaseAtomicCounter(FfxUInt32 slice)\n"
                                "{\n    DeviceMemoryBarrier();\n"
                                "    InterlockedAdd(rw_spd_global_atomic[FfxInt32x2(0, 0)], 1, spdCounter);")
    variants = {"ae_oracle.hlsl": oracle, **split_autoexposure(source)}
    tool = ROOT / "tools/bin"
    includes = [
        ROOT / "third_party/fsr4/api/internal/gpu",
        ROOT / "third_party/fsr4/include/gpu",
    ]
    receipt = {"source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
               "oracle_changes": ["write previous exposure before overwriting current"],
               "metal_options": ["-O3", "-fno-fast-math", "-ffp-contract=off"],
               "variants": {}}
    if not args.raw_oracle:
        receipt["oracle_changes"].append("device memory barrier before publishing the tile through its atomic counter")
    for name, text in variants.items():
        label = Path(name).stem
        target = output / label
        target.mkdir(exist_ok=True)
        hlsl = target / name
        hlsl.write_text(portability_source(text))
        spv, metal, air, library = [target / (label + "." + ext)
                                    for ext in ("spv", "metal", "air", "metallib")]
        command = [tool / "dxc", "-spirv", "-T", "cs_6_6", "-E", "main",
                   "-enable-16bit-types", "-HV", "2021", "-O3", "-Vd",
                   "-fspv-target-env=vulkan1.3", "-fvk-t-shift", "32", "0",
                   "-fvk-u-shift", "64", "0", "-Fo", spv]
        for include in includes:
            command += ["-I", include]
        command += ["-DFFX_GPU=1", "-DFFX_HLSL=1", "-DFFX_HLSL_SM=66", hlsl]
        run(command, target / "dxc.log")
        patch_spv(str(spv))
        run([tool / "spirv-val", "--target-env", "vulkan1.3", spv],
            target / "validation.log")
        msl = run([tool / "spirv-cross", spv, "--msl", "--msl-version", "20300"]).decode()
        msl = msl.replace(".xyzz, uint2", ".xyzx, uint2")
        msl = re.sub(r"^([ \t]*)(for|while)\s*\(",
                     r"\1#pragma clang loop unroll(disable)\n\1\2 (", msl, flags=re.M)
        entry = "fsr411_" + label
        msl, count = re.subn(r"(kernel\s+void\s+)\w+(\s*\()",
                             rf"\g<1>{entry}\2", msl, count=1)
        if count != 1:
            raise ValueError("missing Metal entry")
        if label != "ae_oracle" and ("atomic_fetch" in msl or "rw_spd_global_atomic" in msl):
            raise ValueError("split AE still contains an atomic path")
        first = msl.index("rw_auto_exposure_texture.write(")
        second = msl.index("rw_auto_exposure_texture.write(", first + 1)
        if ".read(" not in msl[first:second]:
            raise ValueError("previous exposure read was sunk past current write")
        metal.write_text(msl)
        run(["xcrun", "metal", "-std=macos-metal2.3", *receipt["metal_options"],
             "-c", metal, "-o", air], target / "metal.log")
        run(["xcrun", "metallib", air, "-o", library])
        buffers, textures, samplers = parse_signature(msl, entry)
        info = {"entry": entry, "library": library.name,
                "wgsize": wgsize(str(spv), "main", str(tool / "spirv-cross")),
                "buffers": buffers, "textures": textures, "samplers": samplers}
        (target / "binding.json").write_text(json.dumps(info, indent=2) + "\n")
        receipt["variants"][label] = {
            "hlsl_sha256": hashlib.sha256(hlsl.read_bytes()).hexdigest(),
            "metallib_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
        }
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    binary = output / "fsr411_autoexposure_test"
    run(["xcrun", "clang++", "-std=c++17", "-O2", "-fno-fast-math", "-ffp-contract=off",
         "-fobjc-arc", "-framework", "Foundation", "-framework", "Metal",
         ROOT / "tests/fsr411_autoexposure_test.mm", "-o", binary], output / "harness.log")
    print(f"AE probe artifacts: {output}", flush=True)
    if not args.compile_only:
        result = subprocess.run([str(binary), str(output)], check=False)
        return result.returncode
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError) as error:
        print(f"FSR 4.1.1 AE probe: {error}", file=sys.stderr)
        raise SystemExit(1)
