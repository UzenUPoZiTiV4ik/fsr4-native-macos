#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Exercise the imported PRE color-distance expression through the Metal toolchain.

The expression is read verbatim from the imported 2160p PRE, embedded in a
buffer-fed HLSL kernel, and passed through production portability_source.
The GPU harness compares it with an independent double-precision std::hypot
reference for signed chroma deltas. --raw bypasses the portability transform
to reproduce the former negative-base powr failure; it is expected to fail on
an affected toolchain and is not a passing CI mode. --compile-only prepares
both the shader and harness without using the GPU.
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


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--builddir", type=Path, default=ROOT / "build/fsr411-chroma-probe",
                        help="artifact parent; production and raw use separate subdirectories")
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--raw", action="store_true",
                        help="diagnose the untransformed expression; failures retain a nonzero exit status")
    args = parser.parse_args()
    mode = "raw" if args.raw else "production"
    output = args.builddir.resolve() / mode
    output.mkdir(parents=True, exist_ok=True)

    source_path = ROOT / "third_party/fsr411/shaders/reimpl2160/pre_rev_411_v15_2160.hlsl"
    source = source_path.read_text()
    matches = list(re.finditer(r"^[ \t]*const float color_distance = [^;\n]+;", source, re.M))
    if len(matches) != 1:
        raise ValueError("expected exactly one imported PRE color_distance declaration")
    expression = matches[0].group().lstrip()
    source_line = source.count("\n", 0, matches[0].start()) + 1
    # Inputs come from a runtime buffer, so DXC/Metal cannot constant-fold the
    # negative-base cases. The ABI is float4(cbCurrent, cbHistory, crCurrent, crHistory).
    snippet = """StructuredBuffer<float4> inputPairs : register(t0);
RWStructuredBuffer<float> outputDistances : register(u0);

[numthreads(1, 1, 1)]
void main(uint3 dtid : SV_DispatchThreadID)
{
    const float4 pair = inputPairs[dtid.x];
    float3 input_ycbcr = float3(0.0f, pair.x, pair.z);
    float3 history_ycbcr = float3(0.0f, pair.y, pair.w);
    """ + expression + """
    outputDistances[dtid.x] = color_distance;
}
"""
    (output / "chroma.imported.hlsl").write_text(snippet)
    hlsl = output / "chroma.hlsl"
    hlsl.write_text(snippet if args.raw else portability_source(snippet))
    tool = ROOT / "tools/bin"
    spv, metal, air, library = [output / ("chroma." + ext)
                               for ext in ("spv", "metal", "air", "metallib")]
    commands = []

    def execute(command, log=None):
        command = [str(arg) for arg in command]
        commands.append(command)
        return run(command, log)

    # Match compile_fsr411.py's PRE DXC/SPIRV-Cross flags. This small kernel
    # has no imported includes, feature defines, loop hints or literal tables.
    execute([tool / "dxc", "-spirv", "-T", "cs_6_6", "-E", "main",
             "-enable-16bit-types", "-HV", "2021", "-O3", "-Vd",
             "-fspv-target-env=vulkan1.3", "-fvk-t-shift", "32", "0",
             "-fvk-u-shift", "64", "0", "-Fo", spv, hlsl], output / "dxc.log")
    patch_count = patch_spv(str(spv))
    execute([tool / "spirv-val", "--target-env", "vulkan1.3", spv],
            output / "validation.log")
    msl = execute([tool / "spirv-cross", spv, "--msl", "--msl-version", "20300"]).decode()
    entry = "fsr411_chroma_probe"
    msl, renamed = re.subn(r"(kernel\s+void\s+)\w+(\s*\()",
                           rf"\g<1>{entry}\2", msl, count=1)
    if renamed != 1:
        raise ValueError("missing Metal probe entry")
    metal.write_text(msl)
    math_options = ["-O3", "-fno-fast-math", "-ffp-contract=off"]
    execute(["xcrun", "metal", "-std=macos-metal2.3", *math_options,
             "-c", metal, "-o", air], output / "metal.log")
    execute(["xcrun", "metallib", air, "-o", library], output / "metallib.log")

    buffers, textures, samplers = parse_signature(msl, entry)
    if sorted(binding[1] for binding in buffers) != ["inputPairs", "outputDistances"] or textures or samplers:
        raise ValueError("unexpected reflected probe resources")
    workgroup = wgsize(str(spv), "main", str(tool / "spirv-cross"))
    if workgroup != [1, 1, 1]:
        raise ValueError("probe requires exactly one invocation per dispatched threadgroup")
    info = {"entry": entry, "library": library.name, "wgsize": workgroup,
            "buffers": buffers, "textures": textures, "samplers": samplers,
            "mode": mode}
    (output / "binding.json").write_text(json.dumps(info, indent=2) + "\n")

    harness = ROOT / "tests/fsr411_chroma_test.mm"
    binary = output / "fsr411_chroma_test"
    execute(["xcrun", "clang++", "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
             "-fno-fast-math", "-ffp-contract=off", "-fobjc-arc",
             "-framework", "Foundation", "-framework", "Metal", harness, "-o", binary],
            output / "harness.log")
    receipt = {
        "mode": mode,
        "source": str(source_path.relative_to(ROOT)),
        "source_sha256": sha256(source_path),
        "source_line": source_line,
        "imported_expression": expression,
        "production_transform": None if args.raw else "compile_fsr411.portability_source",
        "input_layout": ["cbCurrent", "cbHistory", "crCurrent", "crHistory"],
        "expected_math": "std::hypot(double(cbCurrent)-cbHistory, double(crCurrent)-crHistory, 1e-3)",
        "metal_options": ["-std=macos-metal2.3", *math_options],
        "metal_version": run(["xcrun", "metal", "--version"]).decode().strip(),
        "spirv_copy_patches": patch_count,
        "tool_sha256": {name: sha256(tool / name) for name in ("dxc", "spirv-cross", "spirv-val")},
        "production_source_sha256": {
            name: sha256(ROOT / name)
            for name in ("tools/compile_fsr411.py", "tools/compile_variant.py", "tools/fsr411/portability.py")
        },
        "artifact_sha256": {path.name: sha256(path) for path in
                            (hlsl, spv, metal, library, binary)},
        "harness_source_sha256": sha256(harness),
        "runner_source_sha256": sha256(Path(__file__)),
        "commands": commands,
        "reflection_command": [str(tool / "spirv-cross"), str(spv), "--reflect"],
    }
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    # A compile-only run must not leave an older GPU result appearing current.
    (output / "results.json").unlink(missing_ok=True)
    print(f"Chroma probe ({mode}) artifacts: {output}", flush=True)
    if args.compile_only:
        print("Shader and harness compiled; GPU not run.")
        return 0
    return subprocess.run([str(binary), str(output)], check=False).returncode


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError) as error:
        print(f"FSR 4.1.1 chroma probe: {error}", file=sys.stderr)
        raise SystemExit(1)
