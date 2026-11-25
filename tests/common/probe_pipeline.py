# SPDX-License-Identifier: MIT
"""Shared compile-probe steps; importing puts ROOT/"tools" on sys.path."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
from fsr411.compile_fsr411 import run
from fsr402.compile_variant import patch_spv

TOOL = ROOT / "tools/bin"


def hlsl_to_msl(hlsl, spv, *, includes=(), defines=(), execute=run):
    """DXC -> patch_spv -> spirv-val -> spirv-cross; returns (msl, patch count)."""
    out = Path(spv).parent
    command = [TOOL / "dxc", "-spirv", "-T", "cs_6_6", "-E", "main",
               "-enable-16bit-types", "-HV", "2021", "-O3", "-Vd",
               "-fspv-target-env=vulkan1.3", "-fvk-t-shift", "32", "0",
               "-fvk-u-shift", "64", "0", "-Fo", spv]
    for include in includes:
        command += ["-I", include]
    execute(command + list(defines) + [hlsl], out / "dxc.log")
    patches = patch_spv(str(spv))
    execute([TOOL / "spirv-val", "--target-env", "vulkan1.3", spv],
            out / "validation.log")
    msl = execute([TOOL / "spirv-cross", spv, "--msl", "--msl-version", "20300"]).decode()
    return msl, patches


def run_probe(label, main):
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError) as error:
        print(f"{label}: {error}", file=sys.stderr)
        raise SystemExit(1)
