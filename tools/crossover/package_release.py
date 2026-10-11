#!/usr/bin/env python3
"""Build the FSR 4.1-only CrossOver release zip from a git revision.

The zip holds what crossover_fsr4.py install needs (hook, live bridge, panel,
FSR 4.1.1 model, shader tools) plus INSTALL.md. FSR 4.0.2 weights, tests,
benchmark data and 4.0.2 docs are left out; the panel then offers only
FSR 4.1 and FSR 4.1 Lite.

  python3 tools/crossover/package_release.py v1.0 [--rev HEAD] [--out dist/release]
"""

import argparse
import io
import subprocess
import tarfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXCLUDE_PREFIXES = (
    "third_party/fsr4/internal/shaders/fsr4_model_v07_i8_",  # FSR 4.0.2 weights
    "tests/", "tools/bench/", "tools/fsr411/test_", "docs/benchmarks/",
    "docs/FSR402_", "docs/GUIDE_FSR402", "docs/HARDWARE_FSR402", "patches/",
    "build_shaders.sh", "TESTDATA.md", "VERIFICATION.md", ".gitignore",
)


def included(path: str) -> bool:
    return not path.startswith(EXCLUDE_PREFIXES)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("version")
    p.add_argument("--rev", default="HEAD")
    p.add_argument("--out", default=str(ROOT / "dist" / "release"))
    args = p.parse_args()

    name = f"fsr4-mac-{args.version}-crossover"
    out = Path(args.out) / f"{name}.zip"
    out.parent.mkdir(parents=True, exist_ok=True)
    tar = subprocess.check_output(["git", "archive", "--format=tar", args.rev], cwd=ROOT)
    count = 0
    with tarfile.open(fileobj=io.BytesIO(tar)) as src, \
            zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as dst:
        for member in src.getmembers():
            if not member.isfile() or not included(member.name):
                continue
            info = zipfile.ZipInfo(f"{name}/{member.name}", date_time=(2026, 1, 1, 0, 0, 0))
            info.external_attr = (0o100000 | member.mode) << 16  # keep exec bits for tools/bin
            info.compress_type = zipfile.ZIP_DEFLATED
            dst.writestr(info, src.extractfile(member).read())
            count += 1
        guide = (ROOT / "tools" / "crossover" / "INSTALL.md").read_bytes()
        info = zipfile.ZipInfo(f"{name}/INSTALL.md", date_time=(2026, 1, 1, 0, 0, 0))
        info.external_attr = 0o100644 << 16
        info.compress_type = zipfile.ZIP_DEFLATED
        dst.writestr(info, guide)
    print(f"{out}: {count + 1} files, {out.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
