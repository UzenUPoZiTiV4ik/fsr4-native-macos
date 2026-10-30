#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Fetch the FSR 4.1.2 Lossy integer weights into build/fsr411_lossy2/.

The weights belong to the GPL-2.0-or-later upstream project and are not
distributed with this MIT repository. Upstream release dll-2026-10-29 ships
them only inside its DLL, so this downloads the pinned RX 7000 archive
(about 9 MB), unpacks its lossy DLL with bsdtar, decodes the 18 pinned
trained-pass shaders with the vendored DXC (tools/fsr411/lossy412_weights.py)
and writes a byte-reproducible .npz. Archive, DLL, every shader container and
the resulting file are checked against pinned SHA-256 values. The Windows DLL
is never executed.
"""

import argparse
import hashlib
import importlib.util
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=ROOT / "build/fsr411_lossy2/weights_lossy2.npz")
    parser.add_argument("--archive", type=Path, help="use an already downloaded release .7z")
    parser.add_argument("--dll", type=Path, help="use the release's lossy DLL directly")
    parser.add_argument("--url", help="override the pinned archive URL (for a local mirror)")
    parser.add_argument("--dxc", type=Path, default=ROOT / "tools/bin/dxc")
    args = parser.parse_args()
    spec = load("fsr411_lossy2", ROOT / "tools/fsr411/lossy2.py")
    decoder = load("fsr411_lossy412_weights", ROOT / "tools/fsr411/lossy412_weights.py")
    target = args.out
    if target.is_file() and sha256(target) == spec.UPSTREAM_SHA256:
        print("Already present:", target)
        return 0
    with tempfile.TemporaryDirectory() as work:
        work = Path(work)
        dll = args.dll
        if dll is None:
            archive = args.archive
            if archive is None:
                archive = work / "release.7z"
                try:
                    with urllib.request.urlopen(args.url or spec.UPSTREAM_URL, timeout=120) as response, \
                            open(archive, "wb") as stream:
                        shutil.copyfileobj(response, stream)
                except OSError as error:
                    print("Lossy weights download failed:", error, file=sys.stderr)
                    return 2
            if sha256(archive) != spec.UPSTREAM_ARCHIVE_SHA256:
                print("Release archive hash mismatch: %s" % archive, file=sys.stderr)
                return 2
            tar = shutil.which("bsdtar") or shutil.which("tar")
            try:
                subprocess.run([tar, "-xf", str(archive), "-C", str(work), spec.UPSTREAM_DLL_MEMBER], check=True)
            except (OSError, subprocess.CalledProcessError) as error:
                print("Could not unpack %s with %s: %s" % (spec.UPSTREAM_DLL_MEMBER, tar, error), file=sys.stderr)
                return 2
            dll = work / spec.UPSTREAM_DLL_MEMBER
        if sha256(dll) != spec.UPSTREAM_DLL_SHA256:
            print("Lossy DLL hash mismatch: %s" % dll, file=sys.stderr)
            return 2
        try:
            arrays, containers = decoder.extract(dll, args.dxc, pinned=spec.UPSTREAM_CONTAINERS)
        except (OSError, ValueError, subprocess.CalledProcessError) as error:
            print("Could not decode the trained passes:", error, file=sys.stderr)
            return 2
        staged = work / "weights.npz"
        digest = decoder.write_npz(arrays, staged)
        if digest != spec.UPSTREAM_SHA256:
            print("Decoded weights hash mismatch: got %s" % digest, file=sys.stderr)
            return 2
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(".tmp")
        shutil.copyfile(staged, temporary)
        temporary.replace(target)
    print("Decoded %d trained-pass shaders of %s into %s" % (len(containers), spec.UPSTREAM_RELEASE, target))
    return 0


if __name__ == "__main__":
    sys.exit(main())
