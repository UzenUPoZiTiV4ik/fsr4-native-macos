#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Fetch the pinned lossy2 integer weights into build/fsr411_lossy2/.

The archive belongs to the GPL-2.0-or-later upstream project and is not
distributed with this MIT repository, so the lossy2 profile needs this
one-time download (about 50 KB). The exact revision and SHA-256 are pinned.
"""

import argparse
import hashlib
import importlib.util
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]



def load_spec():
    spec = importlib.util.spec_from_file_location("fsr411_lossy2", ROOT / "tools/fsr411/lossy2.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "build/fsr411_lossy2/weights_lossy2.npz")
    parser.add_argument("--url", help="override the pinned source (for a local mirror)")
    args = parser.parse_args()
    module = load_spec()
    target = args.out
    if target.is_file() and hashlib.sha256(target.read_bytes()).hexdigest() == module.UPSTREAM_SHA256:
        print("Already present:", target)
        return 0
    try:
        with urllib.request.urlopen(args.url or module.UPSTREAM_URL, timeout=60) as response:
            data = response.read(1 << 20)
    except OSError as error:
        print("Lossy2 weights download failed:", error, file=sys.stderr)
        return 2
    digest = hashlib.sha256(data).hexdigest()
    if digest != module.UPSTREAM_SHA256:
        print("Lossy2 weights hash mismatch: got %s" % digest, file=sys.stderr)
        return 2
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".tmp")
    temporary.write_bytes(data)
    temporary.replace(target)
    print("Fetched %d bytes from %s into %s" % (len(data), module.UPSTREAM_COMMIT, target))
    return 0


if __name__ == "__main__":
    sys.exit(main())
