#!/usr/bin/env python3
"""Restore the exact MIT shader sources listed in source_manifest.json.

Reads an existing upstream checkout or downloads the same pinned raw files.
No upstream setup/build scripts are executed. All files are verified before
any destination is changed. Adaptation belongs in the compiler/Metal backend,
so the vendored HLSL remains auditable against upstream.
"""

import argparse
import concurrent.futures
import json
import sys
import urllib.request
from pathlib import Path

from import_model import DEFAULT_OUTPUT, sha256, write_if_changed


UPSTREAM_COMMIT = "f508ddecd46fb407d1be41c3102b54dddf57d220"
RAW_BASE = ("https://raw.githubusercontent.com/int3rrobang/"
            "fsr4-int8-reverse-engineering/" + UPSTREAM_COMMIT + "/")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkout", type=Path, help="local upstream checkout; every file hash must match")
    source.add_argument("--download", action="store_true", help="retrieve pinned raw source files")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    try:
        manifest = json.loads((args.output / "source_manifest.json").read_text())
        if manifest["upstream_commit"] != UPSTREAM_COMMIT:
            raise ValueError("source manifest refers to a different upstream commit")

        def read_file(row):
            upstream = Path(row["upstream_path"])
            destination = Path(row["path"])
            if any(path.is_absolute() or ".." in path.parts for path in (upstream, destination)):
                raise ValueError("unsafe source-manifest path")
            if args.checkout:
                data = (args.checkout / upstream).read_bytes()
            else:
                with urllib.request.urlopen(RAW_BASE + upstream.as_posix(), timeout=60) as response:
                    data = response.read(row["size"] + 1)
            if len(data) != row["size"] or sha256(data) != row["sha256"]:
                raise ValueError("upstream file hash/size mismatch: " + str(upstream))
            return destination, data

        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
            files = list(executor.map(read_file, manifest["files"]))
        for relative, data in files:
            write_if_changed(args.output / relative, data)
        print("Restored %d verified MIT source files from %s." % (len(files), UPSTREAM_COMMIT))
        return 0
    except (OSError, KeyError, ValueError) as error:
        print("FSR 4.1.1 source import failed: " + str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
