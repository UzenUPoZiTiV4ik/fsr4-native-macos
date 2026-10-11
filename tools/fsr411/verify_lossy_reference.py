#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Compare generated lossy weights with the pinned published DLL as data.

This optional validation does not build, link, load or execute the Windows
provider or upstream GPL tools. Native DXC disassembles verified containers
in a temporary directory. Only identities and aggregate evidence are saved;
the reference DLL and its shaders are never vendored into the project.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

import import_model
import lossy
import recover_ultraperformance as recovery


ZIP_URL = (lossy.REFERENCE_REPOSITORY + "/releases/download/" +
           lossy.REFERENCE_RELEASE + "/test-lossy.zip")
ZIP_SHA256 = "02b4e1a468962204345dd2195c58e764166f2a9b66526658d22d3bd5a2938a46"
DLL_SHA256 = "f9541e4f897e6eb029d470bac5338c8c62bb5988412bd31ae995e2531a236bd9"
DLL_BYTES = 45273088


def words_sha256(words):
    return hashlib.sha256(b"".join(word.to_bytes(4, "little") for word in words)).hexdigest()


def inspect_reference(dll, source_dir, dxc):
    data = Path(dll).read_bytes()
    if len(data) != DLL_BYTES or hashlib.sha256(data).hexdigest() != DLL_SHA256:
        raise ValueError("expected the pinned dll-2026-10-07 test-lossy.dll payload")
    source_dir = Path(source_dir)
    import_model.verify_sources(source_dir)
    importer_manifest = import_model.read_verified_manifest(source_dir)
    original_hashes = {row["sha256"] for row in importer_manifest["shaders"]}
    expected = {}
    for stage in lossy.FOLD_WORDS:
        source = (source_dir / "shaders/reimpl411" / ("reimpl_pass%d.hlsl" % stage)).read_text()
        expected[stage] = recovery.source_words(lossy.apply_hlsl_lossy(source, stage))
    names = {"fsr4_model_v07_fp8_no_scale_pass%d" % stage: stage for stage in expected}
    candidates = [row for row in import_model.inventory_int8_shaders(data)
                  if row["entry_point"] in names and row["sha256"] not in original_hashes]
    with tempfile.TemporaryDirectory(prefix="fsr411-lossy-oracle-") as directory:
        temporary = Path(directory)

        def compare(row):
            path = temporary / (row["sha256"] + ".dxil")
            path.write_bytes(data[row["offset"]:row["offset"] + row["size"]])
            result = subprocess.run([str(dxc), "-dumpbin", str(path)], capture_output=True, text=True)
            if result.returncode:
                raise ValueError("native DXC could not disassemble " + row["sha256"] + ": " + result.stderr[-2000:])
            stage = names[row["entry_point"]]
            actual = recovery.embedded_words(result.stdout)
            if actual != expected[stage]:
                return None  # Other preset's unchanged coefficient tables.
            return {
                "pass": stage, "offset": row["offset"], "size": row["size"],
                "shader_sha256": row["sha256"], "container_checksum": row["container_checksum"],
                "baked_word_count": len(actual), "baked_weights_sha256": words_sha256(actual),
            }

        with ThreadPoolExecutor(max_workers=3) as pool:
            matched = [row for row in pool.map(compare, candidates) if row is not None]
    for stage in expected:
        if sum(row["pass"] == stage for row in matched) != 3:
            raise ValueError("expected all three published output classes to match for pass " + str(stage))
    return {
        "schema": 1,
        "reference": {
            "repository": lossy.REFERENCE_REPOSITORY, "commit": lossy.REFERENCE_COMMIT,
            "release": lossy.REFERENCE_RELEASE, "asset_url": ZIP_URL,
            "asset_sha256": ZIP_SHA256, "dll_sha256": DLL_SHA256, "dll_bytes": DLL_BYTES,
            "execution": "none; native DXC disassembly only",
        },
        "policy": {
            "common_fold_words": lossy.FOLD_WORDS,
            "common_intermediate_half_up_passes": sorted(lossy.ROUNDING_PASSES),
            "ultraperformance_intermediate_half_up_passes": sorted(lossy.SHARED_ROUNDING_PASSES),
            "tail_rounding": "nearest, ties to even; unchanged",
            "folding": "minimum squared int8x4 norm, nearest retained spatial tap in the same input quartet, saturating int8 addition",
        },
        "validation": "All baked weight tables in all twelve selected published shaders match the independent generated HLSL exactly. This is coefficient validation, not GPU output or temporal quality equivalence.",
        "passes": [
            {"pass": stage, "baked_word_count": len(words), "baked_weights_sha256": words_sha256(words)}
            for stage, words in sorted(expected.items())
        ],
        "shaders": sorted(matched, key=lambda row: (row["pass"], row["offset"])),
    }


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dll", type=Path, required=True, help="DLL extracted from the pinned test-lossy.zip; read as data")
    parser.add_argument("--source-dir", type=Path, default=root / "third_party/fsr411")
    parser.add_argument("--dxc", type=Path, default=root / "tools/bin/dxc")
    parser.add_argument("--output", type=Path, default=root / "third_party/fsr411/lossy_model_manifest.json")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    receipt = inspect_reference(args.dll, args.source_dir, args.dxc)
    serialized = json.dumps(receipt, indent=2) + "\n"
    if args.verify:
        if not args.output.exists() or args.output.read_text() != serialized:
            raise SystemExit("published lossy coefficient receipt differs; validation failed")
    else:
        import_model.write_if_changed(args.output, serialized.encode())
    print("Verified 12 published lossy shader coefficient tables (4 model passes x 3 output classes).")


if __name__ == "__main__":
    main()
