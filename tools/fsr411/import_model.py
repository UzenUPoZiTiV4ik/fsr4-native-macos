#!/usr/bin/env python3
"""Import the official FSR 4.1.1 INT8 initializers and optional DXIL oracles.

The AMD DLL is read as data and is never loaded or executed. Only the pinned
FidelityFX SDK 2.3.0 artifact is accepted. All byte ranges and hashes are
validated before output files are written. Uses Python's standard library.
"""

import argparse
import hashlib
import json
import re
import struct
import sys
import tempfile
import urllib.request
from pathlib import Path


SDK_COMMIT = "60f4ea81909200d8542eca14dccb2628b763a9a3"
DLL_GIT_BLOB = "199de3a500a1d153b7e73fa5ca0adbd2a4ac1229"
DLL_RELATIVE_PATH = "Kits/FidelityFX/signedbin/amd_fidelityfx_upscaler_dx12.dll"
DLL_URL = ("https://raw.githubusercontent.com/GPUOpen-LibrariesAndSDKs/"
           "FidelityFX-SDK/" + SDK_COMMIT + "/" + DLL_RELATIVE_PATH)
DLL_SIZE = 28761864
DLL_SHA256 = "d0dcccc74a43c44ba435b7a369b456e0970d8a4464e4bd683119b374f2c9fb46"
INITIALIZER_OFFSET = 0xC04A00
INITIALIZER_SIZE = 131072
INITIALIZER_SHA256 = "a54e552ff69a3f7199861417f7ae461d1f67f636ab532c6ec9d8ea5d2963735a"
ULTRA_INITIALIZER_OFFSET = 0xC24A00
ULTRA_INITIALIZER_SHA256 = "3c42fc6eabfa9b5b1e489df0e9b2f58e3f9fa22048c584f6839bb27d32a0640e"
INITIALIZERS = {
    "initializer": {"model": "common_int8", "path": "initializers.bin",
                    "dll_offset": INITIALIZER_OFFSET, "size": INITIALIZER_SIZE,
                    "sha256": INITIALIZER_SHA256, "binding": "t18"},
    "ultraperformance_initializer": {
        "model": "ultraperformance_int8", "path": "initializers-ultraperformance.bin",
        "dll_offset": ULTRA_INITIALIZER_OFFSET, "size": INITIALIZER_SIZE,
        "sha256": ULTRA_INITIALIZER_SHA256, "binding": "t18"},
}
EXPECTED_INT8_CONTAINERS = 426
EXPECTED_INVENTORY_SHA256 = "1ef68b434f6500a5aa887a74529fadba694c0540256ab72e7ff03fef46769b6c"
ENTRY_PATTERN = re.compile(rb"fsr4_model_v07_fp8_no_scale_(?:prepass|postpass|pass\d+(?:_post)?)\x00")
DEFAULT_OUTPUT = Path(__file__).resolve().parents[2] / "third_party" / "fsr411"


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def inventory_sha256(rows):
    return sha256(json.dumps(rows, sort_keys=True, separators=(",", ":")).encode())


def parse_container(data, offset):
    """Return validated DXBC metadata, or None for a non-container match.

    DXBC is a chunked container even when its program is DXIL. Chunk offsets
    are relative to its header; a signature in an arbitrary byte string alone
    is insufficient evidence of a shader. No external disassembler is needed.
    """
    if offset < 0 or offset + 32 > len(data) or data[offset:offset + 4] != b"DXBC":
        return None
    version, size, count = struct.unpack_from("<III", data, offset + 20)
    header_size = 32 + 4 * count
    if version != 1 or not 1 <= count <= 128 or size < header_size or offset + size > len(data):
        return None
    ranges = []
    chunks = []
    program_version = None
    for index in range(count):
        relative = struct.unpack_from("<I", data, offset + 32 + 4 * index)[0]
        if relative < header_size or relative + 8 > size:
            return None
        tag, length = struct.unpack_from("<4sI", data, offset + relative)
        end = relative + 8 + length
        if end > size or not all(32 <= value < 127 for value in tag):
            return None
        if any(relative < previous_end and previous_start < end
               for previous_start, previous_end in ranges):
            return None
        ranges.append((relative, end))
        chunks.append({"fourcc": tag.decode("ascii"), "offset": relative, "size": length})
        if tag == b"DXIL":
            if length < 24 or program_version is not None:
                return None
            program_version = struct.unpack_from("<I", data, offset + relative + 8)[0]
    if program_version is None:
        return None
    return {"offset": offset, "size": size, "program_version": program_version,
            "container_checksum": data[offset + 4:offset + 20].hex(), "chunks": chunks}


def inventory_int8_shaders(data):
    """Enumerate this artifact's INT8 model shaders, including debug variants.

    AMD retained an entry name containing 'fp8_no_scale' in its INT8 shaders.
    This *pinned* DLL's compute shader model 6.6 variants implement INT8; the
    similarly named shader model 6.4 variants are FP8 and are excluded.
    """
    position = 0
    rows = []
    while True:
        position = data.find(b"DXBC", position)
        if position < 0:
            break
        container = parse_container(data, position)
        position += 4
        if container is None or container["program_version"] != 0x00050066:
            continue
        begin, length = container["offset"], container["size"]
        blob = data[begin:begin + length]
        names = sorted(set(match.group()[:-1].decode("ascii") for match in ENTRY_PATTERN.finditer(blob)))
        if not names:
            continue
        if len(names) != 1:
            raise ValueError("ambiguous model entry names at DLL offset 0x%x" % begin)
        digest = sha256(blob)
        # Several DLL permutation-table entries embed identical containers.
        # Keep each source offset in the inventory; materialization deduplicates
        # naturally through the content-addressed filename.
        container.update({"sha256": digest, "entry_point": names[0], "shader_model": "cs_6_6",
                          "oracle_path": "oracle/dxil/" + digest + ".dxil"})
        rows.append(container)
    return rows


def inspect_official_payload(data):
    """Fail before any write if the official artifact or its payload changed."""
    digest = sha256(data)
    if len(data) != DLL_SIZE or digest != DLL_SHA256:
        raise ValueError("expected official FSR 4.1.1.2740 DLL (%d bytes, SHA-256 %s); got %d bytes, %s"
                         % (DLL_SIZE, DLL_SHA256, len(data), digest))
    git_blob = hashlib.sha1(b"blob " + str(len(data)).encode("ascii") + b"\0" + data).hexdigest()
    if git_blob != DLL_GIT_BLOB:
        raise ValueError("official Git blob identity mismatch")
    for model in INITIALIZERS.values():
        initializer = data[model["dll_offset"]:model["dll_offset"] + model["size"]]
        if len(initializer) != model["size"] or sha256(initializer) != model["sha256"]:
            raise ValueError("FSR 4.1.1 %s initializer mismatch" % model["model"])
    shaders = inventory_int8_shaders(data)
    if len(shaders) != EXPECTED_INT8_CONTAINERS:
        raise ValueError("expected %d INT8 containers; found %d" % (EXPECTED_INT8_CONTAINERS, len(shaders)))
    if inventory_sha256(shaders) != EXPECTED_INVENTORY_SHA256:
        raise ValueError("original INT8 shader inventory mismatch")
    manifest = {
        "schema": "fsr411_import_manifest_v1",
        "provider_version": "4.1.1.2740",
        "sdk_version": "2.3.0",
        "source": {"url": DLL_URL, "sdk_commit": SDK_COMMIT, "git_blob": DLL_GIT_BLOB,
                   "size": DLL_SIZE, "sha256": DLL_SHA256},
        **INITIALIZERS,
        "selection": {
            "common_model": ["native", "quality", "balanced", "performance", "drs"],
            "ultraperformance": "Separate initializer imported. It requires its own baked weights and quantization parameters; common kernels alone do not implement it.",
            "shader_inventory": "All 426 original INT8 containers, including debug and Ultra Performance variants. Entry names alone do not select a variant.",
            "validation_scope": "Artifact identities and source provenance verified; cross-backend output equivalence requires GPU validation."
        },
        "shader_count": len(shaders),
        "unique_shader_count": len(set(row["sha256"] for row in shaders)),
        "shaders": shaders,
    }
    return data[INITIALIZER_OFFSET:INITIALIZER_OFFSET + INITIALIZER_SIZE], manifest


def write_if_changed(path, data):
    if path.is_file() and path.read_bytes() == data:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=str(path.parent), prefix=path.name + ".", delete=False) as file:
        temporary = Path(file.name)
        file.write(data)
    try:
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def verify_sources(output):
    manifest_path = output / "source_manifest.json"
    if not manifest_path.is_file():
        raise ValueError("missing vendored-source manifest: " + str(manifest_path))
    source_manifest = json.loads(manifest_path.read_text())
    for row in source_manifest["files"]:
        relative = Path(row["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("unsafe source path in source manifest")
        path = output / relative
        if not path.is_file() or sha256(path.read_bytes()) != row["sha256"]:
            raise ValueError("vendored source differs from pinned upstream: " + str(path))
    return len(source_manifest["files"])


def verify_initializers(output):
    """Check both model payloads against fixed identities, not mutable receipts."""
    for model in INITIALIZERS.values():
        path = output / model["path"]
        data = path.read_bytes()
        if len(data) != model["size"] or sha256(data) != model["sha256"]:
            raise ValueError("%s initializer hash/size mismatch" % model["model"])


def read_verified_manifest(output):
    manifest = json.loads((output / "import_manifest.json").read_text())
    if manifest["source"]["sha256"] != DLL_SHA256 or manifest["shader_count"] != EXPECTED_INT8_CONTAINERS:
        raise ValueError("import receipt does not describe the pinned official artifact")
    if any(manifest.get(key) != value for key, value in INITIALIZERS.items()):
        raise ValueError("initializer receipt does not describe both pinned official models")
    if inventory_sha256(manifest["shaders"]) != EXPECTED_INVENTORY_SHA256:
        raise ValueError("original INT8 shader inventory mismatch")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--dll", type=Path, help="read the pinned official DLL as bytes")
    source.add_argument("--download", action="store_true", help="download only the pinned official AMD artifact")
    source.add_argument("--verify", action="store_true", help="verify an existing import and vendored source hashes")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--with-shaders", action="store_true", help="also materialize original DXIL under the ignored oracle/ directory")
    args = parser.parse_args(argv)
    try:
        if args.verify:
            count = verify_sources(args.output)
            verify_initializers(args.output)
            manifest = read_verified_manifest(args.output)
            if args.with_shaders:
                for row in manifest["shaders"]:
                    path = args.output / row["oracle_path"]
                    blob = path.read_bytes()
                    if len(blob) != row["size"] or sha256(blob) != row["sha256"]:
                        raise ValueError("oracle shader differs: " + str(path))
            print("Verified %d vendored sources and both FSR 4.1.1 initializers%s." %
                  (count, " plus all 426 original INT8 shaders" if args.with_shaders else ""))
            return 0
        if args.download:
            with urllib.request.urlopen(DLL_URL, timeout=60) as response:
                data = response.read(DLL_SIZE + 1)
        else:
            data = args.dll.read_bytes()
        _, manifest = inspect_official_payload(data)
        for model in INITIALIZERS.values():
            begin = model["dll_offset"]
            write_if_changed(args.output / model["path"], data[begin:begin + model["size"]])
        if args.with_shaders:
            for row in manifest["shaders"]:
                begin = row["offset"]
                write_if_changed(args.output / row["oracle_path"], data[begin:begin + row["size"]])
        write_if_changed(args.output / "import_manifest.json", (json.dumps(manifest, indent=2) + "\n").encode())
        print("Imported official FSR 4.1.1.2740 common and Ultra Performance INT8 models: two 131072-byte initializers; %d original shader records%s."
              % (len(manifest["shaders"]), "; DXIL files extracted" if args.with_shaders else ""))
        print(str(args.output.resolve()))
        return 0
    except (OSError, ValueError, KeyError) as error:
        print("FSR 4.1.1 import failed: " + str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
