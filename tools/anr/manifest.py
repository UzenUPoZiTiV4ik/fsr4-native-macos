#!/usr/bin/env python3
"""Verify a locally prepared ANR runtime (v1 B14 or v2 B20) before attaching it to a game."""
import hashlib
import json
from pathlib import Path, PurePosixPath

# Legacy FG-V.B14 pins.
CHECKPOINT = "e09deb7012c96c172e49e72eb0b747b4611a8b6e3f14e5f5563189c580772ac1"
WEIGHTS = "af6c2f200ddb7baeaf4d82b5515af6f077f67896b2a9d26163ef6ea23975acb5"
KERNEL_PACK = "aeb6f568a241a63daf804a213de9c17dad2bdd7f7e79779f3002a9a3e1493767"
SIZES = {(1920, 1080), (2560, 1440), (3840, 2160)}

# FG-V.B20 dynamic-resolution pins.
V20_CHECKPOINTS = {
    "packed": "b7481951e4f54e4f2d534ec41c8f3d8e721f4fd22839d2ba2bd236cbd28bea64",
    "dp4a": "bef1b3e76649887e6cb36f178a3011752d51b4e868f22e70f7c4a5a8d62408f4",
}
V20_KERNELS_HASH = "15b864a39495d8d9a0a92cb3799928055e6b9e5d1ad4136233f818570a2d2876"
V20_KERNELS_PORTABLE_HASH = "d41b9a87806d8529f18c898a83c4e4a83e720b3e1cafbece7daf06175fa622a2"
V20_WEIGHTS = {
    "packed": {
        "pq10": "781148c820e87ac6",
        "srgb8": "15e6cbe88e0d5633",
        "scrgb16": "5627f40cadb51a34",
    },
    "dp4a": {
        "pq10": "10c219f7fc7fdf4a",
        "srgb8": "6c1d6d46d8db2f71",
        "scrgb16": "02921ad73599d27f",
    },
}
MAX_BYTES = 32 * 1024 * 1024


def regular(root, name):
    if not isinstance(name, str) or not name or "\\" in name or "\0" in name or ":" in name:
        raise ValueError("unsafe ANR asset path")
    path = PurePosixPath(name)
    if path.is_absolute() or any(part in ("", ".", "..") for part in name.split("/")):
        raise ValueError("unsafe ANR asset path")
    value = root
    for part in path.parts:
        value = value / part
        if value.is_symlink():
            raise ValueError("symlink in ANR runtime")
    if not value.is_file():
        raise ValueError("missing ANR asset: " + name)
    return value


def validate_files_inventory(root, data):
    rows = data.get("files")
    if not isinstance(rows, dict) or not rows or len(rows) > 1000:
        raise ValueError("invalid ANR runtime inventory")
    for name, row in rows.items():
        asset = regular(root, name)
        if not isinstance(row, dict) or type(row.get("bytes")) is not int or not 0 <= row["bytes"] <= MAX_BYTES:
            raise ValueError("invalid ANR asset size")
        if asset.stat().st_size != row["bytes"] or hashlib.sha256(asset.read_bytes()).hexdigest() != row.get("sha256"):
            raise ValueError("ANR asset hash/size mismatch: " + name)
    for child in root.rglob("*"):
        if child.is_symlink() or (not child.is_file() and not child.is_dir()):
            raise ValueError("nonregular ANR asset")
        if child.is_file() and child.relative_to(root).as_posix() not in (set(rows) | {"manifest.json"}):
            raise ValueError("unlisted ANR asset")
    return rows


def validate_v2(root, data):
    backend = data.get("backend")
    if backend not in V20_CHECKPOINTS or data.get("checkpoint_id") != V20_CHECKPOINTS[backend]:
        raise ValueError("unsupported ANR V.B20 runtime/checkpoint")
    if (data.get("source_kernels_sha256") != V20_KERNELS_HASH
            or data.get("source_kernels_portable_sha256") != V20_KERNELS_PORTABLE_HASH):
        raise ValueError("ANR V.B20 kernel source identity mismatch")
    rows = validate_files_inventory(root, data)
    metallib = data.get("metallib")
    if not isinstance(metallib, str) or metallib not in rows:
        raise ValueError("missing ANR V.B20 metallib in inventory")
    metallib_mpp = data.get("metallib_mpp")
    if metallib_mpp is not None and (not isinstance(metallib_mpp, str) or metallib_mpp not in rows):
        raise ValueError("missing ANR V.B20 MPP metallib in inventory")
    encodings = data.get("encodings")
    if not isinstance(encodings, dict):
        raise ValueError("missing ANR V.B20 encodings table")
    for enc_name, prefix in V20_WEIGHTS[backend].items():
        entry = encodings.get(enc_name)
        if not isinstance(entry, dict):
            raise ValueError("missing ANR V.B20 encoding: " + enc_name)
        wfile = entry.get("weights")
        pfile = entry.get("program")
        if wfile not in rows or pfile not in rows:
            raise ValueError("missing ANR V.B20 files for " + enc_name)
        if not rows[wfile]["sha256"].startswith(prefix):
            raise ValueError("unverified ANR V.B20 weights for " + enc_name)
    return root


def validate_one(root):
    if root.is_symlink():
        raise ValueError("ANR runtime is a symlink")
    root = root.resolve(strict=True)
    path = regular(root, "manifest.json")
    if path.stat().st_size > MAX_BYTES:
        raise ValueError("ANR manifest is too large")
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError("unsupported ANR runtime/checkpoint")
    schema = data.get("schema")
    if schema == "fsr4_anr_runtime_v2":
        return validate_v2(root, data)
    if schema != "fsr4_anr_runtime_v1" or data.get("checkpoint_id") != CHECKPOINT:
        raise ValueError("unsupported ANR runtime/checkpoint")
    if (data.get("width"), data.get("height")) not in SIZES:
        raise ValueError("unsupported ANR runtime size")
    if data.get("backend", "native") not in ("native", "packed"):
        raise ValueError("unsupported ANR backend")
    if data.get("weights_sha256") != WEIGHTS or data.get("source_kernel_pack_sha256") != KERNEL_PACK:
        raise ValueError("ANR model/kernel source identity mismatch")
    rows = validate_files_inventory(root, data)
    if rows.get("weights.bin", {}).get("sha256") != WEIGHTS:
        raise ValueError("unverified ANR weights")
    return root


def validate_runtime(directory):
    root = Path(directory).expanduser().absolute()
    if (root / "manifest.json").is_file():
        return validate_one(root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("ANR runtime directory missing")
    found = []
    for width, height in sorted(SIZES):
        profile = root / (str(width) + "x" + str(height))
        if profile.exists():
            checked = validate_one(profile)
            data = json.loads((checked / "manifest.json").read_text())
            if (data["width"], data["height"]) != (width, height):
                raise ValueError("ANR runtime profile name disagrees with dimensions")
            found.append(checked)
    if not found:
        raise ValueError("supply a prepared ANR runtime, not the original test kit")
    return root.resolve()
