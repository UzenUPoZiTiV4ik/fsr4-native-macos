#!/usr/bin/env python3
"""Verify a locally prepared ANR runtime before attaching it to a game."""
import hashlib
import json
from pathlib import Path, PurePosixPath

CHECKPOINT = "e09deb7012c96c172e49e72eb0b747b4611a8b6e3f14e5f5563189c580772ac1"
WEIGHTS = "af6c2f200ddb7baeaf4d82b5515af6f077f67896b2a9d26163ef6ea23975acb5"
KERNEL_PACK = "aeb6f568a241a63daf804a213de9c17dad2bdd7f7e79779f3002a9a3e1493767"
SIZES = {(1920, 1080), (2560, 1440), (3840, 2160)}
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


def validate_one(root):
    if root.is_symlink():
        raise ValueError("ANR runtime is a symlink")
    root = root.resolve(strict=True)
    path = regular(root, "manifest.json")
    if path.stat().st_size > MAX_BYTES:
        raise ValueError("ANR manifest is too large")
    data = json.loads(path.read_text())
    if not isinstance(data, dict) or data.get("schema") != "fsr4_anr_runtime_v1" or data.get("checkpoint_id") != CHECKPOINT:
        raise ValueError("unsupported ANR runtime/checkpoint")
    if (data.get("width"), data.get("height")) not in SIZES:
        raise ValueError("unsupported ANR runtime size")
    if data.get("backend", "native") not in ("native", "packed"):
        raise ValueError("unsupported ANR backend")
    if data.get("weights_sha256") != WEIGHTS or data.get("source_kernel_pack_sha256") != KERNEL_PACK:
        raise ValueError("ANR model/kernel source identity mismatch")
    rows = data.get("files")
    if not isinstance(rows, dict) or not rows or len(rows) > 1000:
        raise ValueError("invalid ANR runtime inventory")
    for name, row in rows.items():
        asset = regular(root, name)
        if not isinstance(row, dict) or type(row.get("bytes")) is not int or not 0 <= row["bytes"] <= MAX_BYTES:
            raise ValueError("invalid ANR asset size")
        if asset.stat().st_size != row["bytes"] or hashlib.sha256(asset.read_bytes()).hexdigest() != row.get("sha256"):
            raise ValueError("ANR asset hash/size mismatch: " + name)
    if rows.get("weights.bin", {}).get("sha256") != WEIGHTS:
        raise ValueError("unverified ANR weights")
    for child in root.rglob("*"):
        if child.is_symlink() or (not child.is_file() and not child.is_dir()):
            raise ValueError("nonregular ANR asset")
        if child.is_file() and child.relative_to(root).as_posix() not in (set(rows) | {"manifest.json"}):
            raise ValueError("unlisted ANR asset")
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
