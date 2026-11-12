#!/usr/bin/env python3
"""Inspect NRX1 assets from a user-supplied ANR test kit, without running it.

The format was reconstructed from FG-V.B14's arm64 bin/check, SHA-256
64aa5480521e6bdfff499b759c6f2732eed4a12a3f88953244a25272fce0285d,
loader 0x10003f6b8 and transform 0x100040954. This is not an ANR source
release or a permission to redistribute model weights or compiled shaders.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import struct
import subprocess
import tempfile

MASK64 = (1 << 64) - 1
MAX_BYTES = 512 * 1024 * 1024


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def unmask(data):
    if len(data) < 16 or data[:4] != b"NRX1":
        raise ValueError("not an NRX1 asset")
    kind, nonce = struct.unpack_from("<IQ", data, 4)
    if kind not in (1, 2):
        raise ValueError("unsupported NRX1 kind: " + str(kind))
    state = nonce ^ 0x6E725F6E72785F31
    payload = bytearray(data[16:])
    for offset in range(0, len(payload), 8):
        state = (state + 0x9E3779B97F4A7C15) & MASK64
        word = ((state ^ (state >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
        word = ((word ^ (word >> 27)) * 0x94D049BB133111EB) & MASK64
        word ^= word >> 31
        for index in range(min(8, len(payload) - offset)):
            payload[offset + index] ^= (word >> (8 * index)) & 255
    return kind, bytes(payload)


def decompress(data):
    try:
        import zstandard
    except ImportError:
        # Optional external zstd already installed by the user. No install.
        with tempfile.TemporaryFile() as output:
            result = subprocess.run(["zstd", "-d", "-c", "--quiet"], input=data,
                                    stdout=output, stderr=subprocess.PIPE, timeout=30)
            if result.returncode:
                raise ValueError("zstd rejected NRX payload: " + result.stderr.decode(errors="replace"))
            if output.tell() > MAX_BYTES:
                raise ValueError("decompressed NRX asset is too large")
            output.seek(0)
            return output.read()
    try:
        with zstandard.ZstdDecompressor().stream_reader(io.BytesIO(data)) as reader:
            plain = reader.read(MAX_BYTES + 1)
        if len(plain) > MAX_BYTES:
            raise ValueError("decompressed NRX asset is too large")
        return plain
    except zstandard.ZstdError as error:
        raise ValueError("zstd rejected NRX payload: " + str(error)) from error


def decode(data):
    if len(data) > MAX_BYTES:
        raise ValueError("NRX asset is too large")
    kind, compressed = unmask(data)
    plain = decompress(compressed)
    if len(plain) > MAX_BYTES:
        raise ValueError("decompressed NRX asset is too large")
    if kind == 1:
        return kind, {"asset.bin": plain}
    if len(plain) < 4:
        raise ValueError("truncated pack directory")
    count, = struct.unpack_from("<I", plain)
    if count > 10000:
        raise ValueError("too many pack members")
    offset, directory, names = 4, [], set()
    for _ in range(count):
        if offset + 4 > len(plain):
            raise ValueError("truncated pack member")
        length, = struct.unpack_from("<I", plain, offset)
        offset += 4
        if not length or length > 4096 or offset + length + 16 > len(plain):
            raise ValueError("invalid pack member name length")
        name = plain[offset:offset + length].decode("utf-8")
        offset += length
        path = PurePosixPath(name)
        if (not name or "\0" in name or "\\" in name or ":" in name or path.is_absolute()
                or any(part in ("", ".", "..") for part in name.split("/"))
                or name.casefold() in names or name.casefold() == "nrx-inspection.json"):
            raise ValueError("unsafe or duplicate pack member: " + repr(name))
        names.add(name.casefold())
        start, size = struct.unpack_from("<QQ", plain, offset)
        offset += 16
        directory.append((name, start, size))
    members = {}
    for name, start, size in directory:
        if start + size > len(plain) - offset:
            raise ValueError("pack member escapes payload: " + name)
        members[name] = plain[offset + start:offset + start + size]
    return kind, members


def inspect(path, export=None):
    if path.is_symlink() or not path.is_file():
        raise ValueError("input must be a regular, non-symlink file")
    if path.stat().st_size > MAX_BYTES:
        raise ValueError("NRX asset is too large")
    data = path.read_bytes()
    kind, members = decode(data)
    report = {"schema": "anr_nrx_inspection_v1", "input_sha256": sha256(data),
              "kind": kind, "members": [], "runtime_executed": False}
    for name, payload in members.items():
        row = {"path": name, "bytes": len(payload), "sha256": sha256(payload)}
        if name.endswith(".json"):
            value = json.loads(payload)
            row["json_keys"] = sorted(value) if isinstance(value, dict) else None
        report["members"].append(row)
    if export is not None:
        # Never overwrite the supplied kit, a symlink, or earlier results.
        export.mkdir(parents=True, exist_ok=False)
        for name, payload in members.items():
            target = export / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
        (export / "nrx-inspection.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("asset", type=Path)
    parser.add_argument("--export", type=Path, help="new local output directory; never overwrites")
    args = parser.parse_args()
    try:
        print(json.dumps(inspect(args.asset, args.export), indent=2))
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        parser.exit(1, "ANR NRX: " + str(error) + "\n")


if __name__ == "__main__":
    main()
