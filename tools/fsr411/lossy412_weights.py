#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Recover the FSR 4.1.2 Lossy integer weights from the upstream release DLL.

Upstream release dll-2026-10-29 retrained six of the twelve passes (1, 2, 4,
5, 10, 12) of its former lossy2 build but published the weights only inside
its DLL, as literal operands of the passes' packed INT8 dot products. This
module reads them back without executing the DLL: it takes the 18 pinned DXBC
containers (six passes x three tensor tiers), disassembles each with the
vendored DXC, walks the straight-line body symbolically and rebuilds
bias + sum(word . literal) for every channel of the 3x3, expansion and
contraction stages. All three tiers must decode to identical weights.

The decoder was checked against the previous release, whose weights are
published: decoding dll-2026-10-22's lossy2 DLL reproduces
research/lossy2/weights_lossy2.npz exactly, channel order included
(`--check DLL NPZ`).

The weights are GPL-2.0-or-later data of the upstream project and are not
distributed with this MIT repository; tools/fsr411/fetch_lossy2_weights.py
writes them into build/ once.
"""

import argparse
import hashlib
import re
import struct
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ARRAYS = ("A", "bA", "Bw", "bB", "Cw", "bC")
PASSES = (1, 2, 4, 5, 10, 12)

# Input tensor base -> (pass, tier); the 3x3 stage's first word selects both.
_CNB_BASE = {8342464: (2, 1080), 33273664: (2, 2160), 132902464: (2, 4320),
             12537792: (12, 1080), 49958592: (12, 2160), 199449792: (12, 4320)}
_FNB_BASE = {8342464: (4, 1080), 33273664: (4, 2160), 132902464: (4, 4320),
             12537792: (5, 1080), 49958592: (5, 2160), 199449792: (5, 4320),
             14659648: (10, 1080), 58349248: (10, 2160), 232819648: (10, 4320)}
_ROW_BYTES = {"cnb": {1080: 15392, 2160: 30752, 4320: 61472},
              "fnb": {1080: 7712, 2160: 15392, 4320: 30752}}
_PLANE_BYTES = {1080: 2097664, 2160: 8342464, 4320: 33273664}
_INST = re.compile(r'^\s*%(\w+) = (.*)$')
_BIN = re.compile(r'(add|sub|mul|shl|or|and|xor|lshr|ashr)( nuw| nsw)* i32 ([^,]+), (.+)$')


def _s32(value):
    value &= 0xFFFFFFFF
    return value - (1 << 32) if value & 0x80000000 else value


def _i8(word, byte):
    value = (word >> (8 * byte)) & 255
    return value - 256 if value > 127 else value


def dxbc_containers(data):
    """Yield every DXBC container of a PE image as bytes."""
    position = 0
    while (position := data.find(b"DXBC", position)) >= 0:
        size = struct.unpack_from("<I", data, position + 24)[0] if position + 28 <= len(data) else 0
        if 32 < size < 5_000_000 and position + size <= len(data):
            yield data[position:position + size]
            position += size
        else:
            position += 4


class _Lin:
    """bias + sum(symbol . literal), built by chained dot4AddPacked/add."""
    __slots__ = ("const", "terms")

    def __init__(self, const=0, terms=()):
        self.const, self.terms = const, tuple(terms)

    def key(self):
        return frozenset(self.terms)


class _Deriv:
    __slots__ = ("deps",)

    def __init__(self, deps=None):
        self.deps = deps or {}


def _args(call):
    inner = call[call.index("(") + 1:call.rindex(")")]
    out, depth, current = [], 0, ""
    for ch in inner:
        depth += ch in "{("
        depth -= ch in "})"
        if ch == "," and depth == 0:
            out.append(current)
            current = ""
        else:
            current += ch
    out.append(current)
    return [a.strip().split(" ")[-1] for a in out]


def _walk(text):
    values, packs, stores, consts, order = {}, [], [], {}, []

    def get(token):
        token = token.strip()
        if token.startswith("%"):
            return values.get(token[1:], _Deriv())
        try:
            return int(token)
        except ValueError:
            return _Deriv()

    def deps(value):
        if isinstance(value, _Lin):
            return {value.key(): {value.const}}
        return value.deps if isinstance(value, _Deriv) else {}

    def merge(*items):
        out = {}
        for item in items:
            for key, seen in deps(item).items():
                out.setdefault(key, set()).update(seen)
        return _Deriv(out)

    def note(lin):
        key = lin.key()
        if key not in consts:
            order.append(key)
        consts.setdefault(key, set()).add(lin.const)
        return lin

    for raw in text.splitlines():
        line = raw.split(";")[0].rstrip()
        if "rawBufferStore" in line and line.lstrip().startswith("call"):
            a = _args(line)
            stores.append((get(a[2]), [get(x) for x in a[4:8]]))
            continue
        match = _INST.match(line)
        if not match:
            continue
        name, rhs = match.groups()
        if rhs.startswith("call"):
            op = re.search(r"@dx\.op\.([\w.]+)", rhs)[1]
            a = _args(rhs)
            if op.startswith(("threadId", "groupId", "flattenedThreadIdInGroup")):
                values[name] = 0
            elif op.startswith("rawBufferLoad"):
                address = get(a[2])
                values[name] = ("LOAD", address if isinstance(address, int) else None)
            elif op.startswith("dot4AddPacked"):
                acc, x, y = get(a[1]), get(a[2]), get(a[3])
                if isinstance(x, int) and not isinstance(y, int):
                    x, y = y, x
                symbol = x if isinstance(x, tuple) and x[0] in ("L", "P") else None
                if symbol is not None and isinstance(y, int) and isinstance(acc, (int, _Lin)):
                    base = _Lin(acc) if isinstance(acc, int) else acc
                    values[name] = note(_Lin(base.const, base.terms + ((symbol, _s32(y)),)))
                else:
                    values[name] = merge(acc, x, y)
            elif op.startswith("pack4x8"):
                keys = []
                for component in (get(t) for t in a[2:6]):
                    found = deps(component)
                    keys.append(max(found, key=len) if found else None)
                values[name] = ("P", len(packs))
                packs.append(keys)
            else:
                values[name] = merge(*[get(t) for t in a[1:]])
            continue
        if rhs.startswith("extractvalue"):
            source = get(rhs.split()[2].rstrip(","))
            index = int(rhs.split(",")[-1])
            values[name] = ("L", source[1]) if isinstance(source, tuple) and source[0] == "LOAD" and index == 0 \
                else merge(source)
            continue
        match = _BIN.match(rhs)
        if match:
            op, x, y = match[1], get(match[3]), get(match[4])
            if isinstance(x, int) and isinstance(y, int):
                values[name] = _s32({"add": x + y, "sub": x - y, "mul": x * y, "shl": x << y, "or": x | y,
                                     "and": x & y, "xor": x ^ y, "lshr": (x & 0xFFFFFFFF) >> y,
                                     "ashr": _s32(x) >> y}[op])
            elif op == "add" and isinstance(x, (int, _Lin)) and isinstance(y, (int, _Lin)):
                xl = x if isinstance(x, _Lin) else _Lin(x)
                yl = y if isinstance(y, _Lin) else _Lin(y)
                values[name] = note(_Lin(_s32(xl.const + yl.const), xl.terms + yl.terms))
            else:
                values[name] = merge(x, y)
            continue
        values[name] = merge(*[get(t) for t in re.findall(r"%\w+", rhs)])
    keys = [k for k in order if not any(k < other for other in order)]
    return keys, consts, packs, stores


def _zeros(*shape):
    if len(shape) == 1:
        return [0] * shape[0]
    return [_zeros(*shape[1:]) for _ in range(shape[0])]


def decode_pass(text):
    """Disassembled trained pass -> (pass, tier, {A, bA, Bw, bB, Cw, bC})."""
    keys, consts, packs, stores = _walk(text)
    kind = {64: "cnb", 112: "fnb"}.get(len(keys))
    if kind is None:
        raise ValueError("not a trained pass: %d linear forms" % len(keys))
    c1, hidden, outputs = (16, 32, 16) if kind == "cnb" else (16, 64, 32)
    planes = outputs // 16
    stage = {}
    for key in keys:
        symbols = {symbol[0] for symbol, _ in key}
        stage[key] = "A" if symbols == {"L"} and len(key) == 36 else \
            "C" if symbols == {"P"} and len(key) == hidden // 4 else "B"
    a_keys = [k for k in keys if stage[k] == "A"]
    offsets = sorted({symbol[1] for k in a_keys for symbol, _ in k})
    base = offsets[0]
    if base == 0:
        logical = 1
        tier = next(t for t, row in _ROW_BYTES[kind].items() if offsets[-1] // row == 2 and offsets[-1] % row < 48)
    else:
        logical, tier = (_CNB_BASE if kind == "cnb" else _FNB_BASE)[base]
    if (kind == "cnb") != (logical in (1, 2, 12)):
        raise ValueError("pass %d decoded with the wrong block kind" % logical)
    row = _ROW_BYTES[kind][tier]
    bias = {k: min(consts[k]) for k in keys}

    def stage_packs(name, count):
        found = [i for i, ks in enumerate(packs) if all(k is not None and stage.get(k) == name for k in ks)]
        if len(found) != count:
            raise ValueError("pass %d: %d %s packs, expected %d" % (logical, len(found), name, count))
        return found

    a_packs = stage_packs("A", c1 // 4)
    b_packs = stage_packs("B", hidden // 4)
    W = {"A": _zeros(c1, 16, 3, 3), "bA": _zeros(c1), "Bw": _zeros(hidden, c1 + 16 * (planes - 1)),
         "bB": _zeros(hidden), "Cw": _zeros(outputs, hidden), "bC": _zeros(outputs)}
    for q, pack in enumerate(a_packs):
        for j, key in enumerate(packs[pack]):
            channel = 4 * q + j
            W["bA"][channel] = bias[key]
            for symbol, literal in key:
                ky, rest = divmod(symbol[1] - base, row)
                kx, rest = divmod(rest, 16)
                for byte in range(4):
                    W["A"][channel][4 * (rest // 4) + byte][ky][kx] = _i8(literal, byte)
    a_word = {pack: q for q, pack in enumerate(a_packs)}
    second_plane = base + _PLANE_BYTES[tier] + row + 16
    for g, pack in enumerate(b_packs):
        for j, key in enumerate(packs[pack]):
            channel = 4 * g + j
            W["bB"][channel] = bias[key]
            for symbol, literal in key:
                word = a_word[symbol[1]] if symbol[0] == "P" else c1 // 4 + (symbol[1] - second_plane) // 4
                for byte in range(4):
                    W["Bw"][channel][4 * word + byte] = _i8(literal, byte)
    b_word = {pack: g for g, pack in enumerate(b_packs)}
    first_store = min(offset for offset, _ in stores)
    seen = set()
    for offset, components in stores:
        plane = 0 if offset == first_store else 1
        for w, component in enumerate(components):
            for j, key in enumerate(packs[component[1]]):
                channel = 16 * plane + 4 * w + j
                seen.add(channel)
                W["bC"][channel] = bias[key]
                for symbol, literal in key:
                    for byte in range(4):
                        W["Cw"][channel][4 * b_word[symbol[1]] + byte] = _i8(literal, byte)
    if len(seen) != outputs:
        raise ValueError("pass %d: %d of %d output channels stored" % (logical, len(seen), outputs))
    return logical, tier, W


def disassemble(dxc, container):
    with tempfile.TemporaryDirectory() as work:
        source, listing = Path(work) / "pass.dxbc", Path(work) / "pass.ll"
        source.write_bytes(container)
        subprocess.run([str(dxc), "-dumpbin", str(source), "-Fc", str(listing)], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        return listing.read_text()


def extract(dll_path, dxc, pinned=None):
    """Decode the six trained passes; every tier must give the same weights.

    The generated passes carry no shader name, unlike AMD's own; without a
    pinned set every unnamed container is tried and non-matching ones skipped.
    """
    data = Path(dll_path).read_bytes()
    containers = [c for c in dxbc_containers(data)
                  if pinned is None or hashlib.sha256(c).hexdigest() in pinned]
    if pinned is not None and len({hashlib.sha256(c).hexdigest() for c in containers}) != len(pinned):
        raise ValueError("the DLL lacks some of the pinned trained-pass containers")
    if pinned is None:
        containers = [c for c in containers if not re.search(rb"fsr4_model_v07_fp8_no_scale_", c)]
    decoded = {}
    for container in containers:
        try:
            logical, tier, weights = decode_pass(disassemble(dxc, container))
        except (ValueError, KeyError, StopIteration):
            if pinned is not None:
                raise
            continue
        if (logical, tier) in decoded:
            raise ValueError("pass %d tier %d appears twice" % (logical, tier))
        decoded[(logical, tier)] = (hashlib.sha256(container).hexdigest(), weights)
    expected = {(p, t) for p in PASSES for t in (1080, 2160, 4320)}
    if set(decoded) != expected:
        raise ValueError("decoded %s, expected six passes in three tiers" % sorted(decoded))
    for logical in PASSES:
        reference = decoded[(logical, 2160)][1]
        for tier in (1080, 4320):
            if decoded[(logical, tier)][1] != reference:
                raise ValueError("pass %d: tier %d weights differ from the 2160 tier" % (logical, tier))
    out = {"p%d_%s" % (p, n): decoded[(p, 2160)][1][n] for p in PASSES for n in ARRAYS}
    return out, sorted(h for h, _ in decoded.values())


def _shape(value):
    shape = []
    while isinstance(value, list):
        shape.append(len(value))
        value = value[0]
    return tuple(shape)


def _flat(value):
    if isinstance(value, list):
        for item in value:
            yield from _flat(item)
    else:
        yield value


def npy_bytes(value):
    shape = _shape(value)
    header = "{'descr': '<i4', 'fortran_order': False, 'shape': (%s), }" % (
        "".join("%d, " % d for d in shape).rstrip(" ") if len(shape) > 1 else "%d," % shape[0])
    header += " " * ((64 - (10 + len(header) + 1) % 64) % 64) + "\n"
    flat = list(_flat(value))
    return b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header)) + header.encode("latin1") + \
        struct.pack("<%di" % len(flat), *flat)


def write_npz(arrays, path):
    """Byte-reproducible .npz: fixed member order and timestamps, stored."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_STORED) as bundle:
        for name in sorted(arrays):
            info = zipfile.ZipInfo(name + ".npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.external_attr = 0o644 << 16
            bundle.writestr(info, npy_bytes(arrays[name]))
    temporary.replace(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_npz(path):
    sys.path.insert(0, str(ROOT / "tools"))
    from fsr411 import lossy2
    archive = lossy2._read_npz(path)
    out = {}
    for name, (shape, flat) in archive.items():
        def build(dims, offset):
            if len(dims) == 1:
                return flat[offset:offset + dims[0]], offset + dims[0]
            rows = []
            for _ in range(dims[0]):
                row, offset = build(dims[1:], offset)
                rows.append(row)
            return rows, offset
        out[name] = build(list(shape), 0)[0]
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dll", type=Path, help="the release's lossy amd_fidelityfx_upscaler_dx12.dll")
    parser.add_argument("--out", type=Path, help="write the weights as .npz")
    parser.add_argument("--check", type=Path, metavar="NPZ", help="compare the decoded weights with a published .npz")
    parser.add_argument("--dxc", type=Path, default=ROOT / "tools/bin/dxc")
    args = parser.parse_args()
    arrays, containers = extract(args.dll, args.dxc)
    print("decoded 6 passes x 3 tiers from %d containers" % len(containers))
    if args.check:
        reference = read_npz(args.check)
        bad = [name for name in sorted(arrays) if reference.get(name) != arrays[name]]
        if bad:
            print("differs from %s: %s" % (args.check, ", ".join(bad)), file=sys.stderr)
            return 1
        print("identical to", args.check)
    if args.out:
        print("wrote %s sha256 %s" % (args.out, write_npz(arrays, args.out)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
