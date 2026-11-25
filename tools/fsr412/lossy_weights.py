#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Shared static DXIL decoding primitives for the pinned November 22 Lite Lossy importer.

No DLL/shader execution, GPU, NumPy, upstream Python, or GPL source is used.
Only the POST decoder needed by `nov22_weights.py` remains here; the obsolete
November 1 original-graph recovery route is intentionally not part of this
module anymore.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import re
import struct
import subprocess
import tempfile

COMMON_SHA256 = "4b9898da719dc2dca5dc45fd71dec5f80bdb9886a580fe3038af59efe637d55e"
INITIALIZER_BYTES = 27392

# (name, initializer byte offset, byte count). Includes the POST CT2D tail
# which the native external-CNB operator itself does not consume.
LAYOUT = {
    "post": (("A", 23040, 2304), ("bA", 25344, 64), ("B", 25472, 512),
             ("bB", 25984, 128), ("C", 26112, 512), ("bC", 26624, 64),
             ("CT", 26752, 512), ("bCT", 27264, 32)),
}
_DOT = re.compile(r"^\s*%([\w.]+) = call i32 @dx\.op\.dot4AddPacked\.i32\(i32 (\d+), i32 ([^,]+), i32 ([^,]+), i32 ([^)]+)\)", re.M)
_INT = re.compile(r"-?\d+\Z")


def _require(condition, reason):
    if not condition:
        raise ValueError("Lite Lossy decoder: " + reason)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def _words(values):
    _require(all(isinstance(v, int) and -(1 << 31) <= v < (1 << 32) for v in values), "non-INT32 literal")
    return struct.pack("<%dI" % len(values), *(v & 0xffffffff for v in values))


def _table_map(stage, base):
    return {name: base[offset:offset + size] for name, offset, size in LAYOUT[stage]}


def _section(data):
    _require(data[:2] == b"MZ" and len(data) > 64, "not a PE image")
    pe = struct.unpack_from("<I", data, 0x3c)[0]
    _require(pe + 24 <= len(data) and data[pe:pe + 4] == b"PE\0\0", "PE header changed")
    sections = struct.unpack_from("<H", data, pe + 6)[0]
    optional = struct.unpack_from("<H", data, pe + 20)[0]
    start = pe + 24 + optional
    _require(start + sections * 40 <= len(data), "truncated PE sections")
    found = []
    for index in range(sections):
        pos = start + index * 40
        if data[pos:pos + 8].rstrip(b"\0") == b".fsr4":
            size, offset = struct.unpack_from("<II", data, pos + 16)
            found.append((offset, size))
    _require(len(found) == 1 and found[0][0] == 28750336 and
             found[0][0] + found[0][1] <= len(data), ".fsr4 section changed")
    return found[0]


def dxbc_containers(data, start=0, end=None):
    """Yield (absolute offset, container bytes), validating DXBC chunk bounds."""
    end = len(data) if end is None else end
    _require(0 <= start <= end <= len(data), "invalid DXBC scan bounds")
    position = start
    while True:
        position = data.find(b"DXBC", position, end)
        if position < 0:
            return
        if position + 32 <= end:
            version, size, chunks = struct.unpack_from("<III", data, position + 20)
            valid = (version == 1 and 32 < size < 5000000 and chunks <= 64 and
                     32 + chunks * 4 <= size and position + size <= end)
            if valid:
                for index in range(chunks):
                    offset = struct.unpack_from("<I", data, position + 32 + index * 4)[0]
                    if offset < 32 + chunks * 4 or offset + 8 > size:
                        valid = False
                        break
                    count = struct.unpack_from("<I", data, position + offset + 4)[0]
                    if offset + 8 + count > size:
                        valid = False
                        break
            if valid:
                yield position, data[position:position + size]
                position += size
                continue
        position += 4


def disassemble(dxc, container):
    """DXC reads bytes as data; the PE and shader entry point are never run."""
    with tempfile.TemporaryDirectory(prefix="fsr412-lossy-dxil-") as work:
        source, output = Path(work) / "shader.dxbc", Path(work) / "shader.ll"
        source.write_bytes(container)
        result = subprocess.run([str(dxc), "-dumpbin", str(source), "-Fc", str(output)],
                                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        _require(result.returncode == 0 and output.is_file(),
                 "DXC disassembly failed: " + result.stderr.decode(errors="replace")[-2000:])
        return output.read_text()


def _dots(text, count):
    dots = list(_DOT.finditer(text))
    _require(len(dots) == count and all(m[2] == "163" for m in dots),
             "signed packed INT8 dot count/opcode changed")
    return dots


def _literal_dots(dots):
    _require(all(_INT.fullmatch(m[5].strip()) for m in dots), "unknown packed weight operand")
    return [int(m[5]) for m in dots]


def _cnb_weights(dots):
    """1408 unrolled calls: A interior/border copies, B output-major, C output-major."""
    _require(len(dots) == 1408, "CNB call boundary changed")
    words = _literal_dots(dots)
    _require(words[:576] == words[576:1152], "interior/border 3x3 weights differ")
    # Interior A is tap-major, channel-major, four packed input words.
    for tap in range(9):
        for channel in range(16):
            group = dots[tap * 64 + channel * 4:tap * 64 + channel * 4 + 4]
            _require(len({m[4] for m in group}) == 4, "3x3 packed input word shape")
    for start, outputs, inputs in ((1152, 32, 4), (1280, 16, 8)):
        for channel in range(outputs):
            group = dots[start + channel * inputs:start + (channel + 1) * inputs]
            for previous, current in zip(group, group[1:]):
                _require(current[3] == "%" + previous[1], "pointwise accumulator chain changed")
    return {"A": _words(words[:576]), "B": _words(words[1152:1280]), "C": _words(words[1280:1408])}


def _post_biases(dots):
    def roots(start, outputs, inputs):
        values = [dots[start + channel * inputs][3] for channel in range(outputs)]
        _require(all(_INT.fullmatch(value) for value in values), "unknown POST bias root")
        return _words([int(value) for value in values])
    return {"bA": roots(0, 16, 4), "bB": roots(1152, 32, 4), "bC": roots(1280, 16, 8)}


def _edge_biases(text, dots):
    """First missing tap PHIs, in reverse channel order, then accumulated taps."""
    region = text[dots[576].start():dots[640].start()]
    phis = re.findall(r"^\s*%([\w.]+) = phi i32 \[ (-?\d+), %[^\]]+ \], \[ %([\w.]+), %[^\]]+ \]", region, re.M)
    _require(len(phis) == 16, "POST first-tap conditional bias PHIs changed")
    by_result = {dots[576 + channel * 4 + 3][1]: channel for channel in range(16)}
    edge = [None] * 16
    for _, literal, result in phis:
        _require(result in by_result, "POST PHI is not the first tap's accumulator")
        edge[by_result[result]] = int(literal)
    _require(None not in edge, "POST edge bias channel coverage")
    return _words(edge)


def decode_post(text, base):
    count = len(list(_DOT.finditer(text)))
    _require(count in (1536, 5760), "POST neural shape changed")
    dots = _dots(text, count)
    copies = 1 if count == 1536 else 4
    tables = _table_map("post", base)
    primary, edge = None, None
    for copy in range(copies):
        group = dots[copy * 1408:(copy + 1) * 1408]
        decoded = {**_cnb_weights(group), **_post_biases(group)}
        branch = _edge_biases(text, group)
        if primary is None:
            primary, edge = decoded, branch
        else:
            _require(primary == decoded and edge == branch, "POST cloned CNBs disagree")
    tables.update(primary)
    # CT2D LLVM literals are recurrent-head first, filter-head second. The AMD
    # initializer stores filter first; swapping these halves is essential.
    # Channels 0..3 use tap weights +0 and bias 27264 (Gaussian/history);
    # channels 4..7 use tap weights +64 and bias 27280 (recurrent sigmoid).
    tail = dots[copies * 1408:]
    weights = _literal_dots(tail)
    _require(len(weights) == 128, "POST CT2D tail dimensions")
    reordered, bias, tap_biases = [], None, []
    for tap in range(4):
        chunk = weights[tap * 32:(tap + 1) * 32]
        reordered.extend(chunk[16:] + chunk[:16])
        roots = [tail[tap * 32 + channel * 4][3] for channel in range(8)]
        _require(all(_INT.fullmatch(value) for value in roots), "unknown POST CT2D bias")
        current = _words([int(value) for value in roots[4:] + roots[:4]])
        if bias is None:
            bias = current
        tap_biases.append(struct.unpack("<8i", current))
    tables["CT"], tables["bCT"] = _words(reordered), bias
    # Edge biases are always the unchanged AMD values, including UP/4320.
    _require(edge == _table_map("post", base)["bA"], "unknown conditional POST edge biases")
    primary_bias = struct.unpack("<16i", primary["bA"])
    edge_bias = struct.unpack("<16i", edge)
    conditional = [{"pass": "post", "layer": "A", "channel": channel,
                    "primary_bias": p, "edge_bias": e,
                    "condition": "first_3x3_tap_out_of_bounds",
                    "predicate": "not (0 <= cnb_half_x - 1 < half_width and 0 <= cnb_half_y - 1 < half_height)",
                    "half_width": "tier * 8 // 9", "half_height": "tier // 2",
                    "applies_to": "all four current/cached-coordinate CNB copies",
                    "llvm_first_tap_phi": "edge literal if first tap skipped; trained dot accumulator otherwise"}
                   for channel, (p, e) in enumerate(zip(primary_bias, edge_bias)) if p != e]
    # AMD's tail has eight shared biases, while Lossy trains separate biases
    # for each subpixel tap. Tap 0 is the explicit primary initializer value;
    # the complete remaining overrides are retained, never averaged/dropped.
    for tap in range(1, 4):
        for channel, (primary_value, conditional_value) in enumerate(zip(tap_biases[0], tap_biases[tap])):
            if primary_value != conditional_value:
                conditional.append({"pass": "post", "layer": "CT", "channel": channel,
                                    "tap": tap, "tap_xy": [tap & 1, tap >> 1],
                                    "primary_bias": primary_value, "edge_bias": conditional_value,
                                    "conditional_bias": conditional_value,
                                    "condition": "post_subpixel_tap",
                                    "predicate": f"(output.x & 1) == {tap & 1} and (output.y & 1) == {tap >> 1}",
                                    "initializer_policy": "tap-0 bias at byte 27264; apply overrides for other taps",
                                    "channel_order": "filter float4, then recurrent float4",
                                    "head": "filter" if channel < 4 else "recurrent",
                                    "head_channel": channel % 4})
    return tables, conditional


def _tier(text, stage):
    rows = {int(v) for v in re.findall(r"\b(7712|15392|30752|61472)\b", text)}
    mapping = {7712: 1080, 15392: 2160, 30752: 4320} if stage == "pass2" else {15392: 1080, 30752: 2160, 61472: 4320}
    _require(rows and max(rows) in mapping, "unknown tensor tier/row pitch")
    return mapping[max(rows)]
