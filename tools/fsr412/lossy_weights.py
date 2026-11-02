#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Static DXIL decoder for the 2026-11-01 FSR 4.1.2 Lite Lossy release.

No DLL/shader execution, GPU, NumPy, upstream Python, or GPL source is used.
The layout decoder is independently derived from DXC's LLVM listings and
checked against every table in the exact sibling's AMD initializer.

recover_primary_training(dll, dxc, base_initializer) returns (bytes, receipt).
This is an explicitly independent Metal adaptation, NOT edge-parity with the
upstream DLL. Its trained common 1080/2160 tables agree; 4320 and UP retain AMD
weights. POST's first-tap-missing PHIs still contain AMD biases upstream, and
POST CT2D biases vary by subpixel tap; the receipt preserves both conditional
overrides. The initializer's CT bias vector explicitly selects tap 0.
recover() rejects these nonuniform cases.
Only vendor-independent layout/identity metadata is tracked in this module.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import struct
import subprocess
import tempfile

DLL_SHA256 = "d9b0a1f5781c2d633a327272f6f1c8607057df7cafd0233cf38c7395c8f7dbc5"
EXACT_DLL_SHA256 = "56896078aef294ea875f4b977695701d827a9273a616d1f5c501b9963e559bd7"
COMMON_SHA256 = "4b9898da719dc2dca5dc45fd71dec5f80bdb9886a580fe3038af59efe637d55e"
ULTRA_SHA256 = "4c57292d9963e82dec86134060eef0c2b66214cd1a31a9e9bc73739b5fe2d1d7"
# Filled only after independently validating the exact sibling and both tiers.
LOSSY_INITIALIZER_SHA256 = "c77c2d1a4d119ed76d94b8c23335490cc38280db3346c72396a1e1acb0523cfc"
INITIALIZER_BYTES = 27392
INITIALIZER_OFFSETS = {"normal": 0xc04a00, "ultraperformance": 0xc24a00}
TIERS = (1080, 2160, 4320)

# (name, initializer byte offset, byte count). Includes the POST CT2D tail
# which the native external-CNB operator itself does not consume.
LAYOUT = {
    "pass1": (("A", 1152, 2304), ("bA", 3456, 64), ("B", 3584, 512),
              ("bB", 4096, 128), ("C", 4224, 512), ("bC", 4736, 64),
              ("DS", 4864, 2048), ("bDS", 6912, 128)),
    "pass2": (("A", 7040, 2304), ("bA", 9344, 64), ("B", 9472, 2048),
              ("bB", 11520, 256), ("C", 11776, 2048), ("bC", 13824, 128)),
    "pass3": (("A", 13952, 2304), ("bA", 16256, 64), ("B", 16384, 2048),
              ("bB", 18432, 256), ("C", 18688, 2048), ("bC", 20736, 128),
              ("CT", 20864, 2048), ("bCT", 22912, 64)),
    "post": (("A", 23040, 2304), ("bA", 25344, 64), ("B", 25472, 512),
             ("bB", 25984, 128), ("C", 26112, 512), ("bC", 26624, 64),
             ("CT", 26752, 512), ("bCT", 27264, 32)),
}
_ENTRY = "fsr4_model_v07_fp8_no_scale_"
_DOT = re.compile(r"^\s*%([\w.]+) = call i32 @dx\.op\.dot4AddPacked\.i32\(i32 (\d+), i32 ([^,]+), i32 ([^,]+), i32 ([^)]+)\)", re.M)
_INT = re.compile(r"-?\d+\Z")


def _require(condition, reason):
    if not condition:
        raise ValueError("Lite Lossy decoder: " + reason)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def _data(value):
    return bytes(value) if isinstance(value, (bytes, bytearray)) else Path(value).read_bytes()


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


def decode_pass1(text, base):
    """Decode all three CNB matrices; validate the unchanged dynamic DS/table biases."""
    dots = _dots(text, 1536)
    tables = _table_map("pass1", base)
    tables.update(_cnb_weights(dots[:1408]))
    _require(all(m[5].startswith("%") for m in dots[1408:]), "unexpected trained downsample operands")
    # DS and biases are not trained literals: these are real SRV initializer
    # loads, not an inference from a missing layer. Exact/known DLL identity
    # and the unchanged initializer bytes bind these accesses to AMD data.
    _require("i8 0, i32 0, i32 18, i1 false" in text, "initializer SRV binding changed")
    for offset in (3456, 4096, 4736, 4864, 6912):
        _require(re.search(r"(?:\bi32 |, )" + str(offset) + r"\b", text), "missing pass1 initializer range")
    return tables


def _literal_store_blocks(text):
    """Recover all 16-word local weight arrays without executing LLVM loops.

    Each packed array is populated at constant indices 0..15 before the
    four-channel loop consumes it through load/phi. No eager integer evaluator
    is needed; an unrelated negative shift cannot affect literal recovery.
    """
    pointers = {}
    stores = []
    for raw in text.splitlines():
        line = raw.split(";")[0].strip()
        pointer = re.fullmatch(r"%([\w.]+) = getelementptr \[16 x i32\], \[16 x i32\]\* %([\w.]+), i32 0, i32 (\d+)", line)
        if pointer:
            pointers[pointer[1]] = (pointer[2], int(pointer[3]))
        store = re.match(r"store i32 (-?\d+), i32\* %([\w.]+), align 4(?:, .*)?$", line)
        if store:
            _require(store[2] in pointers, "literal store is not a bounded weight-array element")
            array, index = pointers[store[2]]
            stores.append((array, index, int(store[1])))
    _require(len(stores) == 1600, "FNB literal store count changed")
    blocks = []
    for start in range(0, len(stores), 16):
        chunk = stores[start:start + 16]
        _require([index for _, index, _ in chunk] == list(range(16)) and
                 len({array for array, _, _ in chunk}) == 1, "FNB local-array initialization order changed")
        blocks.append((chunk[0][0], [value for _, _, value in chunk]))
    return blocks


def _grouped_matrix(words, outputs, blocks, convolution=False):
    """[output group of 4][tap/input bank][output within group][input word]."""
    count = outputs * blocks * 4
    _require(len(words) == count and outputs % 4 == 0, "grouped matrix dimensions")
    result = [None] * count
    for group in range(outputs // 4):
        for block in range(blocks):
            for channel in range(4):
                for word in range(4):
                    source = (((group * blocks + block) * 4 + channel) * 4 + word)
                    output = group * 4 + channel
                    destination = ((block * outputs + output) * 4 + word) if convolution else \
                        (output * blocks * 4 + block * 4 + word)
                    _require(result[destination] is None, "duplicate matrix element")
                    result[destination] = words[source]
    return _words(result)


def decode_pass2(text, base):
    _dots(text, 400)
    blocks = _literal_store_blocks(text)
    stage_arrays = [set(array for array, _ in blocks[start:end]) for start, end in ((0, 36), (36, 68), (68, 100))]
    _require(all(len(arrays) == 1 for arrays in stage_arrays) and
             len(set.union(*stage_arrays)) == 3, "FNB stage weight arrays changed")
    tables = _table_map("pass2", base)
    for name, start, end, outputs, banks in (("A", 0, 36, 16, 9), ("B", 36, 68, 64, 2), ("C", 68, 100, 32, 4)):
        words = [word for _, chunk in blocks[start:end] for word in chunk]
        tables[name] = _grouped_matrix(words, outputs, banks, name == "A")
    _require("i8 0, i32 0, i32 18, i1 false" in text, "initializer SRV binding changed")
    for offset in (9344, 11520, 13824):
        _require(re.search(r"(?:\bi32 |, )" + str(offset) + r"\b", text), "missing FNB bias range")
    return tables


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


def _quantizers(stage, preset, text, post_copies=1):
    counts = Counter(re.findall(r"\bashr i32 [^,]+, (\d+)", text))
    if stage == "pass1":
        expected = {"6": 16, "7": 24, "9": 32, "24": 12} if preset == "normal" else {"6": 16, "8": 56, "24": 12}
        mapping = {"A": {"round_even_right_shift": 7 if preset == "normal" else 8},
                   "B": {"round_even_right_shift": 9 if preset == "normal" else 8, "relu": True},
                   "C_residual": {"input_shift": 5, "sum_shift": 6},
                   "DS": {"lower_16_shift": 6 if preset == "normal" else 7,
                          "upper_16_shift": 7 if preset == "normal" else 8}}
    elif stage == "pass2":
        expected = {"7": 32, "8": 80, "24": 24}
        mapping = {"A": {"round_even_right_shift": 8}, "B": {"round_even_right_shift": 8, "relu": True},
                   "C_residual": {"lower_16_input_shift": 5, "lower_16_sum_shift": 6,
                                  "upper_16_input_shift": 6, "upper_16_sum_shift": 7}}
    elif stage == "pass3":
        expected = {"8": 20, "7": 4, "6": 4, "24": 15} if preset == "normal" else {"8": 16, "7": 8, "9": 4, "24": 15}
        mapping = {"A": {"round_even_right_shift": 8 if preset == "normal" else 9},
                   "B": {"round_even_right_shift": 7, "relu": True},
                   "C_residual": {"lower_16_input_shift": 6, "upper_16_input_shift": 7,
                                  "sum_shift": 6 if preset == "normal" else 7},
                   "CT_skip": {"input_shift": 8 if preset == "normal" else 7, "sum_shift": 8}}
    else:
        expected = {"24": 12 * post_copies, "8": 48 * post_copies} if preset == "normal" else \
            {"24": 12 * post_copies, "7": 16 * post_copies, "8": 32 * post_copies}
        if post_copies == 4:
            expected.update({"1": 4, "2": 2, "16": 24})  # Lossy cache metadata, not neural quantizers.
        mapping = {"A": {"round_even_right_shift": 8 if preset == "normal" else 7},
                   "B": {"round_even_right_shift": 8, "relu": True},
                   "C_residual": {"formula": "rint((input / 32 + acc / 1024) * 16)" if preset == "normal" else
                                  "rint((input / 16 + acc / 1024) * 8)",
                                  "native_input_shift": 5 if preset == "normal" else 6,
                                  "native_sum_shift": 6 if preset == "normal" else 7},
                   "CT2D_tail": {"accumulator_to_float_scale": "1/1024", "quantized_output": False,
                                  "head_order_in_initializer": ["filter_float4", "recurrent_float4"],
                                  "bias_offsets": {"filter": 27264, "recurrent": 27280},
                                  "per_tap_weight_offsets": {"filter": 0, "recurrent": 64}}}
        floats = Counter(re.findall(r"fmul fast float [^,]+, ([^ \n]+)", text))
        residual_literals = ("9.765625e-04", "3.125000e-02" if preset == "normal" else "6.250000e-02",
                             "1.600000e+01" if preset == "normal" else "8.000000e+00")
        if all(floats[literal] == 0 for literal in residual_literals) and preset == "normal" and post_copies == 1:
            expected["6"] = 16  # Exact sibling's integer residual specialization.
            mapping["C_residual"]["lowering"] = "integer nearest-even"
        else:
            for literal in residual_literals:
                _require(floats[literal] == 16 * post_copies, "POST float residual quantizer changed")
            mapping["C_residual"]["lowering"] = "float32 round-nearest-even"
    _require(dict(counts) == expected, stage + " shift/quantizer shape changed")
    # Each quantized layer uses q=(x>>shift)&1, h=x+(2^(shift-1)-1),
    # r=(h+q)>>shift, followed by INT8 clamp. The DLL is pinned in recover;
    # structural evidence and decoded mappings are retained, not guessed.
    return {"mapping": mapping, "signed_ashr_literal_histogram": dict(sorted(counts.items())),
            "rounding": "nearest-even; signed INT32 epilogues; INT8 clamp [-128,127]",
            "listing_sha256": digest(text.encode()), "source_binding": "pinned DLL/container and independently checked exact sibling"}


def _tier(text, stage):
    rows = {int(v) for v in re.findall(r"\b(7712|15392|30752|61472)\b", text)}
    mapping = {7712: 1080, 15392: 2160, 30752: 4320} if stage == "pass2" else {15392: 1080, 30752: 2160, 61472: 4320}
    _require(rows and max(rows) in mapping, "unknown tensor tier/row pitch")
    return mapping[max(rows)]


def _decode_dll(dll, dxc):
    data = _data(dll)
    identity = digest(data)
    _require(identity in (DLL_SHA256, EXACT_DLL_SHA256), "unrecognized DLL identity")
    section_offset, section_size = _section(data)
    bases = {name: data[offset:offset + INITIALIZER_BYTES] for name, offset in INITIALIZER_OFFSETS.items()}
    _require(digest(bases["normal"]) == COMMON_SHA256 and digest(bases["ultraperformance"]) == ULTRA_SHA256,
             "embedded AMD initializers changed")
    all_containers = list(dxbc_containers(data, section_offset, section_offset + section_size))
    decoded, seen = [], set()
    for offset, container in all_containers:
        sha = digest(container)
        if sha in seen:
            continue
        seen.add(sha)
        if not any((_ENTRY + entry).encode() in container for entry in ("pass1", "pass2", "postpass")):
            continue
        text = disassemble(dxc, container)
        entry = re.search(r"^define void @(\w+)\(\)", text, re.M)
        _require(entry is not None, "missing DXIL entry")
        stage = {"pass1": "pass1", "pass2": "pass2", "postpass": "post"}.get(entry[1].removeprefix(_ENTRY))
        if stage is None:
            continue
        # Classification uses the unchanged A matrix, not biased filename/order
        # guesses. Both preset A matrices are independently verified AMD data.
        if stage == "pass1":
            first = _cnb_weights(_dots(text, 1536)[:1408])["A"]
        elif stage == "pass2":
            blocks = _literal_store_blocks(text)
            first = _grouped_matrix([v for _, chunk in blocks[:36] for v in chunk], 16, 9, True)
        else:
            dots = _dots(text, len(list(_DOT.finditer(text))))
            first = _cnb_weights(dots[:1408])["A"]
        presets = [name for name, base in bases.items() if _table_map(stage, base)["A"] == first]
        _require(len(presets) == 1, "unknown or ambiguous trained 3x3/preset")
        preset = presets[0]
        if stage == "pass1":
            tables, conditional = decode_pass1(text, bases[preset]), []
        elif stage == "pass2":
            tables, conditional = decode_pass2(text, bases[preset]), []
        else:
            tables, conditional = decode_post(text, bases[preset])
        copies = 4 if stage == "post" and len(list(_DOT.finditer(text))) == 5760 else 1
        evidence = _quantizers(stage, preset, text, copies)
        decoded.append({"stage": stage, "preset": preset, "tier": _tier(text, stage), "tables": tables,
                        "conditional_biases": conditional, "quantizer_evidence": evidence,
                        "container_sha256": sha, "container_offset": offset, "container_bytes": len(container)})
    # pass3 is genuinely loaded from the unchanged DLL initializer, not an
    # absent trained layer silently replaced. Validate its real table references.
    for offset, container in dxbc_containers(data, 0, section_offset):
        if (_ENTRY + "pass3").encode() not in container:
            continue
        sha = digest(container)
        if sha in seen:
            continue
        seen.add(sha)
        text = disassemble(dxc, container)
        if not re.search(r"^define void @" + _ENTRY + r"pass3\(\)", text, re.M):
            continue
        if not list(_DOT.finditer(text)):
            continue  # The unrelated 32-thread native/MPP graph has the same entry name.
        _dots(text, 368)
        _require(not re.search(r"dot4AddPacked\.i32[^\n]+, i32 -?\d+\)", text), "unexpected trained pass3 literals")
        for name, table_offset, _ in LAYOUT["pass3"]:
            _require(re.search(r"(?:\bi32 |, )" + str(table_offset) + r"\b", text), "missing pass3 initializer range " + name)
        counts = Counter(re.findall(r"\bashr i32 [^,]+, (\d+)", text))
        preset = "normal" if counts.get("6") == 4 else "ultraperformance"
        evidence = _quantizers("pass3", preset, text)
        decoded.append({"stage": "pass3", "preset": preset, "tier": _tier(text, "pass3"),
                        "tables": _table_map("pass3", bases[preset]), "conditional_biases": [],
                        "quantizer_evidence": evidence, "container_sha256": sha,
                        "container_offset": offset, "container_bytes": len(container)})
    return data, identity, bases, decoded


def recover_primary_training(dll, dxc, base_initializer):
    """Recover primary trained normal 1080/2160 tables, with explicit edge/tier limits.

    Exact sibling and unchanged UP inputs return byte-identical AMD data and
    mark trained=False. No files are written except temporary DXC input/output.
    """
    base = _data(base_initializer)
    _require(len(base) == INITIALIZER_BYTES and digest(base) in (COMMON_SHA256, ULTRA_SHA256), "unknown base initializer")
    preset = "normal" if digest(base) == COMMON_SHA256 else "ultraperformance"
    _, identity, bases, decoded = _decode_dll(dll, dxc)
    _require(base == bases[preset], "base initializer does not match the supplied DLL")
    groups = {}
    for item in decoded:
        groups.setdefault((item["preset"], item["stage"], item["tier"]), []).append(item)
    expected = {(p, s, t) for p in ("normal", "ultraperformance") for s in LAYOUT for t in TIERS}
    _require(set(groups) == expected, "neural stage/preset/tier coverage incomplete")
    for key, items in groups.items():
        reference = items[0]
        _require(all(item["tables"] == reference["tables"] and item["conditional_biases"] == reference["conditional_biases"] for item in items),
                 "colorspace/exposure variants disagree: " + str(key))
    # The exact sibling is the independent layout oracle. Every recovered
    # matrix, bias, DS, pass3 CT and POST CT tail must match its AMD bytes.
    for (p, stage, tier), items in groups.items():
        should_be_base = identity == EXACT_DLL_SHA256 or p == "ultraperformance" or tier == 4320
        if should_be_base:
            _require(items[0]["tables"] == _table_map(stage, bases[p]) and not items[0]["conditional_biases"],
                     "unexpected weights in an exact/UP/4320 variant")
    result, changes, conditional = bytearray(base), [], []
    trained = identity == DLL_SHA256 and preset == "normal"
    for stage in LAYOUT:
        first = groups[(preset, stage, 1080)][0]
        second = groups[(preset, stage, 2160)][0]
        _require(first["tables"] == second["tables"] and first["conditional_biases"] == second["conditional_biases"],
                 "independent 1080/2160 trained tables disagree: " + stage)
        for name, offset, size in LAYOUT[stage]:
            value = first["tables"][name]
            _require(len(value) == size, "decoded initializer range size changed")
            result[offset:offset + size] = value
            changed = sum(a != b for a, b in zip(base[offset:offset + size], value))
            changes.append({"pass": stage, "table": name, "offset": offset, "bytes": size,
                            "sha256": digest(value), "changed_bytes": changed,
                            "provenance": "DXIL literal operands" if stage == "post" or
                                          stage in ("pass1", "pass2") and name in ("A", "B", "C") else
                                          "verified unchanged initializer SRV loads"})
        conditional.extend(first["conditional_biases"])
    result = bytes(result)
    _require(not trained or result != base, "Lossy profile contains no recovered training")
    if trained and LOSSY_INITIALIZER_SHA256 is not None:
        _require(digest(result) == LOSSY_INITIALIZER_SHA256, "recovered initializer identity changed")
    name = "initializers-lite-lossy.bin" if trained else ("initializers-lite.bin" if preset == "normal" else "initializers-lite-ultraperformance.bin")
    receipts = []
    for item in sorted(decoded, key=lambda x: (x["stage"], x["preset"], x["tier"], x["container_sha256"])):
        receipts.append({key: value for key, value in item.items() if key not in ("tables", "conditional_biases")})
    ct_taps = [list(struct.unpack("<8i", groups[(preset, "post", 1080)][0]["tables"]["bCT"])) for _ in range(4)]
    for override in conditional:
        if override["layer"] == "CT":
            ct_taps[override["tap"]][override["channel"]] = override["conditional_bias"]
    receipt = {"schema": "fsr412_lite_lossy_import_v1", "source_dll_sha256": identity,
               "exact_reference_dll_sha256": EXACT_DLL_SHA256,
               "dxc_sha256": digest(Path(dxc).read_bytes()),
               "reference_verification": "decoder layout independently tested against the pinned Exact sibling; this invocation binds DLL and embedded initializer identities",
               "initializers": {name: digest(result)}, "initializer_bytes": len(result),
               "base_initializer_sha256": digest(base), "preset": preset, "trained": trained,
               "supported_trained_tiers": [1080, 2160] if trained else [],
               "upstream_base_only_profiles": {"4320": "AMD initializer, not trained", "ultraperformance": "AMD initializer, not trained"},
               "conditional_biases": conditional, "post_ct_biases_by_tap": ct_taps,
               "post_ct_head_order": ["filter_float4", "recurrent_float4"],
               "post_ct_channel_order": ["filter.x", "filter.y", "filter.z", "filter.w",
                                         "recurrent.x", "recurrent.y", "recurrent.z", "recurrent.w"],
               "post_ct_bias_byte_offsets": {"filter": 27264, "recurrent": 27280},
               "post_ct_weight_layout": {"offset": 26752, "tap_stride_bytes": 128,
                                         "filter_offset_in_tap": 0, "recurrent_offset_in_tap": 64},
               "post_ct_primary_tap": 0, "shader_edge_parity": not conditional,
               "adaptation": "primary-training independent Metal adaptation; not upstream cache/temporal/runtime parity",
               "unused_bytes_preserved": True, "tables": changes, "containers": receipts,
               "limitations": ["4320 and UP must use base initializers", "POST missing-first-tap edge biases differ when conditional_biases is nonempty",
                               "POST CT bias overrides are selected by subpixel tap; the initializer stores tap-0 biases only",
                               "Only static decoding and CPU/source proofs; no DLL, shader, or GPU execution"]}
    return result, receipt


def recover(dll, dxc, base_initializer):
    """Strict table recovery. Conditional upstream biases require explicit adaptation API."""
    result, receipt = recover_primary_training(dll, dxc, base_initializer)
    _require(not receipt["conditional_biases"], "conditional POST biases cannot be represented by one initializer; use recover_primary_training explicitly")
    return result, receipt


def _write_external(path, data):
    """Atomic artifact publication; never expose partial weights/receipts to the runtime."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and path.read_bytes() == data:
        return
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(prefix="." + path.name + ".", dir=path.parent, delete=False) as output:
            temporary = Path(output.name)
            output.write(data)
        temporary.replace(path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dll", type=Path)
    parser.add_argument("--dxc", type=Path, default=Path(__file__).resolve().parents[1] / "bin/dxc")
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--primary-training", action="store_true", help="explicit independent Metal adaptation, not edge parity")
    args = parser.parse_args()
    function = recover_primary_training if args.primary_training else recover
    initializer, receipt = function(args.dll, args.dxc, args.base)
    if args.out:
        target = args.out.resolve()
        allowed_build = Path(__file__).resolve().parents[2] / "build/fsr412_lossy"
        _require(target.is_relative_to(allowed_build) or target.is_relative_to(Path("/tmp").resolve()) or
                 target.is_relative_to(Path(tempfile.gettempdir()).resolve()),
                 "weights must remain external in build/fsr412_lossy or temp")
        _write_external(target, initializer)
    if args.receipt:
        _write_external(args.receipt, (json.dumps(receipt, indent=2) + "\n").encode())
    print(json.dumps({"initializers": receipt["initializers"], "trained": receipt["trained"],
                      "conditional_bias_channels": len(receipt["conditional_biases"]), "containers": len(receipt["containers"])}))


if __name__ == "__main__":
    main()
