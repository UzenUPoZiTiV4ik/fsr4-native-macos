#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Pack the imported FSR 4.1.1 CNB model for native Metal matrix kernels.

Reads the MIT reimplementation's actual weight tables and the verified AMD
4.1.1 initializer. It does not use the older v07 weights or execute upstream
scripts. The twelve neural passes and POST's CNB run natively; PRE, the
thirteen border epilogues and POST's floating filter retain the imported HLSL
path. Rejects an unexpected source shape instead of guessing.
"""
import argparse
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import struct
import sys
import tempfile

INITIALIZER_SHA256 = "a54e552ff69a3f7199861417f7ae461d1f67f636ab532c6ec9d8ea5d2963735a"
UP_INITIALIZER_SHA256 = "3c42fc6eabfa9b5b1e489df0e9b2f58e3f9fa22048c584f6839bb27d32a0640e"
U32_MAX = 0xFFFFFFFF
ABI_BYTES = 212
PARAMETER_STRIDE = 256
INPUT_SLOTS = {1: 0, **{i: i for i in range(2, 13)}}
LOSSY_ROUNDING_PASSES = (1, 2, 4, 5, 7, 8, 9, 10, 12)
LOSSY_FOLD_WORDS = {1: 4, 5: 18, 10: 18, 12: 20}
PUBLICATION_LOCK = ".fsr411_native.lock"


def _write_changed_atomic(path, data):
    """Keep a cached file's inode/mtime; publish changed bytes in one rename."""
    try:
        if path.read_bytes() == data:
            return
    except FileNotFoundError:
        pass
    descriptor, temporary = tempfile.mkstemp(prefix="." + path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _publish_native(output_dir, manifest, weights, parameters):
    """Serialize writers to one model directory and publish its manifest last.

    Each data file is replaced atomically, so concurrent readers never see
    truncation or a partial write. Metadata describes the exact emitted
    bytes, and unchanged cached models require no file replacements.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = dict(manifest, weights_sha256=hashlib.sha256(weights).hexdigest(), weights_bytes=len(weights),
                    params_sha256=hashlib.sha256(parameters).hexdigest(), params_bytes=len(parameters))
    manifest_path = output_dir / "fsr411_native_manifest.json"
    with (output_dir / PUBLICATION_LOCK).open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        _write_changed_atomic(output_dir / "fsr411_native_weights.bin", weights)
        _write_changed_atomic(output_dir / "fsr411_native_params.bin", parameters)
        _write_changed_atomic(manifest_path, (json.dumps(metadata, indent=2) + "\n").encode())
    return manifest_path


def require(pattern, text, description, flags=0):
    match = re.search(pattern, text, flags)
    if not match:
        raise ValueError("unexpected 4.1.1 source: " + description)
    return match


def table(text, name, rows):
    body = require(r"static const uint " + name + r"\[" + str(rows) + r"\]\[16\]\s*=\s*\{(.*?)\};",
                   text, name, re.S)[1]
    words = [int(x, 16) for x in re.findall(r"0x([0-9a-fA-F]+)u", body)]
    if len(words) != rows * 16:
        raise ValueError(name + ": wrong packed word count")
    return [words[i:i + 16] for i in range(0, len(words), 16)]


def byte(word, channel):
    bits = (word >> (8 * channel)) & 255
    return bits if bits < 128 else bits - 256


def dyadic_exponent(value):
    fraction, exponent = math.frexp(value)
    if fraction != 0.5:
        raise ValueError("native integer tail requires a positive power of two")
    return exponent - 1


def dot_bounds(weights, bias, input_min=-128, input_max=127):
    lo = bias + sum(w * (input_min if w >= 0 else input_max) for w in weights)
    hi = bias + sum(w * (input_max if w >= 0 else input_min) for w in weights)
    return lo, hi


def geometry(tier):
    levels = []
    for level in range(3):
        width, height = (tier * 8 // 9) >> level, (tier // 2) >> level
        row = (width + 2) * 16
        levels.append((width, height, row, row * (height + 2)))
    return levels


def tensor_at(tier, level, base, channels):
    width, height, row, bank = geometry(tier)[level]
    return [base + row + 16, 16, row, width, height, channels, bank if channels > 16 else 0]


def source_identity(path):
    original = path.read_text()
    return original, {"source": path.name, "source_sha256": hashlib.sha256(original.encode()).hexdigest()}


def int_table(text, name, count):
    body = require(r"static const int " + name + r"\[" + str(count) + r"\]\s*=\s*\{(.*?)\};",
                   text, name, re.S)[1]
    values = [int(x.strip()) for x in body.split(",") if x.strip()]
    if len(values) != count or any(x < -(1 << 31) or x >= (1 << 31) for x in values):
        raise ValueError(name + ": wrong int32 literal table")
    return values


def parse_post_cnb(path, tables_path, tier):
    original, result = source_identity(path)
    text = re.sub(r"//[^\n]*|/\*.*?\*/", "", original, flags=re.S)
    tables_text = tables_path.read_text()
    w0, w1, w2 = (int_table(tables_text, name, count) for name, count in
                   (("DW_W", 576), ("PW1_W", 128), ("PW2_W", 128)))
    biases = [int_table(tables_text, name, count) for name, count in
              (("DW_S", 16), ("PW1_S", 32), ("PW2_S", 16))]
    conv_function = require(r"conv_result\[f2 / 4\] = \(uint\)pack_clamp_s8\(int4\((\w+)\(acc\[f2\]\)", text, "POST conv requant function")[1]
    expand_function = require(r"relu_result\[i2 / 4\] = \(uint\)pack_clamp_s8\(int4\((\w+)\(r_out\[i2\]\)", text, "POST expansion requant function")[1]
    conv = require(r"int " + conv_function + r"\(int v\)\s*\{ return \(v \+ (\d+) \+ \(\(v >> (\d+)\) & 1\)\) >> \2; \}", text, "POST conv rounding")
    expand = require(r"int " + expand_function + r"\(int v\)\s*\{ v = max\(v, 0\); return \(v \+ (\d+) \+ \(\(v >> (\d+)\) & 1\)\) >> \2; \}", text, "POST expansion rounding")
    for match in (conv, expand):
        if not 0 < int(match[2]) < 31 or int(match[1]) != (1 << (int(match[2]) - 1)) - 1:
            raise ValueError("POST rounding is not a supported nearest-even shift")
    shifts = [int(conv[2]), int(expand[2])]
    require(r"float4 a = float4\([^\n]+\) \* 9\.765625e-04f;", text, "POST accumulator scale")
    require(r"float4 b4 = float4\(unpack_s8s32\(rd\[i4 / 4\]\)\) \* 6\.25e-02f;", text, "POST residual scale")
    require(r"int4 q = int4\(round\(\(b4 \+ a\) \* 16\.0f\)\);\s*packedVss\[i4 / 4\] = \(uint\)pack_clamp_s8\(q\);", text, "POST tail without int16 narrowing")
    width, height, row, _ = geometry(tier)[0]
    require(r"uint4 rd = ScratchBuffer.Load4\(\(uint\)\(\(iy \+ 1\) \* " + str(row) + r" \+ \(ix \+ 1\) \* 16\)\);", text, "POST input pitch")
    c0 = [byte(w0[tap * 64 + oc * 4 + ic // 4], ic % 4)
          for tap in range(9) for ic in range(16) for oc in range(16)]
    c1 = [byte(w1[oc * 4 + ic // 4], ic % 4) for ic in range(16) for oc in range(32)]
    c2 = [byte(w2[oc * 8 + ic // 4], ic % 4) for ic in range(32) for oc in range(16)]
    bounds = []
    for stage, (matrix, channels, count) in enumerate(zip((c0, c1, c2), (16, 32, 16), (144, 16, 32))):
        channel_bounds = [dot_bounds([matrix[k * channels + oc] for k in range(count)], biases[stage][oc],
                                     0 if stage == 2 else -128, 127) for oc in range(channels)]
        low, high = min(x[0] for x in channel_bounds), max(x[1] for x in channel_bounds)
        if stage == 2:
            low -= 128 << 6
            high += 127 << 6
        if max(abs(low), abs(high)) >= (1 << 24):
            raise ValueError("POST accumulator exceeds exact fp32 integer range")
        bounds.append([low, high])
    result.update(kind="cnb", kernel="fsr411_cnb_coop_chain_32x1", tile=[32, 1], threads=32,
                  input=tensor_at(tier, 0, 0, 16), output=[0, 16, width * 16, width, height, 16, 0],
                  output_slot=13, output_offset_base="scratch_append", scratch_append_alignment=256,
                  scratch_append_bytes=width * height * 16,
                  weights=[c0, c1, c2], bias=biases, post_literal_weights=(w0, w1, w2),
                  initializer_bias_offsets=[], accumulator_bounds=bounds,
                  tables_source=tables_path.name, tables_sha256=hashlib.sha256(tables_text.encode()).hexdigest(),
                  q=[[0, 0, shifts[0], 0], [0, 0, shifts[1], 1], [0, 6, 6, 0], [0, 6, 6, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    return result


def parse_cnb(path, initializer):
    original = path.read_text()
    text = re.sub(r"//[^\n]*|/\*.*?\*/", "", original, flags=re.S)
    w1, w2, w3 = table(text, "W1", 36), table(text, "W2", 8), table(text, "W3", 8)
    offsets = require(r"static const uint2 OFF\[9\]\s*=\s*\{(.*?)\};", text, "3x3 offsets", re.S)[1]
    taps = [tuple(map(int, m)) for m in re.findall(r"\{\s*(\d+),\s*(\d+)\s*\}", offsets)]
    if taps != [(y, x * 16) for y in range(3) for x in range(3)]:
        raise ValueError("unrecognized 3x3 tap ordering")
    bias_offsets = [int(x) for x in re.findall(r"t18_initializer.Load4\((\d+)u\s*\+", text)]
    if len(bias_offsets) != 3:
        raise ValueError("expected three int32 bias arrays")
    bias = [list(struct.unpack_from("<" + str(n) + "i", initializer, off))
            for off, n in zip(bias_offsets, (16, 32, 16))]
    r0 = require(r"requant_shift\(A8\[f \+ j\],\s*(\d+)u,\s*(\d+)\)", text, "conv requant")
    r1 = require(r"requant_shift\(max\(A10\[f2 \+ j2\], 0\),\s*(\d+)u,\s*(\d+)\)", text, "ReLU requant")
    shifts = [int(r0[1]), int(r1[1])]
    if any(int(m[2]) != (1 << (int(m[1]) - 1)) - 1 for m in (r0, r1)):
        raise ValueError("unexpected round-even offset")
    residual_scale = float(require(r"const float t1 = float\(cbyte\) \* ([\d.]+)f", text, "residual scale")[1])
    acc_scale = float(require(r"const float t2 = float\(A12\[4 \* d \+ j3\]\) \* ([\d.]+)f", text, "accumulator scale")[1])
    out_scale = float(require(r"const float r = round\(\(t1 \+ t2\) \* ([\d.]+)f\)", text, "output scale")[1])
    require(r"lanes\[j3\] = int16_wrap\(int\(r\)\)", text, "tail int16 narrowing")
    ae = dyadic_exponent(acc_scale * out_scale)
    rexp = dyadic_exponent(residual_scale * out_scale)
    right = max(0, -ae, -rexp)
    tail = [ae + right, rexp + right, right, 2]
    if max(tail[:3]) > 30:
        raise ValueError("integer tail shift exceeds ABI")

    # Retain both representations for independent CPU/GPU checks: literal
    # output-major words and the native input-major matrix blob.
    c0 = [byte(w1[(oc // 4) * 9 + tap][4 * (oc % 4) + ic // 4], ic % 4)
          for tap in range(9) for ic in range(16) for oc in range(16)]
    c1 = [byte(w2[oc // 4][4 * (oc % 4) + ic // 4], ic % 4)
          for ic in range(16) for oc in range(32)]
    c2 = [byte(w3[(oc // 4) * 2 + ic // 16][4 * (oc % 4) + (ic % 16) // 4], ic % 4)
          for ic in range(32) for oc in range(16)]
    bounds = []
    for stage, (matrix, cout, count, shift) in enumerate(zip((c0, c1, c2), (16, 32, 16), (144, 16, 32), (*shifts, right))):
        channel_bounds = [dot_bounds([matrix[k * cout + oc] for k in range(count)], bias[stage][oc], 0 if stage == 2 else -128, 127)
                          for oc in range(cout)]
        low = min(b[0] for b in channel_bounds)
        high = max(b[1] for b in channel_bounds)
        if stage == 2:
            low = low * (1 << tail[0]) - 128 * (1 << tail[1])
            high = high * (1 << tail[0]) + 127 * (1 << tail[1])
            if max(abs(low), abs(high)) >= (1 << 24):
                raise ValueError("float tail cannot be replaced exactly: fp32 integer range")
        if low < -(1 << 31) or high + ((1 << shift) // 2) >= (1 << 31):
            raise ValueError("integer requant may overflow its model range")
        bounds.append([low, high])

    input_match = require(r"const uint centerAddr = (\d+)u \+ (\d+)u \* \(ty \+ 1u\) \+ 16u \* tx \+ 16u", text, "input plane")
    output_match = require(r"const uint outRow = (?:(\d+)u \+ )?(\d+)u \* \(ty \+ 1u\) \+ 16u \* \(tx \+ 1u\)", text, "output plane")
    limits = require(r"if \(tx > (\d+)u \|\| ty > (\d+)u\)", text, "static limits")
    slot = int(require(r"const bool inRange = tx < ml2c_cb\[(\d+)\].x && ty < ml2c_cb\[\1\].y", text, "runtime output size")[1])
    width, height = int(limits[1]) + 1, int(limits[2]) + 1
    source_base, source_row = map(int, input_match.groups())
    dest_base, dest_row = int(output_match[1] or 0), int(output_match[2])
    if source_row != dest_row or source_row != (width + 2) * 16:
        raise ValueError("unexpected physical CNB row pitch")
    tensor = lambda base: [base + source_row + 16, 16, source_row, width, height, 16, 0]
    return {"source": path.name, "source_sha256": hashlib.sha256(original.encode()).hexdigest(),
            "kernel": "fsr411_cnb_coop_chain_32x1", "tile": [32, 1], "threads": 32, "kind": "cnb",
            "input": tensor(source_base), "output": tensor(dest_base), "output_slot": slot,
            "literal_weights": (w1, w2, w3), "weights": (c0, c1, c2), "bias": bias,
            "initializer_bias_offsets": bias_offsets, "accumulator_bounds": bounds,
            "q": [[0, 0, shifts[0], 0], [0, 0, shifts[1], 1], tail, tail, [0, 0, 0, 0], [0, 0, 0, 0]]}


def parse_ds(path, initializer, tier, logical, ultraperformance=False):
    original, result = source_identity(path)
    text = re.sub(r"//[^\n]*|/\*.*?\*/", "", original, flags=re.S)
    cin, cout, wb, bb, shift, level = (16, 32, 8576, 10624, 7, 1) if logical == 3 else (32, 64, 24576, 32768, 8, 2)
    require(r"t18_initializer.Load4\(" + str(bb) + r"u \+ 16u \* b\)", text, "downsample bias")
    require(r"\b" + str(wb) + r"u \+ " + str(cin * cout) + r"u \*", text, "downsample weights")
    upper_shift = shift
    if ultraperformance and logical == 3:
        scales = require(r"round\(float\(slot\[16 \* q \+ 4 \* d \+ j3\]\) \* \(\(q == 0\) \? ([\deE.+-]+)f : ([\deE.+-]+)f\)\)", text, "Ultra Performance split downsample scale")
        if [dyadic_exponent(float(x)) for x in scales.groups()] != [-7, -8]:
            raise ValueError("unexpected Ultra Performance downsample scale")
        upper_shift = 8
    else:
        scale = float(require(r"round\(float\(slot\[[^\n]+?\]\) \* ([\deE.+-]+)f\)", text, "downsample scale")[1])
        if dyadic_exponent(scale) != -shift:
            raise ValueError("unexpected downsample scale")
    require(r"lanes\[j3\] = int16_wrap\(int\(r\)\)", text, "downsample int16 narrowing")
    matrix = [byte(initializer[wb + tap * cin * cout + oc * cin + ic], 0)
              for tap in range(4) for ic in range(cin) for oc in range(cout)]
    bias = list(struct.unpack_from("<" + str(cout) + "i", initializer, bb))
    bounds = [dot_bounds([matrix[k * cout + oc] for k in range(cin * 4)], bias[oc]) for oc in range(cout)]
    lo, hi = min(x[0] for x in bounds), max(x[1] for x in bounds)
    if max(abs(lo), abs(hi)) >= (1 << 24):
        raise ValueError("downsample sum exceeds exact fp32 integer domain")
    g = geometry(tier)
    source_base = 0 if logical == 3 else g[0][3]
    output_base = g[0][3] if logical == 3 else g[0][3] + 2 * g[1][3]
    output = tensor_at(tier, level, output_base, cout)
    require(r"const uint addr = " + str(output_base) + r"u \+ " + str(output[6]) + r"u \*", text, "downsample output banks")
    q = [[0, 0, 0, 0] for _ in range(6)]
    q[4], q[5] = [0, 0, shift, 2], [0, 0, upper_shift, 2]
    result.update(kind="ds", kernel="fsr411_ds" + str(cin), tile=[64 if cin == 16 else 32, 1],
                  input=tensor_at(tier, level - 1, source_base, cin), output=output, output_slot=logical + 1,
                  weights=[matrix], bias=[bias], initializer_bias_offsets=[bb],
                  accumulator_bounds=[[lo, hi]], q=q)
    return result


# These are operator contracts read from reimpl_pass4/5/7/8/9/10/11.hlsl,
# not weights from the old model. Embedded weights are decoded from the HLSL;
# runtime weight bytes are taken from the pinned 4.1.1 initializer.
FNB_CONTRACTS = {
    4: dict(nf=32, shifts=(7, 8), bias=(13056, 15232, 17536),
            tail=((0, 8, 8, 2), (0, 8, 9, 2)), floats=(0.015625, 0.00006103515625)),
    5: dict(nf=32, shifts=(8, 8), bias=(19968, 22144, 24448),
            tail=((0, 7, 8, 2), (0, 8, 8, 2)), floats=(0.015625, 0.03125, 0.0001220703125)),
    7: dict(nf=64, shifts=(7, 7), bias=(37632, 45952, 54656), runtime=(33024, 37760, 46464),
            tail=((0, 7, 7, 2), (0, 7, 8, 2)), floats=(0.03125, 0.000244140625)),
    8: dict(nf=64, shifts=(9, 6), bias=(59520, 67840, 76544), runtime=(54912, 59648, 68352),
            tail=((0, 7, 8, 2), (0, 8, 8, 2)), floats=(0.03125, 0.0625, 0.000244140625)),
    9: dict(nf=64, shifts=(8, 7), bias=(81408, 89728, 98432, 106880), runtime=(76800, 81536, 90240, 98688),
            tail=((0, 7, 7, 2), (0, 7, 7, 2)), ct=(0, 8, 9, 2), floats=(0.0625, 0.00048828125, 0.03125, 0.0001220703125)),
    10: dict(nf=32, shifts=(8, 7), bias=(109312, 111488, 113792),
             tail=((0, 7, 7, 2), (0, 7, 7, 2)), floats=(0.0625, 0.00048828125)),
    11: dict(nf=32, shifts=(8, 8), bias=(116224, 118400, 120704, 122880), runtime=(113920, 116352, 118656, 120832),
             tail=((0, 8, 9, 2), (0, 8, 9, 2)), ct=(0, 7, 7, 2), floats=(0.0625, 0.000244140625, 0.03125)),
}


def parse_fnb(path, initializer, tier, logical, ultraperformance=False):
    original, result = source_identity(path)
    text = re.sub(r"//[^\n]*|/\*.*?\*/", "", original, flags=re.S)
    spec = dict(FNB_CONTRACTS[logical])
    if ultraperformance:
        # Independently validated against the changed provider DXIL pairs in
        # tools/fsr411/native_ultraperformance_contracts.json. All other FNB
        # arithmetic is shared, but its initializer bytes still differ.
        if logical == 4:
            spec.update(shifts=(8, 7), tail=((0, 8, 8, 2), (0, 9, 9, 2)),
                        floats=(0.015625, 0.03125, 0.00006103515625))
        elif logical == 10:
            spec.update(shifts=(7, 8), tail=((0, 8, 8, 2), (0, 8, 8, 2)), floats=(0.0625, 0.000244140625))
        elif logical == 11:
            spec.update(tail=((0, 8, 8, 2), (0, 8, 8, 2)), ct=(0, 8, 8, 2),
                        floats=(0.0625, 0.000244140625, 0.03125, 0.0001220703125))
    nf = spec["nf"]
    counts = (nf // 2, 2 * nf, nf) + ((nf // 2,) if "ct" in spec else ())
    biases = [list(struct.unpack_from("<" + str(count) + "i", initializer, off)) for off, count in zip(spec["bias"], counts)]
    for off in spec["bias"] + spec.get("runtime", ()):
        require(r"\b" + str(off) + r"u\b", text, "FNB initializer offset " + str(off))
    for shift in spec["shifts"]:
        require(r",\s*" + str(shift) + r"u,\s*" + str((1 << (shift - 1)) - 1) + r"\)", text, "FNB requant shift")
    actual_floats = {float(x) for x in re.findall(r"\b(\d+\.\d+(?:[eE][+-]?\d+)?)f\b", text)}
    if not set(spec["floats"]).issubset(actual_floats):
        raise ValueError("FNB floating tail contract changed")
    require(r"int16_wrap\(", text, "FNB int16 narrowing")
    if "runtime" not in spec:
        w0, w1, w2 = table(text, "W0", 36), table(text, "W1", 32), table(text, "W2", 32)
        c0 = [byte(w0[(oc // 4) * 9 + tap][4 * (oc % 4) + ic // 4], ic % 4)
              for tap in range(9) for ic in range(16) for oc in range(16)]
        c1 = [byte(w1[(oc // 4) * 2 + ic // 16][4 * (oc % 4) + (ic % 16) // 4], ic % 4)
              for ic in range(32) for oc in range(64)]
        c2 = [byte(w2[(oc // 4) * 4 + ic // 16][4 * (oc % 4) + (ic % 16) // 4], ic % 4)
              for ic in range(64) for oc in range(32)]
        result["literal_weights"] = (w0, w1, w2)
    else:
        wb = spec["runtime"]
        c0 = [byte(initializer[wb[0] + tap * (nf // 2) * 16 + (bank * 16 + oc) * 16 + ic], 0)
              for bank in range(nf // 32) for tap in range(9) for ic in range(16) for oc in range(16)]
        c1 = [byte(initializer[wb[1] + oc * nf + ic], 0) for ic in range(nf) for oc in range(2 * nf)]
        c2 = [byte(initializer[wb[2] + oc * 2 * nf + ic], 0) for ic in range(2 * nf) for oc in range(nf)]
    matrices = [c0, c1, c2]
    q = [[0, 0, spec["shifts"][0], 0], [0, 0, spec["shifts"][1], 1],
         list(spec["tail"][0]), list(spec["tail"][1]), [0, 0, 0, 0], [0, 0, 0, 0]]
    if "ct" in spec:
        out = nf // 2
        matrices.append([byte(initializer[spec["runtime"][3] + tap * out * nf + oc * nf + ic], 0)
                         for ic in range(nf) for tap in range(4) for oc in range(out)])
        q[4] = q[5] = list(spec["ct"])

    bounds = []
    for stage in range(len(matrices)):
        channel_bounds = []
        for packed_oc in range(counts[stage] * (4 if stage == 3 else 1)):
            oc = packed_oc % counts[stage]
            if stage == 0:
                values = [c0[(oc // 16) * 9 * 16 * 16 + k * 16 + oc % 16] for k in range(144)]
            elif stage == 3:
                values = [matrices[3][ic * (4 * counts[stage]) + packed_oc] for ic in range(nf)]
            else:
                values = [matrices[stage][ic * counts[stage] + oc] for ic in range(nf if stage == 1 else 2 * nf)]
            low, high = dot_bounds(values, biases[stage][oc], 0 if stage == 2 else -128, 127)
            if stage >= 2:
                eq = q[(2 if oc < nf // 2 else 3) if stage == 2 else 4]
                low = (low << eq[0]) - (128 << eq[1])
                high = (high << eq[0]) + (127 << eq[1])
                if max(abs(low), abs(high)) >= (1 << 24):
                    raise ValueError("FNB/CT tail exceeds exact fp32 integer range")
            if min(low, high) < -(1 << 31) or max(low, high) >= (1 << 31) - 32768:
                raise ValueError("FNB integer accumulator may overflow")
            channel_bounds.append((low, high))
        bounds.append([min(x[0] for x in channel_bounds), max(x[1] for x in channel_bounds)])

    g = geometry(tier)
    p0, p1, p2 = (level[3] for level in g)
    quarter0, quarter1 = p0 + 2 * p1, p0 + 2 * p1 + 4 * p2
    inbase, outbase, level = {
        4: (p0, quarter0, 1), 5: (quarter0, p0, 1),
        7: (quarter0, quarter1, 2), 8: (quarter1, quarter0, 2),
        9: (quarter0, quarter1, 2), 10: (quarter1, p0, 1), 11: (p0, quarter0, 1),
    }[logical]
    output_level = level - 1 if "ct" in spec else level
    output_channels = nf // 2 if "ct" in spec else nf
    # Record every physical plane contract explicitly. Border passes remain in
    # the imported graph; these kernels write only logical tensor interiors.
    kernel = ("fsr411_ct" if "ct" in spec else "fsr411_fnb") + str(nf)
    # NF32 blocks run the register-resident chain on one SIMD32 group; it
    # falls back to the 16x1 SG1 kernel if the cooperative layout differs.
    # NF64 keeps its measured 16x1/four-group schedule. Dedicated entry names
    # preserve the shapes used by older cached manifests.
    if nf == 64:
        kernel += "_16x1"
    else:
        kernel += "_coop_chain_16x1"
    result.update(kind="ct" if "ct" in spec else "fnb", kernel=kernel,
                  tile=[16, 1], threads=32 if nf == 32 else 128,
                  input=tensor_at(tier, level, inbase, nf),
                  output=tensor_at(tier, output_level, outbase, output_channels), output_slot=logical + 1,
                  weights=matrices, bias=biases, initializer_bias_offsets=list(spec["bias"]), q=q,
                  accumulator_bounds=bounds)
    if "ct" in spec:
        result["skip"] = tensor_at(tier, output_level, p0 if logical == 9 else 0, output_channels)
        result["skip_slot"] = 6 if logical == 9 else 3
    return result


def encode_model(source_dir, initializer_path, tier, output_dir, preset="normal", ultraperformance_dir=None, profile="exact"):
    if preset not in ("normal", "ultraperformance"):
        raise ValueError("expected normal or ultraperformance preset")
    if profile not in ("exact", "lossy"):
        raise ValueError("expected exact or lossy profile")
    ultra = preset == "ultraperformance"
    expected_hash = UP_INITIALIZER_SHA256 if ultra else INITIALIZER_SHA256
    initializer = initializer_path.read_bytes()
    if len(initializer) != 131072 or hashlib.sha256(initializer).hexdigest() != expected_hash:
        raise ValueError("expected the verified 131072-byte AMD 4.1.1 " + preset + " initializer")
    ultraperformance_dir = ultraperformance_dir or source_dir.parent / "ultraperformance"
    dirname = "reimpl411" if tier == 1080 else "reimpl" + str(tier)
    suffix = "" if tier == 1080 else "_" + str(tier)
    blob, param_blob, passes = bytearray(), bytearray(), {}
    lossy_rounding = (7, 8, 9) if ultra else LOSSY_ROUNDING_PASSES
    lossy_fold = None
    if profile == "lossy" and not ultra:
        spec = importlib.util.spec_from_file_location("fsr411_native_lossy", Path(__file__).with_name("fsr411") / "lossy.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        lossy_fold = module.fold_conv3x3

    def append(data):
        blob.extend(bytes((-len(blob)) % 64))
        offset = len(blob)
        blob.extend(data)
        return offset

    def pack_model(model, input_slot):
        weights = [append(bytes(x & 255 for x in matrix)) for matrix in model["weights"]]
        biases = [append(struct.pack("<" + str(len(a)) + "i", *a)) for a in model["bias"]]
        weights += [U32_MAX] * (4 - len(weights))
        biases += [U32_MAX] * (4 - len(biases))
        words = model["input"] + model["output"] + model.get("skip", [0] * 7) + weights + biases + sum(model["q"], [])
        packed = struct.pack("<53I", *words)
        if len(packed) != ABI_BYTES:
            raise ValueError("native ABI mismatch")
        parameter_offset = len(param_blob)
        param_blob.extend(packed)
        param_blob.extend(bytes(PARAMETER_STRIDE - ABI_BYTES))
        entry = {
            "kernel": model["kernel"], "tile": model["tile"], "kind": model["kind"], "threads": model.get("threads", 128),
            "function": model["kernel"],
            "reference": {"function": "fsr411_ref_" + model["kind"], "tile": [32, 1], "threads": 32},
            "dispatch_space": "input" if model["kind"] == "ct" else "output",
            "parameter_offset": parameter_offset, "input_slot": input_slot,
            "output_slot": model["output_slot"], "input": model["input"], "output": model["output"],
            "weights": weights, "biases": biases, "q": model["q"],
            "source": model["source"], "source_sha256": model["source_sha256"],
            "initializer_bias_offsets": model["initializer_bias_offsets"],
            "accumulator_bounds": model["accumulator_bounds"],
        }
        if "skip" in model:
            entry["skip"] = model["skip"]
            entry["skip_slot"] = model["skip_slot"]
        for key in ("output_offset_base", "scratch_append_alignment", "scratch_append_bytes", "tables_source", "tables_sha256", "lossy_transform"):
            if key in model:
                entry[key] = model[key]
        return entry

    for logical in range(1, 13):
        # Changed UP sources must exist; silently falling back to normal
        # baked tables would combine two different models.
        stage_dir = ultraperformance_dir if ultra and logical in (1, 2, 3, 4, 5, 10, 11, 12) else source_dir
        path = stage_dir / dirname / ("reimpl_pass" + str(logical) + suffix + ".hlsl")
        if logical in (1, 2, 12): model = parse_cnb(path, initializer)
        elif logical in (3, 6): model = parse_ds(path, initializer, tier, logical, ultra)
        else: model = parse_fnb(path, initializer, tier, logical, ultra)
        if profile == "lossy" and logical in lossy_rounding:
            model["q"][0][3] |= 4
            model["q"][1][3] |= 4
            model["lossy_transform"] = {"intermediate_rounding": "nearest_ties_toward_positive"}
        if lossy_fold and logical in LOSSY_FOLD_WORDS:
            matrices = list(model["weights"])
            matrices[0], stats = lossy_fold(matrices[0], LOSSY_FOLD_WORDS[logical])
            model["weights"] = matrices
            model.setdefault("lossy_transform", {})["conv3x3_folding"] = stats
        passes[str(logical)] = pack_model(model, INPUT_SLOTS[logical])
        passes[str(logical)].update(stage=logical * 2, logical_pass=logical)
    post_dir = ultraperformance_dir if ultra else source_dir
    post = parse_post_cnb(post_dir / dirname / ("post_rev_411" + suffix + ".hlsl"),
                          post_dir / "reimpl411/post_rev_411_tables.hlsli", tier)
    post_entry = pack_model(post, 13)
    post_entry["before_stage"] = 26
    manifest = {"schema": "fsr411_native_v1", "abi_version": 1, "parameter_size": ABI_BYTES,
                "parameter_stride": PARAMETER_STRIDE, "tier": tier, "passes": passes,
                "auxiliary_passes": {"post_cnb": post_entry},
                "initializer_sha256": expected_hash, "model": preset, "profile": profile,
                "presets": ["ultraperformance"] if ultra else ["native", "quality", "balanced", "performance", "drs"],
                "kernel_source": "src/fsr411_mpp.metal", "header_source": "src/fsr411_model.h",
                "reference_source": "src/fsr411_reference.metal",
                "defines": {"FSR411_MAXW": tier * 16 // 9 // 2, "FSR411_MAXH": tier // 2},
                "dynamic_sizes": "Replace input/output widths and heights from tensor constant slots. Preserve physical pitches. Keep graph border stages. For post_cnb, relocate output.offset to aligned end of original scratch and dispatch before the split imported POST."}
    if profile == "lossy":
        manifest["lossy_rules"] = {"rounding_passes": list(lossy_rounding),
                                   "fold_drop_words_per_channel": {} if ultra else LOSSY_FOLD_WORDS,
                                   "tails": "nearest_even_unchanged", "post_cnb": "exact",
                                   "folding_speed_note": "Dense MPP still executes zero MACs; folding alone does not reduce its work."}
    return _publish_native(output_dir, manifest, blob, param_blob)


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=root / "third_party/fsr411/shaders")
    parser.add_argument("--initializer", type=Path)
    parser.add_argument("--preset", choices=("normal", "ultraperformance"), default="normal")
    parser.add_argument("--profile", choices=("exact", "lossy"), default="exact")
    parser.add_argument("--ultraperformance-dir", type=Path, default=root / "third_party/fsr411/ultraperformance")
    parser.add_argument("--tier", type=int, choices=(1080, 2160, 4320), default=1080)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    initializer = args.initializer or root / ("third_party/fsr411/initializers-ultraperformance.bin" if args.preset == "ultraperformance" else "third_party/fsr411/initializers.bin")
    print(encode_model(args.source_dir, initializer, args.tier, args.outdir, args.preset, args.ultraperformance_dir, args.profile))


if __name__ == "__main__":
    main()
