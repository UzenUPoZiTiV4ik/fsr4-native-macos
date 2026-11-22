#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Trained weights for six of the twelve neural passes (FSR 4.1.2 Lossy).

The profile keeps its internal name lossy2. Since upstream release
dll-2026-11-22 its weights are that release's retrained `lossy` (the former
lossy2 trained further): new integer weights in passes 1, 2, 4, 5, 10 and 12;
layer arithmetic, folding policy and Ultra Performance stay lossy's. This
module maps those integer weights onto the AMD-layout literal tables of the
imported MIT sources. The weights are GPL-2.0-or-later data and are not
vendored: tools/fsr411/fetch_lossy2_weights.py recovers them once from the
pinned release DLL (tools/fsr411/lossy412_weights.py) into build/.

Layouts are the exact inverses of the decoders in tools/gen_fsr411_native.py
(parse_cnb/parse_fnb), which are themselves validated against AMD's shaders.
"""

import hashlib
import os
import re
import struct
from pathlib import Path
import zipfile

DEFAULT_WEIGHTS = Path("build/fsr411_lossy2/weights_lossy2.npz")
UPSTREAM_RELEASE = "dll-2026-11-22"
UPSTREAM_URL = ("https://github.com/zgauthier2000/fsr4-shader-performance-overrides/releases/download/"
                "dll-2026-11-22/fsr4.1.2-cyboman-rx6000-2026-11-22.zip")
UPSTREAM_ARCHIVE_SHA256 = "2d81d93647e5c78c04401cf1437bd706e2c634ac72ed051817571c5c98be7d15"
UPSTREAM_DLL_MEMBER = "lossy/amd_fidelityfx_upscaler_dx12.dll"
UPSTREAM_DLL_SHA256 = "8d462f462efd681aa26b51d6d1217864d907679fa4e6ff4e9381aecdae2ad4fb"
# The 18 trained-pass DXBC containers of that DLL (six passes x three tiers).
UPSTREAM_CONTAINERS = frozenset((
    "0199674e02137608aa1aad54d48279b89efce1e2d5db69a4a1921fa5a8242903",
    "0709c8489199be350e8d61df602a3185ae14fb9eded190a5de513b9fd241ae2b",
    "0ccb18f43a64278cc6f667af1cce508dd54308ddca2dfabcdeb70cce50c1cafa",
    "1677285597cf551b75833716a48ddd24980f2e7be455d7def3e8fbf2d03c8c5a",
    "1cf347ccd2ba7b38ddabd2f18d0f0165e9ac83a49faf7e6cc247cebf408ae266",
    "2cbc3984b62c96dc53e6da59ae21cba93c981734232bff8ba6e83e5b81251e42",
    "3014dca760b500a9f8a2395f7a171e23db24b2389b41b0b7ab498b855b41b316",
    "3f464270992928185da82968d41c04b427a82eef7c977438a3afff857c111503",
    "41d16f4c4067c6e68252583dc223909459c6a21459504a2bc3bc6a64cad2f6ba",
    "48155b7179d5c3c74542f1ec01cab6aff8493f9ddf17c2aa7ba5d9c23fc9f308",
    "4da8b8a62ba8ea94def3bb47ef6b36bedebd7c3c9f37e8629d7313544171f0ab",
    "55cdb4fd73287bab95c6ad854684650c212843a89c0f4916dda841cc36c5e0a2",
    "9108b15cef0278813629585d133e2506f66dbd5a09e7ed0c246db43d94fd8772",
    "988554b59509b4a2f35d3784abeba464766c4debeece43b4665fe9320f437b1b",
    "99965b3b9f8b2ddae345364c27853dcf8a4ae25c390da7132d0bce8235c55960",
    "a8d2063c14aa6b9f782c15f0afe4e62816931739625052f469fc492d6603b0fc",
    "ece23f5c6a6ec6aebec90d3b2b6012f0ca17b4dcda13b3687e6160e78055ca80",
    "ff2be2d590c8552af9f2e78bb024f17e101cf5a17b014e88de099cbadfb19ed9",
))
# SHA-256 of the byte-reproducible .npz written from them.
UPSTREAM_SHA256 = "33ef308990fd749af2d7c833dfbb5ad1879220d7cf4002d3abd982c3c9a17143"
# Previous pin (dll-2026-10-19 lossy2), still accepted through --lossy2-weights.
PREVIOUS_SHA256 = "9a458f65812848b273dace03edcbd346d66364d5d41a99d0963adaa601a3c93c"

# logical pass -> (kind, 3x3 table, expansion table, contraction table)
TABLES = {
    1: ("cnb", "W1", "W2", "W3"),
    2: ("cnb", "W1", "W2", "W3"),
    12: ("cnb", "W1", "W2", "W3"),
    4: ("fnb", "W0", "W1", "W2"),
    5: ("fnb", "W0", "W1", "W2"),
    10: ("fnb", "W0", "W1", "W2"),
}
# Bias arrays (bias0, bias1, bias2) read from the pinned initializer, bytes.
BIAS_OFFSETS = {
    1: (3456, 4096, 4736),
    2: (7168, 7808, 8448),
    12: (125312, 125952, 126592),
    4: (13056, 15232, 17536),
    5: (19968, 22144, 24448),
    10: (109312, 111488, 113792),
}
# Bias counts and pointwise (outputs, inputs) per block kind.
CNB = (16, 32, 16)   # 3x3 outputs, expansion outputs, contraction outputs
FNB = (16, 64, 32)
BIAS_CHANNELS = {"cnb": CNB, "fnb": FNB}
POINTWISE = {"cnb": ((32, 16), (16, 32)), "fnb": ((64, 32), (32, 64))}
MARKER = "// fsr411-generated-lossy2: pass=%d weights=%s\n"


def resolve_path(root, path=None):
    """Explicit path, then FSR4_LOSSY2_WEIGHTS, then the build directory."""
    candidate = Path(path) if path else None
    if candidate is None:
        env = os.environ.get("FSR4_LOSSY2_WEIGHTS")
        if env:
            candidate = Path(env)
    if candidate is None:
        candidate = Path(root) / DEFAULT_WEIGHTS
    if not candidate.is_file():
        raise ValueError("lossy2 weights not found: %s (run make lossy2-weights)" % candidate)
    return candidate


def _read_npy(data):
    """Minimal reader for the little-endian int32 arrays in the pinned archive."""
    if not data.startswith(b"\x93NUMPY") or data[6] not in (1, 2):
        raise ValueError("lossy2: unsupported .npy header")
    if data[6] == 1:
        length, start = struct.unpack_from("<H", data, 8)[0], 10
    else:
        length, start = struct.unpack_from("<I", data, 8)[0], 12
    header = data[start:start + length].decode("latin1")
    descr = re.search(r"'descr':\s*'([<>=|])([iu])(\d+)'", header)
    shape = re.search(r"'shape':\s*\(([^)]*)\)", header)
    order = re.search(r"'fortran_order':\s*(True|False)", header)
    if not descr or not shape or not order or order.group(1) != "False":
        raise ValueError("lossy2: unsupported .npy layout")
    if descr.group(1) != "<" or descr.group(3) != "4":
        raise ValueError("lossy2: expected little-endian 32-bit integers")
    dims = tuple(int(value) for value in shape.group(1).replace(" ", "").split(",") if value)
    count = 1
    for dim in dims:
        count *= dim
    body = data[start + length:]
    if len(body) != 4 * count:
        raise ValueError("lossy2: truncated .npy array")
    code = "i" if descr.group(2) == "i" else "I"
    return dims, list(struct.unpack("<%d%s" % (count, code), body))


def _read_npz(path):
    archive = {}
    with zipfile.ZipFile(path) as bundle:
        for name in bundle.namelist():
            if name.endswith(".npy"):
                archive[name[:-4]] = _read_npy(bundle.read(name))
    return archive


def _value(shape, flat, index):
    """Element of a C-order nested array stored flat."""
    offset = 0
    for dim, position in zip(shape, index):
        offset = offset * dim + position
    return flat[offset]

class Weights:
    def __init__(self, path, layers, sha256):
        self.path = path
        self.layers = layers
        self.sha256 = sha256


def load(root, path=None):
    """Read and validate the six trained layers; returns Weights."""
    resolved = resolve_path(root, path)
    data = resolved.read_bytes()
    sha256 = hashlib.sha256(data).hexdigest()
    archive = _read_npz(resolved)
    layers = {}
    for logical, (kind, _, _, _) in TABLES.items():
        prefix = "p%d_" % logical
        try:
            (a_shape, a_flat), (bw_shape, bw_flat), (cw_shape, cw_flat) = (
                archive[prefix + name] for name in ("A", "Bw", "Cw"))
            bias_entries = [archive[prefix + name] for name in ("bA", "bB", "bC")]
        except KeyError as error:
            raise ValueError("lossy2: missing array %s" % error)
        channels, (expansion, contraction) = BIAS_CHANNELS[kind], POINTWISE[kind]
        if a_shape != (16, 16, 3, 3):
            raise ValueError("lossy2 pass %d: unexpected 3x3 weights %s" % (logical, a_shape))
        if bw_shape != expansion or cw_shape != contraction:
            raise ValueError("lossy2 pass %d: unexpected pointwise shapes %s %s"
                             % (logical, bw_shape, cw_shape))
        if any(shape != (count,) for (shape, _), count in zip(bias_entries, channels)):
            raise ValueError("lossy2 pass %d: unexpected bias shapes" % logical)
        for name, flat in (("A", a_flat), ("Bw", bw_flat), ("Cw", cw_flat)):
            if min(flat) < -128 or max(flat) > 127:
                raise ValueError("lossy2 pass %d: %s weights outside int8" % (logical, name))
        a = [[[[_value(a_shape, a_flat, (oc, ic, ky, kx)) for kx in range(3)] for ky in range(3)]
              for ic in range(16)] for oc in range(16)]
        bw = [[_value(bw_shape, bw_flat, (oc, ic)) for ic in range(bw_shape[1])]
              for oc in range(bw_shape[0])]
        cw = [[_value(cw_shape, cw_flat, (oc, ic)) for ic in range(cw_shape[1])]
              for oc in range(cw_shape[0])]
        layers[logical] = {"a": a, "bw": bw, "cw": cw,
                           "bias": [list(flat) for _, flat in bias_entries]}
    return Weights(str(resolved), layers, sha256)


def conv3x3_words(a):
    """[out][in][ky][kx] -> the 36x16 packed word table read by the shaders."""
    words = [0] * (36 * 16)
    for oc in range(16):
        for ic in range(16):
            for tap in range(9):
                ky, kx = divmod(tap, 3)
                value = a[oc][ic][ky][kx]
                if not -128 <= value <= 127:
                    raise ValueError("lossy2 3x3 weights outside int8")
                row = (oc // 4) * 9 + tap
                column = (oc % 4) * 4 + ic // 4
                words[row * 16 + column] |= (value & 255) << (8 * (ic % 4))
    return words


def pointwise_words(weights):
    """[out][in] -> packed words; each input bank of 16 adds one row per output group."""
    cout, cin = len(weights), len(weights[0])
    if cin % 16 or cout % 4:
        raise ValueError("pointwise weights need 16-input banks and 4-output groups")
    rows = (cout // 4) * (cin // 16)
    words = [0] * (rows * 16)
    for oc in range(cout):
        for ic in range(cin):
            value = weights[oc][ic]
            if not -128 <= value <= 127:
                raise ValueError("lossy2 pointwise weights outside int8")
            row = (oc // 4) * (cin // 16) + ic // 16
            column = 4 * (oc % 4) + (ic % 16) // 4
            words[row * 16 + column] |= (value & 255) << (8 * (ic % 4))
    return words


def _literal_table(source, name, rows):
    pattern = re.compile(r"(static const uint " + name + r"\[" + str(rows) + r"\]\[16\]\s*=\s*\{)(.*?)(\};)", re.S)
    matches = list(pattern.finditer(source))
    if len(matches) != 1:
        raise ValueError("lossy2: expected one %s[%d][16] literal table" % (name, rows))
    return matches[0]


def _write_table(source, match, words):
    if len(words) != len(re.findall(r"0x[0-9a-fA-F]+u", match.group(2))):
        raise ValueError("lossy2: literal table size changed")
    values = iter("0x%08xu" % (word & 0xFFFFFFFF) for word in words)
    body = re.sub(r"0x[0-9a-fA-F]+u", lambda _: next(values), match.group(2))
    return source[:match.start(2)] + body + source[match.end(2):]


def _bias_helper(offset, values):
    words = ", ".join("0x%08xu" % (value & 0xFFFFFFFF) for value in values)
    return ("static const uint Fsr411Lossy2Bias%d[%d] = { %s };\n"
            "uint4 Fsr411Lossy2Load%d(uint byteOffset)\n{\n"
            "    uint index = byteOffset >> 2u;\n"
            "    return uint4(Fsr411Lossy2Bias%d[index + 0u], Fsr411Lossy2Bias%d[index + 1u],\n"
            "                 Fsr411Lossy2Bias%d[index + 2u], Fsr411Lossy2Bias%d[index + 3u]);\n}\n\n"
            % (offset, len(values), words, offset, offset, offset, offset, offset))


def replace_bias_loads(source, logical):
    """Point the three initializer bias loads at literal constant arrays."""
    token = "t18_initializer.Load4("
    offsets = BIAS_OFFSETS[logical]
    position, seen, replaced = 0, [], []
    while True:
        start = source.find(token, position)
        if start < 0:
            replaced.append(source[position:])
            break
        cursor, depth = start + len(token), 1
        while depth:
            if cursor >= len(source):
                raise ValueError("lossy2: unbalanced initializer load")
            depth += source[cursor] == "("
            depth -= source[cursor] == ")"
            cursor += 1
        inner = source[start + len(token):cursor - 1]
        match = re.fullmatch(r"\s*(\d+)u\s*\+\s*(.+)", inner, re.S)
        if not match:
            raise ValueError("lossy2: unrecognized initializer bias load: " + inner[:80])
        offset, rest = int(match.group(1)), match.group(2).strip()
        seen.append(offset)
        replaced.append(source[position:start])
        replaced.append("Fsr411Lossy2Load%d(%s)" % (offset, rest))
        position = cursor
    if seen != list(offsets):
        raise ValueError("lossy2 pass %d: expected bias loads %s, found %s" % (logical, offsets, seen))
    return "".join(replaced)


def apply_source(source, logical, layer, bias_literals=True):
    """Replace this pass's weight tables (and optionally its bias loads)."""
    if logical not in TABLES:
        raise ValueError("lossy2 covers passes 1, 2, 4, 5, 10 and 12 only")
    marker = MARKER % (logical, hashlib.sha256(repr(layer).encode()).hexdigest()[:16])
    if source.startswith(marker):
        return source
    if "fsr411-generated-lossy2:" in source:
        raise ValueError("source already carries different lossy2 weights")
    kind, conv_name, expand_name, contract_name = TABLES[logical]
    expansion, contraction = pointwise_words(layer["bw"]), pointwise_words(layer["cw"])
    source = _write_table(source, _literal_table(source, conv_name, 36), conv3x3_words(layer["a"]))
    source = _write_table(source, _literal_table(source, expand_name, len(expansion) // 16), expansion)
    source = _write_table(source, _literal_table(source, contract_name, len(contraction) // 16), contraction)
    if bias_literals:
        declaration = "ByteAddressBuffer t18_initializer : register(t18);"
        if source.count(declaration) != 1:
            raise ValueError("lossy2: expected one initializer declaration")
        helpers = "".join(_bias_helper(offset, values)
                          for offset, values in zip(BIAS_OFFSETS[logical], layer["bias"]))
        source = source.replace(declaration, declaration + "\n" + helpers, 1)
        source = replace_bias_loads(source, logical)
    return marker + source


def patched_initializer(data, layers):
    """Copy of the pinned initializer with the six passes' biases replaced.

    The native weight generator reads biases at fixed offsets; patching a copy
    keeps the verified initializer file untouched.
    """
    if len(data) != 131072:
        raise ValueError("expected the 131072-byte FSR 4.1.1 initializer")
    patched = bytearray(data)
    for logical, layer in layers.items():
        for offset, values in zip(BIAS_OFFSETS[logical], layer["bias"]):
            if offset + 4 * len(values) > len(patched):
                raise ValueError("lossy2 bias outside the initializer")
            struct.pack_into("<%di" % len(values), patched, offset, *values)
    return bytes(patched)
