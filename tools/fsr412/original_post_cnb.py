#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Replace only four original isolated POST CNB miss computations.

Original external shader/data remain in build/. No GPU/DLL execution here.
The primary cache stores, hit decoding, side hit/miss gates, CT and temporal
code are retained byte-for-byte. Precomputed cells must implement the removed
CNB math including active-ROI taps and conditional first-tap biases.
"""
from pathlib import Path
import argparse
import hashlib
import json
import re

IDENTITY = "a69f41a0bb37376178001a7f73087d4726e21c0d531e7635653277053ac4b268"
SOURCE_SHA256 = "de56b08c7abaf362f3a913cd1d1dff002ff64e478746d3da3df20edc5807e498"
CACHE_BASE = 99821056
CACHE_BYTES = 33273664
DEFAULT_BASE = CACHE_BASE + CACHE_BYTES
DEFAULT_STRIDE = 1920 * 16
CLASS_HEIGHT = 1080


def require(ok, why):
    if not ok:
        raise ValueError("original POST CNB: " + why)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def unique(source, token):
    require(source.count(token) == 1, "unexpected anchor count: " + token)
    return source.index(token)


def math_spans(source):
    """Pinned MSL anchors; scopes/outputs also checked independently."""
    specs = [
        ("primary", "            uint _1749 =", "            uint _26492 =", "_1697", "_1701",
         ("_26315", "_26371", "_26427", "_26483"), True),
        ("x", "                uint _26563 =", "                _107014 = ((((", "_1705", "_1701",
         ("_107011", "_107012", "_107013", "_107014"), False),
        ("y", "                uint _50420 =", "                _106862 = ((((", "_1697", "_1709",
         ("_106859", "_106860", "_106861", "_106862"), False),
        ("xy", "                uint _74277 =", "                _106710 = ((((", "_1705", "_1709",
         ("_106707", "_106708", "_106709", "_106710"), False),
    ]
    spans = []
    for name, first, last, x, y, outputs, declare in specs:
        start, end = unique(source, first), unique(source, last)
        if not declare:
            end = source.index("\n", end) + 1
        require(start < end, "reversed math scope")
        body = source[start:end]
        require(body.count("{") == body.count("}"), "unbalanced CNB math scope")
        require("OracleStore" not in body, "store inside replaceable math")
        require(body.count("extract_bits") == 11264, "CNB dot count changed")
        require("66547328u" in body and "30752u" in body, "CNB plane changed")
        for output in outputs:
            require(len(re.findall(r"(?:uint )?" + output + r" = \(\(\(\(", body)) == 1,
                    "unexpected output definition " + output)
        indent = "            " if declare else "                "
        replacement = indent + "uint4 oracleCnb_" + name + " = OraclePrecomputedCnb(oracle_u11_buffer0._m0, " + x + ", " + y + ");\n"
        replacement += "".join(indent + ("uint " if declare else "") + out + " = oracleCnb_" + name + "." + lane + ";\n"
                               for out, lane in zip(outputs, "xyzw"))
        spans.append(dict(name=name, start=start, end=end, x=x, y=y, outputs=outputs,
                          replacement=replacement, removed_sha256=digest(body.encode())))
    require(all(a["end"] <= b["start"] for a, b in zip(spans, spans[1:])), "overlapping CNB scopes")
    return spans


def rewrite(source, base=DEFAULT_BASE, stride=DEFAULT_STRIDE):
    require(digest(source.encode()) == SOURCE_SHA256, "unpinned isolated source SHA256")
    require(isinstance(base, int) and isinstance(stride, int), "base/stride must be integers")
    require(base >= DEFAULT_BASE and base % 16 == 0, "precompute base overlaps original/isolated planes or is unaligned")
    require(stride >= DEFAULT_STRIDE and stride % 16 == 0, "precompute stride too small or unaligned")
    allocation = base + stride * CLASS_HEIGHT
    require(allocation <= 0xffffffff, "precompute address exceeds uint32")
    require(source.count("^ 2779096485u") == 20, "cache XOR sites changed")
    spans = math_spans(source)
    out = source
    for span in reversed(spans):
        out = out[:span["start"]] + span["replacement"] + out[span["end"]:]
    entry = unique(out, "kernel void main0(")
    helper = ("// Independent compact NHWC precomputed CNB reader; x/y already clamped by original POST.\n"
              "inline uint4 OraclePrecomputedCnb(device uint *p, uint x, uint y)\n{\n"
              f"    const uint i = ({base}u + y * {stride}u + x * 16u) >> 2u;\n"
              "    return uint4(p[i], p[i+1u], p[i+2u], p[i+3u]);\n}\n\n")
    out = out[:entry] + helper + out[entry:]
    require(out.count("extract_bits") == source.count("extract_bits") - 45056, "unexpected surviving math count")
    receipt = dict(schema="fsr412_original_post_cnb_precomputed_v1", dxil_sha256=IDENTITY,
                   literal_isolated_metal_sha256=SOURCE_SHA256, metal_sha256=digest(out.encode()),
                   precomputed_base=base, precomputed_stride=stride, precomputed_bytes=stride * CLASS_HEIGHT,
                   physical_scratch_bytes=allocation, class_half=[1920, 1080], active_roi_taps=True,
                   original_cache_stores_preserved=True, compiler_profile="default-fno-fast-math",
                   production_execute_policy="run frames only; unchanged original isolated POST on skipped frames",
                   skipped_frame_limitation="original mutable-cache reads/writes have schedule-dependent results on cache misses",
                   transformations=[{k: v for k, v in span.items() if k != "replacement"} for span in spans])
    return out, receipt


def ordered_isolated(source):
    """Scheduling diagnostic only; retains isolated addresses and original math."""
    require(digest(source.encode()) == SOURCE_SHA256, "unpinned isolated source SHA256")
    token = "uint3 gl_WorkGroupID [[threadgroup_position_in_grid]]"
    require(source.count(token) == 1, "original group ID signature changed")
    source = source.replace(token, "uint3 oracleRawGroupID [[threadgroup_position_in_grid]]")
    entry = unique(source, "kernel void main0(") + len("kernel void main0(")
    source = source[:entry] + "constant uint& oracleGroupYOffset [[buffer(4)]], " + source[entry:]
    brace = source.index("\n{", entry) + 2
    source = source[:brace] + "\n    const uint3 gl_WorkGroupID=oracleRawGroupID+uint3(0u,oracleGroupYOffset,0u);" + source[brace:]
    return source


def external(path):
    path = Path(path).resolve()
    root = Path(__file__).resolve().parents[2] / "build"
    require(path.is_relative_to(root), "original-derived output must stay in build/")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("build/fsr412_upstream") / (IDENTITY + ".isolated.metal"))
    parser.add_argument("--out", type=Path, default=Path("build/fsr412_upstream") / (IDENTITY + ".precomputed.metal"))
    parser.add_argument("--ordered-diagnostic", action="store_true")
    parser.add_argument("--base", type=int, default=DEFAULT_BASE)
    parser.add_argument("--stride", type=int, default=DEFAULT_STRIDE)
    args = parser.parse_args()
    source = external(args.source).read_text()
    if args.ordered_diagnostic:
        text = ordered_isolated(source)
        receipt = dict(schema="fsr412_original_post_ordered_isolated_diagnostic_v1", metal_sha256=digest(text.encode()),
                       source_sha256=SOURCE_SHA256, scheduling_buffer_slot=4)
    else:
        text, receipt = rewrite(source, args.base, args.stride)
    output = external(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text)
    output.with_suffix(".json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
