#!/usr/bin/env python3
"""Recover FSR 4.1.1 Ultra Performance HLSL overlays from official INT8 DXIL.

Run import_model.py --with-shaders first. The DLL is never executed: native
DXC disassembles its already extracted, SHA-256 verified shader containers.
The common MIT HLSL stays unchanged. Generated sources have independent
weights and quantization parameters and need initializers-ultraperformance.bin.
Static reconstruction is not a claim of Metal or D3D12 output parity.
"""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re
import struct
import subprocess
import sys

import import_model as importer


TIERS = {1080: "reimpl411", 2160: "reimpl2160", 4320: "reimpl4320"}
CHANGED_PASSES = {1, 2, 3, 4, 5, 10, 11, 12}
WEIGHT_PASSES = {1, 2, 4, 5, 10, 12}
ARRAY = re.compile(r"static const uint (W\d+)\[(\d+)\]\[16\]\s*=\s*\{(.*?)\};", re.S)
STORE = re.compile(r"^\s*store i32 (-?\d+), i32\* %\d+, align 4", re.M)
DOT = re.compile(r"@dx.op.dot4AddPacked.i32\(i32 163, i32 (%\d+|-?\d+), "
                 r"i32 (%\d+|-?\d+), i32 (%\d+|-?\d+)\)")
POST_COMMON = "f5c9eaf26bba29d7c743f565d8278e3f36f1613156e56dbfe09a5853687d90e8"
POST_ULTRA = "d977f9b2b80836d0d292b2fd1a69e781e78b35f9d04100c6759ae9e17b4ef1d3"


def fmul_delta(before, after, count):
    return {(f"%v = fmul fast float %v, {before}",
             f"%v = fmul fast float %v, {after}"): count}


def requant_delta(before, after, count):
    return {(f"%v = {op} i32 %v, {a}", f"%v = {op} i32 %v, {b}"): count
            for op, a, b in (("lshr", before, after), ("ashr", before, after),
                             ("add", (1 << (before - 1)) - 1, (1 << (after - 1)) - 1))}


EXPECTED_DELTAS = {
    1: fmul_delta("6.400000e+01", "3.200000e+01", 16),
    2: {**fmul_delta("1.562500e-02", "3.125000e-02", 16),
        **fmul_delta("0x3F30000000000000", "0x3F40000000000000", 16)},
    3: {**fmul_delta("7.812500e-03", "%v", 16),
        **fmul_delta("7.812500e-03", "3.906250e-03", 16)},
    4: {**requant_delta(7, 8, 16), **requant_delta(8, 7, 64),
        **fmul_delta("1.562500e-02", "%v", 16),
        **fmul_delta("1.562500e-02", "3.125000e-02", 16)},
    5: {}, 6: {}, 7: {}, 8: {}, 9: {},
    10: {**requant_delta(8, 7, 16), **requant_delta(7, 8, 64),
         **fmul_delta("0x3F40000000000000", "0x3F30000000000000", 32)},
    11: {**fmul_delta("8.000000e+00", "1.600000e+01", 4),
         **fmul_delta("0x3F30000000000000", "0x3F20000000000000", 4)},
    12: {**requant_delta(7, 8, 16), **requant_delta(8, 7, 32)},
    "post": {**requant_delta(8, 7, 16), **requant_delta(7, 8, 32)},
}

# Replacements are deliberately scoped to whole expressions. Blindly swapping
# every 7/8 or 63/127 would also mutate addresses or change a later stage twice.
SOURCE_EDITS = {
    1: [("round((t1 + t2) * 64.0f)", "round((t1 + t2) * 32.0f)")],
    2: [("float(cbyte) * 0.015625f", "float(cbyte) * 0.03125f"),
        ("float(A12[4 * d + j3]) * 0.000244140625f", "float(A12[4 * d + j3]) * 0.00048828125f")],
    3: [("float(slot[16 * q + 4 * d + j3]) * 7.8125e-03f",
         "float(slot[16 * q + 4 * d + j3]) * ((q == 0) ? 7.8125e-03f : 3.90625e-03f)")],
    4: [("requant_shift(acc0[4 * d0 + j0], 7u, 63)", "requant_shift(acc0[4 * d0 + j0], 8u, 127)"),
        ("requant_shift(max(acc1[4 * d2 + j2], 0), 8u, 127)", "requant_shift(max(acc1[4 * d2 + j2], 0), 7u, 63)"),
        ("float(rb) * 0.015625f", "float(rb) * ((qq == 0) ? 0.015625f : 0.03125f)")],
    5: [],
    10: [("requant_shift(acc0[4 * d0 + j0], 8u, 127)", "requant_shift(acc0[4 * d0 + j0], 7u, 63)"),
         ("requant_shift(max(acc1[4 * d2 + j2], 0), 7u, 63)", "requant_shift(max(acc1[4 * d2 + j2], 0), 8u, 127)"),
         ("float(acc2[16 * qq + 4 * d4 + j3]) * 0.00048828125f", "float(acc2[16 * qq + 4 * d4 + j3]) * 0.000244140625f")],
    11: [("round(v * 8.0f)", "round(v * 16.0f)"),
         ("float(rb) * 0.03125f + float(av) * 0.000244140625f", "float(rb) * 0.03125f + float(av) * 0.0001220703125f")],
    12: [("requant_shift(A8[f + j], 7u, 63)", "requant_shift(A8[f + j], 8u, 127)"),
         ("requant_shift(max(A10[f2 + j2], 0), 8u, 127)", "requant_shift(max(A10[f2 + j2], 0), 7u, 63)")],
    "post": [("int rq256(int v) { return (v + 127 + ((v >> 8) & 1)) >> 8; }",
              "int rq256(int v) { return (v + 63 + ((v >> 7) & 1)) >> 7; }"),
             ("int rq128relu(int v) { v = max(v, 0); return (v + 63 + ((v >> 7) & 1)) >> 7; }",
              "int rq128relu(int v) { v = max(v, 0); return (v + 127 + ((v >> 8) & 1)) >> 8; }")],
}


def strip_comments(text):
    return re.sub(r"//[^\n]*|/\*.*?\*/", "", text, flags=re.S)


def source_words(source):
    words = []
    for match in ARRAY.finditer(strip_comments(source)):
        values = [int(v, 0) for v in re.findall(r"(0x[0-9a-fA-F]+|\d+)u?", match[3])]
        if len(values) != int(match[2]) * 16:
            raise ValueError("unexpected common weight-table shape")
        words.extend(values)
    return words


def embedded_words(disassembly):
    return [int(value) & 0xFFFFFFFF for value in STORE.findall(disassembly)]


def post_tables(disassembly):
    calls = DOT.findall(disassembly)
    words, seeds = [], []
    for accumulator, left, right in calls:
        literals = [value for value in (left, right) if not value.startswith("%")]
        if len(literals) != 1:
            raise ValueError("POST dot must have exactly one immediate weight")
        words.append(int(literals[0]))
        if not accumulator.startswith("%"):
            seeds.append(int(accumulator))
    if len(words) != 960 or len(seeds) != 96:
        raise ValueError("unexpected POST weight/seed count")
    return words, seeds


def patch_once(source, before, after):
    if source.count(before) != 1:
        raise ValueError("expected one source expression: " + before)
    return source.replace(before, after)


def patch_weights(source, words):
    position = 0
    def replace(match):
        nonlocal position
        rows = int(match[2])
        table = words[position:position + rows * 16]
        if len(table) != rows * 16:
            raise ValueError("truncated recovered weight table")
        position += len(table)
        body = "\n".join("    { " + ", ".join("0x%08xu" % w for w in table[i:i + 16]) + " },"
                         for i in range(0, len(table), 16))
        return "static const uint %s[%d][16] = {\n%s\n};" % (match[1], rows, body)
    result = ARRAY.sub(replace, source)
    if position != len(words):
        raise ValueError("unconsumed recovered weight words")
    return result


def validate_extra_select(disassembly, stage):
    values = ("7.812500e-03, float 3.906250e-03" if stage == 3 else
              "1.562500e-02, float 3.125000e-02")
    select = re.findall(r"^\s*(%\d+) = select i1 (%\d+), float " + re.escape(values) + "$",
                        disassembly, re.M)
    if len(select) != 1:
        raise ValueError("missing Ultra Performance channel-select scale")
    definitions = dict(re.findall(r"^\s*(%\d+) = (.*)$", disassembly, re.M))
    predicate = select[0][1]
    compare = re.fullmatch(r"icmp ult i32 (%\d+), 16", definitions[predicate])
    shift = re.fullmatch(r"shl i32 (%\d+), 5", definitions[compare[1]]) if compare else None
    if not shift or not definitions[shift[1]].startswith("call i32 @dx.op.threadId.i32(i32 93, i32 2)"):
        raise ValueError("channel-select predicate is not tid.z * 32 < 16")
    return set(select[0] if stage == 3 else select[0][:1])


def normalized_ir(disassembly, stage, ultra=False):
    body = disassembly.split("define void ", 1)[1].split("\n}", 1)[0]
    omit = validate_extra_select(disassembly, stage) if ultra and stage in (3, 4) else set()
    lines = []
    for line in body.splitlines()[1:]:
        line = line.split(";", 1)[0].strip()
        if not line or line.split(" = ", 1)[0] in omit:
            continue
        line = re.sub(r"%\d+", "%v", line)
        line = re.sub(r"^store i32 -?\d+,", "store i32 TABLE,", line)
        if "phi i32 " in line:
            line = re.sub(r"\[ -?\d+,", "[ PHI,", line)
        if "@dx.op.dot4AddPacked.i32(" in line and stage == "post":
            line = re.sub(r"i32 -?\d+", "i32 IMM", line)
        lines.append(line)
    return lines


def check_delta(common, ultra, stage):
    before = normalized_ir(common, stage)
    after = normalized_ir(ultra, stage, True)
    if len(before) != len(after):
        raise ValueError("unexpected Ultra Performance IR structure in stage %s" % stage)
    observed = Counter((a, b) for a, b in zip(before, after) if a != b)
    if observed != Counter(EXPECTED_DELTAS[stage]):
        raise ValueError("unaccounted Ultra Performance IR delta in stage %s: %r" % (stage, observed))
    return [{"from": a, "to": b, "count": count} for (a, b), count in sorted(observed.items())]


def header(common_source, oracle):
    return ("// SPDX-License-Identifier: MIT\n"
            "// Copyright (C) 2025 Advanced Micro Devices, Inc.\n"
            "// Copyright (C) 2026 int3rrobang (common HLSL reimplementation)\n"
            "// Generated by tools/fsr411/recover_ultraperformance.py; do not edit.\n"
            "// Common source: " + common_source + "\n"
            "// Official AMD Ultra Performance DXIL SHA-256: " + oracle + "\n"
            "// Use initializers-ultraperformance.bin. Static reconstruction only;\n"
            "// independent GPU output parity has not been established.\n\n")


def post_signature(disassembly):
    # Only known model-dependent literal sites are erased. Geometry, transfers,
    # debug paths, and every other instruction remain part of the permutation.
    text = "\n".join(normalized_ir(disassembly, "post"))
    text = re.sub(r"(lshr|ashr) i32 %v, [78]\b", r"\1 i32 %v, SHIFT", text)
    text = re.sub(r"add i32 %v, (63|127)\b", "add i32 %v, ROUND", text)
    return importer.sha256(text.encode())


def generate(output, dxc):
    importer.verify_sources(output)
    importer.verify_initializers(output)
    manifest = importer.read_verified_manifest(output)
    selected = {row["sha256"]: row for row in manifest["shaders"]
                if re.search(r"_(?:pass\d+|postpass)$", row["entry_point"])}
    def disassemble(item):
        digest, row = item
        path = output / row["oracle_path"]
        data = path.read_bytes()
        if len(data) != row["size"] or importer.sha256(data) != digest:
            raise ValueError("official DXIL hash/size mismatch: " + str(path))
        result = subprocess.run([str(dxc), "-dumpbin", str(path)], check=True,
                                capture_output=True, text=True)
        return digest, result.stdout
    with ThreadPoolExecutor(max_workers=4) as workers:
        ir = dict(workers.map(disassemble, selected.items()))
    files, model_rows = {}, []
    for tier, directory in TIERS.items():
        for stage in range(1, 13):
            suffix = "" if tier == 1080 else "_" + str(tier)
            relative = "shaders/%s/reimpl_pass%d%s.hlsl" % (directory, stage, suffix)
            original = (output / relative).read_text()
            source = strip_comments(original)
            guards = re.search(r"if \(tx > (\d+)u \|\| ty > (\d+)u\)", source).groups()
            candidates = {digest: text for digest, text in ir.items()
                          if selected[digest]["entry_point"].endswith("_pass" + str(stage)) and
                          tuple(re.findall(r"icmp ugt i32 %\d+, (\d+)", text)[:2]) == guards}
            if stage in WEIGHT_PASSES:
                matching = [digest for digest, text in candidates.items() if embedded_words(text) == source_words(source)]
            elif stage == 3:
                matching = [digest for digest, text in candidates.items()
                            if "float 7.812500e-03, float 3.906250e-03" not in text]
            elif stage == 11:
                matching = [digest for digest, text in candidates.items()
                            if re.search(r"fmul fast float %\d+, 8.000000e\+00", text)]
            else:
                matching = list(candidates)
            changed = stage in CHANGED_PASSES
            if len(matching) != 1 or len(candidates) != (2 if changed else 1):
                raise ValueError("ambiguous original model pair for %d pass %d" % (tier, stage))
            common = matching[0]
            ultra = next(digest for digest in candidates if digest != common) if changed else common
            delta = check_delta(ir[common], ir[ultra], stage)
            recovered = embedded_words(ir[ultra])
            if changed:
                if stage in WEIGHT_PASSES:
                    if len(recovered) != len(source_words(source)):
                        raise ValueError("Ultra Performance model table dimensions changed")
                    source = patch_weights(source, recovered)
                for before, after in SOURCE_EDITS[stage]:
                    source = patch_once(source, before, after)
                target = "ultraperformance/%s/%s" % (directory, Path(relative).name)
                files[target] = (header(relative, ultra) + source.strip() + "\n").encode()
            else:
                target = relative
            model_rows.append({"tier": tier, "pass": stage, "source": target,
                               "common_source": relative, "common_dxil_sha256": common,
                               "ultraperformance_dxil_sha256": ultra,
                               "baked_weight_words": len(recovered),
                               "baked_weight_sha256": importer.sha256(struct.pack("<%dI" % len(recovered), *recovered)),
                               "scalar_instruction_deltas": delta})

    common_tables = post_tables(ir[POST_COMMON])
    ultra_tables = post_tables(ir[POST_ULTRA])
    table_source = output / "shaders/reimpl411/post_rev_411_tables.hlsli"
    arrays = {name: [int(value) for value in re.findall(r"-?\d+", body)]
              for name, body in re.findall(r"static const int (\w+)\[\d+\]\s*=\s*\{(.*?)\};",
                                          table_source.read_text(), re.S)}
    if common_tables != (sum((arrays[name] for name in ("DW_W", "PW1_W", "PW2_W", "CT2D_W")), []),
                         sum((arrays[name] for name in ("DW_S", "PW1_S", "PW2_S", "CT2D_S")), [])):
        raise ValueError("common POST HLSL tables do not match the official oracle")
    post_groups = defaultdict(list)
    for digest, text in ir.items():
        if selected[digest]["entry_point"].endswith("_postpass"):
            post_groups[post_signature(text)].append(digest)
    post_rows = []
    if len(post_groups) != 48:
        raise ValueError("expected 48 common/Ultra Performance POST pairs")
    for digests in post_groups.values():
        common = [digest for digest in digests if post_tables(ir[digest]) == common_tables]
        ultra = [digest for digest in digests if post_tables(ir[digest]) == ultra_tables]
        if len(digests) != 2 or len(common) != 1 or len(ultra) != 1:
            raise ValueError("POST permutation does not have exactly one pair of models")
        check_delta(ir[common[0]], ir[ultra[0]], "post")
        post_rows.append({"common_dxil_sha256": common[0], "ultraperformance_dxil_sha256": ultra[0]})
    table_text = header("shaders/reimpl411/post_rev_411_tables.hlsli", POST_ULTRA)
    word_pos = seed_pos = 0
    for name, values in arrays.items():
        if name.endswith("_W"):
            recovered = ultra_tables[0][word_pos:word_pos + len(values)]
            word_pos += len(values)
        else:
            recovered = ultra_tables[1][seed_pos:seed_pos + len(values)]
            seed_pos += len(values)
        table_text += "static const int %s[%d] = {\n" % (name, len(values))
        table_text += "\n".join("    " + ", ".join(str(value) for value in recovered[i:i + 8]) + ","
                                for i in range(0, len(recovered), 8)) + "\n};\n"
    files["ultraperformance/reimpl411/post_rev_411_tables.hlsli"] = table_text.encode()
    post_sources = []
    for tier, directory in TIERS.items():
        suffix = "" if tier == 1080 else "_" + str(tier)
        relative = "shaders/%s/post_rev_411%s.hlsl" % (directory, suffix)
        source = strip_comments((output / relative).read_text())
        for before, after in SOURCE_EDITS["post"]:
            source = patch_once(source, before, after)
        source = source.replace("rq256(", "rq128(").replace("rq128relu(", "rq256relu(")
        stride = (tier * 8 // 9 + 2) * 16
        basis = "\n".join(normalized_ir(ir[POST_COMMON], "post")).replace("i32 15392,", "i32 %d," % stride)
        common = [row["common_dxil_sha256"] for row in post_rows
                  if "\n".join(normalized_ir(ir[row["common_dxil_sha256"]], "post")) == basis]
        if len(common) != 1:
            raise ValueError("cannot identify linear POST geometry for tier %d" % tier)
        pair = next(row for row in post_rows if row["common_dxil_sha256"] == common[0])
        target = "ultraperformance/%s/%s" % (directory, Path(relative).name)
        files[target] = (header(relative, pair["ultraperformance_dxil_sha256"]) + source.strip() + "\n").encode()
        post_sources.append({"tier": tier, "source": target, "common_source": relative, **pair})
    shared = []
    for entry in sorted(set(row["entry_point"] for row in manifest["shaders"])):
        if entry.endswith("_prepass") or re.search(r"_pass\d+_post$", entry):
            counts = Counter(row["sha256"] for row in manifest["shaders"] if row["entry_point"] == entry)
            if set(counts.values()) != {2}:
                raise ValueError("PRE/padding shader differs between model variants")
            shared.append({"entry": entry, "shared_dxil_sha256": sorted(counts)})
    receipt = {"schema": "fsr411_ultraperformance_manifest_v1", "provider_version": "4.1.1.2740",
               "dll_sha256": importer.DLL_SHA256,
               "initializer": importer.INITIALIZERS["ultraperformance_initializer"],
               "generator": "tools/fsr411/recover_ultraperformance.py",
               "validation": "Pinned artifacts, table order, shapes and enumerated scalar IR deltas verified; no independent GPU output parity test.",
               "dispatch": "Identical graph, tensor layout, stage order and dispatch geometry to common. Use z=1 dispatch for all neural passes.",
               "neural_passes": model_rows, "post_sources": post_sources,
               "post_pairs": sorted(post_rows, key=lambda row: row["common_dxil_sha256"]),
               "post_scalar_instruction_deltas": [{"from": a, "to": b, "count": count}
                                                  for (a, b), count in sorted(EXPECTED_DELTAS["post"].items())],
               "shared_pre_and_padding": shared,
               "files": [{"path": path, "size": len(data), "sha256": importer.sha256(data)}
                         for path, data in sorted(files.items())]}
    files["ultraperformance_manifest.json"] = (json.dumps(receipt, indent=2) + "\n").encode()
    return files


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=importer.DEFAULT_OUTPUT)
    parser.add_argument("--dxc", type=Path, default=Path(__file__).resolve().parents[1] / "bin/dxc")
    parser.add_argument("--verify", action="store_true", help="regenerate in memory and compare every output without writes")
    args = parser.parse_args(argv)
    try:
        files = generate(args.output, args.dxc)
        for relative, data in files.items():
            path = args.output / relative
            if args.verify:
                if not path.is_file() or path.read_bytes() != data:
                    raise ValueError("Ultra Performance generated source differs: " + str(path))
            else:
                importer.write_if_changed(path, data)
        print(("Verified" if args.verify else "Recovered") +
              " Ultra Performance overlays: 28 source files, 36 neural pairs, 48 POST pairs; separate 131072-byte initializer required.")
        return 0
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
        print("FSR 4.1.1 Ultra Performance recovery failed: " + str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
