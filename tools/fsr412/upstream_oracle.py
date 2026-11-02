#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Extract a static replay inventory from the shipped FSR 4 Lite DLL.

This independently authored loader never loads a DLL or runs a shader. It reads
PE sections, follows file-backed shader-table pointers and asks DXC to decode
DXIL as data. AMD/upstream shader bytes and listings are external build assets,
not source files. The inventory is NOT evidence of Windows runtime behavior.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import struct
import subprocess
import tempfile

LOSSY_SHA256 = "d9b0a1f5781c2d633a327272f6f1c8607057df7cafd0233cf38c7395c8f7dbc5"
EXACT_SHA256 = "56896078aef294ea875f4b977695701d827a9273a616d1f5c501b9963e559bd7"
AMD_SHA256 = "d0dcccc74a43c44ba435b7a369b456e0970d8a4464e4bd683119b374f2c9fb46"
ENTRY_PREFIX = "fsr4_model_v07_fp8_no_scale_"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError("upstream oracle: " + message)


@dataclass(frozen=True)
class Section:
    name: str
    rva: int
    virtual_size: int
    file_offset: int
    file_size: int


class PE:
    """Bounds-checked PE32+ reader. Virtual-only bytes cannot be extracted."""
    def __init__(self, data: bytes):
        self.data = data
        require(len(data) >= 64 and data[:2] == b"MZ", "not a PE image")
        pe = struct.unpack_from("<I", data, 60)[0]
        require(pe + 24 <= len(data) and data[pe:pe + 4] == b"PE\0\0", "invalid PE header")
        count, optional_size = struct.unpack_from("<H", data, pe + 6)[0], struct.unpack_from("<H", data, pe + 20)[0]
        optional = pe + 24
        require(optional_size >= 112 and optional + optional_size + 40 * count <= len(data), "truncated headers")
        require(struct.unpack_from("<H", data, optional)[0] == 0x20b, "expected PE32+")
        self.image_base = struct.unpack_from("<Q", data, optional + 24)[0]
        self.sections = []
        for index in range(count):
            pos = optional + optional_size + 40 * index
            virtual_size, rva, file_size, file_offset = struct.unpack_from("<4I", data, pos + 8)
            require(file_offset + file_size <= len(data), "truncated section")
            self.sections.append(Section(data[pos:pos + 8].rstrip(b"\0").decode("ascii"), rva, virtual_size, file_offset, file_size))

    def offset(self, va: int, size: int = 1) -> int:
        require(size > 0, "empty address range")
        rva = va - self.image_base
        sections = [s for s in self.sections if s.rva <= rva and rva + size <= s.rva + s.file_size]
        require(len(sections) == 1, "pointer not in one file-backed section")
        section = sections[0]
        return section.file_offset + rva - section.rva

    def address(self, offset: int) -> int:
        sections = [s for s in self.sections if s.file_offset <= offset < s.file_offset + s.file_size]
        require(len(sections) == 1, "file offset outside sections")
        section = sections[0]
        return self.image_base + section.rva + offset - section.file_offset

    def shader_at(self, record: int) -> tuple[int, bytes]:
        require(0 <= record <= len(self.data) - 16, "table record outside file")
        size, pointer = struct.unpack_from("<QQ", self.data, record)
        require(32 <= size <= 5_000_000, "invalid table size")
        offset = self.offset(pointer, size)
        blob = self.data[offset:offset + size]
        require(valid_dxbc(blob), "table pointer is not a complete DXBC container")
        return offset, blob


def valid_dxbc(blob: bytes) -> bool:
    if len(blob) < 32 or blob[:4] != b"DXBC":
        return False
    version, size, count = struct.unpack_from("<III", blob, 20)
    if version != 1 or size != len(blob) or count > 64 or 32 + 4 * count > size:
        return False
    ranges = []
    for index in range(count):
        offset = struct.unpack_from("<I", blob, 32 + 4 * index)[0]
        if offset < 32 + 4 * count or offset + 8 > size:
            return False
        end = offset + 8 + struct.unpack_from("<I", blob, offset + 4)[0]
        if end > size or any(offset < other_end and start < end for start, other_end in ranges):
            return False
        ranges.append((offset, end))
    return True


def containers(pe: PE):
    for section in pe.sections:
        cursor, end = section.file_offset, section.file_offset + section.file_size
        while (cursor := pe.data.find(b"DXBC", cursor, end)) >= 0:
            if cursor + 32 <= end:
                size = struct.unpack_from("<I", pe.data, cursor + 24)[0]
                if 32 <= size <= 5_000_000 and cursor + size <= end:
                    blob = pe.data[cursor:cursor + size]
                    if valid_dxbc(blob):
                        yield cursor, blob
                        cursor += size
                        continue
            cursor += 4


def shader_records(reference: PE):
    """Match each named network container to its exact size/pointer record.

    This preserves old pass names even if the replacement is a stripped no-op.
    Container header hashes are opaque identifiers, NOT recomputed SHA-256.
    """
    data_section = [s for s in reference.sections if s.name == ".data"]
    require(len(data_section) == 1, "expected one shader table section")
    section = data_section[0]
    lo, hi = section.file_offset, section.file_offset + section.file_size
    result = []
    for offset, blob in containers(reference):
        entries = set(re.findall(rb"fsr4_model_v07_fp8_no_scale_(?:prepass|postpass|pass\d+(?:_post)?)\x00", blob))
        if not entries:
            continue
        require(len(entries) == 1, "ambiguous entry name")
        target = struct.pack("<QQ", len(blob), reference.address(offset))
        cursor, matches = lo, []
        while (cursor := reference.data.find(target, cursor, hi)) >= 0:
            if cursor % 8 == 0:
                matches.append(cursor)
            cursor += 1
        require(len(matches) == 1, f"expected one live record for {blob[4:20].hex()}, got {len(matches)}")
        program_version = None
        for chunk in range(struct.unpack_from("<I", blob, 28)[0]):
            chunk_offset = struct.unpack_from("<I", blob, 32 + 4 * chunk)[0]
            if blob[chunk_offset:chunk_offset + 4] == b"DXIL":
                program_version = struct.unpack_from("<I", blob, chunk_offset + 8)[0]
        result.append({"table_offset": matches[0], "reference_offset": offset,
                       "reference_program_version": program_version,
                       "reference_entry": next(iter(entries)).rstrip(b"\0").decode(),
                       "reference_header_hash": blob[4:20].hex(),
                       "reference_sha256": sha256(blob), "reference_bytes": len(blob)})
    return sorted(result, key=lambda item: item["table_offset"])


def disassemble(dxc: Path, blob: bytes) -> str:
    with tempfile.TemporaryDirectory(prefix="fsr412-oracle-") as work:
        source, output = Path(work) / "shader.dxil", Path(work) / "shader.ll"
        source.write_bytes(blob)
        completed = subprocess.run([str(dxc), "-dumpbin", str(source), "-Fc", str(output)], capture_output=True)
        require(completed.returncode == 0 and output.is_file(), "DXC failed: " + completed.stderr.decode(errors="replace")[-2000:])
        return output.read_text()


def listing_metadata(text: str) -> dict:
    """Retain reflection declarations and exact accesses, not guessed semantics."""
    entry = re.search(r"^define void @([\w.]+)\(", text, re.M)
    threads = re.search(r"NumThreads=\((\d+),\s*(\d+),\s*(\d+)\)", text)
    body = text[text.index("define void @"):text.index("\n}", text.index("define void @")) + 2] if entry else ""
    calls = Counter(re.findall(r"\bcall [^\n@]*@dx\.op\.([\w.]+)\(", body))
    resources, in_resources = [], False
    for line in text.splitlines():
        if line == "; Resource Bindings:":
            in_resources = True
            continue
        if in_resources and line.startswith("target datalayout"):
            break
        if in_resources and line.startswith(";"):
            fields = line[1:].split()
            if len(fields) >= 6 and fields[-6] in ("cbuffer", "sampler", "texture", "UAV"):
                name = " ".join(fields[:-6])
                kind, fmt, dimension, identifier, binding, count = fields[-6:]
                resources.append({"name": name, "type": kind, "format": fmt,
                                  "dimension": dimension, "id": identifier, "hlsl_binding": binding, "count": count})
    cbuffer_members = []
    for match in re.finditer(r"^;\s+([\w<>]+)\s+([\w.]+)(?:\[(\d+)\])?;\s*; Offset:\s*(\d+)(?: Size:\s*(\d+))?", text, re.M):
        cbuffer_members.append({"type": match[1], "name": match[2], "array_count": int(match[3]) if match[3] else None,
                                "byte_offset": int(match[4]), "byte_size": int(match[5]) if match[5] else None})
    operations = []
    for line in body.splitlines():
        if re.search(r"@dx\.op\.(?:rawBuffer(?:Load|Store)|cbufferLoad|createHandle|atomicBinOp)", line):
            # This is a file-backed disassembly observation, not original source.
            operations.append(line.strip())
    noop = bool(entry and not calls and re.fullmatch(r"define void @[^\n]+\{\s*(?:[\w.]+:\s*)?ret void\s*}", body))
    return {"entry": entry[1] if entry else None, "workgroup": [int(v) for v in threads.groups()] if threads else None,
            "noop": noop, "dx_operations": dict(sorted(calls.items())), "resources": resources,
            "cbuffer_members": cbuffer_members, "accesses": operations,
            "integer_literals": sorted(set(int(v) for v in re.findall(r"\bi32 (-?\d+)\b", body))),
            "listing_sha256": sha256(text.encode())}


def external_output(path: Path) -> Path:
    path = path.resolve()
    # Resolve the workspace root too: a build/ directory may itself be a
    # symlink (shared caches, release worktrees), and the containment check
    # must compare final destinations rather than the link's spelling.
    build = (Path(__file__).resolve().parents[2] / "build/fsr412_upstream").resolve()
    temporary = Path(tempfile.gettempdir()).resolve()
    require(path.is_relative_to(build) or path.is_relative_to(temporary), "assets must remain in build/fsr412_upstream or system temp")
    return path


def extract(reference_path: Path, dll_path: Path, output: Path, dxc: Path, exact_path: Path | None = None, workers: int = 4) -> dict:
    reference_data, data = reference_path.read_bytes(), dll_path.read_bytes()
    require(sha256(reference_data) == AMD_SHA256, "reference is not the pinned AMD 4.1.1 provider")
    require(sha256(data) == LOSSY_SHA256, "target is not the pinned RX7000 Lite Lossy DLL")
    reference, target = PE(reference_data), PE(data)
    exact = PE(exact_path.read_bytes()) if exact_path else None
    if exact:
        require(sha256(exact.data) == EXACT_SHA256, "wrong Exact sibling")
    output = external_output(output)
    output.mkdir(parents=True, exist_ok=True)
    records = shader_records(reference)
    assets = {}
    for record in records:
        for role, pe in (("reference", reference), ("lossy", target), ("exact", exact)):
            if pe is None:
                continue
            offset, blob = pe.shader_at(record["table_offset"])
            identity = sha256(blob)
            assets[identity] = blob
            record[role] = {"offset": offset, "sha256": identity, "header_hash": blob[4:20].hex(), "bytes": len(blob)}
        record["changed"] = record["reference"]["sha256"] != record["lossy"]["sha256"]
    def decode(item):
        identity, blob = item
        dxil, listing = output / (identity + ".dxil"), output / (identity + ".ll")
        dxil.write_bytes(blob)
        text = disassemble(dxc, blob)
        listing.write_text(text)
        return identity, listing_metadata(text)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        decoded = dict(pool.map(decode, sorted(assets.items())))
    initializers = {}
    for preset, offset in (("normal", 0xc04a00), ("ultraperformance", 0xc24a00)):
        blob = data[offset:offset + 131072]
        require(len(blob) == 131072, "truncated initializer")
        path = output / ("initializers-" + preset + ".bin")
        path.write_bytes(blob)
        initializers[preset] = {"path": path.name, "bytes": len(blob), "sha256": sha256(blob),
                                "dll_offset": offset, "binding": "t18", "lite_used_prefix_bytes": 27392}
    manifest = {"schema": "fsr412_upstream_oracle_v1", "authority": "shipped-artifact/static-DXC-decode",
                "initializers": initializers,
                "dll_sha256": sha256(data), "reference_dll_sha256": sha256(reference_data),
                "exact_dll_sha256": sha256(exact.data) if exact else None, "dxc_sha256": sha256(dxc.read_bytes()),
                "records": records, "shaders": decoded,
                "limitations": ["No DLL/shader/GPU execution; dispatch scheduling and resource allocation require separate provider or runtime evidence.",
                                "A table record proves replacement identity, not which permutation a Windows application selected.",
                                "DXIL->SPIR-V->Metal replay is not Windows-driver bit parity until independently tested."]}
    (output / "inventory.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def _spv_instructions(data: bytes):
    require(len(data) % 4 == 0 and len(data) >= 20, "invalid SPIR-V byte length")
    words = list(struct.unpack("<%dI" % (len(data) // 4), data))
    require(words[0] == 0x07230203, "invalid SPIR-V magic")
    result, cursor = [], 5
    while cursor < len(words):
        count, opcode = words[cursor] >> 16, words[cursor] & 65535
        require(count > 0 and cursor + count <= len(words), "truncated SPIR-V instruction")
        result.append((opcode, words[cursor + 1:cursor + count]))
        cursor += count
    return words[:5], result


def _spv_string(value: str) -> list[int]:
    data = value.encode() + b"\0"
    data += b"\0" * (-len(data) % 4)
    return list(struct.unpack("<%dI" % (len(data) // 4), data))


def metal_descriptor_abi(data: bytes, raw_resource_classes: dict[int, int] | None = None) -> tuple[bytes, list[dict]]:
    """Only rename descriptors and remap descriptor decorations.

    The converter CLI maps all D3D register classes to set 0, so t3/u3/b3
    collide. Separate them into SRV=0, UAV=1, CBV=2, sampler=3 namespaces.
    Each scalar/vector SSBO alias gets its own Metal argument, bound to the
    SAME underlying buffer by the host. No heap, BDA, memory or arithmetic
    instructions are modified. Sequential Metal slots are local to a shader.
    """
    header, instructions = _spv_instructions(data)
    types, variables, decorations, nonwritable = {}, {}, {}, set()
    for opcode, operands in instructions:
        if opcode in (25, 26, 28, 29, 30, 32):
            types[operands[0]] = (opcode, operands[1:])
        elif opcode == 59:
            variables[operands[1]] = (operands[0], operands[2])
        elif opcode == 71:
            decorations.setdefault(operands[0], {})[operands[1]] = operands[2:]
            if operands[1] == 24:
                nonwritable.add(operands[0])
        elif opcode == 72 and operands[2] == 24:
            nonwritable.add(operands[0])
    counters = Counter()
    bindings = []
    replacements = {}
    for variable, (pointer, storage) in sorted(variables.items()):
        deco = decorations.get(variable, {})
        if 33 not in deco:
            continue
        require(34 in deco and deco[34] == [0], "only register space 0 is supported by this replay ABI")
        opcode, operands = types[pointer]
        require(opcode == 32, "resource is not an OpTypePointer")
        pointee = operands[1]
        kind, type_operands = types[pointee]
        require(kind not in (28, 29), "descriptor arrays/heaps are not supported")
        register = deco[33][0]
        if storage == 0 and kind == 26:
            cls, bucket, prefix = 3, "sampler", "s"
        elif storage == 0 and kind == 25:
            sampled = type_operands[5]
            require(sampled in (1, 2), "unknown image access class")
            cls, bucket, prefix = (0 if sampled == 1 else 1), "texture", ("t" if sampled == 1 else "u")
        elif storage == 2 and kind == 30 and 2 in decorations.get(pointee, {}):
            cls, bucket, prefix = 2, "buffer", "b"
        elif storage == 12 and kind == 30:
            readonly = variable in nonwritable or pointee in nonwritable
            # Read-only optimization is not a D3D register class: a UAV can
            # be used only for reads (the above-4K original POST does this).
            cls = raw_resource_classes[register] if raw_resource_classes and register in raw_resource_classes else (0 if readonly else 1)
            require(cls in (0,1), "invalid raw resource class")
            bucket, prefix = "buffer", ("t" if cls == 0 else "u")
        else:
            raise ValueError(f"upstream oracle: unsupported descriptor type/storage {kind}/{storage}")
        slot = counters[bucket]
        counters[bucket] += 1
        name = f"oracle_{prefix}{register}_{bucket}{slot}"
        replacements[variable] = (cls, slot, name)
        bindings.append({"spirv_id": variable, "name": name, "d3d_register": prefix + str(register),
                         "register_space": 0, "resource_class": cls, "metal_kind": bucket,
                         "metal_slot": slot, "same_buffer_alias": prefix + str(register) if bucket == "buffer" else None})
    require(counters["buffer"] <= 31 and counters["texture"] <= 31 and counters["sampler"] <= 16, "Metal argument limit exceeded")
    # OpName belongs before type declarations, not among annotations.
    result, names_inserted = [], False
    for opcode, operands in instructions:
        if not names_inserted and (opcode in (71, 72) or 19 <= opcode <= 39):
            result.extend((5, [variable] + _spv_string(name)) for variable, (_, _, name) in replacements.items())
            names_inserted = True
        if opcode == 5 and operands[0] in replacements:
            continue
        if opcode == 71 and operands[0] in replacements:
            namespace, slot, _ = replacements[operands[0]]
            if operands[1] == 34:
                operands = operands[:2] + [namespace]
            elif operands[1] == 33:
                operands = operands[:2] + [slot]
        result.append((opcode, operands))
    words = header[:]
    for opcode, operands in result:
        words += [((len(operands) + 1) << 16) | opcode] + operands
    return struct.pack("<%dI" % len(words), *words), bindings


def translate(output: Path, identity: str, converter: Path, spirv_cross: Path, scratch_bytes: int | None = None) -> dict:
    """Produce external SPIR-V/MSL plus an explicit D3D-to-Metal binding ABI."""
    output = external_output(output)
    require(re.fullmatch(r"[0-9a-f]{64}", identity), "invalid shader identity")
    dxil = output / (identity + ".dxil")
    require(sha256(dxil.read_bytes()) == identity, "DXIL identity mismatch")
    raw, spv, msl = (output / (identity + suffix) for suffix in (".raw.spv", ".spv", ".metal"))
    command = [str(converter), str(dxil), "--output", str(raw), "--ssbo-uav", "--ssbo-srv", "--ssbo-scalar", "--no-bda", "--use-reflection-names"]
    completed = subprocess.run(command, capture_output=True)
    require(completed.returncode == 0 and raw.is_file(), "DXIL->SPIR-V failed: " + completed.stderr.decode(errors="replace")[-2000:])
    metadata = listing_metadata((output / (identity + ".ll")).read_text())
    raw_classes = {}
    for resource in metadata["resources"]:
        if resource["format"] == "byte" and resource["type"] in ("texture", "UAV"):
            register = int(resource["hlsl_binding"][1:])
            cls = 1 if resource["type"] == "UAV" else 0
            require(register not in raw_classes or raw_classes[register] == cls, "ambiguous raw register classes")
            raw_classes[register] = cls
    patched, bindings = metal_descriptor_abi(raw.read_bytes(), raw_classes)
    spv.write_bytes(patched)
    cross_command = [str(spirv_cross), str(spv), "--msl", "--msl-version", "30100", "--msl-decoration-binding", "--output", str(msl)]
    completed = subprocess.run(cross_command, capture_output=True)
    require(completed.returncode == 0 and msl.is_file(), "SPIR-V->Metal failed: " + completed.stderr.decode(errors="replace")[-2000:])
    completed = subprocess.run([str(spirv_cross), str(spv), "--reflect"], capture_output=True)
    require(completed.returncode == 0, "SPIR-V reflection failed")
    reflection = json.loads(completed.stdout)
    text = msl.read_text()
    unguarded = text
    (output / (identity + ".unguarded.metal")).write_text(unguarded)
    from oracle_io import guard_scalar_io
    if scratch_bytes is None:
        require(identity in ACTIVE_2160.values(), "standalone translation without an explicit allocation pins the selected 2160 stage set")
        scratch_bytes = 99820992
    require(scratch_bytes in (25027392, 99820992, 398707392), "unknown pinned scratch allocation")
    text, io_receipt = guard_scalar_io(text, bindings, {"t18": 131072, "u11": scratch_bytes})
    msl.write_text(text)
    for binding in bindings:
        attribute = f"[[{binding['metal_kind']}({binding['metal_slot']})]]"
        require(attribute in text, "MSL binding decoration was not preserved")
    entry = re.search(r"kernel void (\w+)\(", text)
    receipt = {"schema": "fsr412_upstream_translation_v1", "dxil_sha256": identity,
               "converter_sha256": sha256(converter.read_bytes()), "spirv_cross_sha256": sha256(spirv_cross.read_bytes()),
               "converter_command": command, "spirv_cross_command": cross_command,
               "raw_spirv_sha256": sha256(raw.read_bytes()), "spirv_sha256": sha256(patched),
               "unguarded_metal_sha256": sha256(unguarded.encode()), "io_guard": io_receipt,
               "metal_sha256": sha256(text.encode()), "entry": entry[1] if entry else None,
               "bindings": bindings, "reflection": reflection,
               "instruction_transform": "SPIR-V OpName/Binding/DescriptorSet decorations only; independent Metal resource IO bounds guards",
               "limitations": ["Translated source only; not compiled/executed on GPU by this tool.",
                               "All arguments with the same D3D buffer register MUST alias one physical buffer.",
                               "Per-word bounds guards return zero/discard for t18/u11 and physical-extent integer texture loads/stores; sampling and arithmetic remain unchanged."]}
    (output / (identity + ".translation.json")).write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


ACTIVE_2160 = {
    "pre": "9523e223136343d56c1c3b2c89fbdea29adf511343bfcb7fb35ba7afbead4bba",
    "pass1": "2638456df248932a18dfd82b556d2c06d8ae3ab0fcd6f67eeeb037782bf6a0ea",
    "pass2": "6a5d98fc27a87938ab35d09af117f62238355cdab06aeaf3a9b26d91c434fb5e",
    "border": "779dd783525dbe328bf6bfa06509f813f928f9d1c59edf6993fa4bd278a502de",
    "pass3": "4a8443a4c81abc7c08a64f4131e8a1aeb86e33c1f49c5be530c886d33507a238",
    "post": "a69f41a0bb37376178001a7f73087d4726e21c0d531e7635653277053ac4b268",
}
NOOP_SHA256 = "17afa4239aa66d37267373a5ef8d53f483e698d89a222d91fa3c1993fdc85316"


def scratch_geometry(tier: int) -> dict:
    """Tensor geometry observed in the pinned three-tier DXIL graph."""
    require(tier in (1080, 2160, 4320), "unsupported tensor tier")
    width, height = tier * 8 // 9, tier // 2
    stride, quarter_stride = 16 * (width + 2), 16 * (width // 2 + 2)
    plane, quarter_plane = stride * (height + 2), quarter_stride * (height // 2 + 2)
    return {"tier": tier, "half_width": width, "half_height": height, "half_row_bytes": stride,
            "half_plane_bytes": plane, "quarter_row_bytes": quarter_stride, "quarter_plane_bytes": quarter_plane,
            "allocation_bytes": 3 * plane, "skip_marker_byte_offset": plane + 128 if tier != 4320 else None,
            "saved_jitter_byte_offset": plane + 136 if tier != 4320 else None,
            "motion_owner_region_byte_offset": plane + stride if tier != 4320 else None,
            "saved_cnb_byte_offset": 3 * plane // 2 + stride + 16 if tier != 4320 else None,
            "neural_post_input_byte_offset": 2 * plane, "oob_store_sentinel": 134217728}


def contract_manifest(manifest: dict, dll_data: bytes, output: Path) -> dict:
    """A checked metadata contract, not a reimplementation of shader policy."""
    require(sha256(dll_data) == LOSSY_SHA256, "contract source identity mismatch")
    allocation_offsets = {1080: 0x80d6, 2160: 0x80d0, 4320: 0x80ca}
    geometry = {tier: scratch_geometry(tier) for tier in allocation_offsets}
    for tier, offset in allocation_offsets.items():
        require(dll_data[offset - 1] == 0xb8 and dll_data[offset + 4] == 0xc3, "allocation function instruction changed")
        require(struct.unpack_from("<I", dll_data, offset)[0] == geometry[tier]["allocation_bytes"], "allocation/tensor extent mismatch")
        geometry[tier]["allocation_immediate_file_offset"] = offset
    shaders = manifest["shaders"]
    int8 = [r for r in manifest["records"] if r["reference_program_version"] == 0x00050066]
    require(len(int8) == 426, "old INT8 table coverage changed")
    groups = {}
    for record in int8:
        entry = record["reference_entry"].removeprefix(ENTRY_PREFIX)
        groups.setdefault(entry, []).append(record)
    stages = [("prepass", "pre"), ("pass0_post", "noop")]
    for number in range(1, 13):
        replacement = {1: "pass1", 2: "pass2", 3: "border", 4: "pass3"}.get(number, "noop")
        stages.extend([(f"pass{number}", replacement), (f"pass{number}_post", "noop")])
    stages.append(("postpass", "post"))
    mapping = []
    for index, (old_entry, stage) in enumerate(stages):
        records = groups[old_entry]
        if stage == "noop":
            require(all(r["lossy"]["sha256"] == NOOP_SHA256 and shaders[r["lossy"]["sha256"]]["noop"] for r in records), "no-op mapping changed")
        mapping.append({"old_dispatch_index": index, "old_entry": old_entry, "replacement": stage,
                        "table_offsets": [r["table_offset"] for r in records],
                        "replacement_sha256": sorted(set(r["lossy"]["sha256"] for r in records))})
    require(len(mapping) == 27, "old graph stage count mismatch")
    active = []
    expected_entry = {"pre": "prepass", "pass1": "pass1", "pass2": "pass2", "border": "pass3", "pass3": "pass4", "post": "postpass"}
    for stage, identity in ACTIVE_2160.items():
        records = [r for r in int8 if r["lossy"]["sha256"] == identity and r["reference_entry"] == ENTRY_PREFIX + expected_entry[stage]]
        require(records and not shaders[identity]["noop"], "selected replay stage mapping changed")
        active.append({"stage": stage, "sha256": identity, "workgroup": shaders[identity]["workgroup"],
                       "table_offsets": [r["table_offset"] for r in records],
                       "reference_header_hashes": [r["reference_header_hash"] for r in records],
                       "translation_receipt": identity + ".translation.json"})
    result = {"schema": "fsr412_upstream_contract_v1", "source_dll_sha256": LOSSY_SHA256,
              "evidence": "Pinned PE bytes, unchanged table-record coordinates, DXC disassembly; no execution",
              "old27_mapping": mapping, "geometry": geometry,
              # Historical key; this set is AMD FFX_MLSR_AUTOEXPOSURE_ENABLED=1
              # (reads r_auto_exposure_texture only, no r_input_exposure t6).
              "active_2160_normal_di1_lm1_jm0_cs0_ae0": active,
              "active_2160_auto_exposure": 1,
              "constant_buffers": {"b0": {"bytes": 144, "type": "uint4[9]", "p2_and_p3_size_slot": 4},
                                   "b1": {"bytes": 104, "metal_padded_bytes": 112, "members": next(s["cbuffer_members"] for s in shaders.values() if len(s["cbuffer_members"]) == 18)}},
              "scratch_marker": {"words": [1511506142, 168889943], "saved_jitter_words": 2,
                                 "reset_byte_offset_in_b1": 72, "jitter_byte_offset_in_b1": 24,
                                 "up_threshold_f32_bits": 1042983597,
                                 "predicate": "nonzero sign-bit-negative jitter.x AND reset==0 AND (2*width_lr>=width OR NOT(sign-bit-negative jitter.y AND absbits(jitter.y)>1042983597))",
                                 "above_4k": "No marker path in every 4320 PRE permutation"},
              "saved_cnb": {"word_encoding_xor": 2779096485, "missing_condition": "any of the four encoded words is zero",
                            "miss": "recompute CNB from retained neural input at a motion-reprojected half-resolution coordinate",
                            "hit": "decode four INT8-packed words, optionally blend neighbouring cache cells for odd integer displacement",
                            "alias_warning": "Saved CNB range overlaps neural POST input. Cross-workgroup alias/order behavior is not established by static analysis."},
              "motion_owner": {"granularity_output_pixels": 4, "motion_entry_bytes": 8, "owner_entry_bytes": 4,
                               "2160_motion_parity_offsets_bytes": [192, 16384], "2160_owner_parity_offsets_bytes": [8192, 24576],
                               "row_bytes": 30752, "region_base": 33304416,
                               "hole_offsets": "For row r: d1=32*r+16, d2=d1+15392; insert 32 bytes at/after each diagonal, comparing after the first insertion",
                               "owner_packing_2160": "depth_key[12 bits]<<20 | cell_y<<10 | cell_x; atomic unsigned maximum",
                               "fine_motion_alpha": "on skipped PRE, even/even output pixels carry roundEven(dx)+64*roundEven(dy)+32 within +/-31; otherwise -2048",
                               "persistence_alpha": "other skipped PRE pixels and every neural-run PRE pixel carry the center Catmull-Rom history tap alpha unchanged as half, or zero if not reached"},
              "dispatch": {"pre": "ceil(outputW/16),ceil(outputH/16),1",
                           "pass1_provider_compatible": "ceil((outputW/2)/64),outputH/2,1; shader remaps group.x/group.y and has a border-only group (0,2,0)",
                           "pass2_minimal": "ceil((outputW/4)/64),outputH/4,1",
                           "border_minimal": "1,1,1; every nonzero group is a proven no-op; border still runs on skipped frames",
                           "pass3": "ceil((outputW/4)/64),outputH/4,1",
                           "post": "ceil(outputW/32),ceil(outputH/32),1; four output pixels per thread"},
              "replay_limits": ["Descriptor ABI translation must bind same-register SSBO aliases to one physical buffer.",
                                "Original D3D OOB stores use sentinel byte offset 134217728; naked Metal pointers do not preserve D3D bounds behavior.",
                                "No Windows provider dispatch, texture format, sampler state or driver output was observed.",
                                "Current native Lite Lossy temporal policy is independent/approximate and cannot be certified from initializer equality."]}
    from lossy_weights import decode_post, INITIALIZER_BYTES
    base = dll_data[0xc04a00:0xc04a00 + INITIALIZER_BYTES]
    post_text = (output / (ACTIVE_2160["post"] + ".ll")).read_text()
    post_tables, conditional_biases = decode_post(post_text, base)
    for bias in conditional_biases:
        if bias["layer"] == "A":
            bias["half_width"] = "b1.width / 2 (integer sdiv in original POST)"
            bias["half_height"] = "b1.height / 2 (integer sdiv in original POST)"
    ct_taps = [list(struct.unpack("<8i", post_tables["bCT"])) for _ in range(4)]
    for bias in conditional_biases:
        if bias["layer"] == "CT":
            ct_taps[bias["tap"]][bias["channel"]] = bias["conditional_bias"]
    result["conditional_post_biases"] = conditional_biases
    result["post_ct_biases_by_tap"] = ct_taps
    result["initializer_contract"] = {"runtime_full_t18": manifest["initializers"],
                                      "normal_lite_base_prefix_bytes": INITIALIZER_BYTES,
                                      "normal_lite_base_prefix_sha256": sha256(base),
                                      "trained_weights": "Baked literal operands in pass1/pass2/POST; full t18 remains AMD Lite base data"}
    (external_output(output) / "contract.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def replay_bindings_manifest(output: Path) -> dict:
    """Publish production-like loader metadata for the selected real permutation.

    Every resource slot comes from translation reflection, not a guessed root
    table. Other flags/tiers must be separately selected and verified.
    """
    output = external_output(output)
    contract = json.loads((output / "contract.json").read_text())
    resources = {
        "t18": {"name": "InitializerBuffer", "kind": "buffer", "bytes": 131072, "file": "initializers-normal.bin"},
        "u11": {"name": "ScratchBuffer", "kind": "buffer", "bytes": 99820992, "initial_state": "zero on context creation; persistent between frames"},
        "b0": {"name": "CsTensorSizes", "kind": "buffer", "bytes": 144, "shape": "uint4[9]", "slot4": "outputW/4,outputH/4,0,0"},
        "b1": {"name": "MLSR_Optimized_Constants", "kind": "buffer", "bytes": 104, "metal_allocation_bytes": 112},
        "t0": {"name": "r_history_color", "kind": "texture", "size": "output", "storage": "RGBA16Float replay fixture", "alias": "previous history ping-pong"},
        "t1": {"name": "r_velocity", "kind": "texture", "size": "render", "storage": "RG32Float replay fixture"},
        "t2": {"name": "r_depth", "kind": "texture", "size": "render", "storage": "R32Float replay fixture"},
        "t3": {"name": "r_input_color", "kind": "texture", "size": "render", "storage": "RGBA32Float replay fixture"},
        "t4": {"name": "r_recurrent_0", "kind": "texture", "size": "output", "storage": "RGBA8Unorm default replay fixture; selectable RGBA16Float", "alias": "previous recurrent ping-pong"},
        "t9": {"name": "r_reprojected_color", "kind": "texture", "size": "output", "alias": "u3"},
        "t17": {"name": "r_auto_exposure_texture", "kind": "texture", "size": [2, 1], "storage": "R32Float replay fixture", "values": "current exposure, previous exposure"},
        "u1": {"name": "rw_history_color", "kind": "texture", "size": "output", "storage": "RGBA16Float replay fixture", "alias": "next history ping-pong"},
        "u2": {"name": "rw_mlsr_output_color", "kind": "texture", "size": "output", "storage": "RGBA16Float replay fixture"},
        "u3": {"name": "rw_reprojected_color", "kind": "texture", "size": "output", "storage": "RGBA16Float replay fixture"},
        "u6": {"name": "rw_recurrent_0", "kind": "texture", "size": "output", "storage": "RGBA8Unorm default replay fixture; selectable RGBA16Float", "alias": "next recurrent ping-pong"},
        "s0": {"name": "g_history_sampler", "kind": "sampler", "state": "linear clamp-to-edge normalized replay fixture"},
    }
    kernels = {}
    for stage, identity in ACTIVE_2160.items():
        path = output / (identity + ".translation.json")
        require(path.is_file(), "translate every active stage before publishing bindings")
        profile_path = output / (identity + ".o0.translation.json")
        if stage in ("pass2", "pass3"):
            require(profile_path.is_file(), "FNB correctness references require an explicit -O0 profile")
            receipt = json.loads(profile_path.read_text())
        else:
            receipt = json.loads(path.read_text())
        require(receipt["dxil_sha256"] == identity, "translation receipt identity mismatch")
        stem = identity + ".o0" if stage in ("pass2", "pass3") else identity
        row = {"entry": receipt["entry"], "library": stem + ".metallib", "source": stem + ".metal",
               "optimization_profile": receipt.get("optimization_profile", "default"),
               "dxil_sha256": identity, "metal_sha256": receipt["metal_sha256"], "wgsize": json.loads((output / "inventory.json").read_text())["shaders"][identity]["workgroup"],
               "buffers": [], "textures": [], "samplers": [], "resource_reads": [], "resource_writes": []}
        for binding in receipt["bindings"]:
            register = binding["d3d_register"]
            resource = resources[register]
            bucket = {"buffer": "buffers", "texture": "textures", "sampler": "samplers"}[binding["metal_kind"]]
            row[bucket].append([binding["metal_slot"], resource["name"]])
            access = "resource_writes" if register.startswith("u") else "resource_reads"
            if register not in row[access]:
                row[access].append(register)
            if register == "u11" and register not in row["resource_reads"]:
                row["resource_reads"].append(register)
        library = output / row["library"]
        if library.is_file():
            row["library_sha256"] = sha256(library.read_bytes())
        kernels[stage] = row
    initializer = (output / "initializers-normal.bin").read_bytes()
    resources["t18"]["sha256"] = sha256(initializer)
    result = {"schema": "fsr412_original_shader_bindings_v1", "source_dll_sha256": LOSSY_SHA256,
              "options": {"tier": 2160, "ultra_performance": 0, "depth_inverted": 1, "low_res_mv": 1,
                          "jittered_mv": 0, "colorspace": 0, "auto_exposure": 1},
              "verified_permutation_only": True, "scratch_bytes": 99820992, "initializers": "initializers-normal.bin",
              "initializers_sha256": sha256(initializer), "kernels": kernels, "resources_by_register": resources,
              "active_order": list(ACTIVE_2160), "old27_mapping": contract["old27_mapping"], "dispatch": contract["dispatch"],
              "geometry": contract["geometry"]["2160"], "constant_buffers": contract["constant_buffers"],
              "conditional_post_biases": contract["conditional_post_biases"], "post_ct_biases_by_tap": contract["post_ct_biases_by_tap"],
              "limits": ["Only this selected actual flag/tier/preset permutation is ready; other options must fail closed.",
                         "Texture formats and sampler are explicit Metal replay choices; Windows allocation/sampler state was not observed.",
                         "Full 4K literal POST is nondeterministic in controlled Metal replay, consistent with savedCNB/live-input overlap; ordered/isolated modes require explicit adaptation labels.",
                         "P2/P3 use an explicit -O0 correctness profile; legacy default-optimized P2 GPU captures failed the original DXIL CPU SSA check.",
                         "No extra dense 32-byte/pixel post-input cache is part of the original DLL contract."]}
    helper_path=output/(ACTIVE_2160['pass1']+".border-only.translation.json")
    if helper_path.is_file():
        helper=json.loads(helper_path.read_text())
        row={"entry":helper["entry"],"library":helper["library"],"source":ACTIVE_2160['pass1']+".border-only.metal",
             "dxil_sha256":helper["dxil_sha256"],"metal_sha256":helper["metal_sha256"],"wgsize":helper["wgsize"],
             "dispatch_groups":helper["dispatch_groups"],"skip_marker_gate_preserved":helper["skip_marker_gate_preserved"],
             "execute_policy":helper["execute_policy"],"buffers":[],"textures":[],"samplers":[]}
        for binding in helper["bindings"]:
            row["buffers"].append([binding["metal_slot"],resources[binding["d3d_register"]]["name"]])
        if "library_sha256" in helper:row["library_sha256"]=helper["library_sha256"]
        result["kernels"]["pass1_border"]=row
        result["native_replacement_order"]=["pre","native_pass1_cnb","native_pass1_ds","pass1_border","native_pass2","border","native_pass3","post"]
    result["neural_replacement_contract"] = {
        "native_initializer_bytes": 27392,
        "native_primary_training_sha256": "c77c2d1a4d119ed76d94b8c23335490cc38280db3346c72396a1e1acb0523cfc",
        "not_runtime_t18": True,
        "operations": [
            {"name": "pass1_cnb", "original_stage": "pass1", "input_byte_offset": 0, "output_byte_offset": 33273664, "row_bytes": 30752},
            {"name": "pass1_downsample", "original_stage": "pass1", "input_byte_offset": 33273664, "output_byte_offsets": [66547328, 74889792], "output_row_bytes": 15392},
            {"name": "pass2_fnb", "original_stage": "pass2", "input_byte_offsets": [66547328, 74889792], "output_byte_offsets": [0, 8342464], "row_bytes": 15392},
            {"name": "pass3_fnb_upsample", "original_stage": "pass3", "input_byte_offsets": [0, 8342464], "skip_byte_offset": 33273664, "output_byte_offset": 66547328, "output_row_bytes": 30752},
        ],
        "must_preserve": ["original PRE", "original border on every frame", "original POST/CT/temporal math", "skip marker and saved jitter", "same pre-state for every neural byte A/B"],
        "conditional_biases": "POST missing-first-tap biases and CT subpixel biases remain separately represented, not flattened into one native initializer",
    }
    result["controlled_post_variants"] = {}
    for mode in ("ordered", "isolated"):
        variant_path = output / (ACTIVE_2160["post"] + "." + mode + ".translation.json")
        if variant_path.is_file():
            result["controlled_post_variants"][mode] = json.loads(variant_path.read_text())
    (output / "bindings.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dll", type=Path)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--exact", type=Path)
    parser.add_argument("--out", type=Path, default=Path("build/fsr412_upstream"))
    parser.add_argument("--dxc", type=Path, default=Path(__file__).resolve().parents[1] / "bin/dxc")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--classify", action="store_true", help="classify every active original INT8 permutation from typed DXIL evidence")
    parser.add_argument("--select", action="store_true", help="select and optionally build one verified original permutation")
    parser.add_argument("--tier", type=int, choices=(1080,2160,4320), default=2160)
    parser.add_argument("--preset", choices=("normal","ultraperformance"), default="normal")
    parser.add_argument("--ultra-performance", type=int, choices=(0,1), default=None, help="preset alias for production callers")
    for flag, default in (("depth-inverted",1),("low-res-mv",1),("jittered-mv",0),("auto-exposure",0)):
        parser.add_argument("--"+flag,type=int,choices=(0,1),default=default)
    parser.add_argument("--colorspace", type=int, choices=(0,1,2,3), default=0)
    parser.add_argument("--bundle", type=Path, help="external bundle directory for the selected actual shader set")
    parser.add_argument("--compile-selected", action="store_true", help="compile selected external shader data; never runs the GPU")
    parser.add_argument("--post-mode", choices=("literal","ordered","isolated"), default="isolated")
    parser.add_argument("--bindings-only", action="store_true", help="publish selected original resource/dispatch ABI from external translations")
    parser.add_argument("--contract-only", action="store_true", help="verify and publish contract from an existing inventory")
    parser.add_argument("--translate", action="append", default=[], metavar="SHA256", help="translate selected already-extracted shader assets only")
    parser.add_argument("--converter", type=Path)
    parser.add_argument("--spirv-cross", type=Path, default=Path(__file__).resolve().parents[1] / "bin/spirv-cross")
    args = parser.parse_args()
    if args.classify or args.select:
        require(sha256(args.dll.read_bytes()) == LOSSY_SHA256, "selection target DLL identity mismatch")
        if args.ultra_performance is not None:
            args.preset = "ultraperformance" if args.ultra_performance else "normal"
        from oracle_selection import classify, select
        index = classify(args.out)
        if args.select:
            options = {"tier":args.tier,"preset":args.preset,"depth_inverted":args.depth_inverted,"low_res_mv":args.low_res_mv,
                       "jittered_mv":args.jittered_mv,"colorspace":args.colorspace,"auto_exposure":args.auto_exposure}
            selected = select(index, options)
            if args.bundle:
                require(args.converter is not None, "selected bundle requires the caller's existing converter")
                from oracle_bundle import build_bundle
                manifest = build_bundle(args.out,args.bundle,selected,args.converter,args.spirv_cross,args.compile_selected,args.post_mode)
                print(json.dumps({"bundle":str(args.bundle),"options":options,"kernels":{k:v["dxil_sha256"] for k,v in manifest["kernels"].items()}}))
            else:
                print(json.dumps(selected,indent=2))
        else:
            print(json.dumps({"table_counts":index["table_counts"],"index":str(args.out/"permutation-index.json")}))
        return
    if args.bindings_only:
        result = replay_bindings_manifest(args.out)
        print(json.dumps({"bindings": str(args.out / "bindings.json"), "kernels": list(result["kernels"])}))
        return
    if args.contract_only:
        manifest = json.loads((args.out / "inventory.json").read_text())
        contract_manifest(manifest, args.dll.read_bytes(), args.out)
        print(json.dumps({"contract": str(args.out / "contract.json")}))
        return
    if args.translate:
        require(args.converter is not None, "translation requires caller-supplied converter")
        for identity in args.translate:
            receipt = translate(args.out, identity, args.converter, args.spirv_cross)
            print(json.dumps({"shader": identity, "entry": receipt["entry"], "bindings": receipt["bindings"]}))
        return
    require(1 <= args.workers <= 16, "workers out of range")
    manifest = extract(args.reference, args.dll, args.out, args.dxc, args.exact, args.workers)
    contract_manifest(manifest, args.dll.read_bytes(), args.out)
    print(json.dumps({"records": len(manifest["records"]), "unique_shaders": len(manifest["shaders"]),
                      "noops": sum(s["noop"] for s in manifest["shaders"].values()), "out": str(args.out)}))


if __name__ == "__main__":
    main()
