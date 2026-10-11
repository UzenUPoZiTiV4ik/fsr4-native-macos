"""Importer corruption/provenance checks; no GPU or network is required."""

import hashlib
import importlib.util
import json
import struct
import tempfile
import unittest
from pathlib import Path


MODULE = Path(__file__).with_name("import_model.py")
SPEC = importlib.util.spec_from_file_location("fsr411_import_model", MODULE)
importer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(importer)


def make_container(version=0x00050066, entry="fsr4_model_v07_fp8_no_scale_pass1"):
    # The test exercises container framing, not LLVM bitcode decoding.
    chunks = [(b"DXIL", struct.pack("<6I", version, 6, 0x4C495844, 0x100, 16, 0)),
              (b"PSV0", entry.encode("ascii") + b"\0")]
    offset = 32 + 4 * len(chunks)
    offsets = []
    payload = bytearray()
    for tag, body in chunks:
        offsets.append(offset + len(payload))
        payload += tag + struct.pack("<I", len(body)) + body
    header = b"DXBC" + bytes(16) + struct.pack("<III", 1, offset + len(payload), len(chunks))
    return header + struct.pack("<2I", *offsets) + bytes(payload)


class ContainerValidationTests(unittest.TestCase):
    def test_embedded_container_uses_relative_offsets(self):
        prefix = b"PE-data-not-a-shader" * 3
        blob = make_container()
        parsed = importer.parse_container(prefix + blob, len(prefix))
        self.assertEqual(parsed["size"], len(blob))
        self.assertEqual(parsed["program_version"], 0x00050066)
        rows = importer.inventory_int8_shaders(prefix + blob)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["offset"], len(prefix))
        self.assertEqual(rows[0]["sha256"], hashlib.sha256(blob).hexdigest())

    def test_truncated_container_rejected(self):
        self.assertIsNone(importer.parse_container(make_container()[:-1], 0))

    def test_overlapping_chunks_rejected(self):
        blob = bytearray(make_container())
        first = struct.unpack_from("<I", blob, 32)[0]
        struct.pack_into("<I", blob, 36, first)
        self.assertIsNone(importer.parse_container(blob, 0))

    def test_chunk_pointing_into_header_rejected(self):
        blob = bytearray(make_container())
        struct.pack_into("<I", blob, 32, 20)
        self.assertIsNone(importer.parse_container(blob, 0))

    def test_fp8_entry_name_does_not_select_fp8_program(self):
        fp8 = make_container(version=0x00050064)
        int8 = make_container(version=0x00050066)
        rows = importer.inventory_int8_shaders(fp8 + int8)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["offset"], len(fp8))

    def test_duplicate_permutation_bytes_keep_both_source_offsets(self):
        blob = make_container()
        rows = importer.inventory_int8_shaders(blob + blob)
        self.assertEqual([row["offset"] for row in rows], [0, len(blob)])
        self.assertEqual(rows[0]["oracle_path"], rows[1]["oracle_path"])

    def test_unrelated_compute_shader_not_imported(self):
        self.assertEqual(importer.inventory_int8_shaders(make_container(entry="unrelated_compute")), [])


class ProvenanceTests(unittest.TestCase):
    def test_wrong_dll_never_accepted_as_fsr411(self):
        with self.assertRaisesRegex(ValueError, "expected official FSR"):
            importer.inspect_official_payload(b"MZ" + make_container())

    def test_common_model_cannot_substitute_for_ultraperformance(self):
        source = importer.DEFAULT_OUTPUT
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            for model in importer.INITIALIZERS.values():
                (output / model["path"]).write_bytes((source / model["path"]).read_bytes())
            importer.verify_initializers(output)
            (output / "initializers-ultraperformance.bin").write_bytes(
                (output / "initializers.bin").read_bytes())
            with self.assertRaisesRegex(ValueError, "ultraperformance_int8 initializer"):
                importer.verify_initializers(output)

    def test_tampered_source_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            data = b"expected shader source\n"
            (output / "source.hlsl").write_bytes(data)
            manifest = {"files": [{"path": "source.hlsl", "sha256": importer.sha256(data)}]}
            (output / "source_manifest.json").write_text(json.dumps(manifest))
            self.assertEqual(importer.verify_sources(output), 1)
            (output / "source.hlsl").write_bytes(data + b"changed")
            with self.assertRaisesRegex(ValueError, "differs from pinned"):
                importer.verify_sources(output)

    def test_manifest_cannot_escape_source_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            manifest = {"files": [{"path": "../outside", "sha256": "0" * 64}]}
            (output / "source_manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "unsafe source path"):
                importer.verify_sources(output)


if __name__ == "__main__":
    unittest.main()
