"""CPU proofs for the pinned November 22 Lite Lossy import flow.

No DLL, GPU, network, NumPy, or upstream Python code is used. Synthetic LLVM
fixtures always run. Tests marked as artifact tests use only already imported
files in build/ and skip cleanly on machines that have not fetched the
optional trained data.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import re
import struct
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

from fsr412 import fetch_lossy_weights as fetch
from fsr412 import lossy_weights as dxil
from fsr412 import model_data
from fsr412 import native_model
from fsr412 import nov22_weights as nov22


def generated_words(count, seed=1):
    return [(index * 104729 + seed * 257) & 0xffffffff for index in range(count)]


def cnb_listing(copy=0, primary_bias=None, edge_bias=None, matrix_words=None):
    """Four-output-word groups, full/edge copies and bounded pointwise chains."""
    primary_bias = list(range(16)) if primary_bias is None else primary_bias
    edge_bias = primary_bias if edge_bias is None else edge_bias
    matrix_words = generated_words(576, 1) if matrix_words is None else matrix_words
    b_words = generated_words(128, 2)
    c_words = generated_words(128, 3)
    lines = []
    prefix = "copy%d." % copy
    index = 0

    def dot(accumulator, operand, value):
        nonlocal index
        name = prefix + str(index)
        index += 1
        lines.append(
            "  %{name} = call i32 @dx.op.dot4AddPacked.i32(i32 163, i32 {acc}, i32 %{operand}, i32 {value})".format(
                name=name,
                acc=accumulator,
                operand=operand,
                value=value,
            )
        )
        return "%" + name

    for interior in (True, False):
        accumulators = [str(value) for value in primary_bias]
        for tap in range(9):
            for channel in range(16):
                for word in range(4):
                    operand = "%sinput.%d.%d" % (prefix, tap, word)
                    weight = matrix_words[tap * 64 + channel * 4 + word]
                    accumulators[channel] = dot(accumulators[channel], operand, weight)
            if not interior and tap == 0:
                for channel in reversed(range(16)):
                    lines.append(
                        "  %{prefix}phi.{channel} = phi i32 [ {edge}, %skip ], [ {acc}, %inbounds ]".format(
                            prefix=prefix,
                            channel=channel,
                            edge=edge_bias[channel],
                            acc=accumulators[channel],
                        )
                    )
                    accumulators[channel] = "%" + prefix + "phi." + str(channel)

    for start, outputs, inputs, word_values, bias_seed in (
        (1152, 32, 4, b_words, 100),
        (1280, 16, 8, c_words, 200),
    ):
        del start
        for channel in range(outputs):
            accumulator = str(bias_seed + channel)
            for word in range(inputs):
                accumulator = dot(accumulator, "%spacked.%d" % (prefix, word), word_values[channel * inputs + word])

    assert index == 1408
    return "\n".join(lines) + "\n"


def post_fixture(copies=1, conditional=False):
    base = bytearray(dxil.INITIALIZER_BYTES)
    edge_bias = list(range(16))
    primary_bias = [value + 1 for value in edge_bias] if conditional else edge_bias
    base[25344:25408] = dxil._words(edge_bias)
    lines = [cnb_listing(copy, primary_bias, edge_bias) for copy in range(copies)]
    expected_tail = []
    biases = []
    index = copies * 1408

    for tap in range(4):
        source_words = generated_words(32, 10 + tap)
        expected_tail.extend(source_words[16:] + source_words[:16])
        # Source order is recurrent, filter. Initializer order is the reverse.
        roots = list(range(300, 308))
        if conditional and tap:
            roots[4] += tap
        biases.append(roots[4:] + roots[:4])
        for channel in range(8):
            accumulator = str(roots[channel])
            for word in range(4):
                name = "tail." + str(index)
                index += 1
                lines.append(
                    "  %{name} = call i32 @dx.op.dot4AddPacked.i32(i32 163, i32 {acc}, i32 %tail_input.{word}, i32 {weight})\n".format(
                        name=name,
                        acc=accumulator,
                        word=word,
                        weight=source_words[channel * 4 + word],
                    )
                )
                accumulator = "%" + name

    return "".join(lines), bytes(base), dxil._words(expected_tail), biases


class SyntheticPostDecoderTests(unittest.TestCase):
    def test_cnb_full_border_and_accumulator_chains(self):
        text = cnb_listing()
        dots = dxil._dots(text, 1408)
        tables = dxil._cnb_weights(dots)
        self.assertEqual(tables["A"], dxil._words(generated_words(576, 1)))
        self.assertEqual(tables["B"], dxil._words(generated_words(128, 2)))
        self.assertEqual(tables["C"], dxil._words(generated_words(128, 3)))

        bad_opcode = text.replace("i32 163,", "i32 164,", 1)
        with self.assertRaisesRegex(ValueError, "opcode"):
            dxil._dots(bad_opcode, 1408)

        position = dots[576].start()
        bad_split = text[:position] + text[position:].replace("i32 257)", "i32 258)", 1)
        with self.assertRaisesRegex(ValueError, "interior/border"):
            dxil._cnb_weights(dxil._dots(bad_split, 1408))

        first = dots[1153]
        broken = text[:first.start()] + text[first.start():].replace(first[3], "%unknown_accumulator", 1)
        with self.assertRaisesRegex(ValueError, "accumulator chain"):
            dxil._cnb_weights(dxil._dots(broken, 1408))

    def test_post_ct_head_order_and_all_bias_taps(self):
        text, base, tail, biases = post_fixture()
        tables, conditional = dxil.decode_post(text, base)
        self.assertEqual(tables["CT"], tail)
        self.assertEqual(tables["bCT"], dxil._words(biases[0]))
        self.assertEqual(tables["bA"], base[25344:25408])
        self.assertEqual(conditional, [])

    def test_cloned_cnb_and_conditional_biases_are_not_dropped(self):
        text, base, tail, biases = post_fixture(4, True)
        tables, conditional = dxil.decode_post(text, base)
        first = [entry for entry in conditional if entry["layer"] == "A"]
        taps = [entry for entry in conditional if entry["layer"] == "CT"]
        self.assertEqual(len(first), 16)
        self.assertEqual(len(taps), 3)
        self.assertEqual(tables["CT"], tail)
        self.assertEqual(tables["bCT"], dxil._words(biases[0]))
        self.assertEqual(first[0]["primary_bias"], 1)
        self.assertEqual(first[0]["edge_bias"], 0)
        self.assertIn("cnb_half_x - 1", first[0]["predicate"])
        self.assertEqual([entry["tap_xy"] for entry in taps], [[1, 0], [0, 1], [1, 1]])
        self.assertEqual([entry["conditional_bias"] for entry in taps], [biases[tap][0] for tap in (1, 2, 3)])

        old = "%copy1.1152 = call i32 @dx.op.dot4AddPacked.i32(i32 163, i32 100,"
        new = "%copy1.1152 = call i32 @dx.op.dot4AddPacked.i32(i32 163, i32 101,"
        altered = text.replace(old, new, 1)
        with self.assertRaisesRegex(ValueError, "cloned CNBs"):
            dxil.decode_post(altered, base)

    def test_unknown_post_shapes_bias_roots_and_extra_operands_fail_closed(self):
        text, base, _, _ = post_fixture()
        dots = dxil._dots(text, 1536)
        replacements = (
            ("i32 257)", "i32 %weight_from_unknown_buffer)"),
            ("i32 0, i32 %copy0.input", "i32 %unknown_bias, i32 %copy0.input"),
        )
        for old, new in replacements:
            self.assertIn(old, text)
            with self.assertRaises(ValueError):
                dxil.decode_post(text.replace(old, new, 1), base)

        with self.assertRaisesRegex(ValueError, "neural shape"):
            dxil.decode_post(text + dots[0][0] + "\n", base)
        with self.assertRaisesRegex(ValueError, "conditional bias PHIs"):
            dxil.decode_post(text.replace("phi i32", "select i1", 1), base)

    def test_int8_packed_signed_literals_and_post_initializer_ranges(self):
        packed = dxil._words([-1, -2147483648, 4294967295])
        self.assertEqual(packed, struct.pack("<III", 0xffffffff, 0x80000000, 0xffffffff))
        with self.assertRaises(ValueError):
            dxil._words([1 << 32])

        ranges = [(offset, offset + count) for _, offset, count in dxil.LAYOUT["post"]]
        for (start, end), (next_start, _) in zip(sorted(ranges), sorted(ranges)[1:]):
            self.assertLessEqual(end, next_start)
        self.assertEqual(dict((name, off) for name, off, _ in dxil.LAYOUT["post"])["bCT"], 27264)
        self.assertLess(max(end for _, end in ranges), dxil.INITIALIZER_BYTES)
        self.assertEqual(sum(count for _, _, count in dxil.LAYOUT["post"]), 4128)

    def test_dxbc_container_bounds_fail_closed(self):
        header = b"DXBC" + bytes(16) + struct.pack("<III", 1, 44, 1) + struct.pack("<I", 36)
        container = header + b"DXIL" + struct.pack("<I", 0)
        self.assertEqual(len(container), 44)
        self.assertEqual(list(dxil.dxbc_containers(b"prefix" + container)), [(6, container)])

        mutated = (
            container[:24] + struct.pack("<I", 999999) + container[28:],
            container[:32] + struct.pack("<I", 43) + container[36:],
            container[:-4] + struct.pack("<I", 10),
        )
        for changed in mutated:
            self.assertEqual(list(dxil.dxbc_containers(changed)), [])

    def test_tier_row_pitch_mapping_is_exact(self):
        cases = (
            ("pass1", 15392, 1080),
            ("pass1", 30752, 2160),
            ("pass1", 61472, 4320),
            ("pass2", 7712, 1080),
            ("pass2", 15392, 2160),
            ("pass2", 30752, 4320),
        )
        for stage, pitch, tier in cases:
            with self.subTest(stage=stage, pitch=pitch):
                self.assertEqual(dxil._tier("rows %d" % pitch, stage), tier)
        with self.assertRaisesRegex(ValueError, "tier"):
            dxil._tier("no tensor row pitch here", "post")


class November22PinsAndErrorTests(unittest.TestCase):
    def test_release_pins_are_complete_unique_sha256_values(self):
        pins = (
            nov22.ARCHIVE_SHA256,
            nov22.DLL_SHA256,
            nov22.LOSSY_INITIALIZER_SHA256,
            nov22.BIAS_MATRIX_SHA256,
            nov22.BUFFER_SHA256,
            dxil.COMMON_SHA256,
        )
        for value in tuple(pins) + tuple(nov22.POST_CONTAINERS.values()):
            self.assertRegex(value, r"^[0-9a-f]{64}$")
        self.assertEqual(len(set(pins)), len(pins))
        self.assertEqual(set(nov22.POST_CONTAINERS), {1080, 2160})
        self.assertEqual(nov22.RELEASE, "dll-2026-11-22")
        self.assertIn(nov22.RELEASE, nov22.URL)
        self.assertEqual(fetch.MEMBER, "4.1.2l-lossy/amd_fidelityfx_upscaler_dx12.dll")
        self.assertIs(fetch.current, nov22)

    def test_unknown_identity_is_rejected_before_using_dxc(self):
        with tempfile.TemporaryDirectory(prefix="fsr412-nov22-test-") as directory:
            folder = Path(directory)
            dll = folder / "bad.dll"
            base = folder / "base.bin"
            dll.write_bytes(b"not the pinned DLL")
            base.write_bytes(bytes(dxil.INITIALIZER_BYTES))
            with mock.patch.object(dxil, "disassemble", side_effect=AssertionError("DXC must not run")):
                with self.assertRaisesRegex(ValueError, "unexpected Lite Lossy DLL"):
                    nov22.recover(dll, folder / "missing-dxc", base)

    def test_ratio_buffer_mismatch_is_rejected_before_dxil_scanning(self):
        with tempfile.TemporaryDirectory(prefix="fsr412-nov22-test-") as directory:
            folder = Path(directory)
            image = bytes(4096)
            base = bytes(dxil.INITIALIZER_BYTES)
            dll = folder / "shape.dll"
            base_path = folder / "base.bin"
            dll.write_bytes(image)
            base_path.write_bytes(base)
            with mock.patch.object(nov22, "DLL_SHA256", dxil.digest(image)), \
                 mock.patch.object(dxil, "COMMON_SHA256", dxil.digest(base)), \
                 mock.patch.object(dxil, "disassemble", side_effect=AssertionError("DXC must not run")):
                with self.assertRaisesRegex(ValueError, "ratio buffers"):
                    nov22.recover(dll, folder / "missing-dxc", base_path)

    def test_fetcher_requires_the_amd_base_before_archive_or_network_work(self):
        with tempfile.TemporaryDirectory(prefix="fsr412-fetch-test-") as directory:
            folder = Path(directory)
            output = folder / "out"
            argv = [
                "fetch_lossy_weights.py",
                "--root", str(folder),
                "--outdir", str(output),
                "--dll", str(folder / "unused.dll"),
            ]
            with mock.patch.object(sys, "argv", argv):
                with self.assertRaisesRegex(ValueError, "Import the official AMD Lite model first"):
                    fetch.main()
            self.assertFalse((output / "initializers-lite-lossy.bin").exists())

    def test_fetcher_atomic_and_streaming_digest_helpers(self):
        with tempfile.TemporaryDirectory(prefix="fsr412-fetch-test-") as directory:
            path = Path(directory) / "nested" / "weights.bin"
            path.parent.mkdir()
            payload = b"generated test data" * 1000
            fetch.atomic(path, payload)
            self.assertEqual(path.read_bytes(), payload)
            self.assertEqual(fetch.digest(path), hashlib.sha256(payload).hexdigest())
            self.assertEqual(sorted(item.name for item in path.parent.iterdir()), ["weights.bin"])

    def test_validation_and_native_packing_reject_wrong_input_identities(self):
        data = bytes(dxil.INITIALIZER_BYTES)
        receipt = {
            "schema": model_data.SCHEMA,
            "source_dll_sha256": nov22.DLL_SHA256,
            "initializers": {model_data.FILE: nov22.LOSSY_INITIALIZER_SHA256},
            "post_ct_biases_by_tap": [[0] * 8 for _ in range(4)],
        }
        with self.assertRaisesRegex(ValueError, "pinned release"):
            model_data.validate_lossy_data(data, receipt)
        with self.assertRaisesRegex(ValueError, "unknown Lite native profile"):
            native_model.model(1080, "normal", data, "not-a-lite-profile")
        with self.assertRaisesRegex(ValueError, "initializer identity mismatch"):
            native_model.model(1080, "normal", data, "int8-lite-lossy")


LOSSY_DIRECTORY = ROOT / "build/fsr412_lossy"
LOSSY_PATH = LOSSY_DIRECTORY / model_data.FILE
RECEIPT_PATH = LOSSY_DIRECTORY / "import.json"
BASE_PATH = ROOT / "build/fsr412_model/initializers-lite.bin"
ARTIFACTS_AVAILABLE = LOSSY_PATH.is_file() and RECEIPT_PATH.is_file() and BASE_PATH.is_file()


@unittest.skipUnless(ARTIFACTS_AVAILABLE, "fetch the optional Lite Lossy data first")
class November22ArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = LOSSY_PATH.read_bytes()
        cls.receipt = json.loads(RECEIPT_PATH.read_text())
        cls.base = BASE_PATH.read_bytes()

    def test_imported_initializer_and_receipt_match_every_november22_pin(self):
        self.assertEqual(len(self.data), dxil.INITIALIZER_BYTES)
        self.assertEqual(dxil.digest(self.data), nov22.LOSSY_INITIALIZER_SHA256)
        self.assertEqual(dxil.digest(self.base), dxil.COMMON_SHA256)
        self.assertEqual(self.receipt["schema"], model_data.SCHEMA)
        self.assertEqual(self.receipt["source_dll_sha256"], nov22.DLL_SHA256)
        self.assertEqual(self.receipt["initializers"][model_data.FILE], nov22.LOSSY_INITIALIZER_SHA256)
        self.assertEqual(self.receipt["ratio_buffer_offsets"], list(nov22.BUFFER_OFFSETS))
        self.assertEqual(self.receipt["ratio_buffer_sha256"], nov22.BUFFER_SHA256)
        self.assertEqual(self.receipt["post_containers"], {str(key): value for key, value in nov22.POST_CONTAINERS.items()})
        self.assertEqual(self.receipt["supported_trained_tiers"], [1080, 2160])
        self.assertFalse(self.receipt["windows_dll_executed"])
        self.assertFalse(self.receipt["upstream_dll_parity"])

    def test_validation_preserves_complete_subpixel_and_border_bias_evidence(self):
        matrix = model_data.validate_lossy_data(self.data, self.receipt)
        matrix_json = json.dumps(matrix, separators=(",", ":")).encode()
        self.assertEqual(hashlib.sha256(matrix_json).hexdigest(), nov22.BIAS_MATRIX_SHA256)
        self.assertEqual(tuple(matrix[0]), struct.unpack_from("<8i", self.data, 27264))

        conditional = self.receipt["conditional_biases"]
        first_tap = [entry for entry in conditional if entry["layer"] == "A"]
        ct_taps = [entry for entry in conditional if entry["layer"] == "CT"]
        self.assertEqual(len(first_tap), 9)
        self.assertEqual(len(ct_taps), 8)
        self.assertEqual(len(conditional), 17)
        for entry in first_tap:
            self.assertEqual(entry["condition"], "first_3x3_tap_out_of_bounds")
            self.assertIn("cnb_half_x - 1", entry["predicate"])
            self.assertNotEqual(entry["primary_bias"], entry["edge_bias"])
        for entry in ct_taps:
            self.assertEqual(entry["condition"], "post_subpixel_tap")
            self.assertEqual(entry["conditional_bias"], matrix[entry["tap"]][entry["channel"]])
            self.assertEqual(entry["primary_bias"], matrix[0][entry["channel"]])
            self.assertEqual(entry["tap_xy"], [entry["tap"] & 1, entry["tap"] >> 1])

    def test_initializer_selection_uses_training_only_for_supported_normal_tiers(self):
        for tier in (1080, 2160):
            path, data, trained = model_data.initializer_for(ROOT, tier, "normal", "int8-lite-lossy")
            self.assertEqual(path, LOSSY_PATH)
            self.assertEqual(data, self.data)
            self.assertTrue(trained)
        for tier, preset in ((4320, "normal"), (1080, "ultraperformance")):
            path, data, trained = model_data.initializer_for(ROOT, tier, preset, "int8-lite-lossy")
            self.assertNotEqual(path, LOSSY_PATH)
            self.assertNotEqual(dxil.digest(data), nov22.LOSSY_INITIALIZER_SHA256)
            self.assertFalse(trained)

    def test_native_model_lossy_quantizers_and_conditional_post_contract(self):
        passes, post = native_model.model(1080, "normal", self.data, "int8-lite-lossy")
        for key in (1, 2, 3):
            block = passes[key]
            for layer in (0, 1):
                quantizer = block["q"][layer]
                self.assertTrue(quantizer[3] & 4)
                count = len(block["bias_values"][layer])
                offset = block["initializer_bias_offsets"][layer]
                raw = struct.unpack_from("<%di" % count, self.data, offset)
                half = 1 << (quantizer[2] - 1)
                self.assertEqual(block["bias_values"][layer], [value - half for value in raw])

        self.assertEqual(len(post["bias_values"]), 4)
        self.assertEqual(post["bias_values"][3], list(struct.unpack_from("<16i", self.base, 25344)))
        self.assertTrue(post["q"][0][3] & 8)
        self.assertEqual(
            post["first_tap_bias_policy"],
            "AMD bias when first 3x3 tap is outside input; trained bias otherwise",
        )

    def test_native_pack_writes_hash_consistent_complete_artifacts_atomically(self):
        with tempfile.TemporaryDirectory(prefix="fsr412-native-pack-") as directory:
            output = Path(directory) / "model"
            manifest = native_model.pack(1080, "normal", self.data, output, "int8-lite-lossy")
            weights = (output / "fsr411_native_weights.bin").read_bytes()
            params = (output / "fsr411_native_params.bin").read_bytes()
            written = json.loads((output / "fsr411_native_manifest.json").read_text())

            self.assertEqual(written, manifest)
            self.assertEqual(manifest["schema"], "fsr411_native_v1")
            self.assertEqual(manifest["profile"], "int8-lite-lossy")
            self.assertTrue(manifest["trained"])
            self.assertEqual(manifest["initializer_sha256"], nov22.LOSSY_INITIALIZER_SHA256)
            self.assertEqual(hashlib.sha256(weights).hexdigest(), manifest["weights_sha256"])
            self.assertEqual(hashlib.sha256(params).hexdigest(), manifest["params_sha256"])
            self.assertEqual(manifest["weights_bytes"], len(weights))
            self.assertEqual(manifest["params_bytes"], len(params))
            self.assertIn("conditional original border biases", manifest["passes"]["1"]["coefficient_provenance"])
            self.assertIn("first_tap_bias_policy", manifest["auxiliary_passes"]["post_cnb"])
            self.assertEqual(sorted(item.name for item in output.iterdir()), [
                ".fsr412_native.lock",
                "fsr411_native_manifest.json",
                "fsr411_native_params.bin",
                "fsr411_native_weights.bin",
            ])

    def test_receipt_tampering_is_rejected_by_validation(self):
        mutated = copy.deepcopy(self.receipt)
        mutated["post_ct_biases_by_tap"][1][0] += 1
        with self.assertRaisesRegex(ValueError, "subpixel biases"):
            model_data.validate_lossy_data(self.data, mutated)

        mutated = copy.deepcopy(self.receipt)
        mutated["initializers"][model_data.FILE] = dxil.digest(self.base)
        with self.assertRaisesRegex(ValueError, "pinned release"):
            model_data.validate_lossy_data(self.data, mutated)


if __name__ == "__main__":
    unittest.main()
