"""Independent CPU/layout and real-DXIL proofs for Lite Lossy recovery.

No DLL or GPU execution. Real-artifact tests are skipped when the separately
acquired release or DXC is unavailable. Synthetic fixtures contain generated
numbers, not vendor/trained coefficients.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import re
import struct
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("fsr412_lossy_weights", ROOT / "tools/fsr412/lossy_weights.py")
weights = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(weights)


def _generated_words(count, seed=1):
    return [(index * 104729 + seed * 257) & 0xffffffff for index in range(count)]


def _cnb_listing(copy=0, primary_bias=None, edge_bias=None, matrix_words=None):
    """Four-output-word groups, full/edge copies and bounded pointwise chains."""
    primary_bias = list(range(16)) if primary_bias is None else primary_bias
    edge_bias = primary_bias if edge_bias is None else edge_bias
    matrix_words = _generated_words(576, 1) if matrix_words is None else matrix_words
    bwords, cwords = _generated_words(128, 2), _generated_words(128, 3)
    lines, names = [], []
    prefix = f"copy{copy}."
    index = 0

    def dot(acc, operand, value):
        nonlocal index
        name = prefix + str(index)
        index += 1
        names.append(name)
        lines.append(f"  %{name} = call i32 @dx.op.dot4AddPacked.i32(i32 163, i32 {acc}, i32 %{operand}, i32 {value})")
        return "%" + name

    for interior in (True, False):
        accumulators = [str(value) for value in primary_bias]
        for tap in range(9):
            for channel in range(16):
                for word in range(4):
                    accumulators[channel] = dot(accumulators[channel], f"{prefix}input.{tap}.{word}", matrix_words[tap * 64 + channel * 4 + word])
            if not interior and tap == 0:
                for channel in reversed(range(16)):
                    lines.append(f"  %{prefix}phi.{channel} = phi i32 [ {edge_bias[channel]}, %skip ], [ {accumulators[channel]}, %inbounds ]")
                    accumulators[channel] = "%" + prefix + "phi." + str(channel)
    for start, outputs, inputs, word_values, bias_seed in ((1152, 32, 4, bwords, 100), (1280, 16, 8, cwords, 200)):
        for channel in range(outputs):
            accumulator = str(bias_seed + channel)
            for word in range(inputs):
                accumulator = dot(accumulator, f"{prefix}packed.{word}", word_values[channel * inputs + word])
    assert index == 1408
    return "\n".join(lines) + "\n"


def _post_fixture(copies=1, conditional=False):
    base = bytearray(weights.INITIALIZER_BYTES)
    edge_bias = list(range(16))
    primary_bias = [x + 1 for x in edge_bias] if conditional else edge_bias
    base[25344:25408] = weights._words(edge_bias)
    lines = [_cnb_listing(copy, primary_bias, edge_bias) for copy in range(copies)]
    expected_tail, biases = [], []
    index = copies * 1408
    for tap in range(4):
        source_words = _generated_words(32, 10 + tap)
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
                lines.append(f"  %{name} = call i32 @dx.op.dot4AddPacked.i32(i32 163, i32 {accumulator}, i32 %tail_input.{word}, i32 {source_words[channel * 4 + word]})\n")
                accumulator = "%" + name
    return "".join(lines), bytes(base), weights._words(expected_tail), biases


def _store_listing():
    lines = []
    for block in range(100):
        array = "A" if block < 36 else "B" if block < 68 else "C"
        for index in range(16):
            pointer = f"ptr.{block}.{index}"
            lines.append(f"  %{pointer} = getelementptr [16 x i32], [16 x i32]* %{array}, i32 0, i32 {index}")
            lines.append(f"  store i32 {block * 16 + index}, i32* %{pointer}, align 4")
    return "\n".join(lines) + "\n"


class LossyLayoutTests(unittest.TestCase):
    def test_grouped_conv_and_pointwise_channel_order(self):
        for outputs, blocks, convolution in ((16, 9, True), (64, 2, False), (32, 4, False)):
            count = outputs * blocks * 4
            values = list(range(count))
            decoded = struct.unpack("<%dI" % count, weights._grouped_matrix(values, outputs, blocks, convolution))
            for group in range(outputs // 4):
                for block in range(blocks):
                    for channel in range(4):
                        for word in range(4):
                            source = (((group * blocks + block) * 4 + channel) * 4 + word)
                            output = group * 4 + channel
                            destination = (block * outputs + output) * 4 + word if convolution else output * blocks * 4 + block * 4 + word
                            self.assertEqual(decoded[destination], source)

    def test_literal_fnb_arrays_are_bounded_and_no_eager_shift_evaluator(self):
        source = _store_listing() + "  %unrelated = shl i32 1, -1\n"
        blocks = weights._literal_store_blocks(source)
        self.assertEqual(len(blocks), 100)
        self.assertEqual(blocks[0], ("A", list(range(16))))
        self.assertEqual(blocks[-1], ("C", list(range(1584, 1600))))
        for before, after in (("i32 15\n", "i32 16\n"), ("i32 0\n", "i32 1\n")):
            with self.assertRaisesRegex(ValueError, "initialization order"):
                weights._literal_store_blocks(source.replace(before, after, 1))
        with self.assertRaisesRegex(ValueError, "store count"):
            weights._literal_store_blocks(source.replace("  store i32 0, i32* %ptr.0.0, align 4\n", "", 1))
        with self.assertRaisesRegex(ValueError, "bounded weight-array"):
            weights._literal_store_blocks(source + "store i32 7, i32* %unknown, align 4\n")

    def test_cnb_full_border_and_accumulator_chains(self):
        text = _cnb_listing()
        dots = weights._dots(text, 1408)
        tables = weights._cnb_weights(dots)
        self.assertEqual(tables["A"], weights._words(_generated_words(576, 1)))
        self.assertEqual(tables["B"], weights._words(_generated_words(128, 2)))
        self.assertEqual(tables["C"], weights._words(_generated_words(128, 3)))
        bad = text.replace("i32 163,", "i32 164,", 1)
        with self.assertRaisesRegex(ValueError, "opcode"):
            weights._dots(bad, 1408)
        position = dots[576].start()
        bad = text[:position] + text[position:].replace("i32 257)", "i32 258)", 1)
        with self.assertRaisesRegex(ValueError, "interior/border"):
            weights._cnb_weights(weights._dots(bad, 1408))
        first = dots[1153]
        broken = text[:first.start()] + text[first.start():].replace(first[3], "%unknown_accumulator", 1)
        with self.assertRaisesRegex(ValueError, "accumulator chain"):
            weights._cnb_weights(weights._dots(broken, 1408))

    def test_post_ct_head_order_and_all_bias_taps(self):
        text, base, tail, biases = _post_fixture()
        tables, conditional = weights.decode_post(text, base)
        self.assertEqual(tables["CT"], tail)
        self.assertEqual(tables["bCT"], weights._words(biases[0]))
        self.assertEqual(tables["bA"], base[25344:25408])
        self.assertEqual(conditional, [])

    def test_cloned_cnb_and_conditional_biases_are_not_dropped(self):
        text, base, tail, biases = _post_fixture(4, True)
        tables, conditional = weights.decode_post(text, base)
        first = [entry for entry in conditional if entry["layer"] == "A"]
        taps = [entry for entry in conditional if entry["layer"] == "CT"]
        self.assertEqual(len(first), 16)
        self.assertEqual(len(taps), 3)
        self.assertEqual(tables["CT"], tail)
        self.assertEqual(tables["bCT"], weights._words(biases[0]))
        self.assertEqual(first[0]["primary_bias"], 1)
        self.assertEqual(first[0]["edge_bias"], 0)
        self.assertIn("cnb_half_x - 1", first[0]["predicate"])
        self.assertEqual([entry["tap_xy"] for entry in taps], [[1, 0], [0, 1], [1, 1]])
        self.assertEqual([entry["conditional_bias"] for entry in taps], [biases[tap][0] for tap in (1, 2, 3)])
        altered = text.replace("%copy1.1152 = call i32 @dx.op.dot4AddPacked.i32(i32 163, i32 100,", "%copy1.1152 = call i32 @dx.op.dot4AddPacked.i32(i32 163, i32 101,", 1)
        with self.assertRaisesRegex(ValueError, "cloned CNBs"):
            weights.decode_post(altered, base)

    def test_unknown_post_shapes_bias_roots_and_extra_operands_fail_closed(self):
        text, base, _, _ = _post_fixture()
        dots = weights._dots(text, 1536)
        for old, new in (("i32 257)", "i32 %weight_from_unknown_buffer)"),
                         ("i32 0, i32 %copy0.input", "i32 %unknown_bias, i32 %copy0.input")):
            self.assertIn(old, text)
            with self.assertRaises(ValueError):
                weights.decode_post(text.replace(old, new, 1), base)
        with self.assertRaisesRegex(ValueError, "neural shape"):
            weights.decode_post(text + dots[0][0] + "\n", base)
        with self.assertRaisesRegex(ValueError, "conditional bias PHIs"):
            weights.decode_post(text.replace("phi i32", "select i1", 1), base)

    def test_int8_packed_signed_literals_and_initializer_ranges(self):
        self.assertEqual(weights._words([-1, -2147483648, 4294967295]), struct.pack("<III", 0xffffffff, 0x80000000, 0xffffffff))
        with self.assertRaises(ValueError):
            weights._words([1 << 32])
        ranges = [(offset, offset + count) for items in weights.LAYOUT.values() for _, offset, count in items]
        for (start, end), (next_start, _) in zip(sorted(ranges), sorted(ranges)[1:]):
            self.assertLessEqual(end, next_start)
        self.assertEqual(dict((name, off) for name, off, _ in weights.LAYOUT["post"])["bCT"], 27264)
        self.assertLess(max(end for _, end in ranges), weights.INITIALIZER_BYTES)

    def test_unknown_identity_is_rejected_before_using_dxc(self):
        with mock.patch.object(weights, "disassemble", side_effect=AssertionError("DXC must not run")):
            with self.assertRaisesRegex(ValueError, "unrecognized DLL"):
                weights._decode_dll(b"not a DLL", "invalid dxc")
            with self.assertRaisesRegex(ValueError, "unknown base"):
                weights.recover_primary_training(b"not a DLL", "invalid dxc", bytes(27392))

    def test_atomic_external_artifacts_are_complete_and_unchanged_writes_noop(self):
        with tempfile.TemporaryDirectory(prefix="fsr412-lossy-test-") as directory:
            path = Path(directory) / "nested/weights.bin"
            weights._write_external(path, b"generated test data")
            self.assertEqual(path.read_bytes(), b"generated test data")
            inode = path.stat().st_ino
            weights._write_external(path, b"generated test data")
            self.assertEqual(path.stat().st_ino, inode)
            weights._write_external(path, b"different generated data")
            self.assertEqual(path.read_bytes(), b"different generated data")
            self.assertEqual(sorted(p.name for p in path.parent.iterdir()), ["weights.bin"])

    def test_dxbc_container_bounds_fail_closed(self):
        header = b"DXBC" + bytes(16) + struct.pack("<III", 1, 44, 1) + struct.pack("<I", 36)
        container = header + b"DXIL" + struct.pack("<I", 0)
        self.assertEqual(len(container), 44)
        self.assertEqual(list(weights.dxbc_containers(b"prefix" + container)), [(6, container)])
        for changed in (container[:24] + struct.pack("<I", 999999) + container[28:],
                        container[:32] + struct.pack("<I", 43) + container[36:],
                        container[:-4] + struct.pack("<I", 10)):
            self.assertEqual(list(weights.dxbc_containers(changed)), [])


class LossyActualDXILTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path("/tmp/fsr4-lite-lossy-20261101")
        cls.exact = root / "4.1.2l-exact/amd_fidelityfx_upscaler_dx12.dll"
        cls.lossy = root / "4.1.2l-lossy/amd_fidelityfx_upscaler_dx12.dll"
        cls.dxc = ROOT / "tools/bin/dxc"
        if not all(path.is_file() for path in (cls.exact, cls.lossy, cls.dxc)):
            raise unittest.SkipTest("acquire pinned Exact/Lossy release and DXC for real-artifact decoder proof")
        raw = cls.exact.read_bytes()
        cls.common = raw[0xc04a00:0xc04a00 + 27392]
        cls.ultra = raw[0xc24a00:0xc24a00 + 27392]
        cls.exact_common, cls.exact_receipt = weights.recover(cls.exact, cls.dxc, cls.common)
        cls.lossy_common, cls.lossy_receipt = weights.recover_primary_training(cls.lossy, cls.dxc, cls.common)

    def test_every_exact_table_bias_tail_matches_amd_initializer(self):
        self.assertEqual(self.exact_common, self.common)
        self.assertEqual(weights.digest(self.exact_common), weights.COMMON_SHA256)
        self.assertEqual(len(self.exact_receipt["containers"]), 66)
        self.assertTrue(all(table["changed_bytes"] == 0 for table in self.exact_receipt["tables"]))
        self.assertEqual(self.exact_receipt["conditional_biases"], [])
        self.assertFalse(self.exact_receipt["trained"])
        self.assertEqual(self.exact_receipt["source_dll_sha256"], weights.EXACT_DLL_SHA256)

    def test_real_lossy_sha_complete_training_and_unused_bytes_preserved(self):
        result, receipt = self.lossy_common, self.lossy_receipt
        self.assertEqual(len(result), 27392)
        self.assertEqual(weights.digest(result), weights.LOSSY_INITIALIZER_SHA256)
        self.assertEqual(receipt["schema"], "fsr412_lite_lossy_import_v1")
        self.assertEqual(receipt["source_dll_sha256"], weights.DLL_SHA256)
        self.assertEqual(receipt["initializers"], {"initializers-lite-lossy.bin": weights.LOSSY_INITIALIZER_SHA256})
        self.assertTrue(receipt["trained"])
        self.assertEqual(receipt["supported_trained_tiers"], [1080, 2160])
        changed = {index for index, (a, b) in enumerate(zip(result, self.common)) if a != b}
        allowed = {index for table in receipt["tables"] for index in range(table["offset"], table["offset"] + table["bytes"])}
        self.assertLessEqual(changed, allowed)
        self.assertEqual(len(changed), 154)
        self.assertEqual(result[:1152], self.common[:1152])
        self.assertEqual(result[27296:], self.common[27296:])
        self.assertEqual({(t["pass"], t["table"]) for t in receipt["tables"] if t["changed_bytes"]},
                         {("pass1", "B"), ("pass1", "C"), ("pass2", "B"), ("pass2", "C"),
                          ("post", "bA"), ("post", "B"), ("post", "bB"), ("post", "C"), ("post", "bC"), ("post", "bCT")})

    def test_independent_tier_variants_all_presets_and_quantizer_evidence(self):
        receipt = self.lossy_receipt
        containers = receipt["containers"]
        self.assertEqual(len(containers), 62)
        self.assertEqual(len({c["container_sha256"] for c in containers}), 62)
        self.assertEqual({(c["stage"], c["preset"], c["tier"]) for c in containers},
                         {(s, p, t) for s in weights.LAYOUT for p in ("normal", "ultraperformance") for t in weights.TIERS})
        for item in containers:
            self.assertRegex(item["container_sha256"], r"^[0-9a-f]{64}$")
            evidence = item["quantizer_evidence"]
            self.assertRegex(evidence["listing_sha256"], r"^[0-9a-f]{64}$")
            self.assertIn("nearest-even", evidence["rounding"])
            if item["stage"] == "pass1" and item["preset"] == "normal":
                self.assertEqual(evidence["mapping"]["A"]["round_even_right_shift"], 7)
                self.assertEqual(evidence["mapping"]["B"]["round_even_right_shift"], 9)
            if item["stage"] == "post":
                self.assertEqual(evidence["mapping"]["CT2D_tail"]["accumulator_to_float_scale"], "1/1024")

    def test_post_firsttap_and_subpixel_biases_declared_not_silently_averaged(self):
        receipt = self.lossy_receipt
        self.assertFalse(receipt["shader_edge_parity"])
        firsttap = [c for c in receipt["conditional_biases"] if c["layer"] == "A"]
        ct = [c for c in receipt["conditional_biases"] if c["layer"] == "CT"]
        self.assertEqual(len(firsttap), 16)
        self.assertEqual(len(ct), 12)
        baseline = struct.unpack_from("<16i", self.common, 25344)
        primary = struct.unpack_from("<16i", self.lossy_common, 25344)
        for item in firsttap:
            self.assertEqual(item["edge_bias"], baseline[item["channel"]])
            self.assertEqual(item["primary_bias"], primary[item["channel"]])
            self.assertIn("first_3x3_tap", item["condition"])
        taps = receipt["post_ct_biases_by_tap"]
        self.assertEqual(len(taps), 4)
        self.assertTrue(all(len(tap) == 8 for tap in taps))
        self.assertEqual(tuple(taps[0]), struct.unpack_from("<8i", self.lossy_common, 27264))
        for item in ct:
            self.assertEqual(item["conditional_bias"], taps[item["tap"]][item["channel"]])
            self.assertEqual(item["primary_bias"], taps[0][item["channel"]])
        with mock.patch.object(weights, "recover_primary_training", return_value=(self.lossy_common, receipt)):
            with self.assertRaisesRegex(ValueError, "use recover_primary_training explicitly"):
                weights.recover(self.lossy, self.dxc, self.common)

    def test_real_llvm_ct_semantics_filter_first_amd_layout_not_swapped(self):
        receipt = self.lossy_receipt
        self.assertEqual(receipt["post_ct_head_order"], ["filter_float4", "recurrent_float4"])
        self.assertEqual(receipt["post_ct_bias_byte_offsets"], {"filter": 27264, "recurrent": 27280})
        self.assertEqual(receipt["post_ct_channel_order"][:4], ["filter.x", "filter.y", "filter.z", "filter.w"])
        raw = self.lossy.read_bytes()
        offset, size = weights._section(raw)
        for _, container in weights.dxbc_containers(raw, offset, offset + size):
            if weights.digest(container) != "e25cc24c676935c8ead71ec124942f82763799003b277c52af9067f6ff63dde9":
                continue
            text = weights.disassemble(self.dxc, container)
            tail = list(weights._DOT.finditer(text))[-128:]
            for tap in range(4):
                recurrent = tail[tap * 32:tap * 32 + 16]
                filtering = tail[tap * 32 + 16:tap * 32 + 32]
                rec_roots = [int(recurrent[channel * 4][3]) for channel in range(4)]
                filter_roots = [int(filtering[channel * 4][3]) for channel in range(4)]
                self.assertEqual(rec_roots, receipt["post_ct_biases_by_tap"][tap][4:])
                self.assertEqual(filter_roots, receipt["post_ct_biases_by_tap"][tap][:4])
                rec_weights = weights._words([int(dot[5]) for dot in recurrent])
                filter_weights = weights._words([int(dot[5]) for dot in filtering])
                start = 26752 + tap * 128
                self.assertEqual(filter_weights, self.common[start:start + 64])
                self.assertEqual(rec_weights, self.common[start + 64:start + 128])
                # Independently establish head semantics from consumers, not
                # from bias values or decoder labels: recurrent has four 1/x
                # sigmoids; the filter head has width 2/x and tanh/cross terms.
                recurrent_consumer = text[recurrent[-1].end():filtering[0].start()]
                self.assertEqual(recurrent_consumer.count("fdiv fast float 1.000000e+00,"), 4)
                end = tail[(tap + 1) * 32].start() if tap < 3 else len(text)
                filter_consumer = text[filtering[-1].end():end]
                self.assertIn("fdiv fast float 2.000000e+00,", filter_consumer)
                self.assertIn("fsub fast float", filter_consumer)
            self.assertEqual(tuple(receipt["post_ct_biases_by_tap"][0]), struct.unpack_from("<8i", self.lossy_common, 27264))
            break
        else:
            self.fail("pinned trained POST LLVM container unavailable")

    def test_upstream_up_is_explicitly_base_not_claimed_trained(self):
        result, receipt = weights.recover_primary_training(self.lossy, self.dxc, self.ultra)
        self.assertEqual(result, self.ultra)
        self.assertEqual(weights.digest(result), weights.ULTRA_SHA256)
        self.assertFalse(receipt["trained"])
        self.assertEqual(receipt["supported_trained_tiers"], [])
        self.assertEqual(receipt["conditional_biases"], [])
        self.assertTrue(all(table["changed_bytes"] == 0 for table in receipt["tables"]))

    def test_changed_quantizer_and_matrix_shapes_rejected_in_real_listing(self):
        raw = self.lossy.read_bytes()
        offset, size = weights._section(raw)
        for _, container in weights.dxbc_containers(raw, offset, offset + size):
            if b"fsr4_model_v07_fp8_no_scale_pass1" not in container:
                continue
            text = weights.disassemble(self.dxc, container)
            if not re.search(r"define void @fsr4_model_v07_fp8_no_scale_pass1", text):
                continue
            try:
                tables = weights.decode_pass1(text, self.common)
            except ValueError:
                continue
            if tables["A"] != weights._table_map("pass1", self.common)["A"]:
                continue
            self.assertIn("ashr i32", text)
            altered = re.sub(r"ashr i32 ([^,]+), (\d+)", r"ashr i32 \1, 30", text, count=1)
            with self.assertRaisesRegex(ValueError, "quantizer shape"):
                weights._quantizers("pass1", "normal", altered)
            break
        else:
            self.fail("normal trained pass1 listing unavailable")


if __name__ == "__main__":
    unittest.main()
