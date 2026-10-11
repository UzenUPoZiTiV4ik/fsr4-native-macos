"""Numerical invariants and fail-closed source transforms for the lossy model."""

from pathlib import Path
import hashlib
import json
import re
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    import lossy
    import lossy_shaders
finally:
    sys.path.pop(0)


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "third_party/fsr411/shaders/reimpl411"


def position(tap, input_channel, output_channel=0):
    return (tap * 16 + input_channel) * 16 + output_channel


def spatial_sum(matrix, input_channel, output_channel):
    return sum(matrix[position(tap, input_channel, output_channel)] for tap in range(9))


class FoldingTests(unittest.TestCase):
    def test_equal_norm_and_distance_ties_use_row_major_tap(self):
        matrix = [0] * 2304
        for tap in range(9):
            for quartet in range(4):
                matrix[position(tap, quartet * 4)] = 1
        original = matrix[:]
        result, stats = lossy.fold_conv3x3(matrix, 4)
        for quartet in range(4):
            self.assertEqual(result[position(0, quartet * 4)], 0)
            self.assertEqual(result[position(1, quartet * 4)], 2)
            self.assertEqual(result[position(3, quartet * 4)], 1)
        self.assertEqual(stats["dropped_per_channel"], [4] + [0] * 15)
        self.assertEqual(matrix, original)

    def test_distance_tie_prefers_larger_original_norm(self):
        matrix = [10] * 2304
        for tap, weight in ((0, 1), (1, 2), (3, -3)):
            for component in range(4):
                matrix[position(tap, component)] = weight if component == 0 else 0
        result, _ = lossy.fold_conv3x3(matrix, 1)
        self.assertEqual(result[position(0, 0)], 0)
        self.assertEqual(result[position(1, 0)], 2)
        self.assertEqual(result[position(3, 0)], -2)

    def test_int8_saturation_is_reported_and_does_not_wrap(self):
        result, stats = lossy.fold_conv3x3([127] * 2304, 4)
        self.assertEqual(stats["saturated_components"], 256)
        self.assertEqual(stats["dropped_words"], 64)
        self.assertEqual(set(result), {0, 127})
        self.assertEqual(spatial_sum(result, 0, 0), 8 * 127)

    def test_zero_words_are_not_counted_and_each_quartet_keeps_a_tap(self):
        result, stats = lossy.fold_conv3x3([0] * 2304, 32)
        self.assertEqual(stats["dropped_words"], 0)
        result, stats = lossy.fold_conv3x3([1] * 2304, 32)
        self.assertEqual(stats["dropped_words"], 512)
        for channel in range(16):
            self.assertEqual(spatial_sum(result, channel, 0), 9)

    def test_actual_model_preserves_dc_except_reported_saturation(self):
        for stage, count in lossy.FOLD_WORDS.items():
            source = (SOURCE / ("reimpl_pass%d.hlsl" % stage)).read_text()
            name = "W1" if stage in (1, 12) else "W0"
            body = re.search(r"static const uint " + name + r"\[36\]\[16\]\s*=\s*\{(.*?)\};", source, re.S)[1]
            words = [int(value, 16) for value in re.findall(r"0x([0-9a-fA-F]+)u", body)]
            matrix = lossy.literal_to_matrix(words)
            self.assertEqual(lossy.matrix_to_literal(matrix), words)
            result, stats = lossy.fold_conv3x3(matrix, count)
            self.assertEqual(stats["dropped_words"], 16 * count)
            self.assertEqual(stats["saturated_components"], 18 if stage == 10 else 0)
            if stage != 10:
                for ic in range(16):
                    for oc in range(16):
                        self.assertEqual(spatial_sum(matrix, ic, oc), spatial_sum(result, ic, oc))

    def test_all_baked_tables_match_the_published_lossy_receipt(self):
        receipt = json.loads((ROOT / "third_party/fsr411/lossy_model_manifest.json").read_text())
        for stage in receipt["passes"]:
            source = (SOURCE / ("reimpl_pass%d.hlsl" % stage["pass"])).read_text()
            transformed = lossy.apply_hlsl_lossy(source, stage["pass"])
            tables = re.findall(r"static const uint W\d+\[\d+\]\[16\]\s*=\s*\{(.*?)\};", transformed, re.S)
            words = [int(value, 16) for table in tables for value in re.findall(r"0x([0-9a-fA-F]+)u", table)]
            digest = hashlib.sha256(b"".join(word.to_bytes(4, "little") for word in words)).hexdigest()
            self.assertEqual(len(words), stage["baked_word_count"])
            self.assertEqual(digest, stage["baked_weights_sha256"])

    def test_invalid_shapes_and_unsigned_weights_are_rejected(self):
        for matrix, count in (([0] * 2303, 4), ([255] * 2304, 4), ([0] * 2304, 33), ([0] * 2304, -1)):
            with self.assertRaises(ValueError):
                lossy.fold_conv3x3(matrix, count)


class SourceTransformsTests(unittest.TestCase):
    def test_only_common_and_shared_up_passes_change(self):
        for preset in ("normal", "ultraperformance"):
            for stage in range(1, 13):
                path = SOURCE / ("reimpl_pass%d.hlsl" % stage)
                source = path.read_text()
                transformed = lossy.apply_hlsl_lossy(source, stage, preset)
                self.assertEqual(transformed != source, stage in lossy.rounding_passes(preset))
                self.assertEqual(lossy.apply_hlsl_lossy(transformed, stage, preset), transformed)
                if stage in lossy.rounding_passes(preset):
                    # Float residual/output rounding is outside requant_shift.
                    self.assertEqual(re.findall(r"round\([^;]+", transformed), re.findall(r"round\([^;]+", source))

    def test_unknown_quantizer_and_table_shapes_fail(self):
        source = (SOURCE / "reimpl_pass1.hlsl").read_text()
        with self.assertRaises(ValueError):
            lossy.apply_hlsl_lossy(source.replace("int requant_shift(", "int other_quantizer("), 1)
        with self.assertRaises(ValueError):
            lossy.apply_hlsl_lossy(source.replace("W1[36][16]", "W1[35][16]"), 1)

    def test_pre_post_are_idempotent_and_unknown_shapes_fail(self):
        for stage, transform in (("pre", lossy_shaders.apply_lossy_pre), ("post", lossy_shaders.apply_lossy_post)):
            source = (SOURCE / (stage + "_rev_411.hlsl")).read_text()
            result = transform(source, 1080)
            self.assertEqual(transform(result, 1080), result)
            with self.assertRaises(ValueError):
                transform(result, 2160)
            with self.assertRaises(ValueError):
                transform("", 1080)


if __name__ == "__main__":
    unittest.main()
