"""Independent UP arithmetic and reconstruction checks; no GPU or network."""

import json
from pathlib import Path
import struct
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    import recover_ultraperformance as recovery
finally:
    sys.path.pop(0)


CONTRACTS = json.loads(Path(__file__).with_name("native_ultraperformance_contracts.json").read_text())
ROOT = Path(__file__).resolve().parents[2]


def f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def round_ratio_even(numerator, denominator):
    quotient, remainder = divmod(abs(numerator), denominator)
    if 2 * remainder > denominator or (2 * remainder == denominator and quotient % 2):
        quotient += 1
    return -quotient if numerator < 0 else quotient


def narrow(value, flags):
    if flags & 2:
        value = ((value + 32768) % 65536) - 32768
    return max(-128, min(127, value))


def float_tail(accumulator, residual, tail):
    rs, scale, output = tail["float"]
    value = f32(f32(f32(residual) * rs) + f32(f32(accumulator) * scale))
    return narrow(round(f32(value * output)), tail["q"][3])


def integer_tail(accumulator, residual, tail):
    a, r, shift, flags = tail["q"]
    numerator = accumulator * (1 << a) + residual * (1 << r)
    return narrow(round_ratio_even(numerator, 1 << shift), flags)


class QuantizationContractsTests(unittest.TestCase):
    def test_up_integer_tails_match_separate_float32_operations(self):
        for stage in CONTRACTS["stages"]:
            tails = stage["tails"] + ([stage["ct_tail"]] if "ct_tail" in stage else [])
            for tail in tails:
                residuals = [0] if stage["kind"] == "ds" else [-128, -127, -3, -1, 0, 1, 3, 126, 127]
                step = 1 << tail["q"][2]
                accumulators = [-65537, -32769, -32768, -16385, -1, 0, 1, 16385, 32768, 65537]
                # Both signs of exact half ties, adjacent integers and odd/even quotients.
                accumulators += [base * step + step // 2 + delta
                                 for base in (-129, -128, -3, -2, -1, 0, 1, 2, 126, 127)
                                 for delta in (-1, 0, 1)]
                for residual in residuals:
                    for accumulator in accumulators:
                        with self.subTest(stage=stage["pass"], channels=tail["channels"],
                                          accumulator=accumulator, residual=residual):
                            self.assertEqual(integer_tail(accumulator, residual, tail),
                                             float_tail(accumulator, residual, tail))

    def test_downsample3_keeps_distinct_channel_scales(self):
        stage = next(stage for stage in CONTRACTS["stages"] if stage["pass"] == 3)
        self.assertEqual([integer_tail(192, 0, tail) for tail in stage["tails"]], [2, 1])
        self.assertEqual([integer_tail(-192, 0, tail) for tail in stage["tails"]], [-2, -1])

    def test_post_tail_does_not_wrap_to_int16(self):
        post = next(stage for stage in CONTRACTS["stages"] if stage["pass"] == "post")["tails"][0]
        self.assertEqual(float_tail(32769 * 64, 0, post), 127)
        self.assertEqual(integer_tail(32769 * 64, 0, post), 127)
        self.assertEqual(integer_tail(-32769 * 64, 0, post), -128)


class ReconstructionTests(unittest.TestCase):
    def test_patch_fails_for_missing_or_ambiguous_expression(self):
        for text in ("something else", "old + old"):
            with self.assertRaisesRegex(ValueError, "expected one source expression"):
                recovery.patch_once(text, "old", "new")

    def test_weight_reconstruction_cannot_drop_words(self):
        source = "static const uint W1[1][16] = { { " + ", ".join("0u" for _ in range(16)) + " } };"
        with self.assertRaisesRegex(ValueError, "truncated"):
            recovery.patch_weights(source, [0] * 15)
        with self.assertRaisesRegex(ValueError, "unconsumed"):
            recovery.patch_weights(source, [0] * 17)

    def test_post_extracts_swapped_dot_operands_and_signed_seeds(self):
        lines = []
        for i in range(960):
            seed = str(i - 48) if i < 96 else "%1"
            weight = str(i - 480)
            operands = "%2, i32 " + weight if i < 832 else weight + ", i32 %2"
            lines.append("@dx.op.dot4AddPacked.i32(i32 163, i32 %s, i32 %s)" % (seed, operands))
        weights, seeds = recovery.post_tables("\n".join(lines))
        self.assertEqual(weights, list(range(-480, 480)))
        self.assertEqual(seeds, list(range(-48, 48)))

    def test_every_embedded_up_table_differs_from_common(self):
        for tier, directory in recovery.TIERS.items():
            for stage in recovery.WEIGHT_PASSES:
                filename = "reimpl_pass%d%s.hlsl" % (stage, "" if tier == 1080 else "_" + str(tier))
                common = ROOT / "third_party/fsr411/shaders" / directory / filename
                ultra = ROOT / "third_party/fsr411/ultraperformance" / directory / filename
                with self.subTest(tier=tier, stage=stage):
                    a, b = recovery.source_words(common.read_text()), recovery.source_words(ultra.read_text())
                    self.assertEqual(len(a), len(b))
                    self.assertNotEqual(a, b)


if __name__ == "__main__":
    unittest.main()
