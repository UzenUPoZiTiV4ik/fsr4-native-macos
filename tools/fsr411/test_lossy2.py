#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Checks that lossy2 weights reach the native blob exactly as trained.

The transform writes AMD-layout literal tables; the native generator reads
those same tables back with its own decoder and builds the input-major
matrices and bias lists for the MPP kernels. Comparing that decoder's output
against the archive written independently is the strongest check available
without a lossy2 reference frame.
"""

import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import gen_fsr411_native as native  # noqa: E402
from fsr411 import lossy2  # noqa: E402

WEIGHTS = ROOT / lossy2.DEFAULT_WEIGHTS


@unittest.skipUnless(WEIGHTS.is_file(), "lossy2 weights not fetched")
class Lossy2Weights(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.weights = lossy2.load(ROOT)
        cls.initializer = (ROOT / "third_party/fsr411/initializers.bin").read_bytes()

    @staticmethod
    def source(tier, logical):
        folder = "reimpl411" if tier == 1080 else "reimpl%d" % tier
        suffix = "" if tier == 1080 else "_%d" % tier
        return ROOT / "third_party/fsr411/shaders" / folder / ("reimpl_pass%d%s.hlsl" % (logical, suffix))

    @staticmethod
    def expected(layer):
        """The native blob is [tap][in][out], [in][out], [in][out]."""
        a, bw, cw = layer["a"], layer["bw"], layer["cw"]
        return ([a[oc][ic][tap // 3][tap % 3] for tap in range(9) for ic in range(16) for oc in range(16)],
                [bw[oc][ic] for ic in range(len(bw[0])) for oc in range(len(bw))],
                [cw[oc][ic] for ic in range(len(cw[0])) for oc in range(len(cw))])

    def check(self, tier, logical):
        layer = self.weights.layers[logical]
        path = self.source(tier, logical)
        text = lossy2.apply_source(path.read_text(), logical, layer, bias_literals=False)
        initializer = lossy2.patched_initializer(self.initializer, {logical: layer})
        if logical in (1, 2, 12):
            model = native.parse_cnb(path, initializer, text=text)
        else:
            model = native.parse_fnb(path, initializer, tier, logical, text=text)
        self.assertEqual(model["bias"], layer["bias"], "pass %d bias" % logical)
        self.assertEqual([list(matrix) for matrix in model["weights"]],
                         [list(matrix) for matrix in self.expected(layer)], "pass %d weights" % logical)

    def test_native_weights_1080(self):
        for logical in lossy2.TABLES:
            with self.subTest(pass_=logical):
                self.check(1080, logical)

    def test_native_weights_4k(self):
        for logical in lossy2.TABLES:
            with self.subTest(pass_=logical):
                self.check(4320, logical)

    def test_bias_literals_replace_loads(self):
        for logical in lossy2.TABLES:
            with self.subTest(pass_=logical):
                layer = self.weights.layers[logical]
                text = lossy2.apply_source(self.source(1080, logical).read_text(), logical, layer)
                self.assertNotIn("t18_initializer.Load4(", text)
                self.assertEqual(text.count("Fsr411Lossy2Load"), 6)  # three calls and three definitions
                for offset, values in zip(lossy2.BIAS_OFFSETS[logical], layer["bias"]):
                    block = re.search(r"Bias%d\[\d+\] = \{ (.*?) \};" % offset, text)
                    self.assertIsNotNone(block, "pass %d bias %d" % (logical, offset))
                    words = [int(word, 16) & 0xFFFFFFFF for word in re.findall(r"0x([0-9a-f]{8})u", block.group(1))]
                    self.assertEqual(words, [value & 0xFFFFFFFF for value in values])

    def test_transform_is_idempotent_and_rejects_other_weights(self):
        layer = self.weights.layers[1]
        source = self.source(1080, 1).read_text()
        once = lossy2.apply_source(source, 1, layer)
        self.assertEqual(lossy2.apply_source(once, 1, layer), once)
        other = {**layer, "bias": [list(reversed(b)) for b in layer["bias"]]}
        with self.assertRaises(ValueError):
            lossy2.apply_source(once, 1, other)

    def test_patched_initializer_changes_only_bias_regions(self):
        patched = lossy2.patched_initializer(self.initializer, self.weights.layers)
        changed = {start for start, (a, b) in enumerate(zip(self.initializer, patched)) if a != b}
        allowed = set()
        for logical, layer in self.weights.layers.items():
            for offset, values in zip(lossy2.BIAS_OFFSETS[logical], layer["bias"]):
                allowed.update(range(offset, offset + 4 * len(values)))
        self.assertTrue(changed)
        self.assertLessEqual(changed, allowed)

    def test_rejects_weights_outside_int8(self):
        layer = {key: value for key, value in self.weights.layers[1].items()}
        layer["bw"] = [list(row) for row in layer["bw"]]
        layer["bw"][0][0] = 200
        with self.assertRaises(ValueError):
            lossy2.apply_source(self.source(1080, 1).read_text(), 1, layer)


if __name__ == "__main__":
    unittest.main()
