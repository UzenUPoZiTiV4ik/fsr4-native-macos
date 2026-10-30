#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""FSR 4.1.2 Lite POST: the CT2D float-dot rewrite is exact and strict."""

import random
import re
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fsr412.optimization import ct2d_float_dot  # noqa: E402


def signed(byte):
    return byte - 256 if byte > 127 else byte


def kernel(words, products=32):
    table = "constant spvUnsafeArray<uint, %d> _1099 = spvUnsafeArray<uint, %d>({ %s });" % (
        len(words), len(words), ", ".join("%du" % word for word in words))
    term = "reduce_add(int4(as_type<char4>(_2540)) * int4(as_type<char4>(_1099[_2717 + 1u])))"
    return table + "\nkernel void post() {\n" + "\n".join("    acc += %s;" % term for _ in range(products)) + "\n}\n"


class Ct2dFloatDot(unittest.TestCase):
    def test_widened_table_holds_the_signed_bytes(self):
        rng = random.Random(3)
        words = [rng.randrange(1 << 32) for _ in range(128)]
        result = ct2d_float_dot(kernel(words))
        table = re.search(r"_1099F = spvUnsafeArray<float4, 128>\(\{ (.*) \}\);", result)[1]
        values = [tuple(int(v) for v in item.split(", ")) for item in re.findall(r"float4\(([^)]*)\)", table)]
        self.assertEqual(values, [tuple(signed((w >> (8 * i)) & 255) for i in range(4)) for w in words])
        self.assertEqual(result.count("int(dot(float4(as_type<char4>(_2540)), _1099F[_2717 + 1u]))"), 32)
        self.assertNotIn("reduce_add(int4", result)

    def test_float_dot_is_exact_for_every_int8_extreme(self):
        # |sum| <= 4 * 128 * 128 = 65536 < 2^24, so float32 sums are exact.
        import numpy as np
        for a in (-128, 127, 0):
            for b in (-128, 127):
                x = np.full(4, a, np.float32)
                w = np.full(4, b, np.float32)
                self.assertEqual(int(np.float32(np.dot(x, w))), 4 * a * b)

    def test_unknown_shapes_fail(self):
        with self.assertRaises(ValueError):
            ct2d_float_dot(kernel([0] * 128, products=31))
        with self.assertRaises(ValueError):
            ct2d_float_dot("kernel void post() {}")


if __name__ == "__main__":
    unittest.main()
