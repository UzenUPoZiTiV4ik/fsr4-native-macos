# SPDX-License-Identifier: MIT
"""CPU-only generator, preserved-source, address and cache-contract tests."""
from pathlib import Path
import importlib.util
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("original_post_cnb", ROOT / "tools/fsr412/original_post_cnb.py")
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)
SOURCE = ROOT / "build/fsr412_upstream" / (M.IDENTITY + ".isolated.metal")


class ContractTests(unittest.TestCase):
    def test_motion_coordinates_and_roi(self):
        def coords(x, y, dx, dy, w, h):
            dx, dy = max(-254, min(254, dx)), max(-254, min(254, dy))
            bx, by = x + dx // 2, y + dy // 2
            clamp = lambda n, size: max(0, min(size - 1, n))
            return (clamp(bx, w), clamp(by, h), clamp(bx + 1, w), clamp(by + 1, h), dx & 1, dy & 1)
        self.assertEqual(coords(0, 0, -1, -1, 322, 194), (0, 0, 0, 0, 1, 1))
        self.assertEqual(coords(321, 193, 1, 1, 322, 194), (321, 193, 321, 193, 1, 1))
        self.assertEqual(coords(160, 96, 3, -3, 322, 194), (161, 94, 162, 95, 1, 1))
        for x, y in [(0, 0), (321, 193), (160, 96)]:
            first_missing = x == 0 or y == 0
            self.assertEqual(first_missing, not (0 <= x - 1 < 322 and 0 <= y - 1 < 194))

    def test_cache_word_hit(self):
        xor = 0xa5a5a5a5
        raw = [1, 2, 3, 4]
        encoded = [x ^ xor for x in raw]
        self.assertTrue(all(encoded))
        self.assertEqual([x ^ xor for x in encoded], raw)
        self.assertFalse(all([encoded[0], 0, encoded[2], encoded[3]]))
        # An all-zero decoded word is not a miss; zero ENCODED word is.
        self.assertTrue(all([xor, xor, xor, xor]))


@unittest.skipUnless(SOURCE.exists(), "external original MSL unavailable")
class GeneratorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = SOURCE.read_text()
        cls.output, cls.receipt = M.rewrite(cls.source)

    def test_exact_untouched_source(self):
        spans = M.math_spans(self.source)
        reduced = self.source
        for span in reversed(spans):
            reduced = reduced[:span["start"]] + span["replacement"] + reduced[span["end"]:]
        helper_start = self.output.index("// Independent compact NHWC precomputed")
        helper_end = self.output.index("kernel void main0(")
        self.assertEqual(self.output[:helper_start] + self.output[helper_end:], reduced)
        self.assertEqual(self.output.count("extract_bits"), 1024)
        stores = lambda s: [l for l in s.splitlines() if "OracleStore(" in l]
        self.assertEqual(stores(self.output), stores(self.source))
        self.assertEqual(self.output.count("^ 2779096485u"), 20)

    def test_coordinates_and_outputs(self):
        self.assertEqual([(s["x"], s["y"]) for s in self.receipt["transformations"]],
                         [("_1697", "_1701"), ("_1705", "_1701"), ("_1697", "_1709"), ("_1705", "_1709")])
        self.assertEqual(self.output.count("OraclePrecomputedCnb(oracle_u11_buffer0._m0,"), 4)
        self.assertIn("uint _26315 = oracleCnb_primary.x;", self.output)
        self.assertIn("_106710 = oracleCnb_xy.w;", self.output)

    def test_layout_parameters(self):
        self.assertEqual(self.receipt["precomputed_base"], 133094720)
        self.assertEqual(self.receipt["precomputed_stride"], 30720)
        self.assertEqual(self.receipt["physical_scratch_bytes"], 166272320)
        output, receipt = M.rewrite(self.source, 133095424, 30848)
        self.assertIn("133095424u + y * 30848u + x * 16u", output)
        self.assertEqual(receipt["physical_scratch_bytes"], 133095424 + 1080 * 30848)
        for base, stride in [(0, 30720), (133094720 + 1, 30720), (133094720, 30704), (0xfffffff0, 30720)]:
            with self.assertRaises(ValueError):
                M.rewrite(self.source, base, stride)

    def test_fail_closed_changed_source(self):
        for text in [self.source + "\n", self.source.replace("99821056u", "99821072u", 1),
                     self.source.replace("_107319", "_107999", 1)]:
            with self.assertRaisesRegex(ValueError, "unpinned"):
                M.rewrite(text)
        changed = self.source.replace("uint _1749 =", "uint _1748 =", 1)
        with self.assertRaises(ValueError):
            M.math_spans(changed)

    def test_ordered_diagnostic_only_changes_group_coordinates(self):
        ordered = M.ordered_isolated(self.source)
        self.assertIn("constant uint& oracleGroupYOffset [[buffer(4)]]", ordered)
        self.assertIn("const uint3 gl_WorkGroupID=oracleRawGroupID+uint3(0u,oracleGroupYOffset,0u);", ordered)
        self.assertEqual(ordered.count("extract_bits"), self.source.count("extract_bits"))
        self.assertEqual([l for l in ordered.splitlines() if "OracleStore(" in l],
                         [l for l in self.source.splitlines() if "OracleStore(" in l])
        with self.assertRaises(ValueError):
            M.ordered_isolated(self.source + "\n")

    def test_required_run_only_receipt(self):
        self.assertIn("run frames only", self.receipt["production_execute_policy"])
        bench = (ROOT / "tools/bench/fsr412_original_post_cnb_ab.mm").read_text()
        self.assertIn("bool accelerated=!skip || allFrames;", bench)
        self.assertIn("accelerated?rewrittenPipeline:originalCandidatePipeline", bench)
        self.assertIn("if(accelerated)cnbMs+=native.dispatch(b);", bench)

    def test_original_cache_and_guard_contract(self):
        self.assertIn("if (_1642)\n        {\n            uint _1715", self.output)
        self.assertIn("if (_107319)", self.output)
        self.assertIn("if (_1685)", self.output)
        self.assertIn("if (_1686)", self.output)
        self.assertIn("if (_1685 && _1686)", self.output)
        self.assertEqual(sum("OracleStore(" in l and "^ 2779096485u" in l for l in self.output.splitlines()), 4)


if __name__ == "__main__":
    unittest.main()
