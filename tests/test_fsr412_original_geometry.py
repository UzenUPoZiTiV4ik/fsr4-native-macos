# SPDX-License-Identifier: MIT
"""CPU-only geometry checks, derived independently from saved original DXIL.

No shader math/GPU/DLL execution. Run with:
python3 -m unittest discover -s tests -p test_fsr412_original_geometry.py -v
External-text checks skip when ignored build assets are absent.
"""
from pathlib import Path
import hashlib
import json
import re
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "build/fsr412_upstream"
P1 = "2638456df248932a18dfd82b556d2c06d8ae3ab0fcd6f67eeeb037782bf6a0ea"
P2 = "6a5d98fc27a87938ab35d09af117f62238355cdab06aeaf3a9b26d91c434fb5e"
P2_4320 = "84e57a8c5f5e10a0fa04f69dbd0e0b257d36d76a1e1fe7704ca3a1f862a173b1"
P3 = "4a8443a4c81abc7c08a64f4131e8a1aeb86e33c1f49c5be530c886d33507a238"
P1_4320 = "f01148e8802285c16f7582e7743893916a39860b16f0890af39f00d0c0f5c80f"
LISTING_HASHES = {
    P1: "781903d101bad1b9011d64ba378b7fa3b222c1d650f0a563e00baa6247432440",
    P2: "6b87732399f1054ca0208b5fca96772f733abadce18b233df4fd2c837514c203",
    P2_4320: "9f298673bd60a022771f1b4e4bd83f95cc75c904351415d50d8b199ccd745135",
    P3: "f5b2abbb2b507c60b2f5fd8a1477013e73831b83059b9b0e42f8d9d015ec311b",
    P1_4320: "c989ab010627c35d9e7dcf453036d2a94d493428b9ea3136583d910ad77dc8be",
}


def ceil_div(n, d):
    return (n + d - 1) // d


def geometry(w, h, half=(1920, 1080)):
    """Dimensions only; distinct dispatch, write domain, and read domains."""
    assert w > 0 and h > 0 and w % 4 == h % 4 == 0
    assert w <= 2 * half[0] and h <= 2 * half[1]
    gx, gy = ceil_div(w // 2, 64), h // 2
    cnb = min(64 * gx, half[0]), min(4 * ceil_div(gy, 4), half[1])
    quarter_class = half[0] // 2, half[1] // 2
    p2_invocations = min(64 * gx, quarter_class[0]), min(gy, quarter_class[1])
    p2_output = w // 4, h // 4
    p2_minimal = min(64 * ceil_div(p2_output[0], 64), quarter_class[0]), p2_output[1]
    return {
        "groups": (gx, gy), "cnb": cnb,
        "ds": (cnb[0] // 2, cnb[1] // 2),
        "p1_input_inbounds": (min(cnb[0] + 1, half[0]), min(cnb[1] + 1, half[1])),
        "p2_invocations": p2_invocations, "p2_output": p2_output,
        # Raw padded coordinates start at (0,0) for first plane, (1,1) for second.
        "p2_first_raw_read": (p2_invocations[0] + 2, p2_invocations[1] + 2),
        "p2_second_raw_read": p2_invocations,
        "p2_valid_first_raw_read": (p2_output[0] + 2, p2_output[1] + 2),
        "p2_valid_second_raw_read": p2_output,
        "p2_dynamic_invalid": p2_invocations[0] * p2_invocations[1] - p2_output[0] * p2_output[1],
        "p2_minimal_dynamic_invalid": p2_minimal[0] * p2_minimal[1] - p2_output[0] * p2_output[1],
    }


def literal_lane(gx, gy, tx, ty):
    """Literal integer lane/group expressions, not rounded-rectangle formula."""
    return ((tx % 2) + (((gx << 1) | (gy & 1)) << 5) + ((tx // 4) << 1),
            (ty << 1) + ((gy >> 2) << 2) + ((tx % 4) // 2))


def eval_coordinate_ssa(source, gx, gy, tx, ty):
    """Small evaluator for the original P1 straight-line coordinate SSA only."""
    env = {"%g1.x": gx, "%g1.ya": gy, "%g1.yb": gy, "%5": tx, "%6": ty}
    lines = source.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("  %g1.b ="))
    end = next(i for i, line in enumerate(lines) if line.startswith("  %18 ="))
    ops = {
        "add": lambda a, b: a + b, "or": lambda a, b: a | b,
        "and": lambda a, b: a & b, "shl": lambda a, b: a << b,
        "lshr": lambda a, b: a >> b, "sdiv": lambda a, b: a // b,
        "srem": lambda a, b: a % b,
    }
    for line in lines[start:end + 1]:
        match = re.match(r"  (%[\w.]+) = (\w+)(?: nsw)? i32 ([%\w.-]+), ([%\w.-]+)", line)
        if not match:
            continue
        name, op, a, b = match.groups()
        env[name] = ops[op](env[a] if a.startswith("%") else int(a),
                            env[b] if b.startswith("%") else int(b))
    return env["%17"], env["%18"]


class GeometryTests(unittest.TestCase):
    def test_expected_rectangles(self):
        cases = [
            ((640, 384), (5, 192), (320, 192), (160, 96), (321, 193), (320, 192), 46080),
            ((644, 388), (6, 194), (384, 196), (192, 98), (385, 197), (384, 194), 58879),
            ((3840, 2160), (30, 1080), (1920, 1080), (960, 540), (1920, 1080), (960, 540), 0),
        ]
        for size, groups, cnb, ds, input_read, p2_threads, invalid in cases:
            with self.subTest(size=size):
                g = geometry(*size)
                self.assertEqual(g["groups"], groups)
                self.assertEqual(g["cnb"], cnb)
                self.assertEqual(g["ds"], ds)
                self.assertEqual(g["p1_input_inbounds"], input_read)
                self.assertEqual(g["p2_invocations"], p2_threads)
                self.assertEqual(g["p2_dynamic_invalid"], invalid)
                self.assertEqual(g["p2_output"], (size[0] // 4, size[1] // 4))

    def test_exhaustive_p1_store_coverage(self):
        # Count every CNB cell and every uint word in both DS planes.
        # Includes the full 2,073,600-cell class, with compact CPU counters.
        for size in [(640, 384), (644, 388), (3840, 2160)]:
            with self.subTest(size=size):
                g = geometry(*size)
                cw, ch = g["cnb"]
                dw, dh = g["ds"]
                cnb = bytearray(cw * ch)
                ds = bytearray(2 * dw * dh * 4)
                border_groups = []
                for gy in range(g["groups"][1]):
                    for gx in range(g["groups"][0]):
                        if gy & 2:
                            if (gx, gy) == (0, 2):
                                border_groups.append((gx, gy))
                            continue
                        for ty in range(2):
                            for tx in range(64):
                                x, y = literal_lane(gx, gy, tx, ty)
                                if x >= 1920 or y >= 1080:
                                    continue
                                cnb[y * cw + x] += 1
                                plane = ((tx % 4) // 2)
                                word = (tx % 2) * 2
                                i = ((plane * dh + y // 2) * dw + x // 2) * 4 + word
                                ds[i] += 1
                                ds[i + 1] += 1
                self.assertEqual(border_groups, [(0, 2)])
                self.assertEqual(cnb.count(1), len(cnb))
                self.assertEqual(ds.count(1), len(ds))

    def test_p2_read_and_write_coverage(self):
        for w, h in [(640, 384), (644, 388), (3840, 2160)]:
            g = geometry(w, h)
            nx, ny = g["p2_invocations"]
            first = bytearray((nx + 2) * (ny + 2))
            second = bytearray(nx * ny)
            output = bytearray((w // 4) * (h // 4))
            invalid = 0
            for y in range(ny):
                for x in range(nx):
                    # First DS plane has all nine padded taps; second only center.
                    for dy in range(3):
                        for dx in range(3):
                            first[(y + dy) * (nx + 2) + x + dx] = 1
                    second[y * nx + x] = 1
                    if x < w // 4 and y < h // 4:
                        output[y * (w // 4) + x] += 1
                    else:
                        invalid += 1
            self.assertEqual(first.count(1), len(first))
            self.assertEqual(second.count(1), len(second))
            self.assertEqual(output.count(1), len(output))
            self.assertEqual(invalid, g["p2_dynamic_invalid"])

    def test_border_cells_and_marker(self):
        cells = set()
        for ty in range(2):
            for tx in range(64):
                for i in range(64 * ty + tx, 6008, 128):
                    plane, j = divmod(i, 3004)
                    if j < 962:
                        x, y = j, 0
                    elif j < 1924:
                        x, y = j - 962, 541
                    else:
                        y, side = divmod(j - 1924, 2)
                        x, y = 961 * side, y + 1
                    address = 66547328 + plane * 8342464 + y * 15392 + x * 16
                    self.assertNotIn(address, cells)
                    cells.add(address)
        self.assertEqual(len(cells), 6008)
        self.assertTrue(all(66547328 <= a < 99820992 for a in cells))
        # No interior cell belongs to this border, including active ROI edges.
        self.assertNotIn(66547328 + 97 * 15392 + 161 * 16, cells)
        self.assertNotIn(33273792, cells)

    def test_sentinel_4320(self):
        sentinel, allocation, plane, stride = 134217728, 398707392, 132902464, 61472
        self.assertLess(sentinel + 16, allocation)
        self.assertGreaterEqual(sentinel, plane)
        y, row_offset = divmod(sentinel - plane, stride)
        self.assertEqual(row_offset % 16, 0)
        self.assertEqual((row_offset // 16 - 1, y - 1), (1521, 20))
        for (w, h), minimal_invalid in [((640, 384), 3072), ((644, 388), 3007)]:
            g = geometry(w, h, (3840, 2160))
            self.assertGreater(g["p2_dynamic_invalid"], 0)
            self.assertEqual(g["p2_minimal_dynamic_invalid"], minimal_invalid)
        # Quarter-grid shortening does not always remove the sentinel: X is padded.
        g = geometry(256, 384, (3840, 2160))
        self.assertEqual(g["p2_minimal_dynamic_invalid"], 0)
        self.assertGreater(g["p2_dynamic_invalid"], 0)
        self.assertEqual(geometry(7680, 4320, (3840, 2160))["p2_dynamic_invalid"], 0)
        for allocation in (25027392, 99820992):
            self.assertGreaterEqual(sentinel, allocation)


@unittest.skipUnless((ASSETS / (P1 + ".ll")).exists(), "external original text absent")
class ExternalEvidenceTests(unittest.TestCase):
    def test_saved_text_identities(self):
        for identity, digest in LISTING_HASHES.items():
            path = ASSETS / (identity + ".ll")
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)
        for identity in (P1, P2):
            receipt = json.loads((ASSETS / (identity + ".translation.json")).read_text())
            self.assertEqual(hashlib.sha256((ASSETS / (identity + ".metal")).read_bytes()).hexdigest(),
                             receipt["metal_sha256"])

    def test_coordinates_against_saved_ssa(self):
        for identity, points in [(P1, [(0, 0), (0, 1), (5, 192), (5, 193), (29, 1077)]),
                                 (P1_4320, [(0, 0), (5, 193), (59, 2157)])]:
            source = (ASSETS / (identity + ".ll")).read_text()
            for gx, gy in points:
                for ty in range(2):
                    for tx in range(64):
                        self.assertEqual(eval_coordinate_ssa(source, gx, gy, tx, ty),
                                         literal_lane(gx, gy, tx, ty))

    def test_helper_preserves_marker_gate(self):
        sys.path.insert(0, str(ROOT / "tools/fsr412"))
        try:
            from oracle_pass1_border import force_border_group
            source = (ASSETS / (P1 + ".metal")).read_text()
            result = force_border_group(source)
            marker = source[source.index("    if ((OracleLoad"):source.index("        if ((gl_WorkGroupID.y")]
            self.assertIn(marker, result)
            self.assertIn("const uint3 gl_WorkGroupID=uint3(0u,2u,0u);", result)
        finally:
            sys.path.pop(0)

    def test_p3_dynamic_invalid_returns_before_stores(self):
        source = (ASSETS / (P3 + ".ll")).read_text()
        self.assertIn("%1195 = icmp uge i32 %4, %1193", source)
        self.assertIn("%1196 = icmp uge i32 %5, %1194", source)
        self.assertIn("br i1 %1197, label %1815, label %1198", source)
        self.assertIn("%1199 = shl i32 %4, 1", source)
        self.assertIn("%1200 = shl i32 %5, 1", source)
        gate = source.index("br i1 %1197, label %1815, label %1198")
        self.assertLess(gate, source.index("call void @dx.op.rawBufferStore.i32"))

    def test_p2_dynamic_invalid_is_zero_store_not_early_return(self):
        for identity in (P2, P2_4320):
            source = (ASSETS / (identity + ".ll")).read_text()
            self.assertIn("%4819 = select i1 %4818, i32 %4812, i32 0", source)
            self.assertIn("%4843 = select i1 %4818, i32 %4842, i32 134217728", source)
            self.assertRegex(source, r"rawBufferStore\.i32\(i32 140, %dx.types.Handle %1, i32 %4843,")
            self.assertIn("i32 %4839, i32 %4885, i32 %4929, i32 %4973, i8 15", source)


if __name__ == "__main__":
    unittest.main()
