"""Contracts for splitting the pinned SPD source without changing its math.

The independent GPU/CPU behavior checks live in
tests/run_fsr411_autoexposure.py and are not repeated by these source tests.
"""

from pathlib import Path
import re
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from autoexposure import split_autoexposure
finally:
    sys.path.pop(0)


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "third_party/fsr411/shaders/ae_411_ours/ae_ours.hlsl"


def block(source, pattern):
    match = re.search(pattern, source)
    if match is None:
        raise ValueError("missing source block")
    start = source.index("{", match.start())
    depth = 0
    for i in range(start, len(source)):
        depth += (source[i] == "{") - (source[i] == "}")
        if depth == 0:
            return source[match.start():i + 1]
    raise ValueError("unterminated source block")


class AutoExposureTests(unittest.TestCase):
    def test_both_dispatches_preserve_arithmetic_helpers_and_abi(self):
        original = SOURCE.read_text()
        shaders = split_autoexposure(original)
        self.assertEqual(set(shaders), {"ae_tiles.hlsl", "ae_finalize.hlsl"})
        for generated in shaders.values():
            for function in ("computeEV100FromAvgLuminance", "convertEV100ToExposure", "GetExposure",
                             "SpdLoadSourceImage", "SpdReduce4", "SpdLoadIntermediate",
                             "SpdStoreIntermediate"):
                pattern = r"^\w+ " + function + r"\("
                self.assertEqual(block(generated, "(?m)" + pattern), block(original, "(?m)" + pattern))
            self.assertEqual(block(generated, r"cbuffer AutoExposureSPDConstants"),
                             block(original, r"cbuffer AutoExposureSPDConstants"))
            self.assertIn('#include "spd/ffx_spd.h"', generated)
            self.assertIn("[numthreads(256, 1, 1)]", generated)

    def test_mip_six_store_and_previous_exposure_order_are_explicit(self):
        for generated in split_autoexposure(SOURCE.read_text()).values():
            store = block(generated, r"void SpdStore\(")
            self.assertIn("if (index == 5)", store)
            self.assertIn("if (index == MipCount() - 1)", store)
            self.assertNotIn("else if", store)
            previous = store.index("rw_auto_exposure_texture[int2(1,0)] = prev_exposre;")
            current = store.index("rw_auto_exposure_texture[int2(0,0)] = GetExposure(outValue.x);")
            self.assertLess(previous, current)
            read = block(generated, r"FfxFloat32x4 SpdLoad\(")
            self.assertIn("(validSize + FfxUInt32x2(63u, 63u)) / 64u", read)
            self.assertIn("inRect ? loaded : 0.0f", read)

    def test_no_atomic_entry_path_and_full_tail_remain(self):
        generated = split_autoexposure(SOURCE.read_text())
        tiles = block(generated["ae_tiles.hlsl"], r"void main\(")
        final = block(generated["ae_finalize.hlsl"], r"void main\(")
        self.assertIn("FfxUInt32x2(WorkGroupId.xy) + WorkGroupOffset()", tiles)
        self.assertIn("SpdDownsampleMips_0_1(", tiles)
        self.assertIn("2u, min(MipCount(), 6u)", tiles)
        self.assertIn("SpdDownsampleMips_6_7(", final)
        self.assertIn("8u, MipCount()", final)
        for main in (tiles, final):
            self.assertNotIn("SpdDownsample(", main)
            self.assertNotIn("SpdExitWorkgroup(", main)
            self.assertNotIn("SpdIncreaseAtomicCounter(", main)
            self.assertIn("ffxRemapForWaveReduction(FfxUInt32(LocalThreadIndex) % 64u)", main)

    def test_changed_entry_and_unknown_source_are_rejected(self):
        original = SOURCE.read_text()
        for broken in ("", original.replace("FfxUInt32(MipCount()),", "FfxUInt32(12),"),
                       original.replace("return FfxFloat32x4(rw_autoexp_mip_5[tex], 0, 0, 0);",
                                        "return FfxFloat32x4(0, 0, 0, 0);")):
            with self.assertRaises(ValueError):
                split_autoexposure(broken)
        for generated in split_autoexposure(original).values():
            with self.assertRaises(ValueError):
                split_autoexposure(generated)


if __name__ == "__main__":
    unittest.main()
