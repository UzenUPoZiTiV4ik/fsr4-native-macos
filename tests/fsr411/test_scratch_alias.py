"""No-GPU contracts for relocating only the external POST-CNB cache."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
try:
    from fsr411.compile_fsr411 import external_post_cnb
    from fsr411.gen_fsr411_native import parse_post_cnb
    from fsr411.lossy_shaders import apply_lossy_post, apply_lossy_post_cells
    from fsr411.post_optimization import apply_post_optimization
    from fsr411.scratch_alias import alias_post_cnb_cache, post_cnb_layout
finally:
    sys.path.pop(0)

ROOT = Path(__file__).resolve().parents[2]


def sources():
    for tier, folder in ((1080, "reimpl411"), (2160, "reimpl2160"), (4320, "reimpl4320")):
        for preset in ("shaders", "ultraperformance"):
            root = ROOT / "third_party/fsr411" / preset
            path = next((root / folder).glob("post_rev_411*.hlsl"))
            yield tier, preset, path, root / "reimpl411/post_rev_411_tables.hlsli"


class ScratchAliasTests(unittest.TestCase):
    def test_pinned_native_post_contract_all_tiers_and_presets(self):
        for tier, preset, path, tables in sources():
            with self.subTest(tier=tier, preset=preset):
                model = parse_post_cnb(path, tables, tier)
                width, height = tier * 8 // 9, tier // 2
                layout = post_cnb_layout(tier, True)
                self.assertEqual(model["kind"], "cnb")
                self.assertEqual(model["input"], [(width+3)*16, 16, (width+2)*16, width, height, 16, 0])
                self.assertEqual(model["output"], [0, 16, width*16, width, height, 16, 0])
                self.assertEqual(model["scratch_append_bytes"], layout["bytes"])
                self.assertEqual(layout["offset"] % 256, 0)
                self.assertGreaterEqual(layout["offset"], layout["source_plane_bytes"])
                self.assertLessEqual(layout["offset"]+layout["bytes"], layout["allocated_bytes"])
                self.assertEqual(layout["allocated_bytes"], layout["arena_bytes"])

    def test_reverse_relocation_restores_every_arithmetic_byte(self):
        for tier, preset, path, _ in sources():
            exact = external_post_cnb(path.read_text(), tier)
            lossy = apply_lossy_post(exact, tier)
            variants = (exact, lossy, apply_post_optimization(exact),
                        apply_post_optimization(lossy), apply_lossy_post_cells(apply_post_optimization(lossy)))
            for index, source in enumerate(variants):
                with self.subTest(tier=tier, preset=preset, variant=index):
                    result = alias_post_cnb_cache(source, tier)
                    self.assertEqual(alias_post_cnb_cache(result, tier), result)
                    original_base = post_cnb_layout(tier)["offset"]
                    alias_base = post_cnb_layout(tier, True)["offset"]
                    restored = result.split("\n", 1)[1].replace(
                        "ScratchBuffer.Load4(%du +" % alias_base, "ScratchBuffer.Load4(%du +" % original_base)
                    self.assertEqual(restored, source)
                    self.assertNotIn("ScratchBuffer.Load4(%du +" % original_base, result)

    def test_rejects_inline_post_and_unexpected_scratch_access(self):
        for tier, _, path, _ in sources():
            original = path.read_text()
            with self.assertRaises(ValueError):
                alias_post_cnb_cache(original, tier)
            split = external_post_cnb(original, tier)
            for access in ("ScratchBuffer.Load(0u);", "ScratchBuffer.Load4(0u);", "ScratchBuffer.Store(0u, 0u);"):
                with self.assertRaises(ValueError):
                    alias_post_cnb_cache(split + "\n" + access, tier)
            with self.assertRaises(ValueError):
                alias_post_cnb_cache(alias_post_cnb_cache(split, tier), 2160 if tier != 2160 else 1080)


if __name__ == "__main__":
    unittest.main()
