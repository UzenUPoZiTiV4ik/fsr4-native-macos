# SPDX-License-Identifier: MIT
"""Coverage of both staged PRE footprints and all linear PRE inventory shapes."""
import json
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
from fsr412 import compile_fsr412 as compiler
from fsr412.lossy_shaders import apply_lossy_pre
from fsr412.optimization import input_color_cache, pre_quad_routing
from fsr412.pre_opt import PRE_TILE_SIDE, pre_exposure, pre_tile


class PreTile(unittest.TestCase):
    def test_fp32_coverage(self):
        # Include the first/last groups, half-way ties and the host guard's
        # inclusive limit. All operations before floor/rint stay in float32.
        groups = np.arange(0, 7680, 16, dtype=np.float32)[:, None]
        pixels = groups + np.arange(16, dtype=np.float32)
        limit = np.float32((PRE_TILE_SIDE - 3) / 18)
        for inv in np.float32([1 / 7680, 1 / 3, .5, 2 / 3, np.nextafter(limit, np.float32(0)), limit]):
            origin = np.floor(groups * inv) - 2
            offset = np.float32(.5) * inv - np.float32(.5)
            for jitter in (-1, -.5, 0, .5, 1):
                color = np.floor((pixels * inv + offset) + jitter) - origin
                self.assertGreaterEqual(color.min(), 0)
                self.assertLess(color.max() + 1, PRE_TILE_SIDE)
            for delta in (-1, 0, 1):
                depth = np.rint((pixels + delta) * inv + offset) - origin
                self.assertGreaterEqual(depth.min(), 0)
                self.assertLess(depth.max(), PRE_TILE_SIDE)

    def test_linear_pre_variants(self):
        if not (ROOT / "build/fsr412_model/lite-shaders").is_dir():
            self.skipTest("no imported Lite shader inventory")
        rows = json.loads(compiler.CATALOG.read_text())["shaders"]
        for row in rows:
            if row["stage"] != "pre" or row.get("colorspace") != 0:
                continue
            prepared, _ = input_color_cache(compiler.signed_square(compiler.source_for(row).read_text()), 4)
            for stage in ("strict", "lossy", "reuse") if row["tier"] <= 2160 else ("strict",):
                with self.subTest(shader=row["sha256"], stage=stage):
                    text = prepared
                    if stage != "strict":
                        text = apply_lossy_pre(text, row["tier"], low_res_mv=bool(row["low_res_mv"]),
                                               depth_inverted=bool(row["depth_inverted"]), reuse_only=stage == "reuse")
                    text = pre_exposure(text, skip_roundtrip=stage != "strict")
                    if stage != "reuse":
                        text = pre_quad_routing(text)
                    tiled = pre_tile(text)
                    self.assertEqual(tiled.count("r_prepared_color.read("), 1)
                    self.assertEqual(tiled.count("fsr412DepthTile[fsr412LaneIndex] = r_depth.read(uint2(fsr412Coord)"),
                                     int(bool(row["low_res_mv"])))
                    self.assertNotIn("r_depth.read(uint2(uint2(_", tiled)
                    self.assertLess(tiled.index("threadgroup_barrier("), tiled.index("break;", tiled.index("kernel void")))
                    self.assertEqual("fsr412Unchanged" in tiled, stage != "strict")
                    self.assertEqual(tiled.count("r_input_exposure.read("), 0 if row["auto_exposure"] else 1)
                    self.assertEqual(tiled.count("r_auto_exposure_texture.gather(") +
                                     tiled.count("r_auto_exposure_texture.read("), 1)
                    with self.assertRaises(ValueError):
                        pre_tile(tiled)
