# SPDX-License-Identifier: MIT
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
HAS_NUMPY = importlib.util.find_spec("numpy") is not None
if HAS_NUMPY:
    sys.path[:0] = [str(ROOT / "tools/diagnostics"), str(ROOT / "tools")]
    import fsr411_quality_scenes as scenes
    import fsr412_new_look_ablation as ablation
    from fsr412 import new_look
    import numpy as np
    del sys.path[:2]


@unittest.skipUnless(HAS_NUMPY, "quality fixtures require NumPy")
class NewLookAblationTests(unittest.TestCase):
    def test_post_edit_only_disables_the_depth_clamp_block(self):
        original = new_look.COLOR_RULES
        modified = ablation.without_depth_clamps(original)
        self.assertEqual(modified.replace("if (false) {", ablation.GATE), original)
        self.assertNotIn(ablation.GATE, modified)
        for unsupported in ("", original * 2):
            with self.assertRaises(ValueError):
                ablation.without_depth_clamps(unsupported)

    def test_flat_depth_keeps_all_other_fixture_bytes(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(scenes, "FRAMES", 2):
            step, flat = Path(tmp) / "step", Path(tmp) / "flat"
            scenes.generate(step, ("moving_object",), "linear")
            scenes.generate(flat, ("moving_object",), "linear", moving_object_depth="flat")
            step, flat = step / "moving_object", flat / "moving_object"
            self.assertEqual(sorted(p.name for p in step.iterdir()), sorted(p.name for p in flat.iterdir()))
            for p in step.iterdir():
                other = flat / p.name
                if p.name.endswith("_depth.bin"):
                    self.assertEqual(set(np.fromfile(p, dtype="<f4")), {np.float32(.7), np.float32(.2)})
                    self.assertTrue(np.all(np.fromfile(other, dtype="<f4") == np.float32(.7)))
                else:
                    self.assertEqual(p.read_bytes(), other.read_bytes(), p.name)
