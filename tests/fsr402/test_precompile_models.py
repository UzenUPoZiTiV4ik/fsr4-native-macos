import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location(
    "precompile_models", Path(__file__).resolve().parents[2] / "tools/fsr402/precompile_models.py")
precompile = importlib.util.module_from_spec(spec)
spec.loader.exec_module(precompile)


class PrecompileModelsTest(unittest.TestCase):
    def test_auto_uses_native_policy_result(self):
        for backend in ("f16", "i8mpp"):
            self.assertEqual(precompile.selected_models("auto", {"backend": backend}), (backend,))
        with self.assertRaises(ValueError):
            precompile.selected_models("auto", {"backend": "int8"})

    def test_offline_explicit_models_do_not_need_device_info(self):
        self.assertEqual(precompile.selected_models("all", {}), ("f16", "i8mpp"))
        self.assertEqual(precompile.selected_models("f16", {}), ("f16",))

    def test_fp16_does_not_bind_native_post_parameters(self):
        cmd = precompile.compile_command(Path("/tmp/root with spaces"), "f16", "quality", "2160", 1)
        self.assertEqual(cmd[cmd.index("--model") + 1], "f16")
        self.assertEqual(cmd[cmd.index("--res-idx") + 1], "1")
        self.assertNotIn("--post-ext-ct2d", cmd)
        self.assertIn("/tmp/root with spaces", cmd)

    def test_int8_uses_the_generated_tier_offset(self):
        with self.assertRaises(ValueError):
            precompile.compile_command(Path("/tmp/root"), "i8mpp", "quality", "1080", 0)
        cmd = precompile.compile_command(Path("/tmp/root"), "i8mpp", "quality", "4320", 0, 332352256)
        self.assertEqual(cmd[cmd.index("--model") + 1], "i8")
        self.assertEqual(cmd[cmd.index("--res-idx") + 1], "2")
        self.assertEqual(cmd[cmd.index("--post-ext-ct2d") + 1], "332352256")


if __name__ == "__main__":
    unittest.main()
