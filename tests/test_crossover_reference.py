import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location(
    "crossover_fsr4", Path(__file__).resolve().parents[1] / "tools/crossover/crossover_fsr4.py")
installer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(installer)


class CrossoverReferenceTest(unittest.TestCase):
    def test_install_defaults_to_411_lossy(self):
        with patch("sys.argv", ["crossover_fsr4.py", "install", "--no-bottle"]), \
             patch.object(installer, "cmd_install") as install:
            installer.main()
        args = install.call_args.args[0]
        self.assertEqual((args.fsr_version, args.profile, args.bottle), ("4.1.1", "lossy", ""))

    def test_old_gpu_uses_fresh_fp16_reference_even_with_int8_override(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            testcase = root / "testcases/testcase_quality_720to1080"
            testcase.mkdir(parents=True)
            (testcase / "meta.json").write_text("{}")
            (root / "out").mkdir()
            (root / "out/f00_out.bin").write_bytes(b"stale M5 INT8 frame")
            info = {"device": "Apple M1 Pro", "backend": "f16", "reason": "no GPU tensor acceleration"}

            def run(command, **kwargs):
                if command[0] == str(root / "tools/bin/fsr4mac"):
                    self.assertEqual(command[-1], "--fp16")
                    destination = Path(command[2])
                    destination.mkdir(parents=True)
                    (destination / "f00_out.bin").write_bytes(b"fresh FP16 frame")

            with patch.object(installer, "ROOT", root), \
                 patch.object(installer, "DIST", root / "dist/crossover"), \
                 patch.object(installer, "run", side_effect=run), \
                 patch.object(installer.subprocess, "check_output", return_value=json.dumps(info)) as query:
                reference = installer.build_reference({"FSR4_VERSION": "4.0.2", "FSR4_MODEL": "int8"})
                self.assertEqual(reference.read_bytes(), b"fresh FP16 frame")
                self.assertEqual(query.call_args.args[0][-4:], ["--fsr-version", "4.0.2", "--backend-info", "i8mpp"])
                self.assertEqual((root / "out/f00_out.bin").read_bytes(), b"stale M5 INT8 frame")

    def test_reference_defaults_to_411_lossy_integer_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            testcase = root / "testcases/testcase_quality_720to1080"
            testcase.mkdir(parents=True)
            (testcase / "meta.json").write_text("{}")
            info = {"device": "Apple M5", "backend": "int8-reference", "reason": "reference requested"}

            def run(command, **kwargs):
                if command[0] == str(root / "tools/bin/fsr4mac"):
                    self.assertEqual(command[-5:], ["--fsr-version", "4.1.1", "--profile", "lossy", "--reference"])
                    destination = Path(command[2])
                    destination.mkdir(parents=True)
                    (destination / "f00_out.bin").write_bytes(b"new 4.1.1 model")
                    (destination / "run.json").write_text(json.dumps(
                        {"fsr_version": "4.1.1", "profile": "lossy", "backend": "int8-reference"}))

            with patch.object(installer, "ROOT", root), \
                 patch.object(installer, "DIST", root / "dist/crossover"), \
                 patch.object(installer, "run", side_effect=run), \
                 patch.object(installer.subprocess, "check_output", return_value=json.dumps(info)) as query:
                reference = installer.build_reference({"FSR4_MODEL": "int8"})
                self.assertEqual(reference.read_bytes(), b"new 4.1.1 model")
                self.assertEqual(query.call_args.args[0][-4:],
                                 ["--fsr-version", "4.1.1", "--backend-info", "reference"])

    def test_411_rejects_old_fp16_model(self):
        with self.assertRaisesRegex(SystemExit, "no FP16 model"), patch.object(installer, "run") as run:
            installer.build_reference({"FSR4_VERSION": "4.1.1", "FSR4_MODEL": "f16"})
        run.assert_not_called()

    def test_411_scalar_reference_preserves_exact_profile(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            testcase = root / "testcases/testcase_quality_720to1080"
            testcase.mkdir(parents=True)
            (testcase / "meta.json").write_text("{}")
            info = {"device": "Apple M1 Pro", "backend": "int8-scalar", "reason": "portable integer model"}

            def run(command, **kwargs):
                if command[0] == str(root / "tools/bin/fsr4mac"):
                    self.assertEqual(command[-5:], ["--fsr-version", "4.1.1", "--profile", "exact", "--scalar"])
                    destination = Path(command[2])
                    self.assertEqual(destination.name, "reference-4.1.1-exact-scalar")
                    destination.mkdir(parents=True)
                    (destination / "f00_out.bin").write_bytes(b"scalar 4.1.1 frame")
                    (destination / "run.json").write_text(json.dumps(
                        {"fsr_version": "4.1.1", "profile": "exact", "backend": "int8-scalar"}))

            with patch.object(installer, "ROOT", root), \
                 patch.object(installer, "DIST", root / "dist/crossover"), \
                 patch.object(installer, "run", side_effect=run), \
                 patch.object(installer.subprocess, "check_output", return_value=json.dumps(info)):
                reference = installer.build_reference({"FSR4_VERSION": "4.1.1", "FSR4_MODEL": "scalar",
                                                       "FSR4_PROFILE": "exact"})
                self.assertEqual(reference.read_bytes(), b"scalar 4.1.1 frame")

    def test_411_rejects_unknown_profile(self):
        with self.assertRaisesRegex(SystemExit, "FSR4_PROFILE"), patch.object(installer, "run") as run:
            installer.build_reference({"FSR4_VERSION": "4.1.1", "FSR4_PROFILE": "fastest"})
        run.assert_not_called()

    def test_fsr4_profile_builds_the_old_model_reference(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            testcase = root / "testcases/testcase_quality_720to1080"
            testcase.mkdir(parents=True)
            (testcase / "meta.json").write_text("{}")
            info = {"device": "Apple M5", "backend": "i8mpp", "reason": "auto"}

            def run(command, **kwargs):
                if command[0] == str(root / "tools/bin/fsr4mac"):
                    self.assertEqual(command[-3:], ["--fsr-version", "4.0.2", "--i8mpp"])
                    destination = Path(command[2])
                    destination.mkdir(parents=True)
                    (destination / "f00_out.bin").write_bytes(b"4.0.2 frame")

            with patch.object(installer, "ROOT", root), \
                 patch.object(installer, "DIST", root / "dist/crossover"), \
                 patch.object(installer, "run", side_effect=run), \
                 patch.object(installer.subprocess, "check_output", return_value=json.dumps(info)) as query:
                reference = installer.build_reference({"FSR4_VERSION": "4.1.1", "FSR4_PROFILE": "fsr4"})
                self.assertEqual(reference.read_bytes(), b"4.0.2 frame")
                self.assertEqual(query.call_args.args[0][-4:], ["--fsr-version", "4.0.2", "--backend-info", "auto"])

    def test_bottle_accepts_name_directory_or_config(self):
        with tempfile.TemporaryDirectory() as directory:
            bottles = Path(directory)
            conf = bottles / "Steam" / "cxbottle.conf"
            conf.parent.mkdir()
            conf.write_text("[EnvironmentVariables]\n")
            with patch.object(installer, "BOTTLES", bottles):
                for value in ("Steam", str(conf.parent), str(conf)):
                    self.assertEqual(installer.resolve_bottle(value), conf)
                with self.assertRaisesRegex(SystemExit, "available bottles: Steam"):
                    installer.resolve_bottle("Games")

    def test_restore_keeps_metalfx_switch(self):
        with tempfile.TemporaryDirectory() as directory:
            conf = Path(directory) / "Steam" / "cxbottle.conf"
            conf.parent.mkdir()
            conf.write_text("[EnvironmentVariables]\n")
            with patch.object(installer, "DIST", Path(directory) / "dist"):
                installer.conf_set(conf, {installer.KEY: "/x", installer.METALFX_KEY: "1"})
                self.assertIn('"D3DM_ENABLE_METALFX" = "1"', conf.read_text())
                installer.conf_unset(conf)
            text = conf.read_text()
            self.assertIn('"D3DM_ENABLE_METALFX" = "1"', text)
            self.assertNotIn(installer.KEY, text)

    def test_framework_uses_first_installed_crossover(self):
        with tempfile.TemporaryDirectory() as directory:
            missing, preview = Path(directory) / "CrossOver.app", Path(directory) / "CrossOver Preview.app"
            (preview / installer.FRAMEWORK_IN_APP).mkdir(parents=True)
            with patch.object(installer, "CROSSOVER_APPS", (missing, preview)):
                self.assertEqual(installer.find_framework(), preview / installer.FRAMEWORK_IN_APP)
                with self.assertRaisesRegex(SystemExit, "--crossover"):
                    installer.find_framework(str(missing))

    def test_411_only_copy_rejects_fsr4_profile_before_building(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch("sys.argv", ["crossover_fsr4.py", "install", "--no-bottle", "--profile", "fsr4"]), \
             patch.object(installer, "FSR402_WEIGHTS", Path(directory) / "missing.bin"), \
             patch.object(installer, "build_hook") as build, \
             self.assertRaisesRegex(SystemExit, "no FSR 4.0.2 weights"):
            installer.main()
        build.assert_not_called()

    def test_release_zip_leaves_out_fsr402_weights_and_tests(self):
        spec = importlib.util.spec_from_file_location(
            "package_release", Path(__file__).resolve().parents[1] / "tools/crossover/package_release.py")
        package = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(package)
        self.assertFalse(package.included("third_party/fsr4/internal/shaders/fsr4_model_v07_i8_quality/initializers.bin"))
        self.assertFalse(package.included("tests/test_crossover_reference.py"))
        for path in ("third_party/fsr4/internal/shaders/rcas.hlsl", "third_party/fsr411/initializers.bin",
                     "tools/crossover/crossover_fsr4.py", "src/fsr4_live.mm", "tools/bin/dxc", "Makefile"):
            self.assertTrue(package.included(path), path)

    def test_411_rejects_reference_report_from_old_version(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            testcase = root / "testcases/testcase_quality_720to1080"
            testcase.mkdir(parents=True)
            (testcase / "meta.json").write_text("{}")
            refdir = root / "dist/crossover/reference-4.1.1-lossy-auto"
            refdir.mkdir(parents=True)
            info = {"device": "Apple M5", "backend": "int8-mpp", "reason": "auto"}
            for version, profile in (("4.0.2", "lossy"), ("4.1.1", "exact")):
                (refdir / "run.json").write_text(json.dumps(
                    {"fsr_version": version, "profile": profile, "backend": "int8-mpp"}))
                with self.subTest(version=version, profile=profile), \
                     patch.object(installer, "ROOT", root), \
                     patch.object(installer, "DIST", root / "dist/crossover"), \
                     patch.object(installer, "run"), \
                     patch.object(installer.subprocess, "check_output", return_value=json.dumps(info)), \
                     self.assertRaisesRegex(SystemExit, "requested FSR 4.1.1"):
                    installer.build_reference({"FSR4_VERSION": "4.1.1"})


if __name__ == "__main__":
    unittest.main()
