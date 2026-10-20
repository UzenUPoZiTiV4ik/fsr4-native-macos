import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location(
    "crossover_fsr4", Path(__file__).resolve().parents[1] / "tools/crossover/crossover_fsr4.py")
installer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(installer)


class CrossoverReferenceTest(unittest.TestCase):
    def test_external_bottles_follow_config_and_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bottles = root / "CrossOver/Bottles"
            bottles.mkdir(parents=True)
            external = root / "External SSD/My Bottles"
            conf = external / "Steam/cxbottle.conf"
            conf.parent.mkdir(parents=True)
            conf.write_text("[EnvironmentVariables]\n")
            config = bottles.parent / "CrossOver.conf"
            config.write_text('[Other]\n"BottlePath" = "/ignored"\n'
                              '[CrossOver]\n;"BottlePath" = "/ignored"\n'
                              '"BottlePath" = "${TEST_BOTTLE_ROOT}" ; external disk\n')
            with patch.object(installer, "BOTTLES", bottles), \
                 patch.object(installer, "crossover_apps", return_value=[]), \
                 patch.dict(os.environ, {"TEST_BOTTLE_ROOT": str(external)}, clear=True):
                self.assertEqual(installer.default_bottle(), conf)
                self.assertEqual(installer.resolve_bottle("Steam"), conf)
                conf.write_text(installer.MARKER)
                self.assertEqual(installer.default_bottle(for_restore=True), conf)
                other = root / "Other/Steam/cxbottle.conf"
                other.parent.mkdir(parents=True)
                other.write_text("")
                with patch.dict(os.environ, {"CX_BOTTLE_PATH": str(other.parent.parent)}):
                    self.assertEqual(installer.default_bottle(), other)
                with patch.dict(os.environ, {"CX_BOTTLE_PATH": os.pathsep.join((str(external), str(other.parent.parent)))}):
                    with self.assertRaisesRegex(SystemExit, "ambiguous bottle name"):
                        installer.resolve_bottle("Steam")
                    self.assertEqual(installer.resolve_bottle(str(conf)), conf)
                    with self.assertRaisesRegex(SystemExit, "several bottles found"):
                        installer.default_bottle()

    def test_external_symlink_is_not_listed_twice(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            conf = root / "external/Steam/cxbottle.conf"
            conf.parent.mkdir(parents=True)
            conf.write_text("")
            bottles = root / "default"
            bottles.mkdir()
            (bottles / "Steam").symlink_to(conf.parent, target_is_directory=True)
            with patch.object(installer, "BOTTLES", bottles), \
                 patch.object(installer, "crossover_apps", return_value=[]), \
                 patch.dict(os.environ, {"CX_BOTTLE_PATH": os.pathsep.join((str(bottles), str(conf.parent.parent)))}, clear=True):
                self.assertEqual(len(installer.discover_bottles()), 1)
                self.assertEqual(installer.default_bottle().resolve(), conf.resolve())

    def test_custom_app_bottle_config_and_user_override(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            app = root / "CrossOver.app"
            config = app / "Contents/SharedSupport/CrossOver/etc/CrossOver.conf"
            config.parent.mkdir(parents=True)
            config.write_text('[CrossOver]\n"BottlePath" = "' + str(root / "external") + '"\n')
            conf = root / "external/Steam/cxbottle.conf"
            conf.parent.mkdir(parents=True)
            conf.write_text("")
            bottles = root / "user/Bottles"
            bottles.mkdir(parents=True)
            with patch.object(installer, "BOTTLES", bottles), patch.dict(os.environ, {}, clear=True):
                self.assertEqual(installer.default_bottle(crossover=str(app)), conf)
                (bottles.parent / "CrossOver.conf").write_text('[CrossOver]\n"BottlePath" = ""\n')
                self.assertEqual(installer.bottle_roots(str(app)), [bottles])

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

    def test_411_fp16_alias_selects_simd_backend(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            testcase = root / "testcases/testcase_quality_720to1080"
            testcase.mkdir(parents=True)
            (testcase / "meta.json").write_text("{}")
            info = {"device": "Apple M1", "backend": "int8-simd", "reason": "simdgroup half matrices"}

            def run(command, **kwargs):
                if command[0] == str(root / "tools/bin/fsr4mac"):
                    self.assertEqual(command[-3:], ["--profile", "lossy", "--simd"])
                    destination = Path(command[2])
                    destination.mkdir(parents=True)
                    (destination / "f00_out.bin").write_bytes(b"simd frame")
                    (destination / "run.json").write_text(json.dumps(
                        {"fsr_version": "4.1.1", "profile": "lossy", "backend": "int8-simd"}))

            with patch.object(installer, "ROOT", root), \
                 patch.object(installer, "DIST", root / "dist/crossover"), \
                 patch.object(installer, "run", side_effect=run), \
                 patch.object(installer.subprocess, "check_output", return_value=json.dumps(info)) as query:
                reference = installer.build_reference({"FSR4_VERSION": "4.1.1", "FSR4_MODEL": "f16"})
                self.assertEqual(reference.read_bytes(), b"simd frame")
                self.assertEqual(query.call_args.args[0][-1], "simd")

    def test_411_rejects_unknown_model(self):
        with self.assertRaisesRegex(SystemExit, "FSR4_MODEL"), patch.object(installer, "run") as run:
            installer.build_reference({"FSR4_VERSION": "4.1.1", "FSR4_MODEL": "half"})
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

    def test_default_bottle_is_never_guessed(self):
        with tempfile.TemporaryDirectory() as directory:
            bottles = Path(directory)
            with patch.object(installer, "BOTTLES", bottles):
                with self.assertRaisesRegex(SystemExit, "no CrossOver bottles"):
                    installer.default_bottle()
                steam = bottles / "Steam" / "cxbottle.conf"
                steam.parent.mkdir()
                steam.write_text("[EnvironmentVariables]\n")
                self.assertEqual(installer.default_bottle(), steam)
                games = bottles / "Games" / "cxbottle.conf"
                games.parent.mkdir()
                games.write_text("[EnvironmentVariables]\n")
                with self.assertRaisesRegex(SystemExit, "several bottles found: Games, Steam"):
                    installer.default_bottle()
                # restore picks the single bottle that carries our keys
                with self.assertRaisesRegex(SystemExit, "nothing to restore"):
                    installer.default_bottle(for_restore=True)
                steam.write_text(f'[EnvironmentVariables]\n{installer.MARKER}\n"{installer.KEY}" = "/x"\n')
                self.assertEqual(installer.default_bottle(for_restore=True), steam)

    def test_installer_has_no_personal_defaults(self):
        source = (Path(__file__).resolve().parents[1] / "tools/crossover/crossover_fsr4.py").read_text()
        self.assertNotRegex(source, "[\u0400-\u04ff]")
        self.assertNotIn("/Users/", source)
        self.assertNotIn("version-min=14", source)

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
            with patch.object(installer, "crossover_apps", return_value=[preview]):
                self.assertEqual(installer.find_framework(), preview / installer.FRAMEWORK_IN_APP)
                with self.assertRaisesRegex(SystemExit, "--crossover"):
                    installer.find_framework(str(missing))
            with patch.object(installer, "crossover_apps", return_value=[]), \
                 self.assertRaisesRegex(SystemExit, "--crossover"):
                installer.find_framework()

    def test_finds_renamed_crossover_copies(self):
        # Issue #9: copies in ~/Applications named CrossOver_patched etc.
        with tempfile.TemporaryDirectory() as directory:
            system, home = Path(directory) / "system", Path(directory) / "home"
            system.mkdir()
            patched = home / "CrossOver_patched.app"
            dated = home / "CrossOver Preview-20261006.app"
            nested = home / "Games/CrossOver.app"
            for app in (patched, dated, nested):
                (app / installer.CROSSOVER_MARKER).mkdir(parents=True)
            (home / "CrossOver/Game.app/Contents").mkdir(parents=True)   # bottle launcher, not CrossOver
            (home / "Steam.app/Contents").mkdir(parents=True)
            (dated / installer.FRAMEWORK_IN_APP).mkdir(parents=True)
            (nested / installer.FRAMEWORKS_IN_APP[1]).mkdir(parents=True)
            apps = installer.crossover_apps((system, home), spotlight=False)
            self.assertEqual(apps, [dated, patched, nested])
            with patch.object(installer, "crossover_apps", return_value=apps):
                self.assertEqual(installer.find_framework(), dated / installer.FRAMEWORK_IN_APP)
            # GPTK 3-only layout
            self.assertEqual(installer.find_framework(str(nested)), nested / installer.FRAMEWORKS_IN_APP[1])
            # The standard name in /Applications comes first.
            standard = system / "CrossOver.app"
            (standard / installer.CROSSOVER_MARKER).mkdir(parents=True)
            self.assertEqual(installer.crossover_apps((system, home), spotlight=False)[0], standard)

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
