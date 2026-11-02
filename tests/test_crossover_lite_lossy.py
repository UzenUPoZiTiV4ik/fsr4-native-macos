import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location(
    "crossover_lite_lossy_installer",
    Path(__file__).resolve().parents[1] / "tools/crossover/crossover_fsr4.py")
installer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(installer)
PACKAGE_SPEC = importlib.util.spec_from_file_location(
    "crossover_lite_lossy_package",
    Path(__file__).resolve().parents[1] / "tools/crossover/package_release.py")
package = importlib.util.module_from_spec(PACKAGE_SPEC)
PACKAGE_SPEC.loader.exec_module(package)


class LiteLossyInstallerTest(unittest.TestCase):
    def test_package_keeps_lossy_weights_external_and_records_input_paths(self):
        metadata = package.optional_profile_metadata()
        self.assertEqual(set(metadata), set(installer.LITE_LOSSY_PROFILES))
        for profile, options in metadata.items():
            self.assertEqual(options["fetch_command"], "make lite-lossy-weights")
            self.assertFalse(options["weights_included"])
            self.assertEqual(options["experimental"], profile.endswith("-tiled"))
            self.assertEqual(options["panel_selectable"], not profile.endswith("-tiled"))
            self.assertEqual(options["required_inputs"],
                             ["build/fsr412_lossy/initializers-lite-lossy.bin", "build/fsr412_lossy/import.json"])
            for path in options["required_inputs"]:
                self.assertFalse(package.included(path), path)
        self.assertFalse(package.included("build/fsr412_lossy/provider.dll"))
        self.assertTrue(package.included("tools/fsr412/fetch_lossy_weights.py"))

    def test_install_options_accept_both_actual_profiles(self):
        for profile in installer.LITE_LOSSY_PROFILES:
            with self.subTest(profile=profile), \
                 patch("sys.argv", ["crossover_fsr4.py", "install", "--no-bottle",
                                    "--fsr-version", "4.1.2", "--profile", profile]), \
                 patch.object(installer, "cmd_install") as install:
                installer.main()
                args = install.call_args.args[0]
                self.assertEqual((args.fsr_version, args.profile), ("4.1.2", profile))

    def test_other_versions_rejected_before_build_or_bottle_changes(self):
        for version in ("4.0.2", "4.1.1"):
            for profile in installer.LITE_LOSSY_PROFILES:
                with self.subTest(version=version, profile=profile), \
                     patch("sys.argv", ["crossover_fsr4.py", "install", "--no-bottle",
                                        "--fsr-version", version, "--profile", profile]), \
                     patch.object(installer, "run") as run, \
                     patch.object(installer, "find_framework") as framework, \
                     patch.object(installer, "conf_set") as conf, \
                     self.assertRaisesRegex(SystemExit, "require --fsr-version 4.1.2"):
                    installer.main()
                run.assert_not_called()
                framework.assert_not_called()
                conf.assert_not_called()

    def test_both_inputs_required_without_automatic_fetch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "build/fsr412_lossy"
            base.mkdir(parents=True)
            for present in ((), ("initializers-lite-lossy.bin",), ("import.json",)):
                for child in base.iterdir():
                    child.unlink()
                for name in present:
                    (base / name).write_bytes(b"fixture")
                for profile in installer.LITE_LOSSY_PROFILES:
                    with self.subTest(present=present, profile=profile), \
                         patch.object(installer, "ROOT", root), \
                         patch("sys.argv", ["crossover_fsr4.py", "install", "--no-bottle",
                                            "--profile", profile]), \
                         patch.object(installer, "run") as run, \
                         patch.object(installer, "find_framework") as framework, \
                         self.assertRaisesRegex(SystemExit, "make lite-lossy-weights"):
                        installer.main()
                    run.assert_not_called()
                    framework.assert_not_called()

    def test_receipt_and_initializer_digest_validated(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "build/fsr412_lossy"
            base.mkdir(parents=True)
            data = b"fixture trained INT8 weights"
            name = "initializers-lite-lossy.bin"
            receipt = {"schema": "fsr412_lite_lossy_import_v1",
                       "source_dll_sha256": installer.LITE_LOSSY_SOURCE_DLL_SHA256,
                       "initializers": {name: hashlib.sha256(data).hexdigest()}}
            (base / name).write_bytes(data)
            (base / "import.json").write_text(json.dumps(receipt))
            with patch.object(installer, "ROOT", root), patch.object(installer, "run") as run:
                self.assertEqual(installer.require_lite_lossy_weights.__defaults__,
                                 (installer.LITE_LOSSY_INITIALIZER_SHA256,))
                with self.assertRaisesRegex(SystemExit, "pinned initializer"):
                    installer.require_lite_lossy_weights()
                with self.assertRaisesRegex(SystemExit, "pinned initializer"):
                    installer.require_lite_lossy_weights(expected_hash=None)
                installer.require_lite_lossy_weights(expected_hash=hashlib.sha256(data).hexdigest())
                with self.assertRaisesRegex(SystemExit, "pinned initializer"):
                    installer.require_lite_lossy_weights(expected_hash="0" * 64)
                run.assert_not_called()
                for malformed in ("not JSON", "[]", "{}", json.dumps({"schema": "other"}),
                                  json.dumps({**receipt, "source_dll_sha256": "0" * 64}),
                                  json.dumps({**receipt, "source_dll_sha256": None}),
                                  json.dumps({**receipt, "initializers": []}),
                                  json.dumps({**receipt, "initializers": {name: "bad"}})):
                    (base / "import.json").write_text(malformed)
                    with self.subTest(receipt=malformed), self.assertRaisesRegex(SystemExit, "Invalid.*make lite-lossy-weights"):
                        installer.require_lite_lossy_weights()
                (base / "import.json").write_text(json.dumps(receipt))
                for invalid in (b"", b"ordinary Lite or corrupted bytes"):
                    (base / name).write_bytes(invalid)
                    with self.subTest(data=invalid), self.assertRaisesRegex(SystemExit, "SHA-256.*make lite-lossy-weights"):
                        installer.require_lite_lossy_weights()

    def test_selected_profile_keeps_sharpness_and_rcas(self):
        for profile in installer.LITE_LOSSY_PROFILES:
            with self.subTest(profile=profile), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                panel = root / "panel.conf"
                previous = "sharpness=0.498\nrcas=0\nprofile=int8-lite\n"
                panel.write_text(previous)
                with patch.object(installer, "DIST", root):
                    installer.select_panel_profile(profile)
                self.assertEqual(panel.read_text(), "sharpness=0.498\nrcas=0\nprofile=" + profile + "\n")
                self.assertEqual((root / "backups/panel.conf").read_text(), previous)

    def test_reference_verifies_each_actual_profile_and_queries_it(self):
        for profile in installer.LITE_LOSSY_PROFILES:
            for alias in (profile, profile.removeprefix("int8-")):
                with self.subTest(profile=profile, alias=alias), tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    testcase = root / "testcases/testcase_quality_720to1080"
                    testcase.mkdir(parents=True)
                    (testcase / "meta.json").write_text("{}")
                    info = {"device": "fixture CPU", "backend": "int8-scalar", "reason": "mocked"}

                    def run(command, **kwargs):
                        if command[0] == str(root / "tools/bin/fsr4mac"):
                            self.assertEqual(command[-5:], ["--fsr-version", "4.1.2", "--profile", profile, "--scalar"])
                            destination = Path(command[2])
                            self.assertIn(profile, destination.name)
                            destination.mkdir(parents=True)
                            (destination / "f00_out.bin").write_bytes(profile.encode())
                            (destination / "run.json").write_text(json.dumps(
                                {"fsr_version": "4.1.2", "profile": profile, "backend": "int8-scalar"}))

                    with patch.object(installer, "ROOT", root), \
                         patch.object(installer, "DIST", root / "dist/crossover"), \
                         patch.object(installer, "require_lite_lossy_weights") as require, \
                         patch.object(installer, "run", side_effect=run), \
                         patch.object(installer.subprocess, "check_output", return_value=json.dumps(info)) as query:
                        reference = installer.build_reference(
                            {"FSR4_VERSION": "4.1.2", "FSR4_PROFILE": alias, "FSR4_MODEL": "scalar"})
                        self.assertEqual(reference.read_bytes(), profile.encode())
                        require.assert_called_once_with()
                        self.assertEqual(query.call_args.args[0][-6:],
                                         ["--fsr-version", "4.1.2", "--profile", profile, "--backend-info", "scalar"])

    def test_reference_backend_rejected_before_build_or_install(self):
        for profile in installer.LITE_LOSSY_PROFILES:
            for model in ("reference", "int8"):
                with self.subTest(profile=profile, model=model), \
                     patch.object(installer, "run") as run, \
                     patch.object(installer.subprocess, "check_output") as query, \
                     patch.object(installer, "require_lite_lossy_weights") as weights, \
                     patch.object(installer, "find_framework") as framework:
                    with self.assertRaisesRegex(SystemExit, "does not support the AMD reference backend"):
                        installer.build_reference({"FSR4_VERSION": "4.1.2", "FSR4_PROFILE": profile, "FSR4_MODEL": model})
                    with patch.dict(installer.os.environ, {"FSR4_MODEL": model}, clear=True), \
                         patch("sys.argv", ["crossover_fsr4.py", "install", "--no-bottle", "--profile", profile]), \
                         self.assertRaisesRegex(SystemExit, "does not support the AMD reference backend"):
                        installer.main()
                    run.assert_not_called()
                    query.assert_not_called()
                    weights.assert_not_called()
                    framework.assert_not_called()

    def test_reference_rejects_silent_lite_or_other_lossy_variant(self):
        for profile in installer.LITE_LOSSY_PROFILES:
            bad_profiles = ["int8-lite", *(p for p in installer.LITE_LOSSY_PROFILES if p != profile)]
            bad_reports = [(actual, "int8-mpp") for actual in bad_profiles] + [(profile, "int8-reference")]
            for actual, actual_backend in bad_reports:
                with self.subTest(profile=profile, actual=actual, backend=actual_backend), tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    testcase = root / "testcases/testcase_quality_720to1080"
                    testcase.mkdir(parents=True)
                    (testcase / "meta.json").write_text("{}")
                    refdir = root / "dist/crossover" / ("reference-4.1.2-" + profile + "-auto")
                    refdir.mkdir(parents=True)
                    (refdir / "run.json").write_text(json.dumps(
                        {"fsr_version": "4.1.2", "profile": actual, "backend": actual_backend}))
                    info = {"device": "fixture CPU", "backend": "int8-mpp", "reason": "mocked"}
                    with patch.object(installer, "ROOT", root), \
                         patch.object(installer, "DIST", root / "dist/crossover"), \
                         patch.object(installer, "require_lite_lossy_weights"), \
                         patch.object(installer, "run"), \
                         patch.object(installer.subprocess, "check_output", return_value=json.dumps(info)), \
                         self.assertRaisesRegex(SystemExit, "requested FSR 4.1.2 model/profile"):
                        installer.build_reference({"FSR4_VERSION": "4.1.2", "FSR4_PROFILE": profile})


if __name__ == "__main__":
    unittest.main()
