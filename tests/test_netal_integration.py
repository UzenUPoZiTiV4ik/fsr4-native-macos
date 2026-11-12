import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("netal_integration_test", ROOT / "tools/netal_integration.py")
netal = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(netal)
INSTALL_SPEC = importlib.util.spec_from_file_location("netal_installer_test", ROOT / "tools/crossover/crossover_fsr4.py")
installer = importlib.util.module_from_spec(INSTALL_SPEC)
INSTALL_SPEC.loader.exec_module(installer)


class NetalIntegrationTest(unittest.TestCase):
    def source_fixture(self, root):
        source = root / "source"
        source.mkdir()
        contents = {
            "src/macos_host/dlss_sr_session.mm": (
                b'NeuralStudioDLSSSRSession\nstd::getenv("NEURALSTUDIO_SR_PROFILE")\n'
                b'std::getenv("NEURALSTUDIO_SR_RENDER")\nstd::getenv("NEURALSTUDIO_SR_RENDER")\n'),
            "src/macos_host/dlss_sr_session.h": b"NeuralStudioDLSSSRSession",
            "LICENSE": b"MIT fixture",
        }
        for name, data in contents.items():
            path = source / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        manifest = root / "manifest.json"
        manifest.write_text(json.dumps({"schema": "fsr4_netal_source_v1", "revision": "fixture",
                                        "files": {name: netal.digest(data) for name, data in contents.items()}}))
        return source, manifest

    def runtime_fixture(self, root, contents):
        root.mkdir()
        rows = []
        for name, data in contents.items():
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            rows.append({"path": name, "bytes": len(data), "sha256": netal.digest(data)})
        material = (netal.SCHEMA + "\n").encode()
        for row in sorted(rows, key=lambda row: row["path"]):
            material += (row["path"] + "\0" + str(row["bytes"]) + "\0" + row["sha256"] + "\n").encode()
        manifest = {"schema": netal.SCHEMA, "model": netal.MODEL, "files": rows,
                    "asset_root_sha256": netal.digest(material)}
        (root / "manifest.json").write_text(json.dumps(manifest))
        return manifest

    def test_source_is_pinned_and_only_build_copies_change(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, manifest = self.source_fixture(root)
            original = (source / "src/macos_host/dlss_sr_session.mm").read_bytes()
            output = netal.prepare_sources(source, root / "output", manifest)
            self.assertEqual((source / "src/macos_host/dlss_sr_session.mm").read_bytes(), original)
            generated = (output / "dlss_sr_session.mm").read_text()
            self.assertIn('fsr4_netal_geometry_profile()', generated)
            self.assertEqual(generated.count('fsr4_netal_geometry_render()'), 2)
            self.assertIn("Fsr4NetalSRSession", generated)
            self.assertNotIn("NeuralStudioDLSSSRSession", generated)
            self.assertEqual(netal.prepare_sources(source, output, manifest), output)
            (output / "dlss_sr_session.mm").write_text("corrupt")
            with self.assertRaisesRegex(ValueError, "existing Netal build source differs"):
                netal.prepare_sources(source, output, manifest)

    def test_source_hash_mismatch_and_symlink_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, manifest = self.source_fixture(root)
            header = source / "src/macos_host/dlss_sr_session.h"
            header.write_bytes(b"other")
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                netal.prepare_sources(source, root / "output", manifest)
            header.unlink()
            header.symlink_to(source / "LICENSE")
            with self.assertRaisesRegex(ValueError, "symlink"):
                netal.prepare_sources(source, root / "output", manifest)
            self.assertFalse((root / "output").exists())

    def test_changed_geometry_source_contract_refused(self):
        with self.assertRaisesRegex(ValueError, "geometry bridge contract changed"):
            netal.adapt_source("src/macos_host/dlss_sr_session.mm", b"changed source")

    def test_runtime_integrity_and_missing_model_boundaries(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "runtime"
            weights = b"fixture weights, never an NVIDIA model"
            contents = {"model/weights.bin": weights, "metallib/kernel.metallib": b"fixture"}
            metadata = self.runtime_fixture(root, contents)
            models = {"model/weights.bin": (len(weights), netal.digest(weights))}
            with patch.object(netal, "REQUIRED", set(contents)), patch.object(netal, "MODELS", models):
                self.assertEqual(netal.validate_runtime(root), root.resolve())
                (root / "model/weights.bin").write_bytes(b"bad")
                with self.assertRaisesRegex(ValueError, "asset size mismatch"):
                    netal.validate_runtime(root)
                (root / "model/weights.bin").write_bytes(weights)
                (root / "extra.bin").write_bytes(b"extra")
                with self.assertRaisesRegex(ValueError, "unlisted"):
                    netal.validate_runtime(root)
                (root / "extra.bin").unlink()
                metadata["asset_root_sha256"] = "0" * 64
                (root / "manifest.json").write_text(json.dumps(metadata))
                with self.assertRaisesRegex(ValueError, "inventory root"):
                    netal.validate_runtime(root)
            # Self-consistent arbitrary weights never satisfy production pins.
            metadata["asset_root_sha256"] = netal.digest((netal.SCHEMA + "\n").encode())
            (root / "manifest.json").write_text(json.dumps({"schema": "source-build-receipt"}))
            with self.assertRaisesRegex(ValueError, "complete.*bundle"):
                netal.validate_runtime(root)

    def test_runtime_model_pin_and_symlink_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "runtime"
            contents = {"model/weights.bin": b"wrong model"}
            self.runtime_fixture(root, contents)
            with patch.object(netal, "REQUIRED", set(contents)), self.assertRaisesRegex(ValueError, "unsupported Netal model"):
                netal.validate_runtime(root)
            weights = root / "model/weights.bin"
            weights.unlink()
            weights.symlink_to(root / "manifest.json")
            with self.assertRaisesRegex(ValueError, "symlink"):
                netal.validate_runtime(root)

    def test_relative_paths_are_bounded(self):
        with tempfile.TemporaryDirectory() as directory:
            for name in ("/absolute", "../escape", "a/../escape", "a//b", "a\\b", "a/./b", ""):
                with self.subTest(name=name), self.assertRaisesRegex(ValueError, "unsafe"):
                    netal.regular_file(Path(directory), name)

    def test_install_rejects_missing_inputs_before_mutation(self):
        with patch("sys.argv", ["crossover_fsr4.py", "install", "--profile", "netal"]), \
             patch.dict("os.environ", {}, clear=True), patch.object(installer, "find_framework") as framework, \
             patch.object(installer, "conf_set") as config, self.assertRaisesRegex(SystemExit, "requires --netal-source"):
            installer.main()
        framework.assert_not_called()
        config.assert_not_called()

    def test_legacy_version_pin_rejects_netal(self):
        with patch("sys.argv", ["crossover_fsr4.py", "install", "--profile", "netal", "--fsr-version", "4.0.2"]), \
             patch.object(installer, "build_hook") as build, self.assertRaisesRegex(SystemExit, "legacy model pin"):
            installer.main()
        build.assert_not_called()

    def test_netal_option_and_restore_key(self):
        with patch("sys.argv", ["crossover_fsr4.py", "install", "--no-bottle", "--profile", "netal",
                               "--netal-source", "/source", "--netal-runtime", "/runtime"]), \
             patch.object(installer, "cmd_install") as install:
            installer.main()
        args = install.call_args.args[0]
        self.assertEqual((args.profile, args.netal_source, args.netal_runtime), ("netal", "/source", "/runtime"))
        self.assertIn("FSR4_NETAL_RUNTIME", installer.OURS)


if __name__ == "__main__":
    unittest.main()
