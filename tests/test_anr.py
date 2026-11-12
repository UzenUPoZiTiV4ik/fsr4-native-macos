import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


nrx = module("anr_nrx_test", "tools/diagnostics/anr_nrx.py")
prepare = module("anr_prepare_test", "tools/anr/prepare.py")
manifest = module("anr_manifest_test", "tools/anr/manifest.py")
installer = module("anr_installer_test", "tools/crossover/crossover_fsr4.py")


class AnrTest(unittest.TestCase):
    def pack(self, members):
        try:
            import zstandard
        except ImportError:
            self.skipTest("optional zstandard not installed")
        payload = b"".join(data for _, data in members)
        directory = struct.pack("<I", len(members))
        offset = 0
        for name, data in members:
            directory += struct.pack("<I", len(name)) + name + struct.pack("<QQ", offset, len(data))
            offset += len(data)
        compressed = zstandard.ZstdCompressor().compress(directory + payload)
        wrapper = b"NRX1" + struct.pack("<IQ", 2, 17) + compressed
        # XOR stream is symmetric; re-use only the independently reconstructed
        # public mask primitive, not the archive parser under test.
        _, masked = nrx.unmask(wrapper)
        return wrapper[:16] + masked

    def test_nrx_parse_and_export_never_overwrites(self):
        data = self.pack([(b"program.json", b'{"family":"fixture"}'), (b"weights.bin", b"weights")])
        kind, files = nrx.decode(data)
        self.assertEqual(kind, 2)
        self.assertEqual(files, {"program.json": b'{"family":"fixture"}', "weights.bin": b"weights"})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "pack.nrx"
            source.write_bytes(data)
            output = root / "output"
            report = nrx.inspect(source, output)
            self.assertFalse(report["runtime_executed"])
            self.assertEqual((output / "weights.bin").read_bytes(), b"weights")
            with self.assertRaises(FileExistsError):
                nrx.inspect(source, output)
            self.assertEqual(source.read_bytes(), data)

    def test_nrx_names_and_duplicate_rejected(self):
        for name in (b"../escape", b"/absolute", b"a\\b", b"a//b", b"nrx-inspection.json"):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "unsafe"):
                nrx.decode(self.pack([(name, b"fixture")]))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            nrx.decode(self.pack([(b"A", b"a"), (b"a", b"b")]))

    def test_nrx_truncated_invalid_magic_and_budget(self):
        for data in (b"", b"NRX1", b"not an asset"):
            with self.assertRaises(ValueError):
                nrx.decode(data)
        data = self.pack([(b"a", b"payload")])
        with patch.object(nrx, "MAX_BYTES", 16), self.assertRaisesRegex(ValueError, "large"):
            nrx.decode(data)

    def test_uniform_abi_typed_offsets_and_unused_names(self):
        reflection = {"parameters": [
            {"name": "w", "binding": {"kind": "uniform", "offset": 4, "size": 4}, "type": {"scalarType": "uint32"}},
            {"name": "scale", "binding": {"kind": "uniform", "offset": 12, "size": 4}, "type": {"scalarType": "float32"}},
        ]}
        data = prepare.uniform_bytes(reflection, {"w": 4, "scale": .5, "shared_unused": 13})
        self.assertEqual(struct.unpack_from("<I", data, 4)[0], 4)
        self.assertEqual(struct.unpack_from("<f", data, 12)[0], .5)
        for invalid in (-1, .1, 1 << 32, float("inf")):
            with self.subTest(value=invalid), self.assertRaises(ValueError):
                prepare.uniform_bytes(reflection, {"w": invalid})

    def test_preparation_refuses_unknown_release_before_gpu_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pack.nrx"
            path.write_bytes(b"fake runtime")
            with self.assertRaisesRegex(ValueError, "identity"):
                prepare.read_pack(path, "0" * 64)

    def test_missing_runtime_and_unsafe_asset_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "prepared ANR"):
                manifest.validate_runtime(root)
            for name in ("../escape", "/absolute", "a//b", "a\\b"):
                with self.subTest(name=name), self.assertRaisesRegex(ValueError, "unsafe"):
                    manifest.regular(root, name)

    def test_installer_rejects_without_touching_bottle(self):
        with patch("sys.argv", ["crossover_fsr4.py", "install", "--frame-generation", "anr"]), \
             patch.dict("os.environ", {}, clear=True), patch.object(installer, "conf_set") as config, \
             patch.object(installer, "build_hook") as build, self.assertRaisesRegex(SystemExit, "requires --anr-runtime"):
            installer.main()
        config.assert_not_called()
        build.assert_not_called()

    def test_fg_switch_has_no_upscaler_profile_change(self):
        with patch("sys.argv", ["crossover_fsr4.py", "frame-generation", "metalfx", "--bottle", "Games"]), \
             patch.object(installer, "resolve_bottle", return_value=Path("/fixture/config")), \
             patch.object(installer, "conf_set") as config, patch.object(installer, "select_panel_profile") as profile, \
             patch.object(installer, "select_frame_generation") as fg:
            installer.main()
        self.assertEqual(config.call_args.args[1], {"FSR4_FG": "metalfx"})
        profile.assert_not_called()
        fg.assert_called_once_with("metalfx")

    def test_fg_file_keeps_sr_settings_and_rejects_unknown_modes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            panel = root / "panel.conf"
            original = "sharpness=0.200\nrcas=0\nprofile=int8-lite-lossy\n"
            panel.write_text(original)
            with patch.object(installer, "DIST", root):
                installer.select_frame_generation("anr")
                self.assertEqual((root / "fg-settings.conf").read_text(), "schema=fsr4.fg.settings.v1\nmode=anr\n")
                installer.select_frame_generation("metalfx")
                self.assertIn("mode=metalfx", (root / "fg-settings.conf").read_text())
                self.assertIn("mode=anr", (root / "backups/fg-settings.conf").read_text())
                with self.assertRaises(ValueError):
                    installer.select_frame_generation("other")
            self.assertEqual(panel.read_text(), original)

    def test_fg_preserves_existing_sharpness_profile(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old = root / "previous"
            source = old / "dist/crossover/panel.conf"
            source.parent.mkdir(parents=True)
            source.write_text("sharpness=0.200\nrcas=0\nprofile=exact-fast\n")
            new = root / "new"
            new.mkdir()
            config = root / "bottle.conf"
            config.write_text('"FSR4_ROOT" = "' + str(old) + '"\n')
            with patch.object(installer, "DIST", new), patch.object(installer, "ROOT", new):
                installer.preserve_panel_preferences(config)
            self.assertEqual((new / "panel.conf").read_text(), source.read_text())
            self.assertEqual(source.read_text(), "sharpness=0.200\nrcas=0\nprofile=exact-fast\n")

    def test_packed_source_identities_are_separate_but_model_is_same(self):
        self.assertEqual(set(prepare.PACKED_PROGRAMS), set(prepare.PROGRAM_PACKS))
        for size in prepare.PROGRAM_PACKS:
            self.assertNotEqual(prepare.PACKED_PROGRAMS[size], prepare.PROGRAM_PACKS[size])

    def test_source_package_leaves_out_proprietary_anr_data(self):
        package = module("anr_package_test", "tools/crossover/package_release.py")
        for path in ("build/anr_runtime_1080p/weights.bin", "build/anr_runtime_1080p/kernels/k1/kernel.metallib",
                     "build/anr_reference_buffers/b14.bin"):
            self.assertFalse(package.included(path))
        self.assertTrue(package.included("tests/anr_live_test.mm"))
        self.assertTrue(package.included("tests/anr_ui_test.mm"))
        self.assertTrue(package.included("tests/anr_motion_test.mm"))
        self.assertTrue(package.included("src/anr_runtime.mm"))


if __name__ == "__main__":
    unittest.main()
