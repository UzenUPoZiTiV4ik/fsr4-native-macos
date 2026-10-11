# SPDX-License-Identifier: MIT
"""Check imported literal-weight semantics against the native matrix layout."""
import importlib.util
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import struct
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("gen_fsr411_native", ROOT / "tools/gen_fsr411_native.py")
native = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(native)


def s8(x):
    x &= 255
    return x if x < 128 else x - 256


def dot4(activation, weight):
    return sum(s8(activation >> (8 * i)) * s8(weight >> (8 * i)) for i in range(4))


def pack(values):
    return sum((min(127, max(-128, x)) & 255) << (8 * i) for i, x in enumerate(values))


def requant(x, shift, flags=0):
    # Independent exact rational oracle; Python round uses ties to even.
    rounded = math.floor(x / (1 << shift) + 0.5) if flags & 4 else round(x / (1 << shift))
    return max(-128, min(127, rounded))


def literal_pixel(model, image, x, y):
    """Execute the source's packed-dot block/tap/word iteration directly."""
    w1, w2, w3 = model["literal_weights"]
    rows = []
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            p = image[y + dy][x + dx] if 0 <= y + dy < len(image) and 0 <= x + dx < len(image[0]) else [0] * 16
            rows.append([pack(p[i:i + 4]) for i in range(0, 16, 4)])
    a0 = list(model["bias"][0])
    for group in range(4):
        for tap in range(9):
            for lane in range(4):
                a0[group * 4 + lane] += sum(dot4(rows[tap][word], w1[group * 9 + tap][lane * 4 + word]) for word in range(4))
    h0 = [pack([requant(x, model["q"][0][2], model["q"][0][3]) for x in a0[i:i + 4]]) for i in range(0, 16, 4)]
    a1 = list(model["bias"][1])
    for group in range(8):
        for lane in range(4):
            a1[group * 4 + lane] += sum(dot4(h0[word], w2[group][lane * 4 + word]) for word in range(4))
    h1 = [pack([requant(max(x, 0), model["q"][1][2], model["q"][1][3]) for x in a1[i:i + 4]]) for i in range(0, 32, 4)]
    a2 = list(model["bias"][2])
    for group in range(4):
        for half in range(2):
            for lane in range(4):
                a2[group * 4 + lane] += sum(dot4(h1[half * 4 + word], w3[group * 2 + half][lane * 4 + word]) for word in range(4))
    return finish_tail(model, a2, image[y][x])


def finish_tail(model, acc, residual):
    q = model["q"][2]
    result = []
    for a, r in zip(acc, residual):
        rounded = round(((a << q[0]) + (r << q[1])) / (1 << q[2]))
        wrapped = ((rounded + 32768) % 65536) - 32768 if q[3] & 2 else rounded
        result.append(min(127, max(-128, wrapped)))
    return result


def literal_post_pixel(model, image, x, y):
    """POST has flat tap/output-major tables and a floating residual tail."""
    dw, pw1, pw2 = model["post_literal_weights"]
    a0 = list(model["bias"][0])
    for tap in range(9):
        ix, iy = x + tap % 3 - 1, y + tap // 3 - 1
        if not (0 <= ix < len(image[0]) and 0 <= iy < len(image)):
            continue
        pixel = image[iy][ix]
        words = [pack(pixel[i:i + 4]) for i in range(0, 16, 4)]
        for oc in range(16):
            a0[oc] += sum(dot4(words[i], dw[tap * 64 + oc * 4 + i]) for i in range(4))
    h0 = [pack([requant(a, model["q"][0][2]) for a in a0[i:i + 4]]) for i in range(0, 16, 4)]
    a1 = [model["bias"][1][oc] + sum(dot4(h0[i], pw1[oc * 4 + i]) for i in range(4)) for oc in range(32)]
    h1 = [pack([requant(max(a, 0), model["q"][1][2]) for a in a1[i:i + 4]]) for i in range(0, 32, 4)]
    a2 = [model["bias"][2][oc] + sum(dot4(h1[i], pw2[oc * 8 + i]) for i in range(8)) for oc in range(16)]
    return [min(127, max(-128, round((r * 0.0625 + a * 0.0009765625) * 16.0))) for r, a in zip(image[y][x], a2)]


def matrix_pixel(model, image, x, y):
    c0, c1, c2 = model["weights"]
    vector = []
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            vector.extend(image[y + dy][x + dx] if 0 <= y + dy < len(image) and 0 <= x + dx < len(image[0]) else [0] * 16)
    mat = lambda a, w, n, b: [b[o] + sum(v * w[k * n + o] for k, v in enumerate(a)) for o in range(n)]
    a0 = [requant(v, model["q"][0][2], model["q"][0][3]) for v in mat(vector, c0, 16, model["bias"][0])]
    a1 = [requant(max(v, 0), model["q"][1][2], model["q"][1][3]) for v in mat(a0, c1, 32, model["bias"][1])]
    a2 = mat(a1, c2, 16, model["bias"][2])
    return finish_tail(model, a2, image[y][x])


@unittest.skipUnless((ROOT / "third_party/fsr411/initializers.bin").exists(), "import the official FSR 4.1.1 initializer first")
class NativeModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.initializer = (ROOT / "third_party/fsr411/initializers.bin").read_bytes()
        cls.source_dir = ROOT / "third_party/fsr411/shaders"
        cls.up_dir = ROOT / "third_party/fsr411/ultraperformance"
        source = (ROOT / "src/fsr411_mpp.metal").read_text()
        cls.kernel_tiles = {
            name: (int(nf), int(width), int(height), up == "true", int(groups))
            for name, nf, width, height, up, groups in re.findall(
                r"FSR411_FNB\((fsr411_\w+),\s*(\d+),\s*(\d+),\s*(\d+),\s*(true|false),\s*(\d+)\)", source)
        }

        cls.cnb_tiles = {
            name: (int(width), int(height), int(groups))
            for name, width, height, groups in re.findall(
                r"FSR411_CNB\((fsr411_\w+),\s*(\d+),\s*(\d+),\s*(\d+)\)", source)
        }
        # Cooperative chains must dispatch exactly like their SG1 fallback,
        # which runs whenever the runtime's cooperative layout differs.
        fnb_fallback = re.search(r"fsr411_gpu::fnb<(\d+),(\d+),(\d+),UPSAMPLE,(\d+)>", source)
        nf, width, height, groups = map(int, fnb_fallback.groups())
        for name, up in re.findall(r"FSR411_FNB_CHAIN\((fsr411_\w+),\s*(true|false)\)", source):
            cls.kernel_tiles[name] = (nf, width, height, up == "true", groups)
        cnb_fallback = re.search(r"kernel void (fsr411_cnb_coop_chain_\w+)\(.*?fsr411_gpu::cnb<(\d+),(\d+),(\d+)>",
                                 source, re.S)
        cls.cnb_tiles[cnb_fallback.group(1)] = tuple(map(int, cnb_fallback.groups()[1:]))

    def assert_native_dispatch_contract(self, manifest):
        entries = list(manifest["passes"].values()) + list(manifest["auxiliary_passes"].values())
        for entry in entries:
            if entry["kind"] == "cnb":
                self.assertIn(entry["kernel"], self.cnb_tiles)
                width, height, groups = self.cnb_tiles[entry["kernel"]]
                self.assertEqual(entry["tile"], [width, height])
                self.assertEqual(entry["threads"], groups * 32)
                self.assertEqual(entry["function"], entry["kernel"])
                continue
            if entry["kind"] not in ("fnb", "ct"):
                continue
            self.assertIn(entry["kernel"], self.kernel_tiles)
            nf, width, height, up, groups = self.kernel_tiles[entry["kernel"]]
            self.assertEqual(entry["tile"], [width, height])
            self.assertEqual(entry["input"][5], nf)
            self.assertEqual(entry["kind"] == "ct", up)
            self.assertEqual(entry["function"], entry["kernel"])
            self.assertEqual(entry["threads"], groups * 32)

    def test_legacy_kernel_entries_keep_cached_dispatch_shapes(self):
        # A --no-compile cache can outlive tile tuning. Its original entry
        # name must continue to mean the same spatial tile.
        for name, shape in {
            "fsr411_fnb32": (32, 32, 2, False, 4),
            "fsr411_fnb64": (64, 32, 1, False, 4),
            "fsr411_ct32": (32, 32, 2, True, 4),
            "fsr411_ct64": (64, 32, 1, True, 4),
            "fsr411_fnb64_16x1": (64, 16, 1, False, 4),
            "fsr411_ct64_16x1": (64, 16, 1, True, 4),
            "fsr411_ct32_32x1": (32, 32, 1, True, 4),
        }.items():
            self.assertEqual(self.kernel_tiles[name], shape)
        self.assertEqual(self.cnb_tiles["fsr411_cnb"], (32, 4, 4))
        self.assertEqual(self.cnb_tiles["fsr411_cnb_32x1_sg1"], (32, 1, 1))
        self.assertEqual(self.cnb_tiles["fsr411_cnb_coop_chain_32x1"], (32, 1, 1))
        self.assertEqual(self.kernel_tiles["fsr411_fnb32_coop_chain_16x1"], (32, 16, 1, False, 1))
        self.assertEqual(self.kernel_tiles["fsr411_ct32_coop_chain_16x1"], (32, 16, 1, True, 1))

    def test_literal_dot_order_and_native_matrix_layout(self):
        rng = random.Random(411)
        for number in (1, 2, 12):
            model = native.parse_cnb(self.source_dir / "reimpl411" / f"reimpl_pass{number}.hlsl", self.initializer)
            for width, height, extent in ((1, 1, 0), (3, 2, 3), (7, 5, 128)):
                image = [[[rng.randrange(-extent, extent) if extent else 0 for _ in range(16)] for _ in range(width)] for _ in range(height)]
                for y in range(height):
                    for x in range(width):
                        self.assertEqual(literal_pixel(model, image, x, y), matrix_pixel(model, image, x, y), (number, width, height, x, y))

    def test_all_tiers_and_abi(self):
        with tempfile.TemporaryDirectory() as tmp:
            for tier in (1080, 2160, 4320):
                out = Path(tmp) / str(tier)
                native.encode_model(self.source_dir, ROOT / "third_party/fsr411/initializers.bin", tier, out)
                params = (out / "fsr411_native_params.bin").read_bytes()
                self.assertEqual(len(params), 13 * 256)
                manifest = json.loads((out / "fsr411_native_manifest.json").read_text())
                self.assert_native_dispatch_contract(manifest)
                weights = (out / "fsr411_native_weights.bin").read_bytes()
                self.assertEqual(manifest["weights_bytes"], len(weights))
                self.assertEqual(manifest["weights_sha256"], hashlib.sha256(weights).hexdigest())
                self.assertEqual(manifest["params_bytes"], len(params))
                self.assertEqual(manifest["params_sha256"], hashlib.sha256(params).hexdigest())
                self.assertEqual(set(manifest["passes"]), {str(x) for x in range(1, 13)})
                for index, divisor, channels in ((0, 1, 16), (1, 1, 16), (2, 1, 16), (3, 2, 32),
                                                 (4, 2, 32), (5, 2, 32), (6, 4, 64), (7, 4, 64),
                                                 (8, 4, 64), (9, 2, 32), (10, 2, 32), (11, 1, 16)):
                    offset = index * 256
                    words = struct.unpack_from("<53I", params, offset)
                    self.assertEqual(words[3:6], (tier * 8 // 9 // divisor, tier // 2 // divisor, channels))
                    self.assertEqual(params[offset + 212:offset + 256], bytes(44))
                post = manifest["auxiliary_passes"]["post_cnb"]
                self.assertEqual(post["parameter_offset"], 12 * 256)
                self.assertEqual(post["input_slot"], 13)
                self.assertEqual(post["output_slot"], 13)
                self.assertEqual(post["output_offset_base"], "scratch_append")
                self.assertEqual(post["output"], [0, 16, tier * 8 // 9 * 16, tier * 8 // 9, tier // 2, 16, 0])
                self.assertEqual(post["scratch_append_bytes"], (tier * 8 // 9) * (tier // 2) * 16)
                self.assertEqual(post["q"][2:4], [[0, 6, 6, 0], [0, 6, 6, 0]])

    def test_post_literal_weights_and_residual_tail(self):
        model = native.parse_post_cnb(self.source_dir / "reimpl411/post_rev_411.hlsl",
                                      self.source_dir / "reimpl411/post_rev_411_tables.hlsli", 1080)
        rng = random.Random(41126)
        for width, height, extent in ((1, 1, 0), (3, 5, 3), (7, 5, 128)):
            image = [[[rng.randrange(-extent, extent) if extent else 0 for _ in range(16)] for _ in range(width)] for _ in range(height)]
            for y in range(height):
                for x in range(width):
                    self.assertEqual(literal_post_pixel(model, image, x, y), matrix_pixel(model, image, x, y), (width, height, x, y))

    def test_post_rejects_changed_residual_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "post.hlsl"
            path.write_text((self.source_dir / "reimpl411/post_rev_411.hlsl").read_text().replace("(b4 + a) * 16.0f", "(b4 + a) * 15.0f"))
            with self.assertRaisesRegex(ValueError, "POST tail"):
                native.parse_post_cnb(path, self.source_dir / "reimpl411/post_rev_411_tables.hlsli", 1080)

    def test_ultraperformance_original_shader_contracts(self):
        # This table is an independent transcription of the original provider
        # DXIL, not the generator's implementation of the native epilogues.
        evidence = json.loads((ROOT / "tools/fsr411/native_ultraperformance_contracts.json").read_text())
        initializer = ROOT / evidence["initializer"]
        with tempfile.TemporaryDirectory() as tmp:
            for tier in (1080, 2160, 4320):
                out = Path(tmp) / str(tier)
                manifest_path = native.encode_model(self.source_dir, initializer, tier, out, "ultraperformance", self.up_dir)
                manifest = json.loads(manifest_path.read_text())
                self.assert_native_dispatch_contract(manifest)
                self.assertEqual(manifest["initializer_sha256"], evidence["initializer_sha256"])
                self.assertEqual(manifest["presets"], ["ultraperformance"])
                self.assertEqual(len((out / "fsr411_native_params.bin").read_bytes()), 13 * 256)
                for source in evidence["stages"]:
                    entry = manifest["auxiliary_passes"]["post_cnb"] if source["pass"] == "post" else manifest["passes"][str(source["pass"])]
                    q = entry["q"]
                    if source["requant"]:
                        self.assertEqual([q[0][2], q[1][2]], source["requant"], (tier, source["pass"]))
                    tails = source["tails"]
                    expected = [tails[0]["q"], tails[-1]["q"]]
                    self.assertEqual(q[4:6] if source["kind"] == "ds" else q[2:4], expected, (tier, source["pass"]))
                    if source["kind"] == "ct":
                        self.assertEqual(q[4:6], [source["ct_tail"]["q"]] * 2, (tier, source["pass"]))

    def test_ultraperformance_literal_weights(self):
        initializer = (ROOT / "third_party/fsr411/initializers-ultraperformance.bin").read_bytes()
        models = [(native.parse_cnb(self.up_dir / "reimpl411" / f"reimpl_pass{number}.hlsl", initializer), literal_pixel) for number in (1, 2, 12)]
        models.append((native.parse_post_cnb(self.up_dir / "reimpl411/post_rev_411.hlsl",
                                             self.up_dir / "reimpl411/post_rev_411_tables.hlsli", 1080), literal_post_pixel))
        rng = random.Random(4113)
        for model, oracle in models:
            for width, height, extent in ((1, 1, 0), (3, 5, 3), (7, 5, 128)):
                image = [[[rng.randrange(-extent, extent) if extent else 0 for _ in range(16)] for _ in range(width)] for _ in range(height)]
                for y in range(height):
                    for x in range(width):
                        self.assertEqual(oracle(model, image, x, y), matrix_pixel(model, image, x, y), (model["source"], width, height, x, y))

    def test_rejects_mixed_model_initializer(self):
        with tempfile.TemporaryDirectory() as tmp:
            for preset, name in (("normal", "initializers-ultraperformance.bin"), ("ultraperformance", "initializers.bin")):
                with self.assertRaisesRegex(ValueError, "verified"):
                    native.encode_model(self.source_dir, ROOT / "third_party/fsr411" / name, 1080, Path(tmp), preset, self.up_dir)

    def test_requires_ultraperformance_overlay(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                native.encode_model(self.source_dir, ROOT / "third_party/fsr411/initializers-ultraperformance.bin", 1080,
                                    Path(tmp) / "out", "ultraperformance", Path(tmp))

    def test_lossy_profile_isolated_weights_and_rounding(self):
        with tempfile.TemporaryDirectory() as tmp:
            for tier in (1080, 2160, 4320):
                for preset in ("normal", "ultraperformance"):
                    initializer = ROOT / "third_party/fsr411" / ("initializers.bin" if preset == "normal" else "initializers-ultraperformance.bin")
                    manifests, blobs = {}, {}
                    for profile in ("exact", "lossy"):
                        out = Path(tmp) / f"{tier}_{preset}_{profile}"
                        path = native.encode_model(self.source_dir, initializer, tier, out, preset, self.up_dir, profile)
                        manifests[profile] = json.loads(path.read_text())
                        self.assert_native_dispatch_contract(manifests[profile])
                        blobs[profile] = (out / "fsr411_native_weights.bin").read_bytes()
                        self.assertEqual(manifests[profile]["profile"], profile)
                    selected = (1, 2, 4, 5, 7, 8, 9, 10, 12) if preset == "normal" else (7, 8, 9)
                    folded = {1: 4, 5: 18, 10: 18, 12: 20} if preset == "normal" else {}
                    for number in range(1, 13):
                        exact = manifests["exact"]["passes"][str(number)]
                        loss = manifests["lossy"]["passes"][str(number)]
                        q = [list(x) for x in exact["q"]]
                        if number in selected:
                            q[0][3] |= 4
                            q[1][3] |= 4
                        self.assertEqual(loss["q"], q, (tier, preset, number))
                        self.assertEqual(exact["q"][2:], loss["q"][2:], "lossy must retain exact residual tails")
                        self.assertEqual(exact["weights"], loss["weights"])
                        begin = exact["weights"][0]
                        end = exact["weights"][1] if exact["weights"][1] != native.U32_MAX else exact["biases"][0]
                        if number in folded:
                            self.assertNotEqual(blobs["exact"][begin:end], blobs["lossy"][begin:end])
                            stats = loss["lossy_transform"]["conv3x3_folding"]
                            self.assertEqual(stats["drop_words_per_channel"], folded[number])
                            self.assertEqual(stats["dropped_words"], 16 * folded[number])
                        else:
                            self.assertEqual(blobs["exact"][begin:end], blobs["lossy"][begin:end])
                    self.assertEqual(manifests["exact"]["auxiliary_passes"], manifests["lossy"]["auxiliary_passes"])
                    if preset == "ultraperformance":
                        self.assertEqual(blobs["exact"], blobs["lossy"], "UP lossy changes shared rounding only")

    def test_fnb_embedded_weight_layouts(self):
        rng = random.Random(41164)
        for number in (4, 5, 10):
            model = native.parse_fnb(self.source_dir / "reimpl411" / f"reimpl_pass{number}.hlsl", self.initializer, 1080, number)
            w0, w1, w2 = model["literal_weights"]
            c0, c1, c2 = model["weights"]
            pixels = [[rng.randrange(-128, 128) for _ in range(16)] for _ in range(9)]
            for oc in range(16):
                ref = sum(dot4(pack(pixels[tap][word * 4:word * 4 + 4]), w0[(oc // 4) * 9 + tap][4 * (oc % 4) + word])
                          for tap in range(9) for word in range(4))
                got = sum(pixels[tap][ic] * c0[(tap * 16 + ic) * 16 + oc] for tap in range(9) for ic in range(16))
                self.assertEqual(ref, got)
            for inputs, outputs, table, matrix, halves in ((32, 64, w1, c1, 2), (64, 32, w2, c2, 4)):
                vector = [rng.randrange(-128, 128) for _ in range(inputs)]
                words = [pack(vector[i:i + 4]) for i in range(0, inputs, 4)]
                for oc in range(outputs):
                    ref = sum(dot4(words[half * 4 + word], table[(oc // 4) * halves + half][4 * (oc % 4) + word])
                              for half in range(halves) for word in range(4))
                    got = sum(v * matrix[ic * outputs + oc] for ic, v in enumerate(vector))
                    self.assertEqual(ref, got)

    def test_rejects_wrong_initializer(self):
        with tempfile.TemporaryDirectory() as tmp:
            wrong = Path(tmp) / "wrong.bin"
            wrong.write_bytes(bytes(131072))
            with self.assertRaisesRegex(ValueError, "verified"):
                native.encode_model(self.source_dir, wrong, 1080, Path(tmp) / "out")

    def test_rejects_non_power_of_two_tail(self):
        source = self.source_dir / "reimpl411/reimpl_pass1.hlsl"
        with tempfile.TemporaryDirectory() as tmp:
            wrong = Path(tmp) / source.name
            wrong.write_text(source.read_text().replace("(t1 + t2) * 64.0f", "(t1 + t2) * 63.0f"))
            with self.assertRaisesRegex(ValueError, "power of two"):
                native.parse_cnb(wrong, self.initializer)


class NativePublicationTests(unittest.TestCase):
    def test_cached_publication_preserves_unchanged_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            path = native._publish_native(out, {"schema": "test"}, b"weights", b"params")
            files = [out / "fsr411_native_weights.bin", out / "fsr411_native_params.bin", path]
            for file in files:
                os.utime(file, ns=(1234567890000000000, 1234567890000000000))
            before = [(file.stat().st_ino, file.stat().st_mtime_ns) for file in files]
            native._publish_native(out, {"schema": "test"}, b"weights", b"params")
            self.assertEqual(before, [(file.stat().st_ino, file.stat().st_mtime_ns) for file in files])
            native._publish_native(out, {"schema": "test"}, b"weights", b"new parameters")
            self.assertEqual(before[0], (files[0].stat().st_ino, files[0].stat().st_mtime_ns))
            self.assertNotEqual(before[1], (files[1].stat().st_ino, files[1].stat().st_mtime_ns))
            metadata = json.loads(path.read_text())
            self.assertEqual(metadata["params_bytes"], len(b"new parameters"))
            self.assertEqual(metadata["params_sha256"], hashlib.sha256(b"new parameters").hexdigest())

    def test_open_readers_keep_complete_prior_files(self):
        # Open descriptors must retain the entire old model while new opens
        # see the complete replacement. In-place truncate/write fails this.
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            old_weights, old_params = b"A" * 128448, b"a" * 3328
            path = native._publish_native(out, {"schema": "test"}, old_weights, old_params)
            old_manifest = path.read_bytes()
            with (out / "fsr411_native_weights.bin").open("rb") as weights, \
                    (out / "fsr411_native_params.bin").open("rb") as params, path.open("rb") as manifest:
                native._publish_native(out, {"schema": "test"}, b"B" * 128448, b"b" * 3328)
                self.assertEqual(weights.read(), old_weights)
                self.assertEqual(params.read(), old_params)
                self.assertEqual(manifest.read(), old_manifest)
            self.assertEqual((out / "fsr411_native_weights.bin").read_bytes(), b"B" * 128448)
            self.assertEqual((out / "fsr411_native_params.bin").read_bytes(), b"b" * 3328)
            self.assertEqual(list(out.glob("*.tmp")), [])

    def test_writer_lock_is_per_output_directory(self):
        code = """import importlib.util, sys
from pathlib import Path
spec = importlib.util.spec_from_file_location('native', sys.argv[1])
native = importlib.util.module_from_spec(spec)
spec.loader.exec_module(native)
print('ready', flush=True)
native._publish_native(Path(sys.argv[2]), {'schema': 'test'}, b'weights', b'params')
"""
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "locked"
            out.mkdir()
            with (out / native.PUBLICATION_LOCK).open("a+b") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                command = [sys.executable, "-c", code, str(ROOT / "tools/gen_fsr411_native.py"), str(out)]
                with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) as child:
                    try:
                        self.assertEqual(child.stdout.readline().strip(), "ready")
                        with self.assertRaises(subprocess.TimeoutExpired):
                            child.wait(timeout=0.15)
                        self.assertFalse((out / "fsr411_native_manifest.json").exists())
                        independent = native._publish_native(Path(tmp) / "other", {"schema": "test"}, b"other", b"params")
                        self.assertTrue(independent.exists())
                    finally:
                        fcntl.flock(lock, fcntl.LOCK_UN)
                    _, errors = child.communicate(timeout=10)
                    self.assertEqual(child.returncode, 0, errors)
            self.assertEqual((out / "fsr411_native_weights.bin").read_bytes(), b"weights")


if __name__ == "__main__":
    unittest.main()
