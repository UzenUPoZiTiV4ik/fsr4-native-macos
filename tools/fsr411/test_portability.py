"""PRE math and motion correctness, including DXIL and SPIR-V/MSL inspection."""

from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest

from portability import (apply_portability_fixes, VELOCITY_BEFORE, VELOCITY_AFTER,
                         CHROMA_DISTANCE_BEFORE, CHROMA_DISTANCE_AFTER)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    from compile_fsr411 import portability_source, quad_routing
    from fsr411.lossy_shaders import apply_lossy_pre
finally:
    sys.path.pop(0)


ROOT = Path(__file__).resolve().parents[2]
DXC = ROOT / "tools/bin/dxc"
SPIRV_CROSS = ROOT / "tools/bin/spirv-cross"


class SignedChromaDistanceTests(unittest.TestCase):
    def test_all_tiers_presets_profiles_and_routing_use_signed_squares(self):
        for tier, folder in ((1080, "reimpl411"), (2160, "reimpl2160"), (4320, "reimpl4320")):
            for preset in ("shaders", "ultraperformance"):
                # UP overlays model weights and POST; its PRE is inherited
                # from the common source unless the overlay supplies one.
                directory = ROOT / "third_party/fsr411" / preset / folder
                paths = list(directory.glob("pre_rev_411*.hlsl"))
                if not paths:
                    paths = list((ROOT / "third_party/fsr411/shaders" / folder).glob("pre_rev_411*.hlsl"))
                self.assertEqual(len(paths), 1)
                original = paths[0].read_text()
                direct = apply_portability_fixes(original)
                self.assertEqual(direct.replace(CHROMA_DISTANCE_AFTER, CHROMA_DISTANCE_BEFORE), original)
                for profile in ("exact", "lossy"):
                    for optimize in (0, 1):
                        with self.subTest(tier=tier, preset=preset, profile=profile, optimize=optimize):
                            source = quad_routing(original, tier) if optimize else original
                            if profile == "lossy":
                                source = apply_lossy_pre(source, tier)
                            generated = portability_source(source, keep_unroll=True)
                            self.assertEqual(generated.count(CHROMA_DISTANCE_AFTER), 1)
                            self.assertNotIn(CHROMA_DISTANCE_BEFORE, generated)
                            self.assertEqual(apply_portability_fixes(generated), generated)

    def test_unrecognized_or_duplicate_distance_fails(self):
        for source in (CHROMA_DISTANCE_BEFORE.replace(", 2)", ", 3)", 1),
                       CHROMA_DISTANCE_BEFORE + "\n" + CHROMA_DISTANCE_BEFORE,
                       CHROMA_DISTANCE_AFTER + "\n" + CHROMA_DISTANCE_BEFORE):
            with self.subTest(source=source):
                with self.assertRaisesRegex(ValueError, "PRE chroma distance"):
                    apply_portability_fixes(source)

    def test_other_powers_are_unchanged(self):
        unrelated = "float gain = pow(exposure, 2.0);\nfloat transfer = pow(max(color, 0.0), 2.4);\n"
        self.assertEqual(apply_portability_fixes(unrelated), unrelated)
        self.assertEqual(apply_portability_fixes(unrelated + CHROMA_DISTANCE_BEFORE),
                         unrelated + CHROMA_DISTANCE_AFTER)

    @staticmethod
    def shader(expression):
        # Runtime inputs keep positive/negative differences dynamic through
        # compilation. The feature is independent of color conversion/storage.
        return """
StructuredBuffer<float3> inputColors : register(t0);
StructuredBuffer<float3> historyColors : register(t1);
RWStructuredBuffer<float> result : register(u0);
[numthreads(1,1,1)]
void main(uint3 id : SV_DispatchThreadID) {
    float3 input_ycbcr = inputColors[id.x];
    float3 history_ycbcr = historyColors[id.x];
    """ + expression + "\n    result[id.x] = color_distance;\n}\n"

    @unittest.skipUnless(DXC.is_file(), "native DXC is not installed")
    def test_rewrite_preserves_native_dxil_signed_square_body(self):
        expressions = []
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "chroma.hlsl"
            output = Path(directory) / "chroma.dxil"
            for expression in (CHROMA_DISTANCE_BEFORE, apply_portability_fixes(CHROMA_DISTANCE_BEFORE)):
                source.write_text(self.shader(expression))
                built = subprocess.run([str(DXC), "-T", "cs_6_6", "-E", "main", "-HV", "2021", "-O3",
                                        "-Fo", str(output), str(source)], capture_output=True, text=True)
                self.assertEqual(built.returncode, 0, built.stderr)
                ir = subprocess.run([str(DXC), "-dumpbin", str(output)], check=True,
                                    capture_output=True, text=True).stdout
                body = re.search(r"^define void @main\(.*?^\}", ir, re.M | re.S)[0]
                self.assertEqual(len(re.findall(r"fmul fast float (%\d+), \1\b", body)), 2)
                self.assertIn("Sqrt(value)", body)
                self.assertNotRegex(body, r"\b(?:Log|Exp|FAbs)\(")
                # Naming both deltas can schedule their independent subtractions
                # earlier. Compare the store's SSA expression, not instruction IDs.
                values, stores = {}, []
                for line in body.splitlines():
                    expression = re.sub(r"%\d+\b", lambda m: "(" + values[m[0]] + ")" if m[0] in values else m[0],
                                        line.split(";", 1)[0].strip())
                    definition = re.match(r"(%\d+) = (.*)", line.strip())
                    if definition:
                        rhs = definition[2].split(";", 1)[0].strip()
                        values[definition[1]] = re.sub(r"%\d+\b", lambda m: "(" + values[m[0]] + ")" if m[0] in values else m[0], rhs)
                    elif re.search(r"call void @dx\.op\.(?:rawBufferStore|bufferStore)\.f32", line):
                        stores.append(expression)
                self.assertEqual(len(stores), 1)
                expressions.append(stores[0])
        self.assertEqual(expressions[0], expressions[1])

    @unittest.skipUnless(DXC.is_file() and SPIRV_CROSS.is_file(), "DXC/SPIRV-Cross are not installed")
    def test_spirv_to_msl_keeps_signed_products_instead_of_powr(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "chroma.hlsl"
            output = Path(directory) / "chroma.spv"
            source.write_text(self.shader(apply_portability_fixes(CHROMA_DISTANCE_BEFORE)))
            built = subprocess.run([str(DXC), "-spirv", "-T", "cs_6_6", "-E", "main", "-HV", "2021", "-O3",
                                    "-fspv-target-env=vulkan1.3", "-fvk-t-shift", "32", "0",
                                    "-fvk-u-shift", "64", "0", "-Fo", str(output), str(source)],
                                   capture_output=True, text=True)
            self.assertEqual(built.returncode, 0, built.stderr)
            msl = subprocess.run([str(SPIRV_CROSS), str(output), "--msl", "--msl-version", "20300"],
                                 check=True, capture_output=True, text=True).stdout
            self.assertNotRegex(msl, r"\b(?:powr?|abs|log2?|exp2?)\s*\(")
            self.assertEqual(len(re.findall(r"\b(_\d+)\s*\*\s*\1\b", msl)), 2)
            self.assertIn("sqrt(", msl)


class MotionViewportTests(unittest.TestCase):
    def test_only_motion_helper_changes_and_fix_is_idempotent(self):
        source = (ROOT / "third_party/fsr411/shaders/reimpl411/mlsr_optimized_includes.hlsli").read_text()
        changed = apply_portability_fixes(source)
        self.assertEqual(changed.replace(VELOCITY_AFTER, VELOCITY_BEFORE), source)
        self.assertEqual(apply_portability_fixes(changed), changed)
        self.assertIn("InValidRegion(center_pos) ? depth_tex[center_pos] : 0.0f", changed)

    def test_unknown_motion_helper_fails(self):
        with self.assertRaisesRegex(ValueError, "unrecognized"):
            apply_portability_fixes("float2 LoadInputVelocity(int2 pos) { return 0; }")

    @unittest.skipUnless(DXC.is_file(), "native DXC is not installed")
    def test_display_motion_bounds_come_from_texture_dimensions(self):
        # Compile the actual generated helper. A high-resolution permutation
        # must query the bound texture, rather than clip to a render-size CBV
        # or to padded output constants. Low-resolution motion needs no query.
        prefix = """
Texture2D<float2> r_velocity : register(t1);
RWStructuredBuffer<float2> result : register(u0);
cbuffer C : register(b1) {
    uint width_lr, height_lr;
    float2 mv_scale, fMotionVectorJitterCancellation;
};
"""
        suffix = """
[numthreads(1,1,1)]
void main(uint3 id : SV_DispatchThreadID) {
    result[id.x] = LoadInputVelocity(int2(id.xy));
}
"""
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "motion.hlsl"
            compiled = Path(directory) / "motion.dxil"
            source.write_text(prefix + apply_portability_fixes(VELOCITY_BEFORE) + suffix)
            for low_res in (0, 1):
                command = [str(DXC), "-T", "cs_6_6", "-E", "main", "-O3", "-HV", "2021",
                           "-DFFX_MLSR_LOW_RES_MV=" + str(low_res),
                           "-DFFX_MLSR_JITTERED_MOTION_VECTORS=1", "-Fo", str(compiled), str(source)]
                built = subprocess.run(command, capture_output=True, text=True)
                self.assertEqual(built.returncode, 0, built.stderr)
                dumped = subprocess.run([str(DXC), "-dumpbin", str(compiled)],
                                        check=True, capture_output=True, text=True).stdout
                self.assertEqual("@dx.op.getDimensions" in dumped, low_res == 0)


class DiscardedStoreTests(unittest.TestCase):
    def test_all_imported_dummy_addresses_become_guarded_stores(self):
        count = 0
        for directory in ("shaders", "ultraperformance"):
            for path in (ROOT / "third_party/fsr411" / directory).rglob("*.hlsl"):
                source = path.read_text()
                if ": 134217728u;" not in source:
                    continue
                changed = apply_portability_fixes(source)
                self.assertNotIn("134217728u", changed)
                self.assertIn("if (inRange) ScratchBuffer.Store(addr, val);", changed)
                self.assertEqual(apply_portability_fixes(changed), changed)
                count += 1
        self.assertEqual(count, 21)  # 4 common stages and 3 UP stages, each in 3 tiers

    @unittest.skipUnless(DXC.is_file(), "native DXC is not installed")
    def test_discarded_lane_has_no_store_in_compiled_shader(self):
        source = """
RWByteAddressBuffer ScratchBuffer : register(u11);
[numthreads(64,1,1)]
void main(uint3 id : SV_DispatchThreadID) {
    bool inRange = id.x < 5;
    uint outRow = id.x * 4, d = 0, val = inRange ? 1 : 0;
    const uint addr = inRange ? (outRow + 4u * d) : 134217728u;
    ScratchBuffer.Store(addr, val);
}
"""
        with tempfile.TemporaryDirectory() as directory:
            path, output = Path(directory) / "store.hlsl", Path(directory) / "store.dxil"
            path.write_text(apply_portability_fixes(source))
            subprocess.run([str(DXC), "-T", "cs_6_6", "-E", "main", "-O3", "-Fo", str(output), str(path)],
                           check=True, capture_output=True)
            ir = subprocess.run([str(DXC), "-dumpbin", str(output)],
                                check=True, capture_output=True, text=True).stdout
            self.assertNotIn("134217728", ir)
            self.assertIn("br i1", ir)


if __name__ == "__main__":
    unittest.main()
