#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Compile and run focused GPU checks for the actual lossy HLSL helpers.

The fixtures call the generated PRE/POST helpers and the generated POST cache
lookup. Only the cache's byte base and physical row pitch are relocated to a
small test allocation; motion, depth, ownership and parity logic is unchanged.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def run(args):
    result = subprocess.run([str(x) for x in args], capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError("command failed: " + " ".join(map(str, args)) + "\n" + result.stdout + result.stderr)
    return result.stdout


def source(tier):
    path = ROOT / "tools/fsr411/lossy_shaders.py"
    spec = importlib.util.spec_from_file_location("fsr411_lossy_qa_source", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    dirname = "reimpl411" if tier == 1080 else "reimpl" + str(tier)
    suffix = "" if tier == 1080 else "_" + str(tier)
    post = module.apply_lossy_post((ROOT / "third_party/fsr411/shaders" / dirname / ("post_rev_411" + suffix + ".hlsl")).read_text(), tier)
    begin = post.index("        uint4 pixelCnb =")
    end = post.index("        int ct[8];", begin)
    lookup = post[begin:end]
    offset = (module.SCRATCH_BYTES[tier] + 255) & ~255
    lookup = lookup.replace(str(offset) + "u", "0u").replace(str(tier * 8 // 9 * 16) + "u", "qaCacheStride")
    if "ScratchBuffer.Load4(0u" not in lookup:
        raise ValueError("POST cache relocation did not match the generated shader")
    declarations = r"""
cbuffer QaFrame : register(b0) {
    uint width, height, width_lr, height_lr;
    uint recordCount, queryCount, qaCacheStride, qaReserved;
};
struct QaRecordInput {
    uint2 pixel;
    float2 velocity;
    int2 selectedDepth;
    uint invalidMotion;
    uint padding;
};
StructuredBuffer<QaRecordInput> QaRecords : register(t0);
Texture2D<float> r_depth : register(t1);
StructuredBuffer<int2> QaCells : register(t2);
RWStructuredBuffer<uint4> QaResults : register(u0);
RWByteAddressBuffer ScratchBuffer : register(u11);
bool InValidRegion(int2 p) {
    return all(p >= int2(0, 0)) && p.x < int(width_lr) && p.y < int(height_lr);
}
"""
    record = r"""
[numthreads(32, 1, 1)]
void qa_record(uint id : SV_DispatchThreadID) {
    if (id >= recordCount) return;
    QaRecordInput r = QaRecords[id];
    Fsr411LossyRecord(r.pixel, r.velocity, r.selectedDepth, r.invalidMotion != 0u);
}
"""
    query = r"""
[numthreads(32, 1, 1)]
void qa_query(uint id : SV_DispatchThreadID) {
    if (id >= queryCount) return;
    int2 cell = QaCells[id];
    int ix = cell.x, iy = cell.y;
    int w2 = int(width / 2u), h2 = int(height / 2u);
    uint4 externalCnb = ScratchBuffer.Load4(uint(iy) * qaCacheStride + uint(ix) * 16u);
    uint packedVss[4] = {externalCnb.x, externalCnb.y, externalCnb.z, externalCnb.w};
    uint4 lossyOwn = uint4(0u, 0u, 0u, 0u);
    bool atRest = false, uncovered = false;
    if (lossySkip != 0u) {
        lossyOwn = Fsr411LossyReadMotion(cell);
        Fsr411LossyClassify(cell, lossyOwn, atRest, uncovered);
    }
    QaResults[id * 9u] = uint4(lossyOwn.xy, uint(atRest), uint(uncovered));
    [loop] for (uint pk = 0; pk < 4; ++pk) {
        uint2 tid = uint2(ix * 2, iy * 2) + uint2(pk & 1u, pk >> 1u);
LOOKUP
        QaResults[id * 9u + pk * 2u + 1u] = pixelCnb;
        QaResults[id * 9u + pk * 2u + 2u] = uint4(cnbPixel, 0x411u, tid);
    }
}
""".replace("LOOKUP", lookup)
    return declarations + module._declarations(tier) + module.PRE_HELPERS + module.POST_HELPERS + record + query


def signature(msl, entry):
    body = re.search(r"kernel\s+void\s+" + entry + r"\s*\((.*?)\)\s*\{", msl, re.S)
    if not body:
        raise ValueError("Metal entry missing: " + entry)
    return {kind + "s": [[int(slot), name] for name, slot in re.findall(r"(\w+)\s*\[\[" + kind + r"\((\d+)\)\]\]", body[1])]
            for kind in ("buffer", "texture")}


def compile_fixture(args):
    out, tier, inverted, low = args
    folder = out / f"tier{tier}_di{inverted}_lm{low}"
    folder.mkdir(parents=True, exist_ok=True)
    hlsl = source(tier)
    path = folder / "fixture.hlsl"
    path.write_text(hlsl)
    metadata = {"tier": tier, "depth_inverted": inverted, "low_res_mv": low,
                "source_sha256": hashlib.sha256(hlsl.encode()).hexdigest(), "kernels": {}}
    for entry in ("qa_record", "qa_query"):
        spv, metal, air, lib = [folder / (entry + suffix) for suffix in (".spv", ".metal", ".air", ".metallib")]
        run([ROOT / "tools/bin/dxc", "-spirv", "-T", "cs_6_6", "-E", entry, "-HV", "2021", "-O3",
             "-fspv-target-env=vulkan1.3", "-fvk-t-shift", "32", "0", "-fvk-u-shift", "64", "0",
             "-DFFX_MLSR_LOW_RES_MV=" + str(low), "-DFFX_MLSR_DEPTH_INVERTED=" + str(inverted), "-Fo", spv, path])
        msl = run([ROOT / "tools/bin/spirv-cross", spv, "--msl", "--msl-version", "20300"])
        msl = re.sub(r"^([ \t]*)(for|while)\s*\(", r"\1#pragma clang loop unroll(disable)\n\1\2 (", msl, flags=re.M)
        metal.write_text(msl)
        run(["xcrun", "metal", "-std=macos-metal2.3", "-O3", "-c", metal, "-o", air])
        run(["xcrun", "metallib", air, "-o", lib])
        metadata["kernels"][entry] = {"library": lib.name, **signature(msl, entry)}
    (folder / "fixture.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return str(folder)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=ROOT / "build/tests/fsr411_lossy")
    parser.add_argument("--quick", action="store_true", help="1080 tier, ordinary depth and low-resolution motion only")
    parser.add_argument("--build-only", action="store_true")
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    configurations = [(args.outdir, 1080, 0, 1)] if args.quick else [
        (args.outdir, tier, inverted, low) for tier in (1080, 2160, 4320) for inverted in (0, 1) for low in (0, 1)]
    with ThreadPoolExecutor(max_workers=4) as executor:
        directories = list(executor.map(compile_fixture, configurations))
    if args.build_only:
        print("Built", len(directories), "lossy GPU fixtures")
        return
    binary = args.outdir / "fsr411_lossy_test"
    run(["xcrun", "clang++", "-std=c++17", "-O2", "-fobjc-arc", ROOT / "tests/fsr411_lossy_test.mm",
         "-framework", "Foundation", "-framework", "Metal", "-o", binary])
    for directory in directories:
        print(run([binary, directory]).strip(), flush=True)


if __name__ == "__main__":
    main()
