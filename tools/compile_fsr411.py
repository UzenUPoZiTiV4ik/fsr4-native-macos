#!/usr/bin/env python3
"""Build the real FSR 4.1.1 INT8 graph for Metal, with a content-addressed cache.

The imported MIT HLSL remains unchanged. Metal portability transforms live in
the generated include tree; every output records its inputs and tool options.
The model has 27 ordered stages (including thirteen padding stages), plus AE
and RCAS. Model selection is independent of the render/output scale ratio.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

from compile_variant import build_shadow_tree, parse_signature, patch_spv, wgsize
from fsr411.portability import apply_portability_fixes
from fsr411_msl import buffer_integer_constants
from fsr411.lossy import apply_hlsl_lossy
from fsr411.lossy_shaders import apply_lossy_pre, apply_lossy_post, apply_lossy_post_cells
from fsr411.post_optimization import apply_post_optimization, POST_PIXELS_PER_GROUP
from fsr411.autoexposure import split_autoexposure

TIERS = {1080: "reimpl411", 2160: "reimpl2160", 4320: "reimpl4320"}


def run(args: list[str], log: Path | None = None) -> bytes:
    result = subprocess.run([str(x) for x in args], capture_output=True)
    if log:
        log.write_bytes(result.stdout + result.stderr)
    if result.returncode:
        raise RuntimeError(f"{Path(args[0]).name} failed:\n" +
                           (result.stdout + result.stderr).decode(errors="replace")[-12000:])
    return result.stdout


def portability_source(text: str, keep_unroll: bool = False) -> str:
    # Vulkan forbids half sampled-image types. Actual RGBA16F storage performs
    # the same widening/narrowing; explicit half arithmetic/casts are preserved.
    text = re.sub(r"(RW)?Texture2D<half([234]?)>",
                  lambda m: (m[1] or "") + "Texture2D<float" + m[2] + ">", text)
    # The full ML2Code unroll exceeds the Metal driver's pipeline compiler
    # budget. Keeping loop order also gives a stable reference for A/B checks.
    text = apply_portability_fixes(text)
    text = text.replace('"../pre_rev_411/ffx_fsr4_gpu_colorspace_common.h"',
                        '"ffx_fsr4_gpu_colorspace_common.h"')
    return text if keep_unroll else re.sub(r"\[\s*unroll[^\]]*\]", "[loop]", text)


def quad_routing(text: str, tier: int) -> str:
    """Route the already-rounded half inputs, then compute one packed word/lane.

    Each lane evaluates four channels with the original dot2add chain and
    (tap0+tap1)+(tap2+tap3) reduction. Three uint4 exchanges replace thirty-two
    float exchanges; four neighbouring lanes issue neighbouring dword stores.
    This is a fresh implementation on the MIT source, not a GPL patch import.
    """
    start = text.index("        float accumulators[16];")
    end = text.index("            ScratchBuffer.Store4(byteAddress, storageBytes);", start)
    end = text.index("\n        }", end) + len("\n        }")
    stride = (tier * 8 // 9 + 2) * 16
    replacement = r"""        const uint lane = LocalThreadId.x & 3u;
        uint4 local = uint4(Pack2h(downscaleInputs[0].xy), Pack2h(downscaleInputs[0].zw),
                            Pack2h(downscaleInputs[1].xy), Pack2h(downscaleInputs[1].zw));
        uint4 acrossX = QuadReadAcrossX(local);
        uint4 acrossY = QuadReadAcrossY(local);
        uint4 acrossD = QuadReadAcrossDiagonal(local);
        // Constant-index construction avoids a dynamically indexed private
        // array (and register spills) after the quad exchange.
        uint4 inputs[4] = {
            lane == 0u ? local : lane == 1u ? acrossX : lane == 2u ? acrossY : acrossD,
            lane == 1u ? local : lane == 0u ? acrossX : lane == 3u ? acrossY : acrossD,
            lane == 2u ? local : lane == 3u ? acrossX : lane == 0u ? acrossY : acrossD,
            lane == 3u ? local : lane == 2u ? acrossX : lane == 1u ? acrossY : acrossD
        };
        float4 sums;
        [unroll] for (uint ch = 0; ch < 4; ++ch) {
            uint channel = lane * 4u + ch;
            float taps[4];
            [unroll] for (uint tap = 0; tap < 4; ++tap) {
                uint offset = channel * 32u + (tap >> 1u) * 512u + (tap & 1u) * 16u;
                uint4 weights = InitializerBuffer.Load4(offset);
                float a = 0;
                a = dot2add(Unpack2h(weights.x), Unpack2h(inputs[tap].x), a);
                a = dot2add(Unpack2h(weights.y), Unpack2h(inputs[tap].y), a);
                a = dot2add(Unpack2h(weights.z), Unpack2h(inputs[tap].z), a);
                a = dot2add(Unpack2h(weights.w), Unpack2h(inputs[tap].w), a);
                taps[tap] = a;
            }
            sums[ch] = (taps[0] + taps[1]) + (taps[2] + taps[3]);
        }
        float4 bias = asfloat(InitializerBuffer.Load4(1024u + lane * 16u));
        int4 quantized = round((sums + bias) * 64.0f);
        uint packed = pack_clamp_s8(quantized);
        uint sx = (gid.x << 3u) | (tileStartX >> 1u);
        uint sy = ((gid.y << 3u) | 1u) + (LocalThreadId.x >> 5u);
        ScratchBuffer.Store(sy * STRIDEu + (sx + 1u) * 16u + lane * 4u, packed);"""
    return text[:start] + replacement.replace("STRIDEu", str(stride) + "u") + text[end:]


def external_post_cnb(text: str, tier: int) -> str:
    start = text.index("    int acc[16];")
    end = text.index("#if FFX_MLSR_COLORSPACE == FFX_MLSR_COLORSPACE_LINEAR", start)
    scratch = {1080: 20880256, 2160: 83232256, 4320: 332352256}[tier]
    offset = (scratch + 255) & ~255
    stride = tier * 8 // 9 * 16
    replacement = ("    // Native CNB produced exactly the original packedVss values.\n"
                   f"    uint4 externalCnb = ScratchBuffer.Load4({offset}u + (uint)iy * {stride}u + (uint)ix * 16u);\n"
                   "    uint packedVss[4] = {externalCnb.x, externalCnb.y, externalCnb.z, externalCnb.w};\n\n")
    return text[:start] + replacement + text[end:]


FAST_MATH_PROFILES = ("exact-fast", "lossy")


def math_flags(name: str, profile: str, strict: bool = False) -> list[str]:
    """Metal floating-point policy for a generated stage.

    Exact PRE/POST and AE keep ordered IEEE arithmetic: no reassociation, no
    FMA contraction, precise division and functions. Exact is byte-identical
    to the imported reference; AE matches its imported oracle.
    Exact-fast and Lossy PRE/POST use Metal's relaxed math, as FSR 4.0.2
    does with its default fast math: additions may be reordered and fused,
    division and exp/log/pow/sqrt are approximate. Relaxed (not fast) mode
    keeps Inf/NaN semantics, so PRE's colour sanitisation stays intact.
    Reordering moves a few INT8 PRE values by one; on the game captures the
    result stays ≥ 63 dB from the ordered build, with no non-finite output.
    `strict` (tests only, its own `_sm1` build) keeps additions ordered and
    unfused, so `--optimize 0/1` and the imported HLSL match bit for bit.
    """
    if name not in ("pre", "post", "ae", "ae_finalize"):
        return []
    if name in ("pre", "post") and profile in FAST_MATH_PROFILES:
        if strict:
            return ["-fno-fast-math", "-freciprocal-math", "-fmetal-math-fp32-functions=fast", "-ffp-contract=off"]
        return ["-fmetal-math-mode=relaxed", "-fmetal-math-fp32-functions=fast"]
    return ["-fno-fast-math", "-ffp-contract=off"]


def input_hash(root: Path, options: dict) -> str:
    digest = hashlib.sha256(json.dumps(options, sort_keys=True).encode())
    paths = [Path(__file__).resolve(), root / "tools/compile_variant.py", root / "tools/fsr411/portability.py", root / "tools/fsr411_msl.py",
             root / "tools/fsr411/lossy.py", root / "tools/fsr411/lossy_shaders.py",
             root / "tools/fsr411/post_optimization.py", root / "tools/fsr411/autoexposure.py"]
    for directory in ("third_party/fsr411/shaders", "third_party/fsr411/ultraperformance", "third_party/fsr4/dx12",
                      "third_party/fsr4/include/gpu", "third_party/fsr4/api/internal/gpu"):
        paths.extend(p for p in (root / directory).rglob("*") if p.is_file())
    for path in sorted(set(paths)):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    for tool in ("dxc", "spirv-cross", "spirv-val"):
        digest.update(hashlib.sha256((root / "tools/bin" / tool).read_bytes()).digest())
    digest.update(run(["xcrun", "metal", "--version"]))
    return digest.hexdigest()


def stage_name(index: int) -> str:
    if index == 0:
        return "pre"
    if index == 26:
        return "post"
    return f"pad{index // 2}" if index & 1 else f"pass{index // 2}"


def build(args) -> dict:
    root = Path(args.root).resolve()
    builddir = Path(args.builddir).resolve()
    tier = int(args.res)
    if args.profile == "lossy":
        args.external_post_cnb = 1
    strict = bool(args.strict_math) and args.profile in FAST_MATH_PROFILES
    res_index = list(TIERS).index(tier)
    options = {"version": "4.1.1", "tier": tier, "depth_inverted": args.depth_inverted,
               "low_res_mv": args.low_res_mv, "jittered_mv": args.jittered_mv,
               "colorspace": args.colorspace, "auto_exposure": args.auto_exposure,
               "optimize": args.optimize, "ultra_performance": args.ultra_performance,
               "external_post_cnb": args.external_post_cnb, "profile": args.profile}
    if strict:
        options["strict_math"] = 1
    key = (f"fsr411_{tier}_di{args.depth_inverted}_lm{args.low_res_mv}"
           f"_jm{args.jittered_mv}_cs{args.colorspace}_ae{args.auto_exposure}_o{args.optimize}"
           + ("_up1" if args.ultra_performance else "")
           + {"lossy": "_lossy", "exact-fast": "_fast"}.get(args.profile, "")
           + ("_pc1" if args.external_post_cnb else "")
           + ("_sm1" if strict else ""))
    output = builddir / key
    output.mkdir(parents=True, exist_ok=True)
    shared_digest = input_hash(root, {})
    source_digest = hashlib.sha256((shared_digest + json.dumps(options, sort_keys=True)).encode()).hexdigest()
    manifest_path = output / "bindings.json"
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    if previous.get("source_hash") != source_digest:
        previous = {}

    # A local copy ensures quoted includes find the converted resource types.
    hlsl = output / "hlsl"
    hlsl.mkdir(exist_ok=True)
    source_root = root / "third_party/fsr411/shaders"
    directories = [source_root / folder for folder in dict.fromkeys(("reimpl411", TIERS[tier], "ae_411_ours"))]
    if args.ultra_performance:
        overlay = root / "third_party/fsr411/ultraperformance"
        if not (overlay / "reimpl411/reimpl_pass1.hlsl").exists():
            raise ValueError("Ultra Performance source overlay has not been imported")
        directories += [overlay / folder for folder in dict.fromkeys(("reimpl411", TIERS[tier]))]
    for directory in directories:
        for path in directory.glob("*"):
            if path.suffix in (".h", ".hlsl", ".hlsli"):
                source = path.read_text()
                if args.profile == "lossy" and (match := re.match(r"reimpl_pass(\d+)(?:_\d+)?\.hlsl$", path.name)):
                    source = apply_hlsl_lossy(source, int(match[1]), "ultraperformance" if args.ultra_performance else "normal")
                if path.name.startswith("pre_rev_") and path.suffix == ".hlsl" and args.optimize:
                    source = quad_routing(source, tier)
                if path.name.startswith("post_rev_") and path.suffix == ".hlsl" and args.external_post_cnb:
                    source = external_post_cnb(source, tier)
                if args.profile == "lossy" and path.suffix == ".hlsl":
                    if path.name.startswith("pre_rev_"):
                        source = apply_lossy_pre(source, tier)
                    elif path.name.startswith("post_rev_"):
                        source = apply_lossy_post(source, tier)
                if (path.name.startswith("post_rev_") and path.suffix == ".hlsl"
                        and args.external_post_cnb and args.optimize):
                    source = apply_post_optimization(source)
                    if args.profile == "lossy":
                        source = apply_lossy_post_cells(source)
                # Tiny PRE/filter loops should remain unrolled: their dynamic
                # arrays otherwise spill heavily. Only the large scalar model
                # oracle needs loop preservation to keep the driver stable.
                keep_unroll = (path.name.startswith("pre_rev_") or path.name == "mlsr_optimized_includes.hlsli"
                               or (path.name.startswith("post_rev_") and args.external_post_cnb))
                (hlsl / path.name).write_text(portability_source(source, keep_unroll))
    for name, source in split_autoexposure((source_root / "ae_411_ours/ae_ours.hlsl").read_text()).items():
        (hlsl / name).write_text(portability_source(source))
    rcas_source = root / "third_party/fsr4/internal/shaders/rcas.hlsl"
    rcas_text = rcas_source.read_text().replace('"fsr4/ffx_fsr4_', '"ffx_fsr4_')
    (hlsl / "rcas.hlsl").write_text(portability_source(rcas_text))
    shared_shadow = Path(build_shadow_tree(str(root), str(builddir / "fsr411_support")))
    includes = [hlsl, shared_shadow, root / "third_party/fsr4/dx12",
                root / "third_party/fsr4/include/gpu", root / "third_party/fsr4/include/gpu/fsr4",
                root / "third_party/fsr4/api/internal/gpu"]
    imported_manifest = next((source_root / TIERS[tier]).glob("manifest*.json"))
    imported = json.loads(imported_manifest.read_text())
    stages = []
    for stage in imported["passes"]:
        match = re.fullmatch(r"FFX_FSR4_PASS_(\d+)", stage["pix_pass"])
        if match:
            index = int(match[1])
            stages.append({"name": stage_name(index), "index": index,
                           "source": Path(stage["kernel"]).name})
    stages.sort(key=lambda s: s["index"])
    if [s["index"] for s in stages] != list(range(27)):
        raise ValueError("FSR 4.1.1 manifest must contain all 27 ordered graph stages")
    stages += [{"name": "ae", "index": -1, "source": "ae_tiles.hlsl"},
               {"name": "ae_finalize", "index": -1, "source": "ae_finalize.hlsl"},
               {"name": "rcas", "index": 27, "source": "rcas.hlsl"}]
    toolroot = root / "tools/bin"
    kernels = previous.get("kernels", {})
    def save_manifest():
        result = {**options, "key": key, "source_hash": source_digest,
                  "scratch_bytes": imported["scratch_arena"]["size"],
                  "post_pixels_per_group": POST_PIXELS_PER_GROUP if args.optimize and args.external_post_cnb else 32,
                  "auto_exposure_dispatches": 2,
                  "order": [s["name"] for s in stages if 0 <= s["index"] <= 26],
                  "kernels": kernels}
        temp = manifest_path.with_suffix(".tmp")
        temp.write_text(json.dumps(result, indent=2) + "\n")
        temp.replace(manifest_path)

    # Most flags affect PRE/POST only. Reuse identical compiled model stages
    # across MV, colour and exposure variants, with an input/tool content hash.
    kernel_cache = builddir / "fsr411_kernel_cache"
    def kernel_cache_dir(stage, source):
        name = stage["name"]
        relevant = ({k: options[k] for k in ("tier", "depth_inverted", "low_res_mv", "jittered_mv", "colorspace", "auto_exposure", "optimize")}
                    if name == "pre" else {"colorspace": args.colorspace} if name == "post"
                    else {"colorspace": args.colorspace, "auto_exposure": args.auto_exposure} if name == "rcas" else {})
        identity = (shared_digest + name + json.dumps(relevant, sort_keys=True) + source
                    + json.dumps(math_flags(name, args.profile, strict)))
        return kernel_cache / hashlib.sha256(identity.encode()).hexdigest()
    only = set(args.only or [s["name"] for s in stages])
    unknown = only - {s["name"] for s in stages}
    if unknown:
        raise ValueError(f"unknown stages: {sorted(unknown)}")
    defines = [f"FFX_MLSR_RESOLUTION={res_index}", f"FFX_MLSR_DEPTH_INVERTED={args.depth_inverted}",
               f"FFX_MLSR_LOW_RES_MV={args.low_res_mv}",
               f"FFX_MLSR_JITTERED_MOTION_VECTORS={args.jittered_mv}",
               f"FFX_MLSR_COLORSPACE={args.colorspace}",
               f"FFX_MLSR_AUTOEXPOSURE_ENABLED={args.auto_exposure}",
               "FFX_DEBUG_VISUALIZE=0", "DATA_TYPE=float", "DATA_TYPE2=float2",
               "DATA_TYPE3=float3", "DATA_TYPE_VECTOR=float4", "NETWORK_DATA_TYPE=float16_t"]
    for stage in stages:
        name = stage["name"]
        if name not in only:
            continue
        lib_path = output / f"{name}.metallib"
        if name in kernels and lib_path.exists():
            continue
        source = hlsl / stage["source"]
        text = source.read_text()
        cache = kernel_cache_dir(stage, text)
        cached_meta = cache / "binding.json"
        if cached_meta.exists() and (cache / f"{name}.metallib").exists():
            info = json.loads(cached_meta.read_text())
            if info.get("library_sha256") == hashlib.sha256((cache / f"{name}.metallib").read_bytes()).hexdigest():
                for file in cache.glob(name + ".*"):
                    shutil.copyfile(file, output / file.name)
                kernels[name] = info
                save_manifest()
                continue
        entry = re.search(r"\[numthreads\([^\]]+\)\]\s*void\s+(\w+)\s*\(", text)
        if not entry:
            raise ValueError(f"missing compute entry in {source}")
        entry = entry[1]
        spv, metal, air = [output / f"{name}.{ext}" for ext in ("spv", "metal", "air")]
        command = [toolroot / "dxc", "-spirv", "-T", "cs_6_6", "-E", entry,
                   "-enable-16bit-types", "-HV", "2021", "-O3", "-Vd",
                   "-fspv-target-env=vulkan1.3", "-fvk-t-shift", "32", "0",
                   "-fvk-u-shift", "64", "0", "-Fo", spv]
        for inc in includes:
            command += ["-I", inc]
        command += ["-D" + define for define in defines]
        if name in ("ae", "ae_finalize"):
            command += ["-DFFX_GPU=1", "-DFFX_HLSL=1", "-DFFX_HLSL_SM=66"]
        command.append(source)
        print(f"FSR 4.1.1: compiling {name}", file=sys.stderr, flush=True)
        run(command, output / f"{name}.dxc.log")
        patch_spv(str(spv))
        run([toolroot / "spirv-val", "--target-env", "vulkan1.3", spv], output / f"{name}.validation.log")
        result = run([toolroot / "spirv-cross", spv, "--msl", "--msl-version", "20300"])
        msl = result.decode().replace(".xyzz, uint2", ".xyzx, uint2")
        # SPIRV-Cross drops HLSL [loop] / SPIR-V DontUnroll metadata. Without
        # a Metal hint the driver expands the nested integer loops again and
        # its pipeline compiler can exhaust its XPC service memory budget.
        msl = re.sub(r"^([ \t]*)(for|while)\s*\(",
                     r"\1#pragma clang loop unroll(disable)\n\1\2 (", msl, flags=re.M)
        metal_entry = "fsr411_" + name
        msl, renamed = re.subn(r"(kernel\s+void\s+)\w+(\s*\()",
                              rf"\g<1>{metal_entry}\2", msl, count=1)
        if renamed != 1:
            raise ValueError(f"missing Metal entry for {name}")
        msl, table_bytes = buffer_integer_constants(msl)
        tables_path = output / f"{name}.tables.bin"
        if table_bytes:
            tables_path.write_bytes(table_bytes)
        elif tables_path.exists():
            tables_path.unlink()
        # Main is shared by all padding shaders, so each has its own library.
        metal.write_text(msl)
        # Apple's current driver also crashes optimizing the large FNB scalar
        # control flow, independently of constant propagation. These kernels
        # are the diagnostic HLSL oracle; native MPP kernels compile separately
        # with their normal optimization level. PRE/POST/padding stay fast.
        optimization = "-O0" if name.startswith("pass") else "-O3"
        # Moving PRE's quad exchanges exposes the floating-point reduction to
        # reassociation. Fast-math then changes a few INT8 rounding decisions,
        # which the network amplifies. Use the same ordered arithmetic in the
        # original and optimized PRE/POST, preserving half products as well.
        stage_math = math_flags(name, args.profile, strict)
        run(["xcrun", "metal", "-std=macos-metal2.3", optimization, *stage_math,
             "-c", metal, "-o", air], output / f"{name}.metal.log")
        run(["xcrun", "metallib", air, "-o", lib_path])
        buffers, textures, samplers = parse_signature(msl, metal_entry)
        kernels[name] = {**stage, "entry": metal_entry, "library": lib_path.name,
                         "wgsize": wgsize(str(spv), entry, str(toolroot / "spirv-cross")),
                         "buffers": buffers, "textures": textures, "samplers": samplers,
                         "spv_sha256": hashlib.sha256(spv.read_bytes()).hexdigest(),
                         "library_sha256": hashlib.sha256(lib_path.read_bytes()).hexdigest(),
                         "metal_optimization": optimization,
                         "metal_math_mode": ("relaxed-fastfn" if "-fmetal-math-mode=relaxed" in stage_math
                                             else "ordered-reciprocal-fastfn" if "-freciprocal-math" in stage_math
                                             else "ordered-no-contract" if stage_math else "default"),
                         "metal_math_flags": stage_math}
        if table_bytes:
            kernels[name]["literal_tables"] = tables_path.name
            kernels[name]["literal_tables_sha256"] = hashlib.sha256(table_bytes).hexdigest()
        cache.mkdir(parents=True, exist_ok=True)
        for file in output.glob(name + ".*"):
            shutil.copyfile(file, cache / file.name)
        (cache / "binding.json").write_text(json.dumps(kernels[name], indent=2) + "\n")
        save_manifest()
    return {"dir": str(output), "key": key, "version": "4.1.1", "kernels": len(kernels),
            "complete": len(kernels) == len(stages), "source_hash": source_digest}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--builddir", default="build")
    parser.add_argument("--res", type=int, choices=TIERS, default=1080)
    parser.add_argument("--profile", choices=("exact", "exact-fast", "lossy"), default="exact")
    for flag, default in (("depth-inverted", 0), ("low-res-mv", 1), ("jittered-mv", 0),
                          ("auto-exposure", 0), ("optimize", 0), ("ultra-performance", 0),
                          ("external-post-cnb", 0), ("strict-math", 0)):
        parser.add_argument("--" + flag, type=int, choices=(0, 1), default=default)
    parser.add_argument("--colorspace", type=int, choices=range(4), default=0)
    parser.add_argument("--only", nargs="+")
    try:
        args = parser.parse_args()
        builddir = Path(args.builddir).resolve()
        builddir.mkdir(parents=True, exist_ok=True)
        with (builddir / ".fsr411-build.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            print(json.dumps(build(args)))
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"FSR 4.1.1 build: {exc}\n")


if __name__ == "__main__":
    main()
