#!/usr/bin/env python3
"""Prepare an explicit ANR macOS kit for the independent Metal 4 runtime.

No game installation, binary patch, network access or model download. All
weights and compiled shaders stay in the local ignored build directory.
"""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import struct
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("anr_nrx_prepare", ROOT / "tools/diagnostics/anr_nrx.py")
nrx = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(nrx)
CHECKPOINT = "e09deb7012c96c172e49e72eb0b747b4611a8b6e3f14e5f5563189c580772ac1"
KERNEL_PACK = "aeb6f568a241a63daf804a213de9c17dad2bdd7f7e79779f3002a9a3e1493767"
PROGRAM_PACKS = {
    "1080p": "87340ac7131f5b19e5433f6bcc46df63b44d536ea6c4c763cfb68039b4df39e8",
    "1440p": "b280b9b380f456b65ea05b1ce7e1ffe769061df8aad1b2c340f37673a10fd7e8",
    "2160p": "bf0b64db424bb912f24d46944d005be650086585217eb66bddf638840c03704e",
}
PACKED_PROGRAMS = {
    "1080p": "f7a01ee7111b04af0dda5b3a359b4437c5e8e3c1baada9de29510ca7d6b8ac46",
    "1440p": "9d8d7ef138234a6e7b3aa98a2dbe3d04d91b619299b55eef6b3e1622ea4b16e7",
    "2160p": "34b1e07f06c4a2f83d25b2f352f1e6f81bb3ab8a2a5e8ca99408f273718b58d2",
}
CONVERTER_HASH = "1af7a62b73330e56e7e47ad7cf3582b3b79b5cdee1c8fef9ab1387fb15b71d2e"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read_pack(path, expected):
    if path.is_symlink() or not path.is_file() or path.stat().st_size > nrx.MAX_BYTES:
        raise ValueError("missing/invalid pack: " + str(path))
    data = path.read_bytes()
    if digest(data) != expected:
        raise ValueError("unsupported ANR FG-V.B14 pack identity: " + str(path))
    kind, files = nrx.decode(data)
    if kind != 2:
        raise ValueError("ANR package must be an NRX directory")
    return files


def uniform_bytes(reflection, values):
    fields = [p for p in reflection["parameters"] if p["binding"]["kind"] == "uniform"]
    size = max((p["binding"]["offset"] + p["binding"]["size"] for p in fields), default=0)
    if size > 4096:
        raise ValueError("oversized uniform block")
    result = bytearray(size)
    for field in fields:
        name, binding, scalar = field["name"], field["binding"], field["type"].get("scalarType")
        if binding["size"] != 4 or scalar not in ("uint32", "int32", "float32"):
            raise ValueError("unsupported ANR uniform type: " + name)
        if name not in values:
            continue
        value = values[name]
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError("invalid uniform: " + name)
        if scalar != "float32":
            lo, hi = (0, 1 << 32) if scalar == "uint32" else (-(1 << 31), 1 << 31)
            if int(value) != value or not lo <= value < hi:
                raise ValueError("invalid integer uniform: " + name)
            value = int(value)
        struct.pack_into({"uint32": "<I", "int32": "<i", "float32": "<f"}[scalar], result, binding["offset"], value)
    # The graph reuses common uniform dictionaries across kernels. Unreflected
    # entries are unused by this shader; do not invent offsets for them.
    return bytes(result)


def prepare(kit, headers, output, size="1080p", backend="native"):
    kit, headers = Path(kit).expanduser().resolve(strict=True), Path(headers).expanduser().resolve(strict=True)
    output = Path(output).expanduser().absolute()
    if output.exists() or output.is_symlink():
        raise ValueError("output already exists; choose a new local directory")
    if backend not in ("native", "packed"):
        raise ValueError("unknown ANR backend")
    prefix = "metal-packed-" if backend == "packed" else "metal-"
    program_files = read_pack(kit / ("programs/" + prefix + size + "/pack.nrx"),
                              PACKED_PROGRAMS[size] if backend == "packed" else PROGRAM_PACKS[size])
    kernels = read_pack(kit / "kernels-metal/pack.nrx", KERNEL_PACK)
    program = json.loads(program_files["program.json"])
    if program["version"] != 2 or program["checkpoint_id"] != CHECKPOINT or program["path"] != ("packed" if backend == "packed" else "metal"):
        raise ValueError("unsupported ANR graph/checkpoint")
    converter = kit / "bin/libmetalirconverter.dylib"
    if converter.is_symlink() or digest(converter.read_bytes()) != CONVERTER_HASH:
        raise ValueError("unsupported Apple converter identity")
    if not (headers / "metal_irconverter/metal_irconverter.h").is_file():
        raise ValueError("supply the existing Apple converter include directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".anr-prepare-", dir=output.parent) as temporary:
        temp = Path(temporary)
        build = temp / "runtime"
        build.mkdir()
        converter_tool = temp / "convert_dxil"
        subprocess.run(["xcrun", "--sdk", "macosx", "clang++", "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
                        "-fobjc-arc", "-mmacosx-version-min=26.0", "-I", str(headers),
                        str(ROOT / "tools/anr/convert_dxil.mm"), "-framework", "Metal", "-framework", "Foundation",
                        "-o", str(converter_tool)], check=True, timeout=60)
        (build / "weights.bin").write_bytes(program_files["weights.bin"])
        compiled = {}
        all_kernels = sorted({stage["kernel"] for stages in program["stages"].values() for stage in stages})
        for index, key in enumerate(all_kernels):
            module, entry = key.split(":")
            prefix = module.removesuffix(".slang") + "/" + module + "." + entry
            reflection = json.loads(kernels[prefix + ".json"])
            directory = build / "kernels" / ("k" + str(index))
            native = reflection.get("nativeMetal")
            if native:
                directory.mkdir(parents=True)
                (directory / "kernel.metallib").write_bytes(kernels[prefix + ".metallib"])
                bindings = native["bindings"]
                entry_name = native["entryPoint"]
                route = "native"
            else:
                source = temp / ("kernel-" + str(index))
                source.mkdir()
                (source / "shader.dxil").write_bytes(kernels[prefix + ".dxil"])
                (source / "reflection.json").write_text(json.dumps(reflection))
                subprocess.run([str(converter_tool), str(source / "shader.dxil"), str(source / "reflection.json"),
                                str(converter), str(directory)], check=True, timeout=60)
                metadata = json.loads((directory / "bindings.json").read_text())
                bindings = metadata["bindings"]
                entry_name = metadata["entry"]
                route = "converted"
            compiled[key] = {"path": directory.relative_to(build).as_posix() + "/kernel.metallib",
                             "entry": entry_name, "route": route, "bindings": bindings,
                             "metal_binding_model": "root-buffer-2-to-24-byte-resource-table",
                             "resource_count": sum(not x.get("sampler", False) for x in bindings),
                             "sampler_count": sum(x.get("sampler", False) for x in bindings)}
        stages_out = {}
        uniform_dir = build / "uniforms"
        uniform_dir.mkdir()
        for stage_name, stages in program["stages"].items():
            records = []
            for stage in stages:
                if stage["op"] != "dispatch":
                    raise ValueError("unsupported graph operation")
                module, entry = stage["kernel"].split(":")
                prefix = module.removesuffix(".slang") + "/" + module + "." + entry
                reflection = json.loads(kernels[prefix + ".json"])
                constants = uniform_bytes(reflection, stage["uniforms"])
                filename = stage["label"] + ".bin"
                (uniform_dir / filename).write_bytes(constants)
                records.append({"kernel": stage["kernel"], "groups": stage["groups"], "threads": stage["threads"],
                                "buffers": stage["buffers"], "images": stage.get("images", {}),
                                "optional": stage.get("optional", []), "label": stage["label"],
                                "uniforms": "uniforms/" + filename,
                                "unused_uniform_names": sorted(set(stage["uniforms"]) - {
                                    p["name"] for p in reflection["parameters"] if p["binding"]["kind"] == "uniform"})})
            stages_out[stage_name] = records
        manifest = {"schema": "fsr4_anr_runtime_v1", "release": "FG-V.B14", "checkpoint_id": CHECKPOINT, "backend": backend,
                    "width": program["width"], "height": program["height"], "family": program["family"],
                    "source_program_sha256": digest(program_files["program.json"]), "weights_sha256": digest(program_files["weights.bin"]),
                    "source_kernel_pack_sha256": KERNEL_PACK, "converter_sha256": CONVERTER_HASH,
                    "buffers": program["buffers"], "inputs": program["inputs"], "output": program["output"],
                    "formats": program["formats"], "images": program["images"], "swaps": program["swaps"],
                    "kernels": compiled, "stages": stages_out, "runtime_executed": False,
                    "game_integration_verified": False}
        manifest["files"] = {p.relative_to(build).as_posix(): {"bytes": p.stat().st_size, "sha256": digest(p.read_bytes())}
                             for p in sorted(build.rglob("*")) if p.is_file()}
        (build / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        build.rename(output)
    return {"output": str(output), "kernels": len(compiled), "native": sum(x["route"] == "native" for x in compiled.values()),
            "converted": sum(x["route"] == "converted" for x in compiled.values()), "inference_executed": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kit", type=Path, required=True)
    parser.add_argument("--converter-headers", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", choices=tuple(PROGRAM_PACKS), default="1080p")
    parser.add_argument("--backend", choices=("native", "packed"), default="native",
                        help="native Metal matrix kernels or alternative packed FP16 (both require quality verification)")
    args = parser.parse_args()
    try:
        print(json.dumps(prepare(args.kit, args.converter_headers, args.output, args.size, args.backend), indent=2))
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as error:
        parser.exit(1, "ANR prepare: " + str(error) + "\n")


if __name__ == "__main__":
    main()
