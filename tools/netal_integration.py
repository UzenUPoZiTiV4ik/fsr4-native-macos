#!/usr/bin/env python3
"""Hash-pinned, optional Netal SR source and runtime integration.

Does not download or execute NVIDIA DLLs, fetch weights, or patch Netal's tree.
Only the SR session and texture adapter are built; upstream game hooks and FG
are deliberately excluded. Model files remain in the user's local bundle.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import tempfile

ROOT = Path(__file__).resolve().parents[1]
SOURCE_MANIFEST = ROOT / "third_party/netal/source_manifest.json"
MODEL = "DLSS_SR_Feature1_Quality_K"
SCHEMA = "netal.sr.runtime.v2"
MODELS = {
    "model/weights.bin": (3545824, "ddaa59cd82deed53bad9d65b82b454f1439fc4875e4220b3ffafafb6cd0296a1"),
    "model/constants.bin": (524288, "305f4c6773d1cb333b577094e28a46beb1867898716d8290121862a2cb06e773"),
}
REQUIRED = set(MODELS) | {
    "metallib/hiluma_input_poison_a.metallib", "metallib/hiluma_output_exact_math_safe.metallib",
    "metallib/hiluma_input_depthinv_exact.metallib", "metallib/hiluma_output_depthinv_exact.metallib",
    "metallib/hiluma_input_depthinv.metallib", "metallib/netal_copy_exposure.metallib",
    "metallib/dlss_sr_live_texture_convert.metallib", "math/rcp-correction-rom.bin",
    "math/rsq-correction-rom.bin", "math/rcp-f16-to-f16-lut.bin", "math/ex2-f16x2-lut.bin",
    "math/sm120-texture-weights.bin", "math/sm120-mufu/lg2-input-table.bin",
    "math/sm120-mufu/lg2-output-table.bin", "math/sm120-mufu/ex2-input-table.bin",
    "math/sm120-mufu/ex2-output-table.bin", "abi/hiluma-input-reset-template.bin",
    "abi/hiluma-output-reset-template.bin", "abi/exposure-reset-template.bin",
} | {"metallib/dltss_pwin_" + stage + "_layer_exact.metallib"
     for stage in ("enc0", "enc1", "enc2", "enc3", "enc4", "dec5", "dec4", "dec3", "dec2", "dec1", "dec0")}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def regular_file(root, name):
    path = PurePosixPath(name)
    if (not name or path.is_absolute() or "\\" in name or "\0" in name
            or any(part in ("", ".", "..") for part in name.split("/"))):
        raise ValueError("unsafe relative path: " + repr(name))
    current = root
    for part in path.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError("symlink is not supported: " + str(current))
    if not current.is_file():
        raise ValueError("required file is missing: " + str(current))
    return current


def validate_source(source, manifest_path=SOURCE_MANIFEST):
    source = Path(source).expanduser().resolve(strict=True)
    manifest = json.loads(Path(manifest_path).read_text())
    if manifest.get("schema") != "fsr4_netal_source_v1":
        raise ValueError("unsupported Netal source manifest")
    files = {}
    for name, expected in manifest["files"].items():
        data = regular_file(source, name).read_bytes()
        if digest(data) != expected:
            raise ValueError("Netal source SHA-256 mismatch: " + name
                             + "; use revision " + manifest["revision"])
        files[name] = data
    return source, manifest, files


def adapt_source(name, data):
    if name == "LICENSE":
        return data
    text = data.decode("utf-8")
    # Private Objective-C class/symbol names avoid conflicts with a Netal app
    # or another injected runtime. No numerical or shader source changes.
    for old, new in (("NeuralStudioDLSSSR", "Fsr4NetalSR"),
                     ("NeuralStudioDLSSTextureAdapter", "Fsr4NetalTextureAdapter"),
                     ("NeuralStudioDLSSTextureAdapterErrorDomain", "Fsr4NetalTextureAdapterErrorDomain")):
        text = text.replace(old, new)
    if name.endswith("dlss_sr_session.mm"):
        replacements = {
            'std::getenv("NEURALSTUDIO_SR_PROFILE")': ("fsr4_netal_geometry_profile()", 1),
            'std::getenv("NEURALSTUDIO_SR_RENDER")': ("fsr4_netal_geometry_render()", 2),
        }
        for old, (new, count) in replacements.items():
            if text.count(old) != count:
                raise ValueError("Netal geometry bridge contract changed: " + old)
            text = text.replace(old, new)
        text = '#include "netal_live.h"\n' + text
    return text.encode("utf-8")


def prepare_sources(source, output=None, manifest_path=SOURCE_MANIFEST):
    source, manifest, files = validate_source(source, manifest_path)
    output = Path(output) if output is not None else ROOT / "build/netal_src"
    if output.is_symlink():
        raise ValueError("Netal build output is a symlink")
    output.parent.mkdir(parents=True, exist_ok=True)
    expected = {Path(name).name: adapt_source(name, data) for name, data in files.items()}
    if len(expected) != len(files):
        raise ValueError("duplicate source basenames")
    receipt = {"schema": "fsr4_netal_build_v1", "revision": manifest["revision"],
               "original_files": manifest["files"],
               "generated_files": {name: digest(data) for name, data in expected.items()},
               "changes": ["private Objective-C symbols", "thread-local session geometry"],
               "models_included": False, "upstream_hooks_included": False}
    if output.exists():
        for name, data in expected.items():
            if regular_file(output, name).read_bytes() != data:
                raise ValueError("existing Netal build source differs: " + str(output / name)
                                 + "; choose a new output directory")
        saved = json.loads(regular_file(output, "source-receipt.json").read_text())
        if saved != receipt:
            raise ValueError("Netal build receipt mismatch")
        return output
    with tempfile.TemporaryDirectory(prefix=".netal-", dir=output.parent) as temporary:
        stage = Path(temporary) / "src"
        stage.mkdir()
        for name, data in expected.items():
            (stage / name).write_bytes(data)
        (stage / "source-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
        stage.rename(output)
    return output


def validate_runtime(bundle):
    root = Path(bundle).expanduser().resolve(strict=True)
    manifest = json.loads(regular_file(root, "manifest.json").read_text())
    if not isinstance(manifest, dict) or manifest.get("schema") != SCHEMA or manifest.get("model") != MODEL:
        raise ValueError("a complete netal.sr.runtime.v2 DLSS Preset K bundle is required; source assets are not runnable")
    rows = manifest.get("files")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Netal runtime has no asset inventory")
    ledger = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("invalid Netal asset row")
        name, size, expected = row.get("path"), row.get("bytes"), row.get("sha256")
        allowed = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_./-"
        if (not isinstance(name, str) or name == "manifest.json" or name in ledger
                or any(char not in allowed for char in name)
                or "expected" in name.lower() or "capture" in name.lower()
                or type(size) is not int or size < 0 or not isinstance(expected, str)
                or len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected)):
            raise ValueError("invalid Netal asset metadata")
        path = regular_file(root, name)
        if path.stat().st_size != size:
            raise ValueError("Netal runtime asset size mismatch: " + name)
        if digest(path.read_bytes()) != expected:
            raise ValueError("Netal runtime asset SHA-256 mismatch: " + name)
        ledger[name] = row
    material = (SCHEMA + "\n").encode()
    for name in sorted(ledger):
        row = ledger[name]
        material += (name + "\0" + str(row["bytes"]) + "\0" + row["sha256"] + "\n").encode()
    if digest(material) != manifest.get("asset_root_sha256"):
        raise ValueError("Netal runtime inventory root mismatch")
    missing = REQUIRED - ledger.keys()
    if missing:
        raise ValueError("Netal runtime is incomplete: " + ", ".join(sorted(missing)))
    for name, (size, expected) in MODELS.items():
        if ledger[name]["bytes"] != size or ledger[name]["sha256"] != expected:
            raise ValueError("unsupported Netal model bytes: " + name)
    for child in root.rglob("*"):
        if child.is_symlink() or (not child.is_file() and not child.is_dir()):
            raise ValueError("nonregular asset in Netal runtime")
        if child.is_file() and child.relative_to(root).as_posix() not in (set(ledger) | {"manifest.json"}):
            raise ValueError("unlisted Netal runtime asset: " + str(child))
    return root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--source", type=Path, required=True)
    prepare.add_argument("--output", type=Path)
    validate = sub.add_parser("validate")
    validate.add_argument("--runtime", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = prepare_sources(args.source, args.output) if args.cmd == "prepare" else validate_runtime(args.runtime)
        print(result)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, "Netal: " + str(error) + "\n")


if __name__ == "__main__":
    main()
