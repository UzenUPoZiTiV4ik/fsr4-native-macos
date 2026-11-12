#!/usr/bin/env python3
"""Build the CrossOver release zip from a git revision and verified Lite data.

The zip holds what crossover_fsr4.py install needs (hook, live bridge, panel,
FSR 4.1.1 sources, FSR 4.1.2 Lite data, shader tools) plus INSTALL.md.
FSR 4.0.2 weights, optional GPL-derived Lite Lossy weights, Windows DLLs,
test recordings and local caches are left out. Lite Lossy is an explicit
external fetch; the portable receipt documents its required input paths.

  python3 tools/crossover/package_release.py v1.0 [--rev HEAD] [--out dist/release]
"""

import argparse
import hashlib
import io
import json
import subprocess
import tarfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXCLUDE_PREFIXES = (
    "third_party/fsr4/internal/shaders/fsr4_model_v07_i8_",  # FSR 4.0.2 weights
    "build/fsr412_lossy/",  # GPL-derived initializer stays an explicit external fetch
    "build/fsr412_upstream/",  # translated original shaders/data stay external
    "build/netal_src/", "build/netal_runtime/",  # optional source/model stay local
    "build/anr",  # proprietary models, kernel packs and recordings stay local
    "tests/netal",  # initialization harness is a source-only developer tool
    "tests/", "tools/bench/", "tools/installer/", "tools/fsr411/test_", "docs/benchmarks/",
    "docs/FSR402_", "docs/GUIDE_FSR402", "docs/HARDWARE_FSR402", "patches/",
    "build_shaders.sh", "TESTDATA.md", "VERIFICATION.md", ".gitignore",
)


def included(path: str) -> bool:
    # Keep the source-only optional initialization/seam/UI probes, unlike
    # the development test suites and their recordings.
    if path in ("tests/anr_live_test.mm", "tests/anr_ui_test.mm", "tests/anr_motion_test.mm", "tests/netal_probe.mm"):
        return True
    return not path.startswith(EXCLUDE_PREFIXES)


def optional_profile_metadata():
    # Reference paths only. Do not distribute the external initializer or
    # turn an ordinary Lite release into an implicit GPL-weight install.
    required = ["build/fsr412_lossy/initializers-lite-lossy.bin", "build/fsr412_lossy/import.json"]
    return {profile: {"required_inputs": list(required), "weights_included": False,
                      "fetch_command": "make lite-lossy-weights", "experimental": tiled,
                      "panel_selectable": not tiled}
            for profile, tiled in (("int8-lite-lossy", False), ("int8-lite-lossy-tiled", True))}


def model_payload(root=ROOT):
    """Portable build inputs only, pinned to the source catalog and receipts."""
    base=root/"build/fsr412_model"
    expected={"initializers-lite.bin":"4b9898da719dc2dca5dc45fd71dec5f80bdb9886a580fe3038af59efe637d55e",
              "initializers-lite-ultraperformance.bin":"4c57292d9963e82dec86134060eef0c2b66214cd1a31a9e9bc73739b5fe2d1d7"}
    files=[]
    for name,digest in expected.items():
        data=(base/name).read_bytes()
        if len(data)!=27392 or hashlib.sha256(data).hexdigest()!=digest:
            raise ValueError("FSR 4.1.2 initializer identity mismatch: "+name)
        files.append(("build/fsr412_model/"+name,data,0o644))
    catalog=(root/"tools/fsr412/shader_catalog.json").read_bytes()
    receipt_path=base/"lite-shaders/receipt.json"
    receipt=json.loads(receipt_path.read_bytes())
    if receipt["catalog_sha256"]!=hashlib.sha256(catalog).hexdigest():
        raise ValueError("Lite shader receipt does not match this source catalog")
    catalog_data=json.loads(catalog)
    if receipt["source_sha256"]!=catalog_data["source_sha256"] or {s["sha256"] for s in receipt["shaders"]}!={s["sha256"] for s in catalog_data["shaders"]}:
        raise ValueError("Lite shader receipt is incomplete or from another provider")
    for item in receipt["shaders"]:
        name=item["sha256"]+".metal"
        data=(base/"lite-shaders"/name).read_bytes()
        if hashlib.sha256(data).hexdigest()!=item["metal_sha256"]:
            raise ValueError("Lite shader identity mismatch: "+name)
        files.append(("build/fsr412_model/lite-shaders/"+name,data,0o644))
    files.append(("build/fsr412_model/lite-shaders/receipt.json",receipt_path.read_bytes(),0o644))
    # The local import receipt has machine-specific paths; archive metadata
    # contains only paths relative to the portable payload.
    portable={"schema":"fsr412_lite_release_v1","model":"FSR 4.1.2 INT8 Lite",
              "provider_sha256":receipt["source_sha256"],"initializers":expected,
              "shader_count":len(receipt["shaders"]),"provenance":"Static import; the Windows provider is not bundled or executed.",
              "optional_profiles":optional_profile_metadata()}
    files.append(("build/fsr412_model/import.json",(json.dumps(portable,indent=2)+"\n").encode(),0o644))
    return files


def main():
    p = argparse.ArgumentParser()
    p.add_argument("version")
    p.add_argument("--rev", default="HEAD")
    p.add_argument("--out", default=str(ROOT / "dist" / "release"))
    args = p.parse_args()

    name = f"fsr4-mac-{args.version}-crossover"
    out = Path(args.out) / f"{name}.zip"
    out.parent.mkdir(parents=True, exist_ok=True)
    tar = subprocess.check_output(["git", "archive", "--format=tar", args.rev], cwd=ROOT)
    count = 0
    model_files=model_payload()
    with tarfile.open(fileobj=io.BytesIO(tar)) as src, \
            zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as dst:
        for member in src.getmembers():
            if not member.isfile() or not included(member.name):
                continue
            info = zipfile.ZipInfo(f"{name}/{member.name}", date_time=(2026, 1, 1, 0, 0, 0))
            info.external_attr = (0o100000 | member.mode) << 16  # keep exec bits for tools/bin
            info.compress_type = zipfile.ZIP_DEFLATED
            dst.writestr(info, src.extractfile(member).read())
            count += 1
        guide = (ROOT / "tools" / "crossover" / "INSTALL.md").read_bytes()
        info = zipfile.ZipInfo(f"{name}/INSTALL.md", date_time=(2026, 1, 1, 0, 0, 0))
        info.external_attr = 0o100644 << 16
        info.compress_type = zipfile.ZIP_DEFLATED
        dst.writestr(info, guide)
        for path,data,mode in model_files:
            info=zipfile.ZipInfo(f"{name}/{path}",date_time=(2026,1,1,0,0,0))
            info.external_attr=(0o100000|mode)<<16;info.compress_type=zipfile.ZIP_DEFLATED
            dst.writestr(info,data)
    print(f"{out}: {count + 1 + len(model_files)} files, {out.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
