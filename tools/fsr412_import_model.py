#!/usr/bin/env python3
"""Import the user supplied AMD FSR 4.1.2 driver model as data.

The driver DLL is never loaded.  This command validates the exact file hashes,
copies the two 27,392-byte Lite initializer arenas into the ignored build
directory, and records the already decompressed SPIR-V/MSL research inventory.
The original driver remains outside the repository; generated shader sources
are treated as local build inputs.
"""
from __future__ import annotations

import argparse, concurrent.futures, hashlib, json, subprocess, zlib
from pathlib import Path

DX_DLL_SHA = "c33a287f3048bf02311cae430b6a977500b94ca7b315cd7bff751e6b03e483e3"
VK_DLL_SHA = "80e761f7c8c0ec8ead7b321df01911baaf917defae51388d29eaecb1d6bc58bd"
LITE_COMMON_OFFSET = 0x34CE3F0
LITE_UP_OFFSET = 0x34E9020
DX_LITE_COMMON_OFFSET = 0x3C12390
DX_LITE_UP_OFFSET = 0x3C2CFC0
LITE_INIT_SIZE = 27392
LITE_COMMON_SHA = "4b9898da719dc2dca5dc45fd71dec5f80bdb9886a580fe3038af59efe637d55e"
LITE_UP_SHA = "4c57292d9963e82dec86134060eef0c2b66214cd1a31a9e9bc73739b5fe2d1d7"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def validate_shader(data: bytes, row: dict) -> bytes:
    """Decompress exactly one catalog-pinned stream and validate its bytes."""
    start=row["offset"]; size=row["compressed_size"]
    if start < 0 or size <= 0 or start+size > len(data):
        raise ValueError("shader compressed range is outside the pinned DLL")
    decoder=zlib.decompressobj()
    raw=decoder.decompress(data[start:start+size],row["size"]+1)
    if not decoder.eof or decoder.unused_data or decoder.unconsumed_tail:
        raise ValueError("shader zlib boundary does not match the catalog")
    if len(raw)!=row["size"] or sha(raw)!=row["sha256"] or raw[:4]!=b"\x03\x02#\x07":
        raise ValueError("shader decompressed identity mismatch")
    return raw


def import_shaders(data: bytes, root: Path, out: Path) -> dict:
    """Reconstruct the portable shader inputs from operator supplied bytes."""
    catalog_path=root/"tools/fsr412/shader_catalog.json"
    catalog=json.loads(catalog_path.read_text())
    if sha(data)!=catalog["source_sha256"]:
        raise ValueError("shader import requires the pinned amdvkffx64.dll")
    rows={x["sha256"]:x for x in catalog["shaders"]}
    # Validate every compressed payload before creating any shader files.
    payloads={digest:validate_shader(data,row) for digest,row in rows.items()}
    dst=out/"lite-shaders"; dst.mkdir(exist_ok=True)
    tool=root/"tools/bin/spirv-cross"
    tool_sha=sha(tool.read_bytes())
    def one(item):
        digest,raw=item; spv=dst/(digest+".spv"); metal=dst/(digest+".metal")
        if not spv.exists() or sha(spv.read_bytes())!=digest: spv.write_bytes(raw)
        p=subprocess.run([str(tool),str(spv),"--msl","--msl-version","30000"],capture_output=True)
        if p.returncode: raise ValueError(p.stderr.decode(errors="replace")[-4000:])
        metal.write_bytes(p.stdout)
        return {"sha256":digest,"spv":spv.name,"metal":metal.name,
                "metal_sha256":sha(p.stdout),"bytes":len(raw)}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        receipts=list(pool.map(one,payloads.items()))
    receipt={"source_sha256":sha(data),"catalog_sha256":sha(catalog_path.read_bytes()),
             "spirv_cross_sha256":tool_sha,"msl_version":30000,
             "shaders":sorted(receipts,key=lambda x:x["sha256"])}
    (dst/"receipt.json").write_text(json.dumps(receipt,indent=2)+"\n")
    return {"count":len(receipts),"path":str(dst.resolve()),
            "receipt_sha256":sha((dst/"receipt.json").read_bytes())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dll", required=True, type=Path,
                    help="amdxcffx64.dll or amdvkffx64.dll supplied by the operator")
    ap.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--with-shaders", action="store_true",
                    help="decompress pinned Vulkan Lite streams and translate them with SPIRV-Cross")
    args = ap.parse_args()
    root = Path(args.root).resolve()
    dll = args.dll.expanduser().resolve()
    data = dll.read_bytes()
    digest = sha(data)
    expected = VK_DLL_SHA if dll.name.lower() == "amdvkffx64.dll" else DX_DLL_SHA
    if digest != expected:
        raise SystemExit(f"unsupported FSR 4.1.2 provider hash: {digest} (expected {expected})")
    out = Path(args.outdir).resolve() if args.outdir else root / "build" / "fsr412_model"
    out.mkdir(parents=True, exist_ok=True)
    common_offset, ultra_offset = (DX_LITE_COMMON_OFFSET, DX_LITE_UP_OFFSET) if dll.name.lower() == "amdxcffx64.dll" else (LITE_COMMON_OFFSET, LITE_UP_OFFSET)
    common = data[common_offset:common_offset + LITE_INIT_SIZE]
    ultra = data[ultra_offset:ultra_offset + LITE_INIT_SIZE]
    if len(common) != LITE_INIT_SIZE or sha(common) != LITE_COMMON_SHA:
        raise SystemExit("FSR 4.1.2 Lite common initializer identity mismatch")
    if len(ultra) != LITE_INIT_SIZE or sha(ultra) != LITE_UP_SHA:
        raise SystemExit("FSR 4.1.2 Lite Ultra Performance initializer identity mismatch")
    (out / "initializers-lite.bin").write_bytes(common)
    (out / "initializers-lite-ultraperformance.bin").write_bytes(ultra)
    receipt = {
        "schema": "fsr412_lite_import_v1", "provider": dll.name,
        "provider_sha256": digest, "provider_bytes": len(data),
        "model": "FSR 4.1.2 INT8 Lite", "model_passes": [1, 2, 3],
        "initializer": {"path": str((out / "initializers-lite.bin").resolve()),
                        "dll_offset": hex(common_offset), "bytes": LITE_INIT_SIZE,
                        "sha256": LITE_COMMON_SHA},
        "ultraperformance_initializer": {
            "path": str((out / "initializers-lite-ultraperformance.bin").resolve()),
            "dll_offset": hex(ultra_offset), "bytes": LITE_INIT_SIZE,
            "sha256": LITE_UP_SHA},
        "provenance": "operator supplied driver DLL; static bytes only; no DLL execution",
    }
    if args.with_shaders:
        if expected!=VK_DLL_SHA:
            raise SystemExit("--with-shaders needs amdvkffx64.dll; DX12 import supports initializers only")
        receipt["shader_inventory"]=import_shaders(data,root,out)
    tmp = out / "import.json.tmp"
    tmp.write_text(json.dumps(receipt, indent=2) + "\n")
    tmp.replace(out / "import.json")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
