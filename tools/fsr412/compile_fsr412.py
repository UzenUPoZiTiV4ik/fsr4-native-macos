#!/usr/bin/env python3
"""Compile the local FSR 4.1.2 INT8 Lite shader inventory to Metal.

The inventory is generated from the supplied Vulkan driver bytes by the REA
research pass.  It contains only content addressed SPIR-V translations; this
tool never loads the driver. Native MPP operators use the same manifest and
scratch ABI. Optimized linear variants share a float32 input-color transform
between PRE and POST, without changing the neural or temporal graph.
--pixelization-fix widens POST's current-frame filter where history is rejected.
"""
from __future__ import annotations

import argparse, fcntl, hashlib, json, os, re, shutil, subprocess, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fsr412.optimization import (REVISION, POST_TILE_SIDE, input_color_cache, color_cache_source, pixelization_fix,
                                 ct2d_float_dot, post_color_tile)

ROOT = Path(__file__).resolve().parents[2]
CATALOG = ROOT / "tools" / "fsr412" / "shader_catalog.json"


def run(cmd, cwd=None):
    p = subprocess.run([str(x) for x in cmd], cwd=cwd, capture_output=True)
    if p.returncode:
        raise RuntimeError((p.stdout + p.stderr).decode(errors="replace")[-12000:])
    return p.stdout


def pick(rows, stage, tier, opts):
    c = [x for x in rows if x.get("stage") == stage and x.get("tier") == tier]
    if stage == "pre":
        # Output-resolution motion does not dilate depth, so its inversion
        # flag is unused. Nonlinear color does not use exposure. The REA
        # catalog records these absent operations as null, not another value.
        c = [x for x in c if all(x.get(name) is None or x[name] == opts[flag]
             for name,flag in (("depth_inverted","di"),("low_res_mv","lm"),("jittered_mv","jm"),
                               ("colorspace","cs"),("auto_exposure","ae")))]
    elif stage == "post":
        c = [x for x in c if x.get("colorspace") == opts["cs"] and
             (x.get("auto_exposure") is None or x.get("auto_exposure") == opts["ae"])]
        c = [x for x in c if x.get("preset") == ("ultraperformance" if opts["up"] else "normal")]
    elif stage in ("pass1", "pass2", "pass3"):
        c = [x for x in c if x.get("preset") == ("ultraperformance" if opts["up"] else "normal")]
    if not c:
        raise RuntimeError(f"no FSR4.1.2 inventory shader for {stage}, tier {tier}, {opts}")
    # Duplicate entries are distinct compiler containers but identical stage
    # contracts. Pick the first content identity deterministically.
    return sorted(c, key=lambda x: (x["sha256"], x["offset"]))[0]


def source_for(row):
    p = ROOT / "build" / "fsr412_model" / "lite-shaders" / (row["sha256"] + ".metal")
    if not p.exists():
        p = ROOT / "build" / "fsr412_research" / "vk" / "lite" / (row["sha256"] + ".metal")
    if not p.exists():
        raise RuntimeError(f"missing decompressed source: {p}")
    return p


def support_bundle(root, builddir, colorspace, auto_exposure):
    # AE/RCAS have no Lite neural coefficients. Use the matching color space
    # and the split exposure ABI; a clean build can compile these three alone.
    names = ("ae", "ae_finalize", "rcas")
    for path in sorted(builddir.glob("fsr411_1080_*/bindings.json")):
        try:
            manifest = json.loads(path.read_text())
            if manifest.get("colorspace") != colorspace or manifest.get("auto_exposure") != auto_exposure or manifest.get("auto_exposure_dispatches") != 2:
                continue
            if all(name in manifest["kernels"] and (path.parent / manifest["kernels"][name]["library"]).is_file() for name in names):
                return path.parent, manifest
        except (OSError, ValueError, KeyError):
            continue
    command = [sys.executable, root / "tools/compile_fsr411.py", "--root", root,
               "--builddir", builddir, "--res", "1080", "--colorspace", str(colorspace),
               "--auto-exposure",str(auto_exposure),
               "--optimize", "1", "--profile", "exact-fast", "--only", *names]
    result = json.loads(run(command))
    path = Path(result["dir"]) / "bindings.json"
    return path.parent, json.loads(path.read_text())


def external_post_cnb(text: str, tier: int) -> str:
    """Replace only the POST CNB with a native-produced INT8 cache read.

    The four packed variables are the ABI between the CNB residual and the
    final CT2D/filter.  A compact native tensor at scratch byte zero stores
    precisely those sixteen bytes per half-resolution pixel.  The selected
    shader's CT2D, temporal resolve and output transfer remain in place.
    """
    body = text.index("\n{", text.index("kernel void"))
    # This coordinate is the original POST thread's half-resolution pixel.
    start = re.search(r"^        int4 _\d+ = int4\(int\((_[0-9]+)\.x\), int\(\1\.y\), 0, 0\);", text[body:], re.M)
    if not start:
        raise ValueError("POST CNB coordinate boundary not found")
    begin=body+start.start(); coordinate=start[1]
    packed=list(re.finditer(r"^        uint (_\d+) = \(\(\(uint\(clamp\(_\d+\.x, -128, 127\)\).*;\n",text[begin:],re.M))
    if len(packed)<4:
        raise ValueError("POST CNB packed output boundary not found")
    tail=packed[-4:]
    end=begin+tail[-1].end()
    stride=(tier*8//9)*16
    replacement=(f"        // Native POST-CNB cache, compact NHWC INT8 at scratch byte zero.\n"
                 f"        uint fsr412CacheWord = ({coordinate}.y * {stride}u + {coordinate}.x * 16u) >> 2u;\n")
    replacement += "".join(f"        uint {m[1]} = ScratchBuffer._m0[fsr412CacheWord + {i}u];\n" for i,m in enumerate(tail))
    return text[:begin]+replacement+text[end:]


def _call_span(text: str, open_paren: int) -> tuple[int, str]:
    """Return the argument list of the call whose '(' is at open_paren."""
    depth = 0
    for index in range(open_paren, len(text)):
        if text[index] == "(":
            depth += 1
        elif text[index] == ")":
            depth -= 1
            if depth == 0:
                return index + 1, text[open_paren + 1:index]
    raise ValueError("unbalanced SPIR-V pow call")


def _split_arguments(arguments: str) -> list[str]:
    parts, depth, start = [], 0, 0
    for index, character in enumerate(arguments):
        if character in "([":
            depth += 1
        elif character in ")]":
            depth -= 1
        elif character == "," and depth == 0:
            parts.append(arguments[start:index]); start = index + 1
    parts.append(arguments[start:])
    return parts


def signed_square(text: str) -> str:
    """Replace every `powr(x, 2.0)` with a multiply.

    SPIR-V Pow with a constant exponent 2 becomes Metal `powr`, whose base
    must be nonnegative, while AMD's DXIL lowers the same source to `fmul`.
    The 4.1.2 PRE colour-distance feature squares signed chroma differences,
    so a negative base returns NaN on Metal and corrupts the temporal colour
    feature the network uses to separate current colour from stale history.
    The 4.1.1 path fixes the same expression in its HLSL sources; this is the
    equivalent correction for the translated SPIR-V.
    """
    result, cursor, replacements = [], 0, 0
    while True:
        found = text.find("powr(", cursor)
        if found < 0:
            result.append(text[cursor:]); break
        end, arguments = _call_span(text, found + len("powr"))
        parts = _split_arguments(arguments)
        if len(parts) != 2:
            raise ValueError("unrecognized SPIR-V pow call")
        base, exponent = parts[0].strip(), parts[1].strip()
        if exponent not in ("2.0", "2"):
            raise ValueError(f"unexpected SPIR-V pow exponent {exponent!r}")
        result.append(text[cursor:found])
        result.append(f"(({base}) * ({base}))")
        cursor = end; replacements += 1
    return "".join(result)


def compile_metal(src: Path, out: Path, entry: str, external: bool = False, tier: int = 1080, prepared=False, stage="", pixel_fix=False):
    text = src.read_text()
    if external:
        text=external_post_cnb(text,tier)
    text = re.sub(r"kernel\s+void\s+(\w+)(\s*\()", r"kernel void fsr412_\1\2", text, count=1)
    text = text.replace(".xyzz, uint2", ".xyzx, uint2")
    text = signed_square(text)
    if "powr(" in text:
        raise RuntimeError(f"unhandled SPIR-V pow in {src.name}")
    if prepared and stage in ("pre", "post"):
        text, formula = input_color_cache(text, 4 if stage == "pre" else 9)
        if stage == "pre":
            (out.parent / "prepare_color.metal").write_text(color_cache_source(text, formula))
    if pixel_fix and stage == "post":
        text = pixelization_fix(text)
    if prepared and stage == "post":
        text = ct2d_float_dot(text)
    generated = out.with_suffix(".metal")
    generated.write_text(text)
    air = out.with_suffix(".air")
    lib = out.with_suffix(".metallib")
    run(["xcrun", "metal", "-std=macos-metal2.3", "-O3", "-c", generated, "-o", air])
    run(["xcrun", "metallib", air, "-o", lib])
    return generated, lib


def reflect_bindings(src: str, entry: str):
    m = re.search(r"kernel\s+void\s+" + re.escape(entry) + r"\s*\((.*?)\)\s*\{", src, re.S)
    if not m:
        raise RuntimeError(f"missing generated kernel {entry}")
    args = m.group(1); bufs=[]; tex=[]; sam=[]
    for z in re.finditer(r"\b([A-Za-z_]\w*)\s*\[\[(buffer|texture|sampler)\((\d+)\)\]\]", args):
        item=[int(z[3]), z[1]]
        if z[2]=="buffer": bufs.append(item)
        elif z[2]=="texture": tex.append(item)
        else: sam.append(item)
    m2 = re.search(r"kernel\s+void\s+" + re.escape(entry) + r".*?thread_position_in_threadgroup", src, re.S)
    # SPIR-V inventory carries the authoritative workgroup dimensions.
    return bufs, tex, sam


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",default=str(ROOT)); ap.add_argument("--builddir",default="build")
    ap.add_argument("--outdir",default=None); ap.add_argument("--res",type=int,choices=(1080,2160,4320),default=1080)
    ap.add_argument("--depth-inverted",type=int,choices=(0,1),default=0); ap.add_argument("--low-res-mv",type=int,choices=(0,1),default=1)
    ap.add_argument("--jittered-mv",type=int,choices=(0,1),default=0); ap.add_argument("--colorspace",type=int,choices=(0,1,2,3),default=0)
    ap.add_argument("--auto-exposure",type=int,choices=(0,1),default=0); ap.add_argument("--optimize",type=int,choices=(0,1),default=0)
    ap.add_argument("--ultra-performance",type=int,choices=(0,1),default=0); ap.add_argument("--external-post-cnb",type=int,choices=(0,1),default=0)
    ap.add_argument("--pixelization-fix","--hair-fix",dest="pixelization_fix",type=int,choices=(0,1),default=0)
    a=ap.parse_args(); root=Path(a.root).resolve(); bdir=Path(a.builddir).resolve()
    prepared=bool(a.optimize and a.external_post_cnb and a.colorspace==0)
    key=f"fsr412_{a.res}_di{a.depth_inverted}_lm{a.low_res_mv}_jm{a.jittered_mv}_cs{a.colorspace}_ae{a.auto_exposure}_o{a.optimize}_up{a.ultra_performance}_pc{a.external_post_cnb}"
    if prepared: key+=f"_v{REVISION}"
    if a.pixelization_fix: key+="_px2"
    out=Path(a.outdir).resolve() if a.outdir else bdir/key; out.mkdir(parents=True,exist_ok=True)
    lock=(out/".compile.lock").open("a")
    fcntl.flock(lock,fcntl.LOCK_EX)
    rows=json.loads(CATALOG.read_text())["shaders"]
    opts={"di":a.depth_inverted,"lm":a.low_res_mv,"jm":a.jittered_mv,"cs":a.colorspace,"ae":a.auto_exposure,"up":a.ultra_performance}
    order=["pre","pad0","pass1","pad1","pad1b","pass2","pad2","pass3","pad3","post"]
    selected={stage:pick(rows,stage,a.res,opts) for stage in order}
    support_dir,support_manifest=support_bundle(root,bdir,a.colorspace,a.auto_exposure)
    digest=hashlib.sha256(json.dumps(vars(a),sort_keys=True).encode())
    for path in [Path(__file__),Path(__file__).with_name("optimization.py"),CATALOG,
                 *[source_for(row) for row in selected.values()],
                 *[support_dir/support_manifest["kernels"][s]["library"] for s in ("ae","ae_finalize","rcas")]]:
        digest.update(path.read_bytes())
    digest.update(run(["xcrun","metal","--version"]))
    fingerprint=digest.hexdigest()
    manifest_path=out/"bindings.json"
    if manifest_path.exists():
        previous=json.loads(manifest_path.read_text())
        if previous.get("source_hash")==fingerprint and all(
                (out/k["library"]).is_file() and (not k.get("literal_tables") or (out/k["literal_tables"]).is_file())
                for k in previous.get("kernels",{}).values()):
            print(json.dumps({"dir":str(out),"key":key,"version":"4.1.2","model":"int8-lite","cached":True}))
            return
    kernels={}; tensor_slot={"pad0":0,"pass1":0,"pad1":2,"pad1b":3,"pass2":3,"pad2":4,"pass3":4,"pad3":5}
    for stage in order:
        row=selected[stage]
        src=source_for(row); base=out/stage
        generated,lib=compile_metal(src,base,row["entry"],stage=="post" and bool(a.external_post_cnb),a.res,prepared,stage,bool(a.pixelization_fix))
        entry="fsr412_"+row["entry"]
        bufs,tex,sam=reflect_bindings(generated.read_text(),entry)
        info={"stage":stage,"entry":entry,"library":lib.name,"wgsize":row.get("wgsize",[256,1,1]),"buffers":bufs,"textures":tex,"samplers":sam,"source_sha256":row["sha256"]}
        if stage in tensor_slot:
            info["tensor_slot"]=tensor_slot[stage]
            info["tile"]=[32,4] if stage=="pass1" else [64,1]
        elif stage=="pre":info["tile"]=[16,16]
        elif stage=="post":info["tile"]=[16,16]; info["post_pixels_per_group"]=32
        kernels[stage]=info
        if stage=="post" and prepared:
            # Same POST with its colour taps read from a threadgroup tile;
            # the host selects it when the tile covers every tap.
            tile=out/"post_tile"
            tile.with_suffix(".metal").write_text(post_color_tile(generated.read_text()))
            run(["xcrun","metal","-std=macos-metal2.3","-O3","-c",tile.with_suffix(".metal"),"-o",tile.with_suffix(".air")])
            run(["xcrun","metallib",tile.with_suffix(".air"),"-o",tile.with_suffix(".metallib")])
            kernels["post_tile"]=dict(info,stage="post_tile",library="post_tile.metallib",
                                      tile_side=POST_TILE_SIDE)
    if prepared:
        base=out/"prepare_color"
        run(["xcrun","metal","-std=macos-metal2.3","-O3","-c",base.with_suffix(".metal"),"-o",base.with_suffix(".air")])
        run(["xcrun","metallib",base.with_suffix(".air"),"-o",base.with_suffix(".metallib")])
        kernels["prepare_color"]={"stage":"prepare_color","entry":"fsr412_prepare_color","library":"prepare_color.metallib",
            "wgsize":[16,16,1],"buffers":[[0,"MLSR_Optimized_Constants"]],"textures":[[0,"r_input_color"],[1,"r_auto_exposure_texture"],[2,"rw_prepared_color"]],"samplers":[]}
    for name in ("ae","ae_finalize","rcas"):
        entry=support_manifest["kernels"][name]
        for field in ("library","literal_tables"):
            if entry.get(field): shutil.copyfile(support_dir/entry[field],out/entry[field])
        kernels[name]=entry
    scratch={1080:25027392,2160:99886528,4320:398707392}[a.res]
    init=root/"build/fsr412_model"/("initializers-lite-ultraperformance.bin" if a.ultra_performance else "initializers-lite.bin")
    manifest={"version":"4.1.2","model":"int8-lite","key":key,"source_hash":fingerprint,"optimization_revision":REVISION,
        "prepared_color":prepared,"pixelization_fix":bool(a.pixelization_fix),"order":order,"kernels":kernels,"scratch_bytes":scratch,"initializer":str(init.resolve()),"external_post_cnb":bool(a.external_post_cnb),"post_pixels_per_group":32,"geometry":{"pre":{"kind":"pre","tile":[16,16],"threads":[256,1,1]},"post":{"kind":"post","tile":[16,16],"threads":[256,1,1]}}}
    temporary=out/f".bindings.{os.getpid()}.json"
    temporary.write_text(json.dumps(manifest,indent=2)+"\n")
    temporary.replace(manifest_path)
    print(json.dumps({"dir":str(out),"key":key,"version":"4.1.2","model":"int8-lite","stages":len(kernels)}))

if __name__=="__main__":main()
