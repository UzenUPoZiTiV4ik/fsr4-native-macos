#!/usr/bin/env python3
"""
Compile one FSR4 INT8 shader variant for Metal:

  HLSL --(dxc -spirv)--> SPIR-V --(patch OpCopyLogical-->OpCopyObject)-->
  --(spirv-cross --msl)--> .metal --(xcrun metal/metallib)--> .metallib

and emit bindings.json describing each kernel's resource signature.

Usage:
  compile_variant.py --root <repo> --preset quality --res 1080 \
      --depth-inverted 0 --low-res-mv 1 --jittered-mv 0 \
      --colorspace 0 --auto-exposure 0 --debug-view 0 --rcas 0

Writes into <builddir>/fsr4_<key>/:
    shaders.metallib      one metallib containing all kernels needed
    bindings.json         {"kernels": {name: {"entry":..., "wgsize":..,
                            "buffers":[[idx,name],..], "textures":..,
                            "samplers":..}}}
    *.metal / *.spv / *.log (intermediates, kept for debugging)
"""
import argparse, json, os, re, struct, subprocess, sys
from pathlib import Path

def sh(cmd, log=None):
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if log:
        open(log, "wb").write(r.stdout)
    if r.returncode != 0:
        sys.stderr.write("FAIL: %s\n%s\n" % (" ".join(cmd), r.stdout.decode("utf-8", "replace")[:4000]))
        sys.exit(1)
    return r.stdout

def patch_spv(path):
    """DXC (this build) emits OpCopyLogical with identical result/operand types
    for struct members holding RWByteAddressBuffer. OpCopyObject (83) is the
    correct opcode for a same-type copy; rewrite in place."""
    data = bytearray(open(path, "rb").read())
    if struct.unpack("<I", data[:4])[0] != 0x07230203:
        sys.exit("bad spirv: " + path)
    off = 20
    n = 0
    while off < len(data):
        w0 = struct.unpack("<I", data[off:off + 4])[0]
        op, wc = w0 & 0xFFFF, w0 >> 16
        if wc == 0:
            break
        if op == 400:  # OpCopyLogical
            struct.pack_into("<I", data, off, (wc << 16) | 83)
            n += 1
        off += wc * 4
    open(path, "wb").write(data)
    return n

BIND_RE = re.compile(
    r"(constant|device|const device|const constant)\s+[\w:<>,\*\s]+?&?\s*(\w+)\s*\[\[buffer\((\d+)\)\]\]"
    r"|texture2d<[^>]*>\s*(\w+)\s*\[\[texture\((\d+)\)\]\]"
    r"|sampler\s+(\w+)\s*\[\[sampler\((\d+)\)\]\]")

def parse_signature(metal_src, entry):
    """Return resources of the kernel whose body follows `kernel void <entry>`."""
    m = re.search(r"kernel\s+void\s+%s\s*\((.*?)\)\s*\{" % re.escape(entry),
                  metal_src, re.S)
    if not m:
        sys.exit("kernel %s not found in generated MSL" % entry)
    args = m.group(1)
    bufs, texs, samps = [], [], []
    for mm in BIND_RE.finditer(args):
        if mm.group(3) is not None:
            bufs.append([int(mm.group(3)), mm.group(2)])
        elif mm.group(5) is not None:
            texs.append([int(mm.group(5)), mm.group(4)])
        elif mm.group(7) is not None:
            samps.append([int(mm.group(7)), mm.group(6)])
    return bufs, texs, samps

def wgsize(spv_path, entry, spirv_cross):
    out = sh([spirv_cross, spv_path, "--reflect"]).decode()
    for ep in json.loads(out)["entryPoints"]:
        if ep["name"] == entry:
            return ep["workgroup_size"]
    sys.exit("entry %s not in reflect output" % entry)

# --- shadow include tree -----------------------------------------------------
# Two HLSL-level transforms are applied to copies of the AMD sources (the
# originals stay untouched):
#
# 1) [unroll]/[unroll(n)] -> [loop].  The INT8 fused operators fully unroll
#    their channel loops; the resulting ~500KB kernels exceed the 40s GPU
#    watchdog when the paravirtual driver JIT-compiles them.  [loop] is a pure
#    scheduling hint: iteration order and arithmetic are identical.
# 2) The SPIR-V backend rejects the eight template assertions in the INT8
#    fused block. Disable them in the shadow copy used for either unroll mode.
INT8_FUSED_HEADER = "ml2code_runtime/operators/int8_NHWC/Fused/FNB_CT2D_ADD.hlsli"

def source_files(root):
    import glob
    files = [os.path.abspath(__file__), os.path.join(root, "tools/fsr402/gen_f16.py")]
    for d in ["patches/hlsl", "third_party/fsr4"]:
        files += glob.glob(os.path.join(root, d, "**", "*"), recursive=True)
    return [f for f in files if os.path.isfile(f)]

def source_mtime(root):
    return max(os.path.getmtime(f) for f in source_files(root))

def source_hash(root):
    """hash of everything a variant is compiled from (patches, AMD sources,
    this script) — variant dirs are rebuilt when it changes"""
    import hashlib
    h = hashlib.sha1()
    for f in sorted(source_files(root)):
        h.update(os.path.relpath(f, root).encode())
        h.update(Path(f).read_bytes())
    return h.hexdigest()

def build_shadow_tree(root, builddir, keep_unroll=False):
    import glob
    srcroots = [  # include roots, mirroring the -I layout
        os.path.join(root, "third_party/fsr4/dx12"),
        os.path.join(root, "third_party/fsr4/include/gpu"),
        os.path.join(root, "third_party/fsr4/include/gpu/fsr4"),
        os.path.join(root, "third_party/fsr4/api/internal/gpu"),
    ]
    # keep_unroll: same tree (patched copies included) with the [unroll]
    # hints left in place — used for the small pre/post/rcas kernels, where
    # the rolled loops keep per-thread arrays dynamically indexed (stack
    # traffic) and the watchdog concern of the fused model passes does not
    # apply.
    shadow = os.path.join(builddir, "hlsl_shadow_unroll" if keep_unroll else "hlsl_shadow")
    # Older shadow trees did not contain the generated FP16 PRE include.
    stamp = os.path.join(shadow, ".overlays-v2.done")
    # Rebuild the shadow when any input, including an overlay delta, changes.
    if os.path.exists(stamp) and os.path.getmtime(stamp) >= source_mtime(root):
        return shadow
    import shutil
    shutil.rmtree(shadow, ignore_errors=True)
    os.makedirs(shadow, exist_ok=True)
    n = 0
    for sr in srcroots:
        for f in glob.glob(sr + "/**/*.hls*", recursive=True) + \
                 glob.glob(sr + "/**/*.h", recursive=True):
            rel = os.path.relpath(f, sr)
            patched = os.path.join(root, "patches/hlsl", rel)
            has_patch = os.path.isfile(patched)
            with open(patched if has_patch else f, errors="replace") as source:
                txt = source.read()
            if rel == INT8_FUSED_HEADER:
                txt, count = re.subn(r"^([ \t]*)_Static_assert\(", r"\1// FSR4MAC-PATCH _Static_assert(", txt, flags=re.M)
                if count != 8:
                    raise ValueError(f"expected 8 template assertions in {rel}, found {count}")
            elif not has_patch and not re.search(r"\[\s*unroll\b", txt):
                continue
            if not keep_unroll:
                txt = re.sub(r"\[\s*unroll[^\]]*\]", "[loop]", txt)
            dst = os.path.join(shadow, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            Path(dst).write_text(txt)
            n += 1
    # The optional 4.0.2 source payload ships deltas against the pinned AMD
    # headers. PRE FP16 historically came directly from the include path and
    # keeps its unroll hints in both modes. INT8 POST follows the shadow policy.
    patch_dir = os.path.join(root, "patches/hlsl")
    if os.path.isdir(patch_dir):
        for upstream, target in [("pre_common.hlsli", "pre_common_f16.hlsli"),
                                 ("post_common.hlsli", "post_common.hlsli")]:
            source = os.path.join(root, "third_party/fsr4/include/gpu/fsr4", upstream)
            delta = os.path.join(patch_dir, target + ".patch")
            text = subprocess.check_output(["patch", "-s", "-f", "-F", "0", "-o", "-", source, delta], text=True)
            if target == "post_common.hlsli" and not keep_unroll:
                text = re.sub(r"\[\s*unroll[^\]]*\]", "[loop]", text)
            with open(os.path.join(shadow, target), "w") as header:
                header.write(text)
    Path(stamp).write_text("%d files\n" % n)
    return shadow

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--preset", required=True)
    ap.add_argument("--res", required=True, choices=["1080", "2160", "4320"])
    ap.add_argument("--res-idx", type=int, required=True)
    ap.add_argument("--depth-inverted", type=int, default=0)
    ap.add_argument("--low-res-mv", type=int, default=1)
    ap.add_argument("--jittered-mv", type=int, default=0)
    ap.add_argument("--colorspace", type=int, default=0)
    ap.add_argument("--auto-exposure", type=int, default=0)
    ap.add_argument("--debug-view", type=int, default=0)
    ap.add_argument("--rcas", type=int, default=0)
    ap.add_argument("--model", default="i8", choices=["i8", "f16"])
    ap.add_argument("--post-ext-cnb", type=int, default=-1,
                    help="int8 post loads its ConvNextBlock result from scratch at this byte offset (i8mpp)")
    ap.add_argument("--post-ext-ct2d", type=int, default=-1,
                    help="int8 post loads its model parameters (native CNB+CT2D pass 13) from scratch at this byte offset")
    ap.add_argument("--keep-unroll", type=int, default=0,
                    help="pre/post/rcas/spd/debug keep the source [unroll] hints (post ~9%% faster on M5, "
                         "but DXC then contracts/reorders differently: output is not bit-identical to the "
                         "[loop] build, same distance to the DX12 golden); the model passes always get [loop]")
    ap.add_argument("--builddir", required=True)
    a = ap.parse_args()

    if a.model == "f16":
        gendir = os.path.join(a.builddir, "gen_f16_" + a.preset + "_" + a.res)
        gen = os.path.join(a.root, "tools/fsr402/gen_f16.py")
        generated = os.path.join(gendir, "passes_%s.hlsl" % a.res)
        initializer = os.path.join(gendir, "initializers_f16.bin")
        if not os.path.exists(generated) or not os.path.exists(initializer) or os.path.getmtime(generated) < os.path.getmtime(gen):
            sh([sys.executable, gen, "--root", a.root, "--preset", a.preset,
                "--res", a.res, "--outdir", gendir])

    key = "%s_%s_di%d_lm%d_jm%d_cs%d_ae%d_db%d" % (
        a.preset, a.res, a.depth_inverted, a.low_res_mv, a.jittered_mv,
        a.colorspace, a.auto_exposure, a.debug_view)
    if a.model == "f16":
        key += "_f16"
    if a.post_ext_ct2d >= 0:
        key += "_pt"
    elif a.post_ext_cnb >= 0:
        key += "_pc"
    out = os.path.join(a.builddir, "fsr4_" + key)
    os.makedirs(out, exist_ok=True)
    meta = os.path.join(out, "bindings.json")
    # content key: the variant is reused only if it was built from the same
    # patches/sources/flags (the dir name alone is not enough)
    srchash = source_hash(a.root) + " keep_unroll=%d" % a.keep_unroll
    hashfile = os.path.join(out, ".srchash")
    if os.path.exists(meta) and os.path.exists(os.path.join(out, "shaders.metallib")):
        if os.path.exists(hashfile) and open(hashfile).read() == srchash:
            print(json.dumps({"dir": out, "key": key, "cached": True}))
            return
        for f in os.listdir(out):
            os.remove(os.path.join(out, f))

    up = os.path.join(a.root, "third_party/fsr4")
    dxc = os.path.join(a.root, "tools/bin/dxc")
    spvc = os.path.join(a.root, "tools/bin/spirv-cross")
    if a.model == "f16":
        pmdl = gendir
        model_tag = "f16"
    else:
        pmdl = os.path.join(up, "internal/shaders/fsr4_model_v07_i8_" + a.preset)
        model_tag = "i8"

    shadow = build_shadow_tree(a.root, a.builddir)
    inc_tail = [os.path.join(a.root, "patches/hlsl"),
                up + "/dx12", up + "/include/gpu", up + "/include/gpu/fsr4",
                os.path.join(a.root, "third_party/fsr4/api/internal/gpu")]
    incs = ["-I" + p for p in [shadow] + inc_tail]
    # small per-pixel kernels: [unroll] kept (see build_shadow_tree)
    incs_keep = incs
    if a.keep_unroll:
        incs_keep = ["-I" + p for p in [build_shadow_tree(a.root, a.builddir, True)] + inc_tail]

    # t/u register spaces get distinct SPV bindings so same-numbered SRV and
    # UAV resources don't collide; spirv-cross (no --msl-decoration-binding)
    # then packs [[buffer(n)]]/[[texture(n)]] indices sequentially per type.
    dxcflags = ["-spirv", "-T", "cs_6_6", "-enable-16bit-types", "-HV", "2021",
                "-O3", "-Vd", "-fspv-target-env=vulkan1.3",
                "-fvk-t-shift", "32", "0", "-fvk-u-shift", "64", "0"]

    # fp16 textures are illegal in Vulkan SPIR-V; DATA_TYPE*/TEXTURE types are
    # overridden to float. Loads then widen fp16->fp32 (exact); stores convert
    # fp32->fp16/unorm at the surface, identical to the DX12 path. Half
    # arithmetic (NETWORK_DATA_TYPE, half4 accumulators) is untouched.
    dtypes = ["-DDATA_TYPE=float", "-DDATA_TYPE2=float2", "-DDATA_TYPE3=float3",
              "-DDATA_TYPE_VECTOR=float4", "-DNETWORK_DATA_TYPE=float16_t"]

    jobs = []  # (name, entry, file, extra_defines, msl_name)
    pre_src = pmdl + "/pre_f16.hlsl" if a.model == "f16" else pmdl + "/pre.hlsl"
    post_src = pmdl + "/post_f16.hlsl" if a.model == "f16" else pmdl + "/post.hlsl"
    if a.model == "i8" and a.post_ext_ct2d >= 0:
        post_src = os.path.join(a.root, "patches/hlsl/post_filter.hlsl")
    jobs.append(("pre", "main", pre_src, [
        "-DFFX_MLSR_RESOLUTION=%d" % a.res_idx,
        "-DFFX_MLSR_DEPTH_INVERTED=%d" % a.depth_inverted,
        "-DFFX_MLSR_LOW_RES_MV=%d" % a.low_res_mv,
        "-DFFX_MLSR_AUTOEXPOSURE_ENABLED=%d" % a.auto_exposure,
        "-DFFX_MLSR_COLORSPACE=%d" % a.colorspace,
        "-DFFX_MLSR_JITTERED_MOTION_VECTORS=%d" % a.jittered_mv] + dtypes,
        "fsr4_pre"))
    for i in range(1, 13):
        jobs.append(("pass%d" % i, "fsr4_model_v07_%s_pass%d" % (model_tag, i),
                     pmdl + ("/passes_%s.hlsl" % a.res),
                     ["-DMLSR_PASS_%d=1" % i],
                     "fsr4_model_v07_%s_pass%d" % (model_tag, i)))
    post_defs = []
    if a.post_ext_ct2d >= 0:
        post_defs = ["-DFSR4_POST_EXT_CT2D=1", "-DFSR4_POST_EXT_CT2D_OFF=%d" % a.post_ext_ct2d]
    elif a.post_ext_cnb >= 0:
        post_defs = ["-DFSR4_POST_EXT_CNB=1", "-DFSR4_POST_EXT_CNB_OFF=%d" % a.post_ext_cnb]
    jobs.append(("post", "main", post_src, [
        "-DFFX_MLSR_RESOLUTION=%d" % a.res_idx,
        "-DFFX_MLSR_COLORSPACE=%d" % a.colorspace,
        "-DFFX_MLSR_AUTOEXPOSURE_ENABLED=%d" % a.auto_exposure,
        "-DFFX_DEBUG_VISUALIZE=%d" % a.debug_view] + post_defs + dtypes, "fsr4_post"))
    jobs.append(("rcas", "main", up + "/internal/shaders/rcas.hlsl", [
        "-DFFX_MLSR_COLORSPACE=%d" % a.colorspace,
        "-DFFX_MLSR_AUTOEXPOSURE_ENABLED=%d" % a.auto_exposure] + dtypes,
        "fsr4_rcas"))
    if a.auto_exposure:
        jobs.append(("spd", "main", up + "/internal/shaders/spd_auto_exposure.hlsl",
                     ["-DFFX_GPU", "-DFFX_HLSL", "-DFFX_HLSL_SM=66"], "fsr4_spd"))
    if a.debug_view:
        jobs.append(("dbg", "main", up + "/internal/shaders/debug_view.hlsl", [
            "-DFFX_MLSR_RESOLUTION=%d" % a.res_idx,
            "-DFFX_MLSR_DEPTH_INVERTED=%d" % a.depth_inverted,
            "-DFFX_MLSR_LOW_RES_MV=%d" % a.low_res_mv,
            "-DFFX_MLSR_AUTOEXPOSURE_ENABLED=%d" % a.auto_exposure,
            "-DFFX_MLSR_COLORSPACE=%d" % a.colorspace,
            "-DFFX_MLSR_JITTERED_MOTION_VECTORS=%d" % a.jittered_mv,
            "-DFFX_DEBUG_VISUALIZE=1"] + dtypes, "fsr4_dbg"))

    manifest = {"kernels": {}, "key": key, "preset": a.preset, "res": a.res}
    airs = []
    for name, entry, f, defs, msl_name in jobs:
        spv = os.path.join(out, name + ".spv")
        mtl = os.path.join(out, name + ".metal")
        air = os.path.join(out, name + ".air")
        job_incs = incs if name.startswith("pass") else incs_keep
        sh([dxc] + dxcflags + job_incs + ["-E", entry] + defs + [f, "-Fo", spv],
           log=os.path.join(out, name + ".dxc.log"))
        patch_spv(spv)
        src = sh([spvc, spv, "--msl", "--msl-version", "20300"],
                 log=os.path.join(out, name + ".xlog")).decode()
        # 'main' kernels get renamed for unique entry names in the metallib
        src = re.sub(r"kernel\s+void\s+main0", "kernel void " + msl_name, src)
        # D3D12 RWTexture2D<float3> stores expand to 4 components with
        # alpha = .x (verified against WARP golden dumps); spirv-cross emits
        # .xyzz. Every ".xyzz, uint2" occurrence is a texture write arg.
        src = src.replace(".xyzz, uint2", ".xyzx, uint2")
        open(mtl, "w").write(src)
        bufs, texs, samps = parse_signature(src, msl_name)
        manifest["kernels"][msl_name] = {
            "buffers": bufs, "textures": texs, "samplers": samps,
            "wgsize": wgsize(spv, entry, spvc)}
        sh(["xcrun", "metal", "-std=macos-metal2.3", "-c", mtl, "-o", air],
           log=os.path.join(out, name + ".mlog"))
        airs.append(air)

    sh(["xcrun", "metallib"] + airs + ["-o", os.path.join(out, "shaders.metallib")],
       log=os.path.join(out, "metallib.log"))
    json.dump(manifest, open(meta, "w"), indent=1)
    open(hashfile, "w").write(srchash)
    # flat, C-friendly view of the manifest
    with open(os.path.join(out, "bindings.tsv"), "w") as f:
        for kname, k in manifest["kernels"].items():
            f.write("K\t%s\t%d\t%d\t%d\n" % (kname, *k["wgsize"]))
            for i, n in k["buffers"]:
                f.write("B\t%s\t%d\t%s\n" % (kname, i, n))
            for i, n in k["textures"]:
                f.write("T\t%s\t%d\t%s\n" % (kname, i, n))
            for i, n in k["samplers"]:
                f.write("S\t%s\t%d\t%s\n" % (kname, i, n))
    print(json.dumps({"dir": out, "key": key, "cached": False}))

if __name__ == "__main__":
    main()
