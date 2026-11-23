#!/usr/bin/env python3
"""
Generate the FP16 FSR4 model shaders + dequantized weight blob from the
INT8 sources in third_party/fsr4.

  gen_f16.py --root <repo> --preset <preset> --res 1080|2160|4320 --outdir <dir>

Outputs into <outdir>/:
  passes_<res>.hlsl      fp16 model passes (fsr4_model_v07_f16_passN entries)
  pre_f16.hlsl           fp16 pre pass (includes fsr4/pre_common_f16.hlsli)
  post_f16.hlsl          fp16 post pass (includes fsr4/post_common_f16.hlsli)
  initializers_f16.bin   fp16 weight blob (same layout, byte offsets x2)

Transform rules
  - QuantizedTensor4i8_* weight decls  -> Tensor4h_*   (byte strides x2)
  - QuantizedTensor3i8_NHWC act decls  -> Tensor3h_NHWC (byte strides x2)
  - Tensor3i8_NHWC act decls           -> Tensor3h_NHWC (byte strides x2)
  - embedded int8 weight dword arrays  -> fp16-packed arrays (x2 dwords)
  - initializer-buffer int8 regions    -> fp16 blob, byte offsets x2
  - op call names -> f16_* ; an output-scale float is appended where the
    int8 op took its store clamp from output.quantizationScale
"""
import argparse, json, os, re, struct, sys

# ---------------------------------------------------------------- helpers

def f16pack(vals):
    out = []
    for i in range(0, len(vals), 2):
        a = vals[i]
        b = vals[i + 1] if i + 1 < len(vals) else 0.0
        out.append(struct.unpack("<I", struct.pack("<e", a) + struct.pack("<e", b))[0])
    return out

def unpack_i8_dwords(dwords):
    vals = []
    for d in dwords:
        for k in range(4):
            b = (d >> (8 * k)) & 0xFF
            vals.append(b - 256 if b >= 128 else b)
    return vals

def split_fields(s):
    """top-level comma split (commas inside parens stay together);
    line comments are stripped first."""
    s = re.sub(r"//[^\n]*", "", s)
    fields, depth, cur = [], 0, ""
    for ch in s:
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        if ch == "," and depth == 0:
            fields.append(cur.strip())
            cur = ""
        else:
            cur += ch
    if cur.strip():
        fields.append(cur.strip())
    return fields

def is_tuple(f):
    return bool(re.match(r"^u?int\d\(", f))

def is_float(f):
    return bool(re.match(r"^-?\d+\.\d+(e-?\d+)?$", f.strip()))

DECL_RE = re.compile(
    r"const (QuantizedTensor4i8_NHWC|QuantizedTensor4i8_HWCN|QuantizedTensor4i8_HWNC|"
    r"QuantizedTensor3i8_NHWC|QuantizedTensor3i8|Tensor3i8_NHWC|"
    r"Tensor3h_NHWC|Tensor4h_NHWC|Tensor4h_HWCN|Tensor4h_HWNC|Tensor1h|Tensor1f|QuantizedTensor1f32)"
    r"<\s*([A-Za-z0-9_<>]+)\s*>\s*(\w+)\s*=\s*\{(.*?)\};",
    re.S)

TYPE_MAP = {
    "QuantizedTensor4i8_NHWC": "Tensor4h_NHWC",
    "QuantizedTensor4i8_HWCN": "Tensor4h_HWCN",
    "QuantizedTensor4i8_HWNC": "Tensor4h_HWNC",
    "QuantizedTensor3i8_NHWC": "Tensor3h_NHWC",
    "QuantizedTensor3i8": "Tensor3h_NHWC",
    "Tensor3i8_NHWC": "Tensor3h_NHWC",
}

def tensor_scale(fields):
    """quantizationScale = field before the trailing storage ref (literal or
    const-float symbol)."""
    if len(fields) >= 2 and re.match(r"^[A-Za-z_0-9.-]+$", fields[-2]) and \
            not fields[-2].startswith("storage") and \
            not is_tuple(fields[-2]) and not re.match(r"^\d+$", fields[-2]):
        return fields[-2]
    return None

def transform_decl(m):
    """-> (new_text, name, kind) where kind: act|wbuf|wconst|keep"""
    typ, storage, name = m.group(1), m.group(2), m.group(3)
    init = m.group(4)
    if typ not in TYPE_MAP:
        return m.group(0), name, "keep"
    fields = split_fields(init)
    # tuple fields: logicalSize, sliceStart, sliceSize, storageSize, STRIDES,
    # padBegin, padEnd -> strides at index 4 get x2.
    # quantized decls have a trailing (offset, scale, storage); non-quantized
    # are (offset, storage). The byte-offset field -> x2.
    quantized = typ.startswith("Quantized")
    off_idx = len(fields) - (3 if quantized else 2)
    newfields = []
    for i, f in enumerate(fields):
        if is_tuple(f) and i == 4:
            dims = re.findall(r"\d+", f[f.index("("):])
            dim = re.match(r"(u?int\d)", f).group(1)
            newfields.append("%s(%s)" % (dim, ", ".join(str(int(x) * 2) for x in dims)))
        elif quantized and i == off_idx + 1:
            continue  # quantizationScale dropped
        elif i == off_idx:
            mm = re.search(r"\+\s*(\d+)", f)
            if mm:
                f = re.sub(r"\+\s*\d+", "+ %d" % (int(mm.group(1)) * 2), f)
            elif re.match(r"^\d+$", f):
                f = str(int(f) * 2)
            newfields.append(f)
        else:
            newfields.append(f)
    storage = re.sub(r"ConstantBufferStorage<(\d+)>",
                     lambda mm: "ConstantBufferStorage<%d>" % (int(mm.group(1)) * 2),
                     storage)
    txt = "const %s< %s > %s = {\n        %s };" % (
        TYPE_MAP[typ], storage, name, ",\n        ".join(newfields))
    if typ.startswith("QuantizedTensor4i8"):
        kind = "wbuf" if storage == "BufferStorage" else "wconst"
    else:
        kind = "act"
    return txt, name, kind

def dword_array(src, name):
    m = re.search(r"static const uint %s_dwords\[(\d+)\] = \{(.*?)\};" % name, src, re.S)
    if not m:
        return None
    body = re.sub(r"//[^\n]*", "", m.group(2))
    return m.start(), m.end(), int(m.group(1)), [int(x, 16) for x in re.findall(r"0x[0-9a-fA-Fx]+", body)]

def region_extent(fields):
    """byte extent of an int8 tensor = element count (1 byte each)."""
    tup = [f for f in fields if is_tuple(f)]
    if len(tup) < 4:
        return 0
    ssz = [int(x) for x in re.findall(r"\d+", tup[3][tup[3].index("("):])]
    if len(tup) >= 5:
        st = [int(x) for x in re.findall(r"\d+", tup[4][tup[4].index("("):])]
        return sum((s - 1) * t for s, t in zip(ssz, st)) + 1
    n = 1
    for s in ssz:
        n *= s
    return n

_fconsts = {}
def resolve(v):
    return _fconsts.get(v, v)

def scan_args(s, pos):
    """top-level comma split of the call args whose '(' is just before pos;
    -> (fields, index just past the matching ')')."""
    depth, j = 1, pos
    while depth:
        if s[j] == "(":
            depth += 1
        elif s[j] == ")":
            depth -= 1
        j += 1
    return split_fields(s[pos:j - 1]), j

def apply_edits(src, edits):
    """splice (start, end, text) edits into src; spans inside an earlier
    edit are dropped."""
    edits.sort(key=lambda t: (t[0], -t[1]))
    out, pos = [], 0
    for s, e, txt in edits:
        if s < pos:
            continue
        out.append(src[pos:s]); out.append(txt); pos = e
    out.append(src[pos:])
    return "".join(out)

def collect_tensors(src):
    """-> (scales, tensors)  scales: tensor name -> resolved quantization
    scale (quantized decls); tensors: name -> decl info."""
    _fconsts.update(re.findall(r"const float (\w+) = ([-0-9.eE]+)", src))
    scales = {}
    tensors = {}
    for m in DECL_RE.finditer(src):
        typ, storage, name = m.group(1), m.group(2), m.group(3)
        fields = split_fields(m.group(4))
        sc = tensor_scale(fields)
        if sc is not None:
            scales[name] = resolve(sc)
        tensors[name] = dict(typ=typ, storage=storage, scale=resolve(sc) if sc else None,
                             fields=fields, span=(m.start(), m.end()))
    return scales, tensors

def dequant_edits(src, tensors, regions=None):
    """embedded array dequant: storage decl `{ NAME_dwords }` for a
    QuantizedTensor4i8 constant tensor -> emit *_f16dwords. BufferStorage
    QuantizedTensor4i8 decls append (off, extent, scale) to regions."""
    edits = []
    for name, ti in tensors.items():
        if not ti["typ"].startswith("QuantizedTensor4i8"):
            continue
        sm = re.search(
            r"const ConstantBufferStorage<(\d+)> storage_%s = \{ (\w+)_dwords \};" % name, src)
        if ti["storage"].startswith("ConstantBufferStorage") and sm:
            arr = sm.group(2)
            pa = dword_array(src, arr)
            if not pa:
                sys.exit("array %s not found" % arr)
            s, e, n, dw = pa
            packed = f16pack([v * float(ti["scale"]) for v in unpack_i8_dwords(dw)])
            lines = ["    " + ", ".join("0x%x" % d for d in packed[i:i + 8]) + ","
                     for i in range(0, len(packed), 8)]
            edits.append((s, e, "static const uint %s_f16dwords[%d] = {\n%s\n};" %
                          (arr, len(packed), "\n".join(lines))))
            edits.append((sm.start(), sm.end(),
                          "const ConstantBufferStorage<%d> storage_%s = { %s_f16dwords };" %
                          (int(sm.group(1)) * 2, name, arr)))
        elif ti["storage"] == "BufferStorage" and regions is not None:
            off = None
            for f in ti["fields"]:
                if re.match(r"^\d+$", f):
                    off = int(f)
            if off is None:
                sys.exit("no offset for " + name)
            regions.append((off, region_extent(ti["fields"]), float(ti["scale"])))
    return edits

def common_edits(src):
    """decl rewrites — every DECL_RE match (tensor names can repeat across
    #if variants, so iterate matches not the name dict) — plus the slice
    scalar decls."""
    edits = []
    for m in DECL_RE.finditer(src):
        if m.group(1) in TYPE_MAP:
            newtxt, _, _ = transform_decl(m)
            edits.append((m.start(), m.end(), newtxt))
    for mm in re.finditer(r"const uint3 tensorByteStrides_(\w+) = uint3\((\d+), (\d+), (\d+)\);", src):
        a, b, c = int(mm.group(2)) * 2, int(mm.group(3)) * 2, int(mm.group(4)) * 2
        edits.append((mm.start(), mm.end(),
                      "const uint3 tensorByteStrides_%s = uint3(%d, %d, %d);" % (mm.group(1), a, b, c)))
    return edits

def transform_model_source(src):
    """-> (new_src, blob_regions)  blob_regions: [(off, extent, scale)]"""
    scales, tensors = collect_tensors(src)
    blob_regions = []
    edits = dequant_edits(src, tensors, blob_regions) + common_edits(src)
    src = apply_edits(src, edits)

    # entry/include renames
    src = src.replace("fsr4_model_v07_i8_", "fsr4_model_v07_f16_")
    src = re.sub(r'#include "ml2code_runtime/operators/int8_NHWC/[^"]+"',
                 '#include "ml2code_runtime/fp16_ops.hlsli"\n#include "ml2code_runtime/tensor_float16.hlsli"', src)
    src = src.replace('#include "ml2code_runtime/tensor_int8.hlsli"',
                      '#include "ml2code_runtime/tensor_float16.hlsli"')
    # padding batch ops: int8 16B batch -> half4 16B batch on fp16 tensors
    src = re.sub(
        r"StoreConstBatchOperation\s*<\s*int8_t4_packed\s*,\s*16\s*,\s*QuantizedTensor3i8_NHWC",
        "StoreConstBatchOperation < half4, 8, Tensor3h_NHWC", src)

    # --- call rewrites (operate on transformed text; tensor names unchanged)
    call_edits = []
    for m in re.finditer(
            r"(?<![A-Za-z0-9_])(Conv2D_k2s2b|ConvNextBlock|FusedConv2D_k2s2b_QuantizedOutput|"
            r"FasterNetBlock|FNB_CT2D_ADD)\s*(<[0-9, ]+>)?\s*\(", src):
        op, tmpl = m.group(1), m.group(2) or ""
        args, end = scan_args(src, m.end())
        tail = args[-1].strip()
        if tail != "computeShaderParams":
            continue    # comment/prototype matches, not a real dispatch call
        floats = [a for a in args[:-1] if is_float(a)]
        tens = [a for a in args[:-1] if not is_float(a)]
        if op == "Conv2D_k2s2b":
            nc = "f16_Conv2D_k2s2b(%s, %s, %s)" % (scales[tens[-1]], ", ".join(tens), tail)
        elif op == "ConvNextBlock":
            nc = "f16_ConvNextBlock(%s, %s, %s, %s, %s, %s, %s)" % (
                floats[0], floats[1], floats[2], floats[3], scales[tens[-1]],
                ", ".join(tens), tail)
        elif op == "FusedConv2D_k2s2b_QuantizedOutput":
            nc = "f16_FusedConv2D_k2s2b_QuantizedOutput(%s)" % ", ".join(args)
        elif op == "FasterNetBlock":
            fl = list(floats)
            if len(fl) == 3:    # output split scales come from the output decl
                fl += [scales[tens[-1]], scales[tens[-1]]]
            nc = "f16_FNB%s(%s, %s, %s)" % (tmpl, ", ".join(fl), ", ".join(tens), tail)
        elif op == "FNB_CT2D_ADD":
            fl = list(floats)
            if len(fl) == 4:        # <32,1>: i8 took clamp from output.scale
                fl += [scales[tens[-1]], scales[tens[-1]]]
            nc = "f16_FNB_CT2D_ADD%s(%s, %s)" % (tmpl, ", ".join(fl + tens), tail)
        else:
            continue
        call_edits.append((m.start(), end, nc))

    return apply_edits(src, call_edits), blob_regions

def transform_post(src):
    """post.hlsl: same machinery but simpler call rewrite (the fused
    FusedConv2D_DW_Conv2D_PW_Relu_Conv2D_Add_ConvTranspose2D -> same name is
    defined in post_common_f16.hlsli with fp16 signature)."""
    scales, tensors = collect_tensors(src)
    # post has no InitializerBuffer tensors -> regions stays empty
    src = apply_edits(src, dequant_edits(src, tensors) + common_edits(src))
    src = src.replace('#include "post_common.hlsli"', '#include "post_common_f16.hlsli"')
    src = re.sub(
        r"StoreConstBatchOperation\s*<\s*int8_t4_packed\s*,\s*16\s*,\s*QuantizedTensor3i8_NHWC",
        "StoreConstBatchOperation < half4, 8, Tensor3h_NHWC", src)
    return src, scales

def gen_pre(pre_src):
    """pre.hlsl -> pre_f16.hlsl: swap include only; pre_common_f16 carries
    the fp16 slice_0 decl + store."""
    src = pre_src.replace('#include "pre_common.hlsli"',
                          '#include "pre_common_f16.hlsli"')
    if "pre_common_f16" not in src:
        sys.exit("gen_f16: pre_common include not found in pre.hlsl")
    return src

# ---------------------------------------------------------------- mpp sidecar
# Emit mpp_manifest.json + mpp_weights.bin for the native MSL matmul2d FNB
# kernels (src/fnb_mpp.metal). Parses the *generated f16* passes source:
# per f16_FNB<NF,NG> call: in/out scratch tensors, scales, and weight tensors
# (embedded *_f16dwords / *_dwords arrays or InitializerBuffer regions).
# pw weights are emitted transposed to [K][N] row-major (N contiguous) with
# K padded to 64, which is the operand-B layout matmul2d slices.

def _u32s(arr_text):
    return [int(x, 16) for x in re.findall(r"0x[0-9a-fA-F]+", arr_text)]

def _halves_of_dwords(dw):
    out = []
    for d in dw:
        out.append(struct.unpack("<e", struct.pack("<H", d & 0xFFFF))[0])
        out.append(struct.unpack("<e", struct.pack("<H", (d >> 16) & 0xFFFF))[0])
    return out

def kn_transpose(w, dims, strd, kpad):
    """weight halves -> [K][N] row-major (N contiguous), K padded to kpad;
    src elem (c,i) at c*strd[2] + i*strd[3]."""
    k, n = dims[2], dims[3]
    out = [0.0] * (kpad * n)
    for c in range(k):
        for i in range(n):
            out[c * n + i] = w[int(c * strd[2] + i * strd[3])]
    return out

def emit_mpp(src, initf16, outdir):
    manifest = {"passes": {}}
    blob = bytearray()

    def put(vals):
        nonlocal blob
        off = len(blob)
        for v in vals:
            blob += struct.pack("<e", v)
        return off

    def ctx(body):
        """(tinfo, wdata) lookup helpers bound to one pass body."""

        def decl(name):
            m = re.search(
                r"const (?:Tensor3h_NHWC|Tensor4h_NHWC|Tensor4h_HWCN|Tensor1h)"
                r"\s*<[^;=]*?>\s+" + re.escape(name) + r"\s*=\s*\{(.*?)\};",
                body, re.S)
            return split_fields(m.group(1)) if m else None

        def resolve_tup(f):
            """field -> list of ints; handles literal tuples and
            `const uintN NAME = uintN(...)` symbolic refs."""
            if is_tuple(f):
                return [int(v) for v in
                        re.findall(r"\d+", f[f.index("("):])]
            m = re.search(r"const u?int\d\s+" + re.escape(f) +
                          r"\s*=\s*u?int\d\(([^)]*)\)", body)
            if not m:
                sys.exit("mpp: cannot resolve " + f)
            return [int(v) for v in re.findall(r"\d+", m.group(1))]

        def tinfo(name):
            """-> dict(off, strides[bytes], dims) for Tensor3h/4h decls"""
            f = decl(name)
            if f is None:
                sys.exit("mpp: no decl for " + name)
            return dict(off=int(re.search(r"(\d+)\s*$", f[len(f) - 2]).group(1)),
                        strides=resolve_tup(f[4]), dims=resolve_tup(f[0]))

        def wdims_strd(f):
            """(dims, elem strides) from decl fields; Tensor1h fields are
            scalars, Tensor*h use tuple fields."""
            tup = [x for x in f if is_tuple(x)]
            if len(tup) >= 5:
                return ([int(v) for v in
                         re.findall(r"\d+", tup[0][tup[0].index("("):])],
                        [int(v) // 2 for v in
                         re.findall(r"\d+", tup[4][tup[4].index("("):])])
            # Tensor1h: logicalSize, sliceStart, sliceSize, storageSize,
            # stride(bytes), padB, padE, offset, storage
            nums = [x for x in f if re.match(r"^-?\d+$", x)]
            return [int(nums[0])], [int(nums[4]) // 2]

        def wdata(name):
            """-> (halves, dims, strides[elems]) for a weight Tensor4h/1h"""
            f = decl(name)
            dims, strd = wdims_strd(f)
            sm = re.search(r"const ConstantBufferStorage<\d+> storage_" +
                           re.escape(name) + r"\s*=\s*\{\s*(\w+)\s*\};", body)
            if sm:
                am = re.search(r"static const uint %s\[(\d+)\]\s*=\s*\{(.*?)\};"
                               % re.escape(sm.group(1)), src, re.S)
                if not am:
                    sys.exit("mpp: array " + sm.group(1) + " not found")
                return _halves_of_dwords(_u32s(am.group(2))), dims, strd
            off = int(re.search(r"(\d+)\s*$", f[len(f) - 2]).group(1))
            n = 1
            for d in dims:
                n *= d
            raw = initf16[off:off + n * 2]
            return [struct.unpack("<e", raw[k * 2:k * 2 + 2])[0]
                    for k in range(n)], dims, strd

        return tinfo, wdata

    def call_args(body, cm):
        args, _ = scan_args(body, cm.end())
        floats = [a for a in args if is_float(a)]
        tens = [a for a in args if not is_float(a)]
        if len(floats) != 5 or len(tens) != 9 or tens[-1] != "computeShaderParams":
            return None
        return floats, tens

    passes = re.split(r"(void fsr4_model_v07_f16_pass\d+\()", src)
    bodies = [(int(re.search(r"pass(\d+)", passes[i]).group(1)), passes[i + 1])
              for i in range(1, len(passes), 2)]
    for pn, body in bodies:
        cm = re.search(r"f16_FNB<(\d+),\s*(\d+)>\s*\(", body)
        if not cm:
            continue
        nf, ng = int(cm.group(1)), int(cm.group(2))
        ft = call_args(body, cm)
        if ft is None:
            continue
        floats, tens = ft
        input_, w0, b0, w1, b1, w2, b2, output_ = tens[:8]
        tinfo, wdata = ctx(body)
        ti = tinfo(input_)
        to = tinfo(output_)
        w0h, w0d, w0s = wdata(w0)
        b0h, _, _ = wdata(b0)
        w1h, w1d, w1s = wdata(w1)   # dims (1,1,Cin,Cout)
        b1h, _, _ = wdata(b1)
        w2h, w2d, w2s = wdata(w2)
        b2h, _, _ = wdata(b2)
        cin, cout = w1d[2], w1d[3]
        kpad = (cin + 63) // 64 * 64
        k2, n2 = w2d[2], w2d[3]
        manifest["passes"][str(pn)] = dict(
            kind="fnb",
            nf=nf, ng=ng,
            ciPG=w0d[2], fPG=(nf // 2) // ng,
            inT=ti, outT=to,
            w0s=w0s, w0d=w0d,
            K1=kpad, N1=cout, K2=k2, N2=n2,
            scales=[floats[1], floats[2], floats[3], floats[4]],
            w0=put(w0h), b0=put(b0h),
            w1=put(kn_transpose(w1h, w1d, w1s, kpad)), b1=put(b1h),
            w2=put(kn_transpose(w2h, w2d, w2s, k2)), b2=put(b2h))
    # second pass: CNB calls in the passes no FNB entry claimed
    for pn, body in bodies:
        if str(pn) in manifest["passes"]:
            continue
        cm = re.search(r"f16_ConvNextBlock\s*\(", body)
        if not cm:
            continue
        ft = call_args(body, cm)
        if ft is None:
            continue
        floats, tens = ft
        input_, w0, b0, w1, b1, w2, b2, output_ = tens[:8]
        tinfo, wdata = ctx(body)
        w0h, w0d, w0s = wdata(w0)
        w1h, w1d, w1s = wdata(w1)
        w2h, w2d, w2s = wdata(w2)
        # kernel supports only the CNB16 shape: 16->16(3x3), 16->32, 32->16
        if not (w0d == [3, 3, 16, 16] and w1d == [1, 1, 16, 32]
                and w2d == [1, 1, 32, 16]):
            continue
        b0h, _, _ = wdata(b0)
        b1h, _, _ = wdata(b1)
        b2h, _, _ = wdata(b2)
        manifest["passes"][str(pn)] = dict(
            kind="cnb",
            inT=tinfo(input_), outT=tinfo(output_),
            w0s=w0s, w0d=w0d, w1s=w1s, w2s=w2s,
            # kernel wants final requant scales: conv1/2 take 1/rcpr,
            # contract store takes outputScale directly
            scales=[1.0 / float(floats[0]), 1.0 / float(floats[2]),
                    float(floats[4])],
            w0=put(w0h), b0=put(b0h), w1=put(w1h), b1=put(b1h),
            w2=put(w2h), b2=put(b2h))
    if manifest["passes"]:
        open(os.path.join(outdir, "mpp_weights.bin"), "wb").write(blob)
        open(os.path.join(outdir, "mpp_manifest.json"), "w").write(
            json.dumps(manifest, indent=1))
        print("gen_f16: mpp manifest for passes %s" %
              ",".join(sorted(manifest["passes"], key=int)))

def write_blob(regions, init_blob, outpath):
    size = len(init_blob) * 2
    out = bytearray(size)
    for off, ext, scale in regions:
        for b in range(ext):
            q = struct.unpack("b", init_blob[off + b:off + b + 1])[0]
            struct.pack_into("<e", out, (off + b) * 2, q * scale)
    open(outpath, "wb").write(out)
    return bytes(out)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--preset", required=True)
    ap.add_argument("--res", required=True, choices=["1080", "2160", "4320"])
    ap.add_argument("--outdir", required=True)
    a = ap.parse_args()

    pdir = os.path.join(a.root, "third_party/fsr4",
                        "internal/shaders/fsr4_model_v07_i8_" + a.preset)
    init_blob = open(os.path.join(pdir, "initializers.bin"), "rb").read()

    passes = open(os.path.join(pdir, "passes_%s.hlsl" % a.res)).read()
    newsrc, regions = transform_model_source(passes)
    os.makedirs(a.outdir, exist_ok=True)
    open(os.path.join(a.outdir, "passes_%s.hlsl" % a.res), "w").write(newsrc)

    post = open(os.path.join(pdir, "post.hlsl")).read()
    postsrc, _ = transform_post(post)
    open(os.path.join(a.outdir, "post_f16.hlsl"), "w").write(postsrc)

    pre = open(os.path.join(pdir, "pre.hlsl")).read()
    open(os.path.join(a.outdir, "pre_f16.hlsl"), "w").write(gen_pre(pre))

    initf16 = write_blob(regions, init_blob,
               os.path.join(a.outdir, "initializers_f16.bin"))
    emit_mpp(newsrc, initf16, a.outdir)
    print("gen_f16: %s @%s -> %s (%d buffer weights)" %
          (a.preset, a.res, a.outdir, len(regions)))

if __name__ == "__main__":
    main()
