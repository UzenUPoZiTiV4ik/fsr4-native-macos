#!/usr/bin/env python3
"""
gen_i8mpp.py -- manifest + weight blob for the native Metal 4 int8 model
kernels (src/fsr402/i8mpp.metal). Reads the AMD int8 HLSL model passes and emits

  <outdir>/i8mpp_manifest.json   per-pass kind, tensor geometry, byte offsets
                                 of weights/biases in the blob, requant scales
  <outdir>/i8mpp_weights.bin     int8 weights re-laid for mpp::tensor_ops
                                 (conv HWIO, pointwise [K][N]) + fp16 biases

The requant expressions are transcribed from the MSL that DXC+spirv-cross
produced for the same passes (see docs/I8MPP.md): products of two literal
scales are folded in fp32, everything else keeps the source association.

  gen_i8mpp.py --root <repo> --preset <preset> --res 1080|2160|4320 --outdir <dir>
"""
import argparse, json, os, re, struct, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen_f16 import split_fields, is_tuple, is_float, unpack_i8_dwords, _u32s, _halves_of_dwords  # noqa

f32 = np.float32


def fmul(a, b):
    return float(f32(f32(a) * f32(b)))


def frcp(a):
    return float(f32(f32(1.0) / f32(a)))


class Pass:
    def __init__(self, src, n, init):
        a = src.find("#ifdef MLSR_PASS_%d\n" % n)
        b = src.find("#endif // #ifdef MLSR_PASS_%d\n" % n)
        if a < 0 or b < 0:
            sys.exit("pass %d not found" % n)
        self.n = n
        self.src = src
        self.init = init
        self.body = src[a:b]
        m = re.search(r"\n\s+(FasterNetBlock|FNB_CT2D_ADD|ConvNextBlock|"
                      r"FusedConv2D_k2s2b_QuantizedOutput)(<[^>]*>)?\s*\((.*?)\);",
                      self.body, re.S)
        if not m:
            sys.exit("pass %d: no operator call" % n)
        self.op = m.group(1)
        self.tmpl = [int(v) for v in re.findall(r"\d+", m.group(2) or "")]
        self.parse_args(m.group(3))

    def parse_args(self, argtext):
        args = split_fields(argtext)
        self.floats = [float(x) for x in args if is_float(x)]
        self.tens = [x for x in args if not is_float(x) and x != "computeShaderParams"]

    def decl(self, name):
        m = re.search(r"const\s+((?:Quantized)?Tensor\w+)\s*<\s*([A-Za-z0-9_<>]+)\s*>\s+"
                      + re.escape(name) + r"\s*=\s*\{(.*?)\};", self.body, re.S)
        if not m:
            sys.exit("pass %d: no decl for %s" % (self.n, name))
        return m.group(1), m.group(2), split_fields(m.group(3))

    def tup(self, f):
        if is_tuple(f):
            return [int(v) for v in re.findall(r"-?\d+", f[f.index("("):])]
        m = re.search(r"const\s+u?int\d\s+" + re.escape(f) + r"\s*=\s*u?int\d\(([^)]*)\)", self.body)
        if not m:
            sys.exit("pass %d: cannot resolve tuple %s" % (self.n, f))
        return [int(v) for v in re.findall(r"-?\d+", m.group(1))]

    def scalar(self, f):
        f = f.strip()
        if re.match(r"^-?\d+$", f):
            return int(f)
        m = re.match(r"^(\w+)\s*\+\s*(\d+)$", f)
        if m:
            return self.scalar(m.group(1)) + int(m.group(2))
        m = re.search(r"const\s+u?int\s+" + re.escape(f) + r"\s*=\s*(\d+)\s*;", self.body)
        if m:
            return int(m.group(1))
        m = re.search(r"const\s+u?int\s+" + re.escape(f) + r"\s*=\s*dot\(\s*(\w+)\s*,\s*(\w+)\s*\)\s*;", self.body)
        if m:
            g = self.tup(m.group(1))
            st = self.tup(m.group(2))
            return sum(x * y for x, y in zip(g, st))
        sys.exit("pass %d: cannot resolve scalar %s" % (self.n, f))

    def fscalar(self, f):
        f = f.strip()
        if is_float(f):
            return float(f)
        m = re.search(r"const\s+float\s+" + re.escape(f) + r"\s*=\s*(-?[\d.]+(?:e-?\d+)?)\s*;", self.body)
        if not m:
            sys.exit("pass %d: cannot resolve float %s" % (self.n, f))
        return float(m.group(1))

    def act(self, name):
        """activation tensor -> dict(off, sx, sy, w, h, c, qs)"""
        ty, _, f = self.decl(name)
        dims = self.tup(f[0])
        strides = self.tup(f[4])
        if strides[2] != 1:
            sys.exit("pass %d: %s channel stride %d" % (self.n, name, strides[2]))
        pad = self.tup(f[5]) + self.tup(f[6])
        if any(pad):
            sys.exit("pass %d: %s has padding" % (self.n, name))
        off = self.scalar(f[7])
        qs = None
        if ty.startswith("Quantized"):
            qs = self.fscalar(f[8])
        return dict(off=off, sx=strides[0], sy=strides[1], w=dims[0], h=dims[1], c=dims[2], qs=qs)

    def dwords(self, name):
        """embedded uint array behind `storage_<name>`, or None when the
        tensor lives in the initializer blob."""
        sm = re.search(r"const ConstantBufferStorage<\d+>\s+storage_" + re.escape(name)
                       + r"\s*=\s*\{\s*(\w+)\s*\};", self.body)
        if not sm:
            return None
        am = re.search(r"static const uint %s\[(\d+)\]\s*=\s*\{(.*?)\};" % re.escape(sm.group(1)),
                       self.src, re.S)
        if not am:
            sys.exit("pass %d: dword array %s missing" % (self.n, sm.group(1)))
        return _u32s(am.group(2))

    def wdata(self, name):
        """-> (int8 list, dims, elem strides, qscale)"""
        ty, storage, f = self.decl(name)
        dims = self.tup(f[0])
        strides = self.tup(f[4])
        qs = self.fscalar(f[8]) if ty.startswith("Quantized") else None
        n = 1
        for d in dims:
            n *= d
        dw = self.dwords(name)
        if dw is not None:
            vals = unpack_i8_dwords(dw)[:n]
        else:
            off = self.scalar(f[7])
            raw = self.init[off:off + n]
            vals = [struct.unpack("b", raw[k:k + 1])[0] for k in range(n)]
        return vals, dims, strides, qs

    def hdata(self, name):
        """fp16 bias -> list of floats"""
        ty, storage, f = self.decl(name)
        n = int(re.findall(r"\d+", f[0])[0])
        dw = self.dwords(name)
        if dw is not None:
            return _halves_of_dwords(dw)[:n]
        off = self.scalar(f[7])
        raw = self.init[off:off + n * 2]
        return [struct.unpack("<e", raw[k * 2:k * 2 + 2])[0] for k in range(n)]


def select_resolution(src, res):
    """keep only the `#if FFX_MLSR_RESOLUTION == RESOLUTION_<res>` branches"""
    tag = "RESOLUTION_" + res
    out, keep, inblk = [], True, False
    for line in src.splitlines(True):
        st = line.strip()
        if st.startswith("#if FFX_MLSR_RESOLUTION"):
            inblk, keep = True, tag in st
            continue
        if inblk and st.startswith("#elif FFX_MLSR_RESOLUTION"):
            keep = tag in st
            continue
        if inblk and st.startswith("#else"):
            keep = False
            continue
        if inblk and st.startswith("#endif"):
            inblk, keep = False, True
            continue
        if keep:
            out.append(line)
    return "".join(out)


class PostPass(Pass):
    def __init__(self, src, init):
        self.n = 13
        self.src = src
        self.init = init
        self.body = src
        m = re.search(r"\n\s+FusedConv2D_DW_Conv2D_PW_Relu_Conv2D_Add_ConvTranspose2D\s*\((.*?)\);", src, re.S)
        if not m:
            sys.exit("post: no fused CNB+CT2D call")
        self.op = "PostCNB"
        self.tmpl = []
        self.parse_args(m.group(1))


class Blob:
    def __init__(self):
        self.b = bytearray()

    def pad64(self):
        while len(self.b) % 64:
            self.b.append(0)
        return len(self.b)

    def put_i8(self, vals):
        off = self.pad64()
        self.b += bytes((v & 0xFF) for v in vals)
        return off

    def put_h(self, vals):
        off = self.pad64()
        for v in vals:
            self.b += struct.pack("<e", v)
        return off


def conv_hwio(vals, dims, strides, f0, nf):
    """weights (kx,ky,ci,co) with elem strides -> HWIO block [ky][kx][ci][co']
    for co' in [f0, f0+nf)"""
    kx, ky, ci, _ = dims
    out = []
    for y in range(ky):
        for x in range(kx):
            for c in range(ci):
                for o in range(f0, f0 + nf):
                    out.append(vals[x * strides[0] + y * strides[1] + c * strides[2] + o * strides[3]])
    return out


def pw_kn(vals, dims, strides):
    """(1,1,ci,co) -> [K=ci][N=co]"""
    _, _, ci, co = dims
    out = []
    for c in range(ci):
        for o in range(co):
            out.append(vals[c * strides[2] + o * strides[3]])
    return out


def ds_kn(vals, dims, strides):
    """(2,2,ci,co) -> [K=(ky*2+kx)*ci+c][N=co]"""
    kx, ky, ci, co = dims
    out = []
    for y in range(ky):
        for x in range(kx):
            for c in range(ci):
                for o in range(co):
                    out.append(vals[x * strides[0] + y * strides[1] + c * strides[2] + o * strides[3]])
    return out


def ct_kn(vals, dims, strides):
    """HWCN (kx,ky,oc,ic) -> [K=ic][N=4*oc], columns ordered [tap][oc]."""
    kx, ky, oc, ic = dims
    if (kx, ky) != (2, 2):
        raise ValueError("native CT2D requires a 2x2 kernel")
    out = []
    for c in range(ic):
        for t in range(4):
            px, py = t & 1, t >> 1
            for o in range(oc):
                out.append(vals[px * strides[0] + py * strides[1] + o * strides[2] + c * strides[3]])
    return out


# fsr4_shaders::GetScratchSize per output tier; the pass-13 model-parameter
# tensor (half4 per output pixel) is appended after it in the same buffer.
SCRATCH_SIZE = {"1080": 20880256, "2160": 83232256, "4320": 332352256}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--preset", required=True)
    ap.add_argument("--res", required=True, choices=["1080", "2160", "4320"])
    ap.add_argument("--outdir", required=True)
    a = ap.parse_args()
    pdir = os.path.join(a.root, "third_party/fsr4/internal/shaders/fsr4_model_v07_i8_" + a.preset)
    src = open(os.path.join(pdir, "passes_%s.hlsl" % a.res)).read()
    init = open(os.path.join(pdir, "initializers.bin"), "rb").read()
    blob = Blob()
    manifest = {"preset": a.preset, "res": a.res, "passes": {}}

    for n in range(1, 13):
        p = Pass(src, n, init)
        if p.op == "ConvNextBlock":
            if len(p.floats) != 4 or len(p.tens) != 8:
                sys.exit("pass %d: unexpected CNB signature" % n)
            inp, w0, b0, w1, b1, w2, b2, outp = p.tens
            ti, to = p.act(inp), p.act(outp)
            w0v, w0d, w0s, w0q = p.wdata(w0)
            w1v, w1d, w1s, w1q = p.wdata(w1)
            w2v, w2d, w2s, w2q = p.wdata(w2)
            if w0d != [3, 3, 16, 16] or w1d != [1, 1, 16, 32] or w2d != [1, 1, 32, 16]:
                sys.exit("pass %d: CNB shape %s" % (n, [w0d, w1d, w2d]))
            rcpr0, in1, rcpr1, in2 = p.floats
            e = dict(kind="cnb", kernel="i8_cnb", nf=16, ng=1, tile=[32, 4],
                     w0=blob.put_i8(conv_hwio(w0v, w0d, w0s, 0, 16)),
                     w1=blob.put_i8(pw_kn(w1v, w1d, w1s)),
                     w2=blob.put_i8(pw_kn(w2v, w2d, w2s)),
                     b0=blob.put_h(p.hdata(b0)), b1=blob.put_h(p.hdata(b1)), b2=blob.put_h(p.hdata(b2)),
                     s=[fmul(ti["qs"], w0q), rcpr0, fmul(in1, w1q), rcpr1,
                        fmul(w2q, in2), ti["qs"], frcp(to["qs"])])
        elif p.op == "FusedConv2D_k2s2b_QuantizedOutput":
            if len(p.floats) != 2 or len(p.tens) != 4:
                sys.exit("pass %d: unexpected DS signature" % n)
            inp, w, b, outp = p.tens
            ti, to = p.act(inp), p.act(outp)
            wv, wd, ws, wq = p.wdata(w)
            if wd[0] != 2 or wd[1] != 2 or wd[2] not in (16, 32) or wd[3] != 2 * wd[2]:
                sys.exit("pass %d: DS shape %s" % (n, wd))
            qf0, qf1 = p.floats
            e = dict(kind="ds", kernel="i8_ds%d" % wd[2], nf=wd[3], ng=1,
                     tile=[64, 1] if wd[2] == 16 else [32, 1],
                     w0=blob.put_i8(ds_kn(wv, wd, ws)), b0=blob.put_h(p.hdata(b)),
                     s=[wq, ti["qs"], frcp(qf0), frcp(qf1)])
        elif p.op == "FasterNetBlock":
            nf, ng = p.tmpl
            if len(p.tens) != 8:
                sys.exit("pass %d: unexpected FNB signature" % n)
            inp, w0, b0, w1, b1, w2, b2, outp = p.tens
            ti, to = p.act(inp), p.act(outp)
            w0v, w0d, w0s, w0q = p.wdata(w0)
            w1v, w1d, w1s, w1q = p.wdata(w1)
            w2v, w2d, w2s, w2q = p.wdata(w2)
            hf = nf // 2
            fpg = hf // ng
            if w0d != [3, 3, 16, hf] or w1d != [1, 1, nf, 2 * nf] or w2d != [1, 1, 2 * nf, nf]:
                sys.exit("pass %d: FNB shape %s" % (n, [w0d, w1d, w2d]))
            w0blk = [v for g in range(ng)
                     for v in conv_hwio(w0v, w0d, w0s, g * fpg, fpg)]
            if len(p.floats) == 5:
                qs0, qs1, act, qo0, qo1 = p.floats
                ro0, ro1 = frcp(qo0), frcp(qo1)
                kname = "i8_fnb%d_5f" % nf
            elif len(p.floats) == 3:
                qs0, qs1, act = p.floats
                ro0 = ro1 = frcp(to["qs"])
                kname = "i8_fnb%d_3f" % nf
            else:
                sys.exit("pass %d: FNB floats %d" % (n, len(p.floats)))
            # the Metal compiler folds every literal scale product of the
            # reference MSL (fast-math), so every stage is (acc*K + b)*R
            s = [fmul(qs0, w0q), 0.0, frcp(qs1), fmul(qs1, w1q), 0.0, frcp(act),
                 fmul(w2q, act), 0.0, qs0, qs1, ro0, ro1]
            tile = [32, 2] if nf == 32 else [32, 1]
            e = dict(kind="fnb", kernel=kname, nf=nf, ng=ng, tile=tile,
                     w0=blob.put_i8(w0blk), w1=blob.put_i8(pw_kn(w1v, w1d, w1s)),
                     w2=blob.put_i8(pw_kn(w2v, w2d, w2s)),
                     b0=blob.put_h(p.hdata(b0)), b1=blob.put_h(p.hdata(b1)), b2=blob.put_h(p.hdata(b2)),
                     s=s)
        elif p.op == "FNB_CT2D_ADD":
            nf, ng = p.tmpl
            if len(p.tens) != 11:
                sys.exit("pass %d: unexpected FNB_CT2D_ADD signature" % n)
            inp, addp, w0, b0, w1, b1, w2, b2, w3, b3, outp = p.tens
            ti, ta, to = p.act(inp), p.act(addp), p.act(outp)
            w0v, w0d, w0s, w0q = p.wdata(w0)
            w1v, w1d, w1s, w1q = p.wdata(w1)
            w2v, w2d, w2s, w2q = p.wdata(w2)
            w3v, w3d, w3s, w3q = p.wdata(w3)
            hf = nf // 2
            fpg = hf // ng
            nout = w3d[2]
            if w0d != [3, 3, 16, hf] or w1d != [1, 1, nf, 2 * nf] or w2d != [1, 1, 2 * nf, nf] \
                    or w3d[:2] != [2, 2] or w3d[3] != nf:
                sys.exit("pass %d: CT2D shape %s" % (n, [w0d, w1d, w2d, w3d]))
            w0blk = [v for g in range(ng)
                     for v in conv_hwio(w0v, w0d, w0s, g * fpg, fpg)]
            if len(p.floats) == 6:
                qs0, qs1, act, fo, qo0, qo1 = p.floats
                ro0, ro1 = frcp(qo0), frcp(qo1)
            elif len(p.floats) == 4:
                qs0, qs1, act, fo = p.floats
                ro0 = ro1 = frcp(to["qs"])
            else:
                sys.exit("pass %d: CT2D floats %d" % (n, len(p.floats)))
            # fnb_core slots: [c1a, c1b, R1, c2a, c2b, R2, c3a, c3b, IQ0, IQ1,
            #                  RF, KC, AQ, RO0, RO1]  (c1/c2/c3 all folded)
            s = [fmul(qs0, w0q), 0.0, frcp(qs1), fmul(qs1, w1q), 0.0, frcp(act),
                 fmul(w2q, act), 0.0, qs0, qs1, frcp(fo), fmul(fo, w3q), ta["qs"], ro0, ro1]
            # Entry-point names encode the new tap layout so an old manifest
            # cannot silently bind [tap][K][N] weights to the fused kernel.
            e = dict(kind="ct2d", kernel="i8_ct2d%d_tap4" % nf, nf=nf, ng=ng, nout=nout,
                     tile=[32, 1] if nf == 64 else [32, 2],
                     w0=blob.put_i8(w0blk), w1=blob.put_i8(pw_kn(w1v, w1d, w1s)),
                     w2=blob.put_i8(pw_kn(w2v, w2d, w2s)), w3=blob.put_i8(ct_kn(w3v, w3d, w3s)),
                     b0=blob.put_h(p.hdata(b0)), b1=blob.put_h(p.hdata(b1)),
                     b2=blob.put_h(p.hdata(b2)), b3=blob.put_h(p.hdata(b3)),
                     add=ta, s=s)
        else:
            sys.exit("pass %d: unknown op %s" % (n, p.op))
        e["in"] = ti
        e["out"] = to
        manifest["passes"][str(n)] = e

    # pass 13: the post pass's ConvNextBlock (same op as passes 1/2/12) plus its
    # 2x2 s2 ConvTranspose (16 -> 4 model parameters + 4 recurrent channels).
    # The kernel writes the recurrent texture itself and stores the model
    # parameters as half4 per output pixel right after the scratch tensors
    # (post_ext_ct2d_off); the patched post (FSR4_POST_EXT_CT2D) loads them
    # instead of recomputing CNB + CT2D.
    post = open(os.path.join(pdir, "post.hlsl")).read()
    post = select_resolution(post, a.res)
    pp = PostPass(post, init)
    if len(pp.floats) != 5 or len(pp.tens) < 10:
        sys.exit("post: unexpected CNB call (%d floats, %d tensors)" % (len(pp.floats), len(pp.tens)))
    inp, w0, b0, w1, b1, w2, b2, w3, b3 = pp.tens[:9]
    ti = pp.act(inp)
    w0v, w0d, w0s, w0q = pp.wdata(w0)
    w1v, w1d, w1s, w1q = pp.wdata(w1)
    w2v, w2d, w2s, w2q = pp.wdata(w2)
    w3v, w3d, w3s, w3q = pp.wdata(w3)
    if w0d != [3, 3, 16, 16] or w1d != [1, 1, 16, 32] or w2d != [1, 1, 32, 16]:
        sys.exit("post: CNB shape %s" % [w0d, w1d, w2d])
    if w3d != [2, 2, 8, 16]:
        sys.exit("post: CT2D shape %s" % w3d)
    rcpr0, in1, rcpr1, in2, cnbOut = pp.floats
    to = dict(ti)
    to["off"] = manifest["passes"]["1"]["out"]["off"]
    to["qs"] = cnbOut
    prm_off = SCRATCH_SIZE[a.res]
    manifest["passes"]["13"] = dict(
        kind="cnb_ct2d", kernel="i8_cnb_ct2d_tap4", nf=16, ng=1, tile=[32, 4],
        w0=blob.put_i8(conv_hwio(w0v, w0d, w0s, 0, 16)),
        w1=blob.put_i8(pw_kn(w1v, w1d, w1s)),
        w2=blob.put_i8(pw_kn(w2v, w2d, w2s)),
        w3=blob.put_i8(ct_kn(w3v, w3d, w3s)),
        b0=blob.put_h(pp.hdata(b0)), b1=blob.put_h(pp.hdata(b1)), b2=blob.put_h(pp.hdata(b2)),
        b3=blob.put_h(pp.hdata(b3)),
        s=[fmul(ti["qs"], w0q), rcpr0, fmul(in1, w1q), rcpr1, fmul(w2q, in2), ti["qs"], frcp(cnbOut),
           fmul(cnbOut, w3q)],
        prm={"off": prm_off},
        **{"in": ti, "out": to})
    manifest["post_ext_cnb_off"] = to["off"]
    manifest["post_ext_ct2d_off"] = prm_off

    os.makedirs(a.outdir, exist_ok=True)
    open(os.path.join(a.outdir, "i8mpp_weights.bin"), "wb").write(bytes(blob.b))
    open(os.path.join(a.outdir, "i8mpp_manifest.json"), "w").write(json.dumps(manifest, indent=1))
    print("gen_i8mpp: %s @%s -> %s (%d bytes of weights)" % (a.preset, a.res, a.outdir, len(blob.b)))


if __name__ == "__main__":
    main()
