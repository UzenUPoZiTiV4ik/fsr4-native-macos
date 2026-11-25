#!/usr/bin/env python3
"""Prepare a locally supplied ANR package for the independent Metal runtime.

Supports:
  - ANR FG-V.B20 addon (--addon <zip-or-dir>): dynamic resolution 640x360..7680x4320,
    native pq10/srgb8/scrgb16 encodings, SPIR-V -> Metal 3.1 single metallib (schema v2).
  - Legacy ANR FG-V.B14 macOS kit (--kit <dir> --converter-headers <dir>): fixed sizes (schema v1).

No game installation, binary patch, network access or model download. All weights
and compiled shaders stay in the local ignored build directory.
"""
import argparse
import hashlib
import importlib.util
import json
import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import re
import shutil
import struct
import subprocess
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("anr_nrx_prepare", Path(__file__).resolve().with_name("anr_nrx.py"))
nrx = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(nrx)
FINAL_TRUST_SPEC = importlib.util.spec_from_file_location("opt_final_trust", Path(__file__).with_name("opt_final_trust.py"))
opt_final_trust = importlib.util.module_from_spec(FINAL_TRUST_SPEC)
FINAL_TRUST_SPEC.loader.exec_module(opt_final_trust)
GATHER_SPEC = importlib.util.spec_from_file_location("opt_gather", Path(__file__).with_name("opt_gather.py"))
opt_gather = importlib.util.module_from_spec(GATHER_SPEC)
GATHER_SPEC.loader.exec_module(opt_gather)

OPT_SPEC = importlib.util.spec_from_file_location("anr_opt_msl", Path(__file__).with_name("opt_msl.py"))
opt_msl = importlib.util.module_from_spec(OPT_SPEC)
OPT_SPEC.loader.exec_module(opt_msl)

# Legacy FG-V.B14 macOS kit pins (schema v1).
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

# FG-V.B20 addon pins (schema v2, dynamic resolution 640x360..7680x4320).
V20_RELEASE = "FG-V.B20"
V20_KERNELS_HASH = "15b864a39495d8d9a0a92cb3799928055e6b9e5d1ad4136233f818570a2d2876"
V20_KERNELS_PORTABLE_HASH = "d41b9a87806d8529f18c898a83c4e4a83e720b3e1cafbece7daf06175fa622a2"
V20_CHECKPOINTS = {
    "packed": "b7481951e4f54e4f2d534ec41c8f3d8e721f4fd22839d2ba2bd236cbd28bea64",
    "dp4a": "bef1b3e76649887e6cb36f178a3011752d51b4e868f22e70f7c4a5a8d62408f4",
}
V20_PROGRAMS = {
    "packed": {
        "pq10": ("fg-lite-any-packed-pq10", "ee3fcc13a2862ed55fe2993c1d9e06cd4d13495f7a5ecf9d66b6ff071ed886db"),
        "srgb8": ("fg-lite-any-packed-srgb8", "8f80cadf7c46adab11d3523d31b5db69f78b32d0f42ab2ce701c6ec73580855f"),
        "scrgb16": ("fg-lite-any-packed-scrgb16", "471da1067a25e58e4cc97f5fd0fb4e45222ee4ce22db5fb86cdb7c5e36ac400a"),
    },
    "dp4a": {
        "pq10": ("fg-lite-any-dp4a-pq10", "64d3d6376456811b6bdd9774bfd7863af8b43ff4336b13a50c6fc03de916dc7e"),
        "srgb8": ("fg-lite-any-dp4a-srgb8", "2b9a3cd500246c40cf1a4604ab0f6c19d666c09ed59ebbec29ea9d8f3838aea7"),
        "scrgb16": ("fg-lite-any-dp4a-scrgb16", "ac13a9c8a7622e3d0c036b4e04b0fb3dcd4fde5aeca3baae0a9aa592ab21053e"),
    },
}
# Metal lacks native threadgroup atomic_float; rewrite those two 32x32 gather
# kernels from kernels/pack.nrx using a bit-exact threadgroup atomic_uint CAS helper
# while preserving the exact 32x32 / 512-thread workgroup geometry of the packed/dp4a graphs.
V20_TG_FLOAT_ATOMIC_KERNELS = {
    "fg.slang:phase_gather_sil_pd",
    "fg.slang:reverse_gather",
}
SCALAR_TYPE_CODES = {"uint32": 0, "int32": 1, "float32": 2}
SPV_REFLECT_CATEGORIES = ("ubos", "ssbos", "textures", "separate_images", "separate_samplers", "images")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read_pack(path, expected):
    if path.is_symlink() or not path.is_file() or path.stat().st_size > nrx.MAX_BYTES:
        raise ValueError("missing/invalid pack: " + str(path))
    data = path.read_bytes()
    if digest(data) != expected:
        raise ValueError("unsupported ANR pack identity: " + str(path))
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


def resolve_v20_dlssg_root(addon_path, scratch_dir):
    source = Path(addon_path).expanduser().resolve(strict=True)
    if source.is_file() and source.suffix.lower() == ".zip":
        extract = scratch_dir / "addon_unzipped"
        extract.mkdir()
        with zipfile.ZipFile(source) as archive:
            archive.extractall(extract)
        source = extract
    candidates = [
        source / "Apex Neural Rendering FG" / "dlssg",
        source / "dlssg",
        source,
    ]
    for candidate in candidates:
        if (candidate / "kernels" / "pack.nrx").is_file():
            return candidate
    raise ValueError("could not find ANR V.B20 dlssg/ directory inside: " + str(addon_path))


def rewrite_pushpull_tail_msl(msl_code):
    # Replace the two 2752-entry threadgroup arrays with a thin accessor backed by
    # device m_4/m_6 for level 0 (index < level0Count) and 1024-entry threadgroup
    # arrays for coarser levels (exact bit-for-bit semantics, 12 KB threadgroup RAM).
    helper = """
struct PushPullTailVec2 {
    device float2 *dev;
    threadgroup float2 *tg;
    uint level0;
    inline float2 read(uint idx) const { return idx < level0 ? dev[idx] : tg[idx - level0]; }
    inline void write(uint idx, float2 val) const { if (idx < level0) dev[idx] = val; else tg[idx - level0] = val; }
};
struct PushPullTailF32 {
    device float *dev;
    threadgroup float *tg;
    uint level0;
    inline float read(uint idx) const { return idx < level0 ? dev[idx] : tg[idx - level0]; }
    inline void write(uint idx, float val) const { if (idx < level0) dev[idx] = val; else tg[idx - level0] = val; }
};
"""
    # Insert helper right before the kernel declaration and rewrite _3 / _5 accesses cleanly.
    kpos = msl_code.index("kernel void ")
    head, body = msl_code[:kpos], msl_code[kpos:]
    body = body.replace(
        "    threadgroup spvUnsafeArray<float2, 2752> _3;\n    threadgroup spvUnsafeArray<float, 2752> _5;",
        "    threadgroup float2 _3_tg[1024];\n"
        "    threadgroup float _5_tg[1024];\n"
        "    const uint _level0_n = (*spvDescriptorSet0.m_8)._m2 * (*spvDescriptorSet0.m_8)._m3;\n"
        "    PushPullTailVec2 _3_acc{(*spvDescriptorSet0.m_4)._m0, _3_tg, _level0_n};\n"
        "    PushPullTailF32 _5_acc{(*spvDescriptorSet0.m_6)._m0, _5_tg, _level0_n};",
    )
    # Remove redundant level-0 load & store loops (m_4/m_6 already hold level 0 in place),
    # and rewrite _3[expr] / _5[expr] reads and assignments.
    body = re.sub(
        r"_3\[([^\]]+)\]\s*=\s*([^;]+);",
        r"_3_acc.write(\1, \2);",
        body,
    )
    body = re.sub(
        r"_5\[([^\]]+)\]\s*=\s*([^;]+);",
        r"_5_acc.write(\1, \2);",
        body,
    )
    body = re.sub(r"_3\[([^\]]+)\]", r"_3_acc.read(\1)", body)
    body = re.sub(r"_5\[([^\]]+)\]", r"_5_acc.read(\1)", body)
    # Device memory coherence across threads in the single workgroup for level 0:
    body = body.replace("threadgroup_barrier(mem_flags::mem_threadgroup);",
                        "threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);")
    return head + helper + body


# Native Metal 4 MPP (Apple10 / M5 NAX) replacement for conv_mf.slang:conv_packed_m1.
# Preserves the exact descriptor table ABI (buffer(0) ids 0..5), dispatch geometry
# ([ceil(W/32), ceil(H/8), B * ceil(Cout/16)], threads [64, 2, 1]), and epilogue flags.
# Accumulates in cooperative_tensor registers and aliases output staging over input/weight
# threadgroup memory (15.5 KB total -> 2x occupancy vs naive staging).
CONV_PACKED_M1_MPP_MSL = r"""
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

struct ConvParams {
    float _m0;
    uint inH;
    uint inW;
    uint outH;
    uint outW;
    uint Cin;
    uint Cout;
    uint K;
    uint stride;
    uint padH;
    uint padW;
    uint flags;
    uint cBlocks;
    uint batch;
    float lreluAlpha;
    uint _m15;
    float _m16;
};

struct spvDescriptorSetBuffer0 {
    constant ConvParams* m_12 [[id(0)]];
    const device half* m_11 [[id(1)]];
    const device half* m_5 [[id(2)]];
    const device float* m_8 [[id(3)]];
    const device half* m_6 [[id(4)]];
    device half* m_7 [[id(5)]];
};

inline float apply_epilogue(float v, float res, uint flags, float alpha) {
    const bool hasRes = (flags & 4u) != 0u;
    const bool preRes = (flags & 36u) == 36u;
    if (preRes) v += res;
    if ((flags & 2u) != 0u && v < 0.0f) v *= alpha;
    if ((flags & 8u) != 0u) {
        v = 0.5f * v * (1.0f + precise::tanh(fast::clamp(0.79788458347320556640625f * (v + 0.0447149984538555145263671875f * v * v * v), -15.0f, 15.0f)));
    }
    if ((flags & 16u) != 0u) v = fast::clamp(v, -6.0f, 6.0f);
    if (hasRes && !preRes) v += res;
    return v;
}

kernel void __ENTRY_NAME__(
    constant spvDescriptorSetBuffer0& ds [[buffer(0)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]],
    uint lane_id [[thread_index_in_simdgroup]])
{
    const ConvParams p = *ds.m_12;
    const uint batchIdx = gid.z / p.cBlocks;
    const uint ocBase = (gid.z % p.cBlocks) * 16u;
    const uint oyBase = gid.y * 8u;
    const uint oxBase = gid.x * 32u;
    const uint tid = lid.y * 64u + lid.x;
    const uint tileH = 7u + p.K;
    const uint tileW = 31u + p.K;
    const uint totalPix = tileH * tileW;
    const int iy0 = int(oyBase) - int(p.padH);
    const int ix0 = int(oxBase) - int(p.padW);
    const bool inLrelu = (p.flags & 1u) != 0u;
    const half hAlpha = half(p.lreluAlpha);
    const uint taps = p.K * p.K;

    threadgroup union {
        struct {
            half in[340 * 16];
            half w[9 * 16 * 16];
        } stage;
        float out[4 * 32 * 16];
    } tg;

    const device half *inBase = ds.m_11 + size_t(batchIdx) * p.inH * p.inW * p.Cin;
    constexpr auto descMul = matmul2d_descriptor(32, 16, 16, false, false, false, matmul2d_descriptor::mode::multiply);
    constexpr auto descMac = matmul2d_descriptor(32, 16, 16, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<descMul, execution_simdgroup> mmFirst;
    matmul2d<descMac, execution_simdgroup> mmAcc;

    using InTensor = tensor<threadgroup half, extents<int32_t, 16, 32>, tensor_inline>;
    using WTensor  = tensor<threadgroup half, extents<int32_t, 16, 16>, tensor_inline>;
    auto acc0 = mmAcc.template get_destination_cooperative_tensor<InTensor, WTensor, float>();
    auto acc1 = mmAcc.template get_destination_cooperative_tensor<InTensor, WTensor, float>();
    bool firstTap = true;

    for (uint icBase = 0; icBase < p.Cin; icBase += 16u) {
        for (uint idx = tid; idx < totalPix * 2u; idx += 128u) {
            const uint pix = idx >> 1u;
            const uint half8 = idx & 1u;
            const uint ty = pix / tileW;
            const uint tx = pix % tileW;
            const int iy = iy0 + int(ty);
            const int ix = ix0 + int(tx);
            half4 v0 = half4(0.0h), v1 = half4(0.0h);
            if (iy >= 0 && iy < int(p.inH) && ix >= 0 && ix < int(p.inW)) {
                const device half *src = inBase + ((size_t(iy) * p.inW + uint(ix)) * p.Cin + icBase + half8 * 8u);
                v0 = *(const device half4 *)(src);
                v1 = *(const device half4 *)(src + 4);
                if (inLrelu) {
                    v0 = select(v0, v0 * hAlpha, v0 < half4(0.0h));
                    v1 = select(v1, v1 * hAlpha, v1 < half4(0.0h));
                }
            }
            threadgroup half *dst = tg.stage.in + pix * 16u + half8 * 8u;
            *(threadgroup half4 *)(dst) = v0;
            *(threadgroup half4 *)(dst + 4) = v1;
        }
        for (uint idx = tid; idx < taps * 32u; idx += 128u) {
            const uint row16 = idx >> 1u;
            const uint half8 = (idx & 1u) * 8u;
            const uint tap = row16 >> 4u;
            const uint ic = row16 & 15u;
            half4 w0 = half4(0.0h), w1 = half4(0.0h);
            if (ocBase + half8 < p.Cout) {
                const device half *wsrc = ds.m_5 + ((tap * p.Cin + icBase + ic) * p.Cout + ocBase + half8);
                w0 = *(const device half4 *)(wsrc);
                w1 = *(const device half4 *)(wsrc + 4);
            }
            threadgroup half *wdst = tg.stage.w + row16 * 16u + half8;
            *(threadgroup half4 *)(wdst) = w0;
            *(threadgroup half4 *)(wdst + 4) = w1;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const uint r0 = simd_id * 2u;
        const uint r1 = r0 + 1u;
        for (uint ky = 0; ky < p.K; ++ky) {
            for (uint kx = 0; kx < p.K; ++kx) {
                const uint tap = ky * p.K + kx;
                WTensor wT(tg.stage.w + tap * 256u, extents<int32_t, 16, 16>());
                InTensor inT0(tg.stage.in + ((r0 + ky) * tileW + kx) * 16u, extents<int32_t, 16, 32>());
                InTensor inT1(tg.stage.in + ((r1 + ky) * tileW + kx) * 16u, extents<int32_t, 16, 32>());
                if (firstTap) {
                    mmFirst.run(inT0, wT, acc0);
                    mmFirst.run(inT1, wT, acc1);
                    firstTap = false;
                } else {
                    mmAcc.run(inT0, wT, acc0);
                    mmAcc.run(inT1, wT, acc1);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    threadgroup float *myOut = tg.out + simd_id * (32u * 16u);
    auto outT = tensor<threadgroup float, extents<int32_t, 16, 32>, tensor_inline>(myOut, extents<int32_t, 16, 32>());
    const device half *resBase = ds.m_6 + size_t(batchIdx) * p.outH * p.outW * p.Cout;
    device half *outBase = ds.m_7 + size_t(batchIdx) * p.outH * p.outW * p.Cout;
    const bool hasRes = (p.flags & 4u) != 0u;
    const float4 b0 = *(const device float4 *)(ds.m_8 + ocBase);
    const float4 b1 = *(const device float4 *)(ds.m_8 + ocBase + 4);
    const float4 b2 = *(const device float4 *)(ds.m_8 + ocBase + 8);
    const float4 b3 = *(const device float4 *)(ds.m_8 + ocBase + 12);

    for (uint rr = 0; rr < 2u; ++rr) {
        const uint oy = oyBase + simd_id * 2u + rr;
        if (oy >= p.outH) continue;
        if (rr == 0) acc0.store(outT);
        else acc1.store(outT);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        const uint ox = oxBase + lane_id;
        if (ox < p.outW) {
            const size_t off = (size_t(oy) * p.outW + ox) * p.Cout + ocBase;
            const threadgroup float *src = myOut + lane_id * 16u;
            const float4 bs[4] = {b0, b1, b2, b3};
            for (uint ci = 0; ci < 4u; ++ci) {
                const uint c = ci * 4u;
                if (ocBase + c >= p.Cout) break;
                float4 val = *(const threadgroup float4 *)(src + c) + bs[ci];
                float4 rv = hasRes ? float4(*(const device half4 *)(resBase + off + c)) : float4(0.0f);
                half4 out4;
                out4.x = half(apply_epilogue(val.x, rv.x, p.flags, p.lreluAlpha));
                out4.y = half(apply_epilogue(val.y, rv.y, p.flags, p.lreluAlpha));
                out4.z = half(apply_epilogue(val.z, rv.z, p.flags, p.lreluAlpha));
                out4.w = half(apply_epilogue(val.w, rv.w, p.flags, p.lreluAlpha));
                *(device half4 *)(outBase + off + c) = out4;
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
}
"""


def strip_optnone_msl(msl_code):
    # spirv-cross translates SPIR-V NoContraction into [[clang::optnone]] helper calls
    # (spvFMul/spvFAdd/spvFSub), blocking inlining, FMA fusion, and register scheduling
    # on Apple GPUs. Inlining them cuts occl_flags by ~49% and trust_half by ~22% with
    # identical PSNR.
    msl_code = re.sub(
        r"template<typename T>\s*\[\[clang::optnone\]\]\s*T spvFMul\(T l, T r\)\s*\{\s*return fma\(l, r, T\(0\)\);\s*\}",
        "template<typename T>\ninline T spvFMul(T l, T r) { return l * r; }",
        msl_code,
    )
    msl_code = re.sub(
        r"template<typename T>\s*\[\[clang::optnone\]\]\s*T spvFAdd\(T l, T r\)\s*\{\s*return fma\(T\(1\), l, r\);\s*\}",
        "template<typename T>\ninline T spvFAdd(T l, T r) { return l + r; }",
        msl_code,
    )
    msl_code = re.sub(
        r"template<typename T>\s*\[\[clang::optnone\]\]\s*T spvFSub\(T l, T r\)\s*\{\s*return fma\(T\(-1\), r, l\);\s*\}",
        "template<typename T>\ninline T spvFSub(T l, T r) { return l - r; }",
        msl_code,
    )
    return msl_code.replace("[[clang::optnone]]", "inline")


def rewrite_tg_float_atomics_msl(msl_code):
    # All atomic_store/atomic_load on threadgroup float in the V.B20 gather kernels are
    # separated from atomic_fetch_add by threadgroup_barrier, so plain accesses are exact
    # and avoid atomic instruction overhead. Actual float additions use a lean CAS helper.
    helper = """
inline void tg_atomic_add_f32_void(threadgroup float *ptr, float delta) {
    if (delta == 0.0f) return;
    threadgroup atomic_uint *ap = (threadgroup atomic_uint *)ptr;
    uint expected = atomic_load_explicit(ap, memory_order_relaxed);
    for (;;) {
        uint desired = as_type<uint>(as_type<float>(expected) + delta);
        if (atomic_compare_exchange_weak_explicit(ap, &expected, desired, memory_order_relaxed, memory_order_relaxed)) {
            return;
        }
    }
}
inline float tg_atomic_add_f32(threadgroup float *ptr, float delta) {
    threadgroup atomic_uint *ap = (threadgroup atomic_uint *)ptr;
    uint expected = atomic_load_explicit(ap, memory_order_relaxed);
    if (delta == 0.0f) return as_type<float>(expected);
    for (;;) {
        uint desired = as_type<uint>(as_type<float>(expected) + delta);
        if (atomic_compare_exchange_weak_explicit(ap, &expected, desired, memory_order_relaxed, memory_order_relaxed)) {
            return as_type<float>(expected);
        }
    }
}
"""
    kpos = msl_code.index("kernel void ")
    head, body = msl_code[:kpos], msl_code[kpos:]
    body = re.sub(
        r"atomic_store_explicit\(\(threadgroup atomic_float\*\)&([^,]+),\s*([^,]+),\s*memory_order_relaxed\)",
        r"(\1 = \2)",
        body,
    )
    body = re.sub(
        r"atomic_load_explicit\(\(threadgroup atomic_float\*\)&([^,]+),\s*memory_order_relaxed\)",
        r"(\1)",
        body,
    )
    body = re.sub(
        r"float _\d+\s*=\s*atomic_fetch_add_explicit\(\(threadgroup atomic_float\*\)(&[^,]+),\s*([^,]+),\s*memory_order_relaxed\);",
        r"tg_atomic_add_f32_void(\1, \2);",
        body,
    )
    body = re.sub(
        r"atomic_fetch_add_explicit\(\(threadgroup atomic_float\*\)(&[^,]+),\s*([^,]+),\s*memory_order_relaxed\)",
        r"tg_atomic_add_f32(\1, \2)",
        body,
    )
    return head + helper + body


def translate_spv_kernel(spv_bytes, slang_reflection, entry_name, work_dir, kernel_key=None):
    spv_path = work_dir / (entry_name + ".spv")
    msl_path = work_dir / (entry_name + ".metal")
    air_path = work_dir / (entry_name + ".air")
    spv_path.write_bytes(spv_bytes)

    reflect_out = subprocess.check_output(["spirv-cross", str(spv_path), "--reflect"], text=True, timeout=60)
    spv_refl = json.loads(reflect_out)

    subprocess.run(
        [
            "spirv-cross",
            str(spv_path),
            "--msl",
            "--msl-version",
            "30100",
            "--msl-argument-buffers",
            "--msl-argument-buffer-tier",
            "1",
            "--msl-fixed-subgroup-size",
            "32",
            "--msl-no-readwrite-texture-fences",
            "--rename-entry-point",
            "main",
            entry_name,
            "comp",
            "--output",
            str(msl_path),
        ],
        check=True,
        timeout=60,
    )

    msl_code = strip_optnone_msl(msl_path.read_text())
    # Metal limits threadgroup memory to 32768 bytes. pushpull_tail allocates
    # float2[2752] + float[2752] = 33024 bytes when staging level 0 + coarse levels.
    # Keep level 0 in-place in the L1/L2-cached device buffers (m_4/m_6) and stage
    # only coarse levels (>= level0Count, <= 1024 entries = 12 KB) in threadgroup memory.
    if "threadgroup spvUnsafeArray<float2, 2752>" in msl_code and "threadgroup spvUnsafeArray<float, 2752>" in msl_code:
        msl_code = rewrite_pushpull_tail_msl(msl_code)
    if "(threadgroup atomic_float*)" in msl_code:
        msl_code = rewrite_tg_float_atomics_msl(msl_code)
    msl_code = opt_final_trust.optimize_final_trust_msl(kernel_key, msl_code)
    msl_code = opt_gather.optimize_gather_msl(kernel_key, msl_code)
    msl_code = opt_msl.optimize(msl_code, slang_reflection)
    msl_path.write_text(msl_code)

    subprocess.run(
        ["xcrun", "metal", "-std=metal3.1", "-O3", "-Wno-sometimes-uninitialized",
         "-c", str(msl_path), "-o", str(air_path)],
        check=True,
        timeout=60,
    )
    sig_match = re.search(r"kernel void " + re.escape(entry_name) + r"\([^{]*\)", msl_code, re.S)
    if not sig_match:
        raise ValueError("missing translated entry point " + entry_name)
    buf_slots = re.findall(r"\[\[buffer\((\d+)\)\]\]", sig_match.group(0))
    if buf_slots != ["0"]:
        raise ValueError("unexpected extra Metal buffer bindings in " + entry_name + ": " + str(buf_slots))

    spv_by_var = {}
    for category in SPV_REFLECT_CATEGORIES:
        for item in spv_refl.get(category, []):
            var_id = item["name"].split("_")[-1]
            spv_by_var[var_id] = (category, item["set"], item["binding"])

    slang_slots = {}
    uniform_fields = {}
    uniform_size = 0
    for param in slang_reflection["parameters"]:
        binding = param["binding"]
        kind = binding["kind"]
        if kind == "descriptorTableSlot":
            slang_slots[binding["index"]] = param["name"]
        elif kind == "uniform":
            scalar = param["type"].get("scalarType")
            if binding["size"] != 4 or scalar not in SCALAR_TYPE_CODES:
                raise ValueError("unsupported V.B20 uniform field: " + param["name"])
            uniform_fields[param["name"]] = (binding["offset"], SCALAR_TYPE_CODES[scalar])
            uniform_size = max(uniform_size, binding["offset"] + binding["size"])

    if uniform_size > 4096:
        raise ValueError("oversized V.B20 uniform block in " + entry_name)

    struct_match = re.search(r"struct spvDescriptorSetBuffer0\s*\{(.*?)\};", msl_code, re.S)
    if not struct_match:
        raise ValueError("missing spvDescriptorSetBuffer0 in " + entry_name)

    bindings = []
    for raw_line in struct_match.group(1).strip().splitlines():
        line = raw_line.strip()
        if not line:
            continue
        field_match = re.search(r"m_(\d+)\s+\[\[id\((\d+)\)\]\]", line)
        if not field_match:
            raise ValueError("unparsed descriptor field in " + entry_name + ": " + line)
        var_id = field_match.group(1)
        slot_id = int(field_match.group(2))
        if slot_id != len(bindings):
            raise ValueError("non-dense descriptor slot in " + entry_name)
        category, desc_set, desc_binding = spv_by_var[var_id]
        if desc_set != 0:
            raise ValueError("non-zero descriptor set in " + entry_name)

        if category == "ubos" and desc_binding == 0:
            bindings.append({"slot": slot_id, "kind": "uniform", "name": "globalParams", "r32uint": False, "writable": False})
        else:
            if desc_binding not in slang_slots:
                raise ValueError("unmapped SPIR-V binding " + str(desc_binding) + " in " + entry_name)
            slang_name = slang_slots[desc_binding]
            if category == "ssbos" or category == "ubos":
                bind_kind = "buffer"
            elif category == "separate_samplers":
                bind_kind = "sampler"
            elif category == "images":
                bind_kind = "storage_texture"
            else:
                bind_kind = "texture"
            is_r32uint = "texture2d<uint" in line
            is_readonly = (
                "const device" in line
                or "constant " in line
                or "sampler " in line
                or ("texture2d<" in line and "access::write" not in line and "access::read_write" not in line)
            )
            bindings.append({
                "slot": slot_id,
                "kind": bind_kind,
                "name": slang_name,
                "r32uint": is_r32uint,
                "writable": not is_readonly,
            })

    return air_path, {
        "entry": entry_name,
        "slot_count": len(bindings),
        "uniform_bytes": uniform_size,
        "bindings": bindings,
    }, uniform_fields


def pack_v20_program(program_dict, kernel_uniform_fields):
    packed_variants = []
    for variant in program_dict["variants"]:
        stages_out = {}
        for stage_name, stage_list in variant["stages"].items():
            # Keep pair and all texture-output (:image) phases (2x, 3x, 4x, 5x, 6x).
            if stage_name != "pair" and not stage_name.endswith(":image"):
                continue
            packed_stages = []
            for stage in stage_list:
                if stage["op"] != "dispatch":
                    raise ValueError("unsupported V.B20 graph op: " + str(stage["op"]))
                ufields = kernel_uniform_fields[stage["kernel"]]
                uniform_items = []
                for uname, uval in stage["uniforms"].items():
                    if uname not in ufields:
                        continue
                    offset, type_code = ufields[uname]
                    if isinstance(uval, dict) and list(uval.keys()) == ["x"]:
                        uniform_items.append([offset, type_code, {"x": uval["x"]}])
                    else:
                        uniform_items.append([offset, type_code, uval])
                uniform_items.sort(key=lambda item: item[0])
                packed_stages.append({
                    "label": stage["label"],
                    "kernel": stage["kernel"],
                    "groups": stage["groups"],
                    "threads": stage["threads"],
                    "buffers": stage["buffers"],
                    "images": stage.get("images", {}),
                    "uniforms": uniform_items,
                })
            stages_out[stage_name] = packed_stages
        packed_variants.append({
            "guards": variant["guards"],
            "exprs": variant["exprs"],
            "buffers": variant["buffers"],
            "inputs": variant["inputs"],
            "output": variant["output"],
            "swaps": variant["swaps"],
            "textures": variant.get("textures") or {},
            "texture_swaps": variant.get("texture_swaps") or [],
            "stages": stages_out,
        })
    return {
        "version": 3,
        "family": program_dict["family"],
        "checkpoint_id": program_dict["checkpoint_id"],
        "path": program_dict["path"],
        "phases": program_dict["phases"],
        "sizes": program_dict["sizes"],
        "variants": packed_variants,
    }


def prepare_v20(addon, output, backend="packed"):
    if not shutil.which("spirv-cross"):
        raise ValueError("spirv-cross is required to prepare ANR V.B20 Metal shaders")
    if backend not in V20_PROGRAMS:
        raise ValueError("unknown V.B20 backend: " + backend)
    output = Path(output).expanduser().absolute()
    if output.exists() or output.is_symlink():
        raise ValueError("output already exists; choose a new local directory")
    output.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix=".anr-prepare-v20-", dir=output.parent) as temporary:
        temp = Path(temporary)
        dlssg = resolve_v20_dlssg_root(addon, temp)
        kernels_main = read_pack(dlssg / "kernels" / "pack.nrx", V20_KERNELS_HASH)
        kernels_portable = read_pack(dlssg / "kernels-portable" / "pack.nrx", V20_KERNELS_PORTABLE_HASH)

        raw_programs = {}
        raw_weights = {}
        all_kernels = set()
        expected_ckpt = V20_CHECKPOINTS[backend]

        for enc_name, (prog_dir, expected_hash) in V20_PROGRAMS[backend].items():
            pack_files = read_pack(dlssg / "programs" / prog_dir / "pack.nrx", expected_hash)
            prog = json.loads(pack_files["program.json"])
            if prog.get("version") != 3:
                raise ValueError("expected V.B20 dynamic graph version 3 in " + prog_dir)
            if prog.get("checkpoint_id") != expected_ckpt:
                raise ValueError("unexpected V.B20 checkpoint_id in " + prog_dir)
            raw_programs[enc_name] = prog
            raw_weights[enc_name] = pack_files["weights.bin"]
            for variant in prog["variants"]:
                for stage_name, stage_list in variant["stages"].items():
                    if stage_name != "pair" and not stage_name.endswith(":image"):
                        continue
                    for stage in stage_list:
                        all_kernels.add(stage["kernel"])

        build = temp / "runtime"
        build.mkdir()
        compile_dir = temp / "kernels"
        compile_dir.mkdir()

        sorted_kernels = sorted(all_kernels)

        def build_one_kernel(item):
            index, kernel_key = item
            module, entry = kernel_key.split(":")
            prefix = module.removesuffix(".slang") + "/" + module + "." + entry
            slang_refl = json.loads(kernels_main[prefix + ".json"])
            spv_bytes = kernels_main[prefix + ".spv"]
            air_path, meta, ufields = translate_spv_kernel(spv_bytes, slang_refl, "k" + str(index), compile_dir, kernel_key)
            return kernel_key, str(air_path), meta, ufields

        with ThreadPoolExecutor(max_workers=8) as pool:
            built = list(pool.map(build_one_kernel, enumerate(sorted_kernels)))

        air_paths = []
        kernel_meta = {}
        kernel_uniform_fields = {}
        for kernel_key, air_str, meta, ufields in built:
            air_paths.append(air_str)
            kernel_meta[kernel_key] = meta
            kernel_uniform_fields[kernel_key] = ufields

        metallib_path = build / "kernels.metallib"
        subprocess.run(["xcrun", "metallib", *air_paths, "-o", str(metallib_path)], check=True, timeout=60)

        # Optional Metal 4 MPP NAX kernel library for Apple10 / M5+ GPUs (falls back cleanly to
        # kernels.metallib on M1-M4 Macs or if the Metal 4 toolchain is unavailable).
        mpp_kernels = {}
        metallib_mpp_name = None
        conv_key = "conv_mf.slang:conv_packed_m1"
        if backend == "packed" and conv_key in kernel_meta:
            mpp_entry = kernel_meta[conv_key]["entry"] + "_mpp"
            mpp_src = compile_dir / (mpp_entry + ".metal")
            mpp_air = compile_dir / (mpp_entry + ".air")
            mpp_src.write_text(CONV_PACKED_M1_MPP_MSL.replace("__ENTRY_NAME__", mpp_entry))
            mpp_compile = subprocess.run(
                ["xcrun", "metal", "-std=metal4.0", "-O2", "-c", str(mpp_src), "-o", str(mpp_air)],
                capture_output=True,
                text=True,
                timeout=60,
            )
            if mpp_compile.returncode == 0:
                metallib_mpp_name = "kernels_mpp.metallib"
                subprocess.run(
                    ["xcrun", "metallib", str(mpp_air), "-o", str(build / metallib_mpp_name)],
                    check=True,
                    timeout=60,
                )
                mpp_kernels[conv_key] = mpp_entry

        encodings_meta = {}
        for enc_name in ("pq10", "srgb8", "scrgb16"):
            weights_file = "weights_" + enc_name + ".bin"
            program_file = "program_" + enc_name + ".json"
            (build / weights_file).write_bytes(raw_weights[enc_name])
            packed_prog = pack_v20_program(raw_programs[enc_name], kernel_uniform_fields)
            (build / program_file).write_text(json.dumps(packed_prog, separators=(",", ":")) + "\n")
            encodings_meta[enc_name] = {
                "program": program_file,
                "weights": weights_file,
                "weights_sha256": digest(raw_weights[enc_name]),
                "weights_bytes": len(raw_weights[enc_name]),
            }

        manifest = {
            "schema": "fsr4_anr_runtime_v2",
            "release": V20_RELEASE,
            "backend": backend,
            "checkpoint_id": expected_ckpt,
            "family": raw_programs["pq10"]["family"],
            "sizes": raw_programs["pq10"]["sizes"],
            "phases": raw_programs["pq10"]["phases"],
            "metallib": "kernels.metallib",
            "source_kernels_sha256": V20_KERNELS_HASH,
            "source_kernels_portable_sha256": V20_KERNELS_PORTABLE_HASH,
            "encodings": encodings_meta,
            "kernels": kernel_meta,
            "runtime_executed": False,
        }
        if metallib_mpp_name:
            manifest["metallib_mpp"] = metallib_mpp_name
            manifest["mpp_kernels"] = mpp_kernels
        manifest["files"] = {
            p.relative_to(build).as_posix(): {"bytes": p.stat().st_size, "sha256": digest(p.read_bytes())}
            for p in sorted(build.rglob("*"))
            if p.is_file()
        }
        (build / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        build.rename(output)

    return {
        "schema": "fsr4_anr_runtime_v2",
        "release": V20_RELEASE,
        "backend": backend,
        "output": str(output),
        "kernels": len(kernel_meta),
        "mpp_kernels": len(mpp_kernels),
        "encodings": list(encodings_meta.keys()),
        "metallib_bytes": (output / "kernels.metallib").stat().st_size,
        "inference_executed": False,
    }


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
    parser.add_argument("--addon", type=Path, help="ANR FG-V.B20 addon zip or extracted directory (recommended, any resolution)")
    parser.add_argument("--kit", type=Path, help="legacy ANR FG-V.B14 macOS test kit directory")
    parser.add_argument("--converter-headers", type=Path, help="Apple converter include directory (legacy --kit only)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", choices=tuple(PROGRAM_PACKS), default="1080p", help="legacy --kit fixed resolution")
    parser.add_argument("--backend", choices=("native", "packed", "dp4a"), default="packed",
                        help="packed (FP16, default for V.B20), dp4a (INT8, V.B20), or native (legacy V.B14)")
    args = parser.parse_args()
    try:
        if args.addon:
            backend = "packed" if args.backend == "native" else args.backend
            print(json.dumps(prepare_v20(args.addon, args.output, backend), indent=2))
        elif args.kit and args.converter_headers:
            print(json.dumps(prepare(args.kit, args.converter_headers, args.output, args.size, args.backend), indent=2))
        else:
            parser.error("supply either --addon (V.B20) or both --kit and --converter-headers (legacy V.B14)")
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as error:
        parser.exit(1, "ANR prepare: " + str(error) + "\n")


if __name__ == "__main__":
    main()
