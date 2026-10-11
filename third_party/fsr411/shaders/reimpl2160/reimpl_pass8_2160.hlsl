// SPDX-License-Identifier: MIT
// Reimplemented FSR 4.1.1 kernel: semantics reconstructed from the AMD 4.1.1
// provider (Kits/FidelityFX/signedbin/amd_fidelityfx_upscaler_dx12.dll,
// SHA-256 d0dcccc74a43c44ba435b7a369b456e0970d8a4464e4bd683119b374f2c9fb46,
// MIT-licensed per FidelityFX-SDK docs/license.md) and validated byte-exact
// against the AMD container. See README.md and docs/evidence411/.
// Copyright (C) 2025 Advanced Micro Devices, Inc. (upstream implementation)
// Copyright (C) 2026 int3rrobang (reimplementation)

// ============================================================================
// 4.1.1 logical pass 8 reimplementation, 2160 class (our_reimplementation)
// Global 5117-equivalent (FFX_FSR4_PASS_16, nocbv 26044, dispatch 16,540,1).
// 2x spatial scaling of bench/shaders/candidates/pass8_411/reimpl_pass8.hlsl
// (1080 class: 240x135 threads, dispatch 8,270,1). Transform ONLY.
//
// Banked 4-slice INT8 block, t18-resident weights (no DXIL-embedded
// literals), 480x270 threads. Same skeleton as pass7-2160, different t18
// bases/planes/requant (see delta note in bench/results/411_5117_pass8_our_reimpl.json).
// Geometry law verified vs AMD 2160 shipping container 331098ae093a02fa
// (analysis/extracted/4.1.1-official/0320_0092fc80_331098ae093a02fa.dxbc,
// entry fsr4_model_v07_fp8_no_scale_pass8) by dxc -dumpbin diff vs the 1080
// container fccef09cae8bbee8: ONLY guard/stride/base immediates differ.
//
// geometry: W=480 H=270, ROW=7712 (=480*16+32), BANK=2097664 (=7712*272),
//   4 banks of 272 rows; input plane IN=58349248, output plane OUT=49958592.
// stage0: 3x3 conv 16ch->32ch (o in 0..7, bank=o>>2), requant (>>9,+255),
//   no ReLU, no edge masking. W0[o][t][j][k] @54912+o*64+t*512+j*16+k*4,
//   bias0 @59520+o*16.
// passthrough: act0[8..15] = raw input dwords, banks 2,3.
// stage1: pw 64->128 + ReLU (o2 in 0..31), requant (>>6,+31).
//   W1[o2][g][j][k] @59648+o2*256+j*64+g*16+k*4, bias1 @67840+o2*16.
// residual refill: banks 0,1 input pixels at (ty+1, tx+1).
// stage2+tail: 16 iters (o3, bank=o3>>2):
//   W2[o3][g][j][k] @68352+o3*512+j*128+g*16+k*4, bias2 @76544+o3*16.
//   tail: out = s8clamp(i16wrap(round_ne((s8(resid)*rscale + s8(acc)*2^-12)
//     * 16))), rscale = 2^-5 (o3<8) / 2^-4 (o3>=8).
// output: OUT + lane + BANK*(o3>>2) + ROW*(ty+1) + 16*(tx+1), lane=(o3*4)&12.
// ============================================================================
ByteAddressBuffer t18_initializer : register(t18);
RWByteAddressBuffer ScratchBuffer : register(u11);

static const uint IN_PLANE = 58349248u;
static const uint OUT_PLANE = 49958592u;
static const uint ROW = 7712u;
static const uint BANK = 2097664u;
static const uint W0_BASE = 54912u;
static const uint B0_BASE = 59520u;
static const uint W1_BASE = 59648u;
static const uint B1_BASE = 67840u;
static const uint W2_BASE = 68352u;
static const uint B2_BASE = 76544u;

int requant_shift(int x, uint shift, int add)
{
    uint bit = (uint(x) >> shift) & 1u;
    return (x + add + int(bit)) >> int(shift);
}

uint pack_bytes(int4 q)
{
    // SM 6.6 intrinsic -> DXIL Pack4x8 packMode 2 (SClamp). (Do NOT name this
    // helper pack_s8: it shadows the intrinsic and dxc emits truncating packs.)
    return (uint)pack_clamp_s8(q);
}

int int16_wrap(int v)
{
    return ((v & 0xFFFF) ^ 0x8000) - 0x8000;
}

[numthreads(64, 1, 1)]
void reimpl_pass8_2160(uint3 tid : SV_DispatchThreadID)
{
    const uint tx = tid.x;
    const uint ty = tid.y;
    if (tx > 479u || ty > 269u)
        return;

    uint act0[16];
    uint act1[32];
    uint res[8];

    // ---- stage 0: 3x3 conv 16 -> 32 ----
    [unroll]
    for (int o = 0; o < 8; o++) {
        const uint bank = (uint)o >> 2u;
        const uint4 b = t18_initializer.Load4(B0_BASE + (uint)o * 16u);
        int acc0 = asint(b.x);
        int acc1 = asint(b.y);
        int acc2 = asint(b.z);
        int acc3 = asint(b.w);
        [unroll]
        for (int t = 0; t < 9; t++) {
            const uint dy = (uint)t / 3u;
            const uint dx = (uint)t % 3u;
            const uint4 pix = ScratchBuffer.Load4(
                IN_PLANE + bank * BANK + (ty + dy) * ROW + (tx + dx) * 16u);
            const uint wbase = W0_BASE + (uint)o * 64u + (uint)t * 512u;
            acc0 = dot4add_i8packed(pix.x, t18_initializer.Load(wbase + 0u), acc0);
            acc0 = dot4add_i8packed(pix.y, t18_initializer.Load(wbase + 4u), acc0);
            acc0 = dot4add_i8packed(pix.z, t18_initializer.Load(wbase + 8u), acc0);
            acc0 = dot4add_i8packed(pix.w, t18_initializer.Load(wbase + 12u), acc0);
            acc1 = dot4add_i8packed(pix.x, t18_initializer.Load(wbase + 16u), acc1);
            acc1 = dot4add_i8packed(pix.y, t18_initializer.Load(wbase + 20u), acc1);
            acc1 = dot4add_i8packed(pix.z, t18_initializer.Load(wbase + 24u), acc1);
            acc1 = dot4add_i8packed(pix.w, t18_initializer.Load(wbase + 28u), acc1);
            acc2 = dot4add_i8packed(pix.x, t18_initializer.Load(wbase + 32u), acc2);
            acc2 = dot4add_i8packed(pix.y, t18_initializer.Load(wbase + 36u), acc2);
            acc2 = dot4add_i8packed(pix.z, t18_initializer.Load(wbase + 40u), acc2);
            acc2 = dot4add_i8packed(pix.w, t18_initializer.Load(wbase + 44u), acc2);
            acc3 = dot4add_i8packed(pix.x, t18_initializer.Load(wbase + 48u), acc3);
            acc3 = dot4add_i8packed(pix.y, t18_initializer.Load(wbase + 52u), acc3);
            acc3 = dot4add_i8packed(pix.z, t18_initializer.Load(wbase + 56u), acc3);
            acc3 = dot4add_i8packed(pix.w, t18_initializer.Load(wbase + 60u), acc3);
        }
        int4 q = int4(requant_shift(acc0, 9u, 255),
                      requant_shift(acc1, 9u, 255),
                      requant_shift(acc2, 9u, 255),
                      requant_shift(acc3, 9u, 255));
        act0[o] = pack_bytes(q);
    }

    // ---- passthrough: raw input dwords, banks 2,3 ----
    [unroll]
    for (int i = 8; i < 16; i++) {
        const uint bank = (uint)i >> 2u;
        const uint lane = ((uint)i * 4u) & 12u;
        act0[i] = ScratchBuffer.Load(
            IN_PLANE + lane + bank * BANK + (ty + 1u) * ROW + (tx + 1u) * 16u);
    }

    // ---- stage 1: pw 64 -> 128 + ReLU ----
    [unroll]
    for (int o2 = 0; o2 < 32; o2++) {
        const uint4 b = t18_initializer.Load4(B1_BASE + (uint)o2 * 16u);
        int a0 = asint(b.x);
        int a1 = asint(b.y);
        int a2 = asint(b.z);
        int a3 = asint(b.w);
        [unroll]
        for (int g = 0; g < 4; g++) {
            const uint p0 = act0[4 * g + 0];
            const uint p1 = act0[4 * g + 1];
            const uint p2 = act0[4 * g + 2];
            const uint p3 = act0[4 * g + 3];
            const uint wbase = W1_BASE + (uint)o2 * 256u + (uint)g * 16u;
            a0 = dot4add_i8packed(p0, t18_initializer.Load(wbase + 0u), a0);
            a0 = dot4add_i8packed(p1, t18_initializer.Load(wbase + 4u), a0);
            a0 = dot4add_i8packed(p2, t18_initializer.Load(wbase + 8u), a0);
            a0 = dot4add_i8packed(p3, t18_initializer.Load(wbase + 12u), a0);
            a1 = dot4add_i8packed(p0, t18_initializer.Load(wbase + 64u), a1);
            a1 = dot4add_i8packed(p1, t18_initializer.Load(wbase + 68u), a1);
            a1 = dot4add_i8packed(p2, t18_initializer.Load(wbase + 72u), a1);
            a1 = dot4add_i8packed(p3, t18_initializer.Load(wbase + 76u), a1);
            a2 = dot4add_i8packed(p0, t18_initializer.Load(wbase + 128u), a2);
            a2 = dot4add_i8packed(p1, t18_initializer.Load(wbase + 132u), a2);
            a2 = dot4add_i8packed(p2, t18_initializer.Load(wbase + 136u), a2);
            a2 = dot4add_i8packed(p3, t18_initializer.Load(wbase + 140u), a2);
            a3 = dot4add_i8packed(p0, t18_initializer.Load(wbase + 192u), a3);
            a3 = dot4add_i8packed(p1, t18_initializer.Load(wbase + 196u), a3);
            a3 = dot4add_i8packed(p2, t18_initializer.Load(wbase + 200u), a3);
            a3 = dot4add_i8packed(p3, t18_initializer.Load(wbase + 204u), a3);
        }
        int4 q = int4(requant_shift(max(a0, 0), 6u, 31),
                      requant_shift(max(a1, 0), 6u, 31),
                      requant_shift(max(a2, 0), 6u, 31),
                      requant_shift(max(a3, 0), 6u, 31));
        act1[o2] = pack_bytes(q);
    }

    // ---- residual refill: banks 0,1 input pixels at (ty+1, tx+1) ----
    [unroll]
    for (int i = 0; i < 8; i++) {
        const uint bank = (uint)i >> 2u;
        const uint lane = ((uint)i * 4u) & 12u;
        res[i] = ScratchBuffer.Load(
            IN_PLANE + lane + bank * BANK + (ty + 1u) * ROW + (tx + 1u) * 16u);
    }

    // ---- stage 2 + tail ----
    [unroll]
    for (int o3 = 0; o3 < 16; o3++) {
        const uint4 b = t18_initializer.Load4(B2_BASE + (uint)o3 * 16u);
        int c0 = asint(b.x);
        int c1 = asint(b.y);
        int c2 = asint(b.z);
        int c3 = asint(b.w);
        [unroll]
        for (int g = 0; g < 8; g++) {
            const uint p0 = act1[4 * g + 0];
            const uint p1 = act1[4 * g + 1];
            const uint p2 = act1[4 * g + 2];
            const uint p3 = act1[4 * g + 3];
            const uint wbase = W2_BASE + (uint)o3 * 512u + (uint)g * 16u;
            c0 = dot4add_i8packed(p0, t18_initializer.Load(wbase + 0u), c0);
            c0 = dot4add_i8packed(p1, t18_initializer.Load(wbase + 4u), c0);
            c0 = dot4add_i8packed(p2, t18_initializer.Load(wbase + 8u), c0);
            c0 = dot4add_i8packed(p3, t18_initializer.Load(wbase + 12u), c0);
            c1 = dot4add_i8packed(p0, t18_initializer.Load(wbase + 128u), c1);
            c1 = dot4add_i8packed(p1, t18_initializer.Load(wbase + 132u), c1);
            c1 = dot4add_i8packed(p2, t18_initializer.Load(wbase + 136u), c1);
            c1 = dot4add_i8packed(p3, t18_initializer.Load(wbase + 140u), c1);
            c2 = dot4add_i8packed(p0, t18_initializer.Load(wbase + 256u), c2);
            c2 = dot4add_i8packed(p1, t18_initializer.Load(wbase + 260u), c2);
            c2 = dot4add_i8packed(p2, t18_initializer.Load(wbase + 264u), c2);
            c2 = dot4add_i8packed(p3, t18_initializer.Load(wbase + 268u), c2);
            c3 = dot4add_i8packed(p0, t18_initializer.Load(wbase + 384u), c3);
            c3 = dot4add_i8packed(p1, t18_initializer.Load(wbase + 388u), c3);
            c3 = dot4add_i8packed(p2, t18_initializer.Load(wbase + 392u), c3);
            c3 = dot4add_i8packed(p3, t18_initializer.Load(wbase + 396u), c3);
        }
        const float rscale = (o3 < 8) ? 0.03125f : 0.0625f;  // 2^-5 / 2^-4
        const uint rdw = (o3 < 8) ? res[o3] : act0[o3];
        int4 lanes;
        [unroll]
        for (int j = 0; j < 4; j++) {
            const int rb = (int)(((rdw >> (8 * j)) & 0xFFu) << 24) >> 24;
            const int ab = (j == 0) ? c0 : ((j == 1) ? c1 : ((j == 2) ? c2 : c3));
            const float v = float(rb) * rscale + float(ab) * 0.000244140625f;  // 2^-12
            const float rr = round(v * 16.0f);
            lanes[j] = int16_wrap(int(rr));
        }
        const uint val = pack_bytes(lanes);
        const uint bank = ((uint)o3 * 4u) >> 4u;   // o3 >> 2
        const uint lane = ((uint)o3 * 4u) & 12u;
        ScratchBuffer.Store(
            OUT_PLANE + lane + bank * BANK + (ty + 1u) * ROW + (tx + 1u) * 16u, val);
    }
}
