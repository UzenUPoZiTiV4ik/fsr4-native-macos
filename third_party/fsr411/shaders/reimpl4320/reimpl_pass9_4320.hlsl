// SPDX-License-Identifier: MIT
// Reimplemented FSR 4.1.1 kernel: semantics reconstructed from the AMD 4.1.1
// provider (Kits/FidelityFX/signedbin/amd_fidelityfx_upscaler_dx12.dll,
// SHA-256 d0dcccc74a43c44ba435b7a369b456e0970d8a4464e4bd683119b374f2c9fb46,
// MIT-licensed per FidelityFX-SDK docs/license.md) and validated byte-exact
// against the AMD container. See README.md and docs/evidence411/.
// Copyright (C) 2025 Advanced Micro Devices, Inc. (upstream implementation)
// Copyright (C) 2026 int3rrobang (reimplementation)

// ============================================================================
// 4.1.1 pass9 reimplementation, 2160 class (our_reimplementation)
// Global 5120-equivalent: 2x-upsample bottleneck block, 480x270 -> 960x540
// (FFX_FSR4_PASS_18, compute root 26039, dispatch 8,270,1).
// 2x spatial scaling of bench/shaders/candidates/pass9_411/reimpl_pass9.hlsl
// (1080 class: 240x135 -> 480x270, dispatch 4,135,1). Transform ONLY.
// Geometry law verified vs AMD 2160 shipping container 824595cf71596381
// (analysis/extracted/4.1.1-official/0094_00337550_824595cf71596381.dxbc,
// entry fsr4_model_v07_fp8_no_scale_pass9) by dxc -dumpbin diff vs the 1080
// container 8c27af0446e1a7be: ONLY guard/stride/base immediates differ.
//   guards: tx 239->959, ty 134->539 (threads 480x270)
//   input tiles (12537792): row stride 3872->15392, pitch 530464->8342464,
//           base ->199449792 (== pass6-2160 output / pass7-2160 input)
//   stage-4 planes: row stride 15392->30752, tile stride 8342464->33273664,
//           resid base 33273664->132902464 (== pass6-2160 input P0),
//           store base 14659648->232819648 (== pass7-2160 output)
//   stage-4 footprint: col<480->1920, row<270->1080 (output 960x540)
//   CBV tensor-size guard: same regs; 2160 pyramid doubles every level,
//           slot9 (480,270) — still a no-op past the entry guard.
//   weights/biases: identical t18 offsets (76800..107008); tail scales unchanged.
//
//   stage1 3x3 conv on plane-199449792 tiles 0,1 (482x272x16B, pitch 8342464,
//            row stride 15392) -> requant >>8 -> 8 dwords (32ch)
//   stage2 1x1 64ch -> 128ch + ReLU (second half = raw tiles 2,3 pixels)
//            -> requant >>7 -> 32 dwords
//   stage3 1x1 128ch -> 64ch (raw acc) + residual tail A
//            (r*2^-4 + a*2^-11) * 16 -> 16 dwords
//   stage4 1x1 64ch -> 128ch over the 2x2 block x 2 planes + residual
//            tail B (r*2^-5 + a*2^-13) * 16, guarded store to 232819648.
//
// All weights AND biases are t18 (runtime initializer) constants at the
// decoded offsets (76800..107008); no DXIL-embedded tables are needed.
// ============================================================================
ByteAddressBuffer t18_initializer : register(t18);
RWByteAddressBuffer ScratchBuffer : register(u11);

cbuffer MLSR_Optimized_Constants : register(b0)
{
    uint4 ml2c_cb[17]; // ml2c_cb[9].xy = (480, 270) output tensor size
}

int requant_shift(int x, uint shift, int add)
{
    uint bit = (uint(x) >> shift) & 1u;
    return (x + add + int(bit)) >> int(shift); // arithmetic shift
}

uint pack_bytes(int4 q)
{
    // SM 6.6 intrinsic -> DXIL Pack4x8 packMode 2 (SClamp), identical to the
    // 4.1.1 container's requant packs. (A hand-written clamp+byte-compose gets
    // pattern-matched by dxc into a TRUNCATING pack4x8 mode 0.)
    return (uint)pack_clamp_s8(q);
}

int int16_wrap(int v)
{
    return ((v & 0xFFFF) ^ 0x8000) - 0x8000;
}

[numthreads(64, 1, 1)]
void reimpl_pass9_4320(uint3 tid : SV_DispatchThreadID)
{
    const uint tx = tid.x;
    const uint ty = tid.y;
    const uint tz = tid.z;
    if (tx > 959u || ty > 539u)
        return;

    // ---- stage 1: 3x3 conv, tiles 0/1 -> s1[8] ----
    uint s1[8];
    [unroll]
    for (int i = 0; i < 8; i++) {
        const uint4 b1 = t18_initializer.Load4(81408u + 16u * i + (tz << 8));
        int acc0 = asint(b1.x);
        int acc1 = asint(b1.y);
        int acc2 = asint(b1.z);
        int acc3 = asint(b1.w);
        const uint tile = uint(i >> 2);
        [unroll]
        for (int t = 0; t < 9; t++) {
            const uint dy = uint(t / 3);
            const uint dx = uint(t % 3);
            const uint addr = 199449792u + 8342464u * tile + 15392u * (ty + dy) + 16u * tx + 16u * dx;
            const uint4 px = ScratchBuffer.Load4(addr);
            const uint wbase = 76800u + 512u * t + 64u * i;
            acc0 = dot4add_i8packed(px.x, t18_initializer.Load(wbase), acc0);
            acc0 = dot4add_i8packed(px.y, t18_initializer.Load(wbase + 4u), acc0);
            acc0 = dot4add_i8packed(px.z, t18_initializer.Load(wbase + 8u), acc0);
            acc0 = dot4add_i8packed(px.w, t18_initializer.Load(wbase + 12u), acc0);
            acc1 = dot4add_i8packed(px.x, t18_initializer.Load(wbase + 16u), acc1);
            acc1 = dot4add_i8packed(px.y, t18_initializer.Load(wbase + 20u), acc1);
            acc1 = dot4add_i8packed(px.z, t18_initializer.Load(wbase + 24u), acc1);
            acc1 = dot4add_i8packed(px.w, t18_initializer.Load(wbase + 28u), acc1);
            acc2 = dot4add_i8packed(px.x, t18_initializer.Load(wbase + 32u), acc2);
            acc2 = dot4add_i8packed(px.y, t18_initializer.Load(wbase + 36u), acc2);
            acc2 = dot4add_i8packed(px.z, t18_initializer.Load(wbase + 40u), acc2);
            acc2 = dot4add_i8packed(px.w, t18_initializer.Load(wbase + 44u), acc2);
            acc3 = dot4add_i8packed(px.x, t18_initializer.Load(wbase + 48u), acc3);
            acc3 = dot4add_i8packed(px.y, t18_initializer.Load(wbase + 52u), acc3);
            acc3 = dot4add_i8packed(px.z, t18_initializer.Load(wbase + 56u), acc3);
            acc3 = dot4add_i8packed(px.w, t18_initializer.Load(wbase + 60u), acc3);
        }
        s1[i] = pack_bytes(int4(requant_shift(acc0, 8u, 127),
                                requant_shift(acc1, 8u, 127),
                                requant_shift(acc2, 8u, 127),
                                requant_shift(acc3, 8u, 127)));
    }

    // ---- stage 1b: raw tiles 2,3 -> in16[16] ----
    uint in16[16];
    [unroll]
    for (int r = 0; r < 8; r++)
        in16[r] = s1[r];
    [unroll]
    for (int g2 = 0; g2 < 2; g2++) {
        const uint4 px = ScratchBuffer.Load4(199449792u + 8342464u * (2u + g2) + 15392u * (ty + 1u) + 16u * tx + 16u);
        in16[8 + 4 * g2 + 0] = px.x;
        in16[8 + 4 * g2 + 1] = px.y;
        in16[8 + 4 * g2 + 2] = px.z;
        in16[8 + 4 * g2 + 3] = px.w;
    }

    // ---- stage 2: 1x1 64ch -> 128ch + ReLU -> s2[32] ----
    uint s2[32];
    [unroll]
    for (int j = 0; j < 32; j++) {
        const uint4 b2 = t18_initializer.Load4(89728u + 16u * j + (tz << 10));
        int a0 = asint(b2.x);
        int a1 = asint(b2.y);
        int a2 = asint(b2.z);
        int a3 = asint(b2.w);
        const uint wj = 81536u + 256u * j;
        [unroll]
        for (int p = 0; p < 8; p++) {
            const uint w = wj + 8u * p;
            a0 = dot4add_i8packed(in16[2 * p], t18_initializer.Load(w), a0);
            a0 = dot4add_i8packed(in16[2 * p + 1], t18_initializer.Load(w + 4u), a0);
            a1 = dot4add_i8packed(in16[2 * p], t18_initializer.Load(w + 64u), a1);
            a1 = dot4add_i8packed(in16[2 * p + 1], t18_initializer.Load(w + 68u), a1);
            a2 = dot4add_i8packed(in16[2 * p], t18_initializer.Load(w + 128u), a2);
            a2 = dot4add_i8packed(in16[2 * p + 1], t18_initializer.Load(w + 132u), a2);
            a3 = dot4add_i8packed(in16[2 * p], t18_initializer.Load(w + 192u), a3);
            a3 = dot4add_i8packed(in16[2 * p + 1], t18_initializer.Load(w + 196u), a3);
        }
        s2[j] = pack_bytes(int4(requant_shift(max(a0, 0), 7u, 63),
                                requant_shift(max(a1, 0), 7u, 63),
                                requant_shift(max(a2, 0), 7u, 63),
                                requant_shift(max(a3, 0), 7u, 63)));
    }

    // ---- stage 2b: residual dwords r9[16] (tiles 0,1 fresh; 2,3 reused) ----
    uint r9[16];
    [unroll]
    for (int g3 = 0; g3 < 2; g3++) {
        const uint4 px = ScratchBuffer.Load4(199449792u + 8342464u * g3 + 15392u * (ty + 1u) + 16u * tx + 16u);
        r9[4 * g3 + 0] = px.x;
        r9[4 * g3 + 1] = px.y;
        r9[4 * g3 + 2] = px.z;
        r9[4 * g3 + 3] = px.w;
    }
    [unroll]
    for (int r2 = 0; r2 < 8; r2++)
        r9[8 + r2] = in16[8 + r2];

    // ---- CBV tensor-size guard (matches container block 1073) ----
    if (tx >= ml2c_cb[9].x || ty >= ml2c_cb[9].y)
        return;

    // ---- stage 3: 1x1 128ch -> 64ch + tail A -> out10[16] ----
    uint out10[16];
    [unroll]
    for (int si = 0; si < 16; si++) {
        const uint4 b3 = t18_initializer.Load4(98432u + 16u * si + (tz << 10));
        int c0 = asint(b3.x);
        int c1 = asint(b3.y);
        int c2 = asint(b3.z);
        int c3 = asint(b3.w);
        const uint wi = 90240u + 512u * si;
        [unroll]
        for (int p2 = 0; p2 < 16; p2++) {
            const uint w = wi + 8u * p2;
            c0 = dot4add_i8packed(s2[2 * p2], t18_initializer.Load(w), c0);
            c0 = dot4add_i8packed(s2[2 * p2 + 1], t18_initializer.Load(w + 4u), c0);
            c1 = dot4add_i8packed(s2[2 * p2], t18_initializer.Load(w + 128u), c1);
            c1 = dot4add_i8packed(s2[2 * p2 + 1], t18_initializer.Load(w + 132u), c1);
            c2 = dot4add_i8packed(s2[2 * p2], t18_initializer.Load(w + 256u), c2);
            c2 = dot4add_i8packed(s2[2 * p2 + 1], t18_initializer.Load(w + 260u), c2);
            c3 = dot4add_i8packed(s2[2 * p2], t18_initializer.Load(w + 384u), c3);
            c3 = dot4add_i8packed(s2[2 * p2 + 1], t18_initializer.Load(w + 388u), c3);
        }
        const uint rd = r9[si];
        const int rb0 = int(((rd) & 0xFFu) << 24) >> 24;
        const int rb1 = int(((rd >> 8u) & 0xFFu) << 24) >> 24;
        const int rb2 = int(((rd >> 16u) & 0xFFu) << 24) >> 24;
        const int rb3 = int(((rd >> 24u) & 0xFFu) << 24) >> 24;
        int4 lanes;
        lanes.x = int16_wrap(int(round((float(rb0) * 0.0625f + float(c0) * 0.00048828125f) * 16.0f)));
        lanes.y = int16_wrap(int(round((float(rb1) * 0.0625f + float(c1) * 0.00048828125f) * 16.0f)));
        lanes.z = int16_wrap(int(round((float(rb2) * 0.0625f + float(c2) * 0.00048828125f) * 16.0f)));
        lanes.w = int16_wrap(int(round((float(rb3) * 0.0625f + float(c3) * 0.00048828125f) * 16.0f)));
        out10[si] = pack_bytes(lanes);
    }

    // ---- stage 4: 1x1 64ch -> 128ch over the 2x2 block x 2 planes ----
    [unroll]
    for (int o = 0; o < 2; o++) {
        [unroll]
        for (int m = 0; m < 2; m++) {
            const uint col = uint(m) + 2u * tx;
            const uint row = uint(o) + 2u * ty;
            [unroll]
            for (int n = 0; n < 8; n++) {
                const uint tile = uint(n >> 2);
                const uint lane = uint(n & 3);
                const uint4 b4 = t18_initializer.Load4(106880u + 16u * n + (tz << 10));
                int d0 = asint(b4.x);
                int d1 = asint(b4.y);
                int d2 = asint(b4.z);
                int d3 = asint(b4.w);
                const uint wn = 98688u + 4096u * o + 2048u * m + 256u * n;
                [unroll]
                for (int p3 = 0; p3 < 8; p3++) {
                    const uint w = wn + 8u * p3;
                    d0 = dot4add_i8packed(out10[2 * p3], t18_initializer.Load(w), d0);
                    d0 = dot4add_i8packed(out10[2 * p3 + 1], t18_initializer.Load(w + 4u), d0);
                    d1 = dot4add_i8packed(out10[2 * p3], t18_initializer.Load(w + 64u), d1);
                    d1 = dot4add_i8packed(out10[2 * p3 + 1], t18_initializer.Load(w + 68u), d1);
                    d2 = dot4add_i8packed(out10[2 * p3], t18_initializer.Load(w + 128u), d2);
                    d2 = dot4add_i8packed(out10[2 * p3 + 1], t18_initializer.Load(w + 132u), d2);
                    d3 = dot4add_i8packed(out10[2 * p3], t18_initializer.Load(w + 192u), d3);
                    d3 = dot4add_i8packed(out10[2 * p3 + 1], t18_initializer.Load(w + 196u), d3);
                }
                const uint resid = ScratchBuffer.Load(132902464u + 33273664u * tile + 30752u * (row + 1u) + 16u * col + 16u + 4u * lane);
                const int sb0 = int(((resid) & 0xFFu) << 24) >> 24;
                const int sb1 = int(((resid >> 8u) & 0xFFu) << 24) >> 24;
                const int sb2 = int(((resid >> 16u) & 0xFFu) << 24) >> 24;
                const int sb3 = int(((resid >> 24u) & 0xFFu) << 24) >> 24;
                int4 lanes2;
                lanes2.x = int16_wrap(int(round((float(sb0) * 0.03125f + float(d0) * 0.0001220703125f) * 16.0f)));
                lanes2.y = int16_wrap(int(round((float(sb1) * 0.03125f + float(d1) * 0.0001220703125f) * 16.0f)));
                lanes2.z = int16_wrap(int(round((float(sb2) * 0.03125f + float(d2) * 0.0001220703125f) * 16.0f)));
                lanes2.w = int16_wrap(int(round((float(sb3) * 0.03125f + float(d3) * 0.0001220703125f) * 16.0f)));
                const bool inRange = col < 1920u && row < 1080u;
                const uint val = inRange ? pack_bytes(lanes2) : 0u;
                const uint addr = inRange
                    ? (232819648u + 33273664u * tile + 30752u * (row + 1u) + 16u * col + 16u + 4u * lane)
                    : 134217728u;
                ScratchBuffer.Store(addr, val);
            }
        }
    }
}
