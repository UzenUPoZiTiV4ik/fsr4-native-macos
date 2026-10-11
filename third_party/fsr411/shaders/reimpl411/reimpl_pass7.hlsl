// SPDX-License-Identifier: MIT
// Reimplemented FSR 4.1.1 kernel: semantics reconstructed from the AMD 4.1.1
// provider (Kits/FidelityFX/signedbin/amd_fidelityfx_upscaler_dx12.dll,
// SHA-256 d0dcccc74a43c44ba435b7a369b456e0970d8a4464e4bd683119b374f2c9fb46,
// MIT-licensed per FidelityFX-SDK docs/license.md) and validated byte-exact
// against the AMD container. See README.md and docs/evidence411/.
// Copyright (C) 2025 Advanced Micro Devices, Inc. (upstream implementation)
// Copyright (C) 2026 int3rrobang (reimplementation)

// ============================================================================
// 4.1.1 logical pass 7 reimplementation (our_reimplementation) - Global 5114
// 3-stage INT8 chain over the 4-channel int8 plane image at u11@12537792
// (channel stride 530464 = 137 rows * 3872 pitch):
//   phase1: 3x3 conv 16ch -> 32ch (8 groups of 4ch), bias1 = t18@37632,
//           weights W1 @33024 + j*64 + g*512 (16 dwords per (j,g)),
//           requant (acc + 63 + ((uint(acc)>>7)&1)) >> 7, pack SClamp.
//   stage2: 1x1 64ch -> 128ch + ReLU, bias2 = t18@45952,
//           weights W2 @37760 + i*256 (64 dwords per i), requant >>7.
//   stage3: 1x1 128ch -> 16ch, bias3 = t18@54656,
//           weights W3 @46464 + i2*512 (128 dwords per i2), raw acc.
//   tail: out = s8clamp(i16wrap(round_ne((s8(resid)*2^-5 + s8(acc)*2^-12)*S)))
//           resid = u11 reloads (rows ty+1, ch i2), S = 32 (i2<8) / 16.
//   store 1 dword per i2 @14659648 + (i2>>2)*530464 + 3872*(ty+1) + tx*16
//           + 16 + (i2&3)*4.
//
// Decoded from container 0232_006e05c0_9c56e2d6eeb7457d.dxbc
// (fsr4_model_v07_fp8_no_scale_pass7) and validated byte-exact vs the AMD
// oracle by bench/tools/pass7_sim.py before this HLSL was authored.
// Root: nocbv (UAV12 + SRV19, no CBV). Dispatch 8,270,1, NumThreads 64,1,1.
// ============================================================================
ByteAddressBuffer t18_initializer : register(t18);
RWByteAddressBuffer ScratchBuffer : register(u11);

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
void reimpl_pass7(uint3 tid : SV_DispatchThreadID)
{
    const uint tx = tid.x;
    const uint ty = tid.y;
    if (tx > 239u || ty > 134u)
        return;

    // ---- phase 1: 3x3 conv 16ch -> 32ch ----
    uint tmp[16];
    for (int j = 0; j < 8; j++) {
        uint4 b = t18_initializer.Load4(37632u + uint(j) * 16u);
        int acc0 = asint(b.x);
        int acc1 = asint(b.y);
        int acc2 = asint(b.z);
        int acc3 = asint(b.w);
        const uint chBase = 12537792u + uint(j >> 2) * 530464u + 3872u * ty + tx * 16u;
        for (int g = 0; g < 9; g++) {
            const uint4 act = ScratchBuffer.Load4(chBase + uint(g / 3) * 3872u + uint(g % 3) * 16u);
            const uint wBase = 33024u + uint(j) * 64u + uint(g) * 512u;
            // DXIL: UMad(16,(j<<2)|k,g*512)+33024 -> k stride 16B, m stride 4B.
            uint wrow[4];
            wrow[0] = t18_initializer.Load(wBase);
            wrow[1] = t18_initializer.Load(wBase + 16u);
            wrow[2] = t18_initializer.Load(wBase + 32u);
            wrow[3] = t18_initializer.Load(wBase + 48u);
            // acc[k] += sum_m dot4(act[m], row_k[m]); row k dword m = +k*16+m*4
            acc0 = dot4add_i8packed(act.x, wrow[0], acc0);
            acc0 = dot4add_i8packed(act.y, t18_initializer.Load(wBase + 4u), acc0);
            acc0 = dot4add_i8packed(act.z, t18_initializer.Load(wBase + 8u), acc0);
            acc0 = dot4add_i8packed(act.w, t18_initializer.Load(wBase + 12u), acc0);
            acc1 = dot4add_i8packed(act.x, wrow[1], acc1);
            acc1 = dot4add_i8packed(act.y, t18_initializer.Load(wBase + 20u), acc1);
            acc1 = dot4add_i8packed(act.z, t18_initializer.Load(wBase + 24u), acc1);
            acc1 = dot4add_i8packed(act.w, t18_initializer.Load(wBase + 28u), acc1);
            acc2 = dot4add_i8packed(act.x, wrow[2], acc2);
            acc2 = dot4add_i8packed(act.y, t18_initializer.Load(wBase + 36u), acc2);
            acc2 = dot4add_i8packed(act.z, t18_initializer.Load(wBase + 40u), acc2);
            acc2 = dot4add_i8packed(act.w, t18_initializer.Load(wBase + 44u), acc2);
            acc3 = dot4add_i8packed(act.x, wrow[3], acc3);
            acc3 = dot4add_i8packed(act.y, t18_initializer.Load(wBase + 52u), acc3);
            acc3 = dot4add_i8packed(act.z, t18_initializer.Load(wBase + 56u), acc3);
            acc3 = dot4add_i8packed(act.w, t18_initializer.Load(wBase + 60u), acc3);
        }
        tmp[j] = pack_bytes(int4(requant_shift(acc0, 7u, 63),
                                 requant_shift(acc1, 7u, 63),
                                 requant_shift(acc2, 7u, 63),
                                 requant_shift(acc3, 7u, 63)));
    }
    // passthrough rows for the tail residual (device 784-loop)
    const uint rBase = 12537792u + 3872u * (ty + 1u) + tx * 16u + 16u;
    for (int p = 0; p < 8; p++)
        tmp[8 + p] = ScratchBuffer.Load(rBase + (uint(p / 4) + 2u) * 530464u + uint(p % 4) * 4u);

    // ---- stage 2: 1x1 64ch -> 128ch + ReLU ----
    uint hid[32];
    for (int i = 0; i < 32; i++) {
        uint4 b2 = t18_initializer.Load4(45952u + uint(i) * 16u);
        int h0 = asint(b2.x);
        int h1 = asint(b2.y);
        int h2 = asint(b2.z);
        int h3 = asint(b2.w);
        const uint wiBase = 37760u + uint(i) * 256u;
        for (int s = 0; s < 16; s++) {
            const uint a = tmp[s];
            const uint gBase = wiBase + uint(s / 4) * 16u + uint(s % 4) * 4u;
            h0 = dot4add_i8packed(a, t18_initializer.Load(gBase), h0);
            h1 = dot4add_i8packed(a, t18_initializer.Load(gBase + 64u), h1);
            h2 = dot4add_i8packed(a, t18_initializer.Load(gBase + 128u), h2);
            h3 = dot4add_i8packed(a, t18_initializer.Load(gBase + 192u), h3);
        }
        h0 = max(h0, 0); h1 = max(h1, 0); h2 = max(h2, 0); h3 = max(h3, 0);
        hid[i] = pack_bytes(int4(requant_shift(h0, 7u, 63),
                                 requant_shift(h1, 7u, 63),
                                 requant_shift(h2, 7u, 63),
                                 requant_shift(h3, 7u, 63)));
    }

    // ---- tail residual reload (device 1165-loop) ----
    uint res[8];
    for (int r2 = 0; r2 < 8; r2++)
        res[r2] = ScratchBuffer.Load(rBase + uint(r2 / 4) * 530464u + uint(r2 % 4) * 4u);

    // ---- stage 3 + tail ----
    const uint oBase = 14659648u + 3872u * (ty + 1u) + tx * 16u + 16u;
    for (int i2 = 0; i2 < 16; i2++) {
        uint4 b3 = t18_initializer.Load4(54656u + uint(i2) * 16u);
        int c0 = asint(b3.x);
        int c1 = asint(b3.y);
        int c2 = asint(b3.z);
        int c3 = asint(b3.w);
        const uint woBase = 46464u + uint(i2) * 512u;
        for (int s2 = 0; s2 < 32; s2++) {
            const uint a2 = hid[s2];
            const uint gBase = woBase + uint(s2 / 4) * 16u + uint(s2 % 4) * 4u;
            c0 = dot4add_i8packed(a2, t18_initializer.Load(gBase), c0);
            c1 = dot4add_i8packed(a2, t18_initializer.Load(gBase + 128u), c1);
            c2 = dot4add_i8packed(a2, t18_initializer.Load(gBase + 256u), c2);
            c3 = dot4add_i8packed(a2, t18_initializer.Load(gBase + 384u), c3);
        }
        const uint rw = (i2 < 8) ? res[i2] : tmp[8 + (i2 - 8)];
        int4 lanes;
        for (int j3 = 0; j3 < 4; j3++) {
            const int cbyte = (int)(((rw >> (8 * j3)) & 0xFFu) << 24) >> 24;
            const float t1 = float(cbyte) * 0.03125f;                // 2^-5
            const int aval = (j3 == 0) ? c0 : ((j3 == 1) ? c1 : ((j3 == 2) ? c2 : c3));
            const float t2 = float(aval) * 0.000244140625f;          // 2^-12
            const float r = round((t1 + t2) * ((i2 < 8) ? 32.0f : 16.0f));
            lanes[j3] = int16_wrap(int(r));
        }
        ScratchBuffer.Store(oBase + uint(i2 / 4) * 530464u + uint(i2 % 4) * 4u, pack_bytes(lanes));
    }
}
