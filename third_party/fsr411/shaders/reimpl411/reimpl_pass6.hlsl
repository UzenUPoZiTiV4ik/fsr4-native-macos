// SPDX-License-Identifier: MIT
// Reimplemented FSR 4.1.1 kernel: semantics reconstructed from the AMD 4.1.1
// provider (Kits/FidelityFX/signedbin/amd_fidelityfx_upscaler_dx12.dll,
// SHA-256 d0dcccc74a43c44ba435b7a369b456e0970d8a4464e4bd683119b374f2c9fb46,
// MIT-licensed per FidelityFX-SDK docs/license.md) and validated byte-exact
// against the AMD container. See README.md and docs/evidence411/.
// Copyright (C) 2025 Advanced Micro Devices, Inc. (upstream implementation)
// Copyright (C) 2026 int3rrobang (reimplementation)

// ============================================================================
// 4.1.1 logical pass 6 reimplementation (our_reimplementation) - Global 5111
// StridedConv2x2: 32ch -> 64ch, /2 spatial. Decoded from container
// 3b6977a46a8cc9b2 (fsr4_model_v07_fp8_no_scale_pass6) and validated
// byte-exact by bench/tools/pass6_sim.py against the captured oracle before
// this HLSL was authored.
//
// Weights and biases are RUNTIME data in the t18 initializer:
//   weights: t18@[24576..32768): W[s][tap][plane] = 16 B at
//            24576 + 32*s + 2048*tap + 16*plane (s = out slot 0..63,
//            tap 0..3 = 2x2 window row-major, plane 0..1 = 16ch half)
//   bias   : t18@[32768..33024): i32[64]
// Input : 2x2 window of 16-ch pixels from the planes at 8342464 / 10440128
//         (stride 7712; tap rows +0/+7712, tap cols +0/+16; row step 2*7712,
//         col step 32 = 2 pixels). 8 taps x uint4 loads.
// Output: four 16-ch planes at 12537792 + p*530464 (p = slot/16),
//         row stride 3872, pixel (ty+1, tx+1).
// Tail  : out byte = s8clamp(i16wrap(round_ne(slot * 2^-8))).
// Dispatch 4x135x1, numthreads 64: tx = DTID.x (guard <240), ty = DTID.y.
// ============================================================================
ByteAddressBuffer t18_initializer : register(t18);
RWByteAddressBuffer ScratchBuffer : register(u11);

uint pack_bytes(int4 q)
{
    // SM 6.6 intrinsic -> DXIL Pack4x8 packMode 2 (SClamp), identical to the
    // 4.1.1 container's packs. (Do NOT name this helper pack_s8: the name
    // shadows the intrinsic and dxc silently emits truncating pack4x8 mode 0.)
    return (uint)pack_clamp_s8(q);
}

int int16_wrap(int v)
{
    return ((v & 0xFFFF) ^ 0x8000) - 0x8000;
}

[numthreads(64, 1, 1)]
void reimpl_pass6(uint3 tid : SV_DispatchThreadID)
{
    const uint tx = tid.x;
    const uint ty = tid.y;
    if (tx > 239u || ty > 134u)
        return;

    // ---- bias[64] from t18@32768 ----
    int slot[64];
    [unroll]
    for (int b = 0; b < 16; b++) {
        const uint4 v = t18_initializer.Load4(32768u + 16u * b);
        slot[4 * b + 0] = asint(v.x);
        slot[4 * b + 1] = asint(v.y);
        slot[4 * b + 2] = asint(v.z);
        slot[4 * b + 3] = asint(v.w);
    }

    // ---- input window: 8 taps (2 planes x 2x2), tap g = tap*2+plane ----
    uint4 INL[8];
    INL[0] = ScratchBuffer.Load4(0x7f69f0u + 32u * tx + 15424u * ty);
    INL[1] = ScratchBuffer.Load4(0x9f6bf0u + 32u * tx + 15424u * ty);
    INL[2] = ScratchBuffer.Load4(0x7f6a00u + 32u * tx + 15424u * ty);
    INL[3] = ScratchBuffer.Load4(0x9f6c00u + 32u * tx + 15424u * ty);
    INL[4] = ScratchBuffer.Load4(0x7f8810u + 32u * tx + 15424u * ty);
    INL[5] = ScratchBuffer.Load4(0x9f8a10u + 32u * tx + 15424u * ty);
    INL[6] = ScratchBuffer.Load4(0x7f8820u + 32u * tx + 15424u * ty);
    INL[7] = ScratchBuffer.Load4(0x9f8a20u + 32u * tx + 15424u * ty);

    // ---- conv: slot[4q+qq] += sum_g sum_j dot4(INL[g][j], Wload[qq][j]) ----
    // W byte base for (q, g): 24576 + 2048*(g>>1) + 128*q + 16*(g&1),
    // four uint4 loads at +{0, 32, 64, 96}; slot qq uses all four dwords
    // of load qq (wblock[4*qq+j]), paired with input lanes j=0..3.
    [unroll]
    for (int q = 0; q < 16; q++) {
        [unroll]
        for (int g = 0; g < 8; g++) {
            const uint wb = 24576u + 2048u * (g >> 1) + 128u * q + 16u * (g & 1);
            uint4 W[4];
            W[0] = t18_initializer.Load4(wb);
            W[1] = t18_initializer.Load4(wb + 32u);
            W[2] = t18_initializer.Load4(wb + 64u);
            W[3] = t18_initializer.Load4(wb + 96u);
            [unroll]
            for (int qq = 0; qq < 4; qq++) {
                int acc = slot[4 * q + qq];
                acc = dot4add_i8packed(INL[g].x, W[qq].x, acc);
                acc = dot4add_i8packed(INL[g].y, W[qq].y, acc);
                acc = dot4add_i8packed(INL[g].z, W[qq].z, acc);
                acc = dot4add_i8packed(INL[g].w, W[qq].w, acc);
                slot[4 * q + qq] = acc;
            }
        }
    }

    // ---- tail + stores: dword d = pack(slots 4d..4d+3) -> plane d/4 ----
    [unroll]
    for (int d = 0; d < 16; d++) {
        int4 lanes;
        [unroll]
        for (int j3 = 0; j3 < 4; j3++) {
            const float r = round(float(slot[4 * d + j3]) * 3.90625e-03f); // 2^-8
            lanes[j3] = int16_wrap(int(r));
        }
        const uint val = pack_bytes(lanes);
        const uint addr = 12537792u + 530464u * (d >> 2) + 3872u * (ty + 1u)
                        + 16u * (tx + 1u) + 4u * (d & 3);
        ScratchBuffer.Store(addr, val);
    }
}
