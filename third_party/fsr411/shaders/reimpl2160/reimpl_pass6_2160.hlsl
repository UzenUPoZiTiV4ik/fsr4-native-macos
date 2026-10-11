// SPDX-License-Identifier: MIT
// Reimplemented FSR 4.1.1 kernel: semantics reconstructed from the AMD 4.1.1
// provider (Kits/FidelityFX/signedbin/amd_fidelityfx_upscaler_dx12.dll,
// SHA-256 d0dcccc74a43c44ba435b7a369b456e0970d8a4464e4bd683119b374f2c9fb46,
// MIT-licensed per FidelityFX-SDK docs/license.md) and validated byte-exact
// against the AMD container. See README.md and docs/evidence411/.
// Copyright (C) 2025 Advanced Micro Devices, Inc. (upstream implementation)
// Copyright (C) 2026 int3rrobang (reimplementation)

// ============================================================================
// 4.1.1 logical pass 6 reimplementation, 2160 class (our_reimplementation)
// Global 5111-equivalent (FFX_FSR4_PASS_12, nocbv 26044, dispatch 8,270,1).
// 2x spatial scaling of bench/shaders/candidates/pass6_411/reimpl_pass6.hlsl
// (1080 class: 240x135, dispatch 4,135,1). Transform ONLY — no logic change.
//
// Geometry law verified vs AMD 2160 shipping container 1526f25ae524ace4
// (analysis/extracted/4.1.1-official/0311_008f3770_1526f25ae524ace4.dxbc,
// entry fsr4_model_v07_fp8_no_scale_pass6) by dxc -dumpbin diff vs the 1080
// container 3b6977a46a8cc9b2: ONLY guard/stride/base immediates differ.
//   guards: tx 239->479, ty 134->269 (threads 480x270)
//   input : row stride 7712->15392, raw plane bases P0 8342464->33273664,
//           P1 10440128->41616128 (spacing 8342464); tap rows +0/+15392,
//           tap cols +0/+16; row step 15424->30784 (=2*15392),
//           col step 32 (unchanged)
//   output: base 12537792->49958592, row stride 3872->7712,
//           plane stride 530464->2097664 (=(270+2)*7712)
//   weights/biases: identical t18 offsets (24576/32768); tail 2^-8 unchanged.
// Weights and biases are RUNTIME data in the t18 initializer (shared across
// resolution classes — same model, same bytes):
//   weights: t18@[24576..32768), bias: t18@[32768..33024) i32[64]
// Dispatch 8x270x1, numthreads 64: tx = DTID.x (guard <480), ty = DTID.y.
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
void reimpl_pass6_2160(uint3 tid : SV_DispatchThreadID)
{
    const uint tx = tid.x;
    const uint ty = tid.y;
    if (tx > 479u || ty > 269u)
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
    // Row step 30784 = 2*15392 (two input rows per thread row, cf. 1080 15424).
    // Bases carry the folded (+15392 row-one, +16 col-one) origin exactly like
    // the 1080 hex taps (0x7f69f0 = 8342464+7728): tap base = raw 2160 plane
    // base + 15408, tap rows +0/+15392, tap cols +0/+16.
    uint4 INL[8];
    INL[0] = ScratchBuffer.Load4(0x1fbf370u + 32u * tx + 30784u * ty);
    INL[1] = ScratchBuffer.Load4(0x27b3f30u + 32u * tx + 30784u * ty);
    INL[2] = ScratchBuffer.Load4(0x1fbf380u + 32u * tx + 30784u * ty);
    INL[3] = ScratchBuffer.Load4(0x27b3f40u + 32u * tx + 30784u * ty);
    INL[4] = ScratchBuffer.Load4(0x1fc2f90u + 32u * tx + 30784u * ty);
    INL[5] = ScratchBuffer.Load4(0x27b7b50u + 32u * tx + 30784u * ty);
    INL[6] = ScratchBuffer.Load4(0x1fc2fa0u + 32u * tx + 30784u * ty);
    INL[7] = ScratchBuffer.Load4(0x27b7b60u + 32u * tx + 30784u * ty);

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
        const uint addr = 49958592u + 2097664u * (d >> 2) + 7712u * (ty + 1u)
                        + 16u * (tx + 1u) + 4u * (d & 3);
        ScratchBuffer.Store(addr, val);
    }
}
