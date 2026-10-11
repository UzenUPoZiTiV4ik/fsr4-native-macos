// SPDX-License-Identifier: MIT
// Reimplemented FSR 4.1.1 kernel: semantics reconstructed from the AMD 4.1.1
// provider (Kits/FidelityFX/signedbin/amd_fidelityfx_upscaler_dx12.dll,
// SHA-256 d0dcccc74a43c44ba435b7a369b456e0970d8a4464e4bd683119b374f2c9fb46,
// MIT-licensed per FidelityFX-SDK docs/license.md) and validated byte-exact
// against the AMD container. See README.md and docs/evidence411/.
// Copyright (C) 2025 Advanced Micro Devices, Inc. (upstream implementation)
// Copyright (C) 2026 int3rrobang (reimplementation)

// 2160-class variant of reimpl_pass3.hlsl (mechanical 2x-spatial transform, no other changes).
// 1080 reference container: analysis/extracted/4.1.1-official/0214_0066ed50_f790b1105a766948.dxbc (1080, guards 479/269, in-stride 15392, out-stride 7712, base 8342464, spacing 2097664)
// 2160 AMD container (ground truth): analysis/extracted/4.1.1-official/0020_0011a890_8e09087b83065612.dxbc (2160, guards 959/539, in-stride 30752, out-stride 15392, base 33273664, spacing 8342464)
// Verified transform (full-DXIL diff, identical line counts, immediates-only deltas):
//   guards x2, inRange x2, full-res stride 15392->30752, half-res stride 7712->15392,
//   plane spacing 2097664->8342464, bases relocated per container (see 2160 line above).
// Weights / biases / requant / tail / loop structure: byte-identical to the 1080 file.
// ============================================================================
// 4.1.1 logical pass 3 reimplementation (our_reimplementation) - Global 5102
// DownscaleStridedConv2x2: 16ch -> 32ch, /2 spatial. Decoded from container
// f790b1105a766948... (fsr4_model_v07_fp8_no_scale_pass3) and validated
// byte-exact (modulo the established 1399-byte rematerialization noise) by
// bench/tools/pass3_sim.py against the captured oracle before this HLSL was
// authored.
//
// Weights and biases are RUNTIME data in the t18 initializer (unlike passes
// 1-2 whose weights are compile-time constants):
//   weights: t18@8576, tile(tap p, out-dword-group g) at 8576 + 512*p + 64*g
//            (16 dwords; iteration i, input-dword k uses tile dword 4*i+k)
//   bias   : t18@10624, i32[32]
// Input : 2x2 window of 16-ch pixels from the plane at scratch base 0
//         (rows 2*ty+1..2*ty+2, byte cols 32*tx+16..32*tx+32, stride 15392).
// Output: two 16-ch half-res planes at 8342464 and 10440128 (=+272*7712),
//         row stride 7712, pixel (ty+1, tx+1) - matches the 5103 epilogue
//         arena deltas [8342464, 10440128].
// Tail  : out byte = s8clamp(i16wrap(round_ne(slot * 2^-7))).
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
void reimpl_pass3_2160(uint3 tid : SV_DispatchThreadID)
{
    const uint tx = tid.x;
    const uint ty = tid.y;
    if (tx > 959u || ty > 539u)
        return;
    const bool inRange = tx < 960u && ty < 540u;

    // ---- bias[32] from t18@10624 (tz-sliced by 128 B in the container; z=1) ----
    int slot[32];
    [unroll]
    for (int b = 0; b < 8; b++) {
        const uint4 v = t18_initializer.Load4(10624u + 16u * b);
        slot[4 * b + 0] = asint(v.x);
        slot[4 * b + 1] = asint(v.y);
        slot[4 * b + 2] = asint(v.z);
        slot[4 * b + 3] = asint(v.w);
    }

    // ---- input window: 2x2 16-ch pixels, tap order (dy,dx) row-major ----
    uint4 P[4];
    [unroll]
    for (int p = 0; p < 4; p++) {
        const uint row = 30752u * (2u * ty + 1u + (p >> 1));
        const uint col = 32u * tx + 16u + 16u * (p & 1u);
        P[p] = ScratchBuffer.Load4(row + col);
    }

    // ---- conv: tile(p,g) iteration i, input dword k -> slot[4g+i] ----
    [unroll]
    for (int pp = 0; pp < 4; pp++) {
        [unroll]
        for (int g = 0; g < 8; g++) {
            uint4 T[4];
            [unroll]
            for (int j = 0; j < 4; j++)
                T[j] = t18_initializer.Load4(8576u + 512u * pp + 64u * g + 16u * j);
            [unroll]
            for (int i = 0; i < 4; i++) {
                int acc = slot[4 * g + i];
                [unroll]
                for (int k = 0; k < 4; k++) {
                    const uint diy = (4 * i + k) >> 2;
                    const uint dix = (4 * i + k) & 3u;
                    acc = dot4add_i8packed(P[pp][k], T[diy][dix], acc);
                }
                slot[4 * g + i] = acc;
            }
        }
    }

    // ---- tail + stores: two 16-ch planes, dword d = pack(slots 16q+4d..+3) ----
    [unroll]
    for (int q = 0; q < 2; q++) {
        [unroll]
        for (int d = 0; d < 4; d++) {
            int4 lanes;
            [unroll]
            for (int j3 = 0; j3 < 4; j3++) {
                const float r = round(float(slot[16 * q + 4 * d + j3]) * 7.8125e-03f); // 2^-7
                lanes[j3] = int16_wrap(int(r));
            }
            const uint val = inRange ? pack_bytes(lanes) : 0u;
            const uint addr = 33273664u + 8342464u * q + 15392u * (ty + 1u) + 16u * (tx + 1u) + 4u * d;
            ScratchBuffer.Store(addr, val);
        }
    }
}
