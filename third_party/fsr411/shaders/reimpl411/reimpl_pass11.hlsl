// SPDX-License-Identifier: MIT
// Reimplemented FSR 4.1.1 kernel: semantics reconstructed from the AMD 4.1.1
// provider (Kits/FidelityFX/signedbin/amd_fidelityfx_upscaler_dx12.dll,
// SHA-256 d0dcccc74a43c44ba435b7a369b456e0970d8a4464e4bd683119b374f2c9fb46,
// MIT-licensed per FidelityFX-SDK docs/license.md) and validated byte-exact
// against the AMD container. See README.md and docs/evidence411/.
// Copyright (C) 2025 Advanced Micro Devices, Inc. (upstream implementation)
// Copyright (C) 2026 int3rrobang (reimplementation)

// ============================================================================
// 4.1.1 logical pass 11 reimplementation (our_reimplementation) - Global 5126
// FFX_FSR4_PASS_22, final x2 upsample conv (480x270 -> 960x540).
// Decoded from container 0284_00820540_0c3304e8e2d28893.dxbc
// (fsr4_model_v07_fp8_no_scale_pass11, 17516 B) and validated BYTE-EXACT
// (0 differing bytes over the full 20880256 B scratch) vs the captured oracle
// by bench/tools/pass11_sim.py before this HLSL was authored.
//
// stage A: 3x3 conv, 16ch in -> 16ch out (4 iters x 1 dword), NO ReLU,
//          input plane @8342464 stride 7712, window rows ty..ty+2,
//          cols tx..tx+2, requant (>>8,+127), pack SClamp(2)
// stage B: 16ch passthrough from plane @10440128 at (ty+1, tx+1)
// stage C: pw 32->64 + ReLU (IMax), 16 iters x 1 dword, requant (>>8,+127)
// stage D: pw 64->32, 8 iters x 1 dword, float residual tail x8:
//          out = s8clamp(i16(round_ne((s8(ref)*2^-4 + acc*2^-12)*8))),
//          ref = INPUT plane @8342464 at (ty+1,tx+1) for m<4 (block 692
//          reloads %9[0..3] from u11 after stage C), else passthrough
// stage E: per (dy,dx,dw) in 2x2x4, pw 32ch -> 1 dword, float residual x32:
//          out = s8clamp(i16(round_ne((s8(resid)*2^-5 + acc*2^-12)*32))),
//          resid reads the BASE-0 tensor at the raw store index (DXIL loads
//          u11[%1252] with no output-base offset), written to base 12537792
//          + (2ty+dy+1)*15392 + (2tx+dx)*16+16 + 4dw (960x540, stride 15392)
// t18 map: W_A @113920[16][9][2][2] (strides 16/256/8/4), bias_A @116224[4][4],
//          W_C @116352[64][4][2] (strides 32/8/4), bias_C @118400[16][4],
//          W_D @118656[32][8][2] (strides 64/8/4), bias_D @120704[8][4],
//          W_E @120832[2][2][4][4][4][2] (strides 1024/512/128/32/8/4),
//          bias_E @122880[4][4]
// static op counts match container: 140 t18 loads, 27 u11 loads, ~273 dot4,
// 4x pack SClamp(2), 1 store, IMax only in stage C, guard dims CBV[11]=(480,270)
// ============================================================================
ByteAddressBuffer t18_initializer : register(t18);
RWByteAddressBuffer ScratchBuffer : register(u11);

cbuffer MLSR_Optimized_Constants : register(b0)
{
    uint4 ml2c_cb[17];
}

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
void reimpl_pass11(uint3 tid : SV_DispatchThreadID)
{
    const uint tx = tid.x;
    const uint ty = tid.y;
    if (tx > 479u || ty > 269u)
        return;

    // ---- stage A: 3x3 conv 16ch -> 16ch (4 dwords) ----
    uint conv_out[4];
    [loop]
    for (int k = 0; k < 4; k++) {
        const uint4 bA = t18_initializer.Load4(116224u + 16u * uint(k));
        int a0 = asint(bA[0]);
        int a1 = asint(bA[1]);
        int a2 = asint(bA[2]);
        int a3 = asint(bA[3]);
        [loop]
        for (int p = 0; p < 9; p++) {
            const uint rowA = ty + uint(p / 3);
            const uint colA = (tx + uint(p % 3)) * 16u;
            const uint4 px = ScratchBuffer.Load4(8342464u + rowA * 7712u + colA);
            [unroll]
            for (int d = 0; d < 4; d++) {
                const uint waddr = 113920u + 16u * uint(4 * k + d) + 256u * uint(p);
                const uint2 w0 = t18_initializer.Load2(waddr);
                const uint2 w1 = t18_initializer.Load2(waddr + 8u);
                if (d == 0) {
                    a0 = dot4add_i8packed(px[0], w0.x, a0);
                    a0 = dot4add_i8packed(px[1], w0.y, a0);
                    a0 = dot4add_i8packed(px[2], w1.x, a0);
                    a0 = dot4add_i8packed(px[3], w1.y, a0);
                } else if (d == 1) {
                    a1 = dot4add_i8packed(px[0], w0.x, a1);
                    a1 = dot4add_i8packed(px[1], w0.y, a1);
                    a1 = dot4add_i8packed(px[2], w1.x, a1);
                    a1 = dot4add_i8packed(px[3], w1.y, a1);
                } else if (d == 2) {
                    a2 = dot4add_i8packed(px[0], w0.x, a2);
                    a2 = dot4add_i8packed(px[1], w0.y, a2);
                    a2 = dot4add_i8packed(px[2], w1.x, a2);
                    a2 = dot4add_i8packed(px[3], w1.y, a2);
                } else {
                    a3 = dot4add_i8packed(px[0], w0.x, a3);
                    a3 = dot4add_i8packed(px[1], w0.y, a3);
                    a3 = dot4add_i8packed(px[2], w1.x, a3);
                    a3 = dot4add_i8packed(px[3], w1.y, a3);
                }
            }
        }
        int4 qA;
        qA[0] = requant_shift(a0, 8u, 127);
        qA[1] = requant_shift(a1, 8u, 127);
        qA[2] = requant_shift(a2, 8u, 127);
        qA[3] = requant_shift(a3, 8u, 127);
        conv_out[k] = pack_bytes(qA);
    }

    // ---- stage B: passthrough (kept in SSA regs for stage C) ----
    const uint passRow = 10440128u + (ty + 1u) * 7712u + (tx + 1u) * 16u;
    const uint ps0 = ScratchBuffer.Load(passRow);
    const uint ps1 = ScratchBuffer.Load(passRow + 4u);
    const uint ps2 = ScratchBuffer.Load(passRow + 8u);
    const uint ps3 = ScratchBuffer.Load(passRow + 12u);

    // ---- stage C: pw 32->64 + ReLU (16 dwords) ----
    uint feat[16];
    [loop]
    for (int l = 0; l < 16; l++) {
        const uint4 bC = t18_initializer.Load4(118400u + 16u * uint(l));
        int c0 = asint(bC[0]);
        int c1 = asint(bC[1]);
        int c2 = asint(bC[2]);
        int c3 = asint(bC[3]);
        [loop]
        for (int h = 0; h < 4; h++) {
            uint sa, sb;
            if (h == 0) { sa = conv_out[0]; sb = conv_out[1]; }
            else if (h == 1) { sa = conv_out[2]; sb = conv_out[3]; }
            else if (h == 2) { sa = ps0; sb = ps1; }
            else { sa = ps2; sb = ps3; }
            [unroll]
            for (int d = 0; d < 4; d++) {
                const uint2 w = t18_initializer.Load2(116352u + 32u * uint(4 * l + d) + 8u * uint(h));
                if (d == 0) {
                    c0 = dot4add_i8packed(sa, w.x, c0);
                    c0 = dot4add_i8packed(sb, w.y, c0);
                } else if (d == 1) {
                    c1 = dot4add_i8packed(sa, w.x, c1);
                    c1 = dot4add_i8packed(sb, w.y, c1);
                } else if (d == 2) {
                    c2 = dot4add_i8packed(sa, w.x, c2);
                    c2 = dot4add_i8packed(sb, w.y, c2);
                } else {
                    c3 = dot4add_i8packed(sa, w.x, c3);
                    c3 = dot4add_i8packed(sb, w.y, c3);
                }
            }
        }
        int4 qC;
        qC[0] = requant_shift(max(c0, 0), 8u, 127);
        qC[1] = requant_shift(max(c1, 0), 8u, 127);
        qC[2] = requant_shift(max(c2, 0), 8u, 127);
        qC[3] = requant_shift(max(c3, 0), 8u, 127);
        feat[l] = pack_bytes(qC);
    }

    // ---- stage D: pw 64->32 + float residual tail x8 ----
    uint mid[8];
    [loop]
    for (int m = 0; m < 8; m++) {
        const uint4 bD = t18_initializer.Load4(120704u + 16u * uint(m));
        int e0 = asint(bD[0]);
        int e1 = asint(bD[1]);
        int e2 = asint(bD[2]);
        int e3 = asint(bD[3]);
        [loop]
        for (int h = 0; h < 8; h++) {
            const uint sa = feat[2 * h];
            const uint sb = feat[2 * h + 1];
            [unroll]
            for (int d = 0; d < 4; d++) {
                const uint2 w = t18_initializer.Load2(118656u + 64u * uint(4 * m + d) + 8u * uint(h));
                if (d == 0) {
                    e0 = dot4add_i8packed(sa, w.x, e0);
                    e0 = dot4add_i8packed(sb, w.y, e0);
                } else if (d == 1) {
                    e1 = dot4add_i8packed(sa, w.x, e1);
                    e1 = dot4add_i8packed(sb, w.y, e1);
                } else if (d == 2) {
                    e2 = dot4add_i8packed(sa, w.x, e2);
                    e2 = dot4add_i8packed(sb, w.y, e2);
                } else {
                    e3 = dot4add_i8packed(sa, w.x, e3);
                    e3 = dot4add_i8packed(sb, w.y, e3);
                }
            }
        }
        // residual: input-plane pixel (block 692 reload) for m<4, passthrough else
        uint raddr;
        if (m < 4)
            raddr = 8342464u + (ty + 1u) * 7712u + (tx + 1u) * 16u + 4u * uint(m);
        else if (m == 4)
            raddr = passRow;
        else if (m == 5)
            raddr = passRow + 4u;
        else if (m == 6)
            raddr = passRow + 8u;
        else
            raddr = passRow + 12u;
        const uint rdw = ScratchBuffer.Load(raddr);
        int4 lanes;
        [unroll]
        for (int j = 0; j < 4; j++) {
            const int rb = (int)(((rdw >> (8 * j)) & 0xFFu) << 24) >> 24;
            const int av = (j == 0) ? e0 : ((j == 1) ? e1 : ((j == 2) ? e2 : e3));
            const float v = float(rb) * 0.0625f + float(av) * 0.000244140625f;
            lanes[j] = int16_wrap(int(round(v * 8.0f)));
        }
        mid[m] = pack_bytes(lanes);
    }

    // ---- stage E: 2x2 upsample tail x32 ----
    [loop]
    for (int dy = 0; dy < 2; dy++) {
        [loop]
        for (int dx = 0; dx < 2; dx++) {
            const uint rowE = uint(dy) + 2u * ty + 1u;
            const uint colE = (uint(dx) + 2u * tx) * 16u + 16u;
            [loop]
            for (int dw = 0; dw < 4; dw++) {
                const uint4 bE = t18_initializer.Load4(122880u + 16u * uint(dw));
                int f0 = asint(bE[0]);
                int f1 = asint(bE[1]);
                int f2 = asint(bE[2]);
                int f3 = asint(bE[3]);
                [loop]
                for (int h = 0; h < 4; h++) {
                    const uint sa = mid[2 * h];
                    const uint sb = mid[2 * h + 1];
                    [unroll]
                    for (int d = 0; d < 4; d++) {
                        const uint2 w = t18_initializer.Load2(120832u + 1024u * uint(dy) + 512u * uint(dx) + 32u * uint(4 * dw + d) + 8u * uint(h));
                        if (d == 0) {
                            f0 = dot4add_i8packed(sa, w.x, f0);
                            f0 = dot4add_i8packed(sb, w.y, f0);
                        } else if (d == 1) {
                            f1 = dot4add_i8packed(sa, w.x, f1);
                            f1 = dot4add_i8packed(sb, w.y, f1);
                        } else if (d == 2) {
                            f2 = dot4add_i8packed(sa, w.x, f2);
                            f2 = dot4add_i8packed(sb, w.y, f2);
                        } else {
                            f3 = dot4add_i8packed(sa, w.x, f3);
                            f3 = dot4add_i8packed(sb, w.y, f3);
                        }
                    }
                }
                // residual reads the base-0 tensor at the raw store index
                const uint rdw = ScratchBuffer.Load(rowE * 15392u + colE + 4u * uint(dw));
                int4 lanes;
                [unroll]
                for (int j = 0; j < 4; j++) {
                    const int rb = (int)(((rdw >> (8 * j)) & 0xFFu) << 24) >> 24;
                    const int av = (j == 0) ? f0 : ((j == 1) ? f1 : ((j == 2) ? f2 : f3));
                    const float v = float(rb) * 0.03125f + float(av) * 0.000244140625f;
                    lanes[j] = int16_wrap(int(round(v * 32.0f)));
                }
                const uint val = pack_bytes(lanes);
                const uint addr = 12537792u + rowE * 15392u + colE + 4u * uint(dw);
                ScratchBuffer.Store(addr, val);
            }
        }
    }
}
