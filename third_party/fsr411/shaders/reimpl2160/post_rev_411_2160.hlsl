// SPDX-License-Identifier: MIT
// Reimplemented FSR 4.1.1 kernel: semantics reconstructed from the AMD 4.1.1
// provider (Kits/FidelityFX/signedbin/amd_fidelityfx_upscaler_dx12.dll,
// SHA-256 d0dcccc74a43c44ba435b7a369b456e0970d8a4464e4bd683119b374f2c9fb46,
// MIT-licensed per FidelityFX-SDK docs/license.md) and validated byte-exact
// against the AMD container. See README.md and docs/evidence411/.
// Copyright (C) 2025 Advanced Micro Devices, Inc. (upstream implementation)
// Copyright (C) 2026 int3rrobang (reimplementation)

// post_rev_411_2160.hlsl - 411 POST (Global 5132 / PASS_26) 2160-class variant.
// Derived from post_rev_411.hlsl (EXACT_411_REPLAY at 1080) by the 2x law
// confirmed against AMD 2160 POST container 0058_00248220_41bf3c2d8c68ea0e
// (45320B = same size as verified 1080 f5c9eaf26bba29d7; 22-line diff: only
// ten u11 UMads 15392->30752). All other semantics CBV-parameterized
// (width/height/width_lr/height_lr/scales from b1); weights/biases identical.
// u11 read plane at 2160 = pass12-2160 output (base 0, 1922x1082x16, stride 30752).
// dispatch 120,68,1 = (ceil((3840/2)/16), ceil((2160/2)/16)) over half-dims.
// post_rev_411_2160.hlsl (2160-class: u11 stride 15392->30752 x2 sites; see note below)
// post_rev_411.hlsl — clean-room 411 POST (Global 5132 / PASS_26).
//
// Reverses container f5c9eaf26bba29d7 (entry fsr4_model_v07_fp8_no_scale_postpass,
// numthreads 256x1x1, dispatch 60x34, root 26065: CBV b1 + SRV t3/t9/t17 +
// UAV u1/u2/u6 + raw u11, no sampler).
//
// Findings baked in (DXIL-measured):
// - Neural body structurally identical to 402 post (960/960 dot4 sites share
//   identical SSA/IMM operand patterns in identical order); only embedded
//   immediates differ: 960 weight dwords + 96 int accumulator seeds (biases
//   folded to int, scales folded: dw /256, pw1 relu + /128, pw2 raw int,
//   residual float with 0.0625 / 1/1024 / x16).
// - 16x16 input tiles, linear 256-lane groups, clip to 960x540; arena u11 is
//   962x542x16 int8 with 1-px padding (row stride 15392 B, element 16 B).
// - CT2D 2x2 per thread: recurrent ch (sigmoid, half, u6) + model ch (u1/u2
//   via 3x3 Gaussian + MuLaw + blend + optional RemoveMuLaw).
// - Exposure = t17[0].r / preExposure; colorspace convert iff rcas_enabled==0.

# include "post_rev_411_tables.hlsli"
// Colorspace transfers (apply/remove SRGB/PQ + the MuLaw pair with its
// NON_LINEAR identity branch). Same vendored copy the PRE kernels already
// compile against (byte-identical to the SDK header bar `#pragma once`), so
// no SDK include path is needed. The LINEAR arms below reproduce the previous
// hardcoded expressions exactly, so the shipped cs0 blob is unchanged.
# include "../pre_rev_411/ffx_fsr4_gpu_colorspace_common.h"

cbuffer MLSR_Optimized_Constants : register(b1)
{
    float2 inv_size;
    float2 scale;
    float2 inv_scale;
    float2 jitter;
    float2 mv_scale;
    float2 tex_size;
    float2 max_renderSize;
    float2 fMotionVectorJitterCancellation;

    uint width;
    uint height;
    uint reset;
    uint width_lr;

    uint height_lr;
    float preExposure;
    float previous_preExposure;
    uint rcas_enabled;

    float rcas_sharpness;
    float pad;
};

Texture2D<float3> r_input_color : register(t3);
Texture2D<half3> r_reprojected_color : register(t9);
Texture2D<float> r_auto_exposure_texture : register(t17);

RWTexture2D<float3> rw_history_color : register(u1);
RWTexture2D<float3> rw_mlsr_output_color : register(u2);
RWTexture2D<half4> rw_recurrent_0 : register(u6);
RWByteAddressBuffer ScratchBuffer : register(u11);

int rq256(int v) { return (v + 127 + ((v >> 8) & 1)) >> 8; }
int rq128relu(int v) { v = max(v, 0); return (v + 63 + ((v >> 7) & 1)) >> 7; }

// SDK ApplyMuLaw(x * exposureValue) with the FFX_MLSR_COLORSPACE composition:
// the LINEAR arm is the exact expression the shipped cs0 blob was built from.
float3 ApplyMuLaw411(float3 x, float exposure)
{
#if FFX_MLSR_COLORSPACE == FFX_MLSR_COLORSPACE_NON_LINEAR
    return max(x * exposure, 0.0f);
#elif FFX_MLSR_COLORSPACE == FFX_MLSR_COLORSPACE_SRGB
    return max(0.1174f * log(1.0f + 150.0f * RemoveSRGB(x * exposure)), 0.0f);
#elif FFX_MLSR_COLORSPACE == FFX_MLSR_COLORSPACE_PQ
    return max(0.1174f * log(1.0f + 150.0f * RemovePQ(x * exposure)), 0.0f);
#else
    return max(0.1174f * log(1.0f + 150.0f * x * exposure), 0.0f);
#endif
}

// SDK RemoveMuLaw with the same composition (identity for NON_LINEAR,
// re-encode for SRGB/PQ). LINEAR arm = the shipped expression verbatim.
float3 RemoveMuLaw411(float3 x)
{
#if FFX_MLSR_COLORSPACE == FFX_MLSR_COLORSPACE_NON_LINEAR
    return max(x, 0.0f);
#elif FFX_MLSR_COLORSPACE == FFX_MLSR_COLORSPACE_SRGB
    return ApplySRGB(max((exp(8.51788f * x) - 1.0f) * 0.006666666828095913f, 0.0f));
#elif FFX_MLSR_COLORSPACE == FFX_MLSR_COLORSPACE_PQ
    return ApplyPQ(max((exp(8.51788f * x) - 1.0f) * 0.006666666828095913f, 0.0f));
#else
    return max((exp(8.51788f * x) - 1.0f) * 0.006666666828095913f, 0.0f);
#endif
}

[numthreads(256, 1, 1)]
void fsr4_model_v07_fp8_no_scale_postpass(uint3 DTID : SV_DispatchThreadID, uint3 GID : SV_GroupID, uint3 GTID : SV_GroupThreadID)
{
    uint gx = GID.x, gy = GID.y;
    uint lane = GTID.x;
    uint x = (gx << 4) | (lane & 1u) | (((lane >> 1) & 14u));
    uint y = (gy << 4) + ((((lane >> 5) << 1) | ((lane >> 1) & 1u)));
    int ix = (int)x, iy = (int)y;
    int w2 = (int)width / 2, h2 = (int)height / 2;
    if (ix >= w2 || iy >= h2)
        return;

    // ---- depthwise 3x3x16 -> 16 (tap-major: tap = ky*3+kx) ----
    int acc[16];
    [unroll] for (uint f = 0; f < 16; f++) acc[f] = DW_S[f];
    [unroll] for (uint ky = 0; ky < 3; ky++)
        [unroll] for (uint kx = 0; kx < 3; kx++)
        {
            int tap = ky * 3 + kx;
            int lx = ix - 1 + (int)kx, ly = iy - 1 + (int)ky;
            if (lx < 0 || ly < 0 || lx >= w2 || ly >= h2)
                continue;
            uint addr = (uint)((iy + (int)ky) * 30752 + (ix + (int)kx) * 16);
            uint4 d = ScratchBuffer.Load4(addr);
            uint vs[4] = { d.x, d.y, d.z, d.w };
            [unroll] for (uint ff = 0; ff < 16; ff++)
            {
                int b = tap * 64 + ff * 4;
                acc[ff] = dot4add_i8packed(vs[0], DW_W[b], acc[ff]);
                acc[ff] = dot4add_i8packed(vs[1], DW_W[b + 1], acc[ff]);
                acc[ff] = dot4add_i8packed(vs[2], DW_W[b + 2], acc[ff]);
                acc[ff] = dot4add_i8packed(vs[3], DW_W[b + 3], acc[ff]);
            }
        }

    uint conv_result[4];
    [unroll] for (uint f2 = 0; f2 < 16; f2 += 4)
        conv_result[f2 / 4] = (uint)pack_clamp_s8(int4(rq256(acc[f2]), rq256(acc[f2 + 1]), rq256(acc[f2 + 2]), rq256(acc[f2 + 3])));

    // ---- pointwise 16->32 + relu ----
    int r_out[32];
    [unroll] for (uint i = 0; i < 32; i++) r_out[i] = PW1_S[i];
    [unroll] for (uint j = 0; j < 32; j++)
    {
        int b = j * 4;
        r_out[j] = dot4add_i8packed(conv_result[0], PW1_W[b], r_out[j]);
        r_out[j] = dot4add_i8packed(conv_result[1], PW1_W[b + 1], r_out[j]);
        r_out[j] = dot4add_i8packed(conv_result[2], PW1_W[b + 2], r_out[j]);
        r_out[j] = dot4add_i8packed(conv_result[3], PW1_W[b + 3], r_out[j]);
    }
    uint relu_result[8];
    [unroll] for (uint i2 = 0; i2 < 32; i2 += 4)
        relu_result[i2 / 4] = (uint)pack_clamp_s8(int4(rq128relu(r_out[i2]), rq128relu(r_out[i2 + 1]), rq128relu(r_out[i2 + 2]), rq128relu(r_out[i2 + 3])));

    // ---- pointwise 32->16 + residual (raw int acc, float add) ----
    int c3[16];
    [unroll] for (uint i3 = 0; i3 < 16; i3++) c3[i3] = PW2_S[i3];
    [unroll] for (uint k = 0; k < 16; k++)
    {
        int b = k * 8;
        c3[k] = dot4add_i8packed(relu_result[0], PW2_W[b], c3[k]);
        c3[k] = dot4add_i8packed(relu_result[1], PW2_W[b + 1], c3[k]);
        c3[k] = dot4add_i8packed(relu_result[2], PW2_W[b + 2], c3[k]);
        c3[k] = dot4add_i8packed(relu_result[3], PW2_W[b + 3], c3[k]);
        c3[k] = dot4add_i8packed(relu_result[4], PW2_W[b + 4], c3[k]);
        c3[k] = dot4add_i8packed(relu_result[5], PW2_W[b + 5], c3[k]);
        c3[k] = dot4add_i8packed(relu_result[6], PW2_W[b + 6], c3[k]);
        c3[k] = dot4add_i8packed(relu_result[7], PW2_W[b + 7], c3[k]);
    }
    uint4 rd = ScratchBuffer.Load4((uint)((iy + 1) * 30752 + (ix + 1) * 16));
    uint packedVss[4];
    [unroll] for (uint i4 = 0; i4 < 16; i4 += 4)
    {
        float4 a = float4((float)c3[i4], (float)c3[i4 + 1], (float)c3[i4 + 2], (float)c3[i4 + 3]) * 9.765625e-04f;
        float4 b4 = float4(unpack_s8s32(rd[i4 / 4])) * 6.25e-02f;
        int4 q = int4(round((b4 + a) * 16.0f));
        packedVss[i4 / 4] = (uint)pack_clamp_s8(q);
    }

    // LoadExposureValue() is LINEAR-only: every other colorspace returns 1.0
    // (ffx_fsr4_upscale_resources.h:187-197) and never reads t17.
#if FFX_MLSR_COLORSPACE == FFX_MLSR_COLORSPACE_LINEAR
    float exposure = r_auto_exposure_texture[int2(0, 0)].r / preExposure;
#else
    float exposure = 1.0f;
#endif
    bool convert = (rcas_enabled == 0);

    // ---- CT2D 2x2 + epilogue per output px ----
    [unroll] for (uint pk = 0; pk < 4; pk++)
    {
        int2 off = int2(pk & 1u, (pk >> 1) & 1u);
        uint2 tid = uint2((uint)(ix * 2) + (uint)off.x, (uint)(iy * 2) + (uint)off.y);

        int ct[8];
        [unroll] for (uint ch = 0; ch < 8; ch++) ct[ch] = CT2D_S[pk * 8 + ch];
        [unroll] for (uint ch2 = 0; ch2 < 8; ch2++)
        {
            int b = (pk * 32 + ch2 * 4);
            ct[ch2] = dot4add_i8packed(CT2D_W[b], packedVss[0], ct[ch2]);
            ct[ch2] = dot4add_i8packed(CT2D_W[b + 1], packedVss[1], ct[ch2]);
            ct[ch2] = dot4add_i8packed(CT2D_W[b + 2], packedVss[2], ct[ch2]);
            ct[ch2] = dot4add_i8packed(CT2D_W[b + 3], packedVss[3], ct[ch2]);
        }

        float4 rec = float4((float)ct[0], (float)ct[1], (float)ct[2], (float)ct[3]) * -0.00048828125f;
        rec = 1.0f / (1.0f + exp(rec));
        rw_recurrent_0[tid] = (half4)rec;

        float4 filt = float4((float)ct[4], (float)ct[5], (float)ct[6], (float)ct[7]) * 0.00048828125f;

        float corr = (exp(filt.x) - exp(-filt.x)) / (exp(filt.x) + exp(-filt.x));
        float sfx = 2.0f / (1.0f + exp(-filt.y));
        float sfy = 2.0f / (1.0f + exp(-filt.z));
        float sfx2 = sfx * sfx, sfy2 = sfy * sfy;
        float cc = (corr * 2.0f * sfx) * sfy;

        float2 lr_f = float2(inv_scale.x * ((float)tid.x + 0.5f) + (jitter.x - 0.5f),
                             inv_scale.y * ((float)tid.y + 0.5f) + (jitter.y - 0.5f));
        int2 lr_i = (int2)round(lr_f);

        float3 up = 0.0f;
        float tw = 0.0f;
        [unroll] for (uint dy = 0; dy < 3; dy++)
            [unroll] for (uint dx = 0; dx < 3; dx++)
            {
                int2 p = lr_i + int2((int)dx - 1, (int)dy - 1);
                int2 pc = clamp(p, int2(0, 0), int2((int)width_lr - 1, (int)height_lr - 1));
                float xd = ((float)(uint)p.x - lr_f.x) * scale.x;
                float yd = ((float)(uint)p.y - lr_f.y) * scale.y;
                float t = xd * (xd * sfx2 + yd * cc) + (yd * yd) * sfy2;
                float g = exp((-0.5f / (0.47f * 0.47f)) * t);
                tw += g;
                up += ApplyMuLaw411(r_input_color[pc], exposure) * g;
            }
        up /= tw;

        float bf = 1.0f / (1.0f + exp(-filt.w));
        float3 reproj = (float3)r_reprojected_color[tid];
        float3 res = up * (1.0f - bf) + reproj * bf;
        rw_history_color[tid] = res;

        float3 final_color = res;
        // Stock shape (container f5c9eaf26bba): K-scale INSIDE max; max-then-scale differs by 1 ULP at rounding boundaries.
        if (convert)
            final_color = RemoveMuLaw411(final_color) / exposure;
        final_color = clamp(final_color, 0.0f, 64000.0f);
        rw_mlsr_output_color[tid] = final_color;
    }
}
