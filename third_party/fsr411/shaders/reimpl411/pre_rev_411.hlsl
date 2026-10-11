// SPDX-License-Identifier: MIT
// Reimplemented FSR 4.1.1 kernel: semantics reconstructed from the AMD 4.1.1
// provider (Kits/FidelityFX/signedbin/amd_fidelityfx_upscaler_dx12.dll,
// SHA-256 d0dcccc74a43c44ba435b7a369b456e0970d8a4464e4bd683119b374f2c9fb46,
// MIT-licensed per FidelityFX-SDK docs/license.md) and validated byte-exact
// against the AMD container. See README.md and docs/evidence411/.
// Copyright (C) 2025 Advanced Micro Devices, Inc. (upstream implementation)
// Copyright (C) 2026 int3rrobang (reimplementation)

// pre_rev_411.hlsl — clean-room 411 PRE (Global 5093 / FFX_FSR4_PASS_0).
// Target container: c15c7bddd277af0e (17968B, entry fsr4_model_v07_fp8_no_scale_prepass).
// Scalar 402-style path retargeted to 411 bindings: single CB b1, t18
// InitializerBuffer weights, u11-before-u3 declaration order. v1:
// deadzone + MuLaw-0.0814 history round-trip + t18 weights + f32 bias x64.
#include "mlsr_optimized_includes.hlsli"

#include "ml2code_runtime/tensor_float16.hlsli"
#include "ml2code_runtime/tensor_int8.hlsli"

#define RESOLUTION_1080 0
#define RESOLUTION_2160 1
#define RESOLUTION_4320 2

#define GROUP_SIZE_ 16

// Samples a texture with Catmull-Rom filtering, using 9 texture fetches instead of 16.
bool SampleTextureCatmullRom(in float2 reprojectedPos, in float2 texSize, out float3 history_color)
{
    float2 samplePos = reprojectedPos * texSize;
    float2 texPos1 = floor(samplePos - 0.5f) + 0.5f;

    float2 f = samplePos - texPos1;

    float2 w0 = f * (-0.5f + f * (1.0f - 0.5f * f));
    float2 w1 = 1.0f + f * f * (-2.5f + 1.5f * f);
    float2 w2 = f * (0.5f + f * (2.0f - 1.5f * f));
    float2 w3 = f * f * (-0.5f + 0.5f * f);

    float2 w12 = w1 + w2;
    float2 offset12 = w2 / (w1 + w2);

    float2 texPos0 = texPos1 - 1;
    float2 texPos3 = texPos1 + 2;
    float2 texPos12 = texPos1 + offset12;

    texPos0 /= texSize;
    texPos3 /= texSize;
    texPos12 /= texSize;

    float3 result = 0.0f;
    result += r_history_color.SampleLevel(g_history_sampler, float2(texPos0.x, texPos0.y), 0.0f) * w0.x * w0.y;
    result += r_history_color.SampleLevel(g_history_sampler, float2(texPos12.x, texPos0.y), 0.0f) * w12.x * w0.y;
    result += r_history_color.SampleLevel(g_history_sampler, float2(texPos3.x, texPos0.y), 0.0f) * w3.x * w0.y;

    result += r_history_color.SampleLevel(g_history_sampler, float2(texPos0.x, texPos12.y), 0.0f) * w0.x * w12.y;
    result += r_history_color.SampleLevel(g_history_sampler, float2(texPos12.x, texPos12.y), 0.0f) * w12.x * w12.y;
    result += r_history_color.SampleLevel(g_history_sampler, float2(texPos3.x, texPos12.y), 0.0f) * w3.x * w12.y;

    result += r_history_color.SampleLevel(g_history_sampler, float2(texPos0.x, texPos3.y), 0.0f) * w0.x * w3.y;
    result += r_history_color.SampleLevel(g_history_sampler, float2(texPos12.x, texPos3.y), 0.0f) * w12.x * w3.y;
    result += r_history_color.SampleLevel(g_history_sampler, float2(texPos3.x, texPos3.y), 0.0f) * w3.x * w3.y;

    history_color = result;
    return true;
}

groupshared float myArray[16][16][16];

uint2 Swizzle1DthreadgroupForQuad(uint a, uint xTile, uint yTile)
{
    uint quadNum = a >> 2;
    uint quadIdx = a & 3;
    uint tileStartX = (quadNum % (xTile >> 1)) << 1;
    uint tileStartY = (quadNum / (xTile >> 1)) << 1;
    uint quadOffsetX = quadIdx & 1;
    uint quadOffsetY = quadIdx >> 1;
    return uint2(tileStartX |  quadOffsetX, tileStartY | quadOffsetY);
}

[numthreads(GROUP_SIZE_*GROUP_SIZE_, 1, 1)]
void fsr4_model_v07_fp8_no_scale_prepass(uint3 gid : SV_GroupID, uint3 LocalThreadId : SV_GroupThreadID, uint3 Dtid : SV_DispatchThreadID)
{
    uint quadNum = LocalThreadId.x >> 2;
    uint quadIdx = LocalThreadId.x & 3;
    uint tileStartX = (quadNum % (GROUP_SIZE_ >> 1)) << 1;
    uint tileStartY = (quadNum / (GROUP_SIZE_ >> 1)) << 1;
    uint quadOffsetX = quadIdx & 1;
    uint quadOffsetY = quadIdx >> 1;
    uint gtid_y = tileStartY | quadOffsetY;
    uint gx16 = gid.x << 4;
    uint gy16 = gid.y << 4;
    uint2 dtid = uint2((quadOffsetX | gx16) | tileStartX, gtid_y + gy16);
    uint2 gtid = uint2(tileStartX | quadOffsetX, gtid_y);

    if (dtid.x >= width || dtid.y >= height)
        return;

    const float exposureValue = LoadExposureValue();

    const uint dst_idx = dtid.y * width + dtid.x;
    float2 origin_offset = 0.5 * inv_scale - 0.5f;

    float4 upscale_input = optimized_input_upscale_user(dtid, origin_offset, exposureValue);

    const float history_rect_alpha = 0.1f;
    float temporal_sample_weight = upscale_input.a * history_rect_alpha;

    int2 mv_dialated_pos = dilate_mv(dtid, origin_offset, r_depth);
    float2 velocity = LoadInputVelocity(mv_dialated_pos);

    bool disocclusion = local_disocclusion_user(dtid, velocity);

    // 411: near-zero velocity deadzone (kill sub-1/32-texel motion).
    velocity.x = abs(velocity.x) <= inv_size.x * 0.03125f ? 0.0f : velocity.x;
    velocity.y = abs(velocity.y) <= inv_size.y * 0.03125f ? 0.0f : velocity.y;

    // on reset frames, model is trained to the current frame as both input and history
    // Otherwise (offscreen reprojection) the default is 0.0f
    float3 history = reset ? upscale_input.rgb : 0.f;

    float4 recurrent;
    // Init recurrent to 0
    recurrent = 0;

    float2 dstSize = { width, height };

    if (disocclusion == 0 && reset == 0)
    {
        ///-----------------------------------------
        /// Reprojection
        ///-----------------------------------------
        const float2 history_xy = float2(dtid)+velocity * dstSize;
        const float2 history_uv = (history_xy + 0.5f) / tex_size;

        bool onscreen = (0 <= history_xy.x && history_xy.x <= (width-1) && 0 <= history_xy.y && history_xy.y <= (height-1));

        if (onscreen)
        {
            //onscreen = Reproject(history_xy, dstSize, history);
            onscreen = SampleTextureCatmullRom(history_uv, tex_size, history);
            {
                recurrent = r_recurrent_0.SampleLevel(g_history_sampler, history_uv, 0);
            }
        }

        // 411: history color-domain round-trip. Only LINEAR history needs
        // previous/current exposure normalization; perceptual modes compile
        // exposure to 1.0 and do not declare an exposure SRV.
        history = max(history, 0);
        history = RemoveMuLaw(history);
#if FFX_MLSR_COLORSPACE == FFX_MLSR_COLORSPACE_LINEAR
        history = history / (r_auto_exposure_texture[int2(1, 0)].r / previous_preExposure);
#endif
        history = ApplyMuLaw(history * exposureValue);
        history = temporal_sample_weight * upscale_input.rgb + (1.0 - temporal_sample_weight) * history;
    }

    rw_reprojected_color[dtid.xy] = DATA_TYPE3(history);

    half4 downscaleInputs[2];

    // Y + Y + Color Distance (cbcr/euclidean)
    {
        float3 input_ycbcr = RGBtoYCbCr(upscale_input.xyz);
        float3 history_ycbcr = RGBtoYCbCr(history);
        const float color_distance = sqrt(pow(input_ycbcr.y - history_ycbcr.y, 2) + pow(input_ycbcr.z - history_ycbcr.z, 2) + 1e-6);

        downscaleInputs[0].x = MODEL_COLOR_NORMALIZE(NETWORK_DATA_TYPE(input_ycbcr.x));
        downscaleInputs[0].y = MODEL_COLOR_NORMALIZE(NETWORK_DATA_TYPE(history_ycbcr.x));
        downscaleInputs[0].z = NETWORK_DATA_TYPE(color_distance);

    }

    // RFM / Recurrent data
    {
        downscaleInputs[0].w = MODEL_RFM_NORMALIZE(NETWORK_DATA_TYPE(recurrent[0]));
        downscaleInputs[1].x = MODEL_RFM_NORMALIZE(NETWORK_DATA_TYPE(recurrent[1]));
        downscaleInputs[1].y = MODEL_RFM_NORMALIZE(NETWORK_DATA_TYPE(recurrent[2]));
        downscaleInputs[1].z = MODEL_RFM_NORMALIZE(NETWORK_DATA_TYPE(recurrent[3]));
        downscaleInputs[1].w = 0;
    }

    // Downscale (411: weights from InitializerBuffer t18, base byte 512).
    {
#if FFX_MLSR_RESOLUTION == RESOLUTION_1080
        const uint3 logicalSize_slice_0 = uint3(960, 540, 16);
        const int3 groupStart_slice_0 = int3(0, 0, 0) + int3(gid.xy, 1) * int3(8, 8, 16);
        const uint3 groupSize_slice_0 = uint3(8, 8, 16);
        const uint3 storageSize_slice_0 = uint3(962, 542, 16);
        const uint3 tensorByteStrides_slice_0 = uint3(16, 15392, 1);
#elif FFX_MLSR_RESOLUTION == RESOLUTION_2160
        const uint3 logicalSize_slice_0 = uint3(1920, 1080, 16);
        const int3 groupStart_slice_0 = int3(0, 0, 0) + int3(gid.xy, 1) * int3(8, 8, 16);
        const uint3 groupSize_slice_0 = uint3(8, 8, 16);
        const uint3 storageSize_slice_0 = uint3(1922, 1082, 16);
        const uint3 tensorByteStrides_slice_0 = uint3(16, 30752, 1);
#elif FFX_MLSR_RESOLUTION == RESOLUTION_4320
        const uint3 logicalSize_slice_0 = uint3(3840, 2160, 16);
        const int3 groupStart_slice_0 = int3(0, 0, 0) + int3(gid.xy, 1) * int3(8, 8, 16);
        const uint3 groupSize_slice_0 = uint3(8, 8, 16);
        const uint3 storageSize_slice_0 = uint3(3842, 2162, 16);
        const uint3 tensorByteStrides_slice_0 = uint3(16, 61472, 1);
#endif
        const int threadGroupByteOffsetInTensor_slice_0 = dot(groupStart_slice_0, tensorByteStrides_slice_0);
        const float quantizationScale_slice_0 = 0.015625f;
        const RWBufferStorage storage_slice_0 = { ScratchBuffer };
        const uint3 paddingBegin_slice_0 = uint3(1, 1, 0);
        const uint3 paddingEnd_slice_0 = uint3(1, 1, 0);
        const QuantizedTensor3i8_NHWC<RWBufferStorage> output = { logicalSize_slice_0, groupStart_slice_0, groupSize_slice_0, storageSize_slice_0, tensorByteStrides_slice_0, paddingBegin_slice_0, paddingEnd_slice_0, threadGroupByteOffsetInTensor_slice_0 + 0, quantizationScale_slice_0, storage_slice_0 };

        float accumulators[16];
        [unroll]
        for (uint f = 0 ; f < 16; f++)
        {
            accumulators[f] = 0;
        }

        const float rcpScale = 1.0f / quantizationScale_slice_0;
        uint2 downscaleLocalID = uint2(quadOffsetX, (LocalThreadId.x >> 1) & 1);

        float quadY[16];
        [unroll]
        for (uint f = 0 ; f < 16; f++)
        {
            const uint weightsByteOffset = mad(32, f, mad(512, downscaleLocalID.y, downscaleLocalID.x * 16));
            const uint4 weightDwords = InitializerBuffer.Load4(weightsByteOffset);

            half2 ws0 = Unpack2h(weightDwords.x);
            half2 ws1 = Unpack2h(weightDwords.y);
            accumulators[f] = dot2add(ws0, downscaleInputs[0].xy, accumulators[f]);
            accumulators[f] = dot2add(ws1, downscaleInputs[0].zw, accumulators[f]);
            half2 ws2 = Unpack2h(weightDwords.z);
            half2 ws3 = Unpack2h(weightDwords.w);
            accumulators[f] = dot2add(ws2, downscaleInputs[1].xy, accumulators[f]);
            accumulators[f] = dot2add(ws3, downscaleInputs[1].zw, accumulators[f]);

            // Add the results from the lanes in the quad to accumulators[f] .. f in [0..15]
            accumulators[f] += QuadReadAcrossX(accumulators[f]);
            quadY[f] = QuadReadAcrossY(accumulators[f]);
        }

        [unroll]
        for (uint f = 0 ; f < 16; f++)
        {
            accumulators[f] += quadY[f];
        }

        if (all(downscaleLocalID == 0))
        {
            // Add bias + Store (411: fp32 bias from t18 at byte 1024).
            uint4 storageBytes;
            uint4 bw[4];
            [unroll]
            for (uint b = 0; b < 4; b++)
            {
                bw[b] = InitializerBuffer.Load4(1024 + b * 16);
            }
            float4 vs = float4(accumulators[0], accumulators[1], accumulators[2], accumulators[3]);
            int4 quantized_vs = round((vs + asfloat(bw[0])) * rcpScale);
            storageBytes.x = pack_clamp_s8(quantized_vs);

            vs = float4(accumulators[4], accumulators[5], accumulators[6], accumulators[7]);
            quantized_vs = round((vs + asfloat(bw[1])) * rcpScale);
            storageBytes.y = pack_clamp_s8(quantized_vs);

            vs = float4(accumulators[8], accumulators[9], accumulators[10], accumulators[11]);
            quantized_vs = round((vs + asfloat(bw[2])) * rcpScale);
            storageBytes.z = pack_clamp_s8(quantized_vs);

            vs = float4(accumulators[12], accumulators[13], accumulators[14], accumulators[15]);
            quantized_vs = round((vs + asfloat(bw[3])) * rcpScale);
            storageBytes.w = pack_clamp_s8(quantized_vs);


            uint sxBase = gid.x << 3;
            uint syBase = gid.y << 3;
            uint lx = tileStartX >> 1;
            uint ly5 = LocalThreadId.x >> 5;
            uint sx = sxBase | lx;
            uint sy = (syBase | 1) + ly5;
            uint byteAddress = mad(15392, sy, (sx << 4) + 16);

            ScratchBuffer.Store4(byteAddress, storageBytes);
        }
    }
}
