// SPDX-License-Identifier: MIT
// Reimplemented FSR 4.1.1 kernel: semantics reconstructed from the AMD 4.1.1
// provider (Kits/FidelityFX/signedbin/amd_fidelityfx_upscaler_dx12.dll,
// SHA-256 d0dcccc74a43c44ba435b7a369b456e0970d8a4464e4bd683119b374f2c9fb46,
// MIT-licensed per FidelityFX-SDK docs/license.md) and validated byte-exact
// against the AMD container. See README.md and docs/evidence411/.
// Copyright (C) 2025 Advanced Micro Devices, Inc. (upstream implementation)
// Copyright (C) 2026 int3rrobang (reimplementation)

// 411 epilogue 4320-class reconstruction: fsr4_model_v07_fp8_no_scale_pass0_post (AMD 2160 sha 581a39d1f3cc)
// Transformed from epi411/*_2160.hlsl (proven 2160) by the 2x law (guards 2x-1, stride 2x, bases per AMD 4320 container); original 1080 in epi411/pass0_post.hlsl by the confirmed 2x law (slot unchanged;
// subA/subB/base/deltas from the AMD 2160 container immediates; zero-fill
// region scales 4x area vs 1080). dispatch/k documented in 411_2160 record.
cbuffer CsTensorSizes : register(b0)
{
    uint4 tensorSizes[17];
};
RWByteAddressBuffer ScratchBuffer : register(u11);

[numthreads(32, 1, 1)]
void main(uint3 dtid : SV_DispatchThreadID)
{
    uint tid = dtid.x;
    uint x = tensorSizes[0].x;
    uint y = tensorSizes[0].y;
    uint a = 3841u - x;
    uint b = 2161u - y;
    uint c = min(a, 5u);
    uint d = min(b, 1u);
    uint e = x + 1;
    uint f = e + c;
    uint g = y * c;
    uint h = f * d;
    uint i = f + y;
    uint j = i + g;
    uint k = j + h;
    if (tid >= k) return;
    uint ox, oy, idx, div;
    if (tid < f) { ox = 0; oy = 0; idx = tid; div = f; }
    else if (tid < i) { ox = 0; oy = 1; idx = tid - f; div = 1; }
    else if (tid < j) { ox = e; oy = 1; idx = (tid - y) - f; div = c; }
    else { ox = 0; oy = y + 1; idx = ((tid - y) - g) - f; div = f; }
    int q = int(idx) / int(div);
    int r = int(idx) % int(div);
    uint ox2 = uint(r) + ox;
    uint oy2 = uint(q) + oy;
    int base = int(61472u * oy2 + ox2 * 16);
    { int _g0 = base + int(0u); if (_g0 >= 0) ScratchBuffer.Store4(uint(_g0), uint4(0, 0, 0, 0)); }
}
