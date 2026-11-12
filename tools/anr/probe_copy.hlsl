// SPDX-License-Identifier: MIT
// Diagnostic-only readback added after an unmodified ANR graph. Pad unused
// bytes to zero, so the upstream host's fixed image-size output is bounded.
RWByteAddressBuffer source : register(u0);
RWByteAddressBuffer output : register(u1);
cbuffer Probe : register(b0) { uint sourceBytes; uint outputBytes; };
[numthreads(256, 1, 1)]
void anr_probe_copy(uint index : SV_DispatchThreadID) {
    uint offset = index * 4;
    if (offset >= outputBytes) return;
    uint value = offset + 4 <= sourceBytes ? source.Load(offset) : 0;
    output.Store(offset, value);
}
