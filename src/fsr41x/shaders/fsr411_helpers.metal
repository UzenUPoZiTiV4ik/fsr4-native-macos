// SPDX-License-Identifier: MIT
#include <metal_stdlib>
using namespace metal;

// Every use has its own immutable, in-flight constant slot. Texture reads
// preserve the sampled colour domain; Metal handles the source pixel format.
struct Region { uint4 source; uint4 destination; };

kernel void fsr411_copy(texture2d<float, access::read> source [[texture(0)]],
                        texture2d<float, access::write> destination [[texture(1)]],
                        constant Region &region [[buffer(0)]], uint2 p [[thread_position_in_grid]]) {
    if (any(p >= region.source.zw)) return;
    destination.write(source.read(p + region.source.xy), p + region.destination.xy);
}

kernel void fsr411_copy_depth(depth2d<float, access::read> source [[texture(0)]],
                              texture2d<float, access::write> destination [[texture(1)]],
                              constant Region &region [[buffer(0)]], uint2 p [[thread_position_in_grid]]) {
    if (any(p >= region.source.zw)) return;
    destination.write(float4(source.read(p + region.source.xy)), p + region.destination.xy);
}

kernel void fsr411_clear_texture(texture2d<float, access::write> destination [[texture(0)]],
                                 uint2 p [[thread_position_in_grid]]) {
    if (p.x < destination.get_width() && p.y < destination.get_height()) destination.write(float4(0), p);
}

kernel void fsr411_clear_buffer(device uint4 *destination [[buffer(0)]],
                                constant uint &count [[buffer(1)]], uint p [[thread_position_in_grid]]) {
    if (p < count) destination[p] = uint4(0);
}

kernel void fsr411_snapshot(device const uint4 *source [[buffer(0)]],
                            device uint4 *destination [[buffer(1)]],
                            constant uint &count [[buffer(2)]], uint p [[thread_position_in_grid]]) {
    if (p < count) destination[p] = source[p];
}

// POST always reads the internal 2x1 exposure pair, even when PRE uses an
// external 1x1 input. Keep the previous value per temporal owner on the GPU.
kernel void fsr411_exposure(texture2d<float, access::read> source [[texture(0)]],
                            texture2d<float, access::read_write> history [[texture(1)]],
                            constant uint &reset [[buffer(0)]]) {
    float current = source.read(uint2(0)).x;
    // Same neutral convention as LoadExposureValue() for an unset texture.
    if (current == 0.0f) current = 1.0f;
    float previous = reset ? current : history.read(uint2(0)).x;
    history.write(float4(previous), uint2(1, 0));
    history.write(float4(current), uint2(0));
}

kernel void fsr411_init_exposure(texture2d<float, access::write> history [[texture(0)]]) {
    history.write(float4(1), uint2(0));
    history.write(float4(1), uint2(1, 0));
}

// Metal 4 command buffers belong to D3DMetal and cannot accept a completion
// handler here. This token retires ONLY our uniforms/tables/temporary holds;
// it is written after a device barrier and all FSR accesses in this encoder.
kernel void fsr411_retire(device atomic_uint *done [[buffer(0)]]) {
    // Metal 2.3 only exposes relaxed atomics. The separate preceding encoder
    // barrier supplies ordering for resource writes before this dispatch.
    atomic_store_explicit(done, 1u, memory_order_relaxed);
}
