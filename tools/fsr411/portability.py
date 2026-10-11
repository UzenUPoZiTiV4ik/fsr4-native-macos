"""Targeted fixes applied only to the generated FSR 4.1.1 include tree."""

import re


CHROMA_DISTANCE_BEFORE = "const float color_distance = sqrt(pow(input_ycbcr.y - history_ycbcr.y, 2) + pow(input_ycbcr.z - history_ycbcr.z, 2) + 1e-6);"

CHROMA_DISTANCE_AFTER = """const float chroma_delta_cb = input_ycbcr.y - history_ycbcr.y;
        const float chroma_delta_cr = input_ycbcr.z - history_ycbcr.z;
        const float color_distance = sqrt(chroma_delta_cb * chroma_delta_cb + chroma_delta_cr * chroma_delta_cr + 1e-6);"""


VELOCITY_BEFORE = """float2 LoadInputVelocity(int2 pos)
{
    float2 velocity = (InValidRegion(pos) ? r_velocity[pos].xy : float2(0.0f, 0.0f)) * mv_scale;
    #if FFX_MLSR_JITTERED_MOTION_VECTORS == 1
        velocity -= fMotionVectorJitterCancellation;
    #endif
    return velocity;
}"""

VELOCITY_AFTER = """float2 LoadInputVelocity(int2 pos)
{
    // Depth and input-resolution motion use the render viewport. With
    // output-resolution motion, dilate_mv returns output coordinates. Query
    // the canonical motion texture: width/height in b1 include 8-pixel padding,
    // while the supplied motion image may have an odd logical extent.
#if FFX_MLSR_LOW_RES_MV
    const int2 motionExtent = int2(width_lr, height_lr);
#else
    uint motionWidth, motionHeight;
    r_velocity.GetDimensions(motionWidth, motionHeight);
    const int2 motionExtent = int2(motionWidth, motionHeight);
#endif
    const bool motionInside = all(pos >= int2(0, 0)) && all(pos < motionExtent);
    const int2 motionLoadPos = clamp(pos, int2(0, 0), motionExtent - 1);
    const float2 motionSample = r_velocity[motionLoadPos].xy;
    float2 velocity = (motionInside ? motionSample : float2(0.0f, 0.0f)) * mv_scale;
    #if FFX_MLSR_JITTERED_MOTION_VECTORS == 1
        velocity -= fMotionVectorJitterCancellation;
    #endif
    return velocity;
}"""


def _signed_chroma_distance(text):
    # Native DXIL squares both signed chroma differences with fmul. SPIR-V
    # Pow becomes Metal powr, whose base must be nonnegative; negative chroma
    # differences can therefore corrupt the model's color-distance feature.
    # Express only these two known squares explicitly, before SPIR-V lowering.
    if "const float color_distance =" not in text:
        return text
    before, after = text.count(CHROMA_DISTANCE_BEFORE), text.count(CHROMA_DISTANCE_AFTER)
    if before == 0 and after == 1:
        return text
    if before != 1 or after != 0:
        raise ValueError("unrecognized FSR 4.1.1 PRE chroma distance")
    return text.replace(CHROMA_DISTANCE_BEFORE, CHROMA_DISTANCE_AFTER)


def _motion_viewport(text):
    """Fix the pinned helper's render-only MV guard without changing depth.

    Input textures are canonicalized to their logical active rectangles by
    fsr4_input_prepare/the runtime before this shader runs. Clamped reads and
    an explicit zero select preserve D3D's invalid-pixel behavior on Metal.
    An unknown helper fails instead of silently retaining a bad viewport gate.
    """
    if "float2 LoadInputVelocity(int2 pos)" not in text:
        return text
    if text.count(VELOCITY_AFTER) == 1:
        return text
    if text.count(VELOCITY_BEFORE) != 1:
        raise ValueError("unrecognized FSR 4.1.1 LoadInputVelocity source")
    return text.replace(VELOCITY_BEFORE, VELOCITY_AFTER)


def _discard_invalid_buffer_stores(text):
    # DXIL relies on its ByteAddressBuffer view to discard this dummy write.
    # MSL emits an unchecked device pointer, so the dummy address is unsafe;
    # 128 MiB is even inside the largest tier's scratch allocation.
    pattern = re.compile(r"(?P<indent>^[ \t]*)const uint addr = inRange\s*\?\s*"
                         r"(?P<address>[^;]+?)\s*:\s*134217728u;\s*"
                         r"ScratchBuffer.Store\(addr, val\);", re.M)
    def replace(match):
        indent = match["indent"]
        return (indent + "const uint addr = " + " ".join(match["address"].split()) + ";\n" +
                indent + "if (inRange) ScratchBuffer.Store(addr, val);")
    result = pattern.sub(replace, text)
    if re.search(r":\s*134217728u\s*;", result):
        raise ValueError("unrecognized FSR 4.1.1 discarded buffer store")
    return result


def apply_portability_fixes(text):
    """Preserve signed chroma squares and viewport/buffer guards in HLSL."""
    return _discard_invalid_buffer_stores(_motion_viewport(_signed_chroma_distance(text)))
