"""Structural Metal optimizations for the imported AMD INT8 Lite graph.

The neural weights and temporal/filter expressions remain those of the
imported model. Transforms reject unknown source shapes instead of guessing.
"""
from __future__ import annotations
import re

REVISION = 2


def closing(text, start, left="(", right=")"):
    assert text[start] == left
    depth = 0
    for index in range(start, len(text)):
        if text[index] == left:
            depth += 1
        elif text[index] == right:
            depth -= 1
            if depth == 0:
                return index
    raise ValueError("unbalanced imported Metal expression")


def input_color_cache(text: str, expected: int) -> tuple[str, str]:
    """Reuse the same float32 MuLaw result in PRE and POST within one frame."""
    replacements = []
    formula = None
    for match in re.finditer(r"precise::max\(log\(", text):
        end = closing(text, text.index("(", match.start())) + 1
        expression = text[match.start():end]
        if "r_input_color.read(" not in expression:
            continue
        read = expression.index("r_input_color.read(")
        rstart = expression.index("(", read)
        rend = closing(expression, rstart)
        arguments = expression[rstart + 1:rend]
        if not arguments.endswith(", 0u"):
            raise ValueError("Lite color load mip changed")
        position = arguments[:-4]
        sample = expression[:read] + "r_input_color.read(pixel, 0u)" + expression[rend + 1:]
        # The only nonliteral value outside the source texel is the exposure.
        sample = re.sub(r"(?<=\* )_\d+(?=\) \* 150.0)", "exposure", sample)
        if formula is not None and formula != sample:
            raise ValueError("Lite input color transforms differ")
        formula = sample
        replacements.append((match.start(), end, f"r_prepared_color.read({position}, 0u).xyz"))
    if len(replacements) != expected:
        raise ValueError(f"Lite color transform count changed: {len(replacements)} != {expected}")
    for start, end, replacement in reversed(replacements):
        text = text[:start] + replacement + text[end:]
    group = "uint3 gl_WorkGroupID [[threadgroup_position_in_grid]]"
    if text.count(group) != 1 or "[[texture(8)]]" in text:
        raise ValueError("Lite color cache binding/dispatch signature changed")
    text = text.replace(group, "texture2d<float> r_prepared_color [[texture(8)]], " + group, 1)
    return text, formula


def color_cache_source(text: str, formula: str) -> str:
    constants = re.search(r"struct type_MLSR_Optimized_Constants\n\{.*?\n\};", text, re.S)
    if not constants:
        raise ValueError("Lite frame constants not found")
    return """#include <metal_stdlib>
using namespace metal;
""" + constants[0] + """
kernel void fsr412_prepare_color(constant type_MLSR_Optimized_Constants& MLSR_Optimized_Constants [[buffer(0)]],
    texture2d<float> r_input_color [[texture(0)]], texture2d<float> r_auto_exposure_texture [[texture(1)]],
    texture2d<float, access::write> rw_prepared_color [[texture(2)]], uint2 pixel [[thread_position_in_grid]]) {
    if (any(pixel >= uint2(MLSR_Optimized_Constants.width_lr, MLSR_Optimized_Constants.height_lr))) return;
    float exposure = r_auto_exposure_texture.read(uint2(0u), 0u).x / MLSR_Optimized_Constants.preExposure;
    float3 value = """ + formula + ";\n    rw_prepared_color.write(float4(value, 0.0), pixel);\n}\n"
