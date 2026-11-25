"""MSL specialization for the hash-pinned V.B20 pack and Apple SIMD32 GPUs."""
import hashlib
import re
from string import Template


DESCRIPTOR = re.compile(r"struct spvDescriptorSetBuffer0\s*\{([^{}]+)\};")
FIELD = re.compile(r"^(\s*)device (\w+\*\s+(m_\d+)\s+\[\[id\(\d+\)\]\];)", re.M)
RESOURCE = re.compile(r"\bspvDescriptorSet0\.(m_\d+)\b")
ELEMENT = re.compile(r"\)\._m\d+\[")
STORE = re.compile(r"\s*(?:\)\s*)*(?:\+\+|--|(?:<<|>>|[-+*/%&|^])?=(?!=))")
REFERENCE = re.compile(r"\b(?:auto|device)\b[^;={}]*&\s*\w+\s*=")
CALL = re.compile(r"\b(\w+)(?:<\w+>)?\s*\(")
VALUE_CALLS = set("""if while for as_type abs all clamp dot exp floor isunordered length level log max min mix
    powr read round sample select sinh tanh spvFAdd spvFMul spvFSub spvFindUMSB""".split())
VALUE_TYPE = re.compile(r"(?:bool|char|short|ushort|int|uint|long|ulong|half|float)[234]?")


def may_write(statement, resource):
    # Accept only indexed loads. Raw pointers, decaying arrays (pushpull_tail),
    # references and addresses passed to atomics all retain write permission.
    start = resource.start() - 2
    element = ELEMENT.match(statement, resource.end())
    if statement[start:resource.start()] != "(*" or not element:
        return True
    prefix = statement[:start].rstrip().rstrip("(").rstrip()
    if prefix.endswith(("&", "++", "--")) or REFERENCE.search(prefix):
        return True
    position = element.end()
    depth = 1
    while depth and position < len(statement):
        char = statement[position]
        if char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
        position += 1
    suffix = statement[position:]
    # Only swizzles are value members in these storage blocks. Keep unfamiliar
    # secondary indexing, method calls, or new generated access forms writable.
    swizzle = re.match(r"(?:\.[xyzwrgba]+\b)*", suffix)
    remaining = suffix[swizzle.end():].lstrip()
    return depth != 0 or remaining.startswith(("[", ".", "->")) or STORE.match(remaining) is not None


def qualify_readonly(msl):
    descriptor = DESCRIPTOR.search(msl)
    if not descriptor:
        return msl
    functions = msl[descriptor.end():]
    kernel = re.search(r"\bkernel void \w+\s*\([^{}]+\)\s*\{", functions)
    if not kernel or "spvDescriptorSet0 [[buffer(0)]]" not in kernel[0]:
        return msl
    body = functions[:kernel.start()] + functions[kernel.end():]
    # If the descriptor table itself escapes, tracking just .m_N is unsound.
    if re.search(r"\bspvDescriptorSet0\b(?!\.m_\d+\b)", body):
        return msl
    fields = descriptor[1]
    candidates = {field[3] for field in FIELD.finditer(fields)}
    for statement in re.split(r"[;{}]", body):
        if "spvDescriptorSet0." not in statement:
            continue
        # Do not infer argument directions for unknown helpers taking references.
        # Atomics always take their destination by address (tested by may_write).
        calls = CALL.findall(statement)
        unknown_call = any(name not in VALUE_CALLS and not VALUE_TYPE.fullmatch(name)
                           and not name.startswith("atomic_") for name in calls)
        for resource in RESOURCE.finditer(statement):
            if resource[1] in candidates and (unknown_call or may_write(statement, resource)):
                candidates.remove(resource[1])

    def qualify(field):
        return field[1] + "const device " + field[2] if field[3] in candidates else field[0]

    return msl[:descriptor.start(1)] + FIELD.sub(qualify, fields) + msl[descriptor.end(1):]


def _test_descriptor_analysis():
    r"""Executable regression cases: python3 -m doctest tools/anr/opt_msl.py.

    >>> desc = 'struct spvDescriptorSetBuffer0 {\n    device _T* m_1 [[id(1)]];\n};\n'
    >>> signature = 'kernel void k(constant spvDescriptorSetBuffer0& spvDescriptorSet0 [[buffer(0)]]) {'
    >>> item = '(*spvDescriptorSet0.m_1)._m0[idx[sub[i]]]'
    >>> def readonly(code):
    ...     return 'const device' in qualify_readonly(desc + signature + code.replace('ITEM', item) + '}')
    >>> all(readonly(code) for code in ['v = ITEM;', 'x = ITEM.x == other;', 'if (ITEM.z <= 1) {}'])
    True
    >>> any(readonly(code) for code in ['ITEM = 1;', 'ITEM.xy += 1;', '++ITEM;', 'ITEM[1] = 0;'])
    False
    >>> any(readonly(code) for code in ['auto p = &ITEM; *p = 1;', 'auto& a = ITEM; a++;', 'mutate(ITEM);'])
    False
    >>> readonly('atomic_fetch_or_explicit((device atomic_uint*)&ITEM, 1u, memory_order_relaxed);')
    False
    >>> readonly('auto *p = (*spvDescriptorSet0.m_1)._m0; p[i] = 1;')
    False
    >>> readonly('auto& ds = spvDescriptorSet0; (*ds.m_1)._m0[i] = 1;')
    False
    """


# Generated SSA IDs are not an ABI. Only specialize audited, complete bodies;
# unknown versions of SPIRV-Cross keep the portable translation.
TRUST_TEX = "46ea968ea5ee00cd6770764d85c1d42a43da50a20d9a6dc7a6d3c24c1be52c30"
TRUST_LINEAR = "81bf9aa7484373059c143818684490d31bc83948942b4b1f2e42cf1b31b5da32"
PAIR_INPUTS = "cd555ad06f34c8a0d7b5e54bbd8b4667073303c2dad9e0d96f3b0e9ff1681fc7"

SIMD_HELPERS = """
// In a 16x16 Apple compute group, each SIMD32 spans two adjacent rows.
template<typename T>
inline T sum2x2(T v) {
    return ((v + simd_shuffle_xor(v, 1)) + simd_shuffle_xor(v, 16)) + simd_shuffle_xor(v, 17);
}

template<typename T>
inline T sum4x2(T v) {
    v += simd_shuffle_xor(v, 1);
    v += simd_shuffle_xor(v, 2);
    return v + simd_shuffle_xor(v, 16);
}

inline float4 mergeBounds(float4 a, float4 b) {
    return float4(precise::min(a.xy, b.xy), precise::max(a.zw, b.zw));
}

inline float4 bounds8x2(float4 v) {
    v = mergeBounds(v, simd_shuffle_xor(v, 1));
    v = mergeBounds(v, simd_shuffle_xor(v, 2));
    v = mergeBounds(v, simd_shuffle_xor(v, 4));
    return mergeBounds(v, simd_shuffle_xor(v, 16));
}

inline float4 bounds8x8(threadgroup const float4 *rows, uint quadrant) {
    const uint i = (quadrant / 2) * 8 + (quadrant % 2);
    return mergeBounds(mergeBounds(rows[i], rows[i+2]), mergeBounds(rows[i+4], rows[i+6]));
}
"""

TRUST_TAIL = Template("""
        const uint2 xy = gl_LocalInvocationID.xy;
        const uint strip = xy.y / 2;
        const uint tid = 16 * xy.y + xy.x;
        const auto params = *spvDescriptorSet0.$params;
        const float4 bounds = bounds8x2($valid ? float4($motion, $motion) : float4(1e30f, 1e30f, -1e30f, -1e30f));
        const float2 depths = simd_sum(float2($depth_a, $depth_b));
        const float4 pair = sum2x2(float4($value, $flow, $weight));
        const float3 flowSum = sum4x2(float3($flow, $weight));
        const float3 colorSum = sum4x2($color);
        if ((tid & 23u) == 0u) boundsRows[strip * 2 + xy.x / 8] = bounds;
        if ((tid & 31u) == 0u) depthRows[strip] = depths;
        if ((tid & 19u) == 0u) {
            const uint part = strip * 4 + xy.x / 4;
            flowRows[part] = flowSum;
            colorRows[part] = colorSum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < 4) {
            const uint2 pos = gl_WorkGroupID.xy * 2 + uint2(tid % 2, tid / 2);
            if (pos.x < params._m40 && pos.y < params._m41) {
                const uint index = (params._m41 + pos.y) * params._m40 + pos.x;
                const float4 b = bounds8x8(boundsRows, tid);
                (*spvDescriptorSet0.$bounds_out)._m0[index] = b;
                if (params._m43) {
                    const float2 peak = precise::max(abs(b.xy), abs(b.zw));
                    atomic_fetch_max_explicit((device atomic_uint*)&(*spvDescriptorSet0.$peak_out)._m0[0],
                                              as_type<uint>(precise::max(peak.x, peak.y)), memory_order_relaxed);
                }
                if (params._m42) (*spvDescriptorSet0.$copy_out)._m0[index] = b;
            }
        }
        if (tid == 0) {
            const uint index = 2 * (gl_WorkGroupID.y * params._m39 + gl_WorkGroupID.x);
            float2 sum = 0;
            for (uint i = 0; i < 8; ++i) sum += depthRows[i];
            (*spvDescriptorSet0.$depth_out)._m0[index] = sum.x;
            (*spvDescriptorSet0.$depth_out)._m0[index+1] = sum.y;
        }
        if ((tid & 17u) == 0u && pair.w > 0) {
            const uint2 pos = gl_WorkGroupID.xy * 8 + xy / 2;
            if (pos.x < params._m59 && pos.y < params._m58) {
                const uint index = pos.y * params._m59 + pos.x;
                (*spvDescriptorSet0.$half_value)._m0[index] = pair.x / pair.w;
                (*spvDescriptorSet0.$half_flow)._m0[index] = pair.yz / pair.w;
            }
        }
        if (tid < 16) {
            const uint2 pos = gl_WorkGroupID.xy * 4 + uint2(tid % 4, tid / 4);
            const uint part = (tid / 4) * 8 + tid % 4;
            const float3 fw = float3(flowRows[part]) + float3(flowRows[part+4]);
            if (pos.x < params._m25 && pos.y < params._m24 && fw.z > 0) {
                const uint index = pos.y * params._m25 + pos.x;
                const float mag = length(fw.xy / fw.z);
                const float plusOne = 1.0f + mag;
                (*spvDescriptorSet0.$quarter_mag)._m0[index] = plusOne == 1.0f ? mag : log(plusOne);
                const float3 rgb = float3(colorRows[part]) + float3(colorRows[part+4]);
                (*spvDescriptorSet0.$quarter_color)._m0[index] = float4(rgb / fw.z, 0);
            }
        }
    } while(false);
}
""")


def replace_region(body, start, end, replacement):
    left = body.index(start)
    right = body.index(end, left)
    return body[:left] + replacement + body[right:]


def trust_half(body, linear):
    # V.B20 emulates a float divide with an approximate quotient followed by a
    # search over +/-2 ULP. Request a precise Metal divide instead of the search.
    quotient = re.compile(r"(?P<i> +)float (?P<out>_\d+);\n(?P=i)do\n(?P=i)\{\n"
                          r"(?P=i)    float _\d+ = (?P<x>_\d+) / (?P<y>_\d+);.*?"
                          r"\n(?P=i)\} while\(false\);", re.S)
    body, count = quotient.subn(r"\g<i>float \g<out> = precise::divide(\g<x>, \g<y>);", body)
    if count != 6:
        raise ValueError("unexpected lowband divide count")
    body = replace_region(body, "    threadgroup", "    do\n", """    threadgroup float4 boundsRows[16];
    threadgroup float2 depthRows[8];
    threadgroup packed_float3 flowRows[32];
    threadgroup packed_float3 colorRows[32];
""")
    first = body.index("        bool2 _2067" if linear else "        bool2 _2057")
    # Bind both translations to the named inputs and outputs of the same tail.
    locals_start = 2059 if linear else 2049
    args = {name: f"_{number}" for number, name in enumerate(
        ("color", "weight", "flow", "value", "depth_b", "depth_a", "motion"), locals_start)}
    args["valid"] = "_322" if linear else "_324"
    args["peak_out"] = "m_9" if linear else "m_7"
    slot_offset = 1 if linear else 0
    for name, number in (("params", 37), ("bounds_out", 23), ("copy_out", 24), ("depth_out", 25),
                         ("half_value", 30), ("half_flow", 31), ("quarter_mag", 32), ("quarter_color", 33)):
        args[name] = f"m_{number + slot_offset}"
    return body[:first] + TRUST_TAIL.substitute(args)


def pair_inputs(body):
    # The warped position is already clamped. A tap beyond the last row/column
    # has zero fractional weight; reuse the last texel for that tap.
    for dx, dy, out, slot, first, end in (("_466", "_469", "_530", 4, "float _470 =", "float3 _540 ="),
                                          ("_545", "_546", "_616", 3, "float _547 =", "float3 _625 =")):
        sample = f"""float3 {out};
            {{
                const float2 pos = float2({dx}, {dy});
                const uint2 lo = uint2(pos);
                const uint2 hi = min(lo + 1u, uint2((*spvDescriptorSet0.m_37)._m19, (*spvDescriptorSet0.m_37)._m18) - 1u);
                const float2 f = pos - float2(lo);
                const float2 g = 1.0f - f;
                const uint row0 = lo.y * (*spvDescriptorSet0.m_37)._m19;
                const uint row1 = hi.y * (*spvDescriptorSet0.m_37)._m19;
                {out} = float3((*spvDescriptorSet0.m_{slot})._m0[row0 + lo.x].xyz) * (g.x * g.y);
                {out} += float3((*spvDescriptorSet0.m_{slot})._m0[row0 + hi.x].xyz) * (f.x * g.y);
                {out} += float3((*spvDescriptorSet0.m_{slot})._m0[row1 + lo.x].xyz) * (g.x * f.y);
                {out} += float3((*spvDescriptorSet0.m_{slot})._m0[row1 + hi.x].xyz) * (f.x * f.y);
            }}
            """
        body = replace_region(body, first, end, sample)
    body = body.replace("threadgroup spvUnsafeArray<float, 5376> _24;",
                        "threadgroup float rows[32][21];\n    threadgroup float4 boundsRows[16];")
    body = replace_region(body, "        uint _890;", "        if (_204 < 4u)", """        const uint x = gl_LocalInvocationID.x;
        const uint strip = gl_LocalInvocationID.y / 2;
        const float4 bounds = bounds8x2(_887);
        if ((_204 & 23u) == 0u) boundsRows[strip * 2 + x / 8] = bounds;
        #pragma clang loop unroll(full)
        for (uint c = 0; c < 21; ++c) {
            const float partial = sum4x2(_193[c]);
            if ((_204 & 19u) == 0u) rows[strip * 4 + x / 4][c] = partial;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = _204; i < 336; i += 256) {
            const uint tile = i / 21;
            const uint row = (tile / 4) * 8 + tile % 4;
            const uint c = i % 21;
            _25[i] = rows[row][c] + rows[row+4][c];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
""")
    return replace_region(body, "            float4 _959;", "            bool _1004;",
                          "            const float4 _959 = bounds8x8(boundsRows, _204);\n")


def optimize(msl, reflection):
    head, kernel = msl.split("kernel void ", 1)
    signature, body = kernel.split("{", 1)
    if reflection["entryPoints"][0]["threadGroupSize"] == [16, 16, 1]:
        key = hashlib.sha256(body.encode()).hexdigest()
        if key in (TRUST_TEX, TRUST_LINEAR, PAIR_INPUTS):
            head += SIMD_HELPERS
            if key == PAIR_INPUTS:
                body = pair_inputs(body)
            else:
                body = trust_half(body, key == TRUST_LINEAR)
    return qualify_readonly(head + "kernel void " + signature + "{" + body)
