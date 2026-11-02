# SPDX-License-Identifier: MIT
"""Independent D3D-style bounds checks for translated Metal resource IO only.

No model, temporal policy, arithmetic rewrite or upstream source is embedded.
The converter may retain uint2/uint4 alias views even with --ssbo-scalar.
Each guarded vector IO is decomposed into individually bounded uint words,
matching DXIL masks at partial buffer boundaries.
"""
from __future__ import annotations
import re

HELPERS = r"""
// Independent oracle IO ABI: preserve every valid access, zero invalid reads,
// discard invalid stores/atomics. No FSR arithmetic or temporal policy here.
inline uint OracleLoad(const device uint *p, uint i, uint n) { return i < n ? p[i] : 0u; }
inline void OracleStore(device uint *p, uint i, uint v, uint n) { if (i < n) p[i] = v; }
template<uint N>
inline vec<uint,N> OracleLoad(const device vec<uint,N> *p, uint i, uint n) {
    const device uint *words=reinterpret_cast<const device uint *>(p); vec<uint,N> result;
    for(uint c=0;c<N;++c) result[c]=OracleLoad(words,i*N+c,n); return result;
}
template<uint N>
inline void OracleStore(device vec<uint,N> *p, uint i, vec<uint,N> v, uint n) {
    device uint *words=reinterpret_cast<device uint *>(p);
    for(uint c=0;c<N;++c) OracleStore(words,i*N+c,v[c],n);
}
inline uint OracleAtomicMax(device uint *p, uint i, uint v, uint n) {
    return i < n ? atomic_fetch_max_explicit((device atomic_uint *)(p + i), v, memory_order_relaxed) : 0u;
}
template<typename T, access A>
inline vec<T,4> OracleTextureRead(texture2d<T,A> t, uint2 p, uint lod=0u) {
    return p.x < t.get_width(lod) && p.y < t.get_height(lod) ? t.read(p,lod) : vec<T,4>(0);
}
template<typename T, access A>
inline void OracleTextureWrite(texture2d<T,A> t, vec<T,4> v, uint2 p) {
    if (p.x < t.get_width() && p.y < t.get_height()) t.write(v,p);
}
"""


def _closing(text, start, opening="(", closing=")"):
    assert text[start] == opening
    depth = 1
    for end in range(start + 1, len(text)):
        depth += (text[end] == opening) - (text[end] == closing)
        if depth == 0:
            return end
    raise ValueError("unterminated Metal IO expression")


def _arguments(text):
    depth, start, result = 0, 0, []
    for i, char in enumerate(text):
        if char in "([{": depth += 1
        elif char in ")]}": depth -= 1
        elif char == "," and depth == 0:
            result.append(text[start:i].strip()); start = i + 1
    return result + [text[start:].strip()]


def guard_scalar_io(source, bindings, buffer_byte_lengths):
    if "OracleLoad(" in source:
        raise ValueError("already guarded")
    counts = {b["name"]: buffer_byte_lengths[b["d3d_register"]] // 4 for b in bindings
              if b["metal_kind"] == "buffer" and b["d3d_register"] in buffer_byte_lengths}
    # uint/uint2/uint4 views all use separately checked physical 32-bit words.
    signature = source[source.index("kernel void "):source.index("\n{", source.index("kernel void "))]
    for variable in counts:
        match = re.search(r"device (\w+)&(?: __restrict)? " + re.escape(variable), signature)
        if not match or not re.search(r"struct " + re.escape(match[1]) + r"\s*\{\s*uint(?:[24])? _m0\[1\];\s*\}", source):
            raise ValueError("unknown guarded SSBO view: " + variable)
    replacements = []
    atomic_sites = []
    for match in re.finditer(r"atomic_fetch_max_explicit\(", source):
        end = _closing(source, match.end() - 1)
        arguments = _arguments(source[match.end():end])
        if len(arguments) != 3:
            raise ValueError("unexpected atomic shape")
        pointer = re.fullmatch(r"\(device atomic_uint\*\)&(\w+)\._m0\[(.*)\]", arguments[0], re.S)
        if not pointer or pointer[1] not in counts or arguments[2] != "memory_order_relaxed":
            raise ValueError("unrecognized atomic resource access")
        variable, index = pointer.groups()
        replacements.append((match.start(), end + 1, f"OracleAtomicMax({variable}._m0, uint({index}), {arguments[1]}, {counts[variable]}u)"))
        atomic_sites.append((match.start(), end + 1))
    resource = re.compile(r"\b(" + "|".join(re.escape(n) for n in counts) + r")\._m0\[") if counts else None
    store_sites = []
    def reads(expression):
        substitutions = []
        for match in resource.finditer(expression):
            end = _closing(expression, match.end() - 1, "[", "]")
            variable, index = match[1], expression[match.end():end]
            substitutions.append((match.start(), end + 1, f"OracleLoad({variable}._m0, uint({index}), {counts[variable]}u)"))
        for start, end, replacement in reversed(substitutions):
            expression = expression[:start] + replacement + expression[end:]
        return expression
    if resource:
        for match in resource.finditer(source):
            if any(start <= match.start() < end for start, end in atomic_sites + store_sites):
                continue
            end = _closing(source, match.end() - 1, "[", "]")
            variable, index = match[1], source[match.end():end]
            tail = re.match(r"\s*=\s*(?!=)", source[end + 1:])
            if tail:
                rhs_start = end + 1 + tail.end()
                rhs_end = source.index(";", rhs_start)
                value = reads(source[rhs_start:rhs_end])
                replacements.append((match.start(), rhs_end + 1, f"OracleStore({variable}._m0, uint({index}), {value}, {counts[variable]}u);"))
                store_sites.append((match.start(), rhs_end + 1))
            else:
                replacements.append((match.start(), end + 1, f"OracleLoad({variable}._m0, uint({index}), {counts[variable]}u)"))
    for binding in bindings:
        if binding["metal_kind"] != "texture":
            continue
        variable = binding["name"]
        for match in re.finditer(r"\b" + re.escape(variable) + r"\.(read|write)\(", source):
            end = _closing(source, match.end() - 1)
            arguments = source[match.end():end]
            helper = "OracleTextureRead" if match[1] == "read" else "OracleTextureWrite"
            replacements.append((match.start(), end + 1, f"{helper}({variable}, {arguments})"))
    replacements.sort()
    if any(right[0] < left[1] for left, right in zip(replacements, replacements[1:])):
        raise ValueError("nested resource IO requires a separate lowering, not overlapping edits")
    for start, end, replacement in reversed(replacements):
        source = source[:start] + replacement + source[end:]
    entry = source.index("kernel void ")
    source = source[:entry] + HELPERS + "\n" + source[entry:]
    # Atomic declarations are only required by the IO shim, not every shader.
    if "#include <metal_atomic>" not in source:
        source = source.replace("#include <metal_stdlib>", "#include <metal_stdlib>\n#include <metal_atomic>")
    return source, {"schema": "fsr412_oracle_io_guard_v1", "access_sites": len(replacements),
                    "buffer_word_limits": counts, "semantics": "per-word zero/discard, texture integer read zero/write discard outside physical extent"}
