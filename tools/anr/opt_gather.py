# SPDX-License-Identifier: MIT
"""Local Metal specializations of three hash-pinned V.B20 compute passes."""
import hashlib
import re
import sys


# IDs below refer to SPIRV-Cross temporaries, not stable Slang symbols. Use
# the generic translator if a different version emits a different function.
_LANDING_KEY = "fg_lite_landing.slang:trust_phase_neither"
_PHASE_KEY = "fg.slang:phase_gather_sil_pd"
_SOURCE_HASH = {
    _LANDING_KEY: "09a776e7d1777211c161f9d1a3a9ffe94c8f9d2558a3184bafc6e8327996640c",
    _PHASE_KEY: "af5790dcdb3ccf22bb598aff41e008a430ca4af6b38b6e2513f91241030d7a96",
    "fg.slang:reverse_gather": "f64b6e32c9910020466921959b5fa920a55178da5ee7876f95a92840a8186b18",
}


_LANDING = r"""
struct Footprint {
    uint4 indices;
    float4 weights;
};

inline Footprint footprint(float2 coord, uint2 size) {
    float2 pos = clamp(coord, 0.0f, float2(size - 1u));
    uint2 base = uint2(pos);
    uint2 next = min(base + 1u, size - 1u);
    float2 frac = pos - float2(base);
    float2 inv = 1.0f - frac;
    // A tap beyond the clamped edge already has zero weight.
    uint row0 = size.x * base.y;
    uint row1 = size.x * next.y;
    return { uint4(row0 + base.x, row0 + next.x, row1 + base.x, row1 + next.x),
             float4(inv.x * inv.y, frac.x * inv.y, inv.x * frac.y, frac.x * frac.y) };
}

template<typename T>
inline float2 sample_motion(Footprint fp, const device T *values) {
    float2 sum = 0.0f;
    #pragma clang loop unroll(full)
    for (uint tap = 0; tap < 4; ++tap) {
        sum += float2(values[fp.indices[tap]]) * fp.weights[tap];
    }
    return sum;
}

inline float2 landing_motion(constant spvDescriptorSetBuffer0 &d, Footprint fp, uint mode, float scale) {
    if (!mode) return sample_motion(fp, d.m_11->_m0);
    if (mode != 2u) return sample_motion(fp, (const device short2 *)d.m_9->_m0) * scale;
    // The affine motion decode commutes with the normalized four-tap sum.
    const device float *params = d.m_10->_m0;
    uint flags = as_type<uint>(params[6]);
    float2 value = sample_motion(fp, (const device half2 *)d.m_9->_m0) * float2(params[0], params[1]);
    if (flags & 1u) value += float2(params[2], params[3]);
    if (flags & 2u) value.x *= params[4];
    if (flags & 4u) value.y *= params[5];
    return value * 0.5f;
}

inline float landing_log(float value) {
    float shifted = 1.0f + value;
    return shifted == 1.0f ? value : log(shifted);
}

kernel void ENTRY(constant spvDescriptorSetBuffer0 &d [[buffer(0)]],
                  uint2 group [[threadgroup_position_in_grid]],
                  uint tid [[thread_index_in_threadgroup]]) {
    constant auto &params = *d.m_23;
    if (tid == 0u && all(group == 0u)) d.m_3->_m0[0] = 0u;

    threadgroup float4 partials[32];
    uint2 pixel = group * 16u + uint2(tid & 15u, tid >> 4);
    uint2 size(params._m19, params._m18);
    float4 stats = 0.0f;
    if (all(pixel < size)) {
        uint index = pixel.y * size.x + pixel.x;
        bool packed = params._m44 != 0u;
        bool needsScale = packed || params._m29 || (params._m28 && params._m28 != 2u);
        float scale = needsScale ? max(2.0f * as_type<float>(d.m_5->_m0[0]), 64.0f) / 32767.0f : 0.0f;
        float4 displacement;
        if (packed) {
            uint2 bits = d.m_6->_m0[index];
            displacement = float4(as_type<short4>(bits)) * scale;
        } else {
            displacement = d.m_4->_m0[index];
        }
        Footprint prev = footprint(float2(pixel) + displacement.xy, size);
        Footprint curr = footprint(float2(pixel) + displacement.zw, size);
        float2 prevMotion = params._m29 ? sample_motion(prev, (const device short2 *)d.m_7->_m0) * scale
                                        : sample_motion(prev, d.m_8->_m0);
        float2 currMotion = landing_motion(d, curr, params._m28, scale);
        float2 confidence = packed ? float2(d.m_13->_m0[index]) : d.m_14->_m0[index];
        float2 missing = 1.0f - min(confidence * 4.0f, 1.0f);
        stats = float4(length(displacement.xy + prevMotion * params._m24),
                       length(displacement.zw + currMotion * (1.0f - params._m24)),
                       1.0f, missing.x * missing.y);
    }
    stats += simd_shuffle_xor(stats, 1);
    stats += simd_shuffle_xor(stats, 2);
    stats += simd_shuffle_xor(stats, 16);
    // Each SIMD32 owns two adjacent 16-pixel rows; four lanes publish 4x2 sums.
    if ((tid & 19u) == 0u) partials[(tid / 32u) * 4u + (tid & 15u) / 4u] = stats;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid >= 16u) return;
    uint partial = (tid / 4u) * 8u + (tid & 3u);
    stats = partials[partial] + partials[partial + 4u];
    uint2 coarse = group * 4u + uint2(tid & 3u, tid >> 2);
    uint2 coarseSize(params._m21, params._m20);
    if (any(coarse >= coarseSize)) return;
    uint dst = coarse.y * coarseSize.x + coarse.x;
    Footprint color = footprint(float2(coarse) + d.m_16->_m0[dst].zw * 0.25f, coarseSize);
    float3 warped = 0.0f;
    #pragma clang loop unroll(full)
    for (uint tap = 0; tap < 4; ++tap) {
        warped += d.m_17->_m0[color.indices[tap]].xyz * color.weights[tap];
    }
    d.m_18->_m0[dst] = float4(warped, landing_log(stats.x / stats.z));
    d.m_18->_m0[coarseSize.x * coarseSize.y + dst] = float4(d.m_17->_m0[dst].xyz, landing_log(stats.y / stats.z));
    d.m_19->_m0[dst] = stats.w / stats.z;
}
"""


def _readonly(msl, ids):
    fields = "|".join(str(field) for field in ids)
    return re.sub(r"    device (\w+\* m_(?:" + fields + r") )", r"    const device \1", msl)


def _compact_tile(body):
    # V.B20 indexes a 32x32 tile with a padded 40-float row pitch. Remove
    # padding in all five planes and at the start of reverse's mip scratch.
    return re.sub(r"\b(6400|5120|3840|2560|1280|40)u\b", lambda m: str(int(m[1]) * 4 // 5) + "u", body).replace(
        "spvUnsafeArray<float, 6400>", "spvUnsafeArray<float, 5120>")


_TILE = r"""
struct BinBatch {
    atomic_uint heads[33 * 33];
    float4 samples[512];
    float2 fractions[512];
    uint links[512];
};

union GatherTile {
    BinBatch bins;
    spvUnsafeArray<float, 5120> sums;
};
"""


_BIN_BATCH = r"""
        {
            // Each source has one floor cell; four target pixels can visit it.
            // Bin -1 is live at top/left. Packing 512 links and payloads into
            // the existing tile keeps the large-list fallback allocation-free.
            threadgroup auto &bins = storage.bins;
            uint tid = gl_LocalInvocationID.x;
            float4 accum[2] = {};
            float depth[2] = {};
            for (uint first = 0; first < COUNT; first += 8u) {
                for (uint bin = tid; bin < 1089u; bin += 512u) {
                    atomic_store_explicit(&bins.heads[bin], 512u, memory_order_relaxed);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                uint item = first + tid / 64u;
                if (item < COUNT) {
                    uint block = LIST[TILE * 128u + item];
                    uint sx = (block % PARAMS._m25) * 8u + (tid & 7u);
                    uint sy = (block / PARAMS._m25) * 8u + ((tid >> 3) & 7u);
                    if (sx < PARAMS._m24 && sy < PARAMS._m23) {
                        LOAD_SAMPLE
                        float2 base = floor(pos);
                        int2 local = int2(clamp(base - float2(gl_WorkGroupID.xy * 32u), -2.0f, 33.0f));
                        if (all(local >= -1) && all(local < 32)) {
                            uint bin = uint(local.y + 1) * 33u + uint(local.x + 1);
                            bins.links[tid] = atomic_exchange_explicit(&bins.heads[bin], tid, memory_order_relaxed);
                            bins.fractions[tid] = pos - base;
                            bins.samples[tid] = sample;
                        }
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                #pragma clang loop unroll(full)
                for (uint row = 0; row < 2; ++row) {
                    uint2 dst(tid & 31u, (tid >> 5) + 16u * row);
                    uint bottomRight = (dst.y + 1u) * 33u + dst.x + 1u;
                    #pragma clang loop unroll(full)
                    for (uint tap = 0; tap < 4; ++tap) {
                        uint bin = bottomRight - (tap & 1u) - (tap >> 1) * 33u;
                        uint node = atomic_load_explicit(&bins.heads[bin], memory_order_relaxed);
                        while (node != 512u) {
                            float2 frac = bins.fractions[node];
                            float weight = ((tap & 1u) ? frac.x : 1.0f - frac.x) *
                                           ((tap & 2u) ? frac.y : 1.0f - frac.y);
                            if (weight != 0.0f) {
                                float4 sample = bins.samples[node];
                                accum[row] += float4(sample.xyz, 1.0f) * weight;
                                if (DEPTH) depth[row] += sample.w * weight;
                            }
                            node = bins.links[node];
                        }
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            threadgroup auto &tile = storage.sums;
            for (uint row = 0; row < 2; ++row) {
                uint dst = tid + row * 512u;
                uint2 pixel = gl_WorkGroupID.xy * 32u + uint2(dst & 31u, dst >> 5);
                if (any(pixel >= uint2(PARAMS._m24, PARAMS._m23))) {
                    accum[row] = 0.0f;
                    depth[row] = 0.0f;
                }
                tile[dst] = accum[row].x;
                tile[dst + 1024u] = accum[row].y;
                tile[dst + 2048u] = accum[row].z;
                tile[dst + 3072u] = accum[row].w;
                if (DEPTH) tile[dst + 4096u] = depth[row];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
"""


def _bin_batch(body, *, condition, count, tile, params, list_buffer, count_buffer, leader,
               first, source_x, source_y, last, pos, motion, weight, depth=None):
    # Reuse upstream's full motion and depth decode and its entire overflow
    # scan. Only replace the short-list accumulator between those two points.
    count_expr = f"(*spvDescriptorSet0.m_{count_buffer})._m0[{tile}]"
    broadcast = body.rindex(f"if ({leader})", 0, body.index(" = " + count_expr))
    predicate = body.index(f"bool {condition} =", broadcast)
    body = body[:broadcast] + f"uint {count} = {count_expr};\n" + body[predicate:]
    start = body.index("{", body.index(f"if ({condition})"))
    end = body.index(f"uint {last};", start)
    prefix = body[body.index(f"uint {first} =", start):body.index(f"float2 {pos} =", start)]
    prefix = re.sub(r"\b" + source_x + r"\b", "sx", prefix)
    prefix = re.sub(r"\b" + source_y + r"\b", "sy", prefix)
    sample_depth = f"{depth} * {weight}" if depth else "0.0f"
    prefix += f"float2 pos = float2(sx, sy) + {motion};\n"
    prefix += f"float4 sample = float4(-{motion} * {weight}, {weight}, {sample_depth});"
    replacement = _BIN_BATCH.replace("LOAD_SAMPLE", prefix).replace("PARAMS", f"(*spvDescriptorSet0.m_{params})")
    replacement = replacement.replace("COUNT", count).replace("TILE", tile)
    replacement = replacement.replace("LIST", f"(*spvDescriptorSet0.m_{list_buffer})._m0").replace("DEPTH", "true" if depth else "false")
    return body[:start] + replacement + body[end:]


def _unroll_fixed_loops(body):
    # These short SPIRV-Cross loops copy/index private 4-tap arrays. Unroll
    # only fixed trip counts, not the data-dependent bin and overflow walks.
    return re.sub(r"for \(;;\)(?=\s*\{\s*(?:bool _\d+_ladder_break = false;\s*do\s*\{\s*)?\s*if \(!\(_\d+ < (?:2|4|7|8)u\)\))",
                  "#pragma clang loop unroll(full)\nfor (;;)", body)


def optimize_gather_msl(kernel_key, msl_code):
    if kernel_key not in _SOURCE_HASH:
        return msl_code
    canonical = re.sub(r"kernel void \w+\(", "kernel void ENTRY(", msl_code)
    if hashlib.sha256(canonical.encode()).hexdigest() != _SOURCE_HASH[kernel_key]:
        print(f"ANR: using generic translation for unrecognized {kernel_key} MSL", file=sys.stderr)
        return msl_code
    header, body = msl_code.split("kernel void ", 1)
    if kernel_key == _LANDING_KEY:
        header = _readonly(header, (4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 16, 17))
        return header + _LANDING.replace("ENTRY", body.split("(", 1)[0])
    body = _compact_tile(body)
    body = re.sub(r"threadgroup spvUnsafeArray<float, 5120> (_\d+);",
                  r"threadgroup GatherTile storage;\n    threadgroup auto &\1 = storage.sums;", body)
    if kernel_key == _PHASE_KEY:
        header = _readonly(header, (*range(3, 13), 15, 17, 18, 19, 21, 22))
        # The final reduction only reads 7*512 sums. Its 7*64 intermediate
        # results fit in the dead part of the same tile after the last barrier.
        body = body.replace("threadgroup spvUnsafeArray<float, 448> _28;", "threadgroup float *_28 = &_13[3584];")
        body = _bin_batch(body, condition="_301", count="_300", tile="_283", params=35, list_buffer=17,
                          count_buffer=15, leader="_240", first="_350", source_x="_329", source_y="_333",
                          last="_534", pos="_408", motion="_379", weight="_406", depth="_405")
        body = _bin_batch(body, condition="_898", count="_897", tile="_884", params=35, list_buffer=17,
                          count_buffer=15, leader="_240", first="_947", source_x="_926", source_y="_930",
                          last="_1188", pos="_1061", motion="_1032", weight="_1059", depth="_1058")
    else:
        header = _readonly(header, (5, 7, 27, 28, 10, 26, 24, 25, 8, 9))
        body = _bin_batch(body, condition="_221", count="_220", tile="_201", params=29, list_buffer=7,
                          count_buffer=5, leader="_215", first="_270", source_x="_249", source_y="_253",
                          last="_487", pos="_365", motion="_351", weight="_363")
    return header + _TILE + "kernel void " + _unroll_fixed_loops(body)
