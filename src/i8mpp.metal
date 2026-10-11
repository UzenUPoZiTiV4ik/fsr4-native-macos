// Native Metal 4 int8 FSR4 model passes on mpp::tensor_ops (M5 neural
// accelerators). Same int8 NHWC scratch tensors as the generated int8 HLSL
// passes, same int32 accumulation and the same fp32 requant expressions as
// the MSL that DXC+spirv-cross produced for them (see tools/gen_i8mpp.py),
// so results match the int8 reference path.
//
// Every pass is one threadgroup (128 threads, 4 simdgroups) per tile of T
// output pixels:
//   3x3 convs      convolution2d  int8 x int8 -> int32 (threadgroup tile)
//   pointwise      matmul2d       [T][K] int8 (threadgroup) x [K][N] int8 (device)
//   requant        fp32, rint, clamp -> int8 threadgroup A operand
// Requant loops handle four channels per step (int4 accumulator load, one
// packed uint store); a thread's channel group is fixed across its loop, so
// the bias is loaded once per thread.
// Compile with -std=metal4.0 and -D MAXW=<enc0 width> -D MAXH=<enc0 height>
// (per output tier; the descriptors only need an upper bound, runtime tensor
// extents do the bounds/zero-padding).
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

#ifndef MAXW
#define MAXW 960
#endif
#ifndef MAXH
#define MAXH 540
#endif

struct I8P {
    uint inOff, inSX, inSY, inW, inH;
    uint outOff, outSX, outSY, outW, outH;
    uint addOff, addSX, addSY, pad0;
    uint w0, w1, w2, w3, b0, b1, b2, b3;
    float s[16];
    // pass 13 (i8_cnb_ct2d_tap4): the post pass's four model parameters per
    // output pixel, half4 at prmOff + oy*prmSY + ox*8, for ox<prmW, oy<prmH
    uint prmOff, prmSY, prmW, prmH;
};

static inline uint pack4(int a, int b, int c, int d) {
    return (uint)(a & 0xFF) | ((uint)(b & 0xFF) << 8) | ((uint)(c & 0xFF) << 16) | ((uint)(d & 0xFF) << 24);
}
// Saturate before narrowing, then store all four channels with one word.
static inline uint pack_s8(float4 v) { return as_type<uint>(char4(clamp(int4(rint(v)), int4(-128), int4(127)))); }
static inline uint pack_u7(float4 v) { return as_type<uint>(char4(clamp(int4(rint(v)), int4(0), int4(127)))); }
static inline int4 unpack4(uint v) { return int4(as_type<char4>(v)); }
static inline int4 ld4(threadgroup const int32_t *c, uint i) { return *(threadgroup const int4 *)(c + i); }
static inline float4 bias4(device const half *b, uint c) { return float4(*(device const half4 *)(b + c)); }

// ---------------------------------------------------------------------------
// 3x3 conv (C=16 in, O=16 out) of a channel slice of the input tensor into a
// threadgroup int32 tile [TH][TW][16].
// ---------------------------------------------------------------------------
template <int TW, int TH, int W, int H>
static inline void conv3x3_16(device int8_t *act, uint sx, uint sy, uint w, uint h,
                              device int8_t *wts, threadgroup int32_t *tile, uint2 tgid)
{
    auto tA = tensor<device int8_t, dextents<int32_t, 4>, tensor_inline>(
        act, dextents<int32_t, 4>(16, (int)w, (int)h, 1),
        array<int, 4>{1, (int)sx, (int)sy, (int)(sy * h)});
    auto tW = tensor<device int8_t, extents<int32_t, 16, 16, 3, 3>, tensor_inline>(
        wts, extents<int32_t, 16, 16, 3, 3>());
    auto tD = tensor<threadgroup int32_t, extents<int32_t, 16, TW, TH, 1>, tensor_inline>(
        tile, extents<int32_t, 16, TW, TH, 1>());
    constexpr auto d = convolution2d_descriptor(
        int4(16, TW, TH, 1), int4(16, W, H, 1), int2(3, 3),
        convolution2d_activation_layout::nhwc, convolution2d_weights_layout::hwio,
        int2(1, 1), int2(1, 1), 1, false, convolution2d_descriptor::mode::multiply);
    convolution2d<d, execution_simdgroups<4>> op;
    op.set_offsets(int2((int)(tgid.x * TW), (int)(tgid.y * TH)));
    op.run(tA, tW, tD);
}

// C[T][N] (int32, threadgroup) = A[T][K] (int8, threadgroup) x B[K][N] (int8, device).
// These tiles have fixed sizes; expose their extents to the MPP compiler.
template <int T, int N, int K>
static inline void gemm_tg(threadgroup int8_t *A, device int8_t *B, threadgroup int32_t *C)
{
    auto tA = tensor<threadgroup int8_t, extents<int32_t, K, T>, tensor_inline>(A, extents<int32_t, K, T>());
    auto tB = tensor<device int8_t, extents<int32_t, N, K>, tensor_inline>(B, extents<int32_t, N, K>());
    auto tC = tensor<threadgroup int32_t, extents<int32_t, N, T>, tensor_inline>(C, extents<int32_t, N, T>());
    constexpr auto d = matmul2d_descriptor(T, N, K, false, false, false,
                                           matmul2d_descriptor::mode::multiply);
    matmul2d<d, execution_simdgroups<4>> op;
    op.run(tA, tB, tC);
}

// ---------------------------------------------------------------------------
// ConvNextBlock body (passes 1/2/12/13): 3x3 16->16 (+b0, requant) -> a1,
// 1x1 16->32 + relu -> a2, 1x1 32->16 -> c32 [T][16]. s: [K1, R0, K2, R1, K3, IQ, RO]
// ---------------------------------------------------------------------------
template <int TW, int TH>
static inline void cnb_core(device uchar *scratch, device uchar *wgt, constant I8P &p,
                            threadgroup int32_t *c32, threadgroup int8_t *a1, threadgroup int8_t *a2,
                            uint2 tgid, uint tid)
{
    constexpr int T = TW * TH;
    device int8_t *in = (device int8_t *)(scratch + p.inOff);
    conv3x3_16<TW, TH, MAXW, MAXH>(in, p.inSX, p.inSY, p.inW, p.inH,
                                   (device int8_t *)(wgt + p.w0), c32, tgid);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device const half *b0 = (device const half *)(wgt + p.b0);
    device const half *b1 = (device const half *)(wgt + p.b1);
    // c1: (acc*K1 + b0) * R0
    {
        const uint c4 = (tid & 3) * 4;
        const float4 bb = bias4(b0, c4);
        for (uint i = tid; i < T * 4; i += 128) {
            const uint px = i >> 2;
            const float4 v = (float4(ld4(c32, px * 16 + c4)) * p.s[0] + bb) * p.s[1];
            *(threadgroup uint *)(a1 + px * 16 + c4) = pack_s8(v);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    gemm_tg<T, 32, 16>(a1, (device int8_t *)(wgt + p.w1), c32);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // c2: relu((acc*K2 + b1) * R1)
    {
        const uint c4 = (tid & 7) * 4;
        const float4 bb = bias4(b1, c4);
        for (uint i = tid; i < T * 8; i += 128) {
            const uint px = i >> 3;
            const float4 v = (float4(ld4(c32, px * 32 + c4)) * p.s[2] + bb) * p.s[3];
            *(threadgroup uint *)(a2 + px * 32 + c4) = pack_u7(v);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    gemm_tg<T, 16, 32>(a2, (device int8_t *)(wgt + p.w2), c32);
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// c3 + residual for pixel px / channel group c4: (((acc*K3) + b2) + in*IQ) * RO
template <int TW, int TH>
static inline float4 cnb_c3(device uchar *scratch, constant I8P &p, threadgroup int32_t *c32,
                            device const half *b2, uint px, uint c4, uint x, uint y)
{
    const uint inB = p.inOff + y * p.inSY + x * p.inSX + c4;
    const int4 iv = unpack4(*(device const uint *)(scratch + inB));
    const float4 bb = bias4(b2, c4);
    return ((float4(ld4(c32, px * 16 + c4)) * p.s[4] + bb) + float4(iv) * p.s[5]) * p.s[6];
}

kernel void i8_cnb(device uchar *scratch [[buffer(0)]],
                   device uchar *wgt [[buffer(1)]],
                   constant I8P &p [[buffer(2)]],
                   uint2 tgid [[threadgroup_position_in_grid]],
                   uint tid [[thread_index_in_threadgroup]])
{
    constexpr int TW = 32, TH = 4, T = TW * TH;
    threadgroup int4 c32v[T * 32 / 4];
    // The expansion consumes a1 before producing a2. cnb_core barriers
    // separate their lifetimes; share the larger, 32-channel activation tile.
    threadgroup uint activations[T * 32 / 4];
    threadgroup int32_t *c32 = (threadgroup int32_t *)c32v;
    threadgroup int8_t *a1 = (threadgroup int8_t *)activations;
    threadgroup int8_t *a2 = a1;

    cnb_core<TW, TH>(scratch, wgt, p, c32, a1, a2, tgid, tid);
    device const half *b2 = (device const half *)(wgt + p.b2);
    const uint c4 = (tid & 3) * 4;
    for (uint i = tid; i < T * 4; i += 128) {
        const uint px = i >> 2;
        const uint lx = px % TW, ly = px / TW;
        const uint x = tgid.x * TW + lx, y = tgid.y * TH + ly;
        if (x >= p.outW || y >= p.outH) continue;
        const float4 r = cnb_c3<TW, TH>(scratch, p, c32, b2, px, c4, x, y);
        *(device uint *)(scratch + p.outOff + y * p.outSY + x * p.outSX + c4) = pack_s8(r);
    }
}

// ---------------------------------------------------------------------------
// Pass 13: the post pass's ConvNextBlock followed by its 2x2 s2
// ConvTranspose (16 -> 8: four model parameters + four recurrent channels),
// i.e. FusedConv2D_DW_Conv2D_PW_Relu_Conv2D_Add_ConvTranspose2D up to the
// filter. The CNB result stays in threadgroup memory; all four taps share
// one matmul2d [T][16] x [16][32], with columns [tap][8 channels]. This
// reuses the CNB accumulator storage and needs no padded zero channels.
// Per output pixel, exactly the reference expressions (post_common.hlsli
// ct2d_iteration_4 as compiled to MSL):
//   v         = half4(float4(acc) * scale) + half4(bias)
//   recurrent = float4(half4(1 / (1 + exp(-v))))      -> recurrent texture
//   params    = v                                     -> half4 at prmOff
// s[7] = cnbOutputScale * ct2d_weights.quantizationScale; b3 = 8 halves
// (params bias, recurrent bias).
// ---------------------------------------------------------------------------
kernel void i8_cnb_ct2d_tap4(device uchar *scratch [[buffer(0)]],
                        device uchar *wgt [[buffer(1)]],
                        constant I8P &p [[buffer(2)]],
                        texture2d<float, access::write> recurrent [[texture(0)]],
                        uint2 tgid [[threadgroup_position_in_grid]],
                        uint tid [[thread_index_in_threadgroup]])
{
    constexpr int TW = 32, TH = 4, T = TW * TH;
    threadgroup int4 c32v[T * 32 / 4];
    // The expansion consumes a1 before producing a2; the contraction
    // consumes a2 before pass 13 produces a3. cnb_core barriers separate
    // those lifetimes, so all activations share the largest (32-channel) tile.
    threadgroup uint activations[T * 32 / 4];
    threadgroup int32_t *c32 = (threadgroup int32_t *)c32v;
    threadgroup int8_t *a1 = (threadgroup int8_t *)activations;
    threadgroup int8_t *a2 = a1;
    threadgroup int8_t *a3 = a1;

    cnb_core<TW, TH>(scratch, wgt, p, c32, a1, a2, tgid, tid);
    device const half *b2 = (device const half *)(wgt + p.b2);
    {
        const uint c4 = (tid & 3) * 4;
        for (uint i = tid; i < T * 4; i += 128) {
            const uint px = i >> 2;
            const uint lx = px % TW, ly = px / TW;
            const uint x = tgid.x * TW + lx, y = tgid.y * TH + ly;
            uint v = 0;
            if (x < p.outW && y < p.outH)
                v = pack_s8(cnb_c3<TW, TH>(scratch, p, c32, b2, px, c4, x, y));
            *(threadgroup uint *)(a3 + px * 16 + c4) = v;
#if defined(I8_DEBUG) && I8_DEBUG == 4
            // debug: also store the CNB int8 output where the pre-fusion pass 13 put it
            if (x < p.outW && y < p.outH)
                *(device uint *)(scratch + p.outOff + y * p.outSY + x * p.outSX + c4) = v;
#endif
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device const half *cb = (device const half *)(wgt + p.b3);
    const half4 bp = *(device const half4 *)(cb);
    const half4 br = *(device const half4 *)(cb + 4);
    const float4 ks = float4(p.s[7]);
    gemm_tg<T, 32, 16>(a3, (device int8_t *)(wgt + p.w3), c32);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < T * 4; i += 128) {
        const uint px = i >> 2, tap = i & 3;
        const uint lx = px % TW, ly = px / TW;
        const uint ox = (tgid.x * TW + lx) * 2 + (tap & 1), oy = (tgid.y * TH + ly) * 2 + (tap >> 1);
        if (ox >= p.prmW || oy >= p.prmH) continue;
        const int4 ap = ld4(c32, px * 32 + tap * 8), ar = ld4(c32, px * 32 + tap * 8 + 4);
        const half4 hp = half4(float4(ap) * ks) + bp;
        const half4 hr = half4(float4(ar) * ks) + br;
        const float4 rec = float4(1.0) / (float4(1.0) + float4(exp(-hr)));
        recurrent.write(float4(half4(rec)), uint2(ox, oy));
        *(device half4 *)(scratch + p.prmOff + oy * p.prmSY + ox * 8) = hp;
    }
}

// ---------------------------------------------------------------------------
// 2x2 stride-2 conv (passes 3/6): gathered [T][4*CIN] x [4*CIN][COUT].
// s: [WQ, IQ, RQ0, RQ1]  out = ((acc*WQ)*IQ + b) * RQ(half)  (verified bit-exact unfolded)
// ---------------------------------------------------------------------------
template <int CIN, int COUT, int T>
static inline void ds_t(device uchar *scratch, device uchar *wgt, constant I8P &p,
                        threadgroup int8_t *A, threadgroup int32_t *C, uint2 tgid, uint tid)
{
    constexpr int K = 4 * CIN;
    // gather: T pixels x 4 taps x CIN bytes, 16-byte chunks
    constexpr int CH = CIN / 16;
    for (uint i = tid; i < T * 4 * CH; i += 128) {
        const uint px = i / (4 * CH), rem = i % (4 * CH), tap = rem / CH, ch = rem % CH;
        const uint x = tgid.x * T + px, y = tgid.y;
        const uint ix = 2 * x + (tap & 1), iy = 2 * y + (tap >> 1);
        uint4 v = uint4(0);
        if (x < p.outW && y < p.outH && ix < p.inW && iy < p.inH)
            v = *(device const uint4 *)(scratch + p.inOff + iy * p.inSY + ix * p.inSX + ch * 16);
        *(threadgroup uint4 *)(A + px * K + tap * CIN + ch * 16) = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    gemm_tg<T, COUT, K>(A, (device int8_t *)(wgt + p.w0), C);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device const half *b = (device const half *)(wgt + p.b0);
    for (uint i = tid; i < T * (COUT / 4); i += 128) {
        const uint px = i / (COUT / 4), c4 = (i % (COUT / 4)) * 4;
        const uint x = tgid.x * T + px, y = tgid.y;
        if (x >= p.outW || y >= p.outH) continue;
        const float4 bb = bias4(b, c4);
        const float rq = c4 < COUT / 2 ? p.s[2] : p.s[3];
        const float4 r = (((float4(ld4(C, px * COUT + c4)) * p.s[0]) * p.s[1]) + bb) * rq;
        *(device uint *)(scratch + p.outOff + y * p.outSY + x * p.outSX + c4) = pack_s8(r);
    }
}

kernel void i8_ds16(device uchar *scratch [[buffer(0)]], device uchar *wgt [[buffer(1)]],
                    constant I8P &p [[buffer(2)]], uint2 tgid [[threadgroup_position_in_grid]],
                    uint tid [[thread_index_in_threadgroup]])
{
    threadgroup uint4 Av[64 * 64 / 16];
    threadgroup int4 Cv[64 * 32 / 4];
    ds_t<16, 32, 64>(scratch, wgt, p, (threadgroup int8_t *)Av, (threadgroup int32_t *)Cv, tgid, tid);
}

kernel void i8_ds32(device uchar *scratch [[buffer(0)]], device uchar *wgt [[buffer(1)]],
                    constant I8P &p [[buffer(2)]], uint2 tgid [[threadgroup_position_in_grid]],
                    uint tid [[thread_index_in_threadgroup]])
{
    threadgroup uint4 Av[32 * 128 / 16];
    threadgroup int4 Cv[32 * 64 / 4];
    ds_t<32, 64, 32>(scratch, wgt, p, (threadgroup int8_t *)Av, (threadgroup int32_t *)Cv, tgid, tid);
}

// ---------------------------------------------------------------------------
// FasterNetBlock core: partial grouped 3x3 (NF/2 outputs), concat with the
// upper input half, 1x1 NF->2NF + relu, 1x1 2NF->NF, residual. Leaves the
// int32 contract result in c32 [T][NF] and the concat vector in a1 [T][NF].
// s: [c1a, c1b, R1, c2a, c2b, R2, c3a, c3b, IQ0, IQ1, RO0, RO1]
//   F1: c1 = (acc*c1a + b0)*R1            else ((acc*c1a)*c1b + b0)*R1
//   c2 always ((acc*c2a)*c2b + b1)*R2 (relu)   [F2 folded variant: (acc*c2a + b1)*R2]
//   F3: c3 term = acc*c3a                  else (acc*c3a)*c3b
// ---------------------------------------------------------------------------
template <int NF, int NG, int TW, int TH, bool F1, bool F2, bool F3>
static inline void fnb_core(device uchar *scratch, device uchar *wgt, constant I8P &p,
                            threadgroup int32_t *c32, threadgroup int8_t *a1, threadgroup int8_t *a2,
                            uint2 tgid, uint tid)
{
    constexpr int T = TW * TH, HF = NF / 2, FPG = HF / NG;
    static_assert(FPG % 4 == 0 && 128 % (FPG / 4) == 0, "grouped conv features per group");
    static_assert(128 % (2 * NF / 4) == 0, "expand channels per thread");
    device int8_t *in = (device int8_t *)(scratch + p.inOff);
    device const half *b0 = (device const half *)(wgt + p.b0);
    device const half *b1 = (device const half *)(wgt + p.b1);
    // grouped conv: group g reads input channels [g*16, g*16+16), writes a1 cols [g*FPG, ...)
    for (int g = 0; g < NG; g++) {
        conv3x3_16<TW, TH, MAXW, MAXH>(in + g * 16, p.inSX, p.inSY, p.inW, p.inH,
                                       (device int8_t *)(wgt + p.w0 + g * (9 * 16 * FPG)), c32, tgid);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const uint c4 = (tid % (FPG / 4)) * 4;
        const float4 bb = bias4(b0, g * FPG + c4);
        for (uint i = tid; i < T * (FPG / 4); i += 128) {
            const uint px = i / (FPG / 4);
            const float4 acc = float4(ld4(c32, px * 16 + c4));
            float4 v;
            if (F1) v = (acc * p.s[0] + bb) * p.s[2];
            else    v = ((acc * p.s[0]) * p.s[1] + bb) * p.s[2];
            *(threadgroup uint *)(a1 + px * NF + g * FPG + c4) = pack_s8(v);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    // passthrough upper half of the input into a1 cols [HF, NF)
    for (uint i = tid; i < T * (HF / 4); i += 128) {
        const uint px = i / (HF / 4), c4 = HF + (i % (HF / 4)) * 4;
        const uint lx = px % TW, ly = px / TW;
        const uint x = tgid.x * TW + lx, y = tgid.y * TH + ly;
        uint v = 0;
        if (x < p.inW && y < p.inH)
            v = *(device const uint *)(scratch + p.inOff + y * p.inSY + x * p.inSX + c4);
        *(threadgroup uint *)(a1 + px * NF + c4) = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    gemm_tg<T, 2 * NF, NF>(a1, (device int8_t *)(wgt + p.w1), c32);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {
        const uint c4 = (tid % (2 * NF / 4)) * 4;
        const float4 bb = bias4(b1, c4);
        for (uint i = tid; i < T * (2 * NF / 4); i += 128) {
            const uint px = i / (2 * NF / 4);
            const float4 acc = float4(ld4(c32, px * 2 * NF + c4));
            float4 v;
            if (F2) v = (acc * p.s[3] + bb) * p.s[5];
            else    v = ((acc * p.s[3]) * p.s[4] + bb) * p.s[5];
            *(threadgroup uint *)(a2 + px * 2 * NF + c4) = pack_u7(v);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    gemm_tg<T, NF, 2 * NF>(a2, (device int8_t *)(wgt + p.w2), c32);
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// residual term for channel group c4 of pixel px: lower half uses the input
// bytes (scale IQ0), upper half the concat vector a1 (scale IQ1)
template <int NF, int TW, int TH, bool F3>
static inline float4 fnb_c3(constant I8P &p, threadgroup int32_t *c32, threadgroup int8_t *a1,
                            device const half *b2, uint px, uint c4, float rq, float4 res)
{
    const float4 acc = float4(ld4(c32, px * NF + c4));
    const float4 bb = bias4(b2, c4);
    float4 t;
    if (F3) t = acc * p.s[6];
    else    t = (acc * p.s[6]) * p.s[7];
    return ((t + bb) + res) * rq;
}

template <int NF, int NG, int TW, int TH, bool F1, bool F2, bool F3>
static inline void fnb_t(device uchar *scratch, device uchar *wgt, constant I8P &p,
                         threadgroup int32_t *c32, threadgroup int8_t *a1, threadgroup int8_t *a2,
                         uint2 tgid, uint tid)
{
    constexpr int T = TW * TH, HF = NF / 2;
    fnb_core<NF, NG, TW, TH, F1, F2, F3>(scratch, wgt, p, c32, a1, a2, tgid, tid);
    device const half *b2 = (device const half *)(wgt + p.b2);
#if defined(I8_DEBUG) && I8_DEBUG > 0
    for (uint i = tid; i < T * (NF / 4); i += 128) {
        const uint px = i / (NF / 4), c4 = (i % (NF / 4)) * 4;
        const uint lx = px % TW, ly = px / TW;
        const uint x = tgid.x * TW + lx, y = tgid.y * TH + ly;
        if (x >= p.outW || y >= p.outH) continue;
        uint v;
        if (I8_DEBUG == 1) v = *(threadgroup const uint *)(a1 + px * NF + c4);
        else if (I8_DEBUG == 2) v = *(threadgroup const uint *)(a2 + px * 2 * NF + c4);
        else { const int4 a = ld4(c32, px * NF + c4);
               v = pack4(clamp(a.x, -128, 127), clamp(a.y, -128, 127), clamp(a.z, -128, 127), clamp(a.w, -128, 127)); }
        *(device uint *)(scratch + p.outOff + y * p.outSY + x * p.outSX + c4) = v;
    }
    return;
#endif
    for (uint i = tid; i < T * (NF / 4); i += 128) {
        const uint px = i / (NF / 4), c4 = (i % (NF / 4)) * 4;
        const uint lx = px % TW, ly = px / TW;
        const uint x = tgid.x * TW + lx, y = tgid.y * TH + ly;
        if (x >= p.outW || y >= p.outH) continue;
        float4 res;
        float rq;
        if (c4 < HF) {
            const int4 iv = unpack4(*(device const uint *)(scratch + p.inOff + y * p.inSY + x * p.inSX + c4));
            res = float4(iv) * p.s[8];
            rq = p.s[10];
        } else {
            const int4 iv = unpack4(*(threadgroup const uint *)(a1 + px * NF + c4));
            res = float4(iv) * p.s[9];
            rq = p.s[11];
        }
        const float4 r = fnb_c3<NF, TW, TH, F3>(p, c32, a1, b2, px, c4, rq, res);
        *(device uint *)(scratch + p.outOff + y * p.outSY + x * p.outSX + c4) = pack_s8(r);
    }
}

#define FNB_KERNEL(NAME, NF, NG, TW, TH)                                                             \
    kernel void NAME(device uchar *scratch [[buffer(0)]], device uchar *wgt [[buffer(1)]],           \
                     constant I8P &p [[buffer(2)]], uint2 tgid [[threadgroup_position_in_grid]],     \
                     uint tid [[thread_index_in_threadgroup]])                                       \
    {                                                                                                \
        threadgroup int4 c32v[(TW * TH) * 2 * NF / 4];                                               \
        threadgroup uint a1v[(TW * TH) * NF / 4];                                                    \
        threadgroup uint a2v[(TW * TH) * 2 * NF / 4];                                                \
        fnb_t<NF, NG, TW, TH, true, true, true>(scratch, wgt, p, (threadgroup int32_t *)c32v,        \
                                                (threadgroup int8_t *)a1v, (threadgroup int8_t *)a2v, \
                                                tgid, tid);                                          \
    }

FNB_KERNEL(i8_fnb32_5f, 32, 1, 32, 2)
FNB_KERNEL(i8_fnb32_3f, 32, 1, 32, 2)
FNB_KERNEL(i8_fnb64_5f, 64, 2, 32, 1)

// ---------------------------------------------------------------------------
// FNB + ConvTranspose2D(2x2 s2) + skip add (passes 9/11).
// s: fnb_core slots [K1, -, R1, K2, -, R2, K3, -, IQ0, IQ1] + [RF, KC, AQ, RO0, RO1]
//   fnb  = ((acc*K3 + b2) + res) * RF        (c1/c2/c3 all folded)
//   out  = ((acc*KC + cb) + add*AQ) * RO(half of NOUT)
// ---------------------------------------------------------------------------
template <int NF, int NG, int NOUT, int TW, int TH>
static inline void ct2d_t(device uchar *scratch, device uchar *wgt, constant I8P &p,
                          threadgroup int32_t *c32, threadgroup int8_t *a1, threadgroup int8_t *a2,
                          threadgroup int8_t *a3, uint2 tgid, uint tid)
{
    constexpr int T = TW * TH, HF = NF / 2;
    // c1 folded (F1), c2 folded (F2): map to the fnb_core slots
    fnb_core<NF, NG, TW, TH, true, true, true>(scratch, wgt, p, c32, a1, a2, tgid, tid);
    device const half *b2 = (device const half *)(wgt + p.b2);
    for (uint i = tid; i < T * (NF / 4); i += 128) {
        const uint px = i / (NF / 4), c4 = (i % (NF / 4)) * 4;
        const uint lx = px % TW, ly = px / TW;
        const uint x = tgid.x * TW + lx, y = tgid.y * TH + ly;
        float4 res;
        if (c4 < HF) {
            uint v = 0;
            if (x < p.inW && y < p.inH)
                v = *(device const uint *)(scratch + p.inOff + y * p.inSY + x * p.inSX + c4);
            res = float4(unpack4(v)) * p.s[8];
        } else {
            res = float4(unpack4(*(threadgroup const uint *)(a1 + px * NF + c4))) * p.s[9];
        }
        const float4 bb = bias4(b2, c4);
        const float4 r = ((float4(ld4(c32, px * NF + c4)) * p.s[6] + bb) + res) * p.s[10];
        *(threadgroup uint *)(a3 + px * NF + c4) = pack_s8(r);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device const half *cb = (device const half *)(wgt + p.b3);
    // Four independent taps are columns of one wider GEMM. 4*NOUT ==
    // 2*NF, so the FNB expansion accumulator already has enough storage.
    static_assert(4 * NOUT == 2 * NF, "CT2D accumulator capacity");
    gemm_tg<T, 4 * NOUT, NF>(a3, (device int8_t *)(wgt + p.w3), c32);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < T * NOUT; i += 128) {
        const uint px = i / NOUT, tap = (i % NOUT) / (NOUT / 4), c4 = (i % (NOUT / 4)) * 4;
        const uint lx = px % TW, ly = px / TW;
        const uint ox = (tgid.x * TW + lx) * 2 + (tap & 1), oy = (tgid.y * TH + ly) * 2 + (tap >> 1);
        if (ox >= p.outW || oy >= p.outH) continue;
        const float4 bb = bias4(cb, c4);
        const int4 av = unpack4(*(device const uint *)(scratch + p.addOff + oy * p.addSY + ox * p.addSX + c4));
        const float ro = c4 < NOUT / 2 ? p.s[13] : p.s[14];
        const float4 r = ((float4(ld4(c32, px * 4 * NOUT + tap * NOUT + c4)) * p.s[11] + bb) + float4(av) * p.s[12]) * ro;
        *(device uint *)(scratch + p.outOff + oy * p.outSY + ox * p.outSX + c4) = pack_s8(r);
    }
}

#define CT2D_KERNEL(NAME, NF, NG, NOUT, TW, TH)                                                      \
    kernel void NAME(device uchar *scratch [[buffer(0)]], device uchar *wgt [[buffer(1)]],           \
                     constant I8P &p [[buffer(2)]], uint2 tgid [[threadgroup_position_in_grid]],     \
                     uint tid [[thread_index_in_threadgroup]])                                       \
    {                                                                                                \
        threadgroup int4 c32v[(TW * TH) * 2 * NF / 4];                                               \
        threadgroup uint a1v[(TW * TH) * NF / 4];                                                    \
        threadgroup uint a2v[(TW * TH) * 2 * NF / 4];                                                \
        /* CT2D reuses a2 after fnb_core's final barrier. */                                         \
        ct2d_t<NF, NG, NOUT, TW, TH>(scratch, wgt, p, (threadgroup int32_t *)c32v,                   \
                                     (threadgroup int8_t *)a1v, (threadgroup int8_t *)a2v,           \
                                     (threadgroup int8_t *)a2v, tgid, tid);                          \
    }

// Different entry points also reject stale manifests with the old [tap][K][N] weights.
CT2D_KERNEL(i8_ct2d64_tap4, 64, 2, 32, 32, 1)
CT2D_KERNEL(i8_ct2d32_tap4, 32, 1, 16, 32, 2)
