// SPDX-License-Identifier: MIT
// Standalone M5 candidates for FSR 4.1.2 Lite / Lite Lossy. No model data is
// embedded here. The caller retains the integer PassParams ABI and supplies
// the verified 4.1.2 coefficients. This file is not a production selection.
#include "fsr411_mpp.metal"

// Each SIMD32 gathers four 16-channel INT8 taps directly into a cooperative
// matmul operand. The existing MIT primitive preserves int32 wrapping,
// nearest-even quantization and the two independently scaled output halves.
// There are no threadgroup allocations or barriers in these DS entry points.
// IMPORTANT: probe the corresponding bit before selecting a fast kernel.
#define FSR412_DS16(NAME, TW, TH) \
kernel void NAME(device uchar *scratch [[buffer(0)]], device uchar *weights [[buffer(1)]], \
    constant fsr411::PassParams &raw [[buffer(2)]], uint2 group [[threadgroup_position_in_grid]]) { \
    FSR411_LOCAL_PARAMS; \
    fsr411_ds_chain::downsample<16,32,TW,TH>(scratch, weights, p, group); \
}
FSR412_DS16(fsr412_ds16_coop_8x1, 8, 1)
FSR412_DS16(fsr412_ds16_coop_16x1, 16, 1)
FSR412_DS16(fsr412_ds16_coop_32x1, 32, 1)
FSR412_DS16(fsr412_ds16_coop_64x1, 64, 1)
FSR412_DS16(fsr412_ds16_coop_8x2, 8, 2)
FSR412_DS16(fsr412_ds16_coop_16x2, 16, 2)
#undef FSR412_DS16

// Four independent register-only SIMD tiles. Same 64x1/128-thread dispatch
// footprint as the established DS16, but no shared activation staging.
kernel void fsr412_ds16_coop_64x1_sg4(device uchar *scratch [[buffer(0)]],
    device uchar *weights [[buffer(1)]], constant fsr411::PassParams &raw [[buffer(2)]],
    uint2 group [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]]) {
    FSR411_LOCAL_PARAMS;
    fsr411_ds_chain::downsample<16,32,16,1>(scratch, weights, p, uint2(group.x * 4u + sg, group.y));
}

// Layout-only probe. It performs no model inference or matrix products.
// Bits 0..5 correspond to the six single-SIMD candidates above; bit 6 is
// the four-SIMD kernel, which uses the same per-SIMD layout as bit 1.
namespace fsr412_tuning {
template <uint TW, uint TH>
__attribute__((always_inline)) inline bool ds_layout() {
    constexpr uint T = TW * TH;
    constexpr auto d = matmul2d_descriptor(T, 32, 64, false, false, false, matmul2d_descriptor::mode::multiply);
    matmul2d<d, execution_simdgroup> op;
    auto input = op.template get_left_input_cooperative_tensor<int8_t,int8_t,int>();
    auto weight = tensor<device int8_t, extents<int,32,64>, tensor_inline>(
        (device int8_t *)nullptr, extents<int,32,64>());
    auto result = op.template get_destination_cooperative_tensor<decltype(input),decltype(weight),int>();
    return fsr411_ds_chain::valid_layout<T * 64u / 32u>(input)
        && fsr411_ds_chain::valid_layout<T * 32u / 32u>(result);
}
template <uint TW, uint TH>
__attribute__((always_inline)) inline bool cnb_layout() {
    constexpr uint T = TW * TH;
    constexpr auto ed = matmul2d_descriptor(T,32,16,false,false,false,matmul2d_descriptor::mode::multiply);
    constexpr auto pd = matmul2d_descriptor(T,16,32,false,false,false,matmul2d_descriptor::mode::multiply);
    matmul2d<ed,execution_simdgroup> expansion;
    matmul2d<pd,execution_simdgroup> projection;
    auto ei = expansion.template get_left_input_cooperative_tensor<int8_t,int8_t,int>();
    auto pi = projection.template get_left_input_cooperative_tensor<int8_t,int8_t,int>();
    auto input = tensor<device int8_t,dextents<int,4>,tensor_inline>(
        (device int8_t *)nullptr,dextents<int,4>(16,1,1,1));
    auto cw = tensor<device int8_t,extents<int,16,16,3,3>,tensor_inline>(
        (device int8_t *)nullptr,extents<int,16,16,3,3>());
    constexpr auto cd = convolution2d_descriptor(
        int4(16,TW,TH,1),int4(16,FSR411_MAXW,FSR411_MAXH,1),int2(3),
        convolution2d_activation_layout::nhwc,convolution2d_weights_layout::hwio,
        int2(1),int2(1),1,false,convolution2d_descriptor::mode::multiply);
    convolution2d<cd,execution_simdgroup> conv;
    auto cr = conv.template get_destination_cooperative_tensor<decltype(input),decltype(cw),int>();
    auto ew = tensor<device int8_t,extents<int,32,16>,tensor_inline>(
        (device int8_t *)nullptr,extents<int,32,16>());
    auto er = expansion.template get_destination_cooperative_tensor<decltype(ei),decltype(ew),int>();
    auto pw = tensor<device int8_t,extents<int,16,32>,tensor_inline>(
        (device int8_t *)nullptr,extents<int,16,32>());
    auto pr = projection.template get_destination_cooperative_tensor<decltype(pi),decltype(pw),int>();
    return fsr411_chain::matching_layout<T / 2u,true,TW>(cr,ei)
        && fsr411_chain::matching_layout<T,false,TW>(er,pi)
        && fsr411_ds_chain::valid_layout<T / 2u>(pr);
}
}
kernel void fsr412_mpp_tuning_probe(device uint *mask [[buffer(0)]],
    uint lane [[thread_index_in_simdgroup]]) {
    uint bits = 0;
    if (fsr412_tuning::ds_layout<8,1>()) bits |= 1u << 0;
    if (fsr412_tuning::ds_layout<16,1>()) bits |= (1u << 1) | (1u << 6);
    if (fsr412_tuning::ds_layout<32,1>()) bits |= 1u << 2;
    if (fsr412_tuning::ds_layout<64,1>()) bits |= 1u << 3;
    if (fsr412_tuning::ds_layout<8,2>()) bits |= 1u << 4;
    if (fsr412_tuning::ds_layout<16,2>()) bits |= 1u << 5;
    if (!lane) mask[0] = bits;
}

// Fused p1 candidates retain every valid B pixel: pass3's skip needs B.
// A SIMD writes its own TWx2 B region, synchronizes device memory, then
// gathers ONLY that same region into DS. CNB's 3x3 halo is read from A by
// convolution2d; no DS tap depends on another threadgroup's B writes.
// Separate 212-byte PassParams at buffers2/3, no common/on-disk ABI change.
// Host must validate disjoint A/B/C, CNB.output == DS.input, and DS.output
// dimensions <= ceil(CNB.output/2). Dispatch covers CNB, not DS, so odd
// floor-DS dimensions never discard the last retained skip row/column.
#if FSR411_SPECIALIZE
constant uint4 fsr412DsSpec0 [[function_constant(12)]];
constant uint4 fsr412DsSpec1 [[function_constant(13)]];
constant uint4 fsr412DsSpec2 [[function_constant(14)]];
constant uint4 fsr412DsSpec3 [[function_constant(15)]];
constant uint4 fsr412DsSpec4 [[function_constant(16)]];
constant uint4 fsr412DsSpec5 [[function_constant(17)]];
constant uint4 fsr412DsSpec10 [[function_constant(22)]];
constant uint4 fsr412DsSpec11 [[function_constant(23)]];
__attribute__((always_inline)) inline fsr411::PassParams fsr412_specialize_ds(constant fsr411::PassParams &r) {
    // DS uses only weights0/bias0/q4/q5; keep dimensions dynamic and ignore
    // the unrelated CNB/FNB quantizers rather than duplicating their ABI.
    return {
        {fsr412DsSpec0.x,fsr412DsSpec0.y,fsr412DsSpec0.z,r.input.width,r.input.height,fsr412DsSpec0.w,fsr412DsSpec1.x},
        {fsr412DsSpec3.w?r.output.offset:fsr412DsSpec1.y,fsr412DsSpec1.z,fsr412DsSpec1.w,r.output.width,r.output.height,fsr412DsSpec2.x,fsr412DsSpec2.y},
        {}, {fsr412DsSpec4.x,0,0,0}, {fsr412DsSpec5.x,fsr411::kNoOffset,fsr411::kNoOffset,fsr411::kNoOffset},
        {{},{},{},{},
         {fsr412DsSpec10.x,fsr412DsSpec10.y,fsr412DsSpec10.z,fsr412DsSpec10.w},
         {fsr412DsSpec11.x,fsr412DsSpec11.y,fsr412DsSpec11.z,fsr412DsSpec11.w}}
    };
}
#define FSR412_LOCAL_DS const fsr411::PassParams ds = fsr412_specialize_ds(raw_ds)
#else
#define FSR412_LOCAL_DS constant fsr411::PassParams &ds = raw_ds
#endif
#define FSR412_CNB_DS(NAME, TW) \
kernel void NAME(device uchar *scratch [[buffer(0)]], device uchar *weights [[buffer(1)]], \
    constant fsr411::PassParams &raw [[buffer(2)]], constant fsr411::PassParams &raw_ds [[buffer(3)]], \
    uint2 group [[threadgroup_position_in_grid]]) { \
    FSR411_LOCAL_PARAMS; FSR412_LOCAL_DS; \
    if (!fsr412_tuning::cnb_layout<TW,2>() || !fsr412_tuning::ds_layout<(TW)/2,1>()) return; \
    if (!fsr411_chain::cnb<TW,2>(scratch,weights,p,group)) return; \
    simdgroup_barrier(mem_flags::mem_device); \
    fsr411_ds_chain::downsample<16,32,(TW)/2,1>(scratch,weights,ds,group); \
}
FSR412_CNB_DS(fsr412_cnb_ds_coop_32x2, 32)
FSR412_CNB_DS(fsr412_cnb_ds_coop_64x2, 64)
#undef FSR412_CNB_DS
#undef FSR412_LOCAL_DS

kernel void fsr412_cnb_ds_tuning_probe(device uint *mask [[buffer(0)]],
    uint lane [[thread_index_in_simdgroup]]) {
    uint bits = 0;
    if (fsr412_tuning::cnb_layout<32,2>() && fsr412_tuning::ds_layout<16,1>()) bits |= 1u;
    if (fsr412_tuning::cnb_layout<64,2>() && fsr412_tuning::ds_layout<32,1>()) bits |= 2u;
    if (!lane) mask[0] = bits;
}
