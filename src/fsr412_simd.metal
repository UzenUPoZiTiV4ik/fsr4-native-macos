// SPDX-License-Identifier: MIT
// Supplemental exact Apple7+ Lite layouts. Append after fsr411_simd.metal.
// Experimental entries only; runtime selection requires GPU parity + A/B.
#ifndef FSR412_SIMD_TUNING
#define FSR412_SIMD_TUNING 1
#define FSR412_SIMD_ARGS \
    device uchar *scratch [[buffer(0)]], device const uchar *weights [[buffer(1)]], \
    constant fsr411::PassParams &raw [[buffer(2)]], device const half *wh [[buffer(3)]], \
    device const half *wg [[buffer(4)]], uint2 group [[threadgroup_position_in_grid]], \
    uint tid [[thread_index_in_threadgroup]], ushort sg [[simdgroup_index_in_threadgroup]], \
    ushort lane [[thread_index_in_simdgroup]]
#define FSR412_BLOCK(NAME,NF,MB,SG,UPSAMPLE) \
kernel void NAME(FSR412_SIMD_ARGS) { \
    FSR411_LOCAL_PARAMS; \
    threadgroup half tile[(8*(MB)+2)*((SG)+2)*16+(8*(MB))*(SG)*((NF)-16)]; \
    fsr411_simd::block<NF,1,MB,SG,UPSAMPLE,false>(scratch,weights,wh,wg,p,tile,group,tid,sg,lane); \
}
FSR412_BLOCK(fsr412_simd_cnb_16x2,16,2,2,false)
FSR412_BLOCK(fsr412_simd_cnb_32x1,16,4,1,false)
FSR412_BLOCK(fsr412_simd_cnb_32x2,16,4,2,false)
FSR412_BLOCK(fsr412_simd_cnb_64x4,16,8,4,false)
FSR412_BLOCK(fsr412_simd_fnb32_16x2,32,2,2,false)
FSR412_BLOCK(fsr412_simd_fnb32_32x1,32,4,1,false)
FSR412_BLOCK(fsr412_simd_fnb32_32x2,32,4,2,false)
FSR412_BLOCK(fsr412_simd_fnb32_64x4,32,8,4,false)
FSR412_BLOCK(fsr412_simd_ct32_16x2,32,2,2,true)
FSR412_BLOCK(fsr412_simd_ct32_32x1,32,4,1,true)
FSR412_BLOCK(fsr412_simd_ct32_32x2,32,4,2,true)
FSR412_BLOCK(fsr412_simd_ct32_64x4,32,8,4,true)
#undef FSR412_BLOCK
#define FSR412_DS(NAME,MB,SG,COMPACT) \
kernel void NAME(FSR412_SIMD_ARGS) { \
    FSR411_LOCAL_PARAMS; \
    threadgroup uint4 storage[16*(MB)*2*(SG)*16*((COMPACT)?1:2)/16]; \
    fsr411_simd::downsample<16,MB,SG,COMPACT>(scratch,weights,wh,p,(threadgroup uchar*)storage,group,tid,sg,lane); \
}
FSR412_DS(fsr412_simd_ds16_16x2,2,2,false)
FSR412_DS(fsr412_simd_ds16_32x1,4,1,false)
FSR412_DS(fsr412_simd_ds16_32x2,4,2,false)
FSR412_DS(fsr412_simd_ds16_64x4,8,4,false)
FSR412_DS(fsr412_simd_ds16_32x1_s8,4,1,true)
FSR412_DS(fsr412_simd_ds16_32x2_s8,4,2,true)
#undef FSR412_DS
#undef FSR412_SIMD_ARGS
#endif
