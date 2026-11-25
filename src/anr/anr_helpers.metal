// SPDX-License-Identifier: MIT
#include <metal_stdlib>
using namespace metal;
struct AnrConvert {
    uint width,height,renderWidth,renderHeight;
    uint encoding,depthReversed;float whiteNits;uint uiMode;
    float uiNoiseFloor;
};
inline float3 pq_encode(float3 linear,float nits) {
    float3 y=max(linear,0.f)*(nits/10000.f);
    float3 p=pow(clamp(y,0.f,1.f),float3(2610.f/16384.f));
    return pow((3424.f/4096.f+(2413.f/128.f)*p)/(1.f+(2392.f/128.f)*p),float3(2523.f/32.f));
}
inline float3 pq_decode(float3 code,float nits) {
    float3 p=pow(clamp(code,0.f,1.f),float3(32.f/2523.f));
    return pow(max(p-3424.f/4096.f,0.f)/max(2413.f/128.f-(2392.f/128.f)*p,1.e-7f),float3(16384.f/2610.f))*(10000.f/nits);
}
inline float3 srgb_decode(float3 v){return select(pow((max(v,0.f)+.055f)/1.055f,float3(2.4f)),v/12.92f,v<=.04045f);}
inline float3 srgb_encode(float3 v){return select(1.055f*pow(max(v,0.f),float3(1.f/2.4f))-.055f,12.92f*v,v<=.0031308f);}
// encoding bits 0..1: 0=linear HDR, 1=SDR code, 2=PQ10 code.
// encoding bit 3 (8): V.B20 native passthrough (skip color transfer conversion).
kernel void anr_import(texture2d<float,access::read> color [[texture(0)]],
                       texture2d<float,access::read> depth [[texture(1)]],
                       texture2d<float,access::read> motion [[texture(2)]],
                       texture2d<float,access::write> pq [[texture(3)]],
                       texture2d<float,access::write> depthCopy [[texture(4)]],
                       texture2d<float,access::write> motionCopy [[texture(5)]],
                       constant AnrConvert &p [[buffer(0)]],uint2 gid [[thread_position_in_grid]]) {
    if(gid.x>=p.width||gid.y>=p.height)return;
    const uint enc=p.encoding&3u;
    const bool nativePass=(p.encoding&8u)!=0u;
    float3 rgb=color.read(gid).rgb;
    if(!nativePass){
        if(enc==1u)rgb=srgb_decode(rgb);
        if(enc!=2u)rgb=pq_encode(rgb,p.whiteNits);
    }
    rgb=select(rgb,float3(0),!isfinite(rgb));pq.write(float4(rgb,1),gid);
    uint2 low=min(uint2((float2(gid)+.5f)*float2(p.renderWidth,p.renderHeight)/float2(p.width,p.height)),
                  uint2(p.renderWidth-1,p.renderHeight-1));
    float z=depth.read(low).x;if(!isfinite(z))z=p.depthReversed?0:1;
    depthCopy.write(float4(z,0,0,0),gid);
    float2 mv=motion.read(low).xy;mv=select(mv,float2(0),!isfinite(mv));motionCopy.write(float4(mv,0,0),gid);
}

// Fast depth/motion-only import when color is passed directly to V.B20 without conversion.
kernel void anr_import_dm(texture2d<float,access::read> depth [[texture(1)]],
                          texture2d<float,access::read> motion [[texture(2)]],
                          texture2d<float,access::write> depthCopy [[texture(4)]],
                          texture2d<float,access::write> motionCopy [[texture(5)]],
                          constant AnrConvert &p [[buffer(0)]],uint2 gid [[thread_position_in_grid]]) {
    if(gid.x>=p.width||gid.y>=p.height)return;
    uint2 low=min(uint2((float2(gid)+.5f)*float2(p.renderWidth,p.renderHeight)/float2(p.width,p.height)),
                  uint2(p.renderWidth-1,p.renderHeight-1));
    float z=depth.read(low).x;if(!isfinite(z))z=p.depthReversed?0:1;
    depthCopy.write(float4(z,0,0,0),gid);
    float2 mv=motion.read(low).xy;mv=select(mv,float2(0),!isfinite(mv));motionCopy.write(float4(mv,0,0),gid);
}

kernel void anr_export(texture2d<float,access::read> pq [[texture(0)]],
                       texture2d<float,access::write> output [[texture(1)]],
                       texture2d<float,access::read> cleanColor [[texture(2)]],
                       texture2d<float,access::read> ui [[texture(3)]],
                       device atomic_uint *uiStats [[buffer(1)]],
                       constant AnrConvert &p [[buffer(0)]],uint2 gid [[thread_position_in_grid]]) {
    if(gid.x>=p.width||gid.y>=p.height)return;
    const uint enc=p.encoding&3u;
    const bool nativePass=(p.encoding&8u)!=0u;
    float3 rgb=pq.read(gid).rgb;
    if(!nativePass&&enc!=2u){rgb=pq_decode(rgb,p.whiteNits);if(enc==1u)rgb=srgb_encode(rgb);}
    const uint uiMode=p.uiMode&255u;
    float uiOpacity=0.f,uiDifference=0.f,uiMagnitude=1.f;
    if(uiMode){
        float4 hud=ui.read(gid);
        if(all(isfinite(hud.rgb))&&(uiMode==2||isfinite(hud.a))){
            if(uiMode==2){
                // UI contains a complete real frame. A one-bit full-screen
                // difference (dither, grain, post effects) is NOT an opaque HUD.
                // Binary changed-pixel copying mixes real and midpoint geometry.
                float3 clean=cleanColor.read(gid).rgb;
                if(all(isfinite(clean))){
                    float3 delta=hud.rgb-clean;
                    uiDifference=max(abs(delta.x),max(abs(delta.y),abs(delta.z)));
                    uiMagnitude=max(1.f,max(max(abs(clean.x),max(abs(clean.y),abs(clean.z))),
                                           max(abs(hud.x),max(abs(hud.y),abs(hud.z)))));
                    // Minimum opacity consistent with a bounded RGB foreground.
                    // Supports negative wide-gamut scRGB values cleanly via lowerBound <= 0.
                    float upper=max(1.f,max(max(clean.x,max(clean.y,clean.z)),max(hud.x,max(hud.y,hud.z))));
                    float lowerBound=min(0.f,min(min(clean.x,min(clean.y,clean.z)),min(hud.x,min(hud.y,hud.z))));
                    float3 room=select(max(clean-float3(lowerBound),float3(1.e-7f)),
                                       max(float3(upper)-clean,float3(1.e-7f)),delta>=0.f);
                    float3 lowerOpacity=abs(delta)/room;
                    float lower=max(lowerOpacity.x,max(lowerOpacity.y,lowerOpacity.z));
                    float floor=max(p.uiNoiseFloor,1.e-7f)*uiMagnitude;
                    float contrast=smoothstep(floor,8.f*floor,uiDifference);
                    uiOpacity=clamp(lower*1.2f,0.f,1.f)*contrast;
                    // Reconstruct the overlay residual on the interpolated
                    // world. Weak effects stay on ANR instead of copying C.
                    rgb=hud.rgb+(1.f-uiOpacity)*(rgb-clean);
                }
            }else{
                float alpha=clamp(hud.a,0.f,1.f);
                if(alpha>0.f)rgb=hud.rgb+(1.f-alpha)*rgb;
            }
        }
    }
    // Sparse, opt-in samples, requested at most twice per second by the host.
    // uiStats lives after the completion word, never in the same atomic slot.
    if(uiMode==2&&(p.uiMode&256u)&&!(gid.x&31u)&&!(gid.y&31u)){
        atomic_fetch_add_explicit(uiStats+0,1u,memory_order_relaxed);
        float oldEpsilon=enc==0u?1.e-4f:1.f/4096.f;
        if(uiDifference>oldEpsilon*uiMagnitude)atomic_fetch_add_explicit(uiStats+1,1u,memory_order_relaxed);
        if(uiOpacity>0.f&&uiOpacity<.999f)atomic_fetch_add_explicit(uiStats+2,1u,memory_order_relaxed);
        if(uiOpacity>=.999f)atomic_fetch_add_explicit(uiStats+3,1u,memory_order_relaxed);
        atomic_fetch_add_explicit(uiStats+4,uint(round(uiOpacity*4096.f)),memory_order_relaxed);
    }
    output.write(float4(rgb,1),gid);
}
kernel void anr_retire(device atomic_uint *done [[buffer(0)]]) {
    atomic_store_explicit(done,1u,memory_order_relaxed);
}

// Fullscreen triangle render blit for 3x extra-frame presentation onto any CAMetalDrawable
// (handles framebufferOnly=YES, Retina scaling, and cross-format swapchains).
struct PresentVert {
    float4 pos [[position]];
    float2 uv;
};
vertex PresentVert anr_present_vs(uint vid [[vertex_id]]) {
    PresentVert out;
    float2 uv = float2((vid << 1u) & 2u, vid & 2u);
    out.uv = float2(uv.x, 1.0f - uv.y);
    out.pos = float4(uv * 2.0f - 1.0f, 0.0f, 1.0f);
    return out;
}
fragment float4 anr_present_fs(PresentVert in [[stage_in]],
                               texture2d<float, access::sample> src [[texture(0)]]) {
    constexpr sampler s(coord::normalized, address::clamp_to_edge, filter::linear);
    float4 c = src.sample(s, in.uv);
    return float4(c.rgb, 1.0f);
}
