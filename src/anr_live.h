// SPDX-License-Identifier: MIT
#pragma once
#include <stdint.h>
#include <math.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif
#define FSR4_ANR_FG_ABI 1u
// No changes to the existing SR frame ABI. FG has its own settings/status.
typedef struct Fsr4AnrFrame {
    uint32_t structSize, abiVersion;
    void *device, *commandBuffer, *owner;
    // ui points either to a premultiplied RGBA overlay, or (uiComposited)
    // to current clean color + UI. It does not mean color contains the HUD.
    void *color, *previousColor, *depth, *motion, *output, *ui, *fence;
    uint32_t rw, rh, uw, uh;
    float motionScaleX, motionScaleY, jitterX, jitterY;
    float nearPlane, farPlane, deltaTime, fieldOfView, aspectRatio;
    int reset, depthReversed, uiComposited;
    // 0 = linear HDR, 1 = SDR code values, 2 = PQ10 code values.
    int colorEncoding;
} Fsr4AnrFrame;
typedef enum Fsr4AnrUIMode {
    FSR4_ANR_UI_NONE=0, FSR4_ANR_UI_OVERLAY=1, FSR4_ANR_UI_PRECOMPOSITED=2
} Fsr4AnrUIMode;
static inline Fsr4AnrUIMode fsr4_anr_ui_mode(const Fsr4AnrFrame *f) {
    return !f || !f->ui ? FSR4_ANR_UI_NONE :
        f->uiComposited ? FSR4_ANR_UI_PRECOMPOSITED : FSR4_ANR_UI_OVERLAY;
}
typedef struct Fsr4AnrStatus {
    int active;
    char reason[160];
    uint32_t width,height,colorFormat,uiFormat;
    uint64_t encodes,generated,primed,fallbacks;
    uint64_t uiComposited,uiOverlay;
    Fsr4AnrUIMode uiMode;
    uint32_t uiSamples,uiChanged,uiSoft,uiOpaque,uiOpacitySum;
    float motionScaleX,motionScaleY,anrMotionScaleX,anrMotionScaleY;
} Fsr4AnrStatus;
// 1 generated and consumed, 2 history primed (run original without invalidating),
// 0 rejected (run original and invalidate ANR history).
int fsr4_anr_encode_frame(const Fsr4AnrFrame *frame);
void fsr4_anr_invalidate_owner(void *device,void *owner);
void fsr4_anr_release_owner(void *device,void *owner);
void fsr4_anr_get_status(Fsr4AnrStatus *out);
void fsr4_anr_set_inactive(const char *reason);

static inline const char *fsr4_anr_frame_reason(const Fsr4AnrFrame *f) {
    if(!f || f->abiVersion!=FSR4_ANR_FG_ABI || f->structSize<sizeof(*f))return "ANR: incompatible FG ABI";
    if(!f->device || !f->commandBuffer || !f->owner || !f->color || !f->depth || !f->motion || !f->output)
        return "ANR: missing FG textures/device/owner";
    if(!f->rw || !f->rh || !f->uw || !f->uh || f->rw>f->uw || f->rh>f->uh)return "ANR: invalid FG geometry";
    if(!isfinite(f->nearPlane)||f->nearPlane<=0 || !isfinite(f->farPlane)||f->farPlane<=f->nearPlane)
        return "ANR: finite camera near/far planes required";
    if(!isfinite(f->motionScaleX)||!isfinite(f->motionScaleY)||!f->motionScaleX||!f->motionScaleY)
        return "ANR: invalid motion scale";
    if(!isfinite(f->jitterX)||!isfinite(f->jitterY)||fabsf(f->jitterX)>.5f||fabsf(f->jitterY)>.5f)
        return "ANR: jitter outside pixel range";
    // uiTextureComposited only describes an optional ui texture. With no
    // texture the flag is ignored, just as there is no overlay to composite.
    if(f->colorEncoding<0||f->colorEncoding>2)return "ANR: unknown color encoding";
    return 0;
}
// 204-byte hard-frame contract recovered from the shipped GPU program.
// No camera-only flow is synthesized: MetalFX motion already contains camera
// motion, so flag 128 selects the model's explicit no-matrix path.
static inline void fsr4_anr_frame_constants(const Fsr4AnrFrame *f,uint8_t out[204],int stateValid) {
    memset(out,0,204);
    float values[]={f->nearPlane,f->farPlane,f->nearPlane,f->farPlane,
                    f->motionScaleX*(float)f->uw/f->rw,f->motionScaleY*(float)f->uh/f->rh,
                    f->jitterX,f->jitterY};
    memcpy(out,values,sizeof(values));
    uint32_t flags=128u|(f->depthReversed?1u:0u)|(stateValid?256u:0u);
    memcpy(out+32,&flags,4);
    uint32_t sizes[]={f->uw,f->uh,f->uw,f->uh,0,0,0,0,0,0};memcpy(out+36,sizes,sizeof(sizes));
    // Identity camera bases are intentionally ignored by flag 128.
    float camera[]={1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0};
    memcpy(out+76,camera,sizeof(camera));
}
#ifdef __cplusplus
}
#endif
