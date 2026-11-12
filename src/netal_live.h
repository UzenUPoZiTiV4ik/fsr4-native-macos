#pragma once
#include "fsr4_live.h"
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

// Optional Netal SR engine, linked from the hash-pinned external source tree.
// No DLL execution, no second MetalFX hook, and no command-buffer commit.
int fsr4_netal_encode_live_frame(const Fsr4LiveFrameV2 *frame, void *fence);
void fsr4_netal_invalidate_owner(void *device, void *owner);
void fsr4_netal_release_owner(void *device, void *owner);

// The generated upstream session uses these instead of process-global
// geometry getenv calls. Values are scoped to this thread during creation,
// so two scalers can select different profiles without changing the game env.
const char *fsr4_netal_geometry_profile(void);
const char *fsr4_netal_geometry_render(void);

static inline const char *fsr4_netal_input_reason(const Fsr4LiveFrameV2 *v2) {
    if (!fsr4_live_frame_v2_valid(v2)) return "Netal: unsupported live frame ABI";
    const Fsr4LiveFrame *f = &v2->frame;
    if (!f->device || !f->commandBuffer || !f->color || !f->depth || !f->motion || !f->output)
        return "Netal: missing textures or command buffer";
    if (!f->owner) return "Netal: per-scaler owner is required";
    if (!f->rw || !f->rh || !f->uw || !f->uh) return "Netal: empty geometry";
    const Fsr4InputGeometry *g = &v2->geometry;
    if (g->color.offsetX || g->color.offsetY || g->depth.offsetX || g->depth.offsetY
        || g->motion.offsetX || g->motion.offsetY || g->output.offsetX || g->output.offsetY
        || f->mvOffX || f->mvOffY) return "Netal: nonzero texture offsets are not supported";
    if (g->color.width < f->rw || g->color.height < f->rh
        || g->depth.width < f->rw || g->depth.height < f->rh
        || g->motion.width < f->rw || g->motion.height < f->rh
        || g->output.width < f->uw || g->output.height < f->uh
        || (g->maxInputWidth && f->rw > g->maxInputWidth)
        || (g->maxInputHeight && f->rh > g->maxInputHeight))
        return "Netal: content exceeds texture or descriptor extent";
    if (f->jitteredMV != 0) return "Netal: jittered/unknown motion-vector contract";
    if (f->outputResMV != 0) return "Netal: output-resolution/unknown motion-vector contract";
    if (!f->depthReversed) return "Netal: only reversed depth is verified";
    if (f->preExposure != 1.f) return "Netal: pre-exposure must be 1";
    if (f->autoExposure != 0) return "Netal: auto/unknown exposure is not supported";
    if (!isfinite(f->jitterX) || !isfinite(f->jitterY) || fabsf(f->jitterX) > .5f || fabsf(f->jitterY) > .5f)
        return "Netal: jitter is outside the supported pixel range";
    if (!isfinite(f->mvScaleX) || !isfinite(f->mvScaleY) || !f->mvScaleX || !f->mvScaleY)
        return "Netal: motion scale must be finite and nonzero";
    return 0;
}

#ifdef __cplusplus
}
#endif
