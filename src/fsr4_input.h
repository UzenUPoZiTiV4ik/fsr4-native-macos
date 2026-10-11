// The MetalFX -> FSR input contract. This header is deliberately usable from
// Objective-C, C and C++ without Metal; GPU resources stay with the caller.
//
// The hook observes MetalFX AFTER D3DMetal has translated NGX/DLSS parameters.
// Do not apply another DLSS sign change here. MTLFXTemporalScaler.h documents
// raw MV * motionVectorScale as previous-position minus current-position in
// pixels (an object moving down/right by 10 has MV (-10,-10)). Jitter is the
// input-pixel sampling offset, also passed through unchanged.
//
// The FSR normalization follows the upstream provider, including its use of
// logical upscaleSize, not AlignUp(upscaleSize,8), for output-resolution MV:
// https://github.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK/blob/01446e6a74888bf349652fcf2cbf5f642d30c2bf/Kits/FidelityFX/upscalers/fsr4/dx12/ffx_provider_fsr4_dx12.cpp#L1755-L1770
// See also TESTDATA.md and mlsr_optimized_includes.hlsli::LoadInputVelocity.
#pragma once

#include <math.h>
#include <stdint.h>
#include <string.h>

typedef struct Fsr4InputSurface {
    uint32_t width, height;       // allocated texture extent
    uint32_t offsetX, offsetY;    // active region's origin within it
} Fsr4InputSurface;

typedef struct Fsr4InputGeometry {
    Fsr4InputSurface color, depth, motion, output;
    // Descriptor's maximum input extent. Both zero means unavailable (v1).
    uint32_t maxInputWidth, maxInputHeight;
} Fsr4InputGeometry;

typedef struct Fsr4InputRect {
    uint32_t x, y, width, height;
} Fsr4InputRect;

typedef struct Fsr4InputDesc {
    Fsr4InputGeometry geometry;
    uint32_t rw, rh, uw, uh;      // logical content, never allocation or padding
    float jitterX, jitterY;
    float mvScaleX, mvScaleY;     // raw motion units -> pixels, preserve sign
    float preExposure;           // input color was multiplied by this value
    int reset, depthReversed;
    // Descriptor flags: -1 is unavailable and uses MetalFX's documented false
    // default. Texture allocation size must never infer either MV convention.
    int jitteredMV, outputResMV, autoExposure, inputContentProperties;
    int hasExposureTexture;
} Fsr4InputDesc;

typedef enum Fsr4InputStatus {
    FSR4_INPUT_VALID = 0,
    FSR4_INPUT_NEEDS_COPY = 1,
    FSR4_INPUT_BAD_ARGUMENT = 100,
    FSR4_INPUT_BAD_SIZE,
    FSR4_INPUT_BAD_MAX_INPUT_SIZE,
    FSR4_INPUT_COLOR_REGION_OUT_OF_BOUNDS,
    FSR4_INPUT_DEPTH_REGION_OUT_OF_BOUNDS,
    FSR4_INPUT_MOTION_REGION_OUT_OF_BOUNDS,
    FSR4_INPUT_OUTPUT_REGION_OUT_OF_BOUNDS,
    FSR4_INPUT_NONFINITE_JITTER,
    FSR4_INPUT_NONFINITE_MOTION_SCALE,
    FSR4_INPUT_INVALID_PRE_EXPOSURE,
    FSR4_INPUT_INVALID_FLAG
} Fsr4InputStatus;

enum {
    FSR4_INPUT_COPY_COLOR = 1u << 0,
    FSR4_INPUT_COPY_DEPTH = 1u << 1,
    FSR4_INPUT_COPY_MOTION = 1u << 2,
    FSR4_INPUT_COPY_OUTPUT = 1u << 3
};

enum {
    FSR4_INPUT_DEFAULT_JITTERED_MV = 1u << 0,
    FSR4_INPUT_DEFAULT_OUTPUT_RES_MV = 1u << 1,
    FSR4_INPUT_DEFAULT_AUTO_EXPOSURE = 1u << 2,
    FSR4_INPUT_DEFAULT_CONTENT_PROPERTIES = 1u << 3
};

typedef enum Fsr4InputExposureMode {
    FSR4_INPUT_EXPOSURE_ONE = 0,  // bind neutral 1x1 exposure, keep preExposure
    FSR4_INPUT_EXPOSURE_TEXTURE,
    FSR4_INPUT_EXPOSURE_AUTO      // SPD; ignores a supplied exposure texture
} Fsr4InputExposureMode;

enum {
    FSR4_INPUT_RESET_FIRST_FRAME = 1u << 0,
    FSR4_INPUT_RESET_REQUESTED = 1u << 1,
    FSR4_INPUT_RESET_MODEL = 1u << 2,
    FSR4_INPUT_RESET_OUTPUT_SIZE = 1u << 3,
    FSR4_INPUT_RESET_CONVENTION = 1u << 4
};

typedef struct Fsr4InputHistory {
    int valid;
    uint32_t outputWidth, outputHeight;
    float jitter[2], preExposure;
    int jitteredMV, outputResMV, depthReversed;
    Fsr4InputExposureMode exposureMode;
} Fsr4InputHistory;

typedef struct Fsr4InputPrepared {
    uint32_t rw, rh, uw, uh, alignedUW, alignedUH;
    uint32_t mvW, mvH;
    uint32_t maxInputWidth, maxInputHeight;
    Fsr4InputRect colorRegion, depthRegion, motionRegion, outputRegion;
    uint32_t copyMask, defaultedFlags;
    int lowResMV, jitteredMV, depthReversed, autoExposure;
    Fsr4InputExposureMode exposureMode;
    float jitter[2], mvScale[2], jitterCancellation[2];
    float preExposure, previousPreExposure;
    uint32_t reset, resetReasons;
    // Preview only. Commit after successfully encoding the whole frame;
    // preparation and failed encodes must not advance temporal bookkeeping.
    Fsr4InputHistory nextHistory;
} Fsr4InputPrepared;

static inline int fsr4_input_status_ok(Fsr4InputStatus status) {
    return status == FSR4_INPUT_VALID || status == FSR4_INPUT_NEEDS_COPY;
}

static inline const char *fsr4_input_status_reason(Fsr4InputStatus status) {
    switch (status) {
        case FSR4_INPUT_VALID: return "input contract valid";
        case FSR4_INPUT_NEEDS_COPY: return "input contract needs region copies";
        case FSR4_INPUT_BAD_ARGUMENT: return "missing input contract";
        case FSR4_INPUT_BAD_SIZE: return "invalid input/output content size";
        case FSR4_INPUT_BAD_MAX_INPUT_SIZE: return "content exceeds scaler input extent";
        case FSR4_INPUT_COLOR_REGION_OUT_OF_BOUNDS: return "color region exceeds texture";
        case FSR4_INPUT_DEPTH_REGION_OUT_OF_BOUNDS: return "depth region exceeds texture";
        case FSR4_INPUT_MOTION_REGION_OUT_OF_BOUNDS: return "motion region exceeds texture";
        case FSR4_INPUT_OUTPUT_REGION_OUT_OF_BOUNDS: return "output region exceeds texture";
        case FSR4_INPUT_NONFINITE_JITTER: return "non-finite jitter";
        case FSR4_INPUT_NONFINITE_MOTION_SCALE: return "non-finite motion scale";
        case FSR4_INPUT_INVALID_PRE_EXPOSURE: return "preExposure must be positive and finite";
        case FSR4_INPUT_INVALID_FLAG: return "invalid input convention flag";
    }
    return "unknown input contract error";
}

static inline int fsr4_input_region_fits(Fsr4InputSurface surface, uint32_t w, uint32_t h) {
    // Subtraction after checking the origin avoids unsigned offset+size wrap.
    return surface.width && surface.height && surface.offsetX <= surface.width
        && surface.offsetY <= surface.height && w <= surface.width - surface.offsetX
        && h <= surface.height - surface.offsetY;
}

static inline int fsr4_input_surface_needs_copy(Fsr4InputSurface surface, uint32_t w, uint32_t h) {
    // Canonicalizing a larger top-left allocation matters too: normalized
    // texture samples must address content, not the unused allocation tail.
    return surface.offsetX || surface.offsetY || surface.width != w || surface.height != h;
}

static inline Fsr4InputRect fsr4_input_region(Fsr4InputSurface surface, uint32_t w, uint32_t h) {
    Fsr4InputRect rect = {surface.offsetX, surface.offsetY, w, h};
    return rect;
}

static inline int fsr4_input_descriptor_flag_valid(int flag) {
    return flag >= -1 && flag <= 1;
}

static inline Fsr4InputStatus fsr4_input_prepare(const Fsr4InputDesc *in,
                                                const Fsr4InputHistory *history,
                                                int forceReset,
                                                Fsr4InputPrepared *out) {
    if (!in || !out) return FSR4_INPUT_BAD_ARGUMENT;
    memset(out, 0, sizeof(*out));
    // Shaders use signed pixel positions. Also leave room for AlignUp(...,8).
    if (!in->rw || !in->rh || !in->uw || !in->uh
        || in->rw > INT32_MAX || in->rh > INT32_MAX
        || in->uw > (uint32_t)INT32_MAX - 7u || in->uh > (uint32_t)INT32_MAX - 7u)
        return FSR4_INPUT_BAD_SIZE;
    const Fsr4InputGeometry *g = &in->geometry;
    if ((g->maxInputWidth == 0) != (g->maxInputHeight == 0)
        || (g->maxInputWidth && (in->rw > g->maxInputWidth || in->rh > g->maxInputHeight)))
        return FSR4_INPUT_BAD_MAX_INPUT_SIZE;
    if (!fsr4_input_descriptor_flag_valid(in->jitteredMV)
        || !fsr4_input_descriptor_flag_valid(in->outputResMV)
        || !fsr4_input_descriptor_flag_valid(in->autoExposure)
        || !fsr4_input_descriptor_flag_valid(in->inputContentProperties)
        || (in->depthReversed != 0 && in->depthReversed != 1)
        || (in->reset != 0 && in->reset != 1)
        || (in->hasExposureTexture != 0 && in->hasExposureTexture != 1))
        return FSR4_INPUT_INVALID_FLAG;
    if (!isfinite(in->jitterX) || !isfinite(in->jitterY)) return FSR4_INPUT_NONFINITE_JITTER;
    if (!isfinite(in->mvScaleX) || !isfinite(in->mvScaleY)) return FSR4_INPUT_NONFINITE_MOTION_SCALE;
    if (!isfinite(in->preExposure) || in->preExposure <= 0.f) return FSR4_INPUT_INVALID_PRE_EXPOSURE;

    Fsr4InputPrepared p;
    memset(&p, 0, sizeof(p));
    p.rw = in->rw; p.rh = in->rh; p.uw = in->uw; p.uh = in->uh;
    p.alignedUW = (in->uw + 7u) & ~7u; p.alignedUH = (in->uh + 7u) & ~7u;
    p.lowResMV = in->outputResMV != 1;
    p.mvW = p.lowResMV ? in->rw : in->uw;
    p.mvH = p.lowResMV ? in->rh : in->uh;
    if (!fsr4_input_region_fits(g->color, in->rw, in->rh)) return FSR4_INPUT_COLOR_REGION_OUT_OF_BOUNDS;
    if (!fsr4_input_region_fits(g->depth, in->rw, in->rh)) return FSR4_INPUT_DEPTH_REGION_OUT_OF_BOUNDS;
    if (!fsr4_input_region_fits(g->motion, p.mvW, p.mvH)) return FSR4_INPUT_MOTION_REGION_OUT_OF_BOUNDS;
    if (!fsr4_input_region_fits(g->output, in->uw, in->uh)) return FSR4_INPUT_OUTPUT_REGION_OUT_OF_BOUNDS;
    p.maxInputWidth = g->maxInputWidth ? g->maxInputWidth : in->rw;
    p.maxInputHeight = g->maxInputHeight ? g->maxInputHeight : in->rh;
    p.colorRegion = fsr4_input_region(g->color, in->rw, in->rh);
    p.depthRegion = fsr4_input_region(g->depth, in->rw, in->rh);
    p.motionRegion = fsr4_input_region(g->motion, p.mvW, p.mvH);
    p.outputRegion = fsr4_input_region(g->output, in->uw, in->uh);
    if (fsr4_input_surface_needs_copy(g->color, in->rw, in->rh)) p.copyMask |= FSR4_INPUT_COPY_COLOR;
    if (fsr4_input_surface_needs_copy(g->depth, in->rw, in->rh)) p.copyMask |= FSR4_INPUT_COPY_DEPTH;
    if (fsr4_input_surface_needs_copy(g->motion, p.mvW, p.mvH)) p.copyMask |= FSR4_INPUT_COPY_MOTION;
    if (fsr4_input_surface_needs_copy(g->output, in->uw, in->uh)) p.copyMask |= FSR4_INPUT_COPY_OUTPUT;
    if (in->jitteredMV < 0) p.defaultedFlags |= FSR4_INPUT_DEFAULT_JITTERED_MV;
    if (in->outputResMV < 0) p.defaultedFlags |= FSR4_INPUT_DEFAULT_OUTPUT_RES_MV;
    if (in->autoExposure < 0) p.defaultedFlags |= FSR4_INPUT_DEFAULT_AUTO_EXPOSURE;
    if (in->inputContentProperties < 0) p.defaultedFlags |= FSR4_INPUT_DEFAULT_CONTENT_PROPERTIES;
    p.jitteredMV = in->jitteredMV == 1;
    p.depthReversed = in->depthReversed;
    p.autoExposure = in->autoExposure == 1;
    p.exposureMode = p.autoExposure ? FSR4_INPUT_EXPOSURE_AUTO
        : in->hasExposureTexture ? FSR4_INPUT_EXPOSURE_TEXTURE : FSR4_INPUT_EXPOSURE_ONE;
    p.jitter[0] = in->jitterX; p.jitter[1] = in->jitterY;
    p.mvScale[0] = in->mvScaleX / (float)p.mvW;
    p.mvScale[1] = in->mvScaleY / (float)p.mvH;
    p.preExposure = in->preExposure;
    const int hasHistory = history && history->valid;
    p.previousPreExposure = hasHistory ? history->preExposure : 0.f;
    if (p.jitteredMV) {
        // FSR's shader subtracts this value. Preserve upstream association
        // and current-frame MV units, including during an input-only DRS change.
        p.jitterCancellation[0] = ((hasHistory ? history->jitter[0] : 0.f) - in->jitterX) / (float)p.mvW;
        p.jitterCancellation[1] = ((hasHistory ? history->jitter[1] : 0.f) - in->jitterY) / (float)p.mvH;
    }
    if (!hasHistory) p.resetReasons |= FSR4_INPUT_RESET_FIRST_FRAME;
    if (in->reset) p.resetReasons |= FSR4_INPUT_RESET_REQUESTED;
    if (forceReset) p.resetReasons |= FSR4_INPUT_RESET_MODEL;
    if (hasHistory) {
        // Compare logical dimensions: 641 -> 647 still requires reset even
        // though both sizes allocate an aligned 648-pixel history surface.
        if (history->outputWidth != in->uw || history->outputHeight != in->uh)
            p.resetReasons |= FSR4_INPUT_RESET_OUTPUT_SIZE;
        if (history->jitteredMV != p.jitteredMV || history->outputResMV != !p.lowResMV
            || history->depthReversed != p.depthReversed || history->exposureMode != p.exposureMode)
            p.resetReasons |= FSR4_INPUT_RESET_CONVENTION;
    }
    p.reset = p.resetReasons != 0;
    p.nextHistory.valid = 1;
    p.nextHistory.outputWidth = in->uw; p.nextHistory.outputHeight = in->uh;
    p.nextHistory.jitter[0] = in->jitterX; p.nextHistory.jitter[1] = in->jitterY;
    p.nextHistory.preExposure = in->preExposure;
    p.nextHistory.jitteredMV = p.jitteredMV;
    p.nextHistory.outputResMV = !p.lowResMV;
    p.nextHistory.depthReversed = p.depthReversed;
    p.nextHistory.exposureMode = p.exposureMode;
    *out = p;
    return p.copyMask ? FSR4_INPUT_NEEDS_COPY : FSR4_INPUT_VALID;
}

static inline void fsr4_input_commit(Fsr4InputHistory *history, const Fsr4InputPrepared *prepared) {
    if (history && prepared) *history = prepared->nextHistory;
}

static inline void fsr4_input_invalidate(Fsr4InputHistory *history) {
    // Call if the owner skips FSR for a real frame or loses its history textures.
    if (history) memset(history, 0, sizeof(*history));
}
