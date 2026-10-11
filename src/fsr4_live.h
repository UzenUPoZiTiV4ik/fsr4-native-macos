#pragma once
#include <stdint.h>
#include "fsr4_input.h"

// One FSR frame encoded into the caller's command buffer. The encoder
// does not commit. Textures are retained by the caller for the life of the
// buffer. Returns 1 if FSR4 consumed the encode, 0 to keep the original scaler.
#ifdef __cplusplus
extern "C" {
#endif

typedef struct Fsr4LiveFrame {
    void *device;
    void *commandBuffer;
    // Identity of the MetalFX scaler object. History, recurrent state and
    // per-frame bookkeeping live per owner: D3DMetal encodes more than one
    // scaler per game frame and they must not share a history buffer.
    void *owner;
    void *color, *depth, *motion, *output, *exposure;
    uint32_t rw, rh, uw, uh;
    float jitterX, jitterY;
    float mvScaleX, mvScaleY;
    float preExposure;
    int reset;
    int depthReversed;
    // Descriptor flags as D3DMetal configured the scaler (-1 = unknown).
    int jitteredMV;      // motion vectors include the jitter pattern
    int outputResMV;     // motion texture is at output resolution
    int autoExposure;    // scaler computes exposure itself (no exposure texture)
    uint32_t mvOffX, mvOffY;   // motionContentOffset (subrect origin), 0 normally
} Fsr4LiveFrame;

int fsr4_encode_live_frame(const Fsr4LiveFrame *frame);

// V1 above remains byte-for-byte unchanged. V2 makes subrects/allocation
// extents explicit without reading beyond an old caller's Fsr4LiveFrame.
#define FSR4_LIVE_FRAME_ABI_V2 2u
typedef struct Fsr4LiveFrameV2 {
    uint32_t structSize;
    uint32_t abiVersion;
    Fsr4LiveFrame frame;
    Fsr4InputGeometry geometry;
    int inputContentProperties;  // descriptor's dynamic-resolution flag; -1 unavailable
    uint32_t reserved;
} Fsr4LiveFrameV2;

int fsr4_encode_live_frame_v2(const Fsr4LiveFrameV2 *frame);

// A frame that uses the original MetalFX scaler creates a temporal gap. The
// next FSR frame must reset. The lifecycle callback drops a destroyed scaler's
// state without retaining the scaler itself.
void fsr4_invalidate_live_owner(void *device, void *owner);
void fsr4_release_live_owner(void *device, void *owner);

static inline int fsr4_live_frame_v2_valid(const Fsr4LiveFrameV2 *frame) {
    return frame && frame->abiVersion == FSR4_LIVE_FRAME_ABI_V2
        && frame->structSize >= sizeof(Fsr4LiveFrameV2);
}

// Legacy entry points obtain allocation dimensions from the actual Metal
// textures, set unknown maximum input dimensions to zero, then use this same
// conversion as V2. The live encoder must verify V2 geometry against its
// actual textures before encoding copies or dispatches.
static inline Fsr4InputDesc fsr4_live_input_desc(const Fsr4LiveFrame *frame,
                                              const Fsr4InputGeometry *geometry,
                                              int inputContentProperties) {
    Fsr4InputDesc input;
    memset(&input, 0, sizeof(input));
    if (!frame || !geometry) return input;
    input.geometry = *geometry;
    input.rw = frame->rw; input.rh = frame->rh;
    input.uw = frame->uw; input.uh = frame->uh;
    input.jitterX = frame->jitterX; input.jitterY = frame->jitterY;
    input.mvScaleX = frame->mvScaleX; input.mvScaleY = frame->mvScaleY;
    input.preExposure = frame->preExposure;
    input.reset = frame->reset; input.depthReversed = frame->depthReversed;
    input.jitteredMV = frame->jitteredMV; input.outputResMV = frame->outputResMV;
    input.autoExposure = frame->autoExposure;
    input.hasExposureTexture = frame->exposure != 0;
    input.inputContentProperties = inputContentProperties;
    return input;
}

// Legacy entry point (owner = NULL, flags unknown).
int fsr4_encode_live(void *device, void *commandBuffer,
                     void *color, void *depth, void *motion, void *output,
                     void *exposure,
                     uint32_t rw, uint32_t rh, uint32_t uw, uint32_t uh,
                     float jitterX, float jitterY,
                     float mvScaleX, float mvScaleY,
                     float preExposure, int reset, int depthReversed);

// --- status / settings for the panel (src/fsr4_panel.mm) and the hook ---

typedef struct Fsr4LiveStatus {
    int active;              // the last real encode ran FSR4
    char reason[96];         // why not, when !active
    char preset[16];
    char model[8];           // i8mpp | f16 | i8-ref | scalar; version is in variant
    char variant[96];        // shader variant key
    uint32_t rw, rh, uw, uh, uwA, uhA;
    float frameMs;           // EMA of the interval between real encodes (game frame time)
    uint64_t encodes, replaced, stubs;
    float sharpness;         // effective RCAS sharpness (0 = off)
} Fsr4LiveStatus;

void fsr4_live_get_status(Fsr4LiveStatus *out);
// Publish a complete snapshot from an independently implemented model bridge.
void fsr4_live_set_status(const Fsr4LiveStatus *status);
// Mark FSR4 inactive with a reason (hook: no textures, FSR4_REPLACE=0).
void fsr4_live_set_inactive(const char *reason);

// RCAS sharpness setting: FSR scale 0..1 plus an on/off flag. Defaults come
// from FSR4_SHARPNESS / FSR4_RCAS, then dist/crossover/panel.conf (written
// by the panel) overrides them; FSR4_SETTINGS=none ignores the file.
void fsr4_live_get_sharpness(float *sharpness, int *rcas);
void fsr4_live_set_sharpness(float sharpness, int rcas);
// Upscaler profile: 0 = Exact, 1 = Lossy (default), 2 = Exact-fast
// (exact-fast in FSR4_PROFILE/panel.conf), 3 = the FSR 4.0.2 model ("fsr4").
// FSR4_PROFILE sets the initial preference; panel.conf overrides it unless
// FSR4_SETTINGS=none. The 4.1.1 bridge includes 0-2 in its session/history
// identity; switching to or from 3 changes the model and resets history.
void fsr4_live_get_profile(int *profile);
void fsr4_live_set_profile(int profile);
// Model the live bridge encodes: "4.0.2" when FSR4_VERSION=4.0.2 or the
// profile is 3, otherwise "4.1.1".
const char *fsr4_live_model_version(void);

// FSR4_PANEL=1: launch dist/crossover/FSR4 Panel.app once (src/fsr4_panel.mm).
void fsr4_panel_request(void);
// Publish the status for the panel app to dist/crossover/status.conf (2 Hz).
void fsr4_panel_publish(void);

#ifdef __cplusplus
}
#endif
