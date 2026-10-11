#include "fsr4_input.h"
#include "fsr4_live.h"
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <initializer_list>
#include <limits>

static int checks = 0;
#define CHECK(expr) do { ++checks; if (!(expr)) { \
    std::fprintf(stderr, "input contract: line %d: %s\n", __LINE__, #expr); std::exit(1); \
} } while (0)

static bool near(float a, float b, float tolerance = 1e-6f) {
    return std::fabs(a - b) <= tolerance;
}

static Fsr4InputDesc pixels(uint32_t rw = 1280, uint32_t rh = 720,
                            uint32_t uw = 1920, uint32_t uh = 1080) {
    Fsr4InputDesc in = {};
    in.rw = rw; in.rh = rh; in.uw = uw; in.uh = uh;
    in.geometry.color = {rw, rh, 0, 0};
    in.geometry.depth = {rw, rh, 0, 0};
    in.geometry.motion = {rw, rh, 0, 0};
    in.geometry.output = {uw, uh, 0, 0};
    in.mvScaleX = 1.f; in.mvScaleY = 1.f;
    in.preExposure = 1.f;
    return in;
}

static Fsr4InputPrepared prepare(const Fsr4InputDesc &in, const Fsr4InputHistory *history = nullptr,
                                  int forceReset = 0) {
    Fsr4InputPrepared out = {};
    CHECK(fsr4_input_status_ok(fsr4_input_prepare(&in, history, forceReset, &out)));
    return out;
}

static void motion_units_and_signs() {
    auto in = pixels();
    in.jitterX = 0.25f; in.jitterY = -0.375f;
    auto p = prepare(in);
    // MetalFX's documented example: an object moves ten input pixels right
    // and down, so its previous position is fifteen output pixels up/left.
    CHECK(near(-10.f * p.mvScale[0] * p.uw, -15.f));
    CHECK(near(-10.f * p.mvScale[1] * p.uh, -15.f));
    CHECK(p.jitter[0] == 0.25f && p.jitter[1] == -0.375f);
    CHECK(p.lowResMV && p.mvW == 1280 && p.mvH == 720);
    CHECK(p.jitterCancellation[0] == 0.f && p.jitterCancellation[1] == 0.f);

    // UV vectors: the caller scales raw values by the content dimensions.
    in.mvScaleX = 1280.f; in.mvScaleY = 720.f;
    p = prepare(in);
    CHECK(p.mvScale[0] == 1.f && p.mvScale[1] == 1.f);
    CHECK(near(-0.125f * p.mvScale[0] * p.uw, -240.f));

    // A signed NDC -> pixel scale supplied by the bridge stays signed.
    // Neither motion's Y component nor jitter receives an extra DLSS flip.
    in.mvScaleX = 640.f; in.mvScaleY = -360.f;
    p = prepare(in);
    CHECK(p.mvScale[0] == 0.5f && p.mvScale[1] == -0.5f);
    CHECK(near(0.125f * p.mvScale[1] * p.uh, -67.5f));
    in.mvScaleX = 0.f;
    p = prepare(in);
    CHECK(p.mvScale[0] == 0.f);
}

static void logical_extent_and_subrects() {
    auto in = pixels(421, 243, 641, 365);
    in.outputResMV = 1;
    in.geometry.motion = {641, 365, 0, 0};
    auto p = prepare(in);
    CHECK(!p.lowResMV && p.alignedUW == 648 && p.alignedUH == 368);
    CHECK(p.mvW == 641 && p.mvH == 365);
    CHECK(near(p.mvScale[0], 1.f / 641.f, 1e-10f));
    CHECK(near(p.mvScale[1], 1.f / 365.f, 1e-10f));
    CHECK(!near(p.mvScale[0], 1.f / 648.f, 1e-8f));

    in = pixels();
    in.geometry.color = {2048, 1024, 13, 17};
    in.geometry.depth = {1536, 1024, 3, 5};
    in.geometry.motion = {1400, 800, 4, 6};
    in.geometry.output = {2304, 1296, 128, 64};
    in.geometry.maxInputWidth = 2048; in.geometry.maxInputHeight = 1024;
    Fsr4InputPrepared cropped = {};
    CHECK(fsr4_input_prepare(&in, nullptr, 0, &cropped) == FSR4_INPUT_NEEDS_COPY);
    CHECK(cropped.copyMask == (FSR4_INPUT_COPY_COLOR | FSR4_INPUT_COPY_DEPTH | FSR4_INPUT_COPY_MOTION | FSR4_INPUT_COPY_OUTPUT));
    CHECK(cropped.colorRegion.x == 13 && cropped.colorRegion.y == 17 && cropped.colorRegion.width == 1280);
    CHECK(cropped.depthRegion.x == 3 && cropped.motionRegion.y == 6);
    CHECK(cropped.outputRegion.x == 128 && cropped.outputRegion.y == 64 && cropped.outputRegion.height == 1080);
    CHECK(cropped.maxInputWidth == 2048 && cropped.maxInputHeight == 1024);
    CHECK(near(cropped.mvScale[0], 1.f / 1280.f, 1e-10f));

    // Large allocations must not imply output-resolution motion. This is a
    // common DRS input: only a top-left subregion of the texture is active.
    in = pixels();
    in.geometry.motion = {1920, 1080, 0, 0};
    p = prepare(in);
    CHECK(p.lowResMV && p.mvW == 1280 && p.motionRegion.width == 1280);
    CHECK(p.copyMask == FSR4_INPUT_COPY_MOTION);
    in.outputResMV = -1;
    p = prepare(in);
    CHECK(p.lowResMV && (p.defaultedFlags & FSR4_INPUT_DEFAULT_OUTPUT_RES_MV));
    in.outputResMV = 1;
    p = prepare(in);
    CHECK(!p.lowResMV && p.mvW == 1920 && p.copyMask == 0);
}

static void temporal_history_and_drs() {
    auto in = pixels(512, 288, 768, 432);
    in.jitteredMV = 1;
    in.inputContentProperties = 1;
    in.jitterX = 0.25f; in.jitterY = -0.25f;
    in.preExposure = 2.f;
    Fsr4InputHistory history = {};
    auto p = prepare(in, &history);
    CHECK(p.reset && (p.resetReasons & FSR4_INPUT_RESET_FIRST_FRAME));
    CHECK(p.previousPreExposure == 0.f && !history.valid);
    fsr4_input_commit(&history, &p);

    in.jitterX = -0.25f; in.jitterY = 0.125f;
    in.preExposure = 4.f;
    p = prepare(in, &history);
    CHECK(!p.reset && p.previousPreExposure == 2.f && p.preExposure == 4.f);
    CHECK(near(p.jitterCancellation[0], 0.5f / 512.f, 1e-10f));
    CHECK(near(p.jitterCancellation[1], -0.375f / 288.f, 1e-10f));
    // A stationary scene with jitter included in its motion becomes zero
    // after FSR subtracts the cancellation vector. This checks the sign.
    CHECK(near(0.5f * p.mvScale[0] - p.jitterCancellation[0], 0.f));
    CHECK(near(-0.375f * p.mvScale[1] - p.jitterCancellation[1], 0.f));
    CHECK(history.preExposure == 2.f && history.jitter[0] == 0.25f);

    // No commit represents an encode failure; a later attempt sees the last
    // successfully encoded frame, not the abandoned candidate's jitter.
    in.jitterX = 0.f;
    auto retry = prepare(in, &history);
    CHECK(near(retry.jitterCancellation[0], 0.25f / 512.f, 1e-10f));
    fsr4_input_commit(&history, &retry);
    in.rw = 426; in.rh = 240; // backing textures still hold the maximum input
    in.jitterX = -0.125f;
    p = prepare(in, &history);
    CHECK(!p.reset && p.copyMask == (FSR4_INPUT_COPY_COLOR | FSR4_INPUT_COPY_DEPTH | FSR4_INPUT_COPY_MOTION));
    CHECK(p.mvW == 426 && p.mvH == 240);
    CHECK(near(p.jitterCancellation[0], 0.125f / 426.f, 1e-10f));
    CHECK(p.previousPreExposure == 4.f);
    fsr4_input_commit(&history, &p);

    in.reset = 1;
    p = prepare(in, &history);
    CHECK(p.resetReasons == FSR4_INPUT_RESET_REQUESTED);
    in.reset = 0;
    p = prepare(in, &history, 1);
    CHECK(p.resetReasons == FSR4_INPUT_RESET_MODEL);
    in.depthReversed = 1;
    p = prepare(in, &history);
    CHECK(p.depthReversed == 1 && (p.resetReasons & FSR4_INPUT_RESET_CONVENTION));
    fsr4_input_invalidate(&history); // a real frame was handled by MetalFX
    p = prepare(in, &history);
    CHECK(p.reset && p.previousPreExposure == 0.f);

    // Logical output changes within the same aligned allocation still reset.
    in = pixels(421, 243, 641, 365);
    in.geometry.output = {648, 368, 0, 0};
    p = prepare(in);
    fsr4_input_commit(&history, &p);
    in.uw = 647;
    auto resized = prepare(in, &history);
    CHECK(resized.alignedUW == p.alignedUW);
    CHECK(resized.resetReasons == FSR4_INPUT_RESET_OUTPUT_SIZE);
}

static void exposure_and_defaults() {
    auto in = pixels();
    in.preExposure = 8.f;
    auto p = prepare(in);
    CHECK(p.exposureMode == FSR4_INPUT_EXPOSURE_ONE && p.preExposure == 8.f);
    in.hasExposureTexture = 1;
    p = prepare(in);
    CHECK(p.exposureMode == FSR4_INPUT_EXPOSURE_TEXTURE && p.preExposure == 8.f);
    // Auto exposure ignores the supplied texture, as MetalFX documents;
    // preExposure remains a separate input for FSR pre/post/SPD/RCAS.
    in.autoExposure = 1;
    p = prepare(in);
    CHECK(p.autoExposure && p.exposureMode == FSR4_INPUT_EXPOSURE_AUTO && p.preExposure == 8.f);
    in.autoExposure = in.jitteredMV = in.outputResMV = in.inputContentProperties = -1;
    p = prepare(in);
    CHECK(!p.autoExposure && !p.jitteredMV && p.lowResMV);
    CHECK(p.defaultedFlags == 15u);
    CHECK(p.exposureMode == FSR4_INPUT_EXPOSURE_TEXTURE);
}

static void invalid_inputs_and_abi() {
    auto good = pixels();
    Fsr4InputPrepared p = {};
    CHECK(fsr4_input_prepare(nullptr, nullptr, 0, &p) == FSR4_INPUT_BAD_ARGUMENT);
    CHECK(fsr4_input_prepare(&good, nullptr, 0, nullptr) == FSR4_INPUT_BAD_ARGUMENT);
    auto in = good;
    in.rw = 0;
    CHECK(fsr4_input_prepare(&in, nullptr, 0, &p) == FSR4_INPUT_BAD_SIZE);
    in = good; in.uw = UINT32_MAX;
    CHECK(fsr4_input_prepare(&in, nullptr, 0, &p) == FSR4_INPUT_BAD_SIZE);
    in = good; in.geometry.maxInputWidth = 1024; in.geometry.maxInputHeight = 720;
    CHECK(fsr4_input_prepare(&in, nullptr, 0, &p) == FSR4_INPUT_BAD_MAX_INPUT_SIZE);
    in = good; in.geometry.maxInputWidth = 2048;
    CHECK(fsr4_input_prepare(&in, nullptr, 0, &p) == FSR4_INPUT_BAD_MAX_INPUT_SIZE);
    in = good; in.geometry.color.offsetX = UINT32_MAX;
    CHECK(fsr4_input_prepare(&in, nullptr, 0, &p) == FSR4_INPUT_COLOR_REGION_OUT_OF_BOUNDS);
    in = good; in.geometry.depth.offsetY = 1;
    CHECK(fsr4_input_prepare(&in, nullptr, 0, &p) == FSR4_INPUT_DEPTH_REGION_OUT_OF_BOUNDS);
    in = good; in.geometry.motion = {4, 4, 0, 0};
    CHECK(fsr4_input_prepare(&in, nullptr, 0, &p) == FSR4_INPUT_MOTION_REGION_OUT_OF_BOUNDS);
    in = good; in.outputResMV = 1; // render-sized MV cannot fulfill this flag
    CHECK(fsr4_input_prepare(&in, nullptr, 0, &p) == FSR4_INPUT_MOTION_REGION_OUT_OF_BOUNDS);
    in = good; in.geometry.output.offsetX = 1;
    CHECK(fsr4_input_prepare(&in, nullptr, 0, &p) == FSR4_INPUT_OUTPUT_REGION_OUT_OF_BOUNDS);
    in = good; in.jitterX = std::numeric_limits<float>::quiet_NaN();
    CHECK(fsr4_input_prepare(&in, nullptr, 0, &p) == FSR4_INPUT_NONFINITE_JITTER);
    in = good; in.mvScaleY = std::numeric_limits<float>::infinity();
    CHECK(fsr4_input_prepare(&in, nullptr, 0, &p) == FSR4_INPUT_NONFINITE_MOTION_SCALE);
    for (float bad : {0.f, -1.f, std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN()}) {
        in = good; in.preExposure = bad;
        CHECK(fsr4_input_prepare(&in, nullptr, 0, &p) == FSR4_INPUT_INVALID_PRE_EXPOSURE);
    }
    in = good; in.outputResMV = 2;
    CHECK(fsr4_input_prepare(&in, nullptr, 0, &p) == FSR4_INPUT_INVALID_FLAG);

    // The old C frame layout is used by existing dylib clients on both Mac
    // architectures. V2 is a separate entry point with an explicit size tag.
    static_assert(sizeof(void *) != 8 || sizeof(Fsr4LiveFrame) == 128, "V1 ABI size changed");
    static_assert(sizeof(void *) != 8 || offsetof(Fsr4LiveFrame, mvOffX) == 120, "V1 ABI offsets changed");
    Fsr4LiveFrameV2 v2 = {};
    v2.abiVersion = FSR4_LIVE_FRAME_ABI_V2;
    v2.structSize = sizeof(v2);
    CHECK(fsr4_live_frame_v2_valid(&v2));
    v2.structSize--;
    CHECK(!fsr4_live_frame_v2_valid(&v2));
    v2.structSize = sizeof(v2); v2.abiVersion++;
    CHECK(!fsr4_live_frame_v2_valid(&v2));
    v2.frame.rw = 1280; v2.frame.rh = 720; v2.frame.uw = 1920; v2.frame.uh = 1080;
    v2.frame.jitterX = -0.25f; v2.frame.jitterY = 0.125f;
    v2.frame.mvScaleX = 1280.f; v2.frame.mvScaleY = -720.f;
    v2.frame.preExposure = 2.f; v2.frame.depthReversed = 1;
    v2.frame.exposure = &v2;
    v2.geometry = good.geometry;
    auto captured = fsr4_live_input_desc(&v2.frame, &v2.geometry, -1);
    p = prepare(captured);
    CHECK(p.mvScale[0] == 1.f && p.mvScale[1] == -1.f);
    CHECK(p.jitter[0] == -0.25f && p.jitter[1] == 0.125f && p.depthReversed);
    CHECK(p.preExposure == 2.f && p.exposureMode == FSR4_INPUT_EXPOSURE_TEXTURE);
}

int main() {
    motion_units_and_signs();
    logical_extent_and_subrects();
    temporal_history_and_drs();
    exposure_and_defaults();
    invalid_inputs_and_abi();
    std::printf("input contract: %d checks passed\n", checks);
}
