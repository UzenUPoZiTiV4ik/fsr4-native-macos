// SPDX-License-Identifier: MIT
// The hook receives MetalFX inputs AFTER D3DMetal translated DLSS/NGX.
// All coordinate/unit conversion is centralized in fsr4_input.h.
#include "fsr412_runtime.h"
#include "fsr4_live_common.h"

namespace {
using namespace fsr4_live_bridge;

bool options_for(const Fsr4LiveFrameV2 &f,fsr412::Frame &frame,fsr412::Options &o,std::string &error) {
    // The public router selected AMD's new INT8 Lite architecture. Legacy
    // panel profile numbers never select old weights within this runtime.
    int panelProfile=5;fsr4_live_get_profile(&panelProfile);
    o.profile=fsr412::profile_from_int(panelProfile);
    o.neuralTileSize=panelProfile==8?512u:0u;
    int newLook=0;fsr4_live_get_new_look(&newLook);
    o.newLook=panelProfile==7 && newLook!=0;
    if(auto tile=env("FSR4_NEURAL_TILE_SIZE")) {
        if(o.profile!=fsr412::Profile::Int8LiteLossy) {
            error="FSR4_NEURAL_TILE_SIZE is supported only by FSR 4.1.2 Lite Lossy";return false;
        }
        if(!fsr412::parse_neural_tile_size(tile,o.neuralTileSize)) {
            error="FSR4_NEURAL_TILE_SIZE must be 0, 128, 256, 512, 1024, or 2048";return false;
        }
    }
    float strength=fsr412::kPixelFixStrength;fsr4_live_get_pixelization_strength(&strength);
    fsr412::apply_pixel_fix(o,panelProfile==6,strength);
    if(!env_options(f,frame,o,"FSR 4.1.2",error))return false;
    {
        std::string optionError;
        if(!fsr412::validate_options(o,optionError)) {error=optionError;return false;}
    }
    return tail_options(o,error);
}

struct Live412 {
    using Options = fsr412::Options;
    static constexpr const char *version="4.1.2", *model="FSR 4.1.2";
    static constexpr fsr411::Profile trace_profile=fsr411::Profile::Int8Lite;
    static const char *profile_name(const Options &o) { return fsr412::profile_name(o); }
    static bool parse_options(const Fsr4LiveFrameV2 &f,fsr412::Frame &frame,Options &o,std::string &error) {
        return options_for(f,frame,o,error);
    }
};
}

extern "C" int fsr412_encode_live_frame(const Fsr4LiveFrameV2 *input) {
    return fsr4_live_bridge::encode_live_frame<Live412>(input);
}
extern "C" void fsr412_invalidate_owner(void *device,void *owner) {
    fsr4_live_bridge::invalidate_owner<Live412>(device,owner);
}
extern "C" void fsr412_release_owner(void *device,void *owner) {
    fsr4_live_bridge::release_owner<Live412>(device,owner);
}
