// SPDX-License-Identifier: MIT
// The hook receives MetalFX inputs AFTER D3DMetal translated DLSS/NGX.
// All coordinate/unit conversion is centralized in fsr4_input.h.
#include "fsr4_live_common.h"

namespace {
using namespace fsr4_live_bridge;

bool options_for(const Fsr4LiveFrameV2 &f,fsr411::Frame &frame,fsr411::Options &o,std::string &error) {
    int profile=1;fsr4_live_get_profile(&profile);
    o.profile=fsr411::profile_from_int(profile);
    if(!env_options(f,frame,o,"FSR 4.1.1",error))return false;
    if(auto quality=env("FSR4_TEMPORAL_QUALITY")) {
        if(std::strcmp(quality,"0") && std::strcmp(quality,"1") && std::strcmp(quality,"2")) {
            error="FSR4_TEMPORAL_QUALITY must be 0, 1 or 2";return false;
        }
        o.temporalQuality=quality[0]-'0';
    }
    if(o.temporalQuality && !fsr411::profile_is_lossy(o.profile)) {
        error="FSR4_TEMPORAL_QUALITY is only available for Lossy/lossy2";return false;
    }
    return tail_options(o,error);
}

struct Live411 {
    using Options = fsr411::Options;
    static constexpr const char *version="4.1.1", *model="FSR 4.1.1";
    static constexpr fsr411::Profile trace_profile=fsr411::Profile::Lossy;
    static const char *profile_name(const Options &o) { return fsr411::profile_name(o.profile); }
    static bool parse_options(const Fsr4LiveFrameV2 &f,fsr411::Frame &frame,Options &o,std::string &error) {
        return options_for(f,frame,o,error);
    }
};
}

extern "C" int fsr411_encode_live_frame(const Fsr4LiveFrameV2 *input) {
    return fsr4_live_bridge::encode_live_frame<Live411>(input);
}
extern "C" void fsr411_invalidate_owner(void *device,void *owner) {
    fsr4_live_bridge::invalidate_owner<Live411>(device,owner);
}
extern "C" void fsr411_release_owner(void *device,void *owner) {
    fsr4_live_bridge::release_owner<Live411>(device,owner);
}
