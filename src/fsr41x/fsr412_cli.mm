// SPDX-License-Identifier: MIT
// FSR 4.1.2 recording runner. Keep ARC ownership separate from the legacy CLI.
#include "fsr412_runtime.h"
#include "fsr4_cli_recording.h"

namespace {
using namespace fsr4_cli;

void usage() {
    std::fprintf(stderr,
        "usage: fsr4mac --fsr-version 4.1.2 <testcase_dir> <out_dir>\n"
        "       [--auto|--reference|--scalar|--simd|--i8mpp] [--profile int8-lite] [--dump-passes|--dump-state]\n"
        "       [--capture-stages pre,pass1,pass2,pass3,post] [--bench N] [--profile-stages]\n"
        "       [--builddir dir] [--root dir] [--tier 1080|2160|4320] [--optimize 0|1] [--no-compile] [--pixelization-fix]\n"
        "       [--pixelization-strength 0..1] [--neural-tile-size 0|128|256|512|1024|2048]\n"
        "       fsr4mac --fsr-version 4.1.2 --backend-info [auto|reference|scalar|simd|i8mpp]\n"
        "--simd (alias --fp16): M1-M4 simdgroup half matrices with float accumulation for the INT8 model.\n"
        "--bench runs one warmup sequence and N measured sequences; only run.json is written.\n"
        "--profile-stages adds diagnostic GPU counters; on Apple GPUs it separates compute encoders.\n"
        "Use an ordinary --bench run for fused production timing; stage counters add overhead.\n"
        "FSR 4.1.2 INT8 Lite is the default: AMD's smaller architecture, evaluated every frame.\n"
        "--pixelization-fix (or --profile int8-lite-pixfix) widens POST's filter where history is rejected;\n"
        "--pixelization-strength sets how much (0 none, 0.5 default, 1 most).\n"
        "Lite Lossy profiles: int8-lite-lossy and int8-lite-lossy-tiled (experimental, default tile size 512).\n"
        "Lite Lossy requires make lite-lossy-weights and does not support --reference/--int8.\n"
        "--neural-tile-size is Lossy-only; 0 disables partial tiles.\n"
        "--new-look: experimental whole-frame Lite Lossy visual rules, linear input up to 4K.\n"
        "Use --fsr-version 4.1.1 for the earlier Exact, Exact Fast, Lite and Lite 2 profiles.\n"
        "Use --fsr-version 4.0.2 to select the legacy model and its FP16 backend.\n"
        "GPU timings cover the full frame command buffer; diagnostic copies are labelled separately.\n");
}

struct Cli412 {
    using Options = fsr412::Options;
    static constexpr const char *version="4.1.2", *model="FSR 4.1.2";
    static constexpr const char *default_profile="int8-lite";
    static constexpr const char *capture_error="capture stage must be pre, post, pass1..3, pad0..3, or pad1b";
    static constexpr const char *alias_policy="alias_after_pass3_v1";
    static constexpr const char *profiling_exception="FSR412Profiling";
    static constexpr int passes=3, post_pass=4;
    static constexpr bool pad1b=true;
    static void print_usage() { usage(); }
    static bool raw_comparable(const fsr411::View &) { return false; }
    float pixelStrength=fsr412::kPixelFixStrength;
    unsigned neuralTileOverride=0;
    bool haveNeuralTileOverride=false;
    void env(Options &) {
        if(const char *tile=std::getenv("FSR4_NEURAL_TILE_SIZE"); tile && *tile) {
            if(!fsr412::parse_neural_tile_size(tile,neuralTileOverride))
                fail("FSR4_NEURAL_TILE_SIZE must be 0, 128, 256, 512, 1024, or 2048");
            haveNeuralTileOverride=true;
        }
    }
    template<typename Value> bool option(const std::string &arg, Value value, Options &options) {
        if(arg=="--new-look") options.newLook=true;
        else if(arg=="--pixelization-fix" || arg=="--hair-fix") options.pixelFix=true;
        else if(arg=="--neural-tile-size") {
            if(!fsr412::parse_neural_tile_size(value(),neuralTileOverride))
                fail("--neural-tile-size must be 0, 128, 256, 512, 1024, or 2048");
            haveNeuralTileOverride=true;
        }
        else if(arg=="--pixelization-strength") {
            char *end=nullptr; const char *text=value(); const float strength=std::strtof(text,&end);
            if(end==text || *end || !(strength>=0.f && strength<=1.f)) fail("--pixelization-strength must be 0..1");
            pixelStrength=strength;
        }
        else return false;
        return true;
    }
    void post_parse(std::string &profile, Options &options) const {
        if(!fsr412::parse_profile(profile,options.profile))
            fail("FSR 4.1.2 --profile / FSR4_PROFILE must be int8-lite, int8-lite-pixfix, int8-lite-lossy, or int8-lite-lossy-tiled; select --fsr-version 4.1.1 for legacy profiles");
        if(fsr412::pixel_fix_profile(profile)) options.pixelFix=true;
        if(fsr412::tiled_profile(profile)) options.neuralTileSize=512;
        if(haveNeuralTileOverride) {
            if(options.profile!=fsr412::Profile::Int8LiteLossy)
                fail("--neural-tile-size / FSR4_NEURAL_TILE_SIZE is supported only by FSR 4.1.2 Lite Lossy");
            options.neuralTileSize=neuralTileOverride;
        }
        fsr412::apply_pixel_fix(options,options.pixelFix,pixelStrength);
        std::string optionError;
        if(!fsr412::validate_options(options,optionError)) fail(optionError);
        profile=fsr412::profile_name(options);
    }
    void after_meta(Options &options) const {
        std::string optionError;
        if(!fsr412::validate_options(options,optionError)) fail(optionError);
    }
    NSDictionary *report_fields(const Options &options) const {
        return @{@"model":@"amd-int8-lite",@"neural_passes":@3,
            @"new_look":@(options.newLook),@"pixelization_fix":@(options.pixelFix),
            @"pixelization_strength":@(pixelStrength),@"neural_tile_size":@(options.neuralTileSize)};
    }
};
} // namespace

int fsr412_main(int argc, char **argv) { return fsr4_cli::run<Cli412>(argc, argv); }
