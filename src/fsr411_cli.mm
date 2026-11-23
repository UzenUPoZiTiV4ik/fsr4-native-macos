// SPDX-License-Identifier: MIT
// FSR 4.1.1 recording runner. Keep ARC ownership separate from the legacy CLI.
#include "fsr411_runtime.h"
#include "fsr4_cli_recording.h"

namespace {
using namespace fsr4_cli;

void usage() {
    std::fprintf(stderr,
        "usage: fsr4mac --fsr-version 4.1.1 <testcase_dir> <out_dir>\n"
        "       [--auto|--reference|--scalar|--simd|--i8mpp] [--profile exact|exact-fast|lossy|lossy2] [--dump-passes|--dump-state]\n"
        "       [--capture-stages pre,pass1,...,post] [--bench N] [--profile-stages]\n"
        "       [--builddir dir] [--root dir] [--tier 1080|2160|4320] [--optimize 0|1] [--no-compile]\n"
        "       fsr4mac --fsr-version 4.1.1 --backend-info [auto|reference|scalar|simd|i8mpp]\n"
        "--simd (alias --fp16): M1-M4 simdgroup half matrices; INT8 model, output identical to --scalar.\n"
        "--bench runs one warmup sequence and N measured sequences; only run.json is written.\n"
        "--profile-stages adds diagnostic GPU counters; on Apple GPUs it separates compute encoders.\n"
        "Use an ordinary --bench run for fused production timing; stage counters add overhead.\n"
        "FSR 4.1.1 and the lossy profile are the defaults; use --profile exact for exact-reference validation.\n"
        "exact-fast keeps Exact's network but uses approximate division and exp/log in PRE/POST.\n"
        "lossy2 has six retrained layers (upstream dll-2026-10-29); it needs the fetched weights.\n"
        "--temporal-quality 0|1|2: off, conservative repairs, or fast Wave32 notes.\n"
        "Use --fsr-version 4.0.2 to select the legacy model and its FP16 backend.\n"
        "GPU timings cover the full frame command buffer; diagnostic copies are labelled separately.\n");
}

struct Cli411 {
    using Options = fsr411::Options;
    static constexpr const char *version="4.1.1", *model="FSR 4.1.1";
    static constexpr const char *default_profile="lossy";
    static constexpr const char *capture_error="capture stage must be pre, post, pass1..12, or pad0..12";
    static constexpr const char *alias_policy="alias_after_pass12_v1";
    static constexpr const char *profiling_exception="FSR411Profiling";
    static constexpr int passes=12, post_pass=13;
    static constexpr bool pad1b=false;
    static void print_usage() { usage(); }
    static bool raw_comparable(const fsr411::View &view) { return !view.postCnbAlias; }
    void env(Options &options) {
        if(const char *quality=std::getenv("FSR4_TEMPORAL_QUALITY");quality && *quality) {
            if(std::strcmp(quality,"0") && std::strcmp(quality,"1") && std::strcmp(quality,"2")) fail("FSR4_TEMPORAL_QUALITY must be 0, 1 or 2");
            options.temporalQuality=quality[0]-'0';
        }
    }
    template<typename Value> bool option(const std::string &arg, Value value, Options &options) {
        if(arg!="--temporal-quality") return false;
        const char *quality=value();
        if(std::strcmp(quality,"0") && std::strcmp(quality,"1") && std::strcmp(quality,"2")) fail("--temporal-quality must be 0, 1 or 2");
        options.temporalQuality=quality[0]-'0';
        return true;
    }
    void post_parse(std::string &profile, Options &options) const {
        if(!fsr411::parse_profile(profile,options.profile))
            fail("--profile / FSR4_PROFILE must be exact, exact-fast, lossy or lossy2");
        if(options.temporalQuality && !fsr411::profile_is_lossy(options.profile))
            fail("--temporal-quality is only available for Lossy/lossy2");
    }
    void after_meta(Options &) const {}
    NSDictionary *report_fields(const Options &options) const {
        return @{@"temporal_quality":@(options.temporalQuality)};
    }
};
} // namespace

int fsr411_main(int argc, char **argv) { return fsr4_cli::run<Cli411>(argc, argv); }
