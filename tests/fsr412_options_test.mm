// SPDX-License-Identifier: MIT
// CPU-only adapter/live/parser contract. Device creation is replaced with a
// stub, so neither accepted nor rejected CLI options can touch a GPU.
#include "../src/fsr412_live.mm"
#include <stdexcept>
#include <vector>

static int deviceQueries = 0;
static id<MTLDevice> fsr412_cpu_only_device() { ++deviceQueries; return nil; }
#define MTLCreateSystemDefaultDevice fsr412_cpu_only_device
#include "../src/fsr412_cli.mm"
#undef MTLCreateSystemDefaultDevice

static int selectedProfile = 5;
extern "C" void fsr4_live_get_profile(int *profile) { if (profile) *profile = selectedProfile; }
extern "C" void fsr4_live_get_sharpness(float *sharpness, int *rcas) {
    if (sharpness) *sharpness = .4f;
    if (rcas) *rcas = 1;
}
extern "C" void fsr4_live_get_pixelization_strength(float *strength) {
    if (strength) *strength = fsr412::kPixelFixStrength;
}
extern "C" void fsr4_live_get_status(Fsr4LiveStatus *status) { if (status) *status = {}; }
extern "C" void fsr4_live_set_status(const Fsr4LiveStatus *) {}

static void require(bool value, const char *message) {
    if (!value) throw std::runtime_error(message);
}
static void clear_settings_env() {
    for (const char *key : {"FSR4_PROFILE", "FSR4_MODEL", "FSR4_NEURAL_TILE_SIZE",
                           "FSR4_OPTIMIZE", "FSR4_PRESET", "FSR4_COLORSPACE", "FSR4_LOW_RES_MV",
                           "FSR4_JITTERED_MV", "FSR4_AUTOEXPOSURE", "FSR4_COMPILE", "FSR4_TEMPORAL_QUALITY",
                           "FSR4_LITE_LOSSY_ORIGINAL"}) unsetenv(key);
    setenv("FSR4_ROOT", "/tmp/fsr412-cpu-options", 1);
}
static void cli(std::vector<std::string> arguments, bool accepted) {
    arguments.insert(arguments.begin(), "fsr4mac");
    std::vector<char *> argv;
    for (auto &argument : arguments) argv.push_back(argument.data());
    deviceQueries = 0;
    require(fsr412_main(int(argv.size()), argv.data()) == 1, "CPU CLI did not report its stubbed device/error");
    require(deviceQueries == (accepted ? 1 : 0), "CLI validation occurred after device creation or rejected valid options");
}

int main() { @autoreleasepool { try {
    clear_settings_env();
    fsr412::Options options;
    require(options.profile == fsr412::Profile::Int8Lite && !options.pixelFix && !options.neuralTileSize,
            "ordinary Lite default changed");
    for (const char *name : {"int8-lite-lossy", "lite-lossy", "int8-lite-lossy-tiled", "lite-lossy-tiled"}) {
        require(fsr412::parse_profile(name, options.profile), "Lossy alias rejected");
        require(options.profile == fsr412::Profile::Int8LiteLossy, "Lossy alias selected ordinary Lite");
        options.neuralTileSize = fsr412::tiled_profile(name) ? 512 : 0;
        require(std::string(fsr412::profile_name(options)) == (options.neuralTileSize ? "int8-lite-lossy-tiled" : "int8-lite-lossy"),
                "actual Lossy report profile mismatch");
    }
    for (const char *name : {"int8-lite-lossy-original", "lite-lossy-original"}) {
        fsr412::Options original;
        require(fsr412::parse_profile(name, original.profile) && fsr412::original_profile(name)
                && original.profile == fsr412::Profile::Int8LiteLossy, "original Lossy alias rejected");
        original.originalLossy = true;
        require(std::string(fsr412::profile_name(original)) == "int8-lite-lossy-original", "original report profile mismatch");
        std::string reason;
        require(fsr412::validate_options(original, reason), "plain original Lossy rejected");
        original.neuralTileSize = 128;
        require(!fsr412::validate_options(original, reason), "original Lossy accepted neural tiles");
        original.neuralTileSize = 0; original.pixelFix = true;
        require(!fsr412::validate_options(original, reason), "original Lossy accepted the pixelization fix");
        original.pixelFix = false;
    }
    {
        fsr412::Options plain;
        require(!plain.originalLossy && !fsr412::original_profile("int8-lite-lossy"), "original route became a default");
    }
    require(fsr412::profile_from_int(7) == fsr412::Profile::Int8LiteLossy
            && fsr412::profile_from_int(8) == fsr412::Profile::Int8LiteLossy, "live Lossy ID mapping rejected");
    for (unsigned tile : {0u, 128u, 256u, 512u, 1024u, 2048u}) {
        unsigned parsed = 999;
        require(fsr412::parse_neural_tile_size(std::to_string(tile), parsed) && parsed == tile, "valid neural tile rejected");
    }
    for (const char *bad : {"", "-1", "+512", "0128", "127", "513", "4096", "1.0", "512junk"}) {
        unsigned parsed = 0;
        require(!fsr412::parse_neural_tile_size(bad, parsed), "invalid neural tile accepted");
    }
    std::string error;
    options.backend = fsr412::Backend::Reference;
    require(!fsr412::validate_options(options, error) && error.find("reference") != std::string::npos,
            "Lossy reference silently accepted");
    options.profile = fsr412::Profile::Int8Lite; options.neuralTileSize = 512;
    require(!fsr412::validate_options(options, error), "ordinary Lite neural tiles accepted");
    options.neuralTileSize = 0; fsr412::apply_pixel_fix(options,true,0.5f);
    require(options.pixelFix && std::string(fsr412::profile_name(options)) == "int8-lite",
            "pixel fix reporting compatibility changed");
    options.pixelFix = false; options.optimize = 1;
    for (auto profile : {fsr412::Profile::Int8Lite, fsr412::Profile::Int8LiteLossy}) {
        options.profile = profile; options.temporalQuality = 1;
        require(!fsr412::validate_options(options, error) && error.find("4.1.1") != std::string::npos,
                "legacy temporal quality silently accepted on Lite");
    }
    options.temporalQuality = 0;

    Fsr4LiveFrameV2 input = {}; input.frame.rw = 1280; input.frame.uw = 1920;
    for (int id : {5, 6, 7, 8}) {
        fsr412::Frame frame; fsr412::Options live;
        selectedProfile = id;
        require(options_for(input, frame, live, error), "valid live mode rejected");
        require(live.profile == (id >= 7 ? fsr412::Profile::Int8LiteLossy : fsr412::Profile::Int8Lite), "live model mismatch");
        require(live.neuralTileSize == (id == 8 ? 512u : 0u) && live.pixelFix == (id == 6),
                "live tile/pixel-fix mapping mismatch");
    }
    selectedProfile = 7; setenv("FSR4_LITE_LOSSY_ORIGINAL", "1", 1);
    {
        fsr412::Frame f7; fsr412::Options l7;
        require(options_for(input, f7, l7, error) && l7.originalLossy && l7.profile == fsr412::Profile::Int8LiteLossy,
                "live original Lossy opt-in rejected");
        for (int id : {5, 6, 8}) {
            selectedProfile = id; fsr412::Options other;
            require(!options_for(input, f7, other, error), "live original opt-in accepted outside Lite Lossy");
        }
        selectedProfile = 7; setenv("FSR4_LITE_LOSSY_ORIGINAL", "yes", 1);
        require(!options_for(input, f7, l7, error), "malformed original opt-in accepted");
        setenv("FSR4_LITE_LOSSY_ORIGINAL", "0", 1);
        require(options_for(input, f7, l7, error) && !l7.originalLossy, "original opt-out rejected");
        unsetenv("FSR4_LITE_LOSSY_ORIGINAL");
    }
    selectedProfile = 8;
    setenv("FSR4_NEURAL_TILE_SIZE", "256", 1);
    fsr412::Frame frame; fsr412::Options live;
    require(options_for(input, frame, live, error) && live.neuralTileSize == 256, "live env overrides rejected");
    setenv("FSR4_NEURAL_TILE_SIZE", "0", 1);
    require(options_for(input, frame, live, error) && !live.neuralTileSize, "zero override did not disable partial tiles");
    selectedProfile = 5;
    require(!options_for(input, frame, live, error), "ordinary Lite accepted an explicit neural tile env");
    selectedProfile = 7; unsetenv("FSR4_NEURAL_TILE_SIZE"); setenv("FSR4_MODEL", "reference", 1);
    require(!options_for(input, frame, live, error) && error.find("reference") != std::string::npos, "live Lossy reference accepted");
    clear_settings_env();
    fsr412::Options liveValidation;
    // FSR4_TEMPORAL_QUALITY belongs to the 4.1.1 hook; the 4.1.2 bridge never
    // turns it into a Lite option, so a stray value cannot change the graph.
    setenv("FSR4_OPTIMIZE", "1", 1); setenv("FSR4_TEMPORAL_QUALITY", "1", 1);
    require(options_for(input, frame, liveValidation, error) && liveValidation.temporalQuality == 0,
            "live 4.1.2 accepted a legacy temporal-quality mode");
    setenv("FSR4_TEMPORAL_QUALITY", "bad", 1);
    require(options_for(input, frame, liveValidation, error), "stray temporal-quality env broke the live bridge");
    clear_settings_env();

    for (const char *profile : {"int8-lite-lossy", "int8-lite-lossy-tiled", "lite-lossy", "lite-lossy-tiled"}) {
        cli({"--profile", profile, "--reference"}, false);
        cli({"--profile", profile, "--scalar", "--neural-tile-size", "0"}, true);
        cli({"--profile", profile, "--simd", "--neural-tile-size", "2048"}, true);
        cli({"--profile", profile, "--scalar", "--pixelization-fix"}, false);
    }
    cli({"--profile", "int8-lite-lossy-original", "--scalar"}, true);
    cli({"--profile", "int8-lite-lossy-original", "--simd", "--neural-tile-size", "0"}, true);
    cli({"--profile", "int8-lite-lossy-original", "--scalar", "--neural-tile-size", "128"}, false);
    cli({"--profile", "int8-lite-lossy-original", "--scalar", "--pixelization-fix"}, false);
    cli({"--profile", "int8-lite-lossy-original", "--reference"}, false);
    cli({"--profile", "int8-lite", "--neural-tile-size", "0"}, false);
    cli({"--profile", "int8-lite-lossy", "--neural-tile-size", "513"}, false);
    cli({"--profile", "int8-lite-hair", "--scalar"}, true);
    cli({"--profile", "int8-lite-pixfix", "--scalar"}, true);
    setenv("FSR4_NEURAL_TILE_SIZE", "128", 1);
    cli({"--profile", "int8-lite-lossy-tiled", "--scalar"}, true);
    cli({"--profile", "int8-lite", "--scalar"}, false);
    setenv("FSR4_NEURAL_TILE_SIZE", "bad", 1);
    cli({"--profile", "int8-lite-lossy", "--scalar"}, false);
    clear_settings_env();
    // FSR4_TEMPORAL_QUALITY is a 4.1.1 setting; the 4.1.2 CLI has no such flag.
    setenv("FSR4_TEMPORAL_QUALITY", "1", 1);
    cli({"--profile", "int8-lite", "--scalar"}, true);
    cli({"--profile", "int8-lite-lossy", "--scalar"}, true);
    clear_settings_env();
    setenv("FSR4_OPTIMIZE", "1", 1);
    cli({"--profile", "int8-lite-lossy", "--scalar"}, true);
    cli({"--profile", "int8-lite-lossy", "--scalar", "--pixelization-fix"}, false);
    clear_settings_env();
    puts("FSR 4.1.2 CPU options passed (no GPU device created)"); return 0;
} catch (const std::exception &error) {
    fprintf(stderr, "FSR 4.1.2 CPU options test: %s\n", error.what()); return 1;
} } }
