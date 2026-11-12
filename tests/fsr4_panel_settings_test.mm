// SPDX-License-Identifier: MIT
// CPU-only panel settings test. Does not launch NSApplication or the game.
#define main fsr4_panel_app_main
#include "../tools/crossover/panel_app.mm"
#undef main
#include <stdexcept>

static void require(bool value, const char *message) {
    if (!value) throw std::runtime_error(message);
}
static void fixture(const std::string &path, const char *text) {
    FILE *file = fopen(path.c_str(), "w"); require(file, "cannot write fixture");
    bool ok = fputs(text, file) >= 0;
    if (fclose(file)) ok = false;
    require(ok, "cannot save fixture");
}

static std::string receipt_fixture(const std::string &base, const char *schema = "fsr412_lite_lossy_import_v1",
                            const char *source = "d9b0a1f5781c2d633a327272f6f1c8607057df7cafd0233cf38c7395c8f7dbc5") {
    const char *weights = "weights";
    unsigned char digest[CC_SHA256_DIGEST_LENGTH];
    CC_SHA256(weights, (CC_LONG)strlen(weights), digest);
    NSMutableString *hash = [NSMutableString stringWithCapacity:64];
    for (unsigned char byte : digest) [hash appendFormat:@"%02x", byte];
    NSDictionary *receipt = @{@"schema": [NSString stringWithUTF8String:schema],
                              @"source_dll_sha256": [NSString stringWithUTF8String:source],
                              @"initializers": @{@"initializers-lite-lossy.bin": hash}};
    NSData *data = [NSJSONSerialization dataWithJSONObject:receipt options:0 error:nil];
    require([data writeToFile:[NSString stringWithUTF8String:(base + "/import.json").c_str()] atomically:YES], "cannot save receipt");
    return hash.UTF8String;
}

int main() { @autoreleasepool { try {
    char temporary[] = "/tmp/fsr4-panel-settings-XXXXXX";
    require(mkdtemp(temporary), "cannot create temporary root");
    g_root = temporary;
    require(mkdir((g_root + "/build").c_str(), 0700) == 0, "cannot create build fixture");
    const std::string base = g_root + "/build/fsr412_lossy";
    require(mkdir(base.c_str(), 0700) == 0, "cannot create weights fixture");
    g_has402 = g_has_lossy2 = false;
    g_has_lite_lossy = has_lite_lossy();
    require(!g_has_lite_lossy, "missing weights offered Lite Lossy");
    require(stop_for_profile(7) == -1 && stop_for_profile(8) == -1, "missing modes are shown");
    require(mode_profile(-1) == 5, "documented unselected fallback changed");
    fixture(base + "/initializers-lite-lossy.bin", "weights");
    require(!has_lite_lossy(), "initializer without receipt offered Lite Lossy");
    unlink((base + "/initializers-lite-lossy.bin").c_str());
    const std::string fixturePin = receipt_fixture(base);
    require(!has_lite_lossy(fixturePin.c_str()), "receipt without initializer offered Lite Lossy");
    fixture(base + "/initializers-lite-lossy.bin", "weights");
    for (const char *malformed : {"not JSON", "[]", "{}", "{\"initializers\": []}"}) {
        fixture(base + "/import.json", malformed);
        require(!has_lite_lossy(fixturePin.c_str()), "malformed receipt offered Lite Lossy");
    }
    receipt_fixture(base, "other");
    require(!has_lite_lossy(fixturePin.c_str()), "wrong receipt schema offered Lite Lossy");
    receipt_fixture(base, "fsr412_lite_lossy_import_v1", "untrusted DLL");
    require(!has_lite_lossy(fixturePin.c_str()), "wrong source DLL offered Lite Lossy");
    receipt_fixture(base);
    fixture(base + "/initializers-lite-lossy.bin", "ordinary Lite or corrupt bytes");
    require(!has_lite_lossy(fixturePin.c_str()), "hash mismatch offered Lite Lossy");
    fixture(base + "/initializers-lite-lossy.bin", "");
    require(!has_lite_lossy(fixturePin.c_str()), "empty initializer offered Lite Lossy");
    fixture(base + "/initializers-lite-lossy.bin", "weights");
    require(!has_lite_lossy(), "production pin accepted self-consistent replacement weights");
    g_has_lite_lossy = has_lite_lossy(fixturePin.c_str());
    require(g_has_lite_lossy, "both required inputs did not enable Lite Lossy");
    require(mode_profile(stop_for_profile(7)) == 7, "Lite Lossy stop ID mismatch");
    require(stop_for_profile(8) == -1, "tiled is still offered in the panel");
    {
        int modes[8]; const int count = panel_modes(modes);
        require(count == 3 && modes[0] == 2 && modes[1] == 5 && modes[2] == 7,
                "quality/performance stops have the wrong order");
        require(stop_for_profile(6) == stop_for_profile(5), "pixelization fix must reuse the Lite stop");
        for (int optional = 0; optional < 4; ++optional) {
            g_has402 = (optional & 1) != 0; g_has_lossy2 = (optional & 2) != 0;
            const int available = panel_modes(modes);
            require(modes[0] == 2 && modes[available - 1] == 7, "slider endpoints changed");
            for (int i = 0; i < available; ++i) {
                require(modes[i] != 8, "tiled is selectable");
                require(mode_profile(i) == modes[i] && stop_for_profile(modes[i]) == i, "stop/ID round trip failed");
            }
        }
        g_has402 = g_has_lossy2 = false;
        require(profile_after_change(8, stop_for_profile(5), false, true) == 8, "sharpness edit silently replaced tiled");
        require(profile_after_change(7, -1, true, false) == 7, "invalid stop replaced the saved mode");
        require(profile_after_change(8, stop_for_profile(5), true, false) == 5, "explicit mode switch failed");
        require(profile_after_change(2, stop_for_profile(5), true, true) == 6, "Lite switch lost the pixelization fix");
        require(profile_after_change(6, stop_for_profile(2), true, true) == 2, "fix forced the wrong model");
        require(profile_after_change(6, -1, false, false) == 5, "disabling the fix did not return to Lite");
        require([kModeExplanation containsString:@"Render and output resolution stay unchanged"], "resolution disclaimer missing");
        require([kModeExplanation containsString:@"stutters or pauses"] && [kModeExplanation containsString:@"Try different modes"],
                "switching guidance missing");
    }
    require(profile_from_name("lite-lossy", -1) == 7 && profile_from_name("7", -1) == 7, "Lossy aliases rejected");
    require(profile_from_name("lite-lossy-tiled", -1) == 8 && profile_from_name("8", -1) == 8, "tiled aliases rejected");
    require(mkdir((g_root + "/dist").c_str(), 0700) == 0, "cannot create dist fixture");
    require(mkdir((g_root + "/dist/crossover").c_str(), 0700) == 0, "cannot create panel fixture");
    unsetenv("FSR4_PROFILE");
    for (int id : {7, 8}) {
        require(write_settings(.6f, false, id, false, .5f), "cannot save Lossy settings");
        unlink((base + "/initializers-lite-lossy.bin").c_str());
        g_has_lite_lossy = has_lite_lossy();
        float sharp, strength; bool rcas, fix; int profile;
        read_settings(sharp, rcas, profile, fix, strength);
        require(profile == id && fabs(sharp - .6f) < 1e-4 && !rcas && !fix, "unavailable saved mode was discarded");
        require(stop_for_profile(profile) == -1, "unavailable saved mode remains selectable");
        fixture(base + "/initializers-lite-lossy.bin", "weights");
        g_has_lite_lossy = has_lite_lossy(fixturePin.c_str());
    }
    unsetenv("FSR4_FG");unsetenv("FSR4_FG_SETTINGS");
    require([frame_generation_warning() containsString:@"!"]&&[frame_generation_warning() containsString:@"Experimental feature"],"FG warning missing");
    for(auto mode:{fsr4_fg::Mode::MetalFX,fsr4_fg::Mode::ANR}){
        require(write_frame_generation(mode),"cannot save FG mode");
        require(read_frame_generation()==mode,"saved FG mode not reflected");
        float sharp,strength;bool rc,fix;int profile;read_settings(sharp,rc,profile,fix,strength);
        require(profile==8&&fabs(sharp-.6f)<1e-4&&!rc,"FG mode changed SR settings");
    }
    require(write_settings(.2f,true,7,false,.5f),"cannot save SR settings");
    require(read_frame_generation()==fsr4_fg::Mode::ANR,"SR change discarded FG mode");
    for(const char *bad:{"", "mode=other\n", "mode=anr\nmode=metalfx\n", "mode=anr\ngarbage\n"})
        require(!fsr4_fg::parse_settings(bad).valid,"malformed FG settings accepted");
    require(fsr4_fg::parse_settings("# mode\nschema=fsr4.fg.settings.v1\nmode=metalfx\n").valid,"valid FG file rejected");
    require(!write_frame_generation(static_cast<fsr4_fg::Mode>(99)),"unknown FG mode saved");
    unlink(dist_path("fg-settings.conf").c_str());
    unlink(dist_path("panel.conf").c_str());
    rmdir((g_root + "/dist/crossover").c_str()); rmdir((g_root + "/dist").c_str());
    unlink((base + "/initializers-lite-lossy.bin").c_str()); unlink((base + "/import.json").c_str());
    rmdir(base.c_str()); rmdir((g_root + "/build").c_str()); rmdir(g_root.c_str());
    puts("panel settings passed"); return 0;
} catch (const std::exception &error) {
    fprintf(stderr, "panel settings test: %s\n", error.what()); return 1;
} } }
