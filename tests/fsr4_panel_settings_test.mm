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
    require(segment_for_profile(7) == -1 && segment_for_profile(8) == -1, "missing modes are shown");
    require(segment_profile(-1) == 5, "documented unselected fallback changed");
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
    require(segment_profile(segment_for_profile(7)) == 7, "Lite Lossy segment ID mismatch");
    require(segment_profile(segment_for_profile(8)) == 8, "tiled segment ID mismatch");
    {
        int modes[8]; const int count = panel_modes(modes);
        require(count == 4 && modes[0] == 5 && modes[1] == 2 && modes[2] == 7 && modes[3] == 8,
                "Lossy segments are not appended after the existing modes");
        require(segment_for_profile(6) == segment_for_profile(5), "pixelization fix must reuse the Lite segment");
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
        require(segment_for_profile(profile) == -1, "unavailable saved mode remains visible");
        fixture(base + "/initializers-lite-lossy.bin", "weights");
        g_has_lite_lossy = has_lite_lossy(fixturePin.c_str());
    }
    unlink(dist_path("panel.conf").c_str());
    rmdir((g_root + "/dist/crossover").c_str()); rmdir((g_root + "/dist").c_str());
    unlink((base + "/initializers-lite-lossy.bin").c_str()); unlink((base + "/import.json").c_str());
    rmdir(base.c_str()); rmdir((g_root + "/build").c_str()); rmdir(g_root.c_str());
    puts("panel settings passed"); return 0;
} catch (const std::exception &error) {
    fprintf(stderr, "panel settings test: %s\n", error.what()); return 1;
} } }
