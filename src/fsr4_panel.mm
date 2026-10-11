// Hook side of the FSR 4 panel: publishes the encoder status for the panel
// app (dist/crossover/FSR4 Panel.app, tools/crossover/panel_app.mm) and
// launches it once when FSR4_PANEL=1. The app is a separate process with its
// own Dock icon; the two talk through two small files in dist/crossover/:
//   status.conf  written here (rate-limited to 2 Hz, atomic rename)
//   panel.conf   written by the app (sharpness/rcas/profile), reloaded by the encoder
#include "fsr4_live.h"
#include <mach/mach_time.h>
#include <spawn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <mutex>
#include <time.h>
#include <unistd.h>

extern char **environ;

static std::string dist_dir() {
    const char *root = getenv("FSR4_ROOT");
    if (!root || !root[0]) return "";
    return std::string(root) + "/dist/crossover";
}

static double mono_ms() {
    static mach_timebase_info_data_t tb;
    if (!tb.denom) mach_timebase_info(&tb);
    return (double)mach_absolute_time() * tb.numer / tb.denom / 1e6;
}

void fsr4_panel_request(void) {
    static bool done = false;
    if (done) return;
    done = true;
    const char *e = getenv("FSR4_PANEL");
    if (!e || e[0] != '1') return;
    const std::string directory = dist_dir();
    if (directory.empty()) return;
    const std::string app = directory + "/FSR4 Panel.app";
    if (access(app.c_str(), R_OK) != 0) {
        fprintf(stderr, "[fsr4] panel app missing: %s\n", app.c_str());
        return;
    }
    // `open -g` starts the app (or just pokes the running one) without
    // stealing focus from the game
    const char *argv[] = {"/usr/bin/open", "-g", "-a", app.c_str(),
                          "--args", "--root", getenv("FSR4_ROOT"), nullptr};
    pid_t pid = 0;
    int rc = posix_spawn(&pid, argv[0], nullptr, nullptr, (char *const *)argv, environ);
    fprintf(stderr, "[fsr4] panel app %s (%s)\n", rc == 0 ? "launched" : "spawn failed", app.c_str());
}

void fsr4_panel_publish(void) {
    static std::mutex mu;
    static double last = 0;
    static std::string path, tmp;
    std::lock_guard<std::mutex> lk(mu);
    const double now = mono_ms();
    if (last > 0 && now - last < 500.0) return;
    last = now;
    if (path.empty()) {
        const std::string d = dist_dir();
        if (d.empty()) return;
        path = d + "/status.conf";
        tmp = path + ".tmp";
    }
    Fsr4LiveStatus st;
    fsr4_live_get_status(&st);
    int profile = 1;
    fsr4_live_get_profile(&profile);
    const char *version = !strncmp(st.variant, "fsr411_", 7) ? "4.1.1" : "4.0.2";
    if (!st.variant[0]) version = fsr4_live_model_version();
    FILE *f = fopen(tmp.c_str(), "w");
    if (!f) return;
    fprintf(f,
            "t=%ld\npid=%d\nactive=%d\nreason=%s\nmodel=%s\npreset=%s\nvariant=%s\n"
            "rw=%u\nrh=%u\nuw=%u\nuh=%u\nuwA=%u\nuhA=%u\nframe_ms=%.3f\n"
            "encodes=%llu\nreplaced=%llu\nstubs=%llu\nsharpness=%.3f\nfsr_version=%s\nprofile=%s\n",
            (long)time(nullptr), (int)getpid(), st.active, st.reason, st.model, st.preset, st.variant,
            st.rw, st.rh, st.uw, st.uh, st.uwA, st.uhA, st.frameMs,
            (unsigned long long)st.encodes, (unsigned long long)st.replaced, (unsigned long long)st.stubs,
            st.sharpness, version,
            profile == 1 ? "lossy" : profile == 2 ? "exact-fast" : profile == 3 ? "fsr4" : "exact");
    fclose(f);
    rename(tmp.c_str(), path.c_str());
}
