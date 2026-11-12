// FSR 4 panel: a small Dock application (dist/crossover/FSR4 Panel.app) that
// shows what the CrossOver hook is doing and sets its profile/RCAS sharpness. It is a
// separate process; the hook publishes dist/crossover/status.conf at 2 Hz and
// reloads dist/crossover/panel.conf (written here) when its mtime changes.
// Diagnostic captures use an atomic capture.request and capture-status.conf.
// The window is a non-activating floating panel, so clicking the slider does
// not take focus away from the game.
//
//   FSR4 Panel [--root <repo>] [--snapshot <png>] [--make-icon <dir>]
#import <AppKit/AppKit.h>
#import <QuartzCore/QuartzCore.h>
#include <CommonCrypto/CommonDigest.h>
#include <sys/stat.h>
#include <string>
#include <map>
#include <cmath>
#include <mach-o/dyld.h>
#include "../../src/fsr4_fg_settings.h"

struct Status {
    bool ok = false, active = false;
    std::string reason, model, preset, version;
    int profile = 5;
    unsigned rw = 0, rh = 0, uw = 0, uh = 0;
    double frameMs = 0, t = 0;
    unsigned long long replaced = 0, stubs = 0;
};

static std::string g_root;
static std::string dist_path(const char *name) { return g_root + "/dist/crossover/" + name; }

static std::map<std::string, std::string> read_kv(const std::string &path) {
    std::map<std::string, std::string> kv;
    FILE *f = fopen(path.c_str(), "r");
    if (!f) return kv;
    char line[512];
    while (fgets(line, sizeof line, f)) {
        char *eq = strchr(line, '=');
        if (!eq) continue;
        *eq = 0;
        std::string v(eq + 1);
        while (!v.empty() && (v.back() == '\n' || v.back() == '\r')) v.pop_back();
        kv[line] = v;
    }
    fclose(f);
    return kv;
}

// IDs 0-6 retain the existing modes; 7/8 are FSR 4.1.2 Lite Lossy.
static int profile_from_name(const std::string &name, int fallback) {
    if (name == "exact" || name == "0") return 0;
    if (name == "lossy" || name == "1") return 1;
    if (name == "exact-fast" || name == "2") return 2;
    if (name == "lossy2" || name == "4") return 4;
    if (name == "int8-lite" || name == "lite" || name == "5") return 5;
    if (name == "int8-lite-pixfix" || name == "lite-pixfix" || name == "int8-lite-hair" || name == "lite-hair" || name == "6") return 6;
    if (name == "int8-lite-lossy" || name == "lite-lossy" || name == "7") return 7;
    if (name == "int8-lite-lossy-tiled" || name == "lite-lossy-tiled" || name == "8") return 8;
    if (name == "fsr4" || name == "3") return 3;
    return fallback;
}
static const char *profile_label(int profile) {
    return profile == 1 ? "lossy" : profile == 2 ? "exact-fast" : profile == 3 ? "fsr4"
         : profile == 4 ? "lossy2" : profile == 5 ? "int8-lite" : profile == 6 ? "int8-lite-pixfix"
         : profile == 7 ? "int8-lite-lossy" : profile == 8 ? "int8-lite-lossy-tiled" : "exact";
}
// The old Lite (profile 1) is retired from the panel. Keep its saved ID
// readable for compatibility; Lite 2 (profile 4) is now displayed as Lossy.
// AMD 4.1.2 Lite (profile 5) uses its own INT8 architecture and weights.
// Profile 6 is the same model with the POST pixelization fix; the panel shows
// it as the Lite stop plus the "Pixelization fix" checkbox. Tiled remains
// readable as a saved/CLI profile, but is not a selectable panel stop.
// FSR 4 is offered only when the 4.0.2 weights are present (the 4.1-only
// release omits them).
// Lossy profiles are offered only with both pinned inputs, never by an
// environment override or a fallback to the ordinary Lite initializer.
static bool g_has402 = true, g_has_lossy2 = false, g_has_lite_lossy = false;
static const char *kLiteLossyInitializerSha256 = "c77c2d1a4d119ed76d94b8c23335490cc38280db3346c72396a1e1acb0523cfc";
static bool has_lite_lossy(const char *expectedHash = kLiteLossyInitializerSha256) {
    struct stat st = {};
    for (const char *name : {"initializers-lite-lossy.bin", "import.json"}) {
        const std::string path = g_root + "/build/fsr412_lossy/" + name;
        if (stat(path.c_str(), &st) != 0 || !S_ISREG(st.st_mode)) return false;
    }
    const std::string base = g_root + "/build/fsr412_lossy/";
    NSData *receiptData = [NSData dataWithContentsOfFile:[NSString stringWithUTF8String:(base + "import.json").c_str()]];
    if (!receiptData) return false;
    id receipt = [NSJSONSerialization JSONObjectWithData:receiptData options:0 error:nil];
    if (![receipt isKindOfClass:NSDictionary.class]
        || ![receipt[@"schema"] isEqual:@"fsr412_lite_lossy_import_v1"]
        || ![receipt[@"source_dll_sha256"] isEqual:@"d9b0a1f5781c2d633a327272f6f1c8607057df7cafd0233cf38c7395c8f7dbc5"]) return false;
    id initializers = receipt[@"initializers"];
    if (![initializers isKindOfClass:NSDictionary.class]) return false;
    id expected = initializers[@"initializers-lite-lossy.bin"];
    if (![expected isKindOfClass:NSString.class] || [expected length] != CC_SHA256_DIGEST_LENGTH * 2) return false;
    NSData *weights = [NSData dataWithContentsOfFile:[NSString stringWithUTF8String:(base + "initializers-lite-lossy.bin").c_str()]];
    if (!weights.length || weights.length > UINT32_MAX) return false;
    unsigned char digest[CC_SHA256_DIGEST_LENGTH];
    CC_SHA256(weights.bytes, (CC_LONG)weights.length, digest);
    NSMutableString *actual = [NSMutableString stringWithCapacity:CC_SHA256_DIGEST_LENGTH * 2];
    for (unsigned char byte : digest) [actual appendFormat:@"%02x", byte];
    return [actual isEqualToString:expected] && expectedHash
        && [actual isEqualToString:[NSString stringWithUTF8String:expectedHash]];
}
static bool has_fsr402() {
    struct stat st = {};
    return stat((g_root + "/third_party/fsr4/internal/shaders/fsr4_model_v07_i8_quality/initializers.bin").c_str(), &st) == 0;
}
static bool has_lossy2() {
    const char *env = getenv("FSR4_LOSSY2_WEIGHTS");
    const std::string path = env && *env ? std::string(env) : g_root + "/build/fsr411_lossy2/weights_lossy2.npz";
    struct stat st = {};
    return stat(path.c_str(), &st) == 0;
}
// A suggested quality/performance progression, not a measured quality score.
// IDs stay stable; positions may change when optional models are unavailable.
static int panel_modes(int *modes) {
    int count = 0;
    modes[count++] = 2;
    if (g_has402) modes[count++] = 3;
    if (g_has_lossy2) modes[count++] = 4;
    modes[count++] = 5;
    if (g_has_lite_lossy) modes[count++] = 7;
    return count;
}
static int mode_profile(NSInteger stop) {
    int modes[8], count = panel_modes(modes);
    return stop >= 0 && stop < count ? modes[stop] : 5;
}
static NSInteger stop_for_profile(int profile) {
    int modes[8], count = panel_modes(modes);
    if (profile == 6) profile = 5;
    for (int i = 0; i < count; ++i) if (modes[i] == profile) return i;
    return -1;
}
static const char *mode_title(int profile, bool abbreviated = false) {
    switch (profile) {
        case 2: return abbreviated ? "Exact" : "FSR 4.1.2 Exact";
        case 3: return abbreviated ? "FSR 4" : "FSR 4 (4.0.2)";
        case 4: return abbreviated ? "Lossy" : "FSR 4.1.2 Lossy";
        case 5: return abbreviated ? "Lite" : "FSR 4.1.2 Lite";
        case 6: return abbreviated ? "Lite" : "FSR 4.1.2 Lite + Pixelization fix";
        case 7: return abbreviated ? "Lite Lossy" : "FSR 4.1.2 Lite Lossy";
        case 8: return "Lite Lossy Tiled (CLI only)";
        case 1: return "Legacy Lossy (CLI only)";
        default: return "Exact reference (CLI only)";
    }
}
// Unrelated controls must never replace a saved unavailable/retired profile.
static int profile_after_change(int savedProfile, NSInteger stop, bool modeChanged, bool fix) {
    int modes[8], count = panel_modes(modes);
    int profile = modeChanged && stop >= 0 && stop < count ? mode_profile(stop) : savedProfile;
    return profile == 5 || profile == 6 ? (fix ? 6 : 5) : profile;
}
static NSString *const kModeExplanation =
    @"Switch between FSR processing modes. Render and output resolution stay unchanged. "
     "Switching may cause stutters or pauses. Try different modes and choose the best balance of "
     "image quality and performance for your game.";

static Status read_status() {
    Status s;
    auto kv = read_kv(dist_path("status.conf"));
    if (kv.empty()) return s;
    s.ok = true;
    s.active = atoi(kv["active"].c_str()) != 0;
    s.reason = kv["reason"]; s.model = kv["model"]; s.preset = kv["preset"];
    s.version = kv["fsr_version"];
    if (s.version.empty()) s.version = "4.0.2";
    s.profile = profile_from_name(kv["profile"], 5);
    s.rw = atoi(kv["rw"].c_str()); s.rh = atoi(kv["rh"].c_str());
    s.uw = atoi(kv["uw"].c_str()); s.uh = atoi(kv["uh"].c_str());
    s.frameMs = atof(kv["frame_ms"].c_str());
    s.t = atof(kv["t"].c_str());
    s.replaced = strtoull(kv["replaced"].c_str(), nullptr, 10);
    s.stubs = strtoull(kv["stubs"].c_str(), nullptr, 10);
    return s;
}

static void read_settings(float &sharp, bool &rcas, int &profile, bool &pixelFix, float &strength,
                          int fallbackProfile = -1) {
    sharp = 0.4f; rcas = true; profile = fallbackProfile >= 0 ? fallbackProfile : 5; pixelFix = false; strength = 0.5f;
    if (const char *value = getenv("FSR4_PROFILE")) profile = profile_from_name(value, profile);
    auto kv = read_kv(dist_path("panel.conf"));
    if (kv.count("sharpness")) sharp = (float)atof(kv["sharpness"].c_str());
    if (kv.count("rcas")) rcas = atoi(kv["rcas"].c_str()) != 0;
    profile = profile_from_name(kv["profile"], profile);
    // Remembered while another mode is selected; profile 6 implies it.
    if (kv.count("pixelization_fix")) pixelFix = atoi(kv["pixelization_fix"].c_str()) != 0;
    if (profile == 6) pixelFix = true;
    if (kv.count("pixelization_strength")) strength = (float)atof(kv["pixelization_strength"].c_str());
    if (!std::isfinite(strength)) strength = .5f;
    strength = strength < 0 ? 0 : strength > 1 ? 1 : strength;
    if (!std::isfinite(sharp)) sharp = .4f;
    if (sharp < 0) sharp = 0; if (sharp > 1) sharp = 1;
}

static bool write_settings(float sharp, bool rcas, int profile, bool pixelFix, float strength) {
    const std::string p = dist_path("panel.conf"), tmp = p + ".tmp." + std::to_string(getpid());
    FILE *f = fopen(tmp.c_str(), "w");
    if (!f) return false;
    bool ok = fprintf(f, "sharpness=%.3f\nrcas=%d\nprofile=%s\npixelization_fix=%d\npixelization_strength=%.3f\n",
                      sharp, rcas ? 1 : 0, profile_label(profile), pixelFix ? 1 : 0, strength) > 0;
    if (fclose(f) != 0) ok = false;
    if (!ok || rename(tmp.c_str(), p.c_str()) != 0) { unlink(tmp.c_str()); return false; }
    return true;
}

static NSString *frame_generation_warning() {
    return @"! Experimental feature. ANR may cause artifacts and judder.";
}
static fsr4_fg::Mode read_frame_generation() {
    fsr4_fg::Mode mode = fsr4_fg::Mode::MetalFX;
    auto status = read_kv(dist_path("fg-status.conf"));
    if (!fsr4_fg::parse_mode(status["selected_mode"], mode) && atoi(status["active"].c_str()))
        mode = fsr4_fg::Mode::ANR;
    if (const char *env = getenv("FSR4_FG")) fsr4_fg::parse_mode(env, mode);
    auto path = fsr4_fg::settings_path(g_root.c_str(), getenv("FSR4_FG_SETTINGS"));
    if (!path.empty()) {
        auto saved = fsr4_fg::read_settings(path);
        if (saved.valid) mode = saved.mode;
    }
    return mode;
}
static bool write_frame_generation(fsr4_fg::Mode mode) {
    auto path = fsr4_fg::settings_path(g_root.c_str(), getenv("FSR4_FG_SETTINGS"));
    return !path.empty() && fsr4_fg::write_settings(path, mode);
}

static NSString *capture_status_stamp() {
    struct stat st = {};
    if (stat(dist_path("capture-status.conf").c_str(), &st) != 0) return @"";
    return [NSString stringWithFormat:@"%llu:%llu:%lld:%ld:%lld",
            (unsigned long long)st.st_dev, (unsigned long long)st.st_ino,
            (long long)st.st_mtimespec.tv_sec, st.st_mtimespec.tv_nsec, (long long)st.st_size];
}

static bool write_capture_request(NSString **reason) {
    NSData *request = [@"frames=8\ndelay_ms=1500\n" dataUsingEncoding:NSUTF8StringEncoding];
    NSString *path = [NSString stringWithUTF8String:dist_path("capture.request").c_str()];
    NSError *error = nil;
    if ([request writeToFile:path options:NSDataWritingAtomic error:&error]) return true;
    if (reason) *reason = error.localizedDescription;
    return false;
}

@interface PanelApp : NSObject <NSApplicationDelegate>
@property(nonatomic, strong) NSPanel *panel;
@property(nonatomic, strong) NSTextField *status, *res, *frame, *sharpVal, *fixVal, *fgStatus, *fgWarning;
@property(nonatomic, strong) NSSegmentedControl *frameGeneration;
@property(nonatomic, strong) NSString *fgSettingsError;
@property(nonatomic, strong) NSSlider *slider, *fixStrength;
@property(nonatomic, strong) NSButton *rcas, *pixelFix;
@property(nonatomic, strong) NSSlider *modeSlider;
@property(nonatomic, strong) NSTextField *modeValue, *modeExplanation, *profileHint;
@property(nonatomic) int savedProfile;
@property(nonatomic, strong) NSString *settingsError;
@property(nonatomic, strong) NSButton *captureButton;
@property(nonatomic, strong) NSTextField *captureMessage;
@property(nonatomic, strong) NSString *captureStatusBaseline, *captureRequestError;
@property(nonatomic) BOOL awaitingCaptureStatus;
@property(nonatomic, strong) NSTimer *timer;
@end

@implementation PanelApp

static NSTextField *label(NSString *s, CGFloat x, CGFloat y, CGFloat w, BOOL bold) {
    NSTextField *t = [NSTextField labelWithString:s];
    t.frame = NSMakeRect(x, y, w, 18);
    t.font = bold ? [NSFont boldSystemFontOfSize:12] : [NSFont systemFontOfSize:12];
    t.textColor = NSColor.whiteColor;
    t.lineBreakMode = NSLineBreakByTruncatingTail;
    return t;
}

- (void)build {
    const CGFloat W = 660, H = 550;
    NSUInteger style = NSWindowStyleMaskTitled | NSWindowStyleMaskClosable | NSWindowStyleMaskUtilityWindow
                     | NSWindowStyleMaskNonactivatingPanel | NSWindowStyleMaskHUDWindow;
    NSPanel *p = [[NSPanel alloc] initWithContentRect:NSMakeRect(40, 80, W, H)
                                            styleMask:style backing:NSBackingStoreBuffered defer:NO];
    p.title = @"FSR 4";
    p.level = NSStatusWindowLevel;
    p.collectionBehavior = NSWindowCollectionBehaviorCanJoinAllSpaces | NSWindowCollectionBehaviorFullScreenAuxiliary
                         | NSWindowCollectionBehaviorStationary;
    p.hidesOnDeactivate = NO;
    p.floatingPanel = YES;
    p.becomesKeyOnlyIfNeeded = YES;
    p.releasedWhenClosed = NO;
    p.alphaValue = 0.94;
    p.frameAutosaveName = @"FSR4Panel";
    // Keep the saved position, but never restore the old 320-point layout size.
    [p setContentSize:NSMakeSize(W, H)];
    NSView *v = p.contentView;

    CGFloat y = H - 26;
    [v addSubview:label(@"FSR 4 replacing DLSS (CrossOver)", 12, y, W - 24, YES)]; y -= 22;
    self.status = label(@"", 12, y, W - 24, NO); [v addSubview:self.status]; y -= 20;
    self.res = label(@"", 12, y, W - 24, NO); [v addSubview:self.res]; y -= 20;
    self.frame = label(@"", 12, y, W - 24, NO); [v addSubview:self.frame]; y -= 20;
    self.fgStatus = label(@"FG: no data", 12, y, W - 24, NO); [v addSubview:self.fgStatus]; y -= 30;
    self.fgStatus.accessibilityLabel = @"Frame generation status";
    [v addSubview:label(@"Frame generation", 12, y, 180, YES)];
    self.frameGeneration = [[NSSegmentedControl alloc] initWithFrame:NSMakeRect(202, y - 4, W - 214, 26)];
    self.frameGeneration.segmentCount = 2;
    [self.frameGeneration setLabel:@"MetalFX" forSegment:0];
    [self.frameGeneration setLabel:@"ANR" forSegment:1];
    self.frameGeneration.trackingMode = NSSegmentSwitchTrackingSelectOne;
    self.frameGeneration.target = self; self.frameGeneration.action = @selector(frameGenerationChanged:);
    self.frameGeneration.accessibilityLabel = @"Frame generation";
    self.frameGeneration.toolTip = @"MetalFX: Apple's original method, no ANR GPU work. ANR: experimental generator "
                                   @"with a local runtime. DLSS Frame Generation must be enabled in the game.";
    [v addSubview:self.frameGeneration]; y -= 30;
    self.fgWarning = label(frame_generation_warning(), 12, y, W - 24, NO);
    self.fgWarning.font = [NSFont systemFontOfSize:11];
    self.fgWarning.textColor = NSColor.systemOrangeColor;
    self.fgWarning.accessibilityLabel = @"Experimental frame generation";
    [v addSubview:self.fgWarning]; y -= 34;

    float sharp, strength; bool rc, fix; int profile;
    read_settings(sharp, rc, profile, fix, strength);
    [v addSubview:label(@"FSR mode", 12, y, 74, YES)];
    self.modeValue = label(@"", 90, y, W - 102, YES);
    self.modeValue.alignment = NSTextAlignmentRight;
    [v addSubview:self.modeValue]; y -= 28;
    g_has402 = has_fsr402();
    g_has_lossy2 = has_lossy2();
    g_has_lite_lossy = has_lite_lossy();
    int modes[8], count = panel_modes(modes);
    self.savedProfile = profile;
    [v addSubview:label(@"Quality", 18, y, 100, YES)];
    NSTextField *performance = label(@"Performance", W - 138, y, 120, YES);
    performance.alignment = NSTextAlignmentRight;
    [v addSubview:performance]; y -= 30;
    NSInteger stop = stop_for_profile(profile);
    self.modeSlider = [NSSlider sliderWithValue:stop >= 0 ? stop : stop_for_profile(5)
        minValue:0 maxValue:count - 1 target:self action:@selector(changed:)];
    self.modeSlider.frame = NSMakeRect(64, y - 4, W - 128, 28);
    self.modeSlider.numberOfTickMarks = count;
    self.modeSlider.allowsTickMarkValuesOnly = YES;
    self.modeSlider.tickMarkPosition = NSTickMarkPositionBelow;
    // One change on release avoids recompiling every mode crossed while dragging.
    self.modeSlider.continuous = NO;
    self.modeSlider.toolTip = kModeExplanation;
    self.modeSlider.accessibilityLabel = @"FSR processing mode, quality to performance";
    [v addSubview:self.modeSlider];
    for (int i = 0; i < count; ++i) {
        const CGFloat center = 70 + (W - 140) * i / (count - 1);
        NSTextField *name = label([NSString stringWithUTF8String:mode_title(modes[i], true)],
                                  center - 58, y - 24, 116, NO);
        name.alignment = NSTextAlignmentCenter;
        name.font = [NSFont systemFontOfSize:11];
        name.textColor = NSColor.lightGrayColor;
        [v addSubview:name];
    }
    y -= 50;
    self.modeExplanation = [NSTextField wrappingLabelWithString:kModeExplanation];
    self.modeExplanation.frame = NSMakeRect(12, y - 48, W - 24, 48);
    self.modeExplanation.font = [NSFont systemFontOfSize:11];
    self.modeExplanation.textColor = NSColor.lightGrayColor;
    self.modeExplanation.maximumNumberOfLines = 3;
    [v addSubview:self.modeExplanation]; y -= 72;
    [v addSubview:label(@"Sharpness", 12, y, 70, NO)];
    self.slider = [NSSlider sliderWithValue:sharp minValue:0.0 maxValue:1.0 target:self action:@selector(changed:)];
    self.slider.frame = NSMakeRect(84, y - 2, W - 84 - 12 - 50, 22);
    self.slider.continuous = YES;
    [v addSubview:self.slider];
    self.sharpVal = label(@"", W - 12 - 46, y, 46, NO);
    self.sharpVal.alignment = NSTextAlignmentRight;
    [v addSubview:self.sharpVal]; y -= 26;
    self.rcas = [NSButton checkboxWithTitle:@"RCAS (FSR sharpening)" target:self action:@selector(changed:)];
    self.rcas.frame = NSMakeRect(12, y, W / 2 - 24, 18);
    self.rcas.state = rc ? NSControlStateValueOn : NSControlStateValueOff;
    self.rcas.attributedTitle = [[NSAttributedString alloc] initWithString:self.rcas.title
        attributes:@{NSForegroundColorAttributeName: NSColor.whiteColor, NSFontAttributeName: [NSFont systemFontOfSize:12]}];
    [v addSubview:self.rcas];
    self.pixelFix = [NSButton checkboxWithTitle:@"Pixelization fix" target:self action:@selector(changed:)];
    self.pixelFix.frame = NSMakeRect(W / 2, y, 140, 18);
    self.pixelFix.state = fix ? NSControlStateValueOn : NSControlStateValueOff;
    self.pixelFix.attributedTitle = [[NSAttributedString alloc] initWithString:self.pixelFix.title
        attributes:@{NSForegroundColorAttributeName: NSColor.whiteColor, NSFontAttributeName: [NSFont systemFontOfSize:12]}];
    self.pixelFix.toolTip = @"FSR 4.1.2 Lite only: a wider current-frame filter where the network rejects history, "
                            @"so moving hair and thin detail are less blocky. Switching resets history.";
    [v addSubview:self.pixelFix];
    // Strength applies on the next frame; no shader rebuild or history reset.
    self.fixStrength = [NSSlider sliderWithValue:strength minValue:0.0 maxValue:1.0 target:self action:@selector(changed:)];
    self.fixStrength.frame = NSMakeRect(W / 2 + 146, y - 2, W / 2 - 146 - 12 - 50, 22);
    self.fixStrength.continuous = YES;
    self.fixStrength.toolTip = @"Pixelization fix strength: 0 keeps AMD's filter, 0.5 is the default, 1 widens it most (softer).";
    self.fixStrength.accessibilityLabel = @"Pixelization fix strength";
    [v addSubview:self.fixStrength];
    self.fixVal = label(@"", W - 12 - 46, y, 46, NO);
    self.fixVal.alignment = NSTextAlignmentRight;
    [v addSubview:self.fixVal]; y -= 24;
    self.profileHint = label(@"", 12, y, W - 24, NO);
    self.profileHint.font = [NSFont systemFontOfSize:11];
    self.profileHint.textColor = NSColor.lightGrayColor;
    [v addSubview:self.profileHint]; y -= 34;

    self.captureButton = [NSButton buttonWithTitle:@"Record diagnostics"
        target:self action:@selector(requestCapture:)];
    self.captureButton.bezelStyle = NSBezelStyleRounded;
    self.captureButton.frame = NSMakeRect(12, y - 4, W - 24, 30);
    self.captureButton.toolTip = @"Record 8 frames 1.5 seconds after clicking.";
    [v addSubview:self.captureButton];
    self.captureMessage = [NSTextField wrappingLabelWithString:@"In 1.5 s: move the camera"];
    self.captureMessage.frame = NSMakeRect(12, 12, W - 24, 54);
    self.captureMessage.font = [NSFont systemFontOfSize:11];
    self.captureMessage.textColor = NSColor.lightGrayColor;
    self.captureMessage.maximumNumberOfLines = 3;
    self.captureMessage.accessibilityLabel = @"Diagnostics recording status";
    [v addSubview:self.captureMessage];

    self.panel = p;
    [self refresh];
    [p orderFrontRegardless];
    self.timer = [NSTimer scheduledTimerWithTimeInterval:0.5 target:self selector:@selector(tick) userInfo:nil repeats:YES];
    [[NSRunLoop mainRunLoop] addTimer:self.timer forMode:NSRunLoopCommonModes];
}

- (void)changed:(id)sender {
    const bool fix = self.pixelFix.state == NSControlStateValueOn;
    const int profile = profile_after_change(self.savedProfile, (NSInteger)llround(self.modeSlider.doubleValue),
                                              sender == self.modeSlider, fix);
    const bool saved = write_settings((float)self.slider.doubleValue, self.rcas.state == NSControlStateValueOn,
                                     profile, fix, (float)self.fixStrength.doubleValue);
    self.settingsError = saved ? nil : @"Could not save panel settings.";
    [self refresh];
}

- (void)frameGenerationChanged:(id)sender {
    const NSInteger selected = self.frameGeneration.selectedSegment;
    if (selected < 0 || selected > 1) return;
    const auto mode = selected == 1 ? fsr4_fg::Mode::ANR : fsr4_fg::Mode::MetalFX;
    self.fgSettingsError = write_frame_generation(mode) ? nil : @"! Could not save the frame generation mode.";
    [self refresh];
}

- (void)refreshFrameGeneration {
    auto fg = read_kv(dist_path("fg-status.conf"));
    const auto requested = read_frame_generation();
    self.frameGeneration.selectedSegment = requested == fsr4_fg::Mode::ANR ? 1 : 0;
    self.frameGeneration.enabled = !fsr4_fg::settings_path(g_root.c_str(), getenv("FSR4_FG_SETTINGS")).empty();
    self.fgWarning.stringValue = self.fgSettingsError ?: frame_generation_warning();
    self.fgWarning.textColor = self.fgSettingsError ? NSColor.systemRedColor : NSColor.systemOrangeColor;
    const double fgAge = fg.empty() ? 1e9 : (double)time(NULL) - atof(fg["t"].c_str());
    if (!fg.empty() && fg.count("selected_mode") && fg["selected_mode"] != fsr4_fg::name(requested))
        self.fgStatus.stringValue = @"FG: switching mode...";
    else if (requested == fsr4_fg::Mode::MetalFX)
        self.fgStatus.stringValue = @"FG: MetalFX selected (original encode, no ANR)";
    else if (fg.empty()) self.fgStatus.stringValue = @"FG: no data (enable DLSS FG in the game)";
    else if (fgAge > 3) self.fgStatus.stringValue = @"FG: no recent frames";
    else if (atoi(fg["active"].c_str())) {
        const char *ui = (fg["ui_mode"] == "precomposited-copy" || fg["ui_mode"] == "precomposited-adaptive") ? ", HUD recomposited"
                       : fg["ui_mode"] == "premultiplied-overlay" ? ", UI overlay" : "";
        self.fgStatus.stringValue = [NSString stringWithFormat:@"FG: ANR active (2x), generated %s%s", fg["generated"].c_str(), ui];
    } else
        self.fgStatus.stringValue = [NSString stringWithFormat:@"FG: original MetalFX, %s", fg["reason"].c_str()];
    self.fgStatus.toolTip = [NSString stringWithFormat:@"UI: %s, precomposited frames %s, overlay frames %s",
        fg["ui_mode"].c_str(), fg["ui_composited"].c_str(), fg["ui_overlay"].c_str()];
}

- (void)requestCapture:(id)sender {
    self.captureStatusBaseline = capture_status_stamp();
    NSString *reason = nil;
    if (!write_capture_request(&reason)) {
        self.captureRequestError = reason ?: @"Could not create the recording request.";
        self.captureMessage.stringValue = @"Could not request diagnostics.";
        self.captureMessage.textColor = NSColor.systemRedColor;
        self.captureMessage.toolTip = self.captureRequestError;
        return;
    }
    self.captureRequestError = nil;
    self.awaitingCaptureStatus = YES;
    self.captureMessage.stringValue = @"In 1.5 s: move the camera";
    self.captureMessage.textColor = NSColor.lightGrayColor;
    self.captureMessage.toolTip = nil;
}

- (void)refreshCapture {
    if (self.captureRequestError) return;
    NSString *stamp = capture_status_stamp();
    // Keep the new request's hint until the writer replaces the old status.
    if (self.awaitingCaptureStatus && [stamp isEqualToString:self.captureStatusBaseline]) return;
    auto kv = read_kv(dist_path("capture-status.conf"));
    const std::string &state = kv["state"];
    if (state != "waiting" && state != "recording" && state != "writing"
        && state != "complete" && state != "error") return;
    NSString *message = [NSString stringWithUTF8String:kv["message"].c_str()];
    if (!message.length) return;
    self.awaitingCaptureStatus = NO;
    self.captureMessage.stringValue = message;
    self.captureMessage.textColor = state == "error" ? NSColor.systemRedColor : NSColor.lightGrayColor;
    NSString *directory = [NSString stringWithUTF8String:kv["directory"].c_str()];
    self.captureMessage.toolTip = directory.isAbsolutePath
        ? [NSString stringWithFormat:@"%@\n%@", message, directory] : message;
}

- (void)tick {
    if (self.panel.visible) [self refresh];
}

- (void)refresh {
    [self refreshFrameGeneration];
    Status st = read_status();
    const double age = st.ok ? (double)time(NULL) - st.t : 1e9;
    if (!st.ok)
        self.status.stringValue = @"Status: no data, the game has not run with the hook yet";
    else if (age > 3.0)
        self.status.stringValue = [NSString stringWithFormat:@"Status: no frames for %.0f s (game closed or not upscaling)", age];
    else if (st.active)
        self.status.stringValue = [NSString stringWithFormat:@"Status: FSR %s active (%s, %s)",
                                   st.version.c_str(), st.model.c_str(), st.preset.c_str()];
    else
        self.status.stringValue = [NSString stringWithFormat:@"Status: inactive, %@",
                                   st.reason.empty() ? @"no frames yet" : [NSString stringWithUTF8String:st.reason.c_str()]];
    if (st.ok && st.rw)
        self.res.stringValue = [NSString stringWithFormat:@"Upscale: %u×%u → %u×%u  (frames %llu, stubs %llu)",
                                st.rw, st.rh, st.uw, st.uh, st.replaced, st.stubs];
    else
        self.res.stringValue = @"Upscale: —";
    if (st.ok && st.frameMs > 0 && age <= 3.0)
        self.frame.stringValue = [NSString stringWithFormat:@"Game frame: %.1f ms (%.0f fps, average)", st.frameMs, 1000.0 / st.frameMs];
    else
        self.frame.stringValue = @"Game frame: —";
    float s, strength; bool rc, fix; int profile;
    read_settings(s, rc, profile, fix, strength, st.ok && age <= 3.0 ? st.profile : -1);
    self.savedProfile = profile;
    const NSInteger stop = stop_for_profile(profile);
    if (!self.modeSlider.highlighted)
        self.modeSlider.doubleValue = stop >= 0 ? stop : stop_for_profile(5);
    self.modeValue.stringValue = [NSString stringWithFormat:@"%s%@", mode_title(profile),
                                  stop < 0 ? @" - not selectable" : @""];
    self.modeSlider.accessibilityValueDescription = self.modeValue.stringValue;
    self.modeSlider.alphaValue = stop >= 0 ? 1.0 : 0.5;
    // FSR4_VERSION=4.0.2 in the bottle pins the old model; the modes then do nothing.
    const bool pinned = st.ok && st.active && age <= 3.0 && st.version == "4.0.2" && profile != 3;
    self.modeSlider.enabled = !pinned;
    const bool lite = profile == 5 || profile == 6;
    self.pixelFix.hidden = !lite || pinned;
    NSControlStateValue wantFix = profile == 6 || (!lite && fix) ? NSControlStateValueOn : NSControlStateValueOff;
    if (self.pixelFix.state != wantFix) self.pixelFix.state = wantFix;
    self.fixStrength.hidden = self.fixVal.hidden = self.pixelFix.hidden;
    self.fixStrength.enabled = profile == 6;
    self.fixVal.textColor = profile == 6 ? NSColor.whiteColor : NSColor.grayColor;
    self.fixVal.stringValue = [NSString stringWithFormat:@"%.2f", strength];
    if (!self.fixStrength.highlighted && fabs(self.fixStrength.doubleValue - strength) > 1e-3)
        self.fixStrength.doubleValue = strength;
    self.profileHint.stringValue = self.settingsError ?: (pinned
        ? @"The bottle pins FSR 4.0.2 (FSR4_VERSION)."
        : (profile == 7 || profile == 8) && !g_has_lite_lossy
            ? @"Lite Lossy weights/import.json missing or invalid. Run make lite-lossy-weights; saved mode is unchanged."
        : profile == 7 ? @"FSR 4.1.2 Lite Lossy: trained INT8 weights, no fallback to ordinary Lite."
        : profile == 8 ? @"Tiled is no longer offered in the panel. Choose another mode; your saved setting is kept until then."
        : profile == 5 ? @"FSR 4.1.2 INT8 Lite: AMD's smaller model, evaluated every frame."
        : profile == 6 ? @"FSR 4.1.2 Lite with pixelization fix: wider filter where history is rejected."
        : profile == 3 ? @"FSR 4: previous model (4.0.2)."
        : profile == 2 ? @"FSR 4.1.2 Exact: the full network every frame, fast PRE/POST math."
        : profile == 4 ? @"FSR 4.1.2 Lossy: six retrained layers, fewer specks in foliage."
        : profile == 1 ? @"This saved mode was removed. Select FSR 4.1.2 Lossy or FSR 4.1.2 Lite."
        : @"FSR 4.1.1 Exact (bit-exact reference), set by FSR4_PROFILE.");
    self.sharpVal.stringValue = [NSString stringWithFormat:@"%.2f", s];
    if (!self.slider.highlighted && fabs(self.slider.doubleValue - s) > 1e-3) self.slider.doubleValue = s;
    NSControlStateValue want = rc ? NSControlStateValueOn : NSControlStateValueOff;
    if (self.rcas.state != want) self.rcas.state = want;
    [self refreshCapture];
}

- (void)applicationDidFinishLaunching:(NSNotification *)n {
    [self build];
}

- (BOOL)applicationShouldHandleReopen:(NSApplication *)sender hasVisibleWindows:(BOOL)flag {
    [self.panel orderFrontRegardless];   // Dock click re-shows a closed panel
    return NO;
}

- (BOOL)applicationShouldTerminateAfterLastWindowClosed:(NSApplication *)sender { return NO; }

// test aid: content view to PNG (the HUD material is a window property, so
// the view gets a dark layer background for the capture)
- (void)snapshot:(NSString *)path {
    NSView *v = self.panel.contentView;
    v.wantsLayer = YES;
    v.layer.backgroundColor = CGColorCreateGenericGray(0.16, 1.0);
    [v displayIfNeeded];
    NSBitmapImageRep *rep = [v bitmapImageRepForCachingDisplayInRect:v.bounds];
    [v cacheDisplayInRect:v.bounds toBitmapImageRep:rep];
    [[rep representationUsingType:NSBitmapImageFileTypePNG properties:@{}] writeToFile:path atomically:YES];
    printf("panel %.0fx%.0f written to %s\n", v.bounds.size.width, v.bounds.size.height, path.UTF8String);
}

@end

// icon: rounded dark tile with "FSR 4", rendered with CoreGraphics (no window needed)
static void make_icon_png(const char *path, int size) {
    CGColorSpaceRef cs = CGColorSpaceCreateDeviceRGB();
    CGContextRef c = CGBitmapContextCreate(NULL, size, size, 8, 0, cs, kCGImageAlphaPremultipliedLast);
    const CGFloat r = size * 0.22, inset = size * 0.04;
    CGRect rect = CGRectMake(inset, inset, size - 2 * inset, size - 2 * inset);
    CGPathRef rr = CGPathCreateWithRoundedRect(rect, r, r, NULL);
    CGContextAddPath(c, rr);
    CGContextClip(c);
    CGFloat comps[8] = {0.55, 0.12, 0.12, 1.0, 0.20, 0.05, 0.05, 1.0};
    CGGradientRef g = CGGradientCreateWithColorComponents(cs, comps, NULL, 2);
    CGContextDrawLinearGradient(c, g, CGPointMake(0, size), CGPointMake(0, 0), 0);
    CGGradientRelease(g);
    CGPathRelease(rr);
    NSFont *f = [NSFont boldSystemFontOfSize:size * 0.30];
    NSDictionary *attrs = @{NSFontAttributeName: f, NSForegroundColorAttributeName: NSColor.whiteColor};
    NSAttributedString *s1 = [[NSAttributedString alloc] initWithString:@"FSR" attributes:attrs];
    NSAttributedString *s2 = [[NSAttributedString alloc] initWithString:@"4" attributes:attrs];
    NSGraphicsContext *gc = [NSGraphicsContext graphicsContextWithCGContext:c flipped:NO];
    [NSGraphicsContext saveGraphicsState];
    [NSGraphicsContext setCurrentContext:gc];
    NSSize a = s1.size, b = s2.size;
    [s1 drawAtPoint:NSMakePoint((size - a.width) / 2, size * 0.50)];
    [s2 drawAtPoint:NSMakePoint((size - b.width) / 2, size * 0.16)];
    [NSGraphicsContext restoreGraphicsState];
    CGImageRef img = CGBitmapContextCreateImage(c);
    NSBitmapImageRep *rep = [[NSBitmapImageRep alloc] initWithCGImage:img];
    [[rep representationUsingType:NSBitmapImageFileTypePNG properties:@{}] writeToFile:[NSString stringWithUTF8String:path] atomically:YES];
    CGImageRelease(img);
    CGContextRelease(c);
    CGColorSpaceRelease(cs);
}

int main(int argc, char **argv) {
    @autoreleasepool {
        const char *snap = nullptr, *icondir = nullptr;
        for (int i = 1; i < argc; i++) {
            if (!strcmp(argv[i], "--root") && i + 1 < argc) g_root = argv[++i];
            else if (!strcmp(argv[i], "--snapshot") && i + 1 < argc) snap = argv[++i];
            else if (!strcmp(argv[i], "--make-icon") && i + 1 < argc) icondir = argv[++i];
        }
        if (icondir) {
            const int sizes[] = {16, 32, 64, 128, 256, 512, 1024};
            for (int sz : sizes) {
                char p[1024];
                if (sz <= 512) { snprintf(p, sizeof p, "%s/icon_%dx%d.png", icondir, sz, sz); make_icon_png(p, sz); }
                if (sz >= 32) { snprintf(p, sizeof p, "%s/icon_%dx%d@2x.png", icondir, sz / 2, sz / 2); make_icon_png(p, sz); }
            }
            return 0;
        }
        if (g_root.empty()) {
            if (const char *e = getenv("FSR4_ROOT")) g_root = e;
        }
        if (g_root.empty()) {
            // dist/crossover/FSR4 Panel.app/Contents/MacOS/<exe> -> repo root
            char buf[4096]; uint32_t sz = sizeof buf;
            if (_NSGetExecutablePath(buf, &sz) == 0) {
                std::string p = buf;
                for (int i = 0; i < 6; i++) p = p.substr(0, p.rfind('/'));
                g_root = p;
            }
        }
        NSApplication *app = [NSApplication sharedApplication];
        [app setActivationPolicy:NSApplicationActivationPolicyRegular];
        // minimal menu so Cmd+Q works
        NSMenu *bar = [NSMenu new];
        NSMenuItem *appItem = [NSMenuItem new];
        [bar addItem:appItem];
        NSMenu *appMenu = [NSMenu new];
        [appMenu addItemWithTitle:@"Quit FSR 4" action:@selector(terminate:) keyEquivalent:@"q"];
        appItem.submenu = appMenu;
        app.mainMenu = bar;
        PanelApp *pa = [PanelApp new];
        app.delegate = pa;
        if (snap) {
            NSString *path = [NSString stringWithUTF8String:snap];
            dispatch_after(dispatch_time(DISPATCH_TIME_NOW, (int64_t)(1.0 * NSEC_PER_SEC)), dispatch_get_main_queue(), ^{
                [pa snapshot:path];
                exit(0);
            });
        }
        [app run];
    }
    return 0;
}
