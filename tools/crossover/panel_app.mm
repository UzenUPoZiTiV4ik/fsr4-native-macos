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
#include <sys/stat.h>
#include <string>
#include <map>
#include <cmath>
#include <mach-o/dyld.h>

struct Status {
    bool ok = false, active = false;
    std::string reason, model, preset, version;
    int profile = 1;
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

// Profile values match fsr4_live: 0 Exact, 1 Lossy, 2 Exact Fast, 3 FSR 4.0.2.
static int profile_from_name(const std::string &name, int fallback) {
    if (name == "exact" || name == "0") return 0;
    if (name == "lossy" || name == "1") return 1;
    if (name == "exact-fast" || name == "2") return 2;
    if (name == "fsr4" || name == "3") return 3;
    return fallback;
}
static const char *profile_label(int profile) {
    return profile == 1 ? "lossy" : profile == 2 ? "exact-fast" : profile == 3 ? "fsr4" : "exact";
}
// Panel modes: FSR 4 (4.0.2 model), FSR 4.1 (Exact Fast), FSR 4.1 Lite
// (Lossy). Exact stays available through FSR4_PROFILE/panel.conf and then
// shows no selected segment. FSR 4 is offered only when the 4.0.2 weights are
// present (the 4.1-only release omits them).
static bool g_has402 = true;
static bool has_fsr402() {
    struct stat st = {};
    return stat((g_root + "/third_party/fsr4/internal/shaders/fsr4_model_v07_i8_quality/initializers.bin").c_str(), &st) == 0;
}
static int segment_profile(NSInteger segment) {
    static const int all[] = {3, 2, 1}, only411[] = {2, 1};
    return g_has402 ? all[segment] : only411[segment];
}
static NSInteger segment_for_profile(int profile) {
    if (profile == 2) return g_has402 ? 1 : 0;
    if (profile == 1) return g_has402 ? 2 : 1;
    return profile == 3 && g_has402 ? 0 : -1;
}

static Status read_status() {
    Status s;
    auto kv = read_kv(dist_path("status.conf"));
    if (kv.empty()) return s;
    s.ok = true;
    s.active = atoi(kv["active"].c_str()) != 0;
    s.reason = kv["reason"]; s.model = kv["model"]; s.preset = kv["preset"];
    s.version = kv["fsr_version"];
    if (s.version.empty()) s.version = "4.0.2";
    s.profile = profile_from_name(kv["profile"], 1);
    s.rw = atoi(kv["rw"].c_str()); s.rh = atoi(kv["rh"].c_str());
    s.uw = atoi(kv["uw"].c_str()); s.uh = atoi(kv["uh"].c_str());
    s.frameMs = atof(kv["frame_ms"].c_str());
    s.t = atof(kv["t"].c_str());
    s.replaced = strtoull(kv["replaced"].c_str(), nullptr, 10);
    s.stubs = strtoull(kv["stubs"].c_str(), nullptr, 10);
    return s;
}

static void read_settings(float &sharp, bool &rcas, int &profile, int fallbackProfile = -1) {
    sharp = 0.4f; rcas = true; profile = fallbackProfile >= 0 ? fallbackProfile : 1;
    if (const char *value = getenv("FSR4_PROFILE")) profile = profile_from_name(value, profile);
    auto kv = read_kv(dist_path("panel.conf"));
    if (kv.count("sharpness")) sharp = (float)atof(kv["sharpness"].c_str());
    if (kv.count("rcas")) rcas = atoi(kv["rcas"].c_str()) != 0;
    profile = profile_from_name(kv["profile"], profile);
    if (!std::isfinite(sharp)) sharp = .4f;
    if (sharp < 0) sharp = 0; if (sharp > 1) sharp = 1;
}

static bool write_settings(float sharp, bool rcas, int profile) {
    const std::string p = dist_path("panel.conf"), tmp = p + ".tmp." + std::to_string(getpid());
    FILE *f = fopen(tmp.c_str(), "w");
    if (!f) return false;
    bool ok = fprintf(f, "sharpness=%.3f\nrcas=%d\nprofile=%s\n", sharp, rcas ? 1 : 0, profile_label(profile)) > 0;
    if (fclose(f) != 0) ok = false;
    if (!ok || rename(tmp.c_str(), p.c_str()) != 0) { unlink(tmp.c_str()); return false; }
    return true;
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
@property(nonatomic, strong) NSTextField *status, *res, *frame, *sharpVal;
@property(nonatomic, strong) NSSlider *slider;
@property(nonatomic, strong) NSButton *rcas;
@property(nonatomic, strong) NSSegmentedControl *profile;
@property(nonatomic, strong) NSTextField *profileHint;
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
    const CGFloat W = 420, H = 320;
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
    NSView *v = p.contentView;

    CGFloat y = H - 26;
    [v addSubview:label(@"FSR 4 replacing DLSS (CrossOver)", 12, y, W - 24, YES)]; y -= 22;
    self.status = label(@"", 12, y, W - 24, NO); [v addSubview:self.status]; y -= 20;
    self.res = label(@"", 12, y, W - 24, NO); [v addSubview:self.res]; y -= 20;
    self.frame = label(@"", 12, y, W - 24, NO); [v addSubview:self.frame]; y -= 30;

    float sharp; bool rc; int profile;
    read_settings(sharp, rc, profile);
    [v addSubview:label(@"Mode", 12, y, 50, NO)];
    g_has402 = has_fsr402();
    NSArray *modes = g_has402 ? @[@"FSR 4", @"FSR 4.1", @"FSR 4.1 Lite"] : @[@"FSR 4.1", @"FSR 4.1 Lite"];
    self.profile = [NSSegmentedControl segmentedControlWithLabels:modes
        trackingMode:NSSegmentSwitchTrackingSelectOne target:self action:@selector(changed:)];
    self.profile.frame = NSMakeRect(62, y - 4, W - 74, 26);
    self.profile.selectedSegment = segment_for_profile(profile);
    self.profile.toolTip = [NSString stringWithFormat:@"%@FSR 4.1: the 4.1.1 network with fast PRE/POST math. "
                           @"FSR 4.1 Lite: reuses inference on eligible frames, fastest. Switching resets history.",
                           g_has402 ? @"FSR 4: the previous 4.0.2 model. " : @""];
    self.profile.accessibilityLabel = @"FSR mode";
    [v addSubview:self.profile]; y -= 34;
    [v addSubview:label(@"Sharpness", 12, y, 70, NO)];
    self.slider = [NSSlider sliderWithValue:sharp minValue:0.0 maxValue:1.0 target:self action:@selector(changed:)];
    self.slider.frame = NSMakeRect(84, y - 2, W - 84 - 12 - 50, 22);
    self.slider.continuous = YES;
    [v addSubview:self.slider];
    self.sharpVal = label(@"", W - 12 - 46, y, 46, NO);
    self.sharpVal.alignment = NSTextAlignmentRight;
    [v addSubview:self.sharpVal]; y -= 26;
    self.rcas = [NSButton checkboxWithTitle:@"RCAS (FSR sharpening)" target:self action:@selector(changed:)];
    self.rcas.frame = NSMakeRect(12, y, W - 24, 18);
    self.rcas.state = rc ? NSControlStateValueOn : NSControlStateValueOff;
    self.rcas.attributedTitle = [[NSAttributedString alloc] initWithString:self.rcas.title
        attributes:@{NSForegroundColorAttributeName: NSColor.whiteColor, NSFontAttributeName: [NSFont systemFontOfSize:12]}];
    [v addSubview:self.rcas]; y -= 24;
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
    const bool saved = write_settings((float)self.slider.doubleValue, self.rcas.state == NSControlStateValueOn,
                                     segment_profile(self.profile.selectedSegment));
    self.settingsError = saved ? nil : @"Could not save panel settings.";
    [self refresh];
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
    float s; bool rc; int profile;
    read_settings(s, rc, profile, st.ok && age <= 3.0 ? st.profile : -1);
    self.profile.selectedSegment = segment_for_profile(profile);
    // FSR4_VERSION=4.0.2 in the bottle pins the old model; the modes then do nothing.
    const bool pinned = st.ok && st.active && age <= 3.0 && st.version != "4.1.1" && profile != 3;
    self.profile.enabled = !pinned;
    self.profileHint.stringValue = self.settingsError ?: (pinned
        ? @"The bottle pins FSR 4.0.2 (FSR4_VERSION); modes need 4.1.1."
        : profile == 3 ? @"FSR 4: previous model (4.0.2)."
        : profile == 2 ? @"FSR 4.1: newer model, about as fast as FSR 4."
        : profile == 1 ? @"FSR 4.1 Lite: fastest, skips the network on some frames."
        : @"FSR 4.1 Exact (bit-exact reference), set by FSR4_PROFILE.");
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
