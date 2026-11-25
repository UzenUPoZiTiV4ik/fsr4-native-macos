// SPDX-License-Identifier: MIT
// Replacement for the exact MetalFX FG encode created by D3DMetal.
// Supports ANR FG-V.B20 (dynamic resolution 640x360..7680x4320, native srgb8/pq10/scrgb16,
// 2x and 3x dual-phase interpolation) as well as legacy FG-V.B14 (fixed 1080p/1440p/2160p 2x).
#include "anr_live.h"
#include "anr_runtime.h"
#include "fsr4_fg_settings.h"
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <exception>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <unistd.h>
#include <vector>

namespace {

constexpr uint32_t kNativeEncodingBit = 8u;

struct Convert {
    uint32_t width, height, rw, rh, encoding, reversed;
    float whiteNits;
    uint32_t uiMode;
    float uiNoiseFloor;
};
static_assert(sizeof(Convert) == 36, "ANR helper constants ABI");

struct FrameSlot {
    id<MTLBuffer> constants = nil, completion = nil;
    id<MTL4ArgumentTable> importArgs = nil, exportArgs = nil, exportExtraArgs = nil, retireArgs = nil;
    id<MTLResidencySet> residency = nil;
    std::array<uintptr_t, 12> residencyKey = {};
    id<MTLTexture> extraOutput = nil;
    NSArray *resources = nil;
    id<MTLFence> fence = nil;
    bool busy = false;
    bool extraInUse = false;
    uint64_t extraTicket = 0;
    uint64_t uiSequence = 0;

    bool finished() const {
        if (!busy) return !extraInUse;
        return __atomic_load_n((uint32_t *)completion.contents, __ATOMIC_ACQUIRE) != 0 && !extraInUse;
    }
};

struct View {
    id<MTLDevice> device = nil;
    std::unique_ptr<anr::Runtime> runtime;
    id<MTLTexture> color = nil, depth = nil, motion = nil, generated = nil;
    id<MTLComputePipelineState> importPSO = nil, importDmPSO = nil, exportPSO = nil, retirePSO = nil;
    FrameSlot slots[4];
    unsigned activeSlot = 0, nextSlot = 0;
    uint32_t width = 0, height = 0, rw = 0, rh = 0;
    int encoding = 0;
    MTLPixelFormat outputFormat = MTLPixelFormatInvalid;
    bool history = false, swapPending = false, failed = false;
    std::chrono::steady_clock::time_point lastFrame, nextUIStats;
    double emaDeltaSec = 1.0 / 60.0;
    uint64_t uiSequence = 0, uiStatsRead = 0;
    std::string directory;

    bool finished() const {
        for (const auto &slot : slots) if (!slot.finished()) return false;
        return true;
    }

    bool acquire() {
        for (unsigned i = 0; i < 4; ++i) {
            unsigned index = (nextSlot + i) % 4;
            if (slots[index].finished()) {
                activeSlot = index;
                nextSlot = (index + 1) % 4;
                return true;
            }
        }
        return false;
    }

    bool gpu_failed() const {
        for (const auto &slot : slots) {
            if (slot.busy && __atomic_load_n((uint32_t *)slot.completion.contents, __ATOMIC_ACQUIRE) == 2) {
                return true;
            }
        }
        return false;
    }
};

enum class PresentPhase { WaitingInterp = 0, WaitingReal = 1 };

struct PendingExtra {
    uintptr_t device = 0;
    uint64_t ticket = 0;
    id<MTLTexture> texture = nil;
    std::weak_ptr<View> view;
    unsigned slotIndex = 0;
    double stepSeconds = 1.0 / 180.0;
    PresentPhase phase = PresentPhase::WaitingInterp;
    std::chrono::steady_clock::time_point created;
};

using Key = std::pair<uintptr_t, uintptr_t>;
struct Failure {
    std::string directory, reason;
    std::chrono::steady_clock::time_point retry;
};

struct LiveState {
    std::mutex mutex;
    std::map<Key, std::shared_ptr<View>> views;
    std::vector<std::shared_ptr<View>> retired;
    std::map<Key, Failure> failures;
    std::deque<PendingExtra> pendingExtras;
    std::map<uint64_t, std::pair<std::weak_ptr<View>, unsigned>> inFlightExtraTickets;
    uint64_t nextExtraTicket = 1;
    int selectedMode = 0;
    Fsr4AnrStatus status = {};
    std::chrono::steady_clock::time_point lastPublish;
};

LiveState &live_state() {
    // The FG dylib constructor may run before this translation unit's globals.
    // Construct containers on first access, independent of link order.
    static LiveState *const state = new LiveState;
    return *state;
}

Key key(void *device, void *owner) {
    return {(uintptr_t)device, (uintptr_t)owner};
}

const char *mode_string(int selectedMode) {
    if (selectedMode == 2) return "anr3x";
    if (selectedMode == 1) return "anr";
    return "metalfx";
}

const char *encoding_string(anr::Encoding enc) {
    switch (enc) {
        case anr::Encoding::SRGB8: return "srgb8";
        case anr::Encoding::SCRGB16: return "scrgb16";
        case anr::Encoding::PQ10:
        default: return "pq10";
    }
}

void release_slot_ticket_locked(LiveState &state, uint64_t ticket) {
    auto it = state.inFlightExtraTickets.find(ticket);
    if (it != state.inFlightExtraTickets.end()) {
        if (auto v = it->second.first.lock()) {
            auto &slot = v->slots[it->second.second];
            if (slot.extraTicket == ticket) {
                slot.extraInUse = false;
                slot.extraTicket = 0;
            }
        }
        state.inFlightExtraTickets.erase(it);
    }
}

void prune_stale_extras_locked(LiveState &state, std::chrono::steady_clock::time_point now) {
    while (!state.pendingExtras.empty()) {
        const auto &front = state.pendingExtras.front();
        if (now - front.created <= std::chrono::milliseconds(250)) break;
        const uint64_t ticket = front.ticket;
        state.pendingExtras.pop_front();
        release_slot_ticket_locked(state, ticket);
    }
}

void clear_pending_extras_locked(LiveState &state) {
    for (const auto &entry : state.pendingExtras) {
        release_slot_ticket_locked(state, entry.ticket);
    }
    state.pendingExtras.clear();
}

void publish() {
    auto &state = live_state();
    auto &status = state.status;
    auto &lastPublish = state.lastPublish;
    // Synthetic GPU checks must not overwrite the running game's receipt.
    if (const char *disabled = getenv("FSR4_ANR_STATUS_DISABLED")) {
        if (!strcmp(disabled, "1")) return;
    }
    const char *root = getenv("FSR4_ROOT");
    if (!root || !*root) return;
    const auto now = std::chrono::steady_clock::now();
    if (lastPublish.time_since_epoch().count() && now - lastPublish < std::chrono::milliseconds(500)) return;
    lastPublish = now;
    const std::string path = std::string(root) + "/dist/crossover/fg-status.conf";
    const std::string temporary = path + ".tmp";
    FILE *f = fopen(temporary.c_str(), "w");
    if (!f) return;
    const char *uiMode = status.uiMode == FSR4_ANR_UI_PRECOMPOSITED ? "precomposited-adaptive" :
        status.uiMode == FSR4_ANR_UI_OVERLAY ? "premultiplied-overlay" : "none";
    const int selMode = state.selectedMode;
    fprintf(f, "t=%ld\npid=%d\nactive=%d\nreason=%s\nwidth=%u\nheight=%u\nencodes=%llu\ngenerated=%llu\nprimed=%llu\nfallbacks=%llu\nui_mode=%s\nui_composited=%llu\nui_overlay=%llu\ncolor_format=%u\nui_format=%u\n",
            (long)time(nullptr), (int)getpid(), status.active, status.reason, status.width, status.height,
            (unsigned long long)status.encodes, (unsigned long long)status.generated,
            (unsigned long long)status.primed, (unsigned long long)status.fallbacks,
            uiMode, (unsigned long long)status.uiComposited, (unsigned long long)status.uiOverlay,
            status.colorFormat, status.uiFormat);
    fprintf(f, "ui_policy=adaptive-residual-v2\nui_samples=%u\nui_raw_changed=%u\nui_soft=%u\nui_opaque=%u\nui_opacity_mean=%.6f\nmotion_scale_x=%.6f\nmotion_scale_y=%.6f\nanr_motion_scale_x=%.6f\nanr_motion_scale_y=%.6f\n",
            status.uiSamples, status.uiChanged, status.uiSoft, status.uiOpaque,
            status.uiSamples ? double(status.uiOpacitySum) / (4096.0 * status.uiSamples) : 0.0,
            status.motionScaleX, status.motionScaleY, status.anrMotionScaleX, status.anrMotionScaleY);
    fprintf(f, "selected_mode=%s\nnative_passthrough=%d\nanr_configured=%d\nmultiplier=%u\ngenerated_extra_3x=%llu\nrelease=%s\nencoding=%s\nmpp_active=%d\n",
            mode_string(selMode), selMode == 0 ? 1 : 0,
            getenv("FSR4_ANR_RUNTIME") && *getenv("FSR4_ANR_RUNTIME") ? 1 : 0,
            status.multiplier ? status.multiplier : (uint32_t)(selMode == 2 ? 3 : (selMode == 1 ? 2 : 1)),
            (unsigned long long)status.generatedExtra3x,
            status.release[0] ? status.release : "",
            status.encodingName[0] ? status.encodingName : "",
            status.mppActive);
    fclose(f);
    rename(temporary.c_str(), path.c_str());
}

int reject(const char *why) {
    auto &status = live_state().status;
    const bool changed = strncmp(status.reason, why, sizeof status.reason) != 0 || status.active;
    if (changed) {
        fprintf(stderr, "[anr-fg] %s\n", why);
        const char *path = getenv("FSR4_HOOK_LOG");
        if (path && *path) {
            if (FILE *file = fopen(path, "a")) {
                fprintf(file, "[anr-fg] %s\n", why);
                fclose(file);
            }
        }
    }
    status.active = 0;
    snprintf(status.reason, sizeof status.reason, "%s", why);
    ++status.fallbacks;
    publish();
    return 0;
}

bool is_v2_manifest_file(NSString *manifestPath) {
    NSData *data = [NSData dataWithContentsOfFile:manifestPath];
    if (!data || data.length > 32u * 1024u * 1024u) return false;
    NSDictionary *obj = [NSJSONSerialization JSONObjectWithData:data options:0 error:nil];
    return [obj isKindOfClass:NSDictionary.class] && [obj[@"schema"] isEqual:@"fsr4_anr_runtime_v2"];
}

std::string runtime_path(uint32_t width, uint32_t height) {
    const char *base = getenv("FSR4_ANR_RUNTIME");
    if (!base || !*base) return "";
    NSString *root = [NSString stringWithUTF8String:base];
    NSString *rootManifest = [root stringByAppendingPathComponent:@"manifest.json"];
    // V.B20 dynamic-resolution runtime lives directly at root and serves 640x360..7680x4320.
    if ([NSFileManager.defaultManager fileExistsAtPath:rootManifest] && is_v2_manifest_file(rootManifest)) {
        return base;
    }
    NSString *profile = [root stringByAppendingPathComponent:[NSString stringWithFormat:@"%ux%u", width, height]];
    if ([NSFileManager.defaultManager fileExistsAtPath:[profile stringByAppendingPathComponent:@"manifest.json"]]) {
        return profile.UTF8String;
    }
    if ([NSFileManager.defaultManager fileExistsAtPath:rootManifest]) {
        return base;
    }
    return "";
}

bool ui_format_matches(MTLPixelFormat a, MTLPixelFormat b) {
    return a == b
        || ((a == MTLPixelFormatBGRA8Unorm || a == MTLPixelFormatRGBA8Unorm)
            && (b == MTLPixelFormatBGRA8Unorm || b == MTLPixelFormatRGBA8Unorm))
        || ((a == MTLPixelFormatBGRA8Unorm_sRGB || a == MTLPixelFormatRGBA8Unorm_sRGB)
            && (b == MTLPixelFormatBGRA8Unorm_sRGB || b == MTLPixelFormatRGBA8Unorm_sRGB));
}

anr::Encoding pick_v2_encoding(int colorEncoding) {
    if (colorEncoding == 1) return anr::Encoding::SRGB8;
    if (colorEncoding == 0) return anr::Encoding::SCRGB16;
    return anr::Encoding::PQ10;
}

MTLPixelFormat internal_color_format(bool isV2, anr::Encoding enc) {
    if (!isV2) return MTLPixelFormatRGB10A2Unorm;
    switch (enc) {
        case anr::Encoding::SRGB8: return MTLPixelFormatRGBA8Unorm;
        case anr::Encoding::SCRGB16: return MTLPixelFormatRGBA16Float;
        case anr::Encoding::PQ10:
        default: return MTLPixelFormatRGB10A2Unorm;
    }
}

bool format_matches_v2_native(MTLPixelFormat fmt, anr::Encoding enc) {
    switch (enc) {
        case anr::Encoding::SRGB8:
            return fmt == MTLPixelFormatRGBA8Unorm || fmt == MTLPixelFormatBGRA8Unorm;
        case anr::Encoding::SCRGB16:
            return fmt == MTLPixelFormatRGBA16Float;
        case anr::Encoding::PQ10:
        default:
            return fmt == MTLPixelFormatRGB10A2Unorm;
    }
}

id<MTLTexture> tex(id<MTLDevice> device, MTLPixelFormat format, uint32_t width, uint32_t height,
                   MTLTextureUsage extraUsage = 0) {
    auto descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format
                                                                         width:width
                                                                        height:height
                                                                     mipmapped:NO];
    descriptor.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite | extraUsage;
    descriptor.storageMode = MTLStorageModePrivate;
    return [device newTextureWithDescriptor:descriptor];
}

std::shared_ptr<View> create(const Fsr4AnrFrame &f, MTLPixelFormat outputFmt,
                             const std::string &directory, std::string &error) {
    auto v = std::make_shared<View>();
    v->device = (__bridge id<MTLDevice>)f.device;
    v->directory = directory;
    v->width = f.uw;
    v->height = f.uh;
    v->rw = f.rw;
    v->rh = f.rh;
    v->encoding = f.colorEncoding;
    v->outputFormat = outputFmt;
    v->runtime = std::make_unique<anr::Runtime>(v->device, directory);

    const anr::Encoding v2Enc = pick_v2_encoding(f.colorEncoding);
    // Load ThreeX phases (0.5, 1/3, 2/3) on V.B20 so live 2x <-> 3x switches require zero reload.
    // On legacy V.B14 (v1), TwoX is used to keep load cost minimal.
    NSString *manifestPath = [[NSString stringWithUTF8String:directory.c_str()] stringByAppendingPathComponent:@"manifest.json"];
    const bool expectV2 = is_v2_manifest_file(manifestPath);
    const anr::PhaseMode pmode = expectV2 ? anr::PhaseMode::ThreeX : anr::PhaseMode::TwoX;
    if (!v->runtime->initialize(error, f.uw, f.uh, v2Enc, pmode)) return {};
    if (v->runtime->width() != f.uw || v->runtime->height() != f.uh) {
        error = "ANR runtime resolution does not match game FG";
        return {};
    }

    const char *coalesced = getenv("FSR4_ANR_COALESCED");
    v->runtime->set_coalesced(!coalesced || *coalesced != '0');

    const char *root = getenv("FSR4_ROOT");
    if (!root || !*root) {
        error = "FSR4_ROOT required for ANR helper shaders";
        return {};
    }
    NSString *libraryPath = [[NSString stringWithUTF8String:root] stringByAppendingPathComponent:@"build/anr_helpers_ui_v2.metallib"];
    NSError *metalError = nil;
    id<MTLLibrary> library = [v->device newLibraryWithURL:[NSURL fileURLWithPath:libraryPath] error:&metalError];
    auto pipeline = [&](NSString *name) -> id<MTLComputePipelineState> {
        id<MTLFunction> fn = [library newFunctionWithName:name];
        return fn ? [v->device newComputePipelineStateWithFunction:fn error:&metalError] : nil;
    };
    if (!library) {
        error = "ANR helpers: " + std::string(metalError.localizedDescription.UTF8String);
        return {};
    }
    v->importPSO = pipeline(@"anr_import");
    v->importDmPSO = pipeline(@"anr_import_dm");
    v->exportPSO = pipeline(@"anr_export");
    v->retirePSO = pipeline(@"anr_retire");

    const MTLPixelFormat colorFmt = internal_color_format(v->runtime->is_v2(), v->runtime->encoding());
    v->color = tex(v->device, colorFmt, f.uw, f.uh);
    v->depth = tex(v->device, MTLPixelFormatR32Float, f.uw, f.uh);
    v->motion = tex(v->device, MTLPixelFormatRG16Float, f.uw, f.uh);
    v->generated = tex(v->device, colorFmt, f.uw, f.uh);
    if (!v->importPSO || !v->exportPSO || !v->retirePSO || !v->color || !v->depth || !v->motion || !v->generated) {
        error = "ANR view allocation failed";
        return {};
    }

    for (auto &slot : v->slots) {
        slot.constants = [v->device newBufferWithLength:256 options:MTLResourceStorageModeShared];
        slot.completion = [v->device newBufferWithLength:256 options:MTLResourceStorageModeShared];
        if (!slot.constants || !slot.completion) {
            error = "ANR frame slot allocation failed";
            return {};
        }
        *static_cast<uint32_t *>(slot.completion.contents) = 1;
        MTLResidencySetDescriptor *rd = [MTLResidencySetDescriptor new];
        rd.initialCapacity = 16;
        slot.residency = [v->device newResidencySetWithDescriptor:rd error:&metalError];
        MTL4ArgumentTableDescriptor *td = [MTL4ArgumentTableDescriptor new];
        td.maxBufferBindCount = 2;
        td.maxTextureBindCount = 6;
        slot.importArgs = [v->device newArgumentTableWithDescriptor:td error:&metalError];
        slot.exportArgs = [v->device newArgumentTableWithDescriptor:td error:&metalError];
        slot.exportExtraArgs = [v->device newArgumentTableWithDescriptor:td error:&metalError];
        slot.retireArgs = [v->device newArgumentTableWithDescriptor:td error:&metalError];
        if (!slot.residency || !slot.importArgs || !slot.exportArgs || !slot.exportExtraArgs || !slot.retireArgs) {
            error = "ANR argument/residency allocation failed";
            return {};
        }
    }
    return v;
}

void collect() {
    auto &retired = live_state().retired;
    for (auto it = retired.begin(); it != retired.end();) {
        if ((*it)->finished()) it = retired.erase(it);
        else ++it;
    }
}

void read_ui_stats(View &v) {
    auto &status = live_state().status;
    for (const auto &slot : v.slots) {
        if (slot.uiSequence > v.uiStatsRead && slot.finished()) {
            uint32_t values[5];
            memcpy(values, (const uint8_t *)slot.completion.contents + 16, sizeof values);
            status.uiSamples = values[0];
            status.uiChanged = values[1];
            status.uiSoft = values[2];
            status.uiOpaque = values[3];
            status.uiOpacitySum = values[4];
            v.uiStatsRead = slot.uiSequence;
        }
    }
}

float ui_noise_floor(MTLPixelFormat format) {
    if (format == MTLPixelFormatRGBA8Unorm || format == MTLPixelFormatBGRA8Unorm
        || format == MTLPixelFormatRGBA8Unorm_sRGB || format == MTLPixelFormatBGRA8Unorm_sRGB) {
        return 2.5f / 255.f;
    }
    if (format == MTLPixelFormatRGB10A2Unorm) return 2.5f / 1023.f;
    return 1.e-4f;
}

// which: 0 = anr_import, 1 = anr_export (primary), 2 = anr_retire, 3 = anr_import_dm, 4 = anr_export (extra 3x)
bool helper(View &v, id command, bool metal4, int which,
            id<MTLTexture> color, id<MTLTexture> depth, id<MTLTexture> motion,
            id<MTLTexture> output, id<MTLFence> fence, id<MTLTexture> ui = nil) {
    auto &slot = v.slots[v.activeSlot];
    const bool isImport = (which == 0 || which == 3);
    const bool isExport = (which == 1 || which == 4);
    const bool isRetire = (which == 2);
    id<MTLComputePipelineState> pso = which == 0 ? v.importPSO
                                    : which == 3 ? (v.importDmPSO ?: v.importPSO)
                                    : isExport ? v.exportPSO
                                    : v.retirePSO;
    const MTLSize threads = isRetire ? MTLSizeMake(1, 1, 1) : MTLSizeMake(v.width, v.height, 1);
    const MTLSize group = isRetire ? MTLSizeMake(1, 1, 1) : MTLSizeMake(16, 16, 1);
    if (metal4) {
        auto enc = [(id<MTL4CommandBuffer>)command computeCommandEncoder];
        if (!enc) return false;
        [enc barrierAfterQueueStages:MTLStageAll beforeStages:MTLStageDispatch visibilityOptions:MTL4VisibilityOptionDevice];
        if (isImport && fence) [enc waitForFence:fence beforeEncoderStages:MTLStageDispatch];
        id<MTL4ArgumentTable> args = isImport ? slot.importArgs
                                   : which == 1 ? slot.exportArgs
                                   : which == 4 ? slot.exportExtraArgs
                                   : slot.retireArgs;
        [args setAddress:(isRetire ? slot.completion : slot.constants).gpuAddress atIndex:0];
        if (isExport) [args setAddress:slot.completion.gpuAddress + 16 atIndex:1];
        if (isImport) {
            id<MTLTexture> textures[] = {color, depth, motion, v.color, v.depth, v.motion};
            for (unsigned i = 0; i < 6; ++i) [args setTexture:textures[i].gpuResourceID atIndex:i];
        }
        if (isExport) {
            id<MTLTexture> textures[] = {v.generated, output, color, ui ?: color};
            for (unsigned i = 0; i < 4; ++i) [args setTexture:textures[i].gpuResourceID atIndex:i];
        }
        [enc setComputePipelineState:pso];
        [enc setArgumentTable:args];
        [enc dispatchThreads:threads threadsPerThreadgroup:group];
        if (isExport && fence) [enc updateFence:fence afterEncoderStages:MTLStageDispatch];
        [enc endEncoding];
    } else {
        auto enc = [(id<MTLCommandBuffer>)command computeCommandEncoder];
        if (!enc) return false;
        if (isImport && fence) [enc waitForFence:fence];
        [enc setComputePipelineState:pso];
        [enc setBuffer:isRetire ? slot.completion : slot.constants offset:0 atIndex:0];
        if (isExport) [enc setBuffer:slot.completion offset:16 atIndex:1];
        if (isImport) {
            id<MTLTexture> textures[] = {color, depth, motion, v.color, v.depth, v.motion};
            for (unsigned i = 0; i < 6; ++i) [enc setTexture:textures[i] atIndex:i];
        }
        if (isExport) {
            id<MTLTexture> textures[] = {v.generated, output, color, ui ?: color};
            for (unsigned i = 0; i < 4; ++i) [enc setTexture:textures[i] atIndex:i];
        }
        [enc dispatchThreads:threads threadsPerThreadgroup:group];
        if (isExport && fence) [enc updateFence:fence];
        [enc endEncoding];
    }
    return true;
}

void update_slot_residency(FrameSlot &slot, id<MTLTexture> c, id<MTLTexture> d, id<MTLTexture> m,
                           id<MTLTexture> o, id<MTLTexture> ui, View &v) {
    id objs[12] = {
        c, d, m, o, v.color, v.depth, v.motion, v.generated,
        slot.constants, slot.completion, ui, slot.extraOutput
    };
    std::array<uintptr_t, 12> next = {};
    for (size_t i = 0; i < 12; ++i) next[i] = (uintptr_t)(__bridge void *)objs[i];
    if (slot.residencyKey == next) {
        return;
    }
    NSMutableArray *arr = [NSMutableArray arrayWithCapacity:12];
    [slot.residency removeAllAllocations];
    for (id obj : objs) {
        if (obj) {
            [arr addObject:obj];
            [slot.residency addAllocation:obj];
        }
    }
    slot.resources = arr;
    [slot.residency commit];
    slot.residencyKey = next;
}

} // namespace

int fsr4_anr_encode_frame(const Fsr4AnrFrame *frame) {
    @autoreleasepool {
        auto &state = live_state();
        std::lock_guard<std::mutex> lock(state.mutex);
        auto &status = state.status;
        auto &views = state.views;
        auto &retired = state.retired;
        auto &failures = state.failures;
        ++status.encodes;
        collect();

        const int selMode = fsr4_anr_selected_mode();
        state.selectedMode = selMode;
        status.selectedAnr = selMode != 0 ? 1 : 0;
        status.multiplier = (uint32_t)fsr4_anr_multiplier();
        if (!status.selectedAnr) return reject("MetalFX FG selected");

        const char *reason = fsr4_anr_frame_reason(frame);
        if (reason) return reject(reason);
        const auto &f = *frame;
        status.width = f.uw;
        status.height = f.uh;
        status.uiMode = fsr4_anr_ui_mode(&f);
        status.motionScaleX = f.motionScaleX;
        status.motionScaleY = f.motionScaleY;
        status.anrMotionScaleX = f.motionScaleX * float(f.uw) / f.rw;
        status.anrMotionScaleY = f.motionScaleY * float(f.uh) / f.rh;

        id command = (__bridge id)f.commandBuffer;
        id<MTLDevice> device = (__bridge id<MTLDevice>)f.device;
        const bool metal4 = [command respondsToSelector:@selector(useResidencySet:)]
            && ![command respondsToSelector:@selector(addCompletedHandler:)];
        if (![command respondsToSelector:@selector(device)] || [command device] != device) {
            return reject("ANR: command buffer device mismatch");
        }
        if (!metal4 && ![command respondsToSelector:@selector(addCompletedHandler:)]) {
            return reject("ANR: unsupported FG command buffer API");
        }

        id<MTLTexture> color = (__bridge id<MTLTexture>)f.color;
        id<MTLTexture> depth = (__bridge id<MTLTexture>)f.depth;
        id<MTLTexture> motion = (__bridge id<MTLTexture>)f.motion;
        id<MTLTexture> output = (__bridge id<MTLTexture>)f.output;
        id<MTLTexture> ui = (__bridge id<MTLTexture>)f.ui;
        status.colorFormat = (uint32_t)color.pixelFormat;
        status.uiFormat = (uint32_t)ui.pixelFormat;

        for (auto texture : {color, depth, motion, output}) {
            if (texture.device != device || texture.textureType != MTLTextureType2D) {
                return reject("ANR: incompatible FG texture/device");
            }
        }
        if (ui) {
            if (ui.device != device || ui.textureType != MTLTextureType2D || ui.width != f.uw || ui.height != f.uh) {
                return reject("ANR: UI texture extent/device mismatch; keeping MetalFX");
            }
            if (!(ui.usage & MTLTextureUsageShaderRead)) {
                return reject("ANR: UI texture must be shader-readable; keeping MetalFX");
            }
            if (!ui_format_matches(ui.pixelFormat, color.pixelFormat)) {
                return reject("ANR: UI and scene color formats differ; keeping MetalFX");
            }
            if (ui == output) return reject("ANR: UI aliases FG output; keeping MetalFX");
            if (color == output) return reject("ANR: scene aliases FG output during UI recomposition; keeping MetalFX");
            if (!f.uiComposited && (ui.pixelFormat == MTLPixelFormatRGB9E5Float || ui.pixelFormat == MTLPixelFormatRG11B10Float)) {
                return reject("ANR: separate UI overlay needs an alpha channel; keeping MetalFX");
            }
        }
        if (color.width != f.uw || color.height != f.uh || output.width != f.uw || output.height != f.uh
            || depth.width < f.rw || depth.height < f.rh || motion.width < f.rw || motion.height < f.rh) {
            return reject("ANR: unsupported FG texture extent");
        }
        if (depth.pixelFormat != MTLPixelFormatR32Float && depth.pixelFormat != MTLPixelFormatDepth32Float
            && depth.pixelFormat != MTLPixelFormatDepth32Float_Stencil8) {
            return reject("ANR: depth format unsupported");
        }
        if (motion.pixelFormat != MTLPixelFormatRG16Float && motion.pixelFormat != MTLPixelFormatRG32Float) {
            return reject("ANR: motion format unsupported");
        }
        if (!(output.usage & MTLTextureUsageShaderWrite) || !(color.usage & MTLTextureUsageShaderRead)
            || !(depth.usage & MTLTextureUsageShaderRead) || !(motion.usage & MTLTextureUsageShaderRead)) {
            return reject("ANR: FG texture shader usage flags unavailable");
        }

        const std::string directory = runtime_path(f.uw, f.uh);
        if (directory.empty()) {
            if (!getenv("FSR4_ANR_RUNTIME") || !*getenv("FSR4_ANR_RUNTIME")) {
                return reject("ANR: FSR4_ANR_RUNTIME not configured");
            }
            char why[160];
            snprintf(why, sizeof why, "ANR: no prepared graph for %ux%u; prepare V.B20 runtime or 1080p/1440p/2160p", f.uw, f.uh);
            return reject(why);
        }
        if (retired.size() >= 8) return reject("ANR: retired GPU resources still pending");

        const Key ownerKey = key(f.device, f.owner);
        auto cached = failures.find(ownerKey);
        if (cached != failures.end() && cached->second.directory == directory
            && std::chrono::steady_clock::now() < cached->second.retry) {
            return reject(cached->second.reason.c_str());
        }
        failures.erase(ownerKey);

        auto &view = views[ownerKey];
        if (view && view->gpu_failed()) return reject("ANR: GPU failure; restart game");
        if (view && (view->directory != directory || view->width != f.uw || view->height != f.uh
                     || view->encoding != f.colorEncoding || view->rw != f.rw || view->rh != f.rh
                     || view->outputFormat != output.pixelFormat || view->failed)) {
            if (!view->finished()) retired.push_back(view);
            view.reset();
        }

        std::string error;
        if (!view) {
            try {
                @try { view = create(f, output.pixelFormat, directory, error); }
                @catch (NSException *exception) {
                    error = exception.reason ? exception.reason.UTF8String : "ANR runtime initialization exception";
                }
            } catch (const std::exception &exception) {
                error = exception.what();
            }
        }
        if (!view) {
            failures[ownerKey] = {directory, error, std::chrono::steady_clock::now() + std::chrono::seconds(2)};
            return reject(error.c_str());
        }

        View &v = *view;
        read_ui_stats(v);
        const auto now = std::chrono::steady_clock::now();
        prune_stale_extras_locked(state, now);

        if (!v.acquire()) {
            v.history = false;
            return reject("ANR: all four frame slots busy; keeping MetalFX");
        }
        auto &slot = v.slots[v.activeSlot];
        v.runtime->set_frame_slot(v.activeSlot);

        const bool isV2 = v.runtime->is_v2();
        const bool want3x = (selMode == 2) && isV2;
        status.multiplier = want3x ? 3u : 2u;
        status.mppActive = v.runtime->mpp_active() ? 1 : 0;
        snprintf(status.release, sizeof status.release, "%s", v.runtime->release_name().c_str());
        snprintf(status.encodingName, sizeof status.encodingName, "%s", encoding_string(v.runtime->encoding()));

        // Update smoothed real-frame interval for 3x presentation pacing.
        if (v.lastFrame.time_since_epoch().count()) {
            const double wallDt = std::chrono::duration<double>(now - v.lastFrame).count();
            if (wallDt >= 1.0 / 360.0 && wallDt <= 0.25) {
                v.emaDeltaSec = v.emaDeltaSec * 0.75 + wallDt * 0.25;
            }
        } else if (std::isfinite(f.deltaTime) && f.deltaTime >= 1.0f / 360.0f && f.deltaTime <= 0.25f) {
            v.emaDeltaSec = (double)f.deltaTime;
        }
        const bool temporalGap = v.lastFrame.time_since_epoch().count()
            && now - v.lastFrame > std::chrono::milliseconds(500);
        const bool priming = !v.history || f.reset || temporalGap;
        v.lastFrame = now;
        if (v.swapPending) {
            v.runtime->swap_history();
            v.swapPending = false;
        }

        // Lazy-allocate per-slot extraOutput texture when 3x mode is active.
        if (want3x && !slot.extraOutput) {
            slot.extraOutput = tex(v.device, output.pixelFormat, f.uw, f.uh, MTLTextureUsageRenderTarget);
            if (!slot.extraOutput) {
                return reject("ANR: 3x extra output texture allocation failed");
            }
        }

        const bool uiStats = !priming && status.uiMode == FSR4_ANR_UI_PRECOMPOSITED && now >= v.nextUIStats;
        if (uiStats) v.nextUIStats = now + std::chrono::milliseconds(500);
        slot.uiSequence = uiStats ? ++v.uiSequence : 0;
        memset((uint8_t *)slot.completion.contents + 16, 0, 5 * sizeof(uint32_t));

        const uint32_t helperEncoding = (uint32_t)f.colorEncoding | (isV2 ? kNativeEncodingBit : 0u);
        Convert constants = {
            f.uw, f.uh, f.rw, f.rh, helperEncoding, (uint32_t)f.depthReversed, 203.f,
            (uint32_t)status.uiMode | (uiStats ? 256u : 0u), ui_noise_floor(color.pixelFormat)
        };
        const char *white = getenv("FSR4_ANR_WHITE_NITS");
        if (white) {
            float value = strtof(white, nullptr);
            if (std::isfinite(value) && value > 0) constants.whiteNits = value;
        }
        memcpy(slot.constants.contents, &constants, sizeof(constants));

        update_slot_residency(slot, color, depth, motion, output, ui, v);
        if (metal4) [(id<MTL4CommandBuffer>)command useResidencySet:slot.residency];

        id<MTLFence> fence = (__bridge id<MTLFence>)f.fence;
        slot.fence = fence;
        if (fence && fence.device != device) return reject("ANR: fence belongs to another device");

        __atomic_store_n((uint32_t *)slot.completion.contents, 0u, __ATOMIC_RELEASE);
        slot.busy = true;
        uint8_t hardFrame[204];
        fsr4_anr_frame_constants(&f, hardFrame, !priming);

        // Zero-conversion fast paths:
        // - V.B20 natively processes srgb8 (RGBA8/BGRA8), scrgb16 (RGBA16F), and pq10 (RGB10A2).
        // - When depth/motion also match full resolution + format, skip anr_import completely.
        // - When only depth/motion differ, pass color directly and run fast anr_import_dm.
        // - When no UI and color format matches native encoding, write output directly (skip anr_export).
        const bool colorNative = isV2
            ? format_matches_v2_native(color.pixelFormat, v.runtime->encoding())
            : (f.colorEncoding == 2 && color.pixelFormat == MTLPixelFormatRGB10A2Unorm);
        const bool dmNative = depth.pixelFormat == MTLPixelFormatR32Float
            && motion.pixelFormat == MTLPixelFormatRG16Float
            && f.rw == f.uw && f.rh == f.uh;
        const bool directImport = colorNative && dmNative;
        const bool dmOnlyImport = !directImport && isV2 && colorNative && v.importDmPSO != nil;
        const bool directExport = !ui && colorNative;

        bool ok = !priming || v.runtime->reset_history(command, metal4, error);
        if (ok && !directImport) {
            ok = helper(v, command, metal4, dmOnlyImport ? 3 : 0, color, depth, motion, output, fence);
        } else if (ok && fence) {
            if (metal4) {
                auto enc = [(id<MTL4CommandBuffer>)command computeCommandEncoder];
                ok = enc != nil;
                if (enc) {
                    [enc waitForFence:fence beforeEncoderStages:MTLStageDispatch];
                    [enc endEncoding];
                }
            } else {
                auto enc = [(id<MTLCommandBuffer>)command computeCommandEncoder];
                ok = enc != nil;
                if (enc) {
                    [enc waitForFence:fence];
                    [enc endEncoding];
                }
            }
        }

        id<MTLTexture> pairColor = (directImport || dmOnlyImport) ? color : v.color;
        id<MTLTexture> pairDepth = directImport ? depth : v.depth;
        id<MTLTexture> pairMotion = directImport ? motion : v.motion;
        id<MTLTexture> target1 = directExport ? output : v.generated;

        if (ok) {
            ok = metal4
                ? v.runtime->encode_pair(command, pairColor, pairDepth, pairMotion, hardFrame, sizeof hardFrame, error)
                : v.runtime->encode_pair_classic(command, pairColor, pairDepth, pairMotion, hardFrame, sizeof hardFrame, error);
        }

        const double phase1 = want3x ? (1.0 / 3.0) : 0.5;
        if (ok && !priming) {
            ok = metal4
                ? v.runtime->encode_interpolation(command, phase1, target1, error)
                : v.runtime->encode_interpolation_classic(command, phase1, target1, error);
        }
        if (ok && !priming && !directExport) {
            // Delay fence update until after phase 2/3 when 3x is active.
            ok = helper(v, command, metal4, 1, color, depth, motion, output, want3x ? nil : fence, ui);
        }

        // In 3x mode, encode the second intermediate frame (t = 2/3) into slot.extraOutput
        // on the same command buffer (reusing the pair features computed once above!).
        bool encodedExtra = false;
        if (ok && !priming && want3x) {
            id<MTLTexture> target2 = directExport ? slot.extraOutput : v.generated;
            ok = metal4
                ? v.runtime->encode_interpolation(command, 2.0 / 3.0, target2, error)
                : v.runtime->encode_interpolation_classic(command, 2.0 / 3.0, target2, error);
            if (ok && !directExport) {
                ok = helper(v, command, metal4, 4, color, depth, motion, slot.extraOutput, fence, ui);
            }
            encodedExtra = ok;
        }

        if (ok && directExport && fence) {
            if (metal4) {
                auto enc = [(id<MTL4CommandBuffer>)command computeCommandEncoder];
                ok = enc != nil;
                if (enc) {
                    [enc barrierAfterQueueStages:MTLStageAll beforeStages:MTLStageDispatch visibilityOptions:MTL4VisibilityOptionDevice];
                    [enc updateFence:fence afterEncoderStages:MTLStageDispatch];
                    [enc endEncoding];
                }
            } else {
                auto enc = [(id<MTLCommandBuffer>)command computeCommandEncoder];
                ok = enc != nil;
                if (enc) {
                    [enc updateFence:fence];
                    [enc endEncoding];
                }
            }
        }

        if (metal4) {
            if (!helper(v, command, true, 2, nil, nil, nil, nil, nil)) {
                v.failed = true;
                return reject("ANR: completion encoder unavailable; resources retained");
            }
        } else {
            const auto retained = view;
            const unsigned index = v.activeSlot;
            [(id<MTLCommandBuffer>)command addCompletedHandler:^(id<MTLCommandBuffer> done) {
                __atomic_store_n((uint32_t *)retained->slots[index].completion.contents,
                                 done.status == MTLCommandBufferStatusCompleted ? 1u : 2u,
                                 __ATOMIC_RELEASE);
            }];
        }

        if (!ok) {
            v.history = false;
            return reject(error.empty() ? "ANR: FG encode failed; keeping MetalFX" : error.c_str());
        }

        v.history = true;
        v.swapPending = true;

        if (encodedExtra) {
            const uint64_t ticket = state.nextExtraTicket++;
            slot.extraInUse = true;
            slot.extraTicket = ticket;
            state.inFlightExtraTickets[ticket] = {view, v.activeSlot};
            const double stepSec = std::clamp(v.emaDeltaSec / 3.0, 1.0 / 360.0, 1.0 / 30.0);
            while (state.pendingExtras.size() >= 3) {
                const uint64_t dropped = state.pendingExtras.front().ticket;
                state.pendingExtras.pop_front();
                release_slot_ticket_locked(state, dropped);
            }
            state.pendingExtras.push_back({
                (uintptr_t)f.device, ticket, slot.extraOutput, view, v.activeSlot,
                stepSec, PresentPhase::WaitingInterp, now
            });
        }

        if (priming) {
            ++status.primed;
            status.active = 0;
            snprintf(status.reason, sizeof status.reason, "ANR: history primed; MetalFX presents first/reset frame");
            publish();
            return 2;
        }

        status.active = 1;
        if (selMode == 2 && !isV2) {
            snprintf(status.reason, sizeof status.reason, "ANR V.B14 runtime only supports 2x; prepare V.B20 for 3x");
        } else {
            status.reason[0] = 0;
        }
        ++status.generated;
        if (status.uiMode == FSR4_ANR_UI_PRECOMPOSITED) ++status.uiComposited;
        else if (status.uiMode == FSR4_ANR_UI_OVERLAY) ++status.uiOverlay;
        publish();
        return 1;
    }
}

int fsr4_anr_on_present_event(void *device, void **outExtraTexture, double *outStepSeconds, uint64_t *outTicket) {
    if (outExtraTexture) *outExtraTexture = nullptr;
    if (outStepSeconds) *outStepSeconds = 0.0;
    if (outTicket) *outTicket = 0;
    if (fsr4_anr_selected_mode() != 2) return 0;

    auto &state = live_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    const auto now = std::chrono::steady_clock::now();
    prune_stale_extras_locked(state, now);

    const uintptr_t devKey = (uintptr_t)device;
    for (auto it = state.pendingExtras.begin(); it != state.pendingExtras.end(); ++it) {
        if (devKey && it->device != devKey) continue;
        if (it->phase == PresentPhase::WaitingInterp) {
            // First present after encode_frame: OutputInterpolated (t = 1/3).
            // Hand over t = 2/3 texture to insert immediately after, then wait for OutputReal.
            if (outExtraTexture) *outExtraTexture = (__bridge_retained void *)it->texture;
            if (outStepSeconds) *outStepSeconds = it->stepSeconds;
            if (outTicket) *outTicket = it->ticket;
            it->texture = nil;
            it->phase = PresentPhase::WaitingReal;
            return 1;
        }
        // Second present: OutputReal (R_n). Pace it with stepSeconds and retire the pending record.
        if (outStepSeconds) *outStepSeconds = it->stepSeconds;
        state.pendingExtras.erase(it);
        return 2;
    }
    return 0;
}

void fsr4_anr_release_extra_ticket(uint64_t ticket, int presented) {
    if (!ticket) return;
    auto &state = live_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    release_slot_ticket_locked(state, ticket);
    if (presented) {
        ++state.status.generatedExtra3x;
        publish();
    }
}

void fsr4_anr_invalidate_owner(void *device, void *owner) {
    auto &state = live_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    auto it = state.views.find(key(device, owner));
    if (it != state.views.end() && it->second) it->second->history = false;
    clear_pending_extras_locked(state);
}

void fsr4_anr_release_owner(void *device, void *owner) {
    auto &state = live_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    auto it = state.views.find(key(device, owner));
    if (it != state.views.end()) {
        if (it->second && !it->second->finished()) state.retired.push_back(it->second);
        state.views.erase(it);
    }
    state.failures.erase(key(device, owner));
    clear_pending_extras_locked(state);
    collect();
}

extern "C" void fsr4_anr_mode_changed(int mode) {
    auto &state = live_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    auto &status = state.status;
    const bool wasAnr = status.selectedAnr != 0;
    const bool wantAnr = mode != 0;
    state.selectedMode = mode;
    status.selectedAnr = wantAnr ? 1 : 0;
    status.multiplier = mode == 2 ? 3u : (mode == 1 ? 2u : 1u);
    clear_pending_extras_locked(state);

    // Switching live between ANR 2x and ANR 3x keeps V.B20 views/pipelines warm (zero hitch).
    if (wasAnr != wantAnr) {
        status.active = 0;
        for (auto &entry : state.views) {
            if (entry.second && !entry.second->finished()) state.retired.push_back(entry.second);
        }
        state.views.clear();
        state.failures.clear();
    }
    collect();
    snprintf(status.reason, sizeof status.reason, "%s",
             mode == 2 ? "ANR 3x selected; waiting for an accepted frame"
                       : mode == 1 ? "ANR 2x selected; waiting for an accepted frame"
                                   : "MetalFX selected; original native encode");
    state.lastPublish = {};
    publish();
}

void fsr4_anr_publish_passthrough(void) {
    auto &state = live_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    collect();
    state.selectedMode = 0;
    state.status.selectedAnr = 0;
    state.status.multiplier = 1u;
    state.status.active = 0;
    snprintf(state.status.reason, sizeof state.status.reason, "MetalFX selected; original native encode");
    publish();
}

void fsr4_anr_set_inactive(const char *why) {
    auto &state = live_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    state.selectedMode = fsr4_anr_selected_mode();
    reject(why ? why : "ANR inactive");
}

void fsr4_anr_get_status(Fsr4AnrStatus *out) {
    if (!out) return;
    auto &state = live_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    *out = state.status;
}
