// SPDX-License-Identifier: MIT
// Steady-state whole-frame model comparison. Native models use the same
// Runtime::encode as the CLI; optional 4.0.2 uses the production live encoder.
// No Windows DLL, game, GPU counters or capture copies run in timed buffers.
#include "../../src/fsr412_runtime.h"
#include "../../src/fsr4_live.h"
#include <chrono>
#include <fcntl.h>
#include <numeric>
#include <stdexcept>
#include <sys/file.h>
#include <unistd.h>

// A benchmark must not launch the panel or overwrite a user's status file.
extern "C" void fsr4_panel_request(void) {}
extern "C" void fsr4_panel_publish(void) {}

namespace {
using Clock = std::chrono::steady_clock;
void require(bool ok, const std::string &message) { if (!ok) throw std::runtime_error(message); }
double seconds(Clock::time_point start) { return std::chrono::duration<double>(Clock::now() - start).count(); }
struct TimingLock {
    int fd = -1;
    TimingLock() {
        fd = open("/tmp/fsr4-gpu-timing.lock", O_CREAT | O_RDWR, 0600);
        require(fd >= 0 && flock(fd, LOCK_EX | LOCK_NB) == 0, "another GPU timing process holds /tmp/fsr4-gpu-timing.lock");
    }
    ~TimingLock() { if (fd >= 0) { flock(fd, LOCK_UN); close(fd); } }
};
id<MTLTexture> texture(id<MTLDevice> device, MTLPixelFormat format, unsigned w, unsigned h) {
    auto d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format width:w height:h mipmapped:NO];
    d.storageMode = MTLStorageModeShared;
    d.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
    auto t = [device newTextureWithDescriptor:d]; require(t != nil, "cannot allocate texture"); return t;
}
struct Inputs { id<MTLTexture> color, depth, motion; };
Inputs inputs(id<MTLDevice> device, unsigned rw, unsigned rh) {
    Inputs in{texture(device, MTLPixelFormatRGBA16Float, rw, rh), texture(device, MTLPixelFormatR32Float, rw, rh),
              texture(device, MTLPixelFormatRG16Float, rw, rh)};
    std::vector<_Float16> color(size_t(rw) * rh * 4), motion(size_t(rw) * rh * 2);
    std::vector<float> depth(size_t(rw) * rh);
    uint32_t seed = 0x4128u;
    for (unsigned y = 0; y < rh; ++y) for (unsigned x = 0; x < rw; ++x) {
        const size_t i = size_t(y) * rw + x; seed = seed * 1664525u + 1013904223u;
        color[4*i] = (_Float16)(.05f + .8f * float(x) / rw + float(seed >> 28) / 64.f);
        color[4*i+1] = (_Float16)(.03f + .6f * float(y) / rh + float((seed >> 24) & 15) / 80.f);
        color[4*i+2] = (_Float16)(float(((x*7 + y*3) % 23) + 1) / 30.f); color[4*i+3] = 1;
        depth[i] = .2f + .6f * float((x/8 + y/8) % 11) / 11.f;
        motion[2*i] = (_Float16)(float(int(x % 9) - 4) * .0625f);
        motion[2*i+1] = (_Float16)(float(int(y % 5) - 2) * .0625f);
    }
    [in.color replaceRegion:MTLRegionMake2D(0,0,rw,rh) mipmapLevel:0 withBytes:color.data() bytesPerRow:rw*8];
    [in.depth replaceRegion:MTLRegionMake2D(0,0,rw,rh) mipmapLevel:0 withBytes:depth.data() bytesPerRow:rw*4];
    [in.motion replaceRegion:MTLRegionMake2D(0,0,rw,rh) mipmapLevel:0 withBytes:motion.data() bytesPerRow:rw*4];
    return in;
}
fsr411::Frame frame(const Inputs &in, id<MTLTexture> output, unsigned rw, unsigned rh, unsigned uw, unsigned uh,
                    unsigned index, bool reset) {
    fsr411::Frame f; f.color = in.color; f.depth = in.depth; f.motion = in.motion; f.output = output;
    auto &p = f.input; p.rw = rw; p.rh = rh; p.uw = uw; p.uh = uh;
    p.geometry.color = p.geometry.depth = p.geometry.motion = {rw,rh,0,0}; p.geometry.output = {uw,uh,0,0};
    p.reset = reset; p.jitterX = index & 1u ? -.25f : .125f;
    p.jitterY = index % 3u ? 1.f/6.f : -1.f/6.f;
    p.mvScaleX = p.mvScaleY = 1; p.depthReversed = 1; p.preExposure = 1; p.autoExposure = 1;
    p.inputContentProperties = 1; return f;
}
NSArray *numbers(const std::vector<double> &v) { auto a = [NSMutableArray new]; for (double n : v) [a addObject:@(n)]; return a; }
double mean(const std::vector<double> &v) { return v.empty() ? 0 : std::accumulate(v.begin(),v.end(),0.) / v.size(); }
double quantile(std::vector<double> v, double p) {
    if (v.empty()) return 0; std::sort(v.begin(),v.end());
    if (p <= 0) return v.front();
    if (p >= 1) return v.back();
    if (p == .5) return v.size() & 1u ? v[v.size()/2] : (v[v.size()/2-1] + v[v.size()/2])*.5;
    return v[std::min(v.size()-1, size_t(std::ceil(p*v.size())-1))];
}
NSDictionary *thermal(double elapsed) {
    return @{ @"elapsed_s": @(elapsed), @"state": @(NSProcessInfo.processInfo.thermalState),
              @"low_power_mode": @(NSProcessInfo.processInfo.lowPowerModeEnabled) };
}
void write_report(const std::string &path, NSDictionary *report) {
    NSError *error = nil;
    NSData *data = [NSJSONSerialization dataWithJSONObject:report options:NSJSONWritingPrettyPrinted | NSJSONWritingSortedKeys error:&error];
    require(data != nil && [data writeToFile:fsr411::ns(path) options:NSDataWritingAtomic error:&error], fsr411::metal_error(error));
}
struct Mode {
    std::string name, label;
    fsr411::Options options;
    fsr411::View view;
    id<MTLTexture> output = nil;
    bool legacy = false;
    std::vector<double> all, full, reuse, partial, roundMeans;
    std::vector<std::vector<double>> rounds;
};
}

int main(int argc, char **argv) { @autoreleasepool { try {
    require(argc >= 2, "usage: fsr_modes_thermal ROOT --size RW RH UW UH [--rounds N --frames-per-round N --warmup-seconds N --legacy --reverse --report FILE]");
    const std::string root = argv[1]; std::string report = root + "/build/fsr-modes-thermal.json";
    unsigned rw = 1280, rh = 720, uw = 1920, uh = 1080, rounds = 6, frames = 64;
    double warmupSeconds = 120; bool legacy = false, reverse = false, allowCompile = true;
    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--size" && i + 4 < argc) { rw=std::stoul(argv[++i]); rh=std::stoul(argv[++i]); uw=std::stoul(argv[++i]); uh=std::stoul(argv[++i]); }
        else if (a == "--rounds" && i + 1 < argc) rounds=std::stoul(argv[++i]);
        else if (a == "--frames-per-round" && i + 1 < argc) frames=std::stoul(argv[++i]);
        else if (a == "--warmup-seconds" && i + 1 < argc) warmupSeconds=std::stod(argv[++i]);
        else if (a == "--report" && i + 1 < argc) report=argv[++i];
        else if (a == "--legacy") legacy=true;
        else if (a == "--reverse") reverse=true;
        else if (a == "--no-compile") allowCompile=false;
        else throw std::runtime_error("unknown/incomplete option: " + a);
    }
    require(rw && rh && rw<=3840 && rh<=2160 && uw>=4 && uh>=4 && uw<=3840 && uh<=2160 && !(uw&7u) && !(uh&7u), "invalid dimensions");
    require(rounds>=2 && rounds<=100 && frames>=8 && frames<=1024 && !(frames&7u)
            && std::isfinite(warmupSeconds) && warmupSeconds>=0 && warmupSeconds<=900, "invalid timing counts");
    TimingLock timingLock;
    require(!NSProcessInfo.processInfo.lowPowerModeEnabled, "disable Low Power Mode before benchmarking");
    auto device = MTLCreateSystemDefaultDevice(); auto queue = [device newCommandQueue];
    require(device && queue && [device supportsFamily:MTLGPUFamilyApple10], "Apple10/M5 native INT8 device required");
    // Use no installer/panel settings or developer overrides. These variables
    // belong only to this child; the converter/data-location settings stay.
    for (const char *key : {"FSR4_MPP_OPTIMIZE", "FSR4_LITE_POST_TILE", "FSR4_PRESET", "FSR4_LOW_RES_MV",
                           "FSR4_JITTERED_MV", "FSR4_AUTOEXPOSURE", "FSR4_COLORSPACE"}) unsetenv(key);
    setenv("FSR4_ROOT", root.c_str(), 1); setenv("FSR4_SETTINGS", "none", 1);
    setenv("FSR4_VERSION", "4.0.2", 1); setenv("FSR4_MODEL", "i8mpp", 1);
    setenv("FSR4_SHARPNESS", "0", 1); setenv("FSR4_RCAS", "0", 1); unsetenv("FSR4_HOOK_LOG");
    fsr4_live_set_profile(3); fsr4_live_set_sharpness(0, 0);
    fsr411::Runtime runtime(device, root, root + "/build");
    const auto in = inputs(device, rw, rh);
    std::vector<std::unique_ptr<Mode>> modes;
    auto add = [&](const char *name, const char *label, fsr411::Profile profile, bool lite, int quality=0) -> Mode & {
        auto m = std::make_unique<Mode>(); m->name=name; m->label=label;
        if (lite) m->options=fsr412::Options();
        m->options.profile=profile; m->options.modelLite=lite; m->options.backend=fsr411::Backend::Native;
        m->options.temporalQuality=quality; m->options.rcas=false; m->options.sharpness=0; m->options.allowCompile=allowCompile;
        m->output=texture(device,MTLPixelFormatRGBA16Float,uw,uh); modes.push_back(std::move(m)); return *modes.back();
    };
    add("exact-fast", "FSR 4.1.2 Exact (panel)", fsr411::Profile::ExactFast, false);
    add("lossy2", "FSR 4.1.2 Lossy (panel)", fsr411::Profile::Lossy2, false);
    add("int8-lite", "FSR 4.1.2 Lite", fsr411::Profile::Int8Lite, true);
    auto &fix = add("int8-lite-pixfix", "Lite + Pixelization fix (0.5)", fsr411::Profile::Int8Lite, true);
    fsr412::Options fixOptions; fsr412::apply_pixel_fix(fixOptions,true,.5f);
    fixOptions.backend=fsr411::Backend::Native; fixOptions.allowCompile=allowCompile; fix.options=fixOptions;
    add("int8-lite-lossy", "FSR 4.1.2 Lite Lossy", fsr411::Profile::Int8LiteLossy, true);
    add("int8-lite-lossy-tiled", "Lite Lossy Tiled (512, CLI only)", fsr411::Profile::Int8LiteLossy, true).options.neuralTileSize=512;
    add("int8-lite-lossy-original", "Lite Lossy Original (CLI only)", fsr411::Profile::Int8LiteLossy, true).options.originalLossy=true;
    add("exact", "Exact IEEE reference (CLI only)", fsr411::Profile::Exact, false);
    add("lossy", "Legacy Lossy (CLI only)", fsr411::Profile::Lossy, false);
    add("lossy2-temporal1", "Lossy + Temporal quality 1 (CLI only)", fsr411::Profile::Lossy2, false, 1);
    add("lossy2-temporal2", "Lossy + Temporal quality 2 (CLI only)", fsr411::Profile::Lossy2, false, 2);
    if (legacy) add("fsr4-legacy", "FSR 4.0.2 (optional legacy model)", fsr411::Profile::Exact, false).legacy=true;
    if (reverse) std::reverse(modes.begin(),modes.end());
    auto encode = [&](Mode &m, unsigned index, bool reset) -> double { @autoreleasepool {
        const auto f = frame(in,m.output,rw,rh,uw,uh,index,reset);
        auto cb = [queue commandBuffer]; require(cb != nil, "cannot allocate command buffer");
        if (m.legacy) {
            Fsr4LiveFrameV2 v={}; v.structSize=sizeof(v); v.abiVersion=FSR4_LIVE_FRAME_ABI_V2;
            auto &p=v.frame; p.device=(__bridge void *)device; p.commandBuffer=(__bridge void *)cb; p.owner=&m;
            p.color=(__bridge void *)in.color; p.depth=(__bridge void *)in.depth; p.motion=(__bridge void *)in.motion;
            p.output=(__bridge void *)m.output; p.rw=rw; p.rh=rh; p.uw=uw; p.uh=uh;
            p.jitterX=f.input.jitterX; p.jitterY=f.input.jitterY; p.mvScaleX=p.mvScaleY=1;
            p.preExposure=1; p.reset=reset; p.depthReversed=1; p.autoExposure=1;
            v.geometry=f.input.geometry; v.inputContentProperties=0;
            const bool ok=fsr4_encode_live_frame_v2(&v); Fsr4LiveStatus status={}; fsr4_live_get_status(&status);
            require(ok && status.active && std::string(status.model)=="i8mpp", m.name+": "+status.reason);
        } else {
            std::string error; const bool ok=runtime.encode(cb,m.view,f,m.options,error); require(ok,m.name+": "+error);
            require(m.view.backendLabel=="int8-mpp",m.name+": unexpected backend");
        }
        [cb commit]; [cb waitUntilCompleted];
        require(cb.status==MTLCommandBufferStatusCompleted,fsr411::metal_error(cb.error));
        const double ms=(cb.GPUEndTime-cb.GPUStartTime)*1000.; require(std::isfinite(ms) && ms>0,"invalid GPU timestamps");
        return ms;
    }};
    auto telemetry=[NSMutableArray new]; const auto started=Clock::now();
    [telemetry addObject:thermal(0)];
    // All compilation/pipeline creation and state allocations finish before warmup.
    for (auto &m : modes) { encode(*m,0,true); m->options.allowCompile=false; }
    std::printf("preflight: %zu models, %ux%u -> %ux%u; all pipelines resident\n",modes.size(),rw,rh,uw,uh); std::fflush(stdout);
    const auto warming=Clock::now(); unsigned sweep=0; double nextLog=30;
    auto sweep_modes=[&](unsigned index, unsigned round, bool measured) {
        const size_t rotation=(round + index) % modes.size();
        for (size_t step=0; step<modes.size(); ++step) {
            const size_t offset=(round + index)&1u ? modes.size()-1-step : step;
            auto &m=*modes[(rotation + offset) % modes.size()]; const double ms=encode(m,index&7u,false);
            if (measured) {
                m.all.push_back(ms); m.rounds.back().push_back(ms);
                if (!m.legacy && m.view.partialInference) m.partial.push_back(ms);
                else if (!m.legacy && m.view.skippedInference) m.reuse.push_back(ms);
                else m.full.push_back(ms);
            }
        }
    };
    do {
        sweep_modes(sweep&7u,sweep/8u,false); ++sweep;
        if (seconds(warming)>=nextLog) {
            [telemetry addObject:thermal(seconds(started))];
            std::printf("warmup %.1f s, thermal state %ld\n",seconds(warming),(long)NSProcessInfo.processInfo.thermalState);
            std::fflush(stdout); nextLog+=30;
        }
    } while (seconds(warming)<warmupSeconds || (sweep&7u));
    const double actualWarmup=seconds(warming); const auto measuring=Clock::now();
    auto roundTelemetry=[NSMutableArray new];
    for (unsigned round=0; round<rounds; ++round) { @autoreleasepool {
        for (auto &m:modes) m->rounds.emplace_back();
        const NSInteger before=NSProcessInfo.processInfo.thermalState;
        for (unsigned index=0; index<frames; ++index) sweep_modes(index,round,true);
        require(!NSProcessInfo.processInfo.lowPowerModeEnabled,"Low Power Mode changed during timings");
        [roundTelemetry addObject:@{@"round":@(round+1),@"thermal_start":@(before),
            @"thermal_end":@(NSProcessInfo.processInfo.thermalState),@"elapsed_s":@(seconds(started))}];
        [telemetry addObject:thermal(seconds(started))];
        std::printf("round %u/%u, thermal %ld -> %ld\n",round+1,rounds,(long)before,(long)NSProcessInfo.processInfo.thermalState);
        for (auto &m:modes) { m->roundMeans.push_back(mean(m->rounds.back())); std::printf("  %-27s %.3f ms\n",m->name.c_str(),m->roundMeans.back()); }
        std::fflush(stdout);
    }}
    const double measuredSeconds=seconds(measuring); auto rows=[NSMutableArray new];
    for (const auto &m:modes) {
        auto blocks=[NSMutableArray new]; for (const auto &block:m->rounds) [blocks addObject:numbers(block)];
        const size_t half=m->roundMeans.size()/2;
        std::vector<double> early(m->roundMeans.begin(),m->roundMeans.begin()+half),late(m->roundMeans.begin()+half,m->roundMeans.end());
        [rows addObject:@{@"name":fsr411::ns(m->name),@"label":fsr411::ns(m->label),
            @"median_round_mean_ms":@(quantile(m->roundMeans,.5)),@"round_means_ms":numbers(m->roundMeans),
            @"round_range_ms":@[@(quantile(m->roundMeans,0)),@(quantile(m->roundMeans,1))],
            @"frame_p99_ms":@(quantile(m->all,.99)),@"frame_max_ms":@(quantile(m->all,1)),
            @"full_mean_ms":@(mean(m->full)),@"reuse_mean_ms":@(mean(m->reuse)),@"partial_mean_ms":@(mean(m->partial)),
            @"full_count":@(m->full.size()),@"reuse_count":@(m->reuse.size()),@"partial_count":@(m->partial.size()),
            @"late_vs_early_percent":@((quantile(late,.5)/quantile(early,.5)-1)*100),
            @"frames_by_round_ms":blocks,@"full_ms":numbers(m->full),@"reuse_ms":numbers(m->reuse),@"partial_ms":numbers(m->partial),
            @"built_in_legacy_probe":@(m->legacy)}];
    }
    NSDictionary *reportData=@{@"schema":@"fsr_modes_thermal_v1",@"device":device.name,
        @"os":NSProcessInfo.processInfo.operatingSystemVersionString,@"date_utc":[[NSISO8601DateFormatter new] stringFromDate:NSDate.date],
        @"render":@[@(rw),@(rh)],@"output":@[@(uw),@(uh)],@"backend":@"native INT8 MPP",@"metal":@"classic command buffers",
        @"scope":@"whole production frame GPU time; no game or Windows DLL; compilation and warmup excluded",
        @"fixture":@"deterministic 0x4128 LCG color/noise, stepped inverted depth, nonzero motion; identical resident inputs for all models",
        @"cadence":@"eight jitter phases, stable pre-exposure; no measured resets; eligible whole-frame modes refresh/reuse 50:50",
        @"auto_exposure":@YES,@"rcas":@NO,@"tile_size":@512,@"pixelization_strength":@0.5,
        @"order":@"serial; rotate first model each frame and reverse direction each frame/round",
        @"rounds":@(rounds),@"frames_per_round":@(frames),@"warmup_requested_s":@(warmupSeconds),@"warmup_actual_s":@(actualWarmup),
        @"measured_wall_s":@(measuredSeconds),@"wall_s":@(seconds(started)),@"low_power_mode":@(NSProcessInfo.processInfo.lowPowerModeEnabled),
        @"thermal_state_key":@{@"0":@"nominal",@"1":@"fair",@"2":@"serious",@"3":@"critical"},
        @"thermal_samples":telemetry,@"round_thermal":roundTelemetry,@"rows":rows};
    write_report(report,reportData); std::printf("report: %s\n",report.c_str()); return 0;
} catch (const std::exception &e) { std::fprintf(stderr,"fsr_modes_thermal: %s\n",e.what()); return 1; } } }
