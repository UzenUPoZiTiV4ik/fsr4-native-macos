// SPDX-License-Identifier: MIT
// Full-frame A/B for the pre-M5 SIMD backend, in ONE process.
// A = direct 3x3 (FSR4_SIMD_WINOGRAD=0 before its first session load),
// B = exact Winograd. Both runtimes share textures, weights and caches and are
// interleaved frame by frame (ABBA order), so power/clock changes affect both.
// Also checks that A and B produce byte-identical output, history and state.
// usage: fsr411_simd_ab ROOT [--rounds N] [--size RW RH UW UH] [--models lite,exact,lossy2]
#include "../../src/fsr412_runtime.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace {
void require(bool ok, const std::string &message) { if (!ok) throw std::runtime_error(message); }
id<MTLTexture> texture(id<MTLDevice> dev, MTLPixelFormat format, unsigned w, unsigned h) {
    auto d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format width:w height:h mipmapped:NO];
    d.storageMode = MTLStorageModeShared; d.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
    auto t = [dev newTextureWithDescriptor:d]; require(t != nil, "texture allocation failed"); return t;
}
struct Inputs { id<MTLTexture> color, depth, motion, exposure; };
Inputs make_inputs(id<MTLDevice> dev, unsigned rw, unsigned rh) {
    Inputs in{texture(dev,MTLPixelFormatRGBA16Float,rw,rh),texture(dev,MTLPixelFormatR32Float,rw,rh),
              texture(dev,MTLPixelFormatRG16Float,rw,rh),texture(dev,MTLPixelFormatR32Float,1,1)};
    std::vector<_Float16> c(size_t(rw)*rh*4), m(size_t(rw)*rh*2);
    std::vector<float> d(size_t(rw)*rh);
    uint32_t r = 0x41a8u;
    for (unsigned y = 0; y < rh; ++y) for (unsigned x = 0; x < rw; ++x) {
        const size_t i = size_t(y)*rw + x;
        r = r*1664525u + 1013904223u;
        // Smooth gradients plus texture-like noise: non-trivial activations.
        c[i*4] = (_Float16)(.05f + .8f*float(x)/rw + float(r >> 28)/64.f);
        c[i*4+1] = (_Float16)(.03f + .6f*float(y)/rh + float((r >> 24) & 15)/80.f);
        c[i*4+2] = (_Float16)(float(((x*7 + y*3) % 23) + 1)/30.f);
        c[i*4+3] = (_Float16)1.f;
        d[i] = .2f + .6f*float((x/8 + y/8) % 11)/11.f;
        m[i*2] = (_Float16)(float(int(x % 9) - 4)*.0625f); m[i*2+1] = (_Float16)(float(int(y % 5) - 2)*.0625f);
    }
    [in.color replaceRegion:MTLRegionMake2D(0,0,rw,rh) mipmapLevel:0 withBytes:c.data() bytesPerRow:rw*8];
    [in.depth replaceRegion:MTLRegionMake2D(0,0,rw,rh) mipmapLevel:0 withBytes:d.data() bytesPerRow:rw*4];
    [in.motion replaceRegion:MTLRegionMake2D(0,0,rw,rh) mipmapLevel:0 withBytes:m.data() bytesPerRow:rw*4];
    const float one = 1.f;
    [in.exposure replaceRegion:MTLRegionMake2D(0,0,1,1) mipmapLevel:0 withBytes:&one bytesPerRow:4];
    return in;
}
fsr411::Frame frame(const Inputs &in, id<MTLTexture> output, unsigned rw, unsigned rh, unsigned uw, unsigned uh, unsigned index) {
    fsr411::Frame f; f.color = in.color; f.depth = in.depth; f.motion = in.motion; f.exposure = in.exposure; f.output = output;
    auto &d = f.input;
    d.rw = rw; d.rh = rh; d.uw = uw; d.uh = uh;
    d.geometry.color = {rw,rh,0,0}; d.geometry.depth = d.geometry.color; d.geometry.motion = d.geometry.color;
    d.geometry.output = {uw,uh,0,0};
    // Same two-frame pattern as testcases/tc_game_balanced: reset + negative-x
    // jitter, so Lossy reuses inference on exactly the frames it does in replay.
    d.reset = index % 2 == 0;
    d.jitterX = index % 2 ? -.25f : 0.f; d.jitterY = index % 2 ? 1.f/6.f : -1.f/6.f;
    d.mvScaleX = d.mvScaleY = 1; d.depthReversed = 1; d.preExposure = 1; d.hasExposureTexture = 1;
    return f;
}
std::vector<uint8_t> read_texture(id<MTLCommandQueue> queue, id<MTLTexture> t, unsigned bytes) {
    const NSUInteger pitch = (t.width*bytes + 255u) & ~NSUInteger(255);
    auto buffer = [t.device newBufferWithLength:pitch*t.height options:MTLResourceStorageModeShared];
    auto cb = [queue commandBuffer]; auto blit = [cb blitCommandEncoder];
    [blit copyFromTexture:t sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0,0,0) sourceSize:MTLSizeMake(t.width,t.height,1)
          toBuffer:buffer destinationOffset:0 destinationBytesPerRow:pitch destinationBytesPerImage:pitch*t.height];
    [blit endEncoding]; [cb commit]; [cb waitUntilCompleted];
    std::vector<uint8_t> result(t.width*bytes*t.height);
    for (NSUInteger y = 0; y < t.height; ++y) memcpy(result.data() + y*t.width*bytes, (uint8_t *)buffer.contents + y*pitch, t.width*bytes);
    return result;
}
double median(std::vector<double> v) { std::sort(v.begin(), v.end()); const size_t n = v.size(); return n ? (n%2 ? v[n/2] : (v[n/2-1]+v[n/2])/2) : 0; }
NSArray *array(const std::vector<double> &v) { NSMutableArray *a = [NSMutableArray new]; for (double x : v) [a addObject:@(x)]; return a; }
}

int main(int argc, char **argv) { @autoreleasepool { try {
    require(argc >= 2, "usage: fsr411_simd_ab ROOT [--rounds N] [--size RW RH UW UH] [--models lite,exact,lossy2]");
    const std::string root = argv[1];
    unsigned rounds = 30, rw = 1000, rh = 648, uw = 1712, uh = 1112; std::string models = "lite,exact,lossy2";
    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--rounds" && i+1 < argc) rounds = unsigned(std::stoul(argv[++i]));
        else if (a == "--size" && i+4 < argc) { rw = std::stoul(argv[++i]); rh = std::stoul(argv[++i]); uw = std::stoul(argv[++i]); uh = std::stoul(argv[++i]); }
        else if (a == "--models" && i+1 < argc) models = argv[++i];
        else throw std::runtime_error("unknown option " + a);
    }
    require(rounds >= 2 && rounds <= 10000, "rounds must be 2..10000");
    auto dev = MTLCreateSystemDefaultDevice(); auto queue = [dev newCommandQueue];
    require(dev && queue && [dev supportsFamily:MTLGPUFamilyApple7], "Apple7+ GPU required");
    const auto inputs = make_inputs(dev, rw, rh);
    // Variant A must load its sessions while Winograd is disabled; the runtime
    // caches its device decision on the first SIMD library load.
    setenv("FSR4_SIMD_WINOGRAD", "0", 1);
    fsr411::Runtime direct(dev, root, root + "/build");
    struct Model { std::string name; bool lite; fsr411::Profile profile; };
    std::vector<Model> list;
    for (const auto &m : std::vector<Model>{{"lite",true,fsr411::Profile::Int8Lite},{"exact",false,fsr411::Profile::Exact},
                                           {"lossy2",false,fsr411::Profile::Lossy2}})
        if (("," + models + ",").find("," + m.name + ",") != std::string::npos) list.push_back(m);
    require(!list.empty(), "no models selected");
    struct State { fsr411::View view[2]; id<MTLTexture> output[2]; fsr411::Options options;
                   std::vector<double> neural[2], reused[2], ratio; };
    std::vector<State> states(list.size());
    auto encode = [&](fsr411::Runtime &rt, State &s, unsigned v, unsigned index) {
        auto cb = [queue commandBuffer]; std::string error;
        require(rt.encode(cb, s.view[v], frame(inputs, s.output[v], rw, rh, uw, uh, index), s.options, error), error);
        [cb commit]; [cb waitUntilCompleted];
        require(cb.status == MTLCommandBufferStatusCompleted, fsr411::metal_error(cb.error));
        return std::make_pair((cb.GPUEndTime - cb.GPUStartTime)*1000., s.view[v].skippedInference);
    };
    for (size_t i = 0; i < list.size(); ++i) {
        auto &s = states[i];
        if (list[i].lite) s.options = fsr412::Options(); else s.options.profile = list[i].profile;
        s.options.backend = fsr411::Backend::Simd; s.options.allowCompile = true; s.options.optimize = 1;
        for (unsigned v = 0; v < 2; ++v) s.output[v] = texture(dev, MTLPixelFormatRGBA16Float, uw, uh);
        encode(direct, s, 0, 0); // loads A's sessions (direct 3x3)
        require(s.view[0].backendLabel == "int8-simd", "SIMD backend not selected");
    }
    unsetenv("FSR4_SIMD_WINOGRAD");
    fsr411::Runtime winograd(dev, root, root + "/build");
    fsr411::Runtime *rt[2] = {&direct, &winograd};
    for (auto &s : states) encode(winograd, s, 1, 0);
    // Warm both, then measure. Pairs alternate order: A,B then B,A.
    for (unsigned round = 0; round < rounds + 2; ++round)
        for (size_t i = 0; i < list.size(); ++i) for (unsigned index = 0; index < 2; ++index) {
            auto &s = states[i]; double t[2]; bool skipped[2];
            for (unsigned k = 0; k < 2; ++k) {
                const unsigned v = (round + index) % 2 ? 1 - k : k;
                auto r = encode(*rt[v], s, v, index); t[v] = r.first; skipped[v] = r.second;
            }
            require(skipped[0] == skipped[1], "A/B inference reuse decisions differ");
            if (round < 2) continue;
            for (unsigned v = 0; v < 2; ++v) (skipped[v] ? s.reused[v] : s.neural[v]).push_back(t[v]);
            if (!skipped[0]) s.ratio.push_back(t[1]/t[0]);
        }
    NSMutableArray *rows = [NSMutableArray new];
    bool identical = true;
    for (size_t i = 0; i < list.size(); ++i) {
        auto &s = states[i];
        const bool same = read_texture(queue,s.output[0],8) == read_texture(queue,s.output[1],8)
            && read_texture(queue,s.view[0].history,8) == read_texture(queue,s.view[1].history,8)
            && read_texture(queue,s.view[0].recurrent,4) == read_texture(queue,s.view[1].recurrent,4);
        identical &= same;
        const double a = median(s.neural[0]), b = median(s.neural[1]);
        std::printf("%-6s neural frame median: direct %.3f ms, winograd %.3f ms, paired %+.1f%% (%zu pairs); reused %zu; output/history/recurrent %s\n",
                    list[i].name.c_str(), a, b, (median(s.ratio)-1)*100, s.ratio.size(), s.reused[0].size(), same ? "IDENTICAL" : "DIFFER");
        [rows addObject:@{@"model":fsr411::ns(list[i].name),@"neural_direct_ms":array(s.neural[0]),@"neural_winograd_ms":array(s.neural[1]),
            @"reused_direct_ms":array(s.reused[0]),@"reused_winograd_ms":array(s.reused[1]),
            @"neural_direct_median_ms":@(a),@"neural_winograd_median_ms":@(b),
            @"paired_change_percent":@((median(s.ratio)-1)*100),@"byte_identical":@(same)}];
    }
    NSDictionary *report = @{@"device":dev.name,@"scope":@"full frame command buffer, forced SIMD, interleaved in one process",
        @"render":@[@(rw),@(rh)],@"output":@[@(uw),@(uh)],@"rounds":@(rounds),
        @"low_power_mode":@(NSProcessInfo.processInfo.lowPowerModeEnabled),
        @"thermal_state_end":@(NSProcessInfo.processInfo.thermalState),@"models":rows};
    NSData *json = [NSJSONSerialization dataWithJSONObject:report options:NSJSONWritingPrettyPrinted error:nil];
    [json writeToFile:@"build/fsr411_simd_ab.json" atomically:YES];
    std::printf("report: build/fsr411_simd_ab.json\n");
    return identical ? 0 : 1;
} catch (const std::exception &e) { std::fprintf(stderr, "fsr411_simd_ab: %s\n", e.what()); return 1; } } }
