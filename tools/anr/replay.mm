// SPDX-License-Identifier: MIT
// Independent ANR replay & Metal 4 GPU profiler: supports v1 (B14) and v2 (B20).
#include "anr_runtime.h"
#import <CommonCrypto/CommonDigest.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>
#include <mach/mach_time.h>

static NSData *read_bin(const std::string &path) {
    NSData *data = [NSData dataWithContentsOfFile:[NSString stringWithUTF8String:path.c_str()]];
    if (!data) throw std::runtime_error("cannot read " + path);
    return data;
}

static id<MTLTexture> make_texture(id<MTLDevice> device, unsigned w, unsigned h, MTLPixelFormat format) {
    MTLTextureDescriptor *descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format
                                                                                          width:w
                                                                                         height:h
                                                                                      mipmapped:NO];
    descriptor.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
    descriptor.storageMode = MTLStorageModeShared;
    auto result = [device newTextureWithDescriptor:descriptor];
    if (!result) throw std::runtime_error("texture allocation failed");
    return result;
}

static void upload_tex(id<MTLTexture> texture, NSData *data, unsigned bpp) {
    if (data.length != texture.width * texture.height * bpp) {
        throw std::runtime_error("input texture size mismatch");
    }
    [texture replaceRegion:MTLRegionMake2D(0, 0, texture.width, texture.height)
               mipmapLevel:0
                 withBytes:data.bytes
               bytesPerRow:texture.width * bpp];
}

static void require(bool condition, const std::string &why) {
    if (!condition) throw std::runtime_error(why);
}

static double median(std::vector<double> values) {
    if (values.empty()) return 0;
    std::sort(values.begin(), values.end());
    return values[values.size() / 2];
}

static NSString *sha256_ns(NSData *data) {
    unsigned char digest[32];
    CC_SHA256(data.bytes, (CC_LONG)data.length, digest);
    NSMutableString *result = [NSMutableString string];
    for (unsigned char value : digest) [result appendFormat:@"%02x", value];
    return result;
}

int main(int argc, char **argv) {
    @autoreleasepool {
        try {
            if (argc < 4) {
                fprintf(stderr, "usage: anr_replay <runtime> <inputs> <new-out-dir> [frames=8] [phase=0.5] [reference] [prime=1] [width=1920] [height=1080]\n");
                return 2;
            }
            const std::string root = argv[1];
            const std::string inputs = argv[2];
            const std::string out = argv[3];
            const int frames = argc > 4 ? std::stoi(argv[4]) : 8;
            const double phase = argc > 5 ? std::stod(argv[5]) : 0.5;
            const int prime = argc > 7 ? std::stoi(argv[7]) : 1;
            const unsigned reqW = argc > 8 ? (unsigned)std::stoul(argv[8]) : 1920u;
            const unsigned reqH = argc > 9 ? (unsigned)std::stoul(argv[9]) : 1080u;
            const bool mode3x = getenv("ANR_3X") && !strcmp(getenv("ANR_3X"), "1");
            const bool doProfile = getenv("ANR_PROFILE") && !strcmp(getenv("ANR_PROFILE"), "1");

            require(frames >= 1 && frames <= 512, "frame count outside 1..512");
            require(!std::filesystem::exists(out), "output exists; choose a new directory");

            id<MTLDevice> device = MTLCreateSystemDefaultDevice();
            require(device != nil, "no Metal device");

            anr::Runtime runtime(device, root);
            std::string error;
            require(runtime.initialize(error, reqW, reqH, anr::Encoding::PQ10, anr::PhaseMode::All), error);

            const unsigned w = runtime.width();
            const unsigned h = runtime.height();
            id<MTLTexture> color = make_texture(device, w, h, MTLPixelFormatRGB10A2Unorm);
            id<MTLTexture> depth = make_texture(device, w, h, MTLPixelFormatR32Float);
            id<MTLTexture> motion = make_texture(device, w, h, MTLPixelFormatRG16Float);
            id<MTLTexture> output = make_texture(device, w, h, MTLPixelFormatRGB10A2Unorm);
            id<MTLTexture> output2 = mode3x ? make_texture(device, w, h, MTLPixelFormatRGB10A2Unorm) : nil;

            NSData *frame0 = read_bin(inputs + "/frame0.bin");
            NSData *frame1 = read_bin(inputs + "/frame.bin");
            require(frame0.length == 204 && frame1.length == 204, "frame input must be 204 bytes");

            id<MTL4CommandQueue> queue = [device newMTL4CommandQueue];
            id<MTL4CommandAllocator> allocator = [device newCommandAllocator];
            id<MTL4CommandBuffer> command = [device newCommandBuffer];
            id<MTLSharedEvent> done = [device newSharedEvent];
            uint64_t token = 0;
            require(queue && allocator && command && done, "Metal 4 command allocation failed");

            NSError *counterError = nil;
            MTL4CounterHeapDescriptor *counterDesc = [MTL4CounterHeapDescriptor new];
            counterDesc.type = MTL4CounterHeapTypeTimestamp;
            counterDesc.count = 4;
            id<MTL4CounterHeap> counters = [device newCounterHeapWithDescriptor:counterDesc error:&counterError];
            require(counters != nil, "GPU timestamp counter unavailable");

            auto begin = [&] {
                [allocator reset];
                [command beginCommandBufferWithAllocator:allocator];
            };
            auto finish = [&] {
                [command endCommandBuffer];
                id<MTL4CommandBuffer> commands[] = {command};
                [queue commit:commands count:1];
                [queue signalEvent:done value:++token];
                require([done waitUntilSignaledValue:token timeoutMS:20000], "GPU wait timeout");
            };

            const char *coalesced = getenv("ANR_COALESCED");
            runtime.set_coalesced(!coalesced || strcmp(coalesced, "0"));

            if (prime) {
                upload_tex(color, read_bin(inputs + "/color0.bin"), 4);
                upload_tex(depth, read_bin(inputs + "/depth0.bin"), 4);
                upload_tex(motion, read_bin(inputs + "/motion0.bin"), 4);
                begin();
                require(runtime.encode_pair(command, color, depth, motion, frame0.bytes, frame0.length, error), error);
                finish();
                runtime.swap_history();
            }

            upload_tex(color, read_bin(inputs + "/color1.bin"), 4);
            upload_tex(depth, read_bin(inputs + "/depth1.bin"), 4);
            upload_tex(motion, read_bin(inputs + "/motion10.bin"), 4);

            std::vector<double> times, gpuTimes, pairTimes, phaseTimes, phase2Times;
            MTLTimestamp cpuStart = 0, gpuStart = 0, cpuEnd = 0, gpuEnd = 0;
            [device sampleTimestamps:&cpuStart gpuTimestamp:&gpuStart];

            for (int i = -8; i < frames; ++i) {
                begin();
                const auto start = std::chrono::steady_clock::now();
                [command writeTimestampIntoHeap:counters atIndex:0];
                require(runtime.encode_pair(command, color, depth, motion, frame1.bytes, frame1.length, error), error);
                [command writeTimestampIntoHeap:counters atIndex:1];
                if (mode3x) {
                    require(runtime.encode_interpolation(command, 1.0 / 3.0, output, error), error);
                    [command writeTimestampIntoHeap:counters atIndex:2];
                    require(runtime.encode_interpolation(command, 2.0 / 3.0, output2, error), error);
                    [command writeTimestampIntoHeap:counters atIndex:3];
                } else {
                    require(runtime.encode_interpolation(command, phase, output, error), error);
                    [command writeTimestampIntoHeap:counters atIndex:2];
                }
                finish();
                const auto end = std::chrono::steady_clock::now();
                const NSUInteger numStamps = mode3x ? 4 : 3;
                NSData *resolved = [counters resolveCounterRange:NSMakeRange(0, numStamps)];
                require(resolved.length == numStamps * sizeof(MTL4TimestampHeapEntry), "invalid GPU timestamp result");
                const auto *stamp = (const MTL4TimestampHeapEntry *)resolved.bytes;
                require(stamp[0].timestamp < stamp[1].timestamp && stamp[1].timestamp < stamp[2].timestamp,
                        "unordered GPU timestamps");
                if (i >= 0) {
                    times.push_back(std::chrono::duration<double, std::milli>(end - start).count());
                    pairTimes.push_back(double(stamp[1].timestamp - stamp[0].timestamp));
                    phaseTimes.push_back(double(stamp[2].timestamp - stamp[1].timestamp));
                    if (mode3x) {
                        phase2Times.push_back(double(stamp[3].timestamp - stamp[2].timestamp));
                        gpuTimes.push_back(double(stamp[3].timestamp - stamp[0].timestamp));
                    } else {
                        gpuTimes.push_back(double(stamp[2].timestamp - stamp[0].timestamp));
                    }
                }
            }

            [device sampleTimestamps:&cpuEnd gpuTimestamp:&gpuEnd];
            mach_timebase_info_data_t timebase = {};
            mach_timebase_info(&timebase);
            require(cpuEnd > cpuStart && gpuEnd > gpuStart, "timestamp calibration failed");
            const double gpuTickMs = double(cpuEnd - cpuStart) * timebase.numer / timebase.denom / double(gpuEnd - gpuStart) / 1e6;
            for (auto values : {&gpuTimes, &pairTimes, &phaseTimes, &phase2Times}) {
                for (double &value : *values) value *= gpuTickMs;
            }

            if (doProfile) {
                MTL4CounterHeapDescriptor *profDesc = [MTL4CounterHeapDescriptor new];
                profDesc.type = MTL4CounterHeapTypeTimestamp;
                profDesc.count = 256;
                id<MTL4CounterHeap> profHeap = [device newCounterHeapWithDescriptor:profDesc error:&counterError];
                require(profHeap != nil, "profile counter heap unavailable");

                std::vector<std::vector<double>> stageSamples;
                std::vector<anr::StageProfile> meta;
                constexpr int kProfileRuns = 16;
                for (int r = 0; r < kProfileRuns; ++r) {
                    std::vector<anr::StageProfile> pairStages, phaseStages;
                    begin();
                    require(runtime.profile_pair(command, profHeap, 0, color, depth, motion,
                                                 frame1.bytes, frame1.length, pairStages, error), error);
                    const unsigned phaseBase = (unsigned)pairStages.size() * 2;
                    require(runtime.profile_interpolation(command, profHeap, phaseBase, phase,
                                                          output, phaseStages, error), error);
                    finish();
                    const unsigned totalStages = (unsigned)(pairStages.size() + phaseStages.size());
                    NSData *res = [profHeap resolveCounterRange:NSMakeRange(0, totalStages * 2)];
                    const auto *st = (const MTL4TimestampHeapEntry *)res.bytes;
                    if (r == 0) {
                        meta = pairStages;
                        meta.insert(meta.end(), phaseStages.begin(), phaseStages.end());
                        stageSamples.resize(totalStages);
                    }
                    for (unsigned s = 0; s < totalStages; ++s) {
                        double dt = double(st[s * 2 + 1].timestamp - st[s * 2].timestamp) * gpuTickMs;
                        stageSamples[s].push_back(dt);
                    }
                }
                fprintf(stderr, "\n=== Per-stage GPU profile (%ux%u, %s %s, median of %d) ===\n",
                        w, h, runtime.release_name().c_str(), runtime.backend_name().c_str(), kProfileRuns);
                double sumMs = 0.0;
                std::map<std::string, double> byKernel;
                for (size_t s = 0; s < meta.size(); ++s) {
                    const double ms = median(stageSamples[s]);
                    sumMs += ms;
                    byKernel[meta[s].kernel] += ms;
                    fprintf(stderr, "  %2zu %-5s %-58s %6.3f ms\n",
                            s, meta[s].label.c_str(), meta[s].kernel.c_str(), ms);
                }
                fprintf(stderr, "  TOTAL summed stage time: %.3f ms\n", sumMs);
                std::vector<std::pair<double, std::string>> ranked;
                for (const auto &kv : byKernel) ranked.emplace_back(kv.second, kv.first);
                std::sort(ranked.rbegin(), ranked.rend());
                fprintf(stderr, "--- Top kernels by total GPU time ---\n");
                for (const auto &item : ranked) {
                    fprintf(stderr, "  %6.3f ms (%4.1f%%)  %s\n",
                            item.first, sumMs > 0 ? 100.0 * item.first / sumMs : 0.0, item.second.c_str());
                }
            }

            NSMutableData *generated = [NSMutableData dataWithLength:size_t(w) * h * 4];
            [output getBytes:generated.mutableBytes bytesPerRow:w * 4 fromRegion:MTLRegionMake2D(0, 0, w, h) mipmapLevel:0];

            NSMutableDictionary *report = [@{
                @"schema": @"anr_independent_replay_v1",
                @"release": [NSString stringWithUTF8String:runtime.release_name().c_str()],
                @"backend": [NSString stringWithUTF8String:runtime.backend_name().c_str()],
                @"device": device.name,
                @"width": @(w),
                @"height": @(h),
                @"frames": @(frames),
                @"phase": @(phase),
                @"mode_3x": @(mode3x),
                @"primed": @(prime),
                @"host_wall_median_ms": @(median(times)),
                @"output_sha256": sha256_ns(generated),
                @"testkit_executable_used": @NO,
                @"game_integration_verified": @NO,
                @"gpu_median_ms": @(median(gpuTimes)),
                @"pair_gpu_median_ms": @(median(pairTimes)),
                @"phase_gpu_median_ms": @(median(phaseTimes)),
                @"warmups": @8,
                @"gpu_tick_ns": @(gpuTickMs * 1e6),
                @"coalesced": @(!coalesced || strcmp(coalesced, "0")),
            } mutableCopy];
            if (mode3x) {
                report[@"phase2_gpu_median_ms"] = @(median(phase2Times));
            }

            if (argc > 6 && strlen(argv[6]) > 0 && strcmp(argv[6], "-") != 0) {
                NSData *reference = read_bin(argv[6]);
                require(reference.length == generated.length, "reference size mismatch");
                const auto *a = (const uint32_t *)generated.bytes;
                const auto *b = (const uint32_t *)reference.bytes;
                double squared = 0;
                uint64_t pixels = 0, components = 0;
                unsigned maxDifference = 0;
                for (size_t i = 0; i < size_t(w) * h; ++i) {
                    if (a[i] != b[i]) ++pixels;
                    for (unsigned shift : {0u, 10u, 20u}) {
                        const int diff = int((a[i] >> shift) & 1023) - int((b[i] >> shift) & 1023);
                        squared += double(diff) * diff;
                        if (diff) ++components;
                        maxDifference = std::max(maxDifference, unsigned(std::abs(diff)));
                    }
                }
                const double rmse = std::sqrt(squared / (double(w) * h * 3));
                report[@"differing_pixels"] = @(pixels);
                report[@"differing_components"] = @(components);
                report[@"max_code_difference"] = @(maxDifference);
                report[@"psnr_rgb10_db"] = rmse > 0 ? @(20 * std::log10(1023 / rmse)) : @"infinity";
            }

            require(std::filesystem::create_directories(out), "cannot create output");
            NSString *outputPath = [NSString stringWithUTF8String:out.c_str()];
            require([generated writeToFile:[outputPath stringByAppendingPathComponent:@"out.bin"] atomically:YES],
                    "cannot save output");
            NSData *json = [NSJSONSerialization dataWithJSONObject:report options:NSJSONWritingPrettyPrinted error:nil];
            require([json writeToFile:[outputPath stringByAppendingPathComponent:@"run.json"] atomically:YES],
                    "cannot save report");
            fwrite(json.bytes, 1, json.length, stdout);
            puts("");
            return 0;
        } catch (const std::exception &e) {
            fprintf(stderr, "ANR replay: %s\n", e.what());
            return 1;
        }
    }
}
