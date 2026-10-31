// SPDX-License-Identifier: MIT
// Specialized Apple7+ neural-pass benchmark. No DLL/PRE/POST shaders needed.
// --compare-root alternates old/new PSOs in ONE process on identical resources.
// Timings are isolated neural passes, not full-frame/game latency.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "../../src/fsr411_model.h"
#include "../../src/fsr411_simd_weights.h"
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <vector>

static void require(bool ok, const char *message) {
    if (!ok) throw std::runtime_error(message);
}
static unsigned end(const fsr411::Tensor &t) {
    return t.channels ? fsr411::tensor_address(t, t.width + 1, t.height + 1, t.channels - 1) + 64 : 0;
}
static double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    const size_t n = values.size();
    return n % 2 ? values[n/2] : (values[n/2-1]+values[n/2])*.5;
}
static NSArray *numbers(const std::vector<double> &values) {
    NSMutableArray *result = [NSMutableArray new];
    for (double value : values) [result addObject:@(value)];
    return result;
}
static id<MTLLibrary> library(id<MTLDevice> dev, NSString *root, unsigned mb16, unsigned mb32, unsigned mb64) {
    NSError *error = nil;
    NSString *source = [NSString stringWithContentsOfFile:[root stringByAppendingPathComponent:@"src/fsr411_simd.metal"] encoding:NSUTF8StringEncoding error:&error];
    NSString *header = [NSString stringWithContentsOfFile:[root stringByAppendingPathComponent:@"src/fsr411_model.h"] encoding:NSUTF8StringEncoding error:&error];
    require(source && header, "missing shader source/header");
    source = [source stringByReplacingOccurrencesOfString:@"#include \"fsr411_model.h\"" withString:header];
    auto options = [MTLCompileOptions new]; options.languageVersion = MTLLanguageVersion3_0;
    options.preprocessorMacros = @{@"FSR411_SPECIALIZE":@1, @"FSR411_SIMD_MB16":@(mb16),
        @"FSR411_SIMD_MB32":@(mb32), @"FSR411_SIMD_MB64":@(mb64)};
    auto lib = [dev newLibraryWithSource:source options:options error:&error];
    if (!lib) throw std::runtime_error(error.description.UTF8String);
    return lib;
}
static id<MTLComputePipelineState> pipeline(id<MTLDevice> dev, id<MTLLibrary> lib,
                                          NSString *name, MTLFunctionConstantValues *values, bool legacy) {
    NSError *error = nil;
    auto fn = [lib newFunctionWithName:name constantValues:values error:&error];
    if (!fn) throw std::runtime_error(error.description.UTF8String);
    auto desc = [MTLComputePipelineDescriptor new]; desc.computeFunction = fn;
    desc.maxTotalThreadsPerThreadgroup = 128; desc.threadGroupSizeIsMultipleOfThreadExecutionWidth = YES;
    auto pso = legacy ? [dev newComputePipelineStateWithFunction:fn error:&error]
        : [dev newComputePipelineStateWithDescriptor:desc options:MTLPipelineOptionNone reflection:nil error:&error];
    if (!pso) throw std::runtime_error(error.description.UTF8String);
    require(pso.maxTotalThreadsPerThreadgroup >= 128 && pso.threadExecutionWidth == 32, "incompatible SIMD32 pipeline");
    require(pso.staticThreadgroupMemoryLength <= dev.maxThreadgroupMemoryLength, "threadgroup memory overflow");
    return pso;
}

int main(int argc, char **argv) {
    @autoreleasepool { try {
        require(argc >= 3, "usage: fsr411_simd_bench ROOT NATIVE_DIR [--compare-root OLD_ROOT] [--winograd] [--compact-ds] [--legacy-pipeline] [--mb16 N] [--mb32 N] [--mb64 N] [--rounds N]");
        bool legacy = false, winograd = false, compactDs = false; unsigned rounds = 12;
        unsigned mb16 = 4, mb32 = 4, mb64 = 2; NSString *compareRoot = nil;
        for (int i = 3; i < argc; ++i) {
            if (!std::strcmp(argv[i], "--legacy-pipeline")) legacy = true;
            else if (!std::strcmp(argv[i], "--winograd")) winograd = true;
            else if (!std::strcmp(argv[i], "--compact-ds")) compactDs = true;
            else if (!std::strcmp(argv[i], "--compare-root")) {
                require(i+1 < argc, "missing comparison root"); compareRoot = @(argv[++i]);
            } else {
                require(i + 1 < argc, "missing argument value");
                const std::string flag = argv[i++];
                const unsigned value = unsigned(std::stoul(argv[i]));
                require(value > 0 && value <= 1000, "invalid argument value");
                if (flag == "--mb16") mb16 = value;
                else if (flag == "--mb32") mb32 = value;
                else if (flag == "--mb64") mb64 = value;
                else if (flag == "--rounds") rounds = value;
                else throw std::runtime_error("unknown flag: " + flag);
            }
        }
        require(mb16 <= 16 && mb32 <= 8 && mb64 <= 8, "matrix block count too large");
        require(!winograd || (mb16 == 4 && mb32 == 4), "Winograd uses the unchanged 32x4 tile");
        NSString *root = @(argv[1]), *dir = @(argv[2]); NSError *error = nil;
        auto dev = MTLCreateSystemDefaultDevice(); auto queue = [dev newCommandQueue];
        require(dev && queue && [dev supportsFamily:MTLGPUFamilyApple7], "Apple7+ Metal GPU required");
        const auto thermalStart = NSProcessInfo.processInfo.thermalState;
        const bool lowPower = NSProcessInfo.processInfo.lowPowerModeEnabled;
        NSData *data = [NSData dataWithContentsOfFile:[dir stringByAppendingPathComponent:@"fsr411_native_manifest.json"]];
        NSDictionary *manifest = data ? [NSJSONSerialization JSONObjectWithData:data options:0 error:&error] : nil;
        NSData *weights = [NSData dataWithContentsOfFile:[dir stringByAppendingPathComponent:@"fsr411_native_weights.bin"]];
        NSData *params = [NSData dataWithContentsOfFile:[dir stringByAppendingPathComponent:@"fsr411_native_params.bin"]];
        require(manifest && weights.length && params.length, "missing native model");
        require([manifest[@"parameter_size"] unsignedIntValue] == sizeof(fsr411::PassParams), "parameter ABI mismatch");
        auto wb = [dev newBufferWithBytes:weights.bytes length:weights.length options:MTLResourceStorageModeShared];
        std::vector<_Float16> widened(weights.length);
        for (size_t i = 0; i < widened.size(); ++i) widened[i] = (_Float16)((const int8_t *)weights.bytes)[i];
        auto wh = [dev newBufferWithBytes:widened.data() length:widened.size()*sizeof(_Float16) options:MTLResourceStorageModeShared];
        auto lib = library(dev,root,mb16,mb32,mb64);
        // The saved baseline uses its ORIGINAL dispatch shapes and pipeline
        // limit, independent of current benchmark-only MB overrides.
        auto beforeLib = compareRoot ? library(dev,compareRoot,4,4,2) : nil;
        NSMutableDictionary *passes = [manifest[@"passes"] mutableCopy];
        passes[@"post_cnb"] = manifest[@"auxiliary_passes"][@"post_cnb"];
        unsigned scratchBytes = 0; std::vector<_Float16> transformed;
        NSMutableSet *winogradPasses = [NSMutableSet new];
        for (NSString *key in passes) {
            fsr411::PassParams p; const size_t off = [passes[key][@"parameter_offset"] unsignedIntValue];
            require(off <= params.length && sizeof(p) <= params.length-off, "parameter offset overflow");
            std::memcpy(&p, (const uint8_t *)params.bytes+off, sizeof(p));
            scratchBytes = std::max({scratchBytes,end(p.input),end(p.output),end(p.skip)});
            if (winograd && fsr411::prepare_simd_winograd_weights((const uint8_t *)weights.bytes,weights.length,p,transformed))
                [winogradPasses addObject:key];
        }
        auto wg = transformed.empty() ? nil : [dev newBufferWithBytes:transformed.data() length:transformed.size()*sizeof(_Float16) options:MTLResourceStorageModeShared];
        require(transformed.empty() || wg, "Winograd weight allocation failed");
        const unsigned postOffset = (scratchBytes+255u)&~255u;
        scratchBytes = postOffset+[passes[@"post_cnb"][@"scratch_append_bytes"] unsignedIntValue]+256;
        auto scratch = [dev newBufferWithLength:scratchBytes options:MTLResourceStorageModePrivate];
        require(wb && wh && scratch, "buffer allocation failed");
        auto clear = [queue commandBuffer]; auto blit = [clear blitCommandEncoder];
        [blit fillBuffer:scratch range:NSMakeRange(0,scratchBytes) value:0x31];
        [blit endEncoding]; [clear commit]; [clear waitUntilCompleted];
        require(clear.status == MTLCommandBufferStatusCompleted, "initialization failed");
        NSMutableArray *results = [NSMutableArray new];
        constexpr unsigned batch = 8, warmups = 3;
        for (NSString *key in @[@"1",@"2",@"3",@"4",@"5",@"6",@"7",@"8",@"9",@"10",@"11",@"12",@"post_cnb"]) {
            NSDictionary *entry = passes[key]; if (!entry) continue;
            fsr411::PassParams p;
            std::memcpy(&p,(const uint8_t *)params.bytes+[entry[@"parameter_offset"] unsignedIntValue],sizeof(p));
            const bool post = [key isEqual:@"post_cnb"], ct = [entry[@"kind"] isEqual:@"ct"], ds = [entry[@"kind"] isEqual:@"ds"];
            const bool useWinograd = [winogradPasses containsObject:key];
            const auto metadata = fsr411::specialize_metadata(p,post);
            auto values = [MTLFunctionConstantValues new];
            [values setConstantValues:metadata.values type:MTLDataTypeUInt4 withRange:NSMakeRange(0,fsr411::kPassSpecializationCount)];
            NSString *name = entry[@"simd"][@"function"];
            if (useWinograd) name = [name stringByAppendingString:@"_winograd"];
            if (ds && compactDs) name = [name stringByAppendingString:@"_compact"];
            auto current = pipeline(dev,lib,name,values,legacy);
            auto before = beforeLib ? pipeline(dev,beforeLib,entry[@"simd"][@"function"],values,legacy) : nil;
            if (post) p.output.offset += postOffset;
            const unsigned nf = p.input.channels;
            const unsigned mb = ds ? (nf == 16 ? mb32 : mb64) : nf == 16 ? mb16 : nf == 32 ? mb32 : mb64;
            const unsigned w = ct ? p.input.width : p.output.width, h = ct ? p.input.height : p.output.height;
            const unsigned beforeX = [entry[@"simd"][@"tile"][0] unsignedIntValue];
            const unsigned beforeY = [entry[@"simd"][@"tile"][1] unsignedIntValue];
            std::vector<double> times[2]; // current, previous; pairing is kept.
            for (unsigned r = 0; r < rounds+warmups; ++r) {
                const unsigned variants = before ? 2 : 1;
                for (unsigned j = 0; j < variants; ++j) {
                    const unsigned v = before ? ((r&1u)^j) : 0;
                    auto cb = [queue commandBuffer]; auto enc = [cb computeCommandEncoder];
                    [enc setComputePipelineState:v ? before : current];
                    [enc setBuffer:scratch offset:0 atIndex:0]; [enc setBuffer:wb offset:0 atIndex:1];
                    [enc setBuffer:wh offset:0 atIndex:3]; [enc setBytes:&p length:sizeof(p) atIndex:2];
                    if (!v && useWinograd) [enc setBuffer:wg offset:0 atIndex:4];
                    const unsigned tx = v ? beforeX : 8*mb, ty = v ? beforeY : 4;
                    for (unsigned k = 0; k < batch; ++k)
                        [enc dispatchThreadgroups:MTLSizeMake((w+tx-1)/tx,(h+ty-1)/ty,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];
                    [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
                    if (cb.status != MTLCommandBufferStatusCompleted) throw std::runtime_error(cb.error.description.UTF8String);
                    if (r >= warmups) times[v].push_back((cb.GPUEndTime-cb.GPUStartTime)*1000./batch);
                }
            }
            NSMutableDictionary *row = [@{@"pass":key,@"kind":entry[@"kind"],@"channels":@(nf),@"tile":@[@(8*mb),@4],
                @"median_ms":@(median(times[0])),@"min_ms":@(*std::min_element(times[0].begin(),times[0].end())),
                @"samples_ms":numbers(times[0]),@"threadgroup_bytes":@(current.staticThreadgroupMemoryLength),
                @"pipeline_max_threads":@(current.maxTotalThreadsPerThreadgroup),@"winograd":@(useWinograd),
                @"thermal_state":@(NSProcessInfo.processInfo.thermalState)} mutableCopy];
            if (before) {
                std::vector<double> ratios;
                for (size_t i = 0; i < times[0].size(); ++i) ratios.push_back(times[0][i]/times[1][i]);
                row[@"before_median_ms"] = @(median(times[1]));
                row[@"before_min_ms"] = @(*std::min_element(times[1].begin(),times[1].end()));
                row[@"before_samples_ms"] = numbers(times[1]);
                row[@"before_threadgroup_bytes"] = @(before.staticThreadgroupMemoryLength);
                row[@"paired_ratio_median"] = @(median(ratios));
                row[@"paired_change_percent"] = @((median(ratios)-1)*100.);
            }
            [results addObject:row];
        }
        NSDictionary *report = @{@"device":dev.name,@"model":manifest[@"fsr_version"]?:@"4.1.1",
            @"profile":manifest[@"profile"],@"tier":manifest[@"tier"],@"legacy_pipeline":@(legacy),
            @"winograd":@(winograd),@"compact_ds":@(compactDs),@"root":root,@"compare_root":compareRoot?:[NSNull null],
            @"scope":@"isolated neural passes; 8 dispatches per GPU command buffer; compilation excluded",
            @"comparison_order":beforeLib?@"alternating current/previous and previous/current each pair":@"single variant",
            @"low_power_mode":@(lowPower),@"thermal_state_start":@(thermalStart),
            @"thermal_state_end":@(NSProcessInfo.processInfo.thermalState),@"warmup_pairs":@(warmups),
            @"rounds":@(rounds),@"passes":results};
        auto json = [NSJSONSerialization dataWithJSONObject:report options:NSJSONWritingPrettyPrinted error:&error];
        require(json != nil, "JSON serialization failed");
        std::fwrite(json.bytes,1,json.length,stdout); std::puts(""); return 0;
    } catch (const std::exception &e) { std::fprintf(stderr,"simd benchmark: %s\n",e.what()); return 1; } }
}
