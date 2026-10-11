// SPDX-License-Identifier: MIT
// FSR 4.1.1 recording runner. Keep ARC ownership separate from the legacy CLI.
#include "fsr411_runtime.h"
#include "pngio.h"
#include <mach-o/dyld.h>
#include <cerrno>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <stdexcept>

namespace {
using fsr411::ns;
using fsr411::utf8;

[[noreturn]] void fail(const std::string &message) { throw std::runtime_error(message); }

// Apple GPUs currently sample at encoder boundaries, not individual classic
// compute dispatches. This mode preserves command-buffer ordering but inserts
// separate encoders, so its times diagnose stages rather than replace the
// uninstrumented, fused --bench result.
class CounterStageProfiler final : public fsr411::StageProfiler {
public:
    struct Timing { std::string name; double milliseconds; };
    std::vector<Timing> timings;
    CounterStageProfiler(id<MTLDevice> device, id<MTLCommandBuffer> commandBuffer)
        : device_(device), commandBuffer_(commandBuffer) {
        dispatchSamples_=[device supportsCounterSampling:MTLCounterSamplingPointAtDispatchBoundary];
        if(!dispatchSamples_ && ![device supportsCounterSampling:MTLCounterSamplingPointAtStageBoundary])
            fail("--profile-stages requires GPU timestamp counter sampling");
        id<MTLCounterSet> timestamps=nil;
        for(id<MTLCounterSet> counter in device.counterSets)
            if([counter.name isEqual:MTLCommonCounterSetTimestamp]) { timestamps=counter;break; }
        if(!timestamps) fail("--profile-stages: no timestamp counter set");
        auto descriptor=[MTLCounterSampleBufferDescriptor new];
        descriptor.counterSet=timestamps;descriptor.sampleCount=kCapacity;
        descriptor.storageMode=MTLStorageModeShared;descriptor.label=@"FSR 4.1.1 stage timestamps";
        NSError *error=nil;
        samples_=[device newCounterSampleBufferWithDescriptor:descriptor error:&error];
        if(!samples_) fail("cannot allocate stage timestamp counters: "+fsr411::metal_error(error));
        [device_ sampleTimestamps:&cpuBefore_ gpuTimestamp:&gpuBefore_];
    }
    const char *method() const { return dispatchSamples_?"dispatch_boundary":"separate_encoder_boundaries"; }
    void begin_stage(id __strong &encoder,const std::string &name) override {
        if(names_.size()*2u+1u>=kCapacity)
            [NSException raise:@"FSR411Profiling" format:@"stage timestamp capacity exceeded"];
        const NSUInteger index=names_.size()*2u;
        names_.push_back(name);
        if(dispatchSamples_) {
            [(id<MTLComputeCommandEncoder>)encoder sampleCountersInBuffer:samples_ atSampleIndex:index withBarrier:YES];
        } else {
            [encoder endEncoding];
            auto descriptor=[MTLComputePassDescriptor computePassDescriptor];
            descriptor.dispatchType=MTLDispatchTypeSerial;
            auto attachment=descriptor.sampleBufferAttachments[0];
            attachment.sampleBuffer=samples_;attachment.startOfEncoderSampleIndex=index;
            attachment.endOfEncoderSampleIndex=index+1u;
            encoder=[commandBuffer_ computeCommandEncoderWithDescriptor:descriptor];
            if(!encoder) [NSException raise:@"FSR411Profiling" format:@"cannot create stage sampling encoder"];
            [encoder setLabel:ns("FSR 4.1.1 "+name)];
        }
    }
    void end_stage(id encoder) override {
        if(dispatchSamples_)
            [(id<MTLComputeCommandEncoder>)encoder sampleCountersInBuffer:samples_
                atSampleIndex:names_.size()*2u-1u withBarrier:YES];
    }
    NSDictionary *resolve(double commandBufferMs) {
        MTLTimestamp cpuAfter=0,gpuAfter=0;
        [device_ sampleTimestamps:&cpuAfter gpuTimestamp:&gpuAfter];
        if(cpuAfter<=cpuBefore_ || gpuAfter<=gpuBefore_)
            fail("nonmonotonic CPU/GPU timestamp calibration");
        // sampleTimestamps reports the CPU clock in nanoseconds, including on
        // Apple Silicon. Applying mach_timebase_info a second time would scale
        // the result by 41.6667 on M5. Calibrate GPU ticks against that ns clock.
        const double tickMs=(double(cpuAfter-cpuBefore_)/double(gpuAfter-gpuBefore_))
            /1.e6;
        NSData *resolved=[samples_ resolveCounterRange:NSMakeRange(0,names_.size()*2u)];
        if(!resolved || resolved.length!=names_.size()*2u*sizeof(MTLCounterResultTimestamp))
            fail("cannot resolve GPU stage timestamps");
        auto values=(const MTLCounterResultTimestamp *)resolved.bytes;
        NSMutableArray *stages=[NSMutableArray array];
        std::vector<std::pair<uint64_t,uint64_t>> intervals;
        double sum=0;
        for(size_t i=0;i<names_.size();++i) {
            const uint64_t start=values[2u*i].timestamp,end=values[2u*i+1u].timestamp;
            if(start==MTLCounterErrorValue || end==MTLCounterErrorValue || end<start)
                fail("invalid GPU timestamp for "+names_[i]);
            const double milliseconds=double(end-start)*tickMs;
            intervals.emplace_back(start,end);
            timings.push_back({names_[i],milliseconds});sum+=milliseconds;
            [stages addObject:@{@"stage":ns(names_[i]),@"gpu_ms":@(milliseconds),
                @"gpu_start_tick":@(start),@"gpu_end_tick":@(end)}];
        }
        // Different encoders can overlap independent clears/copies. Measure
        // their interval union as well as the sum; only the nonoverlapping
        // union and first-to-last span must fit inside the full command buffer.
        std::sort(intervals.begin(),intervals.end());
        uint64_t covered=0,first=0,last=0;
        if(!intervals.empty()) {
            first=intervals[0].first;uint64_t start=first;last=intervals[0].second;
            for(size_t i=1;i<intervals.size();++i) {
                if(intervals[i].first>last) {covered+=last-start;start=intervals[i].first;}
                last=std::max(last,intervals[i].second);
            }
            covered+=last-start;
        }
        const double span=double(last-first)*tickMs,occupied=double(covered)*tickMs;
        const double tolerance=std::max(0.1,commandBufferMs*.1);
        if(!std::isfinite(span) || span<0 || span>commandBufferMs+tolerance
            || occupied>commandBufferMs+tolerance)
            fail("stage timestamp calibration exceeds the command-buffer GPU duration");
        return @{@"method":ns(method()),@"stages":stages,@"stage_sum_ms":@(sum),
            @"sampled_span_ms":@(span),@"nonoverlapping_stage_ms":@(occupied),
            @"inter_stage_gap_ms":@(span-occupied),@"overlapped_stage_ms":@(sum-occupied),
            @"command_buffer_minus_sampled_span_ms":@(commandBufferMs-span),
            @"gpu_tick_ms":@(tickMs),@"calibration_cpu_ticks":@(cpuAfter-cpuBefore_),
            @"calibration_gpu_ticks":@(gpuAfter-gpuBefore_)};
    }
private:
    static constexpr NSUInteger kCapacity=256;
    id<MTLDevice> device_;
    id<MTLCommandBuffer> commandBuffer_;
    id<MTLCounterSampleBuffer> samples_;
    MTLTimestamp cpuBefore_=0,gpuBefore_=0;
    bool dispatchSamples_=false;
    std::vector<std::string> names_;
};

std::string absolute_path(const std::string &path) {
    NSString *value = ns(path);
    if (!value.isAbsolutePath)
        value = [NSFileManager.defaultManager.currentDirectoryPath stringByAppendingPathComponent:value];
    return utf8(value.stringByStandardizingPath.stringByResolvingSymlinksInPath);
}

std::string default_root() {
    const char *configured = std::getenv("FSR4_ROOT");
    if (configured && *configured) return absolute_path(configured);
    uint32_t size = 0;
    _NSGetExecutablePath(nullptr, &size);
    std::vector<char> path(size);
    if (!size || _NSGetExecutablePath(path.data(), &size)) fail("cannot resolve executable path");
    NSString *root = ns(absolute_path(path.data()));
    for (int i = 0; i < 3; ++i) root = root.stringByDeletingLastPathComponent;
    return utf8(root);
}

NSData *read_file(const std::string &path) {
    NSError *error = nil;
    NSData *data = [NSData dataWithContentsOfFile:ns(path) options:0 error:&error];
    if (!data) fail("cannot read " + path + ": " + fsr411::metal_error(error));
    return data;
}

void write_file(const std::string &path, const void *bytes, size_t length) {
    NSError *error = nil;
    NSData *data = [NSData dataWithBytes:bytes length:length];
    if (!data || ![data writeToFile:ns(path) options:NSDataWritingAtomic error:&error])
        fail("cannot write " + path + ": " + fsr411::metal_error(error));
}

void write_json(const std::string &path, id value) {
    NSError *error = nil;
    NSData *data = [NSJSONSerialization dataWithJSONObject:value
        options:NSJSONWritingPrettyPrinted | NSJSONWritingSortedKeys error:&error];
    if (!data) fail("cannot encode JSON for " + path + ": " + fsr411::metal_error(error));
    write_file(path, data.bytes, data.length);
}

double number(NSDictionary *dict, NSString *key, double fallback) {
    id value = dict[key];
    if (!value) return fallback;
    if (![value isKindOfClass:NSNumber.class] || !std::isfinite([value doubleValue]))
        fail("meta.json: " + utf8(key) + " must be a finite number");
    return [value doubleValue];
}

uint32_t dimension(NSDictionary *dict, NSString *key, uint32_t fallback) {
    double value = number(dict, key, fallback);
    if (value < 1 || value > UINT32_MAX || value != std::floor(value))
        fail("meta.json: " + utf8(key) + " must be a positive integer");
    return uint32_t(value);
}

int flag(NSDictionary *dict, NSString *key, int fallback) {
    double value = number(dict, key, fallback);
    if (value != 0 && value != 1) fail("meta.json: " + utf8(key) + " must be true or false");
    return int(value);
}

std::string string(NSDictionary *dict, NSString *key, const char *fallback) {
    id value = dict[key];
    if (!value) return fallback;
    if (![value isKindOfClass:NSString.class]) fail("meta.json: " + utf8(key) + " must be a string");
    return utf8(value);
}

void pair(NSDictionary *dict, NSString *key, float &x, float &y) {
    id value = dict[key];
    if (!value) return;
    if (![value isKindOfClass:NSArray.class] || [value count] != 2)
        fail("meta.json: " + utf8(key) + " must contain two numbers");
    for (int i = 0; i < 2; ++i)
        if (![value[i] isKindOfClass:NSNumber.class] || !std::isfinite([value[i] doubleValue]))
            fail("meta.json: " + utf8(key) + " must contain finite numbers");
    x = [value[0] floatValue]; y = [value[1] floatValue];
}

int positive_integer(const char *text, const std::string &name) {
    errno = 0; char *end = nullptr;
    long value = std::strtol(text, &end, 10);
    if (errno || !*text || *end || value < 1 || value > INT_MAX)
        fail(name + " requires a positive integer");
    return int(value);
}

fsr411::Backend backend(const std::string &name) {
    if (name == "auto") return fsr411::Backend::Auto;
    if (name == "reference" || name == "int8") return fsr411::Backend::Reference;
    if (name == "scalar") return fsr411::Backend::Scalar;
    if (name == "native" || name == "i8mpp") return fsr411::Backend::Native;
    fail("FSR 4.1.1 backend must be auto, reference, scalar, or i8mpp (no FP16 4.1.1 model)");
}

bool capture_stage(const std::string &name) {
    if (name == "pre" || name == "post") return true;
    for (int pass=0; pass<=12; ++pass)
        if (name == "pad"+std::to_string(pass)
            || (pass && name == "pass"+std::to_string(pass))) return true;
    return false;
}

void print_backend_info(id<MTLDevice> device, fsr411::Backend requested, const std::string &profile) {
    bool native = false;
    if (@available(macOS 26.0, *))
        native = [device supportsFamily:static_cast<MTLGPUFamily>(1010)]
            && device.maxThreadgroupMemoryLength >= 20480
            && device.maxThreadsPerThreadgroup.width >= 128;
    if (requested == fsr411::Backend::Native && !native)
        fail("native FSR 4.1.1 requires Apple GPU family 10 and macOS 26");
    const bool selected = native && (requested == fsr411::Backend::Auto || requested == fsr411::Backend::Native);
    NSString *selectedBackend = selected ? @"int8-mpp"
        : requested == fsr411::Backend::Reference ? @"int8-reference" : @"int8-scalar";
    NSString *reason = selected ? @"native INT8 tensor operations available; pipelines validated at encode"
        : requested == fsr411::Backend::Reference ? @"generic integer Metal implementation of the FSR 4.1.1 model"
        : @"portable scalar integer Metal implementation of the FSR 4.1.1 model";
    NSDictionary *info = @{@"fsr_version": @"4.1.1", @"device": device.name,
        @"profile":ns(profile),
        @"backend": selectedBackend, @"selection_only": @YES, @"reason": reason};
    NSData *data = [NSJSONSerialization dataWithJSONObject:info options:NSJSONWritingSortedKeys error:nil];
    if (!data || std::fwrite(data.bytes, 1, data.length, stdout) != data.length || std::putchar('\n') == EOF)
        fail("cannot write backend information");
}

id<MTLTexture> texture(id<MTLDevice> device, uint32_t w, uint32_t h, MTLPixelFormat format) {
    MTLTextureDescriptor *d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format
        width:w height:h mipmapped:NO];
    d.storageMode = MTLStorageModeShared;
    d.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
    id<MTLTexture> result = [device newTextureWithDescriptor:d];
    if (!result) fail("cannot allocate " + std::to_string(w) + "x" + std::to_string(h) + " texture");
    return result;
}

id<MTLTexture> input_texture(id<MTLDevice> device, const std::string &path,
                            uint32_t w, uint32_t h, MTLPixelFormat format, size_t pixelBytes) {
    if (size_t(w) > SIZE_MAX / pixelBytes / h) fail("input texture byte count overflow");
    const size_t bytes = size_t(w) * h * pixelBytes;
    NSData *data = read_file(path);
    if (data.length != bytes)
        fail(path + ": expected " + std::to_string(bytes) + " bytes, got " + std::to_string(data.length));
    auto result = texture(device, w, h, format);
    [result replaceRegion:MTLRegionMake2D(0,0,w,h) mipmapLevel:0
        withBytes:data.bytes bytesPerRow:size_t(w)*pixelBytes];
    return result;
}

void complete(id<MTLCommandBuffer> cb) {
    if (!cb) fail("cannot allocate Metal command buffer");
    [cb commit]; [cb waitUntilCompleted];
    if (cb.status != MTLCommandBufferStatusCompleted)
        fail("GPU command failed: " + fsr411::metal_error(cb.error));
}

std::vector<uint8_t> read_texture(id<MTLDevice> device, id<MTLCommandQueue> queue,
                                  id<MTLTexture> source, size_t pixelBytes) {
    if (!source) fail("missing output texture");
    const size_t row = source.width * pixelBytes, pitch = (row + 255u) & ~size_t(255u);
    const size_t bytes = pitch * source.height;
    auto buffer = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    if (!buffer) fail("cannot allocate texture readback buffer");
    auto cb = [queue commandBuffer]; auto blit = [cb blitCommandEncoder];
    if (!blit) fail("cannot allocate texture readback encoder");
    [blit copyFromTexture:source sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0,0,0)
        sourceSize:MTLSizeMake(source.width,source.height,1) toBuffer:buffer destinationOffset:0
        destinationBytesPerRow:pitch destinationBytesPerImage:bytes];
    [blit endEncoding]; complete(cb);
    std::vector<uint8_t> result(row * source.height);
    for (size_t y = 0; y < source.height; ++y)
        std::memcpy(result.data()+y*row, (const uint8_t *)buffer.contents+y*pitch, row);
    return result;
}

void dump_buffer(const std::string &path, id<MTLDevice> device,
                 id<MTLCommandQueue> queue, id<MTLBuffer> source) {
    if (!source) fail("missing diagnostic buffer");
    if (source.storageMode == MTLStorageModeShared) {
        write_file(path, source.contents, source.length); return;
    }
    auto staging = [device newBufferWithLength:source.length options:MTLResourceStorageModeShared];
    if (!staging) fail("cannot allocate scratch readback buffer");
    auto cb = [queue commandBuffer]; auto blit = [cb blitCommandEncoder];
    if (!blit) fail("cannot allocate scratch readback encoder");
    [blit copyFromBuffer:source sourceOffset:0 toBuffer:staging destinationOffset:0 size:source.length];
    [blit endEncoding]; complete(cb);
    write_file(path, staging.contents, staging.length);
}

std::string frame_prefix(const std::string &directory, size_t frame) {
    char name[48]; std::snprintf(name, sizeof name, "/f%02zu_", frame);
    return directory + name;
}

void write_preview(const std::string &path, const std::vector<uint8_t> &raw,
                   uint32_t w, uint32_t h, int colorspace) {
    std::vector<uint8_t> pixels(size_t(w)*h*4);
    for (size_t i=0; i<pixels.size(); ++i) {
        if (i%4 == 3) { pixels[i]=255; continue; }
        _Float16 half; std::memcpy(&half,raw.data()+i*2,2); float value=float(half);
        if (!std::isfinite(value)) fail("output contains nonfinite colour values");
        if (colorspace == 0)
            value = value <= 0 ? 0 : (value <= .0031308f ? 12.92f*value : 1.055f*std::pow(value,1.f/2.4f)-.055f);
        pixels[i] = uint8_t(std::clamp(value,0.f,1.f)*255.f+.5f);
    }
    if (!png_write(path.c_str(),pixels.data(),int(w),int(h))) fail("cannot write PNG " + path);
}

void usage() {
    std::fprintf(stderr,
        "usage: fsr4mac --fsr-version 4.1.1 <testcase_dir> <out_dir>\n"
        "       [--auto|--reference|--scalar|--i8mpp] [--profile exact|exact-fast|lossy] [--dump-passes|--dump-state]\n"
        "       [--capture-stages pre,pass1,...,post] [--bench N] [--profile-stages]\n"
        "       [--builddir dir] [--root dir] [--tier 1080|2160|4320] [--optimize 0|1] [--no-compile]\n"
        "       fsr4mac --fsr-version 4.1.1 --backend-info [auto|reference|scalar|i8mpp]\n"
        "--bench runs one warmup sequence and N measured sequences; only run.json is written.\n"
        "--profile-stages adds diagnostic GPU counters; on Apple GPUs it separates compute encoders.\n"
        "Use an ordinary --bench run for fused production timing; stage counters add overhead.\n"
        "FSR 4.1.1 and the lossy profile are the defaults; use --profile exact for exact-reference validation.\n"
        "exact-fast keeps Exact's network but uses approximate division and exp/log in PRE/POST.\n"
        "Use --fsr-version 4.0.2 to select the legacy model and its FP16 backend.\n"
        "GPU timings cover the full frame command buffer; diagnostic copies are labelled separately.\n");
}
} // namespace

int fsr411_main(int argc, char **argv) {
    @autoreleasepool {
    try {
        fsr411::Options options;
        const char *profileEnv=std::getenv("FSR4_PROFILE");
        std::string profile=profileEnv && *profileEnv?profileEnv:"lossy";
        if(const char *opt=std::getenv("FSR4_OPTIMIZE"); opt && *opt) {
            if(std::strcmp(opt,"0") && std::strcmp(opt,"1")) fail("FSR4_OPTIMIZE must be 0 or 1");
            options.optimize=opt[0]=='1';
        }
        std::string root = default_root(), buildDir = "build";
        std::vector<std::string> positional;
        int bench = 0; bool dumpState = false, info = false, profileStages = false;
        for (int i=1;i<argc;++i) {
            const std::string arg=argv[i];
            auto value = [&]() -> const char * {
                if (++i >= argc) fail(arg + " requires a value"); return argv[i];
            };
            if (arg=="--help" || arg=="-h") { usage(); return 0; }
            else if(arg=="--auto") options.backend=fsr411::Backend::Auto;
            else if(arg=="--reference" || arg=="--int8") options.backend=fsr411::Backend::Reference;
            else if(arg=="--scalar") options.backend=fsr411::Backend::Scalar;
            else if(arg=="--i8mpp" || arg=="--native") options.backend=fsr411::Backend::Native;
            else if(arg=="--fp16" || arg=="--mpp" || arg=="--cnb")
                fail("FSR 4.1.1 has no FP16 backend; select --reference, --scalar, or --i8mpp");
            else if(arg=="--builddir") buildDir=value();
            else if(arg=="--root") root=value();
            else if(arg=="--profile") profile=value();
            else if(arg=="--no-compile") options.allowCompile=false;
            else if(arg=="--profile-stages") profileStages=true;
            else if(arg=="--optimize") {
                const char *opt=value();
                if(std::strcmp(opt,"0") && std::strcmp(opt,"1")) fail("--optimize must be 0 or 1");
                options.optimize=opt[0]=='1';
            }
            else if(arg=="--bench") {
                bench=positive_integer(value(),arg);
                if(bench==INT_MAX) fail("--bench iteration count is too large");
            }
            else if(arg=="--tier") {
                options.tier=positive_integer(value(),arg);
                if(options.tier!=1080 && options.tier!=2160 && options.tier!=4320)
                    fail("--tier must be 1080, 2160, or 4320");
            }
            else if(arg=="--dump-state") dumpState=true;
            else if(arg=="--dump-passes") {
                options.captureStages={"pre"};
                for(int pass=1;pass<=12;++pass) options.captureStages.push_back("pass"+std::to_string(pass));
                options.captureStages.push_back("post"); dumpState=true;
            }
            else if(arg=="--capture-stages") {
                std::string list=value(); size_t from=0;
                do {size_t end=list.find(',',from); std::string name=list.substr(from,end-from);
                    if(!capture_stage(name)) fail("capture stage must be pre, post, pass1..12, or pad0..12");
                    if(std::find(options.captureStages.begin(),options.captureStages.end(),name)==options.captureStages.end())
                        options.captureStages.push_back(name);
                    if(end==std::string::npos) break; from=end+1;
                } while(true);
            }
            else if(arg=="--backend-info") {
                info=true;
                if(i+1<argc && argv[i+1][0]!='-') options.backend=backend(argv[++i]);
            }
            else if(!arg.empty() && arg[0]=='-') fail("unknown FSR 4.1.1 option " + arg);
            else positional.push_back(arg);
        }
        if (bench && (dumpState || !options.captureStages.empty()))
            fail("diagnostic snapshots change GPU work; use --bench separately from dumps");
        if(!fsr411::parse_profile(profile,options.profile))
            fail("--profile / FSR4_PROFILE must be exact, exact-fast or lossy");
        id<MTLDevice> device=MTLCreateSystemDefaultDevice();
        if(!device) fail("no Metal device");
        if(info) { if(!positional.empty()) fail("--backend-info takes no testcase paths");
            print_backend_info(device,options.backend,profile); return 0; }
        if(positional.size()!=2) {usage(); return 2;}
        const std::string inputDir=absolute_path(positional[0]), outputDir=absolute_path(positional[1]);
        if(inputDir==outputDir) fail("testcase and output directories must differ");
        root=absolute_path(root); buildDir=absolute_path(buildDir);
        NSError *error=nil;
        id object=[NSJSONSerialization JSONObjectWithData:read_file(inputDir+"/meta.json") options:0 error:&error];
        if(![object isKindOfClass:NSDictionary.class]) fail("meta.json must be a JSON object: "+fsr411::metal_error(error));
        NSDictionary *meta=object;
        const uint32_t rw=dimension(meta,@"render_width",0), rh=dimension(meta,@"render_height",0);
        const uint32_t uw=dimension(meta,@"upscale_width",0), uh=dimension(meta,@"upscale_height",0);
        const uint32_t count=dimension(meta,@"frames",1);
        if(count>100000) fail("too many frames in recording");
        id frameObject=meta[@"frame"];
        if(frameObject && (![frameObject isKindOfClass:NSArray.class] || [frameObject count]!=count))
            fail("meta.json: frame array length must equal frames");
        NSArray *frameMeta=frameObject;
        const std::string preset=string(meta,@"preset","quality"), colorspace=string(meta,@"colorspace","linear");
        options.ultraPerformance=preset=="ultraperf" || preset=="ultraperformance" || preset=="ultra_performance";
        if(colorspace=="linear") options.colorspace=0;
        else if(colorspace=="nonlinear" || colorspace=="non_linear") options.colorspace=1;
        else if(colorspace=="srgb") options.colorspace=2;
        else if(colorspace=="pq") options.colorspace=3;
        else fail("meta.json: colorspace must be linear, nonlinear, srgb, or pq");
        options.rcas=flag(meta,@"rcas",0); options.sharpness=float(number(meta,@"sharpness",0));
        if(![NSFileManager.defaultManager createDirectoryAtPath:ns(outputDir) withIntermediateDirectories:YES
            attributes:nil error:&error]) fail("cannot create output directory: "+fsr411::metal_error(error));
        id<MTLCommandQueue> queue=[device newCommandQueue];
        if(!queue) fail("cannot create Metal command queue");
        fsr411::Runtime runtime(device,root,buildDir);
        NSMutableArray *measurements=[NSMutableArray array];
        std::vector<double> times;
        std::map<std::string,std::pair<double,size_t>> stageTotals;
        std::string stageMethod;
        uint64_t neuralRuns=0,reusedFrames=0;
        std::string actualBackend, variant;
        const int runs=bench?bench+1:1;
        for(int run=0;run<runs;++run) {
            fsr411::View view;
            for(uint32_t index=0;index<count;++index) { @autoreleasepool {
                NSDictionary *specific=frameMeta?frameMeta[index]:@{};
                if(![specific isKindOfClass:NSDictionary.class]) fail("meta.json: frame entry must be an object");
                NSMutableDictionary *fm=[meta mutableCopy]; [fm addEntriesFromDictionary:specific];
                fsr411::Frame frame;
                auto &in=frame.input;
                in.rw=dimension(fm,@"render_width",rw); in.rh=dimension(fm,@"render_height",rh);
                in.uw=dimension(fm,@"upscale_width",uw); in.uh=dimension(fm,@"upscale_height",uh);
                in.jitterX=in.jitterY=0; in.mvScaleX=in.mvScaleY=1;
                pair(fm,@"jitter",in.jitterX,in.jitterY); pair(fm,@"mv_scale",in.mvScaleX,in.mvScaleY);
                in.reset=flag(fm,@"reset",0); in.depthReversed=flag(fm,@"depth_inverted",0);
                in.outputResMV=!flag(fm,@"low_res_mv",1); in.jitteredMV=flag(fm,@"jittered_mv",0);
                in.autoExposure=flag(fm,@"auto_exposure",0); in.inputContentProperties=1;
                in.preExposure=float(number(fm,@"pre_exposure",1));
                const uint32_t mvW=in.outputResMV?in.uw:in.rw, mvH=in.outputResMV?in.uh:in.rh;
                in.geometry.color={in.rw,in.rh,0,0}; in.geometry.depth=in.geometry.color;
                in.geometry.motion={mvW,mvH,0,0}; in.geometry.output={in.uw,in.uh,0,0};
                Fsr4InputPrepared prepared={};
                auto status=fsr4_input_prepare(&in,&view.inputHistory,0,&prepared);
                if(!fsr4_input_status_ok(status)) fail(fsr4_input_status_reason(status));
                const std::string src=frame_prefix(inputDir,index), dst=frame_prefix(outputDir,index);
                frame.color=input_texture(device,src+"color.bin",in.rw,in.rh,MTLPixelFormatRGBA16Float,8);
                frame.depth=input_texture(device,src+"depth.bin",in.rw,in.rh,MTLPixelFormatR32Float,4);
                frame.motion=input_texture(device,src+"mv.bin",mvW,mvH,MTLPixelFormatRG16Float,4);
                frame.output=texture(device,in.uw,in.uh,MTLPixelFormatRGBA16Float);
                if([NSFileManager.defaultManager fileExistsAtPath:ns(src+"exposure.bin")]) {
                    frame.exposure=input_texture(device,src+"exposure.bin",1,1,MTLPixelFormatR32Float,4);
                    in.hasExposureTexture=1;
                }
                // Missing exposure is neutral one, just like the live bridge.
                // preExposure independently divides the captured input colour.
                auto cb=[queue commandBuffer]; cb.label=@"FSR 4.1.1 frame";
                std::unique_ptr<CounterStageProfiler> stageProfiler;
                if(profileStages) stageProfiler=std::make_unique<CounterStageProfiler>(device,cb);
                options.stageProfiler=stageProfiler.get();
                std::string why;
                if(!runtime.encode(cb,view,frame,options,why)) fail(why);
                complete(cb);
                const double milliseconds=(cb.GPUEndTime-cb.GPUStartTime)*1000.;
                if(!std::isfinite(milliseconds) || milliseconds<=0) fail("Metal did not report a valid GPU duration");
                NSDictionary *stageReport=stageProfiler?stageProfiler->resolve(milliseconds):nil;
                actualBackend=view.backendLabel; variant=view.variantKey;
                if(!bench || run>0) {
                    times.push_back(milliseconds);
                    if(view.skippedInference) ++reusedFrames; else ++neuralRuns;
                    NSMutableArray *captured=[NSMutableArray array];
                    for(const auto &entry:view.captures) [captured addObject:ns(entry.first)];
                    NSMutableDictionary *measurement=[@{@"iteration":@(bench?run-1:0),@"frame":@(index),
                        @"gpu_ms":@(milliseconds),@"backend":ns(actualBackend),@"variant":ns(variant),
                        @"skipped_inference":@(view.skippedInference),
                        @"sequence_neural_runs":@(view.neuralRuns),@"sequence_reused_frames":@(view.reusedFrames),
                        @"captured_stages":captured,
                        @"render_width":@(in.rw),@"render_height":@(in.rh),
                        @"upscale_width":@(in.uw),@"upscale_height":@(in.uh),
                        @"aligned_width":@(view.prepared.alignedUW),@"aligned_height":@(view.prepared.alignedUH),
                        @"motion_width":@(mvW),@"motion_height":@(mvH),
                        @"effective_reset":@(view.prepared.reset),@"reset_reasons":@(view.prepared.resetReasons),
                        @"pre_exposure":@(in.preExposure),@"exposure_mode":@(view.prepared.exposureMode)} mutableCopy];
                    if(stageReport) {
                        measurement[@"stage_profile"]=stageReport;stageMethod=stageProfiler->method();
                        for(const auto &timing:stageProfiler->timings) {
                            auto &total=stageTotals[timing.name];total.first+=timing.milliseconds;++total.second;
                        }
                    }
                    [measurements addObject:measurement];
                }
                if(!bench) {
                    auto output=read_texture(device,queue,frame.output,8);
                    write_preview(dst+"out.png",output,in.uw,in.uh,options.colorspace);
                    write_file(dst+"out.bin",output.data(),output.size());
                    for(const auto &stage:options.captureStages) {
                        std::string suffix=stage+"_scratch.bin";
                        if(stage.rfind("pad",0)!=0) {
                            const int pass=stage=="pre"?0:(stage=="post"?13:std::atoi(stage.c_str()+4));
                            char name[64]; std::snprintf(name,sizeof name,"pass%02d_scratch.bin",pass); suffix=name;
                        }
                        const auto captured=view.captures.find(stage);
                        if(captured!=view.captures.end()) dump_buffer(dst+suffix,device,queue,captured->second);
                        // A rerun into an existing output folder must not leave
                        // an old neural snapshot in place of a skipped stage.
                        else if(std::remove((dst+suffix).c_str())!=0 && errno!=ENOENT)
                            fail("cannot remove stale skipped-stage snapshot "+dst+suffix);
                    }
                    if(dumpState) {
                        dump_buffer(dst+"scratch.bin",device,queue,view.scratch);
                        const struct {const char *name; id<MTLTexture> texture; size_t bytes;} states[]={
                            {"history",view.history,8},{"reprojected",view.reprojected,8},
                            {"recurrent",view.recurrent,4},{"auto_exposure",view.autoExposure,4}};
                        NSMutableDictionary *layout=[NSMutableDictionary dictionary];
                        for(const auto &state:states) {
                            auto data=read_texture(device,queue,state.texture,state.bytes);
                            write_file(dst+state.name+".bin",data.data(),data.size());
                            layout[ns(state.name)]=@{@"width":@(state.texture.width),@"height":@(state.texture.height),
                                @"metal_pixel_format":@(state.texture.pixelFormat),@"bytes_per_pixel":@(state.bytes)};
                        }
                        layout[@"scratch_bytes"]=@(view.scratch.length);
                        write_json(dst+"state.json",layout);
                    }
                }
                std::fprintf(stderr,"FSR 4.1.1 %s: %sframe %u, %s, GPU %.3f ms%s%s\n",actualBackend.c_str(),
                    bench && run==0?"warmup ":"",index,view.skippedInference?"reuse":"neural",milliseconds,
                    view.captures.empty()?"":" (with snapshots)",profileStages?" (with stage counters)":"");
            }}
        }
        const double mean=std::accumulate(times.begin(),times.end(),0.)/times.size();
        const double minimum=*std::min_element(times.begin(),times.end());
        NSMutableArray *stages=[NSMutableArray array]; for(const auto &s:options.captureStages) [stages addObject:ns(s)];
        NSMutableDictionary *report=[@{@"schema_version":@1,@"fsr_version":@"4.1.1",@"device":device.name,
            @"profile":ns(profile),
            @"backend":ns(actualBackend),@"variant":ns(variant),@"testcase":ns(inputDir),
            @"preset":ns(preset),@"colorspace":ns(colorspace),@"benchmark_iterations":@(bench),
            @"optimize":@(options.optimize),@"requested_tier":@(options.tier),
            @"warmup_sequences":@(bench?1:0),@"timing_scope":@"full frame command buffer",
            @"measured_frames":@(times.size()),@"neural_runs":@(neuralRuns),@"reused_frames":@(reusedFrames),
            @"capture_stages":stages,@"diagnostic_gpu_copies":@(!options.captureStages.empty()),
            @"diagnostic_gpu_counters":@(profileStages),
            @"missing_exposure":@"neutral one; preExposure remains independent",
            @"gpu_mean_ms":@(mean),@"gpu_min_ms":@(minimum),@"measurements":measurements} mutableCopy];
        if(profileStages) {
            NSMutableArray *summary=[NSMutableArray array];
            for(const auto &entry:stageTotals)
                [summary addObject:@{@"stage":ns(entry.first),@"samples":@(entry.second.second),
                    @"gpu_mean_ms":@(entry.second.first/entry.second.second)}];
            report[@"stage_profiling"]=@{@"method":ns(stageMethod),@"summary":summary,
                @"production_timing":@NO,@"includes_trailing_barrier":@YES,
                @"note":@"Counters instrument each dispatch. Encoder-boundary sampling changes encoder scheduling; dispatch sampling adds sampling barriers. Use an uninstrumented --bench run for production GPU time. Independent encoders can overlap, so stage sums can exceed the sampled span. Gaps include sampling and scheduling, not isolated barrier cost."};
        }
        write_json(outputDir+"/run.json",report);
        std::printf("FSR 4.1.1 %s: mean %.3f ms, min %.3f ms (%zu measured frames: %llu neural, %llu reused)\n",
            actualBackend.c_str(),mean,minimum,times.size(),
            static_cast<unsigned long long>(neuralRuns),static_cast<unsigned long long>(reusedFrames));
        return 0;
    } catch(const std::exception &error) {
        std::fprintf(stderr,"fsr4mac 4.1.1: %s\n",error.what()); return 1;
    }
    }
}
