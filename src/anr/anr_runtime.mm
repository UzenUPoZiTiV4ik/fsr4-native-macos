// SPDX-License-Identifier: MIT
#include "anr_runtime.h"
#import <CommonCrypto/CommonDigest.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <set>
#include <vector>

namespace anr {
namespace {

std::string utf8(NSString *s) {
    return s ? std::string(s.UTF8String) : "";
}

NSString *ns(const std::string &s) {
    return [NSString stringWithUTF8String:s.c_str()];
}

NSData *load_file(NSString *root, NSString *relative) {
    return [NSData dataWithContentsOfFile:[root stringByAppendingPathComponent:relative]];
}

// Every hashed asset is capped at 32MB by the manifest check.
std::string sha256_hex(NSData *data) {
    unsigned char bytes[32];
    CC_SHA256(data.bytes, (CC_LONG)data.length, bytes);
    static const char hex[] = "0123456789abcdef";
    std::string result;
    result.reserve(64);
    for (unsigned char byte : bytes) {
        result += hex[byte >> 4];
        result += hex[byte & 15];
    }
    return result;
}

bool safe_path(NSString *path) {
    if (![path isKindOfClass:NSString.class] || !path.length || path.isAbsolutePath) {
        return false;
    }
    for (NSString *part in [path componentsSeparatedByString:@"/"]) {
        if (!part.length || [part isEqual:@"."] || [part isEqual:@".."]) {
            return false;
        }
    }
    return [path rangeOfCharacterFromSet:[NSCharacterSet characterSetWithCharactersInString:@"\\\0"]].location == NSNotFound;
}

int64_t floordiv_i64(int64_t a, int64_t b) {
    int64_t q = a / b;
    int64_t r = a % b;
    if (r != 0 && ((r < 0) != (b < 0))) {
        --q;
    }
    return q;
}

const char *encoding_key(Encoding enc) {
    switch (enc) {
        case Encoding::SRGB8: return "srgb8";
        case Encoding::SCRGB16: return "scrgb16";
        case Encoding::PQ10:
        default: return "pq10";
    }
}

MTLPixelFormat pixel_format_from_name(NSString *name) {
    if ([name isEqual:@"rgb10a2_unorm"]) return MTLPixelFormatRGB10A2Unorm;
    if ([name isEqual:@"rgba8_unorm"]) return MTLPixelFormatRGBA8Unorm;
    if ([name isEqual:@"rgba16f"]) return MTLPixelFormatRGBA16Float;
    if ([name isEqual:@"r32_uint"]) return MTLPixelFormatR32Uint;
    if ([name isEqual:@"r32f"]) return MTLPixelFormatR32Float;
    if ([name isEqual:@"rg16f"]) return MTLPixelFormatRG16Float;
    return MTLPixelFormatInvalid;
}

std::mutex g_pipelineMutex;
std::map<std::pair<uintptr_t, std::string>, id<MTLComputePipelineState>> g_pipelineCache;
std::map<std::pair<uintptr_t, std::string>, id<MTLLibrary>> g_libraryCache;

enum class BindKind {
    Uniform = 0,
    Buffer = 1,
    Texture = 2,
    StorageTexture = 3,
    Sampler = 4,
};

struct Binding {
    std::string name;
    unsigned index = 0;
    BindKind kind = BindKind::Buffer;
    bool r32uint = false;
    bool writable = true;
};

struct Kernel {
    id<MTLComputePipelineState> pipeline = nil;
    id<MTLArgumentEncoder> argEncoder = nil;
    bool native = false;
    unsigned resources = 0;
    unsigned samplers = 0;
    unsigned slotCount = 0;
    unsigned uniformBytes = 0;
    std::vector<Binding> bindings;
};

struct StageArguments {
    id<MTLBuffer> buffer = nil;
    id<MTL4ArgumentTable> table = nil;
    bool initialized = false;
    std::array<uint64_t, 4> externalIDs = {};
};

struct ExternalImage {
    unsigned slot;
    unsigned source; // color, depth, motion, output
};

struct Stage {
    std::string label;
    std::string kernel;
    MTLSize groups = {};
    MTLSize threads = {};
    id<MTLBuffer> constants = nil;
    // v1 (B14) root descriptor table buffers:
    id<MTLBuffer> resources = nil;
    id<MTLBuffer> samplers = nil;
    id<MTLBuffer> root = nil;
    // v2 (B20) compact Tier-2 argument buffer:
    id<MTL4ArgumentTable> arguments = nil; // v1
    StageArguments parityArgs[2];
    std::vector<ExternalImage> externalImages;
    std::map<std::string, size_t> bufferOffsets;
    uint64_t bufferRevision = 0;
    std::map<std::string, unsigned> buffers;
    std::map<std::string, std::string> images;
    bool hasExternalImage = false;
    bool needsBarrier = true;
};

struct Descriptor {
    uint64_t address = 0;
    uint64_t resource = 0;
    uint64_t metadata = 0;
};
static_assert(sizeof(Descriptor) == 24, "Apple descriptor ABI");

struct InternalTexture {
    id<MTLTexture> main = nil;
    id<MTLTexture> r32uintView = nil;
};

MTLSize extent_from_array(NSArray *v) {
    return MTLSizeMake([v[0] unsignedLongLongValue], [v[1] unsignedLongLongValue], [v[2] unsignedLongLongValue]);
}

bool eval_exprs(NSArray *exprs, int64_t width, int64_t height, std::vector<int64_t> &out, std::string &error) {
    out.clear();
    out.reserve(exprs.count);
    for (NSArray *e in exprs) {
        if (![e isKindOfClass:NSArray.class] || !e.count) {
            error = "ANR v2 malformed expr";
            return false;
        }
        NSString *op = e[0];
        auto at = [&](NSUInteger pos, int64_t &val) -> bool {
            if (pos >= e.count) return false;
            NSInteger idx = [e[pos] integerValue];
            if (idx < 0 || (size_t)idx >= out.size()) return false;
            val = out[(size_t)idx];
            return true;
        };
        if ([op isEqual:@"W"]) {
            out.push_back(width);
        } else if ([op isEqual:@"H"]) {
            out.push_back(height);
        } else if ([op isEqual:@"const"]) {
            if (e.count < 2) { error = "ANR v2 bad const expr"; return false; }
            out.push_back([e[1] longLongValue]);
        } else if ([op isEqual:@"neg"]) {
            int64_t a = 0;
            if (!at(1, a)) { error = "ANR v2 bad neg expr"; return false; }
            out.push_back(-a);
        } else if ([op isEqual:@"add"] || [op isEqual:@"sub"] || [op isEqual:@"mul"]
                   || [op isEqual:@"floordiv"] || [op isEqual:@"max"]) {
            int64_t a = 0, b = 0;
            if (!at(1, a) || !at(2, b)) { error = "ANR v2 bad binary expr"; return false; }
            if ([op isEqual:@"add"]) out.push_back(a + b);
            else if ([op isEqual:@"sub"]) out.push_back(a - b);
            else if ([op isEqual:@"mul"]) out.push_back(a * b);
            else if ([op isEqual:@"floordiv"]) {
                if (b == 0) { error = "ANR v2 division by zero"; return false; }
                out.push_back(floordiv_i64(a, b));
            } else {
                out.push_back(std::max(a, b));
            }
        } else {
            error = "ANR v2 unsupported expr op: " + utf8(op);
            return false;
        }
    }
    return true;
}

bool resolve_int(id node, const std::vector<int64_t> &vals, int64_t &out) {
    if ([node isKindOfClass:NSNumber.class]) {
        out = [node longLongValue];
        return true;
    }
    if ([node isKindOfClass:NSDictionary.class]) {
        NSNumber *idx = node[@"x"];
        if ([idx isKindOfClass:NSNumber.class]) {
            const NSInteger i = idx.integerValue;
            if (i >= 0 && (size_t)i < vals.size()) {
                out = vals[(size_t)i];
                return true;
            }
        }
    }
    return false;
}

bool resolve_double(id node, const std::vector<int64_t> &vals, double &out) {
    if ([node isKindOfClass:NSNumber.class]) {
        out = [node doubleValue];
        return true;
    }
    int64_t iv = 0;
    if (resolve_int(node, vals, iv)) {
        out = (double)iv;
        return true;
    }
    return false;
}

bool check_guards(NSArray *guards, const std::vector<int64_t> &vals) {
    for (NSArray *g in guards) {
        if (![g isKindOfClass:NSArray.class] || g.count != 4) return false;
        NSString *op = g[0];
        int64_t a = 0, b = 0;
        if (!resolve_int(g[1], vals, a) || !resolve_int(g[2], vals, b)) return false;
        const bool expect = [g[3] boolValue];
        bool actual = false;
        if ([op isEqual:@"lt"]) actual = a < b;
        else if ([op isEqual:@"le"]) actual = a <= b;
        else if ([op isEqual:@"gt"]) actual = a > b;
        else if ([op isEqual:@"eq"]) actual = a == b;
        else if ([op isEqual:@"ne"]) actual = a != b;
        else return false;
        if (actual != expect) return false;
    }
    return true;
}

} // namespace

struct Runtime::Impl {
    id<MTLDevice> device = nil;
    NSString *directory = nil;
    NSDictionary *manifest = nil;

    bool initialized = false;
    bool isV2 = false;
    bool mppActive = false;
    bool coalesced = true;
    bool deviceInternalBarrier = false;
    bool dagBarriers = true;
    Encoding encoding = Encoding::PQ10;
    std::string releaseName;
    std::string backendName;
    unsigned width = 0;
    unsigned height = 0;
    uint64_t bufferRevision = 1;
    unsigned frameSlot = 0;
    unsigned swapParity = 0;
    id<MTLBuffer> scratchArena = nil;

    unsigned inputColor0 = 0, inputColor1 = 0, inputDepth0 = 0, inputDepth1 = 0, inputFrame = 0;
    std::vector<unsigned> persistentBuffers;
    std::vector<std::pair<unsigned, unsigned>> bufferSwaps;
    std::vector<std::pair<std::string, std::string>> textureSwaps;

    id<MTLBuffer> frameBuffers[4] = {};
    id<MTLResidencySet> borrowedSets[4] = {};
    std::array<uintptr_t, 5> borrowedKeys[4] = {};
    id<MTLResidencySet> residency = nil;
    id<MTLBuffer> empty = nil;
    id<MTLSamplerState> sampler = nil;

    std::vector<id<MTLBuffer>> buffers;
    std::map<std::string, InternalTexture> internalTextures;
    std::map<std::string, Kernel> kernels;
    std::map<std::string, std::vector<Stage>> stages[4];
    std::vector<id<MTLResource>> staticMetal3Resources;

    id<MTLTexture> inputColor = nil, inputDepth = nil, inputMotion = nil, output = nil;

    bool fail(std::string &error, const std::string &why) {
        error = why;
        return false;
    }

    bool add_resident(id<MTLAllocation> resource, std::string &error) {
        if (!resource) return fail(error, "ANR resource allocation failed");
        [residency addAllocation:resource];
        return true;
    }

    id<MTLBuffer> alloc_shared(size_t bytes) {
        return [device newBufferWithLength:std::max(bytes, size_t(256))
                                   options:MTLResourceStorageModeShared];
    }

    id<MTLBuffer> alloc_private(size_t bytes) {
        const bool dumpMode = getenv("ANR_DUMP_BUFFERS") != nullptr;
        // Metal 3 needs tracked hazards across encoders (in particular the reset
        // blit before the pair pass). Metal 4 is synchronized explicitly.
        const MTLResourceOptions opts = dumpMode ? MTLResourceStorageModeShared : MTLResourceStorageModePrivate;
        return [device newBufferWithLength:std::max(bytes, size_t(256)) options:opts];
    }

    void update_borrowed_residency(id<MTLTexture> c, id<MTLTexture> d, id<MTLTexture> m,
                                   id<MTLTexture> o0, id<MTLTexture> o1 = nil) {
        std::array<uintptr_t, 5> next = {
            (uintptr_t)(__bridge void *)c,
            (uintptr_t)(__bridge void *)d,
            (uintptr_t)(__bridge void *)m,
            (uintptr_t)(__bridge void *)o0,
            (uintptr_t)(__bridge void *)o1,
        };
        if (borrowedKeys[frameSlot] == next) {
            return;
        }
        auto set = borrowedSets[frameSlot];
        [set removeAllAllocations];
        if (c) [set addAllocation:c];
        if (d) [set addAllocation:d];
        if (m) [set addAllocation:m];
        if (o0) [set addAllocation:o0];
        if (o1 && o1 != o0) [set addAllocation:o1];
        [set commit];
        borrowedKeys[frameSlot] = next;
    }

    id<MTLTexture> resolve_texture(const std::string &imageName, bool r32uint, std::string &error) {
        if (imageName == "color") {
            if (!inputColor) { fail(error, "ANR stage missing input color"); return nil; }
            return inputColor;
        }
        if (imageName == "depth") {
            if (!inputDepth) { fail(error, "ANR stage missing input depth"); return nil; }
            return inputDepth;
        }
        if (imageName == "motion") {
            if (!inputMotion) { fail(error, "ANR stage missing input motion"); return nil; }
            return inputMotion;
        }
        if (imageName == "output") {
            if (!output) { fail(error, "ANR stage missing output texture"); return nil; }
            return output;
        }
        auto it = internalTextures.find(imageName);
        if (it == internalTextures.end()) {
            fail(error, "ANR stage unknown internal texture: " + imageName);
            return nil;
        }
        if (r32uint) {
            if (!it->second.r32uintView) {
                fail(error, "ANR stage missing r32uint view for: " + imageName);
                return nil;
            }
            return it->second.r32uintView;
        }
        return it->second.main;
    }

    bool prepare_stage_v1(Stage &stage, std::string &error) {
        auto found = kernels.find(stage.kernel);
        if (found == kernels.end()) return fail(error, "ANR kernel missing: " + stage.kernel);
        Kernel &kernel = found->second;
        auto *entries = (Descriptor *)stage.resources.contents;
        auto *samplerEntries = (Descriptor *)stage.samplers.contents;
        const bool buffersUpToDate = stage.bufferRevision == bufferRevision;
        if (buffersUpToDate && !stage.hasExternalImage) {
            return true;
        }
        for (const Binding &binding : kernel.bindings) {
            if (binding.kind == BindKind::Sampler) {
                if (!buffersUpToDate) {
                    samplerEntries[binding.index] = {sampler.gpuResourceID._impl, 0, 0};
                }
                continue;
            }
            auto image = stage.images.find(binding.name);
            if (image != stage.images.end()) {
                id<MTLTexture> tex = resolve_texture(image->second, false, error);
                if (!tex) return false;
                entries[binding.index] = {0, tex.gpuResourceID._impl, 0};
            } else {
                if (buffersUpToDate) continue;
                id<MTLBuffer> target = empty;
                if (binding.kind == BindKind::Uniform) {
                    target = stage.constants;
                } else {
                    auto it = stage.buffers.find(binding.name);
                    if (it != stage.buffers.end()) target = buffers.at(it->second);
                }
                entries[binding.index] = {target.gpuAddress, 0, target.length};
            }
        }
        if (!stage.bufferRevision) {
            [stage.arguments setAddress:stage.root.gpuAddress atIndex:2];
        }
        stage.bufferRevision = bufferRevision;
        return true;
    }

    bool prepare_stage_v2(Stage &stage, std::string &error) {
        auto &args = stage.parityArgs[swapParity];
        if (args.initialized && stage.externalImages.empty()) return true;
        Kernel &kernel = kernels.at(stage.kernel);
        auto encoder = kernel.argEncoder;
        bool bound = false;
        if (!args.initialized) {
            [encoder setArgumentBuffer:args.buffer offset:0];
            bound = true;
            for (const Binding &binding : kernel.bindings) {
                switch (binding.kind) {
                    case BindKind::Sampler:
                        [encoder setSamplerState:sampler atIndex:binding.index];
                        break;
                    case BindKind::Uniform:
                        [encoder setBuffer:stage.constants offset:0 atIndex:binding.index];
                        break;
                    case BindKind::Buffer: {
                        auto it = stage.buffers.find(binding.name);
                        id<MTLBuffer> target = it == stage.buffers.end() ? empty : buffers.at(it->second);
                        size_t offset = it == stage.buffers.end() ? 0 : stage.bufferOffsets.at(binding.name);
                        [encoder setBuffer:target offset:offset atIndex:binding.index];
                        break;
                    }
                    case BindKind::Texture:
                    case BindKind::StorageTexture: {
                        const auto &name = stage.images.at(binding.name);
                        if (!internalTextures.count(name)) break; // bound below
                        auto tex = resolve_texture(name, binding.r32uint, error);
                        if (!tex) return false;
                        [encoder setTexture:tex atIndex:binding.index];
                        break;
                    }
                }
            }
        }
        if (!stage.externalImages.empty()) {
            id<MTLTexture> textures[] = {inputColor, inputDepth, inputMotion, output};
            auto next = args.externalIDs;
            for (const auto &image : stage.externalImages) {
                auto tex = textures[image.source];
                if (!tex) return fail(error, "ANR v2 missing external texture");
                const uint64_t key = tex.gpuResourceID._impl;
                if (args.initialized && args.externalIDs[image.source] == key) continue;
                if (!bound) {
                    [encoder setArgumentBuffer:args.buffer offset:0];
                    bound = true;
                }
                [encoder setTexture:tex atIndex:image.slot];
                next[image.source] = key;
            }
            args.externalIDs = next;
        }
        args.initialized = true;
        return true;
    }

    bool encode_groups(id command, bool metal4,
                       const std::vector<std::pair<std::vector<Stage> *, id<MTLTexture>>> &batches,
                       std::string &error,
                       id<MTL4CounterHeap> profileHeap = nil, unsigned profileBase = 0,
                       std::vector<StageProfile> *profileOut = nullptr, const FrameExtras &extras = {},
                       bool fuseGroups = true) {
        if (metal4) {
            [(id<MTL4CommandBuffer>)command useResidencySet:residency];
            [(id<MTL4CommandBuffer>)command useResidencySet:borrowedSets[frameSlot]];
        }

        std::vector<id<MTLResource>> m3Resources;
        if (!metal4) {
            m3Resources = staticMetal3Resources;
            for (const auto &batch : batches) {
                for (Stage &stage : *batch.first) {
                    m3Resources.push_back(stage.constants);
                    if (isV2) {
                        m3Resources.push_back(stage.parityArgs[swapParity].buffer);
                    } else {
                        m3Resources.push_back(stage.resources);
                        m3Resources.push_back(stage.samplers);
                        m3Resources.push_back(stage.root);
                    }
                }
                if (batch.second) m3Resources.push_back(batch.second);
            }
            if (inputColor) m3Resources.push_back(inputColor);
            if (inputDepth) m3Resources.push_back(inputDepth);
            if (inputMotion) m3Resources.push_back(inputMotion);
        }

        for (const auto &batch : batches) {
            if (batch.second) output = batch.second;
            for (Stage &stage : *batch.first) {
                const bool ok = isV2 ? prepare_stage_v2(stage, error) : prepare_stage_v1(stage, error);
                if (!ok) return false;
            }
        }

        const bool profiling = metal4 && profileHeap != nil;
        const MTL4VisibilityOptions intraVis = deviceInternalBarrier
            ? MTL4VisibilityOptionDevice
            : MTL4VisibilityOptionNone;
        id encoder = nil;
        bool firstDispatch = true;
        unsigned stageIdx = 0;
        auto closeEncoder = [&] {
            [encoder endEncoding];
            encoder = nil;
        };
        // A batch boundary always orders the two dispatches, even when the DAG
        // for either batch alone has no hazard. This also protects helper I/O.
        auto beginDispatch = [&](bool boundary, bool hazard) {
            if (encoder && (!coalesced || (!fuseGroups && boundary))) closeEncoder();
            if (!encoder) {
                encoder = metal4 ? (id)[(id<MTL4CommandBuffer>)command computeCommandEncoder]
                                 : (id)[(id<MTLCommandBuffer>)command computeCommandEncoder];
                if (!encoder) return fail(error, "ANR compute encoder allocation failed");
                if (metal4) {
                    auto enc = (id<MTL4ComputeCommandEncoder>)encoder;
                    [enc barrierAfterQueueStages:MTLStageAll beforeStages:MTLStageDispatch
                               visibilityOptions:MTL4VisibilityOptionDevice];
                    if (firstDispatch && extras.fence) [enc waitForFence:extras.fence beforeEncoderStages:MTLStageDispatch];
                } else {
                    auto enc = (id<MTLComputeCommandEncoder>)encoder;
                    [enc useResources:m3Resources.data() count:m3Resources.size() usage:MTLResourceUsageRead | MTLResourceUsageWrite];
                    if (firstDispatch && extras.fence) [enc waitForFence:extras.fence];
                }
            } else if (boundary || hazard) {
                if (metal4) {
                    [(id<MTL4ComputeCommandEncoder>)encoder barrierAfterEncoderStages:MTLStageDispatch
                                                                 beforeEncoderStages:MTLStageDispatch
                                                                   visibilityOptions:boundary ? MTL4VisibilityOptionDevice : intraVis];
                } else {
                    [(id<MTLComputeCommandEncoder>)encoder memoryBarrierWithScope:MTLBarrierScopeBuffers | MTLBarrierScopeTextures];
                }
            }
            firstDispatch = false;
            return true;
        };
        if (extras.import) {
            if (!beginDispatch(true, false)) return false;
            extras.import(encoder);
        }

        for (size_t bIdx = 0; bIdx < batches.size(); ++bIdx) {
            auto &group = *batches[bIdx].first;
            for (size_t sIdx = 0; sIdx < group.size(); ++sIdx) {
                Stage &stage = group[sIdx];
                if (profiling) {
                    [(id<MTL4CommandBuffer>)command writeTimestampIntoHeap:profileHeap atIndex:profileBase + stageIdx * 2];
                }
                if (!beginDispatch(sIdx == 0, !dagBarriers || stage.needsBarrier)) return false;
                Kernel &kernel = kernels.at(stage.kernel);
                if (metal4) {
                    auto enc = (id<MTL4ComputeCommandEncoder>)encoder;
                    [enc setComputePipelineState:kernel.pipeline];
                    [enc setArgumentTable:isV2 ? stage.parityArgs[swapParity].table : stage.arguments];
                    [enc dispatchThreadgroups:stage.groups threadsPerThreadgroup:stage.threads];
                } else {
                    auto enc = (id<MTLComputeCommandEncoder>)encoder;
                    [enc setComputePipelineState:kernel.pipeline];
                    if (isV2) {
                        [enc setBuffer:stage.parityArgs[swapParity].buffer offset:0 atIndex:0];
                    } else {
                        [enc setBuffer:stage.root offset:0 atIndex:2];
                    }
                    [enc dispatchThreadgroups:stage.groups threadsPerThreadgroup:stage.threads];
                }

                if (profiling) {
                    closeEncoder();
                    [(id<MTL4CommandBuffer>)command writeTimestampIntoHeap:profileHeap atIndex:profileBase + stageIdx * 2 + 1];
                    if (profileOut) {
                        profileOut->push_back({stage.label, stage.kernel, 0.0});
                    }
                }
                ++stageIdx;
            }
            if (bIdx && extras.exportPhase) {
                if (!beginDispatch(true, false)) return false;
                extras.exportPhase(encoder, unsigned(bIdx - 1));
            }
        }
        if (extras.fence) {
            if (metal4) [(id<MTL4ComputeCommandEncoder>)encoder updateFence:extras.fence afterEncoderStages:MTLStageDispatch];
            else [(id<MTLComputeCommandEncoder>)encoder updateFence:extras.fence];
        }
        if (extras.retire) {
            if (!beginDispatch(true, false)) return false;
            extras.retire(encoder);
        }
        closeEncoder();
        return true;
    }

    bool encode_group(id command, bool metal4, std::vector<Stage> &group, std::string &error,
                      id<MTL4CounterHeap> profileHeap = nil, unsigned profileBase = 0,
                      std::vector<StageProfile> *profileOut = nullptr) {
        std::vector<std::pair<std::vector<Stage> *, id<MTLTexture>>> batches = {{&group, output}};
        return encode_groups(command, metal4, batches, error, profileHeap, profileBase, profileOut);
    }

    bool verify_inventory(NSDictionary *files, std::string &error) {
        if (![files isKindOfClass:NSDictionary.class] || !files.count || files.count > 1000) {
            return fail(error, "ANR runtime inventory missing");
        }
        for (NSString *name in files) {
            if (!safe_path(name)) return fail(error, "ANR unsafe asset path");
            NSString *path = [directory stringByAppendingPathComponent:name];
            NSDictionary *attributes = [NSFileManager.defaultManager attributesOfItemAtPath:path error:nil];
            if (![attributes[NSFileType] isEqual:NSFileTypeRegular]) {
                return fail(error, "ANR asset missing/nonregular: " + utf8(name));
            }
            NSDictionary *row = files[name];
            if (![row isKindOfClass:NSDictionary.class]
                || ![row[@"bytes"] isKindOfClass:NSNumber.class]
                || [row[@"bytes"] unsignedLongLongValue] > 32u * 1024u * 1024u
                || ![row[@"sha256"] isKindOfClass:NSString.class]) {
                return fail(error, "ANR malformed inventory entry");
            }
            NSData *data = load_file(directory, name);
            if (!data) return fail(error, "ANR unreadable asset");
            if (data.length != [row[@"bytes"] unsignedLongLongValue] || sha256_hex(data) != utf8(row[@"sha256"])) {
                return fail(error, "ANR asset identity mismatch: " + utf8(name));
            }
        }
        return true;
    }

    bool stage_wanted(NSString *key, PhaseMode mode) const {
        if ([key isEqual:@"pair"]) return true;
        if (mode == PhaseMode::All) return true;
        if (mode == PhaseMode::TwoX) return [key isEqual:@"phase:0.5:image"];
        // ThreeX also keeps 0.5 ready so live 2x <-> 3x switches require no reload.
        return [key isEqual:@"phase:0.5:image"]
            || [key isEqual:@"phase:0.333333:image"]
            || [key isEqual:@"phase:0.666667:image"];
    }

    bool init_v1(std::string &error, PhaseMode phaseMode) {
        if (![manifest[@"checkpoint_id"] isEqual:@"e09deb7012c96c172e49e72eb0b747b4611a8b6e3f14e5f5563189c580772ac1"]) {
            return fail(error, "ANR runtime manifest/checkpoint mismatch");
        }
        NSDictionary *files = manifest[@"files"];
        if (!verify_inventory(files, error)) return false;

        width = [manifest[@"width"] unsignedIntValue];
        height = [manifest[@"height"] unsignedIntValue];
        if (!((width == 1920 && height == 1080) || (width == 2560 && height == 1440) || (width == 3840 && height == 2160))) {
            return fail(error, "ANR unsupported geometry");
        }
        releaseName = utf8(manifest[@"release"] ?: @"FG-V.B14");
        backendName = utf8(manifest[@"backend"] ?: @"native");
        encoding = Encoding::PQ10;

        MTLResidencySetDescriptor *rd = [MTLResidencySetDescriptor new];
        rd.initialCapacity = 1600;
        NSError *metalError = nil;
        residency = [device newResidencySetWithDescriptor:rd error:&metalError];
        if (!residency) return fail(error, "ANR residency: " + utf8(metalError.localizedDescription));

        NSData *weights = load_file(directory, @"weights.bin");
        if (!weights || weights.length != 43520
            || sha256_hex(weights) != "af6c2f200ddb7baeaf4d82b5515af6f077f67896b2a9d26163ef6ea23975acb5") {
            return fail(error, "ANR unsupported model weights");
        }
        if (![manifest[@"buffers"] isKindOfClass:NSArray.class]
            || [manifest[@"buffers"] count] < 80 || [manifest[@"buffers"] count] > 128) {
            return fail(error, "ANR invalid buffer inventory");
        }
        buffers.resize([manifest[@"buffers"] count]);
        for (NSDictionary *record in manifest[@"buffers"]) {
            unsigned index = [record[@"index"] unsignedIntValue];
            size_t bytes = [record[@"size"] unsignedLongLongValue];
            if (index >= buffers.size() || buffers[index] || !bytes || bytes > (size_t(1) << 30)) {
                return fail(error, "ANR buffer shape invalid");
            }
            buffers[index] = alloc_shared(bytes);
            if (!add_resident(buffers[index], error)) return false;
            std::memset(buffers[index].contents, 0, buffers[index].length);
            if ([record[@"kind"] isEqual:@"weight"]) {
                size_t offset = [record[@"data_offset"] unsignedLongLongValue];
                if (offset > weights.length || bytes > weights.length - offset) {
                    return fail(error, "ANR weight span invalid");
                }
                std::memcpy(buffers[index].contents, (const uint8_t *)weights.bytes + offset, bytes);
            } else if ([record[@"kind"] isEqual:@"persistent"]) {
                persistentBuffers.push_back(index);
            }
        }
        for (unsigned slot = 0; slot < 4; ++slot) {
            frameBuffers[slot] = alloc_shared(204);
            if (!add_resident(frameBuffers[slot], error)) return false;
            MTLResidencySetDescriptor *bd = [MTLResidencySetDescriptor new];
            bd.initialCapacity = 4;
            borrowedSets[slot] = [device newResidencySetWithDescriptor:bd error:&metalError];
            if (!borrowedSets[slot]) return fail(error, "ANR frame residency allocation failed");
        }
        empty = alloc_shared(256);
        if (!add_resident(empty, error)) return false;
        std::memset(empty.contents, 0, empty.length);

        MTLSamplerDescriptor *sd = [MTLSamplerDescriptor new];
        sd.supportArgumentBuffers = YES;
        sd.minFilter = sd.magFilter = MTLSamplerMinMagFilterLinear;
        sampler = [device newSamplerStateWithDescriptor:sd];

        NSDictionary *kernelRecords = manifest[@"kernels"];
        NSDictionary *stageRecords = manifest[@"stages"];
        NSMutableSet *neededKernels = [NSMutableSet set];
        for (NSString *key in stageRecords) {
            if (!stage_wanted(key, phaseMode)) continue;
            for (NSDictionary *stage in stageRecords[key]) {
                [neededKernels addObject:stage[@"kernel"]];
            }
        }
        for (NSString *key in kernelRecords) {
            if (![neededKernels containsObject:key]) continue;
            NSDictionary *record = kernelRecords[key];
            Kernel kernel;
            kernel.native = [record[@"route"] isEqual:@"native"];
            NSString *assetPath = record[@"path"];
            if (!safe_path(assetPath) || !files[assetPath]) return fail(error, "ANR unlisted kernel asset");
            const auto cacheKey = std::make_pair((uintptr_t)(__bridge void *)device,
                                                 utf8(files[assetPath][@"sha256"]) + ":" + utf8(record[@"entry"]));
            {
                std::lock_guard<std::mutex> lock(g_pipelineMutex);
                auto cached = g_pipelineCache.find(cacheKey);
                if (cached != g_pipelineCache.end()) {
                    kernel.pipeline = cached->second;
                } else {
                    NSURL *url = [NSURL fileURLWithPath:[directory stringByAppendingPathComponent:assetPath]];
                    id<MTLLibrary> library = [device newLibraryWithURL:url error:&metalError];
                    id<MTLFunction> function = [library newFunctionWithName:record[@"entry"]];
                    kernel.pipeline = function ? [device newComputePipelineStateWithFunction:function error:&metalError] : nil;
                    if (kernel.pipeline) g_pipelineCache.emplace(cacheKey, kernel.pipeline);
                }
            }
            if (!kernel.pipeline) {
                return fail(error, "ANR pipeline " + utf8(key) + ": " + utf8(metalError.localizedDescription));
            }
            kernel.resources = [record[@"resource_count"] unsignedIntValue];
            kernel.samplers = [record[@"sampler_count"] unsignedIntValue];
            if (!kernel.resources || kernel.resources > 256 || kernel.samplers > 16) {
                return fail(error, "ANR descriptor table size invalid");
            }
            unsigned tableSlot = 0;
            for (NSDictionary *b in record[@"bindings"]) {
                const bool isUniform = [b[@"uniform"] boolValue];
                const bool isSampler = [b[@"sampler"] boolValue];
                BindKind kind = isSampler ? BindKind::Sampler : (isUniform ? BindKind::Uniform : BindKind::Buffer);
                kernel.bindings.push_back({
                    utf8(b[@"name"]),
                    kernel.native ? tableSlot++ : [b[@"slot"] unsignedIntValue],
                    kind,
                    false,
                });
            }
            for (const auto &binding : kernel.bindings) {
                const unsigned limit = binding.kind == BindKind::Sampler ? kernel.samplers : kernel.resources;
                if (binding.index >= limit) return fail(error, "ANR descriptor slot out of bounds");
            }
            kernels.emplace(utf8(key), std::move(kernel));
        }

        for (unsigned slot = 0; slot < 4; ++slot) {
            for (NSString *key in stageRecords) {
                if (!stage_wanted(key, phaseMode)) continue;
                auto &slotStages = stages[slot][utf8(key)];
                for (NSDictionary *record in stageRecords[key]) {
                    Stage stage;
                    stage.label = utf8(record[@"label"] ?: @"");
                    stage.kernel = utf8(record[@"kernel"]);
                    stage.groups = extent_from_array(record[@"groups"]);
                    stage.threads = extent_from_array(record[@"threads"]);
                    auto it = kernels.find(stage.kernel);
                    if (it == kernels.end()) return fail(error, "ANR unknown stage kernel");
                    Kernel &kernel = it->second;
                    if (stage.threads.width * stage.threads.height * stage.threads.depth > kernel.pipeline.maxTotalThreadsPerThreadgroup) {
                        return fail(error, "ANR threadgroup exceeds pipeline limit");
                    }
                    if (!safe_path(record[@"uniforms"]) || !files[record[@"uniforms"]]) {
                        return fail(error, "ANR unlisted uniform asset");
                    }
                    NSData *data = load_file(directory, record[@"uniforms"]);
                    if (!data || data.length > 4096) return fail(error, "ANR uniform size invalid");
                    stage.constants = alloc_shared(data.length);
                    if (!add_resident(stage.constants, error)) return false;
                    std::memset(stage.constants.contents, 0, stage.constants.length);
                    std::memcpy(stage.constants.contents, data.bytes, data.length);
                    for (NSString *name in record[@"buffers"]) {
                        unsigned index = [record[@"buffers"][name] unsignedIntValue];
                        if (index >= buffers.size()) return fail(error, "ANR buffer reference invalid");
                        stage.buffers.emplace(utf8(name), index);
                    }
                    for (NSString *name in record[@"images"]) {
                        const std::string target = utf8(record[@"images"][name]);
                        stage.images.emplace(utf8(name), target);
                        if (target == "color" || target == "depth" || target == "motion" || target == "output") {
                            stage.hasExternalImage = true;
                        }
                    }
                    MTL4ArgumentTableDescriptor *td = [MTL4ArgumentTableDescriptor new];
                    td.maxBufferBindCount = 3;
                    stage.arguments = [device newArgumentTableWithDescriptor:td error:&metalError];
                    if (!stage.arguments) return fail(error, "ANR argument table: " + utf8(metalError.localizedDescription));
                    stage.resources = alloc_shared(kernel.resources * sizeof(Descriptor));
                    stage.samplers = alloc_shared(kernel.samplers * sizeof(Descriptor));
                    stage.root = alloc_shared(16);
                    if (!add_resident(stage.resources, error) || !add_resident(stage.samplers, error)
                        || !add_resident(stage.root, error)) {
                        return false;
                    }
                    std::memset(stage.resources.contents, 0, stage.resources.length);
                    std::memset(stage.samplers.contents, 0, stage.samplers.length);
                    uint64_t pointers[] = {stage.resources.gpuAddress, stage.samplers.gpuAddress};
                    std::memcpy(stage.root.contents, pointers, sizeof(pointers));
                    slotStages.push_back(std::move(stage));
                }
            }
        }

        NSDictionary *inputs = manifest[@"inputs"];
        inputColor0 = [inputs[@"color0"] unsignedIntValue];
        inputColor1 = [inputs[@"color1"] unsignedIntValue];
        inputDepth0 = [inputs[@"depth0"] unsignedIntValue];
        inputDepth1 = [inputs[@"depth1"] unsignedIntValue];
        inputFrame = [inputs[@"frame"] unsignedIntValue];
        for (NSArray *pair in manifest[@"swaps"]) {
            bufferSwaps.emplace_back([pair[0] unsignedIntValue], [pair[1] unsignedIntValue]);
        }
        for (id<MTLBuffer> b : buffers) staticMetal3Resources.push_back(b);
        for (unsigned slot = 0; slot < 4; ++slot) staticMetal3Resources.push_back(frameBuffers[slot]);
        staticMetal3Resources.push_back(empty);

        [residency commit];
        initialized = true;
        return true;
    }

    // Plan offsets once for all wanted phases. A phase can run on its own, or
    // after another phase with no intervening pair. Pair features must survive both.
    bool plan_scratch(NSDictionary *variant, PhaseMode mode, const std::vector<size_t> &sizes,
                      std::vector<size_t> &offsets, size_t &arenaBytes, std::string &error) {
        using Interval = std::pair<size_t, size_t>; // inclusive dispatch ordinals
        std::vector<std::vector<Interval>> spans(sizes.size());
        NSDictionary *groups = variant[@"stages"];
        NSMutableArray *order = [NSMutableArray arrayWithObject:@"pair"];
        for (NSString *key in groups) {
            if (![key isEqual:@"pair"] && stage_wanted(key, mode)) [order addObject:key];
        }
        size_t ordinal = 0;
        const size_t pairEnd = [groups[@"pair"] count];
        for (NSString *key in order) {
            const size_t start = ordinal;
            for (NSDictionary *stage in groups[key]) {
                for (NSNumber *number in [stage[@"buffers"] allValues]) {
                    unsigned index = number.unsignedIntValue;
                    if (index >= sizes.size()) return fail(error, "ANR v2 invalid scratch reference");
                    auto &uses = spans[index];
                    if (uses.empty() || uses.back().first < start) uses.emplace_back(ordinal, ordinal);
                    else uses.back().second = ordinal;
                }
                ++ordinal;
            }
        }
        std::set<unsigned> dedicated;
        for (NSNumber *index in [variant[@"inputs"] allValues]) dedicated.insert(index.unsignedIntValue);
        for (NSArray *pair in variant[@"swaps"]) {
            for (NSNumber *index in pair) dedicated.insert(index.unsignedIntValue);
        }
        std::vector<unsigned> scratch;
        for (NSDictionary *record in variant[@"buffers"]) {
            unsigned index = [record[@"index"] unsignedIntValue];
            if (![record[@"kind"] isEqual:@"scratch"] || dedicated.count(index)) continue;
            auto &uses = spans[index];
            if (uses.size() > 1 && uses.front().first < pairEnd) {
                uses = {{uses.front().first, ordinal - 1}};
            }
            scratch.push_back(index);
        }
        // Largest-first, lowest fitting address among already-placed live neighbors.
        std::stable_sort(scratch.begin(), scratch.end(), [&](unsigned a, unsigned b) { return sizes[a] > sizes[b]; });
        std::vector<unsigned> placed;
        for (unsigned index : scratch) {
            std::vector<Interval> occupied; // half-open byte ranges
            for (unsigned other : placed) {
                bool overlaps = false;
                for (auto a : spans[index]) {
                    for (auto b : spans[other]) {
                        if (a.first <= b.second && b.first <= a.second) { overlaps = true; break; }
                    }
                    if (overlaps) break;
                }
                if (overlaps) occupied.emplace_back(offsets[other], offsets[other] + sizes[other]);
            }
            std::sort(occupied.begin(), occupied.end());
            size_t offset = 0;
            for (auto range : occupied) {
                if (offset + sizes[index] <= range.first) break;
                offset = std::max(offset, (range.second + 255) & ~size_t(255));
            }
            offsets[index] = offset;
            arenaBytes = std::max(arenaBytes, offset + sizes[index]);
            placed.push_back(index);
        }
        return true;
    }

    bool init_v2(std::string &error, unsigned reqWidth, unsigned reqHeight, Encoding reqEncoding, PhaseMode phaseMode) {
        isV2 = true;
        releaseName = utf8(manifest[@"release"] ?: @"FG-V.B20");
        backendName = utf8(manifest[@"backend"] ?: @"packed");
        encoding = reqEncoding;

        const std::string ckpt = utf8(manifest[@"checkpoint_id"]);
        const bool validPacked = backendName == "packed"
            && ckpt == "b7481951e4f54e4f2d534ec41c8f3d8e721f4fd22839d2ba2bd236cbd28bea64";
        const bool validDp4a = backendName == "dp4a"
            && ckpt == "bef1b3e76649887e6cb36f178a3011752d51b4e868f22e70f7c4a5a8d62408f4";
        if (!validPacked && !validDp4a) {
            return fail(error, "ANR v2 unsupported backend/checkpoint");
        }

        NSDictionary *files = manifest[@"files"];
        if (!verify_inventory(files, error)) return false;

        NSDictionary *limits = manifest[@"sizes"];
        NSArray *minSz = limits[@"min"], *maxSz = limits[@"max"];
        const unsigned minW = [minSz[0] unsignedIntValue], minH = [minSz[1] unsignedIntValue];
        const unsigned maxW = [maxSz[0] unsignedIntValue], maxH = [maxSz[1] unsignedIntValue];
        if (reqWidth < minW || reqHeight < minH || reqWidth > maxW || reqHeight > maxH) {
            char msg[160];
            snprintf(msg, sizeof msg, "ANR v2 size %ux%u outside supported %ux%u..%ux%u",
                     reqWidth, reqHeight, minW, minH, maxW, maxH);
            return fail(error, msg);
        }
        width = reqWidth;
        height = reqHeight;

        NSDictionary *encMap = manifest[@"encodings"];
        NSDictionary *encEntry = encMap[ns(encoding_key(reqEncoding))];
        if (![encEntry isKindOfClass:NSDictionary.class]) {
            return fail(error, std::string("ANR v2 missing encoding: ") + encoding_key(reqEncoding));
        }

        NSString *progFile = encEntry[@"program"];
        NSString *weightsFile = encEntry[@"weights"];
        NSString *metallibFile = manifest[@"metallib"];
        if (!safe_path(progFile) || !safe_path(weightsFile) || !safe_path(metallibFile)
            || !files[progFile] || !files[weightsFile] || !files[metallibFile]) {
            return fail(error, "ANR v2 unlisted program/weights/metallib asset");
        }

        NSData *progData = load_file(directory, progFile);
        NSDictionary *program = progData ? [NSJSONSerialization JSONObjectWithData:progData options:0 error:nil] : nil;
        if (![program isKindOfClass:NSDictionary.class] || [program[@"version"] intValue] != 3) {
            return fail(error, "ANR v2 invalid dynamic program");
        }
        NSData *weights = load_file(directory, weightsFile);
        if (!weights || !weights.length) return fail(error, "ANR v2 missing weights");

        // Evaluate dynamic resolution variants and find the unique matching variant.
        NSDictionary *selectedVariant = nil;
        std::vector<int64_t> exprVals;
        for (NSDictionary *variant in program[@"variants"]) {
            std::vector<int64_t> candidateVals;
            if (!eval_exprs(variant[@"exprs"], width, height, candidateVals, error)) {
                return false;
            }
            if (check_guards(variant[@"guards"], candidateVals)) {
                selectedVariant = variant;
                exprVals = std::move(candidateVals);
                break;
            }
        }
        if (!selectedVariant) {
            char msg[128];
            snprintf(msg, sizeof msg, "ANR v2 no graph variant matches %ux%u", width, height);
            return fail(error, msg);
        }

        MTLResidencySetDescriptor *rd = [MTLResidencySetDescriptor new];
        rd.initialCapacity = 1600;
        NSError *metalError = nil;
        residency = [device newResidencySetWithDescriptor:rd error:&metalError];
        if (!residency) return fail(error, "ANR residency: " + utf8(metalError.localizedDescription));

        // Allocate Private GPU buffers and upload weights / clear scratch once on a blit pass.
        NSArray *bufSpecs = selectedVariant[@"buffers"];
        if (![bufSpecs isKindOfClass:NSArray.class] || bufSpecs.count < 64 || bufSpecs.count > 256) {
            return fail(error, "ANR v2 invalid buffer inventory");
        }
        buffers.resize(bufSpecs.count);

        id<MTLBuffer> stagingWeights = [device newBufferWithBytes:weights.bytes
                                                           length:weights.length
                                                          options:MTLResourceStorageModeShared];
        if (!stagingWeights) return fail(error, "ANR v2 weight staging allocation failed");

        id<MTLCommandQueue> initQueue = [device newCommandQueue];
        id<MTLCommandBuffer> initCmd = [initQueue commandBuffer];
        id<MTLBlitCommandEncoder> initBlit = [initCmd blitCommandEncoder];
        if (!initBlit) return fail(error, "ANR v2 init blit encoder failed");

        std::vector<size_t> sizes(buffers.size());
        std::vector<size_t> offsets(buffers.size(), SIZE_MAX); // dedicated unless packed below
        for (NSDictionary *record in bufSpecs) {
            unsigned index = [record[@"index"] unsignedIntValue];
            int64_t sizeVal = 0;
            if (!resolve_int(record[@"size"], exprVals, sizeVal) || sizeVal <= 0 || sizeVal > (1LL << 30)) {
                return fail(error, "ANR v2 invalid resolved buffer size");
            }
            if (index >= buffers.size() || sizes[index]) {
                return fail(error, "ANR v2 duplicate/invalid buffer index");
            }
            sizes[index] = (size_t)sizeVal;
        }
        const char *aliasEnv = getenv("FSR4_ANR_SCRATCH_ALIAS");
        if ((!aliasEnv || strcmp(aliasEnv, "0") != 0) && !getenv("ANR_DUMP_BUFFERS")) {
            size_t arenaBytes = 0;
            if (!plan_scratch(selectedVariant, phaseMode, sizes, offsets, arenaBytes, error)) return false;
            if (arenaBytes) {
                scratchArena = alloc_private((arenaBytes + 255) & ~size_t(255));
                if (!add_resident(scratchArena, error)) return false;
                [initBlit fillBuffer:scratchArena range:NSMakeRange(0, scratchArena.length) value:0];
                staticMetal3Resources.push_back(scratchArena);
            }
        }
        for (NSDictionary *record in bufSpecs) {
            unsigned index = [record[@"index"] unsignedIntValue];
            if (offsets[index] != SIZE_MAX) {
                buffers[index] = scratchArena;
                continue;
            }
            const size_t bytes = sizes[index];
            buffers[index] = alloc_private(bytes);
            if (!add_resident(buffers[index], error)) return false;

            if ([record[@"kind"] isEqual:@"weight"]) {
                size_t offset = [record[@"data_offset"] unsignedLongLongValue];
                if (offset > weights.length || bytes > weights.length - offset) {
                    return fail(error, "ANR v2 weight span out of bounds");
                }
                [initBlit copyFromBuffer:stagingWeights
                            sourceOffset:offset
                                toBuffer:buffers[index]
                       destinationOffset:0
                                    size:bytes];
                if (buffers[index].length > bytes) {
                    [initBlit fillBuffer:buffers[index]
                                   range:NSMakeRange(bytes, buffers[index].length - bytes)
                                   value:0];
                }
            } else {
                [initBlit fillBuffer:buffers[index]
                               range:NSMakeRange(0, buffers[index].length)
                               value:0];
                if ([record[@"kind"] isEqual:@"persistent"]) {
                    persistentBuffers.push_back(index);
                }
            }
        }
        [initBlit endEncoding];
        [initCmd commit];
        [initCmd waitUntilCompleted];
        if (initCmd.status != MTLCommandBufferStatusCompleted) {
            return fail(error, "ANR v2 weight upload GPU failure");
        }

        // Allocate internal textures (tex.color0/1 with R32Uint views, tex.gain0/1).
        NSDictionary *texSpecs = selectedVariant[@"textures"];
        if ([texSpecs isKindOfClass:NSDictionary.class]) {
            for (NSString *texName in texSpecs) {
                NSDictionary *spec = texSpecs[texName];
                MTLPixelFormat fmt = pixel_format_from_name(spec[@"format"]);
                if (fmt == MTLPixelFormatInvalid) {
                    return fail(error, "ANR v2 unsupported internal texture format: " + utf8(spec[@"format"]));
                }
                int64_t tw = width, th = height;
                if (spec[@"width"] && !resolve_int(spec[@"width"], exprVals, tw)) {
                    return fail(error, "ANR v2 bad texture width expr");
                }
                if (spec[@"height"] && !resolve_int(spec[@"height"], exprVals, th)) {
                    return fail(error, "ANR v2 bad texture height expr");
                }
                if (tw <= 0 || th <= 0 || tw > 16384 || th > 16384) {
                    return fail(error, "ANR v2 internal texture dimensions invalid");
                }
                MTLTextureDescriptor *td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:fmt
                                                                                              width:(NSUInteger)tw
                                                                                             height:(NSUInteger)th
                                                                                          mipmapped:NO];
                td.storageMode = MTLStorageModePrivate;
                td.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
                NSString *storageFmt = spec[@"storage_format"];
                if ([storageFmt isEqual:@"r32_uint"]) {
                    td.usage |= MTLTextureUsagePixelFormatView;
                }
                InternalTexture itex;
                itex.main = [device newTextureWithDescriptor:td];
                if (!add_resident(itex.main, error)) return false;
                staticMetal3Resources.push_back(itex.main);
                if ([storageFmt isEqual:@"r32_uint"]) {
                    itex.r32uintView = [itex.main newTextureViewWithPixelFormat:MTLPixelFormatR32Uint];
                    if (!add_resident(itex.r32uintView, error)) return false;
                    staticMetal3Resources.push_back(itex.r32uintView);
                }
                internalTextures.emplace(utf8(texName), itex);
            }
        }

        for (unsigned slot = 0; slot < 4; ++slot) {
            frameBuffers[slot] = alloc_shared(204);
            if (!add_resident(frameBuffers[slot], error)) return false;
            MTLResidencySetDescriptor *bd = [MTLResidencySetDescriptor new];
            bd.initialCapacity = 8;
            borrowedSets[slot] = [device newResidencySetWithDescriptor:bd error:&metalError];
            if (!borrowedSets[slot]) return fail(error, "ANR frame residency allocation failed");
        }
        empty = alloc_shared(256);
        if (!add_resident(empty, error)) return false;
        std::memset(empty.contents, 0, empty.length);

        MTLSamplerDescriptor *sd = [MTLSamplerDescriptor new];
        sd.supportArgumentBuffers = YES;
        sd.minFilter = sd.magFilter = MTLSamplerMinMagFilterLinear;
        sd.sAddressMode = sd.tAddressMode = MTLSamplerAddressModeClampToZero;
        sampler = [device newSamplerStateWithDescriptor:sd];

        // Load single combined metallib (cached per device + sha256).
        const std::string libSha = utf8(files[metallibFile][@"sha256"]);
        id<MTLLibrary> library = nil;
        {
            std::lock_guard<std::mutex> lock(g_pipelineMutex);
            const auto libKey = std::make_pair((uintptr_t)(__bridge void *)device, libSha);
            auto foundLib = g_libraryCache.find(libKey);
            if (foundLib != g_libraryCache.end()) {
                library = foundLib->second;
            } else {
                NSURL *url = [NSURL fileURLWithPath:[directory stringByAppendingPathComponent:metallibFile]];
                library = [device newLibraryWithURL:url error:&metalError];
                if (library) g_libraryCache.emplace(libKey, library);
            }
        }
        if (!library) return fail(error, "ANR v2 metallib load failed: " + utf8(metalError.localizedDescription));

        // Optional Metal 4 MPP library for Apple10 / M5+ GPUs (NAX matrix engines).
        const char *mppEnv = getenv("FSR4_ANR_MPP");
        const bool wantMpp = [device supportsFamily:MTLGPUFamilyApple10] && (!mppEnv || strcmp(mppEnv, "0") != 0);
        NSString *metallibMppFile = manifest[@"metallib_mpp"];
        NSDictionary *mppKernelMap = manifest[@"mpp_kernels"];
        id<MTLLibrary> mppLibrary = nil;
        std::string mppLibSha;
        if (wantMpp && [metallibMppFile isKindOfClass:NSString.class]
            && [mppKernelMap isKindOfClass:NSDictionary.class]
            && safe_path(metallibMppFile) && files[metallibMppFile]) {
            mppLibSha = utf8(files[metallibMppFile][@"sha256"]);
            std::lock_guard<std::mutex> lock(g_pipelineMutex);
            const auto mppKey = std::make_pair((uintptr_t)(__bridge void *)device, mppLibSha);
            auto foundMpp = g_libraryCache.find(mppKey);
            if (foundMpp != g_libraryCache.end()) {
                mppLibrary = foundMpp->second;
            } else {
                NSURL *url = [NSURL fileURLWithPath:[directory stringByAppendingPathComponent:metallibMppFile]];
                mppLibrary = [device newLibraryWithURL:url error:&metalError];
                if (mppLibrary) g_libraryCache.emplace(mppKey, mppLibrary);
            }
        }

        NSDictionary *kernelRecords = manifest[@"kernels"];
        NSDictionary *stageRecords = selectedVariant[@"stages"];
        NSMutableSet *neededKernels = [NSMutableSet set];
        std::map<std::string, bool> kernelMul32;
        std::map<std::string, NSUInteger> kernelMaxThreads;
        for (NSString *key in stageRecords) {
            if (!stage_wanted(key, phaseMode)) continue;
            for (NSDictionary *stage in stageRecords[key]) {
                NSString *kstr = stage[@"kernel"];
                [neededKernels addObject:kstr];
                MTLSize th = extent_from_array(stage[@"threads"]);
                const NSUInteger count = th.width * th.height * th.depth;
                const bool mul32 = (count % 32u) == 0u;
                const std::string ks = utf8(kstr);
                kernelMaxThreads[ks] = std::max(kernelMaxThreads[ks], count);
                auto it = kernelMul32.find(ks);
                if (it == kernelMul32.end()) kernelMul32[ks] = mul32;
                else it->second = it->second && mul32;
            }
        }

        for (NSString *key in neededKernels) {
            NSDictionary *record = kernelRecords[key];
            if (![record isKindOfClass:NSDictionary.class]) {
                return fail(error, "ANR v2 missing kernel record: " + utf8(key));
            }
            Kernel kernel;
            NSString *entryName = record[@"entry"];
            id<MTLLibrary> useLib = library;
            std::string useSha = libSha;
            if (mppLibrary && [mppKernelMap[key] isKindOfClass:NSString.class]) {
                id<MTLFunction> mppFn = [mppLibrary newFunctionWithName:mppKernelMap[key]];
                if (mppFn) {
                    entryName = mppKernelMap[key];
                    useLib = mppLibrary;
                    useSha = mppLibSha;
                    mppActive = true;
                }
            }
            id<MTLFunction> function = [useLib newFunctionWithName:entryName];
            if (!function) return fail(error, "ANR v2 missing function: " + utf8(entryName));
            const bool isMul32 = kernelMul32[utf8(key)];
            const NSUInteger maxThreads = kernelMaxThreads.at(utf8(key));
            const auto cacheKey = std::make_pair((uintptr_t)(__bridge void *)device,
                                                 useSha + ":" + utf8(entryName) + (isMul32 ? ":m32" : ":any")
                                                 + ":t" + std::to_string(maxThreads));
            {
                std::lock_guard<std::mutex> lock(g_pipelineMutex);
                auto cached = g_pipelineCache.find(cacheKey);
                if (cached != g_pipelineCache.end()) {
                    kernel.pipeline = cached->second;
                } else {
                    MTLComputePipelineDescriptor *pd = [MTLComputePipelineDescriptor new];
                    pd.computeFunction = function;
                    pd.maxTotalThreadsPerThreadgroup = maxThreads;
                    pd.threadGroupSizeIsMultipleOfThreadExecutionWidth = isMul32 ? YES : NO;
                    kernel.pipeline = [device newComputePipelineStateWithDescriptor:pd
                                                                            options:MTLPipelineOptionNone
                                                                         reflection:nil
                                                                              error:&metalError];
                    if (kernel.pipeline) g_pipelineCache.emplace(cacheKey, kernel.pipeline);
                }
            }
            if (!kernel.pipeline) {
                return fail(error, "ANR v2 pipeline " + utf8(key) + ": " + utf8(metalError.localizedDescription));
            }
            kernel.argEncoder = [function newArgumentEncoderWithBufferIndex:0];
            if (!kernel.argEncoder) return fail(error, "ANR v2 argument encoder missing for: " + utf8(key));
            kernel.slotCount = [record[@"slot_count"] unsignedIntValue];
            kernel.uniformBytes = [record[@"uniform_bytes"] unsignedIntValue];
            if (!kernel.slotCount || kernel.slotCount > 64 || kernel.uniformBytes > 4096) {
                return fail(error, "ANR v2 kernel slot/uniform size invalid");
            }
            for (NSDictionary *b in record[@"bindings"]) {
                NSString *kstr = b[@"kind"];
                BindKind bk = BindKind::Buffer;
                if ([kstr isEqual:@"uniform"]) bk = BindKind::Uniform;
                else if ([kstr isEqual:@"texture"]) bk = BindKind::Texture;
                else if ([kstr isEqual:@"storage_texture"]) bk = BindKind::StorageTexture;
                else if ([kstr isEqual:@"sampler"]) bk = BindKind::Sampler;
                const bool writable = b[@"writable"] ? [b[@"writable"] boolValue]
                                                     : (bk == BindKind::Buffer || bk == BindKind::StorageTexture);
                kernel.bindings.push_back({
                    utf8(b[@"name"]),
                    [b[@"slot"] unsignedIntValue],
                    bk,
                    [b[@"r32uint"] boolValue] != NO,
                    writable,
                });
            }
            kernels.emplace(utf8(key), std::move(kernel));
        }

        // Build per-slot stages with resolved dispatch sizes and packed uniform blocks.
        for (unsigned slot = 0; slot < 4; ++slot) {
            for (NSString *key in stageRecords) {
                if (!stage_wanted(key, phaseMode)) continue;
                auto &slotStages = stages[slot][utf8(key)];
                for (NSDictionary *record in stageRecords[key]) {
                    Stage stage;
                    stage.label = utf8(record[@"label"] ?: @"");
                    stage.kernel = utf8(record[@"kernel"]);
                    auto it = kernels.find(stage.kernel);
                    if (it == kernels.end()) return fail(error, "ANR v2 unknown stage kernel");
                    Kernel &kernel = it->second;

                    NSArray *gArr = record[@"groups"];
                    int64_t gx = 0, gy = 0, gz = 0;
                    if (gArr.count != 3 || !resolve_int(gArr[0], exprVals, gx)
                        || !resolve_int(gArr[1], exprVals, gy) || !resolve_int(gArr[2], exprVals, gz)
                        || gx <= 0 || gy <= 0 || gz <= 0) {
                        return fail(error, "ANR v2 invalid resolved dispatch groups");
                    }
                    stage.groups = MTLSizeMake((NSUInteger)gx, (NSUInteger)gy, (NSUInteger)gz);
                    stage.threads = extent_from_array(record[@"threads"]);
                    if (stage.threads.width * stage.threads.height * stage.threads.depth > kernel.pipeline.maxTotalThreadsPerThreadgroup) {
                        return fail(error, "ANR v2 threadgroup exceeds pipeline limit");
                    }

                    // Pack dynamic uniform block.
                    stage.constants = alloc_shared(std::max(kernel.uniformBytes, 16u));
                    if (!add_resident(stage.constants, error)) return false;
                    std::memset(stage.constants.contents, 0, stage.constants.length);
                    auto *uBytes = (uint8_t *)stage.constants.contents;
                    for (NSArray *item in record[@"uniforms"]) {
                        const unsigned offset = [item[0] unsignedIntValue];
                        const int typeCode = [item[1] intValue];
                        if (offset + 4 > stage.constants.length) {
                            return fail(error, "ANR v2 uniform offset out of bounds");
                        }
                        if (typeCode == 2) {
                            double dv = 0.0;
                            if (!resolve_double(item[2], exprVals, dv) || !std::isfinite(dv)) {
                                return fail(error, "ANR v2 invalid float uniform");
                            }
                            float fv = (float)dv;
                            std::memcpy(uBytes + offset, &fv, 4);
                        } else if (typeCode == 1) {
                            int64_t iv = 0;
                            if (!resolve_int(item[2], exprVals, iv)) return fail(error, "ANR v2 invalid int uniform");
                            int32_t v32 = (int32_t)iv;
                            std::memcpy(uBytes + offset, &v32, 4);
                        } else {
                            int64_t iv = 0;
                            if (!resolve_int(item[2], exprVals, iv) || iv < 0) return fail(error, "ANR v2 invalid uint uniform");
                            uint32_t v32 = (uint32_t)iv;
                            std::memcpy(uBytes + offset, &v32, 4);
                        }
                    }

                    for (NSString *name in record[@"buffers"]) {
                        unsigned index = [record[@"buffers"][name] unsignedIntValue];
                        if (index >= buffers.size()) return fail(error, "ANR v2 buffer reference invalid");
                        stage.bufferOffsets.emplace(utf8(name), offsets[index] == SIZE_MAX ? 0 : offsets[index]);
                        stage.buffers.emplace(utf8(name), index);
                    }
                    for (NSString *name in record[@"images"]) {
                        const std::string target = utf8(record[@"images"][name]);
                        stage.images.emplace(utf8(name), target);
                    }

                    for (const Binding &binding : kernel.bindings) {
                        if (binding.kind != BindKind::Texture && binding.kind != BindKind::StorageTexture) continue;
                        auto image = stage.images.find(binding.name);
                        if (image == stage.images.end()) return fail(error, "ANR v2 unbound texture: " + binding.name);
                        if (internalTextures.count(image->second)) continue;
                        const std::array<std::string, 4> sources = {"color", "depth", "motion", "output"};
                        auto source = std::find(sources.begin(), sources.end(), image->second);
                        if (source == sources.end()) return fail(error, "ANR v2 unknown texture: " + image->second);
                        stage.externalImages.push_back({binding.index, unsigned(source - sources.begin())});
                    }
                    const NSUInteger argLen = std::max<NSUInteger>(kernel.argEncoder.encodedLength, 64);
                    MTL4ArgumentTableDescriptor *td = [MTL4ArgumentTableDescriptor new];
                    td.maxBufferBindCount = 1;
                    for (auto &args : stage.parityArgs) {
                        args.buffer = alloc_shared(argLen);
                        if (!add_resident(args.buffer, error)) return false;
                        args.table = [device newArgumentTableWithDescriptor:td error:&metalError];
                        if (!args.table) return fail(error, "ANR v2 argument table: " + utf8(metalError.localizedDescription));
                        [args.table setAddress:args.buffer.gpuAddress atIndex:0];
                    }

                    slotStages.push_back(std::move(stage));
                }

                // Compute minimal barrier placement from exact read/write resource DAG.
                std::set<std::string> dirty, readSinceBar;
                for (size_t i = 0; i < slotStages.size(); ++i) {
                    Stage &st = slotStages[i];
                    const Kernel &k = kernels.at(st.kernel);
                    std::set<std::string> reads, writes;
                    for (const Binding &b : k.bindings) {
                        if (b.kind == BindKind::Buffer) {
                            auto bit = st.buffers.find(b.name);
                            if (bit == st.buffers.end()) continue;
                            // Logical buffer IDs alone miss WAR/WAW on reused arena bytes.
                            std::string id = offsets[bit->second] == SIZE_MAX ? "b:" + std::to_string(bit->second) : "arena";
                            if (b.writable) { writes.insert(id); reads.insert(id); }
                            else { reads.insert(id); }
                        } else if (b.kind == BindKind::Texture || b.kind == BindKind::StorageTexture) {
                            auto iit = st.images.find(b.name);
                            if (iit == st.images.end()) continue;
                            std::string id = "i:" + iit->second;
                            if (b.writable) { writes.insert(id); }
                            else { reads.insert(id); }
                        }
                    }
                    bool hazard = false;
                    for (const auto &r : reads) if (dirty.count(r)) { hazard = true; break; }
                    if (!hazard) for (const auto &w : writes) if (dirty.count(w) || readSinceBar.count(w)) { hazard = true; break; }
                    st.needsBarrier = (i > 0 && hazard);
                    if (st.needsBarrier) {
                        dirty.clear();
                        readSinceBar.clear();
                    }
                    dirty.insert(writes.begin(), writes.end());
                    readSinceBar.insert(reads.begin(), reads.end());
                }
            }
        }

        NSDictionary *inputs = selectedVariant[@"inputs"];
        inputColor0 = [inputs[@"color0"] unsignedIntValue];
        inputColor1 = [inputs[@"color1"] unsignedIntValue];
        inputDepth0 = [inputs[@"depth0"] unsignedIntValue];
        inputDepth1 = [inputs[@"depth1"] unsignedIntValue];
        inputFrame = [inputs[@"frame"] unsignedIntValue];
        for (NSArray *pair in selectedVariant[@"swaps"]) {
            bufferSwaps.emplace_back([pair[0] unsignedIntValue], [pair[1] unsignedIntValue]);
        }
        for (NSArray *pair in selectedVariant[@"texture_swaps"]) {
            textureSwaps.emplace_back(utf8(pair[0]), utf8(pair[1]));
        }

        for (id<MTLBuffer> b : buffers) if (b != scratchArena) staticMetal3Resources.push_back(b);
        for (unsigned slot = 0; slot < 4; ++slot) staticMetal3Resources.push_back(frameBuffers[slot]);
        staticMetal3Resources.push_back(empty);

        const char *devBar = getenv("FSR4_ANR_DEVICE_BARRIER");
        deviceInternalBarrier = !devBar || strcmp(devBar, "0") != 0;
        const char *dagEnv = getenv("FSR4_ANR_DAG_BARRIER");
        dagBarriers = !dagEnv || strcmp(dagEnv, "0") != 0;

        [residency commit];
        initialized = true;
        return true;
    }
};

Runtime::Runtime(id<MTLDevice> device, const std::string &directory)
    : impl(std::make_unique<Impl>()) {
    impl->device = device;
    impl->directory = ns(directory);
}

Runtime::~Runtime() = default;

unsigned Runtime::width() const { return impl->width; }
unsigned Runtime::height() const { return impl->height; }
bool Runtime::is_v2() const { return impl->isV2; }
bool Runtime::mpp_active() const { return impl->mppActive; }
Encoding Runtime::encoding() const { return impl->encoding; }
std::string Runtime::release_name() const { return impl->releaseName; }
std::string Runtime::backend_name() const { return impl->backendName; }

id<MTLBuffer> Runtime::diagnostic_buffer(unsigned index) const {
    return index < impl->buffers.size() ? impl->buffers[index] : nil;
}

bool Runtime::initialize(std::string &error, bool twoXOnly) {
    return initialize(error, 1920, 1080, Encoding::PQ10, twoXOnly ? PhaseMode::TwoX : PhaseMode::All);
}

bool Runtime::initialize(std::string &error, unsigned width, unsigned height,
                         Encoding encoding, PhaseMode phaseMode) {
    @autoreleasepool {
        auto &r = *impl;
        if (r.initialized) return true;
        NSData *md = load_file(r.directory, @"manifest.json");
        if (!md || md.length > 32u * 1024u * 1024u) {
            return r.fail(error, "ANR missing/oversized manifest");
        }
        r.manifest = [NSJSONSerialization JSONObjectWithData:md options:0 error:nil];
        if (![r.manifest isKindOfClass:NSDictionary.class]) {
            return r.fail(error, "ANR malformed manifest");
        }
        NSString *schema = r.manifest[@"schema"];
        if ([schema isEqual:@"fsr4_anr_runtime_v2"]) {
            return r.init_v2(error, width, height, encoding, phaseMode);
        }
        if ([schema isEqual:@"fsr4_anr_runtime_v1"]) {
            return r.init_v1(error, phaseMode);
        }
        return r.fail(error, "ANR unsupported runtime schema");
    }
}

bool Runtime::encode_pair(id<MTL4CommandBuffer> command, id<MTLTexture> color,
                          id<MTLTexture> depth, id<MTLTexture> motion,
                          const void *frame, size_t frameBytes, std::string &error) {
    return pair(command, true, color, depth, motion, frame, frameBytes, error);
}

bool Runtime::encode_pair_classic(id<MTLCommandBuffer> command, id<MTLTexture> color,
                                  id<MTLTexture> depth, id<MTLTexture> motion,
                                  const void *frame, size_t frameBytes, std::string &error) {
    return pair(command, false, color, depth, motion, frame, frameBytes, error);
}

bool Runtime::pair(id command, bool metal4, id<MTLTexture> color, id<MTLTexture> depth,
                   id<MTLTexture> motion, const void *frame, size_t frameBytes, std::string &error) {
    auto &r = *impl;
    if (!r.initialized) return r.fail(error, "ANR runtime not initialized");
    if (!command || [command device] != r.device || !color || !depth || !motion || !frame || frameBytes != 204) {
        return r.fail(error, "ANR pair contract mismatch");
    }
    if (metal4) {
        r.update_borrowed_residency(color, depth, motion, r.output);
    }
    r.inputColor = color;
    r.inputDepth = depth;
    r.inputMotion = motion;
    r.output = nil;
    if (r.buffers.at(r.inputFrame) != r.frameBuffers[r.frameSlot]) {
        r.buffers.at(r.inputFrame) = r.frameBuffers[r.frameSlot];
        ++r.bufferRevision;
    }
    std::memcpy(r.frameBuffers[r.frameSlot].contents, frame, frameBytes);
    return r.encode_group(command, metal4, r.stages[r.frameSlot].at("pair"), error);
}

bool Runtime::encode_interpolation(id<MTL4CommandBuffer> command, double phase,
                                   id<MTLTexture> output, std::string &error) {
    return interpolate(command, true, phase, output, error);
}

bool Runtime::encode_interpolation_classic(id<MTLCommandBuffer> command, double phase,
                                           id<MTLTexture> output, std::string &error) {
    return interpolate(command, false, phase, output, error);
}

bool Runtime::interpolate(id command, bool metal4, double phase, id<MTLTexture> output, std::string &error) {
    auto &r = *impl;
    if (!r.initialized || !command || [command device] != r.device || !output || output.device != r.device) {
        return r.fail(error, "ANR interpolation contract mismatch");
    }
    char name[64];
    snprintf(name, sizeof name, "phase:%.6g:image", phase);
    auto found = r.stages[r.frameSlot].find(name);
    if (found == r.stages[r.frameSlot].end()) {
        return r.fail(error, std::string("ANR unsupported phase: ") + name);
    }
    r.output = output;
    if (metal4) {
        r.update_borrowed_residency(r.inputColor, r.inputDepth, r.inputMotion, output);
    }
    return r.encode_group(command, metal4, found->second, error);
}

bool Runtime::encode_frame(id<MTL4CommandBuffer> command, id<MTLTexture> color,
                           id<MTLTexture> depth, id<MTLTexture> motion,
                           const void *frame, size_t frameBytes,
                           const double *phases, id<MTLTexture> const *outputs,
                           unsigned phaseCount, std::string &error, const FrameExtras &extras) {
    return frame_combined(command, true, color, depth, motion, frame, frameBytes, phases, outputs, phaseCount, error, extras);
}

bool Runtime::encode_frame_classic(id<MTLCommandBuffer> command, id<MTLTexture> color,
                                   id<MTLTexture> depth, id<MTLTexture> motion,
                                   const void *frame, size_t frameBytes,
                                   const double *phases, id<MTLTexture> const *outputs,
                                   unsigned phaseCount, std::string &error, const FrameExtras &extras) {
    return frame_combined(command, false, color, depth, motion, frame, frameBytes, phases, outputs, phaseCount, error, extras);
}

bool Runtime::frame_combined(id command, bool metal4, id<MTLTexture> color, id<MTLTexture> depth,
                             id<MTLTexture> motion, const void *frame, size_t frameBytes,
                             const double *phases, id<MTLTexture> const *outputs,
                             unsigned phaseCount, std::string &error, const FrameExtras &extras) {
    auto &r = *impl;
    if (!r.initialized) return r.fail(error, "ANR runtime not initialized");
    if (!command || [command device] != r.device || !color || !depth || !motion
        || !frame || frameBytes != 204 || phaseCount > 2 || (phaseCount && (!outputs || !phases))) {
        return r.fail(error, "ANR frame contract mismatch");
    }
    id<MTLTexture> o0 = phaseCount >= 1 ? outputs[0] : nil;
    id<MTLTexture> o1 = phaseCount >= 2 ? outputs[1] : nil;
    if ((phaseCount >= 1 && (!o0 || o0.device != r.device))
        || (phaseCount >= 2 && (!o1 || o1.device != r.device))) {
        return r.fail(error, "ANR frame output contract mismatch");
    }

    const char *fuseEnv = getenv("FSR4_ANR_FUSE_GROUPS");
    const bool fuse = !fuseEnv || strcmp(fuseEnv, "0") != 0;

    std::vector<std::pair<std::vector<Stage> *, id<MTLTexture>>> batches;
    batches.reserve(1 + phaseCount);
    batches.push_back({&r.stages[r.frameSlot].at("pair"), nil});
    for (unsigned p = 0; p < phaseCount; ++p) {
        char name[64];
        snprintf(name, sizeof name, "phase:%.6g:image", phases[p]);
        auto found = r.stages[r.frameSlot].find(name);
        if (found == r.stages[r.frameSlot].end()) {
            return r.fail(error, std::string("ANR unsupported phase: ") + name);
        }
        if (p && batches.back().first == &found->second) return r.fail(error, "ANR duplicate frame phase");
        batches.push_back({&found->second, outputs[p]});
    }

    r.inputColor = color;
    r.inputDepth = depth;
    r.inputMotion = motion;
    r.output = o0;
    if (metal4) {
        r.update_borrowed_residency(color, depth, motion, o0, o1);
    }
    if (r.buffers.at(r.inputFrame) != r.frameBuffers[r.frameSlot]) {
        r.buffers.at(r.inputFrame) = r.frameBuffers[r.frameSlot];
        ++r.bufferRevision;
    }
    std::memcpy(r.frameBuffers[r.frameSlot].contents, frame, frameBytes);
    return r.encode_groups(command, metal4, batches, error, nil, 0, nullptr, extras, fuse);
}

bool Runtime::profile_pair(id<MTL4CommandBuffer> command, id<MTL4CounterHeap> heap,
                           unsigned baseIndex, id<MTLTexture> color, id<MTLTexture> depth,
                           id<MTLTexture> motion, const void *frame, size_t frameBytes,
                           std::vector<StageProfile> &stages, std::string &error) {
    auto &r = *impl;
    if (!r.initialized) return r.fail(error, "ANR runtime not initialized");
    if (!command || !heap || [command device] != r.device || !color || !depth || !motion || !frame || frameBytes != 204) {
        return r.fail(error, "ANR profile_pair contract mismatch");
    }
    r.update_borrowed_residency(color, depth, motion, r.output);
    r.inputColor = color;
    r.inputDepth = depth;
    r.inputMotion = motion;
    r.output = nil;
    if (r.buffers.at(r.inputFrame) != r.frameBuffers[r.frameSlot]) {
        r.buffers.at(r.inputFrame) = r.frameBuffers[r.frameSlot];
        ++r.bufferRevision;
    }
    std::memcpy(r.frameBuffers[r.frameSlot].contents, frame, frameBytes);
    return r.encode_group(command, true, r.stages[r.frameSlot].at("pair"), error, heap, baseIndex, &stages);
}

bool Runtime::profile_interpolation(id<MTL4CommandBuffer> command, id<MTL4CounterHeap> heap,
                                    unsigned baseIndex, double phase, id<MTLTexture> output,
                                    std::vector<StageProfile> &stages, std::string &error) {
    auto &r = *impl;
    if (!r.initialized || !command || !heap || [command device] != r.device || !output || output.device != r.device) {
        return r.fail(error, "ANR profile_interpolation contract mismatch");
    }
    char name[64];
    snprintf(name, sizeof name, "phase:%.6g:image", phase);
    auto found = r.stages[r.frameSlot].find(name);
    if (found == r.stages[r.frameSlot].end()) {
        return r.fail(error, std::string("ANR unsupported phase: ") + name);
    }
    r.output = output;
    r.update_borrowed_residency(r.inputColor, r.inputDepth, r.inputMotion, output);
    return r.encode_group(command, true, found->second, error, heap, baseIndex, &stages);
}

void Runtime::set_coalesced(bool enabled) {
    impl->coalesced = enabled;
}

void Runtime::set_frame_slot(unsigned slot) {
    impl->frameSlot = slot % 4;
}

bool Runtime::reset_history(id command, bool metal4, std::string &error) {
    auto &r = *impl;
    if (!r.initialized || !command || [command device] != r.device) {
        return r.fail(error, "ANR reset command mismatch");
    }
    id encoder = nil;
    if (metal4) {
        [(id<MTL4CommandBuffer>)command useResidencySet:r.residency];
        encoder = [(id<MTL4CommandBuffer>)command computeCommandEncoder];
        if (encoder) {
            [(id<MTL4ComputeCommandEncoder>)encoder barrierAfterQueueStages:MTLStageAll
                                                               beforeStages:MTLStageBlit
                                                          visibilityOptions:MTL4VisibilityOptionDevice];
        }
    } else {
        encoder = [(id<MTLCommandBuffer>)command blitCommandEncoder];
    }
    if (!encoder) return r.fail(error, "ANR reset encoder allocation failed");
    for (unsigned index : r.persistentBuffers) {
        auto buffer = r.buffers.at(index);
        if (metal4) {
            [(id<MTL4ComputeCommandEncoder>)encoder fillBuffer:buffer range:NSMakeRange(0, buffer.length) value:0];
        } else {
            [(id<MTLBlitCommandEncoder>)encoder fillBuffer:buffer range:NSMakeRange(0, buffer.length) value:0];
        }
    }
    [encoder endEncoding];
    return true;
}

void Runtime::swap_history() {
    auto &r = *impl;
    std::swap(r.buffers.at(r.inputColor0), r.buffers.at(r.inputColor1));
    std::swap(r.buffers.at(r.inputDepth0), r.buffers.at(r.inputDepth1));
    for (const auto &swapPair : r.bufferSwaps) {
        std::swap(r.buffers.at(swapPair.first), r.buffers.at(swapPair.second));
    }
    for (const auto &texPair : r.textureSwaps) {
        auto a = r.internalTextures.find(texPair.first);
        auto b = r.internalTextures.find(texPair.second);
        if (a != r.internalTextures.end() && b != r.internalTextures.end()) {
            std::swap(a->second, b->second);
        }
    }
    if (r.isV2) r.swapParity ^= 1;
    else ++r.bufferRevision;
}

} // namespace anr
