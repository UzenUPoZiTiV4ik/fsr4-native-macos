// SPDX-License-Identifier: MIT
#pragma once
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <memory>
#include <string>
#include <vector>

// Independent ANR FG Metal host supporting:
// - V.B20 (fsr4_anr_runtime_v2): dynamic resolution 640x360..7680x4320, native
//   pq10/srgb8/scrgb16 encodings, compact Tier-2 argument buffers, Private GPU memory.
// - Legacy V.B14 (fsr4_anr_runtime_v1): fixed resolution 1080p/1440p/2160p.
// Caller keeps borrowed textures alive and submits work on one ordered queue.
namespace anr {

enum class PhaseMode {
    All = 0,
    TwoX = 1,
    ThreeX = 2,
};

enum class Encoding {
    PQ10 = 0,
    SRGB8 = 1,
    SCRGB16 = 2,
};

struct StageProfile {
    std::string label;
    std::string kernel;
    double gpuMs = 0.0;
};

// Optional synchronous helper dispatches on the same compute encoder as inference.
// Helpers bind their own resources but do not insert barriers or end the encoder.
struct FrameExtras {
    id<MTLFence> fence = nil;
    void (^import)(id encoder) = nil;
    void (^exportPhase)(id encoder, unsigned phaseIndex) = nil;
    void (^retire)(id encoder) = nil;
};

class Runtime {
public:
    Runtime(id<MTLDevice> device, const std::string &directory);
    ~Runtime();
    Runtime(const Runtime &) = delete;
    Runtime &operator=(const Runtime &) = delete;

    // Legacy entry point: v1 uses manifest dimensions; v2 defaults to 1920x1080 PQ10.
    bool initialize(std::string &error, bool twoXOnly = false);

    // Dynamic V.B20 entry point: selects variant for (width, height) and native encoding.
    bool initialize(std::string &error, unsigned width, unsigned height,
                    Encoding encoding, PhaseMode phaseMode = PhaseMode::All);

    unsigned width() const;
    unsigned height() const;
    bool is_v2() const;
    bool mpp_active() const;
    Encoding encoding() const;
    std::string release_name() const;
    std::string backend_name() const;

    bool encode_pair(id<MTL4CommandBuffer> command, id<MTLTexture> color,
                     id<MTLTexture> depth, id<MTLTexture> motion,
                     const void *frame, size_t frameBytes, std::string &error);
    bool encode_interpolation(id<MTL4CommandBuffer> command, double phase,
                              id<MTLTexture> output, std::string &error);
    bool encode_pair_classic(id<MTLCommandBuffer> command, id<MTLTexture> color,
                             id<MTLTexture> depth, id<MTLTexture> motion,
                             const void *frame, size_t frameBytes, std::string &error);
    bool encode_interpolation_classic(id<MTLCommandBuffer> command, double phase,
                                      id<MTLTexture> output, std::string &error);

    // Coalesced single-encoder pair + 1..2 phase interpolation (avoids inter-encoder
    // queue barriers between pair and phase, and between phase 1/3 and 2/3 in 3x mode).
    bool encode_frame(id<MTL4CommandBuffer> command, id<MTLTexture> color,
                      id<MTLTexture> depth, id<MTLTexture> motion,
                      const void *frame, size_t frameBytes,
                      const double *phases, id<MTLTexture> const *outputs,
                      unsigned phaseCount, std::string &error, const FrameExtras &extras = {});
    bool encode_frame_classic(id<MTLCommandBuffer> command, id<MTLTexture> color,
                              id<MTLTexture> depth, id<MTLTexture> motion,
                              const void *frame, size_t frameBytes,
                              const double *phases, id<MTLTexture> const *outputs,
                              unsigned phaseCount, std::string &error, const FrameExtras &extras = {});

    // Per-stage Metal 4 GPU profiling helpers (writes timestamps before/after each dispatch).
    bool profile_pair(id<MTL4CommandBuffer> command, id<MTL4CounterHeap> heap,
                      unsigned baseIndex, id<MTLTexture> color, id<MTLTexture> depth,
                      id<MTLTexture> motion, const void *frame, size_t frameBytes,
                      std::vector<StageProfile> &stages, std::string &error);
    bool profile_interpolation(id<MTL4CommandBuffer> command, id<MTL4CounterHeap> heap,
                               unsigned baseIndex, double phase, id<MTLTexture> output,
                               std::vector<StageProfile> &stages, std::string &error);

    // One compute encoder for an entire stage group (default); false is the
    // reference scheduling route used by the optimization A/B test.
    void set_coalesced(bool enabled);

    // Four CPU-write slots allow GPU-ordered game frames without descriptor
    // overwrite. Caller must only reuse a slot after its completion token.
    void set_frame_slot(unsigned slot);

    // GPU-ordered clear, never CPU memset into an in-flight history buffer.
    bool reset_history(id command, bool metal4, std::string &error);
    void swap_history();
    id<MTLBuffer> diagnostic_buffer(unsigned index) const;

private:
    struct Impl;
    bool pair(id command, bool metal4, id<MTLTexture> color, id<MTLTexture> depth,
              id<MTLTexture> motion, const void *frame, size_t bytes, std::string &error);
    bool interpolate(id command, bool metal4, double phase, id<MTLTexture> output, std::string &error);
    bool frame_combined(id command, bool metal4, id<MTLTexture> color, id<MTLTexture> depth,
                        id<MTLTexture> motion, const void *frame, size_t bytes,
                        const double *phases, id<MTLTexture> const *outputs,
                        unsigned phaseCount, std::string &error, const FrameExtras &extras);
    std::unique_ptr<Impl> impl;
};

} // namespace anr
