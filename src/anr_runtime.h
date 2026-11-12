// SPDX-License-Identifier: MIT
#pragma once
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <memory>
#include <string>

// Experimental independent ANR FG host. Uses locally extracted/converted
// kernels, never the test kit executable. No queue, commit or presentation hook.
// Caller keeps borrowed textures alive and submits work on one ordered queue.
// Four caller-selected CPU-write slots permit in-flight frames; a slot must not
// be reused until its GPU completion. swap_history changes only CPU bindings,
// not shared GPU history data. src/anr_live.mm enforces this lifecycle.
namespace anr {
class Runtime {
public:
    Runtime(id<MTLDevice> device, const std::string &directory);
    ~Runtime();
    Runtime(const Runtime &) = delete;
    Runtime &operator=(const Runtime &) = delete;
    bool initialize(std::string &error, bool twoXOnly = false);
    unsigned width() const;
    unsigned height() const;
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
    std::unique_ptr<Impl> impl;
};
}
