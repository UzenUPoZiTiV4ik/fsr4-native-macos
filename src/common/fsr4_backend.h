// Backend policy shared by the CLI and the live encoder. Keep the policy
// independent of Metal so every hardware/failure combination can be tested.
#pragma once
#include <cstddef>
#include <cstring>

namespace fsr4 {

enum class Backend { Auto, Fp16, Int8, I8Mpp };

inline const char *backend_name(Backend backend) {
    switch (backend) {
        case Backend::Auto: return "auto";
        case Backend::Fp16: return "f16";
        case Backend::Int8: return "int8";
        case Backend::I8Mpp: return "i8mpp";
    }
    return "auto";
}

inline bool parse_backend(const char *name, Backend &backend) {
    if (!name || !*name || !std::strcmp(name, "auto")) backend = Backend::Auto;
    else if (!std::strcmp(name, "f16") || !std::strcmp(name, "fp16")) backend = Backend::Fp16;
    else if (!std::strcmp(name, "int8")) backend = Backend::Int8;
    else if (!std::strcmp(name, "i8mpp")) backend = Backend::I8Mpp;
    else return false;
    return true;
}

struct DeviceCaps {
    int appleFamily = 0;
    bool metal4 = false;
    std::size_t maxThreads = 0;
    std::size_t threadgroupBytes = 0;
};

struct BackendChoice {
    Backend backend;
    const char *reason;
};

inline BackendChoice choose_backend(const DeviceCaps &caps, Backend request = Backend::Auto,
                                    bool nativeFailed = false) {
    if (request == Backend::Fp16) return {Backend::Fp16, "FP16 requested"};
    if (request == Backend::Int8) return {Backend::Int8, "reference INT8 requested (shader arithmetic)"};
    if (nativeFailed) return {Backend::Fp16, "native INT8 initialization failed; using FP16"};
    if (!caps.metal4) return {Backend::Fp16, "Metal 4 requires macOS 26 or newer"};
    if (caps.appleFamily < 10)
        return {Backend::Fp16, "GPU lacks Apple family 10 tensor acceleration; using FP16"};
    // Largest current native tile: 128 threads, 24 KiB of shared storage.
    // Individual compiled pipelines are checked again before dispatch.
    if (caps.maxThreads < 128 || caps.threadgroupBytes < 24576)
        return {Backend::Fp16, "native INT8 threadgroup requirements exceed device limits"};
    return {Backend::I8Mpp, "Apple family 10+ GPU tensor acceleration"};
}

} // namespace fsr4
