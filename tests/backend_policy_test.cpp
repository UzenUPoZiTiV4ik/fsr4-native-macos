#include "../src/fsr4_backend.h"
#include <cassert>
#include <cstdio>

int main() {
    using fsr4::Backend;
    // Pro/Max/Ultra share their generation's feature family. Names, CPU
    // architecture and GPU core counts must not affect backend selection.
    const int families[] = {0, 7, 8, 9, 10, 11, 12, 15}; // unknown, M1, M2, M3/M4, M5, M6, later
    const bool runtimes[] = {false, true};
    int checks = 0;
    for (int family : families) {
        for (bool metal4 : runtimes) {
            fsr4::DeviceCaps caps{family, metal4, 1024, 32768};
            const Backend expected = metal4 && family >= 10 ? Backend::I8Mpp : Backend::Fp16;
            assert(fsr4::choose_backend(caps).backend == expected);
            assert(fsr4::choose_backend(caps, Backend::I8Mpp).backend == expected);
            assert(fsr4::choose_backend(caps, Backend::Fp16).backend == Backend::Fp16);
            assert(fsr4::choose_backend(caps, Backend::Int8).backend == Backend::Int8);
            // A failed native initialization must win over an explicit
            // override; otherwise the live path retries forever.
            assert(fsr4::choose_backend(caps, Backend::I8Mpp, true).backend == Backend::Fp16);
            assert(fsr4::choose_backend(caps, Backend::Auto, true).backend == Backend::Fp16);
            checks += 6;
        }
    }
    fsr4::DeviceCaps future{11, true, 1024, 32768};
    assert(fsr4::choose_backend(future).backend == Backend::I8Mpp);
    assert(fsr4::choose_backend({}).backend == Backend::Fp16);
    future.maxThreads = 127;
    assert(fsr4::choose_backend(future).backend == Backend::Fp16);
    future.maxThreads = 128;
    future.threadgroupBytes = 24575;
    assert(fsr4::choose_backend(future).backend == Backend::Fp16);
    future.threadgroupBytes = 24576;
    assert(fsr4::choose_backend(future).backend == Backend::I8Mpp);
    Backend request = Backend::Auto;
    assert(fsr4::parse_backend(nullptr, request) && request == Backend::Auto);
    assert(fsr4::parse_backend("fp16", request) && request == Backend::Fp16);
    assert(fsr4::parse_backend("int8", request) && request == Backend::Int8);
    assert(!fsr4::parse_backend("m5-pro", request));
    std::printf("backend policy: %d profile checks and resource/failure boundaries passed\n", checks);
}
