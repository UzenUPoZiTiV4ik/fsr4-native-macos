#pragma once
#import <Metal/Metal.h>
#include "fsr4_backend.h"

namespace fsr4 {

inline DeviceCaps device_caps(id<MTLDevice> dev) {
    DeviceCaps caps;
    if (!dev) return caps;
    // Feature families cover base/Pro/Max/Ultra without parsing model names.
    // Use enum values so an SDK need not name the newest family to query it.
    // Probe a few families past the newest known (M6 and later) so diagnostics
    // report the real generation; selection only needs family >= 10.
    for (int family = 15; family >= 7; --family) {
        if ([dev supportsFamily:static_cast<MTLGPUFamily>(1000 + family)]) {
            caps.appleFamily = family;
            break;
        }
    }
    if (@available(macOS 26.0, *))
        caps.metal4 = NSClassFromString(@"MTLTensorDescriptor") != nil;
    caps.maxThreads = dev.maxThreadsPerThreadgroup.width;
    caps.threadgroupBytes = dev.maxThreadgroupMemoryLength;
    return caps;
}

} // namespace fsr4
