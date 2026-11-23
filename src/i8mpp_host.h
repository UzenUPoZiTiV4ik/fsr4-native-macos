// Host side of the native int8 tensor-ops model passes (src/i8mpp.metal).
// Header-only so both the CLI (fsr4mac.mm) and the live encoder
// (fsr4_live.mm) can use it. Manifest + weights come from tools/gen_i8mpp.py.
#pragma once
#import <Metal/Metal.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <sys/stat.h>
#include <vector>
#include "fsr4_device.h"
#include "fsr4_provider.h"

struct I8P {
    uint32_t inOff, inSX, inSY, inW, inH;
    uint32_t outOff, outSX, outSY, outW, outH;
    uint32_t addOff, addSX, addSY, pad0;
    uint32_t w0, w1, w2, w3, b0, b1, b2, b3;
    float s[16];
    uint32_t prmOff, prmSY, prmW, prmH;   // pass 13 model-parameter tensor (half4 per output pixel)
};

// Bytes the pass-13 model-parameter tensor needs after GetScratchSize(tier):
// one half4 per output pixel of the tier canvas, plus slack for the post
// pass's over-dispatch at the right/bottom edge.
static inline size_t i8mpp_params_bytes(int tier) {
    static const size_t canvas[3] = {1920ull * 1080, 3840ull * 2160, 7680ull * 4320};
    return canvas[tier < 0 ? 0 : tier > 2 ? 2 : tier] * 8 + 65536;
}

struct I8Pass {
    I8P p;
    I8P canvas;               // tier-canvas geometry from the manifest
    std::string kind;
    std::string kernel;
    uint32_t tileW = 32, tileH = 1;
    uint32_t gx = 0, gy = 0;
    id<MTLComputePipelineState> pso = nil;
};

struct I8Mpp {
    id<MTLBuffer> weights = nil;
    id<MTLBuffer> params = nil;     // I8P per pass at (pass * 256) for Metal 4 argument tables
    uint32_t enc0W = 0, enc0H = 0;  // tier canvas (manifest) at encoder level 0
    uint32_t regionW = 0, regionH = 0;
    std::map<int, I8Pass> passes;
    std::string error;

    static bool read_all(const std::string &path, std::vector<uint8_t> &out) {
        return fsr4::load_file(path, out);
    }

    // Ensures <builddir>/gen_i8mpp_<preset>_<res> exists (regenerating it via
    // tools/gen_i8mpp.py when the manifest is missing or older than the
    // tool), then returns the manifest's post ext-CT2D scratch offset.
    // `generated`, when given, reports whether the tool actually ran.
    static bool prepare(const std::string &root, const std::string &builddir,
                        const std::string &preset, const char *res,
                        std::string &gendir, long &postExtOff,
                        std::string &err, bool *generated = nullptr,
                        const char *readError = "cannot read i8mpp manifest",
                        const char *offsetError = "manifest lacks post_ext_ct2d_off") {
        gendir = builddir + "/gen_i8mpp_" + preset + "_" + res;
        const std::string manifest = gendir + "/i8mpp_manifest.json";
        const std::string tool = root + "/tools/gen_i8mpp.py";
        struct stat sm, sg;
        const bool fresh = stat(manifest.c_str(), &sm) == 0
            && (stat(tool.c_str(), &sg) != 0 || sm.st_mtime >= sg.st_mtime);
        if (generated) *generated = !fresh;
        if (!fresh) {
            char cmd[2048];
            snprintf(cmd, sizeof cmd,
                     "python3 '%s' --root '%s' --preset %s --res %s --outdir '%s'",
                     tool.c_str(), root.c_str(), preset.c_str(), res, gendir.c_str());
            if (system(cmd) != 0) { err = "gen_i8mpp failed"; return false; }
        }
        std::vector<uint8_t> mjv;
        if (!read_all(manifest, mjv)) { err = readError; return false; }
        std::string mj(mjv.begin(), mjv.end());
        const size_t k = mj.find("\"post_ext_ct2d_off\"");
        if (k == std::string::npos) {
            err = offsetError;
            return false;
        }
        postExtOff = strtol(mj.c_str() + mj.find(':', k) + 1, nullptr, 10);
        return true;
    }

    // gendir: build/gen_i8mpp_<preset>_<res>; metalSrc: src/i8mpp.metal
    bool load(id<MTLDevice> dev, const std::string &gendir, const std::string &metalSrc) {
        passes.clear();
        error.clear();
        const auto choice = fsr4::choose_backend(fsr4::device_caps(dev), fsr4::Backend::I8Mpp);
        if (choice.backend != fsr4::Backend::I8Mpp) { error = choice.reason; return false; }
        std::vector<uint8_t> wb;
        if (!read_all(gendir + "/i8mpp_weights.bin", wb)) { error = "no i8mpp_weights.bin in " + gendir; return false; }
        weights = [dev newBufferWithBytes:wb.data() length:wb.size() options:MTLResourceStorageModeShared];
        if (!weights) { error = "cannot allocate native INT8 weights"; return false; }
        NSData *jd = [NSData dataWithContentsOfFile:[NSString stringWithUTF8String:(gendir + "/i8mpp_manifest.json").c_str()]];
        if (!jd) { error = "no i8mpp_manifest.json"; return false; }
        NSError *err = nil;
        NSDictionary *mj = [NSJSONSerialization JSONObjectWithData:jd options:0 error:&err];
        if (!mj) { error = "manifest parse"; return false; }
        NSDictionary *ps = mj[@"passes"];
        uint32_t maxW = 0, maxH = 0;
        for (NSString *k in ps) {
            NSDictionary *e = ps[k];
            I8Pass ip;
            I8P &p = ip.p;
            memset(&p, 0, sizeof p);
            auto act = [&](NSDictionary *t, uint32_t &off, uint32_t &sx, uint32_t &sy, uint32_t *w, uint32_t *h) {
                off = [t[@"off"] unsignedIntValue];
                sx = [t[@"sx"] unsignedIntValue];
                sy = [t[@"sy"] unsignedIntValue];
                if (w) *w = [t[@"w"] unsignedIntValue];
                if (h) *h = [t[@"h"] unsignedIntValue];
            };
            act(e[@"in"], p.inOff, p.inSX, p.inSY, &p.inW, &p.inH);
            act(e[@"out"], p.outOff, p.outSX, p.outSY, &p.outW, &p.outH);
            if (e[@"add"]) act(e[@"add"], p.addOff, p.addSX, p.addSY, nullptr, nullptr);
            auto u = [&](const char *n) -> uint32_t { id v = e[[NSString stringWithUTF8String:n]]; return v ? [v unsignedIntValue] : 0; };
            p.w0 = u("w0"); p.w1 = u("w1"); p.w2 = u("w2"); p.w3 = u("w3");
            p.b0 = u("b0"); p.b1 = u("b1"); p.b2 = u("b2"); p.b3 = u("b3");
            NSArray *s = e[@"s"];
            for (NSUInteger i = 0; i < s.count && i < 16; i++) p.s[i] = [s[i] floatValue];
            if (e[@"prm"]) {
                p.prmOff = [e[@"prm"][@"off"] unsignedIntValue];
                p.prmW = 2 * p.outW; p.prmH = 2 * p.outH;
                p.prmSY = p.prmW * 8;
            }
            ip.kernel = [e[@"kernel"] UTF8String];
            ip.kind = [e[@"kind"] UTF8String];
            ip.tileW = [e[@"tile"][0] unsignedIntValue];
            ip.tileH = [e[@"tile"][1] unsignedIntValue];
            // ds/ct2d tile over the *input* grid for ct2d (output = 2x), over
            // the output grid for everything else
            std::string kind = [e[@"kind"] UTF8String];
            uint32_t gw = p.outW, gh = p.outH;
            if (kind == "ct2d") { gw = p.inW; gh = p.inH; }
            ip.gx = (gw + ip.tileW - 1) / ip.tileW;
            ip.gy = (gh + ip.tileH - 1) / ip.tileH;
            if (p.inW > maxW) maxW = p.inW;
            if (p.inH > maxH) maxH = p.inH;
            ip.canvas = ip.p;
            passes[k.intValue] = ip;
        }
        enc0W = maxW; enc0H = maxH;
        params = [dev newBufferWithLength:256 * 16 options:MTLResourceStorageModeShared];
        if (!params) { error = "cannot allocate native INT8 parameters"; return false; }
        memset(params.contents, 0, 256 * 16);
        for (auto &kv : passes)
            if (kv.first >= 0 && kv.first < 16)
                memcpy((uint8_t *)params.contents + kv.first * 256, &kv.second.p, sizeof(I8P));
        std::vector<uint8_t> src;
        if (!read_all(metalSrc, src)) { error = "no " + metalSrc; return false; }
        // Metal's newLibraryWithSource cache served stale binaries after source
        // edits, so the library is built with xcrun into a metallib whose name
        // is a hash of (source, macros) and loaded from that file.
        int dbg = getenv("FSR4_I8_DEBUG") ? atoi(getenv("FSR4_I8_DEBUG")) : 0;
        const char *mflags = getenv("FSR4_I8_MFLAGS");   // extra xcrun metal flags (experiments)
        char macros[256];
        snprintf(macros, sizeof macros, "-D MAXW=%u -D MAXH=%u%s%s %s", maxW, maxH, dbg ? " -D I8_DEBUG=" : "",
                 dbg ? std::to_string(dbg).c_str() : "", mflags ? mflags : "");
        uint64_t hsh = 1469598103934665603ull;
        for (uint8_t c : src) hsh = (hsh ^ c) * 1099511628211ull;
        for (const char *c = macros; *c; c++) hsh = (hsh ^ (uint8_t)*c) * 1099511628211ull;
        std::string outdir = gendir.substr(0, gendir.rfind('/'));
        char base[256];
        snprintf(base, sizeof base, "%s/i8mpp_%016llx", outdir.c_str(), (unsigned long long)hsh);
        std::string mlib = std::string(base) + ".metallib";
        if (!read_all(mlib, src)) {
            char cmd[2048];
            snprintf(cmd, sizeof cmd,
                     "$(xcrun -f metal) -std=metal4.0 -O3 %s -c '%s' -o '%s.air' && $(xcrun -f metallib) '%s.air' -o '%s'",
                     macros, metalSrc.c_str(), base, base, mlib.c_str());
            if (system(cmd) != 0) { error = "i8mpp: xcrun metal compile failed"; return false; }
        }
        id<MTLLibrary> lib = [dev newLibraryWithURL:[NSURL fileURLWithPath:[NSString stringWithUTF8String:mlib.c_str()]] error:&err];
        if (!lib) { error = std::string("i8mpp metallib: ") + err.localizedDescription.UTF8String; return false; }
        for (auto &kv : passes) {
            id<MTLFunction> fn = [lib newFunctionWithName:[NSString stringWithUTF8String:kv.second.kernel.c_str()]];
            if (!fn) { error = "missing kernel " + kv.second.kernel; return false; }
            kv.second.pso = [dev newComputePipelineStateWithFunction:fn error:&err];
            if (!kv.second.pso) { error = "pso " + kv.second.kernel + ": " + err.localizedDescription.UTF8String; return false; }
            if (kv.second.pso.maxTotalThreadsPerThreadgroup < 128 ||
                kv.second.pso.staticThreadgroupMemoryLength > dev.maxThreadgroupMemoryLength) {
                error = "pipeline exceeds threadgroup limits: " + kv.second.kernel;
                return false;
            }
            if (getenv("FSR4_I8_VERBOSE"))
                fprintf(stderr, "i8mpp pass %d %s: tgmem %lu maxThreads %lu grid %ux%u\n", kv.first, kv.second.kernel.c_str(),
                        (unsigned long)kv.second.pso.staticThreadgroupMemoryLength,
                        (unsigned long)kv.second.pso.maxTotalThreadsPerThreadgroup, kv.second.gx, kv.second.gy);
        }
        return true;
    }

    bool has(int pass) const { return passes.count(pass) != 0; }

    // The scratch tensors are a fixed canvas per output tier (e.g. 1920x1080
    // at enc0 for the 2160 tier); a frame only uses the top-left
    // upscale>>(level+1) region, like the provider's dispatch sizes. Set the
    // upscale size so the kernels tile just that region and treat its edge as
    // the padding edge (the rest of the canvas is never written).
    void set_region(uint32_t uw, uint32_t uh) {
        if (uw == regionW && uh == regionH) return;
        regionW = uw; regionH = uh;
        for (auto &kv : passes) {
            I8Pass &ip = kv.second;
            I8P &p = ip.p;
            p = ip.canvas;
            auto level = [&](uint32_t w) { int l = 0; while (l < 4 && (enc0W >> l) > w) l++; return l; };
            const int li = level(ip.canvas.inW), lo = level(ip.canvas.outW);
            p.inW = uw >> (li + 1); p.inH = uh >> (li + 1);
            p.outW = uw >> (lo + 1); p.outH = uh >> (lo + 1);
            if (p.inW > ip.canvas.inW) p.inW = ip.canvas.inW;
            if (p.inH > ip.canvas.inH) p.inH = ip.canvas.inH;
            if (p.outW > ip.canvas.outW) p.outW = ip.canvas.outW;
            if (p.outH > ip.canvas.outH) p.outH = ip.canvas.outH;
            uint32_t gw = p.outW, gh = p.outH;
            if (ip.kind == "ct2d") { gw = p.inW; gh = p.inH; }
            ip.gx = (gw + ip.tileW - 1) / ip.tileW;
            ip.gy = (gh + ip.tileH - 1) / ip.tileH;
            if (p.prmOff) {
                // post indexes the parameters with its `width` constant = uw
                p.prmW = uw; p.prmH = uh; p.prmSY = uw * 8;
            }
            if (params && kv.first >= 0 && kv.first < 16)
                memcpy((uint8_t *)params.contents + kv.first * 256, &p, sizeof(I8P));
        }
    }

    // scratch buffer length the passes need for this tier (tensors + pass-13 params)
    static size_t scratch_bytes(int tier) {
        static const size_t kScratch[3] = {20880256, 83232256, 332352256};
        const int t = tier < 0 ? 0 : tier > 2 ? 2 : tier;
        return kScratch[t] + i8mpp_params_bytes(t);
    }

    // Metal 4 encoder: buffers via an argument table (0 scratch, 1 weights, 2 params);
    // pass 13 also takes the recurrent texture at texture slot 0.
    void dispatch_m4(int pass, id enc, id table, id<MTLBuffer> scratch, id<MTLTexture> tex = nil) const {
        const I8Pass &ip = passes.at(pass);
        id<MTL4ArgumentTable> t = table;
        [t setAddress:scratch.gpuAddress atIndex:0];
        [t setAddress:weights.gpuAddress atIndex:1];
        [t setAddress:params.gpuAddress + (uint64_t)pass * 256 atIndex:2];
        if (tex) [t setTexture:tex.gpuResourceID atIndex:0];
        [enc setArgumentTable:t];
        [enc setComputePipelineState:ip.pso];
        [enc dispatchThreadgroups:MTLSizeMake(ip.gx, ip.gy, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        [enc barrierAfterEncoderStages:MTLStageDispatch | MTLStageBlit
                   beforeEncoderStages:MTLStageDispatch | MTLStageBlit
                     visibilityOptions:MTL4VisibilityOptionDevice];
    }

    // classic encoder
    void dispatch(int pass, id<MTLComputeCommandEncoder> enc, id<MTLBuffer> scratch, MTLBarrierScope scope,
                  id<MTLTexture> tex = nil) const {
        const I8Pass &ip = passes.at(pass);
        if (getenv("FSR4_I8_VERBOSE")) {
            fprintf(stderr, "i8 dispatch pass %d %s in %u/%u/%u %ux%u out %u/%u/%u %ux%u w %u %u %u %u b %u %u %u %u s", pass, ip.kernel.c_str(),
                    ip.p.inOff, ip.p.inSX, ip.p.inSY, ip.p.inW, ip.p.inH, ip.p.outOff, ip.p.outSX, ip.p.outSY, ip.p.outW, ip.p.outH,
                    ip.p.w0, ip.p.w1, ip.p.w2, ip.p.w3, ip.p.b0, ip.p.b1, ip.p.b2, ip.p.b3);
            for (int i = 0; i < 16; i++) fprintf(stderr, " %.6g", ip.p.s[i]);
            fprintf(stderr, " prm %u/%u %ux%u grid %ux%u\n", ip.p.prmOff, ip.p.prmSY, ip.p.prmW, ip.p.prmH, ip.gx, ip.gy);
        }
        [enc setComputePipelineState:ip.pso];
        [enc setBuffer:scratch offset:0 atIndex:0];
        [enc setBuffer:weights offset:0 atIndex:1];
        [enc setBytes:&ip.p length:sizeof(I8P) atIndex:2];
        if (tex) [enc setTexture:tex atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(ip.gx, ip.gy, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        if (scope) [enc memoryBarrierWithScope:scope];
    }
};
