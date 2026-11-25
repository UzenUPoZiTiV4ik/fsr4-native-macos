#pragma once
// Minimal PNG writer (RGBA8, deflate via zlib).
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
#include <zlib.h>

inline void png_u32(std::vector<uint8_t>& v, uint32_t x) {
    v.push_back((x >> 24) & 0xff); v.push_back((x >> 16) & 0xff);
    v.push_back((x >> 8) & 0xff);  v.push_back(x & 0xff);
}
inline uint32_t png_crc(const uint8_t* d, size_t n) {
    return (uint32_t)crc32(0, d, (uInt)n);
}
inline void png_chunk(std::vector<uint8_t>& out, const char type[4],
                      const uint8_t* data, uint32_t len) {
    png_u32(out, len);
    size_t base = out.size();
    out.insert(out.end(), type, type + 4);
    out.insert(out.end(), data, data + len);
    png_u32(out, png_crc(out.data() + base, 4 + len));
}
inline bool png_write(const char* path, const uint8_t* rgba, int w, int h) {
    std::vector<uint8_t> raw;
    raw.reserve((size_t)h * (1 + 4 * (size_t)w));
    for (int y = 0; y < h; y++) {
        raw.push_back(0);
        raw.insert(raw.end(), rgba + (size_t)y * w * 4,
                   rgba + (size_t)(y + 1) * w * 4);
    }
    uLongf clen = compressBound((uLong)raw.size());
    std::vector<uint8_t> comp(clen);
    if (compress2(comp.data(), &clen, raw.data(), (uLong)raw.size(), 6) != Z_OK)
        return false;
    std::vector<uint8_t> png;
    const uint8_t sig[8] = {0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A};
    png.insert(png.end(), sig, sig + 8);
    uint8_t ihdr[13];
    memset(ihdr, 0, 13);
    ihdr[0] = (w >> 24) & 0xff; ihdr[1] = (w >> 16) & 0xff;
    ihdr[2] = (w >> 8) & 0xff;  ihdr[3] = w & 0xff;
    ihdr[4] = (h >> 24) & 0xff; ihdr[5] = (h >> 16) & 0xff;
    ihdr[6] = (h >> 8) & 0xff;  ihdr[7] = h & 0xff;
    ihdr[8] = 8; ihdr[9] = 6;  // 8-bit RGBA
    png_chunk(png, "IHDR", ihdr, 13);
    png_chunk(png, "IDAT", comp.data(), (uint32_t)clen);
    png_chunk(png, "IEND", nullptr, 0);
    FILE* f = fopen(path, "wb");
    if (!f) return false;
    bool ok = fwrite(png.data(), 1, png.size(), f) == png.size();
    if (fclose(f) != 0) ok = false;
    return ok;
}
