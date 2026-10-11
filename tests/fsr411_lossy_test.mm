// SPDX-License-Identifier: MIT
// Independent geometric oracle for the generated lossy PRE/POST helpers.
// Run through tests/run_fsr411_lossy.py. No timings are collected.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <algorithm>
#include <array>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using Word = std::array<uint32_t, 4>;
using Cell = std::array<int32_t, 2>;
struct Record {
    uint32_t pixel[2]; float velocity[2]; int32_t depth[2];
    uint32_t invalid, reserved;
};
struct Dimensions { uint32_t w, h, rw, rh, records, queries, cacheStride, reserved; };
struct Policy { uint32_t skip, blocksW, blocksH, reserved; };
static_assert(sizeof(Record) == 32 && sizeof(Dimensions) == 32 && sizeof(Policy) == 16, "HLSL fixture ABI");
static size_t checks = 0, cases = 0;

static uint32_t bits(float value) { uint32_t b; std::memcpy(&b, &value, 4); return b; }
static float floating(uint32_t value) { float f; std::memcpy(&f, &value, 4); return f; }
static int32_t signed_int(uint32_t value) { int32_t i; std::memcpy(&i, &value, 4); return i; }
static int floor_div(int value, int divisor) { return int(std::floor(double(value) / divisor)); }
static void require(bool good, const std::string &message) {
    ++checks;
    if (!good) { std::fprintf(stderr, "%s\n", message.c_str()); std::exit(1); }
}
static Word cache_value(unsigned x, unsigned y, unsigned width) {
    return {x, y, (y * width + x) ^ 0x411ca5u, 0x12340000u | x | (y << 8)};
}

// Interpret the compressed depth as a five-bit exponent with an m-bit
// significand. This oracle uses frexp/floor, not the shader's bit shifts.
static unsigned depth_key(float nearness, int tierIndex) {
    const int mantissa = 9 - 2 * tierIndex;
    const int maximum = (1 << (14 - 2 * tierIndex)) - 1;
    // The published unsigned depth code retains the IEEE sign bit. Even
    // negative zero consequently saturates to its maximum integer key.
    if (std::signbit(nearness)) return unsigned(maximum);
    if (!(nearness > 0)) return 1;
    int exponent = 0;
    const double normalized = std::frexp(double(nearness), &exponent) * 2;
    const int integer = (exponent - 1 + 31) * (1 << mantissa) +
                        int(std::floor((normalized - 1) * (1 << mantissa)));
    return unsigned(std::max(1, std::min(maximum, integer)));
}

struct Oracle {
    std::vector<Word> motion;
    std::vector<uint32_t> plane;
    std::vector<int> owners;
};

static Oracle record_oracle(const Dimensions &d, const Policy &p, const std::vector<Record> &records,
                             const std::vector<float> &depth, bool low, bool inverted, int tierIndex) {
    const size_t blocks = p.blocksW * p.blocksH;
    Oracle result{std::vector<Word>(blocks, Word{0xa5a5a5a5u,0xa5a5a5a5u,0xa5a5a5a5u,0xa5a5a5a5u}),
                  std::vector<uint32_t>(blocks, p.skip ? 0u : 0x5a5a5a5au), std::vector<int>(blocks, -1)};
    if (!p.skip) return result;
    for (const auto &r : records) {
        if (r.pixel[0] % 4 || r.pixel[1] % 4 || r.pixel[0] / 4 >= p.blocksW || r.pixel[1] / 4 >= p.blocksH) continue;
        const unsigned bx = r.pixel[0] / 4, by = r.pixel[1] / 4, block = by * p.blocksW + bx;
        const auto displacement = [&](int axis) {
            const float pixels = r.velocity[axis] * float(axis ? d.h : d.w);
            return r.invalid ? 0 : int(std::max(-254.f, std::min(254.f, std::nearbyint(pixels))));
        };
        const int dx = displacement(0), dy = displacement(1);
        float raw = 0, signedDepth = 0;
        if (low) {
            if (r.depth[0] >= 0 && r.depth[1] >= 0 && unsigned(r.depth[0]) < d.rw && unsigned(r.depth[1]) < d.rh)
                raw = depth[unsigned(r.depth[1]) * d.rw + unsigned(r.depth[0])];
            signedDepth = inverted ? -raw : raw;
        }
        const bool valid = low && !r.invalid && std::isfinite(raw) && bits(signedDepth) != 0;
        result.motion[block] = {uint32_t(dx), uint32_t(dy), bits(signedDepth), uint32_t(valid)};
        // Publication semantics distinguish participation in the owner
        // plane from validity when POST later reads that owner. A normal
        // +0 depth wins as nearest, but its zero depth bits mask occlusion.
        if (!low || r.invalid || !std::isfinite(raw)) continue;
        const float near = inverted ? raw : 1.f - raw;
        const unsigned owner = by * (1u << (9 + tierIndex)) + bx;
        const uint32_t key = depth_key(near, tierIndex) * (1u << (18 + 2 * tierIndex)) + owner;
        // Enumerate all sixteen translated pixels independently. The shader
        // instead traverses the bounding rectangle of intersected blocks.
        for (int y = 0; y < 4; ++y) for (int x = 0; x < 4; ++x) {
            const int targetX = floor_div(int(r.pixel[0]) + dx + x, 4);
            const int targetY = floor_div(int(r.pixel[1]) + dy + y, 4);
            if (targetX < 0 || targetY < 0 || unsigned(targetX) >= p.blocksW || unsigned(targetY) >= p.blocksH) continue;
            const unsigned target = unsigned(targetY) * p.blocksW + unsigned(targetX);
            if (key > result.plane[target]) { result.plane[target] = key; result.owners[target] = int(block); }
        }
    }
    return result;
}

static bool different(const Word &a, const Word &b) {
    return std::abs(signed_int(a[0]) - signed_int(b[0])) >= 2 || std::abs(signed_int(a[1]) - signed_int(b[1])) >= 2;
}

static std::array<bool, 2> classify_oracle(Cell cell, const Dimensions &d, const Policy &p, const Oracle &oracle) {
    const auto read = [&](int x, int y) -> const Word & {
        x = std::max(0, std::min(int(d.w / 2) - 1, x));
        y = std::max(0, std::min(int(d.h / 2) - 1, y));
        return oracle.motion[unsigned(y / 2) * p.blocksW + unsigned(x / 2)];
    };
    const Word own = read(cell[0], cell[1]);
    const int dx = signed_int(own[0]), dy = signed_int(own[1]);
    bool movingNeighbor = different(own, read(cell[0] + floor_div(dx, 2), cell[1] + floor_div(dy, 2)));
    for (int distance : {3, 8}) for (int sign : {-1, 1}) {
        movingNeighbor |= different(own, read(cell[0] + distance * sign, cell[1]));
        movingNeighbor |= different(own, read(cell[0], cell[1] + distance * sign));
    }
    bool uncovered = false;
    const int previousX = floor_div(cell[0] * 2 + dx, 4), previousY = floor_div(cell[1] * 2 + dy, 4);
    if (own[3] && previousX >= 0 && previousY >= 0 && unsigned(previousX) < p.blocksW && unsigned(previousY) < p.blocksH) {
        const int owner = oracle.owners[unsigned(previousY) * p.blocksW + unsigned(previousX)];
        if (owner >= 0) {
            const auto &surface = oracle.motion[size_t(owner)];
            uncovered = surface[3] && floating(surface[2]) < floating(own[2]) && different(surface, own);
        }
    }
    return {dx == 0 && dy == 0 && !movingNeighbor, uncovered};
}

struct Fixture {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    NSDictionary *metadata;
    NSMutableDictionary *pipelines;
    bool low, inverted;
    int tierIndex;

    explicit Fixture(const char *directory) {
        device = MTLCreateSystemDefaultDevice(); queue = [device newCommandQueue];
        require(device && queue, "Metal device unavailable");
        const auto base = [NSString stringWithUTF8String:directory];
        NSData *data = [NSData dataWithContentsOfFile:[base stringByAppendingPathComponent:@"fixture.json"]];
        NSError *error = nil;
        metadata = [NSJSONSerialization JSONObjectWithData:data options:0 error:&error];
        require(metadata != nil, "Cannot read fixture.json");
        low = [metadata[@"low_res_mv"] boolValue]; inverted = [metadata[@"depth_inverted"] boolValue];
        const unsigned tier = [metadata[@"tier"] unsignedIntValue]; tierIndex = tier == 1080 ? 0 : tier == 2160 ? 1 : 2;
        pipelines = [NSMutableDictionary new];
        for (NSString *name in metadata[@"kernels"]) {
            NSDictionary *k = metadata[@"kernels"][name];
            auto library = [device newLibraryWithURL:[NSURL fileURLWithPath:[base stringByAppendingPathComponent:k[@"library"]]] error:&error];
            require(library != nil, std::string("Cannot load ") + name.UTF8String);
            auto function = [library newFunctionWithName:name];
            auto pipeline = function ? [device newComputePipelineStateWithFunction:function error:&error] : nil;
            require(pipeline != nil, std::string("Cannot create ") + name.UTF8String + ": " + (error ? error.description.UTF8String : ""));
            pipelines[name] = pipeline;
        }
    }

    id<MTLBuffer> buffer(const void *bytes, size_t size) {
        auto result = [device newBufferWithBytes:bytes length:size options:MTLResourceStorageModeShared];
        require(result != nil, "Fixture buffer allocation failed"); return result;
    }
    void encode(id<MTLComputeCommandEncoder> encoder, NSString *name, NSDictionary *buffers, NSDictionary *textures, unsigned count) {
        NSDictionary *k = metadata[@"kernels"][name];
        [encoder setComputePipelineState:pipelines[name]];
        for (NSArray *binding in k[@"buffers"]) {
            id<MTLBuffer> value = buffers[binding[1]];
            require(value != nil, std::string("Unknown fixture buffer ") + [binding[1] UTF8String]);
            [encoder setBuffer:value offset:0 atIndex:[binding[0] unsignedIntValue]];
        }
        for (NSArray *binding in k[@"textures"]) {
            id<MTLTexture> value = textures[binding[1]];
            require(value != nil, std::string("Unknown fixture texture ") + [binding[1] UTF8String]);
            [encoder setTexture:value atIndex:[binding[0] unsignedIntValue]];
        }
        [encoder dispatchThreads:MTLSizeMake(count,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
        [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers|MTLBarrierScopeTextures];
    }

    void test(const char *name, float dx, float dy, int scenario = 0, bool skip = true) { @autoreleasepool {
        Dimensions d{64,64,32,32,0,0,32*16,0}; Policy p{unsigned(skip),16,16,0};
        std::vector<Record> records;
        std::vector<float> depth(d.rw * d.rh, 0.5f);
        const auto setDepth = [&](unsigned block, float value) {
            const auto &r = records[block]; depth[unsigned(r.depth[1]) * d.rw + unsigned(r.depth[0])] = value;
        };
        for (unsigned y = 0; y < p.blocksH; ++y) for (unsigned x = 0; x < p.blocksW; ++x)
            records.push_back({{x*4,y*4},{dx/d.w,dy/d.h},{int32_t(x*2),int32_t(y*2)},0,0});
        if (scenario == 1 || scenario == 2) {
            const unsigned a = 4 * p.blocksW + 4, b = a + 1;
            records[b].velocity[0] = -4.f / d.w;
            setDepth(a, scenario == 2 ? 0.5f : inverted ? 0.2f : 0.8f);
            setDepth(b, scenario == 2 ? 0.5f : inverted ? 0.9f : 0.1f);
        }
        if (scenario == 3) {
            records[0].invalid = 1;
            records[1].depth[0] = -1;
            setDepth(2, 0.f); setDepth(3, -0.f);
            setDepth(4, floating(0x7fc00000u)); setDepth(5, floating(0x7f800000u));
            records[6].depth[0] = int(d.rw);
            records[7].velocity[0] = -500.f / d.w; records[7].velocity[1] = 500.f / d.h;
            const float ties[] = {-2.5f,-1.5f,-0.5f,0.5f,1.5f,2.5f};
            for (unsigned i=0;i<6;++i) { records[8+i].velocity[0] = ties[i]/d.w; records[8+i].velocity[1] = ties[5-i]/d.h; }
            records.push_back({{1,0},{9.f,9.f},{0,0},0,0});
            records.push_back({{d.w,0},{9.f,9.f},{0,0},0,0});
        }
        if (scenario == 4 || scenario == 5) {
            const unsigned a=4*p.blocksW+4,b=a+1,c=a+2;
            records[b].velocity[0]=-4.f/d.w;
            records[c].velocity[0]=-8.f/d.w;
            setDepth(a,inverted?.2f:.8f);
            setDepth(b,inverted?.9f:.1f);
            setDepth(c,scenario==4?0.f:-0.f);
        }
        std::vector<Cell> cells;
        std::vector<Word> cache;
        for (unsigned y=0;y<d.h/2;++y) for (unsigned x=0;x<d.w/2;++x) {
            cells.push_back({int(x),int(y)}); cache.push_back(cache_value(x,y,d.w/2));
        }
        d.records = unsigned(records.size()); d.queries = unsigned(cells.size());
        const auto expected = record_oracle(d,p,records,depth,low,inverted,tierIndex);
        if (low && (scenario==4 || scenario==5)) {
            const unsigned a=4*p.blocksW+4,b=a+1,c=a+2;
            require(expected.owners[a]==int(inverted && scenario==4?b:c),"zero-depth scenario did not select the intended owner");
            const auto classification=classify_oracle({8,8},d,p,expected);
            require(classification[1]==(inverted!=(scenario==5)),"zero-depth owner masking scenario is not discriminating");
            require(expected.motion[c][3]==unsigned(inverted!=(scenario==5)),"signed-zero depth validity scenario is not discriminating");
        }
        std::vector<Word> motion(p.blocksW*p.blocksH,Word{0xa5a5a5a5u,0xa5a5a5a5u,0xa5a5a5a5u,0xa5a5a5a5u});
        std::vector<uint32_t> plane(p.blocksW*p.blocksH,p.skip?0u:0x5a5a5a5au);
        std::vector<Word> results(cells.size()*9+4,Word{0xcdcdcdcdu,0xcdcdcdcdu,0xcdcdcdcdu,0xcdcdcdcdu});
        auto motionBuffer=buffer(motion.data(),motion.size()*sizeof(Word));
        auto planeBuffer=buffer(plane.data(),plane.size()*4);
        auto resultBuffer=buffer(results.data(),results.size()*sizeof(Word));
        NSDictionary *buffers=@{@"QaFrame":buffer(&d,sizeof d),@"Fsr411LossyConstants":buffer(&p,sizeof p),
            @"QaRecords":buffer(records.data(),records.size()*sizeof(Record)),@"QaCells":buffer(cells.data(),cells.size()*sizeof(Cell)),
            @"QaResults":resultBuffer,@"ScratchBuffer":buffer(cache.data(),cache.size()*sizeof(Word)),
            @"Fsr411LossyMotion":motionBuffer,@"Fsr411LossyPlane":planeBuffer};
        auto td=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR32Float width:d.rw height:d.rh mipmapped:NO];
        td.storageMode=MTLStorageModeShared;td.usage=MTLTextureUsageShaderRead;
        auto depthTexture=[device newTextureWithDescriptor:td]; require(depthTexture!=nil,"Depth fixture allocation failed");
        [depthTexture replaceRegion:MTLRegionMake2D(0,0,d.rw,d.rh) mipmapLevel:0 withBytes:depth.data() bytesPerRow:d.rw*4];
        auto command=[queue commandBuffer];auto encoder=[command computeCommandEncoder];
        encode(encoder,@"qa_record",buffers,@{@"r_depth":depthTexture},d.records);
        encode(encoder,@"qa_query",buffers,@{},d.queries);
        [encoder endEncoding];[command commit];[command waitUntilCompleted];
        require(command.status==MTLCommandBufferStatusCompleted,std::string(name)+": GPU command failed");
        const auto actualMotion=static_cast<const Word *>(motionBuffer.contents);
        const auto actualPlane=static_cast<const uint32_t *>(planeBuffer.contents);
        for (size_t i=0;i<motion.size();++i) {
            for (unsigned c=0;c<4;++c) {
                const bool nanPayload=c==2 && std::isnan(floating(actualMotion[i][c])) && std::isnan(floating(expected.motion[i][c]));
                require(actualMotion[i][c]==expected.motion[i][c] || nanPayload,
                        std::string(name)+": motion mismatch block="+std::to_string(i)+" component="+std::to_string(c)+
                        " actual="+std::to_string(actualMotion[i][c])+" expected="+std::to_string(expected.motion[i][c]));
            }
            require(actualPlane[i]==expected.plane[i],std::string(name)+": ownership mismatch block="+std::to_string(i));
        }
        const auto actual=static_cast<const Word *>(resultBuffer.contents);
        for (size_t i=0;i<cells.size();++i) {
            const auto cell=cells[i];
            const Word own=skip?expected.motion[unsigned(cell[1]/2)*p.blocksW+unsigned(cell[0]/2)]:Word{0,0,0,0};
            const auto classification=skip?classify_oracle(cell,d,p,expected):std::array<bool,2>{false,false};
            require(actual[i*9]==Word{own[0],own[1],unsigned(classification[0]),unsigned(classification[1])},
                    std::string(name)+": classification mismatch cell="+std::to_string(i));
            for (unsigned pk=0;pk<4;++pk) {
                const int px=cell[0]*2+int(pk%2),py=cell[1]*2+int(pk/2);
                const int sx=px+signed_int(own[0]),sy=py+signed_int(own[1]);
                const int rawX=floor_div(sx,2),rawY=floor_div(sy,2);
                const unsigned x=unsigned(std::clamp(rawX,0,int(d.w/2)-1)),y=unsigned(std::clamp(rawY,0,int(d.h/2)-1));
                const unsigned parity=unsigned(sx-2*rawX)+2u*unsigned(sy-2*rawY);
                require(actual[i*9+pk*2+1]==cache_value(x,y,d.w/2),std::string(name)+": wrong cached cell at="+std::to_string(i)+" tap="+std::to_string(pk));
                require(actual[i*9+pk*2+2]==Word{parity,0x411u,unsigned(px),unsigned(py)},std::string(name)+": wrong CT parity at="+std::to_string(i)+" tap="+std::to_string(pk));
            }
        }
        for (size_t i=cells.size()*9;i<results.size();++i) require(actual[i]==results[i],std::string(name)+": results overrun");
        ++cases;
    } }
};

int main(int argc,char **argv) { @autoreleasepool {
    if(argc!=2){std::fprintf(stderr,"usage: %s <fixture-directory>\n",argv[0]);return 2;}
    std::fesetround(FE_TONEAREST);
    Fixture fixture(argv[1]);
    const Cell shifts[]={{{-3,-3}},{{-2,-1}},{{-1,2}},{{0,0}},{{1,1}},{{2,0}},{{3,-2}},{{254,-254}},{{-254,254}}};
    for(const auto &shift:shifts) fixture.test("uniform motion",float(shift[0]),float(shift[1]));
    fixture.test("nearer owner",0,0,1);
    fixture.test("owner depth tie",0,0,2);
    fixture.test("invalid and signed rounding",1,1,3);
    fixture.test("zero-depth owner masks occlusion",0,0,4);
    fixture.test("negative-zero owner",0,0,5);
    fixture.test("full frame ignores motion cache",3,-1,0,false);
    std::printf("lossy shader QA tier=%u inverted=%d low-res=%d: %zu cases, %zu assertions passed\n",
                [fixture.metadata[@"tier"] unsignedIntValue],fixture.inverted,fixture.low,cases,checks);
    return 0;
} }
