// SPDX-License-Identifier: MIT
// Lite SIMD (M1-M4) must match AMD's Lite shaders translated to Metal
// (Backend::Reference) bit for bit: output, history, recurrent and exposure,
// both presets, sizes, subrects, resets, changing exposure, motion and RCAS.
// The exact float requantization (FSR411_FLOAT_EPILOGUE) is checked here.
#define FSR412_TEST_HELPERS_ONLY
#include "fsr412_optimization_test.mm"

int main(int argc,char **argv) { @autoreleasepool {try {
    require(argc>=2,"usage: fsr412_simd_parity_test <root> [mtl4]");
    const bool metal4=argc>2 && !std::strcmp(argv[2],"mtl4");
    auto dev=MTLCreateSystemDefaultDevice();auto queue=[dev newCommandQueue];
    require(dev && queue,"Metal unavailable");
    fsr412::Runtime runtime(dev,argv[1],std::string(argv[1])+"/build");
    for(unsigned mode=0;mode<4;++mode) {
        fsr412::View dll,simd;
        fsr412::Options options;options.ultraPerformance=mode>=2;options.optimize=0;
        for(unsigned index=0;index<6;++index) {
            auto a=frame(dev,index,mode%2),b=a;
            b.output=texture(dev,MTLPixelFormatRGBA16Float,a.output.width,a.output.height);
            std::vector<_Float16> sentinel(a.output.width*a.output.height*4,(_Float16)-9);
            for(auto t:{a.output,b.output}) [t replaceRegion:MTLRegionMake2D(0,0,t.width,t.height)
                mipmapLevel:0 withBytes:sentinel.data() bytesPerRow:t.width*8];
            options.rcas=index%2;options.sharpness=.498f;
            options.tier=index>=4?2160:1080;
            options.backend=fsr412::Backend::Reference;encode(runtime,dev,queue,dll,a,options,metal4);
            options.backend=fsr412::Backend::Simd;encode(runtime,dev,queue,simd,b,options,metal4);
            std::printf("checking SIMD parity mode=%u frame=%u API=%s\n",mode,index,metal4?"Metal4":"classic");
            compare(a.output,b.output,8,"SIMD output",true);
            compare(dll.history,simd.history,8,"SIMD history",true);
            compare(dll.recurrent,simd.recurrent,4,"SIMD recurrent",true);
            compare(dll.autoExposure,simd.autoExposure,4,"SIMD exposure",true);
        }
    }
    puts("PASS: Lite SIMD matches the AMD Lite shaders bit for bit across both presets");
    return 0;
} catch(const std::exception &e) {std::fprintf(stderr,"FSR SIMD parity: %s\n",e.what());return 1;}} }
