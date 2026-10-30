// SPDX-License-Identifier: MIT
// Two runtimes (root/builddir pairs) must produce bit-identical optimized
// Lite SIMD frames: output, history, recurrent and exposure, both presets,
// sizes, subrects, tiers, resets, exposure, motion, RCAS and the pixelization
// fix. tests/run_fsr412_runtime_ab.py builds the second root with one exact
// source transform disabled.
#define FSR412_TEST_HELPERS_ONLY
#include "fsr412_optimization_test.mm"

int main(int argc,char **argv) { @autoreleasepool {try {
    require(argc>=5,"usage: fsr412_runtime_ab_test <rootA> <builddirA> <rootB> <builddirB> [mtl4]");
    const bool metal4=argc>5 && !std::strcmp(argv[5],"mtl4");
    auto dev=MTLCreateSystemDefaultDevice();auto queue=[dev newCommandQueue];
    require(dev && queue,"Metal unavailable");
    fsr412::Runtime first(dev,argv[1],argv[2]),second(dev,argv[3],argv[4]);
    for(unsigned mode=0;mode<8;++mode) {
        fsr412::View a,b;
        fsr412::Options options;options.backend=fsr412::Backend::Simd;options.optimize=1;
        options.ultraPerformance=mode&2;options.pixelFix=mode&4;
        for(unsigned index=0;index<6;++index) {
            auto fa=frame(dev,index,mode&1),fb=fa;
            fb.output=texture(dev,MTLPixelFormatRGBA16Float,fa.output.width,fa.output.height);
            std::vector<_Float16> sentinel(fa.output.width*fa.output.height*4,(_Float16)-9);
            for(auto t:{fa.output,fb.output}) [t replaceRegion:MTLRegionMake2D(0,0,t.width,t.height)
                mipmapLevel:0 withBytes:sentinel.data() bytesPerRow:t.width*8];
            options.rcas=index%2;options.sharpness=.498f;
            options.tier=index>=4?2160:1080;
            encode(first,dev,queue,a,fa,options,metal4);
            encode(second,dev,queue,b,fb,options,metal4);
            std::printf("checking runtime A/B mode=%u frame=%u API=%s\n",mode,index,metal4?"Metal4":"classic");
            compare(fa.output,fb.output,8,"A/B output",true);
            compare(a.history,b.history,8,"A/B history",true);
            compare(a.recurrent,b.recurrent,4,"A/B recurrent",true);
            compare(a.autoExposure,b.autoExposure,4,"A/B exposure",true);
        }
    }
    puts("PASS: optimized Lite SIMD frames are bit-identical between the two runtimes");
    return 0;
} catch(const std::exception &e) {std::fprintf(stderr,"FSR runtime A/B: %s\n",e.what());return 1;}} }
