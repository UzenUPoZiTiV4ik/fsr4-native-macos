// Removing pad dispatches is valid only when SIMD loads zero its own halo.
#define FSR412_TEST_HELPERS_ONLY
#include "fsr412_optimization_test.mm"

struct PaddingTrace final : fsr412::StageProfiler {
    unsigned pads=0;
    void begin_stage(id __strong &,const std::string &name) override {
        if(name.rfind("pad",0)==0) ++pads;
    }
    void end_stage(id) override {}
};

int main(int argc,char **argv) { @autoreleasepool {try {
    require(argc>=3,"usage: fsr412_padding_test <root> <builddir> [mtl4]");
    const bool metal4=argc>3 && !std::strcmp(argv[3],"mtl4");
    auto dev=MTLCreateSystemDefaultDevice();auto queue=[dev newCommandQueue];
    require(dev && queue,"Metal unavailable");
    fsr412::Runtime runtime(dev,argv[1],argv[2]);
    for(unsigned mode=0;mode<4;++mode) {
        fsr412::View baseline,tuned;
        fsr412::Options options;options.backend=fsr412::Backend::Simd;
        options.ultraPerformance=mode>=2;
        for(unsigned index=0;index<6;++index) {
            auto a=frame(dev,index,mode%2),b=a;
            b.output=texture(dev,MTLPixelFormatRGBA16Float,a.output.width,a.output.height);
            std::vector<_Float16> sentinel(a.output.width*a.output.height*4,(_Float16)-9);
            for(auto t:{a.output,b.output}) [t replaceRegion:MTLRegionMake2D(0,0,t.width,t.height)
                mipmapLevel:0 withBytes:sentinel.data() bytesPerRow:t.width*8];
            options.rcas=index%2;options.sharpness=.498f;
            options.tier=index>=4?2160:1080;
            PaddingTrace trace;
            options.skipSimdPadding=false;options.stageProfiler=metal4?nullptr:&trace;
            encode(runtime,dev,queue,baseline,a,options,metal4);
            if(!metal4)require(trace.pads==5,"baseline did not run its five padding dispatches");
            trace.pads=0;
            options.skipSimdPadding=true;
            encode(runtime,dev,queue,tuned,b,options,metal4);
            if(!metal4)require(trace.pads==0,"tuned path still dispatched padding");
            std::printf("checking padding mode=%u frame=%u API=%s\n",mode,index,metal4?"Metal4":"classic");
            compare(a.output,b.output,8,"padding output",true);
            compare(baseline.history,tuned.history,8,"padding history",true);
            compare(baseline.recurrent,tuned.recurrent,4,"padding recurrent",true);
            compare(baseline.autoExposure,tuned.autoExposure,4,"padding exposure",true);
        }
    }
    puts("PASS: pad dispatch removal is byte-identical across both Lite presets, subrects, sizes, resets, exposure, motion and RCAS");
    return 0;
} catch(const std::exception &e) {std::fprintf(stderr,"FSR SIMD padding: %s\n",e.what());return 1;}} }
