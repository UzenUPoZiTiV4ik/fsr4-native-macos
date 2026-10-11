// SPDX-License-Identifier: MIT
// Verify diagnostic copies against independently generated CPU texels and
// the caller's final output. This does not validate game motion conventions.
#include "../src/fsr411_runtime.h"
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <vector>

namespace {
constexpr uint32_t rw=37, rh=23, uw=72, uh=48;
constexpr float sentinel=-9.f;
size_t assertions=0;

void require(bool condition,const std::string &message) {
    ++assertions;
    if(!condition) throw std::runtime_error(message);
}
uint16_t half_bits(float value) {
    const _Float16 h=(_Float16)value; uint16_t bits;
    std::memcpy(&bits,&h,sizeof bits); return bits;
}
float half_value(uint16_t bits) {
    _Float16 h; std::memcpy(&h,&bits,sizeof h); return float(h);
}
id<MTLTexture> texture(id<MTLDevice> device,MTLPixelFormat format,Fsr4InputSurface surface) {
    auto d=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format
        width:surface.width height:surface.height mipmapped:NO];
    d.storageMode=MTLStorageModeShared; d.usage=MTLTextureUsageShaderRead|MTLTextureUsageShaderWrite;
    auto result=[device newTextureWithDescriptor:d];
    require(result!=nil,"input texture allocation failed"); return result;
}
fsr411::Frame make_frame(id<MTLDevice> device) {
    fsr411::Frame f;
    f.input.geometry={{rw+11,rh+7,3,2},{rw+13,rh+9,5,3},
                      {rw+17,rh+11,7,4},{uw+19,uh+13,9,6},64,40};
    const auto &g=f.input.geometry;
    f.color=texture(device,MTLPixelFormatRGBA16Float,g.color);
    f.depth=texture(device,MTLPixelFormatR32Float,g.depth);
    f.motion=texture(device,MTLPixelFormatRG16Float,g.motion);
    f.output=texture(device,MTLPixelFormatRGBA16Float,g.output);
    f.input.rw=rw; f.input.rh=rh; f.input.uw=uw; f.input.uh=uh;
    f.input.mvScaleX=-1.75f; f.input.mvScaleY=2.5f;
    f.input.preExposure=1.25f; f.input.inputContentProperties=1;
    return f;
}
struct Pixels { std::vector<float> color,depth,motion,output; };

Pixels upload(fsr411::Frame &f,unsigned index) {
    const auto &g=f.input.geometry;
    Pixels expected;
    expected.color.resize(size_t(rw)*rh*4);
    expected.depth.resize(size_t(rw)*rh);
    expected.motion.resize(size_t(rw)*rh*2);
    std::vector<uint16_t> color(size_t(g.color.width)*g.color.height*4,half_bits(7.f));
    std::vector<float> depth(size_t(g.depth.width)*g.depth.height,.9375f);
    std::vector<uint16_t> motion(size_t(g.motion.width)*g.motion.height*2,half_bits(17.f));
    for(uint32_t y=0;y<rh;++y) for(uint32_t x=0;x<rw;++x) {
        const size_t pixel=size_t(y)*rw+x;
        const float rgb[4]={.03125f+float((x+index*7)%37)/48.f,
                            .0625f+float((y+index*3)%23)/32.f,
                            float(1+(x*3+y*5+index*11)%19)/24.f,1.f};
        for(unsigned c=0;c<4;++c) {
            const uint16_t packed=half_bits(rgb[c]);
            color[((size_t(y+g.color.offsetY)*g.color.width)+x+g.color.offsetX)*4+c]=packed;
            expected.color[pixel*4+c]=half_value(packed);
        }
        const float z=.125f+float((x+y*3+index)%31)/64.f;
        depth[size_t(y+g.depth.offsetY)*g.depth.width+x+g.depth.offsetX]=z;
        expected.depth[pixel]=z;
        const float mv[2]={float(int((x+index*3)%13)-6)/8.f,
                           float(int((y+index*5)%11)-5)/16.f};
        for(unsigned c=0;c<2;++c) {
            const uint16_t packed=half_bits(mv[c]);
            motion[((size_t(y+g.motion.offsetY)*g.motion.width)+x+g.motion.offsetX)*2+c]=packed;
            expected.motion[pixel*2+c]=half_value(packed);
        }
    }
    [f.color replaceRegion:MTLRegionMake2D(0,0,g.color.width,g.color.height) mipmapLevel:0
        withBytes:color.data() bytesPerRow:g.color.width*8];
    [f.depth replaceRegion:MTLRegionMake2D(0,0,g.depth.width,g.depth.height) mipmapLevel:0
        withBytes:depth.data() bytesPerRow:g.depth.width*4];
    [f.motion replaceRegion:MTLRegionMake2D(0,0,g.motion.width,g.motion.height) mipmapLevel:0
        withBytes:motion.data() bytesPerRow:g.motion.width*4];
    std::vector<uint16_t> output(size_t(g.output.width)*g.output.height*4,half_bits(sentinel));
    [f.output replaceRegion:MTLRegionMake2D(0,0,g.output.width,g.output.height) mipmapLevel:0
        withBytes:output.data() bytesPerRow:g.output.width*8];
    f.input.reset=index==0; f.input.jitterX=index?-.25f:.375f; f.input.jitterY=index?.25f:-.125f;
    return expected;
}

std::vector<float> read_float(id<MTLTexture> t,unsigned channels) {
    std::vector<float> result(t.width*t.height*channels);
    [t getBytes:result.data() bytesPerRow:t.width*channels*sizeof(float)
        fromRegion:MTLRegionMake2D(0,0,t.width,t.height) mipmapLevel:0];
    return result;
}
std::vector<float> read_output(const fsr411::Frame &f) {
    const auto &g=f.input.geometry.output;
    std::vector<uint16_t> packed(size_t(g.width)*g.height*4);
    [f.output getBytes:packed.data() bytesPerRow:g.width*8
        fromRegion:MTLRegionMake2D(0,0,g.width,g.height) mipmapLevel:0];
    std::vector<float> result(size_t(uw)*uh*4);
    float lo=1e30f,hi=-1e30f;
    for(uint32_t y=0;y<g.height;++y) for(uint32_t x=0;x<g.width;++x) {
        const bool inside=x>=g.offsetX && y>=g.offsetY && x-g.offsetX<uw && y-g.offsetY<uh;
        for(unsigned c=0;c<4;++c) {
            const uint16_t bits=packed[(size_t(y)*g.width+x)*4+c];
            if(!inside) require(bits==half_bits(sentinel),"output copy overwrote a pixel outside its active region");
            else {
                const float value=half_value(bits);
                result[(size_t(y-g.offsetY)*uw+x-g.offsetX)*4+c]=value;
                if(c<3) {
                    require(std::isfinite(value) && value>=0,"invalid final output RGB");
                    lo=std::min(lo,value); hi=std::max(hi,value);
                }
            }
        }
    }
    require(hi>lo,"final output is constant"); return result;
}
Pixels read_snapshot(const fsr411::FrameSnapshot &s) {
    return {read_float(s.color,4),read_float(s.depth,1),read_float(s.motion,2),read_float(s.output,4)};
}
void exact(const std::vector<float> &actual,const std::vector<float> &expected,const char *label) {
    require(actual.size()==expected.size(),std::string(label)+": size mismatch");
    for(size_t i=0;i<actual.size();++i)
        require(!std::memcmp(&actual[i],&expected[i],sizeof(float)),std::string(label)+": mismatch at float "+std::to_string(i));
}
void exact(const Pixels &actual,const Pixels &expected) {
    exact(actual.color,expected.color,"snapshot color"); exact(actual.depth,expected.depth,"snapshot depth");
    exact(actual.motion,expected.motion,"snapshot motion"); exact(actual.output,expected.output,"snapshot output");
}
bool same_surface(Fsr4InputSurface a,Fsr4InputSurface b) {
    return a.width==b.width && a.height==b.height && a.offsetX==b.offsetX && a.offsetY==b.offsetY;
}
void metadata(const fsr411::FrameSnapshot &s,const Fsr4InputDesc &f) {
    const auto &a=s.input; const auto &g=f.geometry; const auto &p=s.prepared;
    require(same_surface(a.geometry.color,g.color) && same_surface(a.geometry.depth,g.depth)
            && same_surface(a.geometry.motion,g.motion) && same_surface(a.geometry.output,g.output),"snapshot lost source geometry");
    require(a.geometry.maxInputWidth==g.maxInputWidth && a.geometry.maxInputHeight==g.maxInputHeight,"snapshot lost descriptor extents");
    require(a.rw==f.rw && a.rh==f.rh && a.uw==f.uw && a.uh==f.uh,"snapshot lost logical extents");
    require(a.jitterX==f.jitterX && a.jitterY==f.jitterY && a.mvScaleX==f.mvScaleX && a.mvScaleY==f.mvScaleY
            && a.preExposure==f.preExposure,"snapshot changed input float constants");
    require(a.reset==f.reset && a.depthReversed==f.depthReversed && a.jitteredMV==f.jitteredMV
            && a.outputResMV==f.outputResMV && a.autoExposure==f.autoExposure
            && a.inputContentProperties==f.inputContentProperties && a.hasExposureTexture==f.hasExposureTexture,"snapshot changed input flags");
    require(p.rw==rw && p.rh==rh && p.uw==uw && p.uh==uh && p.alignedUW==uw && p.alignedUH==uh
            && p.mvW==rw && p.mvH==rh,"snapshot prepared extents mismatch");
    require(p.copyMask==(FSR4_INPUT_COPY_COLOR|FSR4_INPUT_COPY_DEPTH|FSR4_INPUT_COPY_MOTION|FSR4_INPUT_COPY_OUTPUT),"test did not exercise all region copies");
    require(p.reset && (p.resetReasons&FSR4_INPUT_RESET_FIRST_FRAME),"snapshot lost first-frame reset metadata");
    require(p.mvScale[0]==f.mvScaleX/rw && p.mvScale[1]==f.mvScaleY/rh && p.jitter[0]==f.jitterX
            && p.jitter[1]==f.jitterY && p.preExposure==f.preExposure,"snapshot prepared constants mismatch");
}

struct Pending { id command=nil; id<MTL4CommandAllocator> allocator=nil; };
struct Queue {
    id<MTLDevice> device;
    id<MTLCommandQueue> classic=nil;
    id<MTL4CommandQueue> metal4=nil;
    id<MTLSharedEvent> completed=nil;
    uint64_t sequence=0;
    bool useMetal4;
    Queue(id<MTLDevice> d,bool use):device(d),useMetal4(use) {
        if(useMetal4) {
            if(@available(macOS 26.0,*)) {
                metal4=[device newMTL4CommandQueue]; completed=[device newSharedEvent];
                require(metal4!=nil && completed!=nil,"Metal 4 queue/event allocation failed");
            } else throw std::runtime_error("Metal 4 requires macOS 26");
        } else {classic=[device newCommandQueue]; require(classic!=nil,"classic queue allocation failed");}
    }
    Pending encode(fsr411::Runtime &runtime,fsr411::View &view,const fsr411::Frame &frame,const fsr411::Options &options) {
        Pending p;
        if(useMetal4) {
            if(@available(macOS 26.0,*)) {
                p.allocator=[device newCommandAllocator]; p.command=[device newCommandBuffer];
                require(p.allocator!=nil && p.command!=nil,"Metal 4 command allocation failed");
                [p.command beginCommandBufferWithAllocator:p.allocator];
            }
        } else p.command=[classic commandBuffer];
        require(p.command!=nil,"command buffer allocation failed");
        std::string error; const bool ok=runtime.encode(p.command,view,frame,options,error);
        if(useMetal4) [p.command endCommandBuffer];
        require(ok,"snapshot encode failed: "+error); return p;
    }
    void submit(Pending &p) {
        if(useMetal4) {
            id<MTL4CommandBuffer> commands[]={p.command}; [metal4 commit:commands count:1];
            [metal4 signalEvent:completed value:++sequence];
            require([completed waitUntilSignaledValue:sequence timeoutMS:20000],"Metal 4 snapshot execution timed out");
        } else {
            auto cb=(id<MTLCommandBuffer>)p.command; [cb commit]; [cb waitUntilCompleted];
            require(cb.status==MTLCommandBufferStatusCompleted,"snapshot GPU failure: "+fsr411::metal_error(cb.error));
        }
    }
};

void run(id<MTLDevice> device,const std::string &root,bool compile,bool metal4) {
    fsr411::Runtime runtime(device,root,root+"/build"); Queue queue(device,metal4);
    fsr411::Options options; options.backend=fsr411::Backend::Native; options.profile=fsr411::Profile::Exact;
    options.allowCompile=compile; options.rcas=true; options.sharpness=.35f;
    require(!options.snapshot,"snapshots must be disabled by default");
    auto f=make_frame(device); auto expected=upload(f,0);
    std::vector<float> baseline;
    {
        // The same graph without a snapshot must return identical output and
        // must not allocate diagnostic resources/tables in a fresh frame slot.
        fsr411::View control; auto pending=queue.encode(runtime,control,f,options);
        require(control.slots.size()==1 && !control.slots[0]->snapshot,"null option allocated a snapshot");
        for(NSString *name in control.slots[0]->tables)
            require(![name hasPrefix:@"snapshot"],"null option allocated a snapshot argument table");
        queue.submit(pending); baseline=read_output(f);
    }
    expected=upload(f,0); const auto originalInput=f.input;
    fsr411::View view; auto snapshot=std::make_shared<fsr411::FrameSnapshot>();
    require(!snapshot->ready() && !snapshot->completion,"new snapshot is already ready");
    auto capturedOptions=options; capturedOptions.snapshot=snapshot;
    auto captured=queue.encode(runtime,view,f,capturedOptions);
    require(!snapshot->ready() && snapshot->completion!=nil,"unsubmitted snapshot reports ready");
    require(__atomic_load_n((uint32_t *)snapshot->completion.contents,__ATOMIC_ACQUIRE)==0u,"unsubmitted token is not zero");
    const id<MTLTexture> textures[]={snapshot->color,snapshot->depth,snapshot->motion,snapshot->output};
    const MTLPixelFormat formats[]={MTLPixelFormatRGBA32Float,MTLPixelFormatR32Float,MTLPixelFormatRG32Float,MTLPixelFormatRGBA32Float};
    for(unsigned i=0;i<4;++i) {
        require(textures[i].storageMode==MTLStorageModeShared && textures[i].pixelFormat==formats[i],"snapshot texture is not CPU-readable float32");
        require(textures[i].width==(i==3?uw:rw) && textures[i].height==(i==3?uh:rh),"snapshot texture includes allocation margins");
        require([view.slots[0]->resources containsObject:textures[i]],"frame slot does not retain a snapshot texture");
    }
    require(snapshot->completion!=view.slots[0]->constants && [view.slots[0]->resources containsObject:snapshot->completion],"completion token has no independent lifetime");
    metadata(*snapshot,originalInput); const auto originalPrepared=snapshot->prepared;
    // Drop both caller references before submission. The in-flight slot must
    // keep the snapshot alive even if its writer has not retained it yet.
    std::weak_ptr<fsr411::FrameSnapshot> weak=snapshot;
    capturedOptions.snapshot.reset(); snapshot.reset();
    require(!weak.expired(),"in-flight snapshot was released before GPU submission");
    queue.submit(captured); snapshot=weak.lock();
    require(snapshot && snapshot->ready(),"completed snapshot is not ready");
    require(__atomic_load_n((uint32_t *)snapshot->completion.contents,__ATOMIC_ACQUIRE)==1u,"completed token is not one");
    expected.output=read_output(f); exact(expected.output,baseline,"capture changed final RCAS output");
    const auto frozen=read_snapshot(*snapshot); exact(frozen,expected);

    upload(f,1); f.input.preExposure=1.75f;
    auto next=queue.encode(runtime,view,f,options);
    require(view.slots.size()==1 && !view.slots[0]->snapshot,"uncaptured frame retained the previous snapshot in its slot");
    require(snapshot->ready(),"next frame reset the previous completion token");
    for(id resource in @[snapshot->color,snapshot->depth,snapshot->motion,snapshot->output,snapshot->completion])
        require(![view.slots[0]->resources containsObject:resource],"uncaptured frame retained old capture resources");
    queue.submit(next);
    require(snapshot->ready() && __atomic_load_n((uint32_t *)snapshot->completion.contents,__ATOMIC_ACQUIRE)==1u,"later frame changed the completed token");
    metadata(*snapshot,originalInput);
    require(!std::memcmp(&snapshot->prepared,&originalPrepared,sizeof originalPrepared),"later frame changed prepared snapshot metadata");
    exact(read_snapshot(*snapshot),frozen);
    require(read_output(f)!=expected.output,"second input did not exercise a changed output");
    snapshot.reset(); require(weak.expired(),"retired snapshot leaked after caller release");
    std::printf("FSR 4.1.1 snapshots (%s): cropped float copies, metadata, token, RCAS output and lifetime passed; %zu assertions\n",
                metal4?"Metal 4":"classic Metal",assertions);
}
} // namespace

int main(int argc,char **argv) { @autoreleasepool { try {
    if(argc<2) {std::fprintf(stderr,"usage: fsr411_snapshot_test <repository> [--compile] [--mtl4]\n"); return 2;}
    bool compile=false,metal4=false;
    for(int i=2;i<argc;++i) {
        if(!std::strcmp(argv[i],"--compile")) compile=true;
        else if(!std::strcmp(argv[i],"--mtl4")) metal4=true;
        else throw std::runtime_error(std::string("unknown argument ")+argv[i]);
    }
    auto device=MTLCreateSystemDefaultDevice(); require(device!=nil,"no Metal device");
    run(device,argv[1],compile,metal4); return 0;
} catch(const std::exception &error) {
    std::fprintf(stderr,"Snapshot QA: %s\n",error.what()); return 1;
} } }
