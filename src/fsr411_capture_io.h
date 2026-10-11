// Opt-in, bounded game-frame capture for checking the DLSS/MetalFX contract.
// The render thread polls a tiny request file and records GPU copies. Status
// and image writes, directory creation and retirement polling run on workers.
#pragma once
#include "fsr411_runtime.h"
#include "fsr4_log.h"
#include <atomic>
#include <chrono>
#include <thread>
#include <sys/stat.h>
#include <unistd.h>
#include <dispatch/dispatch.h>

namespace fsr411_capture {

inline double now_ms() {
    using namespace std::chrono;
    return duration<double,std::milli>(steady_clock::now().time_since_epoch()).count();
}

inline void publish(const std::string &root,const char *state,NSString *message,const std::string &directory) {
    NSString *text=[NSString stringWithFormat:@"state=%s\nmessage=%@\ndirectory=%s\n",state,message,directory.c_str()];
    NSString *path=fsr411::ns(root+"/dist/crossover/capture-status.conf");
    static dispatch_queue_t queue=dispatch_queue_create("org.fsr4mac.capture-status",DISPATCH_QUEUE_SERIAL);
    dispatch_async(queue,^{
        @autoreleasepool {
            [text writeToFile:path atomically:YES encoding:NSUTF8StringEncoding error:nil];
        }
    });
}

struct RecordedFrame {
    std::shared_ptr<fsr411::FrameSnapshot> snapshot;
    NSDictionary *metadata=nil;
};
struct Job {
    std::string root,directory;
    unsigned count=8;
    unsigned requestedCount=8;
    uint64_t reservedBytes=0;
    void *owner=nullptr;
    bool hasOwner=false,writing=false;
    double start=0;
    std::vector<RecordedFrame> frames;
    std::atomic<bool> done{false};
};

inline bool write_texture(id<MTLTexture> texture,size_t channels,const std::string &path,std::string &error) {
    const size_t row=size_t(texture.width)*channels*sizeof(float);
    NSMutableData *bytes=[NSMutableData dataWithLength:row*texture.height];
    if(!bytes) {error="cannot allocate capture readback";return false;}
    [texture getBytes:bytes.mutableBytes bytesPerRow:row
           fromRegion:MTLRegionMake2D(0,0,texture.width,texture.height) mipmapLevel:0];
    NSError *e=nil;
    if(![bytes writeToFile:fsr411::ns(path) options:NSDataWritingAtomic error:&e]) {
        error=fsr411::utf8(e.localizedDescription);return false;
    }
    return true;
}

inline void write_job(std::shared_ptr<Job> job) {
    @autoreleasepool {
        NSMutableArray *saved=[NSMutableArray array];
        std::string error;
        NSError *directoryError=nil;
        if(![[NSFileManager defaultManager] createDirectoryAtPath:fsr411::ns(job->directory)
                      withIntermediateDirectories:YES attributes:nil error:&directoryError]) {
            publish(job->root,"error",@"Could not create the diagnostics folder.",job->directory);
            fsr4_logf("4.1.1 capture directory failed: %s",fsr411::utf8(directoryError.localizedDescription).c_str());
            job->done.store(true,std::memory_order_release);return;
        }
        const auto write_manifest=[&](const char *state) {
            NSDictionary *manifest=@{@"schema_version":@1,@"fsr_version":@"4.1.1",@"status":fsr411::ns(state),
                @"format":@"little-endian float32, C row-major",@"requested_frames":@(job->requestedCount),
                @"frames":saved,@"error":fsr411::ns(error)};
            NSData *data=[NSJSONSerialization dataWithJSONObject:manifest options:NSJSONWritingPrettyPrinted error:nil];
            return data && [data writeToFile:fsr411::ns(job->directory+"/capture.json") options:NSDataWritingAtomic error:nil];
        };
        publish(job->root,"writing",@"Saving diagnostic frames…",job->directory);
        for(const auto &frame:job->frames) {
            const double deadline=now_ms()+30000;
            while(!frame.snapshot->ready() && now_ms()<deadline)
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            if(!frame.snapshot->ready()) {error="capture GPU completion timed out";break;}
            NSDictionary *files=frame.metadata[@"files"];
            const id<MTLTexture> textures[]={frame.snapshot->color,frame.snapshot->depth,frame.snapshot->motion,frame.snapshot->output};
            NSString *names[]={@"color",@"depth",@"motion",@"output"};
            const size_t channels[]={4,1,2,4};
            bool ok=true;
            @try {
                for(unsigned i=0;i<4 && ok;++i)
                    ok=write_texture(textures[i],channels[i],job->directory+"/"+fsr411::utf8(files[names[i]]),error);
            } @catch(NSException *exception) {
                error=fsr411::utf8(exception.reason);ok=false;
            }
            if(!ok)break;
            [saved addObject:frame.metadata];
            if(!write_manifest("writing")) {error="cannot write capture manifest";break;}
        }
        if(error.empty() && saved.count==job->count && write_manifest("complete")) {
            publish(job->root,"complete",[NSString stringWithFormat:@"Done: recorded %lu frames.",(unsigned long)saved.count],job->directory);
            fsr4_logf("4.1.1 capture complete frames=%lu directory=%s",(unsigned long)saved.count,job->directory.c_str());
        } else {
            if(error.empty())error="incomplete capture";
            write_manifest("error");
            publish(job->root,"error",@"Could not finish recording; see hook.log.",job->directory);
            fsr4_logf("4.1.1 capture failed: %s directory=%s",error.c_str(),job->directory.c_str());
        }
        job->done.store(true,std::memory_order_release);
    }
}

class Recorder {
public:
    explicit Recorder(std::string root):root_(std::move(root)){}

    std::shared_ptr<fsr411::FrameSnapshot> request(void *owner,const fsr411::Frame &frame) {
        const double now=now_ms();
        if(job_ && job_->done.load(std::memory_order_acquire))job_.reset();
        if(job_ && !job_->writing && now>job_->start+10000)finish_partial("capture recording timed out");
        if(!job_ && now>=nextCheck_) {
            nextCheck_=now+250;
            const std::string path=root_+"/dist/crossover/capture.request";
            const std::string claimed=path+"."+std::to_string(getpid());
            if(rename(path.c_str(),claimed.c_str())==0) {
                unsigned frames=8,delay=1500;
                FILE *f=fopen(claimed.c_str(),"r");
                if(f) {
                    char line[128];
                    while(fgets(line,sizeof line,f)) {
                        unsigned value;
                        if(sscanf(line,"frames=%u",&value)==1)frames=std::clamp(value,2u,16u);
                        if(sscanf(line,"delay_ms=%u",&value)==1)delay=std::min(value,10000u);
                    }
                    fclose(f);
                }
                unlink(claimed.c_str());
                job_=std::make_shared<Job>();job_->root=root_;job_->count=frames;job_->requestedCount=frames;job_->start=now+delay;
                const auto stamp=std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
                job_->directory=root_+"/dist/crossover/captures/"+std::to_string(stamp)+"-"+std::to_string(getpid());
                publish(root_,"waiting",@"Move the camera: recording starts in 1.5 s.",job_->directory);
                fsr4_logf("4.1.1 capture requested frames=%u delayMs=%u directory=%s",frames,delay,job_->directory.c_str());
            }
        }
        if(!job_ || job_->writing || now<job_->start || (job_->hasOwner && job_->owner!=owner))return nullptr;
        const auto &in=frame.input;
        const uint64_t mvPixels=in.outputResMV==1?uint64_t(in.uw)*in.uh:uint64_t(in.rw)*in.rh;
        const uint64_t bytes=uint64_t(in.rw)*in.rh*20+mvPixels*8+uint64_t(in.uw)*in.uh*16;
        constexpr uint64_t budget=512ull*1024*1024;
        if(!bytes || bytes>budget) {
            publish(root_,"error",@"Resolution is too large for a diagnostic recording.",job_->directory);
            job_.reset();return nullptr;
        }
        if(!job_->hasOwner) {
            if(budget/bytes<2) {
                publish(root_,"error",@"Diagnostics need two frames; lower the resolution.",job_->directory);
                job_.reset();return nullptr;
            }
            job_->owner=owner;job_->hasOwner=true;
            job_->count=std::min(job_->count,unsigned(budget/bytes));
        }
        if(bytes>budget-job_->reservedBytes) {
            finish_partial("capture reached its 512 MiB limit after a render-size change");
            return nullptr;
        }
        return std::make_shared<fsr411::FrameSnapshot>();
    }

    void record(void *owner,const fsr411::View &view,const fsr411::Options &options,
                const std::shared_ptr<fsr411::FrameSnapshot> &snapshot) {
        if(!snapshot || !job_ || job_->writing)return;
        const auto &in=snapshot->input;const auto &p=snapshot->prepared;
        const unsigned index=unsigned(job_->frames.size());
        const auto name=[&](const char *surface){return [NSString stringWithFormat:@"f%03u_%s.bin",index,surface];};
        const auto surface=[](Fsr4InputSurface s){return @{@"width":@(s.width),@"height":@(s.height),@"offsetX":@(s.offsetX),@"offsetY":@(s.offsetY)};};
        NSDictionary *files=@{@"color":name("color"),@"depth":name("depth"),@"motion":name("motion"),@"output":name("output")};
        NSDictionary *metadata=@{@"index":@(index),@"ownerFrame":@(view.encodes),@"owner":[NSString stringWithFormat:@"%p",owner],
            @"rw":@(p.rw),@"rh":@(p.rh),@"uw":@(p.uw),@"uh":@(p.uh),@"mvW":@(p.mvW),@"mvH":@(p.mvH),
            @"jitter":@[@(in.jitterX),@(in.jitterY)],@"mvScale":@[@(in.mvScaleX),@(in.mvScaleY)],
            @"jitteredMV":@(p.jitteredMV),@"outputResMV":@(!p.lowResMV),@"preExposure":@(p.preExposure),
            @"reset":@(p.reset),@"requestedReset":@(in.reset),@"depthReversed":@(p.depthReversed),
            @"autoExposure":@(p.autoExposure),@"copyMask":@(p.copyMask),@"resetReasons":@(p.resetReasons),
            @"geometry":@{@"color":surface(in.geometry.color),@"depth":surface(in.geometry.depth),
                @"motion":surface(in.geometry.motion),@"output":surface(in.geometry.output),
                @"maxInputWidth":@(in.geometry.maxInputWidth),@"maxInputHeight":@(in.geometry.maxInputHeight)},
            @"inputFlags":@{@"jitteredMV":@(in.jitteredMV),@"outputResMV":@(in.outputResMV),
                @"autoExposure":@(in.autoExposure),@"inputContentProperties":@(in.inputContentProperties)},
            @"defaultedFlags":@(p.defaultedFlags),@"rcas":@(options.rcas),@"sharpness":@(options.sharpness),
            @"normalizedMvScale":@[@(p.mvScale[0]),@(p.mvScale[1])],
            @"jitterCancellation":@[@(p.jitterCancellation[0]),@(p.jitterCancellation[1])],
            @"profile":@(fsr411::profile_name(options.profile)),
            @"backend":fsr411::ns(view.backendLabel),@"variant":fsr411::ns(view.variantKey),
            @"inferenceReused":@(view.skippedInference),@"files":files};
        job_->frames.push_back({snapshot,metadata});
        job_->reservedBytes+=uint64_t(p.rw)*p.rh*20+uint64_t(p.mvW)*p.mvH*8+uint64_t(p.uw)*p.uh*16;
        publish(root_,"recording",[NSString stringWithFormat:@"Recorded %u of %u frames…",index+1,job_->count],job_->directory);
        if(job_->frames.size()>=job_->count) {
            job_->writing=true;
            std::thread(write_job,job_).detach();
        }
    }

    void failed(void *owner,const std::string &reason) {
        if(!job_ || job_->writing || (job_->hasOwner && job_->owner!=owner))return;
        publish(root_,"error",@"Recording stopped: the scaler rejected a frame. See hook.log.",job_->directory);
        fsr4_logf("4.1.1 capture interrupted: %s",reason.c_str());
        job_.reset();
    }

    void release_owner(void *owner) {
        if(job_ && job_->hasOwner && job_->owner==owner && !job_->writing)
            finish_partial("capture owner was released");
    }

private:
    void finish_partial(const char *reason) {
        if(!job_ || job_->writing)return;
        fsr4_logf("4.1.1 capture recording stopped: %s frames=%lu",reason,(unsigned long)job_->frames.size());
        if(job_->frames.size()>=2) {
            job_->count=unsigned(job_->frames.size());job_->writing=true;
            std::thread(write_job,job_).detach();
        } else {
            publish(root_,"error",@"Not enough frames. Record again in game.",job_->directory);
            job_.reset();
        }
    }
    std::string root_;
    double nextCheck_=0;
    std::shared_ptr<Job> job_;
};
}
