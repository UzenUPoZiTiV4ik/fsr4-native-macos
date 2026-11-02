// SPDX-License-Identifier: MIT
// Opt-in FSR 4.1.2 Lite Lossy graph inside fsr411::Runtime. It runs the pinned
// release's original PRE / P1-border / BORDER / POST shaders (external,
// translated, IO-guarded data under build/) and replaces only the integer
// CNB, DS, FNB and CT stages with the verified native kernels.
// Geometry: src/fsr412_original_geometry.h, docs/FSR412_ORIGINAL_GEOMETRY.md.
// Proof and limits: docs/FSR412_UPSTREAM_ORACLE.md.
static constexpr unsigned kOriginalRevision = 3; // tools/fsr412/compile_original.py REVISION

static bool original_lite_skip(int tier,const Fsr4InputPrepared &p) {
    // Original PRE predicate on b1 jitter/reset/width_lr/width. PRE writes the
    // same decision as a GPU marker that also gates every neural stage.
    if(tier==4320 || p.reset) return false;
    uint32_t x=0,y=0; std::memcpy(&x,&p.jitter[0],4); std::memcpy(&y,&p.jitter[1],4);
    const bool negativeX=(x>>31) && (x&0x7fffffffu);
    const bool negativeY=(y>>31) && (y&0x7fffffffu)>1042983597u;
    return negativeX && (uint32_t(p.rw<<1u)>=p.alignedUW || !negativeY);
}

static void fill_original_native_dimensions(const Session &s,int pass,const TensorSizes &sizes,PassParams &p) {
    // Input bounds are the class extent (raw halo semantics); stores keep the
    // original rounded P1 rectangles and the active quarter/half rectangles.
    const auto g=fsr412_original_geometry::make_geometry(uint32_t(s.tier),sizes.v[1][0],sizes.v[1][1]);
    auto set=[](Tensor &t,fsr412_original_geometry::Extent e) {t.width=e.width;t.height=e.height;};
    if(pass==1) {set(p.input,g.classHalf);set(p.output,g.CNBwritten);}
    else if(pass==4) {set(p.input,g.classHalf);set(p.output,g.DSwritten);}
    else if(pass==2) {set(p.input,g.classQuarter);set(p.output,g.actualQuarter);}
    else if(pass==3) {set(p.input,g.classQuarter);set(p.output,g.actualHalf);set(p.skip,g.actualHalf);}
    // Run-frame POST CNB precompute: inline original CNB taps are zero beyond
    // the ACTIVE half ROI (docs/FSR412_ORIGINAL_POST_CNB.md), and its
    // missing-first-tap bias uses the same active b1.width/2, height/2.
    else if(pass==13) {set(p.input,g.actualHalf);set(p.output,g.actualHalf);}
}

void original_scratch_copy(id __strong &enc,bool m4,FrameSlot &slot,const std::string &tag,const View &v,
                           size_t from,size_t to,size_t bytes) {
    const uint32_t count=uint32_t(bytes/16u);
    const NSUInteger offset=tag=="originalCacheIn"?kOriginalCopy:kOriginalCopy+16;
    put(slot,offset,count);
    helper(enc,m4,slot,tag,"fsr411_snapshot",{{v.scratch,from},{v.scratch,to},{slot.constants,offset}},{},MTLSizeMake(count,1,1));
}

void dispatch_original_stage(id __strong &enc,bool m4,FrameSlot &slot,const Session &s,const std::string &name,
                             View &v,const TensorSizes &sizes,const Bindings &b,bool skip,const Fsr4InputPrepared &p) {
    const uint32_t w=p.alignedUW,h=p.alignedUH;
    if(name=="pre") {dispatch(enc,m4,slot,s,"pre",MTLSizeMake(ceil_div(w,16),ceil_div(h,16),1),b);return;}
    if(name=="pass1") {
        if(!skip) {dispatch_native(enc,m4,slot,s,1,v,sizes);dispatch_native(enc,m4,slot,s,4,v,sizes);}
        // Original P1 group (0,2): class-perimeter DS clear. Its own marker
        // gate keeps it a no-op on skipped 1080/2160 frames.
        dispatch(enc,m4,slot,s,"pass1_border",MTLSizeMake(1,1,1),b);
        return;
    }
    if(name=="pass2") {
        if(skip) return;
        dispatch_native(enc,m4,slot,s,2,v,sizes);
        // Original 4320 P2 dynamic-invalid threads store one zero uint4 at
        // byte 134217728 (residual cell (1521,20)); 1080/2160 discard it.
        if(fsr412_original_geometry::make_geometry(uint32_t(s.tier),w,h).P2sentinelInAllocationNeedsClear) {
            put(slot,kOriginalSentinel,uint32_t(1));
            helper(enc,m4,slot,"originalSentinel","fsr411_clear_buffer",
                   {{v.scratch,134217728u},{slot.constants,kOriginalSentinel}},{},MTLSizeMake(1,1,1));
        }
        return;
    }
    if(name=="border") {dispatch(enc,m4,slot,s,"border",MTLSizeMake(1,1,1),b);return;}
    if(name=="pass3") {if(!skip) dispatch_native(enc,m4,slot,s,3,v,sizes);return;}
    if(name=="post") {
        // Isolated POST: the saved-CNB cache overlaps POST's live input. Copy
        // the span out before and back after, so this frame reads immutable
        // input and the next frame sees the same logical scratch state.
        if(s.originalCacheBytes)
            original_scratch_copy(enc,m4,slot,"originalCacheIn",v,s.originalCacheOffset,s.originalCacheRelocatedOffset,s.originalCacheBytes);
        // Run frames: every primary block misses and no side block may hit, so
        // the POST reading native-precomputed CNB is byte-identical. Skipped
        // frames keep the original: its mutable-cache misses are
        // schedule-dependent even original-vs-original.
        const bool fast=s.originalPostFast && !skip;
        if(fast) dispatch_native(enc,m4,slot,s,13,v,sizes);
        dispatch(enc,m4,slot,s,fast?"post_fast":"post",MTLSizeMake(ceil_div(w,32),ceil_div(h,32),1),b);
        if(s.originalCacheBytes)
            original_scratch_copy(enc,m4,slot,"originalCacheOut",v,s.originalCacheRelocatedOffset,s.originalCacheOffset,s.originalCacheBytes);
        return;
    }
    [NSException raise:@"FSR412Original" format:@"unknown original stage %s",name.c_str()];
}

Session *ensure_original_session(int tier,const Fsr4InputPrepared &p,const Options &o,Kernels kind,std::string &error) {
    if(kind==Kernels::Reference) {error="original Lite Lossy needs native/i8mpp, scalar or SIMD integer kernels";return nullptr;}
    if(o.profile!=Profile::Int8LiteLossy || o.neuralTileSize || o.pixelFix || o.temporalQuality) {
        error="original Lite Lossy runs the source graph only: no neural tiles, POST tiling, hair fix or temporal quality";return nullptr;
    }
    const auto geometry=fsr412_original_geometry::make_geometry(uint32_t(tier),p.alignedUW,p.alignedUH);
    if(!geometry.valid) {error="original Lite Lossy output size does not fit its tier";return nullptr;}
    const std::string key="fsr412_original_"+std::to_string(tier)+"_di"+std::to_string(int(p.depthReversed))
        +"_lm"+std::to_string(int(p.lowResMV))+"_jm"+std::to_string(int(p.jitteredMV))+"_cs"+std::to_string(o.colorspace)
        +"_ae"+std::to_string(int(p.autoExposure))+"_up"+std::to_string(int(o.ultraPerformance))+"_r"+std::to_string(kOriginalRevision);
    const std::string cacheKey=key+"_"+kernels_label(kind);
    auto existing=sessions_.find(cacheKey);if(existing!=sessions_.end())return existing->second.get();
    const std::string dir=buildDir_+"/fsr412_upstream/compiled/"+key;
    if(o.allowCompile && !run(@[@"python3",ns(root_+"/tools/fsr412/compile_original.py"),@"--root",ns(root_),
        @"--builddir",ns(buildDir_),@"--outdir",ns(dir),@"--tier",ns(std::to_string(tier)),
        @"--ultra-performance",o.ultraPerformance?@"1":@"0",@"--depth-inverted",ns(std::to_string(int(p.depthReversed))),
        @"--low-res-mv",ns(std::to_string(int(p.lowResMV))),@"--jittered-mv",ns(std::to_string(int(p.jitteredMV))),
        @"--colorspace",ns(std::to_string(o.colorspace)),@"--auto-exposure",ns(std::to_string(int(p.autoExposure)))],error)) return nullptr;
    error.clear();
    auto manifest=json(dir+"/bindings.json",error);
    if(!manifest) {error="missing original Lite Lossy bundle "+key+"; build it with tools/fsr412/compile_original.py";return nullptr;}
    NSDictionary *opts=manifest[@"options"];
    const bool cacheTier=tier!=4320;
    if(![manifest[@"schema"] isEqual:@"fsr412_original_runtime_v1"] || [manifest[@"original_revision"] unsignedIntValue]!=kOriginalRevision
       || ![manifest[@"source_dll_sha256"] isEqual:@"d9b0a1f5781c2d633a327272f6f1c8607057df7cafd0233cf38c7395c8f7dbc5"]
       || ![opts isKindOfClass:NSDictionary.class] || [opts[@"tier"] intValue]!=tier
       || [opts[@"ultra_performance"] intValue]!=int(o.ultraPerformance) || [opts[@"depth_inverted"] intValue]!=int(p.depthReversed)
       || [opts[@"low_res_mv"] intValue]!=int(p.lowResMV) || [opts[@"jittered_mv"] intValue]!=int(p.jitteredMV)
       || [opts[@"colorspace"] intValue]!=o.colorspace || [opts[@"auto_exposure"] intValue]!=int(p.autoExposure)
       || ![manifest[@"post_mode"] isEqual:cacheTier?@"isolated":@"literal-no-skip-cache"]
       || [manifest[@"scratch_bytes"] unsignedLongLongValue]!=geometry.allocationBytes) {
        error="original Lite Lossy bundle does not match this stream or revision; rebuild it";return nullptr;
    }
    auto s=std::make_unique<Session>();
    s->key=cacheKey;s->tier=tier;s->modelLite=true;s->originalLossy=true;s->profile=Profile::Int8LiteLossy;
    s->ultraPerformance=o.ultraPerformance;s->externalPostCnb=false;s->postCnbAlias=false;
    s->retrainedWeights=!o.ultraPerformance && tier<=2160;
    s->splitAutoExposure=true;s->preparedColor=false;s->postPixelsPerGroup=32;
    s->order={"pre","pass1","pass2","border","pass3","post"};
    s->originalLogicalScratchBytes=size_t(geometry.allocationBytes);
    s->scratchBytes=size_t([manifest[@"scratch_allocation_bytes"] unsignedLongLongValue]);
    NSDictionary *entries=manifest[@"kernels"];
    if(cacheTier) {
        NSDictionary *post=entries[@"post"];
        s->originalCacheOffset=size_t([post[@"cache_original_byte_offset"] unsignedLongLongValue]);
        s->originalCacheRelocatedOffset=size_t([post[@"cache_relocated_byte_offset"] unsignedLongLongValue]);
        s->originalCacheBytes=size_t([post[@"cache_copy_bytes"] unsignedLongLongValue]);
        const size_t relocated=(s->originalLogicalScratchBytes+255u)&~size_t(255);
        if(NSDictionary *fast=entries[@"post_fast"]) {
            s->originalPostFast=true;
            s->originalPrecomputedOffset=size_t([fast[@"precomputed_base"] unsignedLongLongValue]);
            s->originalPrecomputedStride=size_t([fast[@"precomputed_stride"] unsignedLongLongValue]);
        }
        const size_t expectedBytes=s->originalPostFast
            ?s->originalPrecomputedOffset+s->originalPrecomputedStride*geometry.classHalf.height:relocated+s->originalCacheBytes;
        if(s->originalCacheOffset!=geometry.savedCNBByteOffset || s->originalCacheBytes!=geometry.P ||
           s->originalCacheRelocatedOffset!=relocated || s->scratchBytes!=expectedBytes ||
           (s->originalPostFast && (s->originalPrecomputedOffset!=relocated+s->originalCacheBytes ||
            s->originalPrecomputedStride<size_t(geometry.classHalf.width)*16u || s->originalPrecomputedStride%16u))) {
            error="original Lite Lossy POST cache layout mismatch";return nullptr;
        }
    } else if(s->scratchBytes!=s->originalLogicalScratchBytes) {error="original 4320 scratch layout mismatch";return nullptr;}
    NSString *full=manifest[@"initializers"];
    NSData *t18=[full isKindOfClass:NSString.class]?[NSData dataWithContentsOfFile:ns(dir+"/"+utf8(full))]:nil;
    if(t18.length!=131072 || ![manifest[@"initializers_sha256"] isEqual:ns(sha256_bytes(t18.bytes,t18.length))]) {
        error="original Lite t18 initializer identity mismatch";return nullptr;
    }
    // Native integer weights: upstream trained data for normal <=2160, else
    // the unchanged base slice (bytes 0..27391 of the original t18).
    NSData *native=s->retrainedWeights?[NSData dataWithContentsOfFile:ns(root_+"/build/fsr412_lossy/initializers-lite-lossy.bin")]
                                      :[t18 subdataWithRange:NSMakeRange(0,27392)];
    const std::string nativeHash=native.length==27392?sha256_bytes(native.bytes,native.length):"";
    if(nativeHash.empty() || ![manifest[@"native_initializer_sha256"] isEqual:ns(nativeHash)] ||
       (s->retrainedWeights && nativeHash!="c77c2d1a4d119ed76d94b8c23335490cc38280db3346c72396a1e1acb0523cfc")) {
        error="original Lite Lossy native initializer mismatch; run make lite-lossy-weights";return nullptr;
    }
    s->originalInitializer=[device_ newBufferWithBytes:t18.bytes length:t18.length options:MTLResourceStorageModeShared];
    s->initializer=[device_ newBufferWithBytes:native.bytes length:native.length options:MTLResourceStorageModeShared];
    if(!s->originalInitializer || !s->initializer) {error="cannot allocate original Lite initializers";return nullptr;}
    for(const char *name:{"pre","pass1","pass1_border","pass2","border","pass3","post","post_fast","ae","ae_finalize","rcas"}) {
        NSDictionary *entry=entries[@(name)];
        const std::string stage=name;
        if(stage=="post_fast" && !s->originalPostFast) continue;
        if(![entry isKindOfClass:NSDictionary.class]) {error=std::string("incomplete original Lite graph: ")+name;return nullptr;}
        if(stage=="pass1" || stage=="pass2" || stage=="pass3") continue; // replaced by native integer kernels
        const std::string library=dir+"/"+utf8(entry[@"library"]);
        NSData *bytes=[NSData dataWithContentsOfFile:ns(library)];
        if(!bytes.length || ![entry[@"library_sha256"] isEqual:ns(sha256_bytes(bytes.bytes,bytes.length))]) {
            error="original Lite library identity mismatch: "+stage;return nullptr;
        }
        Kernel k;NSError *e=nil;
        auto lib=[device_ newLibraryWithURL:[NSURL fileURLWithPath:ns(library)] error:&e];
        auto fn=lib?[lib newFunctionWithName:entry[@"entry"]]:nil;
        NSArray *wg=entry[@"wgsize"];if(wg.count!=3) {error="missing original threadgroup size: "+stage;return nullptr;}
        k.threads=MTLSizeMake([wg[0] unsignedIntValue],[wg[1] unsignedIntValue],[wg[2] unsignedIntValue]);
        MTLComputePipelineDescriptor *descriptor=[MTLComputePipelineDescriptor new];
        descriptor.computeFunction=fn;
        descriptor.maxTotalThreadsPerThreadgroup=k.threads.width*k.threads.height*k.threads.depth;
        descriptor.threadGroupSizeIsMultipleOfThreadExecutionWidth=descriptor.maxTotalThreadsPerThreadgroup%32u==0;
        k.pso=fn?[device_ newComputePipelineStateWithDescriptor:descriptor options:MTLPipelineOptionNone reflection:nil error:&e]:nil;
        if(!k.pso) {error="original Lite pipeline "+stage+": "+metal_error(e);return nullptr;}
        if(NSString *tables=entry[@"literal_tables"]) {
            NSData *table=[NSData dataWithContentsOfFile:ns(dir+"/"+utf8(tables))];
            if(!table.length) {error="missing original Lite shader table";return nullptr;}
            k.literalTables=[device_ newBufferWithBytes:table.bytes length:table.length options:MTLResourceStorageModeShared];
        }
        for(NSArray *a in entry[@"buffers"])k.buffers.emplace_back([a[0] unsignedIntegerValue],utf8(a[1]));
        for(NSArray *a in entry[@"textures"])k.textures.emplace_back([a[0] unsignedIntegerValue],utf8(a[1]));
        for(NSArray *a in entry[@"samplers"])k.samplers.emplace_back([a[0] unsignedIntegerValue],utf8(a[1]));
        s->kernels.emplace(stage,std::move(k));
    }
    if(s->originalPostFast) {
        // Load the validated native POST CNB (pass 13) like the ordinary Lite
        // path, then retarget it: input = original plane C, output = the
        // compact precompute region after the isolated cache plane.
        s->externalPostCnb=true;
        if(!post_cnb_scratch_layout(uint32_t(tier),false,s->scratchLayout)) {error="original POST CNB layout";return nullptr;}
    }
    const size_t scratchBytes=s->scratchBytes;
    if(!load_native(*s,o.allowCompile,kind,error)) return nullptr;
    s->scratchBytes=scratchBytes;s->externalPostCnb=false;s->scratchLayout={};
    for(int n:{1,2,3,4}) if(!s->native.count(n)) {error="incomplete original Lite native graph";return nullptr;}
    if(s->originalPostFast) {
        auto found=s->native.find(13);
        if(found==s->native.end() || !(found->second.parameters.q[0].flags & kFirstTapBias) ||
           found->second.parameters.biases[3]==kNoOffset) {error="original POST CNB needs conditional first-tap biases";return nullptr;}
        auto &q=found->second.parameters;
        q.input.offset=uint32_t(2*geometry.P+geometry.halfRowBytes+16u);q.input.strideX=16u;q.input.strideY=uint32_t(geometry.halfRowBytes);
        q.input.channels=16u;q.input.bankStride=0u;
        q.output.offset=uint32_t(s->originalPrecomputedOffset);q.output.strideX=16u;q.output.strideY=uint32_t(s->originalPrecomputedStride);
        q.output.channels=16u;q.output.bankStride=0u;
    }
    auto result=s.get();sessions_.emplace(cacheKey,std::move(s));return result;
}
