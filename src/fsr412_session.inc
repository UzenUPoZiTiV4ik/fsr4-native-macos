// SPDX-License-Identifier: MIT
// Model-specific loader within fsr411::Runtime. Resource lifetime, input
// conversion and Metal 3/4 synchronization use the common runtime.
Session *ensure_lite_session(int tier,const Fsr4InputPrepared &p,const Options &o,Kernels kind,std::string &error) {
    if(o.profile!=Profile::Int8Lite) {error="FSR 4.1.2 requires the int8-lite profile";return nullptr;}
    const bool portable=kind!=Kernels::Reference;
    const std::string key="fsr412_"+std::to_string(tier)+"_di"+std::to_string(p.depthReversed)
        +"_lm"+std::to_string(p.lowResMV)+"_jm"+std::to_string(p.jitteredMV)+"_cs"+std::to_string(o.colorspace)
        +"_ae"+std::to_string(p.autoExposure)+"_o"+std::to_string(o.optimize)
        +"_up"+std::to_string(o.ultraPerformance)+"_pc"+std::to_string(portable)
        +(o.optimize && portable && o.colorspace==0?"_v2":"")+(o.pixelFix?"_px2":"");
    const std::string cacheKey=key+"_"+kernels_label(kind);
    auto existing=sessions_.find(cacheKey);if(existing!=sessions_.end())return existing->second.get();
    if(o.allowCompile && !run(@[@"python3",ns(root_+"/tools/fsr412/compile_fsr412.py"),@"--root",ns(root_),
        @"--builddir",ns(buildDir_),@"--outdir",ns(buildDir_+"/"+key),@"--res",ns(std::to_string(tier)),
        @"--depth-inverted",ns(std::to_string(p.depthReversed)),@"--low-res-mv",ns(std::to_string(p.lowResMV)),
        @"--jittered-mv",ns(std::to_string(p.jitteredMV)),@"--colorspace",ns(std::to_string(o.colorspace)),
        @"--auto-exposure",ns(std::to_string(p.autoExposure)),@"--optimize",ns(std::to_string(o.optimize)),
        @"--ultra-performance",o.ultraPerformance?@"1":@"0",@"--external-post-cnb",portable?@"1":@"0",
        @"--pixelization-fix",o.pixelFix?@"1":@"0"],error))return nullptr;
    auto manifest=json(buildDir_+"/"+key+"/bindings.json",error);
    const std::vector<std::string> order={"pre","pad0","pass1","pad1","pad1b","pass2","pad2","pass3","pad3","post"};
    if(!manifest || ![manifest[@"version"] isEqual:@"4.1.2"] || [manifest[@"order"] count]!=order.size()) {
        if(error.empty())error="missing FSR 4.1.2 INT8 Lite graph; import the supplied DLL and compile it";return nullptr;
    }
    for(size_t i=0;i<order.size();++i) if(utf8(manifest[@"order"][i])!=order[i]) {error="FSR 4.1.2 graph order mismatch";return nullptr;}
    if([manifest[@"external_post_cnb"] boolValue]!=portable) {error="FSR 4.1.2 POST cache ABI mismatch";return nullptr;}
    if([manifest[@"pixelization_fix"] boolValue]!=o.pixelFix) {error="FSR 4.1.2 pixelization-fix variant mismatch";return nullptr;}
    auto s=std::make_unique<Session>();s->key=cacheKey;s->tier=tier;s->modelLite=true;
    s->profile=Profile::Int8Lite;s->ultraPerformance=o.ultraPerformance;s->externalPostCnb=portable;
    s->splitAutoExposure=true;s->order=order;
    s->preparedColor=[manifest[@"prepared_color"] boolValue];
    if(s->preparedColor!=(o.optimize && portable && o.colorspace==0)) {error="FSR 4.1.2 input color cache ABI mismatch";return nullptr;}
    if(s->preparedColor && !manifest[@"kernels"][@"prepare_color"]) {error="missing FSR 4.1.2 prepared color stage";return nullptr;}
    s->postPixelsPerGroup=manifest[@"post_pixels_per_group"]?[manifest[@"post_pixels_per_group"] unsignedIntValue]:32;
    if(s->postPixelsPerGroup!=16 && s->postPixelsPerGroup!=32) {error="invalid FSR 4.1.2 POST dispatch";return nullptr;}
    const size_t expected=tier==1080?25027392u:tier==2160?99886528u:398707392u;
    s->scratchBytes=[manifest[@"scratch_bytes"] unsignedLongLongValue];
    if(s->scratchBytes!=expected) {error="FSR 4.1.2 scratch ABI mismatch";return nullptr;}
    const uint32_t w=tier*8u/9u,h=tier/2u;
    // POST reads the final tensor in plane C. Plane A is dead after pass 3,
    // so its beginning can hold the compact, borderless CNB result.
    s->scratchLayout={expected,uint64_t(w+2u)*(h+2u)*16u,0,uint64_t(w)*h*16u,expected,w,h,w*16u};
    s->postCnbAlias=portable;
    NSString *initializer=manifest[@"initializer"];
    if(![initializer isKindOfClass:NSString.class] || !initializer.isAbsolutePath) {error="missing FSR 4.1.2 initializer provenance";return nullptr;}
    NSData *init=[NSData dataWithContentsOfFile:initializer];
    if(init.length!=27392) {error="FSR 4.1.2 Lite initializer must be 27392 bytes";return nullptr;}
    s->initializer=[device_ newBufferWithBytes:init.bytes length:init.length options:MTLResourceStorageModeShared];
    if(!s->initializer) {error="cannot allocate FSR 4.1.2 initializer";return nullptr;}
    if(portable) {
        if(!load_native(*s,o.allowCompile,kind,error))return nullptr;
        s->nativeSequence={{"pass1",{1,4}},{"pass2",{2}},{"pass3",{3}}};
        for(int n:{1,2,3,4,13})if(!s->native.count(n)) {error="incomplete FSR 4.1.2 native graph";return nullptr;}
    }
    NSDictionary *entries=manifest[@"kernels"];
    for(NSString *name in entries) {
        NSDictionary *entry=entries[name];Kernel k;const std::string stage=utf8(name);
        k.tensorSlot=entry[@"tensor_slot"]?[entry[@"tensor_slot"] intValue]:-1;
        if([entry[@"tile"] count]==2) {k.tileX=[entry[@"tile"][0] unsignedIntValue];k.tileY=[entry[@"tile"][1] unsignedIntValue];}
        if((stage.rfind("pad",0)==0 || stage.rfind("pass",0)==0) && (k.tensorSlot<0 || k.tensorSlot>=9 || !k.tileX || !k.tileY)) {
            error="invalid FSR 4.1.2 dispatch geometry: "+stage;return nullptr;
        }
        if(portable && s->nativeSequence.count(stage)) {s->kernels.emplace(stage,std::move(k));continue;}
        NSError *e=nil;auto lib=[device_ newLibraryWithURL:[NSURL fileURLWithPath:ns(buildDir_+"/"+key+"/"+utf8(entry[@"library"]))] error:&e];
        auto fn=lib?[lib newFunctionWithName:entry[@"entry"]]:nil;
        NSArray *wg=entry[@"wgsize"];if(wg.count!=3) {error="missing FSR 4.1.2 threadgroup size";return nullptr;}
        k.threads=MTLSizeMake([wg[0] unsignedIntValue],[wg[1] unsignedIntValue],[wg[2] unsignedIntValue]);
        MTLComputePipelineDescriptor *descriptor=[MTLComputePipelineDescriptor new];
        descriptor.computeFunction=fn;
        descriptor.maxTotalThreadsPerThreadgroup=k.threads.width*k.threads.height*k.threads.depth;
        descriptor.threadGroupSizeIsMultipleOfThreadExecutionWidth=descriptor.maxTotalThreadsPerThreadgroup%32u==0;
        k.pso=fn?[device_ newComputePipelineStateWithDescriptor:descriptor options:MTLPipelineOptionNone reflection:nil error:&e]:nil;
        if(!k.pso) {error="FSR 4.1.2 pipeline "+stage+": "+metal_error(e);return nullptr;}
        if(NSString *tables=entry[@"literal_tables"]) {
            NSData *bytes=[NSData dataWithContentsOfFile:ns(buildDir_+"/"+key+"/"+utf8(tables))];
            if(!bytes.length) {error="missing FSR 4.1.2 shader table";return nullptr;}
            k.literalTables=[device_ newBufferWithBytes:bytes.bytes length:bytes.length options:MTLResourceStorageModeShared];
        }
        for(NSArray *a in entry[@"buffers"])k.buffers.emplace_back([a[0] unsignedIntegerValue],utf8(a[1]));
        for(NSArray *a in entry[@"textures"])k.textures.emplace_back([a[0] unsignedIntegerValue],utf8(a[1]));
        for(NSArray *a in entry[@"samplers"])k.samplers.emplace_back([a[0] unsignedIntegerValue],utf8(a[1]));
        s->kernels.emplace(stage,std::move(k));
    }
    for(const auto &name:order)if(!s->kernels.count(name)) {error="incomplete FSR 4.1.2 graph: "+name;return nullptr;}
    for(const char *name:{"ae","ae_finalize","rcas"})if(!s->kernels.count(name)) {error="missing FSR 4.1.2 exposure/sharpening stage";return nullptr;}
    auto result=s.get();sessions_.emplace(cacheKey,std::move(s));return result;
}
