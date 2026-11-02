// SPDX-License-Identifier: MIT
// CPU-only conservative Original opt-in fusion contract. No Metal dispatch.
#include "../src/fsr412_mpp_policy.h"
#include "../tools/bench/fsr412_mpp_oracle.h"
#include <cassert>
#include <cstdio>

int main() {
    using namespace fsr412::mpp_tuning;
    PassParams c{},d{};
    c.weights[0]=0;c.weights[1]=2304;c.weights[2]=2816;
    c.biases[0]=3328;c.biases[1]=3392;c.biases[2]=3520;
    c.q[0]={0,0,7,0};c.q[1]={0,0,9,1};c.q[2]=c.q[3]={0,5,6,0};
    d.weights[0]=3584;d.biases[0]=5632;d.q[4]={0,0,6,0};d.q[5]={0,0,7,0};
    auto p=fsr412_tuning_cpu::fused_shape(c,d,1920,1080);
    FusionRequest r;r.originalModelRoute=true;r.explicitFusionOptIn=true;r.verifiedNormalLiteLossy=true;
    r.apple10Metal4=true;r.probedFeatures=1;r.scratchBytes=fsr412_tuning_cpu::tensor_end(p.cnb.input)+512;
    r.weightsBytes=8192;r.parameterBytes=1280;r.cnbParameterOffset=0;r.dsParameterOffset=256;
    unsigned checks=0;
    auto expect=[&](FusionRequest request,PassParams a,PassParams b,Reject reason) {
        const auto plan=fusion_plan(request,a,b);assert(plan.reject==reason);assert(plan.enabled==(reason==Reject::None));++checks;
    };
    expect(r,p.cnb,p.ds,Reject::None);
    const auto plan=fusion_plan(r,p.cnb,p.ds);
    assert(plan.groupsX==60 && plan.groupsY==540 && plan.threads==32 && plan.tileInputX==32 && plan.tileInputY==2);
    assert(plan.cnbParameterBinding==2 && plan.dsParameterBinding==3 && plan.parameterBytesEach==212);
    assert(plan.cnbParameterOffset==0 && plan.dsParameterOffset==256);
    auto req=r;req.originalModelRoute=false;expect(req,p.cnb,p.ds,Reject::NotOriginalOptIn);
    req=r;req.explicitFusionOptIn=false;expect(req,p.cnb,p.ds,Reject::NotOriginalOptIn);
    req=r;req.verifiedNormalLiteLossy=false;expect(req,p.cnb,p.ds,Reject::UnverifiedModel);
    req=r;req.apple10Metal4=false;expect(req,p.cnb,p.ds,Reject::UnsupportedDevice);
    for(u32 mask:{0u,2u,4u,~0u}){req=r;req.probedFeatures=mask;expect(req,p.cnb,p.ds,Reject::UnsupportedProbe);}
    req=r;req.probedFeatures=3;expect(req,p.cnb,p.ds,Reject::None);
    for(auto wh:{std::pair<unsigned,unsigned>{960,540},{856,556},{1919,1080},{1920,1079},{1921,1080},{3840,2160}}) {
        auto s=fsr412_tuning_cpu::fused_shape(c,d,wh.first,wh.second);
        expect(r,s.cnb,s.ds,Reject::UnmeasuredActiveSize);
    }
    auto a=p.cnb;auto b=p.ds;b.input.offset+=16;expect(r,a,b,Reject::WrongTensorGeometry);
    a=p.cnb;a.input.offset+=1;expect(r,a,p.ds,Reject::WrongTensorGeometry);
    a=p.cnb;a.input.strideY=16;expect(r,a,p.ds,Reject::WrongTensorGeometry);
    b=p.ds;b.output.bankStride=16;expect(r,p.cnb,b,Reject::WrongTensorGeometry);
    req=r;req.scratchBytes=16;expect(req,p.cnb,p.ds,Reject::WrongTensorGeometry);
    a=p.cnb;a.input=p.cnb.output;expect(r,a,p.ds,Reject::AliasedScratch);
    b=p.ds;b.output.offset=p.cnb.output.offset;expect(r,p.cnb,b,Reject::AliasedScratch);
    for(u32 offset:{u32(fsr411::kNoOffset),1u,8192u}){a=p.cnb;a.weights[0]=offset;expect(r,a,p.ds,Reject::InvalidWeightRegions);}
    req=r;req.weightsBytes=64;expect(req,p.cnb,p.ds,Reject::InvalidWeightRegions);
    for(u32 i=0;i<4;++i)for(u32 field=0;field<4;++field) {
        a=p.cnb;auto &q=a.q[i];if(field==0)++q.accumulatorShift;else if(field==1)++q.residualShift;else if(field==2)++q.rightShift;else q.flags^=4;
        expect(r,a,p.ds,Reject::ChangedQuantizers);
    }
    for(u32 i=4;i<6;++i){b=p.ds;b.q[i].flags=4;expect(r,p.cnb,b,Reject::ChangedQuantizers);}
    for(std::uint64_t offset:{std::uint64_t(1),std::uint64_t(128),std::uint64_t(1280),std::uint64_t(-1)}) {
        req=r;req.dsParameterOffset=offset;expect(req,p.cnb,p.ds,Reject::InvalidParameterBindings);
    }
    req=r;req.dsParameterOffset=0;expect(req,p.cnb,p.ds,Reject::InvalidParameterBindings);
    req=r;req.parameterBytes=400;expect(req,p.cnb,p.ds,Reject::InvalidParameterBindings);
    const auto values=fusion_specialization(p.cnb,p.ds);
    const auto ca=fsr411::specialize_metadata(p.cnb,false),da=fsr411::specialize_metadata(p.ds,false);
    for(u32 i=0;i<12;++i)for(u32 j=0;j<4;++j){assert(values.values[i][j]==ca.values[i][j]);assert(values.values[i+12][j]==da.values[i][j]);checks+=2;}
    assert(fusion_constant_ranges[0].first==0 && fusion_constant_ranges[0].count==12);
    assert(fusion_constant_ranges[1].first==12 && fusion_constant_ranges[1].count==6);
    assert(fusion_constant_ranges[2].first==22 && fusion_constant_ranges[2].count==2);
    const auto probe=fusion_probe();assert(probe.threads==32 && probe.outputBytes==4 && !probe.scratchBytes && !probe.weightsBytes);
    std::printf("FSR412 MPP Original opt-in fusion CPU policy: %u checks pass\n",checks);
}
