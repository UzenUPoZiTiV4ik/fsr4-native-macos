// SPDX-License-Identifier: MIT
// Independent scalar integer oracle for the standalone FSR 4.1.2 DS16 tests.
// Intentionally does not call fsr411::finish or any GPU/reference kernel.
#pragma once
#include "../../src/fsr411_model.h"
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace fsr412_tuning_cpu {
inline int32_t signed32(uint32_t x) {
    return x <= 0x7fffffffu ? int32_t(x) : -1 - int32_t(0xffffffffu - x);
}
inline int8_t signed8(uint8_t x) { return x < 128 ? int8_t(x) : int8_t(int(x) - 256); }
inline int32_t read32(const std::vector<uint8_t> &data, size_t offset) {
    if (offset > data.size() || data.size() - offset < 4) throw std::runtime_error("oracle int32 out of range");
    const uint32_t x = uint32_t(data[offset]) | (uint32_t(data[offset+1]) << 8)
        | (uint32_t(data[offset+2]) << 16) | (uint32_t(data[offset+3]) << 24);
    return signed32(x);
}
inline int8_t quantize(int32_t dot, int32_t bias, int32_t residual, fsr411::Epilogue q) {
    if (!fsr411::valid_epilogue(q)) throw std::runtime_error("oracle invalid quantizer");
    uint32_t bits = (uint32_t(dot) + uint32_t(bias)) << q.accumulatorShift;
    bits += uint32_t(residual) << q.residualShift;
    int32_t value = signed32(bits);
    if ((q.flags & 1u) && value < 0) value = 0;
    if (q.rightShift) {
        // Floor division expressed in int64, not the shared helper's shift.
        const int64_t divisor = int64_t(1) << q.rightShift;
        const int64_t floor = value >= 0 ? value / divisor : -((-int64_t(value) + divisor - 1) / divisor);
        const uint32_t increment = (q.flags & 4u) ? uint32_t(divisor / 2)
            : uint32_t(divisor / 2 - 1 + (floor & 1));
        value = signed32(uint32_t(value) + increment);
        value = int32_t(value >= 0 ? int64_t(value) / divisor : -((-int64_t(value) + divisor - 1) / divisor));
    }
    if (q.flags & 2u) {
        const uint32_t low = uint32_t(value) & 65535u;
        value = low < 32768u ? int32_t(low) : int32_t(low) - 65536;
    }
    return int8_t(std::max(-128, std::min(127, int(value))));
}
inline void source_epilogues(fsr411::PassParams cnb,fsr411::PassParams ds,bool ultra) {
    // Pinned Lite source quantizers, independent of the imported manifest.
    // Both normal trained Lite Lossy tiers use the normal nearest-even set.
    const fsr411::Epilogue cq[4]={{0,0,ultra?8u:7u,0},{0,0,ultra?8u:9u,1},{0,5,6,0},{0,5,6,0}};
    const fsr411::Epilogue dq[2]={{0,0,ultra?7u:6u,0},{0,0,ultra?8u:7u,0}};
    if(std::memcmp(cnb.q,cq,sizeof(cq)) || std::memcmp(ds.q+4,dq,sizeof(dq)))
        throw std::runtime_error("p1/DS quantizers differ from pinned nearest-even Lite source");
}
inline size_t tensor_end(fsr411::Tensor t) {
    return t.channels && t.width && t.height
        ? size_t(fsr411::tensor_address(t,t.width-1,t.height-1,t.channels-1)) + 1 : 0;
}
inline fsr411::PassParams shape(fsr411::PassParams p, unsigned w, unsigned h, bool banked=true) {
    // Poisoned one-pixel padding is intentional: shaders must synthesize OOB
    // zeroes, not read allocator border bytes. Output remains disjoint.
    const unsigned ow=(w+1)/2, oh=(h+1)/2;
    p.input = {256u+(w+2)*16u+16u,16u,(w+2)*16u,w,h,16u,0u};
    const unsigned end=unsigned(tensor_end(p.input)+512u);
    const unsigned pitch=(ow+2)*(banked?16u:32u);
    const unsigned bank=pitch*(oh+2);
    p.output={((end+255u)&~255u)+pitch+(banked?16u:32u),banked?16u:32u,pitch,ow,oh,32u,banked?bank:0u};
    return p;
}
inline std::vector<uint8_t> input(fsr411::PassParams p, unsigned pattern) {
    std::vector<uint8_t> scratch(tensor_end(p.output)+512u,0xa7);
    uint32_t seed=0x12345678u;
    for(unsigned y=0;y<p.input.height;++y)for(unsigned x=0;x<p.input.width;++x)for(unsigned c=0;c<16;++c) {
        seed^=seed<<13;seed^=seed>>17;seed^=seed<<5;
        const uint8_t value=pattern==0?uint8_t(seed):pattern==1?uint8_t(((x+y+c)&1u)?127:128)
            :pattern==2?128u:pattern==3?127u:0u;
        scratch.at(fsr411::tensor_address(p.input,x,y,c))=value;
    }
    return scratch;
}
inline std::vector<uint8_t> ds16(fsr411::PassParams p, const std::vector<uint8_t> &scratch,
    const std::vector<uint8_t> &weights, const std::vector<uint8_t> &initializer, bool original) {
    if (initializer.size()!=27392 || p.input.channels!=16 || p.output.channels!=32)
        throw std::runtime_error("oracle expects complete Lite initializer and DS16");
    std::vector<uint8_t> out(size_t(p.output.width)*p.output.height*32u);
    for(unsigned y=0;y<p.output.height;++y)for(unsigned x=0;x<p.output.width;++x)for(unsigned oc=0;oc<32;++oc) {
        int64_t sum=0;
        // Raw AMD coefficients are output-major within each tap; native MPP
        // coefficients are input-major. Separate indexing checks the packing.
        for(unsigned tap=0;tap<4;++tap)for(unsigned ic=0;ic<16;++ic) {
            const unsigned ix=x*2+(tap&1),iy=y*2+(tap>>1);
            if(ix>=p.input.width || iy>=p.input.height)continue;
            const int v=signed8(scratch.at(fsr411::tensor_address(p.input,ix,iy,ic)));
            const size_t wi=original?4864u+tap*512u+oc*16u+ic:p.weights[0]+(tap*16u+ic)*32u+oc;
            const int w=signed8((original?initializer:weights).at(wi));
            sum+=v*w;
        }
        const int32_t bias=original?read32(initializer,6912u+oc*4u)
            :p.biases[0]==fsr411::kNoOffset?0:read32(weights,p.biases[0]+oc*4u);
        out[(size_t(y)*p.output.width+x)*32u+oc]=uint8_t(quantize(signed32(uint32_t(sum)),bias,0,p.q[oc<16?4:5]));
    }
    return out;
}
struct CnbDsParams { fsr411::PassParams cnb,ds; };
inline bool valid_fused_params(CnbDsParams p) {
    if(p.cnb.input.channels!=16 || p.cnb.output.channels!=16 || p.ds.input.channels!=16 || p.ds.output.channels!=32
        || std::memcmp(&p.cnb.output,&p.ds.input,sizeof(fsr411::Tensor))
        || p.ds.output.width>(p.cnb.output.width+1)/2 || p.ds.output.height>(p.cnb.output.height+1)/2)return false;
    const auto overlap=[](fsr411::Tensor a,fsr411::Tensor b) {
        const size_t ae=tensor_end(a),be=tensor_end(b);
        return ae && be && size_t(a.offset)<be && size_t(b.offset)<ae;
    };
    return !overlap(p.cnb.input,p.cnb.output) && !overlap(p.cnb.input,p.ds.output) && !overlap(p.cnb.output,p.ds.output);
}
inline CnbDsParams fused_shape(fsr411::PassParams cnb, fsr411::PassParams ds,
    unsigned w, unsigned h, bool banked=true, bool floorOutput=false) {
    ds=shape(ds,w,h,banked);
    if(floorOutput) {ds.output.width=w/2;ds.output.height=h/2;}
    cnb.output=ds.input;cnb.input=ds.input;
    cnb.input.offset=unsigned((std::max(tensor_end(ds.input),tensor_end(ds.output))+511u)&~255u);
    return {cnb,ds};
}
inline std::vector<uint8_t> fused_input(CnbDsParams p,unsigned pattern) {
    auto a=p.cnb;a.output=p.cnb.input;auto scratch=input(a,pattern);
    scratch.resize(std::max(scratch.size(),tensor_end(p.ds.output)+512u),0xa7);
    return scratch;
}
inline std::vector<uint8_t> cnb(fsr411::PassParams p,const std::vector<uint8_t> &scratch,
    const std::vector<uint8_t> &weights,const std::vector<uint8_t> &initializer,bool original) {
    if(initializer.size()!=27392 || p.input.channels!=16 || p.output.channels!=16 || (p.q[0].flags&8u))
        throw std::runtime_error("oracle expects p1 CNB16 (not conditional POST)");
    std::vector<uint8_t> out(size_t(p.output.width)*p.output.height*16u);
    for(unsigned y=0;y<p.output.height;++y)for(unsigned x=0;x<p.output.width;++x) {
        int8_t a[16],b[32];
        for(unsigned oc=0;oc<16;++oc) {
            int64_t sum=0;
            for(unsigned tap=0;tap<9;++tap)for(unsigned ic=0;ic<16;++ic) {
                const int ix=int(x)+int(tap%3)-1,iy=int(y)+int(tap/3)-1;
                if(ix<0 || iy<0 || unsigned(ix)>=p.input.width || unsigned(iy)>=p.input.height)continue;
                const int v=signed8(scratch.at(fsr411::tensor_address(p.input,unsigned(ix),unsigned(iy),ic)));
                const size_t wi=original?1152u+tap*256u+oc*16u+ic:p.weights[0]+(tap*16u+ic)*16u+oc;
                sum+=v*int(signed8((original?initializer:weights).at(wi)));
            }
            const int32_t bias=original?read32(initializer,3456u+oc*4u)
                :p.biases[0]==fsr411::kNoOffset?0:read32(weights,p.biases[0]+oc*4u);
            a[oc]=quantize(signed32(uint32_t(sum)),bias,0,p.q[0]);
        }
        for(unsigned oc=0;oc<32;++oc) {
            int64_t sum=0;
            for(unsigned ic=0;ic<16;++ic) {
                const size_t wi=original?3584u+oc*16u+ic:p.weights[1]+ic*32u+oc;
                sum+=int(a[ic])*int(signed8((original?initializer:weights).at(wi)));
            }
            const int32_t bias=original?read32(initializer,4096u+oc*4u)
                :p.biases[1]==fsr411::kNoOffset?0:read32(weights,p.biases[1]+oc*4u);
            b[oc]=quantize(signed32(uint32_t(sum)),bias,0,p.q[1]);
        }
        for(unsigned oc=0;oc<16;++oc) {
            int64_t sum=0;
            for(unsigned ic=0;ic<32;++ic) {
                const size_t wi=original?4224u+oc*32u+ic:p.weights[2]+ic*16u+oc;
                sum+=int(b[ic])*int(signed8((original?initializer:weights).at(wi)));
            }
            const int32_t bias=original?read32(initializer,4736u+oc*4u)
                :p.biases[2]==fsr411::kNoOffset?0:read32(weights,p.biases[2]+oc*4u);
            const int residual=x<p.input.width && y<p.input.height
                ?signed8(scratch.at(fsr411::tensor_address(p.input,x,y,oc))):0;
            out[(size_t(y)*p.output.width+x)*16u+oc]=uint8_t(quantize(signed32(uint32_t(sum)),bias,residual,p.q[oc<8?2:3]));
        }
    }
    return out;
}
inline void scatter(std::vector<uint8_t> &scratch,fsr411::Tensor t,const std::vector<uint8_t> &values) {
    if(values.size()!=size_t(t.width)*t.height*t.channels)throw std::runtime_error("oracle scatter size mismatch");
    for(unsigned y=0;y<t.height;++y)for(unsigned x=0;x<t.width;++x)for(unsigned c=0;c<t.channels;++c)
        scratch.at(fsr411::tensor_address(t,x,y,c))=values[(size_t(y)*t.width+x)*t.channels+c];
}
inline std::vector<uint8_t> fused_oracle(CnbDsParams p,const std::vector<uint8_t> &scratch,
    const std::vector<uint8_t> &weights,const std::vector<uint8_t> &initializer,bool original) {
    if(!valid_fused_params(p))throw std::runtime_error("invalid/aliased CNB+DS fusion geometry");
    auto expected=scratch;
    scatter(expected,p.cnb.output,cnb(p.cnb,scratch,weights,initializer,original));
    scatter(expected,p.ds.output,ds16(p.ds,expected,weights,initializer,original));
    return expected;
}
inline size_t fused_cpu_parity(fsr411::PassParams cnb,fsr411::PassParams ds,
    const std::vector<uint8_t> &weights,const std::vector<uint8_t> &initializer) {
    size_t cases=0;
    for(auto wh : {std::pair<unsigned,unsigned>{1,1},{3,5},{17,9},{65,3},{129,5},{130,4}})
        for(bool banked : {true,false})for(bool floorOutput : {true,false})for(unsigned pattern=0;pattern<5;++pattern) {
            const auto p=fused_shape(cnb,ds,wh.first,wh.second,banked,floorOutput);
            const auto data=fused_input(p,pattern);
            if(fused_oracle(p,data,weights,initializer,true)!=fused_oracle(p,data,weights,initializer,false))
                throw std::runtime_error("p1 CNB+DS raw initializer/packed CPU mismatch");
            ++cases;
        }
    return cases;
}
inline size_t cpu_parity(fsr411::PassParams p, const std::vector<uint8_t> &weights,
    const std::vector<uint8_t> &initializer) {
    size_t cases=0;
    for(auto wh : {std::pair<unsigned,unsigned>{1,1},{3,5},{17,9},{129,3},{130,4}})
        for(bool banked : {true,false})for(unsigned pattern=0;pattern<5;++pattern) {
            auto a=shape(p,wh.first,wh.second,banked);auto data=input(a,pattern);
            if(ds16(a,data,weights,initializer,true)!=ds16(a,data,weights,initializer,false))
                throw std::runtime_error("raw initializer / packed CPU oracle mismatch");
            ++cases;
        }
    return cases;
}
}
