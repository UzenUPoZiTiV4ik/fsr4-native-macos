// SPDX-License-Identifier: MIT
// Exact IO test of the original P1 border-only branch and its marker gate.
#include "fsr412_upstream_replay.h"
#include <iostream>
using namespace fsr412_original;
int main(int argc,char **argv) {@autoreleasepool {try {
    Replay r(argc>1?argv[1]:"build/fsr412_upstream",640,384,384,224,"isolated","u8");r.load_pass1_border();
    auto words=static_cast<uint32_t *>(r.scratch.contents);const uint32_t fill=0x5a5a5a5a;std::fill(words,words+r.logicalScratchBytes/4,fill);
    std::vector<size_t> borders;const unsigned w=960,h=540;const size_t base=2*r.planeBytes,row=r.quarterRowBytes;
    for(unsigned bank=0;bank<2;++bank) {
        for(unsigned x=0;x<w+2;++x)for(unsigned y:{0u,h+1})for(unsigned c=0;c<4;++c)borders.push_back((base+bank*r.quarterPlaneBytes+size_t(y)*row+x*16)/4+c);
        for(unsigned y=1;y<=h;++y)for(unsigned x:{0u,w+1})for(unsigned c=0;c<4;++c)borders.push_back((base+bank*r.quarterPlaneBytes+size_t(y)*row+x*16)/4+c);
    }
    r.dispatch("pass1_border");size_t zeros=0;for(size_t i=0;i<r.logicalScratchBytes/4;++i){check(words[i]==0 || words[i]==fill,"border helper modified unexpected value");zeros+=words[i]==0;}
    check(zeros==borders.size(),"border helper changed unexpected word count");for(size_t i:borders)check(words[i]==0,"missed original class border word");
    for(size_t i:borders)words[i]=fill;words[33273792/4]=1511506142;words[33273792/4+1]=168889943;
    std::string before=hash(r.scratch.contents,r.logicalScratchBytes);r.dispatch("pass1_border");check(hash(r.scratch.contents,r.logicalScratchBytes)==before,"original P1 skip gate was not preserved");
    std::cout<<"pass1_border original IO exact: "<<borders.size()<<" uint words cleared on run; skip preserves all logical scratch\n";return 0;
} catch(const std::exception &e){std::cerr<<e.what()<<"\n";return 1;}}}
