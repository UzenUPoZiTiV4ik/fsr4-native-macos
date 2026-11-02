// SPDX-License-Identifier: MIT
#include "../src/fsr412_neural_tile.h"
#include <cstdio>
#include <cstdlib>
#include <initializer_list>

static void check(bool value,const char *what) {
    if(!value) {std::fprintf(stderr,"Lite neural tile: %s\n",what);std::exit(1);}
}
int main() {
    using namespace fsr412_tiling;
    for(unsigned tile:{128u,256u,512u,1024u,2048u}) {
        check(valid_size(tile),"advertised tile rejected");
        for(unsigned phase=0;phase<2;++phase) {
            Control c{tile,phase,32u,2u,32u,1u,3840u,2160u};
            for(unsigned y=0;y<c.outputHeight;y+=2u)
                for(unsigned x=0;x<c.outputWidth;x+=64u) {
                    bool needed=false;
                    for(unsigned yy=y;yy<y+2u && yy<c.outputHeight;++yy)
                        for(unsigned xx=x;xx<x+64u && xx<c.outputWidth;++xx)
                            needed |= fresh_pixel(tile,phase,c.outputWidth,c.outputHeight,xx,yy);
                    check(!needed || active(c,x/64u,y/2u),"fresh pixel producer omitted");
                }
            c.tileSize=0;
            check(active(c,1,300),"disabled mask changed full graph");
        }
    }
    check(!valid_size(0) && !valid_size(127) && !valid_size(768) && !valid_size(4096),"invalid tile accepted");
    Control c{512,0,0,2,32,1,2048,1024};
    check(active(c,2,100),"selected tile omitted");
    check(!active(c,10,100),"inactive interior group was selected");
    check(active(c,10,0),"fresh border omitted");
    c.halo=32;
    check(active(c,8,100),"convolution halo omitted");
    c.tileSize=768;
    check(active(c,10,100),"malformed control must run full graph");
    std::puts("Lite neural tile CPU: all tile sizes, masks, borders and halos passed");
}
