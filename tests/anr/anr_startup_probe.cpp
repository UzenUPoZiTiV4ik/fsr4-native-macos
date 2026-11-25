// SPDX-License-Identifier: MIT
// dlopen an ANR hook with ANR already selected, before any frame/device exists.
#include "../../src/anr/anr_live.h"
#include <chrono>
#include <cstdio>
#include <cstring>
#include <dlfcn.h>
#include <thread>

int main(int argc,char **argv){
    if(argc!=3)return 2;
    const int expectedMode=!std::strcmp(argv[2],"anr3x")?2:!std::strcmp(argv[2],"anr")?1:0;
    if(!expectedMode&&std::strcmp(argv[2],"metalfx"))return 2;
    const int expectedSelected=expectedMode!=0?1:0;
    const int expectedMult=expectedMode==2?3:(expectedMode==1?2:1);
    void *handle=dlopen(argv[1],RTLD_NOW|RTLD_LOCAL);
    if(!handle){std::fprintf(stderr,"dlopen failed: %s\n",dlerror());return 1;}
    auto selected=reinterpret_cast<int(*)()>(dlsym(handle,"fsr4_anr_is_selected"));
    auto selectedMode=reinterpret_cast<int(*)()>(dlsym(handle,"fsr4_anr_selected_mode"));
    auto multiplier=reinterpret_cast<int(*)()>(dlsym(handle,"fsr4_anr_multiplier"));
    auto refresh=reinterpret_cast<void(*)()>(dlsym(handle,"fsr4_anr_refresh_settings"));
    auto getStatus=reinterpret_cast<void(*)(Fsr4AnrStatus*)>(dlsym(handle,"fsr4_anr_get_status"));
    if(!selected||!selectedMode||!multiplier||!refresh||!getStatus){std::fprintf(stderr,"missing ANR startup symbols\n");return 1;}
    for(int i=0;i<2;++i){
        refresh();Fsr4AnrStatus status={};getStatus(&status);
        if(selected()!=expectedSelected||selectedMode()!=expectedMode||multiplier()!=expectedMult
           ||status.selectedAnr!=expectedSelected||status.active||status.encodes||status.generated||status.width||status.height){
            std::fprintf(stderr,"startup selection/status mismatch; selected=%d mode=%d mult=%d status=%d encodes=%llu\n",
                selected(),selectedMode(),multiplier(),status.selectedAnr,(unsigned long long)status.encodes);return 1;
        }
        // Also exercise the background timer while no interpolator exists.
        if(!i)std::this_thread::sleep_for(std::chrono::milliseconds(650));
    }
    // Objective-C hook classes/the controller remain loaded until process exit.
    std::printf("ANR dlopen startup passed: %s, no model/device/frame\n",argv[2]);return 0;
}
