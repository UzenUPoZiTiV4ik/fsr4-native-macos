// SPDX-License-Identifier: MIT
// Run each mode in a fresh process because environment defaults load once.
#include "fsr4_live.h"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>

static void check(bool condition,const char *message) {if(!condition) throw std::runtime_error(message);}
static int profile() {int value=-1;fsr4_live_get_profile(&value);return value;}
static void sharpness(float expected,int enabled) {
    float value=-1;int rcas=-1;fsr4_live_get_sharpness(&value,&rcas);
    check(std::fabs(value-expected)<.0001f && rcas==enabled,"sharpness/RCAS preference changed unexpectedly");
}
static void write(const std::string &path,const char *data) {
    FILE *file=std::fopen(path.c_str(),"w");check(file,"cannot create settings fixture");
    bool ok=std::fputs(data,file)>=0;if(std::fclose(file))ok=false;check(ok,"cannot save settings fixture");
}
static void poll() {std::this_thread::sleep_for(std::chrono::milliseconds(550));}

int main(int argc,char **argv) {try {
    if(argc!=2) return 2;
    setenv("FSR4_SETTINGS","none",1);unsetenv("FSR4_PROFILE");
    if(!std::strcmp(argv[1],"default")) {
        check(profile()==1,"FSR 4.1.1 default is not Lossy");
        fsr4_live_set_profile(0);check(profile()==0,"in-memory Exact switch failed");
    } else if(!std::strcmp(argv[1],"environment")) {
        setenv("FSR4_PROFILE","exact",1);check(profile()==0,"environment Exact preference was ignored");
    } else if(!std::strcmp(argv[1],"environment-fast")) {
        setenv("FSR4_PROFILE","exact-fast",1);check(profile()==2,"environment Exact Fast preference was ignored");
    } else if(!std::strcmp(argv[1],"model")) {
        // The FSR 4.0 mode routes to the old model unless the bottle pins a version.
        unsetenv("FSR4_VERSION");check(!std::strcmp(fsr4_live_model_version(),"4.1.1"),"default model is not 4.1.1");
        fsr4_live_set_profile(3);check(profile()==3,"FSR 4.0 mode switch failed");
        check(!std::strcmp(fsr4_live_model_version(),"4.0.2"),"FSR 4.0 mode did not select 4.0.2");
        setenv("FSR4_VERSION","4.1.1",1);check(!std::strcmp(fsr4_live_model_version(),"4.0.2"),"4.1.1 install ignored the FSR 4.0 mode");
        fsr4_live_set_profile(2);check(!std::strcmp(fsr4_live_model_version(),"4.1.1"),"Exact Fast did not return to 4.1.1");
        setenv("FSR4_VERSION","4.0.2",1);check(!std::strcmp(fsr4_live_model_version(),"4.0.2"),"FSR4_VERSION=4.0.2 was not honoured");
    } else if(!std::strcmp(argv[1],"environment-fsr4")) {
        setenv("FSR4_PROFILE","fsr4",1);check(profile()==3,"environment FSR 4.0 preference was ignored");
    } else if(!std::strcmp(argv[1],"file")) {
        char temporary[]="/tmp/fsr4-settings-XXXXXX";
        check(mkdtemp(temporary),"cannot create fixture directory");
        const std::string path=std::string(temporary)+"/panel.conf";
        write(path,"sharpness=0.200\nrcas=1\nprofile=exact\n");
        setenv("FSR4_SETTINGS",path.c_str(),1);setenv("FSR4_PROFILE","lossy",1);
        check(profile()==0,"saved profile did not override environment default");sharpness(.2f,1);
        fsr4_live_set_profile(1);check(profile()==1,"Lossy switch failed");sharpness(.2f,1);
        fsr4_live_set_sharpness(.6f,0);check(profile()==1,"sharpness setter discarded Lossy preference");sharpness(.6f,0);
        fsr4_live_set_profile(77);check(profile()==1,"invalid profile was accepted");
        fsr4_live_set_profile(2);check(profile()==2,"Exact Fast switch failed");sharpness(.6f,0);
        fsr4_live_set_profile(1);
        struct stat state={};check(!stat(path.c_str(),&state),"cannot inspect fixture");
        timespec times[2]={state.st_atimespec,{state.st_mtime,1000000}};
        write(path,"sharpness=0.300\nrcas=1\nprofile=exact\n");
        check(!utimensat(AT_FDCWD,path.c_str(),times,0),"cannot set fixture timestamp");poll();
        check(profile()==0,"getter did not reload panel profile");sharpness(.3f,1);
        // Same mtime second, different nanoseconds: rapid UI changes must
        // still be seen by a later frame even after the first file was read.
        write(path,"sharpness=0.300\nrcas=1\nprofile=lossy\n");times[1].tv_nsec=2000000;
        check(!utimensat(AT_FDCWD,path.c_str(),times,0),"cannot update fixture timestamp");poll();
        check(profile()==1,"same-second profile change was missed");sharpness(.3f,1);
        write(path,"sharpness=0.300\nrcas=1\nprofile=exact-fast\n");times[1].tv_nsec=3000000;
        check(!utimensat(AT_FDCWD,path.c_str(),times,0),"cannot update fixture timestamp");poll();
        check(profile()==2,"saved Exact Fast profile was not loaded");
        write(path,"sharpness=0.300\nrcas=1\nprofile=fsr4\n");times[1].tv_nsec=4000000;
        check(!utimensat(AT_FDCWD,path.c_str(),times,0),"cannot update fixture timestamp");poll();
        check(profile()==3,"saved FSR 4.0 mode was not loaded");
        fsr4_live_set_profile(1);
        FILE *saved=std::fopen(path.c_str(),"r");check(saved,"cannot reopen fixture");
        char text[128]={};std::fread(text,1,sizeof text-1,saved);std::fclose(saved);
        check(std::strstr(text,"profile=lossy")!=nullptr,"profile was not saved by name");
        unlink(path.c_str());rmdir(temporary);
    } else return 2;
    std::printf("settings %s passed\n",argv[1]);return 0;
} catch(const std::exception &error) {
    std::fprintf(stderr,"settings test: %s\n",error.what());return 1;
}}
