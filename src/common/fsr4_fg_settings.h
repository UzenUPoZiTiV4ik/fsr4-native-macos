// SPDX-License-Identifier: MIT
#pragma once
#ifdef __cplusplus
#include <cstdio>
#include <cstring>
#include <string>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

// Separate from panel.conf: FG changes must never reset the SR profile/RCAS.
namespace fsr4_fg {
enum class Mode { MetalFX=0, ANR=1 };
inline const char *name(Mode mode){return mode==Mode::ANR?"anr":"metalfx";}
inline bool parse_mode(const std::string &value,Mode &mode){
    if(value=="metalfx"){mode=Mode::MetalFX;return true;}
    if(value=="anr"){mode=Mode::ANR;return true;}
    return false;
}
inline std::string trim(std::string value){
    const auto begin=value.find_first_not_of(" \t\r\n");
    if(begin==std::string::npos)return {};
    return value.substr(begin,value.find_last_not_of(" \t\r\n")-begin+1);
}
struct Settings {bool exists=false,valid=false;Mode mode=Mode::MetalFX;};
inline Settings parse_settings(const std::string &text){
    Settings result;result.exists=true;
    bool found=false;size_t begin=0;
    while(begin<text.size()){
        size_t end=text.find('\n',begin);if(end==std::string::npos)end=text.size();
        auto line=trim(text.substr(begin,end-begin));begin=end+1;
        if(line.empty()||line[0]=='#')continue;
        const auto eq=line.find('=');if(eq==std::string::npos)return result;
        const auto key=trim(line.substr(0,eq)),value=trim(line.substr(eq+1));
        if(key=="mode"){
            if(found||!parse_mode(value,result.mode))return result;
            found=true;
        }else if(key!="schema"||value!="fsr4.fg.settings.v1")return result;
    }
    result.valid=found;return result;
}
inline Settings read_settings(const std::string &path){
    Settings result;
    int fd=open(path.c_str(),O_RDONLY|O_CLOEXEC|O_NOFOLLOW);
    if(fd<0)return result;
    result.exists=true;struct stat st={};
    if(fstat(fd,&st)||!S_ISREG(st.st_mode)||st.st_size<=0||st.st_size>4096){close(fd);return result;}
    std::string text(size_t(st.st_size),'\0');size_t done=0;
    while(done<text.size()){
        ssize_t n=read(fd,text.data()+done,text.size()-done);
        if(n<=0){close(fd);return result;}done+=size_t(n);
    }
    close(fd);
    if(text.find('\0')!=std::string::npos)return result;
    return parse_settings(text);
}
inline bool write_settings(const std::string &path,Mode mode){
    if(mode!=Mode::MetalFX&&mode!=Mode::ANR)return false;
    const std::string temporary=path+".tmp."+std::to_string(getpid());
    int fd=open(temporary.c_str(),O_WRONLY|O_CREAT|O_EXCL|O_CLOEXEC|O_NOFOLLOW,0600);
    if(fd<0)return false;
    FILE *file=fdopen(fd,"w");if(!file){close(fd);unlink(temporary.c_str());return false;}
    bool ok=std::fprintf(file,"schema=fsr4.fg.settings.v1\nmode=%s\n",name(mode))>0;
    if(fclose(file))ok=false;
    if(!ok||rename(temporary.c_str(),path.c_str())){unlink(temporary.c_str());return false;}
    return true;
}
inline std::string settings_path(const char *root,const char *overridePath){
    if(overridePath&&*overridePath)return !std::strcmp(overridePath,"none")?"":overridePath;
    return root&&*root?std::string(root)+"/dist/crossover/fg-settings.conf":"";
}
}
extern "C" {
#endif
// Mode queries only read an atomic. Slow file polling/method switching runs on
// the dedicated controller queue, never inside native MetalFX's frame encode.
int fsr4_anr_is_selected(void);
void fsr4_anr_refresh_settings(void);
void fsr4_anr_mode_changed(int anr);
#ifdef __cplusplus
}
#endif
