// SPDX-License-Identifier: MIT
// Native AppKit interaction/layout check; no game, hook or Windows DLL.
#define main fsr4_panel_app_main
#include "../../tools/crossover/panel_app.mm"
#undef main
#include <stdexcept>

static void require(bool ok, const char *message) { if (!ok) throw std::runtime_error(message); }
int main() { @autoreleasepool { try {
    char temporary[] = "/tmp/fsr4-panel-ui-XXXXXX";
    require(mkdtemp(temporary), "cannot create temporary root"); g_root=temporary;
    NSError *error=nil;
    require([NSFileManager.defaultManager createDirectoryAtPath:[NSString stringWithUTF8String:dist_path("").c_str()]
        withIntermediateDirectories:YES attributes:nil error:&error], "cannot create fixture directory");
    [NSApplication sharedApplication];
    PanelApp *panel=[PanelApp new]; [panel build]; [panel.timer invalidate]; [panel.panel orderOut:nil];
    require(panel.modeSlider.numberOfTickMarks==2 && panel.modeSlider.allowsTickMarkValuesOnly && !panel.modeSlider.continuous,
            "mode slider is not discrete or applies while dragging");
    require(panel.modeSlider.minValue==0 && panel.modeSlider.maxValue==1, "mode slider bounds wrong");
    require([panel.modeExplanation.stringValue isEqual:kModeExplanation], "mode explanation hidden/changed");
    require(!NSIntersectsRect(panel.modeExplanation.frame,panel.slider.frame), "explanation overlaps sharpness");
    require(!NSIntersectsRect(panel.captureButton.frame,panel.captureMessage.frame), "capture controls overlap");
    require(panel.panel.contentView.bounds.size.height>=578, "new panel content is clipped");
    require([panel.newLook.title isEqual:@"Experimental new-look"] && !panel.newLook.enabled
            && panel.newLook.state==NSControlStateValueOff, "new-look button/default wrong");
    require(!NSIntersectsRect(panel.newLook.frame,panel.profileHint.frame)
            && !NSIntersectsRect(panel.newLook.frame,panel.rcas.frame), "new-look controls overlap");
    panel.modeSlider.doubleValue=0; [panel changed:panel.modeSlider];
    float sharp,strength; bool rcas,fix; int profile;
    read_settings(sharp,rcas,profile,fix,strength); require(profile==2, "Quality endpoint did not select Exact");
    panel.modeSlider.doubleValue=1; panel.pixelFix.state=NSControlStateValueOn; [panel changed:panel.modeSlider];
    read_settings(sharp,rcas,profile,fix,strength); require(profile==6 && fix, "Lite stop lost the remembered fix");
    panel.modeSlider.doubleValue=0; [panel changed:panel.modeSlider];
    read_settings(sharp,rcas,profile,fix,strength); require(profile==2 && fix && panel.pixelFix.hidden, "Exact switch lost/used Lite fix");
    require(write_settings(.235f,false,8,false,.7f), "cannot save retired fixture"); [panel refresh];
    panel.slider.doubleValue=.3; [panel changed:panel.slider];
    read_settings(sharp,rcas,profile,fix,strength);
    require(profile==8 && fabs(sharp-.3)<.001 && !rcas && fabs(strength-.7)<.001, "unrelated edit replaced retired mode/settings");
    panel.modeSlider.doubleValue=0; [panel changed:panel.modeSlider];
    read_settings(sharp,rcas,profile,fix,strength); require(profile==2 && !rcas, "explicit recovery from tiled failed");
    // Simulate an available optional Lite Lossy stop, then click the real control.
    g_has_lite_lossy=true; panel.modeSlider.maxValue=2; panel.modeSlider.numberOfTickMarks=3;
    require(write_settings(.3f,false,7,false,.7f,0), "cannot save new-look fixture"); [panel refresh];
    require(panel.newLook.enabled && panel.newLook.state==NSControlStateValueOff, "new-look not available for Lite Lossy");
    [panel.newLook performClick:nil];
    require(read_new_look() && [panel.modeValue.stringValue containsString:@"new-look"], "new-look click did not apply");
    read_settings(sharp,rcas,profile,fix,strength);
    require(profile==7 && !rcas && fabs(strength-.7)<.001, "new-look click changed model/RCAS/fix strength");
    panel.modeSlider.doubleValue=0; [panel changed:panel.modeSlider];
    require(!panel.newLook.enabled && read_new_look(), "Exact used/discarded new-look preference");
    panel.slider.doubleValue=.4; [panel changed:panel.slider];
    require(read_new_look(), "sharpness discarded new-look preference");
    panel.modeSlider.doubleValue=2; [panel changed:panel.modeSlider];
    require(panel.newLook.enabled && read_new_look(), "Lite Lossy did not restore new-look preference");
    if(const char *path=getenv("FSR4_PANEL_TEST_SNAPSHOT")) {
        const char *source=getenv("FSR4_PANEL_TEST_SOURCE");
        require(source!=nullptr,"snapshot needs the optional model fixture source");
        NSString *base=[NSString stringWithUTF8String:(g_root+"/build/fsr412_lossy").c_str()];
        require([NSFileManager.defaultManager createDirectoryAtPath:base withIntermediateDirectories:YES
            attributes:nil error:&error],"cannot create snapshot model fixture");
        for(const char *name:{"initializers-lite-lossy.bin","import.json"}) {
            NSString *from=[NSString stringWithUTF8String:(std::string(source)+"/build/fsr412_lossy/"+name).c_str()];
            NSString *to=[base stringByAppendingPathComponent:[NSString stringWithUTF8String:name]];
            require([NSFileManager.defaultManager copyItemAtPath:from toPath:to error:&error],"cannot copy snapshot model fixture");
        }
        [panel.panel close];[panel build];[panel.timer invalidate];[panel.panel orderOut:nil];
        require(panel.newLook.enabled && panel.modeSlider.numberOfTickMarks==3,"snapshot optional modes are wrong");
        [panel snapshot:[NSString stringWithUTF8String:path]];
    }
    [panel.newLook performClick:nil];
    require(!read_new_look() && ![panel.modeValue.stringValue containsString:@"new-look"], "new-look opt-out did not restore old mode");
    for (NSView *view in panel.panel.contentView.subviews)
        require(NSContainsRect(panel.panel.contentView.bounds,view.frame), "panel control clipped");
    [panel.panel close];
    require([NSFileManager.defaultManager removeItemAtPath:[NSString stringWithUTF8String:g_root.c_str()] error:&error], "fixture cleanup failed");
    puts("panel AppKit interactions/layout passed"); return 0;
} catch(const std::exception &e) { fprintf(stderr,"panel AppKit: %s\n",e.what()); return 1; } } }
