# SPDX-License-Identifier: MIT
"""CPU-only dynamic hook startup regression; no model, Wine bottle or GPU dispatch.

Deliberately link the FG constructor BEFORE anr_live.mm, matching build_hook.
Standalone ANR inference tests link the sources the other way and previously
missed the cross-translation-unit container initialization crash.
"""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]


@unittest.skipUnless(sys.platform=="darwin" and shutil.which("xcrun"),"needs the macOS MetalFX SDK")
class AnrStartupTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory(prefix="fsr4-anr-startup-")
        cls.addClassCleanup(cls.temp.cleanup)
        cls.build=Path(cls.temp.name)
        sdk=subprocess.check_output(["xcrun","--sdk","macosx","--show-sdk-path"],text=True).strip()
        common=["xcrun","clang++","-std=c++17","-O2","-arch","x86_64","-mmacosx-version-min=26.0",
                "-isysroot",sdk,"-I",str(ROOT/"src")]
        cls.hook=cls.build/"libanr-startup.dylib"
        # Do not reorder to put the C++ state TU before the constructor TU.
        commands=[common+["-dynamiclib","-fobjc-arc",str(ROOT/"hook/anr_metalfx.mm"),
                           str(ROOT/"src/anr_live.mm"),str(ROOT/"src/anr_runtime.mm"),
                           "-framework","Foundation","-framework","Metal","-framework","MetalFX","-o",str(cls.hook)],
                  common+[str(ROOT/"tests/anr_startup_probe.cpp"),"-o",str(cls.build/"probe")]]
        for command in commands:
            result=subprocess.run(command,capture_output=True,text=True,timeout=180)
            if result.returncode:raise AssertionError(result.stdout+result.stderr)

    def startup(self,environment_mode,file_mode=None,status_io=False):
        with tempfile.TemporaryDirectory(prefix="fsr4-anr-root-") as temporary:
            root=Path(temporary);dist=root/"dist/crossover";dist.mkdir(parents=True)
            settings=dist/"fg-settings.conf"
            if file_mode is not None:
                settings.write_text("schema=fsr4.fg.settings.v1\nmode="+file_mode+"\n")
            env={k:v for k,v in os.environ.items() if not k.startswith("FSR4_")}
            env.update(FSR4_ROOT=str(root),FSR4_FG=environment_mode,
                       FSR4_FG_SETTINGS=str(settings) if file_mode is not None else "none",
                       FSR4_ANR_STATUS_DISABLED="0" if status_io else "1")
            expected=file_mode if file_mode is not None else environment_mode
            result=subprocess.run([str(self.build/"probe"),str(self.hook),expected],env=env,
                                  capture_output=True,text=True,timeout=15)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            self.assertIn("ANR dlopen startup passed",result.stdout)
            if status_io:
                values=dict(line.split("=",1) for line in (dist/"fg-status.conf").read_text().splitlines())
                self.assertEqual(values["selected_mode"],expected)
                self.assertEqual(values["encodes"],"0")
                self.assertEqual(values["anr_configured"],"0")
            else:
                self.assertFalse((dist/"fg-status.conf").exists())

    def test_anr_selected_at_dlopen_before_any_frame(self):
        self.startup("anr")

    def test_metalfx_selected_at_dlopen(self):
        self.startup("metalfx")

    def test_file_anr_overrides_metalfx_before_dlopen(self):
        self.startup("metalfx","anr",status_io=True)

    def test_file_metalfx_overrides_anr_before_dlopen(self):
        self.startup("anr","metalfx",status_io=True)

    def test_anr_startup_without_proprietary_assets_can_publish(self):
        self.startup("anr",status_io=True)


if __name__=="__main__":unittest.main()
