#!/usr/bin/env python3
"""Build "FSR4 Installer.app" (and optionally a DMG) for a release.

The app bundles the same FSR 4.1 files as the CrossOver release zip
(tools/crossover/package_release.py) as Resources/payload.tar.gz. On install
it unpacks them to ~/Library/Application Support/fsr4-native-macos and runs
tools/crossover/crossover_fsr4.py there.

  python3 tools/installer/build_app.py v1.3 [--rev HEAD] [--out dist/release] [--dmg]
"""

import argparse
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "tools" / "crossover"))
import package_release  # noqa: E402

APP_NAME = "FSR4 Installer"
BUNDLE_ID = "org.fsr4mac.installer"
MIN_MACOS = "26.0"
SOURCES = sorted((HERE / "Sources").glob("*.swift"))


def run(cmd, **kw):
    print("+", " ".join(str(c) for c in cmd))
    subprocess.check_call([str(c) for c in cmd], **kw)


def build_payload(rev: str, dest: Path) -> int:
    """Files of `rev` that the CrossOver install needs, as a tar.gz without a
    top-level folder. Returns the file count."""
    tar = subprocess.check_output(["git", "archive", "--format=tar", rev], cwd=ROOT)
    count = 0
    with tarfile.open(fileobj=io.BytesIO(tar)) as src, tarfile.open(dest, "w:gz", compresslevel=9) as dst:
        for member in src.getmembers():
            if not member.isfile() or not package_release.included(member.name):
                continue
            info = tarfile.TarInfo(member.name)
            info.size = member.size
            info.mode = member.mode
            info.mtime = 1767225600  # 2026-01-01: reproducible archive
            dst.addfile(info, src.extractfile(member))
            count += 1
        guide = (ROOT / "tools" / "crossover" / "INSTALL.md").read_bytes()
        info = tarfile.TarInfo("INSTALL.md")
        info.size, info.mode, info.mtime = len(guide), 0o644, 1767225600
        dst.addfile(info, io.BytesIO(guide))
        for path,data,mode in package_release.model_payload():
            info=tarfile.TarInfo(path)
            info.size,info.mode,info.mtime=len(data),mode,1767225600
            dst.addfile(info,io.BytesIO(data))
            count+=1
    return count + 1


def build_icon(resources: Path, scratch: Path):
    """Same icon as the in-game FSR 4 panel (tools/crossover/panel_app.mm)."""
    sdk = subprocess.check_output(["xcrun", "--sdk", "macosx", "--show-sdk-path"], text=True).strip()
    panel = scratch / "panel"
    run(["clang++", "-O2", "-fobjc-arc", "-std=c++17", "-arch", "arm64", f"-mmacosx-version-min={MIN_MACOS}",
         "-isysroot", sdk, ROOT / "tools/crossover/panel_app.mm",
         "-framework", "Foundation", "-framework", "AppKit", "-framework", "QuartzCore", "-o", panel])
    iconset = scratch / "AppIcon.iconset"
    iconset.mkdir()
    run([panel, "--make-icon", iconset])
    run(["iconutil", "-c", "icns", iconset, "-o", resources / "AppIcon.icns"])


def info_plist(version: str) -> str:
    short = version.lstrip("v")
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>CFBundleName</key><string>{APP_NAME}</string>
  <key>CFBundleDisplayName</key><string>{APP_NAME}</string>
  <key>CFBundleExecutable</key><string>{APP_NAME}</string>
  <key>CFBundleIdentifier</key><string>{BUNDLE_ID}</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleShortVersionString</key><string>{short}</string>
  <key>CFBundleVersion</key><string>{short}</string>
  <key>CFBundleIconFile</key><string>AppIcon</string>
  <key>CFBundleDevelopmentRegion</key><string>en</string>
  <key>CFBundleLocalizations</key><array><string>en</string><string>ru</string></array>
  <key>LSApplicationCategoryType</key><string>public.app-category.utilities</string>
  <key>LSMinimumSystemVersion</key><string>{MIN_MACOS}</string>
  <key>LSArchitecturePriority</key><array><string>arm64</string></array>
  <key>NSHighResolutionCapable</key><true/>
  <key>NSHumanReadableCopyright</key><string>MIT License. AMD FSR is a trademark of AMD.</string>
</dict></plist>
"""


def build_app(version: str, rev: str, out: Path) -> Path:
    app = out / f"{APP_NAME}.app"
    if app.exists():
        shutil.rmtree(app)
    macos = app / "Contents" / "MacOS"
    resources = app / "Contents" / "Resources"
    macos.mkdir(parents=True)
    resources.mkdir(parents=True)
    run(["xcrun", "swiftc", "-O", "-parse-as-library", "-swift-version", "6",
         "-target", f"arm64-apple-macos{MIN_MACOS}", *SOURCES, "-o", macos / APP_NAME])
    (app / "Contents" / "Info.plist").write_text(info_plist(version))
    for lproj in sorted(HERE.glob("Resources/*.lproj")):
        shutil.copytree(lproj, resources / lproj.name)
    with tempfile.TemporaryDirectory() as scratch:
        build_icon(resources, Path(scratch))
    commit = subprocess.check_output(["git", "rev-parse", "--short", rev], cwd=ROOT, text=True).strip()
    count = build_payload(rev, resources / "payload.tar.gz")
    (resources / "payload.json").write_text(json.dumps({"version": version, "commit": commit}) + "\n")
    print(f"payload: {count} files from {commit}, "
          f"{(resources / 'payload.tar.gz').stat().st_size / 1e6:.1f} MB")
    # Ad-hoc signature: required to run on Apple Silicon. Not notarized.
    run(["codesign", "--force", "--sign", "-", "--timestamp=none", app])
    run(["codesign", "--verify", "--strict", app])
    return app


def build_dmg(app: Path, version: str, out: Path) -> Path:
    dmg = out / f"FSR4-Installer-{version}.dmg"
    if dmg.exists():
        dmg.unlink()
    with tempfile.TemporaryDirectory() as scratch:
        stage = Path(scratch) / "FSR 4 for CrossOver"
        stage.mkdir()
        run(["ditto", app, stage / app.name])
        os.symlink("/Applications", stage / "Applications")
        run(["hdiutil", "create", "-volname", "FSR 4 for CrossOver", "-srcfolder", stage,
             "-fs", "HFS+", "-format", "ULFO", "-ov", dmg])
    print(f"{dmg}: {dmg.stat().st_size / 1e6:.1f} MB")
    return dmg


def main():
    p = argparse.ArgumentParser()
    p.add_argument("version")
    p.add_argument("--rev", default="HEAD")
    p.add_argument("--out", default=str(ROOT / "dist" / "release"))
    p.add_argument("--dmg", action="store_true", help="also build FSR4-Installer-<version>.dmg")
    args = p.parse_args()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    app = build_app(args.version, args.rev, out)
    print(app)
    if args.dmg:
        build_dmg(app, args.version, out)


if __name__ == "__main__":
    main()
