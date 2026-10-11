#!/usr/bin/env python3
"""Point one CrossOver bottle at a copied D3DMetal that loads libfsr4hook.

The CrossOver.app bundle is not modified. libd3dshared reads
D3DMETAL_FRAMEWORK_PATH and dlopens that path (RTLD_NOW). Restore deletes
only that key from the bottle config.
"""

import argparse
import json
import os
import shutil
import struct
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DIST = ROOT / "dist" / "crossover"
HOOK_SRC = ROOT / "hook" / "fsr4_metalfx.m"
LIVE_SRC = ROOT / "src" / "fsr4_live.mm"
LIVE411_SRC = ROOT / "src" / "fsr411_live.mm"
PANEL_SRC = ROOT / "src" / "fsr4_panel.mm"          # hook side: status publish + app launch
PANEL_APP_SRC = ROOT / "tools" / "crossover" / "panel_app.mm"
PANEL_APP = DIST / "FSR4 Panel.app"
SMOKE_SRC = ROOT / "tools" / "crossover" / "smoke.m"
LIVE_TEST_SRC = ROOT / "tools" / "crossover" / "live_test.mm"
DLOPEN_SRC = ROOT / "tools" / "crossover" / "dlopen_test.c"
FRAMEWORK_IN_APP = "Contents/SharedSupport/CrossOver/lib/apple_gptk/external/D3DMetal.framework"
CROSSOVER_APPS = (Path("/Applications/CrossOver.app"), Path("/Applications/CrossOver Preview.app"),
                  Path.home() / "Applications/CrossOver.app", Path.home() / "Applications/CrossOver Preview.app")
BOTTLES = Path.home() / "Library/Application Support/CrossOver/Bottles"
DEFAULT_BOTTLE = BOTTLES / "Игры" / "cxbottle.conf"
FSR402_WEIGHTS = ROOT / "third_party/fsr4/internal/shaders/fsr4_model_v07_i8_quality/initializers.bin"
LOAD_PATH = b"@loader_path/libfsr4hook.dylib"
LC_LOAD_DYLIB = 0xC
KEY = "D3DMETAL_FRAMEWORK_PATH"
LOG_KEY = "FSR4_HOOK_LOG"
MARKER = ";; fsr4-mac"
ROOT_KEY = "FSR4_ROOT"
REPLACE_KEY = "FSR4_REPLACE"
PANEL_KEY = "FSR4_PANEL"
VERSION_KEY = "FSR4_VERSION"
PROFILE_KEY = "FSR4_PROFILE"
METALFX_KEY = "D3DM_ENABLE_METALFX"
# "fsr4" is the panel's fourth mode: the FSR 4.0.2 model under a 4.1.1 install.
PROFILES = ("exact", "exact-fast", "lossy", "fsr4")
OURS = (KEY, LOG_KEY, ROOT_KEY, REPLACE_KEY, PANEL_KEY, VERSION_KEY, PROFILE_KEY)


def run(cmd, **kw):
    print("+", " ".join(cmd))
    subprocess.check_call(cmd, **kw)


def sdk():
    return subprocess.check_output(["xcrun", "--sdk", "macosx", "--show-sdk-path"], text=True).strip()


def build_hook(log_path: Path) -> Path:
    DIST.mkdir(parents=True, exist_ok=True)
    out = DIST / "libfsr4hook.dylib"
    common = ["-O2", "-fobjc-arc", "-arch", "x86_64", "-mmacosx-version-min=14.0",
              "-isysroot", sdk(), "-I", str(ROOT / "src")]
    run(["clang", "-c", *common, str(HOOK_SRC), "-o", str(DIST / "hook.o")])
    run(["clang++", "-c", "-std=c++17", *common, str(LIVE_SRC), "-o", str(DIST / "live.o")])
    run(["clang++", "-c", "-std=c++17", *common, str(LIVE411_SRC), "-o", str(DIST / "live411.o")])
    run(["clang++", "-c", "-std=c++17", *common, str(PANEL_SRC), "-o", str(DIST / "panel.o")])
    run([
        "clang++", "-dynamiclib", *common,
        str(DIST / "hook.o"), str(DIST / "live.o"), str(DIST / "live411.o"), str(DIST / "panel.o"),
        "-framework", "Foundation", "-framework", "Metal", "-framework", "MetalFX",
        # Darwin ld needs an explicit allowance as well as weak_import when
        # linking an encoder without the optional 4.1.1 symbol. The public
        # version router rejects an unavailable requested model explicitly.
        "-Wl,-U,_fsr411_encode_live_frame",
        "-Wl,-U,_fsr411_invalidate_owner", "-Wl,-U,_fsr411_release_owner",
        "-install_name", "@loader_path/libfsr4hook.dylib",
        "-o", str(out),
    ])
    run(["codesign", "-f", "-s", "-", str(out)])
    return out


def build_smoke() -> Path:
    out = DIST / "smoke"
    run([
        "clang", "-O2", "-fobjc-arc", "-arch", "x86_64",
        "-mmacosx-version-min=14.0", "-isysroot", sdk(),
        str(SMOKE_SRC), str(DIST / "libfsr4hook.dylib"),
        "-framework", "Foundation", "-framework", "Metal", "-framework", "MetalFX",
        "-Wl,-rpath", str(DIST),
        "-o", str(out),
    ])
    return out


def build_live() -> Path:
    out = DIST / "live_test"
    run([
        "clang++", "-O2", "-fobjc-arc", "-std=c++17", "-arch", "x86_64",
        "-mmacosx-version-min=14.0", "-isysroot", sdk(),
        "-I", str(ROOT / "src"),
        str(LIVE_TEST_SRC), str(DIST / "libfsr4hook.dylib"),
        "-framework", "Foundation", "-framework", "Metal", "-framework", "MetalFX",
        "-Wl,-rpath", str(DIST),
        "-o", str(out),
    ])
    return out


def build_panel_app() -> Path:
    """dist/crossover/FSR4 Panel.app: native (arm64) Dock app showing the hook's
    status and setting the RCAS sharpness (tools/crossover/panel_app.mm)"""
    macos = PANEL_APP / "Contents" / "MacOS"
    res = PANEL_APP / "Contents" / "Resources"
    macos.mkdir(parents=True, exist_ok=True)
    res.mkdir(parents=True, exist_ok=True)
    exe = macos / "FSR4 Panel"
    run([
        "clang++", "-O2", "-fobjc-arc", "-std=c++17", "-arch", "arm64",
        "-mmacosx-version-min=14.0", "-isysroot", sdk(),
        str(PANEL_APP_SRC),
        "-framework", "Foundation", "-framework", "AppKit", "-framework", "QuartzCore",
        "-o", str(exe),
    ])
    (PANEL_APP / "Contents" / "Info.plist").write_text("""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>CFBundleName</key><string>FSR 4</string>
  <key>CFBundleDisplayName</key><string>FSR 4</string>
  <key>CFBundleExecutable</key><string>FSR4 Panel</string>
  <key>CFBundleIdentifier</key><string>org.fsr4mac.panel</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleVersion</key><string>1</string>
  <key>CFBundleShortVersionString</key><string>1.0</string>
  <key>CFBundleIconFile</key><string>icon</string>
  <key>LSMinimumSystemVersion</key><string>14.0</string>
  <key>NSHighResolutionCapable</key><true/>
</dict></plist>
""")
    iconset = DIST / "icon.iconset"
    if iconset.exists():
        shutil.rmtree(iconset)
    iconset.mkdir()
    run([str(exe), "--make-icon", str(iconset)])
    run(["iconutil", "-c", "icns", str(iconset), "-o", str(res / "icon.icns")])
    shutil.rmtree(iconset)
    run(["codesign", "-f", "-s", "-", str(PANEL_APP)])
    return PANEL_APP


def build_dlopen() -> Path:
    out = DIST / "dlopen_test"
    run([
        "clang", "-O2", "-arch", "x86_64", "-mmacosx-version-min=14.0",
        "-isysroot", sdk(), str(DLOPEN_SRC), "-o", str(out),
    ])
    return out


def patch_macho(path: Path):
    data = bytearray(path.read_bytes())
    if LOAD_PATH in data:
        print("load command already present")
        return
    magic, cputype, cpusub, filetype, ncmds, sizeofcmds, flags, reserved = struct.unpack_from("<IIIIIIII", data, 0)
    if magic != 0xFEEDFACF:
        raise SystemExit(f"not a 64-bit mach-o: {path}")
    cmds_end = 32 + sizeofcmds
    # name is 8-aligned inside the command; 24-byte header + path
    name = LOAD_PATH + b"\0"
    pad = (8 - (len(name) % 8)) % 8
    name += b"\0" * pad
    cmdsize = 24 + len(name)
    if cmdsize % 8:
        raise SystemExit("cmdsize alignment")
    # do not walk into __text
    text_off = None
    off = 32
    for _ in range(ncmds):
        cmd, csz = struct.unpack_from("<II", data, off)
        if cmd == 0x19:
            nsects = struct.unpack_from("<I", data, off + 64)[0]
            so = off + 72
            for _s in range(nsects):
                sect = data[so:so + 16].split(b"\0", 1)[0]
                foff = struct.unpack_from("<I", data, so + 48)[0]
                if sect == b"__text":
                    text_off = foff
                so += 80
        off += csz
    if text_off is None:
        raise SystemExit("no __text")
    if cmds_end + cmdsize > text_off:
        raise SystemExit(f"no header pad: need {cmdsize}, have {text_off - cmds_end}")
    if any(data[cmds_end:cmds_end + cmdsize]):
        raise SystemExit("header pad is not zero")
    struct.pack_into("<IIIIII", data, cmds_end, LC_LOAD_DYLIB, cmdsize, 24, 0, 0x10000, 0x10000)
    data[cmds_end + 24:cmds_end + 24 + len(name)] = name
    struct.pack_into("<II", data, 16, ncmds + 1, sizeofcmds + cmdsize)
    path.write_bytes(data)
    print(f"added LC_LOAD_DYLIB {LOAD_PATH.decode()} cmdsize={cmdsize}")


def find_framework(app: str = "") -> Path:
    """D3DMetal.framework of --crossover, else the first installed CrossOver."""
    apps = [Path(app)] if app else CROSSOVER_APPS
    for candidate in apps:
        framework = candidate / FRAMEWORK_IN_APP
        if framework.is_dir():
            return framework
    raise SystemExit("D3DMetal.framework not found in " + ", ".join(map(str, apps))
                     + "; install CrossOver 24+ or pass --crossover /path/to/CrossOver.app")


def resolve_bottle(value: str) -> Path:
    """A bottle name or a path to the bottle directory or its cxbottle.conf."""
    path = Path(value).expanduser()
    if path.name != "cxbottle.conf":
        path = (path if path.is_dir() else BOTTLES / value) / "cxbottle.conf"
    if not path.is_file():
        names = sorted(p.parent.name for p in BOTTLES.glob("*/cxbottle.conf"))
        raise SystemExit(f"bottle config not found: {path}\navailable bottles: "
                         + (", ".join(names) if names else f"none in {BOTTLES}")
                         + "\nuse --bottle <name>, or --no-bottle to only build and test")
    return path


def copy_framework(hook: Path, source: Path) -> Path:
    dest = DIST / "D3DMetal.framework"
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(source, dest, symlinks=True)
    binary = dest / "Versions" / "A" / "D3DMetal"
    patch_macho(binary)
    shutil.copy2(hook, dest / "Versions" / "A" / "libfsr4hook.dylib")
    run(["codesign", "-f", "-s", "-", str(binary)])
    run(["codesign", "-f", "-s", "-", str(dest / "Versions" / "A" / "libfsr4hook.dylib")])
    return binary


def conf_set(conf: Path, values: dict):
    text = conf.read_text()
    lines = text.splitlines()
    seen = set()
    out = []
    for raw in lines:
        hit = None
        for key in values:
            if raw.strip().startswith(f'"{key}"'):
                hit = key
                break
        if hit:
            out.append(f'"{hit}" = "{values[hit]}"')
            seen.add(hit)
        else:
            out.append(raw)
    missing = [k for k in values if k not in seen]
    if missing:
        if MARKER not in out:
            out.append(MARKER)
        for key in missing:
            out.append(f'"{key}" = "{values[key]}"')
    bak = DIST / "backups" / (conf.parent.name + ".cxbottle.conf")
    bak.parent.mkdir(parents=True, exist_ok=True)
    if not bak.exists():
        shutil.copy2(conf, bak)
        print(f"backup {bak}")
    conf.write_text("\n".join(out) + ("\n" if text.endswith("\n") else ""))
    print(f"set {', '.join(values)} in {conf}")


def conf_unset(conf: Path):
    if not conf.exists():
        print(f"no bottle config {conf}")
        return
    lines = conf.read_text().splitlines()
    out = []
    for raw in lines:
        s = raw.strip()
        if s == MARKER or any(s.startswith(f'"{key}"') for key in OURS):
            continue
        out.append(raw)
    conf.write_text("\n".join(out) + "\n")
    print(f"removed {', '.join(OURS)} from {conf}")


def build_reference(env):
    # Rebuild the same model and backend as the live bridge. An older model's
    # saved frame cannot validate a 4.1.1 installation.
    version = env.get(VERSION_KEY, "4.1.1")
    if version not in ("4.0.2", "4.1.1"):
        raise SystemExit("FSR4_VERSION must be 4.0.2 or 4.1.1")
    if env.get(PROFILE_KEY) == "fsr4":
        version = "4.0.2"  # the live bridge routes this profile to the old model
    model = env.get("FSR4_MODEL", "auto")
    if version == "4.1.1":
        aliases = {"auto": "auto", "reference": "reference", "int8": "reference", "scalar": "scalar",
                   "native": "i8mpp", "i8mpp": "i8mpp"}
        if model not in aliases:
            raise SystemExit("FSR 4.1.1 requires FSR4_MODEL=auto, reference, scalar, or i8mpp; no FP16 model")
        model = aliases[model]
        version_options = ["--fsr-version", version]
        profile = env.get(PROFILE_KEY, "lossy")
        if profile not in PROFILES:
            raise SystemExit("FSR4_PROFILE must be exact, exact-fast, lossy or fsr4")
    else:
        if model == "int8":
            model = "i8mpp"  # historic 4.0.2 live alias
        if model not in ("auto", "f16", "fp16", "i8mpp"):
            model = "auto"
        version_options = ["--fsr-version", version]
    run(["make", "-B", "tools/bin/fsr4mac"], cwd=ROOT)
    info = json.loads(subprocess.check_output(
        [str(ROOT / "tools/bin/fsr4mac"), *version_options, "--backend-info", model], text=True, env=env))
    backend = info["backend"]
    print(f"{info['device']}: {backend} ({info['reason']})")
    testcase = ROOT / "testcases" / "testcase_quality_720to1080"
    if not (testcase / "meta.json").exists():
        run(["make", "tools/bin/gen_testcase"], cwd=ROOT)
        run([str(ROOT / "tools/bin/gen_testcase"), str(testcase),
             "quality", "1280", "720", "1920", "1080", "2"])
    if version == "4.1.1":
        refdir = DIST / ("reference-4.1.1-" + profile + "-" + model)
        run([str(ROOT / "tools/bin/fsr4mac"), str(testcase), str(refdir),
             *version_options, "--profile", profile, "--" + model], cwd=ROOT, env=env)
        report = json.loads((refdir / "run.json").read_text())
        if (report.get("fsr_version") != version or report.get("profile") != profile
                or report.get("backend") not in ("int8-mpp", "int8-reference", "int8-scalar")):
            raise SystemExit("reference did not run the requested FSR 4.1.1 model/profile")
        print(f"reference verified: FSR {version} {profile} {report['backend']}")
    else:
        refdir = DIST / ("reference-" + backend)
        run([str(ROOT / "tools/bin/fsr4mac"), str(testcase), str(refdir),
             *version_options, "--fp16" if backend == "f16" else "--i8mpp"], cwd=ROOT, env=env)
    return refdir / "f00_out.bin"


def cmd_install(args):
    if (args.profile == "fsr4" or args.fsr_version == "4.0.2") and not FSR402_WEIGHTS.is_file():
        raise SystemExit("this copy has no FSR 4.0.2 weights (FSR 4.1-only release); "
                         "use --profile exact-fast or lossy")
    bottle = resolve_bottle(args.bottle) if args.bottle else None
    framework = find_framework(args.crossover)
    print(f"D3DMetal: {framework}")
    log = DIST / "hook.log"
    DIST.mkdir(parents=True, exist_ok=True)
    log.write_text("")
    hook = build_hook(log)
    binary = copy_framework(hook, framework)
    env = os.environ.copy()
    env["FSR4_HOOK_LOG"] = str(log)
    env["FSR4_ROOT"] = str(ROOT)
    env["FSR4_REPLACE"] = "0"
    env[VERSION_KEY] = args.fsr_version
    if args.fsr_version == "4.1.1":
        env[PROFILE_KEY] = args.profile
    smoke = build_smoke()
    run(["arch", "-x86_64", str(smoke)], env=env)
    text = log.read_text()
    if "swizzled" not in text or "encodeToCommandBuffer:" not in text:
        raise SystemExit(f"smoke did not swizzle encode:\n{text}")
    dl = build_dlopen()
    log.write_text("")
    run(["arch", "-x86_64", str(dl), str(binary)], env=env)
    loaded = log.read_text()
    if "loaded pid=" not in loaded:
        raise SystemExit(f"copied D3DMetal did not load the hook:\n{loaded}")
    live = build_live()
    env["FSR4_REPLACE"] = "1"
    env["FSR4_SHARPNESS"] = "0"   # the CLI reference frame has no RCAS
    env["FSR4_SETTINGS"] = "none"  # ignore the panel's saved sharpness
    ref = build_reference(env)
    run(["arch", "-x86_64", str(live),
         str(ROOT / "testcases" / "testcase_quality_720to1080"), str(ref)], env=env)
    print(loaded.strip())
    build_panel_app()
    if bottle:
        values = {
            KEY: str(binary),
            LOG_KEY: str(DIST / "hook.log"),
            ROOT_KEY: str(ROOT),
            REPLACE_KEY: "1",
            VERSION_KEY: args.fsr_version,
        }
        if args.fsr_version == "4.1.1":
            values[PROFILE_KEY] = args.profile
        if args.panel:
            values[PANEL_KEY] = "1"   # status/sharpness panel in the game process
        # DLSS reaches MetalFX only with this D3DMetal switch. Not in OURS:
        # restore leaves it, as it may have been set before the install.
        values[METALFX_KEY] = "1"
        conf_set(bottle, values)
    print("original CrossOver.app was not modified")
    print(f"restore: python3 {Path(__file__)} restore")


def cmd_restore(args):
    conf_unset(resolve_bottle(args.bottle))
    print("CrossOver.app was not modified; next launch uses the bundled D3DMetal")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    i = sub.add_parser("install")
    i.add_argument("--bottle", default=str(DEFAULT_BOTTLE),
                   help="bottle name (e.g. Steam) or path to the bottle / its cxbottle.conf")
    i.add_argument("--no-bottle", action="store_true")
    i.add_argument("--crossover", default="",
                   help="CrossOver.app to copy D3DMetal from (default: first found in /Applications)")
    i.add_argument("--fsr-version", choices=("4.0.2", "4.1.1"), default="4.1.1",
                   help="FSR model version for the live bridge and installation test (default: 4.1.1)")
    i.add_argument("--profile", choices=PROFILES, default="lossy",
                   help="initial panel mode: FSR 4.1.1 exact/exact-fast/lossy, or fsr4 for the "
                        "FSR 4.0.2 model (default: lossy)")
    i.add_argument("--panel", action="store_true",
                   help="also set FSR4_PANEL=1 in the bottle (status/sharpness panel)")
    i.set_defaults(func=cmd_install)
    r = sub.add_parser("restore")
    r.add_argument("--bottle", default=str(DEFAULT_BOTTLE),
                   help="bottle name or path, as given to install")
    r.set_defaults(func=cmd_restore)
    args = p.parse_args()
    if args.cmd == "install" and args.no_bottle:
        args.bottle = ""
    args.func(args)


if __name__ == "__main__":
    main()
