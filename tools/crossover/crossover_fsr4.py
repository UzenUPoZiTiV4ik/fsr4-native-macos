#!/usr/bin/env python3
"""Point one CrossOver bottle at a D3DMetal that loads libfsr4hook.

CrossOver 24/25: libd3dshared reads D3DMETAL_FRAMEWORK_PATH and dlopens that
path (RTLD_NOW); CrossOver.app is not modified and restore deletes only the
bottle keys.

CrossOver 26+: libd3dshared no longer reads that variable and loads
@rpath/D3DMetal.framework/D3DMetal from the app. The framework's
Versions/A/D3DMetal is patched in place, with the original binary kept beside
it as D3DMetal.fsr4-original; restore copies that back and removes the hook.
"""

import argparse
import hashlib
import json
import os
import re
import shlex
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
LIVE412_SRC = ROOT / "src" / "fsr412_live.mm"
NETAL_SRC = ROOT / "src" / "netal_live.mm"
sys.path.insert(0, str(ROOT / "tools"))
from netal_integration import prepare_sources, validate_runtime
from anr.manifest import validate_runtime as validate_anr_runtime
PANEL_SRC = ROOT / "src" / "fsr4_panel.mm"          # hook side: status publish + app launch
PANEL_APP_SRC = ROOT / "tools" / "crossover" / "panel_app.mm"
PANEL_APP = DIST / "FSR4 Panel.app"
SMOKE_SRC = ROOT / "tools" / "crossover" / "smoke.m"
LIVE_TEST_SRC = ROOT / "tools" / "crossover" / "live_test.mm"
DLOPEN_SRC = ROOT / "tools" / "crossover" / "dlopen_test.c"
# D3DMetal inside a CrossOver.app, in order of preference. CrossOver 26 moved
# GPTK 3 to lib64; earlier builds ship it under lib. Newer builds also have
# apple_gptk3; older ones only have apple_gptk.
FRAMEWORK_LIB64_GPTK = "Contents/SharedSupport/CrossOver/lib64/apple_gptk/external/D3DMetal.framework"
FRAMEWORK_LIB64_GPTK3 = "Contents/SharedSupport/CrossOver/lib64/apple_gptk3/external/D3DMetal.framework"
FRAMEWORK_LIB_GPTK = "Contents/SharedSupport/CrossOver/lib/apple_gptk/external/D3DMetal.framework"
FRAMEWORK_LIB_GPTK3 = "Contents/SharedSupport/CrossOver/lib/apple_gptk3/external/D3DMetal.framework"
FRAMEWORKS_IN_APP = (FRAMEWORK_LIB64_GPTK, FRAMEWORK_LIB64_GPTK3,
                     FRAMEWORK_LIB_GPTK, FRAMEWORK_LIB_GPTK3)
FRAMEWORK_IN_APP = FRAMEWORKS_IN_APP[0]
CROSSOVER_MARKER = "Contents/SharedSupport/CrossOver"
CROSSOVER_BUNDLE_ID = "com.codeweavers.CrossOver"
APP_FOLDERS = (Path("/Applications"), Path.home() / "Applications")
STANDARD_NAMES = ("CrossOver.app", "CrossOver Preview.app")
BOTTLES = Path.home() / "Library/Application Support/CrossOver/Bottles"
# Same deployment target as the Makefile: the live bridge uses Metal 4 APIs.
MIN_MACOS = "26.0"
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
FRAMEWORK_KEY = "FSR4_FRAMEWORK"
NETAL_RUNTIME_KEY = "FSR4_NETAL_RUNTIME"
FG_KEY = "FSR4_FG"
ANR_RUNTIME_KEY = "FSR4_ANR_RUNTIME"
METALFX_KEY = "D3DM_ENABLE_METALFX"
ORIGINAL_SUFFIX = ".fsr4-original"
HOOK_NAME = "libfsr4hook.dylib"
# 24/25: libd3dshared reads this. 26 dropped it and loads the framework by
# @loader_path, so the app's own binary has to carry the hook.
FRAMEWORK_PATH_ENV = b"D3DMETAL_FRAMEWORK_PATH"
# "fsr4" remains the legacy panel mode for the FSR 4.0.2 comparison model.
LITE_LOSSY_SOURCE_DLL_SHA256 = "d9b0a1f5781c2d633a327272f6f1c8607057df7cafd0233cf38c7395c8f7dbc5"
LITE_LOSSY_INITIALIZER_SHA256 = "c77c2d1a4d119ed76d94b8c23335490cc38280db3346c72396a1e1acb0523cfc"
LITE_LOSSY_PROFILES = ("int8-lite-lossy", "int8-lite-lossy-tiled")
LITE_PROFILES = ("int8-lite", "int8-lite-pixfix", "int8-lite-hair", *LITE_LOSSY_PROFILES)
LEGACY_PROFILES = ("exact", "exact-fast", "lossy", "lossy2", "fsr4")
PROFILES = (*LITE_PROFILES, *LEGACY_PROFILES, "netal")
OURS = (KEY, LOG_KEY, ROOT_KEY, REPLACE_KEY, PANEL_KEY, VERSION_KEY, PROFILE_KEY, FRAMEWORK_KEY,
        NETAL_RUNTIME_KEY, FG_KEY, ANR_RUNTIME_KEY)


def run(cmd, **kw):
    print("+", " ".join(cmd))
    subprocess.check_call(cmd, **kw)


def sdk():
    return subprocess.check_output(["xcrun", "--sdk", "macosx", "--show-sdk-path"], text=True).strip()


def build_hook(log_path: Path, netal_source=None) -> Path:
    netal_source = netal_source or os.environ.get("FSR4_NETAL_SOURCE")
    generated = prepare_sources(netal_source) if netal_source else None
    DIST.mkdir(parents=True, exist_ok=True)
    out = DIST / "libfsr4hook.dylib"
    common = ["-O2", "-fobjc-arc", "-arch", "x86_64", f"-mmacosx-version-min={MIN_MACOS}",
              "-isysroot", sdk(), "-I", str(ROOT / "src")]
    run(["clang", "-c", *common, str(HOOK_SRC), "-o", str(DIST / "hook.o")])
    run(["clang++", "-c", "-std=c++17", *common, str(LIVE_SRC), "-o", str(DIST / "live.o")])
    run(["clang++", "-c", "-std=c++17", *common, str(LIVE411_SRC), "-o", str(DIST / "live411.o")])
    run(["clang++", "-c", "-std=c++17", *common, str(LIVE412_SRC), "-o", str(DIST / "live412.o")])
    run(["clang++", "-c", "-std=c++17", *common, str(PANEL_SRC), "-o", str(DIST / "panel.o")])
    anr_objects = []
    for name, source in (("anr_hook", ROOT / "hook/anr_metalfx.mm"),
                         ("anr_live", ROOT / "src/anr_live.mm"), ("anr_runtime", ROOT / "src/anr_runtime.mm")):
        object_path = str(DIST / (name + ".o"))
        run(["clang++", "-c", "-std=c++17", *common, str(source), "-o", object_path])
        anr_objects.append(object_path)
    # Keep the old helper intact for hooks already mapped in a running game.
    # UI v2 adds noise-aware recomposition and sampled statistics (ABI 36 bytes).
    helper = ROOT / "build/anr_helpers_ui_v2.metallib"
    helper.parent.mkdir(parents=True, exist_ok=True)
    run(["xcrun", "metal", "-std=metal3.0", "-O2", "-c", str(ROOT / "src/anr_helpers.metal"),
         "-o", str(ROOT / "build/anr_helpers_ui_v2.air")])
    run(["xcrun", "metallib", str(ROOT / "build/anr_helpers_ui_v2.air"), "-o", str(helper)])
    netal_objects = [str(DIST / "netal.o")]
    netal_flags = ["-DFSR4_WITH_NETAL=1", "-I", str(generated)] if generated else []
    run(["clang++", "-c", "-std=c++17", *common, *netal_flags, str(NETAL_SRC), "-o", netal_objects[0]])
    if generated:
        for name in ("dlss_sr_session", "dlss_sr_texture_adapter"):
            out_object = str(DIST / ("netal_" + name + ".o"))
            run(["clang++", "-c", "-std=c++20", *common, "-I", str(generated),
                 str(generated / (name + ".mm")), "-o", out_object])
            netal_objects.append(out_object)
    run([
        "clang++", "-dynamiclib", *common,
        str(DIST / "hook.o"), str(DIST / "live.o"), str(DIST / "live411.o"), str(DIST / "live412.o"), str(DIST / "panel.o"),
        *netal_objects, *anr_objects,
        "-framework", "Foundation", "-framework", "Metal", "-framework", "MetalFX",
        # Darwin ld needs an explicit allowance as well as weak_import when
        # linking an encoder without the optional 4.1.1 symbol. The public
        # version router rejects an unavailable requested model explicitly.
        "-Wl,-U,_fsr411_encode_live_frame",
        "-Wl,-U,_fsr411_invalidate_owner", "-Wl,-U,_fsr411_release_owner",
        "-Wl,-U,_fsr412_encode_live_frame",
        "-Wl,-U,_fsr412_invalidate_owner", "-Wl,-U,_fsr412_release_owner",
        "-install_name", "@loader_path/libfsr4hook.dylib",
        "-o", str(out),
    ])
    run(["codesign", "-f", "-s", "-", str(out)])
    receipt = DIST / "netal-build.json"
    if generated:
        receipt.write_text(json.dumps({"schema": "fsr4_netal_link_v1", "linked": True,
            "hook_sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
            "source_receipt": json.loads((generated / "source-receipt.json").read_text())}, indent=2) + "\n")
    elif receipt.exists():
        receipt.unlink()
    return out


def build_smoke() -> Path:
    out = DIST / "smoke"
    run([
        "clang", "-O2", "-fobjc-arc", "-arch", "x86_64",
        f"-mmacosx-version-min={MIN_MACOS}", "-isysroot", sdk(),
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
        f"-mmacosx-version-min={MIN_MACOS}", "-isysroot", sdk(),
        "-I", str(ROOT / "src"),
        str(LIVE_TEST_SRC), str(DIST / "libfsr4hook.dylib"),
        "-framework", "Foundation", "-framework", "Metal", "-framework", "MetalFX",
        "-Wl,-rpath", str(DIST),
        "-o", str(out),
    ])
    return out


def build_anr_live() -> Path:
    out = DIST / "anr_live_test"
    run(["clang++", "-O2", "-fobjc-arc", "-std=c++17", "-arch", "x86_64",
         f"-mmacosx-version-min={MIN_MACOS}", "-isysroot", sdk(), "-I", str(ROOT / "src"),
         str(ROOT / "tests/anr_live_test.mm"), str(DIST / "libfsr4hook.dylib"),
         "-framework", "Foundation", "-framework", "Metal", "-framework", "MetalFX",
         "-Wl,-rpath", str(DIST), "-o", str(out)])
    return out


def preserve_panel_preferences(conf: Path):
    text = conf.read_text()
    hit = re.search(r'^\s*"FSR4_ROOT"\s*=\s*"([^"]+)"', text, re.MULTILINE)
    if not hit:
        return
    old_root = Path(hit.group(1)).expanduser()
    if not old_root.is_absolute() or old_root.resolve() == ROOT.resolve():
        return
    source = old_root / "dist/crossover/panel.conf"
    if not source.is_file():
        return
    target = DIST / "panel.conf"
    previous = target.read_bytes() if target.exists() else None
    if previous is not None and previous != source.read_bytes():
        backup = DIST / "backups/panel-before-anr.conf"
        backup.parent.mkdir(parents=True, exist_ok=True)
        if not backup.exists():
            backup.write_bytes(previous)
    temporary = target.with_name(target.name + f".tmp.{os.getpid()}")
    shutil.copy2(source, temporary)
    temporary.replace(target)
    print("preserved existing SR panel sharpness/profile preferences")


def cmd_frame_generation(args):
    runtime = None
    netal_source = getattr(args, "netal_source", "") or os.environ.get("FSR4_NETAL_SOURCE")
    netal_runtime = getattr(args, "netal_runtime", "") or os.environ.get(NETAL_RUNTIME_KEY)
    if args.mode == "anr" and netal_source:
        try:
            prepare_sources(netal_source)
            if netal_runtime:
                netal_runtime = validate_runtime(netal_runtime)
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise SystemExit("Netal: " + str(error)) from error
    elif args.mode == "anr" and netal_runtime:
        raise SystemExit("--netal-runtime requires --netal-source")
    if args.mode == "anr":
        try:
            runtime = validate_anr_runtime(args.anr_runtime)
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise SystemExit("ANR: " + str(error)) from error
    bottle = resolve_bottle(args.bottle) if args.bottle else default_bottle(crossover=args.crossover)
    # Do not change the SR profile or panel preferences. Install the combined
    # hook and update only FG/root/framework keys in the selected bottle.
    if args.mode == "anr":
        framework = find_framework(args.crossover)
        hook = build_hook(DIST / "hook.log", netal_source=netal_source)
        binary = copy_framework(hook, framework)
        test = build_anr_live()
        test_runtime = runtime if (runtime / "manifest.json").is_file() else runtime / "1920x1080"
        if not (test_runtime / "manifest.json").is_file():
            test_runtime = next(p.parent for p in runtime.glob("*/manifest.json"))
        if (test_runtime / "manifest.json").is_file():
            run(["arch", "-x86_64", str(test), str(ROOT), str(test_runtime), "classic"])
            run(["arch", "-x86_64", str(test), str(ROOT), str(test_runtime), "metal4"])
        values = {KEY: str(binary), ROOT_KEY: str(ROOT), LOG_KEY: str(DIST / "hook.log"),
                  FG_KEY: "anr", ANR_RUNTIME_KEY: str(runtime), METALFX_KEY: "1"}
        if netal_source:
            if netal_runtime:
                values[NETAL_RUNTIME_KEY] = str(netal_runtime)
                receipt_path = DIST / "netal-build.json"
                receipt = json.loads(receipt_path.read_text())
                receipt["runtime"] = str(netal_runtime)
                receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
            print("Netal SR engine is linked" + (" with a validated runtime" if netal_runtime else
                  "; SR profile remains unchanged because a complete Netal runtime is not supplied"))
        print("ANR FG hook verified; only the chosen bottle will use the new runtime")
        preserve_panel_preferences(bottle)
        build_panel_app()
        conf_set(bottle, values)
    else:
        conf_set(bottle, {FG_KEY: "metalfx"})
    print("Frame generation selected: " + args.mode + "; SR profile/settings were not changed")


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
        f"-mmacosx-version-min={MIN_MACOS}", "-isysroot", sdk(),
        str(PANEL_APP_SRC),
        "-framework", "Foundation", "-framework", "AppKit", "-framework", "QuartzCore",
        "-o", str(exe),
    ])
    (PANEL_APP / "Contents" / "Info.plist").write_text(f"""<?xml version="1.0" encoding="UTF-8"?>
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
  <key>LSMinimumSystemVersion</key><string>{MIN_MACOS}</string>
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
        "clang", "-O2", "-arch", "x86_64", f"-mmacosx-version-min={MIN_MACOS}",
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


def is_crossover(app: Path) -> bool:
    """A CrossOver.app under any name (copies get renamed, e.g. CrossOver_patched)."""
    return app.suffix == ".app" and (app / CROSSOVER_MARKER).is_dir()


def spotlight_crossovers() -> list:
    """CrossOver copies outside the Applications folders (other disks), by bundle id."""
    try:
        out = subprocess.run(["/usr/bin/mdfind", f"kMDItemCFBundleIdentifier == '{CROSSOVER_BUNDLE_ID}'"],
                             capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    return [Path(line) for line in out.splitlines() if line.endswith(".app")]


def crossover_apps(folders=APP_FOLDERS, spotlight=True) -> list:
    """Installed CrossOver copies: CrossOver.app / CrossOver Preview.app in
    /Applications, then ~/Applications, then any renamed copy in those folders
    or one subfolder down, then Spotlight hits. Same order as the installer app."""
    found = []

    def add(app: Path):
        if app not in found and is_crossover(app):
            found.append(app)

    for folder in folders:
        for name in STANDARD_NAMES:
            add(folder / name)
    for folder in folders:
        try:
            entries = sorted(folder.iterdir(), key=lambda p: p.name.lower())
        except OSError:
            continue
        for entry in entries:
            if entry.suffix == ".app":
                add(entry)
            elif entry.is_dir() and not entry.name.startswith("."):
                try:
                    for sub in sorted(entry.glob("*.app"), key=lambda p: p.name.lower()):
                        add(sub)
                except OSError:
                    pass
    if spotlight:
        for app in spotlight_crossovers():
            add(app)
    return found


def framework_in(app: Path):
    for relative in FRAMEWORKS_IN_APP:
        if (app / relative).is_dir():
            return app / relative
    return None


def find_framework(app: str = "") -> Path:
    """D3DMetal.framework of --crossover, else the first installed CrossOver that has one."""
    apps = [Path(app).expanduser()] if app else crossover_apps()
    for candidate in apps:
        framework = framework_in(candidate)
        if framework:
            return framework
    if app:
        raise SystemExit(f"no D3DMetal.framework in {apps[0]}; pass --crossover with CrossOver.app 24+")
    where = ", ".join(map(str, apps)) if apps else "/Applications or ~/Applications"
    raise SystemExit(f"D3DMetal.framework not found in {where}; install CrossOver 24+ "
                     "or pass --crossover /path/to/CrossOver.app")


def framework_mode(framework: Path) -> str:
    """How this CrossOver loads D3DMetal.

    CrossOver 24/25's libd3dshared reads D3DMETAL_FRAMEWORK_PATH and can be
    pointed at a copy; 26 dropped that variable and loads the framework next
    to libd3dshared, so only the app's own binary can load the hook."""
    shared = framework.parent / "libd3dshared.dylib"
    try:
        return "redirect" if FRAMEWORK_PATH_ENV in shared.read_bytes() else "inplace"
    except OSError:
        return "inplace"


def bottle_roots(crossover=None) -> list:
    """Read CrossOver's configured search path, including external volumes."""
    environment = dict(os.environ)
    bottle_path = ""
    # CrossOver's user config overrides the app config. Preview shares the
    # CrossOver cellar, and --crossover may select an app outside /Applications.
    apps = [Path(crossover).expanduser()] if crossover else crossover_apps()
    configs = [p / "Contents/SharedSupport/CrossOver/etc/CrossOver.conf" for p in apps[:1]]
    configs.append(BOTTLES.parent / "CrossOver.conf")
    for config in configs:
        try:
            lines = config.read_text(errors="replace").splitlines()
        except OSError:
            continue
        section = ""
        for line in lines:
            line = line.strip()
            if line.startswith(";"):
                continue
            header = re.fullmatch(r"\[([^]]+)\]\s*(?:;.*)?", line)
            if header:
                section = header[1].lower()
                continue
            field = re.fullmatch(r'"((?:[^"\\]|\\.)*)"\s*=\s*"((?:[^"\\]|\\.)*)"\s*(?:;.*)?', line)
            if not field:
                continue
            key, value = (re.sub(r'\\(["\\])', r'\1', part) for part in field.groups())
            if section == "crossover" and key.lower() == "bottlepath":
                bottle_path = value
            elif section == "environmentvariables":
                environment[key] = re.sub(r"\$\{([^}]+)\}", lambda m: environment.get(m[1], ""), value)
    # CX_BOTTLE_PATH takes precedence, as in CXBottle::get_crossover_config.
    search = environment.get("CX_BOTTLE_PATH") or bottle_path
    roots = []
    for value in search.split(os.pathsep) if search else [str(BOTTLES)]:
        value = re.sub(r"\$\{([^}]+)\}", lambda m: environment.get(m[1], ""), value)
        path = Path(value).expanduser()
        if value and path.is_absolute() and path not in roots:
            roots.append(path)
    return roots


def discover_bottles(crossover=None) -> list:
    # A symlink to an external bottle can also live in the default directory.
    # Resolve configs to avoid listing the same bottle through two roots.
    confs = {}
    for root in bottle_roots(crossover):
        for conf in root.glob("*/cxbottle.conf"):
            if conf.is_file():
                confs.setdefault(conf.resolve(), conf)
    return sorted(confs.values(), key=lambda p: (p.parent.name, str(p)))


def bottle_names(crossover=None) -> list:
    return sorted({p.parent.name for p in discover_bottles(crossover)})


def has_install(conf: Path) -> bool:
    text = conf.read_text(errors="replace")
    return MARKER in text or f'"{KEY}"' in text or f'"{FRAMEWORK_KEY}"' in text


def default_bottle(for_restore: bool = False, crossover=None) -> Path:
    """The bottle to use when --bottle is omitted: the only bottle, or (for
    restore) the only bottle that carries our keys. Never guesses between
    several."""
    confs = discover_bottles(crossover)
    if for_restore:
        installed = [c for c in confs if has_install(c)]
        if len(installed) == 1:
            return installed[0]
        if not installed:
            raise SystemExit("no discovered bottle has fsr4-mac keys; nothing to restore")
        confs = installed
    if len(confs) == 1:
        return confs[0]
    if not confs:
        raise SystemExit("no CrossOver bottles in " + ", ".join(map(str, bottle_roots(crossover))) + "\n"
                         "pass --bottle /path/to/cxbottle.conf, or --no-bottle to only build and test")
    raise SystemExit("several bottles found: " + ", ".join(c.parent.name for c in confs)
                     + "\n" + "\n".join(str(c) for c in confs)
                     + "\npick one with --bottle <name> or the full config path")


def resolve_bottle(value: str, crossover=None) -> Path:
    """A bottle name or a path to the bottle directory or its cxbottle.conf."""
    path = Path(value).expanduser()
    if path.name != "cxbottle.conf" and not path.is_dir() and len(path.parts) == 1:
        matches = [conf for conf in discover_bottles(crossover) if conf.parent.name == value]
        if len(matches) > 1:
            raise SystemExit("ambiguous bottle name: " + value + "\nuse --bottle with a full path:\n"
                             + "\n".join(str(conf) for conf in matches))
        if matches:
            return matches[0]
        path = BOTTLES / value
    if path.name != "cxbottle.conf":
        path = path / "cxbottle.conf"
    if not path.is_file():
        names = bottle_names(crossover)
        raise SystemExit(f"bottle config not found: {path}\navailable bottles: "
                         + (", ".join(names) if names else f"none in {BOTTLES}")
                         + "\nuse --bottle <name>, or --no-bottle to only build and test")
    return path


def atomic_copy(source: Path, dest: Path):
    """Replace dest with source without exposing a half-written file."""
    temp = dest.with_name(dest.name + f".tmp.{os.getpid()}")
    shutil.copy2(source, temp)
    os.replace(temp, dest)


def remove_file(path: Path):
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def require_writable(directory: Path):
    """Fail with an actionable message when macOS blocks changes to CrossOver.app."""
    probe = directory / f".fsr4-write-test.{os.getpid()}"
    try:
        probe.write_bytes(b"")
    except OSError as error:
        raise SystemExit(
            f"cannot write {directory}: {error}\n"
            "macOS may be blocking changes to CrossOver.app (App Management). "
            "Allow Terminal or FSR 4 Installer in System Settings > Privacy & Security > "
            "App Management, quit CrossOver, and try again.") from error
    finally:
        remove_file(probe)


def preflight_inplace(framework: Path):
    """Check the binary and write access before the long build."""
    binary = framework / "Versions" / "A" / "D3DMetal"
    if not binary.is_file():
        raise SystemExit(f"D3DMetal binary not found: {binary}\n"
                         "CrossOver 26 keeps it in D3DMetal.framework/Versions/A/D3DMetal")
    require_writable(binary.parent)


def copy_framework(hook: Path, source: Path) -> Path:
    """Copy D3DMetal from CrossOver into dist and add the hook to that copy.

    In-place installs keep the pristine binary as
    Versions/A/D3DMetal.fsr4-original next to the patched one; a reinstall
    must patch a clean binary, not one that already has the load command."""
    dest = DIST / "D3DMetal.framework"
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(source, dest, symlinks=True,
                    ignore=shutil.ignore_patterns("D3DMetal" + ORIGINAL_SUFFIX, HOOK_NAME))
    binary = dest / "Versions" / "A" / "D3DMetal"
    current = source / "Versions" / "A" / "D3DMetal"
    pristine = current.with_name(current.name + ORIGINAL_SUFFIX)
    if pristine.is_file() and LOAD_PATH in current.read_bytes():
        shutil.copy2(pristine, binary)
    patch_macho(binary)
    shutil.copy2(hook, dest / "Versions" / "A" / HOOK_NAME)
    run(["codesign", "-f", "-s", "-", str(binary)])
    run(["codesign", "-f", "-s", "-", str(dest / "Versions" / "A" / HOOK_NAME)])
    return binary


def install_inplace(hook: Path, framework: Path, patched: Path) -> Path:
    """Patch the app's own D3DMetal; the original stays next to it."""
    binary = framework / "Versions" / "A" / "D3DMetal"
    original = binary.with_name(binary.name + ORIGINAL_SUFFIX)
    current = binary.read_bytes()
    if original.exists():
        if LOAD_PATH not in current:
            atomic_copy(binary, original)   # CrossOver was updated since the last install
    elif LOAD_PATH in current:
        raise SystemExit(f"{binary} already loads a hook but {original.name} is missing; "
                         "reinstall CrossOver to get an original binary")
    else:
        atomic_copy(binary, original)
    try:
        # The dist copy and the hook are already ad-hoc signed. Copying them
        # keeps their embedded signatures and, unlike codesign here, does not
        # rewrite the framework's _CodeSignature seal, so restore can make the
        # bundle verify again.
        atomic_copy(patched, binary)
        atomic_copy(hook, binary.with_name(HOOK_NAME))
    except BaseException:
        try:
            if original.exists():
                atomic_copy(original, binary)
            remove_file(binary.with_name(HOOK_NAME))
        except OSError as error:
            print(f"warning: rollback failed: {error}")
        raise
    return binary


def restore_needed(binary: Path) -> bool:
    """True when restore would change or clean anything next to the binary."""
    original = binary.with_name(binary.name + ORIGINAL_SUFFIX)
    hook = binary.with_name(HOOK_NAME)
    if original.exists() or hook.exists():
        return True
    try:
        return LOAD_PATH in binary.read_bytes()
    except OSError:
        return False


def restore_inplace(binary: Path) -> bool:
    """Put CrossOver's original D3DMetal back. Returns True if it changed."""
    original = binary.with_name(binary.name + ORIGINAL_SUFFIX)
    hook = binary.with_name(HOOK_NAME)
    try:
        patched = LOAD_PATH in binary.read_bytes()
    except OSError as error:
        print(f"cannot read {binary}: {error}")
        return False
    if original.exists() and patched:
        atomic_copy(original, binary)
        remove_file(original)
        remove_file(hook)
        print(f"restored {binary} from {original.name}")
        try:
            run(["codesign", "--verify", "--strict", str(binary)])
        except subprocess.CalledProcessError:
            print(f"warning: {binary} does not verify after restore")
        return True
    if original.exists():
        # CrossOver replaced the binary after our install; the backup is stale.
        remove_file(original)
        remove_file(hook)
        print(f"{binary} is already original; removed the fsr4 backup")
        return False
    if patched:
        print(f"warning: {binary} still loads the hook but {original.name} is missing; "
              "reinstall CrossOver to undo the patch")
        return False
    remove_file(hook)
    return False


def conf_get(conf: Path, key: str) -> str:
    try:
        text = conf.read_text(errors="replace")
    except OSError:
        return ""
    field = re.compile(r'\s*"' + re.escape(key) + r'"\s*=\s*"((?:[^"\\]|\\.)*)"')
    for line in text.splitlines():
        match = field.match(line)
        if match:
            return re.sub(r'\\(["\\])', r'\1', match.group(1))
    return ""


def conf_remove(conf: Path, keys):
    try:
        text = conf.read_text()
    except OSError:
        return
    lines = text.splitlines()
    out = [line for line in lines if not any(line.strip().startswith(f'"{key}"') for key in keys)]
    if out != lines:
        conf.write_text("\n".join(out) + ("\n" if text.endswith("\n") else ""))
        print(f"removed {', '.join(keys)} from {conf}")


def other_bottles_using(binary: Path, current: Path) -> list:
    """Bottles (besides current) whose FSR4_FRAMEWORK points at this binary."""
    try:
        target = binary.resolve()
    except OSError:
        target = binary
    users = []
    for conf in discover_bottles():
        try:
            if conf.resolve() == current.resolve():
                continue
        except OSError:
            pass
        value = conf_get(conf, FRAMEWORK_KEY)
        if not value:
            continue
        path = Path(value).expanduser()
        try:
            same = path.resolve() == target
        except OSError:
            same = path == binary
        if same:
            users.append(conf)
    return users


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


def select_panel_profile(profile: str):
    # Saved panel settings override the bottle environment. Update the selected
    # model too, while retaining the user's sharpness and RCAS preferences.
    conf = DIST / "panel.conf"
    text = conf.read_text() if conf.exists() else ""
    if conf.exists():
        backup = DIST / "backups" / "panel.conf"
        backup.parent.mkdir(parents=True, exist_ok=True)
        if not backup.exists():
            shutil.copy2(conf, backup)
    lines = [line for line in text.splitlines() if line.partition("=")[0].strip() != "profile"]
    lines.append("profile=" + profile)
    temp = conf.with_name(conf.name + f".tmp.{os.getpid()}")
    temp.write_text("\n".join(lines) + "\n")
    temp.replace(conf)
    print(f"selected {profile} in {conf}")


def reject_lite_lossy_reference(model):
    if model in ("reference", "int8"):
        raise SystemExit("FSR 4.1.2 Lite Lossy does not support the AMD reference backend; "
                         "use FSR4_MODEL=auto, native/i8mpp, scalar, or simd")


def require_lite_lossy_weights(expected_hash=LITE_LOSSY_INITIALIZER_SHA256):
    # Fetch is an explicit user action. Never run a Windows provider/DLL or
    # silently substitute the ordinary Lite initializer for a Lossy install.
    base = ROOT / "build/fsr412_lossy"
    initializer = base / "initializers-lite-lossy.bin"
    receipt = base / "import.json"
    hint = "; run make lite-lossy-weights, then retry"
    if not initializer.is_file() or not receipt.is_file():
        raise SystemExit("FSR 4.1.2 Lite Lossy requires pinned weights and import.json in "
                         "build/fsr412_lossy" + hint)
    try:
        metadata = json.loads(receipt.read_text())
        if not isinstance(metadata, dict) or metadata.get("schema") != "fsr412_lite_lossy_import_v1":
            raise ValueError("unsupported or malformed import receipt")
        if metadata.get("source_dll_sha256") != LITE_LOSSY_SOURCE_DLL_SHA256:
            raise ValueError("import receipt source DLL SHA-256 is not the pinned provider")
        initializers = metadata.get("initializers")
        expected = initializers.get(initializer.name) if isinstance(initializers, dict) else None
        if not isinstance(expected, str) or re.fullmatch(r"[0-9a-f]{64}", expected) is None:
            raise ValueError("import receipt has no valid initializer SHA-256")
        data = initializer.read_bytes()
        actual = hashlib.sha256(data).hexdigest()
        if not data or actual != expected:
            raise ValueError("initializer SHA-256 does not match import receipt")
        if actual != expected_hash:
            raise ValueError("initializer SHA-256 does not match pinned initializer")
    except (OSError, UnicodeError, ValueError) as error:
        raise SystemExit("Invalid FSR 4.1.2 Lite Lossy weights/import.json: " + str(error) + hint) from error


def build_reference(env):
    # Rebuild the same model and backend as the live bridge. An older model's
    # saved frame cannot validate a different model installation.
    version = env.get(VERSION_KEY, "4.1.2")
    if version not in ("4.0.2", "4.1.1", "4.1.2"):
        raise SystemExit("FSR4_VERSION must be 4.0.2, 4.1.1 or 4.1.2")
    if env.get(PROFILE_KEY) == "fsr4":
        version = "4.0.2"  # the live bridge routes this profile to the old model
    model = env.get("FSR4_MODEL", "auto")
    if version == "4.1.1":
        aliases = {"auto": "auto", "reference": "reference", "int8": "reference", "scalar": "scalar",
                   "simd": "simd", "fp16": "simd", "f16": "simd", "native": "i8mpp", "i8mpp": "i8mpp"}
        if model not in aliases:
            raise SystemExit("FSR 4.1.1 requires FSR4_MODEL=auto, reference, scalar, simd, or i8mpp")
        model = aliases[model]
        version_options = ["--fsr-version", version]
        profile = env.get(PROFILE_KEY, "lossy")
        if profile not in LEGACY_PROFILES:
            raise SystemExit("FSR4_PROFILE must be exact, exact-fast, lossy, lossy2 or fsr4")
    elif version == "4.1.2":
        aliases = {"auto": "auto", "native": "i8mpp", "i8mpp": "i8mpp",
                   "int8": "reference", "reference": "reference", "scalar": "scalar",
                   "simd": "simd", "fp16": "simd", "f16": "simd"}
        if model not in aliases:
            raise SystemExit("FSR 4.1.2 requires FSR4_MODEL=auto, reference, scalar, simd, or i8mpp")
        model = aliases[model]
        version_options = ["--fsr-version", version]
        profile = env.get(PROFILE_KEY, "int8-lite")
        profile = {"lite": "int8-lite", "lite-hair": "int8-lite-hair", "lite-pixfix": "int8-lite-pixfix",
                   "lite-lossy": "int8-lite-lossy", "lite-lossy-tiled": "int8-lite-lossy-tiled"}.get(profile, profile)
        if profile not in LITE_PROFILES:
            raise SystemExit("FSR4_PROFILE must be int8-lite, int8-lite-pixfix, int8-lite-lossy "
                             "or int8-lite-lossy-tiled for FSR 4.1.2")
        if profile in LITE_LOSSY_PROFILES:
            reject_lite_lossy_reference(model)
            require_lite_lossy_weights()
    else:
        if model == "int8":
            model = "i8mpp"  # historic 4.0.2 live alias
        if model not in ("auto", "f16", "fp16", "i8mpp"):
            model = "auto"
        version_options = ["--fsr-version", version]
    run(["make", "-B", "tools/bin/fsr4mac"], cwd=ROOT)
    backend_profile = ["--profile", profile] if version == "4.1.2" and profile in LITE_LOSSY_PROFILES else []
    info = json.loads(subprocess.check_output(
        [str(ROOT / "tools/bin/fsr4mac"), *version_options, *backend_profile, "--backend-info", model], text=True, env=env))
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
                or report.get("backend") not in ("int8-mpp", "int8-simd", "int8-reference", "int8-scalar")):
            raise SystemExit("reference did not run the requested FSR 4.1.1 model/profile")
        print(f"reference verified: FSR {version} {profile} {report['backend']}")
    elif version == "4.1.2":
        refdir = DIST / ("reference-4.1.2-" + profile + "-" + model)
        run([str(ROOT / "tools/bin/fsr4mac"), str(testcase), str(refdir),
             *version_options, "--profile", profile, "--" + model], cwd=ROOT, env=env)
        report = json.loads((refdir / "run.json").read_text())
        actual_profile = profile if profile in LITE_LOSSY_PROFILES else "int8-lite"
        allowed_backends = ("int8-mpp", "int8-scalar", "int8-simd")
        if profile not in LITE_LOSSY_PROFILES:
            allowed_backends += ("int8-reference",)
        if (report.get("fsr_version") != version or report.get("profile") != actual_profile
                or report.get("backend") not in allowed_backends):
            raise SystemExit("reference did not run the requested FSR 4.1.2 model/profile: " + profile)
        print(f"reference verified: FSR {version} {profile} {report['backend']}")
    else:
        refdir = DIST / ("reference-" + backend)
        run([str(ROOT / "tools/bin/fsr4mac"), str(testcase), str(refdir),
             *version_options, "--fp16" if backend == "f16" else "--i8mpp"], cwd=ROOT, env=env)
    return refdir / "f00_out.bin"


def cmd_install(args):
    netal_source = getattr(args, "netal_source", "") or os.environ.get("FSR4_NETAL_SOURCE")
    netal_runtime = getattr(args, "netal_runtime", "") or os.environ.get(NETAL_RUNTIME_KEY)
    fg = getattr(args, "frame_generation", "metalfx")
    anr_runtime = getattr(args, "anr_runtime", "") or os.environ.get(ANR_RUNTIME_KEY)
    if fg == "anr":
        if not anr_runtime:
            raise SystemExit("--frame-generation anr requires --anr-runtime (prepared model, not the test kit)")
        try:
            anr_runtime = validate_anr_runtime(anr_runtime)
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise SystemExit("ANR: " + str(error)) from error
    if netal_runtime and not netal_source:
        raise SystemExit("--netal-runtime requires --netal-source (or FSR4_NETAL_SOURCE)")
    if netal_source and args.profile != "netal":
        try:
            prepare_sources(netal_source)
            if netal_runtime:
                netal_runtime = validate_runtime(netal_runtime)
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise SystemExit("Netal: " + str(error)) from error
    if args.profile == "netal":
        if args.fsr_version == "4.0.2":
            raise SystemExit("Netal cannot be selected with --fsr-version 4.0.2 (the legacy model pin)")
        if not netal_source or not netal_runtime:
            raise SystemExit("Netal requires --netal-source and --netal-runtime (complete Preset K SR bundle); models are not downloaded")
        try:
            prepare_sources(netal_source)
            netal_runtime = validate_runtime(netal_runtime)
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise SystemExit("Netal: " + str(error)) from error
    if args.profile in LITE_LOSSY_PROFILES:
        if args.fsr_version != "4.1.2":
            raise SystemExit("Lite Lossy profiles require --fsr-version 4.1.2")
        reject_lite_lossy_reference(os.environ.get("FSR4_MODEL", "auto"))
        require_lite_lossy_weights()
    if (args.profile == "fsr4" or args.fsr_version == "4.0.2") and not FSR402_WEIGHTS.is_file():
        raise SystemExit("this copy has no FSR 4.0.2 weights (FSR 4.1-only release); "
                         "use --profile exact-fast, lossy or lossy2")
    # FSR 4.1.2 Lossy: fetch on request, and refresh weights left by an older
    # release (the fetcher returns at once when the pinned file is present).
    if args.profile == "lossy2" or (ROOT / "build/fsr411_lossy2/weights_lossy2.npz").is_file():
        run([sys.executable, str(ROOT / "tools/fsr411/fetch_lossy2_weights.py")], cwd=ROOT)
    if args.no_bottle:
        bottle = None
    else:
        bottle = resolve_bottle(args.bottle, args.crossover) if args.bottle else default_bottle(crossover=args.crossover)
        print(f"bottle: {bottle}")
    framework = find_framework(args.crossover)
    mode = framework_mode(framework)
    print(f"D3DMetal: {framework} ({'in-place' if mode == 'inplace' else 'redirect'})")
    if mode == "inplace" and bottle:
        preflight_inplace(framework)
    log = DIST / "hook.log"
    DIST.mkdir(parents=True, exist_ok=True)
    log.write_text("")
    hook = build_hook(log, netal_source=netal_source)
    binary = copy_framework(hook, framework)
    env = os.environ.copy()
    env["FSR4_HOOK_LOG"] = str(log)
    env["FSR4_ROOT"] = str(ROOT)
    env["FSR4_REPLACE"] = "0"
    env[VERSION_KEY] = args.fsr_version
    if args.fsr_version in ("4.1.1", "4.1.2"):
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
    # The existing synthetic FSR fixture has normal depth and cannot prove
    # DLSS output. Still verify the FSR router, then require an independent
    # Netal session initialization for the supplied, fully validated runtime.
    reference_env = env.copy()
    if args.profile == "netal":
        reference_env[VERSION_KEY] = "4.1.2"
        reference_env[PROFILE_KEY] = "int8-lite"
    ref = build_reference(reference_env)
    run(["arch", "-x86_64", str(live),
         str(ROOT / "testcases" / "testcase_quality_720to1080"), str(ref)], env=reference_env)
    if args.profile == "netal":
        run([sys.executable, str(ROOT / "tools/netal_probe.py"), "--source", str(netal_source),
             "--runtime", str(netal_runtime), "--arch", "x86_64"], cwd=ROOT)
        print("Netal session initialized; game image quality is not validated by the FSR fixture")
    if netal_source and netal_runtime:
        receipt_path = DIST / "netal-build.json"
        receipt = json.loads(receipt_path.read_text())
        receipt["runtime"] = str(netal_runtime)
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    print(loaded.strip())
    build_panel_app()
    installed = None
    if bottle:
        if mode == "inplace":
            installed = install_inplace(hook, framework, binary)
            log.write_text("")
            run(["arch", "-x86_64", str(dl), str(installed)], env=env)
            installed_log = log.read_text()
            if "loaded pid=" not in installed_log:
                if other_bottles_using(installed, bottle):
                    print("warning: other bottles use this D3DMetal; run install for them again")
                restore_inplace(installed)
                raise SystemExit(f"patched D3DMetal did not load the hook:\n{installed_log}")
            print(f"patched D3DMetal loads the hook: {installed}")
        values = {
            LOG_KEY: str(DIST / "hook.log"),
            ROOT_KEY: str(ROOT),
            REPLACE_KEY: "1",
            VERSION_KEY: args.fsr_version,
            FG_KEY: fg,
        }
        if mode == "inplace":
            values[FRAMEWORK_KEY] = str(installed)
        else:
            values[KEY] = str(binary)
        if fg == "anr":
            values[ANR_RUNTIME_KEY] = str(anr_runtime)
        if args.fsr_version in ("4.1.1", "4.1.2"):
            values[PROFILE_KEY] = args.profile
        if netal_source and netal_runtime:
            values[NETAL_RUNTIME_KEY] = str(netal_runtime)
        if args.panel:
            values[PANEL_KEY] = "1"   # status/sharpness panel in the game process
        # DLSS reaches MetalFX only with this D3DMetal switch. Not in OURS:
        # restore leaves it, as it may have been set before the install.
        values[METALFX_KEY] = "1"
        try:
            # Keep only the key of the mode actually used.
            conf_remove(bottle, [FRAMEWORK_KEY] if mode == "redirect" else [KEY])
            conf_set(bottle, values)
            select_panel_profile("fsr4" if args.fsr_version == "4.0.2" else args.profile)
        except BaseException:
            if installed and not other_bottles_using(installed, bottle):
                restore_inplace(installed)
            raise
    if installed:
        print("patched CrossOver's own D3DMetal; the original is kept as "
              f"{installed.name}{ORIGINAL_SUFFIX} next to it")
    else:
        if mode == "inplace":
            print("CrossOver 26 loads D3DMetal from the app; run install with a bottle to apply the patch")
        print("original CrossOver.app was not modified")
    restore = f"python3 {Path(__file__)} restore"
    if bottle:
        restore += " --bottle " + shlex.quote(str(bottle))
    print(f"restore: {restore}")


def cmd_restore(args):
    conf = resolve_bottle(args.bottle) if args.bottle else default_bottle(for_restore=True)
    binary = conf_get(conf, FRAMEWORK_KEY)
    if binary:
        installed = Path(binary).expanduser()
        users = other_bottles_using(installed, conf) if installed.is_file() else []
        if users:
            conf_unset(conf)
            print("CrossOver's D3DMetal is still used by: " + ", ".join(c.parent.name for c in users))
            print("run restore for those bottles before the framework can be put back")
            print("this bottle no longer enables FSR 4")
            return
        if installed.is_file() and restore_needed(installed):
            # Fail before dropping the keys, so restore can simply be run again.
            require_writable(installed.parent)
    conf_unset(conf)
    if not binary:
        print("CrossOver.app was not modified; next launch uses the bundled D3DMetal")
        return
    installed = Path(binary).expanduser()
    if not installed.is_file():
        print(f"patched D3DMetal not found: {installed}")
    else:
        try:
            restore_inplace(installed)
        except OSError as error:
            print(f"warning: could not restore {installed}: {error}")
            print(f"the original is kept as {installed.name}{ORIGINAL_SUFFIX}; "
                  "grant App Management access and run restore again")
    print("next launch uses the bundled D3DMetal")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    i = sub.add_parser("install")
    i.add_argument("--bottle", default="",
                   help="bottle name (e.g. Steam) or path to the bottle / its cxbottle.conf "
                        "(default: the only bottle; required when there are several)")
    i.add_argument("--no-bottle", action="store_true")
    i.add_argument("--crossover", default="",
                   help="CrossOver.app to use: copied for 24/25, patched in place for 26+, "
                        "any name or folder (default: first CrossOver found in /Applications or ~/Applications)")
    i.add_argument("--fsr-version", choices=("4.0.2", "4.1.1", "4.1.2"), default="4.1.2",
                   help="FSR model version for the live bridge and installation test (default: 4.1.2)")
    i.add_argument("--profile", choices=PROFILES, default="int8-lite",
                   help="initial panel mode: FSR variants or experimental Netal DLSS SR")
    i.add_argument("--netal-source", default="", help="pinned local Netal checkout to link SR (no upstream hook)")
    i.add_argument("--netal-runtime", default="", help="complete local netal.sr.runtime.v2 Preset K bundle; no model download")
    i.add_argument("--frame-generation", choices=("metalfx", "anr"), default="metalfx",
                   help="replace the MetalFX FG used for DLSS FG with experimental ANR")
    i.add_argument("--anr-runtime", default="", help="prepared ANR Metal runtime or resolution-profile directory")
    i.add_argument("--panel", action="store_true",
                   help="also set FSR4_PANEL=1 in the bottle (status/sharpness panel)")
    i.set_defaults(func=cmd_install)
    r = sub.add_parser("restore")
    r.add_argument("--bottle", default="",
                   help="bottle name or path, as given to install "
                        "(default: the only bottle with fsr4-mac keys)")
    r.set_defaults(func=cmd_restore)
    fg = sub.add_parser("frame-generation", help="switch FG without changing the upscaler profile")
    fg.add_argument("mode", choices=("anr", "metalfx"))
    fg.add_argument("--anr-runtime", default="", help="prepared ANR runtime or profile directory")
    fg.add_argument("--bottle", default="", help="bottle name or full cxbottle.conf path")
    fg.add_argument("--crossover", default="", help="CrossOver app to copy D3DMetal from")
    fg.add_argument("--netal-source", default="", help="also link the pinned Netal SR source, without changing the current SR profile")
    fg.add_argument("--netal-runtime", default="", help="optional complete local Netal SR runtime; no automatic model download")
    fg.set_defaults(func=cmd_frame_generation)
    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
