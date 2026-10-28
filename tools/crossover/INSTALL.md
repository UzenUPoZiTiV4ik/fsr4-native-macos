# FSR 4.1.2 in CrossOver: install guide

Replaces DLSS in Windows games running under CrossOver with AMD FSR 4.1.2
(INT8 Lite model on Metal). The game keeps its DLSS option; the hook swaps
the MetalFX upscaler that D3DMetal uses for DLSS. Русская инструкция:
[docs/GUIDE.md](docs/GUIDE.md).

**Easier:** the FSR4 Installer app (`FSR4-Installer-<version>.dmg` on the
[releases page](https://github.com/UzenUPoZiTiV4ik/fsr4-native-macos/releases/latest))
does everything below with buttons: checks Xcode, the Metal Toolchain,
Rosetta and CrossOver, lists the bottles, installs and removes. It is not
notarized: on first launch open **System Settings → Privacy & Security →
Open Anyway**, or run
`xattr -dr com.apple.quarantine "/Applications/FSR4 Installer.app"`.
The app keeps its copy in `~/Library/Application Support/fsr4-native-macos`.
This zip is the same thing for Terminal users.

## Requirements

- Apple Silicon Mac, macOS 26 or newer. Tested on M5; M1–M4 use a
  half-matrix (`simd`) path with identical output, not benchmarked on
  those chips yet.
- CrossOver 24+ in /Applications or ~/Applications (any name, e.g. a renamed
  copy; elsewhere: pass `--crossover`)
  with a bottle where the game runs with D3DMetal and DLSS enabled.
  CrossOver 26 is supported: see [CrossOver 26](#crossover-26-in-place-d3dmetal)
  below.
- Rosetta 2 (`softwareupdate --install-rosetta`).
- Xcode 26+ with the Metal Toolchain, and Python 3 (ships with Xcode).

```sh
xcode-select --install                       # if Xcode is not set up yet
xcodebuild -downloadComponent MetalToolchain
```

## Install

1. Unpack the zip to a permanent place, e.g. `~/fsr4-mac`. The bottle points
   at this folder, so don't move or delete it afterwards.
2. Quit the game and close every process of its bottle.
3. In Terminal:

```sh
cd ~/fsr4-mac
python3 tools/crossover/crossover_fsr4.py install --panel --bottle "Steam"
```

Replace `Steam` with your bottle's name (as shown in CrossOver). With a single
bottle `--bottle` can be omitted; with several the installer lists them and
asks you to pick one. A wrong name prints the list of bottles. The installer:

- builds the hook and the FSR 4 panel app;
- finds D3DMetal in CrossOver and prints the mode it uses,
  `D3DMetal: … (redirect)` for CrossOver 24/25 or `(in-place)` for 26+;
- CrossOver 24/25: copies D3DMetal.framework from CrossOver into
  `dist/crossover/` and patches only that copy (`CrossOver.app` itself is not
  modified);
- CrossOver 26: patches CrossOver's own `D3DMetal.framework`, keeping the
  original binary next to it (the app is changed only while installed; see
  below);
- runs the patched binary on a test sequence and checks the result against
  the CLI;
- adds `D3DMETAL_FRAMEWORK_PATH` (24/25) or `FSR4_FRAMEWORK` (26),
  `D3DM_ENABLE_METALFX=1` (DLSS via MetalFX) and a few `FSR4_*` keys to the
  bottle's `cxbottle.conf`, backing it up to `dist/crossover/backups/` first.

### CrossOver 26: in-place D3DMetal

CrossOver 26 keeps GPTK under `Contents/SharedSupport/CrossOver/lib64/apple_gptk/external/`
(older builds use `lib/`, so both are searched), and its `libd3dshared.dylib`
no longer reads `D3DMETAL_FRAMEWORK_PATH`. It loads
`@rpath/D3DMetal.framework/D3DMetal` with `LC_RPATH @loader_path`, so the
framework inside the app has to carry the hook. Install therefore:

- patches `D3DMetal.framework/Versions/A/D3DMetal` in place and copies
  `D3DMetal.fsr4-original` (the untouched original) next to it;
- copies `libfsr4hook.dylib` into the same `Versions/A/` directory;
- writes `FSR4_FRAMEWORK` (the patched binary) instead of
  `D3DMETAL_FRAMEWORK_PATH` into the bottle.

Only those two files change inside the app. The patched binary and the hook
keep their existing ad-hoc signatures; the installer deliberately does not
re-sign them, so restore can put the original bytes back and the framework
still verifies. While installed, `codesign --verify --deep --strict` on the
app reports the replaced binary — that is expected and goes away on restore.

The framework is shared by every bottle. The hook does nothing unless
`FSR4_ROOT` is set, which only bottles that installed FSR 4 have.

On macOS 26, writing inside `CrossOver.app` can require the App Management
permission. If install stops with `cannot write …/Versions/A`, allow
**Terminal** (or **FSR 4 Installer**) in **System Settings → Privacy &
Security → App Management**, quit CrossOver, and run install again.

Bottles on an external disk are discovered from CrossOver's `BottlePath`
setting or `CX_BOTTLE_PATH`. In the app, **Choose bottle folder…** also lets
you select any folder containing `cxbottle.conf`. In Terminal, use the full
path if the bottle is not configured in CrossOver's search path:

```sh
python3 tools/crossover/crossover_fsr4.py install --panel --bottle "/Volumes/SSD/Bottles/Steam"
```

The first run takes a few minutes. The first game launch also compiles
shaders for the game's resolution (tens of seconds); later launches use the
cache in `build/`.

## Use

Start the game and select **DLSS** in its graphics settings. The **FSR 4**
panel opens next to the game:

- **FSR 4.1**: the full network every frame, best quality.
- **FSR 4.1.2 Lite**: AMD's compact INT8 architecture, evaluated every frame; the default.
- **Pixelization fix** (shown with FSR 4.1.2 Lite): a wider current-frame
  filter where the network rejects history, so moving hair and thin detail
  are less blocky. Off by default. Its slider sets the strength (0 = as
  without the fix, 0.5 default, 1 = widest and softest) and applies at once.
- **FSR 4.1 Lossy**: formerly Lite 2, with six trained layers, shown once
  `make lossy2-weights` has fetched its weights; closer to FSR 4 on thin
  detail, a little slower to follow brightness changes.
- **Sharpness** and **RCAS** apply immediately.

The old Lite mode has been removed. Status "FSR 4.1.2 active" means the
game goes through the new FSR model. If it says
inactive, the reason is shown in the panel and in `dist/crossover/hook.log`.
The panel can be reopened with `open "dist/crossover/FSR4 Panel.app"`.

Without the panel: drop `--panel`. The default is `--fsr-version 4.1.2
--profile int8-lite`. For FSR 4.1 use `--fsr-version 4.1.1 --profile exact-fast`,
or `--fsr-version 4.1.1 --profile lossy2` for FSR 4.1 Lossy after `make lossy2-weights`.

## Update or remove

Update: quit the game, unpack the new version into the same folder and run
the install command again. On CrossOver 26 the install refreshes
`D3DMetal.fsr4-original` from the current framework if CrossOver was updated
in the meantime.

Remove from the bottle (the folder can be deleted afterwards):

```sh
python3 tools/crossover/crossover_fsr4.py restore --bottle "Steam"
```

Restore puts CrossOver's original `D3DMetal` back (CrossOver 26), or removes
the redirect keys (24/25), and prints the state. If another bottle still uses
the same patched framework, restore leaves it in place and says which bottles
to restore first — the last restored bottle puts the original back. Restore
before updating CrossOver: if CrossOver was already updated, restore removes
the stale backup and hook without overwriting the new framework.

Keys written into the bottle (`D3DM_ENABLE_METALFX` is left on restore in
case you set it yourself):

| Key | CrossOver | Value |
| --- | --- | --- |
| `D3DMETAL_FRAMEWORK_PATH` | 24/25 | `dist/crossover/D3DMetal.framework/Versions/A/D3DMetal` |
| `FSR4_FRAMEWORK` | 26+ | `<CrossOver.app>/…/D3DMetal.framework/Versions/A/D3DMetal` |
| `FSR4_HOOK_LOG`, `FSR4_ROOT`, `FSR4_REPLACE`, `FSR4_VERSION`, `FSR4_PROFILE`, `FSR4_PANEL` | both | as before |

## Troubleshooting

- `D3DMetal.framework not found`: pass the CrossOver you play with, e.g.
  `--crossover "$HOME/Applications/CrossOver_patched.app"`. In the app: **Choose…**.
- `cannot write …/D3DMetal.framework/Versions/A`: allow **Terminal** or
  **FSR 4 Installer** in **System Settings → Privacy & Security → App
  Management**, quit CrossOver, and retry.
- Restore says another bottle still uses the D3DMetal: restore that bottle
  first; the framework stays patched until the last bottle is restored.
- `xcrun: error` or `metal: not found`: install the Metal Toolchain (above).
- Panel shows "no data": the game hasn't used DLSS yet, or the bottle wasn't
  restarted after install.
- Problems with ghosting or smearing: press **Record diagnostics** in the
  panel, move the camera, and attach `dist/crossover/captures/<folder>` and
  `hook.log` to a GitHub issue.
