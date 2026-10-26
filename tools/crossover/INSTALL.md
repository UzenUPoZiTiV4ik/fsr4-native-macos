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
- copies D3DMetal.framework from CrossOver into `dist/crossover/` and patches
  only that copy (CrossOver.app itself is not modified);
- runs the copy on a test sequence and checks the result against the CLI;
- adds `D3DMETAL_FRAMEWORK_PATH`, `D3DM_ENABLE_METALFX=1` (DLSS via
  MetalFX) and a few `FSR4_*` keys to the bottle's `cxbottle.conf`, backing
  it up to `dist/crossover/backups/` first.

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
the install command again.

Remove from the bottle (the folder can be deleted afterwards):

```sh
python3 tools/crossover/crossover_fsr4.py restore --bottle "Steam"
```

## Troubleshooting

- `D3DMetal.framework not found`: pass the CrossOver you play with, e.g.
  `--crossover "$HOME/Applications/CrossOver_patched.app"`. In the app: **Choose…**.
- `xcrun: error` or `metal: not found`: install the Metal Toolchain (above).
- Panel shows "no data": the game hasn't used DLSS yet, or the bottle wasn't
  restarted after install.
- Problems with ghosting or smearing: press **Record diagnostics** in the
  panel, move the camera, and attach `dist/crossover/captures/<folder>` and
  `hook.log` to a GitHub issue.
