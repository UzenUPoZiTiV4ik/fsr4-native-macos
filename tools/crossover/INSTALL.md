# FSR 4.1 in CrossOver: install guide

Replaces DLSS in Windows games running under CrossOver with AMD FSR 4.1
(4.1.1 INT8 model on Metal). The game keeps its DLSS option; the hook swaps
the MetalFX upscaler that D3DMetal uses for DLSS. Русская инструкция:
[docs/GUIDE.md](docs/GUIDE.md).

## Requirements

- Apple Silicon Mac, macOS 26 or newer. Tested on M5; M1–M4 use a
  half-matrix (`simd`) path with identical output, not benchmarked on
  those chips yet.
- CrossOver 24+ (CrossOver.app or CrossOver Preview.app in /Applications)
  with a bottle where the game runs with D3DMetal and DLSS enabled.
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

Replace `Steam` with your bottle's name (as shown in CrossOver). A wrong name
prints the list of bottles. The installer:

- builds the hook and the FSR 4 panel app;
- copies D3DMetal.framework from CrossOver into `dist/crossover/` and patches
  only that copy (CrossOver.app itself is not modified);
- runs the copy on a test sequence and checks the result against the CLI;
- adds `D3DMETAL_FRAMEWORK_PATH`, `D3DM_ENABLE_METALFX=1` (DLSS via
  MetalFX) and a few `FSR4_*` keys to the bottle's `cxbottle.conf`, backing
  it up to `dist/crossover/backups/` first.

The first run takes a few minutes. The first game launch also compiles
shaders for the game's resolution (tens of seconds); later launches use the
cache in `build/`.

## Use

Start the game and select **DLSS** in its graphics settings. The **FSR 4**
panel opens next to the game:

- **FSR 4.1**: the full network every frame, best quality.
- **FSR 4.1 Lite**: reuses network output on suitable frames, about 30%
  less GPU time.
- **Sharpness** and **RCAS** apply immediately.

Status "FSR 4.1.1 active" means the game goes through FSR. If it says
inactive, the reason is shown in the panel and in `dist/crossover/hook.log`.
The panel can be reopened with `open "dist/crossover/FSR4 Panel.app"`.

Without the panel: drop `--panel`, and pick the mode with `--profile
exact-fast` (FSR 4.1) or `--profile lossy` (FSR 4.1 Lite, default).

## Update or remove

Update: quit the game, unpack the new version into the same folder and run
the install command again.

Remove from the bottle (the folder can be deleted afterwards):

```sh
python3 tools/crossover/crossover_fsr4.py restore --bottle "Steam"
```

## Troubleshooting

- `D3DMetal.framework not found`: pass `--crossover "/Applications/CrossOver.app"`.
- `xcrun: error` or `metal: not found`: install the Metal Toolchain (above).
- Panel shows "no data": the game hasn't used DLSS yet, or the bottle wasn't
  restarted after install.
- Problems with ghosting or smearing: press **Record diagnostics** in the
  panel, move the camera, and attach `dist/crossover/captures/<folder>` and
  `hook.log` to a GitHub issue.
