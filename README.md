# fsr4-mac

AMD FSR 4.1.2 INT8 Lite upscaling running natively on Apple Silicon (Metal).
Works as a standalone tool and inside games via CrossOver (DLSS → FSR 4).

## M5 performance: 1080p and 1440p (v3.0)

**Fresh 2026-11-02 measurements on a fanless MacBook Air M5, 16 GB.**
Whole-upscaler GPU milliseconds, **not game FPS**. The value is the median
of 12 round means; parentheses show the observed range of those means.

| Mode | 1080p GPU ms | 1440p GPU ms |
| --- | ---: | ---: |
| FSR 4.1.2 Exact (`exact-fast`) | 4.52 (4.10-5.03) | 8.72 (8.42-8.93) |
| FSR 4 (4.0.2, optional data) | 5.48 (5.01-6.13) | 10.47 (10.13-10.77) |
| FSR 4.1.2 Lossy (`lossy2`) | 3.07 (2.75-3.35) | 5.73 (5.50-5.96) |
| **FSR 4.1.2 Lite (default)** | 2.78 (2.52-3.13) | 5.37 (5.20-5.67) |
| Lite + Pixelization fix (0.5) | 2.82 (2.52-3.15) | 5.35 (5.13-5.57) |
| FSR 4.1.2 Lite Lossy (optional data) | 2.96 (2.69-3.23) | 5.84 (5.60-5.94) |

### Other command-line variants

| Mode | 1080p GPU ms | 1440p GPU ms |
| --- | ---: | ---: |
| Exact IEEE reference (`exact`, CLI) | 5.69 (5.19-6.42) | 11.03 (10.75-11.43) |
| Legacy Lossy (`lossy`, retired from panel) | 3.03 (2.72-3.32) | 5.82 (5.59-5.96) |
| Lossy + `--temporal-quality 1` | 4.62 (4.20-5.04) | 8.93 (8.62-9.15) |
| Lossy + `--temporal-quality 2` | 4.14 (3.78-4.70) | 7.98 (7.76-8.27) |
| Lite Lossy Tiled (512, CLI-only) | 3.73 (3.48-4.03) | 7.02 (6.64-7.20) |
| Lite Lossy Original (CLI-only) | 8.64 (7.92-9.62) | 10.87 (10.44-11.28) |

Input/output: **1280x720 -> 1920x1080** and **1706x960 -> 2560x1440**.
Native INT8 MPP, auto-exposure on, RCAS off, identical deterministic synthetic
motion/noise inputs. Reuse-eligible modes use 50:50 full/reused frames;
tiled updates half the tiles on every measured frame.

**Heat is included:** 120 s initial warmup per resolution, 30 s on repeat,
resolution order 1080p/1440p/1440p/1080p; all models interleaved per frame.
Each model/resolution has **768 measured frames**. All measured rounds had
thermal pressure **fair**. The later 1080p Lite series was about **20% slower**
than the earlier already-warmed series, so the table retains both and shows
the range instead of advertising the cold/best result. Thermal-pressure
state is not a temperature/frequency measurement.

Lite Lossy is **not automatically faster** than Lite; quality and speed depend
on the scene. These are observations on one synthetic fixture, not a quality
ranking or guaranteed game timings. CLI-only variants do not appear as panel
choices. [Method, full/reuse costs and thermal history](docs/FSR_V3_BENCHMARKS.md)
| [Numeric report](docs/reports/m5-modes-v3-2026-11-02.json).

Older recorded-capture measurements remain in
[FSR 4.1.2 stability](docs/FSR412_STABILITY.md); they are not the table above.

**Русская инструкция:** [docs/GUIDE.md](docs/GUIDE.md)

## Requirements

- Mac with Apple Silicon (fastest on M5; M1–M4 use the `simd` half-matrix backend, not benchmarked on those chips yet, see [hardware](docs/HARDWARE.md))
- macOS 26+, Xcode 26+ with Metal Toolchain, Python 3 (the macOS 27 SDK and
  Xcode 27 are supported)
- For games: CrossOver 24 or newer (CrossOver 26 is supported) and Rosetta 2

## Easiest: the FSR4 Installer app

1. Download **FSR4-Installer-v3.0.dmg** from the
   [latest release](https://github.com/UzenUPoZiTiV4ik/fsr4-native-macos/releases/latest)
   and drag **FSR4 Installer** to Applications.
2. First launch: the app is not notarized, so macOS blocks it. Open
   **System Settings → Privacy & Security** and press **Open Anyway**
   (or run `xattr -dr com.apple.quarantine "/Applications/FSR4 Installer.app"`).
3. The app checks Xcode, the Metal Toolchain, Rosetta and CrossOver, and
   offers a button for anything missing. CrossOver is found in /Applications
   or ~/Applications under any name; if it is elsewhere, press **Choose…**.
4. Quit the game and its bottle, pick the bottle, choose a mode and press
   **Install**. It takes about a minute. Then start the game and turn on **DLSS**.

![FSR4 Installer](docs/images/installer.png)

The app keeps its files in `~/Library/Application Support/fsr4-native-macos`.
On CrossOver 24/25 it never modifies `CrossOver.app`: the bottle is pointed at
a patched copy of D3DMetal. CrossOver 26 no longer reads that redirect
variable, so there the app patches `D3DMetal.framework/Versions/A/D3DMetal`
inside CrossOver itself, keeping the original as `D3DMetal.fsr4-original`;
only that binary and `libfsr4hook.dylib` change, and macOS may ask for the
**App Management** permission. **Remove…** puts the bottle back on CrossOver's
own D3DMetal and restores the original binary. Step-by-step guide:
[docs/GUIDE.md](docs/GUIDE.md) (Russian),
[tools/crossover/INSTALL.md](tools/crossover/INSTALL.md) (English).

## Quick start from source: play a game in CrossOver

```sh
xcodebuild -downloadComponent MetalToolchain
make -B
python3 tools/fsr412_import_model.py --dll /path/to/amdvkffx64.dll --with-shaders
python3 tools/crossover/crossover_fsr4.py install --panel --fsr-version 4.1.2 --profile int8-lite
```

The model import is needed for a clean source checkout; release ZIP/DMG payloads already contain the verified model.
Close the game and its bottle first. Then launch the game and turn on **DLSS**.
The install prints `D3DMetal: … (redirect)` on CrossOver 24/25 or `(in-place)`
on 26; on 26 it patches CrossOver's own D3DMetal (the app is changed only
while installed) and the restore command below puts the original back.
The **FSR 4** panel switches processing modes with a discrete
**Quality -> Performance** slider. It applies the chosen mode after you release
the thumb, not every intermediate mode you drag across.

![FSR4 Panel](docs/images/panel.png)

This is **not a resolution/preset control**: the game's render and output
resolution stay unchanged. Switching may cause stutters or pauses while
shaders are prepared and temporal history resets. Try different modes and
choose the best image-quality/performance balance for your game. The axis is
a suggested trade-off, not a measured quality ranking or a speed guarantee.

Exact, Lossy, Lite (default) and Lite Lossy are the current stops; optional
models appear only when their required data is installed. Pixelization fix
stays a separate checkbox/strength slider for Lite. **Lossy Tiled is no
longer offered in the panel**, but its CLI profile remains available. A saved
unavailable or retired mode is kept until you explicitly choose another one.

Undo the bottle change (on CrossOver 26 this also restores the original
D3DMetal; if another bottle uses the same patched framework, restore asks you
to undo that one first):

```sh
python3 tools/crossover/crossover_fsr4.py restore --bottle "/path/to/cxbottle.conf"
```

More: [CrossOver details](docs/CROSSOVER.md).

FSR 4.1.2 INT8 Lite is imported from the supplied AMD driver DLL with
`python3 tools/fsr412_import_model.py --dll /path/to/amdvkffx64.dll --with-shaders`.
The importer validates the DLL and 27,392-byte Lite initializers without
loading the Windows binary. The 4.1.2 graph uses an isolated `fsr412_` cache.
Its optimized output is byte-identical to the DLL's Lite shaders translated
to Metal ([comparison](docs/FSR412_DLL_PARITY.md)); the Windows DLL itself is
not executed.

## Standalone run on test frames

```sh
tools/bin/gen_testcase testcases/quality411 quality 1280 720 1920 1080 8
tools/bin/fsr4mac testcases/quality411 out/lite --fsr-version 4.1.2 --profile int8-lite
tools/bin/fsr4mac testcases/quality411 out/bench --fsr-version 4.1.2 --profile int8-lite --bench 10
```

Output: PNG previews, RGBA16F `.bin` files and `run.json`.
FSR 4.1.2 uses `int8-lite` (default). FSR 4.1.2 Lossy retains its internal
`lossy2` name (six trained layers, see
[docs/FSR411_LOSSY2.md](docs/FSR411_LOSSY2.md)). Use `--fsr-version 4.1.1`
for `lossy2`, `exact-fast`, `exact` (bit-exact reference) or the retired `lossy` profile.
`make lossy2-weights` downloads the pinned upstream release once and decodes
its trained weights into `build/`; they are not distributed with this MIT repository.

## Optional Lite Lossy and temporal repairs

`make lite-lossy-weights` statically imports the pinned upstream Lite Lossy
weights into `build/` (GPL-trained data, not included in the MIT release).
Use `--fsr-version 4.1.2 --profile int8-lite-lossy`; the panel also offers it
once the verified inputs are present. `int8-lite-lossy-tiled` is experimental
and CLI-only; it is not offered by the panel.
`--fsr-version 4.1.1 --profile lossy2 --temporal-quality 1` enables fine motion
and persistent-change repairs; `2` uses optional fast Wave32 notes.
Defaults are unchanged. [Implementation, tradeoffs, timings and tests](docs/FSR412_LOSSY.md).

`--profile int8-lite-lossy-original` is a different, opt-in route: it runs the
release's own PRE/BORDER/POST shaders from a verified external bundle
(`tools/fsr412/compile_original.py`) with exact native INT8 neural stages, and
matches the original-shader oracle byte for byte. It is slower than the adjusted
Lite Lossy path and is not a default.

## Tests

```sh
make test                 # includes the installer app tests
make test-fsr4-panel-ui    # native AppKit clicks/layout; requires a GUI session
make build-fsr411-tests
make test-fsr411-gpu
make test-fsr412-optimization
make test-fsr412-lossy      # normal/partial, native/scalar/SIMD, Metal 3/4
make test-fsr412-original   # opt-in original-graph route versus the original-shader oracle
make test-fsr411-mpp-fast  # GPU-probed M5 kernels and forced fallback
make test-fsr-pre-m5      # forced SIMD and exact Winograd
```

Build the installer app and DMG: `make installer INSTALLER_VERSION=v3.0`
(or `python3 tools/installer/build_app.py v3.0 --dmg`).

## Documentation

| Topic | Link |
| --- | --- |
| Performance and validation | [docs/FSR411_PERFORMANCE.md](docs/FSR411_PERFORMANCE.md) |
| Lite image quality update | [docs/FSR411_QUALITY_UPDATE.md](docs/FSR411_QUALITY_UPDATE.md) |
| FSR 4.1.2 Lossy (formerly Lite 2) | [docs/FSR411_LOSSY2.md](docs/FSR411_LOSSY2.md) |
| Lite Lossy, tiled inference and new temporal repairs | [docs/FSR412_LOSSY.md](docs/FSR412_LOSSY.md) |
| Original-shader oracle, geometry and POST CNB | [docs/FSR412_UPSTREAM_ORACLE.md](docs/FSR412_UPSTREAM_ORACLE.md) |
| v3.0 release notes | [docs/releases/v3.0.md](docs/releases/v3.0.md) |
| Heat-aware 1080p/1440p model measurements | [docs/FSR_V3_BENCHMARKS.md](docs/FSR_V3_BENCHMARKS.md) |
| M5 INT8 register-chain optimization | [docs/FSR_M5_INT8_OPTIMIZATION.md](docs/FSR_M5_INT8_OPTIMIZATION.md) |
| Pre-M5 SIMD optimization and exact Winograd | [docs/FSR_PRE_M5_OPTIMIZATION.md](docs/FSR_PRE_M5_OPTIMIZATION.md) |
| Latest memory optimization and timing limits | [docs/FSR411_MEMORY_OPTIMIZATION.md](docs/FSR411_MEMORY_OPTIMIZATION.md) |
| Benchmark method | [BENCHMARK.md](BENCHMARK.md) |
| Input format | [docs/FSR411_INPUT_CONTRACT.md](docs/FSR411_INPUT_CONTRACT.md) |
| GPU profiling | [docs/FSR411_PROFILING.md](docs/FSR411_PROFILING.md) |
| Current status and limits | [docs/STATE.md](docs/STATE.md) |
| Model sources | [third_party/fsr411/README.md](third_party/fsr411/README.md) |
| Old FSR 4.0.2 model | [architecture](docs/FSR402_ARCHITECTURE.md), [guide](docs/GUIDE_FSR402.md) |

The Lossy network keeps the numerical policy of the
[RX 7000 optimization project](https://github.com/zgauthier2000/fsr4-shader-performance-overrides)
(release `dll-2026-10-07`), implemented independently on MIT sources. Selected
temporal fixes through October 14, the October 15 dark-band guard and the
October 29 depth-edge clamp retain thin detail and update effects without motion vectors. A separate PRE for reused frames
removes unused network-input work; see the quality update for measurements.
