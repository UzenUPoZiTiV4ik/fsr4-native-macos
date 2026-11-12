# Experimental ANR frame generation on Metal

## Current result

The ANR `FG-V.B14` macOS test kit was reverse-engineered far enough to build
an **independent native Metal runtime**. It does not load or launch `bin/check`
in a game. Model inference runs on the game's own Metal command buffer.

`hook/anr_metalfx.mm` intercepts the **frame interpolator** D3DMetal creates
for DLSS FG. It is separate from the temporal-upscaling hook:

```text
Game DLSS FG -> D3DMetal -> MetalFX FG encode -> our ANR runtime -> game output
```

No extra presentation or fake FPS counter is introduced. D3DMetal continues
to submit commands and present frames. This seam supports **2x FG**: one
intermediate frame per real-frame pair. The standalone runner also tests
other interpolation phases; that is not a claim of in-game 4x/6x support.

**Verified here:** model replay, reference quality gates, real MetalFX
objects on Metal 3 and Metal 4, x86_64/Rosetta, opt-out/resume, reset and
three frames in flight, plus stationary-image readback.

**Not yet verified:** a running CrossOver game's camera/color conventions,
visual artifacts, UI behavior or presentation/frame pacing. This is an
experimental integration, not a universal game-support claim.

## Source of the model and shaders

Release: https://github.com/apex-compute/ANR/releases/tag/FG-V.B14

The downloaded macOS archive matches GitHub's asset SHA-256:

```text
0e2abcadf32bf3588199908dff23765fb7693cf74e6e2f8d5562867a3512dc50
```

Its `NRX1` containers contain:

- `program.json`, including dispatch order, buffer shapes, bindings and uniforms;
- `weights.bin`: 43,520 bytes in the selected Metal program;
- seven native Metal libraries and 21 DXIL shaders used by the full selected
  graph, plus reflection metadata;
- test input pairs and reference images.

The NRX loader was reconstructed from the arm64 `bin/check`, SHA-256
`64aa5480521e6bdfff499b759c6f2732eed4a12a3f88953244a25272fce0285d`,
transform at `0x100040954` and container parser at `0x10003f6b8`.
Packing is a SplitMix64-derived XOR stream over a Zstd frame, not missing
remote model data. REA/Ghidra static evidence for the transform was retained
as `ev_ca098a0ffdb44ce9bdf2ff1bc16a3bd31ea444f0631eac67618d96129a382b15`.
Addresses/evidence are build-specific, and recovered code is not original
source. The diagnostic extractor never executes the kit.

Native shader reflection reveals `_nr_root` at Metal buffer index 2,
pointing to a table of 24-byte resource entries. The `nativeMetal.index`
JSON field is **not** a direct Metal buffer binding. Treating it as one
produced a wrong image; the independent runtime now follows the actual
compiled ABI and verifies output images, not just dispatch counters.

The DXIL shaders are converted locally using the Apple-signed
`libmetalirconverter.dylib` included in the kit. No modified Apple library,
Windows DLL execution, kernel source reconstruction or NVIDIA model is
required for ANR.

## Prepare local runtimes

Supply an extracted kit and an Apple Metal Shader Converter header directory.
The headers can come from an existing SDK installation. This project does not
install the SDK or silently download it. The conversion tool uses the C API
in `metal_irconverter/metal_irconverter.h`; the locally tested headers were
Apple's 1.1 declarations, also retained in
[MethanePowered/MetalShaderConverterBinary](https://github.com/MethanePowered/MetalShaderConverterBinary/tree/cfed9ebffb64e1d784853edb8cdeea32456a75a0/include).
Headers were used externally and are not vendored here.

```sh
python3 tools/anr/prepare.py --kit /path/to/anr-fg-V.B14-kit-macos \
  --converter-headers /path/to/include --size 1080p \
  --output build/anr_runtime/1920x1080
```

Repeat with `--size 1440p` / `--output build/anr_runtime/2560x1440`, or
`2160p` / `build/anr_runtime/3840x2160`. `--backend packed` prepares an
alternative packed-FP16 graph from the same pinned weights. It uses three
native and 25 converted kernels. Keep it in a separate runtime directory;
run the same verification before choosing it for a game. Existing outputs are refused and a
failed preparation does not expose a half-built runtime. Original kit
files are not modified. The selected packs, weights and converter have
pinned identities, and each generated runtime has an asset hash inventory.

The complete generated runtime is about 3 MB per resolution. The large
original archive consists mostly of test inputs/reference images, not model
weights. All prepared/proprietary assets stay under ignored `build/`.

## Standalone model verification

```sh
python3 tools/anr/verify.py --runtime build/anr_runtime/1920x1080 \
  --kit /path/to/anr-fg-V.B14-kit-macos --size 1080p \
  --frames 64 --out build/anr-verification-new
```

By default this runs only our native executable. `--compare-testkit`
explicitly also invokes the supplied upstream `bin/check` for a comparison.
The suite checks `t=0.5` and `t=0.75` against the release's reference-quality
thresholds. A second phase alone is not a full multi-frame presentation test.

Observed reference PSNR on M5:

| Resolution | `t=0.5` | `t=0.75` | Both kit gates passed |
| --- | ---: | ---: | --- |
| 1920x1080 | 87.87 dB | 89.18 dB | yes |
| 2560x1440 | 88.37 dB | 89.26 dB | yes |
| 3840x2160 | 88.53 dB | 90.12 dB | yes |

Outputs are **not byte-identical**. Even repeated kit runs have small
variations. These checks establish scoped inference/format regression,
not perceptual quality in arbitrary games. ANR itself labels the checkpoint
as trained, with quality not validated.

## Optimizations and timings

- Grouped compute encoders reduce encoder creation and global barrier cost;
  memory barriers preserve pass ordering. Set `FSR4_ANR_COALESCED=0` for
  the separate-encoder A/B route.
- Only 2x phase pipelines and tables are loaded in the game bridge.
- Identical pipelines are cached by shader hash, entry point and GPU device.
  Model/history state is never shared between scalers.
- Four independent CPU-write slots retain constants, resource tables,
  textures and fences until their GPU completion. The shared neural state
  is ordered on the caller's queue; no CPU wait is added to the game.
- Previous/current buffers and recurrent histories switch via CPU bindings,
  without full-frame copies.
- GPU-ordered history clearing makes resets safe with queued work.
- Native PQ10/full-resolution input takes a zero-conversion path directly
  into the model and directly back to the game's output texture.
- Failure initialization is briefly cached to avoid recompiling or hashing
  a broken runtime on every encode.

A calibrated Metal 4 timestamp A/B on M5 measured approximately 4.34 ms
(separate encoders) vs 4.07 ms (grouped) for a 1080p pair plus one intermediate
frame, eight warmups and 64 measured fixed-pair iterations. A later 1080p run
measured 4.069 ms, split into roughly 2.227 ms pair work and 1.839 ms phase work.
These are standalone GPU durations, **not game FPS or a guaranteed ceiling**.
Thermal load and other work vary timings; later short runs included 5.27 ms.

The optional packed graph passed the same reference gates and MetalFX
PQ10/HDR, Metal 3/4 and queued-frame tests. Measured M5 inference for
`t=0.5`: about **3.83 ms at 1080p, 6.91 ms at 1440p, 14.78 ms at 4K**.
Its PSNR was respectively 83.90, 84.21 and 84.34 dB, so it trades a little
numerical precision for speed without failing the test-kit thresholds.
These are separate replay runs, not guaranteed game timings.

At 4K the tested native complete inference was about 16 ms on M5. Increasing display
resolution has a real cost. Import/export color/depth conversions add work
in games whose textures do not already match the model's input.

Raw GPU counter ticks are calibrated with `sampleTimestamps` and the CPU's
Mach timebase. They are not assumed to be nanoseconds: the measured M5 tick
was about 41.667 ns. Host wall time is reported separately.

## Test the real MetalFX seam

```sh
make test-anr-live ANR_RUNTIME="$PWD/build/anr_runtime/1920x1080"
make test-anr-input
python3 -m unittest discover -s tests -p 'test_anr.py'
```

The live test creates real MetalFX interpolators, not fake Objective-C classes.
It verifies first-frame priming, replacement, disable/resume, three queued
frames and a stationary generated image. Classic Metal and Metal 4 both gave
about 0.022 RGB10 code steps of average stationary-image error. The HDR
conversion route also passed, at about 1.22 normalized RGB10 steps average
error on the synthetic linear float scene. That quantization/conversion
error is not an HDR visual-quality guarantee for games. A source frame
is still needed for a proper camera-motion/game quality evaluation.

## Install just FG into CrossOver

Close the game and bottle first. This changes the selected bottle's framework
and FG environment keys and backs up its config. The original CrossOver app
is not modified. Existing SR profile/sharpness preferences are preserved.

```sh
python3 tools/crossover/crossover_fsr4.py frame-generation anr \
  --anr-runtime "$PWD/build/anr_runtime" --bottle "Games"
```

Use the actual bottle name or full `cxbottle.conf` path. The implementation
builds the combined hook, copies and patches D3DMetal, runs a Rosetta MetalFX
FG test and only then updates the selected bottle. Set the game to **DLSS
frame generation** (the NVIDIA option), so D3DMetal reaches the replaced seam.

To use ANR during the regular SR installer, add
`--frame-generation anr --anr-runtime /path/to/runtime`.

Return FG alone to MetalFX without changing the SR mode:

```sh
python3 tools/crossover/crossover_fsr4.py frame-generation metalfx --bottle "Games"
```

The regular `restore` also removes the ANR keys and restores the framework
route to CrossOver's bundled D3DMetal.

Status is separate from SR: `dist/crossover/fg-status.conf` reports active
ANR, generated/primed/fallback counts and the exact fallback reason. UI
mode/counts and input pixel formats are recorded separately. The panel
shows `HUD recomposited` or `UI overlay` for the active mode. `hook.log`
includes `[anr-fg]` diagnostics. Synthetic checks disable receipt publishing
so they cannot impersonate a running game in the panel.

## UI recomposition

MetalFX `uiTextureComposited` describes the optional **UI texture**, not
whether `colorTexture` already includes a HUD. With the flag set, UI is the
current clean scene plus the HUD; without it, UI is a premultiplied RGBA
overlay. A flag without a UI resource is ignored.

ANR receives only the clean color/depth/motion data. UI is applied once in
the export pass, at the current screen coordinates, without adding it to
temporal history. Separate UI uses premultiplied alpha. For a precomposited
resource, the current clean RGB is removed algebraically and the overlay
residual is applied to the neural result, using a bounded-RGB minimum-opacity
estimate and a smooth contrast gate. Small differences have zero opacity,
so full-screen rounding, dither or grain does not switch world geometry
from the midpoint to the current real frame. High-contrast UI is boosted
continuously; there is no binary changed-pixel phase switch. Alpha of a
precomposited texture is not an opacity mask and is ignored.

The noise floor is 2.5 code steps for 8-bit/PQ10 resources and a relative
1e-4 for floating-point inputs. A 1/32-grid opacity sample is collected at
most twice per second. `ui_raw_changed`, `ui_soft`, `ui_opaque` and
`ui_opacity_mean` distinguish the old mask footprint from the new estimated
opacity without a heavy video capture. Only completed GPU slots are read.

Two opaque RGB images cannot uniquely determine the original UI opacity.
This is approximate recomposition, not a claim to match internal MetalFX
decomposition or every translucent/low-contrast UI case.

The comparison requires matching formats/read-space, with BGRA/RGBA
8-bit twins accepted. Unsupported formats, extents, devices, unreadable UI
or UI/output aliasing still use MetalFX. The direct PQ10 fast path is used
only without UI; recomposition has its own generated input texture and
retains UI in each of the four in-flight frame slots. The helper binary
uses a versioned name so an old running hook cannot load the new ABI.

`make test-anr-ui` validates Metal3/4 export over PQ10, linear HDR,
hardware sRGB and 8-bit SDR (32 cases total). Its generated scene deliberately
differs from both real inputs. Clean/noisy composites, near-black one-bit
dither, alpha/NaN controls, immutable inputs, sampled statistics and
out-of-range dispatch canaries are checked. The real MetalFX seam
test also checks 8,192 exact HUD pixels, flag-only input, both UI modes,
three queued changing modes and rejection of unsafe UI resources.
These are synthetic checks, not Spider-Man gameplay/visual acceptance.

## Camera-shake investigation, 2026-11-12

The real game reached 1920x1080 and recorded 5,197 ANR encodes with no
fallbacks. The user reported shake before pressing diagnostics. Eight
SR-only captures contain 64 frames; seven moving sequences favor the
declared signed MV mapping. A stationary sequence retains a stable SR
output despite varying render jitter. No generated ANR image or presentation
timeline exists in those original captures.

A concrete UI fault was reproduced: one code step of full-screen noise
made the old binary mask copy the current real frame, e.g. 0.130987 instead
of the generated-world value 0.430978. UI v2 removes that phase switch.
This is a verified shader defect and a suspected contributor to the reported
game shake, not proof that every in-game artifact has been resolved.

`make test-anr-motion ANR_RUNTIME=/path/to/1920x1080` adds a real MetalFX
seam regression: constant 24-pixel motion, five consecutive midpoints,
full-size/1280x720 motion inputs, and varying jitter. ANR matches the
12/36/60/84/108-pixel positions on Metal3 and Metal4. The jitter-correction
fields are gated by ANR flag 16, which this bridge does not set. D3DMetal
passes NGX motion scale through its wrapper/replayer; no preliminary scale
operation was observed there. Neither hypothesis justified changing game
motion settings. Native static evidence is retained locally in
`build/anr_diagnostics/d3dmetal-native-evidence.json`.

## Remaining game limitations

- Only the three prepared exact output sizes are supported. A different size
  goes to original MetalFX, not an unsafe reinterpretation of the graph.
- No distortion or nonzero input/output offsets. UI recomposition is
  supported as documented above; unsupported UI resource layouts still
  fall back rather than dropping the HUD.
- Different input/output color formats are conservatively rejected.
- Raw RGB10A2 defaults to PQ10; 8-bit unorm defaults to SDR code values;
  float and sRGB hardware textures use a linear route. Metal does not report
  transfer functions, so verify the game's colorspace. Override with
  `FSR4_ANR_COLOR=linear|srgb|pq10` when known. Linear HDR is normalized using
  `FSR4_ANR_WHITE_NITS` (203 by default). This is not game-specific HDR proof.
- Finite near/far planes, known motion scaling and subpixel jitter are required.
  Motion follows MetalFX's previous-minus-current convention. Camera matrices
  are not fabricated: the model's explicit no-matrix path is selected, with
  camera motion already in the game's motion vectors.
- A greater than 500 ms gap primes history again rather than blending across
  a pause/menu discontinuity.
- Four slots are bounded. If the game queues more uncompleted frames, it falls
  back and primes the next accepted history instead of overwriting GPU input.
- Submission order must be one ordered queue per interpolator. This is the
  MetalFX contract used by the tested seam; arbitrary cross-queue use is not
  supported by this bridge.

## Rights and reproducibility

No ANR weights, kernel binaries, original kit executable, Apple converter or
NVIDIA DLL is committed or packaged. Extraction and execution use explicit
local inputs. ANR's release source for the model/runtime is still unavailable;
extractability does not automatically grant redistribution rights. Preserve
upstream and Apple terms and review them before publishing a binary bundle.
The code here is an independently implemented host/format bridge, not a
claim that ANR's model or compiled shaders are MIT-licensed.
