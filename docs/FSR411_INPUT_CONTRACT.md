# FSR 4.1.1 input and recording contract

The live path is `DLSS/NGX → D3DMetal → MetalFX temporal scaler → FSR`.
The hook reads MetalFX values after D3DMetal has converted the NGX inputs.
`src/fsr4_input.h` is the shared, Metal-independent conversion and validation
layer used by both the live bridge and the standalone 4.1.1 runner.

## Coordinates, motion and exposure

| Input | Conversion |
| --- | --- |
| Motion vectors | Previous position minus current position. Multiply raw values by the supplied signed `motionVectorScale`, then divide by the logical motion-vector extent. |
| Low-resolution motion | Divide by current render-content width/height. |
| Output-resolution motion | Divide by logical output width/height, independently of allocation or the model's 8-pixel alignment. |
| Jitter | Pass the pixel sampling offset unchanged. For jittered motion, the cancellation term is `(previousJitter − currentJitter) / motionExtent`; the shader subtracts it. |
| Reversed depth | Select the reversed-depth shader convention. Do not also invert the texture numerically. |
| Pre-exposure | Divide input colour by finite, positive `preExposure`; keep previous pre-exposure per stream. |
| Exposure texture | Independently multiply by exposure. A missing texture means neutral one. Auto exposure ignores a supplied manual exposure texture. |

MetalFX's documented motion-vector sign already agrees with FSR. Applying an
additional DLSS sign flip at the hook would reverse motion. Zero or negative
motion scales are valid; texture allocation does not determine the scale.

The model's internal output, dispatch sizes, `width`, `height`, `scale` and
`inv_size` are aligned to eight pixels, following the SDK provider. Motion
normalization still uses the logical output extent. The public output texture
and its copied content region retain the requested logical dimensions.

## Content regions and temporal history

The V1 `Fsr4LiveFrame` layout is preserved. V2 adds an explicit version and
structure size, separate allocation sizes and offsets for colour, depth,
motion and output, and the descriptor's maximum input extent. Inputs with
valid subrects or larger backing allocations are copied into canonical
textures before normalized sampling. Output writes are restricted to the
requested output rectangle. Out-of-bounds regions are rejected.

Motion resolution and jitter flags come from the scaler descriptor. Unknown
flags use the documented MetalFX false defaults, recorded in the prepared
input structure. An unset 0×0 input-content size uses the scaler's descriptor
input size; one zero dimension is an error.

History belongs to a scaler and device. Input-only dynamic-resolution changes
preserve it. The first frame, explicit reset, a change of logical output size,
model/profile change, or a convention change resets it. Logical output changes
reset even when the old and new sizes share the same aligned allocation.
An encode that falls back to MetalFX invalidates FSR history. Scaler lifetime
tokens release their state without retaining the scaler itself; Metal 4
resources remain retained until in-flight frames retire.

## Standalone recordings

The runner uses the basic `meta.json` and `fNN_*.bin` layout in `TESTDATA.md`.
FSR 4.1.1 with the Lossy profile is the default in the CLI, live bridge and
CrossOver installer. `--fsr-version 4.0.2` (or `FSR4_VERSION=4.0.2` for the
live bridge) explicitly selects the legacy model and its FP16 backend.
The following commands select the new model explicitly:

```sh
tools/bin/fsr4mac --fsr-version 4.1.1 CASE OUT --i8mpp --profile lossy
tools/bin/fsr4mac --fsr-version 4.1.1 CASE EXACT --reference --profile exact --dump-passes
tools/bin/fsr4mac --fsr-version 4.1.1 CASE BENCH --i8mpp --profile lossy --bench 10
```

`--reference` selects the generic integer implementation of the 4.1.1 model.
`--i8mpp`/`--native` selects native Apple tensor operations; `--auto` chooses
from this device's capabilities. `--scalar` selects the portable integer
Metal implementation; Auto uses it on Apple GPUs without native tensor
support. It still runs the 4.1.1 model. The 4.1.1 runner rejects old FP16 backend
flags. Profile and backend are separate settings. Lossy is the default
profile for 4.1.1; use explicit Exact when checking precise arithmetic.

Colour is RGBA16F, depth R32F, and motion RG16F. With `low_res_mv:false`, the
motion file contains `upscale_width × upscale_height` pixels. Every file must
match its declared size. Per-frame objects may override render/output sizes,
pre-exposure, motion/depth/jitter flags and auto exposure. `frame` must contain
exactly `frames` entries when present. Missing manual exposure means one in
the new runner; `pre_exposure` remains an independent divisor. This differs
from the old 4.0.2 recording runner's implicit missing-exposure fallback.

`run.json` records the actual backend, shader variant, selected profile,
effective resets, logical/aligned dimensions and full command-buffer GPU
durations. `measured_frames`, `neural_runs` and `reused_frames` count measured
frames only, excluding benchmark warmup. Each frame records
`skipped_inference` and cumulative `sequence_neural_runs` /
`sequence_reused_frames`; those sequence counters restart at each benchmark
iteration. `--backend-info` only reports capability-based selection; it is
not evidence that shaders compiled or a frame ran.

`--dump-passes` captures scratch immediately after prepass (`pass00`), neural
passes 1–12, and postpass (`pass13`). `--capture-stages` selects individual
pre/post, pass1–12, or pad0–12 boundaries. `--dump-state` writes final scratch,
history, reprojected colour, recurrent state and exposure, with each texture's
actual dimensions and format in `fNN_state.json`. Diagnostic copies are
included in that run's GPU work and cannot be combined with `--bench`.
On a reused Lossy frame the neural and padding passes do not run, so their
scratch snapshots are absent. Each measurement's `captured_stages` lists the
boundaries actually captured for that frame.

`--bench N` performs one warmup sequence, then N measured sequences. It writes
`run.json`, with complete frame command-buffer timings, rather than images
or sums of separately timed stages. File loading, pipeline compilation and
readback are outside the GPU duration.

## CrossOver profile control and checks

The installer accepts `--fsr-version 4.1.1 --profile exact|exact-fast|lossy`. Its smoke
reference is generated from the same model/profile; a stale 4.0.2 output
cannot satisfy the check. The panel writes `profile=exact`, `profile=exact-fast` or `profile=lossy`
alongside existing sharpness and RCAS preferences. The live encoder polls
these settings; `FSR4_SETTINGS=none` disables file overrides. `FSR4_PROFILE`
sets the initial profile, with Lossy as the 4.1.1 default. Profile switches
take effect through a new session and reset temporal history.

```sh
make test-input test-settings
make build-fsr411-tests       # host harnesses and local hook; does not run GPU work
make test-fsr411-live         # CrossOver/Rosetta: classic + Metal 4, including profile switches
make test-fsr411-gpu          # complete 4.1.1 GPU suite, run sequentially
make test-fsr411-scalar       # full graph parity for the portable scalar backend
make test-fsr411-cli          # default model/profile, reuse counts, dumps and benchmark warmup
```

The complete GPU suite includes native/scalar blocks against an independent
CPU integer oracle, full graph comparisons with direct HLSL, Lossy motion
fixtures, cache scheduling and the live bridge. Native tests require Apple
GPU family 10 and macOS 26; the scalar graph target does not require native
tensor operations. `test-gpu` also runs the explicitly selected 4.0.2 legacy
fallback checks. `tools/precompile_models.py` continues to build 4.0.2
variants; the 4.1.1 runtime uses `tools/compile_fsr411.py` on demand.

`tests/fsr411_live_regions_test.mm` is a separate GPU harness linked against
the live hook. It generates odd-sized inputs and independent offsets for all
four textures, compares canonical and padded outputs, checks untouched border
pixels, and exercises invalidation, failed geometry and owner-address reuse.
It accepts a repository path, `reference|scalar|i8mpp`, and `classic|mtl4`;
it explicitly selects Exact so profile approximation does not obscure
geometry checks. Its optional `switch` argument adds Exact → Lossy → Exact
transitions and checks automatic profile resets against explicit resets.

The motion/alignment contract follows the SDK provider at
[`01446e6a74888bf349652fcf2cbf5f642d30c2bf`](https://github.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK/blob/01446e6a74888bf349652fcf2cbf5f642d30c2bf/Kits/FidelityFX/upscalers/fsr4/dx12/ffx_provider_fsr4_dx12.cpp#L1619)
and the installed Apple SDK's `MTLFXTemporalScaler.h` documentation. These
contracts and local Metal comparisons do not establish independent parity
with a Windows 4.1.1 GPU implementation.
