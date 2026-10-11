# fsr4-mac — FSR 4.1.1 on Apple Metal

Native macOS port of **FSR 4.1.1.2740 INT8**, with switchable **Exact**,
**Exact Fast** and **Lossy** profiles. The standalone runner and CrossOver/MetalFX bridge share
one runtime. **4.1.1 + Lossy are the defaults.**

On Apple GPU family 10 (M5) the neural passes use native integer tensor
operations. Other Apple GPUs use portable integer Metal kernels with the same
model. Physical driver and performance validation is currently on M5; see
[hardware support](docs/HARDWARE.md).

![FSR 4.0.2 vs FSR 4.1 on Apple M5](docs/images/fsr4-vs-fsr41-m5.png)

**Инструкция на русском:** [docs/GUIDE.md](docs/GUIDE.md).
**Input and recording contract:** [docs/FSR411_INPUT_CONTRACT.md](docs/FSR411_INPUT_CONTRACT.md).
**Model/source provenance:** [third_party/fsr411/README.md](third_party/fsr411/README.md).
**Latest performance and validation:** [docs/FSR411_PERFORMANCE.md](docs/FSR411_PERFORMANCE.md).
**GPU stage profiling:** [docs/FSR411_PROFILING.md](docs/FSR411_PROFILING.md).

## Build and run

Requires macOS 26+, Xcode 26+ with its Metal Toolchain, and Python 3.
DXC, SPIRV-Cross and SPIRV-Tools are included in `tools/bin`.

```sh
make -B
tools/bin/fsr4mac --backend-info

tools/bin/gen_testcase testcases/quality411 quality 1280 720 1920 1080 8
tools/bin/fsr4mac testcases/quality411 out/lossy --profile lossy
tools/bin/fsr4mac testcases/quality411 out/exact --profile exact
tools/bin/fsr4mac testcases/quality411 out/exact-fast --profile exact-fast
tools/bin/fsr4mac testcases/quality411 out/bench --profile lossy --bench 10
```

The first run compiles the requested permutation and caches it in `build/`.
A normal run writes RGBA16F `.bin` files, PNG previews and `run.json`.
Benchmark mode writes timing and inference/reuse counts to `run.json`.
`--bench N` measures N complete sequences after one warmup sequence; each
sequence starts with a fresh temporal view. It does not accept legacy
per-pass `--reps` timing flags. See [BENCHMARK.md](BENCHMARK.md).

Use `--i8mpp`, `--scalar` or `--reference` to select a backend explicitly.
`--reference` executes the imported 4.1.1 HLSL graph as a slow diagnostic
oracle. `--optimize 0` disables the PRE routing and POST color-cache rewrites
for comparisons. Profile and backend are separate choices.

## Exact, Exact Fast and Lossy

| Profile | Behavior | M5, 864×560 → 1712×1112 |
| --- | --- | ---: |
| Exact | Original model coefficients and integer rounding, IEEE PRE/POST math; evaluates the network each frame. Byte-identical to the imported reference. | ~5.0 ms |
| Exact Fast | Exact's network, with Metal relaxed math in PRE/POST (like FSR 4.0.2's fast math). | ~4.3 ms |
| Lossy | Folded weak taps, selected half-up quantizers, and motion-reprojected reuse of the previous inference on eligible frames; Exact Fast's PRE/POST math. | ~3.0 ms |
| `fsr4` | The previous FSR 4.0.2 model, for comparison. | ~4.5 ms |

Exact Fast stays ≥ 64 dB PSNR from Exact on the game captures; the
Exact–Lossy gap is several times larger. Relaxed math may reorder additions
and form FMAs but keeps Inf/NaN handling. See
[FSR411_PERFORMANCE.md](docs/FSR411_PERFORMANCE.md).

Lossy follows the numerical policy of the pinned
[RX 7000 optimization project](https://github.com/zgauthier2000/fsr4-shader-performance-overrides)
release `dll-2026-10-07`, independently implemented on MIT sources.
Reused frames still process current input, motion, depth, history and output.
A reset, invalid cache, model/profile change, or Ultra Performance requires
fresh inference. Consecutive inference skips are prevented because the input
motion describes only one frame. Dense Apple tensor kernels still evaluate
zero-weight products; coefficient folding alone is not an Apple GPU speedup.

![FSR4 Panel](docs/images/panel.png)

The panel's **FSR 4 / FSR 4.1 / FSR 4.1 Lite** selector (`fsr4`,
`exact-fast`, `lossy`) updates the running scaler and resets temporal
history; FSR 4 switches the bridge to the 4.0.2 model. Bit-exact Exact is not
in the panel. `FSR4_PROFILE=exact|exact-fast|lossy|fsr4` sets the launch
profile; existing `panel.conf` preferences override it unless
`FSR4_SETTINGS=none` is set.

## CrossOver and DLSS input

The path is **DLSS/NGX → D3DMetal → MetalFX → FSR 4.1.1**. The hook receives
values after D3DMetal's conversion, preserves their signs, normalizes motion
by the logical motion image, and handles jitter, exposure, input regions and
dynamic resolution through one shared input contract.

```sh
python3 tools/crossover/crossover_fsr4.py install --fsr-version 4.1.1 --profile lossy --panel
# Build and validate the local distribution without changing a bottle:
python3 tools/crossover/crossover_fsr4.py install --no-bottle --profile lossy
```

The installer uses a copied D3DMetal framework and generates a matching
4.1.1 smoke-test output. [CrossOver details](docs/CROSSOVER.md) describe the
bottle configuration and restoration.

## Implementation and checks

- Actual common and Ultra Performance 4.1.1 initializers and shader sources,
  with pinned hashes and import verification; all three output classes.
- Twelve native INT8 neural passes, separate POST CNB, and the thirteen
  required border passes. Vector requantization preserves the integer
  contract. Per-pass metadata is compiled in as function constants; CNB and
  32-channel FNB/CT blocks run on one SIMD group with their matrix products
  chained in registers (guarded by a layout check with a threadgroup
  fallback), and 64-channel blocks use 10 KiB of threadgroup memory.
- Packed PRE quad exchange and a float32 shared POST color cache. PRE/POST
  use ordered arithmetic with FMA contraction disabled so routing and cache
  rewrites preserve quantized inputs and final output. Division is
  reciprocal and exp/log are fast functions (0.55–0.85 ms per frame on M5);
  autoexposure keeps precise math. POST loads only its active input
  footprint, evaluates CT2D with provably exact FP32 integers and, on Lossy
  reused frames, classifies motion once per 2×2 cell.
- Two-dispatch SPD autoexposure with explicit Metal visibility between tile
  and final reductions; correct exposure history and partial viewport bounds.
- Classic Metal and Metal 4 encoding, per-owner state, retained in-flight
  resources, and atomic publication of generated model caches.

```sh
make test
make build-fsr411-tests
make test-fsr411-gpu          # sequential GPU validation on M5
```

Validation uses independent CPU integer arithmetic, direct imported HLSL
compiled for Metal, optimized native/scalar kernels, and the live bridge.
The old Windows 4.0 golden output is not a 4.1.1 oracle. Local parity checks
do not establish independent Windows 4.1.1 output parity or quality in every
game. Current results and limitations are in [docs/STATE.md](docs/STATE.md).

## Source map and legacy model

| Path | Purpose |
| --- | --- |
| `src/fsr411_runtime.h` | Shared graph, resources, dispatch and temporal policy. |
| `src/fsr411_mpp.metal`, `src/fsr411_reference.metal` | Native tensor and portable scalar INT8 kernels. |
| `src/fsr4_input.h`, `src/fsr411_live.mm` | DLSS/MetalFX input contract and live routing. |
| `tools/compile_fsr411.py`, `tools/gen_fsr411_native.py` | Shader compilation and authentic model packing. |
| `tools/fsr411/`, `third_party/fsr411/` | Transforms, import verification and source provenance. |

FSR 4.0.2 is retained explicitly with `--fsr-version 4.0.2` or
`FSR4_VERSION=4.0.2`, including its older FP16 backend.
Its [architecture](docs/FSR402_ARCHITECTURE.md) and
[guide](docs/GUIDE_FSR402.md) are archived separately. `build_shaders.sh` and
`tools/precompile_models.py` continue to prepare the legacy model.
