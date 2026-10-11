> Historical FSR 4.0.2 documentation. Select `--fsr-version 4.0.2` for its CLI commands or `FSR4_VERSION=4.0.2` for the live bridge. Current defaults and instructions are in [GUIDE.md](GUIDE.md).

# fsr4-mac — standalone native Metal port of AMD FSR 4 (INT8 + FP16)

**Гайд по запуску (RU):** [docs/GUIDE.md](GUIDE.md) — требования, сборка, тестовые кадры, CLI, сверка с эталоном AMD, FSR 4 в играх через CrossOver, откат, решение проблем.

**Handoff/state doc:** [docs/STATE.md](STATE.md) — kernel map, measured results, rejected approaches, M5 benchmark protocol, optimization roadmap.

**M1–M5 / Pro / Max / Ultra:** [docs/HARDWARE.md](HARDWARE.md). The CLI
and live encoder now select a backend automatically: FP16 for M1–M4,
accelerated INT8 for an Apple family 10+ GPU with Metal 4. Native startup
failures fall back to FP16. Use `--int8` explicitly for the generated
reference path, and `--backend-info` to inspect this Mac's selection.

A command-line tool that runs the **AMD FSR 4 INT8 upscaler** natively on Apple
Metal. It reads recorded input frames (colour / depth / motion vectors +
per-frame parameters), executes the full 14-kernel FSR4 INT8 network pipeline
(pre → 12 model passes → post, plus optional SPD auto-exposure, RCAS and debug
view), and writes upscaled RGBA16F + PNG frames — designed to be numerically
identical to the reference DX12 provider so recorded golden dumps can be
compared pass-by-pass.

Model weights (`initializers.bin`, `src/pass0_weights.h`) come from the open
FSR 4 build vendored in `third_party/fsr4`.

Reference implementation being ported:
`third_party/fsr4/src/Kits/FidelityFX/upscalers/fsr4/dx12/ffx_provider_fsr4_dx12.cpp`
(FidelityFX-SDK commit `01446e6a74888bf349652fcf2cbf5f642d30c2bf`).

---

## Table of contents

- [Quickstart](#quickstart)
- [Repository layout](#repository-layout)
- [Shader toolchain](#shader-toolchain)
- [Host architecture](#host-architecture)
- [Tools reference](#tools-reference)
- [Test-data contract](#test-data-contract)
- [Validation status & known caveats](#validation-status--known-caveats)
- [Limitations & next steps](#limitations--next-steps)
- [CrossOver / DLSS](#crossover--dlss)

---

## Quickstart

Requirements: Apple Silicon Mac, Xcode (`clang++`, `xcrun metal`,
`xcrun metallib`) with the Metal Toolchain component
(`xcodebuild -downloadComponent MetalToolchain` on Xcode 26+), python3 with
numpy (`tools/gen_i8mpp.py`: `--i8mpp` and the CrossOver path). `dxc`,
`spirv-cross` and `spirv-val` are vendored in `tools/bin/`. Step by step:
[docs/GUIDE.md](GUIDE.md).

```bash
# 1. Build host + helper tools (-B: the tools/bin binaries are committed, so a
#    plain `make` in a fresh clone keeps them instead of rebuilding)
make -B

# 2. (Optional) pre-compile shaders — fsr4mac also compiles variants on
#    demand into build/ and caches them
./build_shaders.sh                    # all 6 presets @1080
./build_shaders.sh --all              # all presets × {1080,2160,4320} tiers
./build_shaders.sh --preset quality --res 2160

# 3. Generate a deterministic synthetic testcase
tools/bin/gen_testcase testcases/testcase_quality_720to1080 quality 1280 720 1920 1080 8
tools/bin/gen_testcase testcases/testcase_perf_540to1080   performance 960 540 1920 1080 8

# 4. Run the upscaler
tools/bin/fsr4mac testcases/testcase_quality_720to1080 out/

#    with API validation:
MTL_DEBUG_LAYER=1 tools/bin/fsr4mac testcases/testcase_quality_720to1080 out/

#    with per-pass scratch dumps (slow, one command buffer per pass):
tools/bin/fsr4mac testcases/testcase_quality_720to1080 out/ --dump-passes

# 5. Baselines / comparison
tools/bin/bilin testcases/testcase_quality_720to1080/f00_color.bin 1280 720 1920 1080 out/f00_bilin.bin out/f00_bilin.png
tools/bin/compare out/f07_out.bin out/f07_bilin.bin 1920 1080 out/diff.png
tools/bin/compare out/ out_golden/ out/diff/     # per-frame mode (reads meta.json)
```

Outputs per frame: `fNN_out.bin` (RGBA16F raw, `uw*uh*4` half-floats) and
`fNN_out.png`. With `--dump-passes`: `fNN_passPP_scratch.bin` (raw scratch
buffer after model pass PP), `fNN_recurrent.bin` (RGBA8), `fNN_history.bin`
(RGBA16F), `fNN_scratch.bin` (frame-end scratch).

GPU time per frame is printed from `MTLCommandBuffer` `GPUStartTime`/
`GPUEndTime` timestamps.

---

## Repository layout

    src/fsr4mac.mm          Metal host (ObjC++) — faithful port of
                            ffx_provider_fsr4_dx12.cpp
    src/pass0_weights.h     per-preset pass-0 weight tables extracted from the
                            provider's static arrays (6 × 256 uint32)
    src/gen_testcase.cpp    deterministic synthetic input generator
    src/compare.cpp         per-channel max/mean-abs + PSNR + diff PNG
    src/bilin.cpp           CPU bilinear upscaler (visual baseline)
    src/pngio.h             minimal zlib PNG writer (8-bit RGBA)
    tools/compile_variant.py  per-permutation shader build:
                            DXC -spirv → SPIR-V patch → spirv-cross → MSL →
                            metallib + bindings.{json,tsv}; --model i8|f16
    tools/gen_f16.py        offline fp16 transform: rewrites the generated
                            model HLSL (decls, calls, embedded weights) and
                            emits initializers_f16.bin
    patches/hlsl/ml2code_runtime/fp16_ops.hlsli
                            hand-written real-domain fp16 operator library
    patches/hlsl/pre_common_f16.hlsli / post_common_f16.hlsli
                            fp16 variants of the pre/post stage internals
    tools/bin/              vendored dxc / spirv-cross / spirv-val + built tools
    tools/lib/              libdxcompiler.dylib (dxc dependency)
    build_shaders.sh        batch-compile entry point
    third_party/fsr4/       unmodified AMD sources (MIT) — model HLSL +
                            initializers.bin per preset
    patches/hlsl/           hand-patched HLSL copies (see below)
    build/                  generated: hlsl_shadow/, per-variant dirs, metallibs
    TESTDATA.md             the shared recorded-frame contract
    VERIFICATION.md         golden-vs-port verification log (int8 + fp16)
    BENCHMARK.md            how to run --bench / --caps (for the M5 bot)
    docs/GUIDE.md           step-by-step launch guide (RU): CLI, golden check,
                            CrossOver install/restore, troubleshooting
    docs/CROSSOVER.md       DLSS→FSR4 in CrossOver Preview: copied D3DMetal,
                            bottle env, hook, restore

---

## Shader toolchain

### Route used (fallback 3)

    HLSL ──dxc -spirv──▶ SPIR-V ──patch──▶ SPIR-V ──spirv-cross──▶ MSL ──xcrun metal──▶ .air ──metallib──▶ shaders.metallib

The primary route (DXIL → Apple Metal Shader Converter) was not usable: the
converter download is Apple-ID-gated and unavailable from this machine. The
SPIR-V route is numerically exact anyway — `dot4add_i8packed`/`OpSDot` maps to
MSL `reduce_add(int4(as_type<char4>(a)) * int4(as_type<char4>(b)))` with DP4A
semantics, no polyfill required.

### DXC flags (per kernel)

```
-spirv -T cs_6_6 -enable-16bit-types -HV 2021 -O3 -Vd
-fspv-target-env=vulkan1.3 -fvk-t-shift 32 0 -fvk-u-shift 64 0
-DDATA_TYPE=float -DDATA_TYPE2=float2 -DDATA_TYPE3=float3
-DDATA_TYPE_VECTOR=float4 -DNETWORK_DATA_TYPE=float16_t
+ per-kernel -D permutation defines (see below) and -E <entry>
```

Why each flag matters:

- **`-T cs_6_6`** (not the DX12 `.bat`'s `cs_6_4`): the generated i8 shaders
  use `pack_clamp_s8`/`unpack_s8s32`, which emit SM 6.6 `Pack4x8`/`Unpack4x8`
  SPIR-V ops — `cs_6_4` fails SPIR-V validation on this path.
- **`-Vd`** (skip DXC's own SPIR-V validation): DXC 1.9.5399 emits invalid
  `OpCopyLogical` instructions for struct-copied `ByteAddressBuffer`/
  `RWByteAddressBuffer` resources (see patch 1 below).
- **`-fvk-t-shift 32 0 -fvk-u-shift 64 0`**: separates SRV (`t#`) and UAV
  (`u#`) binding spaces so same-numbered resources don't collide — without it
  spirv-cross aliases e.g. `InitializerBuffer` and `ScratchBuffer` onto one
  `[[buffer(0)]]`.
- **No `--msl-decoration-binding`**: Metal's `[[buffer(n)]]` cap is 31, and
  decorated Vulkan bindings exceed it. Instead indices are packed
  sequentially and the host binds **by resource name** via the generated
  `bindings.tsv` (parsed from the MSL signatures).
- **`DATA_TYPE*` overrides**: Vulkan SPIR-V forbids fp16 texture formats, so
  the macro-resolved texture/sampled types compile as `float*`. Widening
  fp16→fp32 on load is exact and fp32→fp16/unorm stores round identically;
  all network arithmetic stays on explicit `half`/`NETWORK_DATA_TYPE` paths.

### Patches applied (AMD sources stay pristine)

| # | What | Where | Why |
|---|------|-------|-----|
| 1 | `OpCopyLogical` (opcode 400) → `OpCopyObject` (83), rewritten in the emitted `.spv` | `tools/compile_variant.py::patch_spv` | DXC emits `OpCopyLogical` for same-type struct copies of resource wrappers, which `spirv-val` rejects |
| 2 | `_Static_assert(...)` commented out at lines 35–36, 366–367, 657–658, 981–982 | `patches/hlsl/ml2code_runtime/operators/int8_NHWC/Fused/FNB_CT2D_ADD.hlsli` | DXC's SPIR-V backend doesn't implement `_Static_assert` |
| 3 | `[unroll]`/`[unroll(n)]` → `[loop]` on every ml2code_runtime/ffx header, applied to generated shadow copies in `build/hlsl_shadow/` | `tools/compile_variant.py` | Full unrolling produced ~500 KB MSL fused-conv kernels; the GPU JIT compiler tripped the ~40 s watchdog (esp. passes 7–9). Rolled loops cut pass-7 SPIR-V ~503 KB → ~25 KB. Scheduling hint only — numerics unchanged. `--keep-unroll 1` leaves the hints in pre/post/rcas (post ~9% faster on M5, output then differs from the `[loop]` build at the fp16-ulp level) |
| 4 | `.xyzz, uint2` → `.xyzx, uint2` on generated MSL | `tools/compile_variant.py` | `RWTexture2D<float3>` stores expand alpha = `.x` on D3D12 (verified on WARP golden dumps); spirv-cross emits `.xyzz` (alpha = blue) |

### Per-variant defines

| Kernel(s) | Defines |
|-----------|---------|
| `pre`, `post`, `debug` | `FFX_MLSR_RESOLUTION` (0=1080/1=2160/2=4320 tier), `FFX_MLSR_DEPTH_INVERTED`, `FFX_MLSR_LOW_RES_MV`, `FFX_MLSR_JITTERED_MOTION_VECTORS`, `FFX_MLSR_COLORSPACE` (0=linear/1=srgb), `FFX_MLSR_AUTOEXPOSURE_ENABLED`, `FFX_DEBUG_VISUALIZE` (post/debug) |
| `pass1..12` | `MLSR_PASS_<n>=1`, entry `fsr4_model_v07_i8_pass<n>` |
| `rcas` | `FFX_MLSR_COLORSPACE`, `FFX_MLSR_AUTOEXPOSURE_ENABLED` |
| `spd` | `FFX_GPU`, `FFX_HLSL`, `FFX_HLSL_SM=66` |

Variant build dir key: `build/fsr4_<preset>_<res>_di<>_lm<>_jm<>_cs<>_ae<>_db<>[_rc]/`.

---

## Host architecture

`src/fsr4mac.mm` is a faithful port of the DX12 provider: same resources,
constant layouts, dispatch order and workgroup geometry.

### Resource set (per context)

| Resource | Format / size | Notes |
|----------|---------------|-------|
| input colour | RGBA16F @ render | uploaded per frame |
| input depth | R32F @ render | |
| input MV | RG16F @ render | |
| input exposure | R32F 1×1 | `fNN_exposure.bin` or `meta.pre_exposure` (always bound, mirroring the reference harness) |
| auto-exposure | R32F 2×1 | SPD output |
| SPD atomic | R32Uint 1×1 texture + buffer alias | counter |
| luma mip 5 | R32F `ceil(render/16)` | |
| **recurrent** | RGBA8 @ upscale | persistent across frames |
| **history / historyReprojected** | RGBA16F @ upscale | persistent |
| RCAS temp | RGBA16F @ upscale | only when `rcas` |
| scratch | `MTLBuffer`, 20,880,256 / 83,232,256 / 332,352,256 B @ 1080/2160/4320 tier | `ByteAddressBuffer` equivalent; `FSR4_SCRATCH_PRIVATE=1` switches to private storage + staging blits for dumps |
| initializer | `MTLBuffer` from preset's `initializers.bin` | model weights |
| debug info | R16_UNORM @ upscale | only when `debug_view` |

### Constant buffers

- `MLSR_Optimized_Constants` — 104 B, mirrors the provider struct
  (`inv_size`, `scale`, `inv_scale`, `jitter`, `mv_scale`, `tex_size`,
  `max_renderSize`, `fMotionVectorJitterCancellation`, dims, `reset`,
  `preExposure`, `previous_preExposure`, `rcas_enabled`, `rcas_sharpness`).
- `AutoExposureSPDConstants` — SPD workgroup offset/mips/invInputSize/
  preExposure (via `ffxSpdSetup`, ported).
- `cbRCAS` — `FsrRcasCon` ported (`-2*sharpness + 2`).
- `cbPass_Weights` — 1 KB per-preset table from `src/pass0_weights.h`.

### Dispatch order (per frame)

`SPD (if auto_exposure) → pre → passes 1..12 → post → RCAS (if rcas) →
debug view (if debug_view)`

| Pass | Encoder res | Group count |
|------|-------------|-------------|
| pre, post, rcas | upscale | `⌈uw/16⌉ × ⌈uh/16⌉` |
| 1, 2, 12 | upscale/2 | `/64` X |
| 3–5, 10, 11 | upscale/4 | `/64` X |
| 6–8 | upscale/8 | `/64` X |
| 9 | upscale/8 | `/8 × /8` |
| debug | upscale | `⌈uw/8⌉ × ⌈uh/8⌉` |

A `memoryBarrierWithScope:Buffers|Textures` after every dispatch replicates
the DX12 UAV barriers. The INT8 selector never dispatches the WMMA
padding-reset passes — mirrored exactly. Recurrent/history state persists
across frames inside one run, matching provider semantics; `reset` on frame 0
is passed through constants and, as in the provider, forced on the dispatch
that creates the model. Internal textures, constants and dispatch sizes use
the provider's `AlignUp(upscaleSize, 8)`; the output texture keeps its size.

### Environment variables

- `FSR4_SCRATCH_PRIVATE=1` — allocate scratch with private storage (dumps go
  through a staging blit). Used to rule out shared-memory effects while
  debugging; output unaffected.

---

## Tools reference

| Tool | Usage |
|------|-------|
| `fsr4mac` | `fsr4mac <testcase_dir> <out_dir> [--auto\|--int8\|--fp16\|--i8mpp] [--dump-passes] [--bench N] [--reps R] [--builddir dir]` |
| `gen_testcase` | `gen_testcase <out_dir> [preset rw rh uw uh frames]` — deterministic procedural scene: checkerboard + gradient + high-frequency stripes + counter-moving square, planar depth, analytic MVs, Halton(2,3) 8-phase jitter, `reset=true` on frame 0 |
| `compare` | `compare a.bin b.bin w h [diff.png]` or `compare dirA dirB [diff_dir]` — per-channel max-abs/mean-abs/PSNR |
| `bilin` | `bilin in.bin rw rh uw uh out.bin [out.png]` — CPU bilinear baseline |
| `build_shaders.sh` | `--all` / `--preset p --res r`; default = 6 presets @1080 |
| `compile_variant.py` | single-variant build + `bindings.{json,tsv}` manifest |

---

## Test-data contract

See **TESTDATA.md** — `meta.json` + `fNN_{color,depth,mv,exposure}.bin` +
`fNN_out.bin/png`. MV convention: `mv = prev_px − cur_px` in render pixels,
`mv_scale` divided by MV-source size in the host. The DX12/WARP golden
testcases (inputs + reference outputs + per-pass dumps) are published as
GitHub release
[`golden-v0.1`](https://github.com/UzenUPoZiTiV4ik/fsr4-mac/releases/tag/golden-v0.1);
`compare` + the `--dump-passes` per-pass scratch snapshots exist for
pass-by-pass bisection.

---

## INT8 tensor-ops path (`--i8mpp`, M5)

On Apple M5 the model passes run as native Metal 4 `mpp::tensor_ops`
kernels (int8 x int8 -> int32 on the GPU neural accelerators) with the
reference int8 requantization: `src/i8mpp.metal`, `src/i8mpp_host.h`,
`tools/gen_i8mpp.py`. The post pass's ConvNextBlock and its 2x2
ConvTranspose (model parameters + recurrent) run there too (pass 13,
`i8_cnb_ct2d`), so the generated post only does the Gaussian resample; the
post variant carries the `_pt` suffix. Quality 720->1080: **5.2–5.6 ms/frame**
(fp16 12.7, int8 47.5), accuracy equal to the int8 path (47/44/42 dB vs the
DX12 golden on frames 0/3/7; model passes bit-exact or within single-LSB flips
against the golden per-pass dumps). Automatic selection uses it on M5-class GPUs (`FSR4_MODEL=f16`
switches back). Details and the one known reference-side quirk (pass 11 left
band): [docs/STATE.md](STATE.md).

## FP16 model path (`--fp16`)

`fsr4mac --fp16` runs a second, faster model path: weights are dequantized
offline to fp16 (`gen_f16.py` emits `initializers_f16.bin` — same record
layout as `initializers.bin`, every BufferStorage payload repacked as
`int8 * tensorScale` at 2× byte size), activations stay in real-domain fp16,
and the int8 ops are replaced by `fp16_ops.hlsli` equivalents that re-apply
the int8 quantization as real-domain requantization
(`clamp(round(v/s), -128, 127) * s`) at every store the int8 pipeline
quantized at — this keeps the fp16 path numerically locked to the int8
reference (dropping requant measurably drifts: 34.5 dB → 42 dB).

Measured on the golden testcases: quality 42–43 dB PSNR / perf 39–40 dB vs
DX12 golden (int8 path: ~44–50 dB). VM perf is ~2.4× faster than int8
(45 ms vs 107 ms/frame at 720→1080); `--bench N` prints per-stage GPU times.
Details and per-pass tables: VERIFICATION.md. Benchmark instructions for real
hardware (M5): BENCHMARK.md, including `fsr4mac --caps` which probes
`mpp::tensor_ops::matmul2d` (Metal 4 neural-accelerator path, not yet used by
the model) plus device families.

## Validation status & known caveats

Machine: **VirtualMac2.1 (Apple M4 virtualized), "Apple Paravirtual device"**,
Xcode 26.6.

**Verified clean:** all 6 presets @1080, quality @2160, and every flag combo
(rcas / srgb / auto_exposure / depth_inverted / jittered_mv) run under
`MTL_DEBUG_LAYER=1` with zero validation errors.

**`MTL_SHADER_VALIDATION=1` — paravirtual bug, and perturbs even on real HW.**
On the paravirtual device it reports no errors but **silently drops all
compute texture writes** (output comes back all zeros; reproduced with a
trivial 3-line compute shader). On a **physical Apple M5** output is no
longer zeroed — the bug is VM-only — but the validated run still deviates
from the clean run (f07 PSNR 46.6 dB vs uninstrumented), so enabling shader
validation changes results even on real hardware. Do not compare
golden dumps against a `MTL_SHADER_VALIDATION=1` run.

**Run-to-run nondeterminism — VM-only, confirmed on real HW.** Two
identical runs on the paravirtual device are bit-identical through model
pass 10 then diverge from pass 11 (f07 self-noise max 0.33, mean 4.5e-4,
PSNR 52 dB; see VERIFICATION.md). On a physical Apple M5 the same binary is
**bit-stable across runs** (all 8 `fNN_out.bin` identical) — the noise is a
virtualized-GPU artifact, not a pipeline race.

**Performance:** ~120 ms/frame at 720→1080 on a **physical Apple M5** —
the real-hardware baseline for the later optimization phase (not a VM
artifact: the earlier ~100–120 ms VM figure is matched by real HW).
GPU timestamps on the paravirtual device itself are unreliable — after
shader caches warm they have been observed to under-report by orders of
magnitude.

**Physical-hardware verification (Apple M5, macOS 27.0, Xcode 27.0):**
bit-stable across runs, per-channel means match the WARP golden to 4 digits
(R 0.5367 / G 0.3380 / B 0.4035 on f07). See VERIFICATION.md.

---

## Limitations & next steps

- **DX12/WARP golden comparison done** — see **VERIFICATION.md**: alpha
  expansion fix, exposure mirroring, per-pass bisection, final metrics
  (PSNR ~42-50 dB, mean-abs ~0.002; residual = int8 quantization-boundary
  flips on the fp16 path + paravirtual-device noise).
- FP8/WMMA model path intentionally not ported (AMD-only intrinsics; the
  selector never picks it on non-AMD anyway).
- CrossOver integration is separate from this CLI: see **docs/CROSSOVER.md**.
  `CrossOver.app` is not modified; one bottle is pointed at a copied D3DMetal.
- Perf tuning (fused passes, memory layout) deliberately deferred.

---

## CrossOver / DLSS

How FSR4 replaces the temporal MetalFX scaler CrossOver uses for DLSS, which files are copied, and how to put the bottle back: **[docs/CROSSOVER.md](CROSSOVER.md)**.

```bash
python3 tools/crossover/crossover_fsr4.py install --panel   # hook + bottle keys + FSR 4 panel
python3 tools/crossover/crossover_fsr4.py restore          # undo the bottle keys
```

- **Model**: int8 on the M5 neural accelerators (`i8mpp`, same kernels as the CLI), pass 13 = post's
  CNB + ConvTranspose, ~5 ms/frame at 1712x1112 balanced. `FSR4_MODEL=f16` falls back to fp16.
- **Sharpness**: DLSS has no sharpness knob, so RCAS runs by default at FSR sharpness **0.4**
  (`FSR4_SHARPNESS=<0..1>`, `0` or `FSR4_RCAS=0` disables). Same constants as the provider.
- **Panel**: `dist/crossover/FSR4 Panel.app` — a Dock application (status: FSR 4 active or why not,
  preset, input -> output resolution, game frame time; live RCAS sharpness slider). Launched by the
  hook when `FSR4_PANEL=1` is in the bottle (`install --panel`), or opened by hand. Talks to the hook
  through `dist/crossover/status.conf` / `panel.conf`; the slider setting persists.
- **Provider fidelity**: constants, dispatch sizes, preset thresholds, `AlignUp(upscaleSize, 8)`,
  forced `reset` on model (re)creation, per-scaler history.
