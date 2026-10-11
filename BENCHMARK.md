# Benchmarking fsr4-mac

## FSR 4.0.2 vs FSR 4.1.1 / M5 — 2026-10-11

All builds in one session on the same recorded 864×560 → 1712×1112
sequence, RCAS 0.504, AE on, Low Power Mode off. Five interleaved rounds ×
15 sequences; median of round means, full command buffer. FSR 4.0.2 runs
`--i8mpp --bench 15 --reps 1` (fused-frame mean) on a copy of the input
with a 4.0.2-format `meta.json`; its output matches 4.1.1 visually.

| Build | ms | Round range | vs FSR 4.0.2 |
| --- | ---: | ---: | ---: |
| FSR 4.0.2 | 4.466 | 4.43–4.56 | — |
| 4.1.1 first port (`ef1fa1b`) Exact | 7.340 | 7.28–7.49 | +64% |
| 4.1.1 first port Lossy | 5.162 | 5.14–5.57 | +16% |
| **4.1.1 Exact** (IEEE, bit-exact) | **5.008** | 4.97–5.15 | +12% |
| **4.1.1 Exact Fast** | **4.304** | 4.20–4.39 | −4% |
| **4.1.1 Lossy** | **3.001** | 2.98–3.05 | −33% |

The 4.1.1 network (twelve passes) is already cheaper than 4.0.2's (stage
counters: 2.1–2.5 vs 2.7 ms); Exact is slower only through IEEE PRE/POST.

**Relaxed PRE/POST math for Exact Fast and Lossy.** FSR 4.0.2 compiles
PRE/POST with Metal's default fast math. Exact Fast and Lossy now use
`-fmetal-math-mode=relaxed -fmetal-math-fp32-functions=fast` (reordering
and FMA allowed; Inf/NaN preserved for PRE's colour checks). Same-session
A/B (5×15): Exact Fast 4.548 → 4.400, Lossy 3.241 → 3.123 ms; `fast` mode
measured the same as `relaxed`. Most of the gain is PRE (POST alone:
4.520 / 3.187 ms). Four game captures (32 frames each): no non-finite
values; versus the previous ordered build PSNR ≥ 64.1 dB (Exact Fast) and
≥ 63.0 dB (Lossy); distance to Exact: Exact Fast ≥ 64.6 dB (was 66.9),
Lossy unchanged at ≥ 48.7 dB. Exact is byte-identical to before.

[Raw results](docs/benchmarks/fsr402-vs-fsr411-m5-2026-10-11.json)

## FSR 4.1.1 profiles / M5 — 2026-10-11

Same recorded sequence, RCAS 0.504, AE on, battery 72%, Low Power Mode off.
Five interleaved rounds × 15 sequences; median of round means.

| Profile | All frames | Full | Round range |
| --- | ---: | ---: | ---: |
| **Exact** (IEEE PRE/POST) | **5.349** | 5.284 | 5.27–5.40 |
| **Exact Fast** | **4.716** | 4.631 | 4.57–4.75 |
| Lossy | 3.357 | 4.534 | 3.33–3.68 |

Exact is byte-identical to `ef1fa1b` again; Exact Fast is byte-identical to
the previous fast-math Exact; Lossy is unchanged (32/32 capture frames each).
Exact Fast vs Exact on displayed 8-bit sRGB: ΔE2000 > 1 on 0.08% of pixels,
PSNR ≥ 60.4 dB, SSIM ≥ 0.9996.

Rejected: RCAS reading POST's history instead of a separate RGBA16F
intermediate. Byte-identical (96/96), but within noise (median Δ
0.01–0.05 ms; stage profile POST −0.014, RCAS +0.025 ms).

[Raw results](docs/benchmarks/fsr411-m5-profiles-2026-10-11.json)

## FSR 4.1.1 native network / M5 — 2026-10-11

Same recorded 864×560 → 1712×1112 sequence, RCAS 0.504, AE on, battery
78–82%, Low Power Mode off. Five interleaved rounds × 15 sequences; median of
round means, full command buffer. Every variant uses the current PRE/POST/AE;
only the native network source and manifest differ.

| Variant | Exact all | Exact full | Lossy all | Lossy full | Lossy reused |
| --- | ---: | ---: | ---: | ---: | ---: |
| Network before (`0738530`) | 5.474 | 5.394 | 3.696 | 5.406 | 1.773 |
| + constants, one SIMD group | 4.822 | 4.745 | 3.258 | 4.607 | 1.760 |
| **+ cooperative chains** | **4.340** | **4.249** | **3.028** | **4.200** | **1.728** |

Exact −20.7%, Lossy −18.1%; round ranges do not overlap
(Exact 5.43–5.52 → 4.19–4.52, Lossy 3.58–3.84 → 3.01–3.08 ms). Integer-only
and byte-identical: all 64 frames of the four game captures, Exact and
Lossy, and the full/reuse schedule match the start of the session.

- **Pass metadata as pipeline constants.** Offsets, strides, channel counts,
  weight/bias offsets and epilogue shifts of each logical pass become
  `[[function_constant]]` values (`FSR411_SPECIALIZE`), one pipeline per pass.
  Widths/heights and POST's session-appended output offset stay dynamic; the
  212-byte `PassParams` ABI and cached manifests are unchanged.
- **One SIMD32 group per tile.** CNB runs 32×1 tiles and NF32 FNB/CT 16×1 tiles
  on 32 threads instead of 128, with 5 KiB threadgroup storage. NF64 keeps its
  16×1 four-group schedule. In a 4×15 ablation each of the two changes alone
  gave ~0.4 ms Exact; together 5.56 → 4.77 ms.
- **Register-resident chains.** CNB and NF32 FNB/CT keep conv3x3 →
  expansion → projection (→ transpose for CT) in cooperative tensors, with no
  threadgroup round trips or barriers. Cooperative-tensor element ownership is
  implementation-defined, so each kernel checks the layouts it relies on with
  `simd_all` and otherwise runs the SG1 kernel; both paths are tested
  (`FSR411_CHAIN_FORCE_FALLBACK=1` in `test-fsr411-native`).

Lossy averages 3.03 ms, at the 3 ms target; its full frame is 4.20 ms. Exact
is 4.34 ms, still above 4 ms. Not adopted: skipping the scratch clear and
padding on native frames (5.50 → 5.55 Exact, 3.65 → 3.62 ms Lossy, noise) and
an unguarded chain (another ~0.05–0.09 ms, but it relies on the M5 layout).

[Raw results](docs/benchmarks/fsr411-m5-network-2026-10-11.json)

## FSR 4.1.1 Exact PRE/POST math / M5 — 2026-10-11

Same fixture and protocol as the Lossy section below (four interleaved
rounds × 15 sequences, RCAS 0.504, AE on, battery 13–16%, Low Power Mode
off). Exact now uses the same PRE/POST math flags as Lossy.

| Variant | All frames | Full | Round range |
| --- | ---: | ---: | ---: |
| Exact before | 6.254 | 6.197 | 5.85–6.71 |
| **Exact after** | **5.381** | **5.289** | 5.28–5.46 |
| Lossy (same run) | 3.609 | 5.289 | 3.58–3.65 |

Exact is 14% faster; round ranges do not overlap. The "before" absolute
value is higher than the 5.90 ms measured earlier in the day; the per-stage
profile agrees with the gain (non-reset frames, three runs each):
PRE 1.22–1.39 → 0.80–0.85 ms, POST 0.90–1.01 → 0.67–0.78 ms, network
unchanged. A full Exact frame now costs the same as a full Lossy frame.

Exact is no longer byte-identical to its earlier recordings. On the four
game captures (32 frames): versus previous Exact mean |Δ| 1.0·10⁻⁵,
max |Δ| 0.48, PSNR ≥ 66.9 dB (mean 82.8 dB). The Exact–Lossy distance is
unchanged (mean |Δ| 5.1·10⁻⁵, PSNR ≥ 48.7 dB). Lossy metallibs are
byte-identical to the previous build. The 17-case runtime matrix (Exact
against direct imported HLSL, `--optimize 0/1`, scalar, UP, RCAS, AE),
CPU, lossy, cadence, CLI, AE, snapshot and live suites pass.

[Raw results](docs/benchmarks/fsr411-m5-exact-math-2026-10-11.json)

## FSR 4.1.1 Lossy PRE/POST math and cell sharing / M5 — 2026-10-11

Same recorded 864×560 → 1712×1112 sequence as below, RCAS 0.504, AE on,
Low Power Mode off (battery). Four interleaved rounds × 15 measured
sequences; median of round means, full command buffer, no stage counters.

| Variant | All frames | Full | Reused | Steady 50:50 |
| --- | ---: | ---: | ---: | ---: |
| Lossy before | 4.266 | 5.820 | 2.499 | 4.156 |
| **Lossy after** | **3.602** | **5.314** | **1.706** | **3.503** |
| Exact (unchanged) | 5.902 | 5.840 | — | — |

Lossy is 15.6% faster; round ranges do not overlap (4.16–4.44 vs
3.55–3.65 ms). Two changes, Lossy only:

- **PRE/POST compiler math.** Stage profiling showed IEEE division and
  precise exp/log, not memory or the network, made PRE 1.22 ms and POST
  0.96 ms. Lossy now builds them with reciprocal division and fast fp32
  functions, keeping `-fno-fast-math -ffp-contract=off` so additions stay
  ordered and NaN/Inf checks remain valid. Measured PRE+POST: 2.14 → 1.57 ms.
- **One skip classification per 2×2 cell.** Reused-frame POST evaluated the
  same ten-probe motion/occlusion classification for all four pixels of a
  cell. 64 lanes now classify the tile's cells before the existing barrier.
  Integer-only and byte-identical: 32/32 replay frames.

Quality on the four game captures (32 frames, 16 reused): versus previous
Lossy mean |Δ| 7·10⁻⁶, PSNR ≥ 67.9 dB; distance to Exact is unchanged
(mean |Δ| 5.1·10⁻⁵, PSNR ≥ 48.7 dB). Exact PRE/POST metallibs and all
32 Exact replay frames are byte-identical to before. `--optimize 0/1` parity
holds for both profiles (runtime matrix, PRE INT8 bytes included).
The Lossy average is now inside the 3–4 ms target; a full-inference frame
is still ~5.3 ms, of which the twelve neural passes plus POST CNB are ~3.6 ms.
Exact keeps IEEE division and precise exp/log and is not in that range.

[Raw results](docs/benchmarks/fsr411-m5-lossy-math-2026-10-11.json)

## FSR 4.1.1 optimization / M5 — 2026-10-10

Comparison against the corrected 4.1.1 baseline `ef1fa1b`: a real recorded
864×560 → 1712×1112 sequence, RCAS on at 0.504, autoexposure on, M5,
Low Power Mode off. Five interleaved rounds, each with one warmup sequence
and 30 measured eight-frame sequences; normal desktop applications remain
running. Values are the median of the five round means in milliseconds.

| Profile | Baseline | Optimized | GPU time reduction |
| --- | ---: | ---: | ---: |
| Exact | 7.695 | 6.390 | 16.97% |
| Lossy | 5.434 | 4.502 | 17.15% |

The optimized Lossy full/reused non-reset means are 6.293 / 2.588 ms; its
balanced 50:50 non-reset estimate is 4.438 ms. A stable 3–4 ms target has
not been reached. Absolute times varied across the session; shorter earlier
comparisons reached 4.08 ms Lossy, but are not substituted for this table.
These are complete upscaler command-buffer times, not game-frame times.

All 64 replay frame comparisons across Exact and Lossy match their
corresponding corrected baseline outputs byte-for-byte. The improvements
preserve integer semantics and history:
vector quantization, selected smaller tiles and SIMD row partitions,
CNB intermediate specialization, bounded POST cache loads and exact FP32
integer CT2D. Stage profiling is opt-in and excluded from these measurements.

[Detailed changes, protocol and checks](docs/FSR411_PERFORMANCE.md) ·
[Raw results and source hashes](docs/benchmarks/fsr411-m5-optimization-2026-10-10.json)

## Initial FSR 4.1.1 port / M5 — 2026-10-10

The initial port default was FSR 4.1.1, Lossy, native INT8 on Apple M5.
The following historical results predate the optimization and chroma fix
above; their commands describe the implementation at that time.
These measurements are independent of the older 4.0.2 measurements below.
They do not establish an upgrade speedup over that different model.

Machine: Apple M5, macOS 27.0.1, battery power, Low Power Mode off.
Input: **synthetic 960×540 → 1920×1080, 16 frames**, RCAS and autoexposure off.
Three interleaved rounds per variant. Each run uses one warmup sequence and
three measured sequences, each with a new temporal View. All 48 measured
frames per round include their full command buffer, including first-frame
clears; file IO, compilation and readback are outside the GPU timestamps.
No parallel GPU test jobs or diagnostic captures ran during these samples.

Values below are the median of the three round means, in milliseconds:

| 4.1.1 profile / PRE+POST optimization | All frames | Full inference, excluding resets | Reused inference | Full/reused frames per round |
| --- | ---: | ---: | ---: | ---: |
| Exact /0 |7.479|7.496|—|48 /0|
| Exact /1 |7.710|7.746|—|48 /0|
| Lossy /0 |5.577|8.122|3.018|24 /24|
| **Lossy /1 (default)** |**5.224**|**7.607**|**2.860**|24 /24|

Lossy/1 uses about 32% less GPU time than Exact/1 on this sequence. Its
additional PRE/POST rewrites reduce the Lossy average by about 6.3% versus
Lossy/0. The Exact rewrite does **not** show a reliable improvement here:
round means overlap and its median is 3.1% higher. Do not attribute speedup
to zero folded weights: the dense MPP kernels still execute their products.
The large profile difference comes from skipping the network on eligible
frames, with an expected image-quality tradeoff.

These are upscaler GPU durations, not whole-game FPS. Only half the frames
reuse inference in this jitter/scale sequence. Other jitter, reset cadence,
Ultra Performance, RCAS, resolution and GPU power conditions change the
result. In particular, UP always evaluates its separate model.

Raw per-round statistics and protocol:
[fsr411-m5-2026-10-10.json](docs/benchmarks/fsr411-m5-2026-10-10.json).
The local run directories contain complete per-frame `run.json` files in
`build/bench411/{exact_base,exact_optimized,lossy_base,lossy_optimized}_1..3/`.

Reproduce from the repository root, repeating the four variant runs in
interleaved order for three rounds:

```sh
make -B tools/bin/fsr4mac
tools/bin/gen_testcase build/bench411/input_perf performance 960 540 1920 1080 16
tools/bin/fsr4mac build/bench411/input_perf build/bench411/exact_base --i8mpp --profile exact --optimize 0 --bench 3
tools/bin/fsr4mac build/bench411/input_perf build/bench411/exact_optimized --i8mpp --profile exact --optimize 1 --bench 3
tools/bin/fsr4mac build/bench411/input_perf build/bench411/lossy_optimized --i8mpp --profile lossy --optimize 1 --bench 3
tools/bin/fsr4mac build/bench411/input_perf build/bench411/lossy_base --i8mpp --profile lossy --optimize 0 --bench 3
```

`run.json` uses `effective_reset`, `skipped_inference`, `gpu_ms`,
`neural_runs` and `reused_frames`; warmup is excluded. Unlike the historical
runner, 4.1.1 has no per-pass `--reps` benchmark. Each sequence starts fresh,
so distinguish reset/full/reused frames rather than mixing different
recording lengths in a performance comparison.

### Image and correctness scope

On both 1712×1112 frames in `tc_game_balanced`, Exact with PRE/POST
optimization 0 and 1 produced **byte-identical RGBA16F files**. Despite its
name, this fixture is a synthetic checkerboard and moving square, not a
capture of a real game.

Lossy intentionally changes its output. Against Exact on those two frames,
RGB mean absolute difference is 0.001885 on the first full-inference frame
and 0.022548 on the reused frame (linear RGB units). Max differences are
0.1851 and0.9608, concentrated around high-frequency/moving content.
These are profile differences, not a quality score against ground truth.
[Recorded comparison statistics](docs/benchmarks/fsr411-synthetic-comparison.json).
The local visual comparison is `build/validation411/game_exact_lossy.png`.

Full graph validation additionally covers 17 small active-viewport cases
with three frames each, including tiers 2160/4320, UP, both motion resolutions,
reversed depth, colorspaces, RCAS and AE. For the precise validation scope,
see [docs/STATE.md](docs/STATE.md). The Windows 4.0 golden output is not used
as a 4.1.1 oracle, and no claim of in-game image quality follows from these
synthetic tests.

## Historical FSR 4.0.2 measurements

The commands and backend descriptions below refer to the old model. Add
`--fsr-version 4.0.2` when running those historical CLI examples.

## 2026-09-28: reuse native INT8 activation storage

Physical Apple M5, macOS 27.0, Low Power Mode off. Baseline: `c846e66`.
Five alternating A/B runs per case, reversing order on odd rounds, using
`--i8mpp --bench 60 --reps 10 --builddir build` (120 fused frames/run).
Both executables and generated pre/post shaders were identical; baseline
loaded a saved copy of `src/i8mpp.metal`, candidate loaded the edited source.
Each native source was compiled under its own content hash.

Median of the five fused-frame means, in milliseconds:

| testcase | before | after | GPU time reduction |
|---|---:|---:|---:|
| quality, 1280x720 → 1920x1080 | 4.8498 | 4.6540 | 4.0% |
| performance, 960x540 → 1920x1080 | 4.7731 | 4.6563 | 2.4% |
| balanced, 1000x648 → 1712x1112 | 4.4849 | 4.5267 | −0.9% |

**Balanced has no established full-frame speedup.** GPU times drifted during
this series and individual pairs sometimes regressed. The median *paired*
reduction was 2.3% / 3.1% / 1.1%; this is a different statistic from the
table above. An initial shorter three-round series (`--bench 20`) gave
4.0% / −3.4% / −0.2%, motivating the longer measurements. These are
upscaler GPU times, not game FPS or a guarantee for every workload.

The affected CT2D stages improved consistently across all five pairs in
all three cases. Quality's median isolated pass 9 / 11 / 13 times fell
from 0.3083 / 0.4865 / 0.6104 to 0.2552 / 0.4076 / 0.5187 ms (15–17%).

Change: reuse threadgroup buffers once their last reader has completed.
CNB a1 and a2 share an allocation; pass 13 also uses it for a3. FNB-CT2D
reuses a2 for a3, retaining a1 for the residual. Existing post-GEMM and
pre-GEMM barriers separate the lifetimes, and matrix inputs never overlap
the accumulator. No math, model weights, tiles or dispatch counts changed.
Metal's pipeline-reported static threadgroup allocation confirms:

| passes | before | after |
|---|---:|---:|
| 1, 2, 12 | 22 KiB | 20 KiB |
| 9, 11 | 24 KiB | 22 KiB |
| 13 | 24 KiB | 20 KiB |

Validation:

- 55/55 final frames byte-identical to the previous version across the
  existing 15-scenario suite: recorded quality/performance/balanced,
  eight-frame sequences, odd sizes, downscale, DRS, native, ultraperf,
  sRGB, PQ, RCAS, 2160 and 4320 shader tiers.
- Fresh before/after runs with `MTL_DEBUG_LAYER=1`: all 200 binary files
  identical for recorded quality (2 frames) and odd-size balanced
  (8 frames), including every pass dump, recurrent/history, scratch,
  reprojected history, constants and final output.
- `make test` and `make test-gpu` passed. x86_64 live tests under Rosetta
  passed with API validation in both `classic` and `mtl4` modes, three
  repetitions each, zero output difference from CLI.
- This does not resolve or establish determinism for the pre-existing
  SPD auto-exposure issue. No new DX12 golden comparison was needed for
  this storage-only change; the comparison is against the previous Metal
  implementation.

Native loop-unrolling hints were also tried and discarded: there was no
consistent gain. Reducing only a3 storage helped; sharing CNB a1/a2 as well
further reduced allocation and pass-13 time. Pre-stage rewrites remain
future experiments.

Measurement samples and validation summaries are preserved in
[the results JSON](docs/benchmarks/native-storage-2026-09-28.json).
Subsequent pre-convolution experiments were removed: they changed INT8
rounding without a reliable full-frame gain in quality/performance. The
selected implementation contains only activation-storage reuse.
CLI and live load the native shader by source hash. Restart the game to
use the updated kernels; no weight regeneration or hook rebuild is needed.

## 2026-09-24: portable backend selection and FP16 boundaries

`auto` is now the default in the CLI and live encoder: generated FP16 on
M1–M4, native INT8 on compatible M5-class GPUs. Pro/Max/Ultra follow their
Metal feature family. See [HARDWARE.md](docs/HARDWARE.md) for capability
checks, fallback, installation and the limits of physical validation.

The FP16 change removes invalid work at tensor/frame edges. Previously,
FNB threads past the end of 480- and 240-pixel rows raced valid writers in
the next row, producing different left-edge pixels on repeated runs.
Bounds checks fix this and skip those calculations; post also skips
off-frame 2x2 blocks before evaluating the network.

Physical M5, Low Power Mode off, `--fp16 --bench 12 --reps 10`, three
alternating A/B runs per case (24 fused frames/run), medians of means:

| testcase | before | after | reduction |
|---|---:|---:|---:|
| quality 720 → 1080 | 12.66 ms | 12.56 ms | 0.7% |
| performance 540 → 1080 | 12.78 ms | 12.60 ms | 1.4% |
| balanced 1000x648 → 1712x1112 | 12.18 ms | 12.09 ms | 0.7% |

The speed gain is small and sensitive to GPU clocks. The principal result
is deterministic, correct boundary handling. No large FP16 speedup is
claimed for M1–M4 without measurements on those chips.

Additional candidates were measured and discarded: FP16 post colour caching
(post roughly unchanged around 1.75 ms), fully unrolled channel reductions
(over two minutes in the driver compiler before cancellation), paired channel
reductions (about 13.7 vs 11.5 ms/frame), native scalar/four-thread CNB kernels
(1.53/1.82 vs about 1.2 ms/pass), 128-thread groups and pipeline sizing hints
(no consistent frame-time improvement).

Validation: 72 capability-policy checks, five Python tests, GPU checks of
automatic selection and forced native failure, repeated byte-identical FP16
outputs, and six classic/Metal 4 live tests under Rosetta. All 55 frames in
the 15-scenario FP16 suite matched the version with corrected tensor bounds
before the post early exit. Native M5 output remains unchanged. Existing
SPD auto-exposure nondeterminism on the small partial-tile testcase remains;
initializing the CLI exposure texture, as the live encoder already does,
does not resolve it.

## 2026-09-24 follow-up: parallel post filter

Baseline: the already optimized colour-cache/CT2D version in the next
section. Same physical M5, Low Power Mode off, three alternating A/B runs
per case with `--i8mpp --bench 20 --reps 10` (40 fused frames/run). Medians
of the three means:

| testcase | previous optimization | this version | additional GPU time reduction |
|---|---:|---:|---:|
| quality, 1280x720 → 1920x1080 | 4.83 ms | 4.67 ms | 3.2% |
| performance, 960x540 → 1920x1080 | 4.86 ms | 4.63 ms | 4.7% |
| balanced, 1000x648 → 1712x1112 | 4.48 ms | 4.33 ms | 3.3% |

Compare within this series: GPU clock variation accounts for the different
absolute baseline times versus the earlier measurements below.

`post_filter.hlsl` assigns one output pixel to each thread in a 16x16
group, replacing the serial four-pixel loop in an 8x8 group. The host still
dispatches one group per 16x16 output tile and reads the actual threadgroup
size from shader metadata. Cache bounds are checked once for the full 3x3
footprint, rather than for each tap. Quality's post stage fell from 0.514
to 0.362 ms (about 30%); performance/balanced measured 0.512 → 0.356 and
0.465 → 0.330 ms. Native matrix operands now expose their fixed tile extents,
and requantization narrows a saturated `int4` directly to a packed `char4`.
For example, quality's pass 1 fell from 0.381 to 0.366 ms.

The same 15-scenario, 55-frame suite below passed again with byte-identical
outputs. Fixed-input post replays at all six exposure values also matched
output and history with Metal API validation enabled. The x86_64 live
Metal 4/Rosetta test matched the CLI exactly. Auto-exposure's previously
observed full-pipeline nondeterminism remains outside this equality claim.

Rejected experiments: cooperative register accumulators with scalar
requantization (CNB about 0.45 vs 0.37 ms); a pre-stage colour cache
(about 0.73 vs 0.68 ms, also changed rounding); four-feature pre-stage
loops (no consistent gain); and filling a smaller post cache based on the
scale (about 0.37 vs 0.35 ms). The selected post source uses an angle-bracket
include so DXC uses the shadow tree's loop policy. A quoted local include
accidentally enabled unrolling and changed the output; that variant was
discarded.

`tools/precompile_i8mpp.sh` has been rerun for the live shader variants.
Restart the game to pick up the new pipelines.

## 2026-09-24: post-filter cache and packed CT2D taps

Physical Apple M5, macOS 27.0 (26A428), Low Power Mode off on both AC and
battery. Baseline: `702b9c7`. Both versions used `--i8mpp --bench 20 --reps 10`
with the same recorded inputs, barriers and generated pre stage. Three
alternating A/B runs per testcase, reversing order on the middle run;
numbers below are medians of the three fused-frame means (40 frames/run).

| testcase | before | after | GPU time reduction |
|---|---:|---:|---:|
| quality, 1280x720 → 1920x1080 | 5.69 ms | 5.10 ms | 10.3% |
| performance, 960x540 → 1920x1080 | 5.57 ms | 4.98 ms | 10.5% |
| balanced, 1000x648 → 1712x1112 | 5.31 ms | 4.64 ms | 12.5% |

These are upscaler GPU times, not total game-frame times or game FPS.
Old 10+ ms results obtained in Low Power Mode should not be used for this
comparison. GPU clock variation still affects individual runs.

The post filter reuses float32 colours transformed with `ApplyMuLaw`
within each threadgroup. CT2D passes 9/11/13 issue one wider matrix multiply
for all four taps, without increasing accumulator storage. On quality,
the isolated post stage fell from about 0.88 to 0.53 ms, pass 9 from 0.35
to 0.28 ms, and pass 13 from 0.60 to 0.55 ms. Pass 11 was roughly neutral
(0.43 → 0.44 ms); the frame improvement is measured across the entire chain.

Validation against the previous Metal version (not a new DX12 golden run):

- **55/55 final frames are byte-identical** across 15 scenarios: the three
  recorded cases (2 frames each); 8-frame quality and performance sequences;
  an 8-frame balanced 421x243 → 641x365 sequence; native, ultraperf and DRS
  with 321x181 output (3 each); DRS 384x240 → 320x192 (3); sRGB, PQ and
  RCAS 0.4 variants of the odd-size case (3 each); quality 3840x2160 (2);
  and a 3856x24 case exercising the 4320 shader tier (2).
- The recorded quality case also matched all per-pass scratch dumps and
  recurrent/history state on both frames.
- The small auto-exposure testcase was **not deterministic in the baseline**:
  baseline-versus-baseline reruns already differed, starting before the
  modified model passes. It cannot establish full-frame equality. Replaying
  identical scratch inputs through passes 9/11/13 gave identical scratch
  and recurrent data; replaying the post stage with fixed exposure values
  0.03, 0.25, 0.7, 1, 2 and 8 gave identical output and history. The existing
  auto-exposure instability is not fixed by this optimization.
- The live Metal 4 encoder compiled for x86_64 and ran under Rosetta with
  output identical to the CLI (`live_test ... mtl4`, sharpness 0).

Refresh the game shader cache with `tools/precompile_i8mpp.sh` and restart
the game. The CLI rebuilds its shader variants automatically. Native kernel
entry points include `_tap4` to reject manifests with the old CT2D layout.

## Build

```
make
./build_shaders.sh        # compiles all presets/res (needs dxc + spirv-cross, see README)
```

Shaders are also JIT-compiled on demand by `fsr4mac` itself — a bare
`fsr4mac <testcase> <out> [--fp16]` works without `build_shaders.sh`.

## Flags

| flag | meaning |
|---|---|
| `--auto` | Default. Native INT8 on compatible M5-class GPUs, FP16 on older GPUs or after native initialization failure. |
| `--int8` | INT8 model path (original, DP4A-emulated). Explicit reference mode. |
| `--fp16` | FP16 model path (offline-dequantized weights, `gen_f16.py` + `fp16_ops.hlsli`). |
| `--i8mpp` | INT8 model passes as native Metal 4 tensor-ops kernels on the M5 neural accelerators (`src/i8mpp.metal`, `tools/gen_i8mpp.py`); post's ConvNextBlock + ConvTranspose included (pass 13, `_pt` post variant). Reference int8 numerics. Fastest. |
| `--bench N` | Run N iterations per frame over all 8 frames, print per-stage GPU ms and TOTAL. Skips PNG/bin writes and pass dumps. |
| `--reps R` | With `--bench`: encode each stage R times per command buffer and report the mean per dispatch (warm GPU clock; per-stage numbers are ±30% noisy with R=1). `--bench 3 --reps 10` is the recommended per-stage protocol. |
| `--dump-passes` | Dump the scratch buffer after every stage (`fNN_passPP_scratch.bin`) + constants + reprojected history. |
| `--caps` | Print device name, GPU families, Metal 4 runtime presence, and a functional `mpp::tensor_ops::matmul2d` compile+pipeline probe. |
| `--bench-tensorops` | Dispatch + time a 1024x1024x4096 fp16 GEMM via `mpp::tensor_ops::matmul2d` vs `simdgroup_matrix` (both verified vs a CPU ref first); prints GFLOP/s per path. Decides which MMA the convs should use. |
| `--builddir DIR` | Shader build dir (default `build/shaders`). |

## Benchmark recipe (on the M5)

```
./tools/bin/fsr4mac --caps                                  # device + matmul2d compile probe
./tools/bin/fsr4mac --bench-tensorops                       # timed matmul2d vs simdgroup GEMM
./tools/bin/fsr4mac testcases/testcase_quality_720to1080 /tmp/b_i8 --int8 --bench 20
./tools/bin/fsr4mac testcases/testcase_quality_720to1080 /tmp/b_f16 --fp16 --bench 20
```

Report the `TOTAL(per-pass sum)` line plus the per-stage table back.

## `--mpp` (opt-in)

Native `mpp::tensor_ops` matmul2d FNB kernels (`src/fnb_mpp.metal`,
manifest+transposed weights from gen_f16.py). **Measured slower than the
generated fp16 kernels on M5** (see VERIFICATION.md for both iterations) —
opt-in via `--mpp`, default off. Bench CSV stages get `_mpp` suffix.

## `--sync` modes

Inter-pass sync strategy (model passes only touch the scratch buffer; pre/post
also touch textures):

| mode | mechanism |
|---|---|
| `barrier` (default) | `memoryBarrierWithScope` after every dispatch — `Buffers` for model passes, `Buffers|Textures` for pre/post/spd/rcas |
| `encoder` | new compute encoder between passes in the same command buffer — driver hazard tracking inserts only needed fences |
| `none` | no sync — wrong results, upper-bound measurement only |

`--bench` reports both per-stage CB timings and a fused single-command-buffer
frame time (`FUSED_FRAME_<sync>` csv line + `fused frame gpu_ms`) so the
per-stage-sum vs fused gap is visible in one run.

GPU frequency scaling on M5 causes large run-to-run variation (observed int8
120->61 ms at identical binary/cached shaders) — only compare numbers from
the same run/session.

## M5 hotspot analysis (fp16, quality 720->1080, per-pass table)

Bytes moved vs measured ms vs ~273 GB/s — every pass is at 8-38 GB/s
(3-14% of peak): all passes are latency/compute-bound, NOT memory-bound.
Fusion/layout changes would not help; the generated kernels' serial
per-thread channel loops (each thread does a whole pixel's channels) are
the bottleneck. Lever: redistribute [pixel,channel] work across threads.

| pass | MB moved | M5 ms | GB/s |
|---|---|---|---|
| pre | ~64 | 1.77 | ~36 |
| pass1/2/12 CNB enc0 | 66 | 3.10/2.56/3.37 | 20-26 |
| pass4/5/10 FNB32 | 17 | ~1.15 | ~14 |
| pass7/8 FNB64 | 8 | ~1.05 | ~8 |
| post | ~83 | 3.50 | ~24 |

## Notes / caveats

- GPU timestamps come from `MTLCommandBuffer` `GPUStartTime`/`GPUEndTime` per
  dispatch. On the VirtualMac paravirtual GPU absolute numbers are unreliable
  (host calls are timed, not real silicon); treat them as directional only.
  The M5 is the perf authority.
- `--bench` keeps the full per-frame dependency chain (recurrent state), so it
  measures a realistic frame, not isolated kernels.
- Baseline measured on physical M5 @ 720→1080, int8 path: ~120 ms/frame.
- On this VM (VirtualMac2.1 paravirtual GPU, macOS 26.5.2): int8 ≈ 107 ms,
  fp16 ≈ 45 ms per frame (directional only).
- `--bench-tensorops` is the decisive check: on this VM matmul2d already
  measures 2.4 TFLOP/s vs simdgroup_matrix 0.64 TFLOP/s (both verified correct
  on a small GEMM first). If matmul2d >> simdgroup on the M5 too, the convs
  get ported to tensor_ops; parity means emulation and simdgroup_matrix wins.

## i8mpp (M5, 2026-09-23, same session)

```
./tools/bin/fsr4mac testcases/testcase_quality_720to1080 /tmp/o --i8mpp --bench 3 --reps 10 --builddir build/shaders
```

| case | int8 | fp16 | i8mpp (CNB-only pass 13) | i8mpp (CNB+CT2D pass 13), mean / min |
|---|---|---|---|---|
| quality 720->1080, fused frame | 47.5 ms | 12.7 ms | 5.95–6.3 ms | **5.57 / 4.97 ms** |
| performance 540->1080, fused frame | — | — | — | 5.41 / 4.95 ms |
| balanced 1000x648->1712x1112, fused frame | — | 11.1 ms | 5.8 ms | 5.06 / 4.74 ms |

Per-stage (quality 1080, `--bench 10 --reps 10`): pass1/2/12 0.37–0.41 ms,
pass3/6 0.06–0.08, pass4/5/10 0.22–0.24, pass7/8 0.16–0.17, pass9 0.34,
pass11 0.42, pass13 (post CNB+CT2D) 0.61, post 0.86, pre 0.68 (sum 5.20). Accuracy vs DX12 golden: quality 47.1/44.0/41.9 dB (f00/f03/f07),
perf 48.7/42.7/41.9 dB; model passes 1–9 bit-exact against the golden
per-pass dumps on frame 0 (frame 1: 4/7/10 have 1–3 single-LSB flips). Single passes: `tools/bench/i8_test <gendir>
<dumpdir> 0 <pass> 5`.

## Machine results log

| machine | chip/os | tool | result |
|---|---|---|---|
| user's machine | Apple M5, macOS 27.0 | `--bench` quality 720→1080 | int8 120 ms/frame, fp16 43.2 ms (2.78x); post 14.6, pass12 4.5, pass2 3.6, pass1 3.1, pre 2.8, other convs 0.6–2.4; whole-CB mean 44.9 ms |
| user's machine | M5 | fp16 vs golden | f07 R41.57/G43.75/B42.48 dB; f00 R 39.28 dB; per-frame 40.5–42.5 dB |
| user's machine | M5, macOS 27.0 | `--bench-tensorops` | matmul2d 2099 GFLOP/s vs simdgroup 281 GFLOP/s (7.5x) — neural accelerator path confirmed |
| user's machine | M5 | fp16 `--bench` post-fix | per-stage sum 23.4 ms (post 3.5), fused CB 38.2 ms — ~15 ms dispatch/sync overhead under investigation |
| this VM | VirtualMac2.1 paravirtual, macOS 26.5.2 | `--bench` | int8 107 ms, fp16 45.2 ms per-pass sum |
| this VM | paravirtual | `--bench-tensorops` | matmul2d 2388 GFLOP/s, simdgroup_matrix 639 GFLOP/s |

## Lane-parallel CNB (`cnb_mpp` kernel)

`src/fnb_mpp.metal` `cnb_mpp`: one 32-lane simdgroup per output pixel
(4 pixels / 128-thread threadgroup). conv1 3x3 (16->16): lanes split
K=8+8, `simd_shuffle_xor(16)` reduce. conv2 1x1 (16->32): lane = output,
dot over 16. conv3 1x1 (32->16): lanes split K=16+16, xor reduce,
bias + residual + requant. Requant semantics identical to the generated
`f16_ConvNextBlock`. Applies to passes 1/2/12 (CNB16 shape guard in the
emitter; other CNB shapes stay on the generated kernel).

M5 result (`9457a84`): lane-parallel loses 4.6-7.9x — pass1 15.8 vs
1.99, pass2 13.9 vs 2.53, pass12 15.7 vs 3.41 ms; frame 58.7 vs 22.8.
The generated kernel's serial channel loops beat the split-K + shuffle
reduction: 32x more threads at ~1/32 work each exposes launch/shuffle
overhead, and the stride-scattered weight reads do not coalesce. Do NOT
roll out to FNB/9/11. Kept behind `--cnb` as a documented dead end.
