# v3.0: heat-aware full-frame M5 model measurements

Date: 2026-11-02. Machine: MacBook Air M5 (Mac17,4), 16 GiB, fanless.
OS: Version 27.0.1 (Build 26A434). Battery power throughout (41% -> 35%); Low Power Mode off.

These are **whole production upscaler command-buffer GPU durations**, not
CPU encode/compilation time, game frame time or FPS. All variants use native
INT8 MPP, classic Metal, linear colour, inverted depth, low-resolution motion
vectors, auto-exposure on, RCAS off and stable pre-exposure 1.0.

The fixture is deterministic resident synthetic data: 0x4128 LCG colour/noise,
stepped depth, nonzero motion and eight jitter phases. Identical textures and
coordinates are supplied to every model. This is not a recorded game or an
image-quality comparison. 1080p/1440p refer to **output resolution**.

## Protocol

- Compile/prepare the external models and every pipeline before timing.
- Warm for at least 120 seconds on the first pass of each resolution.
  Repeat passes warm for at least 30 seconds; no cooling pause is substituted.
- Resolution order: **1080p, 1440p, 1440p, 1080p**. Models execute serially
  in one process for each pass. Rotate the first model and reverse direction
  per frame/round, including during warmup.
- Two passes per resolution, six rounds per pass, 64 frames per model/round:
  **12 rounds and 768 measured frames per model/resolution**.
- Headline: median of all 12 round means, not pooled frame median, the best
  round, or only the first/cold run. Ranges are minimum/maximum **round means**.
- P99 uses all 768 individual frames. Full/reuse/partial costs are arithmetic
  means of the corresponding frames, so they need not equal the headline.
- There are no measured resets. Eligible whole-frame profiles refresh/reuse
  50:50 (384 full + 384 reused frames). Tiled has 768 partial frames.
  Every other model runs full inference on all 768 frames.
- Pipeline creation, GPU layout validation, compilation and all warmup frames
  are excluded. No stage counters or capture/readback copies are added.
- Optional 4.0.2 uses its **production live encoder**, including its small
  built-in diagnostic GPU probe. All other rows use the production Runtime
  used by the CLI. The legacy row is labelled separately, not a clean ablation.
- A nonblocking exclusive lock at `/tmp/fsr4-gpu-timing.lock` prevents a second
  copy of this timing harness. No other benchmark/test or game ran concurrently.
  Normal desktop/background applications stayed open.

## 1080p: 1280x720 -> 1920x1080

| Variant | Median round mean | Round range | Frame P99 | Full mean | Reuse mean | Partial mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `exact-fast` | 4.523 | 4.099-5.030 | 6.484 | 4.554 | - | - |
| `fsr4-legacy` | 5.483 | 5.008-6.128 | 7.477 | 5.492 | - | - |
| `lossy2` | 3.067 | 2.749-3.347 | 6.121 | 4.519 | 1.608 | - |
| `int8-lite` | 2.778 | 2.516-3.132 | 4.600 | 2.797 | - | - |
| `int8-lite-pixfix` | 2.819 | 2.517-3.149 | 4.491 | 2.812 | - | - |
| `int8-lite-lossy` | 2.963 | 2.693-3.229 | 4.568 | 3.494 | 2.441 | - |
| `exact` | 5.695 | 5.190-6.420 | 7.677 | 5.760 | - | - |
| `lossy` | 3.028 | 2.721-3.317 | 5.862 | 4.460 | 1.589 | - |
| `lossy2-temporal1` | 4.625 | 4.198-5.040 | 6.939 | 5.177 | 4.083 | - |
| `lossy2-temporal2` | 4.138 | 3.782-4.699 | 6.647 | 4.713 | 3.654 | - |
| `int8-lite-lossy-tiled` | 3.730 | 3.481-4.026 | 5.358 | - | - | 3.739 |
| `int8-lite-lossy-original` | 8.636 | 7.918-9.617 | 14.338 | 11.649 | 5.693 | - |

## 1440p: 1706x960 -> 2560x1440

| Variant | Median round mean | Round range | Frame P99 | Full mean | Reuse mean | Partial mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `exact-fast` | 8.723 | 8.425-8.929 | 10.416 | 8.677 | - | - |
| `fsr4-legacy` | 10.467 | 10.134-10.772 | 12.467 | 10.462 | - | - |
| `lossy2` | 5.728 | 5.499-5.955 | 10.200 | 8.475 | 3.003 | - |
| `int8-lite` | 5.368 | 5.199-5.666 | 7.172 | 5.381 | - | - |
| `int8-lite-pixfix` | 5.347 | 5.134-5.572 | 7.091 | 5.350 | - | - |
| `int8-lite-lossy` | 5.843 | 5.603-5.938 | 8.647 | 6.914 | 4.632 | - |
| `exact` | 11.032 | 10.752-11.425 | 12.901 | 11.070 | - | - |
| `lossy` | 5.823 | 5.588-5.964 | 10.168 | 8.566 | 3.003 | - |
| `lossy2-temporal1` | 8.935 | 8.618-9.153 | 11.375 | 9.900 | 7.891 | - |
| `lossy2-temporal2` | 7.978 | 7.759-8.266 | 10.553 | 8.965 | 7.032 | - |
| `int8-lite-lossy-tiled` | 7.016 | 6.636-7.198 | 8.716 | - | - | 6.993 |
| `int8-lite-lossy-original` | 10.866 | 10.438-11.275 | 13.774 | 10.008 | 11.725 | - |

## Heat and session drift

Every measured round started and ended at **NSProcessInfo thermal state 1
(fair)**. The first warmup began at nominal (0); no serious/critical state
was observed. No privileged temperature or GPU-frequency sampler was available.
A fair/nominal state **does not prove the absence of clock throttling**.

| Pass | Output | Warmup | Measured wall time | Measured thermal state | Battery |
| --- | --- | ---: | ---: | --- | --- |
| 1 | 1080p | 120.3 s | 19.3 s | 1 (fair) | 41% -> 39% |
| 2 | 1440p | 120.4 s | 36.3 s | 1 (fair) | 39% -> 37% |
| 3 | 1440p | 30.3 s | 37.7 s | 1 (fair) | 37% -> 36% |
| 4 | 1080p | 30.1 s | 22.7 s | 1 (fair) | 36% -> 35% |

Within each pass, early/late round drift was small (recorded for every row
in the JSON). **Between passes the later 1080p series was slower**, even
though the coarse thermal-pressure state remained fair. Plain Lite increased
by 19.9%. Both passes are included, not discarded or replaced with the faster one.
This captures an observed warm-to-hot/background-load envelope; without
temperature/frequency counters the drift cannot be attributed solely to heat.
It is not evidence that the implementation became slower between builds.

| Variant | Earlier -> later 1080p series median | Earlier -> later 1440p series median |
| --- | ---: | ---: |
| `exact-fast` | 4.149 -> 4.986 (20.2%) | 8.512 -> 8.825 (3.7%) |
| `fsr4-legacy` | 5.020 -> 5.896 (17.4%) | 10.288 -> 10.633 (3.4%) |
| `lossy2` | 2.797 -> 3.326 (18.9%) | 5.637 -> 5.873 (4.2%) |
| `int8-lite` | 2.543 -> 3.048 (19.9%) | 5.248 -> 5.508 (5.0%) |
| `int8-lite-pixfix` | 2.557 -> 3.050 (19.3%) | 5.244 -> 5.467 (4.3%) |
| `int8-lite-lossy` | 2.758 -> 3.176 (15.1%) | 5.642 -> 5.862 (3.9%) |
| `exact` | 5.250 -> 6.263 (19.3%) | 10.923 -> 11.170 (2.3%) |
| `lossy` | 2.763 -> 3.270 (18.3%) | 5.629 -> 5.912 (5.0%) |
| `lossy2-temporal1` | 4.256 -> 5.019 (17.9%) | 8.761 -> 9.029 (3.1%) |
| `lossy2-temporal2` | 3.811 -> 4.545 (19.3%) | 7.854 -> 8.122 (3.4%) |
| `int8-lite-lossy-tiled` | 3.526 -> 3.942 (11.8%) | 6.902 -> 7.116 (3.1%) |
| `int8-lite-lossy-original` | 7.985 -> 9.337 (16.9%) | 10.666 -> 11.083 (3.9%) |

## What these numbers do not establish

- They do not rank image quality or prove that Lite Lossy is better/faster
  than Lite. In this fixture Lite is faster at both resolutions.
- A reused frame is not always cheaper: the original-graph route retains its
  original POST on reused frames, while its fast POST only runs on full frames
  for the proven permutation. 1440p reuse therefore costs more in that row.
- They do not measure a game, a Windows GPU/driver, M1-M4 performance, or a
  continuous single-mode gaming session. Interleaving intentionally balances
  order/heat across variants, with different cache traffic than one model alone.
- The CLI Tiled/Original and temporal-quality options are not selectable
  panel modes. Ordinary Lite POST colour tiling is a separate implementation
  detail; removing Lite Lossy Tiled from the panel does not remove it.

## Reproduce from a source checkout

Required external inputs: ordinary Lite import, `make lossy2-weights`,
`make lite-lossy-weights`, and the verified original-shader bundle described
in [FSR412_UPSTREAM_ORACLE.md](FSR412_UPSTREAM_ORACLE.md). To include legacy
4.0.2, its model weights must also be present. Those optional/GPL inputs are
not included in the MIT ZIP/DMG. No Windows DLL is executed.

Build the harness, then prime each resolution once (allowing compilation):

```sh
make build/tests/fsr_modes_thermal
build/tests/fsr_modes_thermal "$PWD" --size 1280 720 1920 1080 --legacy --warmup-seconds 0 --rounds 2 --frames-per-round 8 --report build/prime-1080.json
build/tests/fsr_modes_thermal "$PWD" --size 1706 960 2560 1440 --legacy --warmup-seconds 0 --rounds 2 --frames-per-round 8 --report build/prime-1440.json
python3 tools/bench/benchmark_modes_thermal.py --legacy --out build/v3-thermal-benchmark
```

For a cold original-bundle build, set `FSR412_DXIL_SPIRV` to the converter.
Omit `--legacy` consistently if its weights are unavailable. Do not run any
other GPU benchmark/test concurrently. The runner refuses a nonempty output
directory and validates the backend, frame counts, cadence and timing reports.
Use `--dry-run` to inspect the schedule without starting the GPU.

[Portable numeric results and thermal samples](reports/m5-modes-v3-2026-11-02.json).
Raw per-frame reports/logs remain under the gitignored build directory.
Benchmark source SHA-256: `3f79c550b56ae23685b7e9dcb580adea2b4d02fde21a979091023ef9f4dafda3`.

v3.0 does not change model weights, neural kernels or default profile from v2.6.
