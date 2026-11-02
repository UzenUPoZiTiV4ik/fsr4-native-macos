# v3.0: heat-aware M5 MPP and forced SIMD measurements

Date: 2026-11-02. MacBook Air M5 (Mac17,4), 16 GiB, fanless. Version 27.0.1 (Build 26A434).
Battery power throughout; Low Power Mode off. MPP series: 41% -> 35%;
forced SIMD series: 30% -> 23%. The backends were collected in separate
sessions, not interleaved with each other.

**Scope:** whole production upscaler command-buffer GPU durations, not
CPU encode/compilation time, game frame time or FPS. Linear colour, inverted
depth, low-resolution motion, auto-exposure on, RCAS off, pre-exposure 1.0.
MPP uses native INT8 tensor operations; forced SIMD uses half matrices with
float accumulation and the same INT8 epilogue. The integer model is retained.
Only the optional 4.0.2 SIMD row uses the distinct real FP16 legacy path.

**Both sets are physical M5 measurements, not M1-M4 timings.** Forcing SIMD
does not change the GPU family: M5-specific POST colour tiling stays active.
This is not an emulation of the entire M1-M4 route or a prediction for it.

The fixture is resident deterministic data: 0x4128 LCG colour/noise, stepped
depth, nonzero motion and eight jitter phases. Every model gets the same
textures and coordinates. This is not a game capture or image-quality test.
1080p/1440p refer to **output resolution**.

## Protocol

- Prime every external model and pipeline before timing.
- Initial warmup: at least 120 s per backend/resolution. Repeat warmup: at
  least 30 s. Compilation, pipeline creation, allocations and all warmup
  frames are excluded. No deliberate cooling pause replaces sustained warmup.
- For each backend: **1080p, 1440p, 1440p, 1080p**. In each process models
  execute serially; the first model rotates and direction reverses per frame/round.
- Per model/backend/resolution: two passes x six rounds x 64 frames =
  **12 round means and 768 measured frames**.
- Headline: median of all 12 round means, not best/cold run or pooled frame
  median. Range: minimum/maximum **round means**, not fastest/slowest frame.
- P99 includes all 768 individual frames. Full/reuse/partial costs are
  arithmetic means of those groups, and need not equal the headline median.
- No measured resets. Eligible whole-frame modes use 384 full + 384 reused
  frames; tiled has 768 partial frames; other modes have 768 full frames.
- Actual backend labels are checked at encode and in the JSON. Forced SIMD
  is not substituted with MPP/scalar. Device probes/internal guarded direct
  kernels remain part of the normal backend policy.
- No stage counters or capture/readback copies in timed buffers. Legacy
  4.0.2 uses its actual live encoder, including its small built-in GPU probe.
  That row is a labelled legacy comparison, not a clean numerical ablation.
- `/tmp/fsr4-gpu-timing.lock` is nonblocking/exclusive. No game or other GPU
  benchmark/test ran concurrently; normal desktop/background apps stayed open.

## MPP 1080p: 1280x720 -> 1920x1080

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

## MPP 1440p: 1706x960 -> 2560x1440

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

## Forced SIMD 1080p: 1280x720 -> 1920x1080

| Variant | Median round mean | Round range | Frame P99 | Full mean | Reuse mean | Partial mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `exact-fast` | 9.914 | 9.102-11.560 | 12.675 | 10.051 | - | - |
| `fsr4-legacy` | 13.217 | 12.205-15.303 | 16.484 | 13.419 | - | - |
| `lossy2` | 5.652 | 5.255-6.546 | 12.487 | 10.025 | 1.564 | - |
| `int8-lite` | 4.790 | 4.365-5.533 | 7.154 | 4.814 | - | - |
| `int8-lite-pixfix` | 4.709 | 4.334-5.487 | 7.038 | 4.821 | - | - |
| `int8-lite-lossy` | 3.870 | 3.596-4.477 | 7.728 | 5.510 | 2.352 | - |
| `exact` | 10.953 | 10.090-12.837 | 13.936 | 11.160 | - | - |
| `lossy` | 5.692 | 5.185-6.568 | 12.679 | 10.048 | 1.524 | - |
| `lossy2-temporal1` | 7.159 | 6.615-8.354 | 13.283 | 10.682 | 3.914 | - |
| `lossy2-temporal2` | 6.794 | 6.195-7.879 | 12.724 | 10.272 | 3.515 | - |
| `int8-lite-lossy-tiled` | 5.182 | 4.799-5.956 | 7.427 | - | - | 5.238 |
| `int8-lite-lossy-original` | 9.157 | 8.463-10.390 | 15.914 | 13.233 | 5.347 | - |

## Forced SIMD 1440p: 1706x960 -> 2560x1440

| Variant | Median round mean | Round range | Frame P99 | Full mean | Reuse mean | Partial mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `exact-fast` | 18.222 | 17.558-18.861 | 20.943 | 18.187 | - | - |
| `fsr4-legacy` | 24.122 | 23.080-24.882 | 27.806 | 24.097 | - | - |
| `lossy2` | 10.499 | 10.008-10.761 | 20.460 | 18.073 | 2.775 | - |
| `int8-lite` | 8.661 | 8.405-8.993 | 10.926 | 8.682 | - | - |
| `int8-lite-pixfix` | 8.754 | 8.335-8.940 | 10.731 | 8.680 | - | - |
| `int8-lite-lossy` | 7.211 | 6.920-7.406 | 11.791 | 10.060 | 4.375 | - |
| `exact` | 20.238 | 19.478-21.016 | 24.305 | 20.227 | - | - |
| `lossy` | 10.549 | 10.072-10.793 | 22.078 | 18.195 | 2.745 | - |
| `lossy2-temporal1` | 13.323 | 12.810-13.653 | 21.957 | 19.352 | 7.167 | - |
| `lossy2-temporal2` | 12.504 | 12.076-12.914 | 21.312 | 18.565 | 6.396 | - |
| `int8-lite-lossy-tiled` | 9.410 | 9.218-9.765 | 11.474 | - | - | 9.445 |
| `int8-lite-lossy-original` | 11.849 | 11.272-12.203 | 15.012 | 12.898 | 10.667 | - |

## Heat and drift

MPP measured rounds started/ended at **fair (1)**. The first SIMD measured
series spanned **nominal (0)/fair (1)**; later SIMD rounds were fair.
No serious/critical state was observed. These are NSProcessInfo thermal
pressure levels, not temperature or clock-frequency sensor readings.
**Nominal/fair does not prove the absence of clock throttling.**

| Backend/pass | Output | Warmup | Measured wall time | Measured thermal state | Battery |
| --- | --- | ---: | ---: | --- | --- |
| MPP 1 | 1080p | 120.3 s | 19.3 s | fair | 41% -> 39% |
| MPP 2 | 1440p | 120.4 s | 36.3 s | fair | 39% -> 37% |
| MPP 3 | 1440p | 30.3 s | 37.7 s | fair | 37% -> 36% |
| MPP 4 | 1080p | 30.1 s | 22.7 s | fair | 36% -> 35% |
| SIMD 1 | 1080p | 120.2 s | 32.2 s | nominal/fair | 30% -> 27% |
| SIMD 2 | 1440p | 120.9 s | 59.6 s | fair | 27% -> 25% |
| SIMD 3 | 1440p | 30.5 s | 62.1 s | fair | 25% -> 24% |
| SIMD 4 | 1080p | 30.3 s | 38.1 s | fair | 24% -> 23% |

Within individual passes early/late drift was small and is saved for each
row in the JSON. Between passes the later 1080p Lite series was **19.9%
slower in MPP and 17.8% slower in SIMD**, despite the coarse pressure state.
Both passes remain included; none was discarded in favour of a prettier
number. This is an observed warm-to-hot/background-load envelope. Without
temperature/frequency sensors, heat cannot be isolated as its sole cause.
It is not a regression comparison between builds.

### MPP: earlier -> later series

| Variant | 1080p pass medians (change) | 1440p pass medians (change) |
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

### SIMD: earlier -> later series

| Variant | 1080p pass medians (change) | 1440p pass medians (change) |
| --- | ---: | ---: |
| `exact-fast` | 9.171 -> 10.747 (17.2%) | 17.764 -> 18.564 (4.5%) |
| `fsr4-legacy` | 12.255 -> 14.481 (18.2%) | 23.677 -> 24.573 (3.8%) |
| `lossy2` | 5.294 -> 6.301 (19.0%) | 10.115 -> 10.660 (5.4%) |
| `int8-lite` | 4.398 -> 5.180 (17.8%) | 8.495 -> 8.857 (4.3%) |
| `int8-lite-pixfix` | 4.431 -> 5.267 (18.9%) | 8.458 -> 8.904 (5.3%) |
| `int8-lite-lossy` | 3.630 -> 4.151 (14.4%) | 7.123 -> 7.327 (2.9%) |
| `exact` | 10.160 -> 12.100 (19.1%) | 19.783 -> 20.599 (4.1%) |
| `lossy` | 5.275 -> 6.324 (19.9%) | 10.213 -> 10.712 (4.9%) |
| `lossy2-temporal1` | 6.639 -> 7.947 (19.7%) | 12.954 -> 13.586 (4.9%) |
| `lossy2-temporal2` | 6.287 -> 7.540 (19.9%) | 12.183 -> 12.717 (4.4%) |
| `int8-lite-lossy-tiled` | 4.906 -> 5.522 (12.5%) | 9.310 -> 9.598 (3.1%) |
| `int8-lite-lossy-original` | 8.549 -> 10.013 (17.1%) | 11.498 -> 12.058 (4.9%) |

## Limits and interpretation

- MPP and SIMD were collected separately. Big differences are useful
  observations on this fixture, not a tightly paired cross-backend test.
- In this fixture Lite is faster than Lite Lossy in MPP. In forced SIMD,
  Lite Lossy is about 19% faster at 1080p and 17% faster at 1440p.
  This does not rank image quality or predict which looks best in a game.
- Tiny overlapping differences, e.g. with/without the Pixelization fix, are
  noise, not a demonstrated optimization. Fix strength here is 0.5.
- A reused original-route frame can cost more: its original POST remains
  on reuse while the proven fast POST is run-frame-only. See the group costs.
- This is not a continuous single-mode gaming session, a Windows driver
  benchmark, or physical M1-M4 validation. Interleaving changes cache traffic
  compared with one mode alone; other scenes can have different POST work.
- Tiled/Original/temporal-quality options remain CLI-only panel-wise.
  Ordinary Lite POST colour tiling is separate and is not removed with
  Lite Lossy Tiled from the panel.

## Reproduce from source

External inputs: ordinary Lite import, `make lossy2-weights`,
`make lite-lossy-weights`, and the original bundle described in
[FSR412_UPSTREAM_ORACLE.md](FSR412_UPSTREAM_ORACLE.md). `--legacy` also
requires the old 4.0.2 weights. Optional/GPL inputs are not in ZIP/DMG,
and Windows DLLs are not executed.

Build and prime both sizes/backends once, allowing compilation:

```sh
make build/tests/fsr_modes_thermal
build/tests/fsr_modes_thermal "$PWD" --size 1280 720 1920 1080 --legacy --warmup-seconds 0 --rounds 2 --frames-per-round 8 --report build/prime-mpp-1080.json
build/tests/fsr_modes_thermal "$PWD" --size 1706 960 2560 1440 --legacy --warmup-seconds 0 --rounds 2 --frames-per-round 8 --report build/prime-mpp-1440.json
build/tests/fsr_modes_thermal "$PWD" --simd --size 1280 720 1920 1080 --legacy --warmup-seconds 0 --rounds 2 --frames-per-round 8 --report build/prime-simd-1080.json
build/tests/fsr_modes_thermal "$PWD" --simd --size 1706 960 2560 1440 --legacy --warmup-seconds 0 --rounds 2 --frames-per-round 8 --report build/prime-simd-1440.json
python3 tools/bench/benchmark_modes_thermal.py --legacy --out build/v3-thermal-benchmark
python3 tools/bench/benchmark_modes_thermal.py --simd --legacy --out build/v3-simd-benchmark
```

A cold original-bundle build needs `FSR412_DXIL_SPIRV` set to the converter.
Omit `--legacy` consistently if its data is unavailable. Run no other GPU
benchmark/test concurrently. The runner refuses a nonempty output directory
and validates counts, cadence, backend and timing reports. `--dry-run`
prints the schedule without starting the GPU.

## Reports and source identity

[MPP report](reports/m5-modes-v3-2026-11-02.json)
| [SIMD report](reports/m5-modes-v3-simd-2026-11-02.json).
Raw per-frame files/logs are retained under gitignored build directories.

MPP was captured with the native-only harness snapshot in
[805d0d8](https://github.com/UzenUPoZiTiV4ik/fsr4-native-macos/blob/805d0d8045c41a61d67a794f1e3937589d73c510/tools/bench/fsr_modes_thermal.mm).
The later harness adds explicit `--simd` support; source hashes are retained
separately rather than relabelling the earlier measurements:
- MPP source SHA-256: `3f79c550b56ae23685b7e9dcb580adea2b4d02fde21a979091023ef9f4dafda3`.
- SIMD source SHA-256: `37331bc73d7377305fbbc41708afddff9eafe97b99a1afd8737fbfd0ceb3470b`.

v3.0 changes the panel/reporting, not neural kernels, weights, default profile or hook ABI.
