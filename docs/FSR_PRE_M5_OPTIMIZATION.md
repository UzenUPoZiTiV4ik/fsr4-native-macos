# Pre-M5 SIMD optimization

The M1-M4 backend executes the same INT8 models through Metal 3 half
simdgroup matrices with exact float accumulation. It does not require M5's
INT8 tensor instructions and does not replace the model with a floating-point
network.

## Changes

- `src/fsr411_simd.metal`: port the M5 MPP vector integer quantizer to the
  two adjacent channels owned by a SIMD lane. Unsigned wrapping, ties-to-even
  / explicit Lossy half-up rounding, residuals and int16 narrowing are retained.
- FNB/CT stage only the convolved channel banks with a 3x3 halo. Bypass banks
  are staged only for centre pixels. This removes unused loads/conversions
  and reduces threadgroup storage:

  | Block | Before | After | Reduction |
  | --- | ---: | ---: | ---: |
  | NF32 FNB/CT | 13056 bytes | 10624 bytes | 18.6% |
  | NF64 FNB/CT | 13824 bytes | 11008 bytes | 20.4% |

- `src/fsr411_runtime.h`: extend Lite's existing fixed-thread pipeline limit
  to all SIMD models. Compile for the actual 128 threads rather than an unused
  1024-thread group; check SIMD32 and static shared-memory requirements.
- `tools/fsr412/native_model.py`: publish/check per-output-column float
  accumulator bounds for Lite, as the Exact/Lossy generator already does.
  Maximum absolute INT8 dot bounds are 410112 Normal and 636672 Ultra,
  both below `2^24`. Biases and rounding still run in int32.

Weights, parameter bytes, neural/temporal graph, inference reuse policy,
M5 MPP kernels and automatic backend selection are unchanged. Existing
SIMD entry names retain their original dispatch shapes, so old model caches
remain compatible. Experimental MB overrides are benchmark-only; production
keeps MB16=4, MB32=4, MB64=2.

Naming: AMD Lite is `--fsr-version 4.1.2 --profile int8-lite`. Exact uses
`--fsr-version 4.1.1 --profile exact`. The retrained upstream 4.1.2 Lossy
keeps the internal name `--fsr-version 4.1.1 --profile lossy2`.

## Exploratory timing, 2026-10-30

Physical machine: Apple M5, macOS 27.0.1, forced `--simd`, battery power
at approximately 21-23%, other desktop applications running. **These are not
M1-M4 measurements, and the timing spread is too large for a firm speed claim.**

Input: `testcases/tc_game_balanced`, 1000x648 -> 1712x1112. Four serial rounds,
reversed variant order on alternating rounds, one warmup sequence and
20 measured two-frame sequences per invocation. No compilation, diagnostic
counters or copies in the timed command buffer. Both versions use the same
current host pipeline limit, caches and weights; this isolates shader changes,
not the separate host pipeline-limit change.

Median of four round means, complete GPU command buffer:

| Model | Before | After | Observed change |
| --- | ---: | ---: | ---: |
| Lite | 6.559 ms | 5.464 ms | -16.7% |
| Exact | 11.764 ms | 10.770 ms | -8.5% |
| Lossy 4.1.2 | 6.278 ms | 5.848 ms | -6.8% |

Round ranges overlap substantially. One Exact candidate round was slower
than its baseline. Lossy's sequence has one reset/inference frame and one
reuse frame: its all-frame mean is not an inference-only latency and does not
measure non-reset inference. No reuse count changes are attributed to the
optimization. [Recorded round summaries](reports/fsr-pre-m5-2026-10-30.json).

For reliable acceptance, repeat on physical M1/M2/M3/M4, on AC power,
without competing GPU work and with a longer capture containing both full
inference and reused steady-state frames.

## Exact Winograd 3x3, 2026-10-31

CNB16 and the convolved bank of FNB32 now use Winograd F(2x2,3x3) on the
same simdgroup half matrices. It needs 16 matrix products per bank instead
of 36 and is **exact, not an approximation**:

- input transform `B^T d B` is integral, |v| <= 512, exact in half;
- weights are stored as integers `T4 = H g H^T = 4 G g G^T`,
  |T4| <= 1152, exact in half (`src/fsr411_simd_weights.h`);
- before use, the host computes `512 * sum|T4|` for every output channel
  and output quarter. It must be `< 2^24`, which bounds every float partial
  sum in any order. Otherwise that pass keeps the direct 3x3 kernel;
- the full inverse equals `4 * dot` exactly; it is divided by four once,
  then the unchanged INT32 bias/rounding/clamp epilogue runs.

All real models pass the bound, with at least 1.68x margin: Lite and Lite
Ultra, Exact, retrained Lossy 4.1.2 (weights are identical in all tiers;
the GPU tests cover 1080 and 2160). CT32 and NF64 blocks keep
direct 3x3: Winograd measured 2-4% slower in CT32. Weights, manifest, scratch
layout, dispatch tiles and cache keys are unchanged. The runtime enables it
only after an on-device probe of the larger transformed products;
`FSR4_SIMD_WINOGRAD=0` (or `false`/`off`/`no`), set before launch,
disables it. The log line
`exact Winograd 3x3 in N of M neural passes` shows when it is active.

Forced `--simd` on Apple M5 (not M1-M4), one process, direct and Winograd
runtimes interleaved frame by frame, 30 rounds, median frame with inference.
Output, history and recurrent state were byte-identical:

| Model | 1000x648 -> 1712x1112 | 1280x720 -> 1920x1080 |
| --- | ---: | ---: |
| Lite 4.1.2 | 6.46 -> 5.51 ms (-15.6%) | 6.08 -> 4.96 ms (-14.7%) |
| Exact | 14.37 -> 12.19 ms (-14.4%) | 13.09 -> 11.11 ms (-15.0%) |
| Lossy 4.1.2 | 13.37 -> 10.98 ms (-15.6%) | 11.84 -> 10.00 ms (-15.3%) |

Lossy frames that reuse inference do not run these kernels and are unchanged.
[Recorded runs](reports/fsr-pre-m5-winograd-2026-10-31.json). M1-M4 have
different matrix throughput, and the CNB/FNB/CT selection above was
measured only on M5. Measure there with:

```sh
make bench-fsr-pre-m5
```

Examined but not enabled: removing the padding dispatches (0.2-0.3% of the
frame) and an INT8-staged downsample (`*_compact`, half the threadgroup
memory, exact, no gain on M5; worth measuring on M1-M4 with
`fsr411_simd_bench --compact-ds`).

## Validation and use

Import Lite and obtain the trained Lossy weights first, as described in the
project README. Then:

```sh
make test
make test-fsr-pre-m5
make dist/crossover/libfsr4hook.dylib
```

`test-fsr-pre-m5` forces SIMD even on M5 and runs sequentially:

- Winograd: CPU proof of the transform and bound gate (9.8 million exact
  checks), then GPU results before quantization and full scratch bytes
  against the CPU integer oracle for every real model (`test-fsr411-simd-winograd`).

- Actual production `finish2` vs independent CPU integer arithmetic:
  251651 packed cases, 1006604 channel values, all flags/shifts, int32 wrap,
  rounding ties, residuals, int16 narrowing and output guards.
- Exact, legacy Lossy and retrained Lossy 4.1.2 full-graph parity, Normal and
  Ultra Performance, cropped textures, odd sizes, motion/jitter permutations,
  a multi-tile frame and tier 2160. Exact Fast retains its pre-existing relaxed
  PRE/POST tolerance.
- Lite Normal, auto-exposure/subrect/resets/RCAS/size changes and Ultra,
  on classic Metal and Metal 4. SIMD with the non-cached color transform
  matches translated AMD shaders bit for bit, including temporal state.
  The pre-existing prepared-color optimization retains its small float
  tolerance; no new tolerance is introduced for neural computations.

Isolated block checks also passed against the CPU oracle for Exact,
retrained Lossy 4.1.2 and Lite: 124 cases, comparing all scratch bytes,
including dirty padding and odd boundary tiles. These are correctness checks,
not performance measurements.

The built x86_64 CrossOver hook also passed forced-SIMD tests under Rosetta:
40 frames across classic Metal / Metal 4 and both motion resolutions,
including subrect sentinels, owner lifecycle and Exact/Lossy switches. This
integration check is available as `make test-fsr411-live-simd`.

The isolated benchmark can run without imported PRE/POST shaders or a DLL:

```sh
make build/tests/fsr411_simd_bench
build/tests/fsr411_simd_bench . build/fsr412_native_1080
build/tests/fsr411_simd_bench . build/fsr411_native_2160_lossy2
```

It measures specialized neural passes only, not full-frame latency. Options
`--legacy-pipeline`, `--mb16`, `--mb32`, `--mb64` and `--rounds` support
controlled experiments. Do not use an MB-overridden library with an ordinary
runtime manifest, because its dispatch tile would no longer match.

On M1-M4, `FSR4_MODEL=auto` selects SIMD automatically; `FSR4_MODEL=simd`
forces it. For manual CLI testing use `--simd`. Rebuild/reinstall the CrossOver
hook and restart the game to load host changes. Building the hook alone does
not modify installed bottles or other project checkouts.
