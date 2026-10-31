# Apple Silicon backends

FSR **4.1.1 + Lossy** is the current default. Profile selection (Exact/Lossy)
is independent of backend selection. All backends execute the actual INT8
model, including its separate Ultra Performance initializer.

| GPU family | Automatic FSR 4.1.1 backend |
| --- | --- |
| M1 / family 7 | `simd`: simdgroup half matrices, exact INT8 result. |
| M2 / family 8 | `simd` |
| M3–M4 / family 9 | `simd` |
| M5 / family 10+, macOS 26+ | Native INT8 tensor operations (`i8mpp`). |
| M6 | `i8mpp` (INT8, not FP8; see below). |
| Non-Apple7 GPU | Scalar INT8 Metal compute. |

Selection uses Metal feature families and device limits. Rosetta does not
change GPU capabilities. Native kernels use 128 threads and 20 KiB of
threadgroup memory. This project builds for macOS 26+ with Xcode 26+ and
the Metal Toolchain.

The `simd` backend (`--simd`, alias `--fp16`; `FSR4_MODEL=simd`) runs the
INT8 convolutions as 8×8 `simdgroup_half8x8` products. INT8 activations and
weights are exact in half, each 8×8 product is exact, and accumulation is in
float: every model accumulator stays below 2^24 (max 474,880), so the sum is
exact and the integer epilogue is unchanged. Output is byte-identical to
`--scalar` (runtime parity suite `make test-fsr411-simd`, native block
tests, CLI frames). Half weights are a widened copy of the INT8 blob; the
simdgroup lane layout is probed once per device, with scalar fallback if
it differs.

The 3×3 convolutions with 16 and 32 channels (CNB, FNB32, CT32) use exact
integer Winograd F(2×2,3×3): 16 products per 2×2 output instead of 36. The
generator stores integer `(2G) g (2G)ᵀ` weights (int16, max 390) and proves
every Winograd accumulator stays below 2^24 (max 1,174,016), so output stays
byte-identical. A second device probe checks exact float accumulation for
`|a| <= 512`, `|b| <= 2048`; on failure the direct kernels are used.
`FSR411_SIMD_WINOGRAD=0` forces the direct kernels. The 64-channel and
downsample convolutions stay direct.

FSR 4.1.2 Lite uses the same Winograd kernels for its pass1 CNB, pass2
FNB32, pass3 FNB32+CT2D and POST CNB; its downsample stays direct.
`tools/fsr412/native_model.py` checks the same bounds on the Lite weights
(largest Winograd accumulator 9,997,312, Ultra Performance pass3). Lite
`simd` output is byte-identical to `--scalar` on the recorded sequence,
Ultra Performance and odd sizes.

The tiled Lite POST (`post_tile`, 28x28 colour tile in threadgroup memory)
runs on Apple10 (M5) only. On Apple8 it costs occupancy: M2 Max,
2016x1260 -> 3024x1890, POST 1.17 ms untiled vs 1.60 ms tiled. Both
pipelines are byte-identical; `FSR4_LITE_POST_TILE=1` or `=0` overrides the
choice, e.g. to measure M3/M4.

Measured on Apple M5, low-power mode off, median of three `--bench 10` runs,
mean GPU frame:

| Profile | Case | simd before | simd with Winograd | i8mpp |
| --- | --- | ---: | ---: | ---: |
| Lossy | `q411` | 6.82 ms | 4.88 ms | 3.10 ms |
| Lossy | `tc_game_balanced` | 6.42 ms | 4.98 ms | 3.27 ms |
| Exact Fast | `q411` | 12.02 ms | 8.23 ms | 4.71 ms |
| Exact Fast | `tc_game_balanced` | 10.86 ms | 8.13 ms | 4.72 ms |

FSR 4.1.2 Lite, recorded 864×560 → 1712×1112 sequence, three interleaved
rounds of `--bench 10`, mean of the round means:

| Profile | simd direct | simd with Winograd | i8mpp |
| --- | ---: | ---: | ---: |
| Lite | 4.62 ms | 3.98 ms | 2.71 ms |
| Lite + pixelization fix | — | 3.92 ms | — |

Scalar is about 72 ms (Lossy) and 140 ms (Exact Fast).

M6 has hardware FP8. The FSR 4.1.1 model is INT8: E4M3 holds integers exactly
only up to 16, so mapping INT8 weights or activations to FP8 changes the
result of every layer, not only an invisible last bit. INT8 tensor operations
are already native on M5/M6, and published Metal FP8 matmul measurements
(M4 Max, emulated) are about fp16 throughput, so M6 keeps `i8mpp`. M6 is not
tested here.

Physical validation is on Apple M5 only, including classic Metal and Metal 4
live encoding under Rosetta. M1–M4 timings are not measured; those GPUs have
lower half-matrix throughput than M5, so expect proportionally slower frames.

```sh
make -B
tools/bin/fsr4mac --backend-info
tools/bin/fsr4mac --caps
tools/bin/fsr4mac CASE OUT --profile lossy --auto
tools/bin/fsr4mac CASE OUT_SCALAR --profile exact --scalar
```

`--backend-info [auto|reference|scalar|i8mpp]` reports capabilities and the
selected backend without compiling or executing a model. `--caps` runs the
existing hardware probes. `--reference` uses the imported HLSL graph and is
a diagnostic path, while `--scalar` uses the portable hand-written integer
kernels. `--i8mpp` explicitly requires compatible tensor hardware; a failed
native request is reported instead of silently substituting another model.
On a failed live FSR encode the hook can use the original MetalFX encoder
and invalidates FSR history before the next successful frame.

For CrossOver use `FSR4_MODEL=auto|i8mpp|scalar|reference`. The 4.1.1 live
`int8` alias means the imported reference path. Restart the game after host
code changes; use the panel to change Exact/Lossy while it is running.

```sh
python3 tools/crossover/crossover_fsr4.py install --no-bottle --profile lossy
```

The first use of each model/permutation compiles its shaders and records
content hashes. `tools/compile_fsr411.py` and `tools/gen_fsr411_native.py`
provide explicit offline preparation, and `FSR4_COMPILE=0` prevents live
compilation when all required variants have already been built.

FSR 4.0.2 and its previous FP16 fallback remain available through
`--fsr-version 4.0.2` / `FSR4_VERSION=4.0.2`. Their policy and older physical
measurements are in [HARDWARE_FSR402.md](HARDWARE_FSR402.md).
`tools/precompile_models.py` and `build_shaders.sh` target that older model.
