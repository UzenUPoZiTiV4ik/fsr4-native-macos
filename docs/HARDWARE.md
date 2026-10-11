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

Measured on Apple M5, `tc_game_balanced`, `--bench 5`, mean GPU frame:

| Profile | scalar | simd | i8mpp |
| --- | ---: | ---: | ---: |
| Lossy | 71.7 ms | 6.13 ms | 3.12 ms |
| Exact Fast | 140.4 ms | 10.3 ms | — |

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
