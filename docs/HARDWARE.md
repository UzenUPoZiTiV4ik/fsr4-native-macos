# Apple Silicon backends

FSR **4.1.1 + Lossy** is the current default. Profile selection (Exact/Lossy)
is independent of backend selection. Both accelerated and scalar backends
execute the actual INT8 model, including its separate Ultra Performance
initializer.

| GPU family | Automatic FSR 4.1.1 backend |
| --- | --- |
| M1 / family 7 | Scalar INT8 Metal compute. |
| M2 / family 8 | Scalar INT8 Metal compute. |
| M3–M4 / family 9 | Scalar INT8 Metal compute. |
| M5 / family 10+, macOS 26+ | Native INT8 tensor operations. |

Selection uses Metal feature families and device limits. Rosetta does not
change GPU capabilities. Native kernels use 128 threads and 20 KiB of
threadgroup memory. Scalar kernels use Metal 3 and do not require the M5
tensor instructions. This project builds for macOS 26+ with Xcode 26+ and
the Metal Toolchain.

Physical validation is on Apple M5, including scalar kernels and both
classic Metal and Metal 4 live encoding under Rosetta. M1–M4 driver behavior
and performance remain unmeasured. Scalar support should not be interpreted
as a real-time performance claim on those GPUs.

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
