> Historical FSR 4.0.2 documentation. Select `--fsr-version 4.0.2` for its CLI commands or `FSR4_VERSION=4.0.2` for the live bridge. Current defaults and instructions are in [GUIDE.md](GUIDE.md).

# Apple Silicon backends

The CLI and CrossOver encoder now default to `auto`. Selection uses Metal
feature families and device limits, not CPU names or whether the process is
running under Rosetta.

| GPU generation | Includes variants | Automatic backend |
|---|---|---|
| M1 / Apple family 7 | base, Pro, Max, Ultra | generated FP16 compute |
| M2 / Apple family 8 | base, Pro, Max, Ultra | generated FP16 compute |
| M3 and M4 / Apple family 9 | available Pro, Max, Ultra variants | generated FP16 compute |
| M5 / Apple family 10 or newer | available Pro, Max, Ultra variants | native INT8 tensor operations |

M1–M4 can execute INT8 arithmetic, but they lack the M5 GPU tensor
accelerators that make this port's native INT8 path fast. Metal 4 API
availability by itself does not establish hardware acceleration. The
generated FP16 backend uses Metal 2.3 shaders and does not require tensor
operations. The documented build/runtime requirement is macOS 26+ and
Xcode 26+ with the Metal Toolchain.

The native path additionally requires the Metal 4 runtime, at least 128
threads per group and 24 KiB of threadgroup memory. Every actual pipeline
is checked against its compiled thread and memory limits. If native INT8
initialization fails, the CLI switches to FP16 before allocating the model
resources. The live encoder retries the **same encode** on FP16 and stops
retrying native initialization for that device, including when the caller
set `FSR4_MODEL=i8mpp`. Switching devices creates fresh selection state.

## Commands

```bash
make -B
tools/bin/fsr4mac --backend-info
tools/bin/fsr4mac --caps
tools/bin/fsr4mac testcases/testcase_quality_720to1080 out/ --auto
python3 tools/precompile_models.py
```

`--backend-info [auto|f16|int8|i8mpp]` returns JSON for tools; it reports the
device, family, limits, selected backend and reason without compiling a
model. `--caps` also compiles, dispatches and checks an actual INT8 matrix
product when Metal 4 is available.

`--fp16` selects the portable model explicitly. `--i8mpp` requests the
native model with capability checks and fallback. `--int8` keeps the
generated reference INT8 path for validation; it is not automatically
selected. The last model flag wins. In CrossOver, `FSR4_MODEL=auto` is the
default, `f16`/`fp16` select FP16, and `i8mpp` requests the native path.
The historic live `FSR4_MODEL=int8` alias means `i8mpp`, unlike CLI `--int8`.

The precompiler accepts `--model auto|f16|i8mpp|all`, `--presets ...` and
`--res 1080 2160 4320`. Auto prepares the selected model for this device;
`all` prepares both backends. `tools/precompile_i8mpp.sh` remains an explicit
INT8-only helper for offline preparation.

After updating the live host code, rebuild its dylib:

```bash
python3 tools/precompile_models.py
python3 tools/crossover/crossover_fsr4.py install --no-bottle
```

The installer builds its own reference frame with the selected backend.
It no longer relies on an existing `out/f00_out.bin`, which could have been
produced by another model. The live shader cache's content hash is checked
once when loading a session, so updates to FP16 kernels replace stale
cached libraries. Restart the game after rebuilding.

## FP16 correctness and validation

Fixed an existing FP16 race: over-dispatched threads in the FNB kernels
could write past a tensor row and overwrite valid pixels in the following
row. This affected widths such as 480 and 240 in the 1080 model tier, and
produced a varying strip along the left of the final image. The kernels
now reject invalid tensor positions; the post stage also rejects blocks
outside the frame before reading its input. In-bound arithmetic is unchanged.

Dispatch dimensions come from shader metadata and are checked against the
actual compiled pipeline limits. No GPU-specific name matching or unverified
Pro/Max tile sizes are used. The model and arithmetic precision stay unchanged.

```bash
make test       # feature-family, OS, limits, fallback and installer policy
make test-gpu   # actual GPU output, repeated FP16 runs and forced INT8 failure
```

Physical validation is on the available Apple M5, including FP16 and the
classic/Metal 4 live paths under Rosetta. M1–M4 capability selection is
covered by policy tests; those chips still need physical performance and
driver validation. FP16 is an approximation of the INT8 model, so outputs
from different backends are not expected to be byte-identical.

The FP16 speed improvement measured on M5 is about 1%; the edge correction
also makes repeated runs deterministic. Larger attempted optimizations did
not show a reliable gain and were discarded. The small partial-tile SPD
auto-exposure testcase still has its previously observed nondeterminism;
see [BENCHMARK.md](../BENCHMARK.md) for the validation scope.
