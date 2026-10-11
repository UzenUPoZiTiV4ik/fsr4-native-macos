# FSR 4.1.1 model import

`import_model.py` reads the exact official AMD SDK 2.3.0 provider as bytes.
It validates its SHA-256 and Git blob ID, extracts the common and Ultra
Performance INT8 initializers, and inventories original INT8 DXIL shaders. It never loads or
executes the DLL and needs no Windows environment.

```sh
python3 tools/fsr411/import_model.py --download --with-shaders
python3 tools/fsr411/import_model.py --verify --with-shaders
python3 -m unittest discover -s tools/fsr411 -p 'test_*.py'
```

Use `--dll /path/to/amd_fidelityfx_upscaler_dx12.dll` instead of `--download`
to use a local official copy. `--output` selects an alternate destination.
The optional original shader files are placed in the ignored `oracle/`
directory. The receipt records all 426 source occurrences; identical
permutations yield **285 unique original shaders**.

`vendor_sources.py` restores the exact HLSL sources and notices from the
pinned MIT reimplementation commit, checking every source file's SHA-256:

```sh
python3 tools/fsr411/vendor_sources.py --checkout /path/to/fsr4-int8-reverse-engineering
# Or retrieve the same pinned files directly:
python3 tools/fsr411/vendor_sources.py --download
```

`recover_ultraperformance.py` reconstructs the separate Ultra Performance
weight tables, quantization parameters and POST coefficients from those
verified original containers using native DXC as a disassembler:

```sh
python3 tools/fsr411/recover_ultraperformance.py
python3 tools/fsr411/recover_ultraperformance.py --verify
```

The common HLSL stays unchanged. Generated overlays and their original
shader identities are recorded in `ultraperformance_manifest.json`; that
model uses `initializers-ultraperformance.bin`. Passes 6–9 and PRE/padding
are shared by the models. The native generator explicitly selects the
appropriate weights and source overlays:

```sh
python3 tools/gen_fsr411_native.py --preset normal --profile lossy --tier 1080 --outdir build/fsr411-native-lossy
python3 tools/gen_fsr411_native.py --preset ultraperformance --profile exact --tier 1080 --outdir build/fsr411-up-exact
```

`lossy.py` independently implements the coefficient policy of
[zgauthier2000's October 7 release](https://github.com/zgauthier2000/fsr4-shader-performance-overrides/releases/tag/dll-2026-10-07).
It folds 4/18/18/20 packed dot-product words per output channel in common
passes 1/5/10/12, and changes intermediate rounding in nine common passes.
The shared passes 7/8/9 also have this rounding in Ultra Performance; its
own model passes are otherwise exact. Output tails retain ties-to-even.
Dense Metal matrix multiplications still execute zero weights, so folding
alone has no established Apple GPU speed benefit.

`lossy_shaders.py` generates separate PRE/POST source copies with cached
POST-CNB reprojection, parity-correct CT2D selection, the rest rule,
neighborhood history clamp, dark-edge repair and prior-surface ownership
lookup. Its module docstring defines b2/u12/u13. The host controls jitter
cadence, cache validity, resets, Ultra Performance exclusion and buffer
clears. A skipped frame must be followed by a full refresh before another
skip, since the supplied motion describes only the immediately preceding
frame. It uses dedicated motion/ownership buffers, avoiding the reference
DLL's scratch aliases. All three tier variants compile to Metal, including
the display-resolution motion PRE variant. These compile checks do not
establish frame or live-game equivalence.

`post_optimization.py` optionally changes the external-CNB POST to one
output pixel per thread and a float32 shared cache for overlapping input
color transforms. Apply it after `external_post_cnb` / `apply_lossy_post`
and before portability conversion, preserving small-loop unrolling. Its
256×1×1 threads cover **16×16 output pixels**, so the host must use
`post_pixels_per_group=16` when dispatching this variant. The original
filter coordinates, Gaussian evaluation, tap accumulation, normalization,
blending and output transfer remain unchanged. A barrier precedes partial
tile returns; render-viewport clamps and a direct-load fallback are kept.
Only `ceil(inv_scale * 16) + 4` texels per axis are loaded into the fixed
20×20 cache. CT2D uses FP32 dot products of signed INT8 values: for every
normal/UP table, the absolute bound on every partial sum is at most
110151/87727, below 2^24. This preserves the original integer result for all
valid inputs; no half-precision approximation is introduced.
The common and Ultra Performance, exact and lossy variants compile for all
three tiers, with all four color spaces covered. Geometry and source
contracts are tested; GPU equivalence and speed require separate checks.

`verify_lossy_reference.py` optionally compares all baked tables against
the published DLL **as data**, with a pinned hash and native disassembly.
It vendors no reference shader code. The recorded result is
`third_party/fsr411/lossy_model_manifest.json`:

```sh
python3 tools/fsr411/verify_lossy_reference.py --dll /path/to/extracted/test-lossy/amd_fidelityfx_upscaler_dx12.dll --verify
```

`portability.py` fixes display-resolution MV bounds and replaces dummy
out-of-view buffer writes with actual conditional stores. Apply it to the
generated include tree, leaving the imported originals intact.

`autoexposure.py` generates `ae_tiles.hlsl` and `ae_finalize.hlsl` from the
imported SPD source. The first dispatch executes the original tile
reductions; a texture barrier precedes a single-group final reduction.
The original arithmetic and 40-byte b0 ABI are preserved. Generated Metal
has no atomic-counter resource or cross-group publication dependency.
It also updates exposure at mip six (previously skipped for maxima 64–127),
zeros reads outside the logical tile image, and writes the previous
exposure slot before overwriting the current slot.

The focused AE probe compiles all variants with `-fno-fast-math` and
`-ffp-contract=off`, and then checks current and previous exposure directly:

```sh
python3 tests/run_fsr411_autoexposure.py
# Build the probe without GPU execution:
python3 tests/run_fsr411_autoexposure.py --compile-only
```

On the development Apple GPU, all 54 varying-input frame comparisons
matched a single-dispatch oracle bit-for-bit, including partial input tiles,
poisoned tile storage and oversized backing textures. Thirty small-image
constant-luminance checks matched an independent CPU formula within
3e-5 relative error and preserved the known previous-exposure marker.
The oracle receives only a store-order fix and an explicit device barrier
before atomic publication. Without that barrier (`--raw-oracle`), the
original Metal translation intermittently reads poisoned tile values;
the split production path avoids that synchronization issue by construction.
These are correctness checks, not timing measurements.

The upstream manifest's
`container` fields mix AMD shader hashes and compiled reimplementation
hashes. Consult the corresponding upstream evidence before treating one
of those fields as an AMD oracle identity.

See `third_party/fsr411/README.md` for the model scope, licenses and validation
limits. Neither Radeon timing measurements nor upstream D3D12 exactness
establishes Metal performance or output equivalence.
