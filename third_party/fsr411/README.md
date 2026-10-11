# FSR 4.1.1 INT8 models and source provenance

This directory contains the **FSR 4.1.1.2740 common and Ultra Performance INT8 models**, independently
of the earlier FSR 4.0.2/v07 source drop in `../fsr4`. The `v07_fp8_no_scale`
entry names in the new sources are names preserved by AMD; they do not make
these INT8 kernels the old model or the FP8 hardware path.

## Provenance

- Official provider: AMD FidelityFX SDK **2.3.0**, commit
  `60f4ea81909200d8542eca14dccb2628b763a9a3`,
  `Kits/FidelityFX/signedbin/amd_fidelityfx_upscaler_dx12.dll`.
- Provider version **4.1.1.2740**, 28,761,864 bytes, SHA-256
  `d0dcccc74a43c44ba435b7a369b456e0970d8a4464e4bd683119b374f2c9fb46`.
- HLSL reimplementation: [int3rrobang/fsr4-int8-reverse-engineering](https://github.com/int3rrobang/fsr4-int8-reverse-engineering),
  commit `f508ddecd46fb407d1be41c3102b54dddf57d220`, MIT. `source_manifest.json`
  records each unchanged file's upstream path, length and SHA-256.
- The official SDK explicitly lists this provider DLL in its MIT exception.
  See `LICENSE.AMD.md`, `LICENSE`, and the preserved `NOTICE.upstream.md`.
  No GPL shader-override implementation is included here.

## Files and model contract

- `shaders/reimpl411/`: 1080 output-class kernels and shared includes.
- `shaders/reimpl2160/`, `shaders/reimpl4320/`: 2160/4320 output-class kernels.
- `shaders/ae_411_ours/ae_ours.hlsl`: reimplemented autoexposure.
- `initializers.bin`: **131,072 bytes**, official DLL range
  `[0xC04A00, 0xC24A00)`, SHA-256
  `a54e552ff69a3f7199861417f7ae461d1f67f636ab532c6ec9d8ea5d2963735a`.
  Bind as the read-only byte-address buffer `t18`; do not interpret it using
  the old 89,216-byte initializer format. Some weights are embedded in HLSL,
  while other weights and int32 biases are loaded from this buffer.
- `initializers-ultraperformance.bin`: the separate **131,072-byte** UP
  initializer, DLL range `[0xC24A00, 0xC44A00)`, SHA-256
  `3c42fc6eabfa9b5b1e489df0e9b2f58e3f9fa22048c584f6839bb27d32a0640e`.
- `ultraperformance/`: generated overlays for neural passes 1–5 and 10–12,
  POST and its coefficient table in the three output classes. They are
  reconstructed from the official INT8 DXIL; the original HLSL stays intact.
- `ultraperformance_manifest.json`: all original common/UP shader pairs,
  exact enumerated scalar instruction changes, generated source hashes and
  the distinct initializer contract. PRE/padding and neural passes 6–9 are
  byte-identical original shader containers between the models, although
  passes 6–9 consume the selected model's initializer.
- `import_manifest.json`: original DLL identity, initializer identity and
  all **426 original INT8 DXIL occurrences**, including debug and Ultra
  Performance variants (**285 unique containers**). The optional
  `oracle/dxil/` files are local imports.

The common model serves native, Quality, Balanced, Performance and DRS input
ratios. **Ultra Performance requires both its separate initializer and its
source overlays.** Reusing the common initializer would select a different
model, even if the resolution ratio were otherwise correct.

There are **27 ordered upscaling passes**: PRE, thirteen border/epilogue
passes interleaved with twelve neural passes, and POST. Autoexposure makes
28 entries in the 1080 manifest. Preserve numeric `FFX_FSR4_PASS_0` through
`FFX_FSR4_PASS_26` order; the upstream JSON array is not in execution order.
The static dispatch numbers in those manifests are upstream validation
examples, not formulas for arbitrary render or output dimensions.

Scratch sizes for the three classes are 20,880,256 / 83,232,256 / 332,352,256
bytes. Tensor constants are seventeen `uint4` slots (272 bytes). Shared PRE
includes still use the MIT tensor helper headers from `../fsr4/dx12`, and AE
uses the existing SDK core/SPD helper includes; those helpers do not supply
the old model's weights.

## Reproduce and validate the import

From the repository root:

```sh
python3 tools/fsr411/import_model.py --download --with-shaders
python3 tools/fsr411/import_model.py --verify --with-shaders
python3 tools/fsr411/recover_ultraperformance.py --verify
```

Or use `--dll /path/to/amd_fidelityfx_upscaler_dx12.dll` for a local copy.
The importer never executes the DLL, rejects every different DLL hash, checks
both extracted initializer hashes, and validates all container chunk boundaries
before writing files. Repeating the same import produces the same receipt.

Upstream classifies these HLSL kernels as byte-exact in its D3D12 replay
tests. Its own live runtime is a research preview. Those results establish
the source provenance and intended semantics; they **do not establish** Metal
parity, end-to-end live-game quality, or an Apple GPU speedup. The local Metal
port requires its own tests and must report their actual coverage.

## Lossy profile

The runtime's requested lossy profile follows the numerical policy of
[zgauthier2000/fsr4-shader-performance-overrides](https://github.com/zgauthier2000/fsr4-shader-performance-overrides),
revision `8d72457ab6302dab8e4f918f6bf83a1af3900288`, release
`dll-2026-10-07`. It is implemented independently in `tools/fsr411/lossy.py`
and `lossy_shaders.py`; no GPL implementation is included.

Common-model 3x3 weights are folded in passes 1/5/10/12 with 4/18/18/20 of
36 packed int8x4 contributions removed per output channel. Each removed
vector is added to the closest retained tap for the same input quartet,
with signed-byte saturation. Pass 10 actually saturates 18 byte additions;
the folding therefore does not preserve every constant-input response.
Intermediate rounding uses ties toward positive infinity in common passes
1/2/4/5/7/8/9/10/12 and the shared UP passes 7/8/9. All output tails and POST
network quantizers keep their exact policy.

`lossy_model_manifest.json` records a comparison with the pinned published
lossy DLL, read only as data: all 832/1600 baked weight dwords match in each
of the twelve affected common shader variants. Its reference DLL, chunks
and GPL rewriters are not build inputs and are not distributed here.

Skipped frames use the saved POST-CNB result reprojected with rounded block
motion, select the correct last-layer coefficients for odd-pixel motion,
and apply the release's history/edge/rest/occlusion repairs. The release
default `SB_CONST=0` makes the current-frame blend weight `(1-a)^2`; the
earlier jitter-distance exponential has no effect at this setting.
Dedicated Metal buffers replace the reference DLL's scratch aliases, and
the ownership plane is cleared on every skipped frame, including unusual
consecutive negative-jitter frames. A valid cache is required; resets and
Ultra Performance force a complete model evaluation.

This is an image-changing profile, separate from the retained exact
comparison profile. Coefficient and compile validation do not prove
end-to-end parity with the Windows release or visual quality in games.
Dense matrix kernels do not skip zero-weight arithmetic; the primary
expected saving is the avoided model dispatch on selected frames, to be
measured on the actual Apple GPU.
