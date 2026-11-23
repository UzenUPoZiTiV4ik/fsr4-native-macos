# AMD FSR 4 — source notes

## Provenance

This is AMD's FSR 4 (FidelityFX Super Resolution 4) source, including the **INT8**
neural-network variant. It comes from AMD's official repository
[`GPUOpen-LibrariesAndSDKs/FidelityFX-SDK`](https://github.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK),
where these files were published under the SDK's MIT licence at commit
**`01446e6a74888bf349652fcf2cbf5f642d30c2bf`** (`"AMD FidelityFX SDK 2.0.0"`, 2025-08-18).

To regenerate this exact drop (verified working, ~20 MB):

```bash
mkdir .fetch && cd .fetch
git init -q .
git remote add origin https://github.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK.git
git config core.sparseCheckout true
printf 'Kits/FidelityFX/upscalers/fsr4/\nKits/FidelityFX/api/\nSamples/Upscalers/FidelityFX_FSR/\n' \
  > .git/info/sparse-checkout
git fetch --depth 1 --filter=blob:none origin 01446e6a74888bf349652fcf2cbf5f642d30c2bf
git checkout FETCH_HEAD
```

## Layout

The macOS checkout keeps the INT8 and FP16 operator headers. It omits the
unused `operators/{float32,float32_NHWC,float8_NHWC,uint8_NHWC}` families
from the upstream layout shown below. The pinned checkout above contains them.

```
src/Kits/FidelityFX/upscalers/fsr4/     the FSR 4 upscaler itself (12 MB, 183 files)
  include/ffx_provider_fsr4.h           provider class
  include/gpu/fsr4/                     common types, resource descriptions, pre/post I/O contract
  include/gpu/spd/ffx_spd.h             SPD (single-pass downsampler) for auto-exposure
  include/gpu/fsr1/ffx_fsr1.h           RCAS lives here
  internal/shader_selector.cpp          which shader blob is picked for which hardware/preset
  internal/fsr4_permutations.h          permutation flags
  internal/shaders/                     the generated model + post-processing shaders
    fsr4_model_v07_i8_<preset>/         INT8 model: pre.hlsl, passes_*.hlsl, post.hlsl, initializers.bin
    fsr4_model_v07_fp8_no_scale_*/      FP8 model (requires WMMA, see Notes)
    rcas.hlsl, spd_auto_exposure.hlsl, debug_view.hlsl
  dx12/ml2code_runtime/                 the operator library the generated shaders call into
    operators/int8_NHWC/                the INT8 operator library
    operators/float16_NHWC/             the FP16 operator family
    operators/float32/, uint8_NHWC/, templates/
    tensor_int8.hlsli, globals.hlsli, storage.hlsli, padding.hlsli
  dx12/ffx_provider_fsr4_dx12.cpp       host side: resources, constants, dispatch order (124 KB)
  dx12/BuildFSR4UpscalerShaders.bat     how AMD compiled the shaders (DXC)
src/Kits/FidelityFX/api/                the API the provider implements (ffx_api.h, ffx_api_types.h, ...)
src/Samples/Upscalers/FidelityFX_FSR/   a DX12 sample showing how a game drives FSR 4 per frame
```

There is **no ONNX file** in the drop. The model exists only as generated HLSL plus the
`initializers.bin` weight blob.

## Orientation

FSR 4 is a temporal upscaler: ML-based, but not a transformer. It is an evolution of
FSR 2's architecture where the hand-written heuristics are replaced by a small CNN.
The generated HLSL is self-documenting: every tensor carries its original ONNX layer
name and shape as a comment, e.g.

```
// /encoder1/DownscaleStridedConv2x2/downscale_conv/Conv (7, 1080, 1920), (16, 7, 2, 2), (16,) -> (16, 540, 960)
```

Network shape, read off those comments: `encoder1 → encoder2 → encoder3 → bottleneck →
decoder3 → decoder2 → decoder1` with skip connections, ConvNeXt-style and FasterNet-style
residual blocks, ~39 layers, on the order of 100K parameters. Roughly a 20 MB scratch
working set at 1080p.

The whole model is **79,360 int8 weights** (see below).

## The INT8 model

In `internal/shaders/fsr4_model_v07_i8_quality/passes_1080.hlsl`:

```
#define WMMA_ENABLED  0
#define DOT4_ENABLED  1
```

The INT8 path uses **no AMD wave-matrix intrinsics and no FP8 lookup table** — ordinary
arithmetic and DP4A-style integer dot products. The FP8 path is the opposite (`#error ...
There is no FP8 without WMMA`).

Six presets (`native`, `quality`, `balanced`, `performance`, `drs`, `ultraperf`) x three
max resolutions (1080/2160/4320). Each preset has its **own** weights and its own
input scale — they are separately trained, not one model at different sizes.

Operators used by the INT8 model, all under `dx12/ml2code_runtime/operators/int8_NHWC/`:

| operator | file |
|---|---|
| `Conv2D_k2s2b` | `Conv2D_k2s2b.hlsli` |
| `ConvNextBlock` | `Fused/ConvNextBlock.hlsli` |
| `FasterNetBlock` | `Fused/FasterNetBlock.hlsli` (214 KB — the big one) |
| `FNB_CT2D_ADD` | `Fused/FNB_CT2D_ADD.hlsli` |
| `FusedConv2D_k2s2b_QuantizedOutput` | `Fused/FusedConv2D_k2s2b_QuantizedOutput.hlsli` |
| `Add`, `Concat`, `Relu`, `Pad`, `QuantizeLinear`, `ConvTranspose2D_k2s2b` | same dir |

Activation quantization is per-tensor, stored as a float scale per tensor, applied once
on store (`round(vs * rcpScale)`), accumulation in FP32. `int8_t4_packed` (4 int8 per
uint32) is the packing used throughout.

## Weight blob format (`initializers.bin`)

Verified across all six INT8 presets — every one is 89,216 bytes and parses identically:

```
256-byte header
  +0    float32   downscale_quantizationScale (the prepass activation scale)
  +4    zero padding to 256
695 x 128-byte records
        records 0..74    (75)  scale records: one float32, rest zero
        records 75..694  (620) weight records: 128 int8 values each
              => 620 x 128 = 79,360 int8 weights
```

128 bytes is 32 x uint32 = 32 x `int8_t4_packed`, i.e. the DP4A packing granularity.

`downscale_quantizationScale` for the quality preset is `0.014884727075695992`, which is
exactly the literal `static const float downscale_quantizationScale` in that preset's
`pre.hlsl` — that is how the format was pinned down.

Note that pass 0's weights are **FP16, not int8** — they are embedded directly in
`pre.hlsl` as `uint` dword arrays with the decoded values in comments:

```c
static const uint embedded_encoder1_DownscaleStridedConv2x2_downscale_conv_weight_dwords[256] = {
    // -0.011314, 0.04135, 0.652, ...   (two fp16 per dword, low half first)
    0x294ba1cb, 0x2c603937, ...
```

The FP8 model's `initializers.bin` is 130,088 bytes with a different layout.

## The pipeline

Dispatch order, read out of `dx12/ffx_provider_fsr4_dx12.cpp`:

```
SPD auto-exposure                     (from low-res colour, unless AUTOEXPOSURE_ENABLED is off)
model_pso[0]        pre pass
  [padding_pso[0]]                    zero scratch padding (INT8: not used, see Notes)
model_pso[1..12]    12 model passes
  [padding_pso[i]]
model_pso[13]       post pass
[rcas_pso]                            optional sharpening
[debug_view_pso]                      optional debug visualisation
```

The entire per-frame cost is **14 model dispatches + SPD + optional RCAS**.

Resources the upscaler owns (`getResourceDescriptions`), all at *upscale* (output)
resolution:

| resource | format |
|---|---|
| `FSR4UPSCALER_Recurrent` | RGBA8_UNORM |
| `FSR4UPSCALER_History` | RGBA16F |
| `FSR4UPSCALER_HistoryReprojected` | RGBA16F |
| (rcas intermediary) | RGBA16F |

Scratch: **20,880,256 B @1080p, 83,232,256 B @4K, 332,352,256 B @8K**.

Per-frame host constants (`OptimizedConstants` in the provider):

```c
float inv_size[2], scale[2], inv_scale[2];
float jitter[2];                      // subpixel jitter
float mv_scale[2];                    // motion vector scale
float tex_size[2], max_renderSize[2];
float fMotionVectorJitterCancellation[2];
uint32_t width, height, reset, width_lr, height_lr;
float preExposure, previous_preExposure;
uint32_t rcas_enabled; float rcas_sharpness;
```

So the required inputs are: colour, depth, motion vectors, jitter, exposure, and a reset
flag — the normal temporal-upscaler contract. The sample under `src/Samples/` shows how a
game supplies them per frame.

The model input is a **7-channel NHWC tensor at render resolution**, shape `(W, H, 7)`.
The 4 recurrent channels plus current-frame luma, previous-output luma and a colourfulness
scalar is the commonly repeated description of those 7; that specific channel *meaning*
is secondhand, not something I confirmed from the source.

The permutation flags that select shader variants are in `internal/fsr4_permutations.h`:
`DEPTH_INVERTED`, `LOW_RES_MV`, `AUTOEXPOSURE_ENABLED`, `COLORSPACE` (linear / sRGB / PQ /
generic nonlinear), `JITTERED_MOTION_VECTORS`, `RESOLUTION`, `PRESET`.

## Notes

* **The FP8 model is WMMA-only.** It uses `AmdWaveMatrixMultiply`, `AMD_GROUPSHARED_LOAD`
  (GCN-specific LDS staging) and lane choreography (`WaveReadLaneFirst` /
  `WaveGetLaneIndex`). None of these appear in the INT8 path.
* **Padding reset is never selected for INT8.** `GetPaddingResetBlob` returns `{}` in
  `shader_selector.cpp`, so the `_POST` padding-reset entry points exist in the i8 shader
  sources but are never handed out by the selector.
* **Tensors are not densely packed.** Scratch tensors are flat byte-addressed buffers with
  explicit byte strides and padding; the strides are literals in the generated HLSL.
* The generated shaders in-tree were emitted for the `navi48` target.

## Open questions

* **Which scale record belongs to which layer.** The HLSL pass files carry every tensor's
  scale as a literal (`quantizationScale_slice_0`), so the mapping is recoverable by
  parsing those, but it has not been done.
* Whether the record order in `initializers.bin` matches pass/layer order.

## Secondary sources

* [lhl/fsr4-rdna3-optimization](https://github.com/lhl/fsr4-rdna3-optimization) — public
  MIT repo carrying an upstream snapshot of this same source, plus analysis of the INT8 vs
  FP8 paths and measured micro-benchmarks on RDNA 3.5.
* [Rolaand-Jayz/RE-of-FSR-4.1.0-Upscaling](https://github.com/Rolaand-Jayz/RE-of-FSR-4.1.0-Upscaling)
  — static analysis of the newer FSR 4.1.0 binaries (blob layout, pass catalogue, weight
  extraction). Covers 4.1 only, which has no source.
