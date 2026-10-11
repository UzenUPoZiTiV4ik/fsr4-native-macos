# Reimplemented FSR 4.1.1 kernels (EXACT_411, full 28-pass homegrown chain)

Twenty-eight HLSL compute kernels reimplemented from the AMD FSR 4.1.1
provider and validated **byte-exact** against AMD's own containers:
identical output SHA over the full 20.9 MB scratch arena on the same
pre-state (two-leg replay: our kernel ≡ AMD container; residuals vs.
historical dumps are the documented rematerialization-family noise,
identical on both legs).

## Provenance (read before reuse)

- Semantics reconstructed from the AMD 4.1.1 provider
  (`Kits/FidelityFX/signedbin/amd_fidelityfx_upscaler_dx12.dll`,
  SHA-256 `d0dcccc74a43c44ba435b7a369b456e0970d8a4464e4bd683119b374f2c9fb46`),
  which AMD distributes under MIT (see `LICENSE` and `NOTICE.md`: SDK 2.0.0
  accidental publication + current `docs/license.md` carve-out naming this
  exact file). Constants (weights/biases/scales) are AMD's MIT-distributed
  model data; the HLSL expression is our reimplementation.
- Operator topology follows the FSR 4.0.2 ML2Code sources (same MIT story).
- Per-kernel validation records: `docs/evidence411/` (oracle SHAs, mismatch
  counts, cross-leg hashes). Evidence binds the algorithms as validated; the
  only difference vs. the validated files is the license header above.
- `manifest_411_quality_p4.json` is the machine-readable dispatch table
  (28 passes, root signatures, dispatch geometry, oracle hashes). Kernel
  paths were translated to this directory on export; `container_file` hashes
  identify AMD containers and resolve only in the private tree.

## Contents

| File | Pass | Operator |
|---|---|---|
| `pre_rev_411.hlsl` (+ `post_rev_411_tables.hlsli` deps) | PASS_0 (PRE) | Input-prep pass, cleanroom reimplementation (`411_pre_cleanroom.json`) |
| `reimpl_pass1.hlsl` | PASS_2 (logical 1) | ConvNextBlock 16ch, mirrored planes, tail ×64 |
| `reimpl_pass2.hlsl` | PASS_4 (logical 2) | ConvNextBlock 16ch, swapped requant, tail ×32 |
| `reimpl_pass3.hlsl` | PASS_6 (logical 3) | DownscaleStridedConv2x2 16→32, /2, runtime t18 weights |
| `reimpl_pass4.hlsl` | PASS_8 (logical 4) | FasterNetBlock 32ch, 3-stage + residual |
| `reimpl_pass5.hlsl` | PASS_10 (logical 5) | FasterNetBlock 32ch, zero-mismatch oracle |
| `reimpl_pass6.hlsl` … `reimpl_pass12.hlsl` | PASS_12…24 (even, logical 6–12) | Homegrown neurals 6–12 (`411_homegrown_full_27exact.json`, per-pass `411_5111_pass6`…`411_5129_pass12` records) |
| `pass0_post.hlsl` … `pass5_post.hlsl` | PASS_1…11 (odd) | Epilogue/move kernels after logical passes 1–5 |
| `pass6_post.hlsl` … `pass12_post.hlsl` | PASS_13…25 (odd) | Epilogue/move kernels after logical passes 6–12 (`411_epilogues_6to12_our_reimpl.json`) |
| `post_rev_411.hlsl` (+ `post_rev_411_tables.hlsli` deps) | PASS_26 (POST) | Output-resolve pass, cleanroom reimplementation (`411_post_cleanroom.json`) |

AE note: `bench/shaders/candidates/ae_411_ours/ae_ours.hlsl` is the proven
AE kernel (stock-container-exact modulo the viewport-bounds guard;
`411_ae_ours_reimpl.json`). Shared includes
(`ffx_fsr4_gpu_colorspace_common.h`, `ffx_fsr4_upscale_resources.h`,
`mlsr_optimized_includes.hlsli`) are cleanroom copies of the shape-only
headers PRE/POST need.

## Building

All kernels are `cs_6_6`. Entry points: `main` for the thirteen
`passN_post` epilogues and `ae_ours.hlsl`, `reimpl_passN` for the twelve
neural passes, `fsr4_model_v07_fp8_no_scale_prepass` /
`fsr4_model_v07_fp8_no_scale_postpass` for `pre_rev_411.hlsl` /
`post_rev_411.hlsl` (with `-enable-16bit-types` for PRE/POST/AE).
Bindings mirror the measured roots: scratch arena UAV `u11`, initializer
SRV `t18`, tensor constants CBV `b0` (see manifest `root` fields: epi =
UAV12+CBV b0, compute = UAV12+SRV19+CBV b0). Examples:

```powershell
dxc reimpl_pass1.hlsl -T cs_6_6 -E reimpl_pass1 -HV 2021 -O3 -Fo reimpl_pass1.dxbc
dxc pass0_post.hlsl -T cs_6_6 -E main -HV 2021 -O3 -Fo pass0_post.dxbc
dxc reimpl_pass6.hlsl -T cs_6_6 -E reimpl_pass6 -HV 2021 -O3 -Fo reimpl_pass6.dxbc
dxc pass6_post.hlsl -T cs_6_6 -E main -HV 2021 -O3 -Fo pass6_post.dxbc
dxc pre_rev_411.hlsl -T cs_6_6 -E fsr4_model_v07_fp8_no_scale_prepass -enable-16bit-types -HV 2021 -O3 -Fo pre_rev_411.dxbc
dxc post_rev_411.hlsl -T cs_6_6 -E fsr4_model_v07_fp8_no_scale_postpass -enable-16bit-types -HV 2021 -O3 -Fo post_rev_411.dxbc
```

Running a kernel needs the matching pre-state (scratch + t18 + tensor slice),
which is game-capture-derived and stays local-only. The evidence files record
the exact geometry each kernel was validated at.
