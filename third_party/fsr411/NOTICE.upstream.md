# Third-party notices

This project is a research effort built **on top of** the following
third-party works. **None of them is included in this repository** — the
entries below explain what you must supply yourself and under what terms.

## AMD FidelityFX SDK 2.0.0

- Source: https://github.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK
- License: MIT, Copyright © 2025 Advanced Micro Devices, Inc.
  (see `LICENSE` in this repo; the AMD copyright and permission notice are
  retained there as the MIT license requires).
- Used as: reference HLSL for the FSR 4 INT8 path (`fsr4_model_v07_i8_*`),
  `ffx_*` API headers (headers only — no SDK libraries are linked), and the
  `Build-FSR4Int8Provider` workflow, which operates on **your own** SDK
  checkout. No AMD binaries are redistributed here.

## AMD FSR 4 provider DLLs (4.0.2 / 4.0.2b / 4.0.2c / 4.1.1)

- Proprietary binary releases. **Not included.** The analysis tooling in
  `analysis/` can inspect a local copy you obtained yourself, but no
  container bytes, weights, or extracted shaders from these DLLs are part
  of this repository.

## OptiScaler

- Source: https://github.com/optiscaler/OptiScaler (check its license there)
- Role: external, user-installed upscaler middleware used during live-game
  testing. **Not vendored, not patched, not redistributed.** Referenced only
  as a test configuration.

## Cyberpunk 2077 (CDPR)

- Used solely as a test subject for live-frame observation. **No game
  content of any kind is included** — no captures, textures, depth/motion
  buffers, or PIX files. Fixture paths referenced in docs and code are
  local-only and resolve to nothing in a fresh clone.

## Toolchain

- DirectXShaderCompiler (DXC, LLVM license), Windows SDK / VS Build Tools,
  NVIDIA Nsight / PIX (capture tools, outputs not included).

## Research credit (no code included — ideas and published facts only)

- **Rolaand-Jayz** — static reverse-engineering of the FSR 4.1.0 FP8 provider
  (`github.com/Rolaand-Jayz/RE-of-FSR-4.1.0-Upscaling`) and the FSR 4.0.2
  reference (`github.com/Rolaand-Jayz/FSR-4.0.2-reference`). The published
  tensor map informed our weight sourcing; no code or binaries taken.
- **Wrong On The Internet** — "Understanding FSR 4"
  (`woti.substack.com/p/understanding-fsr-4`, 2025-11-10), an independent
  read-through of the accidentally published 4.0.2 source that corroborates
  the architecture summary (39-layer U-Net, kernel-prediction head,
  recurrent-state reprojection). Conceptual grounding only.

## Provenance notes (why reimplemented kernels can be here)

- FSR 4.0.2 source and INT8 models: published by AMD in FidelityFX SDK 2.0.0
  (commit `01446e6`, August 2025) under the repo's blanket MIT license, then
  removed from `main`; preserved via public forks (e.g. upstream PR #167
  "Readd FSR4 code"). MIT has no revocation clause: lawfully obtained copies
  keep their grant.
- FSR 4.1.1 provider DLL
  (SHA-256 `d0dcccc74a43c44ba435b7a369b456e0970d8a4464e4bd683119b374f2c9fb46`):
  distributed by AMD in the current repo `main` at
  `Kits/FidelityFX/signedbin/amd_fidelityfx_upscaler_dx12.dll`, explicitly
  named in the MIT carve-out of `docs/license.md` (which otherwise forbids
  reverse engineering of non-listed files).
- The kernels in `bench/shaders/reimpl411/` were reconstructed from the 4.1.1
  provider's behavior and validated byte-exact against it; their constants are
  AMD's MIT-distributed model data, shipped with AMD's copyright notice
  (`LICENSE` + per-file headers). No constants were taken from the unsigned
  community 4.0.2b/c binaries, which AMD never distributed.
