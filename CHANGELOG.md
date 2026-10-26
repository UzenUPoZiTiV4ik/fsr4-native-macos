# Releases

## v2.2 (2026-10-26)

Adjustable FSR 4.1.2 Lite pixelization fix.

- Pixelization fix strength slider next to the checkbox (panel.conf
  `pixelization_strength`, CLI `--pixelization-strength 0..1`). The filter
  floor is `1 - 0.8 * strength`: 0.5 is the v2.1 behavior bit for bit, 0
  selects plain Lite, 1 widens the filter up to about 2.2x. POST reads the
  floor from the unused `pad` frame constant, so moving the slider needs no
  shader rebuild or history reset (cache suffix `_px2`).
- The hook keeps `pixelization_fix` and the strength when it rewrites
  panel.conf.
- Checks: `make test`, `make test-fsr412-optimization` (classic Metal and
  Metal 4); strength 0 equals plain Lite and 0.5 equals v2.1 on all 8
  recorded frames with `i8mpp` and `simd`.

## v2.1 (2026-10-26)

FSR 4.1.2 Lite: pixelization fix and faster M1–M4 path.

- **Pixelization fix** checkbox in the panel, shown when FSR 4.1.2 Lite is
  selected (profile `int8-lite-pixfix`, CLI `--pixelization-fix`). Where the
  network rejects history (moving hair, thin detail) Lite narrows POST's
  current-frame filter so much that half-resolution 2×2 blocks show. The fix
  multiplies the filter's precision by `mix(0.6, 1, history weight)`: up to
  29% wider where history is rejected, unchanged where it is used. Weights,
  network and history are untouched; off by default; no measurable cost.
- **M1–M4 (`simd`) Lite is faster:** its 3×3 convolutions (pass1, pass2,
  pass3, POST CNB) now use the exact integer Winograd kernels FSR 4.1 already
  used. M5 measured with `--simd`: 4.62 → 3.98 ms on the recorded sequence.
  Output stays byte-identical to `--scalar` (recorded, Ultra Performance and
  odd sizes). M1–M4 were not measured on real hardware.
- AMD's driver has no lighter model for GPUs without INT8 matrices: its
  RDNA 2 provider (`ffxProvider_FSR4_Int8_Navi2`) runs the same FSR 4.1 model
  with `dp4a`, which Apple GPUs lack. Lite on `simd` remains the fastest path
  for M1–M4.
- FSR 4.1.2 PRE: translated SPIR-V `powr` exponents are squared correctly;
  `tools/diagnostics/capture_to_testcase.py` converts panel recordings for
  offline replay.
- Checks: `make test`, `make test-fsr411-simd`, `make test-fsr412-optimization`
  (classic Metal and Metal 4).

## v2.0 (2026-10-25)

Added the FSR 4.1.2 INT8 Lite path from the supplied AMD driver libraries.

- The default `int8-lite` profile uses the three-pass Lite graph and a separate
  `fsr412_` shader/native cache. Legacy 4.1.1 profiles remain selectable.
- The panel removes the old Lite mode and names former Lite 2 **FSR 4.1 Lossy**.
  Installation updates the saved panel profile while preserving RCAS and sharpness.
- Static import validates the provider DLL hashes and both 27,392-byte Lite
  initializers without executing the Windows DLL.
- A float32 input-color cache removes repeated PRE/POST transformations.
  Metal pipeline descriptors use the actual threadgroup size. AE/RCAS
  helper selection now matches color space and auto exposure.
- In an interleaved 480-frame M5 replay (864×560 → 1712×1112, RCAS off),
  4.1.2 measured mean 2.70 ms, P99 3.87 ms and maximum 4.93 ms; Lossy 2
  measured mean 2.78 ms and maximum 5.56 ms. These are sampled GPU times,
  not a guaranteed per-frame ceiling. See `docs/FSR412_STABILITY.md`.
- The optimized path is byte-identical to AMD's Lite shaders translated to
  Metal on recorded and synthetic cases, including RCAS and Ultra
  Performance. This fixed wrong Ultra Performance POST quantizer shifts in
  the native model. The Windows DLL itself was not executed; see
  `docs/FSR412_DLL_PARITY.md`.
- Release ZIP/DMG payloads include verified portable Lite data and its
  provenance receipts. The Windows driver DLLs are not included.
- `make test-settings`, input/math/layout checks, CLI and Metal compilation
  pass on the updated tree.

## v1.9 (2026-10-21)

CrossOver release of the FSR 4.1.1 stack with the Lite 2 profile.

- Modes in the panel: FSR 4.1 (Exact), FSR 4.1 Exact Fast, FSR 4.1 Lite
  (`lossy`, default), FSR 4.1 Lite 2 (`lossy2`), and FSR 4 for comparison.
- Lite 2 uses the six trained integer layers from upstream
  `dll-2026-10-19`. Its weights are GPL-2.0-or-later upstream data and are not
  vendored into this MIT tree: `make lossy2-weights` fetches one pinned file
  into `build/fsr411_lossy2/` and checks its SHA-256. The panel shows the mode
  once those weights exist.
- No code changes since v1.8, which introduced Lite 2; this is a fresh
  release build of the same source.

Checked with `make test` (input, settings, math, layout and cache suites),
`make test-settings` including `environment-lossy2`, and a Lite 2 run on the
recorded sequence: 6.1 ms for a full frame, 2.0 ms when inference is reused,
4.2 ms mean over the run.

## Older releases

`crossover-v1.0` through `crossover-v1.8` are tagged in this repository. v1.8
added the Lite 2 profile; v1.1 to v1.7 were the FSR 4.1 port, its panel and
CrossOver packaging fixes; v1.0 was the first FSR 4.1-only package.
