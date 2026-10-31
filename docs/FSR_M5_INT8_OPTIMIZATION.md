# M5 INT8 register-chain optimization (v2.5)

The M5 MPP backend runs the same integer model, weights and epilogues.
This change removes intermediate threadgroup storage from verified CNB,
FNB32/64, CT32/64 and DS32 chains. DS16 retains its original implementation:
the tested register-only replacement was slower.

## Safety and compatibility

- One SIMD32 group per chain. Arithmetic, INT32 wrapping, shifts, ties,
  residuals, narrowing and saturation are unchanged.
- A small GPU probe checks the real cooperative-tensor register layouts
  before selecting any new entry point. Unsupported layouts retain the
  established kernel. There is no model data or temporal-state access in
  the probe.
- The dispatch policy accepts only the original kernel name, tile and
  thread count together. Legacy/custom cache contracts are not reinterpreted.
- Pipeline limits use the actual threadgroup size instead of budgeting for
  unused 1024-thread groups. The on-disk PassParams ABI is unchanged.
- Set `FSR4_MPP_OPTIMIZE=0` before launching to restore the old MPP dispatch
  and full-model pipeline limits. SIMD and reference backends are unaffected.

## Measurements

Apple M5, macOS 27, thermal state nominal, low-power mode off. Serial
interleaved A/B pairs in one process; compilation and warmups are excluded.

Isolated neural-pass reductions: FNB64 about 25-26%, CT32 about 13-14%,
CT64 about 12-13%, DS32 about 20-21%. CNB has little isolated timing gain,
but no longer reserves fallback threadgroup storage.

Full GPU command buffer, 1000x648 -> 1712x1112, 60 eight-frame cycles:

| Model | Before median | After median | Median-time reduction |
| --- | ---: | ---: | ---: |
| Lite | 2.515 ms | 2.472 ms | 1.7% |
| Exact | 4.485 ms | 4.420 ms | 1.4% |
| Retrained Lossy, inference frame | 3.684 ms | 3.588 ms | 2.6% |

Lossy reuse frames have no meaningful improvement. This particular replay
uses the working branch's PRE/POST bundle, not a game FPS counter or a
like-for-like v2.4 installer comparison. The public release also retains its
existing tiled POST optimization. Neural-pass and full-frame gains must not
be conflated; these are sampled timings, not a speed guarantee.

## Combined public release

A final interleaved 60-cycle run uses the v2.4 MPP source as the baseline,
with the public release's existing tiled POST retained on both sides.
Non-reset inference-frame medians, same 1000x648 -> 1712x1112 input:

| Model | Before median | After median | Median-time reduction |
| --- | ---: | ---: | ---: |
| Lite | 2.345 ms | 2.267 ms | 3.3% |
| Exact | 4.824 ms | 4.696 ms | 2.7% |
| Retrained Lossy, inference frame | 3.885 ms | 3.770 ms | 2.9% |

Full output, history, recurrent state and exposure are byte-identical on
classic Metal and Metal 4, including the eight-frame reset/inference/reuse
cycles. Quality readbacks are excluded from the timing run.
[Summary](reports/fsr-m5-register-chains-2026-10-31.json).

## Validation

- 1542 CPU dispatch-policy checks, including malformed and legacy caches.
- Lite, Exact and retrained Lossy, Normal/Ultra/2160: 372 isolated cases
  compared byte for byte with the independent CPU oracle, including signed
  INT8 extremes, odd boundaries and dirty padding.
- The same 372 cases pass with a forced failed layout probe and the old
  fallback kernels: 744 fast/fallback cases in total.
- Full-frame output, history, recurrent state and exposure match the
  pre-change MPP source byte for byte on both classic Metal and Metal 4.
  Reset, non-reset inference and Lossy reuse decisions are exercised.

Rejected experiments stay in ignored local build outputs, not production:
larger per-SIMD tiles, smaller/streamed POST workgroups, multi-SIMD MPP
packing, direct SIMD DS loads and compact SIMD DS staging. Some were exact
but slower; multi-SIMD MPP packing failed byte identity.

The existing pre-M5 SIMD optimization and exact Winograd are included too;
see [FSR_PRE_M5_OPTIMIZATION.md](FSR_PRE_M5_OPTIMIZATION.md). The new SIMD DS
experiments did not improve M5 timings and are not enabled. Performance on
physical M1-M4 still requires measurements on those chips.

## Reproduce

```sh
make test
make test-fsr411-mpp-fast
make build/tests/fsr411_mpp_ab
build/tests/fsr411_mpp_ab . --rounds 60
# --before-root selects a saved pre-change source tree.
# --verify-every-frame validates all temporal state outside timed buffers.
# --mtl4 is quality-only: the report does not invent Metal 4 GPU timestamps.
```
