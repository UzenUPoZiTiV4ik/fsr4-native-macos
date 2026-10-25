# FSR 4.1.2 Lite: performance and validation — 2026-10-25

The optimized path retains AMD's 27,392-byte INT8 Lite initializer and its
three neural stages. Every frame evaluates the network. It does not reuse a
previous frame's neural result or use the earlier 4.1.1 model coefficients.

Linear input color is transformed once into an RGBA32F texture and read by
PRE and POST. This preserves float32 precision while avoiding repeated log
and input-texture work. Pipeline descriptors specify the actual threadgroup
size so Metal can allocate resources for the dispatched groups. The new
shader ABI uses a separate `_v2` cache, with content hashing and an atomic
manifest; repeated builds reuse complete matching bundles.

The compiler now selects AE/RCAS helpers by both color space and exposure
mode. The earlier implementation picked the first existing cache directory,
which could supply the nonlinear RCAS shader for a linear model. A clean
release builds the matching helpers if necessary.

On Apple M5, the recorded 864×560 → 1712×1112 sequence was replayed in three
serial rounds, reversing model order every other round. Each run used one
warmup sequence and 20 measured sequences. RCAS was disabled to match the
saved CrossOver preference; auto exposure remained enabled. All reset frames
were included. Timings cover the full GPU command buffer without compilation,
stage counters or diagnostic copies.

| Model | Measured frames | Mean | Median | P99 | Maximum |
| --- | ---: | ---: | ---: | ---: | ---: |
| Previous 4.1.2 Lite | 480 | 2.97 ms | 2.52 ms | 8.31 ms | 9.61 ms |
| Optimized 4.1.2 Lite | 480 | 2.70 ms | 2.48 ms | 3.87 ms | 4.93 ms |
| 4.1 Lossy 2 | 480 | 2.78 ms | 2.98 ms | 5.01 ms | 5.56 ms |

4.1.2 ran 480 full neural frames; Lossy 2 ran 240 full frames and reused
inference on 240 frames. These maxima describe this finite replay, not a
frame-time guarantee in a game. Other runs showed larger outliers for both
models. GPU scheduling and system load remain uncontrolled variables.

Validation: eight recorded RCAS-off output frames are byte-identical before
and after optimization. A GPU regression test compares optimized and uncached
paths across odd sizes, input/output size changes, subrects, exposure changes,
history resets, RCAS on/off and Ultra Performance, on classic Metal and Metal 4.
Recurrent and exposure state match exactly. Isolated output half values may
differ by up to 0.001953125 in synthetic HDR cases (mean error below 1.1e-7);
the test enforces finite output and explicit error bounds.

The same test compares native `optimize=0` with AMD's Lite shaders
translated to Metal bit for bit; see [DLL comparison](FSR412_DLL_PARITY.md).

Run `make test-fsr412-optimization` for the GPU checks. The interleaved
benchmark tool accepts both versions through each plan variant's
`fsr_version` field. Numeric results are stored in
[the benchmark report](reports/fsr412-m5-stability-2026-10-25.json).
