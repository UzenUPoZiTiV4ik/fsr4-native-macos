# FSR 4.1.2 Lite: comparison with the DLL model — 2026-10-25

Three separate claims, with different evidence:

1. **Weights and shaders are AMD's.** Both 27,392-byte Lite initializers
   (normal `4b9898da…`, Ultra Performance `4c57292d…`) are byte-identical to
   the data in `amdxcffx64.dll` and `amdvkffx64.dll`. All 170 unique Lite
   SPIR-V payloads match `amdvkffx64.dll`, and every translated `.metal`
   source matches its import receipt hash.
2. **The optimized path computes the same numbers as AMD's shaders.** The
   reference backend (`--reference --optimize 0`) runs the DLL's Lite SPIR-V
   translated one to one to Metal. The shipping path (`--i8mpp --optimize 1`)
   uses native INT8 matrix kernels and the float32 input-color cache. On every
   compared case, output, history, reprojected color, recurrent state and auto
   exposure are byte-identical, with no nonfinite values.
3. **Not done: running the Windows DLL.** Both paths share the Metal frame
   setup, input upload and the AE/RCAS helpers. This does not prove equality
   with AMD's Windows runtime output.

| Case | Frames | Output | RGB changed | Max abs error | Temporal state |
| --- | ---: | --- | ---: | ---: | --- |
| Recorded, RCAS off | 8 | 1712×1112 | 0 | 0 | identical |
| Recorded, RCAS on (0.498) | 8 | 1712×1112 | 0 | 0 | identical |
| Recorded, Ultra Performance | 4 | 1712×1112 | 0 | 0 | identical |
| Synthetic odd size, Quality | 2 | 127×79 | 0 | 0 | identical |
| Synthetic odd size, Ultra Performance | 2 | 127×79 | 0 | 0 | identical |

Formats follow the runtime dumps: output, history and reprojected are
RGBA16Float, recurrent is RGBA8Unorm and auto exposure is R32Float. PSNR is
infinite for every case. The report also defines RGB-peak, [0, 1] and 8-bit
PSNR ranges. Scratch arenas are not compared, because the optimized path
reuses intermediate memory after pass 3 by design.

## Fixed before release

The first run found a real difference in Ultra Performance only. PRE and
passes 1–3 matched, but the native POST CNB used the normal-preset
quantizers. AMD's Ultra POST rounds the 3×3 convolution by 2^7 and resolves
the residual as `rint((x/16 + acc/1024) * 8)`. Before the fix, the 127×79
case changed 99.97% of RGB pixels: max error 0.189, PSNR 32.1 dB, recurrent
state off by up to 85/255. `tools/fsr412/native_model.py` now uses the Ultra
shifts and every case matches. Normal presets were not affected.

`make test-fsr412-optimization` now also compares `Backend::Reference` with
native `optimize=0` bit for bit for normal, auto-exposure and Ultra
Performance modes on classic Metal and Metal 4. Before the fix, this test
fails in Ultra mode.

## Reproduce

```sh
tools/bin/fsr4mac --fsr-version 4.1.2 CASE REF --reference --optimize 0 --dump-state
tools/bin/fsr4mac --fsr-version 4.1.2 CASE OPT --i8mpp --optimize 1 --dump-state
python3 tools/diagnostics/fsr412_dll_parity.py --case NAME=REF:OPT --report parity.json
```

The reference backend is intentionally slow (about 30 ms per frame at this
size). Per-case metrics and provenance:
[the parity report](reports/fsr412-dll-parity-2026-10-25.json).
