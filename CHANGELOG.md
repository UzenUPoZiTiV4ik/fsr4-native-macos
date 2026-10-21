# Releases

## v1.9 (2026-10-21)

CrossOver release of the FSR 4.1.1 stack with the Lite 2 profile.

- Modes in the panel: FSR 4.1 (Exact), FSR 4.1 Exact Fast, FSR 4.1 Lite
  (`lossy`, default), FSR 4.1 Lite 2 (`lossy2`), and FSR 4 for comparison.
- Lite 2 uses the six trained integer layers from upstream
  `dll-2026-10-19`. Its weights are GPL-2.0-or-later upstream data and are not
  vendored into this MIT tree: `make lossy2-weights` fetches one pinned file
  into `build/fsr411_lossy2/` and checks its SHA-256. The panel shows the mode
  once those weights exist.
- Adds Lite 2 to the public release: v1.8 had the FSR 4.1 and FSR 4.1 Lite
  modes only. Also includes the at-rest dark-band guard.

Checked with `make test` (65 Python tests across the input, settings, math,
layout, cache and lossy2 suites), `make test-settings` including
`environment-lossy2`, `make test-installer`, and a Lite 2 run on the recorded
sequence: 3.9 ms for a full frame, 1.4 ms when inference is reused.

## Older releases

`crossover-v1.0` through `crossover-v1.8` are tagged in this repository. v1.8
fixed external bottle discovery in both installers; v1.1 to v1.7 were the FSR
4.1 port, its SIMD backend, memory work, image-quality update, installer app
and panel changes; v1.0 was the first FSR 4.1-only package.
