# fsr4-int8-reverse-engineering — open tooling and docs

Two related efforts share this repo. **Track 1 is a reverse-engineering
endeavor**: understanding AMD FidelityFX Super Resolution 4 (INT8 path) on
NVIDIA Turing (GTX 1650 / TU117) — static analysis of provider DLLs you
supply yourself, a D3D12 replay/benchmark harness, reimplemented 4.1.1
kernels, and the engineering notes. **Track 2 is an automatic
kernel-optimization flywheel, effectively a separate project built on top**:
it takes the RE track's outputs (operator contracts, weights, oracle shaders)
as inputs and produces faster kernels behind a fixed correctness oracle. The
tracks share fixtures and vocabulary, but neither requires the other — you
can read, use, or contribute to exactly one of them.

This is an **independent research project**. Not affiliated with, endorsed
by, or supported by AMD, NVIDIA, or CD PROJEKT RED.

## Status: research preview with a user-reported clean live run

- The 28-pass homegrown `reimpl411` chain (PRE + 12 neurals + 13 epilogues
  + POST, plus our AE kernel) is byte-exact against AMD's containers
  **in offline replay** (same pre-state → identical bytes; see
  `docs/evidence411/`).
- The assembled end-to-end chain now has a **user-reported visually-clean
  live run**: Quality, Balanced, and Performance all render without the
  earlier severe ghosting or the off-contract corruption (master §§28.5–28.8,
  `docs/evidence411/411_p48_multires.json`). The fix was host-side —
  a resolution-parametric dispatch contract derived from the AMD sample's
  semantics plus a decode of the captured constant-buffer header — not new
  kernels. This is a visual smoke report from one machine, not a pixel-level
  proof against AMD's output and not a performance measurement.
- Still do not ship this, install it into games, or present it as an FSR
  replacement. What you can build from these kernels is research-grade:
  exact offline, visually clean live in one user's testing, with open items
  (official A/B unconverged, 2160/4320-class coverage) tracked in the docs.

## Track 2: the automatic optimization loop (30-second version)

This is the second, standalone project. It is a machine for turning LLM
fluency into faster kernels without trading away correctness, and it runs on
the RE track's outputs without being part of it:

1. **Contract** — one operator's math, layouts, quantization, and edge cases
   written implementation-neutral (`bench/kernel_harness/kernels/*/CONTRACT.md`).
2. **Three backends, one evaluator** — HLSL (the byte-exact production
   oracle), CUDA, and Triton compete on the same fixtures with normalized
   JSON scoring (`correct` / `mismatches` / `gpu_us` / `registers`).
3. **Blind challenge** — `export_challenge.py` mints optimizer tasks with the
   known wins sanitized out, so a run tests rediscovery, not parroting.

The design is inspired by **KernelBench** (correctness gate + speedup scored
against a baseline, hidden tests as the honesty mechanism) — adapted from
"generate a kernel for this PyTorch op" to "beat this operator behind a fixed
oracle." The structural bet: agents write correct CUDA/Triton far faster than
DXIL-tuned HLSL (training-data fluency), so the research backends are where
iteration happens and HLSL is where truth lives. So far: Triton −20.6% and
CUDA −34.5% on two 4.0.2 kernels, all 20/20 exact — see
`bench/kernel_harness/PROVENANCE.md` for numbers, scope (4.0.2 only), and
caveats.

## What is (and isn't) here

| Included | Deliberately excluded |
|---|---|
| Our analysis scripts (PE/DXBC parsing, source-vs-binary comparison) | All AMD binaries (provider DLLs, extracted shader containers, weights) |
| D3D12 replay + benchmark harness (`bench/src`) | All game-derived data (captures, fixtures, PIX files, buffers) |
| Workload fuzz/corpus evaluation framework (`bench/workloads`) + synthetic corpus (runs with no captures) | The private real-data corpus (references local game captures) |
| INT8 provider **build scripts** (run against your own SDK checkout) | Any built provider DLL, especially under AMD's module filename |
| Reimplemented 4.1.1 kernels (`bench/shaders/reimpl411/`, MIT-derivation, byte-exact: full 28-pass homegrown chain + AE-ours) + validation evidence | The AMD DLLs themselves and anything raw extracted from them (containers, fixtures, captures) |
| Proven autoexposure kernel (`bench/shaders/candidates/ae_411_ours/ae_ours.hlsl`, stock-container-exact modulo the viewport guard) | Community-4.0.2b/c-derived constants |
| Engineering docs (`docs/`) | — |

Rule of thumb: **code structure is open, data is closed.** Anything that is
bytes copied from an AMD binary or pixels captured from a game stays on your
machine — with one principled exception: values AMD itself distributed under
MIT (the 4.1.1 provider, the 4.0.2 tree) may return inside reimplemented
kernels carrying AMD's notice. See `bench/shaders/reimpl411/README.md` and
`NOTICE.md` for the provenance record.

## Layout

**Track 1 — reverse engineering FSR4:**

- `analysis/` — DLL/shader inventory, source-vs-binary comparators,
  DXIL access-map tooling, output verifier.
  `analysis/capture_tools/` needs your own local captures (see its README).
- `bench/src/` (+ `bench/CMakeLists.txt`) — D3D12 harness: loads shaders +
  inputs, dispatches, times under a GPU lock, checks exactness.
- `bench/workloads/` — `correctness_fuzz.py` (byte-exactness trials),
  `evaluate_corpus.py` (timed corpus evaluation), `manage_holdout.py`,
  `pix_stats.py`, validators.
- `bench/shaders/candidates/ae_411_ours/ae_ours.hlsl` — proven
  autoexposure kernel (stock-container-exact modulo the viewport guard).
- `bench/shaders/reimpl411/` — 28 byte-exact 4.1.1 kernels (full homegrown
  chain: PRE + 12 neurals + 13 epilogues + POST) + dispatch manifest (see
  its README for provenance, evidence, and build commands).
  `bench/shaders/reimpl2160/` + `bench/shaders/reimpl4320/` carry the same
  27-pass chain at the 2160 class (3840x2160 output) and 4320 class
  (7680x4320 output), each with its own manifest + `docs/evidence411/2160/`
  + `docs/evidence411/4320/` records (all 0-mismatch; 2160/4320 oracles are
  synthetic same-prestate AMD-container legs — no captures exist above 1080).
- `bench/provider411/` — FFX-compatible provider host code (drives the 28
  homegrown kernels) + a bring-your-own-DLL build: supply the official
  4.1.1 DLL, and `bench/tools/build_411_dll.py` carves only the t18
  initializer out of it (hash-verified), compiles our 28 kernels + AE-ours,
  synthesizes the constant buffers, and links a working DLL locally. See
  its README. No AMD bytes vendored; the built DLL stays on your machine.
- `docs/evidence411/` — per-kernel validation records (oracle SHAs,
  cross-leg hashes, chain checks).
- `bench/tools/correctness_taxonomy.py` — shared correctness classifier used
  by the fuzz/eval scripts.
- `bench/workloads/synthetic_corpus.json` (+ generator) — fully synthetic,
  out-of-box runnable evaluation set (no captures).
- `tools/` — PowerShell build/probe/capture helpers for the FSR4 INT8
  provider workflow, plus the D3D12 adapter probe source.

**Track 2 — automatic optimization flywheel (standalone):**

- `bench/kernel_harness/` — multi-backend flywheel (HLSL oracle + CUDA/Triton
  research backends): contracts, our kernels, MIT-traced weight fixtures,
  blind-challenge exporter, curated results (see its README +
  `PROVENANCE.md`). **4.0.2-model scope: wins may not transfer to 4.1.x.**

**Shared by both tracks:**

- `docs/` — investigation notes. Fair warning: `FSR4_TU117_MASTER.md` is a
  **~2,400-line chronological LLM scratchpad, not documentation** — it is
  where the working agent thinks out loud (every measurement, dead end, and
  reversal, in order). Humans should read this README, the per-directory
  READMEs, and `PROVENANCE.md` files instead, and dip into the master only to
  source a specific claim. (`PROVIDER_411_INTEGRATION.md` covers the
  provider-compat work in normal prose.)
  Paths like `<repo>`, `~`, `<game-dir>` are placeholders for the original
  author's local machine.

## Prerequisites (all user-supplied)

- Windows 10/11, Visual Studio Build Tools (MSVC v143+), Windows SDK, DXC
- Your own checkout of FidelityFX-SDK (the `tools/Build-FSR4Int8Provider.ps1`
  flow expects it; nothing from it is vendored here)
- Your own official 4.1.1 provider DLL — only for the
  `bench/provider411` DLL build, which carves its non-reimplemented
  containers out of it at build time (hash-verified, never vendored)
- A Turing-class (or newer) NVIDIA GPU for the timing work; the static
  analysis runs anywhere Python 3 runs
- Optionally: your own PIX captures + a test game for the
  `capture_tools/` workflows

## Quick start (no proprietary data needed)

```powershell
# 1. Inspect a provider DLL you obtained yourself (read-only, never executes it)
python analysis/analyze_dlls.py --help
# 2. (Maintainer receipt — exits 2 with a pointer on fresh clones, which lack
#    the private DLL inventory; nothing is wrong with your checkout.)
python analysis/verify_outputs.py --help
# 3. Build the harness (CMake, VS generator) and run the synthetic path —
#    no captures needed for timing/determinism work:
#      cmake -S bench -B build -G "Visual Studio 18 2026" -A x64
#      cmake --build build --config Release
#      build/Release/fsr4bench.exe micro pass9 --shader stock --warmup 200 --iterations 500 --mode pseudo
#    (synthetic modes: pass6/7/8/9/11 --mode pseudo|zero|constant|ramp,
#    pre --preset <quality|...>, m411 synthetic-tensor timing. Anything
#    comparing against AMD output needs reference bytes you supply.)
# 4. Fuzz your own candidate shader against your own stock shader:
#      python bench/workloads/correctness_fuzz.py --stock <yours> --candidate <yours> --trials 200
# 5. Timed corpus evaluation on the synthetic set:
#      python bench/workloads/evaluate_corpus.py --stock <yours> --candidate <yours> --corpus bench/workloads/synthetic_corpus.json --split DEV --warmup 200 --iterations 500
#    (regenerate/extend the set: python bench/workloads/generate_synthetic_corpus.py --help)
# 6. Compile a reimplemented 4.1.1 kernel (no proprietary input needed):
#      dxc bench/shaders/reimpl411/reimpl_pass1.hlsl -T cs_6_6 -E reimpl_pass1 -HV 2021 -O3 -Fo reimpl_pass1.dxbc
#    (neural passes use -E reimpl_passN; the six passN_post epilogues use -E main)
# 7. (Needs your official 4.1.1 DLL + SDK checkout) Build a working provider DLL:
#      python bench/tools/build_411_dll.py --dll <official amd_fidelityfx_upscaler_dx12.dll> --ffx-sdk-root <SDK> --smoke
#    (see bench/provider411/README.md for what comes from where; output is local-only)
```

## Contributing / red lines

Do **not** open PRs adding: AMD DLLs, game captures or fixtures, built
`*.dll`s, personal paths, or release-group-named files. The `.gitignore`
blocks the obvious ones; the maintainer blocks the rest. Kernel contributions
must trace constants to MIT-distributed AMD files only (4.1.1 signed provider,
4.0.2 tree) and carry the header from `reimpl411/README.md`;
community-4.0.2b/c-derived constants are refused. Discussion of measured facts
(hashes, timings, dispatch geometry) is welcome — bulk binary data is not.

## License

MIT — see `LICENSE` (AMD's copyright and permission notice retained as the
license requires) and `NOTICE.md` for third-party terms, including credit to
the independent community reversing efforts this work builds on.
