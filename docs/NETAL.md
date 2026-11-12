# Experimental Netal DLSS Super Resolution

The CrossOver hook can now route a temporal MetalFX upscale to
[Netal](https://github.com/nschlaepfer/Netal) instead of AMD FSR. This is
**DLSS SR Preset K**, not DLSS frame generation or ray reconstruction.

## What is implemented and tested

- Optional, hash-pinned SR host source, compiled directly into our hook.
- One MetalFX hook and shared panel; mode `netal` (alias `dlss`, ID 9).
- Independent session/history per device and scaler, explicit geometry
  selection without changing the game process environment.
- Metal 3 and Metal 4 adapter entry points, including the scaler fence.
- Completion-aware lifetime: a scaler has at most one in-flight Netal
  encode in this initial integration. A busy scaler falls back to MetalFX
  and resets DLSS history on its next accepted frame. This is conservative
  and may reduce replacement coverage; it is not a performance claim.
- Full upstream runtime validation plus a Python preflight before install.
- arm64 and x86_64 builds, source checks, settings/input tests and a
  fallback test without proprietary models.

**Not yet verified here:** actual Netal inference, visual quality, game
compatibility or frame time. No complete SR model/runtime was available
for this integration. The FSR synthetic fixture is not a DLSS oracle.

## Compile without models

Supply a local Netal checkout at the pinned revision:

```sh
git clone https://github.com/nschlaepfer/Netal /path/to/Netal
git -C /path/to/Netal checkout bc50c7f4cc16bb2f86c25be16ce72ca4c8896356
make netal-source-check NETAL_SOURCE=/path/to/Netal
make test-netal-input
```

This compiles both native and CrossOver initialization probes. It does not
load NVIDIA weights or claim successful inference. Six selected source
files and `LICENSE` must match `third_party/netal/source_manifest.json`.
No automatic clone, model fetch or installation occurs.

## The remaining runtime dependency

A ready bundle must have `manifest.json` with schema
`netal.sr.runtime.v2`, model `DLSS_SR_Feature1_Quality_K`, complete asset
inventory and the supported model hashes. A source-only asset build is
**not** a ready bundle.

Netal's supported SR DLL is `nvngx_dlss.dll`, file version `310.8.0.0`,
SHA-256:

```text
c85f971ce023c9f3492fc7455f0b01a24ba18ea39636407a846902c4360b0b7e
```

A filename/version match alone is insufficient. Supply permitted local
inputs under their applicable terms. The model package and **four scalar
LG2/EX2 input/output tables** must be supplied separately. Each table contains
1,048,576 uint32 entries (4 MiB), 16 MiB in total.

Checked public release `v0.1.0`: the 5.6 MB DMG contains only the menu-bar
app and model identity metadata, not the compiled game runtime, weights
or missing scalar-contract inputs. The release notes explicitly state
that SR/FG must be provisioned separately. The available LG2 synthetic
characterization source is not a validated generator for these four tables.
A prepared permitted SR runtime from the maintainer would close this
input dependency; a custom approximate Metal math route would instead
need new numerical and image-quality validation. Those tables are
not provided by the DLL importer or the public Netal source. We do not
generate fake tables or accept another model under this name.

Upstream instructions:
- [Model setup](https://github.com/nschlaepfer/Netal/blob/bc50c7f4cc16bb2f86c25be16ce72ca4c8896356/docs/MODEL_SETUP.md)
- [Complete SR assembly](https://github.com/nschlaepfer/Netal/blob/bc50c7f4cc16bb2f86c25be16ce72ca4c8896356/docs/COMPLETE_SOURCE_BUILD.md)

## Validate and initialize your bundle

```sh
python3 tools/netal_integration.py validate --runtime /path/to/sr-runtime
python3 tools/netal_probe.py --source /path/to/Netal --runtime /path/to/sr-runtime
```

The probe verifies session/texture-adapter initialization and reports its
geometry and accelerated-layer coverage. It deliberately reports
`inference_executed: false`; an initialized model is not a tested game.

## Opt-in CrossOver installation

Close the game and its bottle first. Installation changes the selected
bottle, like the existing FSR installer; the original CrossOver app is
not modified.

```sh
python3 tools/crossover/crossover_fsr4.py install --panel \
  --profile netal --netal-source /path/to/Netal \
  --netal-runtime /path/to/sr-runtime --bottle /path/to/cxbottle.conf
```

The installer refuses missing or invalid inputs before changing a bottle.
It verifies the existing FSR route separately, then initializes Netal. The
panel offers `DLSS SR - Netal (experimental)` when the hook was built
with Netal and a runtime is configured. Switching to FSR resets the other
engine's history. RCAS and FSR diagnostics capture are disabled for Netal,
with saved sharpening preferences preserved.

To keep FSR initially while making Netal available, use `--profile int8-lite`
with the same `--netal-source` and `--netal-runtime` arguments.
`FSR4_NETAL_SOURCE` can select source for developer builds;
`FSR4_NETAL_RUNTIME` is the game's runtime path. With no configured path
the bridge uses `build/netal_runtime`. An `FSR4_VERSION=4.0.2` pin prevents
Netal selection. Do not also inject Netal's own game bridge.

Restore the bottle with the existing `restore --bottle ...` command; it
also removes `FSR4_NETAL_RUNTIME`.

## Input limits

The first bridge retains Netal's proven contract instead of pretending to
support every MetalFX frame. Unsupported input goes to original MetalFX.

- Known, non-jittered, input-resolution motion vectors, per-scaler owner.
- Reversed depth, pre-exposure 1, finite subpixel jitter.
- No automatic exposure or nonzero content/output offsets.
- Motion scale and texture formats must satisfy the upstream adapter.
- Different render/output extents get their own session/profile.
- macOS 26 APIs are compiled; some descriptor flags are unavailable on
  older MetalFX versions, so an unknown flag is conservatively rejected.

Optional upstream research flags such as
`NEURALSTUDIO_SR_ALLOW_ANY_JITTER=1` or
`NEURALSTUDIO_SR_ALLOW_ANY_MOTION_SCALE=1` are **not enabled automatically**.
They widen unverified game support, not the evidence for correctness.

## Distribution

Release packaging excludes `build/netal_src/` and `build/netal_runtime/`.
NVIDIA DLLs/models and translated GPU kernels are not bundled. Netal has
its own upstream provenance notices; source compile success is not a
license clearance or a universal DLSS compatibility claim.
