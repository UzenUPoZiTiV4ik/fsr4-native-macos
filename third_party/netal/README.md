# Optional Netal DLSS SR source

Upstream: https://github.com/nschlaepfer/Netal

Pinned commit: `bc50c7f4cc16bb2f86c25be16ce72ca4c8896356` (MIT).
`source_manifest.json` records the SHA-256 of the six SR host files and
the upstream license used by this integration. No upstream source, model
weights, NVIDIA DLL, compiled library or SDK header is vendored here.

`tools/netal_integration.py` accepts an explicit local checkout, verifies
every selected file and creates a source-only build copy under `build/`.
Changes in that copy are limited to private Objective-C symbol names and
thread-local profile/render-size accessors. Numerical code is unchanged.
The original checkout is never modified, and its MIT license is copied
into the local source build.

Only `dlss_sr_session.mm` and `dlss_sr_texture_adapter.mm` are linked. Netal
FG, NR, menu-bar app and game bridges are excluded, avoiding two competing
MetalFX hooks. Source availability does not grant NVIDIA-model distribution
rights. See [setup and limits](../../docs/NETAL.md).
