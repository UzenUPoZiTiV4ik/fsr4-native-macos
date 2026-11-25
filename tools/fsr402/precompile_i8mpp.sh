#!/bin/bash
# Precompile the int8 tensor-ops (i8mpp) shader variants the live CrossOver
# path needs, so the first game frame does not stall on DXC/metal.
#   tools/fsr402/precompile_i8mpp.sh [preset ...]      (default: quality balanced performance ultraperf)
# Tiers 1080 + 2160, depth inverted, low-res MV, no jitter cancellation,
# linear colour, auto-exposure off and on.
set -e
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PRESETS=("$@")
[ ${#PRESETS[@]} -eq 0 ] && PRESETS=(quality balanced performance ultraperf)
for preset in "${PRESETS[@]}"; do
  for res in 1080 2160; do
    idx=0; [ "$res" = 2160 ] && idx=1
    gdir="$ROOT/build/gen_i8mpp_${preset}_${res}"
    if [ ! -f "$gdir/i8mpp_manifest.json" ] || [ "$gdir/i8mpp_manifest.json" -ot "$ROOT/tools/fsr402/gen_i8mpp.py" ]; then
      python3 "$ROOT/tools/fsr402/gen_i8mpp.py" --root "$ROOT" --preset "$preset" --res "$res" --outdir "$gdir"
    fi
    off=$(python3 -c "import json;print(json.load(open('$gdir/i8mpp_manifest.json'))['post_ext_ct2d_off'])")
    for ae in 0 1; do
      python3 "$ROOT/tools/fsr402/compile_variant.py" --root "$ROOT" --preset "$preset" --res "$res" --res-idx "$idx" \
        --depth-inverted 1 --low-res-mv 1 --jittered-mv 0 --colorspace 0 --auto-exposure "$ae" \
        --debug-view 0 --rcas 0 --model i8 --post-ext-ct2d "$off" --builddir "$ROOT/build" | tail -1
    done
  done
done
