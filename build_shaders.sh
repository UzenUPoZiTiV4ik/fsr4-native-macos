#!/usr/bin/env bash
# Compile all FSR4 INT8 shader variants needed for the default test
# configurations, via tools/fsr402/compile_variant.py (DXC->SPV patch->SPIRV-Cross->MSL).
#
#   ./build_shaders.sh                    # all 6 presets x 1080 (default flags)
#   ./build_shaders.sh --all              # all presets x {1080,2160,4320}
#   ./build_shaders.sh --preset quality --res 1080
#
# Variants beyond the flag combos listed here are compiled on demand by
# fsr4mac itself (it invokes compile_variant.py when the cache misses).
set -euo pipefail
cd "$(dirname "$0")"

# bash 3.2 compatible (stock macOS /bin/bash): no associative arrays.
res_idx() {
    case "$1" in
        1080) echo 0;;
        2160) echo 1;;
        4320) echo 2;;
        *) echo "unknown res: $1" >&2; exit 1;;
    esac
}

PRESETS="native quality balanced performance ultraperf drs"
RESES="1080"
FILTER_PRESET=""

while [ $# -gt 0 ]; do
    case "$1" in
        --all) RESES="1080 2160 4320";;
        --preset) FILTER_PRESET="$2"; shift;;
        --res) RESES="$2"; shift;;
        *) echo "unknown arg: $1" >&2; exit 1;;
    esac
    shift
done

for p in $PRESETS; do
    [ -n "$FILTER_PRESET" ] && [ "$p" != "$FILTER_PRESET" ] && continue
    for r in $RESES; do
        echo "== $p @ $r =="
        python3 tools/fsr402/compile_variant.py --root "$PWD" --preset "$p" \
            --res "$r" --res-idx "$(res_idx "$r")" \
            --depth-inverted 0 --low-res-mv 1 --jittered-mv 0 \
            --colorspace 0 --auto-exposure 0 --debug-view 0 --rcas 0 \
            --builddir build
    done
done
