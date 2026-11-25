# fsr4-mac build
CXX      := clang++
FSR4_MIN_MACOS ?= 26.0
SRC_INC  := -I src -I src/common -I src/live -I src/fsr402 -I src/fsr41x -I src/anr -I src/netal
CXXFLAGS := -std=c++17 -O2 -Wall -mmacosx-version-min=$(FSR4_MIN_MACOS) $(SRC_INC)
SDKROOT  := $(shell xcrun --sdk macosx --show-sdk-path)
FRAMEWORKS := -framework Metal -framework Foundation
ZLIB     := -lz
BINDIR   := tools/bin

# fsr411_runtime.h hosts both 4.1.x sessions and includes their layout policies.
FSR41_RUNTIME_HEADERS := src/fsr41x/fsr411_runtime.h src/fsr41x/fsr411_model.h src/fsr41x/fsr411_scratch_layout.h \
    src/common/fsr4_input.h src/common/fsr4_log.h src/fsr41x/fsr411_simd_weights.h src/fsr41x/fsr411_mpp_policy.h \
    src/fsr41x/fsr412_neural_tile.h src/fsr41x/fsr412_session.inc

# Single-source test binary: $(1)=output, $(2)=source, $(3)=extra prerequisites,
# $(4)=extra compile flags, $(5)=extra link inputs after $(FRAMEWORKS).
define compile_mm
$(1): $(2) $(3) Makefile
	mkdir -p $$(@D)
	$$(CXX) $$(CXXFLAGS) -fobjc-arc $(4) $$< $$(FRAMEWORKS) $(5) -o $$@
endef

define compile_cpp
$(1): $(2) $(3) Makefile
	mkdir -p $$(@D)
	$$(CXX) $$(CXXFLAGS) $(4) $$< -o $$@
endef

all: $(BINDIR)/fsr4mac $(BINDIR)/gen_testcase $(BINDIR)/compare $(BINDIR)/bilin $(BINDIR)/bin2png

test: test-input test-settings test-fsr411-math test-installer test-fsr411-layout test-fsr411-cache test-fsr411-mpp-policy test-fsr412-options test-fsr412-neural-tiles test-fsr4-panel-settings test-netal-input test-anr-input build/tests/backend_policy_test
	build/tests/backend_policy_test
	python3 -m unittest discover -s tests -t . -p 'test_*.py'

$(eval $(call compile_cpp,build/tests/backend_policy_test,tests/core/backend_policy_test.cpp))
$(eval $(call compile_cpp,build/tests/input_contract_test,tests/core/input_contract_test.cpp))
$(eval $(call compile_cpp,build/tests/fsr411_math_test,tests/fsr411/fsr411_math_test.cpp))
$(eval $(call compile_cpp,build/tests/fsr411_scratch_layout_test,tests/fsr411/fsr411_scratch_layout_test.cpp,src/fsr41x/fsr411_scratch_layout.h,-Wextra -Werror))
$(eval $(call compile_cpp,build/tests/fsr412_neural_tile_test,tests/fsr412/fsr412_neural_tile_test.cpp,src/fsr41x/fsr412_neural_tile.h src/fsr41x/fsr411_model.h,-Wextra -Werror))
$(eval $(call compile_cpp,build/tests/fsr411_mpp_policy_test,tests/fsr411/fsr411_mpp_policy_test.cpp,src/fsr41x/fsr411_mpp_policy.h src/fsr41x/fsr411_model.h,-Wextra -Werror))
$(eval $(call compile_cpp,build/tests/netal_input_test,tests/netal/netal_input_test.cpp,src/netal/netal_live.h src/live/fsr4_live.h src/common/fsr4_input.h,-Wextra -Werror))
$(eval $(call compile_cpp,build/tests/anr_input_test,tests/anr/anr_input_test.cpp,src/anr/anr_live.h,-Wextra -Werror))

test-input: build/tests/input_contract_test
	$<

test-fsr411-math: build/tests/fsr411_math_test
	$<

test-fsr411-layout: build/tests/fsr411_scratch_layout_test
	$<

test-fsr411-mpp-policy: build/tests/fsr411_mpp_policy_test
	$<

test-fsr412-neural-tiles: build/tests/fsr412_neural_tile_test
	$<

test-netal-input: build/tests/netal_input_test
	$<

test-anr-input: build/tests/anr_input_test
	$<

test-settings: build/tests/fsr4_settings_test
	@set -e; for mode in default environment environment-fast environment-lossy2 environment-int8-lite environment-int8-lite-lossy environment-lite-lossy environment-7 environment-int8-lite-lossy-tiled environment-lite-lossy-tiled environment-8 legacy-default environment-fsr4 model netal file; do \
	  build/tests/fsr4_settings_test $$mode; \
	done

build/tests/fsr4_settings_test: tests/core/fsr4_settings_test.cpp src/live/fsr4_live.mm src/live/fsr4_panel.mm Makefile
	mkdir -p build/tests
	$(CXX) $(CXXFLAGS) -fobjc-arc -x objective-c++ tests/core/fsr4_settings_test.cpp src/live/fsr4_live.mm src/live/fsr4_panel.mm $(FRAMEWORKS) -Wl,-U,_fsr411_encode_live_frame -Wl,-U,_fsr411_invalidate_owner -Wl,-U,_fsr411_release_owner -Wl,-U,_fsr412_encode_live_frame -Wl,-U,_fsr412_invalidate_owner -Wl,-U,_fsr412_release_owner -Wl,-U,_fsr4_netal_encode_live_frame -Wl,-U,_fsr4_netal_invalidate_owner -Wl,-U,_fsr4_netal_release_owner -o build/tests/fsr4_settings_test

test-gpu: $(BINDIR)/fsr4mac $(BINDIR)/gen_testcase
	python3 tools/bench/check_backends.py
	$(MAKE) test-fsr411-gpu

# GPU suites run sequentially, even when the outer make uses -j. The native
# suite requires an Apple family 10 GPU; test-fsr411-scalar is portable.
test-fsr411-gpu:
	$(MAKE) test-fsr411-quantize
	$(MAKE) test-fsr411-native
	$(MAKE) test-fsr411-runtime
	$(MAKE) test-fsr411-scalar
	$(MAKE) test-fsr411-simd
	$(MAKE) test-fsr411-lossy
	$(MAKE) test-fsr411-autoexposure
	$(MAKE) test-fsr411-cadence
	$(MAKE) test-fsr411-reset
	$(MAKE) test-fsr411-cache-gpu
	$(MAKE) test-fsr411-live
	$(MAKE) test-fsr411-cli

build-fsr411-tests: build/tests/fsr411_native_test build/tests/fsr411_runtime_test build/tests/fsr411_lossy_cadence_test dist/crossover/fsr411_live_regions_test

$(eval $(call compile_mm,build/tests/fsr411_native_test,tests/fsr411/fsr411_native_test.mm,src/fsr41x/fsr411_model.h src/fsr41x/fsr411_simd_weights.h src/fsr41x/fsr411_mpp_policy.h))
$(eval $(call compile_mm,build/tests/fsr411_quantize_test,tests/fsr411/fsr411_quantize_test.mm,src/fsr41x/fsr411_model.h,-Wextra -Werror))

# $(1) is the Metal version; $(2) holds optional specialization flags.
define build_test_metallib
mkdir -p $(@D)
xcrun metal -std=$(1) -O2 $(strip $(2) -I src/fsr41x -I src/fsr41x/shaders) -c $< -o $(@:.metallib=.air)
xcrun metallib $(@:.metallib=.air) -o $@
endef

build/tests/fsr411_quantize_test.metallib: tests/fsr411/fsr411_quantize_test.metal src/fsr41x/shaders/fsr411_mpp.metal src/fsr41x/fsr411_model.h Makefile
	$(call build_test_metallib,metal4.0)

test-fsr411-quantize: build/tests/fsr411_quantize_test build/tests/fsr411_quantize_test.metallib
	build/tests/fsr411_quantize_test build/tests/fsr411_quantize_test.metallib

$(eval $(call compile_mm,build/tests/fsr411_runtime_test,tests/fsr411/fsr411_runtime_test.mm,$(FSR41_RUNTIME_HEADERS)))
$(eval $(call compile_mm,build/tests/fsr411_snapshot_test,tests/fsr411/fsr411_snapshot_test.mm,$(FSR41_RUNTIME_HEADERS)))
$(eval $(call compile_mm,build/tests/fsr411_reset_clear_test,tests/fsr411/fsr411_reset_clear_test.mm,$(FSR41_RUNTIME_HEADERS)))
$(eval $(call compile_mm,build/tests/fsr411_cache_fallback_test,tests/fsr411/fsr411_cache_fallback_test.mm,$(FSR41_RUNTIME_HEADERS)))
$(eval $(call compile_mm,build/tests/fsr411_lossy_cadence_test,tests/fsr411/fsr411_lossy_cadence_test.mm,$(FSR41_RUNTIME_HEADERS)))

test-fsr411-snapshot: build/tests/fsr411_snapshot_test
	build/tests/fsr411_snapshot_test "$(CURDIR)"
	build/tests/fsr411_snapshot_test "$(CURDIR)" --mtl4

test-fsr411-reset: build/tests/fsr411_reset_clear_test
	build/tests/fsr411_reset_clear_test "$(CURDIR)" "$(CURDIR)/build" --compile --profile lossy
	build/tests/fsr411_reset_clear_test "$(CURDIR)" "$(CURDIR)/build" --compile --profile lossy --mtl4
	build/tests/fsr411_reset_clear_test "$(CURDIR)" "$(CURDIR)/build" --compile --profile exact-fast
	build/tests/fsr411_reset_clear_test "$(CURDIR)" "$(CURDIR)/build" --compile --profile exact-fast --mtl4

test-fsr411-cache: build/tests/fsr411_cache_fallback_test
	build/tests/fsr411_cache_fallback_test

test-fsr411-cache-gpu: build/tests/fsr411_cache_fallback_test
	python3 tools/fsr411/compile_fsr411.py --root "$(CURDIR)" --builddir "$(CURDIR)/build" --res 1080 --profile lossy --optimize 1 --external-post-cnb 1 --post-cnb-alias 0
	python3 tools/fsr411/compile_fsr411.py --root "$(CURDIR)" --builddir "$(CURDIR)/build" --res 1080 --profile lossy --optimize 1 --external-post-cnb 1 --post-cnb-alias 1
	python3 tools/fsr411/gen_fsr411_native.py --tier 1080 --profile lossy --preset normal --outdir "$(CURDIR)/build/fsr411_native_1080_lossy"
	build/tests/fsr411_cache_fallback_test --gpu "$(CURDIR)" "$(CURDIR)/build"

build/tests/fsr411_mpp.metallib: src/fsr41x/shaders/fsr411_mpp.metal src/fsr41x/fsr411_model.h Makefile
	$(call build_test_metallib,metal4.0)

build/tests/fsr411_mpp_specialized.metallib: src/fsr41x/shaders/fsr411_mpp.metal src/fsr41x/fsr411_model.h Makefile
	$(call build_test_metallib,metal4.0,-DFSR411_SPECIALIZE=1)

# M5 always takes the cooperative chains; this library exercises their fallback.
build/tests/fsr411_mpp_fallback.metallib: src/fsr41x/shaders/fsr411_mpp.metal src/fsr41x/fsr411_model.h Makefile
	$(call build_test_metallib,metal4.0,-DFSR411_SPECIALIZE=1 -DFSR411_CHAIN_FORCE_FALLBACK=1)

build/tests/fsr411_reference.metallib: src/fsr41x/shaders/fsr411_reference.metal src/fsr41x/fsr411_model.h Makefile
	$(call build_test_metallib,metal3.0)

build/tests/fsr411_simd.metallib: src/fsr41x/shaders/fsr411_simd.metal src/fsr41x/fsr411_model.h Makefile
	$(call build_test_metallib,metal3.0)

build/tests/fsr411_simd_specialized.metallib: src/fsr41x/shaders/fsr411_simd.metal src/fsr41x/fsr411_model.h Makefile
	$(call build_test_metallib,metal3.0,-DFSR411_SPECIALIZE=1)

test-fsr411-native: build/tests/fsr411_native_test build/tests/fsr411_mpp.metallib build/tests/fsr411_mpp_specialized.metallib build/tests/fsr411_mpp_fallback.metallib build/tests/fsr411_reference.metallib build/tests/fsr411_simd.metallib build/tests/fsr411_simd_specialized.metallib
	@set -e; \
	for fsr411_profile in exact lossy; do \
	  for fsr411_preset in normal ultraperformance; do \
	    python3 tools/fsr411/gen_fsr411_native.py --tier 1080 --profile $$fsr411_profile --preset $$fsr411_preset --outdir build/tests/fsr411_native; \
	    build/tests/fsr411_native_test build/tests/fsr411_mpp.metallib build/tests/fsr411_native; \
	    build/tests/fsr411_native_test build/tests/fsr411_mpp_specialized.metallib build/tests/fsr411_native --specialized; \
	    build/tests/fsr411_native_test build/tests/fsr411_mpp_fallback.metallib build/tests/fsr411_native --specialized; \
	    build/tests/fsr411_native_test build/tests/fsr411_reference.metallib build/tests/fsr411_native --reference; \
	    build/tests/fsr411_native_test build/tests/fsr411_simd.metallib build/tests/fsr411_native --simd; \
	    build/tests/fsr411_native_test build/tests/fsr411_simd_specialized.metallib build/tests/fsr411_native --simd-specialized; \
	  done; \
	done

test-fsr411-runtime: build/tests/fsr411_runtime_test
	python3 tests/fsr411/runners/run_fsr411_runtime.py

test-fsr411-scalar: build/tests/fsr411_runtime_test
	build/tests/fsr411_runtime_test "$(CURDIR)" --compile --optimize --scalar
	build/tests/fsr411_runtime_test "$(CURDIR)" --compile --optimize --scalar --lossy --strict-math
	build/tests/fsr411_runtime_test "$(CURDIR)" --compile --optimize --scalar --ultra
	build/tests/fsr411_runtime_test "$(CURDIR)" --compile --optimize --scalar --ultra --lossy --strict-math

# Apple7+ simdgroup kernels: same exact parity as scalar, plus odd sizes,
# a multi-tile frame and the 2160 tier. Lossy compares against ordered
# PRE/POST math (--strict-math), like the scalar and runtime suites.
test-fsr411-simd: build/tests/fsr411_runtime_test
	build/tests/fsr411_runtime_test "$(CURDIR)" --compile --optimize --simd
	build/tests/fsr411_runtime_test "$(CURDIR)" --compile --optimize --simd --lossy --strict-math
	build/tests/fsr411_runtime_test "$(CURDIR)" --compile --optimize --simd --ultra
	build/tests/fsr411_runtime_test "$(CURDIR)" --compile --optimize --simd --ultra --lossy --strict-math
	build/tests/fsr411_runtime_test "$(CURDIR)" --compile --optimize --simd --odd --high-mv --jittered --inverted
	build/tests/fsr411_runtime_test "$(CURDIR)" --compile --optimize --simd --lossy --strict-math --size 301 173 603 347
	build/tests/fsr411_runtime_test "$(CURDIR)" --compile --optimize --simd --tier 2160 --lossy --strict-math --odd
	build/tests/fsr411_runtime_test "$(CURDIR)" --compile --optimize --simd --fast --relaxed-tolerance

test-fsr411-lossy:
	python3 tests/fsr411/runners/run_fsr411_lossy.py

test-fsr411-cli: $(BINDIR)/fsr4mac
	python3 tests/fsr411/runners/run_fsr411_cli.py

test-fsr411-autoexposure:
	python3 tests/fsr411/runners/run_fsr411_autoexposure.py

test-fsr411-cadence: build/tests/fsr411_lossy_cadence_test
	build/tests/fsr411_lossy_cadence_test "$(CURDIR)" --compile
	build/tests/fsr411_lossy_cadence_test "$(CURDIR)" --compile --mtl4
	build/tests/fsr411_lossy_cadence_test "$(CURDIR)" --compile --inflight
	build/tests/fsr411_lossy_cadence_test "$(CURDIR)" --compile --inflight --mtl4

dist/crossover/libfsr4hook.dylib: src/common/fsr4_fg_settings.h tools/crossover/crossover_fsr4.py hook/fsr4_metalfx.m src/live/fsr4_live.mm src/live/fsr4_panel.mm src/live/fsr4_live.h src/common/fsr4_device.h src/common/fsr4_backend.h src/fsr402/fsr4_provider.h src/fsr402/pass0_weights.h src/fsr402/i8mpp_host.h src/fsr41x/fsr411_live.mm src/live/fsr411_capture_io.h src/fsr41x/fsr412_live.mm src/live/fsr4_live_common.h src/fsr41x/fsr412_runtime.h src/netal/netal_live.h src/netal/netal_live.mm tools/netal/netal_integration.py third_party/netal/source_manifest.json src/anr/anr_runtime.h src/anr/anr_runtime.mm src/anr/anr_live.h src/anr/anr_live.mm src/anr/anr_helpers.metal hook/anr_metalfx.mm $(FSR41_RUNTIME_HEADERS) Makefile
	python3 -c 'import sys; sys.path.insert(0, "tools/crossover"); import crossover_fsr4 as c; c.build_hook(c.DIST / "hook.log")'

dist/crossover/fsr411_live_regions_test: tests/fsr411/fsr411_live_regions_test.mm src/live/fsr4_live.h src/common/fsr4_input.h dist/crossover/libfsr4hook.dylib Makefile
	$(CXX) $(CXXFLAGS) -fobjc-arc -arch x86_64 $< dist/crossover/libfsr4hook.dylib $(FRAMEWORKS) -framework MetalFX -Wl,-rpath,"$(CURDIR)/dist/crossover" -o $@

test-fsr411-live: dist/crossover/fsr411_live_regions_test
	FSR4_COMPILE=1 arch -x86_64 $< "$(CURDIR)" i8mpp classic switch
	FSR4_COMPILE=1 arch -x86_64 $< "$(CURDIR)" i8mpp mtl4 switch

$(eval $(call compile_mm,build/tests/fsr412_optimization_test,tests/fsr412/fsr412_optimization_test.mm,src/fsr41x/fsr412_runtime.h $(FSR41_RUNTIME_HEADERS),-isysroot $(SDKROOT)))

test-fsr412-optimization: build/tests/fsr412_optimization_test
	$< "$(CURDIR)"
	$< "$(CURDIR)" mtl4

build/obj/fsr411_cli.o build/obj/fsr412_cli.o: build/obj/%.o: src/fsr41x/%.mm src/fsr41x/fsr4_cli_recording.h src/common/pngio.h $(FSR41_RUNTIME_HEADERS) Makefile
	mkdir -p build/obj
	$(CXX) $(CXXFLAGS) -fobjc-arc -isysroot $(SDKROOT) -c $< -o $@

build/obj/fsr412_cli.o: src/fsr41x/fsr412_runtime.h

$(BINDIR)/fsr4mac: Makefile src/fsr402/fsr4mac.mm src/common/pngio.h src/fsr402/fsr4_provider.h src/fsr402/pass0_weights.h src/fsr402/i8mpp_host.h src/common/fsr4_backend.h src/common/fsr4_device.h build/obj/fsr411_cli.o build/obj/fsr412_cli.o
	$(CXX) $(CXXFLAGS) -isysroot $(SDKROOT) $(FRAMEWORKS) $(ZLIB) -o $@ src/fsr402/fsr4mac.mm build/obj/fsr411_cli.o build/obj/fsr412_cli.o

$(BINDIR)/gen_testcase: src/cli/gen_testcase.cpp
	$(CXX) $(CXXFLAGS) -o $@ src/cli/gen_testcase.cpp

$(BINDIR)/compare $(BINDIR)/bilin $(BINDIR)/bin2png: $(BINDIR)/%: src/cli/%.cpp src/common/pngio.h
	$(CXX) $(CXXFLAGS) $(ZLIB) -o $@ $<

lossy2-weights:
	python3 tools/fsr411/fetch_lossy2_weights.py

lite-lossy-weights:
	python3 tools/fsr412/fetch_lossy_weights.py

$(eval $(call compile_mm,build/tests/fsr412_options_test,tests/fsr412/fsr412_options_test.mm,src/fsr41x/fsr412_runtime.h src/fsr41x/fsr412_live.mm src/live/fsr4_live_common.h src/fsr41x/fsr412_cli.mm src/fsr41x/fsr4_cli_recording.h src/live/fsr411_capture_io.h src/common/pngio.h $(FSR41_RUNTIME_HEADERS),,$(ZLIB)))

test-fsr412-options: build/tests/fsr412_options_test
	$<

build/tests/fsr4_panel_settings_test build/tests/fsr4_panel_ui_test: build/tests/%: tests/core/%.mm tools/crossover/panel_app.mm Makefile
	mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -fobjc-arc $< -framework Foundation -framework AppKit -framework QuartzCore -o $@

test-fsr4-panel-settings: build/tests/fsr4_panel_settings_test
	$<

# Requires a macOS GUI session; does not launch a game or load the hook.
test-fsr4-panel-ui: build/tests/fsr4_panel_ui_test
	$<

$(eval $(call compile_mm,build/tests/fsr412_modes_ab,tools/bench/fsr412_modes_ab.mm,tools/bench/bench_gpu.h src/fsr41x/fsr412_runtime.h $(FSR41_RUNTIME_HEADERS)))

test-fsr412-lossy: build/tests/fsr412_modes_ab
	$< "$(CURDIR)"
	$< "$(CURDIR)" --mtl4

$(eval $(call compile_mm,build/tests/fsr412_new_look_gpu,tests/fsr412/fsr412_new_look_gpu.mm,tools/fsr412/new_look.py,-Wextra -Werror))

# Experimental new-look: scalar formulas, complete model/backend parity, switches.
test-fsr412-new-look: build/tests/fsr412_new_look_gpu build/tests/fsr412_modes_ab
	FSR412_NEW_LOOK_GPU_TEST=1 python3 -m unittest discover -s tests/fsr412 -p test_fsr412_new_look.py
	build/tests/fsr412_modes_ab "$(CURDIR)" --new-look
	build/tests/fsr412_modes_ab "$(CURDIR)" --new-look --mtl4
	build/tests/fsr412_modes_ab "$(CURDIR)" --new-look --ultra --size 88 48 256 160

bench-fsr412-modes: build/tests/fsr412_modes_ab
	$< "$(CURDIR)" --bench 30 --size 864 560 1712 1112

build/tests/fsr_modes_thermal: tools/bench/fsr_modes_thermal.mm tools/bench/bench_gpu.h src/live/fsr4_live.mm src/fsr41x/fsr411_live.mm src/fsr41x/fsr412_live.mm src/netal/netal_live.mm $(wildcard src/*/*.h src/fsr41x/*.inc) Makefile
	mkdir -p build/tests
	$(CXX) $(CXXFLAGS) -fobjc-arc -isysroot $(SDKROOT) $(filter %.mm,$^) $(FRAMEWORKS) $(ZLIB) -o $@

# INT2 rounding ported from M5, tested through the actual pre-M5 shader.
build/tests/fsr411_simd_quantize_test.metallib: tests/fsr411/fsr411_simd_quantize_test.metal src/fsr41x/shaders/fsr411_simd.metal src/fsr41x/fsr411_model.h Makefile
	$(call build_test_metallib,metal3.0)

test-fsr411-simd-quantize: build/tests/fsr411_quantize_test build/tests/fsr411_simd_quantize_test.metallib
	build/tests/fsr411_quantize_test build/tests/fsr411_simd_quantize_test.metallib --simd

$(eval $(call compile_mm,build/tests/fsr411_mpp_ab,tools/bench/fsr411_mpp_ab.mm,tools/bench/bench_gpu.h src/fsr41x/fsr412_runtime.h $(FSR41_RUNTIME_HEADERS),-isysroot $(SDKROOT)))

# Isolated GPU parity for all real Normal/Ultra/2160 models. No float network.
test-fsr411-mpp-fast: build/tests/fsr411_native_test build/tests/fsr411_mpp_specialized.metallib build/tests/fsr411_mpp_fallback.metallib
	@set -e; for dir in fsr412_native_1080 fsr412_native_1080_ultraperformance fsr412_native_2160 fsr411_native_1080 fsr411_native_1080_ultraperformance fsr411_native_2160 fsr411_native_1080_lossy2 fsr411_native_1080_ultraperformance_lossy2 fsr411_native_2160_lossy2; do \
	  test -f build/$$dir/fsr411_native_manifest.json || { echo "missing build/$$dir; import/generate models first"; exit 1; }; \
	  build/tests/fsr411_native_test build/tests/fsr411_mpp_specialized.metallib build/$$dir --mpp-fast; \
	  build/tests/fsr411_native_test build/tests/fsr411_mpp_fallback.metallib build/$$dir --mpp-fallback; \
	done

bench-fsr-m5: build/tests/fsr411_mpp_ab
	$< "$(CURDIR)" --rounds 60

$(eval $(call compile_mm,build/tests/fsr411_simd_bench,tools/bench/fsr411_simd_bench.mm,tools/bench/bench_gpu.h src/fsr41x/fsr411_model.h src/fsr41x/fsr411_simd_weights.h))

# Exact F(2x2,3x3) for the pre-M5 SIMD backend. CPU proof of the weight
# transform and bound gate, then raw (pre-quantization) and full-block GPU
# parity against the CPU integer oracle for every eligible real model.
build/tests/fsr411_winograd_weights_test: tests/fsr411/fsr411_winograd_weights_test.cpp src/fsr41x/fsr411_simd_weights.h src/fsr41x/fsr411_model.h Makefile
	mkdir -p build/tests
	$(CXX) -std=c++17 -O2 -Wall -Wextra -Werror -fno-fast-math -ffp-contract=off -mmacosx-version-min=$(FSR4_MIN_MACOS) $(SRC_INC) $< -o $@

build/tests/fsr411_winograd_test.metallib: tests/fsr411/fsr411_winograd_test.metal src/fsr41x/shaders/fsr411_simd.metal src/fsr41x/fsr411_model.h Makefile
	$(call build_test_metallib,metal3.0,-DFSR411_SPECIALIZE=1)

test-fsr411-simd-winograd: build/tests/fsr411_winograd_weights_test build/tests/fsr411_native_test build/tests/fsr411_winograd_test.metallib
	build/tests/fsr411_winograd_weights_test
	@set -e; for dir in fsr412_native_1080 fsr412_native_1080_ultraperformance fsr412_native_2160 fsr411_native_1080 fsr411_native_1080_ultraperformance fsr411_native_2160 fsr411_native_1080_lossy2 fsr411_native_1080_ultraperformance_lossy2 fsr411_native_2160_lossy2; do \
	  test -f build/$$dir/fsr411_native_manifest.json || { echo "missing build/$$dir; import/generate models first"; exit 1; }; \
	  build/tests/fsr411_native_test build/tests/fsr411_winograd_test.metallib build/$$dir --simd-winograd; \
	  build/tests/fsr411_native_test build/tests/fsr411_winograd_test.metallib build/$$dir --simd-compact-ds; \
	done

# Full-frame forced-SIMD A/B in one process: direct 3x3 vs exact Winograd,
# interleaved, with a byte-identity check. Run on the target M1-M4 Mac.
$(eval $(call compile_mm,build/tests/fsr411_simd_ab,tools/bench/fsr411_simd_ab.mm,tools/bench/bench_gpu.h src/fsr41x/fsr412_runtime.h $(FSR41_RUNTIME_HEADERS),-isysroot $(SDKROOT)))

bench-fsr-pre-m5: build/tests/fsr411_simd_ab
	$< "$(CURDIR)" --rounds 30

# Requires the imported/trained 4.1.2 Lossy weights (internal lossy2 profile).
test-fsr411-simd-lossy412: build/tests/fsr411_runtime_test
	$< "$(CURDIR)" --compile --optimize --simd --lossy2 --strict-math
	$< "$(CURDIR)" --compile --optimize --simd --lossy2 --strict-math --odd --high-mv --jittered --inverted
	$< "$(CURDIR)" --compile --optimize --simd --lossy2 --strict-math --ultra

# Forces the actual M1-M4 path even on M5. No MPP/tensor acceleration used.
test-fsr412-simd: build/tests/fsr412_optimization_test
	$< "$(CURDIR)" --simd
	$< "$(CURDIR)" mtl4 --simd

# Sequential GPU work; requires both imported Lite and trained Lossy assets.
test-fsr-pre-m5:
	$(MAKE) test-fsr411-simd-quantize
	$(MAKE) test-fsr411-simd-winograd
	$(MAKE) test-fsr411-simd
	$(MAKE) test-fsr411-simd-lossy412
	$(MAKE) test-fsr412-simd

# Rosetta/CrossOver host integration, forced pre-M5 backend on any Apple7+ GPU.
test-fsr411-live-simd: dist/crossover/fsr411_live_regions_test
	FSR4_COMPILE=1 arch -x86_64 $< "$(CURDIR)" simd classic switch
	FSR4_COMPILE=1 arch -x86_64 $< "$(CURDIR)" simd mtl4 switch

# Netal stays opt-in. Source compilation requires no NVIDIA model; an actual
# probe needs a separately supplied complete runtime (see docs/NETAL.md).
netal-source-check:
	python3 tools/netal/netal_probe.py --source "$(NETAL_SOURCE)" --build-only --arch arm64
	python3 tools/netal/netal_probe.py --source "$(NETAL_SOURCE)" --build-only --arch x86_64

build/anr_helpers_ui_v2.metallib: src/anr/anr_helpers.metal
	mkdir -p build
	xcrun metal -std=metal3.0 -O2 -c $< -o build/anr_helpers_ui_v2.air
	xcrun metallib build/anr_helpers_ui_v2.air -o $@

build/anr_tools/anr_replay: tools/anr/replay.mm src/anr/anr_runtime.mm src/anr/anr_runtime.h
	mkdir -p build/anr_tools
	$(CXX) $(CXXFLAGS) -Wextra -Werror -fobjc-arc tools/anr/replay.mm src/anr/anr_runtime.mm $(FRAMEWORKS) -o $@

build/anr_tools/anr_live_test: tests/anr/anr_live_test.mm src/common/fsr4_fg_settings.h src/anr/anr_live.h src/anr/anr_live.mm src/anr/anr_runtime.h src/anr/anr_runtime.mm hook/anr_metalfx.mm build/anr_helpers_ui_v2.metallib
	mkdir -p build/anr_tools
	$(CXX) $(CXXFLAGS) -Wextra -Werror -Wno-unused-parameter -fobjc-arc tests/anr/anr_live_test.mm src/anr/anr_live.mm src/anr/anr_runtime.mm hook/anr_metalfx.mm $(FRAMEWORKS) -framework MetalFX -framework QuartzCore -o $@

build/anr_tools/anr_motion_test: tests/anr/anr_motion_test.mm src/common/fsr4_fg_settings.h src/anr/anr_live.h src/anr/anr_live.mm src/anr/anr_runtime.h src/anr/anr_runtime.mm hook/anr_metalfx.mm build/anr_helpers_ui_v2.metallib
	mkdir -p build/anr_tools
	$(CXX) $(CXXFLAGS) -Wextra -Werror -fobjc-arc tests/anr/anr_motion_test.mm src/anr/anr_live.mm src/anr/anr_runtime.mm hook/anr_metalfx.mm $(FRAMEWORKS) -framework MetalFX -o $@

build/anr_tools/anr_ui_test: tests/anr/anr_ui_test.mm src/anr/anr_live.h build/anr_helpers_ui_v2.metallib
	mkdir -p build/anr_tools
	$(CXX) $(CXXFLAGS) -Wextra -Werror -fobjc-arc $< $(FRAMEWORKS) -o $@

test-anr-motion: build/anr_tools/anr_motion_test
	$< "$(CURDIR)" "$(ANR_RUNTIME)" classic
	$< "$(CURDIR)" "$(ANR_RUNTIME)" metal4

test-anr-ui: build/anr_tools/anr_ui_test
	$< "$(CURDIR)" classic
	$< "$(CURDIR)" metal4

test-anr-live: build/anr_tools/anr_live_test
	$< "$(CURDIR)" "$(ANR_RUNTIME)" classic
	$< "$(CURDIR)" "$(ANR_RUNTIME)" metal4
	ANR_TEST_HDR=1 $< "$(CURDIR)" "$(ANR_RUNTIME)" classic
	ANR_TEST_HDR=1 $< "$(CURDIR)" "$(ANR_RUNTIME)" metal4

test-anr-startup:
	python3 -m unittest discover -s tests/anr -p test_anr_startup.py

clean:
	rm -f $(BINDIR)/fsr4mac $(BINDIR)/gen_testcase $(BINDIR)/compare $(BINDIR)/bilin $(BINDIR)/bin2png
	rm -f build/obj/fsr411_cli.o build/obj/fsr412_cli.o

# Installer app (tools/installer): SwiftUI front end for crossover_fsr4.py.
INSTALLER_CORE := $(filter-out %/App.swift %/ContentView.swift,$(wildcard tools/installer/Sources/*.swift))
INSTALLER_VERSION ?= dev

installer:
	python3 tools/installer/build_app.py $(INSTALLER_VERSION) --dmg

test-installer:
	mkdir -p build/tests
	xcrun swiftc -swift-version 6 -target arm64-apple-macos$(FSR4_MIN_MACOS) $(INSTALLER_CORE) tests/installer/main.swift -o build/tests/installer_test
	build/tests/installer_test
	python3 tests/installer/check_strings.py

# Run the signed-chroma regression before the remaining sequential GPU suites.
test-fsr411-gpu: test-fsr411-chroma

test-fsr411-chroma:
	python3 tests/fsr411/runners/run_fsr411_chroma.py

.PHONY: all clean test test-gpu test-input test-settings test-fsr411-math test-fsr411-layout test-installer build-fsr411-tests test-fsr411-gpu test-fsr411-native test-fsr411-runtime test-fsr411-scalar test-fsr411-simd test-fsr411-lossy test-fsr411-autoexposure test-fsr411-cadence test-fsr411-live test-fsr411-cli test-fsr411-snapshot test-fsr411-reset test-fsr411-cache test-fsr411-cache-gpu test-fsr411-quantize test-fsr411-chroma test-fsr411-simd-quantize test-fsr411-mpp-policy test-fsr411-mpp-fast bench-fsr-m5 test-fsr411-simd-winograd bench-fsr-pre-m5 test-fsr411-simd-lossy412 test-fsr-pre-m5 test-fsr411-live-simd test-fsr412-optimization test-fsr412-simd test-fsr412-options test-fsr412-neural-tiles test-fsr412-lossy test-fsr412-new-look bench-fsr412-modes test-fsr4-panel-settings test-fsr4-panel-ui test-netal-input netal-source-check test-anr-input test-anr-live test-anr-ui test-anr-motion test-anr-startup lite-lossy-weights lossy2-weights installer test-installer
