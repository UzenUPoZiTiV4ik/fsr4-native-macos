# fsr4-mac build
CXX      := clang++
FSR4_MIN_MACOS ?= 26.0
CXXFLAGS := -std=c++17 -O2 -Wall -mmacosx-version-min=$(FSR4_MIN_MACOS)
SDKROOT  := $(shell xcrun --sdk macosx --show-sdk-path)
FRAMEWORKS := -framework Metal -framework Foundation
ZLIB     := -lz
BINDIR   := tools/bin

all: $(BINDIR)/fsr4mac $(BINDIR)/gen_testcase $(BINDIR)/compare $(BINDIR)/bilin $(BINDIR)/bin2png

build/tests/fsr412_runtime_ab_test: tests/fsr412_runtime_ab_test.mm tests/fsr412_optimization_test.mm src/fsr412_runtime.h src/fsr411_runtime.h src/fsr412_session.inc Makefile src/fsr411_simd_weights.h src/fsr411_mpp_policy.h
	mkdir -p build/tests
	$(CXX) $(CXXFLAGS) -Wextra -Werror -fobjc-arc -isysroot $(SDKROOT) $< $(FRAMEWORKS) -o $@

# PRE's constant-address weights must not change a single bit.
test-fsr412-pre-weights: build/tests/fsr412_runtime_ab_test
	python3 tests/run_fsr412_runtime_ab.py --disable constant_initializer

.PHONY: test-fsr412-pre-weights

test: test-input test-settings test-fsr411-math test-installer test-fsr411-layout test-fsr411-cache test-fsr411-mpp-policy
	mkdir -p build/tests
	$(CXX) $(CXXFLAGS) tests/backend_policy_test.cpp -o build/tests/backend_policy_test
	build/tests/backend_policy_test
	python3 -m unittest discover -s tests -p 'test_*.py'
	python3 -m unittest discover -s tools/fsr411 -p 'test_*.py'

test-input:
	mkdir -p build/tests
	$(CXX) $(CXXFLAGS) -I src tests/input_contract_test.cpp -o build/tests/input_contract_test
	build/tests/input_contract_test

test-fsr411-math:
	mkdir -p build/tests
	$(CXX) $(CXXFLAGS) tests/fsr411_math_test.cpp -o build/tests/fsr411_math_test
	build/tests/fsr411_math_test

test-fsr411-layout:
	mkdir -p build/tests
	$(CXX) $(CXXFLAGS) -Wextra -Werror tests/fsr411_scratch_layout_test.cpp -o build/tests/fsr411_scratch_layout_test
	build/tests/fsr411_scratch_layout_test

test-settings:
	mkdir -p build/tests
	$(CXX) $(CXXFLAGS) -fobjc-arc -x objective-c++ -I src tests/fsr4_settings_test.cpp src/fsr4_live.mm src/fsr4_panel.mm $(FRAMEWORKS) -Wl,-U,_fsr411_encode_live_frame -Wl,-U,_fsr411_invalidate_owner -Wl,-U,_fsr411_release_owner -Wl,-U,_fsr412_encode_live_frame -Wl,-U,_fsr412_invalidate_owner -Wl,-U,_fsr412_release_owner -o build/tests/fsr4_settings_test
	build/tests/fsr4_settings_test default
	build/tests/fsr4_settings_test environment
	build/tests/fsr4_settings_test environment-fast
	build/tests/fsr4_settings_test environment-lossy2
	build/tests/fsr4_settings_test environment-int8-lite
	build/tests/fsr4_settings_test legacy-default
	build/tests/fsr4_settings_test environment-fsr4
	build/tests/fsr4_settings_test model
	build/tests/fsr4_settings_test file

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

build/tests/fsr411_native_test: tests/fsr411_native_test.mm src/fsr411_model.h Makefile src/fsr411_simd_weights.h src/fsr411_mpp_policy.h
	mkdir -p build/tests
	$(CXX) $(CXXFLAGS) -fobjc-arc -I src $< $(FRAMEWORKS) -o $@

build/tests/fsr411_quantize_test: tests/fsr411_quantize_test.mm src/fsr411_model.h Makefile
	mkdir -p build/tests
	$(CXX) $(CXXFLAGS) -Wextra -Werror -fobjc-arc -I src $< $(FRAMEWORKS) -o $@

build/tests/fsr411_quantize_test.metallib: tests/fsr411_quantize_test.metal src/fsr411_mpp.metal src/fsr411_model.h Makefile
	mkdir -p build/tests
	xcrun metal -std=metal4.0 -O2 -I src -c $< -o build/tests/fsr411_quantize_test.air
	xcrun metallib build/tests/fsr411_quantize_test.air -o $@

test-fsr411-quantize: build/tests/fsr411_quantize_test build/tests/fsr411_quantize_test.metallib
	build/tests/fsr411_quantize_test build/tests/fsr411_quantize_test.metallib

build/tests/fsr411_runtime_test: tests/fsr411_runtime_test.mm src/fsr411_runtime.h src/fsr411_model.h src/fsr411_scratch_layout.h src/fsr4_input.h Makefile src/fsr411_simd_weights.h src/fsr411_mpp_policy.h
	mkdir -p build/tests
	$(CXX) $(CXXFLAGS) -fobjc-arc -I src $< $(FRAMEWORKS) -o $@

build/tests/fsr411_snapshot_test: tests/fsr411_snapshot_test.mm src/fsr411_runtime.h src/fsr411_model.h src/fsr411_scratch_layout.h src/fsr4_input.h src/fsr4_log.h Makefile src/fsr411_simd_weights.h src/fsr411_mpp_policy.h
	mkdir -p build/tests
	$(CXX) $(CXXFLAGS) -fobjc-arc -I src $< $(FRAMEWORKS) -o $@

test-fsr411-snapshot: build/tests/fsr411_snapshot_test
	build/tests/fsr411_snapshot_test "$(CURDIR)"
	build/tests/fsr411_snapshot_test "$(CURDIR)" --mtl4

build/tests/fsr411_reset_clear_test: tests/fsr411_reset_clear_test.mm src/fsr411_runtime.h src/fsr411_model.h src/fsr411_scratch_layout.h src/fsr4_input.h Makefile src/fsr411_simd_weights.h src/fsr411_mpp_policy.h
	mkdir -p build/tests
	$(CXX) $(CXXFLAGS) -fobjc-arc -I src $< $(FRAMEWORKS) -o $@

test-fsr411-reset: build/tests/fsr411_reset_clear_test
	build/tests/fsr411_reset_clear_test "$(CURDIR)" "$(CURDIR)/build" --compile --profile lossy
	build/tests/fsr411_reset_clear_test "$(CURDIR)" "$(CURDIR)/build" --compile --profile lossy --mtl4
	build/tests/fsr411_reset_clear_test "$(CURDIR)" "$(CURDIR)/build" --compile --profile exact-fast
	build/tests/fsr411_reset_clear_test "$(CURDIR)" "$(CURDIR)/build" --compile --profile exact-fast --mtl4

.PHONY: test-fsr411-reset

build/tests/fsr411_cache_fallback_test: tests/fsr411_cache_fallback_test.mm src/fsr411_runtime.h src/fsr411_model.h src/fsr411_scratch_layout.h src/fsr4_input.h Makefile src/fsr411_simd_weights.h src/fsr411_mpp_policy.h
	mkdir -p build/tests
	$(CXX) $(CXXFLAGS) -fobjc-arc -I src $< $(FRAMEWORKS) -o $@

test-fsr411-cache: build/tests/fsr411_cache_fallback_test
	build/tests/fsr411_cache_fallback_test

test-fsr411-cache-gpu: build/tests/fsr411_cache_fallback_test
	python3 tools/compile_fsr411.py --root "$(CURDIR)" --builddir "$(CURDIR)/build" --res 1080 --profile lossy --optimize 1 --external-post-cnb 1 --post-cnb-alias 0
	python3 tools/compile_fsr411.py --root "$(CURDIR)" --builddir "$(CURDIR)/build" --res 1080 --profile lossy --optimize 1 --external-post-cnb 1 --post-cnb-alias 1
	python3 tools/gen_fsr411_native.py --tier 1080 --profile lossy --preset normal --outdir "$(CURDIR)/build/fsr411_native_1080_lossy"
	build/tests/fsr411_cache_fallback_test --gpu "$(CURDIR)" "$(CURDIR)/build"

.PHONY: test-fsr411-cache test-fsr411-cache-gpu

build/tests/fsr411_lossy_cadence_test: tests/fsr411_lossy_cadence_test.mm src/fsr411_runtime.h src/fsr411_model.h src/fsr411_scratch_layout.h src/fsr4_input.h Makefile src/fsr411_simd_weights.h src/fsr411_mpp_policy.h
	mkdir -p build/tests
	$(CXX) $(CXXFLAGS) -fobjc-arc -I src $< $(FRAMEWORKS) -o $@

build/tests/fsr411_mpp.metallib: src/fsr411_mpp.metal src/fsr411_model.h Makefile
	mkdir -p build/tests
	xcrun metal -std=metal4.0 -O2 -I src -c $< -o build/tests/fsr411_mpp.air
	xcrun metallib build/tests/fsr411_mpp.air -o $@

build/tests/fsr411_mpp_specialized.metallib: src/fsr411_mpp.metal src/fsr411_model.h Makefile
	mkdir -p build/tests
	xcrun metal -std=metal4.0 -O2 -DFSR411_SPECIALIZE=1 -I src -c $< -o build/tests/fsr411_mpp_specialized.air
	xcrun metallib build/tests/fsr411_mpp_specialized.air -o $@

# M5 always takes the cooperative chains; this library exercises their fallback.
build/tests/fsr411_mpp_fallback.metallib: src/fsr411_mpp.metal src/fsr411_model.h Makefile
	mkdir -p build/tests
	xcrun metal -std=metal4.0 -O2 -DFSR411_SPECIALIZE=1 -DFSR411_CHAIN_FORCE_FALLBACK=1 -I src -c $< -o build/tests/fsr411_mpp_fallback.air
	xcrun metallib build/tests/fsr411_mpp_fallback.air -o $@

build/tests/fsr411_reference.metallib: src/fsr411_reference.metal src/fsr411_model.h Makefile
	mkdir -p build/tests
	xcrun metal -std=metal3.0 -O2 -I src -c $< -o build/tests/fsr411_reference.air
	xcrun metallib build/tests/fsr411_reference.air -o $@

build/tests/fsr411_simd.metallib: src/fsr411_simd.metal src/fsr411_model.h Makefile
	mkdir -p build/tests
	xcrun metal -std=metal3.0 -O2 -I src -c $< -o build/tests/fsr411_simd.air
	xcrun metallib build/tests/fsr411_simd.air -o $@

build/tests/fsr411_simd_specialized.metallib: src/fsr411_simd.metal src/fsr411_model.h Makefile
	mkdir -p build/tests
	xcrun metal -std=metal3.0 -O2 -DFSR411_SPECIALIZE=1 -I src -c $< -o build/tests/fsr411_simd_specialized.air
	xcrun metallib build/tests/fsr411_simd_specialized.air -o $@

test-fsr411-native: build/tests/fsr411_native_test build/tests/fsr411_mpp.metallib build/tests/fsr411_mpp_specialized.metallib build/tests/fsr411_mpp_fallback.metallib build/tests/fsr411_reference.metallib build/tests/fsr411_simd.metallib build/tests/fsr411_simd_specialized.metallib
	@set -e; \
	for fsr411_profile in exact lossy; do \
	  for fsr411_preset in normal ultraperformance; do \
	    python3 tools/gen_fsr411_native.py --tier 1080 --profile $$fsr411_profile --preset $$fsr411_preset --outdir build/tests/fsr411_native; \
	    build/tests/fsr411_native_test build/tests/fsr411_mpp.metallib build/tests/fsr411_native; \
	    build/tests/fsr411_native_test build/tests/fsr411_mpp_specialized.metallib build/tests/fsr411_native --specialized; \
	    build/tests/fsr411_native_test build/tests/fsr411_mpp_fallback.metallib build/tests/fsr411_native --specialized; \
	    build/tests/fsr411_native_test build/tests/fsr411_reference.metallib build/tests/fsr411_native --reference; \
	    build/tests/fsr411_native_test build/tests/fsr411_simd.metallib build/tests/fsr411_native --simd; \
	    build/tests/fsr411_native_test build/tests/fsr411_simd_specialized.metallib build/tests/fsr411_native --simd-specialized; \
	  done; \
	done

test-fsr411-runtime: build/tests/fsr411_runtime_test
	python3 tests/run_fsr411_runtime.py

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
	python3 tests/run_fsr411_lossy.py

test-fsr411-cli: $(BINDIR)/fsr4mac
	python3 tests/run_fsr411_cli.py

test-fsr411-autoexposure:
	python3 tests/run_fsr411_autoexposure.py

test-fsr411-cadence: build/tests/fsr411_lossy_cadence_test
	build/tests/fsr411_lossy_cadence_test "$(CURDIR)" --compile
	build/tests/fsr411_lossy_cadence_test "$(CURDIR)" --compile --mtl4
	build/tests/fsr411_lossy_cadence_test "$(CURDIR)" --compile --inflight
	build/tests/fsr411_lossy_cadence_test "$(CURDIR)" --compile --inflight --mtl4

dist/crossover/libfsr4hook.dylib: Makefile tools/crossover/crossover_fsr4.py hook/fsr4_metalfx.m src/fsr4_live.mm src/fsr4_panel.mm src/fsr4_live.h src/fsr4_input.h src/fsr4_log.h src/fsr4_device.h src/fsr4_backend.h src/i8mpp_host.h src/fsr411_live.mm src/fsr411_runtime.h src/fsr411_capture_io.h src/fsr411_model.h src/fsr411_scratch_layout.h src/fsr412_live.mm src/fsr412_runtime.h src/fsr412_session.inc src/fsr411_simd_weights.h src/fsr411_mpp_policy.h
	python3 -c 'import sys; sys.path.insert(0, "tools/crossover"); import crossover_fsr4 as c; c.build_hook(c.DIST / "hook.log")'

dist/crossover/fsr411_live_regions_test: tests/fsr411_live_regions_test.mm src/fsr4_live.h src/fsr4_input.h dist/crossover/libfsr4hook.dylib Makefile
	$(CXX) $(CXXFLAGS) -fobjc-arc -arch x86_64 -I src $< dist/crossover/libfsr4hook.dylib $(FRAMEWORKS) -framework MetalFX -Wl,-rpath,"$(CURDIR)/dist/crossover" -o $@

test-fsr411-live: dist/crossover/fsr411_live_regions_test
	FSR4_COMPILE=1 arch -x86_64 $< "$(CURDIR)" i8mpp classic switch
	FSR4_COMPILE=1 arch -x86_64 $< "$(CURDIR)" i8mpp mtl4 switch

build/tests/fsr412_optimization_test: tests/fsr412_optimization_test.mm src/fsr412_runtime.h src/fsr411_runtime.h src/fsr412_session.inc Makefile src/fsr411_simd_weights.h src/fsr411_mpp_policy.h
	mkdir -p build/tests
	$(CXX) $(CXXFLAGS) -fobjc-arc -isysroot $(SDKROOT) $< $(FRAMEWORKS) -o $@

test-fsr412-optimization: build/tests/fsr412_optimization_test
	$< "$(CURDIR)"
	$< "$(CURDIR)" mtl4

.PHONY: test-fsr412-optimization

build/obj/fsr411_cli.o: Makefile src/fsr411_cli.mm src/fsr411_runtime.h src/fsr411_model.h src/fsr411_scratch_layout.h src/fsr412_session.inc src/fsr4_input.h src/fsr4_log.h src/pngio.h src/fsr411_simd_weights.h src/fsr411_mpp_policy.h
	mkdir -p build/obj
	$(CXX) $(CXXFLAGS) -fobjc-arc -isysroot $(SDKROOT) -c src/fsr411_cli.mm -o $@

build/obj/fsr412_cli.o: Makefile src/fsr412_cli.mm src/fsr412_runtime.h src/fsr411_runtime.h src/fsr411_model.h src/fsr411_scratch_layout.h src/fsr412_session.inc src/fsr4_input.h src/fsr4_log.h src/pngio.h src/fsr411_simd_weights.h src/fsr411_mpp_policy.h
	mkdir -p build/obj
	$(CXX) $(CXXFLAGS) -fobjc-arc -isysroot $(SDKROOT) -c src/fsr412_cli.mm -o $@

$(BINDIR)/fsr4mac: Makefile src/fsr4mac.mm src/pngio.h src/pass0_weights.h src/i8mpp_host.h src/fsr4_backend.h src/fsr4_device.h build/obj/fsr411_cli.o build/obj/fsr412_cli.o
	$(CXX) $(CXXFLAGS) -isysroot $(SDKROOT) $(FRAMEWORKS) $(ZLIB) -o $@ src/fsr4mac.mm build/obj/fsr411_cli.o build/obj/fsr412_cli.o

$(BINDIR)/gen_testcase: src/gen_testcase.cpp
	$(CXX) $(CXXFLAGS) -o $@ src/gen_testcase.cpp

$(BINDIR)/compare: src/compare.cpp src/pngio.h
	$(CXX) $(CXXFLAGS) $(ZLIB) -o $@ src/compare.cpp

$(BINDIR)/bilin: src/bilin.cpp src/pngio.h
	$(CXX) $(CXXFLAGS) $(ZLIB) -o $@ src/bilin.cpp

$(BINDIR)/bin2png: src/bin2png.cpp src/pngio.h
	$(CXX) $(CXXFLAGS) $(ZLIB) -o $@ src/bin2png.cpp
lossy2-weights:
	python3 tools/fsr411/fetch_lossy2_weights.py

# INT2 rounding ported from M5, tested through the actual pre-M5 shader.
build/tests/fsr411_simd_quantize_test.metallib: tests/fsr411_simd_quantize_test.metal src/fsr411_simd.metal src/fsr411_model.h Makefile
	mkdir -p build/tests
	xcrun metal -std=metal3.0 -O2 -I src -c $< -o build/tests/fsr411_simd_quantize_test.air
	xcrun metallib build/tests/fsr411_simd_quantize_test.air -o $@

test-fsr411-simd-quantize: build/tests/fsr411_quantize_test build/tests/fsr411_simd_quantize_test.metallib
	build/tests/fsr411_quantize_test build/tests/fsr411_simd_quantize_test.metallib --simd

# Fixed cache contracts for the on-device M5 register-chain selection.
test-fsr411-mpp-policy: build/tests/fsr411_mpp_policy_test
	$<

build/tests/fsr411_mpp_policy_test: tests/fsr411_mpp_policy_test.cpp src/fsr411_mpp_policy.h src/fsr411_model.h Makefile
	mkdir -p build/tests
	$(CXX) $(CXXFLAGS) -Wextra -Werror $< -o $@

build/tests/fsr411_mpp_bench: tools/bench/fsr411_mpp_bench.mm src/fsr411_model.h src/fsr411_mpp_policy.h Makefile
	mkdir -p build/tests
	$(CXX) $(CXXFLAGS) -fobjc-arc $< $(FRAMEWORKS) -o $@

build/tests/fsr411_mpp_ab: tools/bench/fsr411_mpp_ab.mm src/fsr412_runtime.h src/fsr411_runtime.h src/fsr411_mpp_policy.h src/fsr411_model.h src/fsr411_simd_weights.h src/fsr412_session.inc Makefile
	mkdir -p build/tests
	$(CXX) $(CXXFLAGS) -fobjc-arc -isysroot $(SDKROOT) $< $(FRAMEWORKS) -o $@

# Isolated GPU parity for all real Normal/Ultra/2160 models. No float network.
test-fsr411-mpp-fast: build/tests/fsr411_native_test build/tests/fsr411_mpp_specialized.metallib build/tests/fsr411_mpp_fallback.metallib
	@set -e; for dir in fsr412_native_1080 fsr412_native_1080_ultraperformance fsr412_native_2160 fsr411_native_1080 fsr411_native_1080_ultraperformance fsr411_native_2160 fsr411_native_1080_lossy2 fsr411_native_1080_ultraperformance_lossy2 fsr411_native_2160_lossy2; do \
	  test -f build/$$dir/fsr411_native_manifest.json || { echo "missing build/$$dir; import/generate models first"; exit 1; }; \
	  build/tests/fsr411_native_test build/tests/fsr411_mpp_specialized.metallib build/$$dir --mpp-fast; \
	  build/tests/fsr411_native_test build/tests/fsr411_mpp_fallback.metallib build/$$dir --mpp-fallback; \
	done

bench-fsr-m5: build/tests/fsr411_mpp_ab
	$< "$(CURDIR)" --rounds 60

.PHONY: test-fsr411-mpp-policy test-fsr411-mpp-fast bench-fsr-m5

build/tests/fsr411_simd_bench: tools/bench/fsr411_simd_bench.mm src/fsr411_model.h src/fsr411_simd_weights.h Makefile
	mkdir -p build/tests
	$(CXX) $(CXXFLAGS) -fobjc-arc $< $(FRAMEWORKS) -o $@

# Exact F(2x2,3x3) for the pre-M5 SIMD backend. CPU proof of the weight
# transform and bound gate, then raw (pre-quantization) and full-block GPU
# parity against the CPU integer oracle for every eligible real model.
build/tests/fsr411_winograd_weights_test: tests/fsr411_winograd_weights_test.cpp src/fsr411_simd_weights.h src/fsr411_model.h Makefile
	mkdir -p build/tests
	$(CXX) -std=c++17 -O2 -Wall -Wextra -Werror -fno-fast-math -ffp-contract=off -mmacosx-version-min=$(FSR4_MIN_MACOS) $< -o $@

build/tests/fsr411_winograd_test.metallib: tests/fsr411_winograd_test.metal src/fsr411_simd.metal src/fsr411_model.h Makefile
	mkdir -p build/tests
	xcrun metal -std=metal3.0 -O2 -DFSR411_SPECIALIZE=1 -I src -c $< -o build/tests/fsr411_winograd_test.air
	xcrun metallib build/tests/fsr411_winograd_test.air -o $@

test-fsr411-simd-winograd: build/tests/fsr411_winograd_weights_test build/tests/fsr411_native_test build/tests/fsr411_winograd_test.metallib
	build/tests/fsr411_winograd_weights_test
	@set -e; for dir in fsr412_native_1080 fsr412_native_1080_ultraperformance fsr412_native_2160 fsr411_native_1080 fsr411_native_1080_ultraperformance fsr411_native_2160 fsr411_native_1080_lossy2 fsr411_native_1080_ultraperformance_lossy2 fsr411_native_2160_lossy2; do \
	  test -f build/$$dir/fsr411_native_manifest.json || { echo "missing build/$$dir; import/generate models first"; exit 1; }; \
	  build/tests/fsr411_native_test build/tests/fsr411_winograd_test.metallib build/$$dir --simd-winograd; \
	  build/tests/fsr411_native_test build/tests/fsr411_winograd_test.metallib build/$$dir --simd-compact-ds; \
	done

# Full-frame forced-SIMD A/B in one process: direct 3x3 vs exact Winograd,
# interleaved, with a byte-identity check. Run on the target M1-M4 Mac.
build/tests/fsr411_simd_ab: tools/bench/fsr411_simd_ab.mm src/fsr412_runtime.h src/fsr411_runtime.h src/fsr411_simd_weights.h src/fsr412_session.inc Makefile src/fsr411_model.h src/fsr411_mpp_policy.h
	mkdir -p build/tests
	$(CXX) $(CXXFLAGS) -fobjc-arc -isysroot $(SDKROOT) $< $(FRAMEWORKS) -o $@

bench-fsr-pre-m5: build/tests/fsr411_simd_ab
	$< "$(CURDIR)" --rounds 30

.PHONY: test-fsr411-simd-winograd bench-fsr-pre-m5

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

.PHONY: test-fsr412-optimization test-fsr412-simd test-fsr411-simd-quantize test-fsr411-simd-lossy412 test-fsr-pre-m5

# Rosetta/CrossOver host integration, forced pre-M5 backend on any Apple7+ GPU.
test-fsr411-live-simd: dist/crossover/fsr411_live_regions_test
	FSR4_COMPILE=1 arch -x86_64 $< "$(CURDIR)" simd classic switch
	FSR4_COMPILE=1 arch -x86_64 $< "$(CURDIR)" simd mtl4 switch

.PHONY: test-fsr411-live-simd

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

.PHONY: installer test-installer
.PHONY: all clean lossy2-weights test test-gpu test-input test-settings test-fsr411-math test-fsr411-layout build-fsr411-tests test-fsr411-gpu test-fsr411-native test-fsr411-runtime test-fsr411-scalar test-fsr411-lossy test-fsr411-autoexposure test-fsr411-cadence test-fsr411-live test-fsr411-cli

# Run the signed-chroma regression before the remaining sequential GPU suites.
test-fsr411-gpu: test-fsr411-chroma

test-fsr411-chroma:
	python3 tests/run_fsr411_chroma.py

.PHONY: test-fsr411-chroma test-fsr411-quantize
