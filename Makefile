# fsr4-mac build
CXX      := clang++
FSR4_MIN_MACOS ?= 26.0
CXXFLAGS := -std=c++17 -O2 -Wall -mmacosx-version-min=$(FSR4_MIN_MACOS)
SDKROOT  := $(shell xcrun --sdk macosx --show-sdk-path)
FRAMEWORKS := -framework Metal -framework Foundation
ZLIB     := -lz
BINDIR   := tools/bin

all: $(BINDIR)/fsr4mac $(BINDIR)/gen_testcase $(BINDIR)/compare $(BINDIR)/bilin $(BINDIR)/bin2png

test: test-input test-settings test-fsr411-math
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

test-settings:
	mkdir -p build/tests
	$(CXX) $(CXXFLAGS) -fobjc-arc -x objective-c++ -I src tests/fsr4_settings_test.cpp src/fsr4_live.mm src/fsr4_panel.mm $(FRAMEWORKS) -Wl,-U,_fsr411_encode_live_frame -Wl,-U,_fsr411_invalidate_owner -Wl,-U,_fsr411_release_owner -o build/tests/fsr4_settings_test
	build/tests/fsr4_settings_test default
	build/tests/fsr4_settings_test environment
	build/tests/fsr4_settings_test environment-fast
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
	$(MAKE) test-fsr411-lossy
	$(MAKE) test-fsr411-autoexposure
	$(MAKE) test-fsr411-cadence
	$(MAKE) test-fsr411-live
	$(MAKE) test-fsr411-cli

build-fsr411-tests: build/tests/fsr411_native_test build/tests/fsr411_runtime_test build/tests/fsr411_lossy_cadence_test dist/crossover/fsr411_live_regions_test

build/tests/fsr411_native_test: tests/fsr411_native_test.mm src/fsr411_model.h Makefile
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

build/tests/fsr411_runtime_test: tests/fsr411_runtime_test.mm src/fsr411_runtime.h src/fsr411_model.h src/fsr4_input.h Makefile
	mkdir -p build/tests
	$(CXX) $(CXXFLAGS) -fobjc-arc -I src $< $(FRAMEWORKS) -o $@

build/tests/fsr411_snapshot_test: tests/fsr411_snapshot_test.mm src/fsr411_runtime.h src/fsr411_model.h src/fsr4_input.h src/fsr4_log.h Makefile
	mkdir -p build/tests
	$(CXX) $(CXXFLAGS) -fobjc-arc -I src $< $(FRAMEWORKS) -o $@

test-fsr411-snapshot: build/tests/fsr411_snapshot_test
	build/tests/fsr411_snapshot_test "$(CURDIR)"
	build/tests/fsr411_snapshot_test "$(CURDIR)" --mtl4

build/tests/fsr411_lossy_cadence_test: tests/fsr411_lossy_cadence_test.mm src/fsr411_runtime.h src/fsr411_model.h src/fsr4_input.h Makefile
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

test-fsr411-native: build/tests/fsr411_native_test build/tests/fsr411_mpp.metallib build/tests/fsr411_mpp_specialized.metallib build/tests/fsr411_mpp_fallback.metallib build/tests/fsr411_reference.metallib
	@set -e; \
	for fsr411_profile in exact lossy; do \
	  for fsr411_preset in normal ultraperformance; do \
	    python3 tools/gen_fsr411_native.py --tier 1080 --profile $$fsr411_profile --preset $$fsr411_preset --outdir build/tests/fsr411_native; \
	    build/tests/fsr411_native_test build/tests/fsr411_mpp.metallib build/tests/fsr411_native; \
	    build/tests/fsr411_native_test build/tests/fsr411_mpp_specialized.metallib build/tests/fsr411_native --specialized; \
	    build/tests/fsr411_native_test build/tests/fsr411_mpp_fallback.metallib build/tests/fsr411_native --specialized; \
	    build/tests/fsr411_native_test build/tests/fsr411_reference.metallib build/tests/fsr411_native --reference; \
	  done; \
	done

test-fsr411-runtime: build/tests/fsr411_runtime_test
	python3 tests/run_fsr411_runtime.py

test-fsr411-scalar: build/tests/fsr411_runtime_test
	build/tests/fsr411_runtime_test "$(CURDIR)" --compile --optimize --scalar
	build/tests/fsr411_runtime_test "$(CURDIR)" --compile --optimize --scalar --lossy
	build/tests/fsr411_runtime_test "$(CURDIR)" --compile --optimize --scalar --ultra
	build/tests/fsr411_runtime_test "$(CURDIR)" --compile --optimize --scalar --ultra --lossy

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

dist/crossover/libfsr4hook.dylib: Makefile tools/crossover/crossover_fsr4.py hook/fsr4_metalfx.m src/fsr4_live.mm src/fsr4_panel.mm src/fsr4_live.h src/fsr4_input.h src/fsr4_log.h src/fsr4_device.h src/fsr4_backend.h src/i8mpp_host.h src/fsr411_live.mm src/fsr411_runtime.h src/fsr411_capture_io.h src/fsr411_model.h
	python3 -c 'import sys; sys.path.insert(0, "tools/crossover"); import crossover_fsr4 as c; c.build_hook(c.DIST / "hook.log")'

dist/crossover/fsr411_live_regions_test: tests/fsr411_live_regions_test.mm src/fsr4_live.h src/fsr4_input.h dist/crossover/libfsr4hook.dylib Makefile
	$(CXX) $(CXXFLAGS) -fobjc-arc -arch x86_64 -I src $< dist/crossover/libfsr4hook.dylib $(FRAMEWORKS) -framework MetalFX -Wl,-rpath,"$(CURDIR)/dist/crossover" -o $@

test-fsr411-live: dist/crossover/fsr411_live_regions_test
	FSR4_COMPILE=1 arch -x86_64 $< "$(CURDIR)" i8mpp classic switch
	FSR4_COMPILE=1 arch -x86_64 $< "$(CURDIR)" i8mpp mtl4 switch

build/obj/fsr411_cli.o: Makefile src/fsr411_cli.mm src/fsr411_runtime.h src/fsr411_model.h src/fsr4_input.h src/fsr4_log.h src/pngio.h
	mkdir -p build/obj
	$(CXX) $(CXXFLAGS) -fobjc-arc -isysroot $(SDKROOT) -c src/fsr411_cli.mm -o $@

$(BINDIR)/fsr4mac: Makefile src/fsr4mac.mm src/pngio.h src/pass0_weights.h src/i8mpp_host.h src/fsr4_backend.h src/fsr4_device.h build/obj/fsr411_cli.o
	$(CXX) $(CXXFLAGS) -isysroot $(SDKROOT) $(FRAMEWORKS) $(ZLIB) -o $@ src/fsr4mac.mm build/obj/fsr411_cli.o

$(BINDIR)/gen_testcase: src/gen_testcase.cpp
	$(CXX) $(CXXFLAGS) -o $@ src/gen_testcase.cpp

$(BINDIR)/compare: src/compare.cpp src/pngio.h
	$(CXX) $(CXXFLAGS) $(ZLIB) -o $@ src/compare.cpp

$(BINDIR)/bilin: src/bilin.cpp src/pngio.h
	$(CXX) $(CXXFLAGS) $(ZLIB) -o $@ src/bilin.cpp

$(BINDIR)/bin2png: src/bin2png.cpp src/pngio.h
	$(CXX) $(CXXFLAGS) $(ZLIB) -o $@ src/bin2png.cpp

clean:
	rm -f $(BINDIR)/fsr4mac $(BINDIR)/gen_testcase $(BINDIR)/compare $(BINDIR)/bilin $(BINDIR)/bin2png
	rm -f build/obj/fsr411_cli.o

.PHONY: all clean test test-gpu test-input test-settings test-fsr411-math build-fsr411-tests test-fsr411-gpu test-fsr411-native test-fsr411-runtime test-fsr411-scalar test-fsr411-lossy test-fsr411-autoexposure test-fsr411-cadence test-fsr411-live test-fsr411-cli

# Run the signed-chroma regression before the remaining sequential GPU suites.
test-fsr411-gpu: test-fsr411-chroma

test-fsr411-chroma:
	python3 tests/run_fsr411_chroma.py

.PHONY: test-fsr411-chroma test-fsr411-quantize
