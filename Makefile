# Makefile — one-command entry points for the MatMul-Free LM C++ engine.
#
# This is a thin, optional wrapper: everything here is a plain cmake/python command you
# can also run by hand (each recipe echoes what it does). The engine itself needs only a
# C++17 compiler and CMake — no Python at build or run time.
#
#   make                 # configure + build      -> cpp/build/mmfree-cli
#   make model           # download + pack a checkpoint (needs the tools/ deps once)
#   make run PROMPT="Hello there"
#   make bench
#   make help            # every target and knob
#
# Portability: POSIX hosts (Linux, macOS, *BSD) on any CPU. On Windows use WSL — the
# weight loader mmap()s the blob. Needs GNU make (macOS ships it as `make`; on *BSD use
# `gmake`) — the recipes use $(if ...) and $(shell ...).
.DEFAULT_GOAL := build

# ─── knobs (override on the command line: `make build SIMD=SCALAR`) ──────────────
SRC        := cpp
BUILD      ?= cpp/build
BUILD_TYPE ?= Release
# getconf is POSIX and works on both Linux and macOS (nproc is Linux-only).
JOBS       ?= $(shell getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)
# Empty = let CMake auto-detect (AVX-512/AVX2/NEON/scalar). SCALAR|X86|X86_512|ARM to pin.
SIMD       ?=
TUNE       ?= OFF
CMAKE      ?= cmake
PYTHON     ?= python3

MODEL      ?= ridger/MMfreeLM-370M
OUT        ?= $(SRC)
BLOB       := $(OUT)/model.mmfree

CLI        := $(BUILD)/mmfree-cli
PROMPT     ?= In a shocking finding, scientists discovered
GEN        ?= 32
REPS       ?= 3
ARGS       ?=

CMAKE_ARGS := -DCMAKE_BUILD_TYPE=$(BUILD_TYPE) -DMMFREE_TUNE_NATIVE=$(TUNE) \
              $(if $(SIMD),-DMMFREE_SIMD=$(SIMD),)

.PHONY: all build configure model run bench profile test smoke install clean distclean help

all: build

# ─── build ───────────────────────────────────────────────────────────────────────
# Re-runs configure only when the cache is missing; cmake itself re-configures on
# CMakeLists.txt changes, so `make build` after an edit is always safe.
configure: $(BUILD)/CMakeCache.txt
$(BUILD)/CMakeCache.txt: $(SRC)/CMakeLists.txt
	$(CMAKE) -S $(SRC) -B $(BUILD) $(CMAKE_ARGS)

build: configure
	$(CMAKE) --build $(BUILD) --parallel $(JOBS)
	@echo "# built $(CLI) — try: make run PROMPT=\"Hello\""

install: build
	$(CMAKE) --install $(BUILD) $(if $(PREFIX),--prefix $(PREFIX),)

# ─── model provisioning (offline, one-time per checkpoint) ───────────────────────
# Downloads MODEL from the HF Hub and packs it into $(OUT)/{model.mmfree,tokenizer.mmtok},
# where mmfree-cli finds it with no flags. Needs the tools/ deps once:
#   pip install -r tools/requirements.txt      (or: make model NO_TORCH=1, numpy only)
model:
	$(PYTHON) tools/provision.py --model $(MODEL) --out-dir $(OUT) \
	    $(if $(NO_TORCH),--no-torch,) $(ARGS)

# ─── run ─────────────────────────────────────────────────────────────────────────
run: build
	@[ -f "$(BLOB)" ] || { echo "ERROR: no weight blob at $(BLOB) — run 'make model' first (or pass BLOB=<path>)." >&2; exit 1; }
	$(CLI) --blob $(BLOB) --gen $(GEN) $(ARGS) "$(PROMPT)"

bench: build
	@[ -f "$(BLOB)" ] || { echo "ERROR: no weight blob at $(BLOB) — run 'make model' first (or pass BLOB=<path>)." >&2; exit 1; }
	$(CLI) --blob $(BLOB) --bench --gen $(GEN) --reps $(REPS) $(ARGS)

profile: build
	@[ -f "$(BLOB)" ] || { echo "ERROR: no weight blob at $(BLOB) — run 'make model' first (or pass BLOB=<path>)." >&2; exit 1; }
	$(CLI) --blob $(BLOB) --bench --profile --gen $(GEN) --reps $(REPS) $(ARGS)

# ─── checks ──────────────────────────────────────────────────────────────────────
# The golden-oracle ctest suite lives on the `validation` branch (cpp/tests/). Here on
# main `make test` runs it if present and otherwise falls back to the build smoke test,
# so the target is never a silent no-op.
test: build
	@if [ -f "$(SRC)/tests/test_kernels.cpp" ]; then \
	    cd $(BUILD) && ctest --output-on-failure; \
	else \
	    echo "# no cpp/tests/ on this branch (they live on \`validation\`) — running the smoke test"; \
	    $(MAKE) --no-print-directory smoke; \
	fi

# Proves the binary links and runs. Generation is included only if a blob is present,
# so this works on a machine that has not provisioned a checkpoint.
smoke: build
	$(CLI) --help >/dev/null
	@if [ -f "$(BLOB)" ]; then \
	    echo "# smoke: generating 4 tokens from $(BLOB)"; \
	    $(CLI) --blob $(BLOB) --ids 1,415,310 --gen 4; \
	else \
	    echo "# smoke: --help OK (no blob at $(BLOB); 'make model' to test generation)"; \
	fi
	@echo "# smoke OK"

clean:
	@[ -d "$(BUILD)" ] && $(CMAKE) --build $(BUILD) --target clean || true

distclean:
	rm -rf $(BUILD)

help:
	@echo 'MatMul-Free LM — C++ CPU inference engine'
	@echo ''
	@echo 'targets:'
	@echo '  build        configure + compile -> $(CLI)   [default]'
	@echo '  model        download + pack MODEL -> $(OUT)/{model.mmfree,tokenizer.mmtok}'
	@echo '  run          generate from PROMPT'
	@echo '  bench        throughput run (prefill + decode tok/s)'
	@echo '  profile      bench with a per-op wall-clock breakdown'
	@echo '  test         ctest suite if present, else the smoke test'
	@echo '  smoke        prove the binary links and runs'
	@echo '  install      cmake --install (PREFIX=/usr/local)'
	@echo '  clean        drop build artifacts;  distclean: drop $(BUILD) entirely'
	@echo ''
	@echo 'knobs (current value):'
	@echo '  MODEL=$(MODEL)'
	@echo '  PROMPT=$(PROMPT)'
	@echo '  GEN=$(GEN)   REPS=$(REPS)   JOBS=$(JOBS)'
	@echo '  BUILD=$(BUILD)   BUILD_TYPE=$(BUILD_TYPE)'
	@echo '  SIMD=$(if $(SIMD),$(SIMD),<auto-detect>)   # SCALAR|X86|X86_512|ARM'
	@echo '  TUNE=$(TUNE)                # ON = -mtune/-mcpu=native (host-tuned scheduling)'
	@echo '  OUT=$(OUT)                 # where make model writes the blobs'
	@echo '  NO_TORCH=1                  # pack without torch (numpy only; see README)'
	@echo '  ARGS="..."                  # extra flags passed through to mmfree-cli'
	@echo ''
	@echo 'examples:'
	@echo '  make -j && make model && make run PROMPT="Once upon a time,"'
	@echo '  make bench GEN=128 REPS=5'
	@echo '  make build SIMD=SCALAR BUILD=/tmp/b   # portable binary, out-of-tree'
	@echo '  make model MODEL=ridger/MMfreeLM-1.3B'
