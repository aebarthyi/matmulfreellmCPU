# Makefile — optional wrapper over plain cmake/python commands (each recipe echoes its own).
#
#   make                 # configure + build      -> build/mmfree-cli
#   make model           # download + pack a checkpoint (needs the tools/ deps once)
#   make run PROMPT="Hello there"
#   make chat            # interactive session (mmfree-chat)
#   make bench
#   make help            # every target and knob
#
# POSIX hosts (Windows: WSL). Needs GNU make for $(if ...)/$(shell ...) (`gmake` on *BSD).
.DEFAULT_GOAL := build

# ─── knobs (override on the command line: `make build SIMD=SCALAR`) ───────────────
SRC        := cpp
# Out-of-source build at the repo root; BUILD=cpp/build also works (the CLI's blob search
# handles both).
BUILD      ?= build
BUILD_TYPE ?= Release
# getconf is POSIX and works on both Linux and macOS (nproc is Linux-only).
JOBS       ?= $(shell getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)
# Empty = let CMake auto-detect (AVX-512/AVX2/NEON/scalar). SCALAR|X86|X86_512|ARM to pin.
SIMD       ?=
TUNE       ?= OFF
CMAKE      ?= cmake
PYTHON     ?= python3

# Model size: blobs/$(MODEL)/{model.mmfree,tokenizer.mmtok}, written by `make model`.
SIZES      := 370M 1.3B 2.7B
MODEL      ?= 370M
BLOB       := blobs/$(MODEL)/model.mmfree

CLI        := $(BUILD)/mmfree-cli
CHAT       := $(BUILD)/mmfree-chat
PROMPT     ?= In a shocking finding, scientists discovered
GEN        ?= 32
REPS       ?= 3
# In-memory ternary weight format: int8 (default) | 1.6bit | 2bit. Lossless (bit-identical).
WEIGHTS    ?=
# Streams decoded in lockstep (lossless); >1 also batches prefill, so its timings shift.
BATCH      ?=
ARGS       ?=

CMAKE_ARGS := -DCMAKE_BUILD_TYPE=$(BUILD_TYPE) -DMMFREE_TUNE_NATIVE=$(TUNE) \
              -DMMFREE_SIMD=$(if $(SIMD),$(SIMD),AUTO)
CLI_ARGS   := $(if $(WEIGHTS),--weights $(WEIGHTS),) $(if $(BATCH),--batch $(BATCH),) $(ARGS)

.PHONY: all build configure model run chat bench profile test smoke install clean distclean help FORCE

all: build

# ─── build ───────────────────────────────────────────────────────────────────────
# Re-configures when the cache is missing or the knobs differ from the last configure (the
# stamp is rewritten only when CMAKE_ARGS changes); cmake tracks CMakeLists edits itself.
ARGS_STAMP := $(BUILD)/.cmake-args
$(ARGS_STAMP): FORCE
	@mkdir -p $(@D); [ "$$(cat $@ 2>/dev/null)" = '$(CMAKE_ARGS)' ] || echo '$(CMAKE_ARGS)' > $@
configure: $(BUILD)/CMakeCache.txt
$(BUILD)/CMakeCache.txt: $(SRC)/CMakeLists.txt $(ARGS_STAMP)
	$(CMAKE) -S $(SRC) -B $(BUILD) $(CMAKE_ARGS)
	@touch $@  # cmake leaves an unchanged cache's mtime alone; without this, make re-configures every run

build: configure
	$(CMAKE) --build $(BUILD) --parallel $(JOBS)
	@echo "# built $(CLI) — try: make run PROMPT=\"Hello\""

install: build
	$(CMAKE) --install $(BUILD) $(if $(PREFIX),--prefix $(PREFIX),)

# ─── model provisioning (offline, one-time per checkpoint) ───────────────────────
# Downloads MODEL and packs blobs/$(MODEL)/, where the CLIs find it by --model.
# Needs `pip install -r tools/requirements.txt` once (NO_TORCH=1: no torch).
define check_model
	@case " $(SIZES) " in *" $(MODEL) "*) ;; *) echo "ERROR: MODEL must be one of $(SIZES) (got '$(MODEL)')." >&2; exit 1 ;; esac
endef
define need_blob
	$(check_model)
	@[ -f "$(BLOB)" ] || { echo "ERROR: the $(MODEL) model is not provisioned ($(BLOB) missing) — run 'make model MODEL=$(MODEL)'." >&2; exit 1; }
endef

model:
	$(check_model)
	$(PYTHON) tools/provision.py --model $(MODEL) --out-dir blobs/$(MODEL) \
	    $(if $(NO_TORCH),--no-torch,) $(ARGS)

# ─── run ─────────────────────────────────────────────────────────────────────────
run: build
	$(need_blob)
	$(CLI) --model $(MODEL) --gen $(GEN) $(CLI_ARGS) "$(PROMPT)"

chat: build
	$(need_blob)
	$(CHAT) --model $(MODEL) $(if $(WEIGHTS),--weights $(WEIGHTS),) $(ARGS)

bench: build
	$(need_blob)
	$(CLI) --model $(MODEL) --bench --gen $(GEN) --reps $(REPS) $(CLI_ARGS)

profile: build
	$(need_blob)
	$(CLI) --model $(MODEL) --bench --profile --gen $(GEN) --reps $(REPS) $(CLI_ARGS)

# ─── checks ──────────────────────────────────────────────────────────────────────
# ctest (test_tq1; plus the golden-oracle suite on the `validation` branch), then smoke.
test: build
	@if [ -d "$(SRC)/tests" ]; then \
	    cd $(BUILD) && ctest --output-on-failure; \
	else \
	    echo "# no cpp/tests/ in this tree — smoke test only"; \
	fi
	@$(MAKE) --no-print-directory smoke

# Proves the binary links and runs; also generates if a blob is present.
smoke: build
	$(CLI) --help >/dev/null
	@if [ -f "$(BLOB)" ]; then \
	    echo "# smoke: generating 4 tokens from $(BLOB)"; \
	    $(CLI) --model $(MODEL) --ids 1,415,310 --gen 4; \
	else \
	    echo "# smoke: --help OK (no blob at $(BLOB); 'make model MODEL=$(MODEL)' to test generation)"; \
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
	@echo '  model        download + pack MODEL -> blobs/$(MODEL)/{model.mmfree,tokenizer.mmtok}'
	@echo '  run          generate from PROMPT'
	@echo '  chat         interactive session: type text, the model continues it'
	@echo '  bench        throughput run (prefill + decode tok/s)'
	@echo '  profile      bench with a per-op wall-clock breakdown'
	@echo '  test         ctest suite if present, else the smoke test'
	@echo '  smoke        prove the binary links and runs'
	@echo '  install      cmake --install (PREFIX=/usr/local)'
	@echo '  clean        drop build artifacts;  distclean: drop $(BUILD) entirely'
	@echo ''
	@echo 'knobs (current value):'
	@echo '  MODEL=$(MODEL)                   # $(SIZES); provisioned: $(or $(foreach s,$(SIZES),$(if $(wildcard blobs/$(s)/model.mmfree),$(s))),none)'
	@echo '  PROMPT=$(PROMPT)'
	@echo '  GEN=$(GEN)   REPS=$(REPS)   JOBS=$(JOBS)'
	@echo '  WEIGHTS=$(if $(WEIGHTS),$(WEIGHTS),int8)          # int8|1.6bit|2bit (packed = same results, less traffic)'
	@echo '  BATCH=$(if $(BATCH),$(BATCH),1)                   # concurrent streams decoded in lockstep (bit-identical)'
	@echo '  BUILD=$(BUILD)   BUILD_TYPE=$(BUILD_TYPE)'
	@echo '  SIMD=$(if $(SIMD),$(SIMD),<auto-detect>)   # SCALAR|X86|X86_512|ARM'
	@echo '  TUNE=$(TUNE)                # ON = -mtune/-mcpu=native (host-tuned scheduling)'
	@echo '  NO_TORCH=1                  # pack without torch (numpy only; see README)'
	@echo '  ARGS="..."                  # extra flags passed through to mmfree-cli / mmfree-chat'
	@echo ''
	@echo 'examples:'
	@echo '  make -j && make model && make run PROMPT="Once upon a time,"'
	@echo '  make bench GEN=128 REPS=5'
	@echo '  make bench WEIGHTS=2bit           # compare against WEIGHTS=int8'
	@echo '  make bench BATCH=8                # batched throughput (aggregate + per-stream)'
	@echo '  make chat ARGS="--temp 0.7 --continue"'
	@echo '  make build SIMD=SCALAR BUILD=/tmp/b   # portable binary, out-of-tree'
	@echo '  make model MODEL=1.3B && make chat MODEL=1.3B'
