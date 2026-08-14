<div align=center>
<img src="__assets__/logo.png" width="200px">
</div>
<h2 align="center">MatMul-Free LM — C++ CPU inference</h2>
<h5 align="center"> This repo is adapted from <a href="https://github.com/sustcsonglin/flash-linear-attention">flash-linear-attention</a>, whose MatMul-Free LM it reimplements in C++. </h5>

<h5 align="center">

[![hf_model](https://img.shields.io/badge/🤗-Models-blue.svg)](https://huggingface.co/collections/ridger/matmulfree-lm-665f4d2b4e4648756e0dd13c) [![arXiv](https://img.shields.io/badge/Arxiv-2406.02528-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2406.02528)
</h5>

A **standalone C++17 inference engine** for MatMul-Free LM (HGRN-Bit). The original
PyTorch/Triton model has been removed now that the C++ implementation is verified to match
it bit-for-bit; what remains is the engine (`cpp/`) plus a small offline Python toolchain
(`tools/`) that packs weights and the tokenizer.

The projections use ternary weights (`{-1, 0, +1}`), so the hot loop is add/subtract/skip —
no multiplies. Nothing here needs a GPU, and the engine itself needs no Python.

## Quickstart

```sh
make                                    # configure + build -> build/mmfree-cli
pip install -r tools/requirements.txt   # one-time, for provisioning only
make model                              # download + pack MMfreeLM-370M (~0.7 GB download)
make run PROMPT="In a shocking finding, scientists discovered"
```

`make help` lists every target and knob. The Makefile is a convenience wrapper — the plain
commands work just as well:

```sh
cmake -S cpp -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
python tools/provision.py                       # -> cpp/{model.mmfree,tokenizer.mmtok}
./build/mmfree-cli "Once upon a time," --gen 32
```

## Requirements

| | needed for | notes |
|---|---|---|
| C++17 compiler + CMake ≥ 3.16 | building the engine | GCC, Clang, or AppleClang |
| OpenMP | multi-threading | optional; without it the build is correct but single-threaded |
| GNU make | the `Makefile` wrapper | optional; macOS ships it as `make`, *BSD as `gmake` |
| Python 3.8+ with numpy | provisioning a checkpoint | one-time, offline; not needed at run time |
| torch + safetensors | the reference weight packer | skippable — see [`--no-torch`](#provisioning-without-torch) |

**Platforms.** CPU support is detected at configure time — AVX-512 or AVX2 on x86-64, NEON on
ARM, and a portable scalar fallback everywhere else. Regularly built and run on **Linux
x86-64** and **Linux aarch64** (a Kria K26 / Cortex-A53). macOS and \*BSD are expected to work
— the only platform-specific code is POSIX `mmap` in the weight loader and the SIMD
intrinsics, both covered above — but are not part of the routine test matrix, so treat them
as untested rather than guaranteed. **Windows is not supported natively** (that `mmap`); build
under WSL.

## Build options

All are `cmake -S cpp -B build -D<OPT>=<VALUE>`, or `make build <OPT>=<VALUE>`:

| option | default | what it does |
|---|---|---|
| `MMFREE_SIMD` | `AUTO` | `AUTO` \| `X86_512` \| `X86` \| `ARM` \| `SCALAR` — see below |
| `MMFREE_OPT_FLAGS` | `-O3` | optimization flags. **Never add `-ffast-math`**: it reorders fp reductions and breaks the numerics contract |
| `MMFREE_TUNE_NATIVE` | `OFF` | `ON` adds `-mtune=native` / `-mcpu=native` — scheduling only, emits no new instructions, so the binary still runs on other CPUs of the same architecture |
| `CMAKE_BUILD_TYPE` | `Release` | |

`AUTO` picks the **fastest backend this machine can run**, which makes the binary
host-specific: on x86-64 it probes for AVX-512 (compiling *and* running a test, since the
toolchain knowing the intrinsics doesn't mean the CPU executes them) and otherwise uses
AVX2, which itself needs a 2013-or-newer CPU. The 512-bit ternary-dot paths were worth
~29% throughput over AVX2 when this was benchmarked on a Xeon, and they produce
**bit-identical logits** — the two backends differ only in speed, which is why AUTO takes
the faster one rather than making you ask. To build something portable to an unknown
machine, ask for it: `make build SIMD=SCALAR`.

Cross-compiling skips the AVX-512 probe (host code can't run on the target) — pass
`-DMMFREE_SIMD=X86_512` explicitly if the target has it.

## Provisioning a model

The engine reads two files: a packed weight blob (`model.mmfree`) and a tokenizer
(`tokenizer.mmtok`). `tools/provision.py` produces both from a Hugging Face checkpoint in
one command — download, pack weights, pack tokenizer:

```sh
python tools/provision.py                                   # 370M -> cpp/
python tools/provision.py --model ridger/MMfreeLM-1.3B
python tools/provision.py --out-dir /srv/models/mmfree-370m  # anywhere you like
python tools/provision.py --skip-download                    # reuse the HF cache
python tools/provision.py --weights-only | --tokenizer-only  # one artifact
```

Checkpoints come from the HF Hub via `huggingface_hub` (cached under `~/.cache/huggingface`;
for gated or private repos run `huggingface-cli login` first — all three MMfreeLM models are
public). Both blobs are git-ignored: regenerate, don't commit.

| Model | Layers | Hidden | Trained tokens | blob size |
|:---|:---:|:---:|:---:|:---:|
| [370M](https://huggingface.co/ridger/MMfreeLM-370M) | 24 | 1024 | 15B | 0.47 GB |
| [1.3B](https://huggingface.co/ridger/MMfreeLM-1.3B) | 24 | 2048 | 100B | 1.56 GB |
| [2.7B](https://huggingface.co/ridger/MMfreeLM-2.7B) | 32 | 2560 | 100B | 2.95 GB |

The blob stores ternary weights as one `int8` per weight, so it is roughly the parameter
count in bytes and is `mmap`ed at load (no copy into RAM).

### Provisioning without torch

torch is by far the largest install here, and it is only used by the reference packer. To
pack with numpy alone:

```sh
python tools/provision.py --no-torch      # or: make model NO_TORCH=1
```

Measured on all 145 ternary tensors of MMfreeLM-370M, `--no-torch` produces **bit-identical
ternary weights**; the only differences are in the per-tensor `scale_w` values (108 of 145
differ, max relative difference 7.4e-7, i.e. 1-2 ULP) and in `lower_bounds` (max absolute
difference 2.4e-7), because numpy and torch order the `mean()`/`softmax()` summations
differently. End to end that shifted logits by ≤0.034 on a ±16 range and left the greedy
token stream unchanged over 16 tokens — but it is *not* bit-exact, so the torch path stays
the default and remains the numerics of record.

## Running

```sh
mmfree-cli "The capital of France is" --gen 32   # prompt -> streamed text
echo "Once upon a time," | mmfree-cli            # prompt on stdin
mmfree-cli --ids 1,415,310 --gen 8               # raw ids in/out (no tokenizer, no EOS)
mmfree-cli --bench --gen 128 --reps 5            # prefill + decode tok/s
mmfree-cli --bench --profile --gen 128           # per-op breakdown (matmul vs the rest)
mmfree-cli --help                                # every flag
```

Sampling is greedy by default (deterministic); `--temp`/`--top-k`/`--seed` switch it to
sampling. `--mode float|fixed` selects the activation numerics — the default `fixed` is the
Q5.10 integer datapath that mirrors the FPGA accelerator.

### Packed ternary weights

The blob stores one `int8` per ternary weight, which is 4-5x more weight traffic per token
than the values need. `--weights` picks a denser in-memory format instead:

```sh
mmfree-cli --bench --gen 128 --weights 1.6bit    # TQ1_0, 5 trits/byte -> 1.625 bits/weight
mmfree-cli --bench --gen 128 --weights 2bit      # 2.0 bits/weight
mmfree-cli --bench --gen 128                     # int8, 8.0 bits/weight (default)
```

Both packed formats are **lossless**, so this is a pure speed/space knob: the round trip
preserves every ternary value and integer addition is associative, so the int32 accumulators
— and therefore the logits — are **bit-identical to `int8`**. Verified two ways:
`cpp/tests/test_tq1.cpp` proves the encoding exhaustively over its whole input domain (all
243 five-trit and 81 four-trit groups, not a sample), and all three formats produce
byte-identical logits from the 370M model.

Mechanics: nothing changes about provisioning — the same `model.mmfree` serves all three.
Each projection is repacked from the blob's `int8` on first touch and cached, so the one-off
cost lands on the first token (`--warmup` absorbs it under `--bench`). Resident heap is
1.625/8 or 2/8 of the `int8` size (370M ≈ 69/85 MB, 2.7B ≈ 532/656 MB), against which the
`int8` mmap pages go cold and become evictable — for the large models this can *lower*
steady-state RSS. Projections whose inner dimension isn't a multiple of 256 fall back to the
`int8` kernel automatically (every real MMfreeLM shape is a multiple). Integer path only:
`--mode float` ignores `--weights` and says so.

Which is faster depends on the machine — packing trades memory traffic for extraction work,
so it wins where the matmul is bandwidth-bound and loses where it is ALU-bound (on a
Cortex-A53 the 1.6-bit path measured *slower* than `int8`). Benchmark both on the target
rather than assuming: `make bench WEIGHTS=2bit` against `make bench`.

With no `--blob`, the engine looks for `model.mmfree` in the working directory, then beside
the executable, its parent directories and a `cpp/` sibling, so `./build/mmfree-cli` from
the repo root finds the `make model` blob in `cpp/` without a flag. `$MMFREE_BLOB` / `$MMFREE_TOKENIZER`
override the search; `--blob` / `--tokenizer` override everything. The tokenizer is looked up
next to the blob.

Threading follows OpenMP: `OMP_NUM_THREADS=4 mmfree-cli ...`. Only independent rows and
disjoint elementwise ranges are parallelized, never a floating-point reduction, so thread
count does not change the output (`include/mmfree/parallel.hpp`).

## Numerics contract

The spec the engine reproduces, from the original `fusedbitnet.py` and the fused Triton
kernel. Changing any of it changes the output:

- **Weight quant** (offline, per tensor): `scale_w = 1 / mean(|W|)`;
  `Wq = clamp(round(W * scale_w), -1, 1)` → ternary `{-1, 0, +1}`; effective weight is
  `Wq / scale_w`. Baked into the blob at pack time — the engine never quantizes weights.
- **Activation quant** (runtime, per row): `fixed` mode uses a static Q5.10 binary point
  (`xq = clamp(round(x * 2^10), -32768, 32767)`, `--frac` to change it); `float` mode keeps
  activations in fp32.
- **Projection**: `y = (Wq @ xq_row) / (scale_w * 2^frac)`, accumulated in int32 — the
  matmul-free part: per output, a sum of `+x` / `-x` / skip over ternary lanes.
- **RMSNorm**: `x * rsqrt(mean(x²) + eps) * weight`, `eps = 1e-6`, reductions in fp32.
- **SiLU / SwiGLU**: `silu(x) = x * sigmoid(x)`; `swiglu(a, b) = silu(a) * b`.

Two consequences worth keeping in mind: **`-ffast-math` is banned** (it reorders the fp
reductions, changing RMSNorm and therefore the quantized activations feeding the integer
accumulators), and the integer projection is exactly reproducible, which is what lets a
hardware backend be checked against the CPU bit-for-bit.

## Validation

`make test` runs whatever this tree registers with `ctest`, then a smoke test:

```sh
make test          # here: cpp/tests/test_tq1 (packed-weight kernels) + smoke
./build/test_tq1 --bench    # the same checks plus a kernel throughput report
```

`test_tq1` needs no model blob and no hardware, and it exercises whichever SIMD path the
library was built for — so it is also the check that a new platform's NEON/AVX-512
extraction is right.

The **model-level** correctness suite — `test_kernels` and `test_e2e` plus the
PyTorch-comparison tooling that generates their golden oracles (`dump_golden.py`,
`compare_prompts.py`, `check_tokenizer.py`, …) — lives on the **`validation`** branch,
because the oracles and their torch dependency are far larger than the engine:

```sh
git checkout validation   # adds the golden-oracle tests + tools/ comparison scripts
make test                 # same command, more tests
```

It was verified end-to-end: the C++ output matches the PyTorch reference exactly in both the
float and fixed-point numeric modes.

## Troubleshooting

| symptom | cause / fix |
|---|---|
| `error: no weight blob found` | not provisioned yet — `make model`, or pass `--blob` |
| `Illegal instruction` / `SIGILL` | binary built on a newer CPU than it runs on. Rebuild with `make build SIMD=SCALAR` (or `SIMD=X86`) |
| `OpenMP not found` warning, slow | AppleClang has no OpenMP runtime: `brew install libomp`, then re-run cmake. Debian/Ubuntu with clang: `apt install libomp-dev` |
| slower than expected on x86 | check the configure line says `SIMD backend: X86_512`; if it says `X86`, this CPU has no AVX-512 |
| `unsupported safetensors dtype` from `--no-torch` | the checkpoint isn't BF16/F16/F32 — drop `--no-torch` |
| CMake warning about Win32 | native Windows isn't supported; build under WSL |

## About the architecture

<div align=center>
<img src="__assets__/main.png">
</div>

MatMul-Free LM replaces every matrix multiplication in a language model: the projections use
ternary weights (add/subtract/skip instead of multiply) and the token mixer is a gated linear
recurrence (HGRN) rather than attention, so inference is a sequential scan with O(1) state per
step instead of a growing KV cache. The result is compatible with the 🤗 Transformers
checkpoints linked above.

<div align=center>
<img src="__assets__/scaling_law.png" width="70%">
</div>

The scaling projection for MatMul-Free LM descends more steeply than Transformer++ across the
370M / 1.3B / 2.7B fits, suggesting the architecture converts additional compute into quality
more efficiently — with each operation treated identically for comparison, though this model
uses more efficient ternary weights in some layers.

## Citation

If you use this repo in your work, please cite the preprint:

```bib
@article{zhu2024scalable,
title={Scalable MatMul-free Language Modeling},
author={Zhu, Rui-Jie and Zhang, Yu and Sifferman, Ethan and Sheaves, Tyler and Wang, Yiqiao and Richmond, Dustin and Zhou, Peng and Eshraghian, Jason K},
journal={arXiv preprint arXiv:2406.02528},
year={2024}
}
```
