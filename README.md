<div align=center>
<img src="__assets__/logo.png" width="200px">
</div>
<h2 align="center">MatMul-Free LM — C++ CPU inference</h2>
<h5 align="center"> This repo is adapted from <a href="https://github.com/sustcsonglin/flash-linear-attention">flash-linear-attention</a>, whose MatMul-Free LM it reimplements in C++. </h5>

<h5 align="center">

[![hf_model](https://img.shields.io/badge/🤗-Models-blue.svg)](https://huggingface.co/collections/ridger/matmulfree-lm-665f4d2b4e4648756e0dd13c) [![arXiv](https://img.shields.io/badge/Arxiv-2406.02528-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2406.02528)
</h5>

A **standalone C++17 inference engine** for MatMul-Free LM (HGRN-Bit): the engine (`cpp/`)
plus a small offline Python toolchain (`tools/`) that packs the weights and tokenizer.

The projections use ternary weights (`{-1, 0, +1}`), so the hot loop is add/subtract/skip —
no multiplies. Nothing here needs a GPU, and the engine itself needs no Python.

## Quickstart

```sh
make                                    # configure + build -> build/mmfree-cli
pip install -r tools/requirements.txt   # one-time, for provisioning only
make model                              # download + pack MMfreeLM-370M -> blobs/370M/ (~0.7 GB)
make run PROMPT="In a shocking finding, scientists discovered"
make chat                               # interactive session (see "Interactive session")
make model MODEL=2.7B && make chat MODEL=2.7B   # the other sizes: 1.3B, 2.7B
```

`make help` lists every target and knob. The Makefile only wraps plain commands:

```sh
cmake -S cpp -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
python tools/provision.py --model 370M          # -> blobs/370M/{model.mmfree,tokenizer.mmtok}
./build/mmfree-cli --model 370M "Once upon a time," --gen 32
```

## Requirements

| | needed for | notes |
|---|---|---|
| C++17 compiler + CMake ≥ 3.16 | building the engine | GCC, Clang, or AppleClang |
| OpenMP | multi-threading | optional; without it the build is correct but single-threaded. macOS: `brew install libomp` (keg-only — CMake finds it for you) |
| GNU make | the `Makefile` wrapper | optional; macOS ships it as `make`, *BSD as `gmake` |
| Python 3.8+ with numpy | provisioning a checkpoint | one-time, offline; not needed at run time |
| torch + safetensors | the reference weight packer | skippable — see [`--no-torch`](#provisioning-without-torch) |

**Platforms.** SIMD support is detected at configure time: AVX-512 or AVX2 on x86-64, NEON
on ARM, a portable scalar fallback elsewhere. Routinely built and run on Linux x86-64 and
Linux aarch64 (Kria K26 / Cortex-A53). macOS and \*BSD should work — the only
platform-specific code is POSIX `mmap` and the SIMD intrinsics — but are untested.
**Windows is not supported natively** (`mmap`); build under WSL.

## Build options

All are `cmake -S cpp -B build -D<OPT>=<VALUE>`:

| option | default | what it does |
|---|---|---|
| `MMFREE_SIMD` | `AUTO` | `AUTO` \| `X86_512` \| `X86` \| `ARM` \| `SCALAR` — see below |
| `MMFREE_OPT_FLAGS` | `-O3` | optimization flags. **Never add `-ffast-math`**: it reorders fp reductions and breaks the numerics contract |
| `MMFREE_TUNE_NATIVE` | `OFF` | `ON` adds `-mtune=native` / `-mcpu=native` — scheduling only, emits no new instructions, so the binary still runs on other CPUs of the same architecture |
| `CMAKE_BUILD_TYPE` | `Release` | |

Through make they are `SIMD=`, `TUNE=` and `BUILD_TYPE=` (no knob for `MMFREE_OPT_FLAGS`);
changing one re-configures the existing build tree.

`AUTO` picks the fastest backend **this machine** can run, so the binary is host-specific:
on x86-64 it compiles *and runs* an AVX-512 probe, otherwise it uses AVX2 (a 2013-or-newer
CPU). The backends differ only in speed — the integer path is bit-identical across them.
For a binary portable to an unknown machine: `make build SIMD=SCALAR`.
Cross-compiling skips the probe; pass `-DMMFREE_SIMD=X86_512` explicitly if the target has
AVX-512.

## Provisioning a model

The engine reads a packed weight blob (`model.mmfree`) and a tokenizer (`tokenizer.mmtok`).
`tools/provision.py` downloads a Hugging Face checkpoint and writes both to `blobs/<size>/`,
where the CLIs find them by size (`--model 1.3B`):

```sh
python tools/provision.py                                   # 370M -> blobs/370M/
python tools/provision.py --model 1.3B                      # or make model MODEL=1.3B
python tools/provision.py --out-dir /srv/models/mmfree-370m  # anywhere (then use --blob)
python tools/provision.py --skip-download                    # reuse the HF cache
python tools/provision.py --weights-only | --tokenizer-only  # one artifact
```

Checkpoints are cached under `~/.cache/huggingface`; for gated or private repos run
`huggingface-cli login` first (all three MMfreeLM models are public). Both blobs are
git-ignored: regenerate, don't commit.

| Model | Layers | Hidden | Trained tokens | blob size |
|:---|:---:|:---:|:---:|:---:|
| [370M](https://huggingface.co/ridger/MMfreeLM-370M) | 24 | 1024 | 15B | 0.47 GB |
| [1.3B](https://huggingface.co/ridger/MMfreeLM-1.3B) | 24 | 2048 | 100B | 1.56 GB |
| [2.7B](https://huggingface.co/ridger/MMfreeLM-2.7B) | 32 | 2560 | 100B | 2.95 GB |

The blob stores one `int8` per ternary weight and is `mmap`ed at load (no copy into RAM).

### Provisioning without torch

torch is by far the largest install and is only used by the reference packer. To pack with
numpy alone:

```sh
python tools/provision.py --no-torch      # or: make model NO_TORCH=1
```

The ternary weights come out **bit-identical**; per-tensor `scale_w` and `lower_bounds` can
differ by 1-2 ULP because numpy and torch order the summations differently. On 370M that
shifted logits by ≤0.034 and left a 16-token greedy stream unchanged — but it is *not*
bit-exact, so the torch path stays the default and the numerics of record.

## Running

```sh
mmfree-cli "The capital of France is" --gen 32   # prompt -> streamed text
echo "Once upon a time," | mmfree-cli            # prompt on stdin
mmfree-cli --ids 1,415,310 --gen 8               # raw ids in/out (no tokenizer, no EOS)
mmfree-cli --bench --gen 128 --reps 5            # prefill + decode tok/s
mmfree-cli --bench --batch 8 --gen 128           # 8-stream batched throughput
mmfree-cli --bench --profile --gen 128           # per-op breakdown (matmul vs the rest)
mmfree-cli --help                                # every flag
```

Sampling is greedy by default (deterministic); `--temp`/`--top-k`/`--top-p`/`--seed` switch it to
sampling. `--mode float|fixed` selects the activation numerics — the default `fixed` is the
Q5.10 integer datapath that mirrors the FPGA accelerator.

Pick the model by size with `--model 370M|1.3B|2.7B` (`-m`); without it, the smallest one
provisioned is used. Models live in `blobs/<size>/`, and `blobs/` is found in the working
directory or beside the executable and its parents, so `./build/mmfree-cli` works from the
repo root with no path. The tokenizer is read from beside the weight blob. To use a blob kept
elsewhere, `--blob PATH` (or `$MMFREE_BLOB`) replaces `--model`, and `--tokenizer PATH` (or
`$MMFREE_TOKENIZER`) overrides the tokenizer.

### Interactive session: `mmfree-chat`

```sh
make chat MODEL=1.3B                          # or: ./build/mmfree-chat --model 1.3B
mmfree-chat --temp 0.7 --top-p 0.6 --gen 200  # sampling and reply-length settings
mmfree-chat --model 2.7B --continue           # one running text across turns
```

Type text and press Enter; the model continues it, streamed token by token. The released
checkpoints are base language models with no chat template, so this is completion, not
dialogue: by default each line starts a fresh text, and `--continue` (or `/continue on`)
keeps one running text that every line extends. Because HGRN is recurrent, a running text
costs nothing to keep — its state carries over and only the new tokens are fed, with no
context window to fill.

| input | does |
|---|---|
| text | continue it (a fresh text, or the running one under `/continue on`) |
| empty line | keep generating from where the last reply stopped |
| line ending in `\` | continue the input on the next line |
| `/temp T`, `/topp P`, `/topk K`, `/penalty R`, `/gen N`, `/seed N` | change sampling (`/temp 0` = greedy) or reply length |
| `/continue on\|off`, `/reset` | switch modes, or forget the running text |
| `/settings`, `/help`, `/quit` | |
| Ctrl-C / Ctrl-D | stop the current reply / exit |

It takes the same `--model`, `--blob`, `--tokenizer`, `--mode`, `--frac` and `--weights` flags
as `mmfree-cli` and finds the model files the same way. `--weights` defaults to `2bit` here,
the fastest format, and the weights are repacked during a warm-up step at startup rather than
on the first reply. Defaults are `--temp 0.6 --top-p 0.4 --repeat-penalty 1.2 --gen 128`.

These checkpoints are small base models that loop or collapse into filler words under plain
sampling — upstream's own examples do this at `top_p=0.4, temperature=0.6`. A sweep on 2.7B
(four prompts × four seeds) found temperature/top-k sampling drifting into word salad and
top-p alone looping, while adding a 1.2 repetition penalty to upstream's settings kept text
coherent; hence the defaults. The same knobs exist on `mmfree-cli` (`--top-p`,
`--repeat-penalty`, `--repeat-last-n`), off by default there so its output stays greedy and
reproducible. Each reply ends with a
short line of stats (tokens, decode tok/s, prefill time) on stderr; replies alone go to stdout,
so piping lines in works for scripting:

```sh
printf 'The capital of France is\n' | mmfree-chat --temp 0 --gen 16 2>/dev/null
```

### Packed ternary weights

One `int8` per ternary weight is 4-5x more weight traffic per token than the values need.
`--weights` picks a denser in-memory format:

```sh
mmfree-cli --bench --gen 128 --weights 1.6bit    # TQ1_0, 5 trits/byte -> 1.625 bits/weight
mmfree-cli --bench --gen 128 --weights 2bit      # 2.0 bits/weight
mmfree-cli --bench --gen 128                     # int8, 8.0 bits/weight (default)
```

Both packed formats are **lossless**: every ternary value survives the round trip and
integer addition is associative, so the int32 accumulators — and the logits — are
**bit-identical to `int8`**. `cpp/tests/test_tq1.cpp` checks the encoding exhaustively over
its whole input domain.

The same `model.mmfree` serves all three formats: each projection is repacked from the
blob's `int8` on first touch and cached, so the one-off cost lands on the first token
(`--warmup` absorbs it under `--bench`). Resident heap is 1.625/8 or 2/8 of the `int8` size
(370M ≈ 69/85 MB, 2.7B ≈ 532/656 MB), while the `int8` mmap pages go cold and become
evictable. Projections whose inner dimension isn't a multiple of 256 fall back to the
`int8` kernel (every real MMfreeLM shape is a multiple). Integer path only: `--mode float`
ignores `--weights` and says so.

Which format is fastest depends on the machine: packing trades memory traffic for
extraction work, so it wins where the matmul is bandwidth-bound and loses where it is
ALU-bound (on a Cortex-A53, 1.6-bit is *slower* than `int8`). Benchmark on the target:
`make bench WEIGHTS=2bit` against `make bench`.

### Batching (`--batch B`)

`--batch B` decodes **B concurrent streams in lockstep**. Each stream has its own recurrent
state and is prefilled independently; the B decode steps then share **one** matmul per
projection, walked weight-row-outer / batch-inner, so each weight row is fetched once and
reused across all B activations — weight traffic per token drops roughly B-fold.

```sh
mmfree-cli --bench --batch 8 --gen 128            # aggregate + per-stream tok/s
mmfree-cli --bench --batch 8 --weights 2bit       # batching composes with the packed formats
mmfree-cli --batch 4 "Once upon a time,"          # prints each stream, labelled
make bench BATCH=8 WEIGHTS=2bit                   # same thing through the Makefile
```

Batching is **opt-in and lossless**: each `acc[i][o]` is still the identical
`ternary_dot_i32` call on identical operands — only the loop order changes — so batched
output is *bit-identical* to B separate runs. `cpp/tests/test_tq1.cpp` (check `[6]`) asserts
this for b=1..8 across all three weight formats, using **distinct** activation rows so a
row/column permutation cannot pass unnoticed.

When reading the numbers:

- `--bench` reports **aggregate** decode tok/s (all streams) with per-stream alongside.
- `--batch B` batches **prefill** too, so prefill timings shift with B — compare like with
  like.
- `--mode float` ignores batching for the matmul (the backend seam is integer-only) and says
  so; the streams still run, just without the weight reuse.

#### What it buys, measured

370M on a 20-core Xeon, `--gen 128 --reps 5`, aggregate decode tok/s (two passes):

| weights | b=1 | b=2 | b=4 | b=8 | b=8 vs b=1 |
|:--|--:|--:|--:|--:|--:|
| `int8`   | 112.6 / 103.5 | 163.5 / 149.7 | 177.5 / 163.5 | 204.0 / 199.1 | 1.81x / 1.92x |
| `2bit`   | 108.1 / 100.6 | 163.9 / 193.0 | 173.4 / 160.7 | 224.3 / 206.3 | 2.08x / 2.05x |
| `1.6bit` | 107.7 /  99.3 | 152.9 / 137.7 | 158.6 / 146.7 | 199.1 / 186.0 | 1.85x / 1.87x |

Roughly 2x aggregate at b=8. Absolute cells carry ~8% run-to-run spread (the `2bit` b=2
cell is bimodal — do not quote it); the *ratio* is stable across passes and formats.

The knee is at b≈10, not at the thread count (b=1..40 ladder, `--gen 32 --reps 1`):

| b | 1 | 2 | 5 | 10 | 20 | 30 | 40 |
|:--|--:|--:|--:|--:|--:|--:|--:|
| aggregate  | 100.8 | 147.9 | 199.3 | 219.4 | 212.9 | 224.7 | 226.3 |
| per stream | 100.8 |  73.9 |  39.9 |  21.9 |  10.6 |   7.5 |   5.7 |

OpenMP already saturates the cores at b=1 (it parallelizes over the output rows); batching
only amortizes weight traffic, and past ~8-10 reuses per weight row the kernel is
compute-bound. Beyond the knee, per-stream latency keeps degrading for no aggregate gain, so
b≈8-12 is the operating point on this host. A 4-core Cortex-A53 knees much earlier and
needs its own ladder.

Threading follows OpenMP (`OMP_NUM_THREADS=4 mmfree-cli ...`). Only independent rows and
disjoint elementwise ranges are parallelized, never a floating-point reduction, so the
thread count does not change the output (`cpp/include/mmfree/parallel.hpp`).

## Numerics contract

The spec the engine reproduces, from the original `fusedbitnet.py` and the fused Triton
kernel. Changing any of it changes the output:

- **Weight quant** (offline, per tensor): `scale_w = 1 / mean(|W|)`;
  `Wq = clamp(round(W * scale_w), -1, 1)` → ternary `{-1, 0, +1}`; effective weight is
  `Wq / scale_w`. Baked into the blob at pack time — the engine never quantizes weights.
- **Activation quant** (runtime): `fixed` mode uses a static Q5.10 binary point
  (`xq = clamp(round(x * 2^10), -32768, 32767)`, `--frac` to change it); `float` mode keeps
  activations in fp32.
- **Projection**: `y = (Wq @ xq_row) / (scale_w * 2^frac)`, accumulated in int32 — the
  matmul-free part: per output, a sum of `+x` / `-x` / skip over ternary lanes.
- **RMSNorm**: `x * rsqrt(mean(x²) + eps) * weight`, `eps = 1e-6`, reductions in fp32.
- **SiLU / SwiGLU**: `silu(x) = x * sigmoid(x)`; `swiglu(a, b) = silu(a) * b`.

**`-ffast-math` is banned**: it reorders the fp reductions, changing RMSNorm and therefore
the quantized activations feeding the integer accumulators. The integer projection is
exactly reproducible, which is what lets a hardware backend be checked against the CPU
bit-for-bit.

## Validation

`make test` runs whatever this tree registers with `ctest`, then a smoke test:

```sh
make test                   # here: cpp/tests/test_tq1 (packed-weight kernels) + smoke
./build/test_tq1 --bench    # the same checks plus a kernel throughput report
```

`test_tq1` needs no model blob and no hardware, and it exercises whichever SIMD path the
library was built for — so it is also the check that a new platform's NEON/AVX-512
extraction is right.

The **model-level** suite — `test_kernels` and `test_e2e`, plus the PyTorch-comparison
tooling that generates their golden oracles (`dump_golden.py`, `compare_prompts.py`,
`check_tokenizer.py`, …) — lives on the **`validation`** branch, because the oracles and
their torch dependency are far larger than the engine:

```sh
git checkout validation   # adds the golden-oracle tests + tools/ comparison scripts
make test                 # same command, more tests
```

In both numeric modes the greedy token stream matches the PyTorch reference exactly and the
logits agree within a relative-L2 tolerance.

## Troubleshooting

| symptom | cause / fix |
|---|---|
| `error: no model provisioned` / `the 1.3B model is not provisioned` | `make model MODEL=<size>`, or pass `--blob` |
| `Illegal instruction` / `SIGILL` | binary built on a newer CPU than it runs on. Rebuild with `make build SIMD=SCALAR` (or `SIMD=X86`) |
| `OpenMP not found` warning, slow | AppleClang ships no OpenMP runtime: `brew install libomp`, then re-run cmake — CMake locates the keg-only install itself. If it still misses (non-standard Homebrew prefix), pass `-DOpenMP_ROOT=$(brew --prefix libomp)`. Debian/Ubuntu with clang: `apt install libomp-dev` |
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
