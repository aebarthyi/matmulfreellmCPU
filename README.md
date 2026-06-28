<div align=center>
<img src="__assets__/logo.png" width="200px">
</div>
<h2 align="center">MatMul-Free LM</h2>
<h5 align="center"> If you like our project, please give us a star ⭐ on GitHub for the latest updates.  </h2>
<h5 align="center"> This repo is adapted from <a href="https://github.com/sustcsonglin/flash-linear-attention">flash-linear-attention</a>. </h2>

<h5 align="center">

[![hf_model](https://img.shields.io/badge/🤗-Models-blue.svg)](https://huggingface.co/collections/ridger/matmulfree-lm-665f4d2b4e4648756e0dd13c) [![arXiv](https://img.shields.io/badge/Arxiv-2406.02528-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2406.02528) 
# Introduction
<div align=center>
<img src="__assets__/main.png">
</div>
MatMul-Free LM is a language model architecture that eliminates the need for Matrix Multiplication (MatMul) operations. This repository provides an implementation of MatMul-Free LM that is compatible with the 🤗 Transformers library.

# Scaling Law
<div align=center>
<img src="__assets__/scaling_law.png">
</div>
We evaluate how the scaling law fits to the 370M, 1.3B and 2.7B parameter models in both Transformer++ and our model. For a fair comparison, each operation is treated identically, though our model uses more efficient ternary weights in some layers. Interestingly, the scaling projection for our model exhibits a steeper descent compared to Transformer++, suggesting our architecture is more efficient in leveraging additional compute to improve performance.

# C++ implementation

This fork is a **standalone C++ inference engine** for MatMul-Free LM (HGRN-Bit). The
original PyTorch/Triton model has been removed now that the C++ implementation is verified
to match it bit-for-bit; what remains is the C++ engine (`cpp/`) plus a small offline Python
toolchain (`tools/`) that packs weights/tokenizer and generates the test oracles.

## Build

Pure CMake/C++17 — no Python needed to build or run the engine:

```sh
cd cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release   # SIMD backend auto-detects (NEON/AVX2/scalar)
cmake --build build -j
# → build/mmfree-cli, build/test_kernels, build/test_e2e
```

## Provision the model (offline, one-time)

The engine reads a packed weight blob (`cpp/model.mmfree`) and tokenizer
(`cpp/tokenizer.mmtok`). Both are produced from a [Hugging Face checkpoint](#pre-trained-model-zoo)
by the offline `tools/` (torch + safetensors). **`tools/provision.py` does all three steps —
download, pack weights, pack tokenizer — in one command:**

```sh
pip install -r tools/requirements.txt
python tools/provision.py                          # ridger/MMfreeLM-370M -> cpp/
# other sizes / locations:
python tools/provision.py --model ridger/MMfreeLM-1.3B
python tools/provision.py --out-dir cpp --skip-download   # checkpoint already cached
```

`provision.py` pulls the checkpoint from the HF Hub via `huggingface_hub` (cached under
`~/.cache/huggingface`; for gated/private repos run `huggingface-cli login` first — the 370M
model is public), then writes `model.mmfree` + `tokenizer.mmtok` into `cpp/`.

<details>
<summary>Run one artifact at a time</summary>

```sh
python tools/provision.py --weights-only        # just cpp/model.mmfree
python tools/provision.py --tokenizer-only       # just cpp/tokenizer.mmtok (stdlib, no torch)
python tools/provision.py --skip-download ...     # reuse an already-cached checkpoint
```
</details>

For the test oracles (only needed for `ctest` — both modes are required, `ctest` runs a
float `e2e` and a fixed-point `e2e_fixed`):

```sh
python tools/dump_golden.py               --out cpp/golden          # float oracle vectors
python tools/dump_golden.py --mode fixedp --out cpp/golden_fixedp   # fixed-point oracle vectors
```

(`model.mmfree`, `tokenizer.mmtok` and `golden/` are git-ignored — regenerate them as above.)

# Usage
## Pre-trained Model Zoo
| Model Size     | Layer | Hidden dimension  | Trained tokens |
|:----------------|:------------:|:----------------:|:------------------:|
| [370M](https://huggingface.co/ridger/MMfreeLM-370M)  | 24  | 1024 | 15B  |
| [1.3B](https://huggingface.co/ridger/MMfreeLM-1.3B)  | 24 | 2048 | 100B  |
| [2.7B](https://huggingface.co/ridger/MMfreeLM-2.7B)  | 32  | 2560 | 100B  |

## Generation

Run the C++ engine with `mmfree-cli` (after building + provisioning above). It tokenizes the
prompt, runs the recurrent forward loop, and decodes the new tokens:

```sh
cd cpp
./build/mmfree-cli "In a shocking finding, scientists discovered" --gen 32
./build/mmfree-cli --ids 1,415,310 --gen 8        # raw ids in/out (no tokenizer, no EOS stop)
./build/mmfree-cli --bench --gen 128 --reps 5     # timing run
```

Key flags: `--blob PATH` (weight blob, default `model.mmfree`), `--tokenizer PATH` (default
`<blob dir>/tokenizer.mmtok`), `--gen N` (max new tokens), `--logits-out PATH` (dump last-position
logits as raw float32). See `cpp/app/main.cpp` for the full list.

## Tests

```sh
cd cpp/build
ctest --output-on-failure     # test_kernels + test_e2e vs the golden/ oracles
```

`ctest` needs `cpp/model.mmfree` and `cpp/golden/` (and `cpp/golden_fixedp/` for the fixed-point
path) — generate them with `tools/provision.py` + `tools/dump_golden.py` (both modes) first. The
C++ output is validated to match the PyTorch reference exactly (`tools/compare_prompts.py`).



# Citation
If you use this repo in your work, please cite our preprint:
```bib
@article{zhu2024scalable,
title={Scalable MatMul-free Language Modeling},
author={Zhu, Rui-Jie and Zhang, Yu and Sifferman, Ethan and Sheaves, Tyler and Wang, Yiqiao and Richmond, Dustin and Zhou, Peng and Eshraghian, Jason K},
journal={arXiv preprint arXiv:2406.02528},
year={2024}
}
```
