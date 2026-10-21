# `matmulfreellmCPU` — C++ inference engine for the MatMul-Free LM (HGRN-Bit) series

Working notes for writing up the project report. Everything below was read out of the tree at
commit `3e91da3` ("CMake: find Homebrew's keg-only libomp on macOS", 2026-08-14), branch
`main`. File:line references are to that state.

Scope: this is the **CPU engine and its offline toolchain only**. The PyTorch/Triton model was
deleted from the repo once the C++ port was verified bit-for-bit against it; the golden-oracle
test suite lives on the `validation` branch, and the FPGA/KRIA side lives in the parent
repository (`matmulfree_KRIA`), which consumes this tree as a submodule.

---

## 1. What the engine is, in one paragraph

A standalone, dependency-free **C++17 forward-only inference engine** (~2.6 kLOC of engine
code, ~0.8 kLOC of Python packer) for the three released MMfreeLM checkpoints (370M / 1.3B /
2.7B). Every projection in the model is a **ternary** matmul — weights are `{-1, 0, +1}`, so
the inner loop is add / subtract / skip and contains no multiplies. The token mixer is a gated
linear recurrence (HGRN) rather than attention, so decode carries an **O(1) recurrent state per
layer** instead of a growing KV cache. There is no GPU path, no Python at run time, no BLAS,
and no framework: the only external dependency is optional OpenMP.

Two artifacts feed it, both produced offline by `tools/provision.py`: a packed weight blob
(`model.mmfree`) and a packed tokenizer (`tokenizer.mmtok`). Ternary quantization is baked into
the blob at pack time — **the engine never quantizes weights**.

### Line-count map (engine + toolchain, excluding assets and build dirs)

| file | lines | role |
|---|--:|---|
| `cpp/include/mmfree/tq1.hpp` | 730 | packed ternary weight formats (1.6-bit TQ1_0 + 2-bit) and their backends |
| `cpp/app/main.cpp` | 484 | `mmfree-cli` front end |
| `cpp/src/model/hgrn_model.cpp` | 456 | full forward, incremental decode, batched serve |
| `cpp/tests/test_tq1.cpp` | 421 | packed-kernel correctness (exhaustive + randomized) |
| `tools/provision.py` | 409 | download → pack weights → pack tokenizer |
| `cpp/include/mmfree/simd.hpp` | 360 | AVX2 / AVX-512 / NEON / scalar primitives |
| `tools/reference.py` | 354 | PyTorch reference numerics (used only by the packer now) |
| `cpp/src/io/weights.cpp` | 261 | `mmap` blob loader + a hand-rolled JSON parser |
| `cpp/CMakeLists.txt` | 256 | SIMD auto-detection, flag policy, OpenMP discovery |
| `cpp/src/io/tokenizer.cpp` | 184 | LLaMA SentencePiece-BPE encode/decode |
| `cpp/src/kernels/bitlinear.cpp` | 161 | the fused RMSNorm → quant → ternary matmul → dequant op |
| `cpp/src/util/bench.cpp` | 145 | shared tok/s harness |
| `cpp/include/mmfree/ternary_backend.hpp` | 141 | the pluggable matmul seam (CPU / packed / FPGA) |
| remaining headers + small kernels | ~450 | model/config/tensor/profile/parallel/kernels decls, rmsnorm, swiglu, scan, swishgate |

---

## 2. Overall architecture

Five layers, each with a clean seam:

```
                   mmfree-cli  (cpp/app/main.cpp)
                        │  flag parsing, blob discovery, streaming text out
                        ▼
   Tokenizer            Model                       run_bench / Profiler
 (io/tokenizer.cpp)  (model/hgrn_model.cpp)      (util/bench.cpp, util/profile.cpp)
                        │  block() × L, generate(), generate_serve()
                        ▼
                     kernels                       Weights  (io/weights.cpp)
      bitlinear / bitlinear_cluster / rmsnorm /     mmap'd model.mmfree,
      rmsnorm_swishgate / hgrn_scan / swiglu        name → TensorRef views
                        │
                        ▼
              TernaryBackend  (ternary_backend.hpp)  ◄── the offload seam
        CpuBackend │ Tq1Backend │ Tq2Backend │ FpgaBackend (parent repo)
                        │
                        ▼
              simd:: primitives  (simd.hpp, tq1.hpp)
        AVX-512 │ AVX2 │ NEON │ portable scalar, chosen at configure time
```

The important architectural decision is the **`TernaryBackend` seam** (§7). Everything above it
is numerics-fixed and platform-independent; everything below it is a swappable implementation of
one single operation. This is what makes the CPU engine and the FPGA accelerator directly
comparable — the FPGA arm reuses this entire tree unchanged and only replaces the backend.

### Layout

```
cpp/include/mmfree/*.hpp   public headers (config, tensor, weights, kernels, model,
                           tokenizer, parallel, profile, bench, simd, tq1, ternary_backend)
cpp/src/{kernels,io,model,util}/   implementations
cpp/app/main.cpp           the CLI
cpp/tests/test_tq1.cpp     the only blob-free test on `main`
cpp/golden*, cpp/golden_fixedp/    .npy oracles (used by the `validation`-branch tests)
tools/{provision.py,reference.py}  offline packer + PyTorch numerics of record
Makefile                   thin wrapper: build / model / run / bench / profile / test
```

---

## 3. The model: HGRN-Bit forward pass

`Config` (`cpp/include/mmfree/config.hpp`) mirrors `HGRNBitConfig` and is read from the blob
header, so one binary serves all three checkpoints:

| | 370M | 1.3B | 2.7B |
|---|--:|--:|--:|
| layers | 24 | 24 | 32 |
| hidden `H` | 1024 | 2048 | 2560 |
| intermediate | 2816 | 5632 | 6912 |
| vocab | 32000 | 32000 | 32000 |
| blob size | 0.47 GB | 1.56 GB | 2.95 GB |

`num_heads = 1`, `expand_ratio = 1`, so `head_dim == hidden_size` and the recurrence runs over
all `H` lanes at once. `use_lower_bound = true`, `use_short_conv = false`, `rms_norm_eps = 1e-6`.

### Per-layer block — `Model::block`, `cpp/src/model/hgrn_model.cpp:80`

Seven ternary projections are involved per layer (six in the block + one shared `lm_head`), i.e.
**145 ternary tensors** for a 24-layer model — which is exactly what the packer emits.

```
resid ← h                                                   (saved for the residual)

── token mixer (gated linear recurrence) ────────────────────────────────────────
hs    = rmsnorm(h, attn_norm.w)                              [T, H]
i,f,g = bitlinear_cluster(hs; i_proj, f_proj, g_proj)        [T, H] each   ← ONE cluster
f     = sigmoid(f)
        if layer > 0 and use_lower_bound:  f = lb[layer] + (1 - lb[layer]) * f
scan  = silu(i) * (1 - f)
recur = hgrn_scan(scan, f)          state = f·state + scan ; out_t = state
oin   = rmsnorm(g, g_norm.w) ⊙ silu(recur)                   ("rmsnorm_swishgate")
oout  = bitlinear(oin; o_proj)
resid += oout

── MLP (pre-norm SwiGLU) ────────────────────────────────────────────────────────
hs    = rmsnorm(resid, mlp_norm.w)
y     = bitlinear(hs; gate_proj)                             [T, 2·inter]
z     = silu(y[:, :inter]) * y[:, inter:]                    (swiglu, chunk-of-2 split)
h     = resid + bitlinear(z; down_proj)                      [T, H]
```

Head: `finalnorm = rmsnorm(h, model.norm.w)` then `logits = bitlinear(finalnorm; lm_head)`.
`lm_head` is a **ternary BitLinear too** (it carries its own `norm.weight`) — there is no dense
GEMM anywhere in the model.

Two deliberate scheduling choices worth calling out in the report:

- **`g_proj` is hoisted upward** so that `i_proj`, `f_proj`, `g_proj` form a single
  *projection cluster* (`hgrn_model.cpp:98-107`). All three read the same post-`attn_norm`
  `hs` (each applying its own inner RMSNorm) and have no dependency on one another, so they are
  the only three projections in the block that an offloading backend can pipeline against each
  other. `g_`'s result is not consumed until the swishgate and nothing in between mutates `hs_`,
  which is what makes the hoist legal.
- **Scratch is grown once and reused**, `Model::ensure_scratch` (`hgrn_model.cpp:197`): thirteen
  buffers sized to the largest `T` ever seen, several of them reused at different widths
  (`scan_`, `onemf_`, `oin_` are sized to `max(T·H, T·inter)`). No allocation happens in steady
  state, and `onemf_` is deliberately double-purposed as the MLP's `y`-half staging buffer.

### Three execution modes

| entry point | shape | recurrent state | used by |
|---|---|---|---|
| `Model::forward` (`:239`) | full prefix `[T, V]` logits | **stateless** — every layer starts from zero and time-scans the whole prefix | `--logits-out`, validation |
| `Model::generate` (`:249`) | one stream, incremental | `rstate_` `[L·H]`, carried across steps | default CLI path, `--bench` at `b = 1` |
| `Model::generate_serve` (`:349`) | `B` streams in lockstep | `serve_state` `[L][B][H]`, layer-major | `--batch B` |

`generate` prefills the whole prompt in one `block(h, Tp, l, &rstate_[l·H])` call, then decodes
one token per step with `T = 1`. Incremental decode is **exact**, not approximate: HGRN's only
cross-token channel is the recurrent state, so replaying the prefix would just re-run the same
linear recurrence.

`generate_serve` prefills each stream **independently** (a real time-scan per prompt, into that
stream's own state slice), then flips the block into `stream_state = true` mode
(`hgrn_model.cpp:139`), where the `T` rows are reinterpreted as `B` independent streams that
each advance their own state by exactly one step — no time scan, and therefore row-parallel and
bit-identical. The state array is laid out layer-major precisely so layer `l`'s `B` states are
contiguous for that batched step.

---

## 4. Numerics contract

This is the part the report must state precisely, because it is the reason a hardware backend can
be gated bit-for-bit against the CPU. Specified in `cpp/include/mmfree/kernels.hpp:9-26` and
implemented against `tools/reference.py`.

- **Weight quant** — offline, per tensor, baked into the blob:
  `scale_w = 1 / mean(|W|)` (clamped at `1e-5`), `Wq = clamp(round(W · scale_w), -1, 1)`,
  effective weight `Wq / scale_w`. `round` is round-half-to-even in both numpy and torch.
- **Activation quant** — runtime, two selectable modes (`ActQuant`, `kernels.hpp:35`):
  - `Float` — activations stay fp32, unrounded. This is faithful to the original *Triton* fused
    kernel, which omits the activation rounding (and whose `clamp` is a no-op because
    `scale = 127/max|y|`). Accumulation is fp32 over signed ternary lanes.
  - `FixedQ510` — **the default** (`--mode fixed`). A *static* Q5.10 binary point:
    `xq = clamp(round(x · 2^10), -32768, 32767)`, then **int32 accumulation**, then
    `out = acc / (2^frac · scale_w)`. `--frac` moves the binary point. This is the datapath the
    FPGA implements, which is why it is the default here.
- **Projection**: `y = (Wq @ xq_row) / (scale_w · 2^frac)` — per output, a sum of `+x` / `−x` /
  skip over ternary lanes.
- **RMSNorm**: `x · rsqrt(mean(x²) + eps) · weight`, `eps = 1e-6`, reductions in fp32, no bias.
  Note the per-projection *inner* RMSNorm inside every BitLinear is in addition to the
  standalone `attn_norm` / `mlp_norm` / `g_norm` / final norm.
- **SiLU / SwiGLU**: `silu(x) = x·sigmoid(x)`, `swiglu(a,b) = silu(a)·b`.
- **`exp` is a Cephes single-precision polynomial**, not libm `expf` (`simd.hpp:212`): range
  reduction plus a 6th-order minimax poly, `2^n` assembled by building the float exponent field,
  input clamped to ±88.3762626647949. Max relative error ~1e-7 (under 1 ULP). The scalar tail
  uses the *same* polynomial so the vector body and tail of one call agree.

### Two hard rules that fall out of this

1. **`-ffast-math` is banned.** It reorders fp reductions, which changes RMSNorm, which changes
   the quantized int16 activations, which changes the int32 accumulators. It is called out in
   `cpp/CMakeLists.txt:26-45` and in the README. `MMFREE_OPT_FLAGS` exists so flags can be A/B'd
   without anyone being tempted to add it.
2. **The integer path is exactly reproducible.** Integer addition is associative, so lane order,
   thread count, weight format and batch width cannot move the result. This is the property the
   CPU==FPGA gate rests on, and it is asserted directly in `cpp/tests/test_tq1.cpp`.

---

## 5. Weight blob, tokenizer blob, and loading

### `model.mmfree` (written by `tools/provision.py:pack_weights`, read by `cpp/src/io/weights.cpp`)

```
[8]    magic "MMFREE1\n"
[8]    u64 header_len (little-endian)
[hl]   header JSON: {"config": {...}, "tensors": {name: {dtype, shape, offset, nbytes}}}
       pad to 64
[...]  data section — each tensor 64-byte aligned, offsets relative to section start
```

Contents: `model.embeddings` (fp32), `model.norm.w`, precomputed `lower_bounds` `[L, H]`, and per
projection a triple `{tag}.wq` (int8 ternary, `nn.Linear` layout `[out, in]`), `{tag}.scale_w`
(fp32 scalar), `{tag}.normw` (fp32 `[in]`). Tags are the HF names, e.g.
`model.layers.3.i_proj`, plus `lm_head`.

`lower_bounds` is **precomputed at pack time** — `softmax(dim=0)`, `cumsum`, minus row 0 — so the
engine only slices it.

The loader `mmap`s the file read-only and builds a `name → TensorRef` table of non-owning views
(`weights.hpp:24`). Nothing is copied into RAM, which matters most for the fp32 embeddings and
`lm_head` on a 4 GB board. Because ternary weights are stored as **one `int8` per weight**, the
blob is roughly the parameter count in bytes — deliberately fat on disk, because the in-memory
format is a separate, run-time choice (§8).

The header JSON is parsed by a ~150-line hand-rolled recursive-descent parser inside
`weights.cpp` — objects/arrays/strings/numbers/bools/null, insertion-ordered — so the engine
takes on no JSON dependency.

### `tokenizer.mmtok` (`tools/provision.py:pack_tokenizer`, `cpp/src/io/tokenizer.cpp`)

Magic `MMTOK1\n\0`, then `u32 vocab_size`, `u32 n_merges`, `i32 bos/eos/unk`, then
length-prefixed vocab pieces in id order, then length-prefixed rank-ordered merge strings.
Stdlib-only to produce; requires no torch.

The C++ tokenizer is a **from-scratch LLaMA SentencePiece-BPE** reimplementation, validated
id-for-id against HF `AutoTokenizer`:

- Normalizer: prepend `U+2581` (`▁`), map every `' '` → `▁`. No pre-tokenizer split. BOS
  prepended by default.
- Encode: codepoints become a doubly-linked symbol list; a min-heap on `(merge_rank, position)`
  merges the lowest-rank adjacent pair first, ties leftmost (llama.cpp's shape).
- Two staleness guards on popped bigrams (`tokenizer.cpp:126-134`): skip dead/non-adjacent
  symbols, **and re-verify the live pair still maps to exactly this merge rank**. Without the
  second one, two symbols that both grew via earlier merges get blindly concatenated into a
  non-vocab piece, silently producing spurious byte-fallback. Worth a sentence in the report as
  a representative subtle-correctness bug in BPE reimplementation.
- Byte-fallback: `<0xNN>` tokens resolved at load into a 256-entry table; any leftover piece not
  in the vocab is emitted byte-wise.
- Decode: `▁` → space, byte-fallback tokens fused back into raw bytes, one leading space
  stripped (the dummy prefix), specials skipped.

Streaming text out (`main.cpp:405-424`) exploits that this decode is **prefix-stable**: the CLI
re-decodes the whole generated id list each step and writes only the new byte suffix. Per-token
decoding would be wrong (multi-byte and byte-fallback fusion, the space mapping, the leading-space
strip); this is always byte-correct.

### Blob discovery

`--blob` > `$MMFREE_BLOB` > a search over `.`, `./cpp`, the exe dir, its parents and a `cpp/`
sibling (`main.cpp:search_model_file`), so `./build/mmfree-cli` from the repo root finds the
`make model` blob in `cpp/` with no flags. The tokenizer is looked up beside the blob, then by
the same search. Tokenizer-only paths (`--decode-ids`, `--gen 0 --print-ids`) never require a
blob at all.

---

## 6. The fused BitLinear op

`bitlinear` (`cpp/src/kernels/bitlinear.cpp:21`) is where most of the engine's wall time lives.
The integer path does, per chunk of `B` rows:

1. **RMSNorm + quant**, `B` rows in parallel (`:62`). Each row: `simd::sumsq` → `rstd` → scale by
   `norm_weight` → `simd::quant_q510` into an int32 staging buffer holding int16-range codes.
2. **One `matmul_batch_chunked` call** into the backend (`:86`).
3. **Dequant** via `simd::dequant_scale`, driven by the backend's column-range callback.

Three details that are load-bearing:

- **`B` comes from the backend**, not from the caller: `B = backend->batch_size()`
  (`bitlinear.cpp:46`). Batch width is a *property of the installed backend*. `B == 1` and
  `rows < B` degrade to the per-row path, and the default `matmul_batch` loops `matmul`, so the
  CPU result is bit-identical either way.
- **Chunked dequant.** `matmul_batch_chunked` lets a backend report completion in *output column
  ranges* as they land, so the caller's per-column dequant runs inside the engine's wait window
  instead of after the whole projection. The default implementation fires exactly one callback
  over `[0, M)` after the matmul, so the CPU path is unchanged and the arithmetic is identical
  either way (each element is scaled exactly once, order-independent).
- **Scratch-buffer discipline, and its footgun.** The `B`-row staging buffers `yqb` / `accb` are
  `static`, deliberately **not** `thread_local` (`bitlinear.cpp:52-58`): the `parallel_rows`
  workers each write their own row's slice and the *serial* `matmul_batch` consumes the whole
  buffer, so `thread_local` would hand each worker its own size-0 copy and produce out-of-bounds
  writes. `static` is only safe because `bitlinear` runs on the single model thread with all
  parallelism internal to the call. Conversely, per-row RMSNorm scratch **is** `thread_local`,
  declared *inside* the row body (`bitlinear.cpp:63-66`, `rmsnorm_swishgate.cpp:18-21`) — a
  function-scope `thread_local` captured by reference would bind one thread's buffer and race.
  This pair of opposite decisions is a good concrete example for a report section on
  parallelizing a numerics-fixed kernel.

`bitlinear_cluster` (`bitlinear.cpp:119`) is the same op for `k` projections sharing one input
and one shape, expressed as `produce(j) → matmul → consume(j)` lambdas so an overlapping backend
can slot `produce(n+1)` / `consume(n−1)` into the engine's wait windows. The default runs strictly
serially and is therefore bit-identical to `k` independent calls. It recomputes `rstd` per
projection even though it only depends on `x` — cheap (one SIMD reduction) and it keeps the
result bit-identical to the per-projection path, which was judged the better trade.

The **float path** is structurally different: no backend, per-row, and parallel over *output
rows* rather than input rows (`bitlinear.cpp:96-117`). It exists for reference/validation, not
speed, and it silently ignores `--weights` and `--batch` (with an explicit warning; see §12).

---

## 7. The `TernaryBackend` seam

`cpp/include/mmfree/ternary_backend.hpp`. The integer BitLinear projection reduces to exactly one
operation:

```
acc[o] = Σ_{n<N} x[n] · wq[o·N + n],     o ∈ [0, M),     wq ∈ {-1, 0, +1}
```

RMSNorm, the int16 quant and the dequant all stay on the CPU *around* it. Four virtuals, each
with a default that preserves bit-identical behaviour:

| method | contract |
|---|---|
| `matmul(proj_id, x, acc, wq, N, M)` | single activation vector. The only method a backend must implement. |
| `batch_size()` | preferred batch width. Drives `bitlinear`'s row chunking. Default 1. |
| `matmul_batch(..., b)` | `b` vectors. Default loops `matmul` → non-overriding backends are bit-identical to per-row. |
| `matmul_batch_chunked(..., on_cols)` | may report output-column ranges as they complete. Default: one callback over `[0, M)` after the matmul. |
| `matmul_seq(proj_ids, wqs, k, produce, consume, ...)` | run `k` same-shape projections as a pipelined cluster. Default: strictly serial. |

Activation contract: `x` carries **int16-range values stored as int32**. The CPU consumes them
directly (so `simd::ternary_dot_i32` can be called verbatim, bit-identical to the historical
inline loop); an FPGA backend narrows to int16 losslessly, since the values are already saturated
to `[-32768, 32767]`. `proj_id` addresses a backend's *resident* per-projection weights; CPU
backends ignore it and use the `wq` pointer.

`Model::set_backend(be, proj_ids)` (`hgrn_model.cpp:40`) installs a backend plus a
`tag → proj_id` registry. **A tag absent from the registry falls back to the CPU reduction for
that projection** (`proj_id = -1`), which is what makes partial offload and per-projection
A/B comparison possible. The backend is *borrowed* — the caller owns its lifetime.

Implementations:

- **`CpuBackend`** (`ternary_backend.hpp:110`) — `#pragma omp parallel for` over the `M` output
  rows, calling `simd::ternary_dot_i32`. Its batched form walks **weight-row-outer /
  batch-inner**: `wq[o]` is hoisted and reused for all `b` activations while still hot in L1/L2.
- **`Tq1Backend` / `Tq2Backend`** = `PackedBackend<Tq1Format|Tq2Format>` (`tq1.hpp:655`) — same
  loop structure over a lazily repacked, cached copy of the weights (§8).
- **`FpgaBackend`** — parent repo, `software/integration/fpga_runner/fpga_backend.{hpp,cpp}`.
  Overrides `batch_size()` (reporting the bitstream's compiled batch width),
  `matmul_batch_chunked` (column tiles as they drain) and `matmul_seq`. Not part of this tree, but
  it is the reason the seam has the exact shape it has.

---

## 8. Packing and unpacking ternary weights

The blob stores one `int8` per ternary weight — 8 bits for 1.58 bits of information, i.e. 4-5×
more weight traffic per token than the values need. A 370M decode step streams ~340 MB of int8
weights, a 2.7B step ~2.6 GB. `--weights` selects a denser **in-memory** format instead. No blob
change is involved: the same `model.mmfree` serves all three formats.

| `--weights` | format | block | bytes/block | bits/weight | 370M resident | 2.7B resident |
|---|---|--:|--:|--:|--:|--:|
| `int8` (default) | one `int8`, straight from the mmap | — | — | 8.000 | 0 (mmap) | 0 (mmap) |
| `1.6bit` | TQ1_0, base-3, 5 trits/byte | 256 | 52 | 1.625 | ~69 MB | ~532 MB |
| `2bit` | base-4, 4 trits/byte | 256 | 64 | 2.000 | ~85 MB | ~656 MB |

Note the resident heap is *offset* by the `int8` mmap pages going cold and becoming evictable
after packing — for the large models this can **lower** steady-state RSS rather than raise it.

### TQ1_0, 1.625 bits/weight (`cpp/include/mmfree/tq1.hpp:1-376`)

llama.cpp's TQ1_0 geometry: 256 weights per block, 5 trits per byte via base-3 packing
(`3^5 = 243 < 256`), with a 4-per-byte remainder group because 256 is not a multiple of 5.

```
qs[0..31]    5 trits each → elements  j + 32·l    (j<32, l<5)  = 160 elements
qs[32..47]   5 trits each → elements  160 + j + 16·l (j<16, l<5) =  80 elements
qh[0..3]     4 trits each → elements  240 + j + 4·l  (j<4,  l<4) =  16 elements
                                                     → 52 bytes / 256 weights
```

The group widths (32 / 16 / 4) are what make extraction vectorize: **the `l`-th trit of a
contiguous run of bytes maps to a contiguous run of elements** at offset `32·l` (resp. `16·l`,
`4·l`), so one vector load of weights pairs with one vector load of activations.

**Encode** (`tq1.hpp:97`), for trits `u_0..u_4` with `u = trit + 1 ∈ {0,1,2}`:

```
v = u_0·81 + u_1·27 + u_2·9 + u_3·3 + u_4        ∈ [0, 242]
q = ceil(v · 256 / 243) = (v·256 + 242) / 243    ∈ [0, 255]
```

**Decode** — no division, no lookup table:

```
q_l = (uint8)(q · 3^l)                 // deliberately wraps mod 256
u_l = floor(3·q_l / 256) = (q_l ≥ 86) + (q_l ≥ 171)
```

The wrap-around *is* the algorithm: `q/256` approximates the base-3 fraction
`0.u_0u_1u_2u_3u_4`; multiplying by `3^l` shifts `l` digits off the top (the mod-256 wrap discards
them) and the compare pair is `floor(3 · fraction)` = the next digit. The compare form is exactly
equivalent to llama.cpp's `((uint16)q_l · 3) >> 8` but needs no widening, so it maps onto a single
NEON `vcgeq_u8` pair.

**One deviation from llama.cpp:** the per-block fp16 scale `d` is dropped. This engine's scale is
per-*projection* (`scale_w`, applied once in the dequant), so a per-block scale would be
redundant and would inflate 52 → 54 bytes (1.6875 bits/weight). Nothing here reads or writes
gguf, so there is no interop to preserve.

### The 2-bit control format (`tq1.hpp:378-611`, namespace `tq2`)

4 trits per byte, 256 weights → **exactly 64 bytes = 2.000 bits/weight**, no remainder group
(4 divides 256). Byte `j` carries elements `j + 64·l`, `l ∈ [0,4)` — the same contiguity property.
Extraction is pure shift/mask, `t = ((q >> 2l) & 3) − 1`: no multiply, no wrap trick, no threshold
compares. So it is **cheaper per weight than TQ1_0** while still cutting weight traffic 4× versus
`int8`.

Why it exists is a measurement-methodology point worth stating in the report: the FPGA is swept
at both 2-bit and 1.6-bit, and the CPU arm needed the same two encodings for a per-format
comparison to mean anything. **2-bit is the one format all three platforms (this engine, the
FPGA, and a Jetson with no 1.6-bit path) share.** The encoding is CPU-internal (`u = trit + 1`,
little end first) and deliberately *not* the FPGA's wire encoding (`0 = hold`, `1 = add`,
`3 = subtract`) — nothing shares these bytes with hardware, and plain base-4 keeps the mask
constants trivial.

### Losslessness ⇒ bit-exactness

Both packed formats are **lossless** round trips over ternary values, so `dot_i32` accumulates the
same integer products as `simd::ternary_dot_i32`. Integer addition is associative, so even the
different summation *order* cannot change the result: the packed kernels are **bit-identical** to
`int8`, not merely close. (`dot_f32` reorders an fp reduction, exactly as the existing SIMD
`ternary_dot_f32` does versus scalar, so it is only equal to within reduction-order noise ~1e-6.)
This is the property that lets the CPU==FPGA gate keep passing with a packed backend installed.

### Lazy repack + cache (`PackedBackend::ensure`, `tq1.hpp:708`)

The packed copy is built **on a projection's first `matmul` and cached, keyed on the `wq`
pointer** the caller passes. That pointer is a stable address inside the mmap'd blob and is unique
per projection, so nothing extra has to be plumbed through `bitlinear` or `Model`. Every
projection is touched during the first decode step, so `--warmup` absorbs the whole one-off cost
and it never lands in a timed rep. The map is unsynchronized, which is safe for the same reason
the shared staging buffers are: `bitlinear` is invoked from the single model thread. `ensure` is
called *before* entering the parallel region for exactly this reason.

**Graceful fallback:** a projection whose inner dim `N` is not a multiple of 256 silently uses the
`int8` kernel (`Fmt::fits(N)`). Every real MMfreeLM shape divides by 256 — 1024, 2048, 2560, 2816,
5632, 6912 — but both `matmul` and `matmul_batch` carry the fallback branch, and both branches are
separately tested.

---

## 9. SIMD backends

Selected at **configure** time and passed down as one compile define
(`MMFREE_ARCH_X86` / `MMFREE_ARCH_ARM` / neither = portable scalar). Header-only, so the flags are
attached `PUBLIC` — see the footgun note in §11.

`cpp/include/mmfree/simd.hpp` provides: `sumsq`, `ternary_dot_f32`, `ternary_dot_i32`,
`quant_q510`, `dequant_scale`, `sigmoid`, `silu`, `swiglu`, `hgrn_step`, plus the `expf_poly` /
`exp8_ps` / `exp4_f32` exponential and the AVX2 horizontal sums. Every one has a scalar tail using
the same arithmetic as the vector body.

Two of these deserve report mentions:

- **`quant_q510`** (`simd.hpp:162`) replaced a per-element libm `nearbyintf`, which was the
  dominant `FixedQ510` CPU cost. It is bit-exact against the scalar reference because
  `vcvtnq_s32_f32` / `_mm256_cvtps_epi32` both round-to-nearest-ties-to-even, matching
  `nearbyintf` under the program's never-changed default rounding mode, and the vector min/max
  matches the scalar saturation.
- **`hgrn_step`** (`simd.hpp:335`) uses **mul-then-add (two roundings), not FMA**, specifically to
  match the scalar reference. A well-meaning FMA "optimization" here would break the parity gate.

### Ternary-dot implementation per ISA

| ISA | 1.6-bit extraction | how a ternary product is formed |
|---|---|---|
| **NEON** (`tq1.hpp:180`) | `vmulq_u8` (wraps mod 256) + `vcgeq_u8`/`vcltq_u8` threshold pair; `0xFF → 1` via logical shift, then subtract → `{-1,0,+1}` | widen int8→int32 and `vmlaq_s32` |
| **AVX2** (`tq1.hpp:198`) | no 8-bit multiply exists, so the mod-256 step runs in 16-bit lanes masked back to 8 by hand; `cmpgt` pair | `_mm256_mullo_epi32` (1.6-bit) / **`_mm256_sign_epi32`** (2-bit) |
| **AVX-512** (`tq1.hpp:237`) | 16-bit compares produce `__mmask32`/`__mmask16` directly | **no multiply and no trit vectors at all** — the masks drive `_mm512_mask_add_epi32` / `_mm512_mask_sub_epi32` on the activations |
| **scalar** | `trit_at` | `if (w > 0) s += x; else if (w < 0) s -= x;` |

Three hard-won micro-architecture lessons are recorded in comments and belong in the report as
findings, not footnotes:

1. **A ternary product is a conditional add, not a multiply.** The first AVX-512 version used
   `vpmulld` into one accumulator and *lost to GCC's auto-vectorized scalar loop* — `vpmulld` is
   2 uops and ~10-cycle latency on Skylake, a poor way to multiply by `{-1,0,+1}`. Replacing it
   with masked add/sub removed both the multiply and the int16→int32 widening. The AVX2 2-bit path
   makes the same move with `vpsignd` (one cheap uop: negate where `t<0`, zero where `t==0`, keep
   otherwise).
2. **Four accumulators, not one.** A single accumulator makes every add depend on the previous
   one; at `N = 1024` that is a ~60-deep serial chain per output column that out-of-order
   execution cannot hide. NEON, AVX2 and AVX-512 all keep `a0..a3`.
3. **Shift amounts must be immediates, and a runtime-indexed vector array spills to stack** —
   that cost ~38% end-to-end in an earlier version of the 2-bit AVX-512 path. Hence the
   unrolled-over-`l` macros rather than loops.

Two dispatch subtleties:

- **`tq2::kX86ScalarBelowN = 2048`** (`tq1.hpp:487`). Both x86 intrinsic paths pay a fixed cost
  per *output column* (zero four accumulators, then a multi-step `reduce_add`) amortized over
  `n/256` blocks. At `n = 1024` that is only four blocks, and for something like 370M's `lm_head`
  the cost is paid 32000 times — GCC's scalar loop has less setup and wins outright. Measured on a
  Xeon W-2155, 2-bit arm, decode tok/s at `b=1`: `n=1024` scalar 90.94 / AVX2 61.01 / AVX-512
  51.70 (**scalar by 43%**); `n=2048` scalar 41.73 / AVX2 75.76 / AVX-512 92.17 (**AVX-512 by
  121%**). The threshold is a round 2048 rather than a fitted value because the data can only
  place the crossover "somewhere between" — a tighter number would be false precision. It is
  deliberately **not** applied to TQ1_0, where the small-`n` difference (97.94 vs 92.07) is inside
  that host's ~11% run-to-run spread. Note also that a batch-aware rule is impossible here: the
  backend is called once per batch row and only ever sees `n`.
- **`MMFREE_SIMD=AUTO` probes the host for AVX-512 by compiling *and running* a test**
  (`CMakeLists.txt:91-121`), because a toolchain knowing the intrinsics does not mean the CPU
  executes them. The probe exercises F + BW + VL, exactly the trio `tq1.hpp` gates on. The 512-bit
  paths were worth ~29% over AVX2 on a Xeon and produce **bit-identical logits**, so "forgot the
  flag" used to be a silent loss of a third of throughput. `AUTO` is therefore deliberately
  **host-specific**, not lowest-common-denominator — even plain `X86` emits AVX2, which needs a
  2013-or-newer CPU. Cross-compiling skips the probe.

---

## 10. Threading

OpenMP only, and **optional** — without it the pragmas vanish and the build stays correct, just
single-threaded. Two helpers in `cpp/include/mmfree/parallel.hpp` express the whole policy:

- `parallel_rows(rows, f)` — `f(r)` for independent rows, `schedule(static)`, guarded on
  `rows > 1`.
- `parallel_chunks(n, f, chunk = 8192)` — contiguous chunks over a pure elementwise map. The chunk
  floor keeps tiny ops serial, where thread-fork cost would dominate.

### The bit-exactness contract

> Only parallelize over **independent rows** or **disjoint elementwise ranges**. Never split a
> floating-point reduction (`sumsq`, `dot`) across threads.

Every row keeps its own reduction serial, so threaded output is **byte-identical** to
single-threaded — verifiable with `OMP_NUM_THREADS=1` vs `N`. Thread count is therefore not a
numerics parameter. This matters because the fixed-point parity gate against the FPGA depends on
it.

### Where the parallelism actually is

| site | axis |
|---|---|
| `CpuBackend` / `PackedBackend` matmul | output rows `M` (batch inner) |
| `bitlinear` integer path | the `B` staged rows for RMSNorm+quant, and again for dequant |
| `bitlinear` float path | output rows `M`, inside the per-row loop |
| `rmsnorm`, `rmsnorm_swishgate` | rows |
| `swiglu`, sigmoid, `1−f`, residual adds, copies | elementwise chunks |
| `hgrn_scan` | **not parallel over time** (state carried); SIMD across the `D` lanes |
| decode `stream_state` scan | rows = streams (each advances its own state) |

The motivation is recorded in `parallel.hpp`: on the FPGA-offload path the CPU-side
RMSNorm/quant/dequant plus elementwise ops are ~60% of decode wall time (measured 2026-07-21) and
had been running single-threaded.

A caveat for anyone reading profiler output: `Profiler`/`ScopedTimer` are single-threaded
constructs used on the main thread only. Because the parallelism inside `bitlinear` is *inside*
the timed region, the enclosing timer records correct **wall-clock** time for the whole matmul,
not per-thread CPU time.

---

## 11. Build system

`cpp/CMakeLists.txt`, wrapped by a thin optional `Makefile` (every recipe echoes the plain
`cmake`/`python` command it runs). C++17, `Release` by default, out-of-source build to `./build`
at the repo root.

| option | default | effect |
|---|---|---|
| `MMFREE_SIMD` | `AUTO` | `AUTO` \| `X86_512` \| `X86` \| `ARM` \| `SCALAR` |
| `MMFREE_OPT_FLAGS` | `-O3` | optimization flags. **Never `-ffast-math`** |
| `MMFREE_TUNE_NATIVE` | `OFF` | adds `-mtune=native` / `-mcpu=native` — *scheduling only*, emits no new instructions, so the binary still runs on other CPUs of the same arch |
| `CMAKE_BUILD_TYPE` | `Release` | |

Notes worth carrying into the report:

- **`-O2` → `-O3` was a measured change, not a default-chasing one** (2026-07-30). On the KRIA at
  2.7B packed `b=4`, `-O3` together with `-mcpu=cortex-a53` cut host CPU time **12.8%** and lifted
  decode **4.6%** (29.56 → 30.91 tok/s), with the CPU==FPGA gate passing on both arms. The
  mechanism: the previous `-O2` setting *overrode* CMake's own Release `-O3`, and at `-O2` GCC 13
  uses `-fvect-cost-model=very-cheap`, which declines loops whose trip count is only known at
  runtime — e.g. the RMSNorm scale loop in `bitlinear.cpp`. `-O3` switches to `dynamic`. These
  flags keep IEEE fp semantics, so the int16 quant output and hence the int32 accumulators are
  identical.
- **`-mcpu`/`-mtune` are deliberately not set in `CMakeLists.txt`** — they are target-specific
  (the A53 is in-order dual-issue, so scheduling matters), so the FPGA runner's own Makefile adds
  `-mcpu=cortex-a53` on aarch64 and `MMFREE_TUNE_NATIVE` covers "build for this machine", leaving
  this file portable.
- **Arch flags are `PUBLIC`, not `PRIVATE`, and that is a correctness requirement.** `tq1.hpp` and
  `simd.hpp` are header-only and gate their 512-bit paths on `__AVX512F__`/`__AVX512BW__`/`
  __AVX512VL__`. A consumer compiled without the flags would silently get the AVX2 path while the
  library got the 512-bit one — an incoherent mix.
- **`-march=native` alone is not enough** to get 512-bit vectors out of *auto-vectorized* code on
  skylake-avx512 (GCC defaults to `-mprefer-vector-width=256` to dodge the AVX-512 downclock), but
  it *is* enough for these explicit, width-fixed intrinsics.
- **macOS OpenMP needs three-way help before `find_package` is even called**
  (`CMakeLists.txt:178-200`): AppleClang ships no OpenMP runtime; Homebrew's libomp is *keg-only*
  so nothing is on a default search path; and AppleClang rejects a bare `-fopenmp`, enabling the
  pragmas only through `-Xpreprocessor -fopenmp`. The file shells out to `brew --prefix libomp`
  (falling back to the two standard prefixes) and pre-fills `OpenMP_ROOT`, `OpenMP_CXX_FLAGS`,
  `OpenMP_CXX_LIB_NAMES`, `OpenMP_omp_LIBRARY`. Doing it *before* `find_package` matters —
  afterwards they are already cached as not-found. A missing OpenMP is a loud `message(WARNING)`,
  because otherwise the only symptom is single-core throughput.
- **Windows is unsupported natively** (the loader uses POSIX `mmap`); there is an explicit
  `message(WARNING)` and the advice is WSL. Routinely built and run on Linux x86-64 and Linux
  aarch64 (Kria K26 / Cortex-A53); macOS and \*BSD are expected to work but are not in the test
  matrix.

Makefile targets: `build` (default), `configure`, `model`, `run`, `bench`, `profile`, `test`,
`smoke`, `install`, `clean`, `distclean`, `help`. Knobs: `SIMD`, `TUNE`, `BUILD`, `BUILD_TYPE`,
`JOBS`, `MODEL`, `OUT`, `GEN`, `REPS`, `WEIGHTS`, `BATCH`, `PROMPT`, `ARGS`, `NO_TORCH`.

---

## 12. Batching (`--batch B`)

`B` concurrent streams decoded **in lockstep**. Each stream carries its own recurrent state and is
prefilled independently; the `B` decode steps then share **one** matmul per projection. Because
the kernels walk that matmul weight-row-outer / batch-inner, each weight row is fetched once and
reused across all `B` activations instead of the whole `[M, N]` matrix being re-swept per stream —
**weight traffic per token drops roughly `B`-fold**.

This is the same lever batching pulls on the FPGA (there the resident weights stream once per
batched matmul; here they land once in cache), which is precisely what makes the two platforms
comparable on this axis.

**Lossless.** Every `acc[i][o]` is still the identical `ternary_dot_i32` call on identical
operands — only the loop order changes — so batched output is **bit-identical** to `B` separate
runs. Without `--batch` the backend width stays 1 and every prior number reproduces exactly.

Plumbing detail: batch width rides the *same* seam as `--weights`, because it is a property of the
backend (`batch_size()`). `int8` normally runs with **no** backend installed (`bitlinear` falls
back to an internal `CpuBackend`), so a batched `int8` run has to install one explicitly — the
very same class, just constructed with a width (`main.cpp:357-359`).

### Measured, 370M on a 20-core Xeon, `--gen 128 --reps 5`, aggregate decode tok/s (two passes)

| weights | b=1 | b=2 | b=4 | b=8 | b=8 vs b=1 |
|:--|--:|--:|--:|--:|--:|
| `int8` | 112.6 / 103.5 | 163.5 / 149.7 | 177.5 / 163.5 | 204.0 / 199.1 | 1.81× / 1.92× |
| `2bit` | 108.1 / 100.6 | 163.9 / 193.0 | 173.4 / 160.7 | 224.3 / 206.3 | 2.08× / 2.05× |
| `1.6bit` | 107.7 / 99.3 | 152.9 / 137.7 | 158.6 / 146.7 | 199.1 / 186.0 | 1.85× / 1.87× |

Absolute cells carry ~8% run-to-run spread and the `2bit` b=2 cell is visibly bimodal (**do not
quote it**); the *ratio* is what is stable across passes and formats, and that is the claim.

### The knee is at b≈10, not at the thread count

`b = 1..40` ladder (`--gen 32 --reps 1`, shape-finding only):

| b | 1 | 2 | 5 | 10 | 20 | 30 | 40 |
|:--|--:|--:|--:|--:|--:|--:|--:|
| aggregate | 100.8 | 147.9 | 199.3 | 219.4 | 212.9 | 224.7 | 226.3 |
| per stream | 100.8 | 73.9 | 39.9 | 21.9 | 10.6 | 7.5 | 5.7 |

Threads are not the limit **because OpenMP already parallelizes over the `M` output rows —
thousands of them — so the cores are saturated at b=1.** Batching's only job is amortizing weight
traffic, and once a weight row is reused ~8-10 times that stops being the bottleneck and the
kernel goes compute-bound. Past the knee you pay per-stream latency (which degrades linearly,
forever) for no aggregate gain, so **b ≈ 8-12 is the operating point on a host this size**.
Prefill saturates identically: 112 ms/prompt at b=1, ~56 ms from b≈10 on. These are x86 numbers —
a 4-core Cortex-A53 knees much earlier and quantizes into scheduling waves over the batch, so the
board needs its own ladder rather than a scaled copy.

### Two honesty rules the CLI enforces loudly

`--mode float` ignores `--weights` (the packed kernels are integer-only) **and** does not batch the
matmul (the backend seam is integer-only). Both print an explicit warning rather than silently
doing something else (`main.cpp:330-345`). This is the same class of decision as validating
`--mode` up front: `--mode` used to accept anything and silently mean `float`, quietly changing the
numerics.

---

## 13. Benchmarking and profiling

**`Profiler` / `ScopedTimer`** (`cpp/include/mmfree/profile.hpp`, `src/util/profile.cpp`): a
label-keyed wall-clock accumulator, **off by default** — when disabled `ScopedTimer` reads one bool
and does nothing, so the instrumented model paths carry essentially zero overhead in normal runs.
Labels emitted by the model: `matmul`, `lm_head` (broken out so the large-vocab projection's cost
is separately visible), `rmsnorm`, `scan`, `swiglu`, `swishgate`, `gate`, `elementwise`, `embed`.
`print_profile` reports ms/run, %, calls/run, µs/call, plus the headline split
**"matmul (incl lm_head) vs other ops"**.

**`run_bench`** (`cpp/include/mmfree/bench.hpp`, `src/util/bench.cpp`) was deliberately extracted
out of `main.cpp` so that *every* front end — this CLI and the parent repo's CPU+FPGA runner —
times generation with byte-identical code. The only thing that differs between a CPU-only and a
CPU+FPGA measurement is the `Model`'s backend, not the measurement. That is what keeps the
comparison apples-to-apples, and it is worth stating in the report's methodology section.

Mechanics: greedy, EOS disabled, so every run produces exactly `--gen` tokens per stream.
`--warmup` runs are untimed (which is where the packed-format repack cost lands); the profiler is
reset and enabled around the timed reps **only**, so the breakdown lines up with the reported
tok/s. Prefill/decode are split by the token callback: the **B-th** callback marks "prefill done
for every stream" (`generate_serve` prefills stream-by-stream, so those callbacks arrive
staggered — which is exactly what a serving TTFT measurement wants), and at `B == 1` that is the
original single-callback condition unchanged.

Reported: `avg prefill` ms and prompt tok/s, `avg decode` tok/s **aggregate** across streams with
per-stream alongside, `avg overall` tok/s. Aggregate is the serving metric; per-stream is what one
user sees. Note `--batch B` batches **prefill** too, so prefill timings shift with `B` — a
`--batch 8` prefill number is not comparable to a `--batch 1` one.

---

## 14. Validation

On `main`, `make test` runs `ctest` (currently just `test_tq1`) plus a smoke test that proves the
binary links, runs `--help`, and — only if a blob is present — generates 4 tokens. So the target is
never a silent no-op, and it works on a machine that has never provisioned a checkpoint.

**`cpp/tests/test_tq1.cpp`** needs no model blob and no hardware, and it exercises whichever SIMD
path the library was built for (it links `mmfreecpu` purely to inherit its `PUBLIC` arch flags).
It is therefore *also* the check that a new platform's NEON/AVX-512 extraction is right. Six
checks, in increasing strength:

1. **Exhaustive over the whole input domain** — all `3^5 = 243` five-trit groups, all `3^4 = 81`
   four-trit groups, all 81 two-bit groups, and the equivalence of the branchless
   `(q≥86)+(q≥171)` form with llama.cpp's `((uint16)q·3)>>8` for every one of the `256×5` and
   `256×4` (byte, position) pairs. This is a **proof over the domain, not a sample**.
2. Pack → unpack round trip over random ternary rows for every real MMfreeLM inner dim
   (256/512/1024/2048/2560/2816/5632/6912).
3. `dot_i32` **exactly** equals `simd::ternary_dot_i32`, for both formats, and for the packed
   *scalar* reference too — so a broken SIMD path cannot hide behind a broken packer.
   (3b: 2000 random vectors at N=1024 and N=2560, to catch order-dependent SIMD bugs.)
4. `dot_f32` within reduction-order tolerance (`rel ≤ 1e-5`).
5. `Tq1Backend`/`Tq2Backend` vs `CpuBackend` over a whole projection (N=1024, M=512), plus:
   the cached second call is identical, `packed_tensors() == 1` (weights packed once, not per
   call), and a non-blockable `N = 1000` falls back to `int8` exactly.
6. **Batched == per-row for b=1..8 across all three formats, using DELIBERATELY DISTINCT
   activation rows.** This is the methodological point: a check that feeds `b` copies of one row
   cannot see a permutation, because every answer is then correct by accident. The test even
   asserts up front that its own reference rows are distinct, so it *can* detect one. The
   non-blockable fallback inside `matmul_batch` is a separate branch from `matmul`'s, so it gets
   its own check.

`./build/test_tq1 --bench` adds a kernel throughput report. The benchmark deliberately uses weight
matrices far larger than L2 (1 MiB on the K26) — a cache-resident matrix would measure only the
extra unpack ALU work and conclude the opposite of the truth about a bandwidth optimization.

**Model-level correctness** (`test_kernels`, `test_e2e`, and the PyTorch-comparison tooling
`dump_golden.py` / `compare_prompts.py` / `check_tokenizer.py`) lives on the **`validation`
branch**, because the `.npy` oracles and their torch dependency dwarf the engine. `CMakeLists.txt`
registers those targets conditionally on the sources existing, so one file serves both branches.
The recorded result: the C++ output matches the PyTorch reference **exactly** in both `float` and
`fixed` modes.

Also available as cheap self-checks: `--weights int8|1.6bit|2bit` must produce byte-identical
logits from the same model (all three verified on 370M); `--batch B` with one prompt and greedy
sampling must produce `B` identical streams.

---

## 15. Offline toolchain (`tools/`)

**`provision.py`** does all three offline steps in one command: HF Hub download (via
`huggingface_hub.snapshot_download`, restricted to `*.safetensors`, `*.json`, `tokenizer.*`,
`*.model`), pack weights → `model.mmfree`, pack tokenizer → `tokenizer.mmtok`. Imports are lazy,
so each path installs only what it uses: `--tokenizer-only` is stdlib-only, `--no-torch` needs
numpy, the default packer needs torch. Flags: `--model`, `--out-dir`, `--skip-download`,
`--weights-only`, `--tokenizer-only`, `--no-torch`.

Two packer backends behind one packing routine:

- **`_TorchBackend`** — loads and quantizes through `tools/reference.py`. This is the **numerics of
  record**; the golden oracles were generated through it.
- **`_NumpyBackend`** (`--no-torch`) — reads the safetensors file directly (a ~30-line minimal
  reader; BF16 is widened by shifting into the high half of an fp32, which is exact and matches
  torch's `.float()`; F16/F32 convert directly; anything else is *rejected rather than guessed
  at*) and quantizes in numpy.

The `--no-torch` divergence was measured, not assumed, over all 145 ternary tensors of
MMfreeLM-370M: **ternary weights bit-identical**; 108 of 145 `scale_w` values differ by at most
7.4e-7 relative (1-2 ULP) and `lower_bounds` by at most 2.4e-7 absolute, because numpy and torch
order the `mean()`/`softmax()` summations differently. End to end that shifted logits by ≤0.034 on
a ±16 range and left the greedy token stream unchanged over 16 tokens — but it is *not* bit-exact,
so the torch path stays the default and remains the numerics of record.

**`reference.py`** is the pure-PyTorch fp32 golden spec: no Triton, no `transformers` model class
(it reads safetensors directly), everything in fp32 with explicit math. It carries five selectable
`QUANT_MODE`s, which is useful context for the report because it documents what was *considered*:
`triton` (the actual fused kernel: ternary weights, activations **not** rounded — the golden), and
`fixedp` (static Q5.10 int16 into an int32 accumulator — what this engine's default mode and the
FPGA implement), plus `bitnet` (textbook b1.58, per-row int8 rounded activations), `int16` (per-row
*dynamic* 32767-scale) and `fullprec` (diagnostic: faithful to the upstream `generate.py`, whose
`prepare_for_inference()` bug leaves weights full-precision and is therefore **not**
matmul-free). `FIXED_FRAC_BITS = 10` is justified in-line: global `max|y| = 16.54` fits Q5.10's
±32, whereas Q4.11's ±16 would saturate.

---

## 16. Numbers and facts worth quoting in the report

- **Formats**: int8 = 8.000 bits/weight; TQ1_0 = 1.625 (52 B / 256 weights); 2-bit = 2.000 exactly
  (64 B / 256). All three produce byte-identical logits.
- **Weight traffic per decode step at int8**: ~340 MB (370M), ~2.6 GB (2.7B). This is the number
  packing and batching both attack.
- **AVX-512 vs AVX2**: ~29% on a Xeon, bit-identical logits.
- **x86 dispatch threshold** (2-bit, `b=1`, Xeon W-2155): `n=1024` scalar beats AVX-512 by 43%;
  `n=2048` AVX-512 beats scalar by 121%.
- **Batching on x86** (370M, 20-core Xeon): ~2× aggregate at b=8; knee at b≈10; per-stream falls
  linearly forever past it.
- **`-O3` + `-mcpu=cortex-a53` on the KRIA** (2.7B packed, b=4): −12.8% host CPU time, +4.6%
  decode (29.56 → 30.91 tok/s), gate still passing.
- **Which format wins is machine-dependent, and this is the headline caveat**: packing trades
  memory traffic for extraction work, so it wins where the matmul is bandwidth-bound and loses
  where it is ALU-bound. On a Cortex-A53 the 1.6-bit path measured **slower** than `int8`.
  Benchmark both on the target.
- **`--no-torch`**: ternary weights bit-identical; `scale_w` within 1-2 ULP; greedy stream
  unchanged over 16 tokens; still not bit-exact.

## 17. Footguns and caveats to keep straight while writing up

1. `-ffast-math` is banned, and the reason is a chain: it reorders fp reductions → changes RMSNorm
   → changes the int16 quant → changes the int32 accumulators → breaks the CPU==FPGA gate.
2. `hgrn_step` must stay mul-then-add, not FMA, or the parity gate breaks.
3. `MMFREE_SIMD=AUTO` produces a **host-specific** binary. `SIGILL` on another machine means
   rebuild with `SIMD=SCALAR` (or `X86` for the AVX2 baseline). Cross-compiles skip the AVX-512
   probe entirely.
4. Arch flags must stay `PUBLIC` or a consumer silently gets a different kernel than the library.
5. `static` vs `thread_local` staging buffers in `bitlinear` are opposite on purpose; getting
   either backwards is an out-of-bounds write or a data race, not a compile error.
6. Batched-kernel tests must use **distinct** rows — identical rows cannot see a permutation.
   (The same trap exists in the parent repo's board-side stream check with a shared prompt.)
7. `--mode float` silently disables both `--weights` and matmul batching; the CLI warns, but a
   scripted sweep that ignores stderr will produce numbers that look like a null result.
8. `--batch B` changes prefill timings too; never compare a b=8 prefill to a b=1 one.
9. Packed formats add resident heap but let the `int8` mmap pages go cold — for the large models
   steady-state RSS can *fall*. Measure RSS, don't reason about it.
10. The first token of a packed run pays the whole repack cost; always `--warmup` before quoting.
11. Aggregate vs per-stream tok/s answer different questions. Say which one a number is.
12. `Model::forward` is stateless (full-prefix) while `generate` is incremental; they agree
    mathematically, but only `generate`'s cost model is O(1) per step.

---

## 18. Citation

```bib
@article{zhu2024scalable,
  title={Scalable MatMul-free Language Modeling},
  author={Zhu, Rui-Jie and Zhang, Yu and Sifferman, Ethan and Sheaves, Tyler and Wang, Yiqiao
          and Richmond, Dustin and Zhou, Peng and Eshraghian, Jason K},
  journal={arXiv preprint arXiv:2406.02528},
  year={2024}
}
```

Upstream lineage: adapted from [flash-linear-attention](https://github.com/sustcsonglin/flash-linear-attention),
whose MatMul-Free LM this reimplements in C++. Checkpoints:
`ridger/MMfreeLM-{370M,1.3B,2.7B}`. The TQ1_0 block geometry and its base-3 arithmetic come from
llama.cpp's `ggml-quants` (with the per-block scale dropped, §8).
