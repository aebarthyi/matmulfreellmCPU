# matmulfreellmCPU — Plan: PyTorch + Triton → consolidated C++ CPU inference

> **2026-06-27: the PyTorch/Triton model was REMOVED.** This is now a C++-only repo
> (`cpp/`) plus the offline `tools/` toolchain (torch+safetensors, no triton/transformers-as-
> model) that packs weights/tokenizer and dumps the test oracles. The C++ port is verified to
> match the old reference exactly, so `mmfreelm/`, `triton-cpu/`, `generate*.py`, `setup.py`
> and the mmfreelm-coupled tools are gone. See README.md.

> **NEXT UP (2026-06-23): PERFORMANCE.** Correctness is done (P1/P4 — C++ port is e2e
> exact, ~9.6 min wall single-threaded scalar). Tomorrow = multithreading + optimizations.
> Order, biggest win first: **(1) KV-cache / incremental decode** in `Model::generate`
> (`src/model/hgrn_model.cpp` recomputes the whole prefix every step; HGRN recurrence
> already carries state, so incremental is natural) → **(2) threading (P3)** — pinned
> pool, parallelize ternary matmul over rows + lm_head over vocab tiles, recurrence stays
> serial in time → **(3) NEON ternary matmul (P2)**, widening NEON only (A53 ARMv8.0).
> After every change: `tools/compare_prompts.py` greedy stream must stay exact + ctest e2e.
> All of cpp/+tools/ is still UNTRACKED — commit the correctness milestone first.

**Status:** planning draft. Written from the `matmulfree_KRIA` runtime as ground truth
(`runtime/mmfreelm/`), to be moved into a standalone `matmulfreellmCPU` repo.

**One-line goal:** replace the PyTorch + triton-cpu HGRN-Bit (MMfreeLM) inference stack
with a single ahead-of-time-compiled C++ library + CLI that runs forward-only inference
on CPU, with **zero JIT**, fitting comfortably in the 4 GB KRIA A53 and portable to x86.

---

## 1. Why we are doing this

triton-cpu JIT-compiles every kernel through LLVM on first use. On the in-order
Cortex-A53 (KRIA, ARMv8.0, 4 GB) the cold compile of the fused RMSNorm+quant kernel
at the `down_proj` feature width (`N≈2731 → BLOCK_N=4096`, one fully-unrolled vector)
swap-thrashes for **10+ hours** and approaches OOM. Thinning autotune configs and
warming `~/.triton/cache` are band-aids; the codegen working set simply does not fit.

An AOT-compiled `.so` removes the entire problem: **no LLVM at runtime, no autotune, no
cache, deterministic memory.** triton-cpu is not a fast CPU backend anyway (LLVM-emitted
scalar/vector loops), so the bar to *match* it is low; the real prize is "never compiles
on the board, fits in RAM, starts instantly."

Scope is deliberately narrow: **forward-only inference**. No autograd, no training, no
backward kernels, no chunked-parallel training path.

---

## 2. The model we are porting (HGRN-Bit / MMfreeLM)

Reference: `mmfreelm/models/hgrn_bit/modeling_hgrn_bit.py`,
`mmfreelm/layers/hgrn_bit.py`, `mmfreelm/ops/`.

Config knobs (`configuration_hgrn_bit.py`) — values are checkpoint-driven, keep them
data-driven in C++:

| field | 370M default | notes |
|---|---|---|
| `vocab_size` | 32000 | tokenizer = LLaMA SentencePiece |
| `hidden_size` | 2048 (config default) | the harness checkpoint shows quant `N=1024` on i/f/g/o/gate |
| `num_hidden_layers` | 24 | |
| `num_heads` | 1 | `head_dim = hidden_size*expand_ratio / num_heads` |
| `expand_ratio` | 1 | `input_dim = hidden_size * expand_ratio` |
| `use_short_conv` | False (default; checkpoint may enable) | depthwise causal conv, `conv_size=4`, `share_conv_kernel` |
| `intermediate_size` | `round256(hidden*ratio*2/3)` | down_proj input width — the big quant `N` (~2731 in harness) |
| `hidden_ratio` | 4 | |
| `hidden_act` | swish (SiLU) | |
| `rms_norm_eps` | 1e-6 | |
| `attn_mode` | `fused_recurrent` | only mode we support |
| `use_lower_bound` | yes | per-layer forget-gate floor |

### 2.1 End-to-end forward (model level)
`HGRNBitModel.forward`:
1. `h = embeddings[input_ids]` — token embedding lookup (fp32, no matmul).
2. If `use_lower_bound`: `lb = softmax(lower_bounds, dim=0).cumsum(0) - lb[0]` →
   one `(num_layers, hidden)` tensor computed **once**, sliced per layer.
3. For each layer: `h = block(h, lower_bound=lb[i])` (the recurrent state carries
   across tokens within the layer, not across layers).
4. `h = final_norm(h)` (RMSNorm).
5. `logits = lm_head(h)` — **dense fp32 matmul** `[T,hidden] x [hidden,vocab]`
   (NOT ternary; it is a plain `nn.Linear`). This is the one true GEMM in the model.

### 2.2 Per-layer forward (`HGRNBitBlock.forward`)
```
residual = h
h = attn_norm(h)                          # standalone RMSNorm
# ---- attention (gated linear recurrence) ----
if use_short_conv: h = h_conv1d(h)        # depthwise causal conv + SiLU
i = i_proj(h);  f = f_proj(h)             # BitLinear (norm+quant+ternary)
f = sigmoid(f)
if layer>0 and use_lower_bound: f = lb + (1-lb)*f
i = swiglu(i, 1-f)                        # = silu(i) * (1-f)
i, f -> reshape (b, heads, T, head_dim)
o, state = fused_recurrent_hgrn(i, f)     # scan: hstate = f*hstate + i ; o[t]=hstate
o = g_norm( g_proj(h), reshape(o) )       # FusedRMSNormSwishGate: RMSNorm(o)*silu(g_proj(h))
o = o_proj(o)                             # BitLinear
# ---- mlp ----
h, residual = mlp_norm(o, residual, prenorm=True)   # RMSNorm + residual add, returns both
y = gate_proj(h); gate, y = chunk(y, 2)             # BitLinear (2*intermediate out)
z = down_proj( swiglu(gate, y) )                    # BitLinear; swiglu = silu(gate)*y
h = residual + z
```

Key structural facts that shape the port:
- **`BitLinear` (`FusedBitLinear`) is itself a fused `RMSNorm → per-token int8 quant →
  ternary matmul`.** Every projection internally normalizes its input. So the standalone
  `attn_norm`/`mlp_norm` are *separate* from the per-projection norm. Port `BitLinear` as
  one primitive that owns its norm weights.
- **The recurrence is the only stateful, inherently-sequential op.** `h_t = f_t*h_{t-1}+i_t`
  along time; trivially parallel across `(batch, head, dim)`. Pure-torch reference is
  `mmfreelm/ops/hgrn/naive.py:naive_recurrent_hgrn` — copy its math verbatim.
- Only `fused_recurrent` mode and forward exist on our path. `chunk.py` and all `_bwd_`
  kernels are training-only — **do not port**.

### 2.3 Triton kernel inventory (what actually JITs today)
Forward hot path (grep: `runtime/mmfreelm/ops`):

| triton kernel | math | port target |
|---|---|---|
| `_layer_norm_fwd_quant_kernel` (fusedbitnet) | RMSNorm + per-token int8 quant, fused, feeds ternary matmul | **the blocker** → `bitlinear_forward` |
| `fused_recurrent_hgrn_fwd_kernel` (recurrent_fuse) | gated linear scan over time | `hgrn_scan` |
| `FusedRMSNormSwishGate` (modules/fused_norm_gate) | `RMSNorm(x) * silu(gate)` | `rmsnorm_swishgate` |
| `swiglu` (modules/activations) | `silu(a) * b` | `swiglu` (elementwise) |
| `RMSNorm` (modules/layernorm) | standalone norm (+ optional residual add) | `rmsnorm` |
| `ShortConvolution` (modules/convolution) | depthwise causal conv1d + SiLU | `short_conv` (only if checkpoint uses it) |

None of these are GEMMs. The only dense matmuls are the **ternary projections**
(int8 act × {-1,0,+1} weight) and the **fp32 `lm_head`**.

---

## 3. Numerics contract (the thing that must not drift)

This is the spec the C++ must satisfy. Source: `activation_quant` / `weight_quant`
(`fusedbitnet.py:16-47`) and the fused kernel.

- **Weight quant (offline, per-tensor):** `scale_w = 1 / mean(|W|)`;
  `Wq = clamp(round(W*scale_w), -1, 1)` → ternary `{-1,0,+1}`; effective weight is
  `Wq / scale_w`. Done once at pack time, baked into the blob.
- **Activation quant (runtime, per-token / per-row):**
  `scale_a = 127 / max(|x_row|).clamp(min=1e-5)`;
  `xq = clamp(round(x*scale_a), -128, 127)`; dequant `= xq / scale_a`.
- **Projection:** `y = (Wq @ xq_row) / (scale_w * scale_a_row)` accumulated in fp32/int32.
  This is the matmul-free part: per output, sum of `+x` / `-x` / skip over ternary lanes.
- **RMSNorm:** `x * rsqrt(mean(x^2) + eps) * weight`, eps=1e-6, accumulate in fp32.
- **SiLU / swiglu:** `silu(x)=x*sigmoid(x)`; `swiglu(a,b)=silu(a)*b`.

**Acceptance:** the ternary projection is integer-exact reproducible (we already do this
HW-vs-CPU, see `phase_e_layer.py`); the C++ projection must match the PyTorch int path
**bit-exactly** at the integer accumulator, and within tight fp tolerance after dequant.
Norms/SiLU/recurrence match PyTorch fp32 within ~1e-5 rel-L2. End-to-end: compare logits
and greedy token stream against PyTorch on a fixed prompt.

> Watch-out: `-ffast-math` reorders fp reductions and changes rounding — it will break the
> "matches PyTorch" check on norms and the recurrence. Keep reductions in plain fp32, only
> fast-path the genuinely associative-safe elementwise ops, or drop `-ffast-math` entirely.

---

## 4. Target C++ architecture

A single static library `libmmfreecpu.{a,so}` + a `mmfree-cli` runner. No PyTorch, no
triton, no Python at runtime. Pure C++17, NEON on aarch64, scalar fallback elsewhere.

```
matmulfreellmCPU/
  include/mmfree/        # public headers
    config.hpp           # HGRNBitConfig mirror, loaded from JSON
    tensor.hpp           # lightweight row-major fp32/int8 views (no framework)
    model.hpp            # HGRNBitModel: load + generate
  src/
    kernels/             # the ported triton kernels (scalar ref + NEON)
      rmsnorm.cpp
      bitlinear.cpp      # norm+quant+ternary matmul  (THE one)
      hgrn_scan.cpp
      rmsnorm_swishgate.cpp
      swiglu.cpp
      short_conv.cpp     # optional
      gemm_f32.cpp       # lm_head only
    model/
      block.cpp          # HGRNBitBlock forward
      attention.cpp      # gated linear recurrence wiring
      mlp.cpp
      hgrn_model.cpp     # embeddings, lower_bounds, layer loop, final norm, lm_head
    io/
      weights.cpp        # load packed blob (config + ternary weights + norm/scale params)
      tokenizer.cpp      # SentencePiece (vendored) or pretokenized-ids passthrough for v1
    runtime/
      threadpool.cpp     # simple fixed pool; pin to A53 cores
    sample.cpp           # greedy / temperature / top-k
  tools/
    pack_weights.py      # OFFLINE: HF safetensors -> mmfree blob (reuse FPGA packer)
    dump_golden.py       # OFFLINE: PyTorch per-op + end-to-end golden tensors for tests
  tests/
    test_kernels.cpp     # each kernel vs golden
    test_e2e.cpp         # logits / token-stream vs golden
  app/
    main.cpp             # mmfree-cli: load model, prompt -> tokens
  CMakeLists.txt
```

### 4.1 Data layout decisions
- Activations: row-major `fp32`, `[T, dim]` (batch=1 for inference v1).
- Ternary weights: **reuse the existing FPGA 2-bit packing** (`0`=skip, `1`=+x, `3`=-x;
  see `software/integration/mmfree_pack.py`). One weight format across FPGA + CPU. Store
  per-tensor `scale_w` alongside. (Alternative: pack to 2 bits/weight in a CPU-friendly
  lane order; decide in P2 once we profile the NEON gather.)
- Per-projection norm weight + eps stored in the blob next to each `BitLinear`.

---

## 5. Component-by-component port plan

| PyTorch / triton piece | C++ plan | lift from ggml? | library |
|---|---|---|---|
| `BitLinear` norm+quant | hand-write: fp32 RMSNorm reduce → per-row max → int8 quant | no | bare NEON / Eigen reduce |
| Ternary matmul (i/f/g/o/gate/down_proj) | custom: per-output conditional add/sub over 2-bit lanes, int32 acc | **inner-loop technique only** (TQ2_0 `vec_dot`) | bare NEON |
| `lm_head` (dense fp32) | tiled GEMM | no | Eigen `MatrixXf` (NEON-vectorized) |
| `RMSNorm` (+residual) | elementwise reduce | no | bare/Eigen |
| `fused_recurrent_hgrn` | sequential scan, parallel over (head,dim) | no (math from `naive.py`) | bare loop + threads |
| `FusedRMSNormSwishGate` | norm × silu(gate) | no | bare |
| `swiglu` / SiLU | elementwise | no | bare (`expf`) |
| `ShortConvolution` | depthwise causal conv + SiLU | no | bare |
| embeddings | gather rows | no | memcpy |
| `lower_bounds` softmax+cumsum | one-time precompute at load | no | bare |
| tokenizer | SentencePiece | vendor `sentencepiece`, or accept pre-tokenized ids in v1 | — |

**ggml's role is a reference cookbook, not a dependency.** Its ternary `vec_dot`
(`TQ1_0`/`TQ2_0`, BitNet b1.58) shows the NEON pattern for sign-masked int8 accumulate,
but its data is coupled to `block_tq2_0` superblocks **and dotted against `Q8_K`
block-scaled activations** — a *different* quant scheme than our per-token quant. Using it
verbatim would change numerics and break the integer-exact check. So: read it, copy the
SIMD idea, write our kernel against *our* packing + per-token quant. Both are MIT, keep
attribution if any substantial snippet is copied. (A53 is ARMv8.0: **no `SDOT`/`i8mm`/SVE**,
so ggml's fastest dotprod paths don't even apply — plain widening NEON is what we'll write.)

**oneDNN / Arm Compute Library: not used.** On aarch64 oneDNN is a wrapper over ACL; it
targets dense int8/fp GEMM (which we don't need for the ternary path and which doesn't
shine without dotprod), and it's a heavy dependency for what is mostly elementwise work.
Eigen covers the one dense GEMM (`lm_head`) at far lower integration cost.

---

## 6. Weight & golden pipeline (offline, in Python — runs on the build host, not the board)

1. `tools/pack_weights.py`: load HF `ridger/MMfreeLM-370M` safetensors → for each
   `BitLinear`, `weight_quant` → ternary → pack 2-bit (reuse `mmfree_pack.py`), store
   `scale_w`, the projection's norm weight, eps; copy embeddings, `lm_head` (fp32),
   `lower_bounds`, final norm; emit one `model.mmfree` blob + a `config.json`.
2. `tools/dump_golden.py`: run the PyTorch model on a fixed prompt with hooks, dump
   per-op input/output tensors (norm, each projection's int accumulator, recurrence
   output, swiglu, gate, final logits) to `.npy`. These are the unit-test oracles and
   reuse the capture-hook approach already in `phase_e_layer_e2e.py`.

---

## 7. Phasing / milestones

- **P0 — Scaffold.** CMake, `tensor.hpp`, config loader, blob format + `pack_weights.py`,
  load weights into memory and round-trip. No compute yet.
- **P1 — Scalar reference (correctness first).** Port every kernel as plain, obvious
  scalar C++. Goal: **bit/`1e-5`-exact vs golden tensors**, single-threaded, slow is fine.
  Lock the numerics contract here. Ship `test_kernels.cpp` green.
- **P2 — NEON optimization.** Vectorize the hot kernels: ternary matmul (the bulk of
  FLOPs after `lm_head`), RMSNorm reduce, SiLU. Use ggml's `vec_dot` as the SIMD
  reference. Re-run golden tests after each kernel (NEON must equal scalar within fp
  tolerance). Microbench each vs the triton-cpu baseline times we already have
  (`phase_e_layer.py` reports ~4 ms/proj on FPGA; CPU glue numbers from `_e2e`).
- **P3 — Threading.** Fixed pool pinned to the 4 A53 cores. Parallelize: ternary matmul
  over output rows, recurrence over `(head, dim)`, `lm_head` over vocab tiles. Recurrence
  stays sequential along time.
- **P4 — End-to-end runner.** `mmfree-cli`: prompt → tokenizer → layer loop with carried
  recurrent state → sampling → detokenize. Validate greedy token stream == PyTorch.
- **P5 — Board bring-up + validation.** Build aarch64 on/for the KRIA, run the validation
  harness, confirm: no JIT, RSS fits in 4 GB, tokens/s vs the current triton path.

Each phase is independently shippable; P1 alone already proves the detritonize thesis.

---

## 8. Validation strategy

- **Per-op:** C++ kernel output vs `dump_golden.py` `.npy` oracle (P1 gate).
- **Integer-exact projection:** reuse the existing HW-vs-CPU integer-equality method
  (`phase_e_layer.py`) — the C++ ternary accumulator must match the PyTorch int path
  exactly before dequant.
- **End-to-end:** logits max|Δ| / cosine, and greedy token stream equality on a fixed
  prompt, vs PyTorch (mirror the metrics block in `phase_e_layer_e2e.py`).
- **Resource:** assert peak RSS and confirm zero LLVM/JIT at runtime (the whole point).

---

## 9. Risks & open questions

- **Exact dims of the shipped 370M checkpoint.** Config defaults (hidden=2048) disagree
  with the harness's observed quant `N` (1024 / ~2731). Resolve by reading the actual
  checkpoint config in `pack_weights.py`; keep everything data-driven, hard-code nothing.
- **Does the 370M checkpoint enable `use_short_conv`?** Decides whether `short_conv` is on
  the critical path. Confirm from the checkpoint before P1.
- **Double-norm question.** `BitLinear` carries its own norm *and* there are standalone
  `attn_norm`/`mlp_norm`. Confirm both are active (not identity) so we don't drop one.
- **Tokenizer scope.** v1 can accept pre-tokenized ids (Python tokenizes, C++ runs the
  model) to defer vendoring SentencePiece; full standalone needs it in C++.
- **fp reduction order vs PyTorch.** May need to match summation order (or use Kahan/
  pairwise) to hit tolerance on norms and the recurrence. No `-ffast-math` on reductions.
- **`lm_head` size.** `[hidden × 32000]` fp32 is the largest dense op and a big chunk of
  RAM + compute; consider int8 or ternary-fying it later if it dominates.
- **Memory ceiling.** Even AOT, the fp32 `lm_head` + embeddings dominate footprint on the
  4 GB board; budget this in P0's blob format (mmap the weights, don't copy).

---

## 10. What we explicitly are NOT building

Training, autograd, backward kernels, the chunked-parallel (`chunk.py`) path, multi-mode
attention, CUDA/GPU, dynamic batching. Forward, batch-1 (then batch-N), CPU, AOT. That's it.
