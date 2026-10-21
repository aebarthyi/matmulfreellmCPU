// mmfree/model.hpp — HGRN-Bit / MMfreeLM forward-only CPU model: token embedding ->
// per-layer block (gated linear recurrence + SwiGLU MLP) -> final RMSNorm -> ternary lm_head.
#pragma once

#include "mmfree/config.hpp"
#include "mmfree/kernels.hpp"
#include "mmfree/weights.hpp"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

namespace mmfree {

struct TernaryBackend;  // mmfree/ternary_backend.hpp — pluggable ternary-matmul seam.

// Decoding controls. temperature <= 0 is greedy argmax (the default); > 0 samples from
// softmax(logits / T), optionally restricted to the top_k highest, with a seeded RNG.
struct Sampling {
  float temperature = 0.0f;     // 0 => greedy
  int top_k = 0;                // <=0 => consider the full vocab
  std::uint64_t seed = 0;       // RNG seed (reproducible when set)
  float top_p = 1.0f;           // nucleus: sample from the smallest set holding this mass; 1 = off
  float repeat_penalty = 1.0f;  // >1 discourages tokens among the last repeat_last_n; 1 = off
  std::size_t repeat_last_n = 64;
};

// Picks the next token from a logits row per a Sampling: greedy argmax, or temperature /
// top-k / top-p sampling drawing once from its own seeded RNG per sampled token. The
// repetition penalty (HF/CTRL semantics: positive logits divided, negative multiplied, once
// per distinct token) applies to `recent`, the stream's tail, before either.
class Sampler {
 public:
  explicit Sampler(std::uint64_t seed) : rng_(seed) {}
  std::int64_t operator()(const float* logits, std::size_t vocab, const Sampling& samp,
                          const std::int64_t* recent = nullptr, std::size_t n_recent = 0);

 private:
  std::mt19937_64 rng_;
  std::vector<std::size_t> cand_;
  std::vector<double> p_;
  std::vector<float> penalized_;
  std::vector<std::int64_t> seen_;
};

class Model {
 public:
  // `aq` selects the BitLinear activation numerics (see ActQuant in kernels.hpp).
  explicit Model(const Weights& w, ActQuant aq = ActQuant::Float, int frac_bits = 10);

  // Forward over `T` token ids. Writes logits [T, vocab_size] into `logits`.
  void forward(const std::int64_t* ids, std::size_t T, std::vector<float>& logits);

  // Incremental decode (recurrent state carried across steps): appends up to `new_tokens`
  // tokens to `ids`, stopping early on `eos_id` (< 0 disables). `on_token`, if set, streams
  // each new id. Returns the full id stream.
  std::vector<std::int64_t> generate(const std::vector<std::int64_t>& ids,
                                     std::size_t new_tokens, std::int64_t eos_id = -1,
                                     const Sampling& samp = {},
                                     const std::function<void(std::int64_t)>& on_token = {});

  // Serve B sequences: prefill each independently (its own recurrent state), then decode
  // all B in lockstep with one batched matmul per projection per step. `new_tokens` is per
  // stream; stream b samples with seed samp.seed + b; a stream that emits eos_id stops
  // growing but keeps its slot. Under greedy each stream matches a separate generate().
  // `on_token(stream, id)` fires per token; prefill runs stream by stream, so the B
  // first-token callbacks arrive staggered. Returns the B full id streams.
  std::vector<std::vector<std::int64_t>> generate_serve(
      const std::vector<std::vector<std::int64_t>>& prompts, std::size_t new_tokens,
      std::int64_t eos_id = -1, const Sampling& samp = {},
      const std::function<void(std::size_t, std::int64_t)>& on_token = {});

  // Interactive decoding on one persistent recurrent state. feed() advances it over `T`
  // tokens (T > 1 is a prefill) and returns the logits [vocab] of the last one;
  // reset_state() starts a new sequence. generate() is exactly these two plus a Sampler.
  void reset_state();
  const std::vector<float>& feed(const std::int64_t* ids, std::size_t T);

  const Config& config() const { return cfg_; }

  // Inject a ternary-matmul backend plus a registry mapping projection tags (e.g.
  // "model.layers.3.i_proj", "lm_head") to its proj_ids; an unregistered tag gets -1. The
  // backend is borrowed and must outlive the Model. Never called = built-in CPU path.
  void set_backend(TernaryBackend* be, std::unordered_map<std::string, int> proj_ids);

 private:
  void ensure_scratch(std::size_t T);
  // Run BitLinear projection `tag` over `rows` positions, dispatching to the configured
  // backend (CPU by default) with the tag's registered proj_id.
  void run_proj(const std::string& tag, float* out, const float* x, std::size_t rows);
  // Run k BitLinear projections sharing input `x` as a cluster (e.g. i/f/g) — same as
  // k run_proj calls, but routed through the backend's pipelined matmul_seq.
  void run_proj_cluster(const std::string* tags, float* const* outs, int k,
                        const float* x, std::size_t rows);
  int proj_id_for(const std::string& tag) const;
  // Run layer `layer` over T positions. `rstate` ([hidden_size]) is the recurrent state
  // carried across calls (initial-in, final-out); nullptr = fresh zero state per call.
  // stream_state=true reinterprets the T rows as T independent streams (serving): rstate
  // is [T, hidden_size] and each row advances its OWN state by one step (no time scan).
  void block(float* h, std::size_t T, std::size_t layer, float* rstate,
             bool stream_state = false);
  // Embed ids, run all layers, final RMSNorm; returns the final-normed hidden [T, H].
  // Stateless: each layer's recurrence starts from zero (full-prefix forward).
  const float* layers_and_norm(const std::int64_t* ids, std::size_t T);

  const Weights& w_;
  Config cfg_;
  ActQuant aq_;
  int frac_bits_;

  // set_backend state; nullptr -> built-in CPU reduction.
  TernaryBackend* backend_ = nullptr;
  std::unordered_map<std::string, int> proj_ids_;

  // scratch, sized for the largest T seen (grown lazily)
  std::size_t cap_T_ = 0;
  std::vector<float> hbuf_, finalnorm_;  // hidden stream + final-norm output
  std::vector<float> hs_, i_, f_, onemf_, scan_, recur_, g_, oin_, oout_, resid_, gate_, y_;

  // Per-layer recurrent state [num_hidden_layers * hidden_size] carried across feed() calls.
  std::vector<float> rstate_;
  std::vector<float> logits_;  // feed()'s output row [vocab]
};

}  // namespace mmfree
