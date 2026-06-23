// mmfree/model.hpp — HGRN-Bit / MMfreeLM forward-only CPU model.
//
// Wires the validated kernels into the full forward: token embedding gather ->
// per-layer block (attn gated-linear-recurrence + swiglu MLP) over 24 layers ->
// final RMSNorm -> ternary lm_head -> logits. Batch=1, fp32, no cache (greedy
// recomputes the prefix each step, mirroring tools/reference.greedy_generate).
#pragma once

#include "mmfree/config.hpp"
#include "mmfree/kernels.hpp"
#include "mmfree/weights.hpp"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>

namespace mmfree {

// Decoding controls. temperature <= 0 selects greedy argmax (deterministic, the default
// and the validation path); temperature > 0 enables sampling: logits are scaled by 1/T,
// optionally restricted to the top_k highest, softmaxed, then sampled with a seeded RNG.
struct Sampling {
  float temperature = 0.0f;     // 0 => greedy
  int top_k = 0;                // <=0 => consider the full vocab
  std::uint64_t seed = 0;       // RNG seed (reproducible when set)
};

class Model {
 public:
  // `aq` selects the BitLinear activation numerics (Float = triton golden;
  // FixedQ510 = static 16-bit fixed-point activations into an integer accumulator).
  explicit Model(const Weights& w, ActQuant aq = ActQuant::Float, int frac_bits = 10);

  // Forward over `T` token ids. Writes logits [T, vocab_size] into `logits`.
  void forward(const std::int64_t* ids, std::size_t T, std::vector<float>& logits);

  // Incremental decode (recurrent state carried across steps). Appends up to `new_tokens`
  // tokens to `ids`, stopping early if `eos_id` is emitted (eos_id < 0 disables it).
  // `samp` selects greedy (default) or sampling. If `on_token` is set it is called with
  // each newly generated token id as it is produced (for streaming). Returns the full id
  // stream.
  std::vector<std::int64_t> generate(const std::vector<std::int64_t>& ids,
                                     std::size_t new_tokens, std::int64_t eos_id = -1,
                                     const Sampling& samp = {},
                                     const std::function<void(std::int64_t)>& on_token = {});

  const Config& config() const { return cfg_; }

 private:
  void ensure_scratch(std::size_t T);
  // Run layer `layer` over T positions. `rstate` ([hidden_size]) is the recurrent state
  // carried across calls (initial-in, final-out); nullptr = fresh zero state per call.
  void block(float* h, std::size_t T, std::size_t layer, float* rstate);
  // Embed ids, run all layers, final RMSNorm; returns the final-normed hidden [T, H].
  // Stateless: each layer's recurrence starts from zero (full-prefix forward).
  const float* layers_and_norm(const std::int64_t* ids, std::size_t T);

  const Weights& w_;
  Config cfg_;
  ActQuant aq_;
  int frac_bits_;

  // scratch, sized for the largest T seen (grown lazily)
  std::size_t cap_T_ = 0;
  std::vector<float> hbuf_, finalnorm_;  // hidden stream + final-norm output
  std::vector<float> hs_, i_, f_, onemf_, scan_, recur_, g_, oin_, oout_, resid_, gate_, y_;

  // Persistent per-layer recurrent state [num_hidden_layers * hidden_size] for incremental
  // decode in generate(). Lives here (host DRAM); on the FPGA split this is the only
  // cross-step state, owned by whichever side runs the scan.
  std::vector<float> rstate_;
};

}  // namespace mmfree
