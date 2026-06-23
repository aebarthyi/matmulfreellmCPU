// test_e2e.cpp — full-model forward vs the PyTorch "triton" golden.
// Loads the packed model.mmfree blob, runs the forward on the golden prompt, and
// checks: (1) logits match golden/logits.npy, (2) the greedy token stream matches
// golden/greedy_ids.npy exactly (the real acceptance gate).
#include "mmfree/model.hpp"
#include "mmfree/weights.hpp"
#include "npy.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

using namespace mmfree;

int main(int argc, char** argv) {
  std::string golden = (argc > 1) ? argv[1] : "golden";
  std::string blob = (argc > 2) ? argv[2] : "model.mmfree";
  std::string mode = (argc > 3) ? argv[3] : "float";  // "float" (triton) | "fixed" (Q5.10)
  ActQuant aq = (mode == "fixed") ? ActQuant::FixedQ510 : ActQuant::Float;
  std::printf("golden dir: %s   blob: %s   mode: %s\n", golden.c_str(), blob.c_str(),
              mode.c_str());

  Weights w(blob);
  Model model(w, aq, 10);
  const Config& cfg = model.config();
  std::printf("config: hidden=%zu layers=%zu inter=%zu vocab=%zu\n", cfg.hidden_size,
              cfg.num_hidden_layers, cfg.intermediate_size, cfg.vocab_size);

  // prompt ids (golden/input_ids.npy is stored as float)
  auto ids_npy = npy::load(golden + "/input_ids.npy");
  std::vector<std::int64_t> ids;
  for (float v : ids_npy.f32) ids.push_back(static_cast<std::int64_t>(std::llround(v)));
  std::size_t T = ids.size();

  int failures = 0;

  // ---- logits ----
  std::vector<float> logits;
  model.forward(ids.data(), T, logits);
  auto ref = npy::load(golden + "/logits.npy");  // [1, T, V]
  double max_abs = 0, num = 0, den = 0;
  for (std::size_t i = 0; i < ref.f32.size(); ++i) {
    double d = static_cast<double>(logits[i]) - ref.f32[i];
    max_abs = std::max(max_abs, std::abs(d));
    num += d * d;
    den += static_cast<double>(ref.f32[i]) * ref.f32[i];
  }
  double rel_l2 = std::sqrt(num / (den + 1e-30));
  // Fixed-point: C++ and torch round y=RMSNorm(x) at slightly different fp values, so a
  // few activations land on opposite sides of a Q5.10 boundary (1 LSB ~= 1e-3) and the
  // 1-LSB flips compound over 24 layers to ~2e-3. This is inherent fixed-point sensitivity
  // (same order as the fixedp-vs-triton quantization gap), not a kernel bug -- the per-op
  // test matches torch fixedp to ~1e-8. The hard acceptance gate is the greedy stream.
  double logit_tol = (mode == "fixed") ? 5e-3 : 1e-3;
  bool logits_ok = rel_l2 <= logit_tol;
  std::printf("[e2e] logits   rel_l2=%.3e  max_abs=%.3e  %s\n", rel_l2, max_abs,
              logits_ok ? "PASS" : "FAIL");
  if (!logits_ok) ++failures;

  // argmax of last position vs golden
  const std::size_t V = cfg.vocab_size;
  auto argmax = [&](const float* p) {
    std::size_t b = 0;
    for (std::size_t v = 1; v < V; ++v)
      if (p[v] > p[b]) b = v;
    return b;
  };
  std::size_t got_last = argmax(logits.data() + (T - 1) * V);
  std::size_t ref_last = argmax(ref.f32.data() + (T - 1) * V);
  std::printf("[e2e] argmax(last) got=%zu ref=%zu  %s\n", got_last, ref_last,
              got_last == ref_last ? "PASS" : "FAIL");
  if (got_last != ref_last) ++failures;

  // ---- greedy stream ----
  auto greedy = npy::load(golden + "/greedy_ids.npy");  // int64 [prompt+new]
  std::size_t new_tokens = greedy.i64.size() - T;
  auto stream = model.generate(ids, new_tokens);
  bool stream_ok = stream.size() == greedy.i64.size();
  for (std::size_t i = 0; stream_ok && i < stream.size(); ++i)
    stream_ok = stream_ok && (stream[i] == greedy.i64[i]);
  std::printf("[e2e] greedy stream (%zu new tokens)  %s\n", new_tokens,
              stream_ok ? "PASS" : "FAIL");
  if (!stream_ok) {
    ++failures;
    std::printf("  got: ");
    for (auto t : stream) std::printf("%lld ", static_cast<long long>(t));
    std::printf("\n  ref: ");
    for (auto t : greedy.i64) std::printf("%lld ", static_cast<long long>(t));
    std::printf("\n");
  }

  std::printf("\n%s (%d failure%s)\n", failures ? "FAILED" : "ALL PASSED", failures,
              failures == 1 ? "" : "s");
  return failures ? 1 : 0;
}
