// hgrn_model.cpp — HGRN-Bit / MMfreeLM forward, batch=1, fp32.
// Mirrors tools/reference.py (QUANT_MODE="triton") op-for-op.
#include "mmfree/model.hpp"

#include "mmfree/kernels.hpp"
#include "mmfree/profile.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <utility>

namespace mmfree {

Model::Model(const Weights& w, ActQuant aq, int frac_bits)
    : w_(w), cfg_(w.config()), aq_(aq), frac_bits_(frac_bits) {}

namespace {
// Run a fused BitLinear projection identified by `tag` (e.g. "model.layers.3.i_proj").
// `be`/`proj_id` route the integer accumulate to an injected backend (CPU when be==nullptr).
void proj(const Weights& w, const std::string& tag, float* out, const float* x,
          std::size_t rows, float eps, ActQuant aq, int frac_bits, TernaryBackend* be,
          int proj_id) {
  const TensorRef& wq = w.get(tag + ".wq");        // int8 [out, in]
  const TensorRef& nw = w.get(tag + ".normw");     // f32 [in]
  const TensorRef& sw = w.get(tag + ".scale_w");   // f32 [1]
  std::size_t out_dim = wq.shape[0], in_dim = wq.shape[1];
  // The ternary BitLinear projections are the "matmul" work; lm_head is the same op but
  // broken out so its (large-vocab) cost is visible separately.
  ScopedTimer t(tag == "lm_head" ? "lm_head" : "matmul");
  bitlinear(out, x, nw.f32(), wq.i8(), sw.f32()[0], rows, in_dim, out_dim, eps, nullptr, aq,
            frac_bits, be, proj_id);
}
}  // namespace

void Model::set_backend(TernaryBackend* be, std::unordered_map<std::string, int> proj_ids) {
  backend_ = be;
  proj_ids_ = std::move(proj_ids);
}

int Model::proj_id_for(const std::string& tag) const {
  auto it = proj_ids_.find(tag);
  return it == proj_ids_.end() ? -1 : it->second;
}

void Model::run_proj(const std::string& tag, float* out, const float* x, std::size_t rows) {
  proj(w_, tag, out, x, rows, cfg_.rms_norm_eps, aq_, frac_bits_, backend_, proj_id_for(tag));
}

void Model::block(float* h, std::size_t T, std::size_t layer, float* rstate) {
  const std::size_t H = cfg_.hidden_size;
  const std::size_t inter = cfg_.intermediate_size;
  const float eps = cfg_.rms_norm_eps;
  const std::string p = "model.layers." + std::to_string(layer);

  // resid = h (save before we overwrite during the attention sub-block)
  float* resid = resid_.data();
  { MMFREE_PROFILE("elementwise");
    for (std::size_t i = 0; i < T * H; ++i) resid[i] = h[i];
  }

  // ---- attention ----
  { MMFREE_PROFILE("rmsnorm");
    rmsnorm(hs_.data(), h, w_.get(p + ".attn_norm.w").f32(), T, H, eps);  // hs
  }

  run_proj(p + ".i_proj", i_.data(), hs_.data(), T);
  run_proj(p + ".f_proj", f_.data(), hs_.data(), T);

  // f = sigmoid(f); if layer>0 && lower_bound: f = lb + (1-lb)*f
  float* f = f_.data();
  { MMFREE_PROFILE("gate");
    for (std::size_t n = 0; n < T * H; ++n) f[n] = 1.0f / (1.0f + std::exp(-f[n]));
    if (cfg_.use_lower_bound && layer > 0) {
      const float* lb = w_.get("lower_bounds").f32() + layer * H;  // [H] slice
      for (std::size_t t = 0; t < T; ++t)
        for (std::size_t c = 0; c < H; ++c) {
          float* ft = f + t * H + c;
          *ft = lb[c] + (1.0f - lb[c]) * (*ft);
        }
    }
  }

  // i = silu(i) * (1 - f);  scan input
  float* onemf = onemf_.data();
  { MMFREE_PROFILE("swiglu");
    for (std::size_t n = 0; n < T * H; ++n) onemf[n] = 1.0f - f[n];
    swiglu(scan_.data(), i_.data(), onemf, T * H);  // scan_ = silu(i)*(1-f)
  }

  // recurrence (heads=1 -> head_dim=H): state = f*state + scan_i ; out_t = state.
  // `rstate` carries the recurrent state across decode steps (nullptr = fresh from zero).
  { MMFREE_PROFILE("scan");
    hgrn_scan(recur_.data(), scan_.data(), f, T, H, rstate);
  }

  // g_norm: rmsnorm(g_proj(hs)) * silu(recurrence)
  run_proj(p + ".g_proj", g_.data(), hs_.data(), T);
  { MMFREE_PROFILE("swishgate");
    rmsnorm_swishgate(oin_.data(), g_.data(), recur_.data(), w_.get(p + ".g_norm.w").f32(), T, H,
                      eps);
  }

  run_proj(p + ".o_proj", oout_.data(), oin_.data(), T);

  // ---- mlp (prenorm: residual = o + residual; hs = rmsnorm(residual)) ----
  { MMFREE_PROFILE("elementwise");
    for (std::size_t n = 0; n < T * H; ++n) resid[n] += oout_.data()[n];
  }
  { MMFREE_PROFILE("rmsnorm");
    rmsnorm(hs_.data(), resid, w_.get(p + ".mlp_norm.w").f32(), T, H, eps);
  }

  run_proj(p + ".gate_proj", y_.data(), hs_.data(), T);  // [T, 2*inter]
  // chunk(y, 2): gate = y[:, :inter], yy = y[:, inter:]; z = silu(gate)*yy
  const std::size_t full = 2 * inter;
  float* gate = gate_.data();
  { MMFREE_PROFILE("swiglu");
    for (std::size_t t = 0; t < T; ++t) {
      const float* yr = y_.data() + t * full;
      for (std::size_t c = 0; c < inter; ++c) {
        gate[t * inter + c] = yr[c];              // gate half
        onemf_.data()[t * inter + c] = yr[inter + c];  // y half (reuse onemf_ as scratch)
      }
    }
    // z = swiglu(gate, yhalf) -> write into oin_ (>= T*inter)
    swiglu(oin_.data(), gate, onemf_.data(), T * inter);
  }

  // down_proj(z) -> [T, H]; h = residual + z_out
  run_proj(p + ".down_proj", oout_.data(), oin_.data(), T);
  { MMFREE_PROFILE("elementwise");
    for (std::size_t n = 0; n < T * H; ++n) h[n] = resid[n] + oout_.data()[n];
  }
}

void Model::ensure_scratch(std::size_t T) {
  if (T <= cap_T_) return;
  const std::size_t H = cfg_.hidden_size;
  const std::size_t inter = cfg_.intermediate_size;
  cap_T_ = T;
  hbuf_.assign(T * H, 0.0f);
  finalnorm_.assign(T * H, 0.0f);
  hs_.assign(T * H, 0.0f);
  i_.assign(T * H, 0.0f);
  f_.assign(T * H, 0.0f);
  recur_.assign(T * H, 0.0f);
  g_.assign(T * H, 0.0f);
  oout_.assign(T * H, 0.0f);
  resid_.assign(T * H, 0.0f);
  gate_.assign(T * inter, 0.0f);
  y_.assign(T * 2 * inter, 0.0f);
  // scratch reused across phases at varying widths -> size to the max
  std::size_t wide = std::max(T * H, T * inter);
  scan_.assign(wide, 0.0f);
  onemf_.assign(wide, 0.0f);
  oin_.assign(wide, 0.0f);
}

const float* Model::layers_and_norm(const std::int64_t* ids, std::size_t T) {
  const std::size_t H = cfg_.hidden_size;
  const float eps = cfg_.rms_norm_eps;
  ensure_scratch(T);

  // token embedding gather (fp32, no matmul)
  float* h = hbuf_.data();
  const float* emb = w_.get("model.embeddings").f32();
  for (std::size_t t = 0; t < T; ++t) {
    const float* row = emb + static_cast<std::size_t>(ids[t]) * H;
    for (std::size_t c = 0; c < H; ++c) h[t * H + c] = row[c];
  }

  for (std::size_t l = 0; l < cfg_.num_hidden_layers; ++l) block(h, T, l, nullptr);

  rmsnorm(finalnorm_.data(), h, w_.get("model.norm.w").f32(), T, H, eps);
  return finalnorm_.data();
}

void Model::forward(const std::int64_t* ids, std::size_t T, std::vector<float>& logits) {
  const std::size_t H = cfg_.hidden_size;
  const std::size_t V = cfg_.vocab_size;
  const float* fn = layers_and_norm(ids, T);
  // lm_head (ternary BitLinear) over all T positions -> logits [T, V]
  logits.assign(T * V, 0.0f);
  run_proj("lm_head", logits.data(), fn, T);
  (void)H;
}

std::vector<std::int64_t> Model::generate(const std::vector<std::int64_t>& ids,
                                          std::size_t new_tokens, std::int64_t eos_id,
                                          const Sampling& samp,
                                          const std::function<void(std::int64_t)>& on_token) {
  const std::size_t H = cfg_.hidden_size;
  const std::size_t V = cfg_.vocab_size;
  const std::size_t L = cfg_.num_hidden_layers;
  const float eps = cfg_.rms_norm_eps;
  const float* emb = w_.get("model.embeddings").f32();
  const float* norm_w = w_.get("model.norm.w").f32();

  std::vector<std::int64_t> stream = ids;
  std::vector<float> last_logits(V);

  std::mt19937_64 rng(samp.seed);
  std::uniform_real_distribution<double> uni(0.0, 1.0);
  std::vector<std::size_t> cand;  // candidate indices (reused across steps)

  // Pick the next token from last_logits: greedy argmax (temperature<=0) or
  // temperature/top-k sampling. Appends it to stream and returns it.
  auto emit = [&]() -> std::int64_t {
    std::size_t chosen = 0;
    if (samp.temperature <= 0.0f) {
      for (std::size_t v = 1; v < V; ++v)
        if (last_logits[v] > last_logits[chosen]) chosen = v;
    } else {
      cand.resize(V);
      for (std::size_t v = 0; v < V; ++v) cand[v] = v;
      auto hotter = [&](std::size_t a, std::size_t b) { return last_logits[a] > last_logits[b]; };
      std::size_t k = (samp.top_k > 0 && static_cast<std::size_t>(samp.top_k) < V)
                          ? static_cast<std::size_t>(samp.top_k)
                          : V;
      if (k < V) {
        std::nth_element(cand.begin(), cand.begin() + k, cand.end(), hotter);
        cand.resize(k);
      }
      // Softmax over the candidates (temperature-scaled, max-shifted), then sample.
      float maxl = last_logits[cand[0]];
      for (std::size_t j = 1; j < cand.size(); ++j) maxl = std::max(maxl, last_logits[cand[j]]);
      double sum = 0.0;
      std::vector<double> p(cand.size());
      for (std::size_t j = 0; j < cand.size(); ++j) {
        p[j] = std::exp((last_logits[cand[j]] - maxl) / samp.temperature);
        sum += p[j];
      }
      double r = uni(rng) * sum, acc = 0.0;
      chosen = cand.back();
      for (std::size_t j = 0; j < cand.size(); ++j) {
        acc += p[j];
        if (r <= acc) { chosen = cand[j]; break; }
      }
    }
    stream.push_back(static_cast<std::int64_t>(chosen));
    return static_cast<std::int64_t>(chosen);
  };
  // RMSNorm + lm_head on a single hidden row -> last_logits.
  auto head = [&](const float* h_row) {
    { MMFREE_PROFILE("rmsnorm");
      rmsnorm(finalnorm_.data(), h_row, norm_w, 1, H, eps);
    }
    run_proj("lm_head", last_logits.data(), finalnorm_.data(), 1);
  };

  // Fresh recurrent state for this generation: each layer starts from zero, then carries
  // across steps. Incremental decode is exact because HGRN's only cross-token channel is
  // this state -- recomputing the prefix would just replay the same linear recurrence.
  rstate_.assign(L * H, 0.0f);

  // ---- prefill: run the whole prompt once, populating per-layer recurrent state ----
  std::size_t Tp = stream.size();
  ensure_scratch(Tp);
  float* h = hbuf_.data();
  { MMFREE_PROFILE("embed");
    for (std::size_t t = 0; t < Tp; ++t) {
      const float* row = emb + static_cast<std::size_t>(stream[t]) * H;
      for (std::size_t c = 0; c < H; ++c) h[t * H + c] = row[c];
    }
  }
  for (std::size_t l = 0; l < L; ++l) block(h, Tp, l, &rstate_[l * H]);
  if (new_tokens == 0) return stream;
  head(h + (Tp - 1) * H);  // logits from the last prompt position
  std::int64_t t = emit();
  if (on_token) on_token(t);
  if (t == eos_id) return stream;

  // ---- decode: one new token at a time, carrying the recurrent state (O(1)/step) ----
  for (std::size_t step = 1; step < new_tokens; ++step) {
    { MMFREE_PROFILE("embed");
      const float* row = emb + static_cast<std::size_t>(stream.back()) * H;
      for (std::size_t c = 0; c < H; ++c) h[c] = row[c];  // single position at h[0..H)
    }
    for (std::size_t l = 0; l < L; ++l) block(h, 1, l, &rstate_[l * H]);
    head(h);
    t = emit();
    if (on_token) on_token(t);
    if (t == eos_id) break;
  }
  return stream;
}

}  // namespace mmfree
