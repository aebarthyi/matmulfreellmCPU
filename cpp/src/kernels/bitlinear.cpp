// bitlinear.cpp — scalar reference fused BitLinear: RMSNorm -> ternary matmul.
//
// Float mode: activations are NOT rounded (the triton golden omits it); the projection
// is a signed float accumulation over ternary lanes, divided by scale_w.
// FixedQ510 mode: the RMSNorm output is quantized to static Q(15-f).f fixed point
// (saturating to int16), then the projection is an INTEGER accumulation over ternary
// lanes (int32 -- the HW datapath), dequantized by acc / (2^f * scale_w).
#include "mmfree/kernels.hpp"

#include "mmfree/simd.hpp"
#include "mmfree/ternary_backend.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace mmfree {

void bitlinear(float* out, const float* x, const float* norm_weight, const int8_t* wq,
               float scale_w, std::size_t rows, std::size_t in_dim, std::size_t out_dim,
               float eps, float* norm_out, ActQuant aq, int frac_bits, TernaryBackend* be,
               int proj_id) {
  // Default to the built-in CPU reduction when no backend is injected.
  static CpuBackend cpu_default;
  TernaryBackend* backend = be ? be : &cpu_default;
  const float inv_scale_w = 1.0f / scale_w;
  const float qs = static_cast<float>(1u << frac_bits);  // 2^frac_bits
  const float inv_fixed = 1.0f / (qs * scale_w);

  std::vector<float> y(in_dim);  // one row's RMSNorm output (fp32)

  if (aq == ActQuant::FixedQ510) {
    // Batch B rows per backend call: RMSNorm+quant B rows, ONE matmul_batch (the engine
    // streams the resident weights once and applies them to all B rows — the batching
    // win), then dequant B rows. B == backend->batch_size(); B==1 (CPU / b=1 bitstream)
    // and rows<B reduce to the per-row path, and matmul_batch's default loops matmul, so
    // the CPU result is bit-identical. The integer accumulate is backend-independent, so
    // the dequant stays bit-identical regardless of B.
    const std::size_t B = std::max<std::size_t>(1, backend->batch_size());
    std::vector<std::int32_t> yqb(B * in_dim);   // B rows of fixed-point activations
    std::vector<std::int32_t> accb(B * out_dim);  // B rows of integer accumulators

    for (std::size_t r0 = 0; r0 < rows; r0 += B) {
      const std::size_t bn = std::min(B, rows - r0);
      for (std::size_t j = 0; j < bn; ++j) {
        const float* xr = x + (r0 + j) * in_dim;
        // RMSNorm into y (fp32 reduction).
        float sumsq = simd::sumsq(xr, in_dim);
        float rstd = 1.0f / std::sqrt(sumsq / static_cast<float>(in_dim) + eps);
        for (std::size_t c = 0; c < in_dim; ++c) y[c] = xr[c] * rstd * norm_weight[c];
        if (norm_out) {
          float* nr = norm_out + (r0 + j) * in_dim;
          for (std::size_t c = 0; c < in_dim; ++c) nr[c] = y[c];
        }
        // Quantize y -> int16 fixed point (round-half-to-even via nearbyint, saturate).
        std::int32_t* yq = yqb.data() + j * in_dim;
        for (std::size_t c = 0; c < in_dim; ++c) {
          float q = std::nearbyintf(y[c] * qs);
          if (q > 32767.0f) q = 32767.0f;
          else if (q < -32768.0f) q = -32768.0f;
          yq[c] = static_cast<std::int32_t>(q);
        }
      }
      backend->matmul_batch(proj_id, yqb.data(), accb.data(), wq, in_dim, out_dim, bn);
      for (std::size_t j = 0; j < bn; ++j) {
        const std::int32_t* acc = accb.data() + j * out_dim;
        float* outr = out + (r0 + j) * out_dim;
        for (std::size_t o = 0; o < out_dim; ++o)
          outr[o] = static_cast<float>(acc[o]) * inv_fixed;
      }
    }
  } else {
    // Float (triton): signed float accumulation, no activation rounding. Per-row (no
    // backend / engine offload in this mode); parallel over output rows preserves the
    // per-row reduction order, so it stays bit-exact vs the scalar reference.
    for (std::size_t r = 0; r < rows; ++r) {
      const float* xr = x + r * in_dim;
      float sumsq = simd::sumsq(xr, in_dim);
      float rstd = 1.0f / std::sqrt(sumsq / static_cast<float>(in_dim) + eps);
      for (std::size_t c = 0; c < in_dim; ++c) y[c] = xr[c] * rstd * norm_weight[c];
      if (norm_out) {
        float* nr = norm_out + r * in_dim;
        for (std::size_t c = 0; c < in_dim; ++c) nr[c] = y[c];
      }
      float* outr = out + r * out_dim;
      const float* yp = y.data();
#pragma omp parallel for schedule(static)
      for (std::size_t o = 0; o < out_dim; ++o) {
        float acc = simd::ternary_dot_f32(yp, wq + o * in_dim, in_dim);
        outr[o] = acc * inv_scale_w;
      }
    }
  }
}

}  // namespace mmfree
