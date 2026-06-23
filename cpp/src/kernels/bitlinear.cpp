// bitlinear.cpp — scalar reference fused BitLinear: RMSNorm -> ternary matmul.
//
// Float mode: activations are NOT rounded (the triton golden omits it); the projection
// is a signed float accumulation over ternary lanes, divided by scale_w.
// FixedQ510 mode: the RMSNorm output is quantized to static Q(15-f).f fixed point
// (saturating to int16), then the projection is an INTEGER accumulation over ternary
// lanes (int32 -- the HW datapath), dequantized by acc / (2^f * scale_w).
#include "mmfree/kernels.hpp"

#include "mmfree/simd.hpp"

#include <cmath>
#include <cstdint>
#include <vector>

namespace mmfree {

void bitlinear(float* out, const float* x, const float* norm_weight, const int8_t* wq,
               float scale_w, std::size_t rows, std::size_t in_dim, std::size_t out_dim,
               float eps, float* norm_out, ActQuant aq, int frac_bits) {
  std::vector<float> y(in_dim);
  std::vector<std::int32_t> yq(in_dim);  // fixed-point activations (FixedQ510)
  const float inv_scale_w = 1.0f / scale_w;
  const float qs = static_cast<float>(1u << frac_bits);  // 2^frac_bits
  const float inv_fixed = 1.0f / (qs * scale_w);

  for (std::size_t r = 0; r < rows; ++r) {
    const float* xr = x + r * in_dim;

    // RMSNorm into y (fp32 reduction).
    float sumsq = simd::sumsq(xr, in_dim);
    float rstd = 1.0f / std::sqrt(sumsq / static_cast<float>(in_dim) + eps);
    for (std::size_t c = 0; c < in_dim; ++c) y[c] = xr[c] * rstd * norm_weight[c];
    if (norm_out) {
      float* nr = norm_out + r * in_dim;
      for (std::size_t c = 0; c < in_dim; ++c) nr[c] = y[c];
    }

    float* outr = out + r * out_dim;
    if (aq == ActQuant::FixedQ510) {
      // Quantize y -> int16 fixed point (round-half-to-even via nearbyint, saturate).
      for (std::size_t c = 0; c < in_dim; ++c) {
        float q = std::nearbyintf(y[c] * qs);
        if (q > 32767.0f) q = 32767.0f;
        else if (q < -32768.0f) q = -32768.0f;
        yq[c] = static_cast<std::int32_t>(q);
      }
      // Ternary projection with integer accumulator; dequant by 2^f * scale_w.
      // Parallel over output rows: each `o` owns its accumulator and the k-reduction
      // stays serial, so the result is bit-identical to the single-threaded path.
      const std::int32_t* yqp = yq.data();
#pragma omp parallel for schedule(static)
      for (std::size_t o = 0; o < out_dim; ++o) {
        std::int32_t acc = simd::ternary_dot_i32(yqp, wq + o * in_dim, in_dim);
        outr[o] = static_cast<float>(acc) * inv_fixed;
      }
    } else {
      // Float (triton): signed float accumulation, no activation rounding.
      // Parallel over output rows (see FixedQ510 branch): per-row reduction order is
      // preserved, so this stays bit-exact vs the scalar reference.
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
