// bitlinear.cpp — fused BitLinear: RMSNorm -> ternary matmul.
//
// Float mode: activations are NOT rounded (the triton golden omits it); the projection
// is a signed float accumulation over ternary lanes, divided by scale_w.
// FixedQ510 mode: the RMSNorm output is quantized to static Q(15-f).f fixed point
// (saturating to int16), then the projection is an INTEGER accumulation over ternary
// lanes (int32 -- the HW datapath), dequantized by acc / (2^f * scale_w).
#include "mmfree/kernels.hpp"

#include "mmfree/parallel.hpp"
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

  // Grow-only scratch reused across calls (avoids a heap alloc per projection).
  thread_local std::vector<float> y_buf;
  if (y_buf.size() < in_dim) y_buf.resize(in_dim);
  float* y = y_buf.data();  // one row's RMSNorm output (fp32)

  if (aq == ActQuant::FixedQ510) {
    // B = backend->batch_size() rows per call: RMSNorm+quant B rows, one matmul_batch,
    // dequant B rows. The integer accumulate is order-independent, so the result is
    // bit-identical for any B.
    const std::size_t B = std::max<std::size_t>(1, backend->batch_size());
    // SHARED staging buffers, not thread_local: parallel_rows workers each write their own
    // row slice and the serial matmul_batch reads the whole buffer (a thread_local would be
    // an unsized per-worker copy). `static` assumes bitlinear is never called concurrently.
    static std::vector<std::int32_t> yqb, accb;
    if (yqb.size() < B * in_dim)  yqb.resize(B * in_dim);    // B rows of fixed-point acts
    if (accb.size() < B * out_dim) accb.resize(B * out_dim);  // B rows of int accumulators

    for (std::size_t r0 = 0; r0 < rows; r0 += B) {
      const std::size_t bn = std::min(B, rows - r0);
      // RMSNorm + quant the bn rows in parallel (each row's reduction stays serial), with
      // per-worker thread_local scratch.
      parallel_rows(bn, [&](std::size_t j) {
        thread_local std::vector<float> ythr;
        if (ythr.size() < in_dim) ythr.resize(in_dim);
        float* yj = ythr.data();
        const float* xr = x + (r0 + j) * in_dim;
        // RMSNorm into yj (fp32 reduction).
        float sumsq = simd::sumsq(xr, in_dim);
        float rstd = 1.0f / std::sqrt(sumsq / static_cast<float>(in_dim) + eps);
        for (std::size_t c = 0; c < in_dim; ++c) yj[c] = xr[c] * rstd * norm_weight[c];
        if (norm_out) {
          float* nr = norm_out + (r0 + j) * in_dim;
          for (std::size_t c = 0; c < in_dim; ++c) nr[c] = yj[c];
        }
        // Quantize yj -> int16 fixed point (round-half-to-even, saturate); SIMD,
        // bit-exact vs the scalar nearbyint+clamp.
        simd::quant_q510(yqb.data() + j * in_dim, yj, qs, in_dim);
      });
      // Dequant per column range as the backend reports it, so an offloading backend
      // overlaps it with the next range. Each element is scaled exactly once either way.
      backend->matmul_batch_chunked(
          proj_id, yqb.data(), accb.data(), wq, in_dim, out_dim, bn,
          [&](std::size_t col0, std::size_t ncols) {
            parallel_rows(bn, [&](std::size_t j) {
              simd::dequant_scale(out + (r0 + j) * out_dim + col0,
                                  accb.data() + j * out_dim + col0, inv_fixed, ncols);
            });
          });
    }
  } else {
    // Float (triton): signed float accumulation, no activation rounding, no backend.
    // Parallel over output rows keeps each dot's reduction order.
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
      const float* yp = y;
#pragma omp parallel for schedule(static)
      for (std::size_t o = 0; o < out_dim; ++o) {
        float acc = simd::ternary_dot_f32(yp, wq + o * in_dim, in_dim);
        outr[o] = acc * inv_scale_w;
      }
    }
  }
}

void bitlinear_cluster(float* const* outs, const float* x, const float* const* norm_weights,
                       const int8_t* const* wqs, const float* scale_w, int k,
                       std::size_t rows, std::size_t in_dim, std::size_t out_dim, float eps,
                       int frac_bits, TernaryBackend* be, const int* proj_ids) {
  static CpuBackend cpu_default;
  TernaryBackend* backend = be ? be : &cpu_default;
  const float qs = static_cast<float>(1u << frac_bits);  // 2^frac_bits
  const std::size_t B = std::max<std::size_t>(1, backend->batch_size());

  // Chunk `rows` into the backend's batch B, exactly as bitlinear() does; the k
  // projections pipeline within each chunk.
  for (std::size_t r0 = 0; r0 < rows; r0 += B) {
    const std::size_t bn = std::min(B, rows - r0);

    // produce(j, xq): RMSNorm(x_row, norm_weights[j]) -> quant -> xq[bn*in_dim]. rstd is
    // recomputed per j to stay bit-identical to bitlinear(); per-worker scratch.
    auto produce = [&](int j, std::int32_t* xq) {
      parallel_rows(bn, [&](std::size_t i) {
        thread_local std::vector<float> ythr;
        if (ythr.size() < in_dim) ythr.resize(in_dim);
        float* y = ythr.data();
        const float* xr = x + (r0 + i) * in_dim;
        float sumsq = simd::sumsq(xr, in_dim);
        float rstd = 1.0f / std::sqrt(sumsq / static_cast<float>(in_dim) + eps);
        for (std::size_t c = 0; c < in_dim; ++c) y[c] = xr[c] * rstd * norm_weights[j][c];
        simd::quant_q510(xq + i * in_dim, y, qs, in_dim);
      });
    };
    auto consume = [&](int j, const std::int32_t* acc) {
      const float inv_fixed = 1.0f / (qs * scale_w[j]);
      parallel_rows(bn, [&](std::size_t i) {
        simd::dequant_scale(outs[j] + (r0 + i) * out_dim, acc + i * out_dim, inv_fixed, out_dim);
      });
    };

    backend->matmul_seq(proj_ids, wqs, k, produce, consume, in_dim, out_dim, bn);
  }
}

}  // namespace mmfree
