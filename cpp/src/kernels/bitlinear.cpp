// bitlinear.cpp — scalar reference fused BitLinear: RMSNorm -> ternary matmul.
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

  // Per-call scratch, reused across calls (bitlinear is invoked serially per projection
  // by the model thread). thread_local + grow-only avoids a heap alloc + zero-init of
  // ~12 KB on every projection (144/token) that profiling charged to the CPU front-end.
  thread_local std::vector<float> y_buf;
  if (y_buf.size() < in_dim) y_buf.resize(in_dim);
  float* y = y_buf.data();  // one row's RMSNorm output (fp32)

  if (aq == ActQuant::FixedQ510) {
    // Batch B rows per backend call: RMSNorm+quant B rows, ONE matmul_batch (the engine
    // streams the resident weights once and applies them to all B rows — the batching
    // win), then dequant B rows. B == backend->batch_size(); B==1 (CPU / b=1 bitstream)
    // and rows<B reduce to the per-row path, and matmul_batch's default loops matmul, so
    // the CPU result is bit-identical. The integer accumulate is backend-independent, so
    // the dequant stays bit-identical regardless of B.
    const std::size_t B = std::max<std::size_t>(1, backend->batch_size());
    // SHARED staging buffers (NOT thread_local): the parallel_rows workers below each
    // write their own row's slice, and the SERIAL matmul_batch consumes the whole buffer.
    // thread_local would give each worker its own size-0 copy (resized only on the main
    // thread) -> out-of-bounds write. Safe as `static` because bitlinear runs on the single
    // model thread (the parallelism is internal to this call); revisit if bitlinear is ever
    // made concurrently callable (e.g. a multi-group interleave -> per-group buffers).
    static std::vector<std::int32_t> yqb, accb;
    if (yqb.size() < B * in_dim)  yqb.resize(B * in_dim);    // B rows of fixed-point acts
    if (accb.size() < B * out_dim) accb.resize(B * out_dim);  // B rows of int accumulators

    for (std::size_t r0 = 0; r0 < rows; r0 += B) {
      const std::size_t bn = std::min(B, rows - r0);
      // RMSNorm + quant the bn rows in parallel (rows independent; each row's sumsq
      // reduction stays serial -> bit-identical to single-threaded). The shared y_buf
      // can't be used across threads, so each worker uses its own thread_local scratch.
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
      // Dequant per COLUMN RANGE rather than after the whole projection. Dequant is
      // elementwise, so a column range can be finished the moment the backend has it — and
      // an offloading backend calls back while the engine is still computing the next
      // range, which puts this work inside the engine's wait window instead of after it.
      // The default backend fires one callback over [0,M) after the matmul, so the CPU path
      // is unchanged and the arithmetic is bit-identical either way (each element is scaled
      // exactly once, order-independent).
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

    // produce(j, xq): RMSNorm(x_row, norm_weights[j]) -> quant -> xq[bn*in_dim].
    // rstd depends only on x (shared across j), but recomputing it keeps this
    // bit-identical to bitlinear()'s per-row path and the cost is a NEON reduction.
    // Row-parallel; per-thread RMSNorm scratch (a shared thread_local can't cross the
    // worker threads — see rmsnorm_swishgate).
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
