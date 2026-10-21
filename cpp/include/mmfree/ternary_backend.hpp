// mmfree/ternary_backend.hpp — pluggable ternary-matmul backend for BitLinear.
//
// The integer (FixedQ510) projection reduces to the one op an accelerator replaces
// (RMSNorm, quant and dequant stay on the CPU around it):
//   acc[o] = sum_{n<N} x[n] * wq[o*N + n],   o in [0,M),   wq[.] in {-1,0,+1} (int8).
// `x` holds int16-range Q5.10 codes stored as int32, so an FPGA backend can narrow them to
// int16 losslessly. `proj_id` addresses resident per-projection weights (FPGA); CPU backends
// ignore it and use `wq`.
#pragma once

#include "mmfree/simd.hpp"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>

namespace mmfree {

struct TernaryBackend {
  // Fill acc[0..M) for a single activation vector x[0..N). wq is the int8 ternary weight
  // matrix [M, N] (nn.Linear layout); backends with resident weights may ignore it and
  // use proj_id instead.
  virtual void matmul(int proj_id, const std::int32_t* x, std::int32_t* acc,
                      const std::int8_t* wq, std::size_t N, std::size_t M) = 0;

  // Preferred batch width: activation rows per call (bitlinear chunks its rows by this).
  // A batching backend fetches each weight once per matmul_batch and applies it to all rows.
  virtual std::size_t batch_size() const { return 1; }

  // Fill acc[i*M + m] for b activation vectors x[i*N + n] (i in [0,b)), row-major;
  // b <= batch_size(). The default loops matmul() per row.
  virtual void matmul_batch(int proj_id, const std::int32_t* x, std::int32_t* acc,
                            const std::int8_t* wq, std::size_t N, std::size_t M,
                            std::size_t b) {
    for (std::size_t i = 0; i < b; ++i)
      matmul(proj_id, x + i * N, acc + i * M, wq, N, M);
  }

  // As matmul_batch, but completion may be reported in column ranges: on_cols(col0, ncols)
  // means acc[i*M + col0 .. col0+ncols) is final for every row i, so the caller's per-column
  // work (dequant) can overlap the rest of the projection. Every column must be reported
  // exactly once; the default reports [0,M) after the whole matmul.
  virtual void matmul_batch_chunked(int proj_id, const std::int32_t* x, std::int32_t* acc,
                                    const std::int8_t* wq, std::size_t N, std::size_t M,
                                    std::size_t b,
                                    const std::function<void(std::size_t, std::size_t)>& on_cols) {
    matmul_batch(proj_id, x, acc, wq, N, M, b);
    if (on_cols) on_cols(0, M);
  }

  // Run k projections sharing (N,M,b) as a cluster (e.g. i/f/g on the same input): for each
  // j, produce(j, xq) fills b*N activations, the backend accumulates b*M, and consume(j, acc)
  // takes the result. proj_ids[j] / wqs[j] address the weights. The default is strictly
  // serial (bit-identical to k matmul_batch calls); an overlapping backend may interleave
  // produce/consume with engine work.
  virtual void matmul_seq(const int* proj_ids, const std::int8_t* const* wqs, int k,
                          const std::function<void(int, std::int32_t*)>& produce,
                          const std::function<void(int, const std::int32_t*)>& consume,
                          std::size_t N, std::size_t M, std::size_t b) {
    std::vector<std::int32_t> xq(b * N), acc(b * M);
    for (int j = 0; j < k; ++j) {
      produce(j, xq.data());
      matmul_batch(proj_ids[j], xq.data(), acc.data(), wqs[j], N, M, b);
      consume(j, acc.data());
    }
  }

  virtual ~TernaryBackend() = default;
};

// CPU reference: per-output int32 ternary reduction, parallel over output rows.
// With batch > 1, matmul_batch walks weight-row-outer / batch-inner, so each weight row is
// fetched once and reused across the b activation rows while hot in cache. Every
// acc[i*M + o] is still the same ternary_dot_i32 call on the same operands, so results are
// bit-identical to batch=1 (cpp/tests/test_tq1.cpp checks this with distinct rows).
struct CpuBackend final : TernaryBackend {
  CpuBackend() = default;
  explicit CpuBackend(std::size_t batch) : batch_(batch ? batch : 1) {}

  std::size_t batch_size() const override { return batch_; }

  void matmul(int /*proj_id*/, const std::int32_t* x, std::int32_t* acc,
              const std::int8_t* wq, std::size_t N, std::size_t M) override {
#pragma omp parallel for schedule(static)
    for (std::size_t o = 0; o < M; ++o)
      acc[o] = simd::ternary_dot_i32(x, wq + o * N, N);
  }

  void matmul_batch(int proj_id, const std::int32_t* x, std::int32_t* acc,
                    const std::int8_t* wq, std::size_t N, std::size_t M,
                    std::size_t b) override {
    if (b <= 1) { matmul(proj_id, x, acc, wq, N, M); return; }
#pragma omp parallel for schedule(static)
    for (std::size_t o = 0; o < M; ++o) {
      const std::int8_t* w = wq + o * N;  // hoisted: reused by every row below
      for (std::size_t i = 0; i < b; ++i)
        acc[i * M + o] = simd::ternary_dot_i32(x + i * N, w, N);
    }
  }

 private:
  std::size_t batch_ = 1;
};

}  // namespace mmfree
