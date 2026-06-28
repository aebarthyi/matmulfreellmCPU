// mmfree/ternary_backend.hpp — pluggable ternary-matmul backend for BitLinear.
//
// The integer (FixedQ510) BitLinear projection reduces to one op:
//   acc[o] = sum_{n<N} x[n] * wq[o*N + n],   o in [0,M),   wq[.] in {-1,0,+1} (int8).
// This is the only part of the model that an accelerator replaces (RMSNorm, the int16
// quant, and the dequant all stay on the CPU around it). `TernaryBackend` is that seam.
//
// Activation contract: `x` carries int16-range integer activations (the Q5.10 codes
// produced in bitlinear.cpp), stored as int32. The CPU backend consumes them directly;
// an FPGA backend (KRIA repo) narrows them to int16 losslessly — the values are already
// saturated to [-32768, 32767]. `proj_id` identifies the projection for backends that
// hold resident per-projection weights (FPGA); the CPU backend ignores it and uses the
// `wq` pointer passed by the caller. Keeping the int32/`wq` shape here lets the CPU path
// call simd::ternary_dot_i32 verbatim — bit-identical to the historical inline loop.
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

  // Preferred batch width: how many activation vectors this backend wants per call.
  // 1 for the CPU (no win) and the b=1 bitstream; the spatial-batch FPGA returns its
  // CoreConfig.batchSize so bitlinear chunks prefill rows into that many per call. The
  // weight-streaming FPGA streams the resident weights ONCE per matmul_batch and applies
  // them to all `b` rows — the whole point of batching (amortizes the DDR weight wall).
  virtual std::size_t batch_size() const { return 1; }

  // Fill acc[i*M + m] for b activation vectors x[i*N + n] (i in [0,b)), row-major. Default
  // loops the scalar matmul per row, so any backend that doesn't override it (CpuBackend)
  // is bit-identical to the per-row path. `b` is <= batch_size().
  virtual void matmul_batch(int proj_id, const std::int32_t* x, std::int32_t* acc,
                            const std::int8_t* wq, std::size_t N, std::size_t M,
                            std::size_t b) {
    for (std::size_t i = 0; i < b; ++i)
      matmul(proj_id, x + i * N, acc + i * M, wq, N, M);
  }

  // Run k projections that share dims (N,M,b) as a CLUSTER (e.g. i/f/g, all reading
  // the same RMSNorm input). For each j in [0,k): produce(j, xq) fills b*N int32
  // activations for projection j, the backend accumulates acc[b*M], then
  // consume(j, acc) consumes it. proj_ids[j] addresses resident weights (FPGA);
  // wqs[j] is the CPU weight matrix [M,N] (the default/CPU path uses it).
  //
  // The default runs strictly serial (produce -> matmul_batch -> consume), so it is
  // BIT-IDENTICAL to k independent matmul_batch calls. A backend whose per-call cost
  // splits into engine work and overlappable CPU work (FPGA) overrides this to slot
  // produce(n+1)/consume(n-1) into the engine's wait windows.
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

// CPU reference: the original per-output int32 ternary reduction. Parallelized over output
// rows exactly as the inline loop in bitlinear.cpp was, so results are bit-identical and
// independent of lane/row order (integer accumulation is associative).
struct CpuBackend final : TernaryBackend {
  void matmul(int /*proj_id*/, const std::int32_t* x, std::int32_t* acc,
              const std::int8_t* wq, std::size_t N, std::size_t M) override {
#pragma omp parallel for schedule(static)
    for (std::size_t o = 0; o < M; ++o)
      acc[o] = simd::ternary_dot_i32(x, wq + o * N, N);
  }
};

}  // namespace mmfree
