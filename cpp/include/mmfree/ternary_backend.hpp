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

namespace mmfree {

struct TernaryBackend {
  // Fill acc[0..M) for a single activation vector x[0..N). wq is the int8 ternary weight
  // matrix [M, N] (nn.Linear layout); backends with resident weights may ignore it and
  // use proj_id instead.
  virtual void matmul(int proj_id, const std::int32_t* x, std::int32_t* acc,
                      const std::int8_t* wq, std::size_t N, std::size_t M) = 0;
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
