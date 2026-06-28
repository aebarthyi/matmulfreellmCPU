// rmsnorm_swishgate.cpp — g_norm: out = RMSNorm(g)*weight ⊙ silu(o).
// g is the g_proj output (gets normalized); o is the recurrence output (gets silu-gated).
#include "mmfree/kernels.hpp"

#include "mmfree/simd.hpp"

#include <cmath>
#include <vector>

namespace mmfree {

void rmsnorm_swishgate(float* out, const float* g, const float* o, const float* weight,
                       std::size_t rows, std::size_t cols, float eps) {
  // silu(o_row) scratch, vectorized exp (see simd.hpp); reused across rows (the model
  // thread calls this serially). The trailing scale-and-multiply is memory-bound.
  thread_local std::vector<float> so;
  if (so.size() < cols) so.resize(cols);
  for (std::size_t r = 0; r < rows; ++r) {
    const float* gr = g + r * cols;
    float* outr = out + r * cols;
    float sumsq = simd::sumsq(gr, cols);
    float rstd = 1.0f / std::sqrt(sumsq / static_cast<float>(cols) + eps);
    simd::silu(so.data(), o + r * cols, cols);
    for (std::size_t c = 0; c < cols; ++c)
      outr[c] = (gr[c] * rstd * weight[c]) * so[c];
  }
}

}  // namespace mmfree
