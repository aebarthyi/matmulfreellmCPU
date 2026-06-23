// rmsnorm_swishgate.cpp — g_norm: out = RMSNorm(g)*weight ⊙ silu(o).
// g is the g_proj output (gets normalized); o is the recurrence output (gets silu-gated).
#include "mmfree/kernels.hpp"

#include "mmfree/simd.hpp"

#include <cmath>

namespace mmfree {

static inline float silu(float x) { return x / (1.0f + std::exp(-x)); }

void rmsnorm_swishgate(float* out, const float* g, const float* o, const float* weight,
                       std::size_t rows, std::size_t cols, float eps) {
  for (std::size_t r = 0; r < rows; ++r) {
    const float* gr = g + r * cols;
    const float* orow = o + r * cols;
    float* outr = out + r * cols;
    float sumsq = simd::sumsq(gr, cols);
    float rstd = 1.0f / std::sqrt(sumsq / static_cast<float>(cols) + eps);
    for (std::size_t c = 0; c < cols; ++c)
      outr[c] = (gr[c] * rstd * weight[c]) * silu(orow[c]);
  }
}

}  // namespace mmfree
