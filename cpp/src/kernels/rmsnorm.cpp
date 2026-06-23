// rmsnorm.cpp — scalar reference RMSNorm. fp32 reductions (no -ffast-math reorder).
#include "mmfree/kernels.hpp"

#include "mmfree/simd.hpp"

#include <cmath>

namespace mmfree {

void rmsnorm(float* y, const float* x, const float* weight, std::size_t rows,
             std::size_t cols, float eps) {
  for (std::size_t r = 0; r < rows; ++r) {
    const float* xr = x + r * cols;
    float* yr = y + r * cols;
    float sumsq = simd::sumsq(xr, cols);
    float rstd = 1.0f / std::sqrt(sumsq / static_cast<float>(cols) + eps);
    for (std::size_t c = 0; c < cols; ++c) yr[c] = xr[c] * rstd * weight[c];
  }
}

}  // namespace mmfree
