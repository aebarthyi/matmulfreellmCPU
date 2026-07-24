// rmsnorm.cpp — scalar reference RMSNorm. fp32 reductions (no -ffast-math reorder).
#include "mmfree/kernels.hpp"

#include "mmfree/parallel.hpp"
#include "mmfree/simd.hpp"

#include <cmath>

namespace mmfree {

void rmsnorm(float* y, const float* x, const float* weight, std::size_t rows,
             std::size_t cols, float eps) {
  // Row-parallel: rows are independent and each row's sumsq reduction stays serial,
  // so the result is bit-identical to the single-threaded path (see parallel.hpp).
  parallel_rows(rows, [&](std::size_t r) {
    const float* xr = x + r * cols;
    float* yr = y + r * cols;
    float sumsq = simd::sumsq(xr, cols);
    float rstd = 1.0f / std::sqrt(sumsq / static_cast<float>(cols) + eps);
    for (std::size_t c = 0; c < cols; ++c) yr[c] = xr[c] * rstd * weight[c];
  });
}

}  // namespace mmfree
