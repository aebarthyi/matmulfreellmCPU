// activations.cpp — SiLU / swiglu elementwise (scalar reference).
#include "mmfree/kernels.hpp"

#include <cmath>

namespace mmfree {

static inline float silu(float x) { return x / (1.0f + std::exp(-x)); }

void swiglu(float* out, const float* a, const float* b, std::size_t n) {
  for (std::size_t i = 0; i < n; ++i) out[i] = silu(a[i]) * b[i];
}

}  // namespace mmfree
