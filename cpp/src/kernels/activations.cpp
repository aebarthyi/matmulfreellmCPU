// activations.cpp — SiLU / swiglu elementwise.
#include "mmfree/kernels.hpp"

#include "mmfree/simd.hpp"

namespace mmfree {

void swiglu(float* out, const float* a, const float* b, std::size_t n) {
  simd::swiglu(out, a, b, n);  // out = silu(a)*b, vectorized exp (see simd.hpp)
}

}  // namespace mmfree
