// activations.cpp — SiLU / swiglu elementwise.
#include "mmfree/kernels.hpp"

#include "mmfree/parallel.hpp"
#include "mmfree/simd.hpp"

namespace mmfree {

void swiglu(float* out, const float* a, const float* b, std::size_t n) {
  // out = silu(a)*b (vectorized exp, see simd.hpp). Elementwise -> chunk across threads,
  // bit-identical to the single-threaded call.
  parallel_chunks(n, [&](std::size_t off, std::size_t len) {
    simd::swiglu(out + off, a + off, b + off, len);
  });
}

}  // namespace mmfree
