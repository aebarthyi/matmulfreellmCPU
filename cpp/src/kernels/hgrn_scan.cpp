// hgrn_scan.cpp — gated linear recurrence: state = f_t*state + i_t ; out_t = state.
// Sequential along time, independent across the D feature lanes. Single (batch,head).
#include "mmfree/kernels.hpp"

#include "mmfree/simd.hpp"

#include <vector>

namespace mmfree {

void hgrn_scan(float* out, const float* i, const float* f, std::size_t T, std::size_t D,
               float* state_io) {
  std::vector<float> local(D, 0.0f);
  float* state = state_io ? state_io : local.data();

  // Sequential in time (state carried), SIMD across the D independent feature lanes.
  for (std::size_t t = 0; t < T; ++t)
    simd::hgrn_step(state, out + t * D, f + t * D, i + t * D, D);
}

}  // namespace mmfree
