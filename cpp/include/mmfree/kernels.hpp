// mmfree/kernels.hpp — forward-only CPU kernels for HGRN-Bit / MMfreeLM.
//
// Numerics contract (tools/reference.py):
//   * RMSNorm: x * rsqrt(mean(x^2)+eps) * weight, reductions in fp32, no bias.
//   * BitLinear: RMSNorm(eps) -> activations (see ActQuant) -> ternary matmul:
//       out[o] = ( sum_k wq[o,k] * y[k] ) / scale_w ,  wq in {-1,0,+1}.
//   * SiLU(x) = x*sigmoid(x);  swiglu(a,b) = silu(a)*b.
//   * hgrn_scan: state = f_t*state + i_t along time; out_t = state.
//   * rmsnorm_swishgate (g_norm): RMSNorm(g)*silu(o).
#pragma once

#include <cstddef>
#include <cstdint>

namespace mmfree {

struct TernaryBackend;  // mmfree/ternary_backend.hpp — pluggable ternary-matmul seam.

// Activation handling for the BitLinear projection input (post-RMSNorm y).
//   Float      : y kept fp32, unrounded (reference QUANT_MODE="triton").
//   FixedQ510  : y -> static signed Q(15-f).f (default Q5.10), saturated to int16, then
//                int32 accumulate; dequant = acc / (2^f * scale_w) (QUANT_MODE="fixedp").
enum class ActQuant { Float, FixedQ510 };

// y[r,:] = rmsnorm(x[r,:]) * weight ; x,y are [rows, cols], weight is [cols].
void rmsnorm(float* y, const float* x, const float* weight, std::size_t rows,
             std::size_t cols, float eps);

// Fused BitLinear forward. x:[rows,in], wq:[out_dim,in] int8 ternary (nn.Linear layout),
// norm_weight:[in], out:[rows,out_dim]. norm_out, if non-null, receives the RMSNorm output
// [rows,in]. `frac_bits` is the fixed-point fraction width (Q5.10 -> 10). On the FixedQ510
// path `be` (nullptr = built-in CPU reduction) does the integer accumulate, addressed by
// `proj_id`; the Float path ignores both.
void bitlinear(float* out, const float* x, const float* norm_weight, const int8_t* wq,
               float scale_w, std::size_t rows, std::size_t in_dim, std::size_t out_dim,
               float eps, float* norm_out = nullptr, ActQuant aq = ActQuant::Float,
               int frac_bits = 10, TernaryBackend* be = nullptr, int proj_id = -1);

// k BitLinear projections sharing input x and dims (e.g. i/f/g), each with its own
// norm_weight/scale_w/wq/proj_id/out. FixedQ510 only. Bit-identical to k bitlinear() calls,
// but routed through TernaryBackend::matmul_seq so an overlapping backend can pipeline them.
void bitlinear_cluster(float* const* outs, const float* x, const float* const* norm_weights,
                       const int8_t* const* wqs, const float* scale_w, int k,
                       std::size_t rows, std::size_t in_dim, std::size_t out_dim, float eps,
                       int frac_bits, TernaryBackend* be, const int* proj_ids);

// Elementwise. out[i] = silu(a[i]) * b[i].
void swiglu(float* out, const float* a, const float* b, std::size_t n);

// Gated linear recurrence over time for a single (batch,head). i,f,out are [T, D]
// row-major. state carried across t; if state_io!=nullptr it is the initial state in
// and final state out ([D]); otherwise state starts at 0.
void hgrn_scan(float* out, const float* i, const float* f, std::size_t T, std::size_t D,
               float* state_io = nullptr);

// g_norm: out[r,:] = rmsnorm(g[r,:])*weight  ⊙  silu(o[r,:]). All [rows,cols].
void rmsnorm_swishgate(float* out, const float* g, const float* o, const float* weight,
                       std::size_t rows, std::size_t cols, float eps);

}  // namespace mmfree
