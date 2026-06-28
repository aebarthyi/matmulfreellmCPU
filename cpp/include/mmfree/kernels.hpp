// mmfree/kernels.hpp — forward-only CPU kernels for HGRN-Bit / MMfreeLM.
//
// Numerics contract ("triton" golden, see tools/reference.py):
//   * RMSNorm: x * rsqrt(mean(x^2)+eps) * weight, reductions in fp32, no bias.
//   * BitLinear: RMSNorm(eps) -> activations NOT rounded -> ternary matmul:
//       out[o] = ( sum_k wq[o,k] * y[k] ) / scale_w ,  wq in {-1,0,+1}.
//     (matmul-free: signed float adds over ternary lanes; the triton kernel omits
//      activation rounding, so there is no int8 quant step.)
//   * SiLU(x) = x*sigmoid(x);  swiglu(a,b) = silu(a)*b.
//   * hgrn_scan: state = f_t*state + i_t along time; out_t = state.
//   * rmsnorm_swishgate (g_norm): RMSNorm(g)*silu(o).
#pragma once

#include <cstddef>
#include <cstdint>

namespace mmfree {

struct TernaryBackend;  // mmfree/ternary_backend.hpp — pluggable ternary-matmul seam.

// Activation handling for the BitLinear projection input (post-RMSNorm y).
//   Float      : y kept fp32 (the "triton" golden -- kernel omits activation rounding).
//   FixedQ510  : y quantized to static signed Q(15-f).f fixed point (default Q5.10),
//                saturating clamp to int16, then INTEGER accumulate over ternary lanes;
//                dequant = acc / (2^f * scale_w). Mirrors reference QUANT_MODE="fixedp".
enum class ActQuant { Float, FixedQ510 };

// y[r,:] = rmsnorm(x[r,:]) * weight ; x,y are [rows, cols], weight is [cols].
void rmsnorm(float* y, const float* x, const float* weight, std::size_t rows,
             std::size_t cols, float eps);

// Fused BitLinear forward. x:[rows,in], weight wq:[out_dim,in] int8 ternary (nn.Linear
// layout), norm_weight:[in], out:[rows,out_dim]. If norm_out!=nullptr, the per-row
// RMSNorm result [rows,in] is written there (for tests/debug). `aq` selects the
// activation numerics; `frac_bits` is the fixed-point fractional width (Q5.10 -> 10).
//
// `be` selects the ternary-matmul backend for the FixedQ510 (integer) path: nullptr uses
// the built-in CPU reduction; a non-null backend (e.g. FPGA) is given `proj_id` to address
// its resident weights. `be`/`proj_id` are ignored by the Float path. Passing nullptr
// reproduces the historical behavior exactly.
void bitlinear(float* out, const float* x, const float* norm_weight, const int8_t* wq,
               float scale_w, std::size_t rows, std::size_t in_dim, std::size_t out_dim,
               float eps, float* norm_out = nullptr, ActQuant aq = ActQuant::Float,
               int frac_bits = 10, TernaryBackend* be = nullptr, int proj_id = -1);

// Cluster of k BitLinear projections sharing one input x and dims (in_dim,out_dim),
// each with its own norm_weight/scale_w/wq/proj_id/out buffer (i/f/g: same post-attn
// hs, distinct per-projection inner RMSNorm). FixedQ510 only. Equivalent to k bitlinear()
// calls (norm_out=nullptr) but routed through TernaryBackend::matmul_seq, letting an
// overlapping backend hide the CPU front/back-end under the engine. `be`==nullptr uses
// the built-in CPU reduction. Bit-identical to the per-projection path.
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
