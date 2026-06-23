// mmfree/simd.hpp — architecture-specific SIMD primitives for the hot kernels.
//
// The backend is selected at compile time by a define passed from CMake:
//   MMFREE_ARCH_X86    -> AVX2 + FMA   (x86_64)
//   MMFREE_ARCH_ARM    -> NEON         (AArch64; widening only, no SDOT/i8mm -- A53 is
//                                       ARMv8.0)
//   (neither)          -> portable scalar fallback
//
// Numerics contract (see kernels.hpp / golden-numerics):
//   * ternary_dot_i32 (FixedQ510 path) is INTEGER accumulation -> associative ->
//     BIT-EXACT vs the scalar reference regardless of lane order.
//   * ternary_dot_f32 / sumsq (fp32) only reorder the summation tree; each ternary
//     product y*(+-1/0) is itself exact, so the deviation from scalar is pure
//     reduction-order noise (~1e-6), well under the e2e logit tolerance.
//   * hgrn_step is elementwise (no reduction) -> bit-exact per lane.
#pragma once

#include <cstddef>
#include <cstdint>

#if defined(MMFREE_ARCH_X86)
#include <immintrin.h>
#elif defined(MMFREE_ARCH_ARM)
#include <arm_neon.h>
#endif

namespace mmfree {
namespace simd {

#if defined(MMFREE_ARCH_X86)
inline const char* backend() { return "x86-avx2"; }
#elif defined(MMFREE_ARCH_ARM)
inline const char* backend() { return "arm-neon"; }
#else
inline const char* backend() { return "scalar"; }
#endif

#if defined(MMFREE_ARCH_X86)
// Horizontal sums for AVX2 registers.
inline float hsum_ps(__m256 v) {
  __m128 lo = _mm256_castps256_ps128(v);
  __m128 hi = _mm256_extractf128_ps(v, 1);
  lo = _mm_add_ps(lo, hi);
  __m128 sh = _mm_movehdup_ps(lo);
  __m128 s = _mm_add_ps(lo, sh);
  sh = _mm_movehl_ps(sh, s);
  s = _mm_add_ss(s, sh);
  return _mm_cvtss_f32(s);
}
inline std::int32_t hsum_epi32(__m256i v) {
  __m128i lo = _mm256_castsi256_si128(v);
  __m128i hi = _mm256_extracti128_si256(v, 1);
  __m128i s = _mm_add_epi32(lo, hi);
  s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(1, 0, 3, 2)));
  s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(2, 3, 0, 1)));
  return _mm_cvtsi128_si32(s);
}
#endif

// sum_c x[c]^2  (fp32).
inline float sumsq(const float* x, std::size_t n) {
  std::size_t c = 0;
  float s = 0.0f;
#if defined(MMFREE_ARCH_X86)
  __m256 acc = _mm256_setzero_ps();
  for (; c + 8 <= n; c += 8) {
    __m256 v = _mm256_loadu_ps(x + c);
    acc = _mm256_fmadd_ps(v, v, acc);
  }
  s = hsum_ps(acc);
#elif defined(MMFREE_ARCH_ARM)
  float32x4_t acc = vdupq_n_f32(0.0f);
  for (; c + 4 <= n; c += 4) {
    float32x4_t v = vld1q_f32(x + c);
    acc = vmlaq_f32(acc, v, v);
  }
  s = vaddvq_f32(acc);
#endif
  for (; c < n; ++c) s += x[c] * x[c];
  return s;
}

// sum_k w[k]*y[k]  with w in {-1,0,+1} (int8).  fp32 accumulate.
inline float ternary_dot_f32(const float* y, const std::int8_t* w, std::size_t n) {
  std::size_t k = 0;
  float s = 0.0f;
#if defined(MMFREE_ARCH_X86)
  __m256 acc = _mm256_setzero_ps();
  for (; k + 8 <= n; k += 8) {
    __m256 yv = _mm256_loadu_ps(y + k);
    __m128i w8 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(w + k));  // 8x int8
    __m256 wf = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(w8));               // -> fp32 +-1/0
    acc = _mm256_fmadd_ps(yv, wf, acc);  // y*(+-1/0) is exact; only the sum reorders
  }
  s = hsum_ps(acc);
#elif defined(MMFREE_ARCH_ARM)
  float32x4_t acc0 = vdupq_n_f32(0.0f), acc1 = vdupq_n_f32(0.0f);
  for (; k + 8 <= n; k += 8) {
    int8x8_t w8 = vld1_s8(w + k);
    int16x8_t w16 = vmovl_s8(w8);
    float32x4_t wf0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(w16)));
    float32x4_t wf1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(w16)));
    acc0 = vmlaq_f32(acc0, vld1q_f32(y + k), wf0);
    acc1 = vmlaq_f32(acc1, vld1q_f32(y + k + 4), wf1);
  }
  s = vaddvq_f32(vaddq_f32(acc0, acc1));
#endif
  for (; k < n; ++k) {
    std::int8_t wk = w[k];
    if (wk > 0) s += y[k];
    else if (wk < 0) s -= y[k];
  }
  return s;
}

// sum_k w[k]*yq[k]  with w in {-1,0,+1} (int8).  int32 accumulate (bit-exact, assoc.).
inline std::int32_t ternary_dot_i32(const std::int32_t* yq, const std::int8_t* w,
                                    std::size_t n) {
  std::size_t k = 0;
  std::int32_t s = 0;
#if defined(MMFREE_ARCH_X86)
  __m256i acc = _mm256_setzero_si256();
  for (; k + 8 <= n; k += 8) {
    __m256i yv = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(yq + k));
    __m128i w8 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(w + k));
    __m256i wi = _mm256_cvtepi8_epi32(w8);
    acc = _mm256_add_epi32(acc, _mm256_mullo_epi32(yv, wi));
  }
  s = hsum_epi32(acc);
#elif defined(MMFREE_ARCH_ARM)
  int32x4_t acc0 = vdupq_n_s32(0), acc1 = vdupq_n_s32(0);
  for (; k + 8 <= n; k += 8) {
    int8x8_t w8 = vld1_s8(w + k);
    int16x8_t w16 = vmovl_s8(w8);
    acc0 = vmlaq_s32(acc0, vld1q_s32(yq + k), vmovl_s16(vget_low_s16(w16)));
    acc1 = vmlaq_s32(acc1, vld1q_s32(yq + k + 4), vmovl_s16(vget_high_s16(w16)));
  }
  s = vaddvq_s32(vaddq_s32(acc0, acc1));
#endif
  for (; k < n; ++k) {
    std::int8_t wk = w[k];
    if (wk > 0) s += yq[k];
    else if (wk < 0) s -= yq[k];
  }
  return s;
}

// One HGRN time step over D lanes: state = f*state + i ; out = state.
// Elementwise (no cross-lane reduction) -> bit-exact per lane. Mul-then-add (two
// roundings) to match the scalar reference rather than a fused multiply-add.
inline void hgrn_step(float* state, float* out, const float* f, const float* in,
                      std::size_t D) {
  std::size_t d = 0;
#if defined(MMFREE_ARCH_X86)
  for (; d + 8 <= D; d += 8) {
    __m256 s = _mm256_add_ps(_mm256_mul_ps(_mm256_loadu_ps(f + d), _mm256_loadu_ps(state + d)),
                             _mm256_loadu_ps(in + d));
    _mm256_storeu_ps(state + d, s);
    _mm256_storeu_ps(out + d, s);
  }
#elif defined(MMFREE_ARCH_ARM)
  for (; d + 4 <= D; d += 4) {
    float32x4_t s = vaddq_f32(vmulq_f32(vld1q_f32(f + d), vld1q_f32(state + d)),
                              vld1q_f32(in + d));
    vst1q_f32(state + d, s);
    vst1q_f32(out + d, s);
  }
#endif
  for (; d < D; ++d) {
    state[d] = f[d] * state[d] + in[d];
    out[d] = state[d];
  }
}

}  // namespace simd
}  // namespace mmfree
