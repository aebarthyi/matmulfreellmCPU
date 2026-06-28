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
//   * exp/sigmoid/silu/swiglu use a Cephes single-precision exp polynomial (range
//     reduction + 6th-order minimax) in place of libm expf. Max relative error
//     ~1e-7 (under 1 ulp), so the deviation from the libm-based scalar reference is
//     far below the e2e logit tolerance (the kernel tests check rel_l2<=1e-5 /
//     max_abs<=1e-4). The scalar tail uses the same polynomial so the vector body
//     and tail of a single call agree.
#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>

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

// Quantize y[0..n) to Q(15-f).f int16-range codes stored as int32: round y*qs to
// nearest (ties to even) then saturate to [-32768, 32767]. The SIMD convert ops
// (vcvtnq_s32_f32 / _mm256_cvtps_epi32) use round-to-nearest-ties-to-even, matching
// nearbyintf under the program's (never-changed) default rounding mode, and the
// min/max clamp matches the scalar saturation -> BIT-EXACT vs the scalar reference.
// Replaces a per-element libm nearbyintf call (the dominant FixedQ510 CPU cost).
inline void quant_q510(std::int32_t* yq, const float* y, float qs, std::size_t n) {
  std::size_t c = 0;
#if defined(MMFREE_ARCH_X86)
  const __m256 vqs = _mm256_set1_ps(qs);
  const __m256i lo = _mm256_set1_epi32(-32768), hi = _mm256_set1_epi32(32767);
  for (; c + 8 <= n; c += 8) {
    __m256i q = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(y + c), vqs));
    q = _mm256_min_epi32(_mm256_max_epi32(q, lo), hi);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(yq + c), q);
  }
#elif defined(MMFREE_ARCH_ARM)
  const float32x4_t vqs = vdupq_n_f32(qs);
  const int32x4_t lo = vdupq_n_s32(-32768), hi = vdupq_n_s32(32767);
  for (; c + 4 <= n; c += 4) {
    int32x4_t q = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(y + c), vqs));
    q = vminq_s32(vmaxq_s32(q, lo), hi);
    vst1q_s32(yq + c, q);
  }
#endif
  for (; c < n; ++c) {
    float q = std::nearbyintf(y[c] * qs);
    if (q > 32767.0f) q = 32767.0f;
    else if (q < -32768.0f) q = -32768.0f;
    yq[c] = static_cast<std::int32_t>(q);
  }
}

// out[o] = (float)acc[o] * scale.  int32->float uses round-to-nearest (both scalar
// and SIMD), single multiply per element -> BIT-EXACT vs the scalar reference.
inline void dequant_scale(float* out, const std::int32_t* acc, float scale, std::size_t n) {
  std::size_t o = 0;
#if defined(MMFREE_ARCH_X86)
  const __m256 vs = _mm256_set1_ps(scale);
  for (; o + 8 <= n; o += 8) {
    __m256 f = _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(acc + o)));
    _mm256_storeu_ps(out + o, _mm256_mul_ps(f, vs));
  }
#elif defined(MMFREE_ARCH_ARM)
  const float32x4_t vs = vdupq_n_f32(scale);
  for (; o + 4 <= n; o += 4) {
    vst1q_f32(out + o, vmulq_f32(vcvtq_f32_s32(vld1q_s32(acc + o)), vs));
  }
#endif
  for (; o < n; ++o) out[o] = static_cast<float>(acc[o]) * scale;
}

// --- exp(x) (Cephes single-precision): fx = floor(x*log2(e)+0.5); r = x - fx*ln2
// (ln2 split hi/lo for a few extra bits); exp(r) via a 6th-order minimax poly;
// 2^fx by building the float exponent field. Input is clamped to +-88.376 so the
// integer exponent never overflows. See the numerics contract at top of file. ---
inline float expf_poly(float x) {
  if (x > 88.3762626647949f) x = 88.3762626647949f;
  else if (x < -88.3762626647949f) x = -88.3762626647949f;
  float fx = std::floor(x * 1.44269504088896341f + 0.5f);
  float r = x - fx * 0.693359375f;
  r = r - fx * (-2.12194440e-4f);
  float p = 1.9875691500E-4f;
  p = p * r + 1.3981999507E-3f;
  p = p * r + 8.3334519073E-3f;
  p = p * r + 4.1665795894E-2f;
  p = p * r + 1.6666665459E-1f;
  p = p * r + 5.0000001201E-1f;
  p = p * (r * r) + r + 1.0f;
  std::int32_t e = (static_cast<std::int32_t>(fx) + 127) << 23;
  float pow2n;
  std::memcpy(&pow2n, &e, sizeof(pow2n));
  return p * pow2n;
}

#if defined(MMFREE_ARCH_X86)
inline __m256 exp8_ps(__m256 x) {
  x = _mm256_min_ps(_mm256_max_ps(x, _mm256_set1_ps(-88.3762626647949f)),
                    _mm256_set1_ps(88.3762626647949f));
  __m256 fx = _mm256_floor_ps(
      _mm256_fmadd_ps(x, _mm256_set1_ps(1.44269504088896341f), _mm256_set1_ps(0.5f)));
  __m256 r = _mm256_fnmadd_ps(fx, _mm256_set1_ps(0.693359375f), x);
  r = _mm256_fnmadd_ps(fx, _mm256_set1_ps(-2.12194440e-4f), r);
  __m256 p = _mm256_set1_ps(1.9875691500E-4f);
  p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.3981999507E-3f));
  p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(8.3334519073E-3f));
  p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(4.1665795894E-2f));
  p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.6666665459E-1f));
  p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(5.0000001201E-1f));
  p = _mm256_fmadd_ps(p, _mm256_mul_ps(r, r), _mm256_add_ps(r, _mm256_set1_ps(1.0f)));
  __m256i e = _mm256_slli_epi32(
      _mm256_add_epi32(_mm256_cvttps_epi32(fx), _mm256_set1_epi32(127)), 23);
  return _mm256_mul_ps(p, _mm256_castsi256_ps(e));
}
#elif defined(MMFREE_ARCH_ARM)
inline float32x4_t exp4_f32(float32x4_t x) {
  x = vminq_f32(vmaxq_f32(x, vdupq_n_f32(-88.3762626647949f)), vdupq_n_f32(88.3762626647949f));
  float32x4_t fx = vrndmq_f32(  // floor(x*log2(e) + 0.5)
      vmlaq_f32(vdupq_n_f32(0.5f), x, vdupq_n_f32(1.44269504088896341f)));
  float32x4_t r = vmlsq_f32(x, fx, vdupq_n_f32(0.693359375f));
  r = vmlsq_f32(r, fx, vdupq_n_f32(-2.12194440e-4f));
  float32x4_t p = vdupq_n_f32(1.9875691500E-4f);
  p = vmlaq_f32(vdupq_n_f32(1.3981999507E-3f), p, r);
  p = vmlaq_f32(vdupq_n_f32(8.3334519073E-3f), p, r);
  p = vmlaq_f32(vdupq_n_f32(4.1665795894E-2f), p, r);
  p = vmlaq_f32(vdupq_n_f32(1.6666665459E-1f), p, r);
  p = vmlaq_f32(vdupq_n_f32(5.0000001201E-1f), p, r);
  p = vmlaq_f32(vaddq_f32(r, vdupq_n_f32(1.0f)), p, vmulq_f32(r, r));
  int32x4_t e = vshlq_n_s32(vaddq_s32(vcvtq_s32_f32(fx), vdupq_n_s32(127)), 23);
  return vmulq_f32(p, vreinterpretq_f32_s32(e));
}
#endif

// out[i] = 1/(1+exp(-x[i]))  (logistic sigmoid).
inline void sigmoid(float* out, const float* x, std::size_t n) {
  std::size_t i = 0;
#if defined(MMFREE_ARCH_X86)
  const __m256 one = _mm256_set1_ps(1.0f), sign = _mm256_set1_ps(-0.0f);
  for (; i + 8 <= n; i += 8) {
    __m256 e = exp8_ps(_mm256_xor_ps(_mm256_loadu_ps(x + i), sign));
    _mm256_storeu_ps(out + i, _mm256_div_ps(one, _mm256_add_ps(one, e)));
  }
#elif defined(MMFREE_ARCH_ARM)
  const float32x4_t one = vdupq_n_f32(1.0f);
  for (; i + 4 <= n; i += 4) {
    float32x4_t e = exp4_f32(vnegq_f32(vld1q_f32(x + i)));
    vst1q_f32(out + i, vdivq_f32(one, vaddq_f32(one, e)));
  }
#endif
  for (; i < n; ++i) out[i] = 1.0f / (1.0f + expf_poly(-x[i]));
}

// out[i] = silu(x[i]) = x[i]/(1+exp(-x[i])).
inline void silu(float* out, const float* x, std::size_t n) {
  std::size_t i = 0;
#if defined(MMFREE_ARCH_X86)
  const __m256 one = _mm256_set1_ps(1.0f), sign = _mm256_set1_ps(-0.0f);
  for (; i + 8 <= n; i += 8) {
    __m256 xv = _mm256_loadu_ps(x + i);
    __m256 e = exp8_ps(_mm256_xor_ps(xv, sign));
    _mm256_storeu_ps(out + i, _mm256_div_ps(xv, _mm256_add_ps(one, e)));
  }
#elif defined(MMFREE_ARCH_ARM)
  const float32x4_t one = vdupq_n_f32(1.0f);
  for (; i + 4 <= n; i += 4) {
    float32x4_t xv = vld1q_f32(x + i);
    float32x4_t e = exp4_f32(vnegq_f32(xv));
    vst1q_f32(out + i, vdivq_f32(xv, vaddq_f32(one, e)));
  }
#endif
  for (; i < n; ++i) out[i] = x[i] / (1.0f + expf_poly(-x[i]));
}

// out[i] = silu(a[i]) * b[i].
inline void swiglu(float* out, const float* a, const float* b, std::size_t n) {
  std::size_t i = 0;
#if defined(MMFREE_ARCH_X86)
  const __m256 one = _mm256_set1_ps(1.0f), sign = _mm256_set1_ps(-0.0f);
  for (; i + 8 <= n; i += 8) {
    __m256 av = _mm256_loadu_ps(a + i);
    __m256 e = exp8_ps(_mm256_xor_ps(av, sign));
    __m256 s = _mm256_div_ps(av, _mm256_add_ps(one, e));
    _mm256_storeu_ps(out + i, _mm256_mul_ps(s, _mm256_loadu_ps(b + i)));
  }
#elif defined(MMFREE_ARCH_ARM)
  const float32x4_t one = vdupq_n_f32(1.0f);
  for (; i + 4 <= n; i += 4) {
    float32x4_t av = vld1q_f32(a + i);
    float32x4_t e = exp4_f32(vnegq_f32(av));
    float32x4_t s = vdivq_f32(av, vaddq_f32(one, e));
    vst1q_f32(out + i, vmulq_f32(s, vld1q_f32(b + i)));
  }
#endif
  for (; i < n; ++i) out[i] = (a[i] / (1.0f + expf_poly(-a[i]))) * b[i];
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
