// mmfree/tq1.hpp — packed ternary weights for the CPU backend: TQ1_0 (1.625 bits/weight)
// and 2-bit (tq2 below), plus PackedBackend, a TernaryBackend that repacks the blob's int8.
//
// TQ1_0 is llama.cpp's TQ1_0 geometry minus its per-block fp16 scale (our scale_w is per
// projection). 256 weights per block, base-3 packed (3^5 = 243 < 256):
//   qs[0..31]   5 trits each -> elements  j + 32*l       (j<32, l<5) = 160 elements
//   qs[32..47]  5 trits each -> elements  160 + j + 16*l (j<16, l<5) =  80 elements
//   qh[0..3]    4 trits each -> elements  240 + j + 4*l  (j<4,  l<4) =  16 elements
//   -> 52 bytes per 256 weights
// The l-th trit of a contiguous byte run maps to a contiguous element run, so one vector
// load of weights pairs with one vector load of activations.
//
// Encode (u = trit + 1, u_0 first):  v = u_0*81 + u_1*27 + u_2*9 + u_3*3 + u_4  in [0, 242]
//                                    q = ceil(v * 256 / 243) = (v*256 + 242) / 243
// Decode trit l:  q_l = (uint8)(q * 3^l)   -- the mod-256 wrap drops the leading digits
//                 u_l = (q_l >= 86) + (q_l >= 171)  == llama.cpp's ((uint16)q_l * 3) >> 8
// cpp/tests/test_tq1.cpp checks the encoding and that equivalence over the whole domain.
//
// Lossless, hence bit-exact: dot_i32 == simd::ternary_dot_i32 on the same weights (integer
// addition is associative, so summation order cannot matter). dot_f32 differs from the
// scalar reference only by fp reduction order.
#pragma once

#include "mmfree/parallel.hpp"
#include "mmfree/simd.hpp"
#include "mmfree/ternary_backend.hpp"

#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

#if defined(MMFREE_ARCH_X86)
#include <immintrin.h>
// AVX-512 paths only when the compiler targets F+BW+VL (MMFREE_SIMD=X86_512); an X86 build
// stays AVX2. TQ1_0's 16-bit extraction needs BW and VL; tq2 alone needs only F.
#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VL__)
#define MMFREE_AVX512 1
#endif
#elif defined(MMFREE_ARCH_ARM)
#include <arm_neon.h>
#endif

namespace mmfree {
namespace tq1 {

constexpr std::size_t kBlockElems = 256;  // weights per block
constexpr std::size_t kQsBytes    = 48;   // 5 trits each: 32 bytes then 16 bytes
constexpr std::size_t kQhBytes    = 4;    // 4 trits each
constexpr std::size_t kBlockBytes = kQsBytes + kQhBytes;  // 52

// Bytes needed for n weights. n MUST be a multiple of kBlockElems (see fits()).
inline std::size_t row_bytes(std::size_t n) { return (n / kBlockElems) * kBlockBytes; }
inline bool fits(std::size_t n) { return n != 0 && n % kBlockElems == 0; }

inline const std::uint8_t* pow3() {
  static const std::uint8_t p[5] = {1, 3, 9, 27, 81};
  return p;
}

// ---- encode ---------------------------------------------------------------------------
// Pack n ternary int8 weights ({-1,0,+1}) into row_bytes(n) bytes.
inline void pack_row(std::uint8_t* dst, const std::int8_t* w, std::size_t n) {
  const std::size_t nb = n / kBlockElems;
  for (std::size_t b = 0; b < nb; ++b) {
    std::uint8_t* qs = dst + b * kBlockBytes;
    std::uint8_t* qh = qs + kQsBytes;
    const std::int8_t* wb = w + b * kBlockElems;
    // group A: 32 bytes x 5 trits, element j + 32*l
    for (std::size_t j = 0; j < 32; ++j) {
      unsigned v = 0;
      for (std::size_t l = 0; l < 5; ++l) v = v * 3 + (unsigned)(wb[j + 32 * l] + 1);
      qs[j] = (std::uint8_t)((v * 256u + 242u) / 243u);
    }
    // group B: 16 bytes x 5 trits, element 160 + j + 16*l
    for (std::size_t j = 0; j < 16; ++j) {
      unsigned v = 0;
      for (std::size_t l = 0; l < 5; ++l) v = v * 3 + (unsigned)(wb[160 + j + 16 * l] + 1);
      qs[32 + j] = (std::uint8_t)((v * 256u + 242u) / 243u);
    }
    // group C: 4 bytes x 4 trits, element 240 + j + 4*l
    for (std::size_t j = 0; j < 4; ++j) {
      unsigned v = 0;
      for (std::size_t l = 0; l < 4; ++l) v = v * 3 + (unsigned)(wb[240 + j + 4 * l] + 1);
      qh[j] = (std::uint8_t)((v * 256u + 80u) / 81u);
    }
  }
}

// ---- decode (reference; for tests and the scalar tail) --------------------------------
inline int trit_at(std::uint8_t q, std::size_t l) {
  const std::uint8_t ql = (std::uint8_t)(q * pow3()[l]);
  return (int)(ql >= 86) + (int)(ql >= 171) - 1;
}

inline void unpack_row(std::int8_t* w, const std::uint8_t* src, std::size_t n) {
  const std::size_t nb = n / kBlockElems;
  for (std::size_t b = 0; b < nb; ++b) {
    const std::uint8_t* qs = src + b * kBlockBytes;
    const std::uint8_t* qh = qs + kQsBytes;
    std::int8_t* wb = w + b * kBlockElems;
    for (std::size_t l = 0; l < 5; ++l)
      for (std::size_t j = 0; j < 32; ++j) wb[j + 32 * l] = (std::int8_t)trit_at(qs[j], l);
    for (std::size_t l = 0; l < 5; ++l)
      for (std::size_t j = 0; j < 16; ++j)
        wb[160 + j + 16 * l] = (std::int8_t)trit_at(qs[32 + j], l);
    for (std::size_t l = 0; l < 4; ++l)
      for (std::size_t j = 0; j < 4; ++j) wb[240 + j + 4 * l] = (std::int8_t)trit_at(qh[j], l);
  }
}

// ---- dot products ---------------------------------------------------------------------
// Scalar reference for both element types, shared by the tails and the SCALAR build.
template <class T, class Acc>
inline Acc dot_scalar(const T* x, const std::uint8_t* src, std::size_t n) {
  Acc s = (Acc)0;
  const std::size_t nb = n / kBlockElems;
  for (std::size_t b = 0; b < nb; ++b) {
    const std::uint8_t* qs = src + b * kBlockBytes;
    const std::uint8_t* qh = qs + kQsBytes;
    const T* xb = x + b * kBlockElems;
    for (std::size_t l = 0; l < 5; ++l)
      for (std::size_t j = 0; j < 32; ++j) s += (Acc)trit_at(qs[j], l) * (Acc)xb[j + 32 * l];
    for (std::size_t l = 0; l < 5; ++l)
      for (std::size_t j = 0; j < 16; ++j)
        s += (Acc)trit_at(qs[32 + j], l) * (Acc)xb[160 + j + 16 * l];
    for (std::size_t l = 0; l < 4; ++l)
      for (std::size_t j = 0; j < 4; ++j) s += (Acc)trit_at(qh[j], l) * (Acc)xb[240 + j + 4 * l];
  }
  return s;
}

#if defined(MMFREE_ARCH_ARM)
// 16 packed bytes -> the l-th trit of each, as int8 in {-1,0,+1}.
// vmulq_u8 wraps mod 256, which is precisely the "shift base-3 digits off the top" step.
inline int8x16_t trits16(uint8x16_t q, std::uint8_t p3) {
  const uint8x16_t a = vmulq_u8(q, vdupq_n_u8(p3));
  const uint8x16_t pos = vcgeq_u8(a, vdupq_n_u8(171));  // trit = +1
  const uint8x16_t neg = vcltq_u8(a, vdupq_n_u8(86));   // trit = -1
  // 0xFF -> 1 via a logical shift, then difference: +1 / 0 / -1 with no multiply.
  return vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(pos, 7)),
                  vreinterpretq_s8_u8(vshrq_n_u8(neg, 7)));
}

// Accumulate 16 int32 activations against 16 trits (sign-extend int8 -> int32, then mla).
inline void mac16_i32(int32x4_t& a0, int32x4_t& a1, int32x4_t& a2, int32x4_t& a3,
                      const std::int32_t* x, int8x16_t t) {
  const int16x8_t lo = vmovl_s8(vget_low_s8(t)), hi = vmovl_s8(vget_high_s8(t));
  a0 = vmlaq_s32(a0, vld1q_s32(x + 0), vmovl_s16(vget_low_s16(lo)));
  a1 = vmlaq_s32(a1, vld1q_s32(x + 4), vmovl_s16(vget_high_s16(lo)));
  a2 = vmlaq_s32(a2, vld1q_s32(x + 8), vmovl_s16(vget_low_s16(hi)));
  a3 = vmlaq_s32(a3, vld1q_s32(x + 12), vmovl_s16(vget_high_s16(hi)));
}

inline void mac16_f32(float32x4_t& a0, float32x4_t& a1, float32x4_t& a2, float32x4_t& a3,
                      const float* x, int8x16_t t) {
  const int16x8_t lo = vmovl_s8(vget_low_s8(t)), hi = vmovl_s8(vget_high_s8(t));
  a0 = vmlaq_f32(a0, vld1q_f32(x + 0), vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo))));
  a1 = vmlaq_f32(a1, vld1q_f32(x + 4), vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo))));
  a2 = vmlaq_f32(a2, vld1q_f32(x + 8), vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi))));
  a3 = vmlaq_f32(a3, vld1q_f32(x + 12), vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi))));
}
#endif  // MMFREE_ARCH_ARM

#if defined(MMFREE_ARCH_X86)
// AVX2 has no 8-bit multiply, so the mod-256 step runs in 16-bit lanes and is masked back
// to 8 bits by hand. 16 packed bytes -> 16 trits as int16 in {-1,0,+1}.
inline __m256i trits16_epi16(const std::uint8_t* q, std::uint8_t p3) {
  const __m256i q16 = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i*)q));
  const __m256i a = _mm256_and_si256(_mm256_mullo_epi16(q16, _mm256_set1_epi16(p3)),
                                     _mm256_set1_epi16(0xFF));  // (q * 3^l) mod 256
  // (a >= 171) + (a >= 86) - 1, via cmpgt on a-1 (all values are non-negative < 256)
  const __m256i pos = _mm256_cmpgt_epi16(a, _mm256_set1_epi16(170));  // -1 where true
  const __m256i neg = _mm256_cmpgt_epi16(_mm256_set1_epi16(86), a);   // -1 where true
  return _mm256_sub_epi16(neg, pos);  // masks are 0/-1, so neg-pos -> -1 / 0 / +1
}

#endif  // MMFREE_ARCH_X86

// Integer dot: BIT-IDENTICAL to simd::ternary_dot_i32 on the same logical weights.
inline std::int32_t dot_i32(const std::int32_t* x, const std::uint8_t* src, std::size_t n) {
#if defined(MMFREE_ARCH_ARM)
  int32x4_t a0 = vdupq_n_s32(0), a1 = vdupq_n_s32(0), a2 = vdupq_n_s32(0), a3 = vdupq_n_s32(0);
  const std::size_t nb = n / kBlockElems;
  for (std::size_t b = 0; b < nb; ++b) {
    const std::uint8_t* qs = src + b * kBlockBytes;
    const std::int32_t* xb = x + b * kBlockElems;
    const uint8x16_t qa0 = vld1q_u8(qs), qa1 = vld1q_u8(qs + 16), qb = vld1q_u8(qs + 32);
    for (std::size_t l = 0; l < 5; ++l) {
      const std::uint8_t p = pow3()[l];
      mac16_i32(a0, a1, a2, a3, xb + 32 * l, trits16(qa0, p));       // elements 32l+0..15
      mac16_i32(a0, a1, a2, a3, xb + 32 * l + 16, trits16(qa1, p));  // elements 32l+16..31
      mac16_i32(a0, a1, a2, a3, xb + 160 + 16 * l, trits16(qb, p));  // elements 160+16l..+15
    }
    // group C is 4 bytes x 4 trits = 16 elements; too narrow to vectorize usefully.
    const std::uint8_t* qh = qs + kQsBytes;
    std::int32_t t = 0;
    for (std::size_t l = 0; l < 4; ++l)
      for (std::size_t j = 0; j < 4; ++j) t += trit_at(qh[j], l) * xb[240 + j + 4 * l];
    a3 = vsetq_lane_s32(vgetq_lane_s32(a3, 0) + t, a3, 0);
  }
  return vaddvq_s32(vaddq_s32(vaddq_s32(a0, a1), vaddq_s32(a2, a3)));
#elif defined(MMFREE_ARCH_X86) && defined(MMFREE_AVX512)
  // No multiply: a ternary product is a conditional add, so the compare masks drive
  // mask_add / mask_sub on the activations directly (no widening, no vpmulld). Four
  // accumulators keep the adds off one serial dependency chain.
  __m512i a0 = _mm512_setzero_si512(), a1 = _mm512_setzero_si512();
  __m512i a2 = _mm512_setzero_si512(), a3 = _mm512_setzero_si512();
  std::int32_t tail = 0;
  const std::size_t nb = n / kBlockElems;
  const __m512i k255 = _mm512_set1_epi16(0xFF), k170 = _mm512_set1_epi16(170),
                k86 = _mm512_set1_epi16(86);
  const __m256i h255 = _mm256_set1_epi16(0xFF), h170 = _mm256_set1_epi16(170),
                h86 = _mm256_set1_epi16(86);
  for (std::size_t b = 0; b < nb; ++b) {
    const std::uint8_t* qs = src + b * kBlockBytes;
    const std::int32_t* xb = x + b * kBlockElems;
    // group A: 32 bytes, 5 trits each -> elements 32l + 0..31 (contiguous)
    const __m512i qA = _mm512_cvtepu8_epi16(_mm256_loadu_si256((const __m256i*)qs));
    // group B: 16 bytes, 5 trits each -> elements 160 + 16l + 0..15
    const __m256i qB = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i*)(qs + 32)));
    for (std::size_t l = 0; l < 5; ++l) {
      const short p = (short)pow3()[l];
      {  // 32 trits: the mask halves index the two 16-lane activation groups
        const __m512i a = _mm512_and_si512(_mm512_mullo_epi16(qA, _mm512_set1_epi16(p)), k255);
        const __mmask32 pos = _mm512_cmpgt_epi16_mask(a, k170);
        const __mmask32 neg = _mm512_cmpgt_epi16_mask(k86, a);
        const __m512i xlo = _mm512_loadu_si512((const void*)(xb + 32 * l));
        const __m512i xhi = _mm512_loadu_si512((const void*)(xb + 32 * l + 16));
        a0 = _mm512_mask_add_epi32(a0, (__mmask16)pos, a0, xlo);
        a1 = _mm512_mask_sub_epi32(a1, (__mmask16)neg, a1, xlo);
        a2 = _mm512_mask_add_epi32(a2, (__mmask16)(pos >> 16), a2, xhi);
        a3 = _mm512_mask_sub_epi32(a3, (__mmask16)(neg >> 16), a3, xhi);
      }
      {  // 16 trits
        const __m256i a = _mm256_and_si256(_mm256_mullo_epi16(qB, _mm256_set1_epi16(p)), h255);
        const __mmask16 pos = _mm256_cmpgt_epi16_mask(a, h170);
        const __mmask16 neg = _mm256_cmpgt_epi16_mask(h86, a);
        const __m512i xv = _mm512_loadu_si512((const void*)(xb + 160 + 16 * l));
        a2 = _mm512_mask_add_epi32(a2, pos, a2, xv);
        a3 = _mm512_mask_sub_epi32(a3, neg, a3, xv);
      }
    }
    // group C (16 elements) is too narrow to vectorize; a separate scalar sum is still exact.
    const std::uint8_t* qh = qs + kQsBytes;
    for (std::size_t l = 0; l < 4; ++l)
      for (std::size_t j = 0; j < 4; ++j) tail += trit_at(qh[j], l) * xb[240 + j + 4 * l];
  }
  return _mm512_reduce_add_epi32(_mm512_add_epi32(_mm512_add_epi32(a0, a1),
                                                  _mm512_add_epi32(a2, a3))) + tail;
#elif defined(MMFREE_ARCH_X86)
  __m256i acc = _mm256_setzero_si256();
  const std::size_t nb = n / kBlockElems;
  for (std::size_t b = 0; b < nb; ++b) {
    const std::uint8_t* qs = src + b * kBlockBytes;
    const std::int32_t* xb = x + b * kBlockElems;
    for (std::size_t l = 0; l < 5; ++l) {
      const std::uint8_t p = pow3()[l];
      const struct { const std::uint8_t* q; const std::int32_t* x; } grp[3] = {
          {qs, xb + 32 * l}, {qs + 16, xb + 32 * l + 16}, {qs + 32, xb + 160 + 16 * l}};
      for (int g = 0; g < 3; ++g) {
        const __m256i t16 = trits16_epi16(grp[g].q, p);
        const __m256i tlo = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(t16));
        const __m256i thi = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(t16, 1));
        acc = _mm256_add_epi32(
            acc, _mm256_mullo_epi32(tlo, _mm256_loadu_si256((const __m256i*)(grp[g].x))));
        acc = _mm256_add_epi32(
            acc, _mm256_mullo_epi32(thi, _mm256_loadu_si256((const __m256i*)(grp[g].x + 8))));
      }
    }
    const std::uint8_t* qh = qs + kQsBytes;
    std::int32_t t = 0;
    for (std::size_t l = 0; l < 4; ++l)
      for (std::size_t j = 0; j < 4; ++j) t += trit_at(qh[j], l) * xb[240 + j + 4 * l];
    acc = _mm256_add_epi32(acc, _mm256_set_epi32(0, 0, 0, 0, 0, 0, 0, t));
  }
  return simd::hsum_epi32(acc);
#else
  return dot_scalar<std::int32_t, std::int32_t>(x, src, n);
#endif
}

// Float dot: reorders the fp reduction exactly as simd::ternary_dot_f32 does (each ternary
// product is itself exact), so it matches the scalar reference to reduction-order noise.
inline float dot_f32(const float* x, const std::uint8_t* src, std::size_t n) {
#if defined(MMFREE_ARCH_ARM)
  float32x4_t a0 = vdupq_n_f32(0), a1 = vdupq_n_f32(0), a2 = vdupq_n_f32(0), a3 = vdupq_n_f32(0);
  const std::size_t nb = n / kBlockElems;
  for (std::size_t b = 0; b < nb; ++b) {
    const std::uint8_t* qs = src + b * kBlockBytes;
    const float* xb = x + b * kBlockElems;
    const uint8x16_t qa0 = vld1q_u8(qs), qa1 = vld1q_u8(qs + 16), qb = vld1q_u8(qs + 32);
    for (std::size_t l = 0; l < 5; ++l) {
      const std::uint8_t p = pow3()[l];
      mac16_f32(a0, a1, a2, a3, xb + 32 * l, trits16(qa0, p));
      mac16_f32(a0, a1, a2, a3, xb + 32 * l + 16, trits16(qa1, p));
      mac16_f32(a0, a1, a2, a3, xb + 160 + 16 * l, trits16(qb, p));
    }
    const std::uint8_t* qh = qs + kQsBytes;
    float t = 0.0f;
    for (std::size_t l = 0; l < 4; ++l)
      for (std::size_t j = 0; j < 4; ++j)
        t += (float)trit_at(qh[j], l) * xb[240 + j + 4 * l];
    a0 = vsetq_lane_f32(vgetq_lane_f32(a0, 0) + t, a0, 0);
  }
  return vaddvq_f32(vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3)));
#else
  return dot_scalar<float, float>(x, src, n);
#endif
}

}  // namespace tq1

// =========================================================================================
// tq2 — 2-bit format (the CPU counterpart of the FPGA's 2-bit build). 4 trits per byte,
// 256 weights -> 64 bytes, exactly 2.0 bits/weight. Byte j holds elements j + 64*l
// (l in [0,4)) as u = trit + 1 in bits [2l, 2l+2), so extraction is shift/mask only:
// t = ((q >> 2l) & 3) - 1. CPU-internal; not the engine's 0=hold/1=add/3=sub wire encoding.
namespace tq2 {

constexpr std::size_t kBlockElems = 256;
constexpr std::size_t kBlockBytes = 64;  // 4 trits/byte, exactly 2.0 bits/weight

inline std::size_t row_bytes(std::size_t n) { return (n / kBlockElems) * kBlockBytes; }
inline bool fits(std::size_t n) { return n != 0 && n % kBlockElems == 0; }

inline int trit_at(std::uint8_t q, std::size_t l) {
  return (int)((q >> (2 * l)) & 0x3) - 1;
}

inline void pack_row(std::uint8_t* dst, const std::int8_t* w, std::size_t n) {
  const std::size_t nb = n / kBlockElems;
  for (std::size_t b = 0; b < nb; ++b) {
    std::uint8_t* q = dst + b * kBlockBytes;
    const std::int8_t* wb = w + b * kBlockElems;
    for (std::size_t j = 0; j < 64; ++j) {
      unsigned v = 0;
      for (std::size_t l = 0; l < 4; ++l)
        v |= (unsigned)(wb[j + 64 * l] + 1) << (2 * l);
      q[j] = (std::uint8_t)v;
    }
  }
}

inline void unpack_row(std::int8_t* w, const std::uint8_t* src, std::size_t n) {
  const std::size_t nb = n / kBlockElems;
  for (std::size_t b = 0; b < nb; ++b) {
    const std::uint8_t* q = src + b * kBlockBytes;
    std::int8_t* wb = w + b * kBlockElems;
    for (std::size_t l = 0; l < 4; ++l)
      for (std::size_t j = 0; j < 64; ++j) wb[j + 64 * l] = (std::int8_t)trit_at(q[j], l);
  }
}

template <class T, class Acc>
inline Acc dot_scalar(const T* x, const std::uint8_t* src, std::size_t n) {
  Acc s = (Acc)0;
  const std::size_t nb = n / kBlockElems;
  for (std::size_t b = 0; b < nb; ++b) {
    const std::uint8_t* q = src + b * kBlockBytes;
    const T* xb = x + b * kBlockElems;
    for (std::size_t l = 0; l < 4; ++l)
      for (std::size_t j = 0; j < 64; ++j) s += (Acc)trit_at(q[j], l) * (Acc)xb[j + 64 * l];
  }
  return s;
}

#if defined(MMFREE_ARCH_ARM)
// The NEON shift must be an immediate, so SH is a template argument. SH=0 must never reach
// vshrq_n_u8: its ACLE range is [1,8] — GCC folds a 0 shift, clang rejects it. `if constexpr`
// (not a plain `if`) keeps that call from being instantiated at all.
template <int SH>
inline int8x16_t tq2_trits(uint8x16_t qv) {
  static_assert(SH >= 0 && SH <= 6 && SH % 2 == 0, "trit level shift is 0/2/4/6");
  uint8x16_t s;
  if constexpr (SH == 0) {
    s = qv;
  } else {
    s = vshrq_n_u8(qv, SH);
  }
  return vsubq_s8(vreinterpretq_s8_u8(vandq_u8(s, vdupq_n_u8(0x3))), vdupq_n_s8(1));
}
#define MMFREE_TQ2_TRITS(qv, SH) ::mmfree::tq2::tq2_trits<(SH)>((qv))
#endif

#if defined(MMFREE_ARCH_X86)
// x86 counterpart of MMFREE_TQ2_TRITS, on bytes already widened to int16 lanes. A macro
// because the shift count must be an immediate.
#define MMFREE_TQ2_TRITS_X86(q16, SH)                                              \
  _mm256_sub_epi16(                                                                \
      _mm256_and_si256(_mm256_srli_epi16((q16), (SH)), _mm256_set1_epi16(0x3)),     \
      _mm256_set1_epi16(1))
#endif

// Inner dim below which the x86 intrinsic paths lose to GCC's auto-vectorized dot_scalar:
// their fixed per-output-column cost (zero accumulators, horizontal reduce) is amortized
// over only n/256 blocks. The measured crossover lies between n=1024 and n=2048, so this is
// a bound, not a fitted value. TQ1_0 showed no such effect and does not dispatch on it.
constexpr std::size_t kX86ScalarBelowN = 2048;

inline std::int32_t dot_i32(const std::int32_t* x, const std::uint8_t* src, std::size_t n) {
#if defined(MMFREE_ARCH_X86)
  if (n < kX86ScalarBelowN) return dot_scalar<std::int32_t, std::int32_t>(x, src, n);
#endif
#if defined(MMFREE_ARCH_ARM)
  int32x4_t a0 = vdupq_n_s32(0), a1 = vdupq_n_s32(0), a2 = vdupq_n_s32(0), a3 = vdupq_n_s32(0);
  const std::size_t nb = n / kBlockElems;
  for (std::size_t b = 0; b < nb; ++b) {
    const std::uint8_t* q = src + b * kBlockBytes;
    const std::int32_t* xb = x + b * kBlockElems;
    for (std::size_t c = 0; c < 64; c += 16) {
      const uint8x16_t qv = vld1q_u8(q + c);
      tq1::mac16_i32(a0, a1, a2, a3, xb + 0 + c, MMFREE_TQ2_TRITS(qv, 0));
      tq1::mac16_i32(a0, a1, a2, a3, xb + 64 + c, MMFREE_TQ2_TRITS(qv, 2));
      tq1::mac16_i32(a0, a1, a2, a3, xb + 128 + c, MMFREE_TQ2_TRITS(qv, 4));
      tq1::mac16_i32(a0, a1, a2, a3, xb + 192 + c, MMFREE_TQ2_TRITS(qv, 6));
    }
  }
  return vaddvq_s32(vaddq_s32(vaddq_s32(a0, a1), vaddq_s32(a2, a3)));
#elif defined(MMFREE_ARCH_X86) && defined(MMFREE_AVX512)
  // Extract in 32-bit lanes so each compare yields a __mmask16 matching a 16-wide activation
  // load: v==2 (trit +1) -> mask_add, v==0 (trit -1) -> mask_sub. AVX512F only, no multiply;
  // four accumulators keep the adds off one dependency chain.
  __m512i aP0 = _mm512_setzero_si512(), aN0 = _mm512_setzero_si512();
  __m512i aP1 = _mm512_setzero_si512(), aN1 = _mm512_setzero_si512();
  const std::size_t nb = n / kBlockElems;
  const __m512i k3 = _mm512_set1_epi32(3), k2 = _mm512_set1_epi32(2), kz = _mm512_setzero_si512();
  for (std::size_t b = 0; b < nb; ++b) {
    const std::uint8_t* q = src + b * kBlockBytes;
    const std::int32_t* xb = x + b * kBlockElems;
    for (std::size_t c = 0; c < 64; c += 16) {
      // 16 bytes -> 16 int32 lanes, reused across all four levels.
      const __m512i q32 = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*)(q + c)));
      // Unrolled: the shift must be an immediate, and a runtime-indexed vector array spills.
#define MMFREE_TQ2_MAC512(SH, L, AP, AN)                                                  \
  do {                                                                                    \
    const __m512i v_ = _mm512_and_si512(_mm512_srli_epi32(q32, SH), k3);                   \
    const __mmask16 pos_ = _mm512_cmpeq_epi32_mask(v_, k2);                                \
    const __mmask16 neg_ = _mm512_cmpeq_epi32_mask(v_, kz);                                \
    const __m512i xv_ = _mm512_loadu_si512((const void*)(xb + 64 * (L) + c));               \
    AP = _mm512_mask_add_epi32(AP, pos_, AP, xv_);                                          \
    AN = _mm512_mask_sub_epi32(AN, neg_, AN, xv_);                                          \
  } while (0)
      MMFREE_TQ2_MAC512(0, 0, aP0, aN0);
      MMFREE_TQ2_MAC512(2, 1, aP1, aN1);
      MMFREE_TQ2_MAC512(4, 2, aP0, aN0);
      MMFREE_TQ2_MAC512(6, 3, aP1, aN1);
#undef MMFREE_TQ2_MAC512
    }
  }
  return _mm512_reduce_add_epi32(_mm512_add_epi32(_mm512_add_epi32(aP0, aN0),
                                                  _mm512_add_epi32(aP1, aN1)));
#elif defined(MMFREE_ARCH_X86)
  // AVX2: _mm256_sign_epi32(x, t) is multiplication by a trit in one cheap uop (vs vpmulld).
  // Four accumulators keep the adds off one dependency chain.
  __m256i acc0 = _mm256_setzero_si256(), acc1 = _mm256_setzero_si256();
  __m256i acc2 = _mm256_setzero_si256(), acc3 = _mm256_setzero_si256();
  const std::size_t nb = n / kBlockElems;
  for (std::size_t b = 0; b < nb; ++b) {
    const std::uint8_t* q = src + b * kBlockBytes;
    const std::int32_t* xb = x + b * kBlockElems;
    for (std::size_t c = 0; c < 64; c += 16) {
      const __m256i q16 = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i*)(q + c)));
      // Unrolled: the shift must be an immediate, and a runtime-indexed vector array spills.
#define MMFREE_TQ2_MAC256(SH, L, A, B)                                                    \
  do {                                                                                    \
    const __m256i t_ = MMFREE_TQ2_TRITS_X86(q16, SH);                                     \
    const std::int32_t* xp_ = xb + 64 * (L) + c;                                           \
    const __m256i tlo_ = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(t_));                \
    const __m256i thi_ = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(t_, 1));           \
    A = _mm256_add_epi32(                                                                  \
        A, _mm256_sign_epi32(_mm256_loadu_si256((const __m256i*)xp_), tlo_));               \
    B = _mm256_add_epi32(                                                                  \
        B, _mm256_sign_epi32(_mm256_loadu_si256((const __m256i*)(xp_ + 8)), thi_));         \
  } while (0)
      MMFREE_TQ2_MAC256(0, 0, acc0, acc1);
      MMFREE_TQ2_MAC256(2, 1, acc2, acc3);
      MMFREE_TQ2_MAC256(4, 2, acc0, acc1);
      MMFREE_TQ2_MAC256(6, 3, acc2, acc3);
#undef MMFREE_TQ2_MAC256
    }
  }
  return simd::hsum_epi32(_mm256_add_epi32(_mm256_add_epi32(acc0, acc1),
                                           _mm256_add_epi32(acc2, acc3)));
#else
  return dot_scalar<std::int32_t, std::int32_t>(x, src, n);
#endif
}

inline float dot_f32(const float* x, const std::uint8_t* src, std::size_t n) {
#if defined(MMFREE_ARCH_ARM)
  float32x4_t a0 = vdupq_n_f32(0), a1 = vdupq_n_f32(0), a2 = vdupq_n_f32(0), a3 = vdupq_n_f32(0);
  const std::size_t nb = n / kBlockElems;
  for (std::size_t b = 0; b < nb; ++b) {
    const std::uint8_t* q = src + b * kBlockBytes;
    const float* xb = x + b * kBlockElems;
    for (std::size_t c = 0; c < 64; c += 16) {
      const uint8x16_t qv = vld1q_u8(q + c);
      tq1::mac16_f32(a0, a1, a2, a3, xb + 0 + c, MMFREE_TQ2_TRITS(qv, 0));
      tq1::mac16_f32(a0, a1, a2, a3, xb + 64 + c, MMFREE_TQ2_TRITS(qv, 2));
      tq1::mac16_f32(a0, a1, a2, a3, xb + 128 + c, MMFREE_TQ2_TRITS(qv, 4));
      tq1::mac16_f32(a0, a1, a2, a3, xb + 192 + c, MMFREE_TQ2_TRITS(qv, 6));
    }
  }
  return vaddvq_f32(vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3)));
#else
  return dot_scalar<float, float>(x, src, n);
#endif
}

}  // namespace tq2

// ---- backend --------------------------------------------------------------------------
// Format policies, so one PackedBackend drives both packed layouts.
struct Tq1Format {
  static const char* name() { return "TQ1_0"; }
  static double bits() { return 8.0 * (double)tq1::kBlockBytes / (double)tq1::kBlockElems; }
  static bool fits(std::size_t n) { return tq1::fits(n); }
  static std::size_t row_bytes(std::size_t n) { return tq1::row_bytes(n); }
  static void pack_row(std::uint8_t* d, const std::int8_t* w, std::size_t n) {
    tq1::pack_row(d, w, n);
  }
  static std::int32_t dot_i32(const std::int32_t* x, const std::uint8_t* q, std::size_t n) {
    return tq1::dot_i32(x, q, n);
  }
};

struct Tq2Format {
  static const char* name() { return "2-bit"; }
  static double bits() { return 8.0 * (double)tq2::kBlockBytes / (double)tq2::kBlockElems; }
  static bool fits(std::size_t n) { return tq2::fits(n); }
  static std::size_t row_bytes(std::size_t n) { return tq2::row_bytes(n); }
  static void pack_row(std::uint8_t* d, const std::int8_t* w, std::size_t n) {
    tq2::pack_row(d, w, n);
  }
  static std::int32_t dot_i32(const std::int32_t* x, const std::uint8_t* q, std::size_t n) {
    return tq2::dot_i32(x, q, n);
  }
};

// Drop-in CpuBackend replacement that keeps each projection's weights packed. The packed
// copy is built on a projection's first matmul and cached, keyed on the `wq` pointer (a
// stable, per-projection address inside the mmap'd blob), so the cost lands on the first
// token. A projection whose N is not a multiple of 256 stays on the int8 kernel.
template <class Fmt>
struct PackedBackend final : TernaryBackend {
  PackedBackend() = default;
  explicit PackedBackend(std::size_t batch) : batch_(batch ? batch : 1) {}

  // Batching as in CpuBackend (ternary_backend.hpp): weight row outer, bit-exact.
  std::size_t batch_size() const override { return batch_; }

  void matmul(int /*proj_id*/, const std::int32_t* x, std::int32_t* acc,
              const std::int8_t* wq, std::size_t N, std::size_t M) override {
    if (!Fmt::fits(N)) {  // shape we can't block — stay on the int8 path
#pragma omp parallel for schedule(static)
      for (std::size_t o = 0; o < M; ++o) acc[o] = simd::ternary_dot_i32(x, wq + o * N, N);
      return;
    }
    const std::uint8_t* packed = ensure(wq, N, M);
    const std::size_t stride = Fmt::row_bytes(N);
#pragma omp parallel for schedule(static)
    for (std::size_t o = 0; o < M; ++o) acc[o] = Fmt::dot_i32(x, packed + o * stride, N);
  }

  void matmul_batch(int proj_id, const std::int32_t* x, std::int32_t* acc,
                    const std::int8_t* wq, std::size_t N, std::size_t M,
                    std::size_t b) override {
    if (b <= 1) { matmul(proj_id, x, acc, wq, N, M); return; }
    if (!Fmt::fits(N)) {
#pragma omp parallel for schedule(static)
      for (std::size_t o = 0; o < M; ++o) {
        const std::int8_t* w = wq + o * N;
        for (std::size_t i = 0; i < b; ++i)
          acc[i * M + o] = simd::ternary_dot_i32(x + i * N, w, N);
      }
      return;
    }
    // ensure() mutates the cache, so it must run BEFORE the parallel region (as in matmul).
    const std::uint8_t* packed = ensure(wq, N, M);
    const std::size_t stride = Fmt::row_bytes(N);
#pragma omp parallel for schedule(static)
    for (std::size_t o = 0; o < M; ++o) {
      const std::uint8_t* q = packed + o * stride;
      for (std::size_t i = 0; i < b; ++i)
        acc[i * M + o] = Fmt::dot_i32(x + i * N, q, N);
    }
  }

  std::size_t packed_bytes() const { return bytes_; }
  std::size_t packed_tensors() const { return cache_.size(); }
  static const char* format_name() { return Fmt::name(); }
  static double format_bits() { return Fmt::bits(); }

 private:
  const std::uint8_t* ensure(const std::int8_t* wq, std::size_t N, std::size_t M) {
    auto it = cache_.find(wq);
    if (it != cache_.end()) return it->second.data();
    const std::size_t stride = Fmt::row_bytes(N);
    std::vector<std::uint8_t> buf(stride * M);
    std::uint8_t* dst = buf.data();
    parallel_rows(M, [&](std::size_t o) { Fmt::pack_row(dst + o * stride, wq + o * N, N); });
    bytes_ += buf.size();
    return cache_.emplace(wq, std::move(buf)).first->second.data();
  }

  // Unsynchronized: bitlinear calls the backend from the single model thread only.
  std::unordered_map<const std::int8_t*, std::vector<std::uint8_t>> cache_;
  std::size_t bytes_ = 0;
  std::size_t batch_ = 1;
};

using Tq1Backend = PackedBackend<Tq1Format>;   // 1.625 bits/weight
using Tq2Backend = PackedBackend<Tq2Format>;   // 2.0   bits/weight

}  // namespace mmfree
