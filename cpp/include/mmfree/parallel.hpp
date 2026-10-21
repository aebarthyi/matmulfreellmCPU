// parallel.hpp — bit-exact CPU threading helpers (OpenMP; serial without it).
//
// Bit-exactness contract: parallelize only over INDEPENDENT rows or disjoint elementwise
// ranges, never across a floating-point reduction (sumsq, dot) — that would reorder the
// sum. Threaded output is then byte-identical to OMP_NUM_THREADS=1.
#pragma once
#include <cstddef>

namespace mmfree {

// Run f(r) for r in [0, rows), one independent row per iteration, across threads.
// Use for per-row work whose only cross-column step is a reduction kept inside f.
template <class F>
inline void parallel_rows(std::size_t rows, F&& f) {
#pragma omp parallel for schedule(static) if (rows > 1)
  for (std::size_t r = 0; r < rows; ++r) f(r);
}

// Run f(off, len) over contiguous chunks tiling [0, n), across threads. Use for pure
// elementwise maps (out[i] = g(a[i], ...)) where iteration order is irrelevant. The
// chunk floor keeps tiny ops serial (thread-fork cost would dominate).
template <class F>
inline void parallel_chunks(std::size_t n, F&& f, std::size_t chunk = 8192) {
#pragma omp parallel for schedule(static) if (n > chunk)
  for (std::size_t off = 0; off < n; off += chunk) {
    std::size_t len = (n - off < chunk) ? (n - off) : chunk;
    f(off, len);
  }
}

}  // namespace mmfree
