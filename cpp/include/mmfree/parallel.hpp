// parallel.hpp — bit-exact CPU threading helpers.
//
// The FPGA-offload forward pass is dominated by CPU-side RMSNorm/quant/dequant and
// the elementwise vector ops (~60% of decode wall time, measured 2026-07-21), and it
// ran single-threaded. These helpers parallelize that work across the ARM cores.
//
// BIT-EXACTNESS CONTRACT (the fixed-point parity gate depends on it): only parallelize
// over INDEPENDENT rows or over disjoint elementwise ranges. Never split a floating-point
// reduction (sumsq, dot) across threads — that reorders the summation and changes the
// result. Each row keeps its own reduction serial, so threaded output is byte-identical
// to single-threaded (verify with OMP_NUM_THREADS=1 vs N).
//
// Without OpenMP the pragmas are ignored and everything stays serial + correct.
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
