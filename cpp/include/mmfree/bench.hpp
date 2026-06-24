// mmfree/bench.hpp — shared generation-timing harness (tok/s) used by the CLIs.
//
// Extracted from app/main.cpp so every front-end (the CPU mmfree-cli and, in the KRIA
// repo, the CPU+FPGA runner) times generation with byte-identical code — the only thing
// that differs between a CPU-only and a CPU+FPGA measurement is the Model's backend, not
// the measurement itself. Keeps the comparison apples-to-apples.
#pragma once

#include "mmfree/model.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace mmfree {

struct BenchOpts {
  std::size_t gen = 32;  // tokens to generate per run (EOS disabled -> exactly this many)
  int warmup = 1;        // untimed warmup runs
  int reps = 3;          // timed runs to average
  bool profile = false;  // reset+enable the Profiler around the timed reps, print breakdown
};

struct BenchResult {
  double avg_prefill_s = 0;
  double avg_decode_tps = 0;   // steady-state decode throughput (the headline metric)
  double avg_overall_tps = 0;
  std::size_t prompt_tokens = 0;
  std::size_t gen_tokens = 0;
};

// Time greedy generation over `ids` (temperature 0, EOS disabled), printing the same
// header / per-run / summary lines the standalone CLI used. `blob_label`/`mode_label`/`frac`
// only annotate the header. Returns the averaged summary.
BenchResult run_bench(Model& model, const std::vector<std::int64_t>& ids, const BenchOpts& opts,
                      const char* blob_label, const char* mode_label, int frac);

// Print the accumulated per-op wall-clock breakdown to stdout, divided by `runs`
// ("per generation"). No-op message when there are no samples.
void print_profile(int runs);

}  // namespace mmfree
