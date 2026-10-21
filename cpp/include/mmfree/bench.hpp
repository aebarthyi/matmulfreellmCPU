// mmfree/bench.hpp — generation-timing harness (tok/s), shared by mmfree-cli and the KRIA
// CPU+FPGA runner so both measure identically; only the Model's backend differs.
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
  // Streams decoded in lockstep (generate_serve); 1 = single-stream generate(). > 1 only
  // batches the matmul if the Model's backend has batch_size() >= this (see --batch).
  std::size_t batch = 1;
};

struct BenchResult {
  double avg_prefill_s = 0;
  double avg_decode_tps = 0;   // steady-state decode throughput, AGGREGATE over all streams
                               // (the headline metric; == per-stream when batch == 1)
  double avg_overall_tps = 0;
  std::size_t prompt_tokens = 0;
  std::size_t gen_tokens = 0;  // per stream
  std::size_t batch = 1;       // streams decoded in lockstep
};

// Time greedy generation over `ids` (temperature 0, EOS disabled), printing a header,
// per-run and summary lines. `blob_label`/`mode_label`/`frac` only annotate the header.
// Returns the averaged summary.
BenchResult run_bench(Model& model, const std::vector<std::int64_t>& ids, const BenchOpts& opts,
                      const char* blob_label, const char* mode_label, int frac);

// Print the accumulated per-op wall-clock breakdown to stdout, divided by `runs`
// ("per generation"). No-op message when there are no samples.
void print_profile(int runs);

}  // namespace mmfree
