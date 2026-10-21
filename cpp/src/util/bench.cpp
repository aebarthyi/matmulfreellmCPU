// bench.cpp — generation-timing harness shared by the CLIs (see mmfree/bench.hpp).
#include "mmfree/bench.hpp"

#include "mmfree/profile.hpp"

#include <chrono>
#include <cstdio>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace mmfree {

void print_profile(int runs) {
  auto& prof = mmfree::Profiler::instance();
  auto entries = prof.sorted();
  double total_ms = prof.total_ns() / 1e6;
  if (total_ms <= 0.0 || runs <= 0) {
    std::printf("\nprofile: no samples\n");
    return;
  }
  std::printf("\nop profile (%d run%s, instrumented ops only):\n", runs, runs == 1 ? "" : "s");
  std::printf("  %-12s %10s %8s %10s %10s\n", "op", "ms/run", "%", "calls/run", "us/call");
  double matmul_ms = 0.0;
  for (const auto& e : entries) {
    double ms = e.ns / 1e6;
    double pct = 100.0 * ms / total_ms;
    double per_run_ms = ms / runs;
    double calls_per_run = static_cast<double>(e.count) / runs;
    double us_per_call = e.count ? (e.ns / 1e3) / e.count : 0.0;
    std::printf("  %-12s %10.2f %7.1f%% %10.1f %10.2f\n", e.label.c_str(), per_run_ms, pct,
                calls_per_run, us_per_call);
    if (e.label == "matmul" || e.label == "lm_head") matmul_ms += ms;
  }
  std::printf("  %-12s %10.2f %7.1f%%\n", "TOTAL", total_ms / runs, 100.0);
  std::printf("\n  matmul (incl lm_head): %.1f%% of profiled time;  other ops: %.1f%%\n",
              100.0 * matmul_ms / total_ms, 100.0 * (total_ms - matmul_ms) / total_ms);
}

BenchResult run_bench(Model& model, const std::vector<std::int64_t>& ids, const BenchOpts& opts,
                      const char* blob_label, const char* mode_label, int frac) {
  using clk = std::chrono::steady_clock;
  auto secs = [](clk::duration d) { return std::chrono::duration<double>(d).count(); };

  const std::size_t B = opts.batch > 0 ? opts.batch : 1;

  // One timed generation: greedy, EOS disabled -> exactly opts.gen tokens per stream.
  // Prefill emits one token per stream, so the B-th callback marks the end of prefill and
  // the start of decode. `gen_n` counts tokens across all streams.
  struct RunStat { double prefill_s, decode_s, total_s; std::size_t gen_n; };
  auto run_once = [&]() -> RunStat {
    Sampling g;  // temperature 0 => greedy
    clk::time_point t0 = clk::now(), t_first = t0, t_last = t0;
    std::size_t n = 0;
    auto tick = [&]() {
      clk::time_point now = clk::now();
      if (++n == B) t_first = now;  // prefill done for every stream
      t_last = now;
    };
    if (B == 1) {
      // Single stream: plain generate().
      model.generate(ids, opts.gen, /*eos=*/-1, g, [&](std::int64_t) { tick(); });
    } else {
      std::vector<std::vector<std::int64_t>> prompts(B, ids);
      model.generate_serve(prompts, opts.gen, /*eos=*/-1, g,
                           [&](std::size_t, std::int64_t) { tick(); });
    }
    double prefill_s = secs(t_first - t0);
    double decode_s = (n > B) ? secs(t_last - t_first) : 0.0;
    return {prefill_s, decode_s, secs(t_last - t0), n};
  };

  int threads = 1;
#ifdef _OPENMP
  threads = omp_get_max_threads();
#endif
  std::printf("mmfree-cli benchmark\n");
  std::printf("  blob:          %s\n", blob_label);
  std::printf("  mode:          %s (frac=%d)\n", mode_label, frac);
  std::printf("  threads:       %d\n", threads);
  std::printf("  prompt tokens: %zu\n", ids.size());
  std::printf("  gen tokens:    %zu%s\n", opts.gen, B > 1 ? " (per stream)" : "");
  if (B > 1) std::printf("  batch:         %zu streams decoded in lockstep\n", B);
  std::printf("  warmup/reps:   %d/%d\n\n", opts.warmup, opts.reps);

  for (int i = 0; i < opts.warmup; ++i) run_once();

  // Profile only the timed reps (warmup excluded), so the breakdown lines up with the
  // reported tok/s. Totals are divided by reps in print_profile -> "per generation".
  if (opts.profile) { Profiler::instance().reset(); Profiler::instance().enable(true); }

  // Decode throughput is AGGREGATE across streams: (tokens after prefill) / decode time;
  // per-stream (aggregate / B) is printed alongside.
  double sum_prefill = 0, sum_total = 0, sum_dec_tps = 0;
  for (int i = 0; i < opts.reps; ++i) {
    RunStat s = run_once();
    double dec_tps = (s.gen_n > B && s.decode_s > 0) ? (s.gen_n - B) / s.decode_s : 0.0;
    double overall_tps = (s.total_s > 0) ? s.gen_n / s.total_s : 0.0;
    if (B > 1) {
      std::printf("  run %d: prefill %7.1f ms   decode %8.2f tok/s (%6.2f/stream)"
                  "   overall %8.2f tok/s\n",
                  i + 1, s.prefill_s * 1e3, dec_tps, dec_tps / B, overall_tps);
    } else {
      std::printf("  run %d: prefill %7.1f ms   decode %8.2f tok/s   overall %8.2f tok/s\n",
                  i + 1, s.prefill_s * 1e3, dec_tps, overall_tps);
    }
    sum_prefill += s.prefill_s;
    sum_total += s.total_s;
    sum_dec_tps += dec_tps;
  }
  double n = opts.reps > 0 ? opts.reps : 1;
  double avg_prefill = sum_prefill / n;
  double avg_dec_tps = sum_dec_tps / n;
  double avg_total = sum_total / n;
  const std::size_t prompt_toks = ids.size() * B;   // every stream is prefilled
  const std::size_t gen_toks = opts.gen * B;
  double prefill_tps = avg_prefill > 0 ? prompt_toks / avg_prefill : 0.0;
  double overall_tps = avg_total > 0 ? (gen_toks / avg_total) : 0.0;
  std::printf("\n  avg prefill: %7.1f ms  (%.2f tok/s over %zu prompt tokens)\n",
              avg_prefill * 1e3, prefill_tps, prompt_toks);
  if (B > 1) {
    std::printf("  avg decode:  %8.2f tok/s aggregate  (%.2f tok/s per stream x %zu)\n",
                avg_dec_tps, avg_dec_tps / B, B);
  } else {
    std::printf("  avg decode:  %8.2f tok/s\n", avg_dec_tps);
  }
  std::printf("  avg overall: %8.2f tok/s  (%zu gen tokens / %.3f s)\n", overall_tps, gen_toks,
              avg_total);
  if (opts.profile) { Profiler::instance().enable(false); print_profile(opts.reps); }
  return {avg_prefill, avg_dec_tps, overall_tps, prompt_toks, opts.gen, B};
}

}  // namespace mmfree
