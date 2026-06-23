// mmfree/profile.hpp — lightweight wall-clock op profiler.
//
// Off by default: when disabled, ScopedTimer reads one bool and does nothing, so the
// instrumented model paths carry essentially zero overhead in normal runs. Enable it
// (Profiler::instance().enable(true)) to accumulate elapsed time and call counts per
// label; the model wraps each op (matmul/rmsnorm/scan/...) in a ScopedTimer.
//
// Single-threaded use: timers are created/destroyed on the main thread only. The
// parallelism inside bitlinear is *inside* the timed region, so the enclosing timer
// records correct wall-clock time for the whole matmul (not per-thread CPU time).
#pragma once

#include <chrono>
#include <cstdint>
#include <string>
#include <vector>

namespace mmfree {

class Profiler {
 public:
  struct Entry {
    std::string label;
    std::uint64_t ns = 0;     // total elapsed nanoseconds
    std::uint64_t count = 0;  // number of timed calls
  };

  static Profiler& instance();

  void enable(bool on) { enabled_ = on; }
  bool enabled() const { return enabled_; }

  void reset() { entries_.clear(); }
  void add(const char* label, std::uint64_t ns);

  // Snapshot sorted by total time descending.
  std::vector<Entry> sorted() const;
  std::uint64_t total_ns() const;

 private:
  bool enabled_ = false;
  std::vector<Entry> entries_;  // few labels -> linear scan is fine
};

// RAII timer: records elapsed time into `label` on destruction when profiling is on.
class ScopedTimer {
 public:
  explicit ScopedTimer(const char* label)
      : label_(label), on_(Profiler::instance().enabled()) {
    if (on_) t0_ = std::chrono::steady_clock::now();
  }
  ~ScopedTimer() {
    if (on_) {
      auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - t0_)
                    .count();
      Profiler::instance().add(label_, static_cast<std::uint64_t>(ns));
    }
  }
  ScopedTimer(const ScopedTimer&) = delete;
  ScopedTimer& operator=(const ScopedTimer&) = delete;

 private:
  const char* label_;
  bool on_;
  std::chrono::steady_clock::time_point t0_;
};

#define MMFREE_PROFILE_CONCAT_(a, b) a##b
#define MMFREE_PROFILE_CONCAT(a, b) MMFREE_PROFILE_CONCAT_(a, b)
// Time the enclosing scope under `name` (a string literal or const char*).
#define MMFREE_PROFILE(name) \
  ::mmfree::ScopedTimer MMFREE_PROFILE_CONCAT(mmfree_timer_, __LINE__)(name)

}  // namespace mmfree
