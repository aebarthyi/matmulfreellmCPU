// profile.cpp — Profiler singleton + accumulation (see mmfree/profile.hpp).
#include "mmfree/profile.hpp"

#include <algorithm>
#include <cstring>

namespace mmfree {

Profiler& Profiler::instance() {
  static Profiler p;
  return p;
}

void Profiler::add(const char* label, std::uint64_t ns) {
  for (auto& e : entries_) {
    if (e.label == label) {
      e.ns += ns;
      ++e.count;
      return;
    }
  }
  entries_.push_back({label, ns, 1});
}

std::vector<Profiler::Entry> Profiler::sorted() const {
  std::vector<Entry> out = entries_;
  std::sort(out.begin(), out.end(),
            [](const Entry& a, const Entry& b) { return a.ns > b.ns; });
  return out;
}

std::uint64_t Profiler::total_ns() const {
  std::uint64_t t = 0;
  for (const auto& e : entries_) t += e.ns;
  return t;
}

}  // namespace mmfree
