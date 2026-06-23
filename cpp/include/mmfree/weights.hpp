// mmfree/weights.hpp — mmap loader for the packed model.mmfree blob.
//
// The blob (tools/pack_weights.py) holds ternary weights (int8 {-1,0,+1}) + per-tensor
// scale_w, per-projection + standalone norm weights, fp32 embeddings, precomputed
// lower_bounds, final norm and lm_head. Weights are mmap'd and viewed in place -- no
// copies (the fp32 embeddings + lm_head dominate footprint on the 4 GB board).
#pragma once

#include "mmfree/config.hpp"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace mmfree {

// Non-owning view of one tensor inside the mmap.
struct TensorRef {
  const void* data = nullptr;
  std::vector<std::size_t> shape;
  bool is_int8 = false;

  std::size_t numel() const {
    std::size_t n = 1;
    for (auto s : shape) n *= s;
    return shape.empty() ? 0 : n;
  }
  std::size_t rows() const { return shape.empty() ? 0 : shape.front(); }
  std::size_t cols() const { return shape.empty() ? 0 : shape.back(); }
  const float* f32() const { return static_cast<const float*>(data); }
  const int8_t* i8() const { return static_cast<const int8_t*>(data); }
};

class Weights {
 public:
  explicit Weights(const std::string& path);
  ~Weights();
  Weights(const Weights&) = delete;
  Weights& operator=(const Weights&) = delete;

  const Config& config() const { return cfg_; }
  bool has(const std::string& name) const { return tensors_.count(name) != 0; }

  const TensorRef& get(const std::string& name) const {
    auto it = tensors_.find(name);
    if (it == tensors_.end()) throw std::runtime_error("weights: missing tensor " + name);
    return it->second;
  }

 private:
  Config cfg_;
  std::unordered_map<std::string, TensorRef> tensors_;
  void* base_ = nullptr;
  std::size_t size_ = 0;
  int fd_ = -1;
};

}  // namespace mmfree
