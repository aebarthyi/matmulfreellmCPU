// mmfree/tensor.hpp — lightweight, framework-free row-major tensor views.
//
// Forward-only CPU inference for HGRN-Bit / MMfreeLM. No ownership semantics beyond
// an optional backing std::vector; views are non-owning (ptr + shape). Everything is
// row-major and fp32 unless stated. Batch=1 inference v1 uses 2D [T, dim] mostly.
#pragma once

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <numeric>
#include <vector>

namespace mmfree {

// Non-owning 2D row-major view over float data: shape [rows, cols], stride = cols.
struct MatF {
  const float* data = nullptr;
  std::size_t rows = 0;
  std::size_t cols = 0;

  MatF() = default;
  MatF(const float* d, std::size_t r, std::size_t c) : data(d), rows(r), cols(c) {}

  const float* row(std::size_t r) const { return data + r * cols; }
  std::size_t size() const { return rows * cols; }
};

// Mutable 2D row-major view.
struct MatFMut {
  float* data = nullptr;
  std::size_t rows = 0;
  std::size_t cols = 0;

  MatFMut() = default;
  MatFMut(float* d, std::size_t r, std::size_t c) : data(d), rows(r), cols(c) {}

  float* row(std::size_t r) { return data + r * cols; }
  const float* row(std::size_t r) const { return data + r * cols; }
  std::size_t size() const { return rows * cols; }
  operator MatF() const { return MatF(data, rows, cols); }
};

// Owning fp32 buffer with a shape, convertible to views. Used for scratch + activations.
struct Tensor {
  std::vector<float> buf;
  std::vector<std::size_t> shape;

  Tensor() = default;
  explicit Tensor(std::initializer_list<std::size_t> s) : shape(s) {
    buf.assign(numel(), 0.0f);
  }
  Tensor(std::vector<std::size_t> s) : shape(std::move(s)) { buf.assign(numel(), 0.0f); }

  std::size_t numel() const {
    return shape.empty() ? 0
                         : std::accumulate(shape.begin(), shape.end(), std::size_t{1},
                                           std::multiplies<>());
  }
  float* data() { return buf.data(); }
  const float* data() const { return buf.data(); }

  // View the last dim as columns; everything before it collapses into rows.
  MatFMut mat() {
    std::size_t cols = shape.empty() ? 0 : shape.back();
    std::size_t rows = cols ? numel() / cols : 0;
    return MatFMut(buf.data(), rows, cols);
  }
  MatF mat() const {
    std::size_t cols = shape.empty() ? 0 : shape.back();
    std::size_t rows = cols ? numel() / cols : 0;
    return MatF(buf.data(), rows, cols);
  }
};

}  // namespace mmfree
