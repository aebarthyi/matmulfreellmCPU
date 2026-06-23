// mmfree/config.hpp — HGRNBitConfig mirror, loaded from the packed blob's config.json.
#pragma once

#include <cstddef>

namespace mmfree {

struct Config {
  std::size_t vocab_size = 32000;
  std::size_t hidden_size = 1024;
  std::size_t num_hidden_layers = 24;
  std::size_t num_heads = 1;
  std::size_t expand_ratio = 1;
  std::size_t intermediate_size = 2816;  // gate_proj out = 2*this; down_proj in = this
  bool use_lower_bound = true;
  bool use_short_conv = false;
  float rms_norm_eps = 1e-6f;

  std::size_t input_dim() const { return hidden_size * expand_ratio; }
  std::size_t head_dim() const { return input_dim() / num_heads; }
};

}  // namespace mmfree
