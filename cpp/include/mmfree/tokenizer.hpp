// mmfree/tokenizer.hpp — LLaMA SentencePiece-BPE tokenizer (forward-only, no deps).
//
// Reproduces the HF `tokenizers` BPE used by MMfreeLM-370M from the data packed by
// tools/pack_tokenizer.py (tokenizer.mmtok): 32000 pieces + rank-ordered merges,
// byte_fallback, normalizer {prepend U+2581, ' ' -> U+2581}, no pre-tokenizer split,
// BOS prepended. Validated id-for-id against HF AutoTokenizer.
#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace mmfree {

class Tokenizer {
 public:
  explicit Tokenizer(const std::string& path);

  // Text -> token ids. Prepends BOS (matches add_bos_token=true) when add_bos.
  std::vector<std::int64_t> encode(const std::string& text, bool add_bos = true) const;

  // Token ids -> text. Skips <s>/</s>/<unk> when skip_special.
  std::string decode(const std::vector<std::int64_t>& ids, bool skip_special = true) const;

  std::int64_t bos_id() const { return bos_; }
  std::int64_t eos_id() const { return eos_; }
  std::int64_t unk_id() const { return unk_; }
  std::size_t vocab_size() const { return id2piece_.size(); }

 private:
  std::vector<std::string> id2piece_;
  std::unordered_map<std::string, int> piece2id_;
  std::unordered_map<std::string, int> merge_rank_;  // "left right" -> rank
  int byte_token_[256];                              // <0xNN> -> id (-1 if absent)
  std::int64_t bos_ = 1, eos_ = 2, unk_ = 0;
};

}  // namespace mmfree
