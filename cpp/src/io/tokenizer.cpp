// tokenizer.cpp — LLaMA SentencePiece-BPE encode/decode (see tokenizer.hpp).
#include "mmfree/tokenizer.hpp"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <queue>
#include <stdexcept>

namespace mmfree {

namespace {
constexpr char kMagic[8] = {'M', 'M', 'T', 'O', 'K', '1', '\n', '\0'};
const std::string kUnderscore = "\xE2\x96\x81";  // U+2581 "▁"

std::uint32_t read_u32(std::ifstream& f) {
  std::uint32_t v;
  f.read(reinterpret_cast<char*>(&v), 4);
  return v;  // pack_tokenizer writes little-endian; build hosts are LE.
}
std::int32_t read_i32(std::ifstream& f) {
  std::int32_t v;
  f.read(reinterpret_cast<char*>(&v), 4);
  return v;
}
std::string read_str(std::ifstream& f) {
  std::uint32_t n = read_u32(f);
  std::string s(n, '\0');
  if (n) f.read(&s[0], n);
  return s;
}

// Split a UTF-8 string into its codepoints (one substring per codepoint).
std::vector<std::string> utf8_chars(const std::string& s) {
  std::vector<std::string> out;
  for (std::size_t i = 0; i < s.size();) {
    std::size_t len = 1;
    unsigned char c = static_cast<unsigned char>(s[i]);
    if (c >= 0xF0) len = 4;
    else if (c >= 0xE0) len = 3;
    else if (c >= 0xC0) len = 2;
    if (i + len > s.size()) len = 1;  // truncated -> byte-fallback territory
    out.emplace_back(s.substr(i, len));
    i += len;
  }
  return out;
}
}  // namespace

Tokenizer::Tokenizer(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("tokenizer: cannot open " + path);
  char magic[8];
  f.read(magic, 8);
  if (std::memcmp(magic, kMagic, 8) != 0)
    throw std::runtime_error("tokenizer: bad magic in " + path);

  std::uint32_t vocab_size = read_u32(f);
  std::uint32_t n_merges = read_u32(f);
  bos_ = read_i32(f);
  eos_ = read_i32(f);
  unk_ = read_i32(f);

  id2piece_.resize(vocab_size);
  piece2id_.reserve(vocab_size * 2);
  for (std::uint32_t i = 0; i < vocab_size; ++i) {
    id2piece_[i] = read_str(f);
    piece2id_.emplace(id2piece_[i], static_cast<int>(i));
  }
  merge_rank_.reserve(n_merges * 2);
  for (std::uint32_t r = 0; r < n_merges; ++r)
    merge_rank_.emplace(read_str(f), static_cast<int>(r));
  if (!f) throw std::runtime_error("tokenizer: truncated " + path);

  // Resolve the 256 byte-fallback tokens <0x00>..<0xFF>.
  for (int b = 0; b < 256; ++b) {
    char buf[8];
    std::snprintf(buf, sizeof(buf), "<0x%02X>", b);
    auto it = piece2id_.find(buf);
    byte_token_[b] = (it != piece2id_.end()) ? it->second : -1;
  }
}

std::vector<std::int64_t> Tokenizer::encode(const std::string& text, bool add_bos) const {
  std::vector<std::int64_t> out;
  if (add_bos) out.push_back(bos_);
  if (text.empty()) return out;  // HF emits only BOS for empty input (no dummy prefix)

  // Normalize: prepend "▁", then replace every ' ' with "▁".
  std::string norm = kUnderscore;
  for (char ch : text) norm += (ch == ' ') ? kUnderscore : std::string(1, ch);
  if (norm.empty()) return out;

  // Initial symbols = codepoints, as a doubly-linked list.
  struct Sym {
    std::string text;
    int prev, next;
  };
  std::vector<Sym> sym;
  for (auto& c : utf8_chars(norm)) sym.push_back({c, -1, -1});
  const int n = static_cast<int>(sym.size());
  for (int i = 0; i < n; ++i) {
    sym[i].prev = i - 1;
    sym[i].next = (i + 1 < n) ? i + 1 : -1;
  }

  // Merge the lowest-rank adjacent pair first (ties: leftmost), llama.cpp-style PQ.
  struct Bigram {
    int left, right, rank, pos;
  };
  auto worse = [](const Bigram& a, const Bigram& b) {
    return a.rank != b.rank ? a.rank > b.rank : a.pos > b.pos;  // min-heap on (rank,pos)
  };
  std::priority_queue<Bigram, std::vector<Bigram>, decltype(worse)> pq(worse);
  auto try_add = [&](int l, int r) {
    if (l < 0 || r < 0) return;
    auto it = merge_rank_.find(sym[l].text + " " + sym[r].text);
    if (it != merge_rank_.end()) pq.push({l, r, it->second, l});
  };
  for (int i = 0; i + 1 < n; ++i) try_add(i, i + 1);

  while (!pq.empty()) {
    Bigram b = pq.top();
    pq.pop();
    // Skip stale entries: a symbol is dead (empty) or no longer adjacent to its pair.
    if (sym[b.left].text.empty() || sym[b.right].text.empty()) continue;
    if (sym[b.left].next != b.right) continue;
    // Re-verify against the CURRENT texts: either endpoint may have grown via an earlier
    // merge, so this queued pair could no longer be the merge we think it is. Only proceed
    // if the live pair still maps to exactly this merge rank (the fresh pair, if any, was
    // queued separately). Without this, two grown symbols get blindly concatenated into a
    // non-vocab piece -> spurious byte-fallback.
    auto mr = merge_rank_.find(sym[b.left].text + " " + sym[b.right].text);
    if (mr == merge_rank_.end() || mr->second != b.rank) continue;
    sym[b.left].text += sym[b.right].text;  // merged piece (guaranteed in vocab)
    sym[b.right].text.clear();
    int nx = sym[b.right].next;
    sym[b.left].next = nx;
    if (nx != -1) sym[nx].prev = b.left;
    try_add(sym[b.left].prev, b.left);
    try_add(b.left, nx);
  }

  // Emit ids; byte-fallback any leftover piece not in the vocab.
  for (int i = 0; i != -1; i = sym[i].next) {
    if (sym[i].text.empty()) continue;
    auto it = piece2id_.find(sym[i].text);
    if (it != piece2id_.end()) {
      out.push_back(it->second);
    } else {
      for (unsigned char c : sym[i].text) out.push_back(byte_token_[c]);
    }
    if (sym[i].next == -1) break;
  }
  return out;
}

std::string Tokenizer::decode(const std::vector<std::int64_t>& ids, bool skip_special) const {
  std::string bytes;
  for (std::int64_t id : ids) {
    if (id < 0 || static_cast<std::size_t>(id) >= id2piece_.size()) continue;
    if (skip_special && (id == bos_ || id == eos_ || id == unk_)) continue;
    const std::string& p = id2piece_[id];
    // Byte-fallback token -> raw byte (fused with neighbours into UTF-8).
    if (p.size() == 6 && p[0] == '<' && p[1] == '0' && p[2] == 'x' && p[5] == '>') {
      bytes.push_back(static_cast<char>(std::strtol(p.substr(3, 2).c_str(), nullptr, 16)));
      continue;
    }
    // Normal piece: replace "▁" with space.
    for (std::size_t i = 0; i < p.size();) {
      if (p.compare(i, kUnderscore.size(), kUnderscore) == 0) {
        bytes.push_back(' ');
        i += kUnderscore.size();
      } else {
        bytes.push_back(p[i]);
        ++i;
      }
    }
  }
  if (!bytes.empty() && bytes.front() == ' ') bytes.erase(bytes.begin());  // strip dummy prefix
  return bytes;
}

}  // namespace mmfree
