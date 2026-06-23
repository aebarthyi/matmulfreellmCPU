// weights.cpp — mmap the model.mmfree blob and build a name->view table.
//
// Blob layout (see tools/pack_weights.py):
//   [8]  magic "MMFREE1\n"
//   [8]  u64 header_len (LE)
//   [hl] header JSON {"config":{...},"tensors":{name:{dtype,shape,offset,nbytes}}}
//   pad to 64; then the data section (each tensor 64-byte aligned, offset relative
//   to the data-section start).
#include "mmfree/weights.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstdint>
#include <cstring>

namespace mmfree {
namespace {

constexpr std::size_t kAlign = 64;
std::size_t align_up(std::size_t n) { return (n + kAlign - 1) / kAlign * kAlign; }

// --- minimal JSON parser (objects/arrays/strings/numbers/bools/null) -----------------
struct Json {
  enum Type { Null, Bool, Num, Str, Arr, Obj } type = Null;
  bool b = false;
  double num = 0;
  std::string str;
  std::vector<Json> arr;
  std::vector<std::pair<std::string, Json>> obj;  // insertion-ordered

  const Json* find(const std::string& key) const {
    for (auto& kv : obj)
      if (kv.first == key) return &kv.second;
    return nullptr;
  }
  double n() const { return num; }
  std::size_t u() const { return static_cast<std::size_t>(num); }
};

struct Parser {
  const char* p;
  const char* end;

  void ws() {
    while (p < end && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')) ++p;
  }
  [[noreturn]] void fail(const char* what) { throw std::runtime_error(std::string("json: ") + what); }

  Json parse() {
    ws();
    Json v = value();
    return v;
  }

  Json value() {
    ws();
    if (p >= end) fail("unexpected end");
    char c = *p;
    if (c == '{') return object();
    if (c == '[') return array();
    if (c == '"') {
      Json v;
      v.type = Json::Str;
      v.str = string();
      return v;
    }
    if (c == 't' || c == 'f') return boolean();
    if (c == 'n') {
      expect("null");
      return Json{};
    }
    return number();
  }

  void expect(const char* lit) {
    std::size_t n = std::strlen(lit);
    if (static_cast<std::size_t>(end - p) < n || std::strncmp(p, lit, n) != 0) fail("literal");
    p += n;
  }

  Json boolean() {
    Json v;
    v.type = Json::Bool;
    if (*p == 't') {
      expect("true");
      v.b = true;
    } else {
      expect("false");
      v.b = false;
    }
    return v;
  }

  std::string string() {
    if (*p != '"') fail("expected string");
    ++p;
    std::string s;
    while (p < end && *p != '"') {
      if (*p == '\\') {
        ++p;
        if (p >= end) fail("bad escape");
        switch (*p) {
          case 'n': s.push_back('\n'); break;
          case 't': s.push_back('\t'); break;
          case 'r': s.push_back('\r'); break;
          case '"': s.push_back('"'); break;
          case '\\': s.push_back('\\'); break;
          case '/': s.push_back('/'); break;
          default: s.push_back(*p); break;
        }
        ++p;
      } else {
        s.push_back(*p++);
      }
    }
    if (p >= end) fail("unterminated string");
    ++p;  // closing quote
    return s;
  }

  Json number() {
    const char* start = p;
    while (p < end && (std::strchr("+-0123456789.eE", *p))) ++p;
    Json v;
    v.type = Json::Num;
    v.num = std::strtod(std::string(start, p).c_str(), nullptr);
    return v;
  }

  Json array() {
    Json v;
    v.type = Json::Arr;
    ++p;  // [
    ws();
    if (p < end && *p == ']') {
      ++p;
      return v;
    }
    while (true) {
      v.arr.push_back(value());
      ws();
      if (p < end && *p == ',') {
        ++p;
        continue;
      }
      if (p < end && *p == ']') {
        ++p;
        break;
      }
      fail("expected , or ]");
    }
    return v;
  }

  Json object() {
    Json v;
    v.type = Json::Obj;
    ++p;  // {
    ws();
    if (p < end && *p == '}') {
      ++p;
      return v;
    }
    while (true) {
      ws();
      std::string key = string();
      ws();
      if (p >= end || *p != ':') fail("expected :");
      ++p;
      v.obj.emplace_back(std::move(key), value());
      ws();
      if (p < end && *p == ',') {
        ++p;
        continue;
      }
      if (p < end && *p == '}') {
        ++p;
        break;
      }
      fail("expected , or }");
    }
    return v;
  }
};

}  // namespace

Weights::Weights(const std::string& path) {
  fd_ = ::open(path.c_str(), O_RDONLY);
  if (fd_ < 0) throw std::runtime_error("weights: cannot open " + path);
  struct stat st {};
  if (::fstat(fd_, &st) != 0) {
    ::close(fd_);
    throw std::runtime_error("weights: fstat failed for " + path);
  }
  size_ = static_cast<std::size_t>(st.st_size);
  base_ = ::mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd_, 0);
  if (base_ == MAP_FAILED) {
    ::close(fd_);
    base_ = nullptr;
    throw std::runtime_error("weights: mmap failed for " + path);
  }

  const char* bytes = static_cast<const char*>(base_);
  if (size_ < 16 || std::strncmp(bytes, "MMFREE1\n", 8) != 0)
    throw std::runtime_error("weights: bad magic in " + path);
  std::uint64_t header_len = 0;
  std::memcpy(&header_len, bytes + 8, 8);

  Parser parser{bytes + 16, bytes + 16 + header_len};
  Json root = parser.parse();

  // config
  const Json* c = root.find("config");
  if (!c) throw std::runtime_error("weights: no config in header");
  auto cu = [&](const char* k, std::size_t def) {
    const Json* j = c->find(k);
    return j ? j->u() : def;
  };
  auto cb = [&](const char* k, bool def) {
    const Json* j = c->find(k);
    return j ? j->b : def;
  };
  cfg_.vocab_size = cu("vocab_size", cfg_.vocab_size);
  cfg_.hidden_size = cu("hidden_size", cfg_.hidden_size);
  cfg_.num_hidden_layers = cu("num_hidden_layers", cfg_.num_hidden_layers);
  cfg_.num_heads = cu("num_heads", cfg_.num_heads);
  cfg_.expand_ratio = cu("expand_ratio", cfg_.expand_ratio);
  cfg_.intermediate_size = cu("intermediate_size", cfg_.intermediate_size);
  cfg_.use_lower_bound = cb("use_lower_bound", cfg_.use_lower_bound);
  cfg_.use_short_conv = cb("use_short_conv", cfg_.use_short_conv);
  if (const Json* e = c->find("rms_norm_eps")) cfg_.rms_norm_eps = static_cast<float>(e->n());

  // data section starts after the 64-aligned header
  std::size_t data_start = align_up(16 + header_len);
  const char* data = bytes + data_start;

  const Json* ts = root.find("tensors");
  if (!ts || ts->type != Json::Obj) throw std::runtime_error("weights: no tensors in header");
  for (auto& kv : ts->obj) {
    const Json& m = kv.second;
    TensorRef ref;
    const Json* dt = m.find("dtype");
    ref.is_int8 = dt && dt->str == "i8";
    if (const Json* sh = m.find("shape"))
      for (auto& d : sh->arr) ref.shape.push_back(d.u());
    std::size_t off = m.find("offset")->u();
    ref.data = data + off;
    tensors_.emplace(kv.first, std::move(ref));
  }
}

Weights::~Weights() {
  if (base_ && base_ != MAP_FAILED) ::munmap(base_, size_);
  if (fd_ >= 0) ::close(fd_);
}

}  // namespace mmfree
