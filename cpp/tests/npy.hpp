// npy.hpp — minimal NumPy .npy v1.0 reader for tests (header-only).
// Supports C-contiguous little-endian float32, int32, int64 arrays — enough to load
// the golden oracles emitted by tools/dump_golden.py.
#pragma once

#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace mmfree::npy {

struct Array {
  std::vector<std::size_t> shape;
  std::string dtype;          // e.g. "<f4", "<i4", "<i8"
  std::vector<float> f32;     // populated for float arrays
  std::vector<int64_t> i64;   // populated for int arrays (i4 widened to i64)

  std::size_t numel() const {
    std::size_t n = 1;
    for (auto s : shape) n *= s;
    return shape.empty() ? 0 : n;
  }
};

inline Array load(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("npy: cannot open " + path);

  char magic[6];
  f.read(magic, 6);
  if (std::string(magic, 6) != "\x93NUMPY")
    throw std::runtime_error("npy: bad magic in " + path);
  uint8_t major = 0, minor = 0;
  f.read(reinterpret_cast<char*>(&major), 1);
  f.read(reinterpret_cast<char*>(&minor), 1);
  uint16_t hlen16 = 0;
  f.read(reinterpret_cast<char*>(&hlen16), 2);
  std::size_t hlen = hlen16;
  std::string header(hlen, '\0');
  f.read(header.data(), static_cast<std::streamsize>(hlen));

  Array a;
  // descr
  auto dpos = header.find("'descr':");
  auto q1 = header.find('\'', dpos + 8);
  auto q2 = header.find('\'', q1 + 1);
  a.dtype = header.substr(q1 + 1, q2 - q1 - 1);
  // shape
  auto spos = header.find("'shape':");
  auto lp = header.find('(', spos);
  auto rp = header.find(')', lp);
  std::string shp = header.substr(lp + 1, rp - lp - 1);
  std::size_t i = 0;
  while (i < shp.size()) {
    while (i < shp.size() && (shp[i] == ' ' || shp[i] == ',')) ++i;
    std::size_t j = i;
    while (j < shp.size() && shp[j] >= '0' && shp[j] <= '9') ++j;
    if (j > i) a.shape.push_back(std::stoul(shp.substr(i, j - i)));
    i = j + 1;
  }

  std::size_t n = a.numel();
  if (a.dtype == "<f4") {
    a.f32.resize(n);
    f.read(reinterpret_cast<char*>(a.f32.data()), static_cast<std::streamsize>(n * 4));
  } else if (a.dtype == "<i4") {
    std::vector<int32_t> tmp(n);
    f.read(reinterpret_cast<char*>(tmp.data()), static_cast<std::streamsize>(n * 4));
    a.i64.assign(tmp.begin(), tmp.end());
  } else if (a.dtype == "<i8") {
    a.i64.resize(n);
    f.read(reinterpret_cast<char*>(a.i64.data()), static_cast<std::streamsize>(n * 8));
  } else if (a.dtype == "|i1") {
    std::vector<int8_t> tmp(n);
    f.read(reinterpret_cast<char*>(tmp.data()), static_cast<std::streamsize>(n));
    a.i64.assign(tmp.begin(), tmp.end());
  } else {
    throw std::runtime_error("npy: unsupported dtype " + a.dtype + " in " + path);
  }
  return a;
}

}  // namespace mmfree::npy
