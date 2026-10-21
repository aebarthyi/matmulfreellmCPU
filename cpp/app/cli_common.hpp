// cli_common.hpp — pieces shared by the command-line front ends (mmfree-cli, mmfree-chat):
// model-file discovery (blobs/<size>/), ternary-backend selection, and error reporting.
#pragma once

#include "mmfree/model.hpp"
#include "mmfree/tq1.hpp"  // CpuBackend / Tq1Backend / Tq2Backend

#include <sys/stat.h>

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

namespace mmfree::cli {

inline bool file_exists(const std::string& p) {
  std::ifstream f(p, std::ios::binary);
  return f.good();
}

inline const char* env_or_null(const char* k) {
  const char* v = std::getenv(k);
  return (v && *v) ? v : nullptr;
}

// Directory holding the running executable, from argv[0] ("" when it has no path).
inline std::string exe_dir(const char* argv0) {
  std::string p = argv0 ? argv0 : "";
  std::size_t slash = p.find_last_of('/');
  return (slash == std::string::npos) ? "" : p.substr(0, slash);
}

// The released checkpoints, smallest first. Each provisions into blobs/<size>/
// (tools/provision.py --model <size>) as model.mmfree + tokenizer.mmtok.
inline const std::vector<std::string>& model_sizes() {
  static const std::vector<std::string> sizes = {"370M", "1.3B", "2.7B"};
  return sizes;
}

// "1.3b" -> "1.3B"; "" if not a known size.
inline std::string canonical_size(const std::string& s) {
  std::string up;
  for (char c : s) up.push_back(static_cast<char>(std::toupper(static_cast<unsigned char>(c))));
  for (const std::string& k : model_sizes())
    if (up == k) return k;
  return "";
}

inline std::string join_sizes(const std::vector<std::string>& v) {
  std::string out;
  for (const std::string& x : v) out += (out.empty() ? "" : ", ") + x;
  return out.empty() ? "none" : out;
}

// blobs/ in the working dir, else beside the executable or one/two levels up (the build/
// and cpp/build/ layouts). "" if there is none.
inline std::string find_blobs_dir(const char* argv0) {
  std::vector<std::string> dirs = {"blobs"};
  const std::string ed = exe_dir(argv0);
  if (!ed.empty()) {
    dirs.push_back(ed + "/blobs");
    dirs.push_back(ed + "/../blobs");
    dirs.push_back(ed + "/../../blobs");
  }
  for (const std::string& d : dirs) {
    struct stat st {};
    if (stat(d.c_str(), &st) == 0 && S_ISDIR(st.st_mode)) return d;
  }
  return "";
}

// Sizes with a weight blob under `blobs`, smallest first.
inline std::vector<std::string> provisioned_sizes(const std::string& blobs) {
  std::vector<std::string> out;
  if (blobs.empty()) return out;
  for (const std::string& k : model_sizes())
    if (file_exists(blobs + "/" + k + "/model.mmfree")) out.push_back(k);
  return out;
}

struct ModelPaths {
  std::string blob, tokenizer;
  std::string size;  // set when the blob came from blobs/<size>/
};

// Resolve the model files. Weights: --blob, else $MMFREE_BLOB, else blobs/<size>/ for
// `size_flag` (default: the smallest provisioned size). Tokenizer: --tokenizer, else
// $MMFREE_TOKENIZER, else beside the weights. Returns 0, or prints what is missing and
// returns an exit code; `need_blob` = false (tokenizer-only uses) tolerates no weights.
inline int resolve_model(const std::string& blob_flag, const std::string& size_flag,
                         const std::string& tok_flag, const char* argv0, bool need_blob,
                         const char* prog, ModelPaths& out) {
  std::string size;
  if (!size_flag.empty()) {
    size = canonical_size(size_flag);
    if (size.empty()) {
      std::fprintf(stderr, "error: unknown --model %s (expected %s)\n", size_flag.c_str(),
                   join_sizes(model_sizes()).c_str());
      return 2;
    }
  }
  std::string dir;  // directory the tokenizer defaults to
  out.blob = blob_flag;
  if (out.blob.empty())
    if (const char* e = env_or_null("MMFREE_BLOB")) out.blob = e;
  if (!out.blob.empty()) {
    const std::size_t slash = out.blob.find_last_of('/');
    dir = (slash == std::string::npos) ? "." : out.blob.substr(0, slash);
  } else {
    const std::string blobs = find_blobs_dir(argv0);
    const std::vector<std::string> have = provisioned_sizes(blobs);
    if (size.empty() && !have.empty()) size = have.front();
    if (!blobs.empty() && !size.empty()) {
      dir = blobs + "/" + size;
      if (file_exists(dir + "/model.mmfree")) {
        out.blob = dir + "/model.mmfree";
        out.size = size;
      }
    }
    if (out.blob.empty() && need_blob) {
      if (!size_flag.empty())
        std::fprintf(stderr, "error: the %s model is not provisioned in %s/ "
                             "(provisioned: %s)\n",
                     size.c_str(), blobs.empty() ? "blobs" : blobs.c_str(),
                     join_sizes(have).c_str());
      else
        std::fprintf(stderr, "error: no model provisioned (looked for blobs/<size>/model.mmfree "
                             "in the working directory and next to the executable)\n");
      const char* want = size.empty() ? "370M" : size.c_str();
      std::fprintf(stderr,
                   "  provision it:    make model MODEL=%s   "
                   "(or: python tools/provision.py --model %s)\n"
                   "  or point at one: %s --blob /path/to/model.mmfree  (or set MMFREE_BLOB)\n",
                   want, want, prog);
      return 2;
    }
  }
  if (!tok_flag.empty()) out.tokenizer = tok_flag;
  else if (const char* e = env_or_null("MMFREE_TOKENIZER")) out.tokenizer = e;
  else out.tokenizer = (dir.empty() ? std::string(".") : dir) + "/tokenizer.mmtok";
  return 0;
}

inline bool valid_weight_format(const std::string& f) {
  return f == "int8" || f == "1.6bit" || f == "2bit";
}

// Owns whichever ternary backend install_backend() put on the model; must outlive its use.
struct Backends {
  std::unique_ptr<CpuBackend> cpu;
  std::unique_ptr<Tq1Backend> tq1;
  std::unique_ptr<Tq2Backend> tq2;
};

// Install the backend for weight format `wfmt` and batch width `batch`. The packed formats
// repack each projection on first use and are bit-identical to int8. Integer (fixed) path
// only: the float path never consults the backend, so it warns instead.
inline void install_backend(Model& model, const std::string& wfmt, ActQuant aq,
                            std::size_t batch, Backends& out) {
  if (batch > 1 && aq != ActQuant::FixedQ510) {
    std::fprintf(stderr, "warning: --batch %zu does not batch the matmul in --mode float "
                         "(the backend seam is integer-only); streams run independently\n",
                 batch);
  }
  if (wfmt != "int8") {
    if (aq != ActQuant::FixedQ510) {
      std::fprintf(stderr, "warning: --weights %s is ignored in --mode float "
                           "(the packed kernels are integer-only); using int8\n",
                   wfmt.c_str());
    } else if (wfmt == "1.6bit") {
      out.tq1 = std::make_unique<Tq1Backend>(batch);
      model.set_backend(out.tq1.get(), {});
      std::fprintf(stderr, "weight format: %s, %.4f bits/weight (%zu-weight blocks, %zu B)\n",
                   Tq1Backend::format_name(), Tq1Backend::format_bits(),
                   tq1::kBlockElems, tq1::kBlockBytes);
    } else {  // 2bit
      out.tq2 = std::make_unique<Tq2Backend>(batch);
      model.set_backend(out.tq2.get(), {});
      std::fprintf(stderr, "weight format: %s, %.4f bits/weight (%zu-weight blocks, %zu B)\n",
                   Tq2Backend::format_name(), Tq2Backend::format_bits(),
                   tq2::kBlockElems, tq2::kBlockBytes);
    }
  } else if (batch > 1 && aq == ActQuant::FixedQ510) {
    // int8 normally runs with no backend (bitlinear's internal CpuBackend, width 1).
    out.cpu = std::make_unique<CpuBackend>(batch);
    model.set_backend(out.cpu.get(), {});
  }
}

// Turn a load failure into one actionable line (Weights/Tokenizer throw on bad files).
inline int report_error(const std::exception& e) {
  std::string msg = e.what();
  std::fprintf(stderr, "error: %s\n", msg.c_str());
  if (msg.compare(0, 9, "weights: ") == 0 || msg.compare(0, 11, "tokenizer: ") == 0)
    std::fprintf(stderr,
                 "hint: the model files come from `python tools/provision.py` "
                 "(see the README); pass --blob/--tokenizer to point elsewhere.\n");
  return 1;
}

}  // namespace mmfree::cli
