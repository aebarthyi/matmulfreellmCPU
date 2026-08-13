// main.cpp — mmfree-cli: run HGRN-Bit / MMfreeLM inference from the command line.
//
// Consolidated text-in / text-out tool. The common case is a text prompt:
//   mmfree-cli "The capital of France is"           # -> generated text on stdout
//   echo "Once upon a time," | mmfree-cli           # prompt from stdin
// It tokenizes (embedded LLaMA BPE, mmfree/tokenizer.hpp), generates greedily with the
// recurrent-state cache, stops at </s> or --gen, and decodes the new tokens back to text.
//
// A raw-ids path is kept for validation/debugging (tools/compare_prompts.py):
//   mmfree-cli --ids 1,415,310 --gen 8              # ids in -> full id stream out
// In --ids mode there is no tokenizer dependency and no EOS stop (exactly --gen tokens),
// so the greedy stream stays directly comparable to the reference.
//
// The flag reference is kUsage below (printed by `--help`), so there is one copy of it.
#include "mmfree/bench.hpp"
#include "mmfree/model.hpp"
#include "mmfree/profile.hpp"
#include "mmfree/tokenizer.hpp"
#include "mmfree/weights.hpp"

#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <vector>

using namespace mmfree;

static const char kUsage[] =
    R"(mmfree-cli — MatMul-Free LM (HGRN-Bit) inference, text in / text out.

usage: mmfree-cli [flags] ["prompt text"]
       echo "prompt" | mmfree-cli [flags]

  mmfree-cli "The capital of France is" --gen 32
  mmfree-cli --bench --gen 128 --reps 5          # timing run, fixed prompt
  mmfree-cli --ids 1,415,310 --gen 8             # raw ids in/out (no tokenizer, no EOS)

model files
  --blob PATH        weight blob (default: $MMFREE_BLOB, else model.mmfree looked up in
                     the working dir, then next to the executable and its parent dirs)
  --tokenizer PATH   tokenizer blob (default: $MMFREE_TOKENIZER, else tokenizer.mmtok
                     beside the weight blob)
  Both are produced by tools/provision.py — see the README.

generation
  --gen N            max new tokens                        (default 32)
  --temp T           sampling temperature, 0 = greedy       (default 0)
  --top-k K          sample from the K highest logits, 0 = full vocab (needs --temp > 0)
  --seed N           RNG seed for sampling                  (default: random each run)
  --no-bos           do not prepend BOS in text mode
  --no-eos           do not stop at EOS in text mode

numerics
  --mode float|fixed activation numerics (default fixed = the Q5.10 FPGA datapath)
  --frac N           fixed-point fraction bits              (default 10)

output / validation
  --print-ids        print the full id stream instead of decoded text
  --show-ids         also print prompt/gen ids to stderr
  --decode-ids a,b,c decode ids to text and exit (no model run)
  --logits-out PATH  write last-position logits of the full stream as raw float32

benchmarking
  --bench            time generation instead of printing text; reports prefill and decode
                     throughput. Greedy with EOS disabled, so every run produces exactly
                     --gen tokens (stable, comparable timing).
  --warmup N         untimed warmup runs before measuring   (default 1)
  --reps N           timed runs to average over             (default 3)
  --profile          per-op wall-clock breakdown (matmul vs rmsnorm/scan/...). Works alone
                     or with --bench.

  -h, --help         this message
)";

static bool file_exists(const std::string& p) {
  std::ifstream f(p, std::ios::binary);
  return f.good();
}

// Directory holding the running executable, derived from argv[0] ("" if argv[0] carries no
// path — e.g. an installed binary found through PATH; the working-dir candidates and
// --blob/$MMFREE_BLOB still cover that case).
static std::string exe_dir(const char* argv0) {
  std::string p = argv0 ? argv0 : "";
  std::size_t slash = p.find_last_of('/');
  return (slash == std::string::npos) ? "" : p.substr(0, slash);
}

// Find `name` in the usual places so the engine runs from wherever it was built or
// installed, not only from the directory holding the blob. In build-tree layouts the
// binary sits in cpp/build/ (or cpp/build/Release/) while the blob sits in cpp/, hence
// walking a couple of parents. Returns "" if nothing matched.
static std::string search_model_file(const std::string& name, const char* argv0) {
  std::vector<std::string> dirs = {"."};
  std::string ed = exe_dir(argv0);
  if (!ed.empty()) {
    dirs.push_back(ed);
    dirs.push_back(ed + "/..");
    dirs.push_back(ed + "/../..");
  }
  for (const std::string& d : dirs) {
    std::string cand = d + "/" + name;
    if (file_exists(cand)) return cand;
  }
  return "";
}

static const char* env_or_null(const char* k) {
  const char* v = std::getenv(k);
  return (v && *v) ? v : nullptr;
}

static std::vector<std::int64_t> parse_ids(const std::string& s) {
  std::vector<std::int64_t> ids;
  std::size_t i = 0;
  while (i < s.size()) {
    while (i < s.size() && !(std::isdigit(s[i]) || s[i] == '-')) ++i;
    if (i >= s.size()) break;
    std::size_t j = i;
    if (s[j] == '-') ++j;
    while (j < s.size() && std::isdigit(s[j])) ++j;
    ids.push_back(std::stoll(s.substr(i, j - i)));
    i = j;
  }
  return ids;
}

// "<dir>/tokenizer.mmtok" alongside the weight blob.
static std::string default_tokenizer_path(const std::string& blob) {
  std::size_t slash = blob.find_last_of('/');
  std::string dir = (slash == std::string::npos) ? "" : blob.substr(0, slash + 1);
  return dir + "tokenizer.mmtok";
}

static int run(int argc, char** argv) {
  std::string blob, tokenizer_path, mode = "fixed", ids_arg, logits_out;
  std::string prompt, decode_arg;
  bool have_prompt = false, no_bos = false, no_eos = false, print_ids = false,
       show_ids = false, bench = false, profile = false;
  int frac = 10;
  int bench_warmup = 1, bench_reps = 3;
  std::size_t gen = 32;
  Sampling samp;
  bool have_seed = false;

  for (int a = 1; a < argc; ++a) {
    std::string k = argv[a];
    auto next = [&]() { return (a + 1 < argc) ? argv[++a] : ""; };
    if (k == "--blob") blob = next();
    else if (k == "--tokenizer") tokenizer_path = next();
    else if (k == "--mode") mode = next();
    else if (k == "--frac") frac = std::atoi(next());
    else if (k == "--gen") gen = static_cast<std::size_t>(std::atoll(next()));
    else if (k == "--temp" || k == "--temperature") samp.temperature = std::atof(next());
    else if (k == "--top-k") samp.top_k = std::atoi(next());
    else if (k == "--seed") { samp.seed = std::strtoull(next(), nullptr, 10); have_seed = true; }
    else if (k == "--ids") ids_arg = next();
    else if (k == "--decode-ids") decode_arg = next();
    else if (k == "--no-bos") no_bos = true;
    else if (k == "--no-eos") no_eos = true;
    else if (k == "--print-ids") print_ids = true;
    else if (k == "--show-ids") show_ids = true;
    else if (k == "--logits-out") logits_out = next();
    else if (k == "--bench") bench = true;
    else if (k == "--warmup") bench_warmup = std::atoi(next());
    else if (k == "--reps") bench_reps = std::atoi(next());
    else if (k == "--profile") profile = true;
    else if (k == "--help" || k == "-h") { std::fputs(kUsage, stdout); return 0; }
    else if (!k.empty() && k[0] == '-') {
      std::fprintf(stderr, "unknown arg: %s\n(try --help)\n", k.c_str());
      return 2;
    } else {
      prompt = k;  // first positional = prompt text
      have_prompt = true;
    }
  }

  // Resolve the model files: explicit flag > environment > search. Report a miss here
  // rather than letting Weights/Tokenizer throw "cannot open model.mmfree", because at
  // this point we still know WHERE we looked and what would fix it.
  //
  // The weight blob is only *required* by the paths that actually run the model -- the
  // tokenizer-only paths (--decode-ids, `--gen 0 --print-ids`) must keep working on a
  // machine that has never provisioned one.
  const bool need_blob = decode_arg.empty() && (gen > 0 || !logits_out.empty() || bench);
  if (blob.empty()) {
    if (const char* e = env_or_null("MMFREE_BLOB")) {
      blob = e;
    } else {
      blob = search_model_file("model.mmfree", argv[0]);
      if (blob.empty() && need_blob) {
        std::fprintf(stderr,
                     "error: no weight blob found (looked for model.mmfree in the working "
                     "directory and next to the executable).\n"
                     "  provision one:  python tools/provision.py\n"
                     "  or point at it: mmfree-cli --blob /path/to/model.mmfree  "
                     "(or set MMFREE_BLOB)\n");
        return 2;
      }
    }
  }
  if (tokenizer_path.empty()) {
    if (const char* e = env_or_null("MMFREE_TOKENIZER")) {
      tokenizer_path = e;
    } else {
      // Beside the blob first; then the same search the blob used, so a build-tree
      // tokenizer.mmtok is found even when --blob points into a model zoo (or when there
      // is no blob at all). Falling back to the bare name lets Tokenizer report the miss.
      if (!blob.empty()) tokenizer_path = default_tokenizer_path(blob);
      if (tokenizer_path.empty() || !file_exists(tokenizer_path)) {
        std::string found = search_model_file("tokenizer.mmtok", argv[0]);
        if (!found.empty()) tokenizer_path = found;
        else if (tokenizer_path.empty()) tokenizer_path = "tokenizer.mmtok";
      }
    }
  }

  // --decode-ids: pure tokenizer decode, no weights/model needed.
  if (!decode_arg.empty()) {
    Tokenizer t(tokenizer_path);
    std::printf("%s\n", t.decode(parse_ids(decode_arg)).c_str());
    return 0;
  }

  const bool ids_mode = !ids_arg.empty();

  // Load the tokenizer only when we actually need text encode/decode.
  const bool ids_out = print_ids || ids_mode;
  const bool need_tok = !ids_mode || !ids_out;
  std::unique_ptr<Tokenizer> tok;
  if (need_tok) tok = std::make_unique<Tokenizer>(tokenizer_path);

  // Resolve the prompt ids.
  std::vector<std::int64_t> ids;
  if (ids_mode) {
    ids = parse_ids(ids_arg);
  } else {
    if (!have_prompt) {
      if (bench) {  // standalone benchmark: use a fixed prompt instead of blocking on stdin
        prompt = "The quick brown fox jumps over the lazy dog.";
      } else {  // read prompt from stdin
        std::string all, line;
        while (std::getline(std::cin, line)) all += line + "\n";
        while (!all.empty() && (all.back() == '\n' || all.back() == ' ')) all.pop_back();
        prompt = all;
      }
    }
    ids = tok->encode(prompt, !no_bos);
  }
  if (ids.empty()) {
    std::fprintf(stderr, "no input (pass a prompt, --ids, or pipe via stdin)\n");
    return 2;
  }

  // Build the model only when we actually run it (skip for pure tokenization, e.g.
  // `--gen 0 --print-ids`, so no weight blob is needed for that path).
  ActQuant aq = (mode == "fixed") ? ActQuant::FixedQ510 : ActQuant::Float;
  std::unique_ptr<Weights> w;
  std::unique_ptr<Model> model;
  if (gen > 0 || !logits_out.empty() || bench) {
    w = std::make_unique<Weights>(blob);
    model = std::make_unique<Model>(*w, aq, frac);
  }

  // ---- benchmark mode: time generation, report prefill/decode tok/s, then exit ----
  if (bench) {
    if (gen == 0) { std::fprintf(stderr, "--bench needs --gen > 0\n"); return 2; }
    BenchOpts opts;
    opts.gen = gen;
    opts.warmup = bench_warmup;
    opts.reps = bench_reps;
    opts.profile = profile;
    run_bench(*model, ids, opts, blob.c_str(), mode.c_str(), frac);
    return 0;
  }

  // Sampling: a random seed each run unless --seed pins it (greedy ignores the seed).
  if (samp.temperature > 0.0f && !have_seed) samp.seed = std::random_device{}();

  // Stream decoded text token-by-token (text-output mode only). BPE decode isn't safe
  // per-token (▁->space, leading-space strip, multi-byte/byte-fallback fusion), so we
  // re-decode the generated-so-far each step and emit only the new byte suffix -- always
  // byte-correct because decode is prefix-stable for this tokenizer.
  const bool stream_text = !ids_out;
  std::vector<std::int64_t> shown_ids;
  std::string shown;
  std::function<void(std::int64_t)> on_token;
  if (stream_text) {
    on_token = [&](std::int64_t id) {
      shown_ids.push_back(id);
      std::string full = tok->decode(shown_ids);
      if (full.size() > shown.size()) {
        std::fwrite(full.data() + shown.size(), 1, full.size() - shown.size(), stdout);
        std::fflush(stdout);
      }
      shown = std::move(full);
    };
  }

  // Generate. EOS stop only in text mode (ids mode stays exactly --gen for comparison).
  if (profile) { Profiler::instance().reset(); Profiler::instance().enable(true); }
  std::int64_t eos = (!ids_mode && !no_eos && tok) ? tok->eos_id() : -1;
  std::vector<std::int64_t> stream =
      (gen > 0) ? model->generate(ids, gen, eos, samp, on_token) : ids;
  if (profile) Profiler::instance().enable(false);
  if (stream_text && gen > 0) std::fputc('\n', stdout);  // finish the streamed line

  if (!logits_out.empty()) {
    std::vector<float> logits;
    model->forward(stream.data(), stream.size(), logits);
    std::size_t V = model->config().vocab_size;
    const float* last = logits.data() + (stream.size() - 1) * V;
    std::ofstream f(logits_out, std::ios::binary);
    f.write(reinterpret_cast<const char*>(last), static_cast<std::streamsize>(V * sizeof(float)));
  }

  std::vector<std::int64_t> gen_ids(stream.begin() + static_cast<std::ptrdiff_t>(ids.size()),
                                    stream.end());
  if (show_ids) {
    std::fprintf(stderr, "prompt ids (%zu):", ids.size());
    for (auto t : ids) std::fprintf(stderr, " %lld", static_cast<long long>(t));
    std::fprintf(stderr, "\ngen ids (%zu):", gen_ids.size());
    for (auto t : gen_ids) std::fprintf(stderr, " %lld", static_cast<long long>(t));
    std::fprintf(stderr, "\n");
  }

  if (ids_out) {
    // Full id stream (prompt + generated), space-separated -- the validation format.
    for (std::size_t i = 0; i < stream.size(); ++i)
      std::printf("%lld%s", static_cast<long long>(stream[i]),
                  i + 1 < stream.size() ? " " : "\n");
  } else if (gen == 0) {
    // Nothing was streamed (no generation) -- emit the (empty) decode for consistency.
    std::printf("%s\n", tok->decode(gen_ids).c_str());
  }
  // (text mode with gen > 0 was already streamed above)
  if (profile && gen > 0) print_profile(1);
  return 0;
}

int main(int argc, char** argv) {
  // Weights/Tokenizer report bad or missing blobs by throwing; catching here turns what
  // was an uncaught-exception abort into one actionable line.
  try {
    return run(argc, argv);
  } catch (const std::exception& e) {
    std::string msg = e.what();
    std::fprintf(stderr, "error: %s\n", msg.c_str());
    if (msg.compare(0, 9, "weights: ") == 0 || msg.compare(0, 11, "tokenizer: ") == 0)
      std::fprintf(stderr,
                   "hint: the model files come from `python tools/provision.py` "
                   "(see the README); pass --blob/--tokenizer to point elsewhere.\n");
    return 1;
  }
}
