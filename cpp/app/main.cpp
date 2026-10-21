// main.cpp — mmfree-cli: HGRN-Bit / MMfreeLM inference from the command line.
//
// Text in / text out: tokenize (mmfree/tokenizer.hpp), generate with the recurrent-state
// cache, stop at </s> or --gen, stream the decoded tokens. `--ids` is the raw-ids
// validation path: no tokenizer and no EOS stop, so exactly --gen tokens.
// The flag reference is kUsage below (printed by --help).
#include "cli_common.hpp"
#include "mmfree/bench.hpp"
#include "mmfree/model.hpp"
#include "mmfree/profile.hpp"
#include "mmfree/tokenizer.hpp"
#include "mmfree/weights.hpp"

#include <cctype>
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
  mmfree-cli --model 2.7B "The capital of France is"
  mmfree-cli --bench --gen 128 --reps 5          # timing run, fixed prompt
  mmfree-cli --ids 1,415,310 --gen 8             # raw ids in/out (no tokenizer, no EOS)
  mmfree-cli --bench --batch 8 --gen 128         # batched (8-stream) throughput

model files
  --model SIZE       370M | 1.3B | 2.7B: use blobs/<SIZE>/ (default: the smallest
                     provisioned). `make model MODEL=<SIZE>` provisions one.
  --blob PATH        weight blob, instead of --model (also: $MMFREE_BLOB)
  --tokenizer PATH   tokenizer blob (default: $MMFREE_TOKENIZER, else tokenizer.mmtok
                     beside the weight blob)

generation
  --gen N            max new tokens                        (default 32)
  --temp T           sampling temperature, 0 = greedy       (default 0)
  --top-k K          sample from the K highest logits, 0 = full vocab (needs --temp > 0)
  --top-p P          sample from the smallest set of tokens holding mass P (needs
                     --temp > 0; 1 = off, the default)
  --repeat-penalty R discourage tokens seen in the last --repeat-last-n (default 64)
                     tokens; 1 = off (default), ~1.1-1.3 curbs the loops these models fall into
  --seed N           RNG seed for sampling                  (default: random each run)
  --no-bos           do not prepend BOS in text mode
  --no-eos           do not stop at EOS in text mode
  --batch B          decode B concurrent streams in lockstep (default 1). Each stream gets
                     its own recurrent state and is prefilled independently; the B decode
                     steps then share ONE matmul per projection, so each weight row is
                     fetched once and reused across the batch instead of the matrix being
                     re-swept per stream. Lossless: batched results are bit-identical to
                     B separate runs. With --bench this reports aggregate and per-stream
                     throughput; otherwise every stream's text is printed, labelled.
                     B > 1 also batches prefill, so prefill timings shift with it.

numerics
  --mode float|fixed activation numerics (default fixed = the Q5.10 FPGA datapath)
  --frac N           fixed-point fraction bits              (default 10)
  --weights FMT      in-memory ternary weight format        (default int8)
                       int8    one int8 per weight, straight from the blob
                       1.6bit  TQ1_0 base-3 packing, 5 trits/byte (1.625 bits/weight)
                       2bit    2 bits/weight
                     The packed formats are lossless, so results are bit-identical to int8;
                     they trade a one-off repack at first use for ~5x/4x less weight traffic
                     per token. `fixed` mode only (float mode ignores this).

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

static int run(int argc, char** argv) {
  std::string blob, size, tokenizer_path, mode = "fixed", ids_arg, logits_out, wfmt = "int8";
  std::string prompt, decode_arg;
  bool have_prompt = false, no_bos = false, no_eos = false, print_ids = false,
       show_ids = false, bench = false, profile = false;
  int frac = 10;
  int bench_warmup = 1, bench_reps = 3;
  std::size_t gen = 32;
  std::size_t batch = 1;
  Sampling samp;
  bool have_seed = false;

  for (int a = 1; a < argc; ++a) {
    std::string k = argv[a];
    auto next = [&]() { return (a + 1 < argc) ? argv[++a] : ""; };
    if (k == "--blob") blob = next();
    else if (k == "--model" || k == "-m") size = next();
    else if (k == "--tokenizer") tokenizer_path = next();
    else if (k == "--mode") mode = next();
    else if (k == "--weights") wfmt = next();
    else if (k == "--frac") frac = std::atoi(next());
    else if (k == "--gen") gen = static_cast<std::size_t>(std::atoll(next()));
    else if (k == "--temp" || k == "--temperature") samp.temperature = std::atof(next());
    else if (k == "--top-k") samp.top_k = std::atoi(next());
    else if (k == "--top-p") samp.top_p = std::strtof(next(), nullptr);
    else if (k == "--repeat-penalty") samp.repeat_penalty = std::strtof(next(), nullptr);
    else if (k == "--repeat-last-n") samp.repeat_last_n = std::strtoull(next(), nullptr, 10);
    else if (k == "--seed") { samp.seed = std::strtoull(next(), nullptr, 10); have_seed = true; }
    else if (k == "--ids") ids_arg = next();
    else if (k == "--decode-ids") decode_arg = next();
    else if (k == "--no-bos") no_bos = true;
    else if (k == "--no-eos") no_eos = true;
    else if (k == "--print-ids") print_ids = true;
    else if (k == "--show-ids") show_ids = true;
    else if (k == "--logits-out") logits_out = next();
    else if (k == "--batch") batch = static_cast<std::size_t>(std::atoll(next()));
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

  // Validate the enum-valued flags before touching a multi-GB blob.
  if (mode != "fixed" && mode != "float") {
    std::fprintf(stderr, "error: unknown --mode %s (expected fixed or float)\n", mode.c_str());
    return 2;
  }
  if (!cli::valid_weight_format(wfmt)) {
    std::fprintf(stderr, "error: unknown --weights %s (expected int8, 1.6bit, or 2bit)\n",
                 wfmt.c_str());
    return 2;
  }
  if (batch < 1) {
    std::fprintf(stderr, "error: --batch must be >= 1 (got %zu)\n", batch);
    return 2;
  }

  // Resolve the model files: flag > environment > search. A missing blob is reported here,
  // where we still know where we looked; the tokenizer-only paths (--decode-ids,
  // `--gen 0 --print-ids`) must work without one.
  const bool need_blob = decode_arg.empty() && (gen > 0 || !logits_out.empty() || bench);
  cli::ModelPaths mp;
  if (int rc = cli::resolve_model(blob, size, tokenizer_path, argv[0], need_blob,
                                  "mmfree-cli", mp))
    return rc;
  blob = mp.blob;
  tokenizer_path = mp.tokenizer;

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

  // Build the model only when it runs (pure tokenization needs no weight blob).
  ActQuant aq = (mode == "fixed") ? ActQuant::FixedQ510 : ActQuant::Float;
  std::unique_ptr<Weights> w;
  std::unique_ptr<Model> model;
  if (gen > 0 || !logits_out.empty() || bench) {
    w = std::make_unique<Weights>(blob);
    model = std::make_unique<Model>(*w, aq, frac);
  }

  // ---- --weights / --batch: install a TernaryBackend (cli_common.hpp) ----
  cli::Backends backends;
  if (model) cli::install_backend(*model, wfmt, aq, batch, backends);

  // ---- benchmark mode: time generation, report prefill/decode tok/s, then exit ----
  if (bench) {
    if (gen == 0) { std::fprintf(stderr, "--bench needs --gen > 0\n"); return 2; }
    BenchOpts opts;
    opts.gen = gen;
    opts.warmup = bench_warmup;
    opts.reps = bench_reps;
    opts.profile = profile;
    opts.batch = batch;
    run_bench(*model, ids, opts, blob.c_str(), mode.c_str(), frac);
    return 0;
  }

  // Sampling: a random seed each run unless --seed pins it (greedy ignores the seed).
  if (samp.temperature > 0.0f && !have_seed) samp.seed = std::random_device{}();

  // ---- --batch B (non-bench): decode B streams in lockstep and print each -------------
  // Per-token streaming can't interleave B streams, so each prints once complete. With one
  // prompt + greedy every stream must come out identical (a cheap self-check).
  if (batch > 1 && gen > 0) {
    std::int64_t eos_b = (!ids_mode && !no_eos && tok) ? tok->eos_id() : -1;
    std::vector<std::vector<std::int64_t>> prompts(batch, ids);
    if (profile) { Profiler::instance().reset(); Profiler::instance().enable(true); }
    std::vector<std::vector<std::int64_t>> streams =
        model->generate_serve(prompts, gen, eos_b, samp);
    if (profile) Profiler::instance().enable(false);

    for (std::size_t s = 0; s < streams.size(); ++s) {
      std::printf("[stream %zu] ", s);
      if (ids_out) {
        for (std::size_t i = 0; i < streams[s].size(); ++i)
          std::printf("%s%lld", i ? " " : "", static_cast<long long>(streams[s][i]));
        std::printf("\n");
      } else {
        std::vector<std::int64_t> gen_only(
            streams[s].begin() + static_cast<std::ptrdiff_t>(ids.size()), streams[s].end());
        std::printf("%s\n", tok->decode(gen_only).c_str());
      }
    }
    if (profile) print_profile(1);
    return 0;
  }

  // Stream decoded text token by token. BPE pieces don't decode independently, so re-decode
  // the generated-so-far each step and emit the new byte suffix (decode is prefix-stable).
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
  try {
    return run(argc, argv);
  } catch (const std::exception& e) {
    return cli::report_error(e);
  }
}
