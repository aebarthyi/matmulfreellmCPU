// chat.cpp — mmfree-chat: a live, interactive session with a MatMul-Free LM.
//
// Type text and the model continues it, streamed token by token. The released checkpoints
// are base LMs (no chat template), so this is completion-style: by default each line starts
// a new text; `/continue on` keeps one running text across turns. HGRN is recurrent, so a
// running text costs nothing to keep: its state carries over and only new tokens are fed.
#include "cli_common.hpp"
#include "mmfree/model.hpp"
#include "mmfree/tokenizer.hpp"
#include "mmfree/weights.hpp"

#include <fcntl.h>
#include <poll.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <memory>
#include <random>
#include <string>
#include <vector>

using namespace mmfree;

namespace {

const char kUsage[] =
    R"(mmfree-chat — interactive MatMul-Free LM (HGRN-Bit) session.

usage: mmfree-chat [--model 370M|1.3B|2.7B] [flags]

  Type text and press Enter; the model continues it. An empty line keeps generating
  from where the last reply stopped. Ctrl-C stops a reply, Ctrl-D (or /quit) exits.
  Commands are listed by /help inside the session.

model files
  --model SIZE       370M | 1.3B | 2.7B: use blobs/<SIZE>/ (default: the smallest
                     provisioned). `make model MODEL=<SIZE>` provisions one.
  --blob PATH        weight blob, instead of --model (also: $MMFREE_BLOB)
  --tokenizer PATH   tokenizer blob (default: $MMFREE_TOKENIZER, else beside the blob)

generation
  --gen N            max tokens per reply                   (default 128)
  --temp T           sampling temperature, 0 = greedy        (default 0.6)
  --top-p P          sample from the smallest set of tokens holding mass P
                                                             (default 0.4, 1 = off)
  --top-k K          sample from the K most likely tokens    (default 0 = all)
  --repeat-penalty R discourage tokens among the last --repeat-last-n (default 64)
                                                             (default 1.2, 1 = off)
  --seed N           RNG seed                                (default: random)
  --continue         keep one running text across turns      (default: each line is fresh)

numerics
  --mode float|fixed activation numerics                     (default fixed)
  --frac N           fixed-point fraction bits               (default 10)
  --weights FMT      int8 | 1.6bit | 2bit                    (default 2bit, the fastest;
                     all three are bit-identical)
  --no-warmup        skip the startup step that repacks weights before the first reply

  -h, --help         this message
)";

const char kCommands[] =
    R"(  /help              this list
  /quit, /exit       leave (Ctrl-D works too)
  /reset             forget the running text
  /continue on|off   keep one running text across turns, or start fresh each line
  /temp T            sampling temperature, 0 = greedy
  /topk K            sample from the K most likely tokens, 0 = all
  /topp P            sample from the smallest set of tokens holding mass P, 1 = off
  /penalty R         repetition penalty over recent tokens, 1 = off
  /gen N             max tokens per reply
  /seed N            reseed the sampler
  /settings          show the current settings
  (empty line)       keep generating from where the last reply stopped
  line ending in \   continue the input on the next line
  Ctrl-C             stop the current reply
)";

// Ctrl-C sets g_interrupt, which the decode loop polls between tokens, and writes to a
// self-pipe that wakes the prompt's poll(). EINTR cannot be relied on for the prompt: the
// signal may be delivered to an idle OpenMP worker instead of the thread blocked in read().
std::atomic<bool> g_interrupt{false};
int g_wake[2] = {-1, -1};

extern "C" void on_sigint(int) {
  g_interrupt = true;
  if (g_wake[1] >= 0) {
    const char c = 0;
    ssize_t r = write(g_wake[1], &c, 1);
    (void)r;
  }
}

void install_sigint() {
  if (pipe(g_wake) == 0) {
    fcntl(g_wake[0], F_SETFL, O_NONBLOCK);
    fcntl(g_wake[1], F_SETFL, O_NONBLOCK);
  }
  struct sigaction sa {};
  sa.sa_handler = on_sigint;
  sigemptyset(&sa.sa_mask);
  sa.sa_flags = SA_RESTART;
  sigaction(SIGINT, &sa, nullptr);
}

void clear_interrupt() {
  char b[64];
  if (g_wake[0] >= 0)
    while (read(g_wake[0], b, sizeof b) > 0) {}
  g_interrupt = false;
}

// Lines from stdin that a Ctrl-C can interrupt (see on_sigint).
class LineReader {
 public:
  enum Result { kLine, kInterrupted, kEof };

  Result next(std::string& line) {
    for (;;) {
      const std::size_t nl = buf_.find('\n');
      if (nl != std::string::npos) {
        line = buf_.substr(0, nl);
        buf_.erase(0, nl + 1);
        return kLine;
      }
      if (eof_) {
        if (buf_.empty()) return kEof;
        line.swap(buf_);
        buf_.clear();
        return kLine;
      }
      pollfd fds[2] = {{STDIN_FILENO, POLLIN, 0}, {g_wake[0], POLLIN, 0}};
      if (poll(fds, g_wake[0] >= 0 ? 2 : 1, -1) < 0) {
        if (errno != EINTR) eof_ = true;
        continue;
      }
      if (fds[1].revents & POLLIN) {
        clear_interrupt();
        buf_.clear();
        return kInterrupted;
      }
      if (fds[0].revents & (POLLIN | POLLHUP | POLLERR)) {
        char tmp[4096];
        const ssize_t n = read(STDIN_FILENO, tmp, sizeof tmp);
        if (n > 0) buf_.append(tmp, static_cast<std::size_t>(n));
        else if (n == 0 || errno != EINTR) eof_ = true;
      }
    }
  }

 private:
  std::string buf_;
  bool eof_ = false;
};

// Sampling defaults chosen on 2.7B: plain temperature / top-k sampling drifts into word salad
// and top-p alone loops; top-p 0.4 at temperature 0.6 (upstream's demo settings) plus a 1.2
// repetition penalty stays coherent.
struct Settings {
  std::size_t gen = 128;
  Sampling samp = [] {
    Sampling s;
    s.temperature = 0.6f;
    s.top_p = 0.4f;
    s.repeat_penalty = 1.2f;
    return s;
  }();
  bool cont = false;
};

struct Style {
  const char* dim = "";
  const char* off = "";
};

class Session {
 public:
  Session(Model& m, const Tokenizer& tok, Settings& s, Style st, std::uint64_t seed)
      : m_(m), tok_(tok), s_(s), st_(st), pick_(seed) {}

  void reseed(std::uint64_t seed) { pick_ = Sampler(seed); }

  void reset() {
    m_.reset_state();
    fed_ = 0;
    pending_ = -1;
    tail_.clear();
  }

  std::size_t context() const { return fed_; }

  // One turn: feed `text` (empty = keep going) and stream the reply to stdout.
  void turn(const std::string& text) {
    const std::int64_t eos = tok_.eos_id();
    std::vector<std::int64_t> in;
    if (text.empty()) {
      if (pending_ < 0 || pending_ == eos) {
        std::fprintf(stderr, "%s(nothing to continue — type some text)%s\n", st_.dim, st_.off);
        return;
      }
      in.push_back(pending_);
    } else {
      if (!s_.cont || pending_ == eos) reset();
      std::vector<std::int64_t> ids = tok_.encode(text, /*add_bos=*/fed_ == 0);
      if (pending_ >= 0) in.push_back(pending_);
      in.insert(in.end(), ids.begin(), ids.end());
      tail_.insert(tail_.end(), ids.begin(), ids.end());
    }

    using clock = std::chrono::steady_clock;
    clear_interrupt();
    const auto t0 = clock::now();
    const std::vector<float>* lg = &m_.feed(in.data(), in.size());
    fed_ += in.size();
    pending_ = -1;
    const auto t1 = clock::now();

    // Stream text as the suffix of decode(context + reply) beyond decode(context): BPE
    // pieces carry their own leading space, which only decodes right with a left context.
    std::vector<std::int64_t> ctx = tail_;
    std::size_t shown = tok_.decode(ctx).size();
    bool at_line_start = true;  // the reply starts on a fresh line: drop one leading space
    std::size_t n = 0;
    bool hit_eos = false;
    for (;;) {
      const std::size_t from =
          ctx.size() > s_.samp.repeat_last_n ? ctx.size() - s_.samp.repeat_last_n : 0;
      const std::int64_t t =
          pick_(lg->data(), lg->size(), s_.samp, ctx.data() + from, ctx.size() - from);
      ++n;
      if (t == eos) { hit_eos = true; pending_ = t; break; }
      ctx.push_back(t);
      const std::string full = tok_.decode(ctx);
      if (full.size() > shown) {
        const char* p = full.data() + shown;
        std::size_t len = full.size() - shown;
        if (at_line_start && *p == ' ') { ++p; --len; }
        if (len) at_line_start = false;
        std::fwrite(p, 1, len, stdout);
        std::fflush(stdout);
      }
      shown = full.size();
      if (n >= s_.gen || g_interrupt) { pending_ = t; break; }
      lg = &m_.feed(&t, 1);
      ++fed_;
    }
    const auto t2 = clock::now();
    if (!at_line_start) std::fputc('\n', stdout);
    std::fflush(stdout);

    // Keep a left context for the next turn: decode spacing and the repetition window.
    const std::size_t keep = std::max<std::size_t>(16, s_.samp.repeat_last_n);
    tail_.assign(ctx.size() > keep ? ctx.end() - keep : ctx.begin(), ctx.end());

    auto ms = [](clock::duration d) {
      return std::chrono::duration<double, std::milli>(d).count();
    };
    const double dec_ms = ms(t2 - t1);
    std::fprintf(stderr, "%s[%zu tok", st_.dim, n);
    if (n > 1 && dec_ms > 0) std::fprintf(stderr, " · %.1f tok/s", (n - 1) * 1000.0 / dec_ms);
    std::fprintf(stderr, " · prefill %zu tok in %.0f ms", in.size(), ms(t1 - t0));
    if (s_.cont) std::fprintf(stderr, " · context %zu", fed_);
    if (hit_eos) std::fprintf(stderr, " · end of text");
    if (g_interrupt) std::fprintf(stderr, " · stopped");
    std::fprintf(stderr, "]%s\n", st_.off);
    clear_interrupt();
  }

 private:
  Model& m_;
  const Tokenizer& tok_;
  Settings& s_;
  Style st_;
  Sampler pick_;
  std::size_t fed_ = 0;              // tokens in the recurrent state since reset()
  std::int64_t pending_ = -1;        // last sampled token, not yet fed to the model
  std::vector<std::int64_t> tail_;   // recent ids: decode context + repetition window
};

void print_settings(const Settings& s, std::uint64_t seed) {
  std::fprintf(stderr, "temp %g · top-k %d · top-p %g · repeat penalty %g · up to %zu tokens "
                       "per reply · seed %llu · %s\n",
               s.samp.temperature, s.samp.top_k, s.samp.top_p, s.samp.repeat_penalty, s.gen,
               static_cast<unsigned long long>(seed),
               s.cont ? "one running text (/continue off to start fresh each line)"
                      : "each line starts fresh (/continue on to keep a running text)");
}

// Read one logical input line ('\\' at the end continues it). Returns false on EOF; a
// Ctrl-C at the prompt sets `interrupted` and discards the partial input.
bool read_input(LineReader& in, std::string& out, bool tty, bool& interrupted) {
  out.clear();
  interrupted = false;
  const char* prompt = "> ";
  for (;;) {
    if (tty) { std::fputs(prompt, stderr); std::fflush(stderr); }
    std::string line;
    const LineReader::Result r = in.next(line);
    if (r == LineReader::kInterrupted) {
      if (tty) std::fputc('\n', stderr);
      interrupted = true;
      out.clear();
      return true;
    }
    if (r == LineReader::kEof) return !out.empty();
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (!line.empty() && line.back() == '\\') {
      line.pop_back();
      out += line + "\n";
      prompt = "… ";
      continue;
    }
    out += line;
    return true;
  }
}

bool parse_size(const std::string& s, std::size_t& v) {
  char* end = nullptr;
  const unsigned long long x = std::strtoull(s.c_str(), &end, 10);
  if (s.empty() || *end) return false;
  v = static_cast<std::size_t>(x);
  return true;
}

// Handle a /command. Returns false when the session should end.
bool command(const std::string& line, Session& sess, Settings& s, std::uint64_t& seed) {
  const std::size_t sp = line.find(' ');
  const std::string cmd = line.substr(0, sp);
  const std::string arg = (sp == std::string::npos) ? "" : line.substr(sp + 1);
  std::size_t n = 0;
  if (cmd == "/quit" || cmd == "/exit") return false;
  if (cmd == "/help") {
    std::fputs(kCommands, stderr);
  } else if (cmd == "/reset") {
    sess.reset();
    std::fputs("(running text cleared)\n", stderr);
  } else if (cmd == "/continue" && (arg == "on" || arg == "off")) {
    s.cont = (arg == "on");
    if (!s.cont) sess.reset();
    print_settings(s, seed);
  } else if (cmd == "/temp" && !arg.empty()) {
    s.samp.temperature = std::max(0.0f, std::strtof(arg.c_str(), nullptr));
    print_settings(s, seed);
  } else if (cmd == "/topp" && !arg.empty()) {
    s.samp.top_p = std::min(1.0f, std::max(0.0f, std::strtof(arg.c_str(), nullptr)));
    print_settings(s, seed);
  } else if (cmd == "/penalty" && !arg.empty()) {
    s.samp.repeat_penalty = std::max(1.0f, std::strtof(arg.c_str(), nullptr));
    print_settings(s, seed);
  } else if (cmd == "/topk" && parse_size(arg, n)) {
    s.samp.top_k = static_cast<int>(n);
    print_settings(s, seed);
  } else if (cmd == "/gen" && parse_size(arg, n) && n > 0) {
    s.gen = n;
    print_settings(s, seed);
  } else if (cmd == "/seed" && parse_size(arg, n)) {
    seed = n;
    sess.reseed(seed);
    print_settings(s, seed);
  } else if (cmd == "/settings") {
    print_settings(s, seed);
  } else {
    std::fprintf(stderr, "unknown command or bad argument: %s  (/help lists them)\n",
                 line.c_str());
  }
  return true;
}

int run(int argc, char** argv) {
  std::string blob, size, tokenizer_path, mode = "fixed", wfmt = "2bit";
  int frac = 10;
  bool warmup = true, have_seed = false;
  std::uint64_t seed = 0;
  Settings s;

  for (int a = 1; a < argc; ++a) {
    const std::string k = argv[a];
    auto next = [&]() { return (a + 1 < argc) ? argv[++a] : ""; };
    if (k == "--blob") blob = next();
    else if (k == "--model" || k == "-m") size = next();
    else if (k == "--tokenizer") tokenizer_path = next();
    else if (k == "--mode") mode = next();
    else if (k == "--weights") wfmt = next();
    else if (k == "--frac") frac = std::atoi(next());
    else if (k == "--gen") s.gen = static_cast<std::size_t>(std::atoll(next()));
    else if (k == "--temp" || k == "--temperature") s.samp.temperature = std::atof(next());
    else if (k == "--top-k") s.samp.top_k = std::atoi(next());
    else if (k == "--top-p") s.samp.top_p = std::strtof(next(), nullptr);
    else if (k == "--repeat-penalty") s.samp.repeat_penalty = std::strtof(next(), nullptr);
    else if (k == "--repeat-last-n") s.samp.repeat_last_n = std::strtoull(next(), nullptr, 10);
    else if (k == "--seed") { seed = std::strtoull(next(), nullptr, 10); have_seed = true; }
    else if (k == "--continue") s.cont = true;
    else if (k == "--no-warmup") warmup = false;
    else if (k == "--help" || k == "-h") { std::fputs(kUsage, stdout); return 0; }
    else {
      std::fprintf(stderr, "unknown arg: %s\n(try --help)\n", k.c_str());
      return 2;
    }
  }
  if (mode != "fixed" && mode != "float") {
    std::fprintf(stderr, "error: unknown --mode %s (expected fixed or float)\n", mode.c_str());
    return 2;
  }
  if (!cli::valid_weight_format(wfmt)) {
    std::fprintf(stderr, "error: unknown --weights %s (expected int8, 1.6bit, or 2bit)\n",
                 wfmt.c_str());
    return 2;
  }
  if (s.gen == 0) s.gen = 1;
  if (!have_seed) seed = std::random_device{}();

  cli::ModelPaths mp;
  if (int rc = cli::resolve_model(blob, size, tokenizer_path, argv[0], /*need_blob=*/true,
                                  "mmfree-chat", mp))
    return rc;
  blob = mp.blob;
  tokenizer_path = mp.tokenizer;

  Weights w(blob);  // first, so a bad --blob is reported as such, not as a missing tokenizer
  Tokenizer tok(tokenizer_path);
  const ActQuant aq = (mode == "fixed") ? ActQuant::FixedQ510 : ActQuant::Float;
  Model model(w, aq, frac);
  cli::Backends backends;
  cli::install_backend(model, wfmt, aq, /*batch=*/1, backends);

  const bool tty_in = isatty(STDIN_FILENO);
  const bool tty_err = isatty(STDERR_FILENO);
  Style st;
  if (tty_err && !std::getenv("NO_COLOR")) { st.dim = "\033[2m"; st.off = "\033[0m"; }

  const Config& c = model.config();
  const std::string name = mp.size.empty() ? blob : "MMfreeLM-" + mp.size + "  (" + blob + ")";
  std::fprintf(stderr, "mmfree-chat · %s\n%s%zu layers · hidden %zu · vocab %zu · %s weights · "
                       "%s activations%s\n",
               name.c_str(), st.dim, c.num_hidden_layers, c.hidden_size, c.vocab_size,
               aq == ActQuant::FixedQ510 ? wfmt.c_str() : "int8",
               aq == ActQuant::FixedQ510 ? "fixed-point" : "float", st.off);
  if (warmup) {  // first touch pages in the blob and repacks packed weights: pay it now
    if (tty_err) { std::fprintf(stderr, "%swarming up…%s", st.dim, st.off); std::fflush(stderr); }
    const std::int64_t bos = tok.bos_id();
    model.feed(&bos, 1);
    model.reset_state();
    if (tty_err) std::fputs("\r\033[K", stderr);
  }
  print_settings(s, seed);
  std::fprintf(stderr, "Type text and the model continues it. Empty line = keep going, "
                       "/help = commands, Ctrl-D = quit.\n");

  install_sigint();
  Session sess(model, tok, s, st, seed);
  LineReader in;
  std::string line;
  bool interrupted = false;
  while (read_input(in, line, tty_in, interrupted)) {
    if (interrupted) {
      if (tty_in) std::fprintf(stderr, "%s(Ctrl-D or /quit to exit)%s\n", st.dim, st.off);
      continue;
    }
    if (!line.empty() && line[0] == '/') {
      if (!command(line, sess, s, seed)) break;
      continue;
    }
    sess.turn(line);
  }
  if (tty_in) std::fputc('\n', stderr);
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    return run(argc, argv);
  } catch (const std::exception& e) {
    return cli::report_error(e);
  }
}
