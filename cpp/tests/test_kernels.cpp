// test_kernels.cpp — validate scalar kernels against the golden oracles
// emitted by tools/dump_golden.py. No test framework; plain asserts + a report.
#include "mmfree/kernels.hpp"
#include "npy.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace mmfree;

static std::string G;  // golden dir

static npy::Array load(const std::string& name) { return npy::load(G + "/" + name + ".npy"); }

struct Metrics {
  double max_abs = 0;
  double rel_l2 = 0;
};

static Metrics compare(const std::vector<float>& got, const std::vector<float>& ref) {
  double max_abs = 0, num = 0, den = 0;
  for (std::size_t i = 0; i < ref.size(); ++i) {
    double d = static_cast<double>(got[i]) - ref[i];
    max_abs = std::max(max_abs, std::abs(d));
    num += d * d;
    den += static_cast<double>(ref[i]) * ref[i];
  }
  return {max_abs, std::sqrt(num / (den + 1e-30))};
}

static int failures = 0;
static void check(const std::string& name, const Metrics& m, double rel_tol, double abs_tol) {
  bool ok = (m.rel_l2 <= rel_tol) || (m.max_abs <= abs_tol);
  std::printf("  %-32s rel_l2=%.3e  max_abs=%.3e  %s\n", name.c_str(), m.rel_l2, m.max_abs,
              ok ? "PASS" : "FAIL");
  if (!ok) ++failures;
}

// ---- individual kernel tests against layer-0 golden ----------------------------------

static void test_rmsnorm_via_bitlinear_norm() {
  // RMSNorm is exercised inside bitlinear; check its norm_out against i_proj.norm.
  auto in = load("model.layers.0.i_proj.in");      // [1,T,1024]
  auto nw = load("model.layers.0.i_proj.normw");    // [1024]
  auto wq = load("model.layers.0.i_proj.wq");        // int8 [1024,1024]
  auto sw = load("model.layers.0.i_proj.scale_w");   // [1]
  auto ref_norm = load("model.layers.0.i_proj.norm");
  std::size_t in_dim = nw.shape[0], rows = in.numel() / in_dim, out_dim = wq.shape[0];

  std::vector<float> out(rows * out_dim), norm(rows * in_dim);
  std::vector<int8_t> w(wq.i64.begin(), wq.i64.end());
  bitlinear(out.data(), in.f32.data(), nw.f32.data(), w.data(), sw.f32[0], rows, in_dim,
            out_dim, 1e-6f, norm.data());
  check("rmsnorm (i_proj.norm)", compare(norm, ref_norm.f32), 1e-5, 1e-4);
}

static void test_bitlinear(const std::string& tag, std::size_t in_dim_expected) {
  auto in = load(tag + ".in");
  auto nw = load(tag + ".normw");
  auto wq = load(tag + ".wq");
  auto sw = load(tag + ".scale_w");
  auto ref_acc = load(tag + ".acc");
  auto ref_out = load(tag + ".out");
  std::size_t in_dim = nw.shape[0], rows = in.numel() / in_dim, out_dim = wq.shape[0];
  (void)in_dim_expected;

  std::vector<float> out(rows * out_dim);
  std::vector<int8_t> w(wq.i64.begin(), wq.i64.end());
  bitlinear(out.data(), in.f32.data(), nw.f32.data(), w.data(), sw.f32[0], rows, in_dim,
            out_dim, 1e-6f, nullptr);
  check("bitlinear " + tag.substr(tag.rfind('.') + 1) + ".out", compare(out, ref_out.f32),
        1e-4, 1e-3);

  // also verify the float accumulator: acc = out * scale_w
  std::vector<float> acc(out.size());
  for (std::size_t i = 0; i < out.size(); ++i) acc[i] = out[i] * sw.f32[0];
  check("bitlinear " + tag.substr(tag.rfind('.') + 1) + ".acc", compare(acc, ref_acc.f32),
        1e-4, 1e-2);
}

// Fixed-point (Q5.10) BitLinear vs a fixed-point golden dir. Validates the int16 quant
// + integer accumulate path matches the torch "fixedp" reference bit-closely.
static void test_bitlinear_fixed(const std::string& gdir, const std::string& tag) {
  auto load_g = [&](const std::string& n) { return npy::load(gdir + "/" + n + ".npy"); };
  auto in = load_g(tag + ".in");
  auto nw = load_g(tag + ".normw");
  auto wq = load_g(tag + ".wq");
  auto sw = load_g(tag + ".scale_w");
  auto ref_out = load_g(tag + ".out");
  std::size_t in_dim = nw.shape[0], rows = in.numel() / in_dim, out_dim = wq.shape[0];

  std::vector<float> out(rows * out_dim);
  std::vector<int8_t> w(wq.i64.begin(), wq.i64.end());
  bitlinear(out.data(), in.f32.data(), nw.f32.data(), w.data(), sw.f32[0], rows, in_dim,
            out_dim, 1e-6f, nullptr, ActQuant::FixedQ510, 10);
  check("fixedQ510 " + tag.substr(tag.rfind('.') + 1) + ".out", compare(out, ref_out.f32), 1e-4,
        1e-3);
}

static void test_swiglu() {
  // mlp: swiglu(gate, y) where [gate|y] = chunk(gate_proj.out); expected = down_proj.in.
  auto gp = load("model.layers.0.gate_proj.out");   // [1,T,5632]
  auto ref = load("model.layers.0.down_proj.in");    // [1,T,2816]
  std::size_t half = ref.shape.back();
  std::size_t rows = ref.numel() / half;
  std::size_t full = gp.shape.back();

  std::vector<float> got(rows * half), a(rows * half), b(rows * half);
  for (std::size_t r = 0; r < rows; ++r) {
    for (std::size_t c = 0; c < half; ++c) {
      a[r * half + c] = gp.f32[r * full + c];
      b[r * half + c] = gp.f32[r * full + half + c];
    }
  }
  swiglu(got.data(), a.data(), b.data(), got.size());
  check("swiglu (down_proj.in)", compare(got, ref.f32), 1e-5, 1e-4);
}

static void test_hgrn_scan() {
  auto si = load("model.layers.0.scan_i");  // [1,1,T,D]
  auto sf = load("model.layers.0.scan_f");
  auto ref = load("model.layers.0.recurrence");  // [1,T,D]
  std::size_t D = si.shape.back();
  std::size_t T = si.numel() / D;
  std::vector<float> got(T * D);
  hgrn_scan(got.data(), si.f32.data(), sf.f32.data(), T, D, nullptr);
  check("hgrn_scan (recurrence)", compare(got, ref.f32), 1e-5, 1e-4);
}

static void test_rmsnorm_swishgate() {
  // g_norm(g_proj.out, recurrence) -> o_proj.in
  auto g = load("model.layers.0.g_proj.out");
  auto o = load("model.layers.0.recurrence");
  auto w = load("model.layers.0.g_norm.w");
  auto ref = load("model.layers.0.o_proj.in");
  std::size_t cols = w.shape[0], rows = ref.numel() / cols;
  std::vector<float> got(rows * cols);
  rmsnorm_swishgate(got.data(), g.f32.data(), o.f32.data(), w.f32.data(), rows, cols, 1e-6f);
  check("rmsnorm_swishgate (o_proj.in)", compare(got, ref.f32), 1e-5, 1e-4);
}

int main(int argc, char** argv) {
  G = (argc > 1) ? argv[1] : "golden";
  std::printf("golden dir: %s\n", G.c_str());

  std::printf("[kernels]\n");
  test_rmsnorm_via_bitlinear_norm();
  test_bitlinear("model.layers.0.i_proj", 1024);
  test_bitlinear("model.layers.0.f_proj", 1024);
  test_bitlinear("model.layers.0.g_proj", 1024);
  test_bitlinear("model.layers.0.o_proj", 1024);
  test_bitlinear("model.layers.0.gate_proj", 1024);
  test_bitlinear("model.layers.0.down_proj", 2816);
  test_bitlinear("lm_head", 1024);
  test_swiglu();
  test_hgrn_scan();
  test_rmsnorm_swishgate();

  // Optional fixed-point (Q5.10) golden dir as argv[2].
  if (argc > 2) {
    std::string gfixed = argv[2];
    std::printf("[fixed-point Q5.10] golden dir: %s\n", gfixed.c_str());
    for (const char* tag : {"model.layers.0.i_proj", "model.layers.0.f_proj",
                            "model.layers.0.g_proj", "model.layers.0.o_proj",
                            "model.layers.0.gate_proj", "model.layers.0.down_proj", "lm_head"})
      test_bitlinear_fixed(gfixed, tag);
  }

  std::printf("\n%s (%d failure%s)\n", failures ? "FAILED" : "ALL PASSED", failures,
              failures == 1 ? "" : "s");
  return failures ? 1 : 0;
}
