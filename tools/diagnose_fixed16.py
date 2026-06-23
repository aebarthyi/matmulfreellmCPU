# -*- coding: utf-8 -*-
"""
Evaluate 16-bit fixed-point activations against the fp32 "triton" golden.

Answers two questions empirically:
  1. Dynamic range: what is max|y| over every BitLinear input (RMSNorm output) across
     all layers? Per-row (sets the int16 dynamic scale) and global (sets a static
     Qm.f binary point). This decides whether a STATIC fixed point can work.
  2. Fidelity: do the 16-bit modes ("int16" per-row, "fixedp" static) still produce
     the SAME greedy token stream and tight logits vs the fp32 "triton" reference?
"""
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import reference as R
from reference import Capture, load_reference, greedy_generate

IDS = torch.tensor([[1, 512, 297, 263, 2913, 9138]])
NEW = 8


def logit_stats(a, b):
    a, b = a.float().reshape(-1), b.float().reshape(-1)
    d = (a - b).abs()
    cos = torch.nn.functional.cosine_similarity(a, b, dim=0).item()
    rel = (d.pow(2).sum() / (b.pow(2).sum() + 1e-30)).sqrt().item()
    return d.max().item(), rel, cos


def main():
    torch.manual_seed(0)
    ref, cfg, ckpt = load_reference()
    print(f"checkpoint: {ckpt}")

    # ---- 1. activation dynamic range over all BitLinear inputs ----
    R.QUANT_MODE = "triton"
    cap = Capture(enabled=True)
    base_logits = ref.forward(IDS, cap)
    norms = {k: v for k, v in cap.store.items() if k.endswith(".norm")}
    global_max = 0.0
    per_row_max = []
    print(f"\n== BitLinear-input ranges over {len(norms)} captured projections (layer0+lm_head) ==")
    for k in sorted(norms):
        y = norms[k]
        rmax = y.abs().amax(dim=-1)        # per-row (per-token) max
        per_row_max.append(rmax.reshape(-1))
        gm = y.abs().max().item()
        global_max = max(global_max, gm)
    per_row_max = torch.cat(per_row_max)
    print(f"  global max|y|           = {global_max:.4f}")
    print(f"  per-row max|y|: min={per_row_max.min():.4f}  "
          f"mean={per_row_max.mean():.4f}  max={per_row_max.max():.4f}")
    # int16 step sizes
    print(f"  int16 per-row step (max/32767)   ~ {per_row_max.max().item()/32767:.3e} .. "
          f"{per_row_max.min().item()/32767:.3e}")
    for fb in (10, 11, 12, 13):
        rng = 32768 / (1 << fb)
        print(f"  fixedp Q.{fb}: range +-{rng:.2f}  step={1.0/(1<<fb):.3e}  "
              f"{'OVERFLOW' if rng < global_max else 'ok'}")

    # ---- 2. fidelity of each mode vs triton (logits + greedy stream) ----
    print("\n== logits vs fp32 triton (full 6-token prompt) ==")
    for mode in ("int16", "fixedp", "bitnet"):
        R.QUANT_MODE = mode
        lg = ref.forward(IDS)
        mx, rel, cos = logit_stats(lg, base_logits)
        print(f"  {mode:8s}  max|Δ|={mx:.4e}  rel_l2={rel:.4e}  cos={cos:.8f}")

    print("\n== greedy token stream (prompt + 8) ==")
    R.QUANT_MODE = "triton"
    g_triton = greedy_generate(ref, IDS, NEW)[0].tolist()
    print(f"  triton : {g_triton}")
    for mode in ("int16", "fixedp", "bitnet"):
        R.QUANT_MODE = mode
        g = greedy_generate(ref, IDS, NEW)[0].tolist()
        match = "MATCH" if g == g_triton else "DIFFERS"
        print(f"  {mode:7s}: {g}   {match}")


if __name__ == "__main__":
    main()
