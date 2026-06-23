# -*- coding: utf-8 -*-
"""
Determine which numerics mode reproduces the real triton-backed model, and
quantify the gap between the matmul-free modes and the (buggy) full-precision path.

Runs the HF model TWICE:
  * without prepare_for_inference  -> fused kernel applies weight_quant (ternary),
    no activation round  == reference mode "triton"
  * with    prepare_for_inference  -> raw full-precision weights (the bug)
    == reference mode "fullprec"
and compares each to the matching reference mode + cross-compares.
"""
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
import reference as R
from reference import load_reference, DEFAULT_MODEL


def stats(a, b, label):
    a, b = a.float().reshape(-1), b.float().reshape(-1)
    d = (a - b).abs()
    cos = torch.nn.functional.cosine_similarity(a, b, dim=0).item()
    print(f"  {label:40s} max|Δ|={d.max():.4e}  mean|Δ|={d.mean():.4e}  cos={cos:.8f}")


def hf_logits(prepare):
    import mmfreelm  # noqa
    from transformers import AutoModelForCausalLM
    from mmfreelm.ops.fusedbitnet import BitLinear
    m = AutoModelForCausalLM.from_pretrained(DEFAULT_MODEL, device_map="cpu").float().eval()
    if prepare:
        for mod in m.modules():
            if isinstance(mod, BitLinear):
                mod.prepare_for_inference()
    with torch.no_grad():
        return m(IDS).logits.float()


IDS = torch.tensor([[1, 512, 297, 263, 2913, 9138]])


def main():
    torch.manual_seed(0)
    ref, cfg, ckpt = load_reference()

    ref_modes = {}
    for mode in ("triton", "bitnet", "fullprec"):
        R.QUANT_MODE = mode
        ref_modes[mode] = ref.forward(IDS).float()

    print("== HF WITHOUT prepare_for_inference (fused weight_quant ternary, no act round) ==")
    hf_noprep = hf_logits(prepare=False)
    stats(ref_modes["triton"], hf_noprep, "ref[triton]   vs hf[no-prepare]")
    stats(ref_modes["bitnet"], hf_noprep, "ref[bitnet]   vs hf[no-prepare]")

    print("== HF WITH prepare_for_inference (generate.py path; full-precision bug) ==")
    hf_prep = hf_logits(prepare=True)
    stats(ref_modes["fullprec"], hf_prep, "ref[fullprec] vs hf[prepare]")

    print("== how different is matmul-free vs the full-precision path? ==")
    stats(ref_modes["triton"], ref_modes["fullprec"], "ref[triton]   vs ref[fullprec]")
    stats(ref_modes["bitnet"], ref_modes["fullprec"], "ref[bitnet]   vs ref[fullprec]")
    stats(hf_noprep, hf_prep, "hf[no-prepare] vs hf[prepare]")

    for tag, lg in [("ref[triton]", ref_modes["triton"]), ("ref[bitnet]", ref_modes["bitnet"]),
                    ("hf[no-prep]", hf_noprep), ("hf[prepare]", hf_prep)]:
        print(f"  argmax {tag:12s}: {lg[0].argmax(-1).tolist()}")


if __name__ == "__main__":
    main()
