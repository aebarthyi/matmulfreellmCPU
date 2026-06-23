# -*- coding: utf-8 -*-
"""
Validate the pure-torch fp32 reference (tools/reference.py) against the real
triton-backed HF model. Confirms the reference is a faithful golden oracle.

NOTE: the first HF forward JIT-compiles the triton-cpu kernels (slow on first run).
"""
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch

from reference import load_reference, DEFAULT_MODEL


def main():
    torch.manual_seed(0)
    prompt_ids = torch.tensor([[1, 512, 297, 263, 2913, 9138]])

    print("[ref] loading pure-torch reference ...", flush=True)
    ref, cfg, ckpt = load_reference()
    ref_logits = ref.forward(prompt_ids).float()
    print(f"[ref] logits {tuple(ref_logits.shape)} argmax(last)={int(ref_logits[0,-1].argmax())}", flush=True)

    print("[hf] loading HF model (fp32) + triton kernels ...", flush=True)
    import mmfreelm  # noqa: registers the architecture
    from transformers import AutoModelForCausalLM
    from mmfreelm.ops.fusedbitnet import BitLinear

    model = AutoModelForCausalLM.from_pretrained(DEFAULT_MODEL, device_map="cpu").float()
    model.eval()
    for m in model.modules():
        if isinstance(m, BitLinear):
            m.prepare_for_inference()

    print("[hf] running forward (JIT compiling first time) ...", flush=True)
    with torch.no_grad():
        hf_logits = model(prompt_ids).logits.float()
    print(f"[hf] logits {tuple(hf_logits.shape)} argmax(last)={int(hf_logits[0,-1].argmax())}", flush=True)

    d = (ref_logits - hf_logits).abs()
    cos = torch.nn.functional.cosine_similarity(
        ref_logits.reshape(-1), hf_logits.reshape(-1), dim=0)
    ref_arg = ref_logits[0].argmax(-1)
    hf_arg = hf_logits[0].argmax(-1)
    print("\n========= REFERENCE vs HF =========")
    print(f"max|Δ logits|     : {d.max().item():.6e}")
    print(f"mean|Δ logits|    : {d.mean().item():.6e}")
    print(f"cosine similarity : {cos.item():.8f}")
    print(f"argmax per pos    : ref={ref_arg.tolist()}  hf={hf_arg.tolist()}")
    print(f"argmax agreement  : {(ref_arg == hf_arg).float().mean().item()*100:.1f}%")


if __name__ == "__main__":
    main()
