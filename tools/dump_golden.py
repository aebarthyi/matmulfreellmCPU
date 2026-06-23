# -*- coding: utf-8 -*-
"""
Dump golden tensors from the pure-torch fp32 reference for C++ unit tests.

Emits a directory of .npy oracles plus a manifest.json:
  * per-op inputs/outputs for layer 0 (every kernel: norms, projections incl. the
    int32 accumulator, recurrence, swiglu, gate),
  * end-to-end final_norm + logits + greedy token stream.

The C++ tests load these to check each kernel bit/1e-5-exact, and the accumulator
oracle (*.acc.npy, integer-valued) is the integer-exactness target for bitlinear.

Usage:
  python3 dump_golden.py [--out ../cpp/golden] [--prompt-ids 1,512,297,263,2913,9138]
                         [--new-tokens 8]
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

import reference as R
from reference import Capture, greedy_generate, load_reference, weight_quant_int


def save(out_dir: str, name: str, t: torch.Tensor, manifest: dict):
    arr = t.detach().cpu().numpy().astype(np.float32)
    # NOTE: in the chosen "triton" numerics, the projection accumulator (*.acc) is a
    # FLOAT sum of signed (non-rounded) activations over ternary lanes -- not an integer.
    path = os.path.join(out_dir, name + ".npy")
    np.save(path, arr)
    manifest[name] = {"shape": list(arr.shape), "dtype": str(arr.dtype)}


def save_raw(out_dir, name, arr, manifest):
    np.save(os.path.join(out_dir, name + ".npy"), arr)
    manifest[name] = {"shape": list(arr.shape), "dtype": str(arr.dtype)}


def dump_weights(ref, out_dir, manifest):
    """Export the weights needed for self-contained layer-0 + lm_head kernel tests:
    per-projection norm weight (.normw), ternary weight (.wq int8) + scale_w (.scale_w),
    standalone norm weights, and lower_bound slices for layers 0 and 1."""
    import torch
    sd = ref.sd
    # (golden tag, linear weight key, per-projection norm weight key)
    projs = [
        ("model.layers.0.i_proj", "model.layers.0.attn.i_proj.weight", "model.layers.0.attn.i_proj.norm.weight"),
        ("model.layers.0.f_proj", "model.layers.0.attn.f_proj.weight", "model.layers.0.attn.f_proj.norm.weight"),
        ("model.layers.0.g_proj", "model.layers.0.attn.g_proj.weight", "model.layers.0.attn.g_proj.norm.weight"),
        ("model.layers.0.o_proj", "model.layers.0.attn.o_proj.weight", "model.layers.0.attn.o_proj.norm.weight"),
        ("model.layers.0.gate_proj", "model.layers.0.mlp.gate_proj.weight", "model.layers.0.mlp.gate_proj.norm.weight"),
        ("model.layers.0.down_proj", "model.layers.0.mlp.down_proj.weight", "model.layers.0.mlp.down_proj.norm.weight"),
        ("lm_head", "lm_head.weight", "lm_head.norm.weight"),
    ]
    for tag, wkey, nkey in projs:
        wq, scale_w = weight_quant_int(sd[wkey])
        save_raw(out_dir, tag + ".wq", wq.to(torch.int8).cpu().numpy().astype(np.int8), manifest)
        save_raw(out_dir, tag + ".scale_w", np.float32(scale_w.item()).reshape(1), manifest)
        save_raw(out_dir, tag + ".normw", sd[nkey].cpu().numpy().astype(np.float32), manifest)

    # standalone norm weights
    for tag, key in [
        ("model.layers.0.attn_norm", "model.layers.0.attn_norm.weight"),
        ("model.layers.0.mlp_norm", "model.layers.0.mlp_norm.weight"),
        ("model.layers.0.g_norm", "model.layers.0.attn.g_norm.weight"),
        ("model.norm", "model.norm.weight"),
    ]:
        save_raw(out_dir, tag + ".w", sd[key].cpu().numpy().astype(np.float32), manifest)

    # precomputed lower_bound slices (softmax+cumsum-lb[0]) for layers 0 and 1
    lb = sd["model.lower_bounds"].softmax(0)
    lb = lb.cumsum(0) - lb[0]
    save_raw(out_dir, "lower_bound.0", lb[0].cpu().numpy().astype(np.float32), manifest)
    save_raw(out_dir, "lower_bound.1", lb[1].cpu().numpy().astype(np.float32), manifest)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "cpp", "golden"))
    ap.add_argument("--prompt-ids", default="1,512,297,263,2913,9138")
    ap.add_argument("--new-tokens", type=int, default=8)
    ap.add_argument("--mode", default="triton", choices=("triton", "int16", "fixedp", "bitnet"),
                    help="activation numerics for the BitLinear projection (reference QUANT_MODE)")
    args = ap.parse_args()

    R.QUANT_MODE = args.mode

    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)

    torch.manual_seed(0)
    ref, cfg, ckpt = load_reference()
    ids = torch.tensor([[int(x) for x in args.prompt_ids.split(",")]])

    manifest: dict = {
        "checkpoint": ckpt,
        "mode": args.mode,
        "fixed_frac_bits": R.FIXED_FRAC_BITS,
        "config": {
            "hidden_size": cfg.hidden_size,
            "num_hidden_layers": cfg.num_hidden_layers,
            "num_heads": cfg.num_heads,
            "head_dim": cfg.head_dim,
            "inter_size": cfg.inter_size,
            "vocab_size": cfg.vocab_size,
            "rms_norm_eps": cfg.rms_norm_eps,
            "use_lower_bound": cfg.use_lower_bound,
        },
        "prompt_ids": ids[0].tolist(),
        "tensors": {},
    }

    # ---- per-op capture on the real prompt ----
    cap = Capture(enabled=True)
    logits = ref.forward(ids, cap)

    # keep a representative subset: all of layer 0 + final norm + logits + lm_head
    keep_prefixes = ("model.layers.0.", "final_norm", "logits", "lm_head")
    for name, t in cap.store.items():
        if name.startswith(keep_prefixes):
            save(out_dir, name, t, manifest["tensors"])
    save(out_dir, "input_ids", ids.float(), manifest["tensors"])
    dump_weights(ref, out_dir, manifest["tensors"])

    # ---- end-to-end greedy token stream ----
    gen = greedy_generate(ref, ids, args.new_tokens)
    np.save(os.path.join(out_dir, "greedy_ids.npy"), gen[0].cpu().numpy().astype(np.int64))
    manifest["greedy_ids"] = gen[0].tolist()

    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"wrote {len(manifest['tensors'])} golden tensors to {out_dir}")
    print(f"greedy stream: {manifest['greedy_ids']}")


if __name__ == "__main__":
    main()
