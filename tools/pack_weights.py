# -*- coding: utf-8 -*-
"""
Pack the HGRN-Bit / MMfreeLM checkpoint into a single mmap-friendly `model.mmfree`
blob for the C++ CPU runtime. OFFLINE step (runs on the build host, needs torch).

Numerics: ternary weights are baked here once via reference.weight_quant_int
(scale_w = 1/mean(|W|); wq = clamp(round(W*scale_w), -1, 1)). The C++ side never
quantizes weights -- it loads wq (int8 {-1,0,+1}) + scale_w and runs the matmul-free
projection. Per-projection norm weights, standalone norms, embeddings (fp32), the
precomputed lower_bounds (softmax(0).cumsum(0)-lb[0]), final norm and lm_head are
all baked in too. This is the same numerics as tools/reference.py (QUANT_MODE=triton).

Blob layout:
  [8]   magic        "MMFREE1\n"
  [8]   u64          header_len  (little-endian)
  [hl]  header JSON  {"config":{...}, "tensors":{name:{dtype,shape,offset,nbytes}}}
  pad to 64
  data section: each tensor 64-byte aligned; `offset` is relative to data-section start.

Usage:
  python pack_weights.py [--out ../cpp/model.mmfree]
"""
from __future__ import annotations

import argparse
import json
import os
import struct

import numpy as np
import torch

from reference import load_reference, weight_quant_int

MAGIC = b"MMFREE1\n"
ALIGN = 64

# the six ternary projections per layer: (blob tag suffix, reference weight key infix)
LAYER_PROJS = [
    ("i_proj", "attn.i_proj"),
    ("f_proj", "attn.f_proj"),
    ("g_proj", "attn.g_proj"),
    ("o_proj", "attn.o_proj"),
    ("gate_proj", "mlp.gate_proj"),
    ("down_proj", "mlp.down_proj"),
]


def _align(n: int) -> int:
    return (n + ALIGN - 1) // ALIGN * ALIGN


def add_f32(tensors: dict, name: str, t: torch.Tensor):
    tensors[name] = ("f32", np.ascontiguousarray(t.detach().cpu().numpy().astype(np.float32)))


def add_i8(tensors: dict, name: str, t: torch.Tensor):
    tensors[name] = ("i8", np.ascontiguousarray(t.detach().cpu().numpy().astype(np.int8)))


def collect(ref) -> dict:
    """Return {name: (dtype_str, np.ndarray)} for every tensor the runtime needs."""
    sd = ref.sd
    cfg = ref.cfg
    tensors: dict = {}

    # global: embeddings, final norm, lm_head (ternary), lower_bounds (precomputed)
    add_f32(tensors, "model.embeddings", sd["model.embeddings.weight"])
    add_f32(tensors, "model.norm.w", sd["model.norm.weight"])

    wq, scale_w = weight_quant_int(sd["lm_head.weight"])
    add_i8(tensors, "lm_head.wq", wq)
    add_f32(tensors, "lm_head.scale_w", scale_w.reshape(1))
    add_f32(tensors, "lm_head.normw", sd["lm_head.norm.weight"])

    if cfg.use_lower_bound:
        lb = sd["model.lower_bounds"].softmax(0)
        lb = lb.cumsum(0) - lb[0]
        add_f32(tensors, "lower_bounds", lb)  # [L, H]

    # per-layer
    for l in range(cfg.num_hidden_layers):
        p = f"model.layers.{l}"
        add_f32(tensors, f"{p}.attn_norm.w", sd[f"{p}.attn_norm.weight"])
        add_f32(tensors, f"{p}.mlp_norm.w", sd[f"{p}.mlp_norm.weight"])
        add_f32(tensors, f"{p}.g_norm.w", sd[f"{p}.attn.g_norm.weight"])
        for tag, infix in LAYER_PROJS:
            w = sd[f"{p}.{infix}.weight"]
            wq, scale_w = weight_quant_int(w)
            add_i8(tensors, f"{p}.{tag}.wq", wq)
            add_f32(tensors, f"{p}.{tag}.scale_w", scale_w.reshape(1))
            add_f32(tensors, f"{p}.{tag}.normw", sd[f"{p}.{infix}.norm.weight"])

    return tensors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "cpp", "model.mmfree"))
    args = ap.parse_args()
    out = os.path.abspath(args.out)

    ref, cfg, ckpt = load_reference()
    print(f"loaded checkpoint: {ckpt}")
    tensors = collect(ref)

    # assign offsets (relative to data-section start), 64-byte aligned
    meta: dict = {}
    offset = 0
    for name, (dtype, arr) in tensors.items():
        nbytes = arr.nbytes
        meta[name] = {"dtype": dtype, "shape": list(arr.shape), "offset": offset, "nbytes": nbytes}
        offset = _align(offset + nbytes)
    data_len = offset

    header = {
        "config": {
            "vocab_size": cfg.vocab_size,
            "hidden_size": cfg.hidden_size,
            "num_hidden_layers": cfg.num_hidden_layers,
            "num_heads": cfg.num_heads,
            "expand_ratio": cfg.expand_ratio,
            "intermediate_size": cfg.inter_size,
            "use_lower_bound": cfg.use_lower_bound,
            "use_short_conv": cfg.use_short_conv,
            "rms_norm_eps": cfg.rms_norm_eps,
        },
        "tensors": meta,
    }
    header_bytes = json.dumps(header).encode("utf-8")
    data_start = _align(len(MAGIC) + 8 + len(header_bytes))

    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<Q", len(header_bytes)))
        f.write(header_bytes)
        f.write(b"\0" * (data_start - f.tell()))
        assert f.tell() == data_start
        base = f.tell()
        for name, (dtype, arr) in tensors.items():
            off = base + meta[name]["offset"]
            f.write(b"\0" * (off - f.tell()))
            f.write(arr.tobytes())
        # pad tail to full data_len so the file size is exactly data_start+data_len
        f.write(b"\0" * (base + data_len - f.tell()))

    total = data_start + data_len
    print(f"wrote {len(tensors)} tensors -> {out} ({total/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
