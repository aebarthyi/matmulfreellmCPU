# -*- coding: utf-8 -*-
"""
Provision the C++ engine from a Hugging Face checkpoint — one script, all three
offline steps:

  1. download the MMfreeLM checkpoint from the HF Hub (weights + tokenizer.json)
  2. pack the weights   -> cpp/model.mmfree     (ternary quant, baked once)
  3. pack the tokenizer -> cpp/tokenizer.mmtok  (BPE vocab + merges)

After this, the C++ engine runs with no Python at all (see ../README.md). The
numerics match tools/reference.py (QUANT_MODE=triton): scale_w = 1/mean(|W|),
wq = clamp(round(W*scale_w), -1, 1); the C++ side never quantizes weights.

Usage:
  pip install -r tools/requirements.txt
  python tools/provision.py                                   # ridger/MMfreeLM-370M -> ../cpp
  python tools/provision.py --model ridger/MMfreeLM-1.3B
  python tools/provision.py --out-dir /path/to/cpp
  python tools/provision.py --skip-download                   # checkpoint already cached
  python tools/provision.py --weights-only | --tokenizer-only # one artifact

Gated/private repos: run `huggingface-cli login` first (the 370M model is public).
torch is imported lazily — `--tokenizer-only` needs only the stdlib.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import struct
import sys

MMFREE_MAGIC = b"MMFREE1\n"      # weight blob magic (matches cpp/src/io/weights.cpp)
MMTOK_MAGIC = b"MMTOK1\n\x00"    # tokenizer magic (matches cpp/src/io/tokenizer.cpp)
ALIGN = 64
DEFAULT_MODEL = "ridger/MMfreeLM-370M"

# the six ternary projections per layer: (blob tag suffix, reference weight key infix)
LAYER_PROJS = [
    ("i_proj", "attn.i_proj"),
    ("f_proj", "attn.f_proj"),
    ("g_proj", "attn.g_proj"),
    ("o_proj", "attn.o_proj"),
    ("gate_proj", "mlp.gate_proj"),
    ("down_proj", "mlp.down_proj"),
]


def _resolve_checkpoint(model: str) -> str:
    """Locate a checkpoint dir: a local path, else the latest HF cache snapshot."""
    if os.path.isdir(model):
        return model
    cache = os.path.expanduser(
        f"~/.cache/huggingface/hub/models--{model.replace('/', '--')}/snapshots/*"
    )
    snaps = sorted(glob.glob(cache))
    if not snaps:
        raise FileNotFoundError(
            f"checkpoint for {model!r} not found locally — run without --skip-download")
    return snaps[-1]


def _align(n: int) -> int:
    return (n + ALIGN - 1) // ALIGN * ALIGN


# ── 1. download ──────────────────────────────────────────────────────────────
def download(model: str) -> str:
    from huggingface_hub import snapshot_download  # lazy: only needed to download
    # Pull just what the packers read; the rest of the repo (pytorch_model.bin,
    # generation config, etc.) isn't needed by the C++ path.
    path = snapshot_download(
        repo_id=model,
        allow_patterns=["*.safetensors", "*.json", "tokenizer.*", "*.model"],
    )
    return path


# ── 2. pack weights (needs torch; imported lazily) ───────────────────────────
def pack_weights(out: str, model: str = DEFAULT_MODEL) -> str:
    """Quantize + pack `model`'s checkpoint into `out` (model.mmfree). The checkpoint
    must already be in the HF cache (download() above). Returns the output path."""
    import numpy as np
    import torch  # noqa: F401  (reference loads torch tensors)
    from reference import load_reference, weight_quant_int

    out = os.path.abspath(out)
    ref, cfg, ckpt = load_reference(model)
    print(f"loaded checkpoint: {ckpt}")
    sd = ref.sd

    tensors: dict = {}

    def add_f32(name, t):
        tensors[name] = ("f32", np.ascontiguousarray(t.detach().cpu().numpy().astype(np.float32)))

    def add_i8(name, t):
        tensors[name] = ("i8", np.ascontiguousarray(t.detach().cpu().numpy().astype(np.int8)))

    # global: embeddings, final norm, lm_head (ternary), lower_bounds (precomputed)
    add_f32("model.embeddings", sd["model.embeddings.weight"])
    add_f32("model.norm.w", sd["model.norm.weight"])
    wq, scale_w = weight_quant_int(sd["lm_head.weight"])
    add_i8("lm_head.wq", wq)
    add_f32("lm_head.scale_w", scale_w.reshape(1))
    add_f32("lm_head.normw", sd["lm_head.norm.weight"])
    if cfg.use_lower_bound:
        lb = sd["model.lower_bounds"].softmax(0)
        lb = lb.cumsum(0) - lb[0]
        add_f32("lower_bounds", lb)  # [L, H]

    # per-layer
    for l in range(cfg.num_hidden_layers):
        p = f"model.layers.{l}"
        add_f32(f"{p}.attn_norm.w", sd[f"{p}.attn_norm.weight"])
        add_f32(f"{p}.mlp_norm.w", sd[f"{p}.mlp_norm.weight"])
        add_f32(f"{p}.g_norm.w", sd[f"{p}.attn.g_norm.weight"])
        for tag, infix in LAYER_PROJS:
            w = sd[f"{p}.{infix}.weight"]
            wq, scale_w = weight_quant_int(w)
            add_i8(f"{p}.{tag}.wq", wq)
            add_f32(f"{p}.{tag}.scale_w", scale_w.reshape(1))
            add_f32(f"{p}.{tag}.normw", sd[f"{p}.{infix}.norm.weight"])

    # offsets (relative to data-section start), 64-byte aligned
    meta: dict = {}
    offset = 0
    for name, (dtype, arr) in tensors.items():
        meta[name] = {"dtype": dtype, "shape": list(arr.shape), "offset": offset, "nbytes": arr.nbytes}
        offset = _align(offset + arr.nbytes)
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
    data_start = _align(len(MMFREE_MAGIC) + 8 + len(header_bytes))

    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "wb") as f:
        f.write(MMFREE_MAGIC)
        f.write(struct.pack("<Q", len(header_bytes)))
        f.write(header_bytes)
        f.write(b"\0" * (data_start - f.tell()))
        assert f.tell() == data_start
        base = f.tell()
        for name, (dtype, arr) in tensors.items():
            off = base + meta[name]["offset"]
            f.write(b"\0" * (off - f.tell()))
            f.write(arr.tobytes())
        f.write(b"\0" * (base + data_len - f.tell()))  # pad tail to exact size

    total = data_start + data_len
    print(f"wrote {len(tensors)} tensors -> {out} ({total / 1e6:.1f} MB)")
    return out


# ── 3. pack tokenizer (stdlib only) ──────────────────────────────────────────
def pack_tokenizer(out: str, model: str = DEFAULT_MODEL) -> str:
    """Pack `model`'s tokenizer.json into `out` (tokenizer.mmtok). Stdlib-only; the
    checkpoint must already be in the HF cache. Returns the output path."""
    ckpt = _resolve_checkpoint(model)
    with open(os.path.join(ckpt, "tokenizer.json"), encoding="utf-8") as f:
        tj = json.load(f)
    m = tj["model"]
    assert m["type"] == "BPE", f"expected BPE, got {m['type']!r}"

    vocab = m["vocab"]            # piece -> id
    merges = m["merges"]          # rank-ordered "left right" (or [left, right])
    vocab_size = len(vocab)

    id2piece = [None] * vocab_size
    for piece, i in vocab.items():
        id2piece[i] = piece
    assert all(p is not None for p in id2piece), "vocab ids are not dense 0..N-1"

    def special_id(want: str, default: int) -> int:
        for t in tj.get("added_tokens", []):
            if t.get("content") == want:
                return int(t["id"])
        return int(vocab.get(want, default))

    bos = special_id("<s>", 1)
    eos = special_id("</s>", 2)
    unk = special_id("<unk>", 0)

    def merge_str(mg) -> str:
        return mg if isinstance(mg, str) else f"{mg[0]} {mg[1]}"

    out = os.path.abspath(out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "wb") as f:
        f.write(MMTOK_MAGIC)
        f.write(struct.pack("<II", vocab_size, len(merges)))
        f.write(struct.pack("<iii", bos, eos, unk))
        for piece in id2piece:
            b = piece.encode("utf-8")
            f.write(struct.pack("<I", len(b)))
            f.write(b)
        for mg in merges:
            b = merge_str(mg).encode("utf-8")
            f.write(struct.pack("<I", len(b)))
            f.write(b)

    print(f"wrote {out}")
    print(f"  vocab={vocab_size} merges={len(merges)} bos={bos} eos={eos} unk={unk}")
    return out


def main() -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, here)  # so `from reference import ...` resolves in pack_weights
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL, help="HF repo id (default: %(default)s)")
    ap.add_argument("--out-dir", default=os.path.join(here, "..", "cpp"),
                    help="where to write model.mmfree + tokenizer.mmtok (default: ../cpp)")
    ap.add_argument("--skip-download", action="store_true",
                    help="use the already-cached checkpoint")
    ap.add_argument("--weights-only", action="store_true", help="pack only model.mmfree")
    ap.add_argument("--tokenizer-only", action="store_true", help="pack only tokenizer.mmtok")
    args = ap.parse_args()
    if args.weights_only and args.tokenizer_only:
        ap.error("--weights-only and --tokenizer-only are mutually exclusive")

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    do_weights = not args.tokenizer_only
    do_tokenizer = not args.weights_only

    n = int(not args.skip_download) + int(do_weights) + int(do_tokenizer)
    step = 0

    if not args.skip_download:
        step += 1
        print(f"==> [{step}/{n}] downloading {args.model} from the HF Hub ...", flush=True)
        print(f"    cached at {download(args.model)}")
    else:
        print("==> skipping download (--skip-download)")

    results = []
    if do_weights:
        step += 1
        print(f"==> [{step}/{n}] packing weights -> model.mmfree", flush=True)
        results.append(pack_weights(os.path.join(out_dir, "model.mmfree"), args.model))
    if do_tokenizer:
        step += 1
        print(f"==> [{step}/{n}] packing tokenizer -> tokenizer.mmtok", flush=True)
        results.append(pack_tokenizer(os.path.join(out_dir, "tokenizer.mmtok"), args.model))

    print("\nDone. C++ inputs ready:")
    for r in results:
        print(f"  {r}")
    print('Build + run:  cd cpp && cmake -B build && cmake --build build -j && '
          './build/mmfree-cli "hello"')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
