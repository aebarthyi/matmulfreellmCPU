# -*- coding: utf-8 -*-
"""
Pack the LLaMA SentencePiece-BPE tokenizer into a single `tokenizer.mmtok` file for the
C++ runtime. OFFLINE step (build host); needs only the stdlib -- reads tokenizer.json
straight from the HF snapshot (no torch / transformers / sentencepiece).

The MMfreeLM-370M tokenizer is byte-level BPE (model.type "BPE"): 32000 pieces, 58980
rank-ordered merges, byte_fallback, normalizer = {prepend "U+2581", replace " "->U+2581},
no pre-tokenizer split, BOS(1) prepended. The C++ tokenizer (mmfree/tokenizer.hpp)
reproduces encode/decode from exactly this data.

tokenizer.mmtok layout (little-endian):
  [8]   magic        "MMTOK1\n"
  u32   vocab_size
  u32   n_merges
  i32   bos_id, eos_id, unk_id
  vocab_size x { u32 len, <len> piece bytes (UTF-8) }      # id order
  n_merges  x { u32 len, <len> "left right" bytes (UTF-8) } # rank order

Usage:
  python pack_tokenizer.py [--out ../cpp/tokenizer.mmtok] [--model ridger/MMfreeLM-370M]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import struct

MAGIC = b"MMTOK1\n\x00"  # 8 bytes; matches the C++ reader (mmfree/tokenizer.cpp)
DEFAULT_MODEL = "ridger/MMfreeLM-370M"


def resolve_checkpoint(model: str) -> str:
    if os.path.isdir(model):
        return model
    cache = os.path.expanduser(
        f"~/.cache/huggingface/hub/models--{model.replace('/', '--')}/snapshots/*"
    )
    snaps = sorted(glob.glob(cache))
    if not snaps:
        raise FileNotFoundError(f"checkpoint for {model!r} not found locally")
    return snaps[-1]


def special_id(tj: dict, vocab: dict, want: str, default: int) -> int:
    for t in tj.get("added_tokens", []):
        if t.get("content") == want:
            return int(t["id"])
    return int(vocab.get(want, default))


def main() -> None:
    ap = argparse.ArgumentParser()
    here = os.path.dirname(__file__)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--out", default=os.path.join(here, "..", "cpp", "tokenizer.mmtok"))
    args = ap.parse_args()

    ckpt = resolve_checkpoint(args.model)
    with open(os.path.join(ckpt, "tokenizer.json"), encoding="utf-8") as f:
        tj = json.load(f)
    model = tj["model"]
    assert model["type"] == "BPE", f"expected BPE, got {model['type']!r}"

    vocab = model["vocab"]            # piece -> id
    merges = model["merges"]          # rank-ordered "left right" (or [left, right])
    vocab_size = len(vocab)

    # id -> piece (dense, ids are 0..vocab_size-1)
    id2piece = [None] * vocab_size
    for piece, i in vocab.items():
        id2piece[i] = piece
    assert all(p is not None for p in id2piece), "vocab ids are not dense 0..N-1"

    bos = special_id(tj, vocab, "<s>", 1)
    eos = special_id(tj, vocab, "</s>", 2)
    unk = special_id(tj, vocab, "<unk>", 0)

    def merge_str(m) -> str:
        # tokenizer.json stores merges as "left right" strings or [left, right] pairs.
        return m if isinstance(m, str) else f"{m[0]} {m[1]}"

    out = os.path.abspath(args.out)
    with open(out, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<II", vocab_size, len(merges)))
        f.write(struct.pack("<iii", bos, eos, unk))
        for piece in id2piece:
            b = piece.encode("utf-8")
            f.write(struct.pack("<I", len(b)))
            f.write(b)
        for m in merges:
            b = merge_str(m).encode("utf-8")
            f.write(struct.pack("<I", len(b)))
            f.write(b)

    print(f"wrote {out}")
    print(f"  vocab={vocab_size} merges={len(merges)} bos={bos} eos={eos} unk={unk}")
    print(f"  size={os.path.getsize(out)} bytes")


if __name__ == "__main__":
    main()
