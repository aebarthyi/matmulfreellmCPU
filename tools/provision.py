# -*- coding: utf-8 -*-
"""
Provision the C++ engine from a Hugging Face checkpoint, all three offline steps:

  1. download the MMfreeLM checkpoint from the HF Hub (weights + tokenizer.json)
  2. pack the weights   -> blobs/<size>/model.mmfree     (ternary quant, baked once)
  3. pack the tokenizer -> blobs/<size>/tokenizer.mmtok  (BPE vocab + merges)

Afterwards the engine needs no Python. Weight quant matches tools/reference.py:
scale_w = 1/mean(|W|), wq = clamp(round(W*scale_w), -1, 1); the C++ side never quantizes.

Usage:
  pip install -r tools/requirements.txt
  python tools/provision.py                                   # 370M -> blobs/370M/
  python tools/provision.py --model 1.3B                      # 370M | 1.3B | 2.7B, or an HF id
  python tools/provision.py --out-dir /path/to/dir
  python tools/provision.py --skip-download                   # checkpoint already cached
  python tools/provision.py --weights-only | --tokenizer-only # one artifact
  python tools/provision.py --no-torch                        # pack with numpy only

Gated/private repos: run `huggingface-cli login` first (the MMfreeLM models are public).
Imports are lazy: downloading needs huggingface_hub, the default weight packer torch,
--no-torch only numpy, and the tokenizer packer only the stdlib.

--no-torch reads the safetensors file directly and quantizes in numpy. The ternary weights
are bit-identical to the torch path; per-tensor `scale_w` (and `lower_bounds`) can differ
in the last 1-2 ULP because numpy and torch sum in a different order. The torch path is
the numerics of record.
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
# The released checkpoints by size: `--model 1.3B` provisions into blobs/1.3B/, which is
# where `mmfree-cli --model 1.3B` looks.
SIZES = {"370M": "ridger/MMfreeLM-370M", "1.3B": "ridger/MMfreeLM-1.3B",
         "2.7B": "ridger/MMfreeLM-2.7B"}


def resolve_model(name: str) -> tuple[str, str]:
    """(HF repo id, blobs/ subdirectory) for a size like "1.3b" or any HF repo id."""
    for size, repo in SIZES.items():
        if name.upper() == size or name == repo:
            return repo, size
    return name, name.rstrip("/").split("/")[-1]

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


# ── 2. pack weights ──────────────────────────────────────────────────────────
# The packer works on numpy fp32 arrays; a small backend supplies them plus the ternary
# quant, so the same packing code serves both the torch reference path and --no-torch.
class _TorchBackend:
    """Reference path: tools/reference.py (torch) loads and quantizes. This is the
    numerics of record — the golden oracles were generated through it."""
    name = "torch"

    def __init__(self, model: str):
        import torch  # noqa: F401  (reference loads torch tensors)
        from reference import load_reference, weight_quant_int
        self._quant_fn = weight_quant_int
        ref, self.cfg, self.ckpt = load_reference(model)
        self._sd = ref.sd

    def f32(self, key: str):
        import numpy as np
        return np.ascontiguousarray(self._sd[key].detach().cpu().numpy().astype(np.float32))

    def quant(self, key: str):
        import numpy as np
        wq, scale = self._quant_fn(self._sd[key])
        return (np.ascontiguousarray(wq.detach().cpu().numpy().astype(np.int8)),
                np.float32(scale.item()))

    def lower_bounds(self):
        import numpy as np
        lb = self._sd["model.lower_bounds"].softmax(0)
        lb = lb.cumsum(0) - lb[0]
        return np.ascontiguousarray(lb.detach().cpu().numpy().astype(np.float32))


class _NumpyBackend:
    """torch-free path (--no-torch): reads the safetensors file itself and quantizes in
    numpy. See the module docstring for exactly how far this diverges from _TorchBackend
    (ternary weights identical; scale_w within 1-2 ULP)."""
    name = "numpy"

    def __init__(self, model: str):
        import numpy as np
        self._np = np
        self.ckpt = _resolve_checkpoint(model)
        self.cfg = _NumpyConfig.from_json(os.path.join(self.ckpt, "config.json"))
        self._sd = {}
        for f in sorted(glob.glob(os.path.join(self.ckpt, "*.safetensors"))):
            self._sd.update(_read_safetensors_f32(f))
        if not self._sd:
            raise FileNotFoundError(f"no *.safetensors in {self.ckpt}")

    def f32(self, key: str):
        return self._np.ascontiguousarray(self._sd[key], dtype=self._np.float32)

    def quant(self, key: str):
        # Mirrors reference.weight_quant_int. np.rint and torch.round both round half to
        # even, so only the fp32 summation order inside mean() can differ.
        np = self._np
        w = self._sd[key]
        scale = np.float32(1.0) / np.maximum(np.abs(w).mean(dtype=np.float32),
                                             np.float32(1e-5))
        wq = np.clip(np.rint(w * scale), -1, 1).astype(np.int8)
        return np.ascontiguousarray(wq), np.float32(scale)

    def lower_bounds(self):
        np = self._np
        x = self._sd["model.lower_bounds"].astype(np.float32)
        e = np.exp(x - x.max(axis=0, keepdims=True))       # softmax over dim 0
        lb = e / e.sum(axis=0, keepdims=True)
        lb = np.cumsum(lb, axis=0) - lb[0]
        return np.ascontiguousarray(lb, dtype=np.float32)


class _NumpyConfig:
    """Torch-free mirror of reference.HGRNBitConfig (same fields and inter_size rule)."""
    _DEFAULTS = dict(vocab_size=32000, hidden_size=1024, num_hidden_layers=24, num_heads=1,
                     expand_ratio=1, hidden_ratio=4, intermediate_size=None,
                     use_short_conv=False, conv_size=4, use_lower_bound=True,
                     rms_norm_eps=1e-6)

    def __init__(self, **kw):
        for k, v in self._DEFAULTS.items():
            setattr(self, k, kw.get(k, v) if kw.get(k, v) is not None else v)
        self.intermediate_size = kw.get("intermediate_size")  # may legitimately be None

    @classmethod
    def from_json(cls, path: str) -> "_NumpyConfig":
        with open(path) as f:
            d = json.load(f)
        return cls(**{k: d[k] for k in cls._DEFAULTS if k in d})

    @property
    def inter_size(self) -> int:
        if self.intermediate_size is not None:
            return self.intermediate_size
        i = int(self.hidden_size * self.hidden_ratio * 2 / 3)
        return 256 * ((i + 256 - 1) // 256)


def _read_safetensors_f32(path: str) -> dict:
    """Minimal safetensors reader -> {name: fp32 ndarray}. BF16 is widened by shifting
    into the high half of an fp32 (exact, same as torch's .float()); F16/F32 convert
    directly. Everything else is rejected rather than guessed at."""
    import numpy as np
    with open(path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))
    base = 8 + header_len
    blob = np.memmap(path, dtype=np.uint8, mode="r")
    out = {}
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        start, end = meta["data_offsets"]
        raw = blob[base + start: base + end]
        dtype = meta["dtype"]
        if dtype == "BF16":
            arr = (raw.view(np.uint16).astype(np.uint32) << 16).view(np.float32)
        elif dtype == "F32":
            arr = raw.view(np.float32).astype(np.float32)
        elif dtype == "F16":
            arr = raw.view(np.float16).astype(np.float32)
        else:
            raise ValueError(f"{name}: unsupported safetensors dtype {dtype!r} "
                             f"(--no-torch handles BF16/F16/F32); drop --no-torch")
        out[name] = arr.reshape(meta["shape"])
    return out


def pack_weights(out: str, model: str = DEFAULT_MODEL, no_torch: bool = False) -> str:
    """Quantize + pack `model`'s checkpoint into `out` (model.mmfree). The checkpoint
    must already be in the HF cache (download() above). Returns the output path."""
    import numpy as np

    out = os.path.abspath(out)
    be = (_NumpyBackend if no_torch else _TorchBackend)(model)
    cfg = be.cfg
    print(f"loaded checkpoint: {be.ckpt}  (packer backend: {be.name})")

    tensors: dict = {}

    def add_f32(name, arr):
        tensors[name] = ("f32", np.ascontiguousarray(arr, dtype=np.float32))

    def add_i8(name, arr):
        tensors[name] = ("i8", np.ascontiguousarray(arr, dtype=np.int8))

    def add_proj(tag, key):
        """One ternary projection: int8 weights + its per-tensor scale + norm weight."""
        wq, scale_w = be.quant(f"{key}.weight")
        add_i8(f"{tag}.wq", wq)
        add_f32(f"{tag}.scale_w", np.asarray(scale_w, dtype=np.float32).reshape(1))
        add_f32(f"{tag}.normw", be.f32(f"{key}.norm.weight"))

    # global: embeddings, final norm, lm_head (ternary), lower_bounds (precomputed)
    add_f32("model.embeddings", be.f32("model.embeddings.weight"))
    add_f32("model.norm.w", be.f32("model.norm.weight"))
    add_proj("lm_head", "lm_head")
    if cfg.use_lower_bound:
        add_f32("lower_bounds", be.lower_bounds())  # [L, H]

    # per-layer
    for l in range(cfg.num_hidden_layers):
        p = f"model.layers.{l}"
        add_f32(f"{p}.attn_norm.w", be.f32(f"{p}.attn_norm.weight"))
        add_f32(f"{p}.mlp_norm.w", be.f32(f"{p}.mlp_norm.weight"))
        add_f32(f"{p}.g_norm.w", be.f32(f"{p}.attn.g_norm.weight"))
        for tag, infix in LAYER_PROJS:
            add_proj(f"{p}.{tag}", f"{p}.{infix}")

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
    ap.add_argument("--model", default="370M",
                    help="370M | 1.3B | 2.7B, or any HF repo id (default: %(default)s)")
    ap.add_argument("--out-dir", default=None,
                    help="where to write model.mmfree + tokenizer.mmtok "
                         "(default: blobs/<size>/ at the repo root)")
    ap.add_argument("--skip-download", action="store_true",
                    help="use the already-cached checkpoint")
    ap.add_argument("--weights-only", action="store_true", help="pack only model.mmfree")
    ap.add_argument("--tokenizer-only", action="store_true", help="pack only tokenizer.mmtok")
    ap.add_argument("--no-torch", action="store_true",
                    help="pack the weights with numpy instead of torch (ternary weights "
                         "identical, scale_w within 1-2 ULP — see the module docstring)")
    args = ap.parse_args()
    if args.weights_only and args.tokenizer_only:
        ap.error("--weights-only and --tokenizer-only are mutually exclusive")

    args.model, subdir = resolve_model(args.model)
    out_dir = os.path.abspath(args.out_dir or os.path.join(here, "..", "blobs", subdir))
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
        results.append(pack_weights(os.path.join(out_dir, "model.mmfree"), args.model,
                                    no_torch=args.no_torch))
    if do_tokenizer:
        step += 1
        print(f"==> [{step}/{n}] packing tokenizer -> tokenizer.mmtok", flush=True)
        results.append(pack_tokenizer(os.path.join(out_dir, "tokenizer.mmtok"), args.model))

    print("\nDone. C++ inputs ready:")
    for r in results:
        print(f"  {r}")
    size = subdir if subdir in SIZES else None
    print('Build + run:  make && make run' + (f' MODEL={size}' if size else '') +
          ' PROMPT="hello"   (or: cmake -S cpp -B build && cmake --build build -j)')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
