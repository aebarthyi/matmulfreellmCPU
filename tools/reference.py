# -*- coding: utf-8 -*-
"""
Pure-PyTorch fp32 reference forward for HGRN-Bit / MMfreeLM (ridger/MMfreeLM-370M).

This is the GOLDEN SPEC the C++ CPU port must reproduce. It deliberately:
  * uses NO triton and NO transformers model class (loads safetensors directly),
  * runs everything in fp32 with explicit, obvious math,
  * faithfully replicates the fused-inference numerics of the original kernels:
      - per-projection RMSNorm with eps=1e-6 (the FusedBitLinear path overrides the
        module's 1e-8 default; the standalone attn/mlp/final norms also use 1e-6),
      - per-row int8 activation quant/dequant,
      - per-tensor ternary weight quant ({-1,0,+1} / scale_w),
      - the projection done as an INTEGER accumulator (int32) then divided by
        (scale_w * scale_a_row) -> matches the float matmul bit-closely and lets the
        C++ assert integer-exactness at the accumulator.

Architectural facts (verified against the checkpoint, June 2026):
  hidden=1024, layers=24, heads=1, expand_ratio=1 -> head_dim=1024,
  intermediate=2816 (gate_proj out=5632, down_proj in=2816), vocab=32000,
  use_lower_bound=True, use_short_conv=False.
  lm_head IS a ternary BitLinear (it has lm_head.norm.weight) -- there is no dense GEMM.

Every op optionally records its inputs/outputs into a `Capture` dict so dump_golden.py
can emit per-op oracles for the C++ unit tests.
"""
from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
from safetensors import safe_open

DEFAULT_MODEL = "ridger/MMfreeLM-370M"


# --------------------------------------------------------------------------------------
# config + weight loading
# --------------------------------------------------------------------------------------
@dataclass
class HGRNBitConfig:
    vocab_size: int = 32000
    hidden_size: int = 1024
    num_hidden_layers: int = 24
    num_heads: int = 1
    expand_ratio: int = 1
    hidden_ratio: int = 4
    intermediate_size: Optional[int] = None
    use_short_conv: bool = False
    conv_size: int = 4
    use_lower_bound: bool = True
    rms_norm_eps: float = 1e-6
    hidden_act: str = "swish"

    @property
    def input_dim(self) -> int:
        return self.hidden_size * self.expand_ratio

    @property
    def head_dim(self) -> int:
        return self.input_dim // self.num_heads

    @property
    def inter_size(self) -> int:
        if self.intermediate_size is not None:
            return self.intermediate_size
        i = int(self.hidden_size * self.hidden_ratio * 2 / 3)
        return 256 * ((i + 256 - 1) // 256)

    @classmethod
    def from_json(cls, path: str) -> "HGRNBitConfig":
        with open(path) as f:
            d = json.load(f)
        known = {k: d[k] for k in cls.__dataclass_fields__ if k in d}
        return cls(**known)


def resolve_checkpoint(model: str = DEFAULT_MODEL) -> str:
    """Return a local directory containing config.json + model.safetensors."""
    if os.path.isdir(model):
        return model
    cache = os.path.expanduser(
        f"~/.cache/huggingface/hub/models--{model.replace('/', '--')}/snapshots/*"
    )
    snaps = sorted(glob.glob(cache))
    if not snaps:
        raise FileNotFoundError(
            f"checkpoint for {model!r} not found locally; run generate.py once to download it."
        )
    return snaps[-1]


def load_state_dict(ckpt_dir: str) -> Dict[str, torch.Tensor]:
    sd: Dict[str, torch.Tensor] = {}
    for f in sorted(glob.glob(os.path.join(ckpt_dir, "*.safetensors"))):
        with safe_open(f, framework="pt") as st:
            for k in st.keys():
                sd[k] = st.get_tensor(k).float()  # everything to fp32
    return sd


# --------------------------------------------------------------------------------------
# capture
# --------------------------------------------------------------------------------------
@dataclass
class Capture:
    """Optional recorder of intermediate tensors keyed by name."""
    enabled: bool = False
    store: Dict[str, torch.Tensor] = field(default_factory=dict)

    def __call__(self, name: str, t: torch.Tensor) -> torch.Tensor:
        if self.enabled:
            self.store[name] = t.detach().clone()
        return t


# --------------------------------------------------------------------------------------
# primitive ops (the numerics contract)
# --------------------------------------------------------------------------------------
def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """x * rsqrt(mean(x^2)+eps) * weight, reductions in fp32. No bias."""
    x = x.float()
    var = x.pow(2).mean(dim=-1, keepdim=True)
    rstd = torch.rsqrt(var + eps)
    return x * rstd * weight


def activation_quant_int(y: torch.Tensor):
    """Per-row int8 quant. Returns (yq_int [int32], scale_a [per-row])."""
    scale = 127.0 / y.abs().amax(dim=-1, keepdim=True).clamp(min=1e-5)
    yq = (y * scale).round().clamp(-128, 127)
    return yq, scale


def weight_quant_int(w: torch.Tensor):
    """Per-tensor ternary quant. Returns (wq_int in {-1,0,1} [int8], scale_w scalar)."""
    scale = 1.0 / w.abs().mean().clamp(min=1e-5)
    wq = (w * scale).round().clamp(-1, 1)
    return wq, scale


# Numerics mode for the activation/weight handling inside BitLinear.
#   "triton"  : faithful to the triton fused kernel actually run by the model --
#               ternary weights (weight_quant), activations NOT rounded (the kernel
#               omits .round(), and since scale=127/max|y| the clamp is a no-op, so
#               activation quant is identity). This is matmul-free (float act x ternary w).
#   "bitnet"  : textbook BitNet b1.58 / docstring intent -- ternary weights AND per-row
#               int8-rounded activations, integer accumulator (plan's assumption).
#   "fullprec": faithful to generate.py as written -- the prepare_for_inference() bug
#               leaves weights full-precision; NOT matmul-free. (diagnostic only)
#   "int16"   : 16-bit activations, per-row DYNAMIC scale (block float, like "bitnet"
#               but 32767 instead of 127). Closest quantized analog to "triton";
#               integer accumulate over ternary lanes. scale_a = 32767/max|y_row|.
#   "fixedp"  : 16-bit activations, STATIC Qm.f scale (true fixed point), one global
#               power-of-2 binary point FIXED_FRAC_BITS for every projection input.
QUANT_MODE = "triton"
FIXED_FRAC_BITS = 10  # for "fixedp": signed Q5.10 in int16 -> range +-32, step 2^-10
                      # (global max|y|=16.54 fits; Q.11 -> +-16 would overflow/saturate)


def bitlinear(
    x: torch.Tensor,
    norm_weight: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    cap: Optional[Capture] = None,
    tag: str = "",
    mode: Optional[str] = None,
) -> torch.Tensor:
    """
    Fused BitLinear inference: RMSNorm(eps) -> activation handling -> projection.
    weight has nn.Linear layout [out, in]; output = act @ weight^T scaled.
    """
    mode = mode or QUANT_MODE
    y = rmsnorm(x, norm_weight, eps)                 # [.., in]

    if mode == "bitnet":
        yq, scale_a = activation_quant_int(y)        # int8 activations (rounded)
        wq, scale_w = weight_quant_int(weight)       # ternary weights
        acc = torch.matmul(yq, wq.t())               # integer accumulator
        out = acc / (scale_w * scale_a)
    elif mode == "int16":
        scale_a = 32767.0 / y.abs().amax(dim=-1, keepdim=True).clamp(min=1e-5)
        yq = (y * scale_a).round().clamp(-32768, 32767)  # int16 activations (per-row scale)
        wq, scale_w = weight_quant_int(weight)
        acc = torch.matmul(yq, wq.t())               # integer accumulator
        out = acc / (scale_w * scale_a)
    elif mode == "fixedp":
        s = float(1 << FIXED_FRAC_BITS)              # static Qm.f, global binary point
        yq = (y * s).round().clamp(-32768, 32767)    # int16 fixed point
        wq, scale_w = weight_quant_int(weight)
        acc = torch.matmul(yq, wq.t())               # integer accumulator (units of 1/s)
        out = acc / (s * scale_w)
    elif mode == "fullprec":
        acc = torch.matmul(y, weight.t())            # raw full-precision weights
        out = acc
    else:  # "triton": ternary weights, activations NOT rounded (kernel omits round)
        scale_a = 127.0 / y.abs().amax(dim=-1, keepdim=True).clamp(min=1e-5)
        ya = (y * scale_a).clamp(-128, 127) / scale_a   # == y (clamp is a no-op); kept for fidelity
        wq, scale_w = weight_quant_int(weight)
        acc = torch.matmul(ya, wq.t())               # float accumulator over signed acts
        out = acc / scale_w

    if cap is not None and cap.enabled:
        cap(f"{tag}.in", x)
        cap(f"{tag}.norm", y)
        cap(f"{tag}.acc", acc)
        cap(f"{tag}.out", out)
    return out


def silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


def swiglu(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return silu(a) * b


# --------------------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------------------
class HGRNBitReference:
    def __init__(self, cfg: HGRNBitConfig, sd: Dict[str, torch.Tensor]):
        self.cfg = cfg
        self.sd = sd

    def w(self, key: str) -> torch.Tensor:
        return self.sd[key]

    def block(
        self,
        h: torch.Tensor,            # [B, T, H]
        layer: int,
        lower_bound: Optional[torch.Tensor],
        cap: Optional[Capture] = None,
    ) -> torch.Tensor:
        cfg = self.cfg
        eps = cfg.rms_norm_eps
        p = f"model.layers.{layer}"

        residual = h
        # ---- attention ----
        hs = rmsnorm(h, self.w(f"{p}.attn_norm.weight"), eps)

        i = bitlinear(hs, self.w(f"{p}.attn.i_proj.norm.weight"),
                      self.w(f"{p}.attn.i_proj.weight"), eps, cap, f"{p}.i_proj")
        f = bitlinear(hs, self.w(f"{p}.attn.f_proj.norm.weight"),
                      self.w(f"{p}.attn.f_proj.weight"), eps, cap, f"{p}.f_proj")

        f = torch.sigmoid(f)
        if lower_bound is not None and layer > 0:
            f = lower_bound + (1 - lower_bound) * f
        i = swiglu(i, 1 - f)                                  # silu(i) * (1-f)

        # recurrence over time; heads=1 so reshape is [B, 1, T, head_dim]
        B, T, _ = i.shape
        hd = cfg.head_dim
        ir = i.view(B, T, cfg.num_heads, hd).transpose(1, 2)  # [B, nh, T, hd]
        fr = f.view(B, T, cfg.num_heads, hd).transpose(1, 2)
        if cap is not None and cap.enabled:
            cap(f"{p}.scan_i", ir)
            cap(f"{p}.scan_f", fr)
        o = torch.zeros_like(ir)
        state = torch.zeros(B, cfg.num_heads, hd, dtype=torch.float32)
        for t in range(T):
            state = fr[:, :, t] * state + ir[:, :, t]
            o[:, :, t] = state
        o = o.transpose(1, 2).reshape(B, T, cfg.input_dim)    # [B, T, input_dim]
        if cap is not None and cap.enabled:
            cap(f"{p}.recurrence", o)

        g = bitlinear(hs, self.w(f"{p}.attn.g_proj.norm.weight"),
                      self.w(f"{p}.attn.g_proj.weight"), eps, cap, f"{p}.g_proj")
        # g_norm: RMSNorm(g_proj(h)) * silu(recurrence_out)
        o = rmsnorm(g, self.w(f"{p}.attn.g_norm.weight"), eps) * silu(o)
        o = bitlinear(o, self.w(f"{p}.attn.o_proj.norm.weight"),
                      self.w(f"{p}.attn.o_proj.weight"), eps, cap, f"{p}.o_proj")

        # ---- mlp (mlp_norm is prenorm: returns (norm(o+residual), o+residual)) ----
        residual = o + residual
        hs = rmsnorm(residual, self.w(f"{p}.mlp_norm.weight"), eps)

        y = bitlinear(hs, self.w(f"{p}.mlp.gate_proj.norm.weight"),
                      self.w(f"{p}.mlp.gate_proj.weight"), eps, cap, f"{p}.gate_proj")
        gate, y = y.chunk(2, dim=-1)
        z = swiglu(gate, y)
        z = bitlinear(z, self.w(f"{p}.mlp.down_proj.norm.weight"),
                      self.w(f"{p}.mlp.down_proj.weight"), eps, cap, f"{p}.down_proj")

        h = residual + z
        if cap is not None and cap.enabled:
            cap(f"{p}.out", h)
        return h

    def forward(self, input_ids: torch.Tensor, cap: Optional[Capture] = None) -> torch.Tensor:
        cfg = self.cfg
        eps = cfg.rms_norm_eps

        h = torch.nn.functional.embedding(input_ids, self.w("model.embeddings.weight"))
        h = h.float()

        lower_bounds = None
        if cfg.use_lower_bound:
            lb = self.w("model.lower_bounds").softmax(0)      # [L, H]
            lower_bounds = lb.cumsum(0) - lb[0]

        for layer in range(cfg.num_hidden_layers):
            lb = lower_bounds[layer] if cfg.use_lower_bound else None
            h = self.block(h, layer, lb, cap)

        h = rmsnorm(h, self.w("model.norm.weight"), eps)
        if cap is not None and cap.enabled:
            cap("final_norm", h)

        logits = bitlinear(h, self.w("lm_head.norm.weight"),
                           self.w("lm_head.weight"), eps, cap, "lm_head")
        if cap is not None and cap.enabled:
            cap("logits", logits)
        return logits


def load_reference(model: str = DEFAULT_MODEL):
    ckpt = resolve_checkpoint(model)
    cfg = HGRNBitConfig.from_json(os.path.join(ckpt, "config.json"))
    sd = load_state_dict(ckpt)
    return HGRNBitReference(cfg, sd), cfg, ckpt


@torch.no_grad()
def greedy_generate(ref: "HGRNBitReference", input_ids: torch.Tensor, max_new_tokens: int):
    """Simple greedy decode recomputing the full prefix each step (no cache, batch=1)."""
    ids = input_ids
    for _ in range(max_new_tokens):
        logits = ref.forward(ids)
        nxt = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        ids = torch.cat([ids, nxt], dim=1)
    return ids


if __name__ == "__main__":
    torch.manual_seed(0)
    ref, cfg, ckpt = load_reference()
    print(f"loaded checkpoint: {ckpt}")
    print(f"config: hidden={cfg.hidden_size} layers={cfg.num_hidden_layers} "
          f"inter={cfg.inter_size} vocab={cfg.vocab_size} heads={cfg.num_heads}")
    ids = torch.tensor([[1, 512, 297, 263, 2913, 9138]])  # arbitrary token ids
    cap = Capture(enabled=True)
    logits = ref.forward(ids, cap)
    print(f"logits shape: {tuple(logits.shape)}  argmax(last)={int(logits[0, -1].argmax())}")
    print(f"captured {len(cap.store)} intermediate tensors")
