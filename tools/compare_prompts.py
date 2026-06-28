# -*- coding: utf-8 -*-
"""
Prompt-for-prompt comparison: C++ CPU port vs the PyTorch golden reference.

For each text prompt: tokenize ONCE (the real LLaMA SentencePiece tokenizer), feed the
SAME ids to (a) the pure-torch reference (reference.py) and (b) the C++ mmfree-cli, then
compare the greedy token streams (the hard acceptance gate) and the last-position logits.

Because both sides consume identical ids, no C++ tokenizer is needed (plan P4). The
C++ activation numerics (--cli-mode) are matched to the reference QUANT_MODE (--ref-mode):
  fixed <-> fixedp (16-bit Q5.10), float <-> triton (fp32).

Usage:
  python compare_prompts.py [--new-tokens 8] [--ref-mode fixedp] [--cli-mode fixed]
                            [--prompts FILE] [--cli ../cpp/build/mmfree-cli]
                            [--blob ../cpp/model.mmfree]
"""
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import subprocess

import numpy as np
import torch

import reference as R
from reference import DEFAULT_MODEL, greedy_generate, load_reference

DEFAULT_PROMPTS = [
    "In a shocking finding, scientist discovered a herd of dragons.",
    "The capital of France is",
    "Once upon a time,",
    "def fibonacci(n):",
    "The meaning of life is",
]

CLI_TO_REF = {"fixed": "fixedp", "float": "triton"}


def rel_l2(a, b):
    a, b = a.reshape(-1), b.reshape(-1)
    return float(np.sqrt(((a - b) ** 2).sum() / ((b ** 2).sum() + 1e-30)))


def run_cli(cli, blob, mode, ids, new_tokens, logits_out):
    cmd = [cli, "--blob", blob, "--mode", mode, "--gen", str(new_tokens),
           "--ids", ",".join(str(int(i)) for i in ids), "--logits-out", logits_out]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return [int(x) for x in out.stdout.split()]


def main():
    here = os.path.dirname(__file__)
    ap = argparse.ArgumentParser()
    ap.add_argument("--new-tokens", type=int, default=8)
    ap.add_argument("--ref-mode", default=None, help="reference QUANT_MODE (default: from --cli-mode)")
    ap.add_argument("--cli-mode", default="fixed", choices=("fixed", "float"))
    ap.add_argument("--prompts", default=None, help="file with one prompt per line")
    ap.add_argument("--cli", default=os.path.join(here, "..", "cpp", "build", "mmfree-cli"))
    ap.add_argument("--blob", default=os.path.join(here, "..", "cpp", "model.mmfree"))
    args = ap.parse_args()

    ref_mode = args.ref_mode or CLI_TO_REF[args.cli_mode]
    cli = os.path.abspath(args.cli)
    blob = os.path.abspath(args.blob)
    tmp_logits = os.path.join(here, "_cli_logits.f32")

    prompts = DEFAULT_PROMPTS
    if args.prompts:
        with open(args.prompts) as f:
            prompts = [ln.rstrip("\n") for ln in f if ln.strip()]

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(DEFAULT_MODEL)
    torch.manual_seed(0)
    ref, cfg, ckpt = load_reference()
    R.QUANT_MODE = ref_mode
    print(f"ref_mode={ref_mode}  cli_mode={args.cli_mode}  new_tokens={args.new_tokens}  "
          f"checkpoint={os.path.basename(ckpt)}\n")

    n_match = 0
    for p in prompts:
        ids = tok(p, return_tensors="pt").input_ids
        ids_list = ids[0].tolist()

        ref_stream = greedy_generate(ref, ids, args.new_tokens)[0].tolist()
        cli_stream = run_cli(cli, blob, args.cli_mode, ids_list, args.new_tokens, tmp_logits)

        match = ref_stream == cli_stream
        n_match += int(match)

        # logits cross-check on the (common) stream, if greedy agreed
        logit_note = ""
        if match:
            full = torch.tensor([cli_stream])
            ref_last = ref.forward(full)[0, -1].float().numpy()
            cli_last = np.fromfile(tmp_logits, dtype=np.float32)
            logit_note = f"  logits rel_l2={rel_l2(cli_last, ref_last):.2e}"

        gen_ref = ref_stream[len(ids_list):]
        gen_cli = cli_stream[len(ids_list):]
        status = "MATCH" if match else "DIFFER"
        print(f"[{status}] {p!r}")
        print(f"    prompt ids ({len(ids_list)}): {ids_list}")
        print(f"    ref gen: {gen_ref}")
        print(f"    cli gen: {gen_cli}{logit_note}")
        if match:
            print(f"    -> {tok.decode(gen_cli, skip_special_tokens=True)!r}")
        print()

    if os.path.exists(tmp_logits):
        os.remove(tmp_logits)
    print(f"{n_match}/{len(prompts)} prompts: greedy stream MATCH")
    raise SystemExit(0 if n_match == len(prompts) else 1)


if __name__ == "__main__":
    main()
