# -*- coding: utf-8 -*-
"""
Text-in / text-out runner for the C++ mmfree-cli.

Tokenizes a prompt with the real LLaMA SentencePiece tokenizer (Python), feeds the
ids to the C++ model (mmfree-cli), then decodes the generated ids back to text. No
PyTorch / reference model is loaded -- this just drives the C++ port (cf. plan P4:
Python tokenizes, C++ runs the model).

Usage:
  python run_prompt.py "The capital of France is" [--new-tokens 16] [--mode fixed]
  python run_prompt.py            # reads the prompt from stdin

  --mode fixed   16-bit Q5.10 fixed-point activations (the FPGA datapath, default)
  --mode float   fp32 activations
"""
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import subprocess
import sys

from transformers import AutoTokenizer

DEFAULT_MODEL = "ridger/MMfreeLM-370M"


def main():
    here = os.path.dirname(__file__)
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt", nargs="?", help="prompt text (reads stdin if omitted)")
    ap.add_argument("--new-tokens", type=int, default=16)
    ap.add_argument("--mode", default="fixed", choices=("fixed", "float"))
    ap.add_argument("--cli", default=os.path.join(here, "..", "cpp", "build", "mmfree-cli"))
    ap.add_argument("--blob", default=os.path.join(here, "..", "cpp", "model.mmfree"))
    ap.add_argument("--show-ids", action="store_true", help="also print prompt/gen ids")
    args = ap.parse_args()

    prompt = args.prompt if args.prompt is not None else sys.stdin.read().strip()
    if not prompt:
        ap.error("no prompt (pass as an argument or via stdin)")

    cli = os.path.abspath(args.cli)
    blob = os.path.abspath(args.blob)

    tok = AutoTokenizer.from_pretrained(DEFAULT_MODEL)
    ids = tok(prompt, return_tensors="pt").input_ids[0].tolist()

    cmd = [cli, "--blob", blob, "--mode", args.mode, "--gen", str(args.new_tokens),
           "--ids", ",".join(str(i) for i in ids)]
    out = subprocess.run(cmd, capture_output=True, text=True)
    if out.returncode != 0:
        sys.stderr.write(out.stderr)
        raise SystemExit(out.returncode)

    stream = [int(x) for x in out.stdout.split()]
    gen_ids = stream[len(ids):]

    if args.show_ids:
        print(f"prompt ids ({len(ids)}): {ids}")
        print(f"gen ids ({len(gen_ids)}): {gen_ids}")
    print(tok.decode(gen_ids, skip_special_tokens=True))


if __name__ == "__main__":
    main()
