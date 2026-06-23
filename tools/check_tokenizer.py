# -*- coding: utf-8 -*-
"""
Validate the C++ embedded tokenizer (mmfree-cli) against HF AutoTokenizer, id-for-id.

For each prompt: compare encode (HF .input_ids vs `mmfree-cli "<prompt>" --gen 0 --print-ids`)
and decode (HF .decode vs `mmfree-cli --decode-ids ...`). Exits non-zero on any mismatch.

Usage:
  python check_tokenizer.py [--cli ../cpp/build/mmfree-cli] [--tokenizer ../cpp/tokenizer.mmtok]
"""
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import subprocess

from transformers import AutoTokenizer

DEFAULT_MODEL = "ridger/MMfreeLM-370M"

PROMPTS = [
    "The capital of France is",
    "Once upon a time,",
    "def fibonacci(n):",
    "In a shocking finding, scientist discovered a herd of dragons.",
    "The meaning of life is",
    "Hello, world!",
    "   leading and  multiple   spaces ",
    "Numbers: 1234567890 and symbols @#$%^&*()",
    "Café au lait — naïve façade, jalapeño 🌶️ emoji test 🚀",
    "Mixed\ttabs\nand newlines",
    "ALLCAPS and CamelCase and snake_case_words",
    "",
    "a",
    "你好，世界",
]


def cli_encode(cli, tok_path, prompt):
    out = subprocess.run([cli, prompt, "--tokenizer", tok_path, "--gen", "0", "--print-ids"],
                         capture_output=True, text=True, check=True)
    return [int(x) for x in out.stdout.split()]


def cli_decode(cli, tok_path, ids):
    out = subprocess.run([cli, "--tokenizer", tok_path, "--decode-ids",
                          ",".join(str(i) for i in ids)],
                         capture_output=True, text=True, check=True)
    return out.stdout.rstrip("\n")


def main():
    here = os.path.dirname(__file__)
    ap = argparse.ArgumentParser()
    ap.add_argument("--cli", default=os.path.join(here, "..", "cpp", "build", "mmfree-cli"))
    ap.add_argument("--tokenizer", default=os.path.join(here, "..", "cpp", "tokenizer.mmtok"))
    args = ap.parse_args()
    cli = os.path.abspath(args.cli)
    tok_path = os.path.abspath(args.tokenizer)

    hf = AutoTokenizer.from_pretrained(DEFAULT_MODEL)

    enc_fail = dec_fail = 0
    for p in PROMPTS:
        hf_ids = hf(p).input_ids
        cpp_ids = cli_encode(cli, tok_path, p)
        enc_ok = hf_ids == cpp_ids
        enc_fail += not enc_ok

        # decode the HF ids on both sides (skip_special to match the CLI default)
        hf_txt = hf.decode(hf_ids, skip_special_tokens=True)
        cpp_txt = cli_decode(cli, tok_path, hf_ids)
        dec_ok = hf_txt == cpp_txt
        dec_fail += not dec_ok

        tag = "ok " if (enc_ok and dec_ok) else "FAIL"
        print(f"[{tag}] {p!r}")
        if not enc_ok:
            print(f"      ENCODE  hf={hf_ids}\n              cpp={cpp_ids}")
        if not dec_ok:
            print(f"      DECODE  hf={hf_txt!r}\n              cpp={cpp_txt!r}")

    n = len(PROMPTS)
    print(f"\nencode: {n - enc_fail}/{n} match   decode: {n - dec_fail}/{n} match")
    raise SystemExit(1 if (enc_fail or dec_fail) else 0)


if __name__ == "__main__":
    main()
