"""Quick smoke test: load tokenizers and verify chat template output.

Runs on CPU (login node) — no GPU needed.  Downloads only the tokenizer
configs (~a few MB each), not the full model weights.

Usage:
    python scripts/smoke_test_models.py
    python scripts/smoke_test_models.py --model mistralai/Mistral-7B-Instruct-v0.3
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

HF_TOKEN = os.environ.get("HF_TOKEN")

MODELS = [
    "microsoft/Phi-3-mini-4k-instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
]

SAMPLE_MESSAGES = [
    {
        "role": "system",
        "content": (
            "You are a clinical QA assistant. Answer based only on the provided context."
        ),
    },
    {
        "role": "user",
        "content": (
            "### Patient Context\n\nPatient is a 65-year-old male admitted for chest pain.\n\n"
            "### Question\n\nWhat was the admission diagnosis?"
        ),
    },
]


def test_tokenizer(model_id: str) -> bool:
    from transformers import AutoTokenizer

    print(f"\n{'='*60}")
    print(f"Testing: {model_id}")
    print(f"{'='*60}")

    try:
        tok = AutoTokenizer.from_pretrained(model_id, token=HF_TOKEN)
        print(f"  vocab_size         : {tok.vocab_size}")
        print(f"  model_max_length   : {tok.model_max_length}")
        print(f"  padding_side       : {tok.padding_side}")
        print(f"  has chat_template  : {tok.chat_template is not None}")

        if tok.chat_template:
            formatted = tok.apply_chat_template(
                SAMPLE_MESSAGES,
                tokenize=False,
                add_generation_prompt=True,
            )
            n_tokens = len(tok.encode(formatted))
            print(f"  sample prompt tokens : {n_tokens}")
            print(f"  formatted (first 300 chars):")
            print("    " + repr(formatted[:300]))
        else:
            print("  WARNING: No chat template — instruct format may not work")

        print(f"  PASS")
        return True

    except Exception as exc:
        print(f"  FAIL: {exc}")
        return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", help="Single model ID to test (default: test all)")
    args = p.parse_args()

    models = [args.model] if args.model else MODELS

    results = {m: test_tokenizer(m) for m in models}

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for m, ok in results.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {m}")

    if not all(results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
