"""Assign weak binary labels to (question, chunk) pairs using answer-chunk overlap.

Supervision signal: a chunk is labeled positive (1) if its token-level F1
with the gold answer exceeds a threshold, else negative (0).

This is "weak supervision via distant supervision" — standard in retrieval
model training (e.g., DPR, BM25 re-ranking). The answers in EHR-DS-QA are
factual sentences extracted from the discharge summary, so high token overlap
reliably indicates that the chunk supports the answer.
"""

from __future__ import annotations

import re
from collections import Counter


def _tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, split on whitespace."""
    return re.findall(r"[a-z0-9]+", text.lower())


def token_f1(text_a: str, text_b: str) -> float:
    """Token-level F1 between two text strings.

    Treats each text as a bag of tokens.  F1 is the harmonic mean of
    precision (fraction of text_b tokens that appear in text_a) and
    recall (fraction of text_a tokens that appear in text_b).
    """
    toks_a = Counter(_tokenize(text_a))
    toks_b = Counter(_tokenize(text_b))
    if not toks_a or not toks_b:
        return 0.0
    common    = sum((toks_a & toks_b).values())
    precision = common / sum(toks_b.values())
    recall    = common / sum(toks_a.values())
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def label_chunks(
    chunks: list[dict],
    answer: str,
    threshold: float = 0.15,
) -> list[int]:
    """Return a binary label per chunk based on token F1 overlap with the answer.

    Args:
        chunks:     List of chunk dicts (must have a 'text' key).
        answer:     Gold answer string from EHR-DS-QA.
        threshold:  Minimum token F1 for a chunk to be labeled positive.

    Returns:
        List of 0/1 labels, one per chunk.
    """
    return [
        1 if token_f1(chunk["text"], answer) >= threshold else 0
        for chunk in chunks
    ]
