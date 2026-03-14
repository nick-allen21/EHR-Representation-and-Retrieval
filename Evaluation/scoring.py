"""Score predicted answers against gold answers.

Implements token F1, ROUGE-L, and LLM-as-judge scoring.
"""

from __future__ import annotations

import re
from rouge_score import rouge_scorer


# Token F1

def _normalize(text: str) -> list[str]:
    """Lowercase, strip punctuation, split on whitespace."""
    return re.findall(r"[a-z0-9]+", text.lower())


def token_f1(predicted: str, gold: str) -> dict:
    """Compute token-level precision, recall, and F1."""
    pred_toks = _normalize(predicted)
    gold_toks = _normalize(gold)
    if not pred_toks and not gold_toks:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not pred_toks or not gold_toks:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    common = sum((min(pred_toks.count(t), gold_toks.count(t)) for t in set(pred_toks)))
    precision = common / len(pred_toks)
    recall = common / len(gold_toks)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


# ROUGE-L

_scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)


def rouge_l(predicted: str, gold: str) -> dict:
    """Compute ROUGE-L precision, recall, and F-measure."""
    scores = _scorer.score(gold, predicted)
    rl = scores["rougeL"]
    return {"precision": rl.precision, "recall": rl.recall, "f1": rl.fmeasure}


### CITATION: Used Claude 4.6 Opus to help draft the LLM-as-judge rubric prompt below.
# LLM-as-Judge

_JUDGE_RUBRIC = """\
You are evaluating a clinical QA system. Given the question, the reference
(gold) answer, and the system's predicted answer, rate the predicted answer
on a scale of 1-5:

  5 — Fully correct, includes all key facts from the reference answer.
  4 — Mostly correct, minor omission or imprecision.
  3 — Partially correct, captures some key facts but misses important ones.
  2 — Mostly incorrect, only marginally related to the reference answer.
  1 — Completely wrong or empty.

Return ONLY a single integer (1-5).
"""


async def llm_judge(
    question: str,
    predicted: str,
    gold: str,
    runner=None,
) -> dict:
    """Rate predicted vs gold using an LLM judge.

    Requires an LLMRunner instance. Returns {"score": int} on success,
    or {"score": 0, "error": ...} on failure.
    """
    if runner is None:
        raise ValueError("llm_judge requires a LLMRunner instance via runner=")

    user_prompt = (
        f"### Question\n{question}\n\n"
        f"### Reference Answer\n{gold}\n\n"
        f"### Predicted Answer\n{predicted}\n\n"
        f"Rate the predicted answer (1-5):"
    )

    result = await runner.generate(
        system_prompt=_JUDGE_RUBRIC,
        user_prompt=user_prompt,
        temperature=0.0,
        max_tokens=8,
    )

    raw = result.get("answer", "").strip()
    try:
        score = int(raw)
        score = max(1, min(5, score))
    except ValueError:
        return {"score": 0, "error": f"unparseable judge response: {raw!r}"}
    return {"score": score}


# Batch scoring

def score_pair(predicted: str, gold: str) -> dict:
    """Compute all deterministic metrics for one (predicted, gold) pair."""
    tf1 = token_f1(predicted, gold)
    rl = rouge_l(predicted, gold)
    return {
        "token_f1":         tf1["f1"],
        "token_precision":  tf1["precision"],
        "token_recall":     tf1["recall"],
        "rouge_l_f1":       rl["f1"],
        "rouge_l_precision": rl["precision"],
        "rouge_l_recall":   rl["recall"],
    }
