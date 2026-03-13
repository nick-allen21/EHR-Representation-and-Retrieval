"""LLM-as-Judge: score predicted answers against gold using a clinical rubric.

This is a **post-processing** script — it reads existing result JSON files
produced by ``run_evaluation.py`` and adds an ``llm_judge_score`` (1–5) to
each record.  It writes an augmented copy of each file with the suffix
``_judged.json``.

The judge model (default: gpt-4o) evaluates each (question, gold, prediction)
triple against a clinical correctness rubric.

Usage
-----
# Pilot run — 50 questions per file, validate rubric
python -m Evaluation.llm_judge --results-dir data/results --limit 50

# Full run — all questions in each result file
python -m Evaluation.llm_judge --results-dir data/results

# Custom judge model and concurrency
python -m Evaluation.llm_judge --judge-model gpt-4o --concurrency 20
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
import re
import time
from pathlib import Path

from Evaluation.llm_runner import LLMRunner

log = logging.getLogger(__name__)

_RESULTS_DIR = Path("data/results")

# ── Clinical correctness rubric ──────────────────────────────────────────────

JUDGE_SYSTEM_PROMPT = """\
You are an expert clinical reviewer evaluating an AI question-answering system \
that answers questions about patient medical records (EHR discharge summaries).

Given the clinical question, the reference (gold-standard) answer, and the \
system's predicted answer, rate the predicted answer on a 1–5 scale:

5 — Completely correct and complete. All clinically relevant facts from the \
reference answer are present. Specific values (lab results, medication doses, \
dates) match the reference. No clinically significant errors or hallucinations.

4 — Mostly correct. The key clinical facts are present, but there are minor \
omissions (e.g., missing a secondary medication) or slight imprecision \
(e.g., approximate date) that would not affect clinical decision-making.

3 — Partially correct. Some important clinical facts are captured, but \
significant details are missing or imprecise. A clinician would need to \
verify the answer before acting on it.

2 — Mostly incorrect. The answer is tangentially related to the question but \
contains major factual errors, confuses different clinical events, or \
fabricates information not present in the record.

1 — Incorrect or irrelevant. The answer does not address the question, is \
empty, or contains dangerous misinformation.

IMPORTANT INSTRUCTIONS:
- Compare the predicted answer ONLY against the reference answer.
- Focus on clinical correctness, not writing style or verbosity.
- A verbose but correct answer should score 4–5, not lower.
- A concise but correct answer should score 4–5, not lower.
- If the predicted answer contains all key facts but in different words, \
that is still correct (score 4–5).
- Return ONLY a single integer (1, 2, 3, 4, or 5). No explanation."""


def _format_judge_prompt(question: str, gold: str, predicted: str) -> str:
    return (
        f"### Clinical Question\n{question}\n\n"
        f"### Reference Answer (Gold Standard)\n{gold}\n\n"
        f"### System's Predicted Answer\n{predicted}\n\n"
        f"Your rating (1–5):"
    )


def _parse_score(raw: str) -> int | None:
    """Extract a 1–5 integer from the judge's response."""
    raw = raw.strip()
    if raw in ("1", "2", "3", "4", "5"):
        return int(raw)
    match = re.search(r"\b([1-5])\b", raw)
    if match:
        return int(match.group(1))
    return None


# ── Core judging logic ───────────────────────────────────────────────────────

async def judge_single(
    runner: LLMRunner,
    question: str,
    gold: str,
    predicted: str,
) -> dict:
    """Judge a single (question, gold, predicted) triple. Returns score dict."""
    user_prompt = _format_judge_prompt(question, gold, predicted)
    result = await runner.generate(
        system_prompt=JUDGE_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        temperature=0.0,
        max_tokens=8,
    )
    raw = result.get("answer", "").strip()
    score = _parse_score(raw)
    out = {
        "llm_judge_score": score if score is not None else 0,
        "llm_judge_raw": raw,
        "llm_judge_cached": result.get("cached", False),
    }
    if score is None:
        out["llm_judge_error"] = f"unparseable: {raw!r}"
        log.warning("Unparseable judge response: %r", raw)
    return out


async def judge_result_file(
    runner: LLMRunner,
    input_path: Path,
    output_path: Path,
    limit: int | None = None,
    seed: int = 42,
) -> dict:
    """Judge all (or a sample of) results in a file. Returns summary stats."""
    records = json.loads(input_path.read_text())
    if not isinstance(records, list):
        log.warning("Skipping %s: not a list", input_path.name)
        return {}

    if limit and limit < len(records):
        rng = random.Random(seed)
        indices = sorted(rng.sample(range(len(records)), limit))
        sample = [records[i] for i in indices]
    else:
        sample = records
        indices = list(range(len(records)))

    log.info(
        "Judging %s: %d / %d records",
        input_path.name, len(sample), len(records),
    )

    tasks = []
    for rec in sample:
        question = rec.get("question", "")
        gold = rec.get("gold_answer", "")
        predicted = rec.get("answer", "")
        tasks.append(judge_single(runner, question, gold, predicted))

    scores = await asyncio.gather(*tasks)

    scored_records = list(records)
    for idx, score_dict in zip(indices, scores):
        scored_records[idx] = {**records[idx], **score_dict}

    output_path.write_text(json.dumps(scored_records, indent=2))
    log.info("Wrote judged results to %s", output_path.name)

    valid_scores = [s["llm_judge_score"] for s in scores if s["llm_judge_score"] > 0]
    errors = sum(1 for s in scores if s.get("llm_judge_error"))
    cached = sum(1 for s in scores if s.get("llm_judge_cached"))

    stats = {
        "file": input_path.name,
        "total": len(sample),
        "scored": len(valid_scores),
        "errors": errors,
        "cached": cached,
        "mean_score": sum(valid_scores) / len(valid_scores) if valid_scores else 0,
        "score_dist": {
            str(i): valid_scores.count(i) for i in range(1, 6)
        },
    }
    return stats


# ── CLI ───────────────────────────────────────────────────────────────────────

async def _run(args: argparse.Namespace) -> None:
    runner = LLMRunner(
        model=args.judge_model,
        cache_dir=args.cache_dir,
        concurrency=args.concurrency,
    )

    results_dir = Path(args.results_dir)
    output_dir = results_dir / "judged"
    output_dir.mkdir(parents=True, exist_ok=True)

    result_files = sorted(results_dir.glob("*__*.json"))
    if not result_files:
        log.error("No result files matching *__*.json in %s", results_dir)
        return

    log.info("Found %d result files to judge", len(result_files))
    all_stats: list[dict] = []
    t0 = time.time()

    for rf in result_files:
        out_path = output_dir / rf.name.replace(".json", "_judged.json")
        stats = await judge_result_file(
            runner, rf, out_path,
            limit=args.limit,
            seed=args.seed,
        )
        if stats:
            all_stats.append(stats)
            log.info(
                "  %s: mean=%.2f, scored=%d/%d, errors=%d, cached=%d",
                stats["file"], stats["mean_score"],
                stats["scored"], stats["total"],
                stats["errors"], stats["cached"],
            )

    elapsed = time.time() - t0
    log.info("Done in %.1f seconds", elapsed)

    summary_path = output_dir / "judge_summary.json"
    summary_path.write_text(json.dumps(all_stats, indent=2))
    log.info("Summary written to %s", summary_path)

    runner_stats = runner.stats()
    log.info(
        "Runner stats: total=%d, cached=%d, errors=%d",
        runner_stats.get("total", 0),
        runner_stats.get("cache_hits", 0),
        runner_stats.get("errors", 0),
    )

    print("\n=== LLM-as-Judge Summary ===\n")
    print(f"{'File':<55s}  {'Mean':>5s}  {'N':>4s}  {'Err':>3s}")
    print("-" * 75)
    for s in all_stats:
        print(
            f"{s['file']:<55s}  {s['mean_score']:>5.2f}  "
            f"{s['scored']:>4d}  {s['errors']:>3d}"
        )
    print("-" * 75)
    if all_stats:
        overall = sum(s["mean_score"] * s["scored"] for s in all_stats) / max(
            sum(s["scored"] for s in all_stats), 1
        )
        print(f"{'Overall weighted mean':<55s}  {overall:>5.2f}")


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    p = argparse.ArgumentParser(
        description="Run LLM-as-judge scoring on evaluation result files",
    )
    p.add_argument(
        "--results-dir", type=Path, default=_RESULTS_DIR,
        help="Directory containing {model}__{method}.json files",
    )
    p.add_argument(
        "--judge-model", default="gpt-4o",
        help="OpenAI model to use as judge (default: gpt-4o)",
    )
    p.add_argument(
        "--limit", type=int, default=None,
        help="Max questions to judge per file (random sample; default: all)",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for sampling (default: 42)",
    )
    p.add_argument(
        "--concurrency", type=int, default=15,
        help="Max concurrent API calls (default: 15)",
    )
    p.add_argument(
        "--cache-dir", type=Path, default=Path("data/results/cache"),
        help="Disk cache directory for judge responses",
    )
    args = p.parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
