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
from collections import defaultdict
from datetime import datetime, timezone
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
<<<<<<< HEAD
        if stats:
            all_stats.append(stats)
            log.info(
                "  %s: mean=%.2f, scored=%d/%d, errors=%d, cached=%d",
                stats["file"], stats["mean_score"],
                stats["scored"], stats["total"],
                stats["errors"], stats["cached"],
=======
        for r in to_score
    ])

    for row, jr in zip(to_score, judge_results):
        row["judge_score"] = jr["score"]
        if "error" in jr:
            row["judge_error"] = jr["error"]
        else:
            row.pop("judge_error", None)

    path.write_text(json.dumps(rows, indent=2))
    n_valid = sum(1 for r in rows if r.get("judge_score", 0) > 0)
    log.info("%s: wrote %d scores (%d / %d rows now have scores)", path.name, len(to_score), n_valid, len(rows))
    return len(to_score)


# ── Phase 2: Rank ──────────────────────────────────────────────────────────────

def _question_key(row: dict) -> tuple:
    """Stable identifier for a question across strategy files."""
    return (int(row["subject_id"]), int(row["hadm_id"]), row["question"].strip())


def _rank_strategies(files: list[Path]) -> dict | None:
    """Cross-reference judge scores across strategy files, print rankings, return data dict."""

    # Load scored rows from each file, indexed by question key
    strategy_index: dict[str, dict[tuple, dict]] = {}
    for path in files:
        rows = json.loads(path.read_text())
        if not isinstance(rows, list) or not rows or "gold_answer" not in rows[0]:
            continue
        name = path.stem  # e.g. "o4-mini__learned_k5"
        strategy_index[name] = {
            _question_key(r): r
            for r in rows
            if r.get("judge_score", 0) > 0
        }

    if not strategy_index:
        print("No scored rows found. Run without --rank-only first.")
        return None

    strategies = sorted(strategy_index.keys())

    # Find questions that appear in ALL strategies (for fair comparison)
    common_keys = set.intersection(*[set(idx.keys()) for idx in strategy_index.values()])
    log.info("Questions scored in all %d strategies: %d", len(strategies), len(common_keys))

    if not common_keys:
        print("No questions scored across all strategies yet. Run scoring first.")
        return None

    # Aggregate per strategy
    wins: dict[str, float] = defaultdict(float)   # fractional wins (ties split)
    scores: dict[str, list[float]] = defaultdict(list)
    score_dist: dict[str, dict[int, int]] = {s: defaultdict(int) for s in strategies}

    for key in common_keys:
        row_scores = {s: strategy_index[s][key]["judge_score"] for s in strategies}
        max_score = max(row_scores.values())
        winners = [s for s, sc in row_scores.items() if sc == max_score]
        for w in winners:
            wins[w] += 1.0 / len(winners)  # split ties evenly
        for s in strategies:
            scores[s].append(row_scores[s])
            score_dist[s][row_scores[s]] += 1

    n = len(common_keys)

    # Per-difficulty breakdown
    difficulty_scores: dict[str, dict[str, list[float]]] = {s: defaultdict(list) for s in strategies}
    for key in common_keys:
        diff = strategy_index[strategies[0]][key].get("difficulty", "unknown") or "unknown"
        for s in strategies:
            difficulty_scores[s][diff].append(strategy_index[s][key]["judge_score"])

    difficulties = sorted({
        strategy_index[strategies[0]][k].get("difficulty", "unknown") or "unknown"
        for k in common_keys
    })

    # Rank by win rate
    ranked = sorted(strategies, key=lambda s: -wins[s])

    # ── Print ──────────────────────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print(f"  LLM-as-Judge Rankings  (n={n} questions, judge shared across all strategies)")
    print(f"{'='*72}")
    print(f"  {'#':<3} {'Strategy':<38} {'Mean Score':>10} {'Wins':>8} {'Win %':>8}")
    print(f"  {'-'*3} {'-'*38} {'-'*10} {'-'*8} {'-'*8}")

    for rank, strat in enumerate(ranked, 1):
        mean = sum(scores[strat]) / len(scores[strat])
        win_pct = 100.0 * wins[strat] / n
        print(f"  {rank:<3} {strat:<38} {mean:>10.3f} {wins[strat]:>8.1f} {win_pct:>7.1f}%")

    print(f"\n  Score distribution (1=wrong → 5=perfect):")
    score_labels = "  " + f"{'Strategy':<38}" + "".join(f"  [{i}]" for i in range(1, 6))
    print(score_labels)
    for strat in ranked:
        dist_str = "".join(f"{score_dist[strat].get(i, 0):>5}" for i in range(1, 6))
        print(f"  {strat:<38}{dist_str}")

    if len(difficulties) > 1:
        print(f"\n  Mean judge score by difficulty:")
        header = f"  {'Strategy':<38}" + "".join(f"  {d:>8}" for d in difficulties)
        print(header)
        for strat in ranked:
            row_str = "".join(
                f"  {sum(difficulty_scores[strat][d])/len(difficulty_scores[strat][d]):>8.3f}"
                if difficulty_scores[strat][d] else f"  {'—':>8}"
                for d in difficulties
>>>>>>> 86dd6d5 (update llm-as-judge score saving to results folder)
            )

    elapsed = time.time() - t0
    log.info("Done in %.1f seconds", elapsed)

<<<<<<< HEAD
    summary_path = output_dir / "judge_summary.json"
    summary_path.write_text(json.dumps(all_stats, indent=2))
    log.info("Summary written to %s", summary_path)
=======
    # ── Return structured data for saving ─────────────────────────────────────
    return {
        "n_questions": n,
        "strategies": strategies,
        "ranked": [
            {
                "rank": rank,
                "strategy": strat,
                "mean_score": round(sum(scores[strat]) / len(scores[strat]), 4),
                "wins": round(wins[strat], 2),
                "win_pct": round(100.0 * wins[strat] / n, 2),
                "n": len(scores[strat]),
                "score_dist": {str(i): score_dist[strat].get(i, 0) for i in range(1, 6)},
            }
            for rank, strat in enumerate(ranked, 1)
        ],
        "by_difficulty": {
            diff: {
                strat: round(sum(difficulty_scores[strat][diff]) / len(difficulty_scores[strat][diff]), 4)
                if difficulty_scores[strat][diff] else None
                for strat in ranked
            }
            for diff in difficulties
        },
        "per_strategy_n_scored": {
            strat: len(strategy_index[strat])
            for strat in strategies
        },
    }


def _save_judge_rankings(
    data: dict,
    judge_model: str,
    files: list[Path],
    out_path: Path,
) -> None:
    """Save the full rankings table to judge_rankings.json."""
    output = {
        "metadata": {
            "judge_model": judge_model,
            "n_questions_common": data["n_questions"],
            "files_judged": [p.stem for p in files],
            "per_strategy_n_scored": data["per_strategy_n_scored"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "note": (
                "n_questions_common is questions scored in ALL judged strategies. "
                "per_strategy_n_scored shows total scored rows per file (may differ if "
                "files were scored at different limits)."
            ),
        },
        "ranked": data["ranked"],
        "by_difficulty": data["by_difficulty"],
    }
    out_path.write_text(json.dumps(output, indent=2))
    log.info("Saved judge rankings → %s", out_path)


def _save_to_summary_json(
    data: dict,
    judge_model: str,
    summary_path: Path,
) -> None:
    """Merge judge_score_mean and judge_n into summary.json for each judged strategy.

    summary.json uses short method names (e.g. "learned_k5"); strategy names here
    include the model prefix (e.g. "o4-mini__learned_k5"). We strip the prefix to match.
    """
    if not summary_path.exists():
        log.warning("summary.json not found at %s — skipping summary update", summary_path)
        return

    summary = json.loads(summary_path.read_text())

    for entry in data["ranked"]:
        strat = entry["strategy"]  # e.g. "o4-mini__learned_k5"
        # Strip model prefix if present to match summary.json keys
        method_key = strat.split("__", 1)[-1]  # "learned_k5"

        if method_key not in summary:
            log.warning("Strategy %r (key %r) not found in summary.json — skipping", strat, method_key)
            continue

        summary[method_key]["judge_score_mean"] = entry["mean_score"]
        summary[method_key]["judge_n"] = entry["n"]
        summary[method_key]["judge_model"] = judge_model

    summary_path.write_text(json.dumps(summary, indent=2))
    log.info("Updated judge_score_mean in %s", summary_path)

>>>>>>> 86dd6d5 (update llm-as-judge score saving to results folder)

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
<<<<<<< HEAD
        print(f"{'Overall weighted mean':<55s}  {overall:>5.2f}")
=======

    data = _rank_strategies(files)

    if data is not None:
        results_dir = files[0].parent
        _save_judge_rankings(data, args.judge_model, files, results_dir / "judge_rankings.json")
        _save_to_summary_json(data, args.judge_model, results_dir / "summary.json")
>>>>>>> 86dd6d5 (update llm-as-judge score saving to results folder)


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
<<<<<<< HEAD
        help="OpenAI model to use as judge (default: gpt-4o)",
=======
        help="OpenAI model used as judge (default: gpt-4o)",
>>>>>>> 86dd6d5 (update llm-as-judge score saving to results folder)
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
