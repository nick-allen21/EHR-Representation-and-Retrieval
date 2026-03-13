"""Post-processing LLM-as-judge scoring for evaluation results.

Reads existing result files (one per retrieval strategy), calls an LLM judge
to rate each (question, predicted_answer, gold_answer) triple on a 1–5
correctness scale, and writes judge_score back into the same result files.

Scoring is idempotent: rows that already have a judge_score are skipped unless
--rescore is passed.  Use --limit to cap the number of newly scored rows per
file (useful for cost-controlled pilots before a full run).

Usage
-----
# Pilot: score up to 200 rows per file using gpt-4o
python -m Evaluation.llm_judge --limit 200

# Score only one model's files
python -m Evaluation.llm_judge --pattern "o4-mini__*.json" --limit 200

# Full run (all rows, all files) — expensive
python -m Evaluation.llm_judge

# Re-score rows that already have a score
python -m Evaluation.llm_judge --rescore --limit 50
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path

from Evaluation.llm_runner import LLMRunner
from Evaluation.scoring import llm_judge

log = logging.getLogger(__name__)

_RESULTS_DIR = Path("data/results")


async def _score_file(
    path: Path,
    runner: LLMRunner,
    limit: int | None,
    rescore: bool,
) -> dict:
    """Add judge scores to one result file in-place.

    Returns a summary dict with n_scored and mean_judge_score, or {} if
    the file was skipped.
    """
    rows = json.loads(path.read_text())

    # Skip non-result files (summary.json, etc.)
    if not isinstance(rows, list) or not rows or "gold_answer" not in rows[0]:
        log.debug("Skipping %s (not a per-row result file)", path.name)
        return {}

    to_score = [
        r for r in rows
        if rescore or "judge_score" not in r
    ]
    if limit is not None:
        to_score = to_score[:limit]

    if not to_score:
        log.info("%s: all rows already scored — skipping", path.name)
        return {}

    log.info("%s: scoring %d / %d rows", path.name, len(to_score), len(rows))

    judge_results = await asyncio.gather(*[
        llm_judge(
            question=r["question"],
            predicted=r.get("answer", ""),
            gold=r["gold_answer"],
            runner=runner,
        )
        for r in to_score
    ])

    for row, jr in zip(to_score, judge_results):
        row["judge_score"] = jr["score"]
        if "error" in jr:
            row["judge_error"] = jr["error"]
        else:
            row.pop("judge_error", None)

    path.write_text(json.dumps(rows, indent=2))

    valid = [r for r in rows if r.get("judge_score", 0) > 0]
    mean_score = sum(r["judge_score"] for r in valid) / len(valid) if valid else 0.0
    log.info(
        "%s: done — mean judge score %.3f (n_valid=%d / n_total=%d)",
        path.name, mean_score, len(valid), len(rows),
    )
    return {"n_scored": len(to_score), "mean_judge_score": mean_score}


async def run(args: argparse.Namespace) -> None:
    runner = LLMRunner(
        model=args.judge_model,
        cache_dir=args.cache_dir,
        concurrency=args.concurrency,
    )

    results_dir = Path(args.results_dir)
    paths = sorted(results_dir.glob(args.pattern))
    if not paths:
        log.error("No files matching %r in %s", args.pattern, results_dir)
        return

    log.info(
        "Judging %d file(s) with %s (limit=%s per file)",
        len(paths), args.judge_model, args.limit,
    )

    summary: dict[str, dict] = {}
    for path in paths:
        stats = await _score_file(
            path, runner, limit=args.limit, rescore=args.rescore,
        )
        if stats:
            summary[path.stem] = stats

    if summary:
        print("\n=== Judge Score Summary ===")
        for stem, s in summary.items():
            print(
                f"  {stem:<40s}  mean={s['mean_judge_score']:.3f}"
                f"  (n_newly_scored={s['n_scored']})"
            )

    rstat = runner.stats
    log.info(
        "Runner: %d cache hits, %d API calls, %d errors",
        rstat["cache_hits"], rstat["api_calls"], rstat["errors"],
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    p = argparse.ArgumentParser(
        description="LLM-as-judge post-processing for evaluation result files",
    )
    p.add_argument(
        "--results-dir", default=str(_RESULTS_DIR),
        help="Directory containing result JSON files",
    )
    p.add_argument(
        "--pattern", default="*.json",
        help="Glob pattern for result files to judge (default: *.json)",
    )
    p.add_argument(
        "--judge-model", default="gpt-4o",
        help="OpenAI model to use as judge (default: gpt-4o)",
    )
    p.add_argument(
        "--limit", type=int, default=None,
        help="Max rows to newly score per file — for cost-controlled pilots",
    )
    p.add_argument(
        "--cache-dir", default="data/results/cache",
        help="LLM response cache directory",
    )
    p.add_argument(
        "--concurrency", type=int, default=10,
        help="Max concurrent judge API calls",
    )
    p.add_argument(
        "--rescore", action="store_true",
        help="Re-score rows that already have a judge_score",
    )
    args = p.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
