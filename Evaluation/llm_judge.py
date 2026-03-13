"""LLM-as-judge scoring and ranking for evaluation result files.

Scores each (question, predicted_answer, gold_answer) triple on a 1–5
correctness scale, then cross-references scores across strategies to rank
which retrieval method produces the best answers per question.

Two phases:
  1. Score  — call the judge LLM for each row in each target file; write
              judge_score back into the result file (idempotent, cached).
  2. Rank   — align scored rows across strategies by question identity,
              compute win counts, mean scores, and a strategy ranking table.

Usage
-----
# Default: score full_context, semantic_rag, learned — 200 rows each, gpt-4o-mini
python -m Evaluation.llm_judge

# Specific files
python -m Evaluation.llm_judge \\
    --files data/results/o4-mini__full_context.json \\
            data/results/o4-mini__semantic_rag_k5.json \\
            data/results/o4-mini__learned_k5.json

# Rank only (no new scoring — use cached scores already in the files)
python -m Evaluation.llm_judge --rank-only

# Re-score everything from scratch
python -m Evaluation.llm_judge --rescore --limit 200
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from collections import defaultdict
from pathlib import Path

from Evaluation.llm_runner import LLMRunner
from Evaluation.scoring import llm_judge

log = logging.getLogger(__name__)

_RESULTS_DIR = Path("data/results")

_DEFAULT_FILES = [
    "data/results/o4-mini__full_context.json",
    "data/results/o4-mini__semantic_rag_k5.json",
    "data/results/o4-mini__learned_k5.json",
]


# ── Phase 1: Score ─────────────────────────────────────────────────────────────

async def _score_file(
    path: Path,
    runner: LLMRunner,
    limit: int | None,
    rescore: bool,
) -> int:
    """Add judge_score to rows that don't have one yet. Returns count newly scored."""
    rows = json.loads(path.read_text())

    if not isinstance(rows, list) or not rows or "gold_answer" not in rows[0]:
        log.warning("Skipping %s — not a per-row result file", path.name)
        return 0

    to_score = [r for r in rows if rescore or "judge_score" not in r]
    if limit is not None:
        to_score = to_score[:limit]

    if not to_score:
        already = sum(1 for r in rows if r.get("judge_score", 0) > 0)
        log.info("%s: all rows already scored (%d total) — skipping", path.name, already)
        return 0

    log.info("%s: scoring %d / %d rows with %s", path.name, len(to_score), len(rows), runner.model)

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
    n_valid = sum(1 for r in rows if r.get("judge_score", 0) > 0)
    log.info("%s: wrote %d scores (%d / %d rows now have scores)", path.name, len(to_score), n_valid, len(rows))
    return len(to_score)


# ── Phase 2: Rank ──────────────────────────────────────────────────────────────

def _question_key(row: dict) -> tuple:
    """Stable identifier for a question across strategy files."""
    return (int(row["subject_id"]), int(row["hadm_id"]), row["question"].strip())


def _rank_strategies(files: list[Path]) -> None:
    """Cross-reference judge scores across strategy files and print rankings."""

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
        return

    strategies = sorted(strategy_index.keys())

    # Find questions that appear in ALL strategies (for fair comparison)
    common_keys = set.intersection(*[set(idx.keys()) for idx in strategy_index.values()])
    log.info("Questions scored in all %d strategies: %d", len(strategies), len(common_keys))

    if not common_keys:
        print("No questions scored across all strategies yet. Run scoring first.")
        return

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

    # Print ranking table sorted by win rate
    ranked = sorted(strategies, key=lambda s: -wins[s])

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

    # Per-difficulty breakdown if available
    difficulty_scores: dict[str, dict[str, list[float]]] = {s: defaultdict(list) for s in strategies}
    for key in common_keys:
        diff = strategy_index[strategies[0]][key].get("difficulty", "unknown") or "unknown"
        for s in strategies:
            difficulty_scores[s][diff].append(strategy_index[s][key]["judge_score"])

    difficulties = sorted({
        strategy_index[strategies[0]][k].get("difficulty", "unknown") or "unknown"
        for k in common_keys
    })
    if len(difficulties) > 1:
        print(f"\n  Mean judge score by difficulty:")
        header = f"  {'Strategy':<38}" + "".join(f"  {d:>8}" for d in difficulties)
        print(header)
        for strat in ranked:
            row_str = "".join(
                f"  {sum(difficulty_scores[strat][d])/len(difficulty_scores[strat][d]):>8.3f}"
                if difficulty_scores[strat][d] else f"  {'—':>8}"
                for d in difficulties
            )
            print(f"  {strat:<38}{row_str}")

    print(f"{'='*72}\n")


# ── Orchestrator ───────────────────────────────────────────────────────────────

async def run(args: argparse.Namespace) -> None:
    files = [Path(f) for f in args.files]

    missing = [f for f in files if not f.exists()]
    if missing:
        log.error("File(s) not found: %s", missing)
        return

    if not args.rank_only:
        runner = LLMRunner(
            model=args.judge_model,
            cache_dir=args.cache_dir,
            concurrency=args.concurrency,
        )
        log.info("Scoring %d file(s) with %s (limit=%s/file)", len(files), args.judge_model, args.limit)

        for path in files:
            await _score_file(path, runner, limit=args.limit, rescore=args.rescore)

        rstat = runner.stats
        log.info(
            "Scoring done — %d cache hits, %d API calls, %d errors",
            rstat["cache_hits"], rstat["api_calls"], rstat["errors"],
        )

    _rank_strategies(files)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    p = argparse.ArgumentParser(
        description="LLM-as-judge scoring and cross-strategy ranking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--files", nargs="+", default=_DEFAULT_FILES, metavar="PATH",
        help="Result JSON files to judge (default: full_context, semantic_rag, learned)",
    )
    p.add_argument(
        "--judge-model", default="gpt-4o-mini",
        help="OpenAI model used as judge (default: gpt-4o-mini)",
    )
    p.add_argument(
        "--limit", type=int, default=200,
        help="Max rows to newly score per file (default: 200)",
    )
    p.add_argument(
        "--cache-dir", default="data/results/cache",
        help="LLM response cache directory",
    )
    p.add_argument(
        "--concurrency", type=int, default=15,
        help="Max concurrent judge API calls",
    )
    p.add_argument(
        "--rescore", action="store_true",
        help="Re-score rows that already have a judge_score",
    )
    p.add_argument(
        "--rank-only", action="store_true",
        help="Skip scoring; just print rankings from existing judge_score fields",
    )
    args = p.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
