"""LLM-as-Judge: score predicted answers against gold using a clinical rubric.

Two modes:

  **Normal run** (default): Adds ``judge_score`` (1–5) in-place to each result
  file and saves rankings to ``judge_rankings.json`` and ``summary.json``.

  **Dry-run** (``--dry-run``): Makes fresh API calls on a sample (bypasses the
  main cache), prints the ranking table to stdout, and writes nothing to disk.
  Useful for spot-checks and sanity tests.

Usage
-----
# Score all *__*.json files in data/results/ in-place
python -m Evaluation.llm_judge

# Score specific files only
python -m Evaluation.llm_judge \\
    --files data/results/o4-mini__full_context.json \\
            data/results/o4-mini__semantic_rag_k5.json \\
            data/results/o4-mini__learned_k5.json

# Pilot / re-score: random sample of 100 questions per file
python -m Evaluation.llm_judge --limit 100

# Dry-run: fresh API calls on 50 questions, print table only, write nothing
python -m Evaluation.llm_judge --dry-run --limit 50

# Dry-run filtered to hard questions
python -m Evaluation.llm_judge --dry-run --limit 30 --difficulty hard

# Cheaper/faster judge model
python -m Evaluation.llm_judge --dry-run --limit 50 --judge-model gpt-4o-mini

# Rank only (no new API calls — reads existing judge_score fields)
python -m Evaluation.llm_judge --rank-only
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
import re
import tempfile
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from Evaluation.llm_runner import LLMRunner

log = logging.getLogger(__name__)

_RESULTS_DIR = Path("data/results")

_DEFAULT_FILES = [
    "data/results/o4-mini__full_context.json",
    "data/results/o4-mini__semantic_rag_k5.json",
    "data/results/o4-mini__learned_k5.json",
]

# ── Clinical correctness rubric ───────────────────────────────────────────────

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


def _question_key(row: dict) -> tuple:
    """Stable identifier for a question across strategy files."""
    return (int(row["subject_id"]), int(row["hadm_id"]), row["question"].strip())


# ── Core judge call ───────────────────────────────────────────────────────────

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
        "score": score if score is not None else 0,
        "raw": raw,
        "cached": result.get("cached", False),
    }
    if score is None:
        out["error"] = f"unparseable: {raw!r}"
        log.warning("Unparseable judge response: %r", raw)
    return out


# ── Phase 1: Score files ──────────────────────────────────────────────────────

async def _score_file(
    runner: LLMRunner,
    path: Path,
    limit: int | None = None,
    seed: int = 42,
    rescore: bool = False,
) -> int:
    """Add ``judge_score`` in-place to rows in *path*.

    Only rows without a score are judged unless *rescore=True*. Returns the
    number of rows judged in this call.
    """
    rows = json.loads(path.read_text())
    if not isinstance(rows, list) or not rows or "gold_answer" not in rows[0]:
        log.warning("Skipping %s: not a valid result file", path.name)
        return 0

    # Identify rows that need scoring
    if rescore:
        to_score = list(rows)
    else:
        to_score = [r for r in rows if r.get("judge_score", 0) == 0]

    if not to_score:
        log.info("%s: all rows already scored — skipping (use --rescore to force)", path.name)
        return 0

    if limit and limit < len(to_score):
        rng = random.Random(seed)
        to_score = rng.sample(to_score, limit)

    log.info("Scoring %d / %d rows in %s", len(to_score), len(rows), path.name)

    judge_results = await asyncio.gather(*[
        judge_single(runner, r.get("question", ""), r.get("gold_answer", ""), r.get("answer", ""))
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
    log.info(
        "%s: wrote %d scores (%d / %d rows now have scores)",
        path.name, len(to_score), n_valid, len(rows),
    )
    return len(to_score)


# ── Phase 2: Rank ─────────────────────────────────────────────────────────────

def _rank_strategies(
    files: list[Path],
    scored_rows: dict[str, list[dict]] | None = None,
    difficulty_filter: str | None = None,
) -> dict | None:
    """Cross-reference judge scores across strategy files, print rankings, return data dict.

    If *scored_rows* is provided (dry-run path), use those rows instead of
    reading from files.
    """
    strategy_index: dict[str, dict[tuple, dict]] = {}

    if scored_rows is not None:
        # Dry-run: use the in-memory scored rows
        for name, rows in scored_rows.items():
            strategy_index[name] = {
                _question_key(r): r
                for r in rows
                if r.get("judge_score", 0) > 0
            }
    else:
        for path in files:
            rows = json.loads(path.read_text())
            if not isinstance(rows, list) or not rows or "gold_answer" not in rows[0]:
                continue
            name = path.stem
            strategy_index[name] = {
                _question_key(r): r
                for r in rows
                if r.get("judge_score", 0) > 0
            }

    if not strategy_index:
        print("No scored rows found. Run without --rank-only first.")
        return None

    strategies = sorted(strategy_index.keys())

    # Questions scored in ALL strategies
    common_keys = set.intersection(*[set(idx.keys()) for idx in strategy_index.values()])

    if difficulty_filter:
        common_keys = {
            k for k in common_keys
            if (strategy_index[strategies[0]][k].get("difficulty") or "unknown") == difficulty_filter
        }

    if not common_keys:
        msg = f"No questions scored across all strategies"
        if difficulty_filter:
            msg += f" with difficulty='{difficulty_filter}'"
        print(msg)
        return None

    log.info("Questions scored in all %d strategies: %d", len(strategies), len(common_keys))

    # Aggregate
    wins: dict[str, float] = defaultdict(float)
    scores: dict[str, list[float]] = defaultdict(list)
    score_dist: dict[str, dict[int, int]] = {s: defaultdict(int) for s in strategies}
    difficulty_scores: dict[str, dict[str, list[float]]] = {s: defaultdict(list) for s in strategies}

    for key in common_keys:
        row_scores = {s: strategy_index[s][key]["judge_score"] for s in strategies}
        max_score = max(row_scores.values())
        winners = [s for s, sc in row_scores.items() if sc == max_score]
        for w in winners:
            wins[w] += 1.0 / len(winners)
        diff = strategy_index[strategies[0]][key].get("difficulty", "unknown") or "unknown"
        for s in strategies:
            scores[s].append(row_scores[s])
            score_dist[s][row_scores[s]] += 1
            difficulty_scores[s][diff].append(row_scores[s])

    n = len(common_keys)
    difficulties = sorted({
        strategy_index[strategies[0]][k].get("difficulty", "unknown") or "unknown"
        for k in common_keys
    })
    ranked = sorted(strategies, key=lambda s: -wins[s])

    # ── Print ──────────────────────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print(f"  LLM-as-Judge Rankings  (n={n} questions)")
    print(f"{'='*72}")
    print(f"  {'#':<3} {'Strategy':<38} {'Mean Score':>10} {'Wins':>8} {'Win %':>8}")
    print(f"  {'-'*3} {'-'*38} {'-'*10} {'-'*8} {'-'*8}")

    for rank, strat in enumerate(ranked, 1):
        mean = sum(scores[strat]) / len(scores[strat])
        win_pct = 100.0 * wins[strat] / n
        print(f"  {rank:<3} {strat:<38} {mean:>10.3f} {wins[strat]:>8.1f} {win_pct:>7.1f}%")

    print(f"\n  Score distribution (1=wrong → 5=perfect):")
    print("  " + f"{'Strategy':<38}" + "".join(f"  [{i}]" for i in range(1, 6)))
    for strat in ranked:
        dist_str = "".join(f"{score_dist[strat].get(i, 0):>5}" for i in range(1, 6))
        print(f"  {strat:<38}{dist_str}")

    if len(difficulties) > 1:
        print(f"\n  Mean judge score by difficulty:")
        print("  " + f"{'Strategy':<38}" + "".join(f"  {d:>8}" for d in difficulties))
        for strat in ranked:
            row_str = "".join(
                f"  {sum(difficulty_scores[strat][d])/len(difficulty_scores[strat][d]):>8.3f}"
                if difficulty_scores[strat][d] else f"  {'—':>8}"
                for d in difficulties
            )
            print(f"  {strat:<38}{row_str}")

    print(f"{'='*72}\n")

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


# ── Save helpers ──────────────────────────────────────────────────────────────

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
    """Merge judge_score_mean and judge_n into summary.json for each judged strategy."""
    if not summary_path.exists():
        log.warning("summary.json not found at %s — skipping", summary_path)
        return

    summary = json.loads(summary_path.read_text())

    for entry in data["ranked"]:
        strat = entry["strategy"]            # e.g. "o4-mini__learned_k5"
        method_key = strat.split("__", 1)[-1]  # "learned_k5"

        if method_key not in summary:
            log.warning("Strategy %r not found in summary.json — skipping", method_key)
            continue

        summary[method_key]["judge_score_mean"] = entry["mean_score"]
        summary[method_key]["judge_n"] = entry["n"]
        summary[method_key]["judge_model"] = judge_model

    summary_path.write_text(json.dumps(summary, indent=2))
    log.info("Updated judge_score_mean in %s", summary_path)


# ── CLI runner ────────────────────────────────────────────────────────────────

async def _run(args: argparse.Namespace) -> None:
    t0 = time.time()

    # Resolve file list
    if args.files:
        files = [Path(f) for f in args.files]
    else:
        files = sorted(Path(args.results_dir).glob("*__*.json"))
        if not files:
            log.error("No result files matching *__*.json in %s", args.results_dir)
            return

    # ── Dry-run: fresh API calls, print only, write nothing ───────────────────
    if args.dry_run:
        tmp_cache = tempfile.mkdtemp(prefix="judge_fresh_")
        runner = LLMRunner(
            model=args.judge_model,
            cache_dir=tmp_cache,
            concurrency=args.concurrency,
        )

        # Load all rows, find common questions across files
        idx: dict[str, dict[tuple, dict]] = {}
        for path in files:
            if not path.exists():
                print(f"[skip] {path} not found")
                continue
            rows = json.loads(path.read_text())
            if not isinstance(rows, list) or not rows or "gold_answer" not in rows[0]:
                print(f"[skip] {path.name} — not a valid result file")
                continue
            idx[path.stem] = {_question_key(r): r for r in rows}

        if not idx:
            print("No valid result files found.")
            return

        strategies = sorted(idx.keys())
        common = list(set.intersection(*[set(v.keys()) for v in idx.values()]))

        if args.difficulty:
            common = [
                k for k in common
                if (idx[strategies[0]][k].get("difficulty") or "unknown") == args.difficulty
            ]
            if not common:
                print(f"No questions with difficulty='{args.difficulty}'")
                return

        limit = args.limit or 50
        if limit < len(common):
            rng = random.Random(args.seed)
            common = rng.sample(common, limit)

        n = len(common)
        n_calls = n * len(strategies)
        diff_note = f", difficulty={args.difficulty}" if args.difficulty else ""
        print(f"\nDry-run: {n} questions{diff_note}, {len(strategies)} strategies, {n_calls} fresh API calls")
        print(f"Judge model: {args.judge_model}  |  seed={args.seed}")
        print("Running (no files will be written)...\n")

        # Score all questions concurrently
        async def _judge_one(key: tuple, strategy: str) -> dict:
            row = idx[strategy][key]
            jr = await judge_single(
                runner,
                row.get("question", ""),
                row.get("gold_answer", ""),
                row.get("answer", ""),
            )
            return {
                **row,
                "judge_score": jr["score"],
                "_strategy": strategy,
                "_key": key,
            }

        tasks = [_judge_one(k, s) for k in common for s in strategies]
        results = await asyncio.gather(*tasks)

        # Rebuild scored_rows for _rank_strategies
        scored_rows: dict[str, list[dict]] = {s: [] for s in strategies}
        for r in results:
            scored_rows[r["_strategy"]].append(r)

        _rank_strategies(files, scored_rows=scored_rows, difficulty_filter=None)
        print(f"API stats: {runner.stats}")
        print(f"(temp cache at {tmp_cache} — not saved to project)\n")
        return

    # ── Rank-only: read existing judge_score fields, no API calls ─────────────
    if args.rank_only:
        existing_files = [f for f in files if f.exists()]
        if not existing_files:
            print("No result files found.")
            return
        data = _rank_strategies(existing_files, difficulty_filter=args.difficulty)
        return

    # ── Normal run: score in-place, then save rankings ─────────────────────────
    runner = LLMRunner(
        model=args.judge_model,
        cache_dir=args.cache_dir,
        concurrency=args.concurrency,
    )

    total_judged = 0
    for path in files:
        if not path.exists():
            log.warning("File not found: %s", path)
            continue
        n = await _score_file(runner, path, limit=args.limit, seed=args.seed, rescore=args.rescore)
        total_judged += n

    elapsed = time.time() - t0
    log.info("Scoring done: %d rows judged in %.1f seconds", total_judged, elapsed)
    log.info("Runner stats: %s", runner.stats)

    # Rank and save
    existing_files = [f for f in files if f.exists()]
    data = _rank_strategies(existing_files, difficulty_filter=args.difficulty)

    if data is not None:
        results_dir = Path(args.results_dir)
        _save_judge_rankings(data, args.judge_model, existing_files, results_dir / "judge_rankings.json")
        _save_to_summary_json(data, args.judge_model, results_dir / "summary.json")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    p = argparse.ArgumentParser(
        description="Run LLM-as-judge scoring on evaluation result files",
    )

    # File selection
    p.add_argument(
        "--files", nargs="+", default=None, metavar="PATH",
        help=(
            "Specific result JSON files to judge (default: all *__*.json in --results-dir). "
            "Example: data/results/o4-mini__full_context.json data/results/o4-mini__learned_k5.json"
        ),
    )
    p.add_argument(
        "--results-dir", type=Path, default=_RESULTS_DIR,
        help="Directory containing {model}__{method}.json files (default: data/results)",
    )

    # Judge settings
    p.add_argument(
        "--judge-model", default="gpt-4o",
        help="OpenAI model used as judge (default: gpt-4o)",
    )
    p.add_argument(
        "--concurrency", type=int, default=15,
        help="Max concurrent API calls (default: 15)",
    )
    p.add_argument(
        "--cache-dir", type=Path, default=Path("data/results/cache"),
        help="Disk cache directory for judge responses (default: data/results/cache)",
    )

    # Sampling
    p.add_argument(
        "--limit", type=int, default=None,
        help="Max questions to judge per file / dry-run (random sample; default: all / 50 for dry-run)",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for sampling (default: 42)",
    )
    p.add_argument(
        "--difficulty", choices=["easy", "medium", "hard"], default=None,
        help="Filter to a specific difficulty level",
    )

    # Mode flags
    p.add_argument(
        "--dry-run", action="store_true",
        help=(
            "Fresh trial run: make real API calls on a sample, print table only, "
            "write nothing to disk. Bypasses the main cache."
        ),
    )
    p.add_argument(
        "--rank-only", action="store_true",
        help="Print rankings from existing judge_score fields only (no API calls, no writes)",
    )
    p.add_argument(
        "--rescore", action="store_true",
        help="Re-score rows that already have a judge_score (normal run only)",
    )

    args = p.parse_args()

    if args.dry_run and args.rank_only:
        p.error("--dry-run and --rank-only are mutually exclusive")

    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
