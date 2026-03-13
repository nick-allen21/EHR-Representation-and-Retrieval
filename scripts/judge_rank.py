"""Read-only LLM-as-judge ranking script.

Reads existing judge_score fields from result JSON files (no API calls, no file writes).
Prints rankings, score distribution, and per-difficulty breakdown to stdout.

Usage
-----
# Default: all 3 strategy files, all scored questions
python3 scripts/judge_rank.py

# Random sample of N questions (reproducible with seed)
python3 scripts/judge_rank.py --sample 100
python3 scripts/judge_rank.py --sample 100 --seed 7

# Filter to a specific difficulty
python3 scripts/judge_rank.py --difficulty hard

# Custom files
python3 scripts/judge_rank.py --files data/results/o4-mini__learned_k5.json data/results/o4-mini__full_context.json
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

_DEFAULT_FILES = [
    "data/results/o4-mini__full_context.json",
    "data/results/o4-mini__semantic_rag_k5.json",
    "data/results/o4-mini__learned_k5.json",
]


def qkey(r: dict) -> tuple:
    return (int(r["subject_id"]), int(r["hadm_id"]), r["question"].strip())


def main() -> None:
    p = argparse.ArgumentParser(description="Print judge rankings from cached scores (no API calls)")
    p.add_argument("--files", nargs="+", default=_DEFAULT_FILES, metavar="PATH",
                   help="Result JSON files to rank (default: full_context, semantic_rag, learned)")
    p.add_argument("--sample", type=int, default=None, metavar="N",
                   help="Randomly sample N questions (default: all scored questions)")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for sampling (default: 42)")
    p.add_argument("--difficulty", choices=["easy", "medium", "hard"], default=None,
                   help="Filter to a specific difficulty level")
    args = p.parse_args()

    # Load scored rows from each file
    idx: dict[str, dict[tuple, dict]] = {}
    for f in args.files:
        path = Path(f)
        if not path.exists():
            print(f"[skip] {path} not found")
            continue
        rows = json.loads(path.read_text())
        if not isinstance(rows, list) or not rows or "gold_answer" not in rows[0]:
            print(f"[skip] {path.name} — not a valid result file")
            continue
        idx[path.stem] = {qkey(r): r for r in rows if r.get("judge_score", 0) > 0}

    if not idx:
        print("No scored files found. Run the judge pipeline first.")
        return

    strategies = sorted(idx.keys())

    # Common questions across all strategies
    common = list(set.intersection(*[set(v.keys()) for v in idx.values()]))

    # Optional difficulty filter
    if args.difficulty:
        common = [k for k in common
                  if (idx[strategies[0]][k].get("difficulty") or "unknown") == args.difficulty]
        if not common:
            print(f"No scored questions with difficulty='{args.difficulty}'")
            return

    # Optional random sample
    if args.sample and args.sample < len(common):
        rng = random.Random(args.seed)
        common = rng.sample(common, args.sample)

    n = len(common)
    filter_note = f", difficulty={args.difficulty}" if args.difficulty else ""
    sample_note = f", sample={n}" if args.sample else ""
    print(f"\nLoaded {n} common scored questions{filter_note}{sample_note}")

    # Aggregate
    wins: dict[str, float] = defaultdict(float)
    scores: dict[str, list] = defaultdict(list)
    score_dist: dict[str, dict] = {s: defaultdict(int) for s in strategies}
    diff_scores: dict[str, dict] = {s: defaultdict(list) for s in strategies}

    for key in common:
        row_scores = {s: idx[s][key]["judge_score"] for s in strategies}
        mx = max(row_scores.values())
        winners = [s for s, sc in row_scores.items() if sc == mx]
        for w in winners:
            wins[w] += 1.0 / len(winners)
        diff = (idx[strategies[0]][key].get("difficulty") or "unknown")
        for s in strategies:
            scores[s].append(row_scores[s])
            score_dist[s][row_scores[s]] += 1
            diff_scores[s][diff].append(row_scores[s])

    ranked = sorted(strategies, key=lambda s: -wins[s])

    # Rankings table
    print(f"\n{'='*72}")
    print(f"  LLM-as-Judge Rankings  (n={n} questions, seed={args.seed})")
    print(f"{'='*72}")
    print(f"  {'#':<3} {'Strategy':<38} {'Mean Score':>10} {'Wins':>8} {'Win %':>8}")
    print(f"  {'-'*3} {'-'*38} {'-'*10} {'-'*8} {'-'*8}")
    for rank, s in enumerate(ranked, 1):
        mean = sum(scores[s]) / len(scores[s])
        wp = 100.0 * wins[s] / n
        print(f"  {rank:<3} {s:<38} {mean:>10.3f} {wins[s]:>8.1f} {wp:>7.1f}%")

    # Score distribution
    print(f"\n  Score distribution (1=wrong → 5=perfect):")
    print("  " + f"{'Strategy':<38}" + "".join(f"  [{i}]" for i in range(1, 6)))
    for s in ranked:
        dist_str = "".join(f"{score_dist[s].get(i, 0):>5}" for i in range(1, 6))
        print(f"  {s:<38}{dist_str}")

    # Per-difficulty breakdown (skip if filtered to single difficulty)
    difficulties = sorted({(idx[strategies[0]][k].get("difficulty") or "unknown") for k in common})
    if len(difficulties) > 1:
        print(f"\n  Mean judge score by difficulty:")
        print("  " + f"{'Strategy':<38}" + "".join(f"  {d:>8}" for d in difficulties))
        for s in ranked:
            row = "".join(
                f"  {sum(diff_scores[s][d])/len(diff_scores[s][d]):>8.3f}"
                if diff_scores[s][d] else f"  {'—':>8}"
                for d in difficulties
            )
            print(f"  {s:<38}{row}")

    print(f"{'='*72}\n")


if __name__ == "__main__":
    main()
