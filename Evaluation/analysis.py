"""Aggregate evaluation results and produce comparison tables and plots.

Reads JSON result files from data/results/ (one per strategy), computes
summary statistics, and optionally generates bar charts and efficiency
curves.

Usage
-----
python -m Evaluation.analysis                          # print summary table
python -m Evaluation.analysis --plots                  # also save plots
python -m Evaluation.analysis --results-dir data/results --plots
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

_RESULTS_DIR = Path("data/results")
_PLOTS_DIR = Path("data/results/plots")

_SKIP_FILES = {"cache", "summary"}


# ── Loading ───────────────────────────────────────────────────────────────────

def load_all_results(results_dir: Path) -> dict[str, list[dict]]:
    """Load all strategy result files into {method_name: [result_dicts]}."""
    results: dict[str, list[dict]] = {}
    for p in sorted(results_dir.glob("*.json")):
        if p.stem in _SKIP_FILES:
            continue
        data = json.loads(p.read_text())
        if not data:
            continue
        method = data[0].get("method", p.stem)
        results[method] = data
        log.info("Loaded %s: %d results", method, len(data))
    return results


# ── Summary table ─────────────────────────────────────────────────────────────

def summary_table(results: dict[str, list[dict]]) -> str:
    """Build a formatted comparison table across strategies."""
    header = (
        f"{'Strategy':<20s}  {'N':>5s}  "
        f"{'Token-F1':>9s}  {'ROUGE-L':>9s}  "
        f"{'Ctx Tokens':>10s}  {'Prompt Tok':>10s}"
    )
    sep = "-" * len(header)
    lines = [sep, header, sep]

    for method in sorted(results.keys()):
        rows = results[method]
        n = len(rows)
        mean_tf1 = np.mean([r.get("token_f1", 0) for r in rows])
        mean_rl = np.mean([r.get("rouge_l_f1", 0) for r in rows])
        mean_ctx = np.mean([r.get("context_tokens", 0) for r in rows])
        mean_prompt = np.mean([r.get("prompt_tokens", 0) for r in rows])
        lines.append(
            f"{method:<20s}  {n:>5d}  "
            f"{mean_tf1:>9.4f}  {mean_rl:>9.4f}  "
            f"{mean_ctx:>10.0f}  {mean_prompt:>10.0f}"
        )

    lines.append(sep)
    return "\n".join(lines)


# ── Per-difficulty breakdown ──────────────────────────────────────────────────

def difficulty_breakdown(results: dict[str, list[dict]]) -> str:
    """Break down scores by question difficulty (easy / medium / hard)."""
    lines: list[str] = []
    for method in sorted(results.keys()):
        rows = results[method]
        by_diff: dict[str, list[float]] = {}
        for r in rows:
            d = r.get("difficulty", "unknown") or "unknown"
            by_diff.setdefault(d, []).append(r.get("token_f1", 0))

        parts = []
        for d in ["easy", "medium", "hard", "unknown"]:
            vals = by_diff.get(d, [])
            if vals:
                parts.append(f"{d}={np.mean(vals):.3f} (n={len(vals)})")
        lines.append(f"  {method}: {', '.join(parts)}")

    return "Per-difficulty Token-F1:\n" + "\n".join(lines)


# ── Plots ─────────────────────────────────────────────────────────────────────

def save_plots(results: dict[str, list[dict]], output_dir: Path) -> None:
    """Generate comparison bar charts and save as PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    methods = sorted(results.keys())

    # -- Bar chart: Token-F1 and ROUGE-L per strategy --
    tf1_means = [np.mean([r.get("token_f1", 0) for r in results[m]]) for m in methods]
    rl_means = [np.mean([r.get("rouge_l_f1", 0) for r in results[m]]) for m in methods]

    x = np.arange(len(methods))
    w = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - w / 2, tf1_means, w, label="Token F1")
    ax.bar(x + w / 2, rl_means, w, label="ROUGE-L F1")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=30, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Retrieval Strategy Comparison")
    ax.legend()
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(output_dir / "strategy_comparison.png", dpi=150)
    plt.close(fig)
    log.info("Saved strategy_comparison.png")

    # -- Efficiency curve: Token-F1 vs context tokens --
    fig, ax = plt.subplots(figsize=(8, 5))
    for m in methods:
        rows = results[m]
        ctx = np.mean([r.get("context_tokens", 0) for r in rows])
        tf1 = np.mean([r.get("token_f1", 0) for r in rows])
        ax.scatter(ctx, tf1, s=80, zorder=3)
        ax.annotate(m, (ctx, tf1), textcoords="offset points", xytext=(5, 5), fontsize=8)
    ax.set_xlabel("Mean Context Tokens")
    ax.set_ylabel("Mean Token F1")
    ax.set_title("Context Efficiency")
    fig.tight_layout()
    fig.savefig(output_dir / "context_efficiency.png", dpi=150)
    plt.close(fig)
    log.info("Saved context_efficiency.png")


# ── Structured export ─────────────────────────────────────────────────────────

def export_summary_json(results: dict[str, list[dict]], output_path: Path) -> None:
    """Write a machine-readable summary JSON."""
    summary = {}
    for method, rows in results.items():
        summary[method] = {
            "n": len(rows),
            "token_f1_mean": float(np.mean([r.get("token_f1", 0) for r in rows])),
            "rouge_l_f1_mean": float(np.mean([r.get("rouge_l_f1", 0) for r in rows])),
            "context_tokens_mean": float(np.mean([r.get("context_tokens", 0) for r in rows])),
            "prompt_tokens_mean": float(np.mean([r.get("prompt_tokens", 0) for r in rows])),
        }
    output_path.write_text(json.dumps(summary, indent=2))
    log.info("Summary JSON saved to %s", output_path)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    p = argparse.ArgumentParser(description="Analyze evaluation results")
    p.add_argument("--results-dir", type=Path, default=_RESULTS_DIR)
    p.add_argument("--plots", action="store_true", help="Generate comparison plots")
    args = p.parse_args()

    results = load_all_results(args.results_dir)
    if not results:
        log.error("No result files found in %s", args.results_dir)
        return

    print(summary_table(results))
    print()
    print(difficulty_breakdown(results))

    export_summary_json(results, args.results_dir / "summary.json")

    if args.plots:
        save_plots(results, _PLOTS_DIR)


if __name__ == "__main__":
    main()
