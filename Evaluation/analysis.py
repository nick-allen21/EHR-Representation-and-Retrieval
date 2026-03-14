"""Aggregate evaluation results and produce comparison tables and plots.

Reads JSON result files from data/results/ (one per model x strategy), computes
summary statistics, and optionally generates bar charts and efficiency curves.

Filenames follow the convention {model_slug}__{method}.json.
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

_SKIP_STEMS = {"summary", "cache"}

MODEL_DISPLAY = {
    "o4-mini": "o4-mini",
    "gpt-4o-mini": "gpt-4o-mini",
    "meta-llama--Llama-3.1-8B-Instruct": "Llama-3.1-8B",
    "mistralai--Mistral-7B-Instruct-v0.3": "Mistral-7B",
    "microsoft--Phi-3-mini-4k-instruct": "Phi-3-mini-4k",
}

MODEL_ORDER = list(MODEL_DISPLAY.keys())

METHOD_ORDER = [
    "discharge_only",
    "full_context",
    "recency_n25",
    "bm25_k5",
    "semantic_rag_k5",
    "learned_k5",
]

METHOD_DISPLAY = {
    "discharge_only": "discharge-only",
    "full_context": "full-context",
    "recency_n25": "recency",
    "bm25_k5": "BM25",
    "semantic_rag_k5": "semantic RAG",
    "learned_k5": "learned",
}


# Loading

def load_all_results(
    results_dir: str | Path = _RESULTS_DIR,
) -> dict[tuple[str, str], list[dict]]:
    """Load result files into {(model_slug, method): [result_dicts]}.

    Parses model and method from {model}__{method}.json filenames.
    Falls back to ("unknown", stem) for legacy files.
    """
    results_dir = Path(results_dir)
    data: dict[tuple[str, str], list[dict]] = {}
    for p in sorted(results_dir.glob("*.json")):
        if p.stem in _SKIP_STEMS or p.is_dir():
            continue
        if "__" in p.stem:
            model_slug, method = p.stem.split("__", 1)
        else:
            model_slug, method = "unknown", p.stem
        try:
            rows = json.loads(p.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            log.warning("Skipping %s: %s", p.name, exc)
            continue
        if not isinstance(rows, list) or not rows:
            continue
        data[(model_slug, method)] = rows
        log.info("Loaded %-55s %d results", p.name, len(rows))
    return data


def _mean(rows: list[dict], key: str) -> float:
    vals = [r.get(key, 0) for r in rows]
    return float(np.mean(vals)) if vals else 0.0


# Multi-model matrix

def model_method_matrix(
    data: dict[tuple[str, str], list[dict]],
    metric: str = "token_f1",
) -> dict[str, dict[str, float | None]]:
    """Build {model_slug: {method: mean_metric}} for every populated cell."""
    matrix: dict[str, dict[str, float | None]] = {}
    for (model, method), rows in data.items():
        matrix.setdefault(model, {})[method] = _mean(rows, metric)
    return matrix


def format_matrix_table(
    data: dict[tuple[str, str], list[dict]],
    metric: str = "token_f1",
) -> str:
    """Return a markdown table of the model × method matrix."""
    matrix = model_method_matrix(data, metric)

    models = [m for m in MODEL_ORDER if m in matrix]
    for m in sorted(matrix.keys()):
        if m not in models:
            models.append(m)

    methods = [m for m in METHOD_ORDER if any(m in matrix.get(mod, {}) for mod in models)]
    for m in sorted({meth for d in matrix.values() for meth in d}):
        if m not in methods:
            methods.append(m)

    header_labels = [METHOD_DISPLAY.get(m, m) for m in methods]
    header = "|  | " + " | ".join(header_labels) + " |"
    sep = "|---|" + "|".join(["---"] * len(methods)) + "|"

    rows_out = [header, sep]
    for model in models:
        label = MODEL_DISPLAY.get(model, model)
        cells = []
        row_vals = matrix.get(model, {})
        best_val = max((v for v in row_vals.values() if v is not None), default=0)
        for method in methods:
            val = row_vals.get(method)
            if val is None:
                cells.append("")
            elif abs(val - best_val) < 1e-4:
                cells.append(f"**{val:.3f}**")
            else:
                cells.append(f"{val:.3f}")
        rows_out.append(f"| {label} | " + " | ".join(cells) + " |")

    return "\n".join(rows_out)


# Single-model summary table

def summary_table(results: dict[str, list[dict]] | dict[tuple, list[dict]]) -> str:
    """Build a formatted comparison table across strategies.

    Accepts either legacy {method: [rows]} or new {(model, method): [rows]}
    format; if new format, flattens to per-method (combining all models).
    """
    flat: dict[str, list[dict]] = {}
    for key, rows in results.items():
        if isinstance(key, tuple):
            _, method = key
        else:
            method = key
        flat.setdefault(method, []).extend(rows)

    header = (
        f"{'Strategy':<20s}  {'N':>5s}  "
        f"{'Token-F1':>9s}  {'ROUGE-L':>9s}  "
        f"{'Ctx Tokens':>10s}  {'Prompt Tok':>10s}"
    )
    sep = "-" * len(header)
    lines = [sep, header, sep]

    for method in sorted(flat.keys()):
        rows = flat[method]
        n = len(rows)
        lines.append(
            f"{method:<20s}  {n:>5d}  "
            f"{_mean(rows, 'token_f1'):>9.4f}  {_mean(rows, 'rouge_l_f1'):>9.4f}  "
            f"{_mean(rows, 'context_tokens'):>10.0f}  {_mean(rows, 'prompt_tokens'):>10.0f}"
        )

    lines.append(sep)
    return "\n".join(lines)


# Per-difficulty breakdown

def difficulty_breakdown(results: dict) -> str:
    """Break down scores by question difficulty (easy / medium / hard)."""
    flat: dict[str, list[dict]] = {}
    for key, rows in results.items():
        method = key[1] if isinstance(key, tuple) else key
        flat.setdefault(method, []).extend(rows)

    lines: list[str] = []
    for method in sorted(flat.keys()):
        rows = flat[method]
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


# Plots

def save_plots(
    data: dict[tuple[str, str], list[dict]],
    output_dir: Path,
) -> None:
    """Generate multi-model comparison plots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    matrix = model_method_matrix(data, "token_f1")

    models = [m for m in MODEL_ORDER if m in matrix]
    methods = [m for m in METHOD_ORDER if any(m in matrix.get(mod, {}) for mod in models)]
    n_models = len(models)
    n_methods = len(methods)

    if not n_models or not n_methods:
        log.warning("No data to plot")
        return

    # -- Grouped bar chart --
    x = np.arange(n_methods)
    total_width = 0.8
    bar_width = total_width / n_models

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, model in enumerate(models):
        vals = [matrix[model].get(m, 0) or 0 for m in methods]
        offset = (i - n_models / 2 + 0.5) * bar_width
        ax.bar(x + offset, vals, bar_width, label=MODEL_DISPLAY.get(model, model))

    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_DISPLAY.get(m, m) for m in methods], rotation=20, ha="right")
    ax.set_ylabel("Token F1")
    ax.set_title("Model × Retrieval Strategy (Token F1)")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_ylim(0, max(0.6, ax.get_ylim()[1] * 1.1))
    fig.tight_layout()
    fig.savefig(output_dir / "multi_model_comparison.png", dpi=150)
    plt.close(fig)
    log.info("Saved multi_model_comparison.png")

    # -- Heatmap --
    vals_2d = np.array(
        [[matrix[mod].get(m, 0) or 0 for m in methods] for mod in models]
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(vals_2d, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(n_methods))
    ax.set_xticklabels([METHOD_DISPLAY.get(m, m) for m in methods], rotation=30, ha="right")
    ax.set_yticks(range(n_models))
    ax.set_yticklabels([MODEL_DISPLAY.get(m, m) for m in models])
    for i in range(n_models):
        for j in range(n_methods):
            ax.text(j, i, f"{vals_2d[i, j]:.3f}", ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax, label="Token F1")
    ax.set_title("Token F1 Heatmap")
    fig.tight_layout()
    fig.savefig(output_dir / "token_f1_heatmap.png", dpi=150)
    plt.close(fig)
    log.info("Saved token_f1_heatmap.png")


# Structured export

def export_summary_json(
    data: dict[tuple[str, str], list[dict]],
    output_path: Path,
) -> None:
    """Write a machine-readable summary JSON grouped by model and method."""
    summary: dict[str, dict] = {}
    for (model, method), rows in sorted(data.items()):
        summary.setdefault(model, {})[method] = {
            "n": len(rows),
            "token_f1_mean": float(_mean(rows, "token_f1")),
            "rouge_l_f1_mean": float(_mean(rows, "rouge_l_f1")),
            "context_tokens_mean": float(_mean(rows, "context_tokens")),
            "prompt_tokens_mean": float(_mean(rows, "prompt_tokens")),
        }
    output_path.write_text(json.dumps(summary, indent=2))
    log.info("Summary JSON saved to %s", output_path)


# CLI

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    p = argparse.ArgumentParser(description="Analyze evaluation results")
    p.add_argument("--results-dir", type=Path, default=_RESULTS_DIR)
    p.add_argument("--plots", action="store_true", help="Generate comparison plots")
    args = p.parse_args()

    data = load_all_results(args.results_dir)
    if not data:
        log.error("No result files found in %s", args.results_dir)
        return

    print("\n=== Model × Method Matrix (Token F1) ===\n")
    print(format_matrix_table(data, "token_f1"))

    print("\n=== Model × Method Matrix (ROUGE-L) ===\n")
    print(format_matrix_table(data, "rouge_l_f1"))

    print("\n=== Aggregate Summary ===\n")
    print(summary_table(data))
    print()
    print(difficulty_breakdown(data))

    export_summary_json(data, args.results_dir / "summary.json")

    if args.plots:
        save_plots(data, _PLOTS_DIR)


if __name__ == "__main__":
    main()
