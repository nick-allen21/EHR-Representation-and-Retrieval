"""Generate publication-quality figures for the CS 229 final report.

Produces 11 figures per dataset (dev / verified) where data permits,
plus 2 model-level figures (feature importance, feature group breakdown).

Usage
-----
conda run -n ehr python scripts/generate_plots.py                  # both datasets
conda run -n ehr python scripts/generate_plots.py --dataset dev     # dev only
conda run -n ehr python scripts/generate_plots.py --dataset verified
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import seaborn as sns

log = logging.getLogger(__name__)

# Paths

ROOT = Path(__file__).resolve().parent.parent
DEV_RESULTS = ROOT / "data" / "results"
VERIFIED_RESULTS = ROOT / "data" / "results" / "verified"
MODEL_DIR = ROOT / "data" / "models" / "logreg"

# Display constants

MODEL_SLUGS = [
    "o4-mini",
    "gpt-4o-mini",
    "meta-llama--Llama-3.1-8B-Instruct",
    "mistralai--Mistral-7B-Instruct-v0.3",
    "microsoft--Phi-3-mini-4k-instruct",
]

MODEL_DISPLAY = {
    "o4-mini": "o4-mini",
    "gpt-4o-mini": "gpt-4o-mini",
    "meta-llama--Llama-3.1-8B-Instruct": "Llama-3.1-8B",
    "mistralai--Mistral-7B-Instruct-v0.3": "Mistral-7B",
    "microsoft--Phi-3-mini-4k-instruct": "Phi-3-mini-4k",
}

METHOD_ORDER = [
    "discharge_only",
    "full_context",
    "recency_n25",
    "bm25_k5",
    "semantic_rag_k5",
    "learned_k5",
]

METHOD_DISPLAY = {
    "discharge_only": "Discharge-Only",
    "full_context": "Full-Context",
    "recency_n25": "Recency",
    "bm25_k5": "BM25",
    "semantic_rag_k5": "Semantic RAG",
    "learned_k5": "Learned (Ours)",
}

# Style

STRATEGY_PALETTE = {
    "discharge_only": "#6C757D",
    "full_context": "#495057",
    "recency_n25": "#0077B6",
    "bm25_k5": "#ADB5BD",
    "semantic_rag_k5": "#E85D04",
    "learned_k5": "#2D6A4F",
}

MODEL_PALETTE = {
    "o4-mini": "#264653",
    "gpt-4o-mini": "#2A9D8F",
    "meta-llama--Llama-3.1-8B-Instruct": "#E9C46A",
    "mistralai--Mistral-7B-Instruct-v0.3": "#F4A261",
    "microsoft--Phi-3-mini-4k-instruct": "#E76F51",
}

METRIC_PALETTE = {"Token F1": "#264653", "ROUGE-L": "#2A9D8F", "Judge (norm.)": "#E76F51"}

SCORE_COLORS = {1: "#D62828", 2: "#F77F00", 3: "#FCBF49", 4: "#90BE6D", 5: "#43AA8B"}


def _apply_style():
    sns.set_theme(style="whitegrid", font_scale=1.1)
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.bbox": "tight",
        "savefig.dpi": 200,
        "font.family": "sans-serif",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.alpha": 0.3,
    })


# Data loading

def load_summary(results_dir: Path) -> dict:
    p = results_dir / "summary.json"
    if not p.exists():
        raise FileNotFoundError(p)
    return json.loads(p.read_text())


def load_judge_rankings(results_dir: Path) -> dict:
    p = results_dir / "judge_rankings.json"
    if not p.exists():
        raise FileNotFoundError(p)
    return json.loads(p.read_text())


def load_per_question(results_dir: Path) -> dict[tuple[str, str], list[dict]]:
    data = {}
    for p in sorted(results_dir.glob("*.json")):
        if p.stem in ("summary", "judge_rankings") or p.is_dir():
            continue
        if "__" not in p.stem:
            continue
        model_slug, method = p.stem.split("__", 1)
        rows = json.loads(p.read_text())
        if isinstance(rows, list) and rows:
            data[(model_slug, method)] = rows
    return data


def _get_metric(summary: dict, model: str, method: str, metric: str) -> float | None:
    return summary.get(model, {}).get(method, {}).get(metric)


def _judge_means(per_q: dict[tuple[str, str], list[dict]]) -> dict[tuple[str, str], float]:
    means = {}
    for (model, method), rows in per_q.items():
        scores = [r["judge_score"] for r in rows if r.get("judge_score") is not None]
        if scores:
            means[(model, method)] = float(np.mean(scores))
    return means


# Plot 1: Efficiency vs Accuracy Scatter

def plot_efficiency_scatter(summary: dict, per_q: dict, out: Path, dataset_label: str):
    judge_means = _judge_means(per_q)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    for ax, (metric_key, metric_label, use_judge) in zip(axes, [
        ("token_f1_mean", "Token F1", False),
        (None, "Judge Score (1–5)", True),
    ]):
        for method in METHOD_ORDER:
            tokens_list, scores_list = [], []
            for model in MODEL_SLUGS:
                tok = _get_metric(summary, model, method, "context_tokens_mean")
                if use_judge:
                    sc = judge_means.get((model, method))
                else:
                    sc = _get_metric(summary, model, method, metric_key)
                if tok is not None and sc is not None:
                    tokens_list.append(tok)
                    scores_list.append(sc)
            if not tokens_list:
                continue
            avg_tok = np.mean(tokens_list)
            avg_sc = np.mean(scores_list)
            color = STRATEGY_PALETTE[method]
            ax.scatter(avg_tok, avg_sc, s=160, color=color, edgecolors="white",
                       linewidth=1.2, zorder=5)
            ax.annotate(METHOD_DISPLAY[method], (avg_tok, avg_sc),
                        textcoords="offset points", xytext=(8, 6),
                        fontsize=9, color=color, fontweight="bold")

        ax.set_xlabel("Average Context Tokens", fontsize=11)
        ax.set_ylabel(metric_label, fontsize=11)
        ax.set_title(f"Efficiency vs. {metric_label}", fontsize=12, fontweight="bold")
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

    fig.suptitle(f"Context Efficiency — {dataset_label}", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(out / "01_efficiency_scatter.png")
    plt.close(fig)
    log.info("Saved 01_efficiency_scatter.png")


# Plot 2: Grouped Bar - All Metrics by Strategy

def plot_metrics_by_strategy(summary: dict, per_q: dict, out: Path, dataset_label: str):
    judge_means = _judge_means(per_q)

    methods = METHOD_ORDER
    metric_names = list(METRIC_PALETTE.keys())
    n_metrics = len(metric_names)
    x = np.arange(len(methods))
    bar_w = 0.22

    fig, ax = plt.subplots(figsize=(12, 5.5))
    for i, (metric_name, color) in enumerate(METRIC_PALETTE.items()):
        vals = []
        for method in methods:
            agg = []
            for model in MODEL_SLUGS:
                if metric_name == "Token F1":
                    v = _get_metric(summary, model, method, "token_f1_mean")
                elif metric_name == "ROUGE-L":
                    v = _get_metric(summary, model, method, "rouge_l_f1_mean")
                else:
                    v = judge_means.get((model, method))
                    if v is not None:
                        v = v / 5.0
                if v is not None:
                    agg.append(v)
            vals.append(np.mean(agg) if agg else 0)
        offset = (i - n_metrics / 2 + 0.5) * bar_w
        bars = ax.bar(x + offset, vals, bar_w, label=metric_name, color=color, edgecolor="white", linewidth=0.6)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=7.5, color=color)

    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_DISPLAY[m] for m in methods], rotation=20, ha="right")
    ax.set_ylabel("Score (0–1 scale)", fontsize=11)
    ax.set_title(f"All Metrics by Retrieval Strategy — {dataset_label}", fontsize=12, fontweight="bold")
    ax.legend(frameon=True, framealpha=0.9, fontsize=9)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.15)
    fig.tight_layout()
    fig.savefig(out / "02_metrics_by_strategy.png")
    plt.close(fig)
    log.info("Saved 02_metrics_by_strategy.png")


# Plot 3: Recall@K Curve

def plot_recall_at_k(out: Path):
    ks = [1, 3, 5, 10]
    baseline = [0.45, 0.71, 0.84, 0.96]
    enriched = [0.41, 0.77, 0.88, 0.96]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(ks, baseline, "o-", color="#0077B6", linewidth=2.2, markersize=8, label="Baseline (130 features)")
    ax.plot(ks, enriched, "s-", color="#2D6A4F", linewidth=2.2, markersize=8, label="Enriched (166 features)")

    for k, b, e in zip(ks, baseline, enriched):
        ax.annotate(f"{b:.2f}", (k, b), textcoords="offset points", xytext=(-18, -16), fontsize=8.5, color="#0077B6")
        ax.annotate(f"{e:.2f}", (k, e), textcoords="offset points", xytext=(6, 8), fontsize=8.5, color="#2D6A4F")

    ax.set_xlabel("K (chunks retrieved)", fontsize=11)
    ax.set_ylabel("Mean Recall@K", fontsize=11)
    ax.set_title("Retrieval Quality: Recall@K", fontsize=12, fontweight="bold")
    ax.set_xticks(ks)
    ax.set_ylim(0.3, 1.02)
    ax.legend(frameon=True, framealpha=0.9, fontsize=10)
    ax.fill_between(ks, baseline, enriched, alpha=0.08, color="#2D6A4F")
    fig.tight_layout()
    fig.savefig(out / "03_recall_at_k.png")
    plt.close(fig)
    log.info("Saved 03_recall_at_k.png")


# Plot 4: Feature Importance

def _load_feature_data() -> tuple[list[str], np.ndarray]:
    SECTION_CATEGORIES = [
        "medications", "diagnosis", "hospital_course", "results", "hpi",
        "allergies", "chief_complaint", "pmh", "exam", "labs", "followup",
        "discharge_instructions", "discharge_condition", "discharge_disposition",
        "social", "family", "vitals", "procedure", "other",
    ]
    QUESTION_TYPES = ["medications", "diagnosis", "labs", "imaging", "procedure", "other"]

    names = ["embed_sim", "tfidf_sim", "lexical_overlap", "position", "log_length"]
    names += ["has_temporal_marker", "temporal_density", "has_critical_event", "has_abnormal_lab"]
    names += ["days_to_discharge", "is_within_24h", "is_within_1week", "has_no_timestamp"]
    names += [f"sec_{s}" for s in SECTION_CATEGORIES]
    names += [f"qt_{t}" for t in QUESTION_TYPES]
    names += [f"int_{s}_{t}" for s in SECTION_CATEGORIES for t in QUESTION_TYPES]
    names += [
        "int_abnormal_lab_x_labs", "int_abnormal_lab_x_diagnosis",
        "int_critical_event_x_diagnosis", "int_critical_event_x_procedure",
        "int_within_24h_x_medications", "int_within_24h_x_labs",
        "int_temporal_marker_x_labs", "int_temporal_marker_x_diagnosis",
    ]
    names += [
        "content_word_overlap", "numeric_density",
        "has_discharge_meds_hdr", "has_admission_meds_hdr",
        "has_discharge_labs_hdr", "has_admission_labs_hdr",
    ]

    with open(MODEL_DIR / "model.pkl", "rb") as f:
        model = pickle.load(f)
    weights = model.coef_.ravel()

    if len(names) != len(weights):
        log.warning("Feature name count (%d) != weight count (%d); truncating", len(names), len(weights))
        names = names[:len(weights)]

    return names, weights


def plot_feature_importance(out: Path):
    names, weights = _load_feature_data()

    order = np.argsort(np.abs(weights))[::-1]
    top_n = 20
    top_idx = order[:top_n]

    top_names = [names[i] for i in top_idx]
    top_weights = weights[top_idx]
    colors = ["#2D6A4F" if w > 0 else "#D62828" for w in top_weights]

    fig, ax = plt.subplots(figsize=(10, 7))
    y = np.arange(top_n)
    ax.barh(y, top_weights, color=colors, edgecolor="white", linewidth=0.5, height=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels(top_names, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Learned Weight", fontsize=11)
    ax.set_title("Top 20 Features by Absolute Weight (L1 Logistic Regression)", fontsize=12, fontweight="bold")
    ax.axvline(0, color="black", linewidth=0.8)

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor="#2D6A4F", label="Positive (predicts relevant)"),
                       Patch(facecolor="#D62828", label="Negative (predicts irrelevant)")]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9, frameon=True)

    fig.tight_layout()
    fig.savefig(out / "04_feature_importance.png")
    plt.close(fig)
    log.info("Saved 04_feature_importance.png")


# Plot 5: Judge Score Distribution - Stacked Bar

def plot_judge_distribution(judge_data: dict, out: Path, dataset_label: str):
    ranked = judge_data.get("ranked", [])
    key_strategies = ["full_context", "semantic_rag_k5", "learned_k5", "discharge_only", "recency_n25"]
    target_model = "o4-mini"

    entries = []
    for entry in ranked:
        strat = entry["strategy"]
        if "__" in strat:
            model, method = strat.split("__", 1)
        else:
            continue
        if model == target_model and method in key_strategies:
            entries.append((method, entry["score_dist"], entry["n"]))

    if not entries:
        log.warning("No judge distribution data for %s", dataset_label)
        return

    order = [m for m in key_strategies if any(e[0] == m for e in entries)]
    entries_dict = {m: (sd, n) for m, sd, n in entries}

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(order))
    bottom = np.zeros(len(order))

    for score in [1, 2, 3, 4, 5]:
        proportions = []
        for method in order:
            sd, n = entries_dict[method]
            count = sd.get(str(score), 0)
            proportions.append(count / n if n > 0 else 0)
        bars = ax.bar(x, proportions, bottom=bottom, color=SCORE_COLORS[score],
                       label=f"Score {score}", edgecolor="white", linewidth=0.5, width=0.6)
        for xi, prop in enumerate(proportions):
            if prop > 0.05:
                ax.text(xi, bottom[xi] + prop / 2, f"{prop:.0%}",
                        ha="center", va="center", fontsize=7.5, fontweight="bold", color="white")
        bottom += proportions

    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_DISPLAY.get(m, m) for m in order], fontsize=10)
    ax.set_ylabel("Proportion of Questions", fontsize=11)
    ax.set_title(f"Judge Score Distribution ({MODEL_DISPLAY[target_model]}) — {dataset_label}",
                 fontsize=12, fontweight="bold")
    ax.legend(title="Judge Score", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    fig.tight_layout()
    fig.savefig(out / "05_judge_distribution.png")
    plt.close(fig)
    log.info("Saved 05_judge_distribution.png")


# Plot 6: Per-Difficulty Breakdown

def plot_per_difficulty(judge_data: dict, out: Path, dataset_label: str):
    by_diff = judge_data.get("by_difficulty", {})
    if not by_diff:
        log.warning("No per-difficulty data for %s — skipping", dataset_label)
        return

    difficulties = ["easy", "medium", "hard"]
    strategies = ["full_context", "semantic_rag_k5", "learned_k5"]
    target_model = "o4-mini"

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(difficulties))
    bar_w = 0.22

    for i, method in enumerate(strategies):
        key = f"{target_model}__{method}"
        vals = [by_diff.get(d, {}).get(key, 0) for d in difficulties]
        offset = (i - len(strategies) / 2 + 0.5) * bar_w
        color = STRATEGY_PALETTE[method]
        bars = ax.bar(x + offset, vals, bar_w, label=METHOD_DISPLAY[method],
                       color=color, edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.03,
                        f"{val:.2f}", ha="center", va="bottom", fontsize=8, color=color)

    ax.set_xticks(x)
    ax.set_xticklabels([d.capitalize() for d in difficulties], fontsize=11)
    ax.set_ylabel("Mean Judge Score (1–5)", fontsize=11)
    ax.set_title(f"Performance by Question Difficulty ({MODEL_DISPLAY[target_model]}) — {dataset_label}",
                 fontsize=12, fontweight="bold")
    ax.legend(frameon=True, framealpha=0.9, fontsize=9)
    ax.set_ylim(0, 5.2)
    fig.tight_layout()
    fig.savefig(out / "06_per_difficulty.png")
    plt.close(fig)
    log.info("Saved 06_per_difficulty.png")


# Plot 8: Model x Retrieval Heatmap

def plot_heatmap(summary: dict, per_q: dict, out: Path, dataset_label: str, metric: str = "token_f1"):
    if metric == "judge_score":
        judge_means = _judge_means(per_q)
        matrix = []
        for model in MODEL_SLUGS:
            row = [judge_means.get((model, m), 0) for m in METHOD_ORDER]
            matrix.append(row)
        cbar_label = "Judge Score (1–5)"
        cmap = "YlGn"
        fmt = ".2f"
        title_metric = "Judge Score"
    else:
        metric_key = "token_f1_mean" if metric == "token_f1" else "rouge_l_f1_mean"
        matrix = []
        for model in MODEL_SLUGS:
            row = [_get_metric(summary, model, m, metric_key) or 0 for m in METHOD_ORDER]
            matrix.append(row)
        cbar_label = "Token F1" if metric == "token_f1" else "ROUGE-L"
        cmap = "YlOrRd"
        fmt = ".3f"
        title_metric = cbar_label

    arr = np.array(matrix)
    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(arr, cmap=cmap, aspect="auto", vmin=arr[arr > 0].min() * 0.9 if (arr > 0).any() else 0,
                   vmax=arr.max() * 1.02)
    ax.set_xticks(range(len(METHOD_ORDER)))
    ax.set_xticklabels([METHOD_DISPLAY[m] for m in METHOD_ORDER], rotation=30, ha="right", fontsize=10)
    ax.set_yticks(range(len(MODEL_SLUGS)))
    ax.set_yticklabels([MODEL_DISPLAY[m] for m in MODEL_SLUGS], fontsize=10)

    for i in range(len(MODEL_SLUGS)):
        row_best = arr[i].max()
        for j in range(len(METHOD_ORDER)):
            val = arr[i, j]
            bold = val >= row_best - 1e-4
            txt = f"{val:{fmt}}"
            color = "white" if val > (arr.max() + arr.min()) / 2 else "black"
            ax.text(j, i, txt, ha="center", va="center", fontsize=9,
                    fontweight="bold" if bold else "normal", color=color)

    fig.colorbar(im, ax=ax, label=cbar_label, shrink=0.8)
    ax.set_title(f"Model × Retrieval Strategy — {title_metric} — {dataset_label}",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()

    suffix = metric.replace("_", "-")
    fig.savefig(out / f"08_heatmap_{suffix}.png")
    plt.close(fig)
    log.info("Saved 08_heatmap_%s.png", suffix)


# Plot 9: Feature Weight Category Breakdown

def plot_feature_group_breakdown(out: Path):
    names, weights = _load_feature_data()

    groups = {
        "Similarity": [],
        "Structural / Positional": [],
        "Section Identity": [],
        "Question Type": [],
        "Section × Q-Type": [],
        "Text Signals (A)": [],
        "Temporal (B)": [],
        "Signal × Q-Type (C)": [],
        "Precision (D)": [],
    }

    for i, name in enumerate(names):
        if name in ("embed_sim", "tfidf_sim", "lexical_overlap", "content_word_overlap"):
            groups["Similarity"].append(i)
        elif name in ("position", "log_length", "numeric_density"):
            groups["Structural / Positional"].append(i)
        elif name.startswith("sec_"):
            groups["Section Identity"].append(i)
        elif name.startswith("qt_"):
            groups["Question Type"].append(i)
        elif name.startswith("int_") and "_x_" not in name:
            groups["Section × Q-Type"].append(i)
        elif name in ("has_temporal_marker", "temporal_density", "has_critical_event", "has_abnormal_lab"):
            groups["Text Signals (A)"].append(i)
        elif name in ("days_to_discharge", "is_within_24h", "is_within_1week", "has_no_timestamp"):
            groups["Temporal (B)"].append(i)
        elif "_x_" in name:
            groups["Signal × Q-Type (C)"].append(i)
        elif name.startswith("has_") and ("meds_hdr" in name or "labs_hdr" in name):
            groups["Precision (D)"].append(i)

    group_names = []
    group_masses = []
    for gname, indices in groups.items():
        mass = float(np.sum(np.abs(weights[indices]))) if indices else 0.0
        if mass > 0:
            group_names.append(gname)
            group_masses.append(mass)

    total = sum(group_masses)
    group_pcts = [m / total * 100 for m in group_masses]

    palette = sns.color_palette("Set2", len(group_names))

    fig, ax = plt.subplots(figsize=(8, 8))
    wedges, texts, autotexts = ax.pie(
        group_pcts, labels=group_names, autopct=lambda p: f"{p:.1f}%" if p > 3 else "",
        colors=palette, startangle=140, pctdistance=0.78,
        wedgeprops=dict(width=0.45, edgecolor="white", linewidth=2),
    )
    for t in texts:
        t.set_fontsize(9)
    for t in autotexts:
        t.set_fontsize(8)
        t.set_fontweight("bold")

    ax.set_title("Feature Weight Mass by Group\n(L1 Logistic Regression)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / "09_feature_group_breakdown.png")
    plt.close(fig)
    log.info("Saved 09_feature_group_breakdown.png")


# Plot 10: Small Model Lift Chart

def plot_model_lift(summary: dict, per_q: dict, out: Path, dataset_label: str):
    judge_means = _judge_means(per_q)

    models_ordered = MODEL_SLUGS
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, (metric_label, get_val) in zip(axes, [
        ("Token F1", lambda m, meth: _get_metric(summary, m, meth, "token_f1_mean")),
        ("Judge Score", lambda m, meth: judge_means.get((m, meth))),
    ]):
        baselines = ["discharge_only", "full_context"]
        x = np.arange(len(models_ordered))
        bar_w = 0.35

        for bi, baseline in enumerate(baselines):
            deltas = []
            for model in models_ordered:
                learned = get_val(model, "learned_k5")
                base = get_val(model, baseline)
                if learned is not None and base is not None:
                    deltas.append(learned - base)
                else:
                    deltas.append(0)

            offset = (bi - 0.5) * bar_w
            colors = ["#2D6A4F" if d >= 0 else "#D62828" for d in deltas]
            ax.bar(x + offset, deltas, bar_w, color=colors, edgecolor="white", linewidth=0.5,
                   label=f"vs {METHOD_DISPLAY[baseline]}")
            for xi, d in enumerate(deltas):
                ax.text(xi + offset, d + (0.003 if d >= 0 else -0.012),
                        f"{d:+.3f}", ha="center", va="bottom" if d >= 0 else "top",
                        fontsize=7.5, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_DISPLAY[m] for m in models_ordered], rotation=20, ha="right", fontsize=9)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_ylabel(f"Δ {metric_label}", fontsize=11)
        ax.set_title(f"Learned Selector Lift ({metric_label})", fontsize=12, fontweight="bold")
        ax.legend(fontsize=9, frameon=True)

    fig.suptitle(f"How Much Does the Learned Selector Help? — {dataset_label}",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(out / "10_model_lift.png")
    plt.close(fig)
    log.info("Saved 10_model_lift.png")


# Plot 11: Phi-3 Context Cliff

def plot_phi3_cliff(summary: dict, per_q: dict, out: Path, dataset_label: str):
    model = "microsoft--Phi-3-mini-4k-instruct"
    judge_means = _judge_means(per_q)

    methods = METHOD_ORDER
    f1_vals = [_get_metric(summary, model, m, "token_f1_mean") or 0 for m in methods]
    judge_vals = [judge_means.get((model, m), 0) for m in methods]
    tok_vals = [_get_metric(summary, model, m, "context_tokens_mean") or 0 for m in methods]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    for ax, vals, ylabel, title_metric in zip(
        axes, [f1_vals, judge_vals],
        ["Token F1", "Judge Score (1–5)"],
        ["Token F1", "Judge Score"],
    ):
        colors = [STRATEGY_PALETTE[m] for m in methods]
        x = np.arange(len(methods))
        bars = ax.bar(x, vals, color=colors, edgecolor="white", linewidth=0.8, width=0.6)

        for xi, (bar, val, tok) in enumerate(zip(bars, vals, tok_vals)):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.3f}\n({tok:.0f} tok)", ha="center", va="bottom", fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels([METHOD_DISPLAY[m] for m in methods], rotation=25, ha="right", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(f"Phi-3-mini-4k — {title_metric}", fontsize=12, fontweight="bold")

        ax.axhline(y=0, color="black", linewidth=0.5)
        ymax = ax.get_ylim()[1]
        ax.annotate("4k context limit", xy=(0.5, ymax * 0.92),
                    fontsize=9, ha="center", color="#D62828", fontstyle="italic",
                    xycoords=("axes fraction", "data"),
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="#FDECEA", edgecolor="#D62828", alpha=0.8))

    fig.suptitle(f"Phi-3-mini-4k: Context Selection Rescues a Tiny Model — {dataset_label}",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(out / "11_phi3_cliff.png")
    plt.close(fig)
    log.info("Saved 11_phi3_cliff.png")


# Plot 12: Pairwise Win Rate - Learned vs Baselines

def plot_pairwise_winrate(per_q: dict, out: Path, dataset_label: str):
    baselines = ["discharge_only", "full_context", "recency_n25", "bm25_k5", "semantic_rag_k5"]
    models = MODEL_SLUGS

    win_matrix = np.zeros((len(models), len(baselines)))

    for mi, model in enumerate(models):
        learned_rows = per_q.get((model, "learned_k5"), [])
        if not learned_rows:
            continue
        learned_by_q = {(r["subject_id"], r["hadm_id"], r["question"]): r.get("judge_score")
                        for r in learned_rows}

        for bi, baseline in enumerate(baselines):
            base_rows = per_q.get((model, baseline), [])
            wins, total = 0, 0
            for r in base_rows:
                key = (r["subject_id"], r["hadm_id"], r["question"])
                l_score = learned_by_q.get(key)
                b_score = r.get("judge_score")
                if l_score is not None and b_score is not None:
                    total += 1
                    if l_score > b_score:
                        wins += 1
            win_matrix[mi, bi] = wins / total if total > 0 else 0

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(win_matrix, cmap="RdYlGn", aspect="auto", vmin=0.0, vmax=0.6)

    ax.set_xticks(range(len(baselines)))
    ax.set_xticklabels([METHOD_DISPLAY[b] for b in baselines], rotation=30, ha="right", fontsize=10)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([MODEL_DISPLAY[m] for m in models], fontsize=10)

    for i in range(len(models)):
        for j in range(len(baselines)):
            val = win_matrix[i, j]
            color = "white" if val > 0.35 else "black"
            ax.text(j, i, f"{val:.0%}", ha="center", va="center", fontsize=10,
                    fontweight="bold", color=color)

    fig.colorbar(im, ax=ax, label="Win Rate (Learned > Baseline)", shrink=0.8,
                 format=mticker.PercentFormatter(1.0))
    ax.set_title(f"Learned Selector Win Rate vs. Each Baseline (Judge Score) — {dataset_label}",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / "12_pairwise_winrate.png")
    plt.close(fig)
    log.info("Saved 12_pairwise_winrate.png")


# Orchestration

def generate_all(results_dir: Path, out_dir: Path, dataset_label: str, is_dev: bool):
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = load_summary(results_dir)
    judge_data = load_judge_rankings(results_dir)
    per_q = load_per_question(results_dir)

    plot_efficiency_scatter(summary, per_q, out_dir, dataset_label)
    plot_metrics_by_strategy(summary, per_q, out_dir, dataset_label)
    plot_heatmap(summary, per_q, out_dir, dataset_label, metric="token_f1")
    plot_heatmap(summary, per_q, out_dir, dataset_label, metric="judge_score")
    plot_judge_distribution(judge_data, out_dir, dataset_label)
    plot_model_lift(summary, per_q, out_dir, dataset_label)
    plot_phi3_cliff(summary, per_q, out_dir, dataset_label)
    plot_pairwise_winrate(per_q, out_dir, dataset_label)

    if is_dev:
        plot_per_difficulty(judge_data, out_dir, dataset_label)
        plot_recall_at_k(out_dir)
        plot_feature_importance(out_dir)
        plot_feature_group_breakdown(out_dir)


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    _apply_style()

    p = argparse.ArgumentParser(description="Generate publication-quality plots")
    p.add_argument("--dataset", choices=["dev", "verified", "both"], default="both",
                   help="Which dataset to plot (default: both)")
    args = p.parse_args()

    plots_root = DEV_RESULTS / "plots"

    if args.dataset in ("dev", "both"):
        log.info("=== Generating dev set plots ===")
        generate_all(DEV_RESULTS, plots_root / "dev",
                     "200-Patient Dev Set (GPT-4o QA)", is_dev=True)

    if args.dataset in ("verified", "both"):
        log.info("=== Generating verified set plots ===")
        generate_all(VERIFIED_RESULTS, plots_root / "verified",
                     "70-Patient Verified Set (Physician QA)", is_dev=False)

    log.info("Done — all plots saved to %s", plots_root)


if __name__ == "__main__":
    main()
