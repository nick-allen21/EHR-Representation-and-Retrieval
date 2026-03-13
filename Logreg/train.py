"""Train an L1-regularized logistic regression chunk selector.

Pipeline:
  1. Chunk each discharge summary
  2. Pair every chunk with each QA question
  3. Label positive if token F1(chunk, answer) ≥ threshold
  4. Sample negatives to control class imbalance
  5. Fit TF-IDF + (optionally) sentence embeddings
  6. Train logistic regression with L1 penalty
  7. Evaluate on a held-out patient split
  8. Save model + feature extractor artifacts

Usage:
  from Logreg.data_loader import load_and_merge
  from Logreg.train import run_training

  records = load_and_merge()
  metrics = run_training(records, output_dir="data/models/logreg")
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from Logreg.chunker import chunk_note
from Logreg.features import FeatureExtractor
from Logreg.labeler import label_chunks


# ── Dataset construction ──────────────────────────────────────────────────────

def build_dataset(
    merged_records: list[dict],
    strategy: str = "section",
    label_threshold: float = 0.2,
    neg_per_pos: int = 5,
    random_seed: int = 42,
) -> tuple[list[str], list[dict], list[int], list[int]]:
    """Convert merged records into (question, chunk) training examples.

    Args:
        merged_records:  Output of data_loader.merge() or load_and_merge().
        strategy:        Chunking strategy ('section' or 'fixed').
        label_threshold: Min token F1 between chunk and answer to label positive.
        neg_per_pos:     How many negatives to sample per positive (class balance).
        random_seed:     RNG seed for reproducibility.

    Returns:
        questions   — list[str]  (length N)
        chunks      — list[dict] (length N)
        labels      — list[int]  0/1   (length N)
        record_ids  — list[int]  index into merged_records for each example
                      (used downstream to implement patient-level splits)
    """
    rng = np.random.default_rng(random_seed)
    questions:   list[str]  = []
    chunks:      list[dict] = []
    labels:      list[int]  = []
    record_ids:  list[int]  = []

    for rec_idx, record in enumerate(merged_records):
        note_chunks = chunk_note(record["note_text"], strategy=strategy)
        if not note_chunks:
            continue
        dischtime = (record.get("admission") or {}).get("dischtime")
        for chunk in note_chunks:
            chunk["_dischtime"] = dischtime

        for qa in record["qa_pairs"]:
            question = qa.get("question", "").strip()
            answer   = qa.get("answer",   "").strip()
            if not question or not answer:
                continue

            chunk_labels = label_chunks(note_chunks, answer, threshold=label_threshold)
            pos_idx = [i for i, l in enumerate(chunk_labels) if l == 1]
            neg_idx = [i for i, l in enumerate(chunk_labels) if l == 0]

            # Always keep all positives
            for i in pos_idx:
                questions.append(question)
                chunks.append(note_chunks[i])
                labels.append(1)
                record_ids.append(rec_idx)

            # Sample negatives
            n_neg = min(len(neg_idx), len(pos_idx) * neg_per_pos)
            if n_neg > 0:
                sampled = rng.choice(neg_idx, size=n_neg, replace=False).tolist()
                for i in sampled:
                    questions.append(question)
                    chunks.append(note_chunks[i])
                    labels.append(0)
                    record_ids.append(rec_idx)

    pos = sum(labels)
    neg = len(labels) - pos
    print(f"Dataset: {len(labels):,} examples  ({pos:,} positive, {neg:,} negative)")
    return questions, chunks, labels, record_ids


def patient_split(
    record_ids: list[int],
    n_records: int,
    val_fraction: float = 0.15,
    random_seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Split example indices into train/val so that no patient spans both sets.

    Returns (train_idx, val_idx) arrays of example-level indices.
    """
    rng = np.random.default_rng(random_seed)
    all_rec_ids = np.arange(n_records)
    rng.shuffle(all_rec_ids)
    split = int(n_records * (1 - val_fraction))
    train_recs = set(all_rec_ids[:split].tolist())

    ids = np.array(record_ids)
    train_idx = np.where(np.isin(ids, list(train_recs)))[0]
    val_idx   = np.where(~np.isin(ids, list(train_recs)))[0]
    return train_idx, val_idx


# ── Model training ────────────────────────────────────────────────────────────

def train_model(
    X: np.ndarray,
    y: np.ndarray,
    C: float = 1.0,
    random_seed: int = 42,
) -> LogisticRegression:
    """Fit an L1-regularized logistic regression model.

    L1 penalty drives irrelevant feature weights to exactly zero, giving an
    interpretable sparse model.  class_weight='balanced' compensates for the
    negative-majority class imbalance even after negative sampling.
    """
    model = LogisticRegression(
        solver="liblinear",
        C=C,
        l1_ratio=1.0,
        max_iter=1000,
        random_state=random_seed,
        class_weight="balanced",
    )
    model.fit(X, y)
    return model


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_model(
    model: LogisticRegression,
    X: np.ndarray,
    y: np.ndarray,
) -> dict:
    """Binary classification metrics on the flat (question, chunk, label) pairs.

    These are diagnostic training metrics — they tell you whether the model can
    distinguish positive chunks from negative ones.  The actual retrieval metric
    (mean Recall@K per question) is computed separately by per_question_recall_at_k().
    """
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]
    y_int  = y.astype(int)

    accuracy = float((y_pred == y_int).mean())
    auc      = float(roc_auc_score(y_int, y_prob)) if len(np.unique(y_int)) > 1 else 0.0
    ap       = float(average_precision_score(y_int, y_prob)) if len(np.unique(y_int)) > 1 else 0.0

    return {
        "accuracy":      accuracy,
        "roc_auc":       auc,
        "avg_precision": ap,
        "n_positive":    int(y_int.sum()),
        "n_total":       len(y_int),
    }


def per_question_recall_at_k(
    model: LogisticRegression,
    X: np.ndarray,
    y: np.ndarray,
    questions: list[str],
    record_ids: list[int],
    k_values: tuple[int, ...] = (1, 3, 5, 10),
) -> dict[str, float]:
    """Compute mean Recall@K across individual questions on the val set.

    This is the primary retrieval metric.  For each unique (patient, question)
    group in the val set:
      1. Score all of that question's chunks with the model.
      2. Take the top-K by predicted probability.
      3. Recall@K = (# positive chunks in top-K) / (# total positive chunks).
    Then average across all question groups.

    This is the correct retrieval metric — it directly measures whether the
    model retrieves the answer-containing chunks within the context budget.
    """
    y_prob = model.predict_proba(X)[:, 1]
    y_int  = np.array(y, dtype=int)

    # Group example indices by (record_id, question_text)
    groups: dict[tuple[int, str], list[int]] = {}
    for i, (rid, q) in enumerate(zip(record_ids, questions)):
        key = (rid, q)
        groups.setdefault(key, []).append(i)

    recall_by_k: dict[int, list[float]] = {k: [] for k in k_values}

    for indices in groups.values():
        probs   = y_prob[indices]
        labels  = y_int[indices]
        n_pos   = labels.sum()
        if n_pos == 0:
            continue  # skip unanswerable questions

        for k in k_values:
            top_k  = np.argsort(probs)[-k:]
            hits   = labels[top_k].sum()
            recall_by_k[k].append(hits / n_pos)

    return {
        f"mean_recall@{k}": float(np.mean(vals)) if vals else 0.0
        for k, vals in recall_by_k.items()
    }


# ── Plotting ─────────────────────────────────────────────────────────────────

def _save_plots(
    model: LogisticRegression,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feat_names: list[str],
    recall_metrics: dict[str, float],
    output_dir: Path,
) -> None:
    """Generate and save diagnostic plots to *output_dir*/plots/."""
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    y_prob = model.predict_proba(X_val)[:, 1]
    y_int  = y_val.astype(int)

    # ── 1. Feature importance (top 20 non-zero) ─────────────────────────────
    coef = model.coef_[0]
    nonzero_mask = coef != 0
    nz_idx   = np.where(nonzero_mask)[0]
    nz_coef  = coef[nz_idx]
    nz_names = [feat_names[i] for i in nz_idx]

    top_n = min(20, len(nz_idx))
    top   = np.argsort(np.abs(nz_coef))[-top_n:]
    top   = top[np.argsort(nz_coef[top])]

    fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.35)))
    colors = ["#e74c3c" if nz_coef[i] < 0 else "#2ecc71" for i in top]
    ax.barh([nz_names[i] for i in top], nz_coef[top], color=colors)
    ax.set_xlabel("L1 Logistic Regression Coefficient")
    ax.set_title(f"Top {top_n} Feature Weights ({int(nonzero_mask.sum())}/{len(coef)} non-zero)")
    ax.axvline(0, color="black", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(plots_dir / "feature_importance.png", dpi=150)
    plt.close(fig)

    # ── 2. ROC curve ────────────────────────────────────────────────────────
    if len(np.unique(y_int)) > 1:
        fpr, tpr, _ = roc_curve(y_int, y_prob)
        auc_val = roc_auc_score(y_int, y_prob)

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(fpr, tpr, linewidth=2, label=f"AUC = {auc_val:.3f}")
        ax.plot([0, 1], [0, 1], "--", color="grey", linewidth=0.8)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve (Validation)")
        ax.legend(loc="lower right")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        fig.tight_layout()
        fig.savefig(plots_dir / "roc_curve.png", dpi=150)
        plt.close(fig)

    # ── 3. Precision-Recall curve ───────────────────────────────────────────
    if len(np.unique(y_int)) > 1:
        prec, rec, _ = precision_recall_curve(y_int, y_prob)
        ap = average_precision_score(y_int, y_prob)

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(rec, prec, linewidth=2, label=f"AP = {ap:.3f}")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall Curve (Validation)")
        ax.legend(loc="upper right")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        fig.tight_layout()
        fig.savefig(plots_dir / "precision_recall_curve.png", dpi=150)
        plt.close(fig)

    # ── 4. Recall@K curve ───────────────────────────────────────────────────
    k_values = sorted(int(k.split("@")[1]) for k in recall_metrics)
    recalls  = [recall_metrics[f"mean_recall@{k}"] for k in k_values]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(k_values, recalls, "o-", linewidth=2, markersize=8, color="#3498db")
    for k, r in zip(k_values, recalls):
        ax.annotate(f"{r:.2f}", (k, r), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=10)
    ax.set_xlabel("K (number of chunks selected)")
    ax.set_ylabel("Mean Recall@K")
    ax.set_title("Retrieval Performance: Recall@K (Validation)")
    ax.set_xticks(k_values)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / "recall_at_k.png", dpi=150)
    plt.close(fig)

    # ── 5. Score distribution ───────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(y_prob[y_int == 0], bins=40, alpha=0.6, label="Negative", color="#e74c3c", density=True)
    ax.hist(y_prob[y_int == 1], bins=40, alpha=0.6, label="Positive", color="#2ecc71", density=True)
    ax.set_xlabel("Predicted Probability")
    ax.set_ylabel("Density")
    ax.set_title("Score Distribution (Validation)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "score_distribution.png", dpi=150)
    plt.close(fig)

    print(f"Plots saved to {plots_dir}/")


# ── Full training pipeline ────────────────────────────────────────────────────

def run_training(
    merged_records: list[dict],
    output_dir: str | Path = "data/models/logreg",
    strategy: str = "section",
    C: float = 1.0,
    label_threshold: float = 0.2,
    use_embeddings: bool = True,
    val_fraction: float = 0.15,
    random_seed: int = 42,
) -> dict:
    """End-to-end training pipeline.

    1. Build the (question, chunk, label) dataset from merged_records.
    2. Split by patient (no patient appears in both train and val).
    3. Fit the feature extractor on training texts only.
    4. Compute embeddings in bulk (one pass per split, not per example).
    5. Train logistic regression with L1 penalty.
    6. Evaluate on val split and report Recall@K.
    7. Save model.pkl, feature_extractor.pkl, metrics.json to output_dir.

    Returns:
        metrics dict with 'train', 'val', 'top_features', 'config' keys.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Build dataset
    print("Building dataset …")
    questions, chunks, labels, record_ids = build_dataset(
        merged_records,
        strategy=strategy,
        label_threshold=label_threshold,
        random_seed=random_seed,
    )
    y = np.array(labels)

    # 2. Patient-level train/val split
    train_idx, val_idx = patient_split(
        record_ids, len(merged_records), val_fraction=val_fraction, random_seed=random_seed
    )
    print(f"Split: {len(train_idx):,} train examples, {len(val_idx):,} val examples")

    q_train = [questions[i] for i in train_idx]
    q_val   = [questions[i] for i in val_idx]
    c_train = [chunks[i]    for i in train_idx]
    c_val   = [chunks[i]    for i in val_idx]
    y_train = y[train_idx]
    y_val   = y[val_idx]

    # 3. Fit feature extractor on training texts only (no val leakage)
    print("Fitting TF-IDF on training corpus …")
    extractor = FeatureExtractor(use_embeddings=use_embeddings)
    extractor.fit(q_train + [c["text"] for c in c_train])

    # 4. Compute embeddings in bulk (avoids per-example encoder calls)
    q_emb_train = c_emb_train = q_emb_val = c_emb_val = None
    if use_embeddings:
        print("Computing sentence embeddings for train split …")
        q_emb_train = extractor.embed(q_train)
        c_emb_train = extractor.embed([c["text"] for c in c_train])
        print("Computing sentence embeddings for val split …")
        q_emb_val   = extractor.embed(q_val)
        c_emb_val   = extractor.embed([c["text"] for c in c_val])

    print("Extracting feature vectors …")
    X_train = extractor.extract_batch(q_train, c_train, q_emb_train, c_emb_train)
    X_val   = extractor.extract_batch(q_val,   c_val,   q_emb_val,   c_emb_val)
    print(f"Feature matrix: train {X_train.shape}, val {X_val.shape}")

    # 5. Train
    print(f"Training logistic regression (C={C}, penalty=l1) …")
    model = train_model(X_train, y_train, C=C, random_seed=random_seed)

    # 6. Evaluate
    # Classification diagnostics (is the model ranking positive chunks higher?)
    train_metrics = evaluate_model(model, X_train, y_train)
    val_metrics   = evaluate_model(model, X_val,   y_val)
    print("Train classification:", train_metrics)
    print("Val   classification:", val_metrics)

    # Per-question Recall@K — the primary retrieval metric
    val_record_ids = [record_ids[i] for i in val_idx]
    val_questions  = [questions[i]  for i in val_idx]
    recall_metrics = per_question_recall_at_k(
        model, X_val, y_val, val_questions, val_record_ids
    )
    print("Val   Recall@K (per-question):", recall_metrics)
    val_metrics.update(recall_metrics)

    # Feature importance summary
    coef = model.coef_[0]
    n_nonzero = int((coef != 0).sum())
    feat_names = extractor.feature_names
    top_idx    = np.argsort(np.abs(coef))[-15:][::-1]
    top_features = [(feat_names[i], float(coef[i])) for i in top_idx]
    print(f"Non-zero features: {n_nonzero}/{len(coef)}")
    print("Top features:", top_features)

    # 7. Save artifacts
    extractor.save(output_dir / "feature_extractor.pkl")
    with open(output_dir / "model.pkl", "wb") as f:
        pickle.dump(model, f)

    metrics = {
        "train":             train_metrics,
        "val":               val_metrics,
        "top_features":      top_features,
        "n_nonzero_features": n_nonzero,
        "config": {
            "strategy":        strategy,
            "C":               C,
            "label_threshold": label_threshold,
            "use_embeddings":  use_embeddings,
        },
    }
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # 8. Save diagnostic plots
    _save_plots(
        model, X_val, y_val,
        feat_names=feat_names,
        recall_metrics=recall_metrics,
        output_dir=output_dir,
    )

    print(f"\nArtifacts saved to {output_dir}/")
    return metrics
