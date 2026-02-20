"""Apply a trained logistic regression model to score and select chunks.

At inference time:
  1. Load model + feature extractor from disk (saved by train.py)
  2. Chunk the discharge summary using the same strategy used during training
  3. Score all chunks with the learned model
  4. Return the top-K chunks by predicted usefulness probability

Also provides Recall@K evaluation for a single (question, note, answer) triple,
which the CLI uses to compute mean Recall@K over a test set.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from Logreg.chunker import chunk_note
from Logreg.features import FeatureExtractor
from Logreg.labeler import label_chunks


class ChunkSelector:
    """Score and select discharge summary chunks for a given question."""

    def __init__(
        self,
        model: LogisticRegression,
        extractor: FeatureExtractor,
        strategy: str = "section",
    ):
        self.model     = model
        self.extractor = extractor
        self.strategy  = strategy

    @classmethod
    def load(cls, model_dir: str | Path, strategy: str = "section") -> "ChunkSelector":
        """Load a trained selector from a directory written by run_training()."""
        model_dir = Path(model_dir)
        with open(model_dir / "model.pkl", "rb") as f:
            model = pickle.load(f)
        extractor = FeatureExtractor.load(model_dir / "feature_extractor.pkl")
        return cls(model, extractor, strategy=strategy)

    # ── Core scoring ──────────────────────────────────────────────────────────

    def score_chunks(
        self,
        question: str,
        chunks: list[dict],
        c_embeddings: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return predicted usefulness probabilities for each chunk. Shape: (N,)."""
        if not chunks:
            return np.array([])
        N = len(chunks)
        questions = [question] * N

        # Compute question embedding once (reused for every chunk)
        q_embeddings = None
        if self.extractor.use_embeddings:
            q_emb = self.extractor.embed([question])  # (1, D)
            q_embeddings = np.tile(q_emb, (N, 1))     # (N, D)

        X = self.extractor.extract_batch(questions, chunks, q_embeddings, c_embeddings)
        return self.model.predict_proba(X)[:, 1]

    # ── Top-K selection ───────────────────────────────────────────────────────

    def select(
        self,
        question: str,
        note_text: str,
        K: int = 5,
    ) -> list[dict]:
        """Select the top-K most useful chunks from a discharge summary.

        Returns a list of chunk dicts sorted by score (descending), each
        with an additional 'score' key containing the predicted probability.
        """
        chunks = chunk_note(note_text, strategy=self.strategy)
        if not chunks:
            return []

        scores  = self.score_chunks(question, chunks)
        top_idx = np.argsort(scores)[-K:][::-1]

        result = []
        for i in top_idx:
            chunk = dict(chunks[i])
            chunk["score"] = float(scores[i])
            result.append(chunk)
        return result

    # ── Evaluation ────────────────────────────────────────────────────────────

    def recall_at_k(
        self,
        question: str,
        note_text: str,
        answer: str,
        K: int = 5,
        label_threshold: float = 0.15,
    ) -> float:
        """Compute Recall@K for a single (question, note, answer) triple.

        Recall@K = (# positive chunks in top-K) / (# total positive chunks)

        A chunk is positive if token F1(chunk, answer) ≥ label_threshold.
        Returns 0.0 if there are no positive chunks (unanswerable question).
        """
        chunks = chunk_note(note_text, strategy=self.strategy)
        if not chunks:
            return 0.0

        chunk_labels = label_chunks(chunks, answer, threshold=label_threshold)
        n_pos = sum(chunk_labels)
        if n_pos == 0:
            return 0.0

        scores      = self.score_chunks(question, chunks)
        top_k_idx   = set(np.argsort(scores)[-K:].tolist())
        hits        = sum(1 for i in top_k_idx if chunk_labels[i] == 1)
        return hits / n_pos
