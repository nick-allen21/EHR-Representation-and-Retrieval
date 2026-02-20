"""Feature extraction for (question, chunk) pairs.

Feature vector layout (F dimensions total):
  [0]       embed_sim          — cosine similarity of sentence embeddings
  [1]       tfidf_sim          — cosine similarity of TF-IDF vectors
  [2]       lexical_overlap    — fraction of question tokens found in chunk
  [3]       position           — normalized chunk position in note [0, 1]
  [4]       log_length         — log(1 + whitespace token count of chunk)
  [5:22]    sec_*              — one-hot section category (17 categories)
  [22:28]   qt_*               — one-hot question type  (6 categories)
  [28:28+17*6] int_*_*         — section × question_type interactions (102 dims)

Total: 5 + 17 + 6 + 102 = 130 features

The FeatureExtractor must be `.fit()` on the training corpus before
calling `.extract_batch()` on any split.
"""

from __future__ import annotations

import pickle
import re
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from Logreg.chunker import SECTION_CATEGORIES

# ── Question type taxonomy ────────────────────────────────────────────────────

_QUESTION_TYPE_KEYWORDS: list[tuple[str, list[str]]] = [
    ("medications", ["medication", "drug", "prescribed", "discharge on",
                     "taking", "medications", "meds", "dose", "dosage"]),
    ("diagnosis",   ["diagnosis", "diagnosed", "condition", "disease",
                     "disorder", "finding", "impression"]),
    ("labs",        ["lab", "laboratory", "level", "result", "value", "test",
                     "sodium", "potassium", "creatinine", "glucose", "hemoglobin"]),
    ("imaging",     ["imaging", "scan", "x-ray", "ct ", "mri", "ultrasound",
                     "radiograph", "chest"]),
    ("procedure",   ["procedure", "surgery", "operation", "performed",
                     "intervention"]),
]
QUESTION_TYPES: list[str] = [qt for qt, _ in _QUESTION_TYPE_KEYWORDS] + ["other"]


def classify_question(question: str) -> str:
    q = question.lower()
    for qtype, keywords in _QUESTION_TYPE_KEYWORDS:
        if any(kw in q for kw in keywords):
            return qtype
    return "other"


def _token_overlap(text_a: str, text_b: str) -> float:
    """Fraction of unique tokens in text_a that also appear in text_b."""
    toks_a = set(re.findall(r"[a-z0-9]+", text_a.lower()))
    toks_b = set(re.findall(r"[a-z0-9]+", text_b.lower()))
    if not toks_a:
        return 0.0
    return len(toks_a & toks_b) / len(toks_a)


# ── Feature extractor ─────────────────────────────────────────────────────────

class FeatureExtractor:
    """Extract a fixed-length numeric feature vector for each (question, chunk) pair.

    Workflow:
        extractor = FeatureExtractor()
        extractor.fit(train_texts)          # fit TF-IDF on training corpus
        X = extractor.extract_batch(qs, cs) # extract features for all pairs

    Set use_embeddings=False to skip sentence-transformer inference (faster,
    but the embed_sim feature will be 0 for all examples).
    """

    def __init__(
        self,
        use_embeddings: bool = True,
        embedding_model: str = "all-MiniLM-L6-v2",
    ):
        self.use_embeddings     = use_embeddings
        self.embedding_model_name = embedding_model
        self.tfidf: TfidfVectorizer | None = None
        self._embed_model = None

    # ── Fitting ───────────────────────────────────────────────────────────────

    def fit(self, texts: list[str]) -> "FeatureExtractor":
        """Fit the TF-IDF vectorizer on a corpus of text (questions + chunks)."""
        self.tfidf = TfidfVectorizer(
            max_features=20_000,
            ngram_range=(1, 2),
            min_df=2,
            sublinear_tf=True,
        )
        self.tfidf.fit(texts)
        return self

    # ── Embedding helpers ─────────────────────────────────────────────────────

    def _get_embed_model(self):
        if self._embed_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._embed_model = SentenceTransformer(self.embedding_model_name)
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required for embedding features.\n"
                    "Install it with: pip install sentence-transformers"
                )
        return self._embed_model

    def embed(self, texts: list[str], batch_size: int = 256) -> np.ndarray:
        """Encode texts to unit-norm sentence embeddings. Shape: (N, D)."""
        model = self._get_embed_model()
        return model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=len(texts) > 500,
            normalize_embeddings=True,
        )

    # ── Core extraction ───────────────────────────────────────────────────────

    def extract_batch(
        self,
        questions: list[str],
        chunks: list[dict],
        q_embeddings: np.ndarray | None = None,
        c_embeddings: np.ndarray | None = None,
    ) -> np.ndarray:
        """Extract features for parallel lists of questions and chunks.

        Args:
            questions:    Question strings (length N).
            chunks:       Chunk dicts (length N).
            q_embeddings: Pre-computed question embeddings (N, D) — optional.
            c_embeddings: Pre-computed chunk embeddings (N, D) — optional.

        Returns:
            Float32 feature matrix of shape (N, F).
        """
        N = len(questions)
        assert len(chunks) == N, "questions and chunks must have the same length"

        chunk_texts = [c["text"] for c in chunks]

        # ── TF-IDF similarity ─────────────────────────────────────────────────
        if self.tfidf is not None:
            q_tfidf = normalize(self.tfidf.transform(questions), norm="l2")
            c_tfidf = normalize(self.tfidf.transform(chunk_texts), norm="l2")
            # Pairwise diagonal: cosine(question_i, chunk_i) for each i
            tfidf_sim = np.asarray(q_tfidf.multiply(c_tfidf).sum(axis=1)).flatten()
        else:
            tfidf_sim = np.zeros(N)

        # ── Embedding similarity ──────────────────────────────────────────────
        if self.use_embeddings:
            if q_embeddings is None:
                q_embeddings = self.embed(questions)
            if c_embeddings is None:
                c_embeddings = self.embed(chunk_texts)
            embed_sim = (q_embeddings * c_embeddings).sum(axis=1)  # dot of unit vecs
        else:
            embed_sim = np.zeros(N)

        # ── Lexical overlap ───────────────────────────────────────────────────
        lexical = np.array([
            _token_overlap(q, c["text"])
            for q, c in zip(questions, chunks)
        ])

        # ── Structural features ───────────────────────────────────────────────
        position   = np.array([c.get("position", 0.5) for c in chunks])
        log_length = np.log1p([len(c["text"].split()) for c in chunks])

        # ── Section one-hot ───────────────────────────────────────────────────
        n_sec = len(SECTION_CATEGORIES)
        sec_onehot = np.zeros((N, n_sec))
        for i, c in enumerate(chunks):
            sec = c.get("section", "other")
            idx = SECTION_CATEGORIES.index(sec) if sec in SECTION_CATEGORIES else SECTION_CATEGORIES.index("other")
            sec_onehot[i, idx] = 1.0

        # ── Question-type one-hot ─────────────────────────────────────────────
        n_qt = len(QUESTION_TYPES)
        qt_onehot = np.zeros((N, n_qt))
        for i, q in enumerate(questions):
            qt  = classify_question(q)
            idx = QUESTION_TYPES.index(qt)
            qt_onehot[i, idx] = 1.0

        # ── Section × question-type interactions ──────────────────────────────
        # Shape: (N, n_sec, n_qt) → flatten to (N, n_sec * n_qt)
        interactions = (sec_onehot[:, :, None] * qt_onehot[:, None, :]).reshape(N, -1)

        # ── Assemble ──────────────────────────────────────────────────────────
        X = np.column_stack([
            embed_sim,
            tfidf_sim,
            lexical,
            position,
            log_length,
            sec_onehot,
            qt_onehot,
            interactions,
        ])
        return X.astype(np.float32)

    @property
    def feature_names(self) -> list[str]:
        names  = ["embed_sim", "tfidf_sim", "lexical_overlap", "position", "log_length"]
        names += [f"sec_{s}"  for s in SECTION_CATEGORIES]
        names += [f"qt_{t}"   for t in QUESTION_TYPES]
        names += [f"int_{s}_{t}" for s in SECTION_CATEGORIES for t in QUESTION_TYPES]
        return names

    # ── Serialization ─────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """Save TF-IDF state to disk. The embedding model is not saved (reloaded on demand)."""
        with open(path, "wb") as f:
            pickle.dump({
                "tfidf":                self.tfidf,
                "use_embeddings":       self.use_embeddings,
                "embedding_model_name": self.embedding_model_name,
            }, f)

    @classmethod
    def load(cls, path: str | Path) -> "FeatureExtractor":
        with open(path, "rb") as f:
            state = pickle.load(f)
        obj = cls(
            use_embeddings=state["use_embeddings"],
            embedding_model=state["embedding_model_name"],
        )
        obj.tfidf = state["tfidf"]
        return obj
