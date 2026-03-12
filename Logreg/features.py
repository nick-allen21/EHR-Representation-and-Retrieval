"""Feature extraction for (question, chunk) pairs.

Feature vector layout (F dimensions total):
  [0]       embed_sim          — cosine similarity of sentence embeddings
  [1]       tfidf_sim          — cosine similarity of TF-IDF vectors
  [2]       lexical_overlap    — fraction of question tokens found in chunk
  [3]       position           — normalized chunk position in note [0, 1]
  [4]       log_length         — log(1 + whitespace token count of chunk)

  Group A — text-signal features (computed from chunk text):
  [5]       has_temporal_marker — chunk mentions time-relative terms (today, POD N, etc.)
  [6]       temporal_density    — temporal marker count / chunk word count
  [7]       has_critical_event  — chunk mentions ICU transfer, intubation, arrest, etc.
  [8]       has_abnormal_lab    — chunk contains abnormal lab flag ([H], [L], critical, etc.)

  Group B — temporal proximity features (require _dischtime on chunk dict):
  [9]       days_to_discharge   — days before discharge, normalized to [0, 1] over 30-day window
  [10]      is_within_24h       — event occurred within 24h of discharge
  [11]      is_within_1week     — event occurred within 7 days of discharge
  [12]      has_no_timestamp    — 1 for note-section chunks (no embedded timestamp)

  [13:30]   sec_*              — one-hot section category (17 categories)
  [30:36]   qt_*               — one-hot question type  (6 categories)
  [36:138]  int_*_*            — section × question_type interactions (102 dims)

  Group C — new signal × question-type interactions (8 dims):
  [138]     int_abnormal_lab_x_labs
  [139]     int_abnormal_lab_x_diagnosis
  [140]     int_critical_event_x_diagnosis
  [141]     int_critical_event_x_procedure
  [142]     int_within_24h_x_medications
  [143]     int_within_24h_x_labs
  [144]     int_temporal_marker_x_labs
  [145]     int_temporal_marker_x_diagnosis

Total: 5 + 4 + 4 + 17 + 6 + 102 + 8 = 146 features

Temporal metadata is passed via chunk["_dischtime"] (an ISO datetime string,
e.g. "2180-06-27 14:30:00"). Set by build_dataset() and selector.select().
Chunks without this field get neutral temporal feature values.

The FeatureExtractor must be `.fit()` on the training corpus before
calling `.extract_batch()` on any split.
"""

from __future__ import annotations

import pickle
import re
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from Logreg.chunker import SECTION_CATEGORIES

# ── Text-signal regexes (Group A) ─────────────────────────────────────────────

# Temporal markers: relative time expressions common in clinical notes
_TEMPORAL_MARKER_RE = re.compile(
    r"\b("
    r"today|tonight|overnight|this\s+(morning|afternoon|evening|night)"
    r"|earlier\s+today|just\s+now|a\s+few\s+hours"
    r"|post[\-\s]?op\s+day\s*\d*|pod\s*#?\s*\d+"
    r"|hospital\s+day\s*\d*|hd\s*#?\s*\d+"
    r"|day\s+[1-9]\d?(?:\s|$)"           # "day 1", "day 14" but not "day-to-day"
    r")",
    re.IGNORECASE,
)

# Critical clinical events: transfers, respiratory interventions, hemodynamic crises
_CRITICAL_EVENT_RE = re.compile(
    r"\b("
    r"icu|intensive\s+care\s+unit|micu|sicu|ccicu|neuro\s*icu"
    r"|intubat|extubat|re[\-\s]?intubat"
    r"|mechanical\s+ventilat|vent(?:ilator)?"
    r"|vasopressor|pressor|norepinephrine|dopamine|epinephrine\s+drip"
    r"|cardiac\s+arrest|code\s+blue|cpr|rapid\s+response"
    r"|emergent(?:ly)?|emergenc"
    r"|transfer(?:red)?\s+to\s+(?:the\s+)?(?:icu|micu|sicu)"
    r")",
    re.IGNORECASE,
)

# Abnormal lab flags: serialized event format ([H], [L], [abnormal]) and free text
_ABNORMAL_LAB_RE = re.compile(
    r"("
    r"\[(?:H|L|abnormal|critical|high|low)\]"   # serialized flag: [H], [L], [abnormal]
    r"|\bcritical\s+(?:value|result|lab)\b"
    r"|\b(?:elevated|abnormal)\s+\w+\s*(?:level|value|result)?\b"
    r"|\bmarkedly\s+(?:elevated|reduced|low|high)\b"
    r")",
    re.IGNORECASE,
)

# ISO timestamp embedded by event serializers: "2180-06-26 22:45:00" or "2180-06-26"
_TIMESTAMP_RE = re.compile(r"\b(\d{4}-\d{2}-\d{2}(?:\s+\d{2}:\d{2}(?::\d{2})?)?)\b")


def _count_temporal_markers(text: str) -> int:
    return len(_TEMPORAL_MARKER_RE.findall(text))


def _extract_temporal_meta(
    chunk_text: str,
    dischtime: str | None,
) -> tuple[float, float, float, float]:
    """Compute Group B temporal proximity features for one chunk.

    Returns (days_to_discharge_norm, is_within_24h, is_within_1week, has_no_timestamp).
      days_to_discharge_norm: days before discharge / 30 (clamped [0, 1]);
                              0.5 when no useful timestamp is available.
      has_no_timestamp: 1.0 when no ISO date found in chunk text (note-section chunks).
    """
    m = _TIMESTAMP_RE.search(chunk_text)
    if m is None or not dischtime:
        return 0.5, 0.0, 0.0, 1.0

    try:
        ts_str = m.group(1)
        fmt = "%Y-%m-%d %H:%M:%S" if len(ts_str) > 10 else "%Y-%m-%d"
        chunk_dt = datetime.strptime(ts_str, fmt)
        disch_dt = datetime.strptime(dischtime[:19], "%Y-%m-%d %H:%M:%S")
        days_before = max(0.0, (disch_dt - chunk_dt).total_seconds() / 86400.0)
        return (
            min(days_before / 30.0, 1.0),   # normalized, 30-day window
            float(days_before <= 1.0),
            float(days_before <= 7.0),
            0.0,
        )
    except (ValueError, OverflowError):
        return 0.5, 0.0, 0.0, 1.0

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

        # ── Group A: text-signal features ────────────────────────────────────
        has_temporal_marker = np.array([
            float(_count_temporal_markers(c["text"]) > 0) for c in chunks
        ])
        temporal_density = np.array([
            _count_temporal_markers(c["text"]) / max(len(c["text"].split()), 1)
            for c in chunks
        ])
        has_critical_event = np.array([
            float(bool(_CRITICAL_EVENT_RE.search(c["text"]))) for c in chunks
        ])
        has_abnormal_lab = np.array([
            float(bool(_ABNORMAL_LAB_RE.search(c["text"]))) for c in chunks
        ])

        # ── Group B: temporal proximity features ─────────────────────────────
        # Reads _dischtime from chunk dict (set by build_dataset / selector.select)
        temporal_rows = np.array([
            _extract_temporal_meta(c["text"], c.get("_dischtime"))
            for c in chunks
        ])  # (N, 4): days_norm, within_24h, within_1week, has_no_timestamp
        days_to_discharge = temporal_rows[:, 0]
        is_within_24h     = temporal_rows[:, 1]
        is_within_1week   = temporal_rows[:, 2]
        has_no_timestamp  = temporal_rows[:, 3]

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
        qt_indices = []
        for i, q in enumerate(questions):
            qt  = classify_question(q)
            idx = QUESTION_TYPES.index(qt)
            qt_onehot[i, idx] = 1.0
            qt_indices.append(idx)

        # ── Section × question-type interactions ──────────────────────────────
        interactions = (sec_onehot[:, :, None] * qt_onehot[:, None, :]).reshape(N, -1)

        # ── Group C: new signal × question-type interactions ──────────────────
        # Indices into QUESTION_TYPES: medications=0, diagnosis=1, labs=2, procedure=4
        _qt = {qt: QUESTION_TYPES.index(qt) for qt in QUESTION_TYPES}
        new_interactions = np.column_stack([
            has_abnormal_lab    * qt_onehot[:, _qt["labs"]],
            has_abnormal_lab    * qt_onehot[:, _qt["diagnosis"]],
            has_critical_event  * qt_onehot[:, _qt["diagnosis"]],
            has_critical_event  * qt_onehot[:, _qt["procedure"]],
            is_within_24h       * qt_onehot[:, _qt["medications"]],
            is_within_24h       * qt_onehot[:, _qt["labs"]],
            has_temporal_marker * qt_onehot[:, _qt["labs"]],
            has_temporal_marker * qt_onehot[:, _qt["diagnosis"]],
        ])

        # ── Assemble ──────────────────────────────────────────────────────────
        X = np.column_stack([
            embed_sim,
            tfidf_sim,
            lexical,
            position,
            log_length,
            has_temporal_marker,
            temporal_density,
            has_critical_event,
            has_abnormal_lab,
            days_to_discharge,
            is_within_24h,
            is_within_1week,
            has_no_timestamp,
            sec_onehot,
            qt_onehot,
            interactions,
            new_interactions,
        ])
        return X.astype(np.float32)

    @property
    def feature_names(self) -> list[str]:
        names  = ["embed_sim", "tfidf_sim", "lexical_overlap", "position", "log_length"]
        names += ["has_temporal_marker", "temporal_density", "has_critical_event", "has_abnormal_lab"]
        names += ["days_to_discharge", "is_within_24h", "is_within_1week", "has_no_timestamp"]
        names += [f"sec_{s}"  for s in SECTION_CATEGORIES]
        names += [f"qt_{t}"   for t in QUESTION_TYPES]
        names += [f"int_{s}_{t}" for s in SECTION_CATEGORIES for t in QUESTION_TYPES]
        names += [
            "int_abnormal_lab_x_labs",
            "int_abnormal_lab_x_diagnosis",
            "int_critical_event_x_diagnosis",
            "int_critical_event_x_procedure",
            "int_within_24h_x_medications",
            "int_within_24h_x_labs",
            "int_temporal_marker_x_labs",
            "int_temporal_marker_x_diagnosis",
        ]
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
