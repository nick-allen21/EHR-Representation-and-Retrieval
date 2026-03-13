"""Assign weak binary labels to (question, chunk) pairs using answer-chunk overlap.

Supervision signal: a chunk is labeled positive (1) if either:
  (a) its token-level F1 with the gold answer exceeds a threshold, OR
  (b) the question and chunk share a recognized clinical entity AND the chunk
      is in a clinically relevant section (entity-boosted labeling).

The entity boost fixes a known failure mode: lab/vital/medication event chunks
contain real numeric values (e.g., "Sodium: 125 mEq/L") but token F1 against
a short answer is diluted by the chunk's length, putting it below the threshold
even though it is clearly the evidence the question is asking about.
"""

from __future__ import annotations

import re
from collections import Counter

# ── Clinical entity vocabulary ────────────────────────────────────────────────
# Terms that, when shared between a question and a chunk, signal relevance.
# Kept intentionally specific to avoid false positives in narrative text.
_CLINICAL_ENTITIES: frozenset[str] = frozenset({
    # Labs
    "sodium", "potassium", "creatinine", "glucose", "hemoglobin", "hematocrit",
    "wbc", "platelet", "bun", "bicarbonate", "chloride", "calcium", "magnesium",
    "phosphorus", "albumin", "bilirubin", "troponin", "lactate", "inr", "ptt",
    "ast", "alt", "alkaline", "phosphatase", "bnp", "lipase", "amylase",
    # Vitals
    "temperature", "pulse", "systolic", "diastolic", "saturation", "respiratory",
    # Common medications
    "aspirin", "metoprolol", "lisinopril", "atorvastatin", "furosemide",
    "warfarin", "insulin", "heparin", "vancomycin", "metformin", "amlodipine",
    "omeprazole", "lorazepam", "morphine", "acetaminophen", "ibuprofen",
    "prednisone", "dexamethasone", "clopidogrel", "amiodarone", "digoxin",
})

# Sections where entity-boosted labels are applied.
# Restricted to structured data sections only — narrative sections like
# hospital_course mention entities everywhere and create noisy positives.
_ENTITY_BOOST_SECTIONS: frozenset[str] = frozenset({
    "labs", "vitals", "medications", "results",
})

# Entity boost also requires a numeric value near the entity (measurement, not mention)
_HAS_NUMERIC_RE = re.compile(r"\b\d+\.?\d*\s*(?:meq|mg|g|dl|ml|mmol|units?|%|bpm|mmhg|kg|iu|ng|mcg|miu|u/l|/ul)\b", re.I)


def _tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, split on whitespace."""
    return re.findall(r"[a-z0-9]+", text.lower())


def token_f1(text_a: str, text_b: str) -> float:
    """Token-level F1 between two text strings.

    Treats each text as a bag of tokens.  F1 is the harmonic mean of
    precision (fraction of text_b tokens that appear in text_a) and
    recall (fraction of text_a tokens that appear in text_b).
    """
    toks_a = Counter(_tokenize(text_a))
    toks_b = Counter(_tokenize(text_b))
    if not toks_a or not toks_b:
        return 0.0
    common    = sum((toks_a & toks_b).values())
    precision = common / sum(toks_b.values())
    recall    = common / sum(toks_a.values())
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _question_entities(question: str) -> frozenset[str]:
    """Return clinical entities mentioned in the question."""
    tokens = frozenset(re.findall(r"[a-z]+", question.lower()))
    return tokens & _CLINICAL_ENTITIES


def label_chunks(
    chunks: list[dict],
    answer: str,
    threshold: float = 0.15,
    question: str | None = None,
) -> list[int]:
    """Return a binary label per chunk.

    Primary signal: token F1(chunk, answer) >= threshold.
    Boost signal (when question is provided): if the question and chunk share
    a clinical entity AND the chunk is in a relevant section, label positive
    even if token F1 is below threshold.

    Args:
        chunks:     List of chunk dicts (must have 'text' and 'section' keys).
        answer:     Gold answer string.
        threshold:  Minimum token F1 for a chunk to be labeled positive.
        question:   Optional question string for entity-boosted labeling.

    Returns:
        List of 0/1 labels, one per chunk.
    """
    labels = [
        1 if token_f1(chunk["text"], answer) >= threshold else 0
        for chunk in chunks
    ]

    if question:
        q_entities = _question_entities(question)
        if q_entities:
            for i, chunk in enumerate(chunks):
                if labels[i] == 1:
                    continue  # already positive
                if chunk.get("section") not in _ENTITY_BOOST_SECTIONS:
                    continue
                chunk_tokens = frozenset(re.findall(r"[a-z]+", chunk["text"].lower()))
                # Also require a numeric measurement in the chunk (not just a mention)
                if q_entities & chunk_tokens and _HAS_NUMERIC_RE.search(chunk["text"]):
                    labels[i] = 1

    return labels
