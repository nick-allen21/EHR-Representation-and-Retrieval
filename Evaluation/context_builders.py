"""Build LLM prompts for each retrieval strategy.

Each builder takes a patient record + question and returns a dict containing
the user-facing prompt, token counts, and method metadata.  The system prompt
is shared across all strategies and exported as SYSTEM_PROMPT.

Strategies
----------
discharge_only   Raw discharge summary text (floor baseline).
full_context     Demographics + admission + all events + discharge summary.
recency          Discharge summary + N most recent events.
bm25             Top-K chunks ranked by BM25 score.
semantic_rag     Top-K chunks ranked by embedding cosine similarity.
learned          Top-K chunks from the trained logistic regression selector.
"""

from __future__ import annotations

import numpy as np
import tiktoken
from rank_bm25 import BM25Okapi

from Logreg.chunker import chunk_note
from Logreg.data_loader import serialize_events

# ── Token helpers ─────────────────────────────────────────────────────────────

_encoder: tiktoken.Encoding | None = None


def _get_encoder() -> tiktoken.Encoding:
    global _encoder
    if _encoder is None:
        _encoder = tiktoken.encoding_for_model("gpt-4o")
    return _encoder


def count_tokens(text: str) -> int:
    return len(_get_encoder().encode(text))


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    enc = _get_encoder()
    toks = enc.encode(text)
    if len(toks) <= max_tokens:
        return text
    return enc.decode(toks[:max_tokens])


# ── Prompt template (stub — will be refined during prompt-design phase) ───────

SYSTEM_PROMPT = (
    "You are a clinical QA assistant. Answer the question based ONLY on the "
    "provided patient context. Be concise and specific — include relevant "
    "values, dates, and medication names where applicable. If the context does "
    "not contain enough information to answer, say so."
)


def format_user_prompt(context: str, question: str) -> str:
    return (
        f"### Patient Context\n\n{context}\n\n"
        f"### Question\n\n{question}"
    )


def _overhead_tokens(question: str) -> int:
    """Tokens consumed by the prompt wrapper (excluding the context body)."""
    return count_tokens(format_user_prompt("", question))


# ── Helpers for full_context / recency ────────────────────────────────────────

def _serialize_demographics(demo: dict) -> str:
    if not demo:
        return ""
    lines = ["DEMOGRAPHICS:"]
    for k, v in demo.items():
        if v is not None:
            lines.append(f"  {k}: {v}")
    return "\n".join(lines)


def _serialize_admission(adm: dict) -> str:
    if not adm:
        return ""
    lines = ["ADMISSION:"]
    for k, v in adm.items():
        if v is not None:
            lines.append(f"  {k}: {v}")
    return "\n".join(lines)


# ── Strategy: discharge_only ──────────────────────────────────────────────────

def build_discharge_only(
    record: dict,
    question: str,
    *,
    token_budget: int = 4096,
    **_kw,
) -> dict:
    ds = record.get("discharge_summary", "")
    context = truncate_to_tokens(ds, token_budget - _overhead_tokens(question))
    return {
        "user_prompt": format_user_prompt(context, question),
        "context_tokens": count_tokens(context),
        "method": "discharge_only",
    }


# ── Strategy: full_context ────────────────────────────────────────────────────

def build_full_context(
    record: dict,
    question: str,
    *,
    token_budget: int = 4096,
    **_kw,
) -> dict:
    parts: list[str] = []
    demo = _serialize_demographics(record.get("demographics", {}))
    if demo:
        parts.append(demo)
    adm = _serialize_admission(record.get("admission", {}))
    if adm:
        parts.append(adm)
    events_text = serialize_events(record.get("events", []))
    if events_text:
        parts.append(events_text)
    ds = record.get("discharge_summary", "")
    if ds:
        parts.append(ds)

    full = "\n\n".join(parts)
    context = truncate_to_tokens(full, token_budget - _overhead_tokens(question))
    return {
        "user_prompt": format_user_prompt(context, question),
        "context_tokens": count_tokens(context),
        "method": "full_context",
    }


# ── Strategy: recency ─────────────────────────────────────────────────────────

def build_recency(
    record: dict,
    question: str,
    *,
    n: int = 25,
    token_budget: int = 4096,
    **_kw,
) -> dict:
    events = record.get("events", [])
    recent = events[-n:] if len(events) > n else events
    events_text = serialize_events(recent)

    parts = [record.get("discharge_summary", "")]
    if events_text:
        parts.append(events_text)
    full = "\n\n".join(p for p in parts if p)
    context = truncate_to_tokens(full, token_budget - _overhead_tokens(question))
    return {
        "user_prompt": format_user_prompt(context, question),
        "context_tokens": count_tokens(context),
        "method": f"recency_n{n}",
    }


# ── Strategy: bm25 ───────────────────────────────────────────────────────────

def build_bm25(
    record: dict,
    question: str,
    *,
    k: int = 5,
    token_budget: int = 4096,
    **_kw,
) -> dict:
    note_text = record.get("note_text", record.get("discharge_summary", ""))
    chunks = chunk_note(note_text, strategy="section")
    if not chunks:
        return build_discharge_only(record, question, token_budget=token_budget)

    tokenized = [c["text"].lower().split() for c in chunks]
    bm25 = BM25Okapi(tokenized)
    scores = bm25.get_scores(question.lower().split())

    top_k = min(k, len(chunks))
    top_idx = np.argsort(scores)[-top_k:][::-1]

    # Re-sort by note position for reading coherence
    selected = sorted(top_idx.tolist())
    full = "\n\n".join(chunks[i]["text"] for i in selected)
    context = truncate_to_tokens(full, token_budget - _overhead_tokens(question))
    return {
        "user_prompt": format_user_prompt(context, question),
        "context_tokens": count_tokens(context),
        "method": f"bm25_k{k}",
    }


# ── Strategy: semantic_rag ────────────────────────────────────────────────────

_embed_model = None


def _get_embed_model():
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer
        _embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embed_model


def build_semantic_rag(
    record: dict,
    question: str,
    *,
    k: int = 5,
    token_budget: int = 4096,
    embedder=None,
    **_kw,
) -> dict:
    note_text = record.get("note_text", record.get("discharge_summary", ""))
    chunks = chunk_note(note_text, strategy="section")
    if not chunks:
        return build_discharge_only(record, question, token_budget=token_budget)

    model = embedder or _get_embed_model()
    chunk_texts = [c["text"] for c in chunks]
    q_emb = model.encode([question], normalize_embeddings=True)
    c_embs = model.encode(chunk_texts, normalize_embeddings=True)
    scores = (c_embs @ q_emb.T).flatten()

    top_k = min(k, len(chunks))
    top_idx = np.argsort(scores)[-top_k:][::-1]

    selected = sorted(top_idx.tolist())
    full = "\n\n".join(chunks[i]["text"] for i in selected)
    context = truncate_to_tokens(full, token_budget - _overhead_tokens(question))
    return {
        "user_prompt": format_user_prompt(context, question),
        "context_tokens": count_tokens(context),
        "method": f"semantic_rag_k{k}",
    }


# ── Strategy: learned ─────────────────────────────────────────────────────────

def build_learned(
    record: dict,
    question: str,
    *,
    k: int = 5,
    token_budget: int = 4096,
    selector=None,
    **_kw,
) -> dict:
    if selector is None:
        raise ValueError(
            "build_learned requires a ChunkSelector via the selector= kwarg"
        )

    note_text = record.get("note_text", record.get("discharge_summary", ""))
    selected_chunks = selector.select(question, note_text, K=k)
    if not selected_chunks:
        return build_discharge_only(record, question, token_budget=token_budget)

    selected_chunks.sort(key=lambda c: c.get("chunk_idx", 0))
    full = "\n\n".join(c["text"] for c in selected_chunks)
    context = truncate_to_tokens(full, token_budget - _overhead_tokens(question))
    return {
        "user_prompt": format_user_prompt(context, question),
        "context_tokens": count_tokens(context),
        "method": f"learned_k{k}",
    }


# ── Public registry ───────────────────────────────────────────────────────────

STRATEGIES: dict[str, callable] = {
    "discharge_only": build_discharge_only,
    "full_context":   build_full_context,
    "recency":        build_recency,
    "bm25":           build_bm25,
    "semantic_rag":   build_semantic_rag,
    "learned":        build_learned,
}
