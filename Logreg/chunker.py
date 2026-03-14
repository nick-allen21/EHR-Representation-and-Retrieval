"""Split discharge summary notes into chunks for retrieval.

Two strategies:
  'section'  — split on MIMIC-IV discharge summary section headers (preferred)
  'fixed'    — split into fixed-size overlapping token windows (fallback)

A chunk is a dict with keys:
    text            str   — the chunk's raw text
    section         str   — broad clinical category (e.g. 'medications', 'hpi')
    section_header  str   — the original header string as it appeared in the note
    position        float — normalized position in the note [0.0, 1.0]
    chunk_idx       int   — sequential index within the note
"""

from __future__ import annotations

import re

### CITATION: Used Claude 4.6 Opus to help generate the regex patterns and
### the _SECTION_MAP clinical category vocabulary below.

# Section header detection
# Matches MIMIC-IV discharge summary headers (Title Case + colon, or ALL CAPS)
_HEADER_RE = re.compile(
    r"^[ \t]*([A-Z][A-Za-z0-9 /\-]+:)\s*$"   # Title Case + colon
    r"|^[ \t]*([A-Z][A-Z0-9 /\-]{3,})\s*$",  # ALL CAPS (4+ chars)
    re.MULTILINE,
)

# Map header text to broad clinical category
_SECTION_MAP: list[tuple[re.Pattern, str]] = [
    (re.compile(r"allerg",                      re.I), "allergies"),
    (re.compile(r"chief\s+complaint",           re.I), "chief_complaint"),
    (re.compile(r"history\s+of\s+present",      re.I), "hpi"),
    (re.compile(r"past\s+medical",              re.I), "pmh"),
    (re.compile(r"social\s+history",            re.I), "social"),
    (re.compile(r"family\s+history",            re.I), "family"),
    (re.compile(r"physical\s+exam",             re.I), "exam"),
    (re.compile(r"pertinent\s+result",          re.I), "results"),
    (re.compile(r"(brief\s+)?hospital\s+course",re.I), "hospital_course"),
    (re.compile(r"medication|drug",             re.I), "medications"),
    (re.compile(r"discharge\s+diag",            re.I), "diagnosis"),
    (re.compile(r"discharge\s+cond",            re.I), "discharge_condition"),
    (re.compile(r"discharge\s+disp",            re.I), "discharge_disposition"),
    (re.compile(r"discharge\s+instruct",        re.I), "discharge_instructions"),
    (re.compile(r"follow\s*[-–]?\s*up",         re.I), "followup"),
    (re.compile(r"lab|laboratory",              re.I), "labs"),
    (re.compile(r"vital\s*sign",                re.I), "vitals"),
    (re.compile(r"procedure|surger|operation",  re.I), "procedure"),
    (re.compile(r"diagnos",                     re.I), "diagnosis"),
]

# All known section category names (order matters — used for one-hot encoding)
SECTION_CATEGORIES: list[str] = [
    "medications",
    "diagnosis",
    "hospital_course",
    "results",
    "hpi",
    "allergies",
    "chief_complaint",
    "pmh",
    "exam",
    "labs",
    "followup",
    "discharge_instructions",
    "discharge_condition",
    "discharge_disposition",
    "social",
    "family",
    "vitals",
    "procedure",
    "other",
]


def _classify_section(header_text: str) -> str:
    for pattern, category in _SECTION_MAP:
        if pattern.search(header_text):
            return category
    return "other"


def split_by_sections(
    note_text: str,
    max_section_tokens: int = 200,
) -> list[dict]:
    """Split a discharge summary on section headers.

    Falls back to the whole note as a single chunk if fewer than 3 sections
    are detected (malformed / very short note).

    Sections exceeding max_section_tokens are automatically sub-split into
    overlapping fixed-size windows so that token-F1 weak supervision works
    well even for long sections (e.g., Hospital Course, large lab blocks).
    """
    sections: list[tuple[str, str]] = []   # (raw_header, body_text)
    current_header = "preamble"
    current_lines: list[str] = []

    for line in note_text.split("\n"):
        m = _HEADER_RE.match(line)
        if m:
            body = "\n".join(current_lines).strip()
            if body:
                sections.append((current_header, body))
            current_header = (m.group(1) or m.group(2) or "").strip().rstrip(":")
            current_lines = [line]
        else:
            current_lines.append(line)

    body = "\n".join(current_lines).strip()
    if body:
        sections.append((current_header, body))

    if len(sections) < 3:
        return split_fixed_size(note_text)

    # Build chunk list, sub-splitting oversized sections
    result: list[dict] = []
    for header, text in sections:
        section_cat = _classify_section(header)
        n_tokens = len(text.split())

        if n_tokens > max_section_tokens:
            sub_chunks = split_fixed_size(
                text, chunk_tokens=200, stride=100,
            )
            for sc in sub_chunks:
                sc["section"] = section_cat
                sc["section_header"] = header
            result.extend(sub_chunks)
        else:
            result.append({
                "text":           text,
                "section":        section_cat,
                "section_header": header,
                "position":       0.0,
                "chunk_idx":      0,
            })

    # Recompute position and chunk_idx across the flattened list
    total = len(result)
    for idx, chunk in enumerate(result):
        chunk["position"]  = idx / max(total - 1, 1)
        chunk["chunk_idx"] = idx

    return result


def split_fixed_size(
    note_text: str,
    chunk_tokens: int = 200,
    stride: int = 100,
) -> list[dict]:
    """Split a note into overlapping fixed-size windows (whitespace tokens)."""
    tokens = note_text.split()
    if not tokens:
        return []

    chunks: list[dict] = []
    idx = 0
    start = 0
    n = len(tokens)

    while start < n:
        end = min(start + chunk_tokens, n)
        text = " ".join(tokens[start:end])
        chunks.append({
            "text":           text,
            "section":        "other",
            "section_header": "",
            "position":       start / max(n - chunk_tokens, 1),
            "chunk_idx":      idx,
        })
        idx += 1
        if end == n:
            break
        start += stride

    return chunks


def chunk_note(
    note_text: str,
    strategy: str = "section",
    chunk_tokens: int = 200,
    stride: int = 100,
) -> list[dict]:
    """Chunk a discharge summary into retrievable units.

    Args:
        note_text:    Raw discharge summary text.
        strategy:     'section' (split on headers) or 'fixed' (token windows).
        chunk_tokens: Approximate tokens per chunk (only used for 'fixed').
        stride:       Token stride between windows (only used for 'fixed').

    Returns:
        List of chunk dicts (text, section, section_header, position, chunk_idx).
    """
    if strategy == "section":
        return split_by_sections(note_text)
    elif strategy == "fixed":
        return split_fixed_size(note_text, chunk_tokens, stride)
    else:
        raise ValueError(f"Unknown chunking strategy: {strategy!r}. Use 'section' or 'fixed'.")
