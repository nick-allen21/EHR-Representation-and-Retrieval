"""Async QA generation pipeline using gpt-4o.

Reads patient timelines, serializes each into a structured text block,
sends them to gpt-4o with a system prompt requesting temporal/cross-source
QA pairs, and writes the results with per-patient caching for resumability.

The pipeline is **append-only**: existing QA pairs (in the output file or
per-patient cache) are never overwritten. Running with a larger --limit
will only generate QA pairs for new (subject_id, hadm_id) pairs.

Usage:
    python -m Generation.generate_qa \
        --input data/processed/patient_timelines.json \
        --output data/generated/qa_pairs.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from pathlib import Path

import tiktoken
from dotenv import load_dotenv
from openai import AsyncOpenAI, RateLimitError, APIStatusError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "qa_generation.txt"
_ROOT_ENV = Path(__file__).resolve().parent.parent / ".env"
_MODEL = "gpt-4o"
_MAX_CONTEXT_TOKENS = 12_000
_MAX_RETRIES = 5
_BASE_BACKOFF = 2.0
_MAX_BACKOFF = 60.0
_CONCURRENCY = 10
_QA_PER_PATIENT = 5

_encoder: tiktoken.Encoding | None = None


def _get_encoder() -> tiktoken.Encoding:
    global _encoder
    if _encoder is None:
        _encoder = tiktoken.encoding_for_model(_MODEL)
    return _encoder


def count_tokens(text: str) -> int:
    return len(_get_encoder().encode(text))


def serialize_timeline(record: dict) -> str:
    """Convert a patient timeline dict into a structured text block for the LLM."""
    sections: list[str] = []

    demo = record.get("demographics", {})
    if demo:
        sections.append(
            "## Demographics\n"
            f"- Gender: {demo.get('gender', 'N/A')}\n"
            f"- Age at anchor: {demo.get('anchor_age', 'N/A')}\n"
            f"- Anchor year group: {demo.get('anchor_year_group', 'N/A')}\n"
            f"- Date of death: {demo.get('dod', 'N/A')}"
        )

    adm = record.get("admission", {})
    if adm:
        sections.append(
            "## Admission\n"
            f"- Admit time: {adm.get('admittime', 'N/A')}\n"
            f"- Discharge time: {adm.get('dischtime', 'N/A')}\n"
            f"- Admission type: {adm.get('admission_type', 'N/A')}\n"
            f"- Admission location: {adm.get('admission_location', 'N/A')}\n"
            f"- Discharge location: {adm.get('discharge_location', 'N/A')}\n"
            f"- Insurance: {adm.get('insurance', 'N/A')}\n"
            f"- Race: {adm.get('race', 'N/A')}"
        )

    events = record.get("events", [])
    diagnoses = [e for e in events if e.get("event_type") == "diagnosis"]
    if diagnoses:
        diag_lines = [f"- [{d.get('seq_num', '')}] {d.get('icd_code', '')} — {d.get('title', '')}" for d in diagnoses]
        sections.append("## Diagnoses\n" + "\n".join(diag_lines))

    clinical_events = [e for e in events if e.get("event_type") != "diagnosis"]
    if clinical_events:
        event_lines: list[str] = []
        for ev in clinical_events:
            etype = ev.get("event_type", "unknown")
            ts = ev.get("timestamp", "")
            if etype == "lab":
                line = f"[{ts}] LAB: {ev.get('label', '')} = {ev.get('value', '')} {ev.get('unit', '')} {ev.get('flag', '')}".strip()
            elif etype == "vital":
                line = f"[{ts}] VITAL: {ev.get('label', '')} = {ev.get('value', '')} {ev.get('unit', '')}".strip()
            elif etype == "medication":
                line = f"[{ts}] MED: {ev.get('drug', '')} {ev.get('dose', '')} {ev.get('dose_unit', '')} ({ev.get('route', '')}) until {ev.get('end_time', '')}".strip()
            elif etype == "procedure":
                line = f"[{ts}] PROCEDURE: {ev.get('icd_code', '')} — {ev.get('title', '')}".strip()
            else:
                line = f"[{ts}] {etype.upper()}: {ev}"
            event_lines.append(line)

        budget = _MAX_CONTEXT_TOKENS - 2000
        header = "## Clinical Events (chronological)\n"
        running_text = "\n".join(sections) + "\n\n" + header
        running_tokens = count_tokens(running_text)

        kept_lines: list[str] = []
        for line in event_lines:
            line_tokens = count_tokens(line + "\n")
            if running_tokens + line_tokens > budget:
                kept_lines.append(f"... [{len(event_lines) - len(kept_lines)} more events truncated] ...")
                break
            kept_lines.append(line)
            running_tokens += line_tokens

        sections.append(header + "\n".join(kept_lines))

    ds = record.get("discharge_summary", "")
    if ds:
        ds_tokens = count_tokens(ds)
        if count_tokens("\n\n".join(sections)) + ds_tokens > _MAX_CONTEXT_TOKENS:
            enc = _get_encoder()
            allowed = _MAX_CONTEXT_TOKENS - count_tokens("\n\n".join(sections)) - 200
            if allowed > 500:
                ds = enc.decode(enc.encode(ds)[:allowed])
                ds += "\n... [discharge summary truncated]"
        sections.append("## Discharge Summary\n" + ds)

    return "\n\n".join(sections)


def _load_system_prompt() -> str:
    return _PROMPT_PATH.read_text()


async def _generate_one(
    record: dict,
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    system_prompt: str,
    cache_dir: Path,
    existing_keys: set[tuple[int, int]],
) -> dict | None:
    """Generate QA pairs for a single patient, with caching and retry.

    Returns None if the pair already exists in the output file (skipped).
    """
    sid = record["subject_id"]
    hid = record["hadm_id"]

    if (sid, hid) in existing_keys:
        log.info("Already in output, skipping: %s_%s", sid, hid)
        return None

    cache_file = cache_dir / f"{sid}_{hid}.json"
    if cache_file.exists():
        log.info("Cache hit: %s_%s", sid, hid)
        return json.loads(cache_file.read_text())

    user_content = serialize_timeline(record)

    async with semaphore:
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                response = await client.chat.completions.create(
                    model=_MODEL,
                    temperature=0.7,
                    max_tokens=4096,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content},
                    ],
                )
                raw = response.choices[0].message.content.strip()
                if raw.startswith("```"):
                    raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
                qa_pairs = json.loads(raw)

                if isinstance(qa_pairs, dict):
                    for key in ("qa_pairs", "questions", "data", "items"):
                        if key in qa_pairs and isinstance(qa_pairs[key], list):
                            qa_pairs = qa_pairs[key]
                            break
                    else:
                        qa_pairs = [qa_pairs]

                if not isinstance(qa_pairs, list):
                    log.warning("Unexpected response shape for %s_%s, wrapping", sid, hid)
                    qa_pairs = [qa_pairs]

                result = {
                    "subject_id": sid,
                    "hadm_id": hid,
                    "qa_pairs": qa_pairs,
                    "model": _MODEL,
                    "tokens_prompt": response.usage.prompt_tokens if response.usage else None,
                    "tokens_completion": response.usage.completion_tokens if response.usage else None,
                }

                cache_file.write_text(json.dumps(result, indent=2))
                log.info("Generated %d QA pairs for %s_%s (attempt %d)", len(qa_pairs), sid, hid, attempt)
                return result

            except RateLimitError as exc:
                retry_after = getattr(exc, "retry_after", None)
                wait = retry_after if retry_after else min(_BASE_BACKOFF * (2 ** (attempt - 1)), _MAX_BACKOFF)
                log.warning("Rate limited on %s_%s, retrying in %.1fs (attempt %d/%d)", sid, hid, wait, attempt, _MAX_RETRIES)
                await asyncio.sleep(wait)

            except APIStatusError as exc:
                if exc.status_code >= 500:
                    wait = min(_BASE_BACKOFF * (2 ** (attempt - 1)), _MAX_BACKOFF)
                    log.warning("Server error %d on %s_%s, retrying in %.1fs (attempt %d/%d)", exc.status_code, sid, hid, wait, attempt, _MAX_RETRIES)
                    await asyncio.sleep(wait)
                else:
                    log.error("API error %d on %s_%s: %s", exc.status_code, sid, hid, exc.message)
                    return {"subject_id": sid, "hadm_id": hid, "qa_pairs": [], "error": str(exc)}

            except json.JSONDecodeError:
                log.warning("JSON parse error for %s_%s (attempt %d/%d)", sid, hid, attempt, _MAX_RETRIES)
                if attempt == _MAX_RETRIES:
                    return {"subject_id": sid, "hadm_id": hid, "qa_pairs": [], "error": "json_parse_error", "raw": raw}
                await asyncio.sleep(_BASE_BACKOFF)

    log.error("All retries exhausted for %s_%s", sid, hid)
    return {"subject_id": sid, "hadm_id": hid, "qa_pairs": [], "error": "max_retries_exhausted"}


def _load_existing_output(output_path: Path) -> dict[tuple[int, int], dict]:
    """Load previously generated QA pairs keyed by (subject_id, hadm_id)."""
    if not output_path.exists():
        return {}
    try:
        existing = json.loads(output_path.read_text())
    except (json.JSONDecodeError, OSError):
        log.warning("Could not parse existing output at %s, starting fresh", output_path)
        return {}
    return {
        (int(r["subject_id"]), int(r["hadm_id"])): r
        for r in existing
        if r.get("qa_pairs")
    }


async def run(input_path: Path, output_path: Path, cache_dir: Path) -> None:
    load_dotenv(_ROOT_ENV)
    client = AsyncOpenAI()
    system_prompt = _load_system_prompt()
    semaphore = asyncio.Semaphore(_CONCURRENCY)

    log.info("Loading timelines from %s", input_path)
    records = json.loads(input_path.read_text())
    log.info("Loaded %d patient records", len(records))

    existing = _load_existing_output(output_path)
    existing_keys = set(existing.keys())
    log.info("Found %d existing QA pairs in output (will not regenerate)", len(existing_keys))

    cache_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    tasks = [
        _generate_one(r, client, semaphore, system_prompt, cache_dir, existing_keys)
        for r in records
    ]
    raw_results = await asyncio.gather(*tasks)
    elapsed = time.time() - t0

    new_results = [r for r in raw_results if r is not None]
    skipped = sum(1 for r in raw_results if r is None)

    merged: dict[tuple[int, int], dict] = dict(existing)
    for r in new_results:
        key = (int(r["subject_id"]), int(r["hadm_id"]))
        if key not in merged:
            merged[key] = r

    all_results = list(merged.values())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(all_results, indent=2))

    total_qa = sum(len(r.get("qa_pairs", [])) for r in all_results)
    errors = sum(1 for r in new_results if r.get("error"))
    cache_hits = sum(1 for r in new_results if not r.get("error") and not r.get("tokens_prompt"))
    api_calls = len(new_results) - cache_hits - errors

    log.info("=" * 50)
    log.info("QA Generation Complete")
    log.info("  Total in output    : %d records, %d QA pairs", len(all_results), total_qa)
    log.info("  Previously existed : %d (preserved)", skipped)
    log.info("  New API calls      : %d", api_calls)
    log.info("  New from cache     : %d", cache_hits)
    log.info("  Errors             : %d", errors)
    log.info("  Wall time          : %.1fs", elapsed)
    log.info("  Output             : %s", output_path)
    log.info("=" * 50)


def main():
    parser = argparse.ArgumentParser(description="Generate QA pairs from patient timelines using gpt-4o")
    parser.add_argument("--input", type=Path, default=Path("data/processed/patient_timelines.json"))
    parser.add_argument("--output", type=Path, default=Path("data/generated/qa_pairs.json"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/generated/cache"))
    args = parser.parse_args()
    asyncio.run(run(args.input, args.output, args.cache_dir))


if __name__ == "__main__":
    main()
