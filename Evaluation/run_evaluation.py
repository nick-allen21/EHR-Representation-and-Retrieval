"""CLI orchestrator for the downstream LLM evaluation.

Loads patient timelines + QA pairs, builds context with each retrieval
strategy, sends prompts to a frozen LLM (default: o4-mini), scores
responses, and saves results.

Usage
-----
# Single strategy
python -m Evaluation.run_evaluation --method discharge_only --limit 200

# All strategies (default K=5 for chunk-based methods)
python -m Evaluation.run_evaluation --method all --limit 200

# Specific K values
python -m Evaluation.run_evaluation --method bm25 --k 10 --limit 200

# Specific recency N
python -m Evaluation.run_evaluation --method recency --n 50 --limit 200
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from pathlib import Path

from Logreg.data_loader import (
    load_timelines,
    load_qa_pairs,
    serialize_events,
    QA_GENERATED,
    QA_EHR_DS_QA,
)
from Evaluation.context_builders import STRATEGIES, SYSTEM_PROMPT
from Evaluation.llm_runner import LLMRunner
from Evaluation.scoring import score_pair


def _make_runner(args: argparse.Namespace):
    """Return an LLMRunner or HFRunner depending on the model name.

    HuggingFace model IDs always contain a '/' (e.g. 'meta-llama/Meta-Llama-3-8B-Instruct').
    OpenAI model names never do (e.g. 'o4-mini', 'gpt-4o-mini').
    """
    if "/" in args.model:
        from Evaluation.hf_runner import HFRunner
        return HFRunner(
            model_id=args.model,
            cache_dir=args.cache_dir,
            batch_size=args.hf_batch_size,
        )
    return LLMRunner(
        model=args.model,
        cache_dir=args.cache_dir,
        concurrency=args.concurrency,
    )

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

_DEFAULT_TIMELINES = Path("data/processed/patient_timelines.json")
_RESULTS_DIR = Path("data/results")


# ── Data loading ──────────────────────────────────────────────────────────────

def _build_eval_records(
    timelines_path: Path,
    qa_path: Path,
    limit: int | None = None,
) -> list[dict]:
    """Join timelines with QA pairs, keeping full timeline fields.

    Returns enriched records with: subject_id, hadm_id, discharge_summary,
    demographics, admission, events, note_text, qa_pairs.
    """
    timelines = load_timelines(timelines_path)
    qa_records = load_qa_pairs(qa_path)

    qa_index: dict[tuple[int, int], list[dict]] = {}
    for rec in qa_records:
        key = (int(rec["subject_id"]), int(rec["hadm_id"]))
        pairs = [
            qa for qa in (rec.get("qa_pairs") or [])
            if qa.get("question") and qa.get("answer")
            and qa.get("correct", True) is not False
        ]
        if pairs:
            qa_index[key] = pairs

    records: list[dict] = []
    for t in timelines:
        key = (int(t["subject_id"]), int(t["hadm_id"]))
        qa_pairs = qa_index.get(key)
        ds = (t.get("discharge_summary") or "").strip()
        if not qa_pairs or not ds:
            continue

        events = t.get("events", [])
        events_text = serialize_events(events)
        note_text = f"{ds}\n\n{events_text}" if events_text else ds

        records.append({
            "subject_id":        int(t["subject_id"]),
            "hadm_id":           int(t["hadm_id"]),
            "discharge_summary": ds,
            "demographics":      t.get("demographics", {}),
            "admission":         t.get("admission", {}),
            "events":            events,
            "note_text":         note_text,
            "qa_pairs":          qa_pairs,
        })

        if limit and len(records) >= limit:
            break

    log.info(
        "Loaded %d evaluation records (%d QA pairs total)",
        len(records),
        sum(len(r["qa_pairs"]) for r in records),
    )
    return records


# ── Running a single strategy ─────────────────────────────────────────────────

async def _run_strategy(
    method: str,
    records: list[dict],
    runner: LLMRunner,
    token_budget: int,
    **strategy_kwargs,
) -> list[dict]:
    """Run one retrieval strategy on all records and return scored results."""
    builder = STRATEGIES[method]
    items: list[dict] = []

    for rec in records:
        for qa in rec["qa_pairs"]:
            prompt_info = builder(
                rec,
                qa["question"],
                token_budget=token_budget,
                **strategy_kwargs,
            )
            items.append({
                "subject_id": rec["subject_id"],
                "hadm_id":    rec["hadm_id"],
                "question":   qa["question"],
                "gold_answer": qa["answer"],
                "difficulty":  qa.get("difficulty", ""),
                "source_types": qa.get("source_types", []),
                **prompt_info,
            })

    log.info(
        "Running %s: %d prompts → %s",
        method, len(items), runner.model,
    )
    t0 = time.time()
    results = await runner.generate_batch(
        items, system_prompt=SYSTEM_PROMPT,
    )
    elapsed = time.time() - t0

    for r in results:
        scores = score_pair(r.get("answer", ""), r["gold_answer"])
        r.update(scores)

    total_prompt = sum(r.get("prompt_tokens", 0) for r in results)
    total_comp = sum(r.get("completion_tokens", 0) for r in results)
    mean_f1 = sum(r["token_f1"] for r in results) / max(len(results), 1)
    mean_rouge = sum(r["rouge_l_f1"] for r in results) / max(len(results), 1)

    log.info(
        "  %s done in %.1fs | %d prompts | "
        "tokens: %d prompt + %d completion | "
        "mean token-F1: %.3f | mean ROUGE-L: %.3f",
        method, elapsed, len(results),
        total_prompt, total_comp, mean_f1, mean_rouge,
    )
    return results


# ── Save results ──────────────────────────────────────────────────────────────

def _save_results(results: list[dict], method: str, output_dir: Path, model: str = "unknown") -> Path:
    """Write results to a JSON file, stripping the full prompt to save space."""
    output_dir.mkdir(parents=True, exist_ok=True)
    tag = results[0].get("method", method) if results else method
    model_slug = model.replace("/", "--")
    path = output_dir / f"{model_slug}__{tag}.json"

    slim = []
    for r in results:
        row = dict(r)
        row.pop("user_prompt", None)
        slim.append(row)

    path.write_text(json.dumps(slim, indent=2))
    log.info("  Results saved to %s", path)
    return path


# ── Main ──────────────────────────────────────────────────────────────────────

async def run(args: argparse.Namespace) -> None:
    records = _build_eval_records(
        timelines_path=args.timelines,
        qa_path=args.qa_data,
        limit=args.limit,
    )
    if not records:
        log.error("No evaluation records found. Exiting.")
        return

    runner = _make_runner(args)

    selector = None
    if args.method in ("learned", "all"):
        from Logreg.selector import ChunkSelector
        model_dir = Path(args.model_dir)
        if (model_dir / "model.pkl").exists():
            selector = ChunkSelector.load(model_dir)
            log.info("Loaded learned selector from %s", model_dir)
        else:
            log.warning(
                "Trained model not found at %s — skipping learned strategy",
                model_dir,
            )

    methods = list(STRATEGIES.keys()) if args.method == "all" else [args.method]

    strategy_kwargs: dict[str, dict] = {
        "discharge_only": {},
        "full_context":   {},
        "recency":        {"n": args.n},
        "bm25":           {"k": args.k},
        "semantic_rag":   {"k": args.k},
        "learned":        {"k": args.k, "selector": selector},
    }

    for method in methods:
        if method == "learned" and selector is None:
            log.warning("Skipping learned (no trained model)")
            continue

        results = await _run_strategy(
            method=method,
            records=records,
            runner=runner,
            token_budget=args.token_budget,
            **strategy_kwargs.get(method, {}),
        )
        _save_results(results, method, Path(args.output_dir), model=args.model)

    stats = runner.stats
    log.info(
        "Runner stats: %d cache hits, %d API calls, %d errors",
        stats["cache_hits"], stats["api_calls"], stats["errors"],
    )


def main():
    p = argparse.ArgumentParser(
        description="Run downstream LLM evaluation across retrieval strategies",
    )
    p.add_argument(
        "--method", default="all",
        choices=list(STRATEGIES.keys()) + ["all"],
        help="Retrieval strategy to evaluate (default: all)",
    )
    p.add_argument("--limit", type=int, default=None, help="Max patient records to evaluate")
    p.add_argument("--k", type=int, default=5, help="Top-K for chunk-based strategies")
    p.add_argument("--n", type=int, default=25, help="N most recent events for recency strategy")
    p.add_argument("--token-budget", type=int, default=4096, help="Max context tokens per prompt")
    p.add_argument("--model", default="o4-mini", help="LLM model for QA inference")
    p.add_argument(
        "--timelines", type=Path, default=_DEFAULT_TIMELINES,
        help="Path to patient_timelines.json",
    )
    p.add_argument(
        "--qa-data", type=Path, default=QA_GENERATED,
        help="Path to QA pairs JSON",
    )
    p.add_argument("--model-dir", default="data/models/logreg", help="Trained logreg model directory")
    p.add_argument("--output-dir", default=str(_RESULTS_DIR), help="Directory for result files")
    p.add_argument("--cache-dir", default="data/results/cache", help="LLM response cache directory")
    p.add_argument("--concurrency", type=int, default=10, help="Max concurrent API calls (OpenAI only)")
    p.add_argument("--hf-batch-size", type=int, default=4, help="GPU batch size for HuggingFace models")
    args = p.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
