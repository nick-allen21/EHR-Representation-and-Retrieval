"""CLI entry point for the Logreg chunk selector.

Subcommands:

  train     — build dataset, train L1 logistic regression, save artifacts
  evaluate  — compute mean Recall@K over a set of records using a saved model
  select    — interactively select top-K chunks for a question + note (demo)

Examples:

  # Train on all available records
  python -m Logreg.run train

  # Train on a smaller subset for a quick smoke-test
  python -m Logreg.run train --limit 100 --no-embeddings

  # Evaluate mean Recall@5 on the full dataset
  python -m Logreg.run evaluate --K 5

  # Select top-3 chunks for a question using a saved note file
  python -m Logreg.run select --question "What medications were given at discharge?" \\
                               --note-file data/example_note.txt --K 3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_DEFAULT_TIMELINES = "data/processed/patient_timelines.json"
_DEFAULT_QA_DATA   = "data/generated/qa_pairs.json"
_DEFAULT_MODEL_DIR = "data/models/logreg"

_EHR_DS_QA = "data/physionet.org/files/ehr-ds-qa/1.0.0/mimic_iv_note_qa.json"


# ── train ─────────────────────────────────────────────────────────────────────

def cmd_train(args: argparse.Namespace) -> None:
    from Logreg.data_loader import load_and_merge
    from Logreg.train import run_training

    include_events = not args.discharge_only
    print(f"Loading data …\n  timelines : {args.timelines}\n  qa data   : {args.qa_data}")
    print(f"  mode      : {'discharge only' if args.discharge_only else 'discharge + structured events'}")
    records = load_and_merge(args.timelines, args.qa_data, include_events=include_events)

    if args.limit:
        records = records[: args.limit]
        print(f"Using first {len(records)} records (--limit {args.limit})")

    metrics = run_training(
        records,
        output_dir=args.model_dir,
        strategy=args.strategy,
        C=args.C,
        label_threshold=args.label_threshold,
        use_embeddings=not args.no_embeddings,
    )

    print("\n── Validation metrics ──")
    print(json.dumps(metrics["val"], indent=2))


# ── evaluate ──────────────────────────────────────────────────────────────────

def cmd_evaluate(args: argparse.Namespace) -> None:
    from Logreg.data_loader import load_and_merge
    from Logreg.selector import ChunkSelector

    include_events = not args.discharge_only
    print(f"Loading selector from {args.model_dir} …")
    selector = ChunkSelector.load(args.model_dir, strategy=args.strategy)

    print(f"Loading data …")
    records = load_and_merge(args.timelines, args.qa_data, include_events=include_events)
    if args.limit:
        records = records[: args.limit]

    recall_values: list[float] = []
    for rec in records:
        for qa in rec["qa_pairs"]:
            question = qa.get("question", "")
            answer   = qa.get("answer",   "")
            if not question or not answer:
                continue
            r = selector.recall_at_k(
                question, rec["note_text"], answer,
                K=args.K,
                label_threshold=args.label_threshold,
            )
            recall_values.append(r)

    mean_recall = sum(recall_values) / len(recall_values) if recall_values else 0.0
    result = {
        "mean_recall_at_k": mean_recall,
        "K":                args.K,
        "n_qa_pairs":       len(recall_values),
        "strategy":         args.strategy,
    }
    print(f"\nMean Recall@{args.K}: {mean_recall:.4f}  (over {len(recall_values)} QA pairs)")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Results saved to {args.output}")


# ── select ────────────────────────────────────────────────────────────────────

def cmd_select(args: argparse.Namespace) -> None:
    from Logreg.selector import ChunkSelector

    selector = ChunkSelector.load(args.model_dir, strategy=args.strategy)

    note_text = args.note or ""
    if args.note_file:
        note_text = Path(args.note_file).read_text()

    if not note_text.strip():
        print("ERROR: provide note text via --note or --note-file", file=sys.stderr)
        sys.exit(1)

    selected = selector.select(question=args.question, note_text=note_text, K=args.K)

    print(f"\nTop-{args.K} selected chunks for:\n  '{args.question}'\n")
    for rank, chunk in enumerate(selected, 1):
        print(f"[{rank}] score={chunk['score']:.4f}  section={chunk['section']}  "
              f"position={chunk['position']:.2f}")
        print(chunk["text"][:400])
        print("─" * 60)


# ── main ──────────────────────────────────────────────────────────────────────

def _add_data_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--timelines", default=_DEFAULT_TIMELINES,
                   help="Path to patient_timelines.json (from Preprocess.run_pipeline)")
    p.add_argument("--qa-data",   default=_DEFAULT_QA_DATA,
                   help=f"Path to QA pairs JSON. Default: generated (gpt-4o). "
                        f"For EHR-DS-QA use: {_EHR_DS_QA}")
    p.add_argument("--model-dir", default=_DEFAULT_MODEL_DIR,
                   help="Directory to save/load model artifacts")
    p.add_argument("--strategy",  default="section", choices=["section", "fixed"],
                   help="Chunking strategy (default: section)")
    p.add_argument("--discharge-only", action="store_true",
                   help="Use only the discharge summary as note text (no structured events). "
                        "Cross-source QA pairs will be filtered out.")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m Logreg.run",
        description="L1 logistic regression chunk selector for EHR discharge summaries",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # train
    p_train = sub.add_parser("train", help="Train the chunk selector")
    _add_data_args(p_train)
    p_train.add_argument("--C",               type=float, default=1.0,
                         help="Inverse L1 regularization strength (smaller → sparser model)")
    p_train.add_argument("--label-threshold", type=float, default=0.15,
                         help="Min token F1 overlap for a chunk to be labeled positive")
    p_train.add_argument("--no-embeddings",   action="store_true",
                         help="Disable sentence-transformer embeddings (faster, ~5%% lower accuracy)")
    p_train.add_argument("--limit",           type=int, default=None,
                         help="Only use the first N patient records (for debugging)")

    # evaluate
    p_eval = sub.add_parser("evaluate", help="Compute Recall@K on the full dataset")
    _add_data_args(p_eval)
    p_eval.add_argument("--K",               type=int,   default=5,
                        help="Number of chunks to select per question")
    p_eval.add_argument("--label-threshold", type=float, default=0.15)
    p_eval.add_argument("--output",          type=str,   default=None,
                        help="Save results JSON to this path")
    p_eval.add_argument("--limit",           type=int,   default=None)

    # select
    p_sel = sub.add_parser("select", help="Select top-K chunks for a question (demo)")
    p_sel.add_argument("--question",  required=True, help="Clinical question string")
    p_sel.add_argument("--note",      default="",    help="Note text (inline string)")
    p_sel.add_argument("--note-file", default=None,  help="Path to a plain-text note file")
    p_sel.add_argument("--model-dir", default=_DEFAULT_MODEL_DIR)
    p_sel.add_argument("--strategy",  default="section", choices=["section", "fixed"])
    p_sel.add_argument("--K",         type=int, default=5)

    args = parser.parse_args()

    if args.command == "train":
        cmd_train(args)
    elif args.command == "evaluate":
        cmd_evaluate(args)
    elif args.command == "select":
        cmd_select(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
