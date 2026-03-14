"""Create submission.zip for CS229 final project.

Includes only source code, config, and documentation — no data, models,
caches, or library files.  Target: < 5 MB.

Usage:
    python create_submission.py
"""

import os
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT = PROJECT_ROOT / "submission.zip"

INCLUDE_FILES = [
    # Preprocess
    "Preprocess/__init__.py",
    "Preprocess/bigquery_client.py",
    "Preprocess/extract_notes.py",
    "Preprocess/extract_structured.py",
    "Preprocess/build_timeline.py",
    "Preprocess/run_pipeline.py",

    # Generation
    "Generation/__init__.py",
    "Generation/generate_qa.py",
    "Generation/prompts/qa_generation.txt",

    # Logreg
    "Logreg/__init__.py",
    "Logreg/data_loader.py",
    "Logreg/chunker.py",
    "Logreg/labeler.py",
    "Logreg/features.py",
    "Logreg/train.py",
    "Logreg/selector.py",
    "Logreg/run.py",

    # Evaluation
    "Evaluation/__init__.py",
    "Evaluation/context_builders.py",
    "Evaluation/llm_runner.py",
    "Evaluation/hf_runner.py",
    "Evaluation/scoring.py",
    "Evaluation/llm_judge.py",
    "Evaluation/run_evaluation.py",
    "Evaluation/analysis.py",

    # Scripts
    "scripts/generate_plots.py",

    # Config and docs
    "config/config.yaml",
    "requirements.txt",
    "README.md",
]


def main():
    missing = [f for f in INCLUDE_FILES if not (PROJECT_ROOT / f).exists()]
    if missing:
        print("WARNING — missing files (will be skipped):")
        for f in missing:
            print(f"  {f}")

    with zipfile.ZipFile(OUTPUT, "w", zipfile.ZIP_DEFLATED) as zf:
        for relpath in INCLUDE_FILES:
            full = PROJECT_ROOT / relpath
            if full.exists():
                zf.write(full, relpath)

    size_mb = OUTPUT.stat().st_size / (1024 * 1024)
    print(f"\nCreated {OUTPUT.name}  ({size_mb:.2f} MB)")
    print(f"  Files: {len(INCLUDE_FILES) - len(missing)}")
    if size_mb > 5.0:
        print("  WARNING: zip exceeds 5 MB limit!")
    else:
        print("  OK — under 5 MB limit.")

    print("\nContents:")
    with zipfile.ZipFile(OUTPUT, "r") as zf:
        for info in zf.infolist():
            kb = info.file_size / 1024
            print(f"  {info.filename:<55s} {kb:6.1f} KB")


if __name__ == "__main__":
    main()
