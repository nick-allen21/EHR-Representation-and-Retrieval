"""CLI entrypoint — run the full extraction + merge pipeline.

Usage:
    python -m Preprocess.run_pipeline [--format parquet|json] [--limit N]
"""

import argparse
import datetime
import json
from pathlib import Path

import pandas as pd

from Preprocess.bigquery_client import load_config, get_client
from Preprocess.build_timeline import build_patient_timelines


def _json_serializer(obj):
    """Handle pandas Timestamps, dates, and other non-serializable types."""
    if isinstance(obj, (pd.Timestamp, datetime.datetime)):
        return obj.isoformat()
    if isinstance(obj, datetime.date):
        return obj.isoformat()
    if isinstance(obj, datetime.timedelta):
        return str(obj)
    if hasattr(obj, "item"):
        return obj.item()
    if pd.isna(obj):
        return None
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def main():
    parser = argparse.ArgumentParser(
        description="Build longitudinal patient records from EHR-DS-QA + MIMIC-IV"
    )
    parser.add_argument(
        "--format",
        choices=["parquet", "json"],
        default=None,
        help="Output format (default: from config.yaml)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N QA rows (useful for testing)",
    )
    args = parser.parse_args()

    config = load_config()
    client = get_client(config)

    out_format = args.format or config["pipeline"]["output_format"]
    out_dir = Path(config["data"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    records = build_patient_timelines(
        config=config, client=client, limit=args.limit
    )

    if out_format == "json":
        out_path = out_dir / "patient_timelines.json"
        with open(out_path, "w") as f:
            json.dump(records, f, indent=2, default=_json_serializer)
    else:
        out_path = out_dir / "patient_timelines.parquet"
        flat = []
        for r in records:
            flat.append({
                "subject_id": r["subject_id"],
                "hadm_id": r["hadm_id"],
                "note_id": r["note_id"],
                "demographics": json.dumps(r["demographics"], default=_json_serializer),
                "admission": json.dumps(r["admission"], default=_json_serializer),
                "discharge_summary": r["discharge_summary"],
                "events": json.dumps(r["events"], default=_json_serializer),
            })
        pd.DataFrame(flat).to_parquet(out_path, index=False)

    print(f"Wrote {len(records)} records to {out_path}")


if __name__ == "__main__":
    main()
