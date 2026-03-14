"""Merge EHR-DS-QA data with MIMIC-IV structured + note data into
longitudinal patient records.

Each output record represents one hospital admission and Has the foloiwng tibgs:
- patient demographics
- admission metadata
- the full discharge summary text
- temporally ordered clinical events (labs, vitals, medications, procedures, diagnoses)

QA pairs are generated separately by the generation pipeline (see Generation/).
"""

from pathlib import Path

import pandas as pd
from google.cloud import bigquery

from Preprocess.bigquery_client import load_config, get_client
from Preprocess.extract_notes import get_discharge_notes
from Preprocess.extract_structured import (
    get_admissions,
    get_diagnoses,
    get_labs,
    get_patients,
    get_prescriptions,
    get_procedures,
    get_vitals,)


def load_qa_dataset(config: dict | None = None) -> pd.DataFrame:
    """Load the local EHR-DS-QA CSV for subject_id / hadm_id mappings."""
    config = config or load_config()
    csv_path = Path(config["data"]["ehr_ds_qa_csv"])
    return pd.read_csv(csv_path)


def _events_from_labs(df: pd.DataFrame) -> list[dict]:
    events = []
    for _, row in df.iterrows():
        events.append({
            "event_type": "lab",
            "timestamp": str(row.get("charttime", "")),
            "label": row.get("lab_label", ""),
            "value": row.get("value", ""),
            "valuenum": row.get("valuenum"),
            "unit": row.get("valueuom", ""),
            "flag": row.get("flag", ""),
        })
    return events


def _events_from_vitals(df: pd.DataFrame) -> list[dict]:
    events = []
    for _, row in df.iterrows():
        events.append({
            "event_type": "vital",
            "timestamp": str(row.get("charttime", "")),
            "label": row.get("vital_label", ""),
            "value": row.get("value", ""),
            "valuenum": row.get("valuenum"),
            "unit": row.get("valueuom", ""),
        })
    return events


def _events_from_prescriptions(df: pd.DataFrame) -> list[dict]:
    events = []
    for _, row in df.iterrows():
        events.append({
            "event_type": "medication",
            "timestamp": str(row.get("starttime", "")),
            "end_time": str(row.get("stoptime", "")),
            "drug": row.get("drug", ""),
            "dose": row.get("dose_val_rx", ""),
            "dose_unit": row.get("dose_unit_rx", ""),
            "route": row.get("route", ""),
        })
    return events


def _events_from_procedures(df: pd.DataFrame) -> list[dict]:
    events = []
    for _, row in df.iterrows():
        events.append({
            "event_type": "procedure",
            "timestamp": str(row.get("chartdate", "")),
            "icd_code": row.get("icd_code", ""),
            "title": row.get("procedure_title", ""),
        })
    return events


def _events_from_diagnoses(df: pd.DataFrame) -> list[dict]:
    events = []
    for _, row in df.iterrows():
        events.append({
            "event_type": "diagnosis",
            "timestamp": "",
            "seq_num": int(row.get("seq_num", 0)),
            "icd_code": row.get("icd_code", ""),
            "title": row.get("diagnosis_title", ""),
        })
    return events


def build_patient_timelines(
    config: dict | None = None,
    client: bigquery.Client | None = None,
    limit: int | None = None,
) -> list[dict]:
    """Main entry point: build longitudinal records for every admission in the
    QA dataset.

    Args:
        limit: If set, only process the first N QA rows (applied before
               querying BigQuery so that test runs are fast).

    Returns a list of dicts, one per (subject_id, hadm_id) pair.
    """
    config = config or load_config()
    client = client or get_client(config)
    batch_size = config["pipeline"]["batch_size"]

    # 1. Load local QA data
    qa_df = load_qa_dataset(config)

    if limit is not None:
        qa_df = qa_df.head(limit)

    subject_ids = qa_df["subject_id"].unique().tolist()
    hadm_ids = qa_df["hadm_id"].unique().tolist()

    print(f"Loaded {len(qa_df)} QA rows — "
          f"{len(subject_ids)} patients, {len(hadm_ids)} admissions")

    # 2. Fetch all MIMIC-IV data from BigQuery
    print("Fetching discharge notes …")
    notes_df = get_discharge_notes(
        subject_ids, hadm_ids, client=client, config=config, batch_size=batch_size
    )

    print("Fetching patient demographics …")
    patients_df = get_patients(subject_ids, client=client, config=config)

    print("Fetching admissions …")
    admissions_df = get_admissions(hadm_ids, client=client, config=config)

    print("Fetching diagnoses …")
    diagnoses_df = get_diagnoses(hadm_ids, client=client, config=config)

    print("Fetching procedures …")
    procedures_df = get_procedures(hadm_ids, client=client, config=config)

    print("Fetching labs …")
    labs_df = get_labs(
        subject_ids, hadm_ids, client=client, config=config
    )

    print("Fetching prescriptions …")
    prescriptions_df = get_prescriptions(hadm_ids, client=client, config=config)

    print("Fetching vitals …")
    vitals_df = get_vitals(
        subject_ids, hadm_ids, client=client, config=config
    )

    # 3. Index auxiliary DataFrames by admission
    patients_map = {
        row["subject_id"]: row.to_dict()
        for _, row in patients_df.iterrows()
    } if not patients_df.empty else {}

    admissions_map = {
        row["hadm_id"]: row.to_dict()
        for _, row in admissions_df.iterrows()
    } if not admissions_df.empty else {}

    notes_map: dict[int, str] = {}
    if not notes_df.empty:
        for _, row in notes_df.iterrows():
            notes_map[row["hadm_id"]] = row["text"]

    # 4. Build one record per admission
    records: list[dict] = []

    for _, qa_row in qa_df.iterrows():
        sid = int(qa_row["subject_id"])
        hid = int(qa_row["hadm_id"])

        # Collect events for this admission
        events: list[dict] = []

        if not labs_df.empty:
            events.extend(
                _events_from_labs(labs_df[labs_df["hadm_id"] == hid])
            )
        if not vitals_df.empty:
            events.extend(
                _events_from_vitals(vitals_df[vitals_df["hadm_id"] == hid])
            )
        if not prescriptions_df.empty:
            events.extend(
                _events_from_prescriptions(
                    prescriptions_df[prescriptions_df["hadm_id"] == hid]
                )
            )
        if not procedures_df.empty:
            events.extend(
                _events_from_procedures(
                    procedures_df[procedures_df["hadm_id"] == hid]
                )
            )
        if not diagnoses_df.empty:
            events.extend(
                _events_from_diagnoses(
                    diagnoses_df[diagnoses_df["hadm_id"] == hid]
                )
            )

        events.sort(key=lambda e: e.get("timestamp", ""))

        record = {
            "subject_id": sid,
            "hadm_id": hid,
            "note_id": qa_row.get("note_id", ""),
            "demographics": patients_map.get(sid, {}),
            "admission": admissions_map.get(hid, {}),
            "discharge_summary": notes_map.get(hid, ""),
            "events": events,
        }
        records.append(record)

    print(f"Built {len(records)} longitudinal patient records")
    return records
