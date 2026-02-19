"""Fetch full discharge summary text from MIMIC-IV-Note on BigQuery."""

import pandas as pd
from google.cloud import bigquery

from Preprocess.bigquery_client import load_config, get_client, build_id_filter


def get_discharge_notes(
    subject_ids: list[int],
    hadm_ids: list[int],
    client: bigquery.Client | None = None,
    config: dict | None = None,
    batch_size: int = 500,
) -> pd.DataFrame:
    """Retrieve discharge summaries for a set of (subject_id, hadm_id) pairs.

    Returns a DataFrame with columns:
        subject_id, hadm_id, note_id, charttime, text
    """
    if config is None:
        config = load_config()
    if client is None:
        client = get_client(config)

    table = config["tables"]["discharge"]

    unique_subjects = list(set(subject_ids))
    frames: list[pd.DataFrame] = []

    for i in range(0, len(unique_subjects), batch_size):
        batch = unique_subjects[i : i + batch_size]
        id_filter = build_id_filter("subject_id", batch)

        sql = f"""
        SELECT
            subject_id,
            hadm_id,
            note_id,
            charttime,
            text
        FROM `{table}`
        WHERE {id_filter}
        ORDER BY subject_id, charttime
        """
        df = client.query(sql).to_dataframe()
        frames.append(df)

    if not frames:
        return pd.DataFrame(
            columns=["subject_id", "hadm_id", "note_id", "charttime", "text"]
        )

    result = pd.concat(frames, ignore_index=True)

    # Filter to only the hadm_ids present in our QA dataset
    valid_hadms = set(hadm_ids)
    result = result[result["hadm_id"].isin(valid_hadms)].reset_index(drop=True)

    return result
