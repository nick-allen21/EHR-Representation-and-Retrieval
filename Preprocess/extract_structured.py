"""Fetch structured clinical data from MIMIC-IV on BigQuery.

Each public function takes a list of subject/hadm IDs, queries BigQuery in
batches, and returns a tidy DataFrame with a timestamp column for temporal
ordering.
"""

import pandas as pd
from google.cloud import bigquery

from Preprocess.bigquery_client import load_config, get_client, build_id_filter


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _batched_query(
    sql_template: str,
    id_column: str,
    id_values: list[int],
    client: bigquery.Client,
    batch_size: int = 500,
) -> pd.DataFrame:
    """Run *sql_template* in batches over *id_values*.

    ``sql_template`` must contain a ``{id_filter}`` placeholder.
    """
    unique_ids = list(set(id_values))
    frames: list[pd.DataFrame] = []

    for i in range(0, len(unique_ids), batch_size):
        batch = unique_ids[i : i + batch_size]
        id_filter = build_id_filter(id_column, batch)
        sql = sql_template.format(id_filter=id_filter)
        frames.append(client.query(sql).to_dataframe())

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Public extractors
# ---------------------------------------------------------------------------

def get_patients(
    subject_ids: list[int],
    client: bigquery.Client | None = None,
    config: dict | None = None,
) -> pd.DataFrame:
    """Demographics from ``mimiciv_hosp.patients``."""
    config = config or load_config()
    client = client or get_client(config)
    table = config["tables"]["patients"]

    sql = f"""
    SELECT subject_id, gender, anchor_age, anchor_year, anchor_year_group, dod
    FROM `{table}`
    WHERE {{id_filter}}
    """
    return _batched_query(sql, "subject_id", subject_ids, client)


def get_admissions(
    hadm_ids: list[int],
    client: bigquery.Client | None = None,
    config: dict | None = None,
) -> pd.DataFrame:
    """Admission-level details from ``mimiciv_hosp.admissions``."""
    config = config or load_config()
    client = client or get_client(config)
    table = config["tables"]["admissions"]

    sql = f"""
    SELECT
        subject_id, hadm_id,
        admittime, dischtime, deathtime,
        admission_type, admit_provider_id,
        admission_location, discharge_location,
        insurance, language, marital_status, race
    FROM `{table}`
    WHERE {{id_filter}}
    ORDER BY admittime
    """
    return _batched_query(sql, "hadm_id", hadm_ids, client)


def get_diagnoses(
    hadm_ids: list[int],
    client: bigquery.Client | None = None,
    config: dict | None = None,
) -> pd.DataFrame:
    """ICD diagnoses with long titles from ``mimiciv_hosp.diagnoses_icd``."""
    config = config or load_config()
    client = client or get_client(config)
    dx = config["tables"]["diagnoses_icd"]
    d_dx = config["tables"]["d_icd_diagnoses"]

    sql = f"""
    SELECT
        d.subject_id, d.hadm_id, d.seq_num,
        d.icd_code, d.icd_version,
        dd.long_title AS diagnosis_title
    FROM `{dx}` d
    JOIN `{d_dx}` dd
        ON d.icd_code = dd.icd_code AND d.icd_version = dd.icd_version
    WHERE {{id_filter}}
    ORDER BY d.hadm_id, d.seq_num
    """
    return _batched_query(sql, "d.hadm_id", hadm_ids, client)


def get_procedures(
    hadm_ids: list[int],
    client: bigquery.Client | None = None,
    config: dict | None = None,
) -> pd.DataFrame:
    """ICD procedures with long titles from ``mimiciv_hosp.procedures_icd``."""
    config = config or load_config()
    client = client or get_client(config)
    px = config["tables"]["procedures_icd"]
    d_px = config["tables"]["d_icd_procedures"]

    sql = f"""
    SELECT
        p.subject_id, p.hadm_id, p.seq_num,
        p.icd_code, p.icd_version, p.chartdate,
        dp.long_title AS procedure_title
    FROM `{px}` p
    JOIN `{d_px}` dp
        ON p.icd_code = dp.icd_code AND p.icd_version = dp.icd_version
    WHERE {{id_filter}}
    ORDER BY p.hadm_id, p.chartdate
    """
    return _batched_query(sql, "p.hadm_id", hadm_ids, client)


def get_labs(
    subject_ids: list[int],
    hadm_ids: list[int],
    client: bigquery.Client | None = None,
    config: dict | None = None,
) -> pd.DataFrame:
    """Lab results from ``mimiciv_hosp.labevents`` with item labels."""
    config = config or load_config()
    client = client or get_client(config)
    lab = config["tables"]["labevents"]
    d_lab = config["tables"]["d_labitems"]

    hadm_set = set(hadm_ids)

    sql = f"""
    SELECT
        le.subject_id, le.hadm_id, le.specimen_id,
        le.itemid, dl.label AS lab_label,
        le.charttime,
        le.value, le.valuenum, le.valueuom,
        le.ref_range_lower, le.ref_range_upper, le.flag
    FROM `{lab}` le
    JOIN `{d_lab}` dl ON le.itemid = dl.itemid
    WHERE {{id_filter}}
    ORDER BY le.subject_id, le.charttime
    """
    df = _batched_query(sql, "le.subject_id", subject_ids, client)
    if not df.empty:
        df = df[df["hadm_id"].isin(hadm_set)].reset_index(drop=True)
    return df


def get_prescriptions(
    hadm_ids: list[int],
    client: bigquery.Client | None = None,
    config: dict | None = None,
) -> pd.DataFrame:
    """Medication prescriptions from ``mimiciv_hosp.prescriptions``."""
    config = config or load_config()
    client = client or get_client(config)
    table = config["tables"]["prescriptions"]

    sql = f"""
    SELECT
        subject_id, hadm_id, pharmacy_id,
        starttime, stoptime,
        drug_type, drug, gsn, ndc,
        prod_strength, form_rx, dose_val_rx, dose_unit_rx,
        form_val_disp, form_unit_disp,
        route
    FROM `{table}`
    WHERE {{id_filter}}
    ORDER BY hadm_id, starttime
    """
    return _batched_query(sql, "hadm_id", hadm_ids, client)


def get_vitals(
    subject_ids: list[int],
    hadm_ids: list[int],
    client: bigquery.Client | None = None,
    config: dict | None = None,
) -> pd.DataFrame:
    """Vital signs from ``mimiciv_icu.chartevents``.

    Filters to common vital-sign item IDs:
      220045=HR, 220050=SysBP, 220051=DiasBP, 220052=MeanBP,
      220179=SysBP(NI), 220180=DiasBP(NI), 220210=RespRate,
      223761=TempF, 223762=TempC, 220277=SpO2
    """
    config = config or load_config()
    client = client or get_client(config)
    ce = config["tables"]["chartevents"]
    di = config["tables"]["d_items"]

    vital_items = (
        220045, 220050, 220051, 220052,
        220179, 220180, 220210,
        223761, 223762, 220277,
    )
    items_csv = ", ".join(str(i) for i in vital_items)
    hadm_set = set(hadm_ids)

    sql = f"""
    SELECT
        ce.subject_id, ce.hadm_id, ce.stay_id,
        ce.charttime,
        ce.itemid, di.label AS vital_label,
        ce.value, ce.valuenum, ce.valueuom
    FROM `{ce}` ce
    JOIN `{di}` di ON ce.itemid = di.itemid
    WHERE {{id_filter}}
      AND ce.itemid IN ({items_csv})
    ORDER BY ce.subject_id, ce.charttime
    """
    df = _batched_query(sql, "ce.subject_id", subject_ids, client)
    if not df.empty:
        df = df[df["hadm_id"].isin(hadm_set)].reset_index(drop=True)
    return df
