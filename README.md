# Learning Minimal, Temporally Coherent Representations of Electronic Health Records for Clinical Decision Support

## Team

| Name | SUNet |
|---|---|
| Nick Allen | nallen21 |
| Niki Yoon | nikiyoon |

**Category:** Life Sciences / Natural Language

---

## Motivation

Large language models are increasingly deployed in healthcare settings to assist clinicians by retrieving and summarizing information from Electronic Health Records (EHRs). While recent systems demonstrate strong language generation capabilities, they largely rely on heuristic retrieval methods — such as semantic similarity, recency-based filtering, or note-type selection — to determine which EHR context is provided to the model. These approaches frequently surface redundant, outdated, or clinically irrelevant information and fail to capture the temporal structure that clinicians rely on when reasoning about patient trajectories.

This project tackles the problem of learning which portions of a longitudinal EHR are clinically decision-relevant under a constrained context budget. Rather than improving the language model itself, we focus on the upstream machine learning problem of **representation learning and feature selection** over heterogeneous, time-ordered clinical data. This is an application-focused project, grounded in real clinical data, that seeks to provide principled learning-based alternatives to heuristic EHR retrieval strategies.

---

## Method

We model each patient's EHR as a temporally ordered sequence consisting of unstructured clinical notes, structured events (e.g., laboratory values, medications, procedures), and timestamps. Our goal is to learn a function that maps the full EHR sequence to a sparse, weighted subset of EHR elements that maximizes downstream task performance under a fixed context size constraint and response structure to be fed into an LLM.

We plan to explore machine learning techniques including:

- **Supervised learning with sparsity constraints** — L1-regularized linear models for learned feature selection over EHR elements
- **Representation learning** — Methods that encode temporal abstraction and clinical state changes
- **Section classification** — Logistic regression over various EHR sections

The learned selection is optimized jointly with downstream task objectives, enabling principled trade-offs between information compression and predictive utility.

---

## Intended Experiments

We evaluate whether **learned EHR representations improve downstream LLM performance** compared to standard EHR retrieval methods, while holding the language model and prompting strategy fixed. The LLM is treated as a frozen downstream consumer; all experiments vary only the EHR context provided to the model.

### Baselines

- Full-context input
- Recency-based filtering
- Semantic similarity retrieval (traditional RAG)
- Note-type filtering

### Evaluation Metrics

- **Task performance** — AUROC, accuracy, or ROUGE depending on the task
- **Context efficiency** — performance as a function of context size

### Data

All experiments are conducted on the [MIMIC-IV](https://physionet.org/content/mimiciv/) dataset. We use the [EHR-DS-QA](https://physionet.org/content/ehr-ds-qa/1.0.0/) dataset which contains physician-evaluated question-answer pairs grounded in MIMIC-IV discharge summaries.

We additionally perform ablation studies to analyze the contribution of temporal modeling and sparsity constraints, as well as qualitative error analysis to identify failure modes and limitations.

---

## Project Structure

```
├── config/
│   └── config.yaml                # BigQuery project ID, dataset paths, pipeline settings
├── Preprocess/
│   ├── __init__.py
│   ├── bigquery_client.py         # Shared BQ client + query helper
│   ├── extract_notes.py           # Fetch discharge summaries from mimiciv_note
│   ├── extract_structured.py      # Fetch labs, vitals, diagnoses, meds, procedures
│   ├── build_timeline.py          # Merge all sources into longitudinal patient records
│   └── run_pipeline.py            # CLI entrypoint: orchestrates full extraction
├── data/
│   ├── physionet.org/             # EHR-DS-QA dataset (local, committed)
│   └── processed/                 # Output: merged longitudinal records (gitignored)
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Setup

### 1. Prerequisites

- A [PhysioNet](https://physionet.org/) credentialed account with access to MIMIC-IV
- A Google Cloud Platform project (free tier works) linked to your PhysioNet account
- The [Google Cloud CLI](https://cloud.google.com/sdk/docs/install) installed locally

### 2. Link PhysioNet to BigQuery

1. Go to https://physionet.org/settings/cloud/ and link your GCP account
2. Accept the data use agreement for MIMIC-IV on BigQuery
3. Both collaborators should be added to the shared GCP project via **IAM & Admin > Grant Access** (Editor role)

### 3. Create a Conda Environment

```bash
conda create -n ehr python=3.11 -y
conda activate ehr
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Authenticate with Google Cloud

```bash
gcloud auth application-default login
```

This stores credentials locally — no API keys or service account files needed. Each collaborator runs this once on their machine.

### 6. Configure the Project

Edit `config/config.yaml` and set your shared GCP project ID:

```yaml
gcp_project_id: "your-gcp-project-id"
```

### 7. Run the Pipeline

```bash
# Full run (all ~21k QA rows)
python -m Preprocess.run_pipeline

# Test with a subset
python -m Preprocess.run_pipeline --limit 10

# Output as JSON instead of Parquet
python -m Preprocess.run_pipeline --format json
```

Output is written to `data/processed/`.

---

## Data Pipeline

The preprocessing pipeline joins the local EHR-DS-QA dataset with MIMIC-IV tables on BigQuery to produce longitudinal patient records:

1. **Load** the EHR-DS-QA CSV (local) to get `subject_id` and `hadm_id` for each QA pair
2. **Fetch** from BigQuery: discharge summaries, demographics, admissions, diagnoses, labs, vitals, prescriptions, procedures
3. **Merge** into a single record per admission containing temporally ordered clinical events, the full discharge summary, and the associated QA pairs
4. **Save** to `data/processed/` as Parquet or JSON
