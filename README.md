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

All experiments are conducted on the [MIMIC-IV v3.1](https://physionet.org/content/mimiciv/3.1/) dataset. We use the [EHR-DS-QA](https://physionet.org/content/ehr-ds-qa/1.0.0/) dataset (~21k QA pairs, ~500 physician-verified) grounded in MIMIC-IV discharge summaries.

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
├── Evaluation/
│   └── PLAN.md                    # Experimental design and evaluation workflow
├── data/
│   ├── physionet.org/             # EHR-DS-QA dataset (local, committed)
│   └── processed/                 # Output: merged longitudinal records (gitignored)
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Getting Started (for collaborators)

### Prerequisites

- A [PhysioNet](https://physionet.org/) credentialed account with access to MIMIC-IV and MIMIC-IV-Note
- The [Google Cloud CLI](https://cloud.google.com/sdk/docs/install) (`brew install --cask google-cloud-sdk` on macOS)

### Step 1: Clone the Repo

```bash
git clone https://github.com/<your-org>/EHR-Representation-and-Retrieval.git
cd EHR-Representation-and-Retrieval
```

### Step 2: Create a Conda Environment

```bash
conda create -n ehr python=3.11 -y
conda activate ehr
pip install -r requirements.txt
```

### Step 3: Link PhysioNet to Google Cloud

1. Go to https://physionet.org/settings/cloud/
2. Under **Google Cloud Platform**, enter your Google email (must be the same email added to the shared GCP project)
3. Click **Save**
4. Request BigQuery access on each dataset page:
   - https://physionet.org/content/mimiciv/3.1/ (MIMIC-IV core)
   - https://physionet.org/content/mimic-iv-note/2.2/ (MIMIC-IV-Note)
   - Look for the **BigQuery** section and accept the data use agreement

### Step 4: Get Added to the GCP Project

Ask a project owner to add your Google email to the shared GCP project `ehr-representation-retrieval` via **IAM & Admin > Grant Access** with the **BigQuery Job User** role. (If you're the owner, this is already done.)

### Step 5: Authenticate Locally

```bash
# Log in to gcloud with the SAME email you linked to PhysioNet
gcloud auth login <your-email>
gcloud config set account <your-email>
gcloud config set project ehr-representation-retrieval

# Set up Application Default Credentials for Python
gcloud auth application-default login
gcloud auth application-default set-quota-project ehr-representation-retrieval
```

When the browser opens, sign in with the email that is linked to PhysioNet (this is the identity that has read access to the `physionet-data` BigQuery tables).

### Step 6: Verify the Connection

```bash
python -c "
from Preprocess.bigquery_client import get_client
client = get_client()
result = client.query('SELECT COUNT(*) as n FROM \`physionet-data.mimiciv_3_1_hosp.patients\`').to_dataframe()
print(result)
"
```

You should see `364627` (the number of patients in MIMIC-IV v3.1).

### Step 7: Run the Preprocessing Pipeline

```bash
# Test with a small subset
python -m Preprocess.run_pipeline --limit 5 --format json

# Full run (all ~21k QA rows)
python -m Preprocess.run_pipeline
```

Output is written to `data/processed/`.

---

## Data Pipeline

The preprocessing pipeline joins the local EHR-DS-QA dataset with MIMIC-IV tables on BigQuery to produce longitudinal patient records:

1. **Load** the EHR-DS-QA CSV (local) to get `subject_id` and `hadm_id` for each QA pair
2. **Fetch** from BigQuery: discharge summaries, demographics, admissions, diagnoses, labs, vitals, prescriptions, procedures
3. **Merge** into a single record per admission containing temporally ordered clinical events, the full discharge summary, and the associated QA pairs
4. **Save** to `data/processed/` as Parquet or JSON

### Output Record Structure

Each record contains:

- `subject_id`, `hadm_id`, `note_id` — identifiers
- `demographics` — gender, age, date of death
- `admission` — admit/discharge times, location, insurance, etc.
- `discharge_summary` — full text of the discharge note
- `events` — temporally sorted list of clinical events, each with:
  - `event_type` — one of: `lab`, `vital`, `medication`, `procedure`, `diagnosis`
  - `timestamp` — when the event occurred
  - Domain-specific fields (lab values, drug names, ICD codes, etc.)
- `qa_pairs` — the original question-answer pairs from EHR-DS-QA

---

## Troubleshooting

**403 Access Denied on BigQuery tables**: Make sure (1) your `gcloud auth list` shows the email linked to PhysioNet as active, and (2) you've requested BigQuery access on the PhysioNet dataset pages. Run `gcloud auth application-default login` and sign in with the correct email.

**"BigQuery Storage module not found" warning**: Harmless. Install `google-cloud-bigquery-storage` to silence it, but it's not required.

**Slow queries**: The `--limit N` flag restricts how many QA rows are processed (and therefore how many patients are queried). Use `--limit 10` for testing.
