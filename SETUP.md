# Setup & Running Guide

How to install dependencies, authenticate with external services, and run each pipeline in this project.

---

## Prerequisites

- **Python 3.11+** (via [Conda](https://docs.conda.io/) or similar)
- A [PhysioNet](https://physionet.org/) credentialed account with access to MIMIC-IV and MIMIC-IV-Note
- The [Google Cloud CLI](https://cloud.google.com/sdk/docs/install) (`brew install --cask google-cloud-sdk` on macOS)
- An [OpenAI API key](https://platform.openai.com/api-keys) (for QA generation and evaluation only)

---

## 1. Clone and Install

```bash
git clone https://github.com/<your-org>/EHR-Representation-and-Retrieval.git
cd EHR-Representation-and-Retrieval

conda create -n ehr python=3.11 -y
conda activate ehr
pip install -r requirements.txt
```

---

## 2. Link PhysioNet to Google Cloud

1. Go to https://physionet.org/settings/cloud/
2. Under **Google Cloud Platform**, enter your Google email (must be the same email added to the shared GCP project)
3. Click **Save**
4. Request BigQuery access on each dataset page:
   - https://physionet.org/content/mimiciv/3.1/ (MIMIC-IV core)
   - https://physionet.org/content/mimic-iv-note/2.2/ (MIMIC-IV-Note)
   - Look for the **BigQuery** section and accept the data use agreement

---

## 3. GCP Project Access

Ask a project owner to add your Google email to the shared GCP project `ehr-representation-retrieval` via **IAM & Admin > Grant Access** with the **BigQuery Job User** role. (If you're the owner, this is already done.)

---

## 4. Authenticate Locally

```bash
gcloud auth login <your-email>
gcloud config set account <your-email>
gcloud config set project ehr-representation-retrieval

gcloud auth application-default login
gcloud auth application-default set-quota-project ehr-representation-retrieval
```

Sign in with the email linked to PhysioNet (the identity that has read access to `physionet-data` BigQuery tables).

**Verify the connection:**

```bash
python -c "
from Preprocess.bigquery_client import get_client
client = get_client()
result = client.query('SELECT COUNT(*) as n FROM \`physionet-data.mimiciv_3_1_hosp.patients\`').to_dataframe()
print(result)
"
```

You should see `364627` (the number of patients in MIMIC-IV v3.1).

---

## 5. Set Up API Keys

Create a `.env` file in the project root:

```bash
echo 'OPENAI_API_KEY="sk-..."' > .env
```

This file is gitignored and never committed. Required for QA generation (`Generation/`) and evaluation (`Evaluation/`).

---

## Running the Pipelines

### Preprocessing (BigQuery → patient timelines)

```bash
# Test with a small subset
python -m Preprocess.run_pipeline --limit 5 --format json

# Preliminary run (500 patients)
python -m Preprocess.run_pipeline --limit 500 --format json

# Full run (all ~21k QA rows)
python -m Preprocess.run_pipeline --format json
```

Output: `data/processed/patient_timelines.json`

### QA Generation (gpt-4o)

```bash
# Generate QA pairs for all patients in the timelines file
python -m Generation.generate_qa

# Or specify paths explicitly
python -m Generation.generate_qa \
    --input data/processed/patient_timelines.json \
    --output data/generated/qa_pairs.json
```

The pipeline is **append-only** — previously generated QA pairs are never overwritten. Per-patient results are cached in `data/generated/cache/` for resumability. Re-running after scaling up preprocessing will only generate pairs for new patients.

**Alternative:** Skip this step entirely and use the shipped EHR-DS-QA dataset at `data/physionet.org/files/ehr-ds-qa/1.0.0/mimic_iv_note_qa.json` (~156k QA pairs, always available).

### Training the Learned Selector (Logreg)

```bash
# Default: generated QA pairs, section chunking, sentence-transformer embeddings
python -m Logreg.run train

# Use EHR-DS-QA instead
python -m Logreg.run train --qa-data data/physionet.org/files/ehr-ds-qa/1.0.0/mimic_iv_note_qa.json

# Fast debug run: 100 patients, no embeddings (~30 seconds)
python -m Logreg.run train --limit 100 --no-embeddings

# Fixed-size chunking instead of sections
python -m Logreg.run train --strategy fixed

# Tune L1 strength (smaller C = sparser model)
python -m Logreg.run train --C 0.1
```

Artifacts saved to `data/models/logreg/`:
- `model.pkl` — trained logistic regression weights
- `feature_extractor.pkl` — fitted TF-IDF vectorizer + config
- `metrics.json` — classification metrics + per-question Recall@K
- `plots/` — feature importance, ROC, PR, Recall@K, score distribution

### Evaluating Recall@K

```bash
python -m Logreg.run evaluate --K 5
python -m Logreg.run evaluate --K 5 --output data/results/logreg_recall.json
```

### Demo: Select Chunks for a Question

```bash
python -m Logreg.run select \
  --question "What medications was the patient discharged on?" \
  --note-file data/example_note.txt \
  --K 3
```

### Downstream LLM Evaluation (not yet implemented)

See `Evaluation/PLAN.md` for the full evaluation plan. Once implemented:

```bash
# Run a retrieval baseline
python -m Evaluation.run_baselines --method discharge_only --model o4-mini --split verified

# Run the learned selector
python -m Evaluation.run_learned --K 5 --model o4-mini --split verified
```

---

## Troubleshooting

**403 Access Denied on BigQuery tables:** Make sure (1) `gcloud auth list` shows the email linked to PhysioNet as active, and (2) you've requested BigQuery access on the PhysioNet dataset pages. Run `gcloud auth application-default login` and sign in with the correct email.

**"BigQuery Storage module not found" warning:** Harmless. Install `google-cloud-bigquery-storage` to silence it, but it's not required.

**Slow queries:** The `--limit N` flag restricts how many QA rows are processed (and therefore how many patients are queried). Use `--limit 10` for testing.

**Missing `patient_timelines.json`:** Run `python -m Preprocess.run_pipeline --format json` first. The Logreg and Evaluation modules read from this file; they do not access BigQuery directly.

**Missing `qa_pairs.json`:** Either run `python -m Generation.generate_qa` (requires OpenAI key) or use the shipped EHR-DS-QA dataset by passing `--qa-data data/physionet.org/files/ehr-ds-qa/1.0.0/mimic_iv_note_qa.json`.
