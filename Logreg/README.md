# Logreg: Learned Chunk Selector for EHR Discharge Summaries

## What this is

A supervised **L1-regularized logistic regression** model that learns to score and select chunks of a discharge summary note based on their usefulness for answering a clinical question.

**This model does not answer questions.** It only decides *which parts of the note* to include in the context passed to a downstream LLM.

---

## The Learning Problem

### Input
- A **clinical question** (e.g., "What medications was the patient discharged on?")
- A **discharge summary** (full note text from MIMIC-IV-Note, fetched via BigQuery)

### Output
- A **ranked list of chunks** from that discharge summary, scored by predicted usefulness
- At inference: the **top-K chunks** are selected and passed to an LLM

### What is learned
A single global function:

```
score(question, chunk) → probability of usefulness ∈ [0, 1]
```

The model is trained once on thousands of (question, chunk) pairs. At test time, it applies the same learned weights to new questions and new patients.

---

## The Training Signal (Weak Supervision)

The EHR-DS-QA dataset provides `(question, answer)` pairs derived from discharge summaries, but does **not** label individual chunks. Labels are derived automatically:

1. Split the discharge summary into chunks (by section header)
2. For each chunk, compute **token F1** between the chunk text and the gold answer
3. Label the chunk **positive (1)** if `token_F1 ≥ 0.2`, else **negative (0)**

This is called **distant supervision** — standard in search and retrieval model training (e.g., how DPR and BM25 re-rankers are trained).

**Why this works:** EHR-DS-QA answers are factual sentences extracted directly from the discharge summary. A chunk with high token overlap with the answer almost certainly contains the supporting evidence.

---

## Feature Vector (130 dimensions)

Each `(question, chunk)` pair is converted into a 130-dimensional numeric vector:

| Feature(s) | Dim | Description |
|---|---|---|
| `embed_sim` | 1 | Cosine similarity of sentence-transformer embeddings |
| `tfidf_sim` | 1 | Cosine similarity of TF-IDF vectors |
| `lexical_overlap` | 1 | Fraction of question tokens found in chunk |
| `position` | 1 | Normalized position in the note (0 = start, 1 = end) |
| `log_length` | 1 | Log(1 + word count of chunk) |
| `sec_*` | 17 | One-hot: section category (medications, hpi, hospital_course, …) |
| `qt_*` | 6 | One-hot: question type (medications, diagnosis, labs, …) |
| `int_*_*` | 102 | Interaction: section × question type |

The **interaction features** are where the model learns things like "the medications section is relevant when the question is about medications, but not when it is about imaging."

---

## Evaluation

There are two distinct metrics:

### 1. Classification metrics (training diagnostics only)
Computed on the flat pool of all `(question, chunk, label)` pairs in the val set.

| Metric | What it measures |
|---|---|
| `accuracy` | Fraction of chunks correctly classified as useful/not |
| `roc_auc` | Ranking quality across all (question, chunk) pairs |
| `avg_precision` | Area under precision-recall curve |

These confirm the model is learning, but are **not the retrieval metric**.

### 2. Per-question Recall@K — THE metric

For each question in the val set:
1. Score all chunks of that patient's note
2. Select the top-K chunks by predicted probability
3. Recall@K = (# positive chunks in top-K) / (# total positive chunks)

Then **average Recall@K across all questions**.

```
mean_recall@K = average over all questions of:
                    (# answer-containing chunks retrieved in top-K)
                  / (# answer-containing chunks total)
```

**Example:** If a question has 2 positive chunks (both contain the answer), and the model puts both in the top-3, `recall@3 = 2/2 = 1.0`. If it retrieves only 1, `recall@3 = 0.5`.

This directly measures the model's value to the downstream LLM: a score of `recall@5 = 0.8` means the LLM receives 80% of the answer-supporting evidence when given 5 chunks.

The per-question Recall@K is reported under `val.mean_recall@K` in `metrics.json` and is the number to report in the paper.

---

## Data Dependencies

This module reads two files, both produced outside of `Logreg/`:

```
data/processed/patient_timelines.json      ← from Preprocess.run_pipeline (BigQuery)
data/physionet.org/.../mimic_iv_note_qa.json  ← shipped in the repo (EHR-DS-QA)
```

The `patient_timelines.json` is the only file that requires BigQuery. Once it exists, all Logreg code runs entirely locally.

**To generate it:**
```bash
# Small test run (100 patients, ~2 min)
python -m Preprocess.run_pipeline --limit 100 --format json

# Full run (~21k patients)
python -m Preprocess.run_pipeline --format json
```

---

## File Structure

```
Logreg/
├── README.md            ← this file
├── __init__.py
├── data_loader.py       loads patient_timelines.json + mimic_iv_note_qa.json, joins them
├── chunker.py           splits discharge notes into chunks (section-based or fixed-size)
├── labeler.py           assigns weak binary labels via token F1 overlap with the answer
├── features.py          extracts the 130-dim feature vector for each (question, chunk) pair
├── train.py             builds dataset, trains L1 logistic regression, saves artifacts
├── selector.py          loads trained model, scores chunks, returns top-K
└── run.py               CLI entry point: train / evaluate / select subcommands
```

---

## How to Run

### Step 1: Generate patient timelines (if not done)

```bash
python -m Preprocess.run_pipeline --limit 500 --format json
```

### Step 2: Train

```bash
# Full run with sentence-transformer embeddings (recommended, slower)
python -m Logreg.run train

# Fast debug run: 100 patients, no embeddings (~30 seconds)
python -m Logreg.run train --limit 100 --no-embeddings

# Tune L1 strength (smaller C = sparser model)
python -m Logreg.run train --C 0.1
```

Artifacts are saved to `data/models/logreg/`:
- `model.pkl` — trained logistic regression weights
- `feature_extractor.pkl` — fitted TF-IDF vectorizer + config
- `metrics.json` — classification metrics + per-question Recall@K

### Step 3: Evaluate Recall@K

```bash
# Mean Recall@5 over all QA pairs
python -m Logreg.run evaluate --K 5

# Save results to a file
python -m Logreg.run evaluate --K 5 --output data/results/logreg_recall.json
```

### Step 4: Demo — select chunks for a question

```bash
python -m Logreg.run select \
  --question "What medications was the patient discharged on?" \
  --note-file data/example_note.txt \
  --K 3
```

---

## Key Design Decisions

**Why section-based chunking?**
Discharge summaries have a well-defined section structure (HPI, Hospital Course, Discharge Medications, etc.). Section chunks are clinically meaningful units — a physician reading the chart would look at a whole section, not a random 200-token window.

**Why L1 (not L2) regularization?**
L1 drives most weights to exactly zero, giving a **sparse** model where only the most informative features remain. This is interpretable — you can read the non-zero weights and understand what the model learned. With real data, you expect ~10–30 non-zero features out of 130.

**Why patient-level train/val split?**
If the same patient appears in both train and val, the model can memorize note-specific language patterns and overfit. Patient-level splitting ensures the val set contains genuinely unseen notes.

**Why not split by question?**
The same patient can have multiple questions, and all questions share the same note text. Splitting by patient is the correct granularity for EHR data.

---

## What to Expect on Real Data

On the full EHR-DS-QA dataset (~21k notes, ~156k QA pairs) with the section chunking strategy:

- Average sections per note: ~10–15
- Average positive sections per question: ~1–2
- Baseline (random selection): `mean_recall@1 ≈ 0.08–0.12`
- BM25 baseline: `mean_recall@5 ≈ 0.5–0.7` (estimate)
- Learned model target: `mean_recall@5 ≥ 0.7` with L1 sparsity

If `mean_recall@K` is barely above random, the most likely causes are:
1. Label threshold is too high/low (`--label-threshold 0.15` or `0.3`)
2. TF-IDF not fitted on enough text (use more records)
3. The discharge note text is missing for many records (check merge output)
