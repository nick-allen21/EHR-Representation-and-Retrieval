# Logreg: Learned Chunk Selector for EHR Discharge Summaries

## What this is

A supervised **L1-regularized logistic regression** model that learns to score and select chunks of a patient record based on their usefulness for answering a clinical question.

**This model does not answer questions.** It is a learned **context selection layer** — it decides *which parts of the record* to include in the context passed to a frozen downstream LLM. The goal is to make smaller LLMs (e.g., LLaMA-scale models) more capable by ensuring they see the right evidence within their limited context window.

> *"Rather than improving the downstream language model itself, we learn a lightweight, interpretable 'middle layer' that maps a patient's record to a minimal set of evidence chunks that are maximally useful for answering a question."*
> — Milestone report

---

## Progress

### Done

- [x] **Preprocess pipeline** — BigQuery extraction of discharge summaries, structured events (labs, vitals, meds, procedures, diagnoses), demographics, and admission metadata into longitudinal patient timelines (`Preprocess/`)
- [x] **QA generation pipeline** — Async gpt-4o generation of grounded QA pairs with per-patient caching and resumability (`Generation/`)
- [x] **Chunking** — Section-based splitting on MIMIC-IV headers with automatic fallback to fixed-size overlapping token windows; oversized sections sub-split (`chunker.py`)
- [x] **Weak labeling** — Token F1 overlap between chunk text and gold answer; threshold at F1 ≥ 0.15 (`labeler.py`)
- [x] **Feature extraction** — 130-dim feature vector: lexical similarity (TF-IDF cosine, token overlap), semantic similarity (sentence-transformer embeddings), structural signals (position, log length), section one-hot (17), question-type one-hot (6), section × question-type interactions (102) (`features.py`)
- [x] **L1 logistic regression training** — Patient-level train/val split, negative sampling, class-weight balancing, bulk embedding computation (`train.py`)
- [x] **Inference selector** — Load trained model, score chunks, return top-K (`selector.py`)
- [x] **CLI** — `train`, `evaluate`, `select` subcommands (`run.py`)
- [x] **Evaluation metrics** — Classification diagnostics (ROC-AUC, average precision) + per-question Recall@K as primary retrieval metric
- [x] **Preliminary results on validation set** — Recall@1 = 0.45, Recall@3 = 0.71, Recall@5 = 0.84, Recall@10 = 0.96
- [x] **Diagnostic plots** — Feature importance, ROC curve, precision-recall curve, Recall@K curve, score distribution

### TODO

- [ ] **Add higher-signal temporal and structured-event features**
  - [ ] Time-gap features: Δt to discharge, coarse temporal buckets (within 24h, 2–7d, >7d), temporal marker detection ("today," "overnight," "post-op day")
  - [ ] Recency and trend features for labs/vitals: abnormal value flags, changes from baseline, simple slopes for frequently queried labs
  - [ ] Event salience indicators: binary flags for critical events (ICU transfer, intubation, surgery keywords), medication initiation/cessation, new diagnoses
  - [ ] Aggregation features: compact representations of structured events (top abnormal labs, most recent medication list, key procedures)
- [ ] **Ablate weak supervision threshold** — Sweep F1 threshold (0.10, 0.15, 0.20, 0.30) to study tradeoff between missing weakly relevant evidence (too strict) and introducing noisy positives (too lax)
- [ ] **Scale data reliably and implement evaluation dataset**
  - [ ] Expand cohorts beyond initial subset to larger, diverse set of admissions
  - [ ] Set up held-out test set (currently only train/val; test needed for final evaluation)
  - [ ] Incorporate clinician-reviewed benchmark (EHRNoteQA) as primary evaluation dataset
- [ ] **Set up downstream LLM evaluation (Phase 1: core comparison)**
  - [ ] Wire selected chunks into a frozen small LLM (e.g., o4-mini)
  - [ ] Implement all retrieval baselines: discharge-only, full-context, recency, BM25, semantic similarity RAG
  - [ ] Run learned selector vs all baselines on the same fixed LLM
  - [ ] Score with ROUGE-L, token F1 / exact match, and LLM-as-judge (1–5 scale)
- [ ] **Multi-model generalization (Phase 2: model × retrieval matrix)**
  - [ ] Run the same retrieval comparison across multiple frozen small LLMs to show the learned retrieval layer generalizes across architectures:

  |  | discharge-only | full-context | recency | BM25 | semantic RAG | **learned** |
  |---|---|---|---|---|---|---|
  | o4-mini | | | | | | |
  | gpt-4o-mini | | | | | | |
  | Llama-3-8B (HF) | | | | | | |
  | Mistral-7B (HF) | | | | | | |
  | Phi-3-mini (HF) | | | | | | |

  - [ ] Set up HuggingFace inference for open-source models (Llama-3-8B, Mistral-7B, Phi-3-mini or similar)
  - [ ] Set up OpenAI API inference for GPT small models (o4-mini, gpt-4o-mini)
  - [ ] Run full matrix and report per-model improvement from learned retrieval vs best heuristic baseline
- [ ] **Learn better scoring functions beyond linear baseline**
  - [ ] MLP ranker on frozen embeddings + engineered temporal/section features
  - [ ] Two-stage retrieval: fast retriever (TF-IDF/bi-encoder) → re-ranker (MLP or cross-encoder)
  - [ ] Structured sparsity / coherence constraints for contiguous note windows or coherent time windows
- [ ] **Strengthen evaluation and analysis**
  - [ ] Budget-efficiency curves: performance vs context size (top-K and token budgets)
  - [ ] Evidence support metrics: whether selected context contains answer spans/entities
  - [ ] Ablations: remove temporal features, remove structured events, remove section features, compare learned vs RAG heuristics
  - [ ] Error analysis: characterize failure modes (list omissions, temporal confusion, distractor overlap)
- [ ] **Final write-up**
- [ ] **Clean up code and turn in**
- [ ] **Project poster**

### Extensions (optional, if time permits)

- [ ] **Budget-efficiency curves** — Plot QA accuracy as a function of context size (top-K = 1, 3, 5, 10, 25 and token budgets) across all retrieval methods; show the learned selector achieves higher accuracy with fewer tokens
- [ ] **Per-difficulty and per-question-type breakdown** — Report metrics split by question difficulty (easy/medium/hard) and question type (medications/diagnosis/labs/imaging/procedure) to identify where the learned selector helps most
- [ ] **Additional open-source models** — Extend the model × retrieval matrix to more HuggingFace models (e.g., Gemma-2-9B, Qwen-2.5-7B) for a broader generalization claim
- [ ] **Cross-encoder re-ranker** — Train a lightweight cross-encoder on (question, chunk) pairs as a second-stage re-ranker on top of the logistic regression's top-M candidates
- [ ] **Qualitative case studies** — Select 5–10 representative questions and show side-by-side: what each retrieval method selected, what the LLM answered, and where errors occurred
- [ ] **Clinician evaluation** — Have a domain expert review a sample of LLM answers to assess clinical correctness beyond automated metrics

---

## Preliminary Results

On the validation set (patient-level split, no leakage), the learned selector achieves strong retrieval performance under small budgets:

| K | mean Recall@K |
|---|---|
| 1 | 0.45 |
| 3 | 0.71 |
| 5 | 0.84 |
| 10 | 0.96 |

Recall improves sharply from K=1 to K=5, indicating that high-scoring chunks are preferentially relevant rather than the model simply classifying most chunks as negative.

**Limitations of current results:** These evaluate retrieval quality using overlap-derived "positives" as ground truth and report validation performance only. The final evaluation will (i) use clinician-reviewed QA benchmarks (EHRNoteQA) and (ii) compare a downstream small LLM baseline to the same model augmented with learned retrieval, isolating gains attributable to improved context selection.

---

## The Learning Problem

### Input
- A **clinical question** (e.g., "What medications was the patient discharged on?")
- A **patient record** (discharge summary + serialized structured events, from `patient_timelines.json`)

### Output
- A **ranked list of chunks** from that record, scored by predicted usefulness
- At inference: the **top-K chunks** are selected and passed to a frozen downstream LLM

### What is learned
A single global scoring function:

```
p(y=1 | q, c) = σ(w⊤ φ(q, c))
```

where φ(q, c) is a 130-dimensional feature vector for (question q, chunk c). The model is trained once on thousands of (question, chunk) pairs. At test time it applies the same learned weights to new questions and new patients.

---

## Data Sources

This module reads two files, both produced outside of `Logreg/`. No BigQuery or network calls happen here.

```
data/processed/patient_timelines.json          ← from Preprocess.run_pipeline (BigQuery)
data/generated/qa_pairs.json                   ← from Generation.generate_qa (gpt-4o)  [default]
data/physionet.org/.../mimic_iv_note_qa.json   ← shipped in repo (EHR-DS-QA)           [alternative]
```

### patient_timelines.json

Written by `python -m Preprocess.run_pipeline`. Each record contains:
- `subject_id`, `hadm_id`
- `discharge_summary` — full note text (this is what gets chunked)
- `events` — chronologically sorted structured clinical events (labs, vitals, medications, procedures)
- `demographics`, `admission` — structured metadata

**To generate:**
```bash
python -m Preprocess.run_pipeline --limit 500 --format json   # quick test
python -m Preprocess.run_pipeline --format json               # full ~21k patients
```

### QA Pairs — two options

#### Option A: Generated pairs (default) — `data/generated/qa_pairs.json`

Generated by `python -m Generation.generate_qa` using gpt-4o on the full patient record (demographics + structured events + discharge summary). Each record contains:
```
subject_id, hadm_id, qa_pairs: [{question, answer, difficulty, source_types, reasoning}]
```

The `source_types` field lists which data sources are needed to answer (e.g., `["discharge_summary", "lab"]`). **Only pairs where `"discharge_summary"` is in `source_types` are kept** — because weak supervision (labeling chunks via token overlap with the answer) only works when the answer text actually appears in the discharge note.

#### Option B: EHR-DS-QA (shipped) — `data/physionet.org/files/ehr-ds-qa/1.0.0/mimic_iv_note_qa.json`

Generated from discharge summaries only using Llama 2. Contains ~156k QA pairs across ~21k admissions. Each record:
```
subject_id, hadm_id, qa_pairs: [{question, answer}]   (no source_types field)
```

All pairs are always discharge-grounded by construction, so no filtering is needed.

#### How to choose

In Python:
```python
from Logreg.data_loader import load_and_merge, QA_GENERATED, QA_EHR_DS_QA

records = load_and_merge(qa_path=QA_GENERATED)   # gpt-4o pairs (default)
records = load_and_merge(qa_path=QA_EHR_DS_QA)   # EHR-DS-QA (always available)
```

From the CLI:
```bash
python -m Logreg.run train                          # uses generated pairs (default)
python -m Logreg.run train --qa-data data/physionet.org/files/ehr-ds-qa/1.0.0/mimic_iv_note_qa.json
```

### Data loading and merging

`data_loader.merge()` joins the two files on `(subject_id, hadm_id)` and:
1. Filters to QA pairs that are discharge-grounded (see above)
2. Drops admissions with no QA pairs or no note text
3. Returns `[{subject_id, hadm_id, note_text, qa_pairs}, ...]`

---

## The Training Signal (Weak Supervision)

QA pairs provide `(question, answer)` but do **not** label individual chunks. Labels are derived automatically:

1. Split the patient record into chunks (by section header or fixed-size windows)
2. For each chunk, compute **token F1** between the chunk text and the gold answer
3. Label the chunk **positive (1)** if `token_F1 ≥ 0.15`, else **negative (0)**

This is **distant supervision** — standard in retrieval model training (e.g., DPR, BM25 re-rankers). The threshold of 0.15 will be ablated in future work (see TODO).

**Why this works:** The answers are factual sentences drawn from the discharge note. A chunk with high token overlap with the answer almost certainly contains the supporting evidence.

**Why cross-source questions are dropped:** If the answer was derived from structured events (labs, vitals) and is not phrased the same way in the note, every chunk gets near-zero token F1 → all negative labels → that QA pair contributes nothing useful.

---

## Chunking

Two strategies, selectable via `--strategy`:

### `section` (default)

Splits on MIMIC-IV discharge summary section headers using regex:
- `History of Present Illness:` (Title Case + colon)
- `PHYSICAL EXAMINATION` (ALL CAPS, 4+ chars)

Each detected section becomes one chunk. Sections exceeding a token limit are automatically sub-split into overlapping fixed-size windows so that token-F1 weak supervision works well even for long sections (e.g., Hospital Course).

**Fallback:** If fewer than 3 section headers are detected (malformed or very short note), falls back automatically to fixed-size token windows.

### `fixed`

Overlapping sliding windows of whitespace tokens.
- Default: 200 tokens per window, 100-token stride (50% overlap)
- All chunks get `section = "other"`

### Section categories (19)

Used for the one-hot encoding in the feature vector:

```
medications, diagnosis, hospital_course, results, hpi, allergies,
chief_complaint, pmh, exam, labs, followup, discharge_instructions,
discharge_condition, discharge_disposition, social, family, vitals,
procedure, other
```

---

## Feature Vector (130 dimensions)

Every `(question, chunk)` pair is converted into a 130-dimensional float32 vector. The L1 penalty drives most weights to zero; only the most informative features survive.

### Scalar features (5)

| Index | Name | Description |
|---|---|---|
| 0 | `embed_sim` | Cosine similarity of `all-MiniLM-L6-v2` sentence embeddings (question vs. chunk) |
| 1 | `tfidf_sim` | Cosine similarity of TF-IDF vectors (up to 20k unigrams+bigrams, sublinear_tf) |
| 2 | `lexical_overlap` | Fraction of unique question tokens that also appear in the chunk |
| 3 | `position` | Normalized chunk position in note: 0.0 = beginning, 1.0 = end |
| 4 | `log_length` | `log(1 + whitespace token count of chunk)` — proxy for chunk size |

### Section one-hot (19, indices 5–23)

Which clinical section the chunk came from. One-hot over the section categories listed above.

### Question-type one-hot (6, indices 24–29)

Classified from question text by keyword matching:

| Type | Keywords checked |
|---|---|
| `medications` | medication, drug, prescribed, discharge on, taking, meds, dose, dosage |
| `diagnosis` | diagnosis, diagnosed, condition, disease, disorder, finding, impression |
| `labs` | lab, laboratory, level, result, value, test, sodium, potassium, creatinine, glucose, hemoglobin |
| `imaging` | imaging, scan, x-ray, ct, mri, ultrasound, radiograph, chest |
| `procedure` | procedure, surgery, operation, performed, intervention |
| `other` | (no keywords matched) |

### Section × Question-type interactions (indices 30+)

Every pairwise product of the section one-hot and question-type one-hot.

Examples of what these capture:
- `int_medications_medications` — medications chunk + medications question → strong positive signal
- `int_hospital_course_diagnosis` — hospital course chunk + diagnosis question → likely positive
- `int_discharge_instructions_labs` — discharge instructions chunk + lab question → likely zero (irrelevant)

The model learns these associations from data rather than hard-coding them. L1 regularization zeroes out the vast majority of interactions — only the combinations that genuinely predict relevance survive.

---

## Evaluation

Two distinct metrics are reported:

### 1. Classification metrics (training diagnostics)

Computed on the flat pool of all `(question, chunk, label)` pairs in the val set.

| Metric | What it measures |
|---|---|
| `accuracy` | Fraction of chunks correctly classified positive/negative |
| `roc_auc` | Ranking quality across all (question, chunk) pairs |
| `avg_precision` | Area under the precision-recall curve |

These confirm the model is learning but are **not the retrieval metric**.

### 2. Per-question Recall@K — the primary metric

For each question in the val set:
1. Score all chunks of that patient's note
2. Select the top-K by predicted probability
3. `recall@K = (# positive chunks in top-K) / (# total positive chunks)`

Then average across all questions.

```
mean_recall@K = average over all questions of:
                    (# answer-containing chunks retrieved in top-K)
                  / (# answer-containing chunks total)
```

This directly measures the model's value to the downstream LLM. `recall@5 = 0.84` means the LLM receives 84% of the answer-supporting evidence within a 5-chunk context budget.

Reported under `val.mean_recall@{K}` in `metrics.json`. **This is the number to report.**

---

## Validation Protocol

- **Patient-level split** — all examples from a given patient are assigned entirely to train or val (no patient spans both)
- **TF-IDF fitted on train only** — vocabulary learned from train questions + train chunks, applied to val via `transform`
- **Frozen sentence embeddings** — computed with a pretrained encoder, no fitting on val data
- **Model trained on train only** — evaluated on held-out validation patients
- **Separate test set** — will be introduced for final evaluation (not yet implemented, see TODO)

---

## File Structure

```
Logreg/
├── README.md            ← this file
├── __init__.py
├── data_loader.py       loads patient_timelines.json + QA pairs JSON, joins on (subject_id, hadm_id)
├── chunker.py           splits discharge notes into chunks (section-based or fixed-size)
├── labeler.py           assigns weak binary labels via token F1 overlap with the answer
├── features.py          extracts the 130-dim feature vector for each (question, chunk) pair
├── train.py             builds dataset, trains L1 logistic regression, saves artifacts
├── selector.py          loads trained model, scores chunks, returns top-K
└── run.py               CLI: train / evaluate / select subcommands
```

---

## How to Run

### Step 1: Generate patient timelines

```bash
python -m Preprocess.run_pipeline --limit 500 --format json
```

### Step 2: (Optional) Generate QA pairs

```bash
python -m Generation.generate_qa   # requires OPENAI_API_KEY in .env
```

Skip this step to use the shipped EHR-DS-QA dataset instead.

### Step 3: Train

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

### Step 4: Evaluate Recall@K

```bash
python -m Logreg.run evaluate --K 5
python -m Logreg.run evaluate --K 5 --output data/results/logreg_recall.json
```

### Step 5: Demo — select chunks for a question

```bash
python -m Logreg.run select \
  --question "What medications was the patient discharged on?" \
  --note-file data/example_note.txt \
  --K 3
```

---

## Key Design Decisions

**Why section-based chunking?**
Discharge summaries have a well-defined section structure (HPI, Hospital Course, Discharge Medications, etc.). Section chunks are clinically meaningful units. When section parsing fails, fixed-size token windows are used as a fallback — every note always produces multiple retrievable chunks.

**Why L1 (not L2) regularization?**
L1 drives most weights to exactly zero, giving a sparse model where only the most informative features remain. This makes the model interpretable — you can read the non-zero weights directly. It also implicitly does feature selection over the interaction features, most of which are expected to be irrelevant.

**Why patient-level train/val split?**
If the same patient appears in both train and val, the model can memorize note-specific language and overfit. Patient-level splitting ensures the val set contains genuinely unseen notes.

**Why bulk embeddings?**
Sentence-transformer inference is slow when called per-example. `run_training` computes embeddings in a single batched pass per split (train then val), then reuses them during feature extraction.

**Why weak supervision via token F1?**
The QA pairs provide (question, answer) but not per-chunk relevance labels. Token F1 overlap is a cheap, scalable proxy: a chunk with high token overlap with the gold answer almost certainly contains the supporting evidence. The threshold (currently 0.15) trades off between missing weakly relevant chunks (too strict) and introducing noisy positives (too lax).
