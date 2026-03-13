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
- [x] **Downstream LLM evaluation pipeline** *(Nick)* — 6 retrieval strategies (discharge-only, full-context, recency, BM25, semantic RAG, learned) evaluated on frozen o4-mini with token F1 and ROUGE-L scoring (`Evaluation/`)
- [x] **Phase 1 core comparison results** *(Nick)* — 200-patient dev set, 1,000 QA pairs, 4,096-token budget; learned selector achieves 0.406 token F1 with 72% fewer tokens than full-context
- [x] **Response caching infrastructure** *(Nick)* — SHA-256 disk cache in `data/results/cache/` for $0 re-runs; reasoning-model detection for o-series API quirks (`max_completion_tokens`, no `temperature`)
- [x] **Git LFS data sharing** *(Nick)* — `patient_timelines.json` and `qa_pairs.json` tracked via LFS so collaborators don't need to regenerate
- [x] **Per-difficulty breakdown** *(Nick)* — Metrics split by question difficulty (easy/medium/hard) implemented in `analysis.py`

### TODO

| Task | Owner | Status |
|---|---|---|
| **Add higher-signal temporal and structured-event features** | | |
| — Time-gap features, temporal buckets, temporal marker detection | | |
| — Recency/trend features for labs/vitals, abnormal value flags | | |
| — Event salience indicators (ICU transfer, intubation, new dx) | | |
| — Aggregation features (top abnormal labs, recent med list) | | |
| **Ablate weak supervision threshold** (sweep 0.10–0.30) | | |
| **Scale data and implement evaluation dataset** | Nick | In progress |
| — Expand to 200-patient dev set with gpt-4o QA pairs | Nick | Done |
| — Scale to 5,000 patients for final results (~$1,300 est.) | Nick | |
| — Set up held-out test set | | |
| — Incorporate EHRNoteQA as primary eval benchmark | | |
| **Downstream LLM evaluation (Phase 1: core comparison)** | Nick | Done (3/11/26) |
| — Wire selected chunks into frozen small LLM (o4-mini) | Nick | Done |
| — Implement retrieval baselines (discharge-only, full-context, recency, BM25, semantic RAG) | Nick | Done |
| — Run learned selector vs all baselines on same fixed LLM | Nick | Done |
| — Score with ROUGE-L, token F1 | Nick | Done |
| — LLM-as-judge scoring | Nick | In progress — designing prompt + running |
| **Multi-model generalization (Phase 2)** | Nick | In progress |
| — Extend `llm_runner.py` to support HuggingFace models | Nick | Done — `Evaluation/hf_runner.py` |
| — Set up HuggingFace inference (Llama-3.1-8B, Mistral-7B, Phi-3-mini-4k) | Nick | Infrastructure done; GPU runs pending |
| — Set up OpenAI API inference (o4-mini, gpt-4o-mini) | Nick | Done — both complete |
| — Run full model × retrieval matrix (5 models × 6 strategies) | Nick | |
| **Learn better scoring functions** | | |
| — MLP ranker on frozen embeddings | | |
| — Two-stage retrieval (fast retriever → re-ranker) | | |
| **Strengthen evaluation and analysis** | Nick | In progress |
| — Budget-efficiency curves (accuracy vs K) | Nick | Infrastructure built, need K/N sweeps |
| — LLM-as-judge prompt design and execution | Nick | In progress |
| — Evidence support metrics | | |
| — Feature group ablations | | |
| — Error analysis (list omissions, temporal confusion, distractor overlap) | | |
| **Final write-up** | | |
| **Clean up code and turn in** | | |
| **Project poster** | | |

**Model × retrieval matrix** (to be filled during Phase 2):

|  | discharge-only | full-context | recency | BM25 | semantic RAG | **learned** |
|---|---|---|---|---|---|---|
| o4-mini (OpenAI, 16k ctx) | 0.347 | 0.415 | 0.391 | 0.321 | **0.424** | 0.406 |
| gpt-4o-mini (OpenAI, 128k ctx) | 0.411 | 0.457 | 0.451 | 0.368 | **0.476** | 0.447 |
| Llama-3.1-8B-Instruct (HF, 128k ctx) | 0.327 | 0.373 | 0.364 | 0.315 | **0.394** | |
| Mistral-7B-Instruct-v0.3 (HF, 32k ctx) | 0.365 | 0.393 | 0.396 | 0.335 | | |
| Phi-3-mini-4k-instruct (HF, 4k ctx) | | | | | | |

### Extensions (optional, if time permits)

| Task | Owner | Status |
|---|---|---|
| **Budget-efficiency curves** — accuracy vs context size across all methods | Nick | Infrastructure built |
| **Per-difficulty breakdown** — metrics by easy/medium/hard | Nick | Done |
| **Per-question-type breakdown** — metrics by medications/diagnosis/labs/etc. | | |
| **Additional open-source models** — Gemma-2-9B, Qwen-2.5-7B, etc. | | |
| **Cross-encoder re-ranker** — second-stage re-ranker on top of logreg top-M | | |
| **Qualitative case studies** — side-by-side comparison of retrieval + LLM answers | | |
| **Clinician evaluation** — domain expert review of answer correctness | | |

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

### Downstream LLM Evaluation (Phase 1: Core Comparison)

Frozen o4-mini evaluated on 200 patients (1,000 QA pairs) with 4,096-token context budget. Gold answers generated by gpt-4o.

| Strategy | Token F1 | ROUGE-L | Avg Context Tokens |
|---|---|---|---|
| discharge-only | 0.348 | 0.289 | 2,679 |
| full-context | 0.415 | 0.338 | 3,815 |
| recency (N=25) | 0.391 | 0.325 | 3,258 |
| BM25 (K=5) | 0.321 | 0.265 | 1,070 |
| semantic RAG (K=5) | **0.424** | **0.350** | 1,336 |
| learned (K=5) | 0.406 | 0.335 | 1,064 |

The learned selector achieves comparable accuracy to semantic RAG and full-context while using **72% fewer tokens** than full-context (1,064 vs 3,815). Semantic RAG slightly outperforms on this dev set; both chunk-based methods substantially beat the discharge-only floor baseline.

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
Preprocess/                  Phase 0 — BigQuery → patient timelines
├── bigquery_client.py       shared BQ client + query helper
├── extract_notes.py         fetch discharge summaries
├── extract_structured.py    fetch labs, vitals, diagnoses, meds, procedures
├── build_timeline.py        merge all sources into longitudinal records
└── run_pipeline.py          CLI entrypoint

Generation/                  Phase 0 — gpt-4o QA pair generation
├── generate_qa.py           async generation with caching + retry
└── prompts/qa_generation.txt

Logreg/                      Phase 1a — learned chunk selector
├── data_loader.py           loads + joins timelines and QA pairs
├── chunker.py               section-based and fixed-size chunking
├── labeler.py               weak binary labels via token F1 overlap
├── features.py              130-dim feature extraction
├── train.py                 L1 logistic regression training
├── selector.py              inference: score chunks, return top-K
└── run.py                   CLI: train / evaluate / select

Evaluation/                  Phase 1b — downstream LLM comparison (Nick)
├── PLAN.md                  experimental design
├── context_builders.py      6 retrieval strategies → prompts
├── llm_runner.py            async OpenAI wrapper + response cache
├── scoring.py               token F1, ROUGE-L, LLM-as-judge (stub)
├── run_evaluation.py        CLI orchestrator
└── analysis.py              summary tables, plots, JSON export

Progress/                    documentation for agent handoff
└── CORE_COMPARISON_AGENT.md Phase 1b progress log (Nick)
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

### Step 6: Downstream LLM evaluation (Phase 1)

```bash
# Run a single strategy
python -m Evaluation.run_evaluation --method discharge_only --limit 200

# Run all 6 strategies
python -m Evaluation.run_evaluation --method all --limit 200

# Custom K/N values and token budget
python -m Evaluation.run_evaluation --method bm25 --k 10 --token-budget 4096

# Analyze results and generate plots
python -m Evaluation.analysis --plots
```

Results saved to `data/results/<strategy>.json`. LLM responses are cached in `data/results/cache/` to avoid re-running identical prompts.

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