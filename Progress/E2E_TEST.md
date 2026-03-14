# Full End-to-End Test — Progress Log

**Date:** March 13, 2026
**Branch:** `main` (no dedicated branch — no significant code changes)
**Agent:** N/A (interactive session)

---

## Objective

Run the full evaluation pipeline end-to-end with 200 patients across all 5 models and 6 retrieval strategies, then score all 30 cells with LLM-as-judge.

---

## What was done

- [x] Pulled partner's LLM-as-judge changes (commit `a0349c3`)
- [x] Resolved merge conflict markers in `Evaluation/llm_judge.py` (4 conflict regions) and `Progress/MULTI_MODEL_GEN.md` (1 region)
- [x] Fixed bug: `runner.stats()` → `runner.stats` (it's a `@property`, not a method)
- [x] Ran `analysis.py --plots` — regenerated `summary.json` and plots for all 30 cells
- [x] Ran LLM-as-judge (`gpt-4o-mini`) on all 30 result files (50–245 rows scored per file)
- [x] Submitted SLURM jobs for 3 HF models (Llama, Mistral, Phi-3) — pending GPU availability
- [x] Updated `MULTI_MODEL_GEN.md` and `README.md` with judge results
- [x] Committed and pushed all results

---

## Results Summary

### Token F1 (n=1000 per cell)

|  | discharge-only | full-context | recency | BM25 | semantic RAG | learned |
|---|---|---|---|---|---|---|
| o4-mini | 0.347 | 0.415 | 0.391 | 0.321 | **0.424** | 0.405 |
| gpt-4o-mini | 0.411 | 0.457 | 0.451 | 0.368 | **0.476** | 0.447 |
| Llama-3.1-8B | 0.327 | 0.373 | 0.364 | 0.315 | **0.394** | 0.391 |
| Mistral-7B | 0.365 | 0.393 | 0.396 | 0.335 | **0.415** | 0.398 |
| Phi-3-mini-4k | 0.124 | 0.184 | 0.239 | 0.240 | **0.299** | 0.287 |

### ROUGE-L (n=1000 per cell)

|  | discharge-only | full-context | recency | BM25 | semantic RAG | learned |
|---|---|---|---|---|---|---|
| o4-mini | 0.289 | 0.338 | 0.325 | 0.265 | **0.350** | 0.331 |
| gpt-4o-mini | 0.354 | 0.386 | 0.385 | 0.316 | **0.404** | 0.379 |
| Llama-3.1-8B | 0.275 | 0.312 | 0.303 | 0.266 | 0.329 | **0.329** |
| Mistral-7B | 0.306 | 0.327 | 0.332 | 0.281 | **0.346** | 0.330 |
| Phi-3-mini-4k | 0.102 | 0.152 | 0.199 | 0.203 | **0.249** | 0.242 |

### LLM-as-Judge (gpt-4o-mini, 1–5 scale, n=50 common questions)

|  | discharge-only | full-context | recency | BM25 | semantic RAG | learned |
|---|---|---|---|---|---|---|
| o4-mini | 3.22 | 3.28 | 3.34 | 2.86 | **3.56** | 3.46 |
| gpt-4o-mini | 2.92 | 3.14 | 3.22 | 2.56 | **3.38** | 2.88 |
| Llama-3.1-8B | 2.82 | 3.06 | 3.08 | 2.48 | **3.12** | 2.72 |
| Mistral-7B | 2.66 | 2.72 | 2.86 | 2.34 | **3.12** | 2.70 |
| Phi-3-mini-4k | 1.60 | 1.42 | 2.52 | 2.14 | **2.94** | 2.62 |

---

## Key Observations

1. **Semantic RAG** is the top strategy across all three metrics for most models.
2. **Learned (K=5)** is competitive with semantic RAG while using fewer context tokens — the efficiency story holds across all 5 models.
3. **Phi-3-mini-4k** performs poorly across the board due to 4k context window truncation.
4. **gpt-4o-mini** has the highest Token F1/ROUGE-L but o4-mini scores higher on judge (reasoning models may produce more clinically precise answers).
5. **Judge scores are moderate (2.5–3.5 mean)** — likely due to gpt-4o-mini as judge (weaker than gpt-4o), reference bias from GPT-4o-generated gold answers, and small n=50 common questions.

---

## Artifacts

- `data/results/summary.json` — aggregated metrics per model per strategy
- `data/results/judge_rankings.json` — LLM-as-judge rankings across all 30 cells
- `data/results/plots/multi_model_comparison.png` — grouped bar chart
- `data/results/plots/token_f1_heatmap.png` — heatmap of Token F1
- All 30 `data/results/{model}__{strategy}.json` files now contain `judge_score` fields

---

## Phase 2: Physician-Verified QA Set Evaluation

**Date:** March 13, 2026
**Branch:** `nallen21/verified-qa-eval`
**Status:** In progress — data ready, evaluation runs pending (FarmShare agent)

### Motivation

The Phase 1 E2E test used GPT-4o-generated QA pairs, creating a circular LLM-generates/LLM-evaluates loop. The physician-verified set (`mimic_iv_note_qa_verified.json`) provides human-validated ground truth: 70 patients, 478 correct QA pairs (28 marked incorrect by physician review), and **zero patient overlap** with the 200-patient dev set.

### Code changes completed

1. **`Preprocess/run_pipeline.py`** — Added `--qa-csv` and `--output` CLI flags so the pipeline can generate timelines for a different patient set without modifying `config.yaml` or overwriting existing timelines.

2. **`Evaluation/run_evaluation.py`** — Added filtering in `_build_eval_records` to drop QA pairs where `correct == False`. Uses `qa.get("correct", True) is not False` so it's backward-compatible with the generated QA set (which has no `correct` field).

### Data preparation completed (March 13, 2026)

- [x] **Generated verified patient timelines** — `python -m Preprocess.run_pipeline --qa-csv data/physionet.org/files/ehr-ds-qa/1.0.0/mimic_iv_note_qa_verified.csv --output data/processed/patient_timelines_verified.json --format json` — 70 records written via BigQuery (discharge summaries, structured events, demographics, admissions, diagnoses, procedures, labs, prescriptions, vitals)
- [x] **Verified data join** — 70/70 patients matched between timelines and QA; 478 correct QA pairs usable (28 physician-rejected pairs filtered by `correct == False` logic already in `run_evaluation.py`)
- [x] **Created SLURM script for verified set** — `scripts/run_verified_eval.sbatch` (HF models) and `scripts/run_verified_openai.sh` (OpenAI models)
- [x] **Pushed all data and scripts to `nallen21/verified-qa-eval`** — no BigQuery access needed from here

### Evaluation runs (March 13–14, 2026)

- [x] **o4-mini**: all 6 strategies complete
- [x] **gpt-4o-mini**: all 6 strategies complete
- [x] **Llama-3.1-8B-Instruct**: all 6 strategies complete
- [x] **Mistral-7B-Instruct-v0.3**: all 6 strategies complete
- [x] **Phi-3-mini-4k-instruct**: all 6 strategies complete
- [x] **LLM-as-judge**: all 30 files scored (gpt-4o, 478 rows per file, 10,940 total judgments)
- [x] **Logreg retrained**: model.pkl updated from 160 → 166 features (feature enrichment mismatch discovered and fixed during eval)

### Results (30/30 cells complete, March 14, 2026)

**Token F1 (n=478 physician-verified QA pairs per cell)**

| Model | discharge-only | full-context | recency | BM25 | semantic RAG | learned |
|---|---|---|---|---|---|---|
| o4-mini | **0.494** | 0.379 | 0.487 | 0.400 | 0.469 | 0.468 |
| gpt-4o-mini | **0.608** | 0.457 | 0.601 | 0.493 | 0.556 | 0.562 |
| Llama-3.1-8B | **0.551** | 0.394 | 0.540 | 0.434 | 0.495 | 0.511 |
| Mistral-7B | **0.545** | 0.386 | 0.525 | 0.417 | 0.480 | 0.486 |
| Phi-3-mini-4k | 0.420 | 0.300 | 0.410 | 0.413 | **0.475** | 0.460 |

**ROUGE-L (n=478 per cell)**

| Model | discharge-only | full-context | recency | BM25 | semantic RAG | learned |
|---|---|---|---|---|---|---|
| o4-mini | **0.455** | 0.346 | 0.449 | 0.363 | 0.427 | 0.429 |
| gpt-4o-mini | **0.575** | 0.423 | 0.565 | 0.453 | 0.516 | 0.524 |
| Llama-3.1-8B | **0.515** | 0.365 | 0.508 | 0.400 | 0.458 | 0.475 |
| Mistral-7B | **0.509** | 0.355 | 0.487 | 0.377 | 0.445 | 0.447 |
| Phi-3-mini-4k | 0.392 | 0.277 | 0.378 | 0.385 | **0.442** | 0.428 |

**LLM-as-Judge (gpt-4o, 1–5 clinical correctness scale, n=478 per cell)**

| Model | discharge-only | full-context | recency | BM25 | semantic RAG | learned |
|---|---|---|---|---|---|---|
| o4-mini | 4.38 | 3.46 | **4.40** | 3.72 | 4.09 | 4.06 |
| gpt-4o-mini | 4.29 | 3.35 | **4.32** | 3.59 | 3.92 | 3.99 |
| Llama-3.1-8B | 4.12 | 3.23 | **4.14** | 3.46 | 3.83 | 3.82 |
| Mistral-7B | **4.02** | 3.17 | 4.00 | 3.38 | 3.69 | 3.76 |
| Phi-3-mini-4k | **3.83** | 2.92 | 3.67 | 3.44 | 3.75 | 3.78 |

**Context efficiency (avg tokens per strategy, across all models)**

| Strategy | Avg context tokens | Avg Token F1 | Avg Judge |
|---|---|---|---|
| discharge-only | 1,975 | 0.523 | 4.13 |
| full-context | 3,485 | 0.383 | 3.23 |
| recency | 2,571 | 0.512 | 4.11 |
| BM25 | 875 | 0.431 | 3.52 |
| semantic RAG | 567 | 0.495 | 3.86 |
| learned | 679 | 0.497 | 3.88 |

### Key findings

1. **Discharge-only wins on lexical metrics for 4/5 models** (best F1 = 0.608 for gpt-4o-mini). The physician-written discharge summary is a natural "best-of" context — a human already curated what matters.

2. **Recency wins on judge scores for 3/5 models** (best judge = 4.40 for o4-mini). Recency includes the discharge summary plus recent events, giving the LLM slightly richer context that the judge rewards even when lexical overlap doesn't change much.

3. **Full-context is consistently the worst strategy** across all three metrics (F1 0.300–0.457, judge 2.92–3.46). Dumping all structured EHR events into the context window adds noise and hurts performance.

4. **Semantic RAG and learned achieve competitive quality with 65–72% fewer tokens**:
   - Semantic RAG: 567 tokens, F1 0.495, judge 3.86
   - Learned: 679 tokens, F1 0.497, judge 3.88
   - Discharge-only: 1,975 tokens, F1 0.523, judge 4.13
   - The efficiency gap is the core thesis result — the learned selector gets ~95% of discharge-only quality with ~1/3 the tokens.

5. **Phi-3 is the exception**: semantic RAG (F1 0.475) and learned (0.460) beat discharge-only (0.420) for Phi-3, suggesting that smaller models with tighter context windows benefit more from intelligent chunk selection.

6. **Token F1 and judge scores diverge on model rankings**: gpt-4o-mini has the highest Token F1 (0.608) but o4-mini has the highest judge scores (4.40). This suggests o4-mini produces more clinically precise answers that a judge rates higher even when they don't lexically match the gold answer.

7. **Open-source models are competitive**: Llama (F1 0.551, judge 4.14) and Mistral (F1 0.545, judge 4.02) approach commercial models on the best strategies, suggesting context quality matters more than raw model capability for this task.

8. **Dev-set vs verified-set patterns differ significantly**: On the dev set (GPT-generated QA), semantic RAG dominated; on the verified set (physician-reviewed QA), discharge-only dominates. This validates the concern about circular LLM evaluation and shows the importance of human-validated benchmarks.

### Remaining steps

- [x] All 30/30 cells complete
- [x] LLM-as-judge scoring on all 30 files (gpt-4o, full 478 rows per file)
- [x] summary.json and judge_rankings.json generated
- [ ] Generate publication-quality plots (see Progress/PLOTS.md)
- [ ] Final commit with all results

### Key data files (all committed, no BigQuery needed)

| File | Description |
|---|---|
| `data/processed/patient_timelines_verified.json` | 70 patient timelines (discharge + structured events) |
| `data/physionet.org/files/ehr-ds-qa/1.0.0/mimic_iv_note_qa_verified.json` | 70 patients, 506 QA pairs (478 correct) |
| `data/physionet.org/files/ehr-ds-qa/1.0.0/mimic_iv_note_qa_verified.csv` | Same data in CSV format (used by Preprocess pipeline) |
| `data/models/logreg/model.pkl` | Retrained logreg selector (166 features, matches current FeatureExtractor) |
| `data/models/logreg/feature_extractor.pkl` | Fitted TF-IDF vectorizer + embedding config |
| `data/results/verified/summary.json` | Aggregated Token F1, ROUGE-L, context tokens per model per strategy |
| `data/results/verified/judge_rankings.json` | LLM-as-judge rankings, score distributions, per-strategy stats |
| `data/results/verified/{model}__{strategy}.json` | 30 result files (478 rows each, all with `judge_score` fields) |
| `scripts/run_hf_eval_verified.sbatch` | SLURM script for HF models on verified set |
| `scripts/run_openai_eval_verified.sbatch` | SLURM script for OpenAI models (per-strategy, GPU for embeddings) |
| `scripts/run_all_learned_verified.sbatch` | SLURM script for all 5 models × learned strategy |
