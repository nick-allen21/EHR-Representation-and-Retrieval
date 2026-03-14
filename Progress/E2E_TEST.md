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
