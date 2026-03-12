# Multi-Model Generalization & LLM-as-Judge — Agent Progress Log

**Owner:** Nick Allen
**Started:** March 11, 2026
**Branch:** `nallen21/multi-model-gen` (to be created)
**Status:** Setup phase — extending infrastructure before model runs

**README TODO mapping:** This file tracks progress for:
- **Multi-model generalization (Phase 2)** — all sub-items
- **LLM-as-judge scoring** — prompt design and execution
- **Strengthen evaluation and analysis** — LLM-as-judge sub-item

---

## Objectives

1. **Fill the model × retrieval matrix** — Run all 6 retrieval strategies across multiple frozen LLMs to show the learned retrieval layer generalizes across architectures
2. **Design and run LLM-as-judge** — Create a clinical rubric prompt and score all strategy outputs on a 1–5 correctness scale using gpt-4o
3. **Produce a complete results table** with Token F1, ROUGE-L, and LLM-judge scores per cell

---

## Current State (inherited from Phase 1)

### What exists

- `Evaluation/llm_runner.py` — async OpenAI wrapper with disk cache and reasoning-model detection
- `Evaluation/run_evaluation.py` — CLI orchestrator supporting `--model` and `--method` flags
- `Evaluation/scoring.py` — `token_f1()`, `rouge_l()`, and `llm_judge()` stub with 1–5 rubric
- `Evaluation/context_builders.py` — 6 retrieval strategies, all model-agnostic
- o4-mini row fully populated (Token F1 and ROUGE-L for all 6 strategies)
- Response cache at `data/results/cache/` (5,859 entries from Phase 1)

### What needs to be built

| Task | Status | Notes |
|---|---|---|
| Extend `llm_runner.py` for HuggingFace models | Not started | Need unified interface for local + API models |
| Run gpt-4o-mini across all 6 strategies | Not started | Trivial — runner already handles OpenAI models |
| Run Llama-3-8B across all 6 strategies | Not started | Requires HuggingFace backend |
| Run Mistral-7B across all 6 strategies | Not started | Requires HuggingFace backend |
| Run Phi-3-mini across all 6 strategies | Not started | Requires HuggingFace backend |
| Design LLM-as-judge prompt | Not started | Rubric stub exists in `scoring.py` lines 54–66 |
| Run LLM-as-judge on all results | Not started | Depends on prompt design |
| Update `analysis.py` for multi-model tables | Not started | Need per-model breakdown |

---

## Plan

### Phase 2a: HuggingFace Runner Setup

Extend `llm_runner.py` (or add a new `hf_runner.py`) to support local model inference:

- Load models via `transformers` (AutoModelForCausalLM + AutoTokenizer)
- Match the same interface: `generate(messages, max_tokens) -> dict`
- Support batched inference for throughput
- Integrate with the same disk-based response cache
- Handle model-specific chat templates (Llama-3 uses `<|begin_of_text|>`, Mistral uses `[INST]`, etc.)
- Environment: HuggingFace API key in `.env` as `HF_TOKEN`

**Target models:**

| Model | HF ID | Parameters | Context Window |
|---|---|---|---|
| Llama-3-8B-Instruct | `meta-llama/Meta-Llama-3-8B-Instruct` | 8B | 8,192 |
| Mistral-7B-Instruct | `mistralai/Mistral-7B-Instruct-v0.3` | 7B | 32,768 |
| Phi-3-mini-128k | `microsoft/Phi-3-mini-128k-instruct` | 3.8B | 128,000 |

### Phase 2b: OpenAI Model Expansion

- Run gpt-4o-mini (already supported, just needs execution)
- Verify reasoning-model detection doesn't trigger for gpt-4o-mini (it shouldn't — prefix is `gpt`)

### Phase 2c: LLM-as-Judge

- Design clinical correctness rubric prompt (1–5 scale)
- Use gpt-4o as the judge model
- Run on all (model × strategy) result files
- Add judge scores to analysis output

### Phase 2d: Analysis

- Extend `analysis.py` to produce per-model comparison tables
- Generate the full model × retrieval matrix with all three metrics
- Budget-efficiency curves per model

---

## Environment Requirements

```
# .env keys needed
OPENAI_API_KEY=sk-...     # For gpt-4o-mini, o4-mini, gpt-4o (judge)
HF_TOKEN=hf_...           # For gated models (Llama-3, Mistral)
```

```
# Additional pip packages (to be added to requirements.txt)
transformers>=4.40
accelerate>=0.27
torch>=2.0
```

---

## Model × Retrieval Matrix (Token F1)

|  | discharge-only | full-context | recency | BM25 | semantic RAG | **learned** |
|---|---|---|---|---|---|---|
| o4-mini | 0.348 | 0.415 | 0.391 | 0.321 | **0.424** | 0.406 |
| gpt-4o-mini | | | | | | |
| Llama-3-8B | | | | | | |
| Mistral-7B | | | | | | |
| Phi-3-mini | | | | | | |

---

## Cost Estimates

| Task | Estimated Cost |
|---|---|
| gpt-4o-mini evaluation (6 strategies × 1,000 Qs) | ~$8 |
| HuggingFace models (local GPU, no API cost) | $0 (compute time only) |
| LLM-as-judge (gpt-4o, all cells) | ~$50–100 depending on scale |
| **Total estimated** | **~$60–110** |

---

## Bugs and Issues

(To be filled as work progresses)

---

## File Changes

(To be filled as work progresses — list every file created or modified)
