# Multi-Model Generalization & LLM-as-Judge — Agent Progress Log

**Owner:** Nick Allen
**Started:** March 11, 2026
**Branch:** `nallen21/multi-model-gen`
**Status:** gpt-4o-mini 5/6 done; Llama-3.1 GPU job running; Mistral/Phi-3 queued

**README TODO mapping:** This file tracks progress for:
- **Multi-model generalization (Phase 2)** — all sub-items
- **LLM-as-judge scoring** — prompt design and execution
- **Strengthen evaluation and analysis** — LLM-as-judge sub-item

---

## Objectives

1. **Fill the model × retrieval matrix** — Run all 6 retrieval strategies across **5 frozen LLMs** to show the learned retrieval layer generalizes across architectures
2. **Design and run LLM-as-judge** — Create a clinical rubric prompt and score all strategy outputs on a 1–5 correctness scale using gpt-4o
3. **Produce a complete results table** with Token F1, ROUGE-L, and LLM-judge scores per cell

---

## Final Model Selection

The matrix uses **5 models**. We originally dropped Llama because gated access was requested for the wrong model (`Meta-Llama-Guard-2-8B`, a safety classifier). Access to `meta-llama/Llama-3.1-8B` (covering all variants including Instruct) was approved on 3/12/26, so Llama is back in. Phi-3-mini-4k is kept because its 4k context window exactly matches our token budget, maximizing variance across strategies.

| Model | HF ID | Context | Source | Status |
|---|---|---|---|---|
| o4-mini | `o4-mini` | 16k | OpenAI API | **Done (Phase 1)** |
| gpt-4o-mini | `gpt-4o-mini` | 128k | OpenAI API | **Done (3/12/26)** — all 6 strategies |
| Llama-3.1-8B-Instruct | `meta-llama/Llama-3.1-8B-Instruct` | 128k | HuggingFace (gated) | **5/6 done** — learned pending (1491053) |
| Mistral-7B-Instruct-v0.3 | `mistralai/Mistral-7B-Instruct-v0.3` | 32k | HuggingFace | SLURM job running (1490849) — 1/6 done |
| Phi-3-mini-4k-instruct | `microsoft/Phi-3-mini-4k-instruct` | **4k** | HuggingFace | SLURM job running (1490850) — model loaded |

**Narrative:** Retrieval strategy matters most when context is tight. Phi-3-mini-4k is the hero model — its 4k window exactly equals our token budget, maximizing variance across strategies. The spread from 4k (Phi-3) to 128k (Llama-3.1, gpt-4o-mini) shows whether the retrieval advantage holds across radically different context windows.

---

## Current State

### What exists (inherited from Phase 1 + Phase 2 infrastructure)

- `Evaluation/llm_runner.py` — async OpenAI wrapper with disk cache and reasoning-model detection
- `Evaluation/run_evaluation.py` — CLI orchestrator with `--model`, `--method`, `--hf-batch-size` flags; `_make_runner()` auto-dispatches to HFRunner or LLMRunner based on model name
- `Evaluation/hf_runner.py` — **Created (3/12/26)** — HuggingFace local inference runner matching LLMRunner interface (generate, generate_batch, stats, SHA-256 cache)
- `Evaluation/scoring.py` — `token_f1()`, `rouge_l()`, and `llm_judge()` stub with 1–5 rubric
- `Evaluation/context_builders.py` — 6 retrieval strategies, all model-agnostic
- `scripts/smoke_test_models.py` — **Created (3/12/26)** — tokenizer verification for all 3 HF models
- `scripts/setup_env.sh` — **Created (3/12/26)** — FarmShare one-time setup (conda env, cache redirects to scratch)
- `scripts/run_hf_eval.sbatch` — **Created (3/12/26)** — SLURM GPU batch job template
- o4-mini row fully populated (Token F1 and ROUGE-L for all 6 strategies)
- Response cache at `data/results/cache/` (5,859 entries from Phase 1)
- Existing Phase 1 result files: `data/results/{method}.json` — **need renaming** to `o4-mini__{method}.json` (see Step 0)

### Critical bug — FIXED (3/12/26)

`_save_results()` previously saved to `{method}.json` with no model prefix. Now saves to `{model_slug}__{method}.json`. The call site passes `model=args.model`. Existing Phase 1 files still need renaming (Step 0).

---

## Implementation Plan

### Step 0: Migrate existing result files — **TODO (manual)**

Rename the 6 existing Phase 1 result files to match the new `{model}__{method}.json` convention:

```bash
cd data/results
for f in bm25_k5.json discharge_only.json full_context.json learned_k5.json recency_n25.json semantic_rag_k5.json; do
  mv "$f" "o4-mini__${f}"
done
```

Verify: `ls data/results/o4-mini__*.json` should show 6 files.

---

### Step 1: Fix output file naming in `run_evaluation.py` — **DONE (3/12/26)**

`_save_results` now takes a `model` parameter and writes `{model_slug}__{tag}.json`. Call site passes `model=args.model`.

---

### Step 2: Update `analysis.py` for multi-model grouping — **TODO**

**File:** `Evaluation/analysis.py`

`load_all_results` currently returns `{method: [results]}`. Change it to parse model and method from the new `{model}__{method}.json` filename pattern, and fall back gracefully for old-style filenames:

```python
def load_all_results(results_dir: str | Path = "data/results") -> dict:
    """Returns {(model, method): [result_dicts]}"""
    results_dir = Path(results_dir)
    data = {}
    for p in sorted(results_dir.glob("*.json")):
        stem = p.stem
        if "__" in stem:
            model_slug, method = stem.split("__", 1)
        else:
            model_slug, method = "unknown", stem
        data[(model_slug, method)] = json.loads(p.read_text())
    return data
```

Update `summary_table()` and `export_summary_json()` to group by model, then method, producing the 5×6 matrix.

---

### Step 3: Add runner routing to `run_evaluation.py` — **DONE (3/12/26)**

Implemented as `_make_runner(args)`. Convention: any model with `/` in its name routes to `HFRunner`; OpenAI model names never contain `/`.

---

### Step 4: Build `Evaluation/hf_runner.py` — **DONE (3/12/26), updated (3/12/26)**

Full implementation matching `LLMRunner` interface. Key details:
- GPU batched inference with left-padding
- SHA-256 disk cache (same scheme as `LLMRunner`)
- `max_length` for input truncation derived from `model.config.max_position_embeddings` (not hardcoded)
- Logs a warning when input truncation occurs
- Supported models: Llama-3.1-8B-Instruct, Mistral-7B-Instruct-v0.3, Phi-3-mini-4k-instruct

---

### Step 5: Build `Evaluation/llm_judge.py` — **TODO**

Post-processing script that runs LLM-as-judge on existing result files. Runs independently of `run_evaluation.py`.

Clinical correctness rubric (1–5 scale):
```
5 — Completely correct and complete. All relevant clinical facts present.
4 — Mostly correct. Minor omissions or imprecision that would not affect clinical decisions.
3 — Partially correct. Key facts present but important details missing or imprecise.
2 — Mostly incorrect. Answer is related to the question but contains significant errors.
1 — Incorrect or irrelevant. Answer does not address the question or contains harmful errors.
```

CLI: `python -m Evaluation.llm_judge --results-dir data/results --judge-model gpt-4o --limit 200`

**Cost note:** gpt-4o judge on 5 models × 6 strategies × 1,000 Qs = 30,000 calls ≈ $60–120. Sample 200 Qs per cell first to validate the rubric.

---

## Execution Order

```
Step 0:  Rename existing result files                                     ✓ DONE (3/12/26)
Step 1:  Fix _save_results naming in run_evaluation.py                    ✓ DONE (3/12/26)
Step 2:  Edit analysis.py — multi-model grouping (30 min)                 ← TODO
Step 3:  Add _make_runner() routing in run_evaluation.py                  ✓ DONE (3/12/26)
Step 4:  Create hf_runner.py                                              ✓ DONE (3/12/26)
Step 5:  Run gpt-4o-mini (fills row 2)                                    ✓ DONE (3/12/26)
Step 6:  Run smoke_test_models.py on FarmShare                            ✓ DONE (3/12/26) — all 3 pass
Step 7:  Run Llama-3.1-8B on FarmShare (fills row 3)                      ⏳ 5/6 done; learned submitted (1491053)
Step 8:  Run Mistral-7B on FarmShare (fills row 4)                        ⏳ SLURM job 1490849 running (1/6)
Step 9:  Run Phi-3-mini-4k on FarmShare (fills row 5)                     ⏳ SLURM job 1490850 running (0/6)
Step 10: Create llm_judge.py + design prompt (1 hr)                       ← TODO
Step 11: Run LLM-as-judge on all result files (~$60)                      ← TODO
Step 12: Update analysis.py + README.md matrix with final numbers         ← TODO
```

Run gpt-4o-mini command (after Step 0):
```bash
python -m Evaluation.run_evaluation \
  --method all --model gpt-4o-mini --limit 200
```

Run HF models on FarmShare:
```bash
sbatch scripts/run_hf_eval.sbatch meta-llama/Llama-3.1-8B-Instruct
sbatch scripts/run_hf_eval.sbatch mistralai/Mistral-7B-Instruct-v0.3
sbatch scripts/run_hf_eval.sbatch microsoft/Phi-3-mini-4k-instruct
```

---

## FarmShare GPU Setup (Stanford)

Docs: https://docs.farmshare.stanford.edu/

FarmShare uses SLURM (v25.11.1). GPU nodes are called **oat** servers and have NVIDIA L40S GPUs (4 per node). The HF models need ≥16 GB VRAM (fp16) or ≥8 GB (4-bit quant). L40S has 48 GB so all models fit comfortably in fp16.

**Important:** FarmShare does **not** have a `micromamba` module. Use `~/miniconda3/bin/conda` (already installed). The conda env lives at `/scratch/users/nallen21/envs/ehr` to avoid home quota issues.

### 1. SSH into FarmShare

```bash
ssh nallen21@rice.stanford.edu
```

### 2. Clone/update the repo

```bash
cd ~/EHR-Representation-and-Retrieval   # or wherever the repo lives
git pull origin nallen21/multi-model-gen
```

### 3. Set up conda environment (one-time)

```bash
bash scripts/setup_env.sh
```

Or manually:
```bash
CONDA=~/miniconda3/bin/conda
$CONDA create --prefix /scratch/users/nallen21/envs/ehr python=3.11 pip -y
$CONDA run --prefix /scratch/users/nallen21/envs/ehr pip install -r requirements.txt
```

### 4. Ensure `.env` exists on FarmShare (never commit this)

```bash
# From local machine:
scp .env nallen21@rice.stanford.edu:~/EHR-Representation-and-Retrieval/.env
```

The `.env` must contain:
```
OPENAI_API_KEY=sk-...     # Not needed for HF models, but needed for judge
HF_TOKEN=hf_...           # Required for gated models (Llama-3.1)
```

### 5. Data files (Git LFS)

```bash
git lfs pull
```

If LFS is not available on FarmShare, copy manually:
```bash
scp -r data/processed/ nallen21@rice.stanford.edu:~/EHR-Representation-and-Retrieval/data/processed/
scp -r data/generated/ nallen21@rice.stanford.edu:~/EHR-Representation-and-Retrieval/data/generated/
```

### 6. Run tokenizer smoke test (login node, no GPU needed)

```bash
~/miniconda3/bin/conda run --prefix /scratch/users/nallen21/envs/ehr \
  python scripts/smoke_test_models.py
```

This verifies tokenizers + chat templates for all 3 HF models (Llama-3.1, Mistral, Phi-3).

### 7. Submit SLURM GPU jobs

```bash
sbatch scripts/run_hf_eval.sbatch meta-llama/Llama-3.1-8B-Instruct
sbatch scripts/run_hf_eval.sbatch mistralai/Mistral-7B-Instruct-v0.3
sbatch scripts/run_hf_eval.sbatch microsoft/Phi-3-mini-4k-instruct
```

Each job runs all 6 retrieval strategies with `--limit 200`. Results go to `data/results/<model-slug>/`.

**Interactive testing** (request a GPU node first):
```bash
srun --partition=normal --qos=gpu --gres=gpu:1 --mem=32G --cpus-per-task=8 --time=4:00:00 --pty bash
~/miniconda3/bin/conda run --prefix /scratch/users/nallen21/envs/ehr \
  python -m Evaluation.run_evaluation \
    --method discharge_only \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --limit 5
```

### 8. Copy results back (if running on a separate FarmShare clone)

```bash
scp -r nallen21@rice.stanford.edu:~/EHR-Representation-and-Retrieval/data/results/ data/
```

---

## File Changes Summary

| File | Action | Status |
|---|---|---|
| `Evaluation/run_evaluation.py` | Edit — `_make_runner()` routing + `_save_results` model prefix + `--hf-batch-size` | **Done (3/12/26)** |
| `Evaluation/hf_runner.py` | **Created** — HuggingFace runner; updated for Llama-3.1 + dynamic max_length | **Done (3/12/26)** |
| `Evaluation/analysis.py` | Edit — parse `(model, method)` from filenames, multi-model tables | Not started |
| `Evaluation/llm_judge.py` | **Create** — post-processing LLM-as-judge script | Not started |
| `requirements.txt` | Edit — bump `transformers>=4.40`, add `accelerate>=0.27` | **Done (3/12/26)** |
| `scripts/setup_env.sh` | **Created** — conda env setup + cache redirects to scratch (fixed: conda, not micromamba) | **Done (3/12/26)** |
| `scripts/run_hf_eval.sbatch` | **Created** — SLURM GPU batch job (fixed: conda, not micromamba; all 3 HF models) | **Done (3/12/26)** |
| `scripts/smoke_test_models.py` | **Created** — tokenizer verification; updated with all 3 HF models | **Done (3/12/26)** |
| `data/results/*.json` | Rename Phase 1 files to `o4-mini__*.json` | **Done (3/12/26)** |
| `Progress/MULTI_MODEL_GEN.md` | Reconciled (3/12/26) — 5-model matrix, fixed discrepancies | **Done (3/12/26)** |
| `README.md` | Updated model × retrieval matrix to 5 rows + Llama-3.1 in TODO table | **Done (3/12/26)** |

---

## Model × Retrieval Matrix (Token F1)

|  | discharge-only | full-context | recency | BM25 | semantic RAG | **learned** |
|---|---|---|---|---|---|---|
| o4-mini | 0.347 | 0.415 | 0.391 | 0.321 | **0.424** | 0.406 |
| gpt-4o-mini | 0.411 | 0.457 | 0.451 | 0.368 | **0.476** | 0.447 |
| Llama-3.1-8B | 0.327 | 0.373 | 0.364 | 0.315 | **0.394** | *pending* |
| Mistral-7B | 0.365 | 0.393 | 0.396 | 0.335 | *running* | |
| Phi-3-mini-4k | *resubmitted (truncation fix)* | | | | | |

---

## Cost Estimates

| Task | Estimated Cost |
|---|---|
| gpt-4o-mini evaluation (6 strategies × 1,000 Qs) | ~$8 |
| HuggingFace models (FarmShare GPU, no API cost) | $0 (compute time only) |
| LLM-as-judge pilot (200 Qs × 30 cells, gpt-4o) | ~$12 |
| LLM-as-judge full run (1,000 Qs × 30 cells, gpt-4o) | ~$60–120 |
| **Total estimated** | **~$80–140** |

---

## Bugs and Issues

### Conda env at /scratch/users/nallen21/envs/ehr (3/12/26)
- `micromamba` module not available on this FarmShare instance
- Used existing `~/miniconda3/bin/conda` instead
- Env created at `/scratch/users/nallen21/envs/ehr` (not in home, avoids quota)
- Python binary: `/scratch/users/nallen21/envs/ehr/bin/python`
- **Scripts fixed (3/12/26):** `setup_env.sh` and `run_hf_eval.sbatch` now use `$CONDA_EXE` / `~/miniconda3/bin/conda` instead of micromamba

### pip and HF caches → scratch (3/12/26)
- `~/.cache/pip` (3.3GB) moved to `/scratch/users/nallen21/pip_cache_home`
- `PIP_CACHE_DIR=/scratch/users/nallen21/pip_cache` added to `~/.bashrc`
- `HF_HOME=/scratch/users/nallen21/hf_cache` already set
- Future pip installs and model downloads will not consume home quota

### Tokenizer smoke tests passed (3/12/26) — all 3 HF models
- `meta-llama/Llama-3.1-8B-Instruct`: vocab=128000, max_length=131072 (128k), chat template OK
- `mistralai/Mistral-7B-Instruct-v0.3`: vocab=32768, chat template uses `[INST]` format, OK
- `microsoft/Phi-3-mini-4k-instruct`: vocab=32000, max_length=4096, chat template OK
- Mistral reports nonsense `model_max_length` (library artifact) — actual limit is 32k
- Mistral tokenizer defaults to `padding_side=right`; `HFRunner.__init__` overrides to `left`

### Llama access — RESOLVED (3/12/26)
- Originally requested `meta-llama/Meta-Llama-Guard-2-8B` by mistake (safety classifier, not QA model)
- Phi-3-mini-4k was added as a replacement
- **HuggingFace approved access to `meta-llama/Llama-3.1-8B`** (covers all variants including Instruct)
- Llama-3.1-8B-Instruct is now back in the matrix as model #3 (Phi-3 remains as model #5)
- `hf_runner.py`, `smoke_test_models.py`, `run_hf_eval.sbatch` all updated with new model ID

### Farmshare home directory quota exceeded (3/12/26)
- **Symptom:** `EDQUOT: Disk quota exceeded` when trying to write any file
- **Cause:** 41GB `cs224n_final_project` + 4.3GB miniconda3 + 3.4GB `.cache/huggingface` ≈ 49GB, over quota
- **Fix:** Moved `~/.cache/huggingface` to `/scratch/users/nallen21/hf_cache` (freed 3.4GB, quota OK again)
- **Permanent fix:** `scripts/setup_env.sh` sets `HF_HOME=/scratch/users/.../hf_cache` so future model downloads go to scratch
- **Note:** The symlink from `~/.cache/huggingface` → scratch failed (also quota blocked), so `HF_HOME` env var is used instead

### _save_results overwrite bug — FIXED (3/12/26)
- `_save_results()` previously wrote `{method}.json` — different models would overwrite each other
- Now writes `{model_slug}__{method}.json` (e.g., `o4-mini__bm25_k5.json`)
- Existing Phase 1 files (`bm25_k5.json`, etc.) still need manual renaming — see Step 0

### SLURM sbatch script failures — FIXED (3/12/26)
- **Issue 1:** `BASH_SOURCE[0]` inside a SLURM job resolves to the spool copy (e.g., `/var/spool/slurmd/...`), not the original script. `REPO_ROOT` was wrong, causing `mkdir` to fail.
- **Fix:** Use `SLURM_SUBMIT_DIR` (the directory from which `sbatch` was run) with fallback to `BASH_SOURCE` for interactive use.
- **Issue 2:** `conda run --no-banner` not supported on FarmShare's conda version.
- **Fix:** Removed `--no-banner` flag.
- **Issue 3:** Script had DOS line endings (`\r\n`) from editing on Windows. `sed -i 's/\r$//'` corrupted the file during a connection cutoff, emptying it entirely.
- **Fix:** Rewrote the script from scratch using heredoc. All line endings are now UNIX.

### Phi-3 right-truncation cuts off question — FIXED (3/13/26)
- **Symptom:** Phi-3 discharge_only had F1=0.124 — model generated medication lists instead of answering questions
- **Cause:** `hf_runner.py` used default right-side truncation. For Phi-3 (4096 max, 3584 after gen reserve), the system_prompt + context + question + chat_template exceeded 3584 tokens. Right truncation removed the question at the end, leaving the model to do text continuation.
- **Fix:** Set `tokenizer.truncation_side = "left"` before tokenizing in `_batch_generate()`, so early context is trimmed rather than the question at the end. Restored after tokenization.
- **Impact:** Only affects Phi-3 — other models have much larger context windows (32k–128k) and never hit the truncation limit with `token_budget=4096`.
- Cancelled bad job (1490850), removed bad results, resubmitted (1491077).

### hf_runner.py hardcoded max_length — FIXED (3/12/26)
- Previously hardcoded `max_length=4096` for tokenizer truncation, which is correct for Phi-3 but wrong for Llama-3.1 (128k) and Mistral (32k)
- Now reads `model.config.max_position_embeddings` and reserves space for `max_new_tokens`
- Logs a warning when truncation occurs

---

## For the Next Agent

**Start here:**

1. Read this file top-to-bottom
2. Check SLURM job status: `squeue -u nallen21` and `sacct -u nallen21 -j 1490759,1490760,1490761`
3. Verify gpt-4o-mini learned_k5 result exists (may still be running)
4. Once HF SLURM jobs complete, score results and fill in matrix rows 3-5
5. Implement Step 2 (analysis.py multi-model grouping)
6. Build llm_judge.py (Step 10)

**Key invariants to preserve:**
- Cache key format: `SHA256(json({model, messages}))` — identical in both `llm_runner.py` and `hf_runner.py`
- Result dict shape: `{answer, prompt_tokens, completion_tokens, cached}` — `run_evaluation.py` expects exactly these keys
- All 6 strategies use the same `token_budget=4096` for fair comparison across models
- `data/results/cache/` is gitignored and regenerable — never commit it
- Result filenames: `{model_slug}__{method}.json` — e.g. `o4-mini__bm25_k5.json`

**Known issues to watch for:**
- Llama-3.1-8B-Instruct has 128k context, but `hf_runner.py` truncates input to `max_position_embeddings - max_new_tokens`. At `token_budget=4096`, this is well within limits.
- Mistral-7B-Instruct-v0.3 uses a legacy `[INST]` chat template. `apply_chat_template()` handles this automatically but verify the output format looks correct on a single example before running at scale.
- HF model weights are large (Llama-3.1-8B ≈ 16 GB fp16, Mistral-7B ≈ 14 GB fp16). First load downloads from HuggingFace — ensure `HF_TOKEN` is set and Llama license is accepted.
- FarmShare L40S GPUs have 48 GB VRAM — all models fit in fp16 without quantization.
