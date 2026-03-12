# Multi-Model Generalization & LLM-as-Judge — Agent Progress Log

**Owner:** Nick Allen
**Started:** March 11, 2026
**Branch:** `nallen21/multi-model-gen`
**Status:** HF infrastructure built — ready to run evaluations on GPU

**README TODO mapping:** This file tracks progress for:
- **Multi-model generalization (Phase 2)** — all sub-items
- **LLM-as-judge scoring** — prompt design and execution
- **Strengthen evaluation and analysis** — LLM-as-judge sub-item

---

## Objectives

1. **Fill the model × retrieval matrix** — Run all 6 retrieval strategies across 4 frozen LLMs to show the learned retrieval layer generalizes across architectures
2. **Design and run LLM-as-judge** — Create a clinical rubric prompt and score all strategy outputs on a 1–5 correctness scale using gpt-4o
3. **Produce a complete results table** with Token F1, ROUGE-L, and LLM-judge scores per cell

---

## Final Model Selection

After deliberation, the final matrix uses **4 models** (Phi-3-mini was dropped — its 128k context window flattens retrieval strategy differences, undermining our core narrative):

| Model | HF ID | Context | Source | Status |
|---|---|---|---|---|
| o4-mini | `o4-mini` | 16k | OpenAI API | **Done (Phase 1)** |
| gpt-4o-mini | `gpt-4o-mini` | 128k | OpenAI API | Not started |
| Llama-3-8B-Instruct | `meta-llama/Meta-Llama-3-8B-Instruct` | 8k | HuggingFace | Not started |
| Mistral-7B-Instruct | `mistralai/Mistral-7B-Instruct-v0.3` | 32k | HuggingFace | Not started |

**Narrative:** Retrieval strategy matters most when context is tight. Llama-3-8B (8k) is the hero model — highest expected variance across strategies.

---

## Current State (inherited from Phase 1)

### What exists

- `Evaluation/llm_runner.py` — async OpenAI wrapper with disk cache and reasoning-model detection
- `Evaluation/run_evaluation.py` — CLI orchestrator with `--model` and `--method` flags
- `Evaluation/scoring.py` — `token_f1()`, `rouge_l()`, and `llm_judge()` stub with 1–5 rubric
- `Evaluation/context_builders.py` — 6 retrieval strategies, all model-agnostic
- o4-mini row fully populated (Token F1 and ROUGE-L for all 6 strategies)
- Response cache at `data/results/cache/` (5,859 entries from Phase 1)
- Existing result files: `data/results/{method}.json` (no model prefix — must be migrated)

### Critical bug in current code (must fix before running anything)

`_save_results()` in `run_evaluation.py` (line 178) saves to `{method}.json`:
```python
path = output_dir / f"{tag}.json"  # BUG: no model prefix
```
Running gpt-4o-mini **will silently overwrite** the o4-mini results. Fix this first.

---

## Implementation Plan

### Step 0: Migrate existing result files

Rename the 6 existing Phase 1 result files to include the model prefix before touching any code:

```bash
cd data/results
for f in discharge_only.json full_context.json recency.json bm25.json semantic_rag.json learned.json; do
  mv "$f" "o4-mini__${f}"
done
```

Verify: `ls data/results/o4-mini__*.json` should show 6 files.

---

### Step 1: Fix output file naming in `run_evaluation.py`

**File:** `Evaluation/run_evaluation.py`

Change `_save_results` signature and path construction:

```python
# OLD (line 174–178):
def _save_results(results: list[dict], method: str, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    tag = results[0]["method"] if results else method
    path = output_dir / f"{tag}.json"

# NEW:
def _save_results(results: list[dict], method: str, output_dir: Path, model: str = "unknown") -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    tag = results[0].get("method", method)
    model_slug = model.replace("/", "--")
    path = output_dir / f"{model_slug}__{tag}.json"
```

Update the call site in `run()` (line 245):
```python
# OLD:
_save_results(results, method, Path(args.output_dir))

# NEW:
_save_results(results, method, Path(args.output_dir), model=args.model)
```

---

### Step 2: Update `analysis.py` for multi-model grouping

**File:** `Evaluation/analysis.py`

`load_all_results` currently returns `{method: [results]}`. Change it to parse model and method from the new `{model}__{method}.json` filename pattern, and fall back gracefully for old-style filenames:

```python
# In load_all_results(), change the key extraction logic:
# For filename "o4-mini__bm25.json" → key = ("o4-mini", "bm25")
# For filename "bm25.json" (old style) → key = ("unknown", "bm25")

def load_all_results(results_dir: str | Path = "data/results") -> dict:
    """Returns {(model, method): [result_dicts]}"""
    results_dir = Path(results_dir)
    data = {}
    for p in sorted(results_dir.glob("*.json")):
        stem = p.stem  # e.g. "o4-mini__bm25" or "bm25"
        if "__" in stem:
            model_slug, method = stem.split("__", 1)
        else:
            model_slug, method = "unknown", stem
        data[(model_slug, method)] = json.loads(p.read_text())
    return data
```

Update `summary_table()` and `export_summary_json()` to group by model, then method, producing the model × retrieval matrix.

---

### Step 3: Add runner routing to `run_evaluation.py`

**File:** `Evaluation/run_evaluation.py`

Add a `_get_runner()` helper near the top of the `run()` function, before the existing `LLMRunner` instantiation (line 203):

```python
def _get_runner(args):
    """Route to OpenAI or HuggingFace runner based on model name."""
    if "/" in args.model:
        from Evaluation.hf_runner import HFRunner
        return HFRunner(
            model_id=args.model,
            cache_dir=args.cache_dir,
        )
    return LLMRunner(
        model=args.model,
        cache_dir=args.cache_dir,
        concurrency=args.concurrency,
    )
```

Replace the `LLMRunner(...)` instantiation block (lines 203–207) with:
```python
runner = _get_runner(args)
```

Convention: any model with a `/` in its name (e.g. `meta-llama/Meta-Llama-3-8B-Instruct`) routes to `HFRunner`. OpenAI model names never contain `/`.

---

### Step 4: Build `Evaluation/hf_runner.py`

**New file.** Must match `LLMRunner` interface exactly so `run_evaluation.py` needs no other changes.

Interface contract:
```python
class HFRunner:
    model: str  # the model_id passed in

    def __init__(self, model_id: str, cache_dir: str | Path = "data/results/cache", batch_size: int = 4): ...
    async def generate(self, system_prompt: str, user_prompt: str, temperature: float = 0.3, max_tokens: int = 512) -> dict: ...
    async def generate_batch(self, items: list[dict], system_prompt: str, temperature: float = 0.3, max_tokens: int = 512) -> list[dict]: ...
    @property
    def stats(self) -> dict: ...  # {cache_hits, api_calls, errors}
```

Implementation notes:
- Load model at `__init__` time: `AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")`
- Apply chat templates via `tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)` — all three target models ship their templates in the tokenizer config, so this is automatic
- HF inference is synchronous. Wrap in `asyncio.to_thread()` inside `generate()` to avoid blocking the event loop
- `generate_batch()` processes items in batches of `batch_size` (default 4) sequentially via `asyncio.to_thread(self._run_batch, batch)`
- Cache using same SHA256 scheme as `LLMRunner` (keyed on `model_id + messages`)
- Token counts: use `len(tokenizer(text).input_ids)` for prompt tokens; derive completion tokens from output length minus input length
- Return dict: `{answer, prompt_tokens, completion_tokens, cached}` — identical to `LLMRunner`
- `max_tokens` defaults to 512 for generation (HF models typically produce verbose completions; keep this lower than the OpenAI default of 4096)

**Reasoning model detection is not needed** — HF models are not reasoning models.

---

### Step 5: Build `Evaluation/llm_judge.py`

**New file.** Post-processing script that runs LLM-as-judge on existing result files. Runs independently of `run_evaluation.py` — does not need to be integrated into the main eval loop.

The stub in `scoring.py` lines 54–66 has a rubric. Extend it with a full clinical correctness prompt:

```
System:
You are a clinical expert evaluating AI-generated answers to questions about patient records.
Score the predicted answer on a 1–5 scale based on clinical correctness and completeness.

5 — Completely correct and complete. All relevant clinical facts present.
4 — Mostly correct. Minor omissions or imprecision that would not affect clinical decisions.
3 — Partially correct. Key facts present but important details missing or imprecise.
2 — Mostly incorrect. Answer is related to the question but contains significant errors.
1 — Incorrect or irrelevant. Answer does not address the question or contains harmful errors.

Respond with only a JSON object: {"score": <int>, "reasoning": "<1 sentence>"}

User:
Question: {question}
Gold answer: {gold_answer}
Predicted answer: {predicted_answer}
```

CLI usage:
```bash
python -m Evaluation.llm_judge --results-dir data/results --judge-model gpt-4o --limit 200
```

Saves judge scores back to result files as `llm_judge_score` and `llm_judge_reasoning` fields (or writes a parallel `{model_slug}__{method}__judge.json`).

**Cost note:** gpt-4o judge on 4 models × 6 strategies × 1,000 Qs = 24,000 calls ≈ $50–100. Consider sampling 200 Qs per cell first to validate the rubric before full run.

---

## Execution Order

```
Step 0:  Rename existing result files (bash, 1 minute)
Step 1:  Edit run_evaluation.py — fix _save_results naming (10 min)
Step 2:  Edit analysis.py — multi-model grouping (30 min)
Step 3:  Edit run_evaluation.py — add _get_runner() routing (10 min)
Step 4:  Create hf_runner.py (1–2 hrs)
Step 5:  Run gpt-4o-mini (local, ~$8, fills row 2 of matrix)
Step 6:  FarmShare GPU setup (see below)
Step 7:  Run Llama-3-8B on FarmShare (compute only, fills row 3)
Step 8:  Run Mistral-7B on FarmShare (compute only, fills row 4)
Step 9:  Create llm_judge.py + design prompt (1 hr)
Step 10: Run LLM-as-judge on all result files (~$50)
Step 11: Update README.md matrix with final numbers
```

Run gpt-4o-mini command (after Steps 0–4):
```bash
conda run -n ehr python -m Evaluation.run_evaluation \
  --method all --model gpt-4o-mini --limit 200
```

---

## FarmShare GPU Setup (Stanford)

Docs: https://docs.farmshare.stanford.edu/

FarmShare uses SLURM. The HF models need a GPU node with ≥16 GB VRAM (fp16) or ≥8 GB (4-bit quant).

### 1. SSH into FarmShare

```bash
ssh <sunetid>@rice.stanford.edu
```

### 2. Clone the repo

```bash
cd $SCRATCH   # or $HOME — use $SCRATCH for large data files
git clone https://github.com/<org>/EHR-Representation-and-Retrieval.git
cd EHR-Representation-and-Retrieval
git checkout nallen21/multi-model-gen
```

### 3. Set up conda environment

```bash
module load anaconda
conda create -n ehr python=3.11 -y
conda activate ehr
pip install -r requirements.txt
```

### 4. Copy `.env` (never commit this)

From your local machine:
```bash
scp .env <sunetid>@rice.stanford.edu:~/EHR-Representation-and-Retrieval/.env
```

The `.env` must contain:
```
OPENAI_API_KEY=sk-...     # Not needed for HF models, but needed for judge
HF_TOKEN=hf_...           # Required for gated models (Llama-3, Mistral)
```

### 5. Copy data files (or use Git LFS)

`patient_timelines.json` and `qa_pairs.json` are tracked via Git LFS. Pull them:
```bash
git lfs pull
```

If LFS is not available on FarmShare, copy manually:
```bash
scp -r data/processed/ <sunetid>@rice.stanford.edu:~/EHR-Representation-and-Retrieval/data/processed/
scp -r data/generated/ <sunetid>@rice.stanford.edu:~/EHR-Representation-and-Retrieval/data/generated/
```

### 6. Request an interactive GPU node (for testing)

```bash
srun --partition=gpu --gres=gpu:1 --mem=32G --cpus-per-task=8 --time=4:00:00 --pty bash
```

For Llama-3-8B fp16, request ≥16 GB VRAM. Mistral-7B is similar.
If VRAM is limited (8–12 GB), use 4-bit quantization — add `load_in_4bit=True` to `from_pretrained()` in `hf_runner.py` (requires `bitsandbytes` package).

### 7. Run evaluation (interactive or as SLURM job)

**Interactive:**
```bash
conda activate ehr
python -m Evaluation.run_evaluation \
  --method all \
  --model meta-llama/Meta-Llama-3-8B-Instruct \
  --limit 200
```

**SLURM batch script** (`run_llama.sh`):
```bash
#!/bin/bash
#SBATCH --job-name=ehr-llama
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=8:00:00
#SBATCH --output=logs/llama_%j.out

source activate ehr
python -m Evaluation.run_evaluation \
  --method all \
  --model meta-llama/Meta-Llama-3-8B-Instruct \
  --limit 200
```

Submit: `sbatch run_llama.sh`

### 8. Copy results back

```bash
scp -r <sunetid>@rice.stanford.edu:~/EHR-Representation-and-Retrieval/data/results/ data/
```

---

## File Changes Summary

| File | Action | Status |
|---|---|---|
| `Evaluation/run_evaluation.py` | Edit — add `_make_runner()` auto-dispatch + `--hf-batch-size` flag | **Done (3/12/26)** |
| `Evaluation/analysis.py` | Edit — parse `(model, method)` from filenames, multi-model tables | Not started |
| `Evaluation/hf_runner.py` | **Created** — HuggingFace local inference runner, same interface as LLMRunner | **Done (3/12/26)** |
| `Evaluation/llm_judge.py` | **Create** — post-processing LLM-as-judge script | Not started |
| `requirements.txt` | Edit — bump `transformers>=4.40`, add `accelerate>=0.27` | **Done (3/12/26)** |
| `scripts/setup_env.sh` | **Created** — micromamba env setup + HF_HOME redirect to scratch | **Done (3/12/26)** |
| `scripts/run_hf_eval.sbatch` | **Created** — Slurm GPU batch job (normal partition, gpu QoS, L40S) | **Done (3/12/26)** |
| `data/results/*.json` | Rename — add model prefix (handled via `--output-dir` per model) | Deferred — use separate output dirs instead |
| `Progress/MULTI_MODEL_GEN.md` | Update continuously as work progresses | In progress |

---

## Model × Retrieval Matrix (Token F1)

|  | discharge-only | full-context | recency | BM25 | semantic RAG | **learned** |
|---|---|---|---|---|---|---|
| o4-mini | 0.348 | 0.415 | 0.391 | 0.321 | **0.424** | 0.406 |
| gpt-4o-mini | | | | | | |
| Llama-3-8B | | | | | | |
| Mistral-7B | | | | | | |

---

## Cost Estimates

| Task | Estimated Cost |
|---|---|
| gpt-4o-mini evaluation (6 strategies × 1,000 Qs) | ~$8 |
| HuggingFace models (FarmShare GPU, no API cost) | $0 (compute time only) |
| LLM-as-judge pilot (200 Qs × 24 cells, gpt-4o) | ~$10 |
| LLM-as-judge full run (1,000 Qs × 24 cells, gpt-4o) | ~$50–100 |
| **Total estimated** | **~$70–120** |

---

## Bugs and Issues

### Farmshare home directory quota exceeded (3/12/26)
- **Symptom:** `EDQUOT: Disk quota exceeded` when trying to write any file
- **Cause:** 41GB `cs224n_final_project` + 4.3GB miniconda3 + 3.4GB `.cache/huggingface` ≈ 49GB, over quota
- **Fix:** Moved `~/.cache/huggingface` to `/scratch/users/nallen21/hf_cache` (freed 3.4GB, quota OK again)
- **Permanent fix:** `scripts/setup_env.sh` sets `HF_HOME=/scratch/users/.../hf_cache` so future model downloads go to scratch
- **Note:** The symlink from `~/.cache/huggingface` → scratch failed (also quota blocked), so `HF_HOME` env var is used instead

---

## For the Next Agent

**Start here:**

1. Read this file top-to-bottom
2. Read `Evaluation/llm_runner.py` and `Evaluation/run_evaluation.py` to understand the existing runner interface — `hf_runner.py` must match it exactly
3. Execute Step 0 (rename files) before any code changes
4. Implement Steps 1–4 in order (they have dependencies)
5. Run gpt-4o-mini as a smoke test before touching FarmShare

**Key invariants to preserve:**
- Cache key format: `SHA256(json({model, messages}))` — must be identical in `hf_runner.py`
- Result dict shape: `{answer, prompt_tokens, completion_tokens, cached}` — `run_evaluation.py` expects exactly these keys
- All 6 strategies use the same `token_budget=4096` for fair comparison across models
- `data/results/cache/` is gitignored and regenerable — never commit it

**Known issues to watch for:**
- Llama-3-8B has an 8k context window. At `token_budget=4096`, the full-context strategy may exceed this when combined with the system prompt overhead. `hf_runner.py` should truncate inputs to `model.config.max_position_embeddings` and log a warning when truncation occurs.
- Mistral-7B-Instruct-v0.3 uses a legacy `[INST]` chat template. `apply_chat_template()` handles this automatically but verify the output format looks correct on a single example before running at scale.
- HF model weights are large (Llama-3-8B ≈ 16 GB fp16). First load will download from HuggingFace — ensure `HF_TOKEN` is set and the Llama-3 license has been accepted at huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct.
