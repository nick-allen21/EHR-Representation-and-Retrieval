# Agent Operating Procedures

This document defines how AI coding agents should operate within this repository. All agents must follow these procedures.

**Core principle: Branches, README TODOs, Progress `.md` files, and Agents are always 1:1:1:1.** One branch per TODO. One Progress file per TODO. One agent per TODO. No exceptions.

---

## 1. Branch, TODO, and Progress File Mapping (1:1:1:1)

Every unit of work follows a strict **1:1:1:1 mapping**:

| Concept | Location | Example |
|---|---|---|
| **TODO item** | `README.md` → TODO table | "Multi-model generalization (Phase 2)" |
| **Branch** | Git branch | `nallen21/multi-model-gen` |
| **Progress file** | `Progress/<NAME>.md` | `Progress/MULTI_MODEL_GEN.md` |
| **Agent** | One agent session owns the work | The agent that creates the branch |

**Rules:**
- Before starting work on a TODO, create a dedicated branch and a corresponding Progress file
- The Progress file name should clearly map to the TODO item (e.g., `CORE_COMPARISON_AGENT.md` ↔ "Downstream LLM evaluation (Phase 1)")
- Never work on multiple TODOs in the same branch
- Never mix progress for different TODOs in the same Progress file

---

## 2. Progress Files (`Progress/`)

### Before starting work

- Read the Progress file for the relevant TODO to understand what has already been done
- Check `README.md` TODO table for current owner and status
- Check the previous Progress files for context on related work

### During work

- Log all progress in the Progress file, including:
  - What was built (modules, functions, files)
  - Infrastructure changes (dependencies, config, git)
  - Data generated or consumed
  - Bugs encountered and how they were fixed
  - Cost tracking (API tokens, estimated spend)
  - CLI commands that reproduce the results
- Update the Progress file continuously as work progresses — don't wait until the end

### After completing work

- Mark the TODO as Done in `README.md` with the completion date
- Update the Progress file with final results, next steps for the following agent, and a "For the Next Agent" section
- Commit all changes and push

---

## 3. README.md Ownership

### TODO table

- Every task you work on must have your name in the **Owner** column
- Update the **Status** column as work progresses: blank → In progress → Done (date)
- Mark sub-items individually as they are completed
- If a task is blocked or deferred, note why in the Status column
- Never remove or modify another owner's entries without explicit instruction

### Done section

- When a TODO is complete, add a corresponding `[x]` entry to the **Done** section with your name in parentheses: `*(Nick)*`
- Include a brief summary of what was accomplished

### Results

- Fill in any results tables (e.g., model × retrieval matrix) as data becomes available
- Always include the metric name, dataset size, and conditions

---

## 4. Git Workflow

- **One branch per TODO** — branch names should be descriptive: `<owner>/short-description`
- **Commit frequently** with clear messages describing the "why" not the "what"
- **Push before ending a session** so work is not lost
- **Merge into main** only when the TODO is complete and results are verified
- **Never force-push to main**
- **Use Git LFS** for large data files (configured in `.gitattributes`)

---

## 5. Environment and Configuration

### Required keys (`.env`)

```
OPENAI_API_KEY=sk-...     # OpenAI API access (o4-mini, gpt-4o-mini, gpt-4o)
HF_TOKEN=hf_...           # HuggingFace API access (gated models)
```

- Never commit `.env` or any file containing secrets
- Always load keys via `dotenv` — never hardcode

### Conda environment

- The project uses a conda environment named `ehr`
- Run commands with `conda run -n ehr python ...` or activate first
- Add new dependencies to `requirements.txt` with minimum version pins

---

## 6. Code Standards

- **Default model for LLM calls**: Use `o4-mini` unless the task specifically requires a different model
- **Caching**: All LLM API calls must be cached to disk (SHA-256 hash of model + messages). Re-runs must cost $0.
- **Reasoning model detection**: Models with prefixes `o1`, `o3`, `o4` use `max_completion_tokens` instead of `max_tokens` and do not support `temperature`
- **Token budgets**: Enforce token limits on context prompts (default 4,096). Use `tiktoken` for counting.
- **No unnecessary files**: Don't create files unless required. Prefer editing existing files.
- **No narration comments**: Code comments should explain non-obvious intent, not describe what the code does.

---

## 7. Evaluation Pipeline Conventions

### Running evaluations

```bash
# Single strategy, single model
python -m Evaluation.run_evaluation --method discharge_only --model o4-mini --limit 200

# All strategies
python -m Evaluation.run_evaluation --method all --limit 200

# Analysis
python -m Evaluation.analysis --plots
```

### Result files

- Per-strategy results go to `data/results/<strategy>.json`
- Response cache lives in `data/results/cache/` (gitignored, regenerable)
- Plots go to `data/results/plots/`
- Summary JSON goes to `data/results/summary.json`

### Adding a new model

1. Ensure the model is supported in `llm_runner.py` (or `hf_runner.py` for local models)
2. Run `--method all` with the new model
3. Results automatically save to separate files
4. Update the model × retrieval matrix in `README.md`
5. Update the Progress file with results and any issues

### Adding a new retrieval strategy

1. Add a `build_<strategy>()` function in `context_builders.py` matching the existing signature
2. Register it in `run_evaluation.py`
3. Run and score
4. Add to comparison tables

---

## 8. Cost Awareness

- Always estimate API costs before running large-scale evaluations
- Log actual costs in the Progress file
- Use the response cache to avoid re-billing on re-runs
- Start with small dev sets (200 patients) for iteration, scale to 5k for final results
- When in doubt, run a smoke test on 5–10 patients first

---

## 9. Handoff Protocol

When finishing a session or completing a TODO:

1. Push all changes to the remote branch
2. Ensure the Progress file has a "For the Next Agent" section with:
   - What was completed
   - What is not done
   - Exact CLI commands to reproduce results
   - Known issues or blockers
   - Key files to understand
3. Update `README.md` TODO table with final status
4. If merging to main, resolve conflicts by preferring the feature branch changes (unless there's a specific reason not to)

---

## 10. File Reference

| Path | Purpose |
|---|---|
| `README.md` | Project overview, TODO table (source of truth for task tracking), results |
| `AGENT_README.md` | This file — agent operating procedures |
| `SETUP.md` | Environment setup instructions |
| `Progress/*.md` | Per-TODO progress logs (1:1 with TODO items) |
| `Evaluation/PLAN.md` | Experimental design for the evaluation pipeline |
| `Evaluation/context_builders.py` | Retrieval strategy implementations |
| `Evaluation/llm_runner.py` | LLM API wrapper + cache |
| `Evaluation/scoring.py` | Scoring functions (token F1, ROUGE-L, LLM-as-judge) |
| `Evaluation/run_evaluation.py` | CLI orchestrator |
| `Evaluation/analysis.py` | Result aggregation and visualization |
| `Logreg/` | Learned chunk selector (training + inference) |
| `Preprocess/` | BigQuery → patient timelines |
| `Generation/` | gpt-4o QA pair generation |
