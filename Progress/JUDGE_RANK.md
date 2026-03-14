# Judge Ranking Script — Reference

**Script:** `scripts/judge_rank.py`

**What it does:** Reads `judge_score` fields already stored in result JSON files and prints a rankings table. Makes **zero API calls** and **writes nothing to disk**. Purely a read + print operation.

**What "judge_score" is:** A 1–5 rating assigned by gpt-4o-mini (pilot) comparing o4-mini's predicted answer against the gpt-4o gold answer for each question. Stored in `data/results/o4-mini__*.json`.

---

## How to run

Run from the repo root:

```bash
# Default: all scored questions across 3 strategies
python3 scripts/judge_rank.py

# Random sample of 100 questions
python3 scripts/judge_rank.py --sample 100

# Specific difficulty only
python3 scripts/judge_rank.py --difficulty hard
python3 scripts/judge_rank.py --difficulty easy
python3 scripts/judge_rank.py --difficulty medium

# Sample + difficulty combined
python3 scripts/judge_rank.py --sample 50 --difficulty hard

# Different random seed (changes which questions are sampled)
python3 scripts/judge_rank.py --sample 100 --seed 7

# Custom set of files
python3 scripts/judge_rank.py \
    --files data/results/o4-mini__learned_k5.json \
            data/results/o4-mini__full_context.json
```

---

## All flags

| Flag | Default | Description |
|---|---|---|
| `--files PATH [...]` | full_context, semantic_rag_k5, learned_k5 | Result JSON files to include |
| `--sample N` | all | Randomly sample N questions from the common scored set |
| `--seed N` | 42 | Random seed (change to get a different sample) |
| `--difficulty` | all | Filter to `easy`, `medium`, or `hard` questions only |

---

## How scoring works (no API calls in this script)

The `judge_score` field was written to each result file by a previous run of `llm_judge.py` using gpt-4o-mini as judge. For each row:

1. Judge sees: `(clinical question, gold answer from gpt-4o, predicted answer from o4-mini)`
2. Judge rates predicted answer 1–5 against the gold
3. Score is stored directly in the result file row

This script just reads those stored scores — no LLM is called.

**Rankings** are computed by:
- For each question, whichever strategy scores highest wins (ties split fractionally)
- `Wins` = total fractional wins across all questions
- `Mean Score` = average 1–5 rating across all questions in the set

---

## Current results (all 395 scored questions, gpt-4o-mini judge)

```
========================================================================
  LLM-as-Judge Rankings  (n=395 questions, all scored)
========================================================================
  #   Strategy                               Mean Score     Wins    Win %
  --- -------------------------------------- ---------- -------- --------
  1   o4-mini__full_context                       3.329    140.7    35.6%
  2   o4-mini__learned_k5                         3.382    128.7    32.6%
  3   o4-mini__semantic_rag_k5                    3.471    125.7    31.8%

  Score distribution (1=wrong → 5=perfect):
  Strategy                                [1]  [2]  [3]  [4]  [5]
  o4-mini__full_context                    62   83   53   57  140
  o4-mini__learned_k5                      50   92   48   67  138
  o4-mini__semantic_rag_k5                 42   82   60   70  141

  Mean judge score by difficulty:
  Strategy                                    easy      hard    medium
  o4-mini__full_context                      3.311     3.101     3.416
  o4-mini__learned_k5                        3.811     3.342     3.226
  o4-mini__semantic_rag_k5                   3.756     3.253     3.434
========================================================================
```

**Key finding:** Learned selector dominates on easy questions (3.811) and holds up well on hard (3.342 vs 3.101 for full_context). Semantic RAG has the highest overall mean but fewest wins — it scores consistently but rarely dominates per-question.
