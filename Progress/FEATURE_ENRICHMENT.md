# Phase 2: Feature Enrichment — Progress Log

**Owner:** Niki Yoon
**Started:** March 12, 2026
**Status:** Planning

---

## Goal

Improve the learned chunk selector (L1 logistic regression) by adding temporal and clinical-salience features to the feature vector. The current 130-dim vector captures lexical/semantic similarity and section structure, but has no sense of *when* a chunk happened or whether it contains clinically critical information.

The hypothesis: chunks describing recent events (within 24h of discharge) and chunks flagging abnormal lab values or critical clinical events are more likely to be answer-supporting, regardless of keyword overlap with the question.

---

## Evaluation Protocol

Two metrics, both tracked before and after:

| Metric | Where computed | What it measures |
|---|---|---|
| **Recall@K** (K=1,3,5,10) | `python -m Logreg.run train` → `metrics.json` | Retrieval quality: do top-K chunks contain the answer-supporting evidence? |
| **Token F1 / ROUGE-L** | `python -m Evaluation.run_evaluation --method learned` | End-to-end task quality: does the LLM answer correctly given selected context? |

Recall@K is the internal retrieval metric (fast, $0 cost). Token F1 is the downstream metric (uses LLM cache if prompts are identical, may cost API credits if context changes). Report both.

**Baseline (current model):**
- Recall@1 = 0.45, Recall@3 = 0.71, Recall@5 = 0.84, Recall@10 = 0.96
- Token F1 = 0.406, ROUGE-L = 0.335 (200-patient dev set, o4-mini)

---

## Planned Features

### Group A: Text-signal features (~4 new dims)
Pure text analysis on `c["text"]`. No pipeline changes required.

| Name | Type | Description |
|---|---|---|
| `has_temporal_marker` | 0/1 | Chunk contains temporal expressions: "today", "overnight", "post-op day N", "hospital day N", "POD #N", "HD #N" |
| `temporal_density` | float | Temporal marker count divided by chunk word count (0.0–1.0) |
| `has_critical_event` | 0/1 | Chunk contains critical clinical keywords: ICU transfer, intubation, intubated, vasopressor, cardiac arrest, emergent, code blue, CPR |
| `has_abnormal_lab` | 0/1 | Chunk contains lab abnormality signals: " H ", " L ", "CRITICAL", "HIGH", "LOW" near a number, or explicit "abnormal" |

### Group B: Temporal proximity features (~4 new dims)
Requires small change: pass `discharge_dt` (parsed from `admission["dischtime"]`) as metadata on each chunk in `train.py:build_dataset`. Event serializers embed ISO timestamps in chunk text (e.g., `2180-06-26 22:45:00`) that can be regex-extracted.

| Name | Type | Description |
|---|---|---|
| `days_to_discharge` | float [0,1] | Days from event timestamp to dischtime, normalized by admission length. 0 = at discharge, 1 = at admission |
| `is_within_24h` | 0/1 | Event occurred within 24h of discharge |
| `is_within_1week` | 0/1 | Event occurred within 7 days of discharge |
| `has_no_timestamp` | 0/1 | 1 for note-section chunks (no embedded timestamp); 0 for event chunks. Prevents spurious "no timestamp = old" signal |

### Group C: New signal × question-type interactions (~8 new dims)
Products of the highest-signal new binary features crossed with relevant question types.

| Interaction | Rationale |
|---|---|
| `has_abnormal_lab × qt_labs` | Abnormal lab chunk + lab question → strong positive |
| `has_abnormal_lab × qt_diagnosis` | Abnormal labs can support diagnosis questions |
| `has_critical_event × qt_diagnosis` | Critical event chunk + diagnosis question → likely relevant |
| `has_critical_event × qt_procedure` | Critical event chunk + procedure question → likely relevant |
| `is_within_24h × qt_medications` | Recent medication events + med question → strong positive |
| `is_within_24h × qt_labs` | Recent labs + lab question → strong positive |
| `has_temporal_marker × qt_labs` | Temporal language in chunk + temporal lab question |
| `has_temporal_marker × qt_diagnosis` | Temporal language + diagnosis question |

**Total: 130 → ~146 features.**

---

## Implementation Plan

### Step 1: Text-signal features (Group A)
**File:** `Logreg/features.py`

1. Add `_count_temporal_markers(text: str) -> int` — regex for "today", "overnight", "post-op day", "hospital day", "POD", "HD" followed by optional number.
2. Add `_has_critical_event(text: str) -> bool` — regex for ICU/critical keywords.
3. Add `_has_abnormal_lab(text: str) -> bool` — regex for H/L flags and explicit abnormal terms.
4. Extend `extract_batch()` to compute and stack these 4 features.
5. Extend `feature_names` property to include the new names.

**Tests:** Verify on a couple of hand-crafted chunk texts that the regexes fire correctly before retraining.

### Step 2: Temporal proximity features (Group B)
**Files:** `Logreg/train.py` (build_dataset), `Logreg/features.py`

In `train.py:build_dataset`:
- Parse `record["admission"]["dischtime"]` into a datetime object.
- After `chunk_note()`, iterate chunks and attempt to regex-extract an ISO timestamp from each chunk's text.
- Compute `days_before_discharge` and binary flags. Add as `chunk["discharge_dt_meta"]` dict.

In `features.py:extract_batch`:
- Read `c.get("discharge_dt_meta", {})` for each chunk.
- Extract `days_to_discharge` (normalized), `is_within_24h`, `is_within_1week`, `has_no_timestamp`.

**Note:** These features are only meaningful at training time when we have the full record. At inference time (`selector.py`), the caller must also pass `dischtime` for them to be non-zero. Update `selector.py:select_chunks()` signature to accept optional `dischtime`.

### Step 3: Interaction features (Group C)
**File:** `Logreg/features.py`

After computing the new binary features and the question-type one-hot, add 8 pairwise products as described above.

### Step 4: Retrain and evaluate
```bash
# Retrain with new features
python -m Logreg.run train

# Check Recall@K (internal metric, $0 cost)
# -> metrics.json: val.mean_recall@{1,3,5,10}

# Re-run downstream eval for learned strategy only
# (LLM cache hit if same questions/prompts)
python -m Evaluation.run_evaluation --method learned --limit 200

# Regenerate analysis
python -m Evaluation.analysis --plots
```

### Step 5: Ablation
Run with each group individually to isolate contribution:
```bash
# TODO: add --feature-groups flag to run.py to enable/disable feature groups
```

---

## File Changes Summary

| File | Change |
|---|---|
| `Logreg/features.py` | Add ~15 lines of regex helpers + 4+4+8 new feature dims in extract_batch + update feature_names |
| `Logreg/train.py` | In build_dataset: parse dischtime, attach temporal metadata to chunks (~10 lines) |
| `Logreg/selector.py` | Accept optional `dischtime` arg for inference-time temporal features |
| `Logreg/run.py` | Thread dischtime through if needed for the `select` subcommand |

---

## Progress

- [ ] Step 1: Text-signal features (Group A)
- [ ] Step 2: Temporal proximity features (Group B)
- [ ] Step 3: Interaction features (Group C)
- [ ] Step 4: Retrain and measure Recall@K delta
- [ ] Step 5: Re-run downstream evaluation, measure Token F1 delta
- [ ] Update README.md with new results

---

## Results (to fill in)

| Metric | Baseline | After Group A | After Group A+B | After A+B+C |
|---|---|---|---|---|
| Recall@1 | 0.45 | | | |
| Recall@3 | 0.71 | | | |
| Recall@5 | 0.84 | | | |
| Recall@10 | 0.96 | | | |
| Token F1 | 0.406 | | | |
| ROUGE-L | 0.335 | | | |
| Context tokens | 1,064 | | | |
