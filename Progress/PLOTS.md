# CS 229 Final Project — Recommended Figures

Priority order: make the first four first. Together they cover the thesis, retrieval quality, and interpretability. The rest are supporting/bonus.

---

## Must-Have (4 figures)

### 1. Efficiency vs. Accuracy Scatter Plot
**What it shows:** The core thesis in one image — learned selector achieves competitive accuracy with drastically fewer tokens than full-context baselines.

- **X-axis:** Average context tokens used
- **Y-axis:** Token F1 (or LLM judge mean score once fully run)
- **Each point:** One retrieval strategy, labeled (discharge-only, full-context, recency, BM25, semantic RAG, learned)
- **Optional:** Add a Pareto frontier line

**Data source:** `data/results/summary.json` — `context_tokens_mean`, `token_f1_mean`, `judge_score_mean`

---

### 2. Grouped Bar Chart: All Metrics by Strategy
**What it shows:** Side-by-side comparison of Token F1, ROUGE-L, and LLM judge mean score for each strategy. The judge score column often tells a different story than lexical metrics — worth showing both.

- **X-axis:** Retrieval strategy
- **Y-axis:** Score (Token F1 and ROUGE-L are 0–1; judge score is 1–5, divide by 5 to normalize onto the same axis)
- **Color:** One color per metric

**Data source:** `data/results/summary.json`

---

### 3. Recall@K Curve
**What it shows:** The retrieval model's quality as a function of context budget. Directly shows how much answer-supporting evidence the logreg chunk selector captures.

- **X-axis:** K (chunks retrieved): 1, 3, 5, 10
- **Y-axis:** Mean Recall@K
- **Two lines:** Baseline (130 features) vs. Enriched (146 features)
- **Known values:**
  - Baseline: K=1→0.45, K=3→0.71, K=5→0.84, K=10→0.96
  - Enriched:  K=1→0.41, K=3→0.77, K=5→0.88, K=10→0.96

**Data source:** `data/models/logreg/metrics.json`

---

### 4. Feature Importance Bar Chart
**What it shows:** Which features the L1 model assigned non-zero weight — what it actually learned. The most "ML" figure; reviewers like it.

- **X-axis:** Feature name (top ~15–20 by absolute weight)
- **Y-axis:** Learned weight (positive = predicts relevant, negative = predicts irrelevant)
- Color positive weights one color, negative another
- Expected top features: `embed_sim`, `tfidf_sim`, `int_medications_medications`, `int_hospital_course_diagnosis`, `position`, `log_length`, etc.

**Data source:** `data/models/logreg/model.pkl` (`.coef_`) + `data/models/logreg/feature_extractor.pkl` (feature names)

---

## Supporting Figures

### 5. Judge Score Distribution (Stacked Bar)
**What it shows:** Not just the mean, but *how* strategies differ — does the learned selector produce more 4–5s and fewer 1–2s than full-context?

- **X-axis:** Retrieval strategy (3 bars: full_context, semantic_rag, learned)
- **Y-axis:** Proportion of questions (stacked, sums to 100%)
- **Colors:** 5 shades, one per score level (1=red → 5=green)

**Data source:** `data/results/judge_rankings.json` → `ranked[*].score_dist`

---

### 6. Per-Difficulty Breakdown (Grouped Bar)
**What it shows:** Where the learned selector helps most. Current data shows it outperforms full-context on easy questions (judge score 3.81 vs 3.31) but is closer on hard ones — a meaningful finding worth visualizing.

- **X-axis:** Difficulty (easy, medium, hard)
- **Y-axis:** Mean judge score
- **Grouped bars:** One per strategy (full_context, semantic_rag, learned)

**Data source:** `data/results/judge_rankings.json` → `by_difficulty`

---

### 7. Budget-Efficiency Curve (requires K sweep)
**What it shows:** As you increase K (chunks retrieved), how does downstream answer quality improve? Shows the full trade-off curve.

- **X-axis:** Context tokens (sweep K=1,3,5,10 for learned/semantic_rag)
- **Y-axis:** Token F1 or judge score
- **Lines:** One per strategy
- **Status:** Infrastructure built; needs K/N sweep runs in `run_evaluation.py`

---

### 8. Model × Retrieval Heatmap (once Phase 2 complete)
**What it shows:** The full 5×6 results matrix as a color grid. The definitive Phase 2 figure.

- **Rows:** Models (o4-mini, gpt-4o-mini, Llama-3.1-8B, Mistral-7B, Phi-3-mini)
- **Columns:** Retrieval strategies (6)
- **Cell color:** Token F1 (or judge score)
- **Status:** Only o4-mini row populated so far

**Data source:** `data/results/{model}__{method}.json` files once HF runs complete

---

## Data Locations Quick Reference

| Data needed | File |
|---|---|
| Token F1, ROUGE-L, context tokens, judge mean per strategy | `data/results/summary.json` |
| Judge win counts, score distribution, per-difficulty | `data/results/judge_rankings.json` |
| Recall@K values | `data/models/logreg/metrics.json` |
| Feature weights | `data/models/logreg/model.pkl` + `feature_extractor.pkl` |
| Per-row results (custom aggregations) | `data/results/o4-mini__{method}.json` |
