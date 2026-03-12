#!/usr/bin/env bash
#SBATCH --job-name=ehr-hf-eval
#SBATCH --partition=normal
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=logs/hf_eval_%j_%x.out
#SBATCH --error=logs/hf_eval_%j_%x.err
#
# Run all 6 retrieval strategies with a HuggingFace model on a GPU node.
#
# Usage (from repo root):
#   sbatch scripts/run_hf_eval.sbatch meta-llama/Meta-Llama-3-8B-Instruct
#   sbatch scripts/run_hf_eval.sbatch mistralai/Mistral-7B-Instruct-v0.3
#
# The first positional argument is the HF model ID.
# Results go to data/results/<model-slug>/ (separate from o4-mini results).

set -euo pipefail

MODEL_ID="${1:-mistralai/Mistral-7B-Instruct-v0.3}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Derive a filesystem-safe slug from the model ID (e.g. mistralai/Mistral-7B → mistral-7b)
MODEL_SLUG="$(echo "${MODEL_ID}" | tr '/' '-' | tr '[:upper:]' '[:lower:]')"
OUTPUT_DIR="${REPO_ROOT}/data/results/${MODEL_SLUG}"
LOG_DIR="${REPO_ROOT}/logs"
mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

# ── Environment ──────────────────────────────────────────────────────────────
module load micromamba

# Redirect HuggingFace downloads to scratch (avoids home quota exhaustion)
SCRATCH_HF="/scratch/users/${USER}/hf_cache"
mkdir -p "${SCRATCH_HF}"
export HF_HOME="${SCRATCH_HF}"
export HUGGINGFACE_HUB_CACHE="${SCRATCH_HF}/hub"
export TRANSFORMERS_CACHE="${SCRATCH_HF}/hub"

# Load .env for HF_TOKEN and OPENAI_API_KEY
set -a
# shellcheck source=../.env
source "${REPO_ROOT}/.env" 2>/dev/null || true
set +a

echo "=============================="
echo "Job ID      : ${SLURM_JOB_ID}"
echo "Node        : $(hostname)"
echo "GPU         : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "Model       : ${MODEL_ID}"
echo "Output dir  : ${OUTPUT_DIR}"
echo "HF cache    : ${SCRATCH_HF}"
echo "=============================="

# ── Run evaluation (all 6 strategies) ────────────────────────────────────────
cd "${REPO_ROOT}"

micromamba run -n ehr python -m Evaluation.run_evaluation \
    --model "${MODEL_ID}" \
    --method all \
    --limit 200 \
    --token-budget 4096 \