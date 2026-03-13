#!/usr/bin/env bash
#SBATCH --job-name=ehr-hf-eval
#SBATCH --partition=gpu
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
#   sbatch scripts/run_hf_eval.sbatch meta-llama/Llama-3.1-8B-Instruct
#   sbatch scripts/run_hf_eval.sbatch mistralai/Mistral-7B-Instruct-v0.3
#   sbatch scripts/run_hf_eval.sbatch microsoft/Phi-3-mini-4k-instruct
#
# The first positional argument is the HF model ID.
# Results go to data/results/ with model prefix in filenames.

set -euo pipefail

MODEL_ID="${1:-mistralai/Mistral-7B-Instruct-v0.3}"
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CONDA="${CONDA_EXE:-${HOME}/miniconda3/bin/conda}"
ENV_PREFIX="/scratch/users/${USER}/envs/ehr"

LOG_DIR="${REPO_ROOT}/logs"
mkdir -p "${LOG_DIR}"

# ── Environment ──────────────────────────────────────────────────────────────
SCRATCH_HF="/scratch/users/${USER}/hf_cache"
mkdir -p "${SCRATCH_HF}"
export HF_HOME="${SCRATCH_HF}"
export HUGGINGFACE_HUB_CACHE="${SCRATCH_HF}/hub"
export TRANSFORMERS_CACHE="${SCRATCH_HF}/hub"
export PIP_CACHE_DIR="/scratch/users/${USER}/pip_cache"

# Load .env for HF_TOKEN and OPENAI_API_KEY
set -a
source "${REPO_ROOT}/.env" 2>/dev/null || true
set +a

echo "=============================="
echo "Job ID      : ${SLURM_JOB_ID}"
echo "Node        : $(hostname)"
echo "GPU         : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "Model       : ${MODEL_ID}"
echo "HF cache    : ${SCRATCH_HF}"
echo "Conda env   : ${ENV_PREFIX}"
echo "=============================="

# ── Run evaluation (all 6 strategies) ────────────────────────────────────────
cd "${REPO_ROOT}"

"${CONDA}" run --prefix "${ENV_PREFIX}" python -m Evaluation.run_evaluation \
    --model "${MODEL_ID}" \
    --method all \
    --limit 200 \
    --token-budget 4096
