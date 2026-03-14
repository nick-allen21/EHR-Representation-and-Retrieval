#!/usr/bin/env bash
#SBATCH --job-name=ehr-hf-verified
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --output=logs/hf_verified_%j_%x.out
#SBATCH --error=logs/hf_verified_%j_%x.err
#
# Run all 6 retrieval strategies with a HuggingFace model on the
# physician-verified QA set (70 patients, 478 correct QA pairs).
#
# Usage (from repo root):
#   sbatch scripts/run_hf_eval_verified.sbatch meta-llama/Llama-3.1-8B-Instruct
#   sbatch scripts/run_hf_eval_verified.sbatch mistralai/Mistral-7B-Instruct-v0.3
#   sbatch scripts/run_hf_eval_verified.sbatch microsoft/Phi-3-mini-4k-instruct

set -euo pipefail

MODEL_ID="${1:-mistralai/Mistral-7B-Instruct-v0.3}"
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
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
export PYTHONUNBUFFERED=1

set -a
source "${REPO_ROOT}/.env" 2>/dev/null || true
set +a

echo "=============================="
echo "Job ID      : ${SLURM_JOB_ID}"
echo "Node        : $(hostname)"
echo "GPU         : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "Model       : ${MODEL_ID}"
echo "Dataset     : verified (70 patients, physician-reviewed)"
echo "HF cache    : ${SCRATCH_HF}"
echo "Conda env   : ${ENV_PREFIX}"
echo "=============================="

# ── Run evaluation on verified set ───────────────────────────────────────────
cd "${REPO_ROOT}"

export PATH="${ENV_PREFIX}/bin:${PATH}"
python -m Evaluation.run_evaluation \
    --model "${MODEL_ID}" \
    --method all \
    --timelines data/processed/patient_timelines_verified.json \
    --qa-data data/physionet.org/files/ehr-ds-qa/1.0.0/mimic_iv_note_qa_verified.json \
    --output-dir data/results/verified \
    --limit 70 \
    --token-budget 4096
