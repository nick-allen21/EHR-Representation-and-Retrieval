#!/usr/bin/env bash
#SBATCH --job-name=ehr-oai-verified
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --output=logs/oai_verified_%j_%x.out
#SBATCH --error=logs/oai_verified_%j_%x.err
#
# Run a SINGLE retrieval strategy with an OpenAI model on the
# physician-verified QA set.  GPU is requested for sentence-transformer
# embedding acceleration (semantic_rag / learned strategies).
#
# Usage (from repo root):
#   sbatch scripts/run_openai_eval_verified.sbatch o4-mini semantic_rag
#   sbatch scripts/run_openai_eval_verified.sbatch o4-mini learned
#   sbatch scripts/run_openai_eval_verified.sbatch gpt-4o-mini semantic_rag
#   sbatch scripts/run_openai_eval_verified.sbatch gpt-4o-mini learned

set -euo pipefail

MODEL="${1:-o4-mini}"
METHOD="${2:-semantic_rag}"
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
echo "Model       : ${MODEL} (OpenAI API)"
echo "Method      : ${METHOD}"
echo "Dataset     : verified (70 patients, physician-reviewed)"
echo "HF cache    : ${SCRATCH_HF}"
echo "Conda env   : ${ENV_PREFIX}"
echo "=============================="

# ── Run evaluation ───────────────────────────────────────────────────────────
cd "${REPO_ROOT}"

export PATH="${ENV_PREFIX}/bin:${PATH}"
python -m Evaluation.run_evaluation \
    --model "${MODEL}" \
    --method "${METHOD}" \
    --timelines data/processed/patient_timelines_verified.json \
    --qa-data data/physionet.org/files/ehr-ds-qa/1.0.0/mimic_iv_note_qa_verified.json \
    --output-dir data/results/verified \
    --limit 70 \
    --token-budget 4096
