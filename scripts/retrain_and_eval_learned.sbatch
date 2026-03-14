#!/usr/bin/env bash
#SBATCH --job-name=ehr-retrain-learned
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=logs/retrain_learned_%j.out
#SBATCH --error=logs/retrain_learned_%j.err
#
# Retrain the logreg model (to match current 166-feature extractor),
# then run the "learned" strategy for ALL 5 models on the verified QA set.
#
# Usage:
#   sbatch scripts/retrain_and_eval_learned.sbatch

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ENV_PREFIX="/scratch/users/${USER}/envs/ehr"

mkdir -p "${REPO_ROOT}/logs"

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

export PATH="${ENV_PREFIX}/bin:${PATH}"
cd "${REPO_ROOT}"

echo "=============================="
echo "Job ID : ${SLURM_JOB_ID}"
echo "Node   : $(hostname)"
echo "GPU    : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "=============================="

echo ""
echo "=== Phase 1: Retrain logreg model (160 -> 166 features) ==="
echo ""
python -m Logreg.run train --qa-source generated

echo ""
echo "=== Phase 2: Run learned strategy for all 5 models ==="
echo ""

COMMON_ARGS="--method learned --timelines data/processed/patient_timelines_verified.json --qa-data data/physionet.org/files/ehr-ds-qa/1.0.0/mimic_iv_note_qa_verified.json --output-dir data/results/verified --limit 70 --token-budget 4096"

for MODEL in o4-mini gpt-4o-mini; do
    echo "--- Running learned for ${MODEL} (OpenAI API) ---"
    python -m Evaluation.run_evaluation --model "${MODEL}" ${COMMON_ARGS}
done

for MODEL in "meta-llama/Llama-3.1-8B-Instruct" "mistralai/Mistral-7B-Instruct-v0.3" "microsoft/Phi-3-mini-4k-instruct"; do
    echo "--- Running learned for ${MODEL} (HF) ---"
    python -m Evaluation.run_evaluation --model "${MODEL}" ${COMMON_ARGS}
done

echo ""
echo "=== All learned evaluations complete ==="
