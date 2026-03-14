#!/usr/bin/env bash
# One-time environment setup for FarmShare.
#
# Run this once from the repo root on a login node:
#   bash scripts/setup_env.sh
#
# What it does:
#   1. Redirects HuggingFace cache and pip cache to /scratch (avoids home quota)
#   2. Creates (or updates) the 'ehr' conda environment from requirements.txt
#
# Note: FarmShare does not have a micromamba module. We use the conda install
# at ~/miniconda3 (or $CONDA_EXE if already on PATH). The env is placed under
# /scratch to avoid home-directory quota issues.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRATCH_HF="/scratch/users/${USER}/hf_cache"
SCRATCH_PIP="/scratch/users/${USER}/pip_cache"
ENV_PREFIX="/scratch/users/${USER}/envs/ehr"
CONDA="${CONDA_EXE:-${HOME}/miniconda3/bin/conda}"

if [[ ! -x "${CONDA}" ]]; then
    echo "ERROR: conda not found at ${CONDA}. Install miniconda3 first."
    exit 1
fi

# ── 1. Redirect caches to scratch ────────────────────────────────────────────
mkdir -p "${SCRATCH_HF}" "${SCRATCH_PIP}"
export HF_HOME="${SCRATCH_HF}"
export HUGGINGFACE_HUB_CACHE="${SCRATCH_HF}/hub"
export TRANSFORMERS_CACHE="${SCRATCH_HF}/hub"
export PIP_CACHE_DIR="${SCRATCH_PIP}"

echo "HF_HOME set to ${SCRATCH_HF}"
echo "PIP_CACHE_DIR set to ${SCRATCH_PIP}"

BASHRC="${HOME}/.bashrc"
if ! grep -q "HF_HOME=${SCRATCH_HF}" "${BASHRC}" 2>/dev/null; then
    cat >> "${BASHRC}" <<EOF

# EHR project cache redirects (added by setup_env.sh)
export HF_HOME="${SCRATCH_HF}"
export HUGGINGFACE_HUB_CACHE="${SCRATCH_HF}/hub"
export TRANSFORMERS_CACHE="${SCRATCH_HF}/hub"
export PIP_CACHE_DIR="${SCRATCH_PIP}"
EOF
    echo "Appended cache exports to ${BASHRC}"
fi

# ── 2. Create or update the conda environment ────────────────────────────────
if "${CONDA}" env list | grep -q "${ENV_PREFIX}"; then
    echo "Environment at ${ENV_PREFIX} already exists — updating packages."
    "${CONDA}" run --prefix "${ENV_PREFIX}" pip install -r "${REPO_ROOT}/requirements.txt" --quiet
else
    echo "Creating environment at ${ENV_PREFIX} with Python 3.11 …"
    "${CONDA}" create --prefix "${ENV_PREFIX}" python=3.11 pip -y
    "${CONDA}" run --prefix "${ENV_PREFIX}" pip install -r "${REPO_ROOT}/requirements.txt" --quiet
fi

echo ""
echo "Setup complete. To activate:"
echo "  ${CONDA} activate ${ENV_PREFIX}"
echo ""
echo "Or run directly:"
echo "  ${CONDA} run --prefix ${ENV_PREFIX} python -m Evaluation.run_evaluation ..."