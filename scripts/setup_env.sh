#!/usr/bin/env bash
# One-time environment setup for farmshare.
#
# Run this once from the repo root on a login node:
#   bash scripts/setup_env.sh
#
# What it does:
#   1. Redirects HuggingFace cache to /scratch (avoids home quota exhaustion)
#   2. Loads micromamba module
#   3. Creates (or updates) the 'ehr' conda environment from requirements.txt

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRATCH_HF="/scratch/users/${USER}/hf_cache"
ENV_NAME="ehr"

# ── 1. Redirect HuggingFace downloads to scratch ─────────────────────────────
mkdir -p "${SCRATCH_HF}"
export HF_HOME="${SCRATCH_HF}"
export HUGGINGFACE_HUB_CACHE="${SCRATCH_HF}/hub"
export TRANSFORMERS_CACHE="${SCRATCH_HF}/hub"

echo "HF_HOME set to ${SCRATCH_HF}"

# Persist the redirect in ~/.bashrc so it survives future sessions
BASHRC="${HOME}/.bashrc"
if ! grep -q "HF_HOME=${SCRATCH_HF}" "${BASHRC}" 2>/dev/null; then
    cat >> "${BASHRC}" <<EOF

# HuggingFace model cache → scratch (added by EHR setup_env.sh)
export HF_HOME="${SCRATCH_HF}"
export HUGGINGFACE_HUB_CACHE="${SCRATCH_HF}/hub"
export TRANSFORMERS_CACHE="${SCRATCH_HF}/hub"
EOF
    echo "Appended HF_HOME export to ${BASHRC}"
fi

# ── 2. Load micromamba ────────────────────────────────────────────────────────
module load micromamba 2>/dev/null || {
    echo "ERROR: 'module load micromamba' failed. Are you on a farmshare node?"
    exit 1
}

# ── 3. Create or update the 'ehr' environment ────────────────────────────────
if micromamba env list | grep -q "^${ENV_NAME} "; then
    echo "Environment '${ENV_NAME}' already exists — installing/updating packages."
    micromamba run -n "${ENV_NAME}" pip install -r "${REPO_ROOT}/requirements.txt" --quiet
else
    echo "Creating environment '${ENV_NAME}' with Python 3.11 …"
    micromamba create -n "${ENV_NAME}" python=3.11 pip -c conda-forge -y
    micromamba run -n "${ENV_NAME}" pip install -r "${REPO_ROOT}/requirements.txt" --quiet
fi

echo ""
echo "Setup complete. To activate:"
echo "  module load micromamba && micromamba activate ${ENV_NAME}"
echo ""
echo "Or in batch scripts use:"
echo "  module load micromamba"