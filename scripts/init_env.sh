#!/usr/bin/env bash

set -euo pipefail

ENV_NAME="${ENV_NAME:-fi-bench}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is not installed or not on PATH" >&2
  exit 1
fi

CONDA_BASE="$(conda info --base)"
source "${CONDA_BASE}/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -Fxq "${ENV_NAME}"; then
  echo "Conda environment '${ENV_NAME}' already exists, reusing it."
else
  echo "Creating conda environment '${ENV_NAME}' with Python ${PYTHON_VERSION}..."
  conda create -y -n "${ENV_NAME}" "python=${PYTHON_VERSION}"
fi

echo "Activating '${ENV_NAME}' and installing required packages..."
conda activate "${ENV_NAME}"
pip install flashinfer-bench modal

echo "Environment setup complete."
echo "Environment name: ${ENV_NAME}"
