#!/usr/bin/env bash

set -euo pipefail

DATASET_REPO="${DATASET_REPO:-https://huggingface.co/datasets/flashinfer-ai/mlsys26-contest}"
DATASET_PARENT="${DATASET_PARENT:-/home/yuwei/Documents/Personal-projects/Kernel-Competition}"
DATASET_DIR_NAME="${DATASET_DIR_NAME:-mlsys26-contest}"
DATASET_PATH="${DATASET_PATH:-${DATASET_PARENT}/${DATASET_DIR_NAME}}"

if ! command -v git >/dev/null 2>&1; then
  echo "git is not installed or not on PATH" >&2
  exit 1
fi

if ! git lfs version >/dev/null 2>&1; then
  echo "git-lfs is not installed or not on PATH" >&2
  exit 1
fi

mkdir -p "${DATASET_PARENT}"

echo "Installing git-lfs hooks..."
git lfs install

if [ -d "${DATASET_PATH}/.git" ]; then
  echo "Dataset already exists at '${DATASET_PATH}', reusing it."
else
  echo "Cloning dataset into '${DATASET_PATH}'..."
  git clone "${DATASET_REPO}" "${DATASET_PATH}"
fi

echo
echo "Dataset setup complete."
echo "Set the following environment variable before running benchmarks:"
echo "export FIB_DATASET_PATH=${DATASET_PATH}"
