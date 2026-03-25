#!/usr/bin/env bash

set -euo pipefail

VOLUME_NAME="${VOLUME_NAME:-flashinfer-trace}"
DATASET_PATH="${DATASET_PATH:-/home/yuwei/Documents/Personal-projects/Kernel-Competition/mlsys26-contest}"
REMOTE_PATH="${REMOTE_PATH:-/}"

if ! command -v modal >/dev/null 2>&1; then
  echo "modal CLI is not installed or not on PATH" >&2
  exit 1
fi

if [ ! -d "${DATASET_PATH}" ]; then
  echo "Dataset path does not exist: ${DATASET_PATH}" >&2
  exit 1
fi

echo "Uploading dataset from '${DATASET_PATH}' to Modal volume '${VOLUME_NAME}'..."
modal volume put "${VOLUME_NAME}" "${DATASET_PATH}/" "${REMOTE_PATH}"

echo
echo "Listing Modal volume contents for verification..."
modal volume ls "${VOLUME_NAME}"
