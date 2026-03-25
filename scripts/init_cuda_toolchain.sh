#!/usr/bin/env bash

set -euo pipefail

CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"

echo "Checking local CUDA toolchain..."

if command -v nvcc >/dev/null 2>&1; then
  echo "Found nvcc on PATH:"
  nvcc --version
else
  echo "nvcc was not found on PATH."
  echo "Please install the NVIDIA CUDA Toolkit for your operating system."
  echo "After installation, ensure nvcc is available on PATH or set CUDA_HOME explicitly."
  exit 1
fi

if [ -d "${CUDA_HOME}" ]; then
  echo "Found CUDA_HOME at '${CUDA_HOME}'."
else
  echo "CUDA_HOME directory not found at '${CUDA_HOME}'."
  echo "If CUDA is installed elsewhere, rerun with CUDA_HOME=/path/to/cuda."
fi

if [ -f "${CUDA_HOME}/include/cuda_runtime.h" ]; then
  echo "Found CUDA headers."
else
  echo "Could not find '${CUDA_HOME}/include/cuda_runtime.h'."
  echo "Please verify the CUDA development headers are installed."
fi

if [ -d "${CUDA_HOME}/lib64" ]; then
  echo "Found CUDA libraries under '${CUDA_HOME}/lib64'."
else
  echo "Could not find '${CUDA_HOME}/lib64'."
  echo "Please verify the CUDA development libraries are installed."
fi

echo
echo "CUDA toolchain check complete."
echo "This script is intended as a one-time local setup helper before using local_compile."
