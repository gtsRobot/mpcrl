#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
ENV_NAME="${1:-mpc_rl_zprl_5090}"

echo "[1/4] Create Conda environment: ${ENV_NAME}"
conda create -n "${ENV_NAME}" python=3.11 -y

echo "[2/4] Install RTX 5090 compatible PyTorch (CUDA 12.8 wheel)"
eval "$(conda shell.bash hook)"
conda activate "${ENV_NAME}"
"$PYTHON" -m pip install --upgrade pip
"$PYTHON" -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

echo "[3/4] Install project dependencies"
"$PYTHON" -m pip install -r requirements.txt

echo "[4/4] Verify GPU"
"$PYTHON" scripts/check_rtx5090.py

echo "Done. Activate later with: conda activate ${ENV_NAME}"
