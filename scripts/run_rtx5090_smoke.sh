#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
"$PYTHON" scripts/check_rtx5090.py
"$PYTHON" -m unittest discover -s tests -v
"$PYTHON" -m experiments.train --config configs/rtx5090/halfcheetah.yaml --method planning_residual --steps 2000 --run-name rtx5090_smoke
