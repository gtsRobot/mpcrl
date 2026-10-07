#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
"$PYTHON" -m experiments.quick_diagnostic \
  --config configs/rtx5090/halfcheetah.yaml \
  --steps 10000 \
  --seeds 0
