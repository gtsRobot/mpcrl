#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
STEPS="${STEPS:-100000}"
SEEDS="${SEEDS:-0 1 2 3 4}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"

for CFG in configs/rtx5090/refine/halfcheetah_planning.yaml configs/rtx5090/refine/hopper_planning.yaml; do
  ENV_NAME="$(basename "$CFG" _planning.yaml)"
  for SEED in $SEEDS; do
    echo "=== refine ${ENV_NAME} planning_residual seed=${SEED} ==="
    "$PYTHON" -m experiments.train \
      --config "$CFG" \
      --method planning_residual \
      --steps "$STEPS" \
      --seed "$SEED" \
      --eval-episodes 5 \
      --run-name "refine__${ENV_NAME}__planning_residual__seed${SEED}__${STAMP}"
  done
done

echo "Refinement runs finished. Re-run aggregate and paper_figures afterwards."
