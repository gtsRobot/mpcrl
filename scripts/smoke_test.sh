#!/usr/bin/env bash
# Minimal GPU + pipeline validation. Run this first on a new machine.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
mpcrl::preflight

MPCRL_FOREGROUND=1

"$PYTHON" scripts/check_rtx5090.py
"$PYTHON" -m unittest discover -s tests -v

echo
echo "=== state-space smoke (all three methods) ==="
for method in mpc_only action_residual planning_residual; do
  "$PYTHON" -m experiments.train \
    --config configs/local/halfcheetah.yaml \
    --method "$method" --steps 1200 --eval-episodes 0 \
    --run-name "smoke_${method}" --no-progress
done

echo
echo "=== paired protocol smoke ==="
"$PYTHON" -m experiments.train_paired \
  --config configs/local/halfcheetah.yaml \
  --steps 3000 --eval-episodes 1 --run-name smoke_paired --no-progress

echo
echo "=== LeWM (ViT + AR predictor) smoke ==="
"$PYTHON" -m experiments.train_lewm_mpc_rl \
  --config configs/local/lewm_halfcheetah.yaml \
  --rl-steps 300 --eval-episodes 1 --run-root runs/smoke_lewm

echo
echo "[mpcrl] smoke test PASSED"
