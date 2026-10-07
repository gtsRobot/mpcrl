#!/usr/bin/env bash
set -e
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
for seed in 0 1 2 3 4; do
  "$PYTHON" -m experiments.ablation --config configs/halfcheetah.yaml --steps 100000 --seed "$seed"
done
