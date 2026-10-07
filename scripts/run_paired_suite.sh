#!/usr/bin/env bash
# Paired MPC vs MPC+residual protocol across seeds, then the seed-level summary.
#
# This is the experiment that can actually detect the effect. The unpaired
# 4-method suite cannot: HalfCheetah mpc_only is 2293 +/- 1054 across 5 seeds, so
# an unpaired 5-seed 95% CI is about +/-990 on effects of order 10-120.
#
# n >= 6 is deliberate: the exact two-sided Wilcoxon floor at n=5 is 2/2^5 =
# 0.0625, so a 5-seed study cannot reach p<0.05 regardless of effect size.
#
# Usage:
#   bash scripts/run_paired_suite.sh
#   CONFIG=configs/rtx5090/hopper.yaml SEEDS="0 1 2 3 4 5" bash scripts/run_paired_suite.sh
#   MPCRL_FOREGROUND=1 bash scripts/run_paired_suite.sh      # do not detach
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
mpcrl::self_detach "$@"
mpcrl::preflight

CONFIG="${CONFIG:-configs/rtx5090/halfcheetah.yaml}"
SEEDS="${SEEDS:-0 1 2 3 4 5 6 7}"
STEPS="${STEPS:-100000}"
EVAL_EPISODES="${EVAL_EPISODES:-10}"
TAG="${TAG:-paired_$(date +%Y%m%d_%H%M%S)}"
OUT="results/${TAG}.csv"

# Deliberately NOT using `set -e` semantics per-seed: one failed seed must not
# discard the seeds that already finished.
for seed in $SEEDS; do
  echo "=== paired seed ${seed} ==="
  if ! "$PYTHON" -m experiments.train_paired \
      --config "$CONFIG" \
      --seed "$seed" \
      --steps "$STEPS" \
      --eval-episodes "$EVAL_EPISODES" \
      --run-name "${TAG}__seed${seed}" \
      --no-progress; then
    echo "[mpcrl] seed ${seed} FAILED; continuing with the remaining seeds"
  fi
done

"$PYTHON" -m experiments.summarize_paired --root runs --out "$OUT"
echo "[mpcrl] paired summary: $OUT"
