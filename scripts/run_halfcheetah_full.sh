#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
# One-click HalfCheetah paper experiment:
#   4 methods x 5 seeds x 100k steps -> deterministic eval -> aggregation -> figures.
#
# Normal use:
#   bash scripts/run_halfcheetah_full.sh
#
# Resume after an interruption (reuse the printed TAG):
#   TAG=20260904_120000 START_JOB=8 bash scripts/run_halfcheetah_full.sh
#
# Optional overrides:
#   STEPS=100000 EVAL_EPISODES=5 SEEDS="0 1 2 3 4" bash scripts/run_halfcheetah_full.sh

export MPLBACKEND=Agg

TAG="${TAG:-$(date +%Y%m%d_%H%M%S)}"
STEPS="${STEPS:-100000}"
EVAL_EPISODES="${EVAL_EPISODES:-5}"
SEEDS="${SEEDS:-0 1 2 3 4}"
START_JOB="${START_JOB:-1}"

RUN_ROOT="runs/halfcheetah_full_${TAG}"
RESULT_ROOT="results/halfcheetah_full_${TAG}"
CFG="${RESULT_ROOT}/halfcheetah_full.yaml"
LOG="${RESULT_ROOT}/full_run.log"

mkdir -p "$RUN_ROOT" "$RESULT_ROOT"

# Make an experiment-local config so old 10k/30k diagnostics can never be
# selected by the final aggregation/plotting scripts.
"$PYTHON" - "$RUN_ROOT" "$CFG" <<'PY'
import sys
from mpcrl.config import load_yaml, save_yaml

run_root, cfg_out = sys.argv[1], sys.argv[2]
cfg = load_yaml('configs/rtx5090/halfcheetah.yaml')
cfg['logging']['root'] = run_root
save_yaml(cfg, cfg_out)
print('experiment config:', cfg_out)
print('run root:', run_root)
PY

{
  echo "TAG=$TAG"
  echo "STEPS=$STEPS"
  echo "EVAL_EPISODES=$EVAL_EPISODES"
  echo "SEEDS=$SEEDS"
  echo "GIT_COMMIT=$(git rev-parse HEAD 2>/dev/null || echo unknown)"
  "$PYTHON" --version 2>&1
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || true
} > "${RESULT_ROOT}/environment.txt"

run_logged() {
  echo -e "\n>>> $*" | tee -a "$LOG"
  "$@" 2>&1 | tee -a "$LOG"
}

# Fail fast on source-level regressions before spending GPU-hours.
run_logged "$PYTHON" -m unittest discover -s tests -v

methods=(mpc_only action_residual planning_residual zprl_style)
read -r -a seed_array <<< "$SEEDS"
job=0
total=$((${#methods[@]} * ${#seed_array[@]}))

for seed in "${seed_array[@]}"; do
  for method in "${methods[@]}"; do
    job=$((job + 1))
    if (( job < START_JOB )); then
      echo "SKIP job ${job}/${total}: seed=${seed} method=${method}" | tee -a "$LOG"
      continue
    fi

    echo -e "\n=== JOB ${job}/${total}: HalfCheetah-v5 | ${method} | seed=${seed} ===" | tee -a "$LOG"
    if [[ "$method" == "zprl_style" ]]; then
      run_logged "$PYTHON" -m experiments.train_zprl_style \
        --config "$CFG" \
        --steps "$STEPS" \
        --seed "$seed" \
        --eval-episodes "$EVAL_EPISODES"
    else
      run_logged "$PYTHON" -m experiments.train \
        --config "$CFG" \
        --method "$method" \
        --steps "$STEPS" \
        --seed "$seed" \
        --eval-episodes "$EVAL_EPISODES"
    fi
  done
done

# Final paper statistics. aggregate.py prioritizes deterministic eval.csv and
# keeps one longest/newest run per (env, method, seed).
run_logged "$PYTHON" -m experiments.aggregate \
  --root "$RUN_ROOT" \
  --out "${RESULT_ROOT}/summary.csv" \
  --tail 20

# Compact deterministic-evaluation comparison, including improvement vs MPC.
run_logged "$PYTHON" -m experiments.summarize_eval \
  --glob "${RUN_ROOT}/*" \
  --out "${RESULT_ROOT}/eval_summary"

# Multi-seed paper plots: each seed is smoothed independently, aligned by
# environment step, then mean +/- std is computed. PNG 300 dpi + PDF are saved.
MIN_STEPS=$((STEPS * 9 / 10))
run_logged "$PYTHON" -m experiments.plot_results \
  --root "$RUN_ROOT" \
  --outdir "${RESULT_ROOT}/figures" \
  --min-steps "$MIN_STEPS" \
  --smooth-window 10 \
  --points 400

echo -e "\nHalfCheetah full experiment completed." | tee -a "$LOG"
echo "Runs:     $RUN_ROOT" | tee -a "$LOG"
echo "Results:  $RESULT_ROOT" | tee -a "$LOG"
echo "Summary:  ${RESULT_ROOT}/eval_summary/comparison_eval.csv" | tee -a "$LOG"
echo "Figures:  ${RESULT_ROOT}/figures" | tee -a "$LOG"
