#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
export MPLBACKEND=Agg

TAG="${TAG:-$(date +%Y%m%d_%H%M%S)}"
STEPS="${STEPS:-100000}"
EVAL_EPISODES="${EVAL_EPISODES:-5}"
SEEDS="${SEEDS:-0 1 2 3 4}"
START_JOB="${START_JOB:-1}"

RUN_ROOT="runs/walker2d_full_${TAG}"
RESULT_ROOT="results/walker2d_full_${TAG}"
CFG="${RESULT_ROOT}/walker2d_full.yaml"
LOG="${RESULT_ROOT}/full_run.log"

mkdir -p "$RUN_ROOT" "$RESULT_ROOT"

"$PYTHON" - "$RUN_ROOT" "$CFG" <<'PY'
import sys
from mpcrl.config import load_yaml, save_yaml
run_root, cfg_out = sys.argv[1], sys.argv[2]
cfg = load_yaml('configs/rtx5090/walker2d.yaml')
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

run_logged "$PYTHON" -m unittest discover -s tests -v

methods=(mpc_only action_residual planning_residual zprl_style)
job=0
total=20

for seed in $SEEDS; do
  for method in "${methods[@]}"; do
    job=$((job + 1))
    if (( job < START_JOB )); then
      echo "SKIP job ${job}/${total}: seed=${seed} method=${method}" | tee -a "$LOG"
      continue
    fi

    echo -e "\n=== JOB ${job}/${total}: Walker2d-v5 | ${method} | seed=${seed} ===" | tee -a "$LOG"

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

run_logged "$PYTHON" -m experiments.aggregate \
  --root "$RUN_ROOT" \
  --out "${RESULT_ROOT}/summary.csv" \
  --env Walker2d-v5 \
  --methods mpc_only action_residual planning_residual zprl_style \
  --min-steps 90000 \
  --require-eval \
  --expected-seeds 0 1 2 3 4 \
  --require-complete \
  --tail 20

run_logged "$PYTHON" -m experiments.summarize_eval \
  --root "$RUN_ROOT" \
  --out "${RESULT_ROOT}/eval_summary" \
  --env Walker2d-v5 \
  --methods mpc_only action_residual planning_residual zprl_style \
  --min-steps 90000 \
  --expected-seeds 0 1 2 3 4 \
  --require-complete

run_logged "$PYTHON" -m experiments.plot_results \
  --root "$RUN_ROOT" \
  --outdir "${RESULT_ROOT}/figures" \
  --env Walker2d-v5 \
  --methods mpc_only action_residual planning_residual zprl_style \
  --min-steps 90000 \
  --expected-methods mpc_only action_residual planning_residual zprl_style \
  --expected-seeds 0 1 2 3 4 \
  --require-complete \
  --smooth-window 10 \
  --points 400 \
  --error-band std

echo -e "\nWalker2d full experiment completed." | tee -a "$LOG"
echo "Runs:     $RUN_ROOT" | tee -a "$LOG"
echo "Results:  $RESULT_ROOT" | tee -a "$LOG"
echo "Summary:  ${RESULT_ROOT}/eval_summary/comparison_eval.csv" | tee -a "$LOG"
echo "Figures:  ${RESULT_ROOT}/figures" | tee -a "$LOG"
