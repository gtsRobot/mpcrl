#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
export MPLBACKEND=Agg

TAG="${TAG:-$(date +%Y%m%d_%H%M%S)}"
START_JOB="${START_JOB:-1}"
SUITE="${SUITE:-configs/rtx5090/lewm_mpc_rl_suite.yaml}"
RUN_ROOT="runs/lewm_mpc_rl_${TAG}"
RESULT_ROOT="results/lewm_mpc_rl_${TAG}"
LOG="${RESULT_ROOT}/full_run.log"

mkdir -p "$RUN_ROOT" "$RESULT_ROOT"

{
  echo "TAG=$TAG"
  echo "GIT_COMMIT=$(git rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "SUITE=$SUITE"
  echo "START_JOB=$START_JOB"
  "$PYTHON" --version 2>&1
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || true
} > "${RESULT_ROOT}/environment.txt"

run_logged() {
  echo -e "\n>>> $*" | tee -a "$LOG"
  "$@" 2>&1 | tee -a "$LOG"
}

run_logged "$PYTHON" -m unittest discover -s tests -v
run_logged "$PYTHON" -m experiments.run_lewm_mpc_rl_suite \
  --suite "$SUITE" \
  --run-root "$RUN_ROOT" \
  --start-job "$START_JOB"

run_logged "$PYTHON" -m experiments.summarize_lewm_mpc_rl \
  --root "$RUN_ROOT" \
  --outdir "$RESULT_ROOT/summary" \
  --envs Hopper-v5 Walker2d-v5 HalfCheetah-v5 Reacher-v5 \
  --seeds 0 1 2 3 4 \
  --require-complete

run_logged "$PYTHON" -m experiments.plot_lewm_mpc_rl \
  --root "$RUN_ROOT" \
  --summary-dir "$RESULT_ROOT/summary" \
  --outdir "$RESULT_ROOT/figures" \
  --points 300 \
  --smooth-window 5

echo
echo "============================================================"
echo "Direct LeWM-MPC vs LeWM-MPC+RL experiment completed."
echo "Runs:      $RUN_ROOT"
echo "Results:   $RESULT_ROOT"
echo "Main table:$RESULT_ROOT/summary/direct_summary.csv"
echo "Paired:    $RESULT_ROOT/summary/paired_summary.csv"
echo "Coverage:  $RESULT_ROOT/summary/direct_coverage.csv"
echo "Figures:   $RESULT_ROOT/figures"
echo "============================================================"
