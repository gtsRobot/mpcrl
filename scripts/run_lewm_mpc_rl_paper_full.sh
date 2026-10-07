#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
export MPLBACKEND=Agg

TAG="${TAG:-$(date +%Y%m%d_%H%M%S)}"
START_JOB="${START_JOB:-1}"
SUITE="${SUITE:-configs/rtx5090/lewm_mpc_rl_suite.yaml}"
RUN_ROOT="runs/lewm_mpc_rl_${TAG}"
RESULT_ROOT="results/lewm_mpc_rl_${TAG}"

mkdir -p "$RUN_ROOT" "$RESULT_ROOT"
{
  echo "TAG=$TAG"
  echo "GIT_COMMIT=$(git rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "SUITE=$SUITE"
  echo "START_JOB=$START_JOB"
  "$PYTHON" --version 2>&1
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || true
} > "${RESULT_ROOT}/environment.txt"

"$PYTHON" -m unittest discover -s tests -v
"$PYTHON" -m experiments.run_lewm_mpc_rl_suite \
  --suite "$SUITE" \
  --run-root "$RUN_ROOT" \
  --results-dir "$RESULT_ROOT" \
  --start-job "$START_JOB"

echo
echo "Direct paired LeWM paper suite completed."
echo "Runs:          $RUN_ROOT"
echo "Summary:       $RESULT_ROOT/summary"
echo "Paper figures: $RESULT_ROOT/paper_figures"
