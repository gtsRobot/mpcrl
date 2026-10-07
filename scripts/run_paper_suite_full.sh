#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
export MPLBACKEND=Agg

TAG="${TAG:-$(date +%Y%m%d_%H%M%S)}"
START_JOB="${START_JOB:-1}"
BASE_SUITE="${BASE_SUITE:-configs/rtx5090/paper_suite.yaml}"
RUN_ROOT="runs/paper_suite_${TAG}"
RESULT_ROOT="results/paper_suite_${TAG}"
LOCAL_SUITE="${RESULT_ROOT}/paper_suite_local.yaml"
LOG="${RESULT_ROOT}/paper_suite.log"

mkdir -p "$RUN_ROOT" "$RESULT_ROOT/configs"

"$PYTHON" - "$BASE_SUITE" "$RUN_ROOT" "$RESULT_ROOT" "$LOCAL_SUITE" <<'PY'
import sys
from pathlib import Path
from mpcrl.config import load_yaml, save_yaml

base_suite, run_root, result_root, local_suite = sys.argv[1:]
suite = load_yaml(base_suite)
out_cfg = Path(result_root) / 'configs'
out_cfg.mkdir(parents=True, exist_ok=True)

local_tasks = []
for src in suite['tasks']:
    cfg = load_yaml(src)
    cfg.setdefault('logging', {})['root'] = run_root
    dst = out_cfg / Path(src).name
    save_yaml(cfg, dst)
    local_tasks.append(str(dst))

local_lewm = []
for src in suite.get('lewm_configs', []):
    cfg = load_yaml(src)
    cfg.setdefault('logging', {})['root'] = run_root
    dst = out_cfg / Path(src).name
    save_yaml(cfg, dst)
    local_lewm.append(str(dst))

suite['tasks'] = local_tasks
suite['lewm_configs'] = local_lewm
suite['run_lewm'] = True
save_yaml(suite, local_suite)
print('local suite:', local_suite)
print('run root:', run_root)
PY

{
  echo "TAG=$TAG"
  echo "GIT_COMMIT=$(git rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "BASE_SUITE=$BASE_SUITE"
  echo "START_JOB=$START_JOB"
  "$PYTHON" --version 2>&1
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || true
} > "${RESULT_ROOT}/environment.txt"

run_logged() {
  echo -e "\n>>> $*" | tee -a "$LOG"
  "$@" 2>&1 | tee -a "$LOG"
}

run_logged "$PYTHON" -m unittest discover -s tests -v
run_logged "$PYTHON" -m experiments.run_suite \
  --suite "$LOCAL_SUITE" \
  --include-lewm \
  --start-job "$START_JOB"

ENVS=(Hopper-v5 Walker2d-v5 HalfCheetah-v5 Reacher-v5)
CORE_METHODS=(mpc_only action_residual planning_residual zprl_style)
ALL_METHODS=(mpc_only action_residual planning_residual zprl_style lewm_mpc)

ENV_ARGS=()
for env in "${ENVS[@]}"; do ENV_ARGS+=(--env "$env"); done

run_logged "$PYTHON" -m experiments.aggregate \
  --root "$RUN_ROOT" \
  --out "${RESULT_ROOT}/main_summary.csv" \
  "${ENV_ARGS[@]}" \
  --methods "${CORE_METHODS[@]}" \
  --min-steps 90000 \
  --require-eval \
  --expected-seeds 0 1 2 3 4 \
  --require-complete \
  --tail 20

run_logged "$PYTHON" -m experiments.summarize_eval \
  --root "$RUN_ROOT" \
  --out "${RESULT_ROOT}/final_eval" \
  "${ENV_ARGS[@]}" \
  --methods "${ALL_METHODS[@]}" \
  --min-steps 0 \
  --expected-seeds 0 1 2 3 4 \
  --require-complete

run_logged "$PYTHON" -m experiments.plot_results \
  --root "$RUN_ROOT" \
  --outdir "${RESULT_ROOT}/figures" \
  "${ENV_ARGS[@]}" \
  --methods "${CORE_METHODS[@]}" \
  --min-steps 90000 \
  --expected-methods "${CORE_METHODS[@]}" \
  --expected-seeds 0 1 2 3 4 \
  --require-complete \
  --smooth-window 10 \
  --points 400 \
  --error-band std

run_logged "$PYTHON" -m experiments.paper_figures \
  --summary "${RESULT_ROOT}/main_summary_per_seed.csv" \
  --runs "$RUN_ROOT" \
  --out "${RESULT_ROOT}/paper_figures" \
  --smooth-episodes 20

echo
echo "Paper suite completed."
echo "Runs:          $RUN_ROOT"
echo "Results:       $RESULT_ROOT"
echo "Comparison:    $RESULT_ROOT/final_eval/comparison_eval.csv"
echo "Figures:       $RESULT_ROOT/figures"
echo "Paper figures: $RESULT_ROOT/paper_figures"
