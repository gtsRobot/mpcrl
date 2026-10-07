#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
"$PYTHON" -m experiments.aggregate --root "${RUNS_ROOT:-runs}" --out "${SUMMARY:-results/summary.csv}"
"$PYTHON" -m experiments.paper_figures \
  --summary "${SUMMARY_PER_SEED:-results/summary_per_seed.csv}" \
  --runs "${RUNS_ROOT:-runs}" \
  --out "${FIG_OUT:-results/paper_figures}" \
  --smooth-episodes "${SMOOTH_EPISODES:-20}"
