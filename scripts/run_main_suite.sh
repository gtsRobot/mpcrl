#!/usr/bin/env bash
# Full 4-method paper suite. Survives SSH disconnects; one failing job no longer
# aborts the rest (see experiments/run_suite.py --keep-going).
#
# NOTE this suite is UNPAIRED -- each method trains its own world model -- so it
# gives absolute levels per method but is not the right instrument for claiming
# "the residual beats MPC by X%". Use scripts/run_paired_suite.sh for that.
#
# Usage:
#   bash scripts/run_main_suite.sh
#   START_JOB=37 bash scripts/run_main_suite.sh        # resume a killed suite
#   MPCRL_FOREGROUND=1 bash scripts/run_main_suite.sh  # do not detach
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
mpcrl::self_detach "$@"
mpcrl::preflight

SUITE="${SUITE:-configs/rtx5090/paper_suite.yaml}"
START_JOB="${START_JOB:-1}"

"$PYTHON" -m experiments.run_suite --suite "$SUITE" --start-job "$START_JOB" || \
  echo "[mpcrl] some jobs failed; see results/suite_failures.csv"

"$PYTHON" -m experiments.aggregate --root runs --out results/summary.csv --tail 20
"$PYTHON" -m experiments.plot_results --root runs --outdir results/figures
echo "[mpcrl] done. summary: results/summary.csv  figures: results/figures"
