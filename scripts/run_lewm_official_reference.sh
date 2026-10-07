#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
# Strict LeWorldModel reference verification on the official benchmark.
# This script evaluates the official LeWM checkpoints only; it does NOT claim
# that the state-space ZPRL implementation in this repository has been ported
# to the official pixel-goal benchmark.
#
# Usage:
#   LEWM_OFFICIAL_DIR=/path/to/LeWorldModel \
#   STABLEWM_HOME=/path/to/.stable-wm \
#   bash scripts/run_lewm_official_reference.sh


: "${LEWM_OFFICIAL_DIR:?Set LEWM_OFFICIAL_DIR to the official LeWorldModel repository}"
: "${STABLEWM_HOME:?Set STABLEWM_HOME to the directory containing official checkpoints/data}"

PYTHON_BIN="${PYTHON_BIN:-python}"
OUT="${OUT:-results/lewm_official_reference_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUT"

if [[ ! -f "$LEWM_OFFICIAL_DIR/eval.py" ]]; then
  echo "Missing $LEWM_OFFICIAL_DIR/eval.py"
  exit 2
fi

export STABLEWM_HOME

run_task() {
  local config_name="$1"
  local policy="$2"
  local log="$OUT/${config_name%.yaml}.log"
  echo "=== Official LeWM: ${config_name} | ${policy} ===" | tee "$log"
  (
    cd "$LEWM_OFFICIAL_DIR"
    "$PYTHON_BIN" eval.py --config-name="$config_name" policy="$policy"
  ) 2>&1 | tee -a "$log"
}

run_task tworoom.yaml tworoom/lewm
run_task pusht.yaml pusht/lewm
run_task cube.yaml cube/lewm
run_task reacher.yaml reacher/lewm

cat > "$OUT/README.txt" <<EOF
Official LeWorldModel reference evaluation completed.
Tasks: TwoRoom, PushT, Cube, Reacher.
Repository: $LEWM_OFFICIAL_DIR
STABLEWM_HOME: $STABLEWM_HOME

These logs verify the official LeWM reference checkpoints. They are not results
for this repository's state-space ZPRL method and should be reported separately
from the unified MuJoCo engineering benchmark.
EOF

echo "Official reference logs saved to: $OUT"
