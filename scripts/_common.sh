#!/usr/bin/env bash
# Shared helpers for every script in this directory.
#
# Sourced, not executed. Fixes three things that repeatedly cost whole runs:
#   1. Scripts assumed the repo root was the CWD.
#   2. A bare `python` picked up whatever happened to be on PATH.
#   3. Nothing detached, so an SSH disconnect SIGHUP'd multi-hour jobs.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-python}"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/logs}"
mkdir -p "$LOG_DIR"

# MuJoCo offscreen rendering on a headless box. mpcrl/envs.py also sets this,
# but exporting here covers any subprocess that imports mujoco first.
if [[ -z "${MUJOCO_GL:-}" && -z "${DISPLAY:-}" ]]; then
  export MUJOCO_GL=egl
fi

# Keep Python's stdout usable when the locale is C/POSIX and output is redirected.
export PYTHONUNBUFFERED=1
export PYTHONIOENCODING="${PYTHONIOENCODING:-utf-8}"

mpcrl::log_path() {
  printf '%s/%s_%s.log' "$LOG_DIR" "$1" "$(date +%Y%m%d_%H%M%S)"
}

# Re-exec THIS script detached from the terminal, so an SSH disconnect cannot
# SIGHUP a multi-hour job. Call it once near the top, after sourcing this file:
#
#   mpcrl::self_detach "$@"
#
# The re-exec sets MPCRL_DETACHED=1 so the second pass falls through and runs the
# real body. Set MPCRL_FOREGROUND=1 to skip detaching (CI, interactive debugging).
# Preferred over wrapping a command, because it works for scripts whose body is a
# loop and needs no function exporting.
mpcrl::self_detach() {
  if [[ "${MPCRL_DETACHED:-0}" == "1" || "${MPCRL_FOREGROUND:-0}" == "1" ]]; then
    return 0
  fi
  local tag log
  tag="$(basename "${BASH_SOURCE[1]:-$0}" .sh)"
  log="$(mpcrl::log_path "$tag")"
  echo "[mpcrl] detaching $tag; log: $log"
  if command -v setsid >/dev/null 2>&1; then
    MPCRL_DETACHED=1 setsid nohup bash "${BASH_SOURCE[1]:-$0}" "$@" >"$log" 2>&1 < /dev/null &
  else
    MPCRL_DETACHED=1 nohup bash "${BASH_SOURCE[1]:-$0}" "$@" >"$log" 2>&1 < /dev/null &
  fi
  echo "$!" > "$LOG_DIR/$tag.pid"
  echo "[mpcrl] pid $(cat "$LOG_DIR/$tag.pid")"
  echo "[mpcrl] follow with: tail -f $log"
  exit 0
}

# Run a single command detached, teeing to a log file.
# Prefer mpcrl::self_detach for whole scripts.
# Usage: mpcrl::detach <tag> <cmd...>
mpcrl::detach() {
  local tag="$1"; shift
  local log
  log="$(mpcrl::log_path "$tag")"
  if [[ "${MPCRL_FOREGROUND:-0}" == "1" || "${MPCRL_DETACHED:-0}" == "1" ]]; then
    echo "[mpcrl] running: $* (log: $log)"
    "$@" 2>&1 | tee "$log"
    return "${PIPESTATUS[0]}"
  fi
  echo "[mpcrl] detaching: $*"
  echo "[mpcrl] log: $log"
  if command -v setsid >/dev/null 2>&1; then
    setsid nohup "$@" >"$log" 2>&1 < /dev/null &
  else
    nohup "$@" >"$log" 2>&1 < /dev/null &
  fi
  local pid=$!
  echo "[mpcrl] pid: $pid"
  echo "[mpcrl] follow with: tail -f $log"
  echo "$pid" > "$LOG_DIR/$tag.pid"
  wait "$pid"
}

mpcrl::preflight() {
  echo "[mpcrl] repo:    $REPO_ROOT"
  echo "[mpcrl] python:  $($PYTHON -c 'import sys;print(sys.executable)')"
  echo "[mpcrl] torch:   $($PYTHON -c 'import torch;print(torch.__version__, "cuda", torch.version.cuda, "avail", torch.cuda.is_available())')"
  echo "[mpcrl] MUJOCO_GL: ${MUJOCO_GL:-<unset>}"
}
