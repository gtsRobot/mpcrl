#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
# Final Walker2d paper analysis from completed 100k runs.
#
# Normal use:
#   bash scripts/analyze_walker2d_full.sh
#
# If the runs were produced by another clone of this repository:
#   ROOT=/absolute/path/to/MPC_RL_ZPRL_RTX5090/runs \
#   bash scripts/analyze_walker2d_full.sh
#
# Optional overrides:
#   OUT=results/walker2d_final MIN_STEPS=90000 \
#   bash scripts/analyze_walker2d_full.sh

export MPLBACKEND=Agg

ROOT="${ROOT:-runs}"
OUT="${OUT:-results/walker2d_final}"
MIN_STEPS="${MIN_STEPS:-90000}"

mkdir -p "$OUT"

# ---------------------------------------------------------------------------
# Preflight: give a useful diagnosis instead of only
# "No valid runs matched the requested filters".
# ---------------------------------------------------------------------------
preflight() {
"$PYTHON" - "$ROOT" "$MIN_STEPS" <<'PY'
from __future__ import annotations
import glob
import os
import sys
from pathlib import Path
import pandas as pd

root = Path(sys.argv[1]).expanduser()
min_steps = float(sys.argv[2])
methods = ('mpc_only', 'action_residual', 'planning_residual', 'zprl_style')
expected_seeds = set(range(5))

print(f'[preflight] repository: {Path.cwd()}')
print(f'[preflight] requested ROOT: {root.resolve() if root.exists() else root}')

files = sorted(glob.glob(str(root / '**' / 'episodes.csv'), recursive=True)) if root.exists() else []
records = []
for f in files:
    try:
        df = pd.read_csv(f)
    except Exception:
        continue
    req = {'env', 'method', 'seed', 'global_step'}
    if df.empty or not req.issubset(df.columns):
        continue
    df = df.copy()
    df['global_step'] = pd.to_numeric(df['global_step'], errors='coerce')
    df = df.dropna(subset=['env', 'method', 'seed', 'global_step'])
    if df.empty:
        continue
    last = df.iloc[-1]
    records.append({
        'path': f,
        'env': str(last['env']),
        'method': str(last['method']),
        'seed': int(float(last['seed'])),
        'max_step': float(df['global_step'].max()),
        'has_eval': (Path(f).parent / 'eval.csv').exists(),
    })

walker = [r for r in records if r['env'] == 'Walker2d-v5']
formal = [r for r in walker if r['method'] in methods and r['max_step'] >= min_steps]

if records:
    env_counts = pd.DataFrame(records).groupby('env').size().sort_values(ascending=False)
    print('[preflight] valid episodes.csv by env:')
    print(env_counts.to_string())
else:
    print('[preflight] no readable episodes.csv found under requested ROOT')

if walker:
    print('\n[preflight] Walker2d candidates under requested ROOT:')
    q = pd.DataFrame(walker)[['method', 'seed', 'max_step', 'has_eval', 'path']]
    print(q.sort_values(['method', 'seed', 'max_step']).to_string(index=False))

coverage = {(r['method'], r['seed']) for r in formal if r['has_eval']}
missing = [(m, s) for m in methods for s in sorted(expected_seeds) if (m, s) not in coverage]

if formal and not missing:
    print(f'\n[preflight] OK: complete 4 methods x 5 seeds with >= {int(min_steps)} steps and eval.csv')
    sys.exit(0)

print('\n[preflight] requested ROOT does not contain a complete formal Walker2d experiment.')
if formal:
    print('[preflight] matched formal runs:', len(formal))
if missing:
    print('[preflight] missing formal method/seed pairs:')
    for m, s in missing:
        print(f'  - {m}, seed={s}')

# Search sibling clones under ~/Private_encrypted. This is diagnostic only;
# never silently merges runs from different experiment roots.
home = Path.home()
search_base = home / 'Private_encrypted'
if search_base.exists():
    candidates = []
    for p in search_base.glob('*/MPC_RL_ZPRL_RTX5090/runs'):
        if not p.is_dir() or p.resolve() == root.resolve() if root.exists() else False:
            continue
        n_walker = 0
        n_formal = 0
        pairs = set()
        for f in glob.glob(str(p / '**' / 'episodes.csv'), recursive=True):
            try:
                df = pd.read_csv(f, usecols=lambda c: c in {'env','method','seed','global_step'})
            except Exception:
                continue
            if df.empty or not {'env','method','seed','global_step'}.issubset(df.columns):
                continue
            df['global_step'] = pd.to_numeric(df['global_step'], errors='coerce')
            df = df.dropna()
            if df.empty:
                continue
            last = df.iloc[-1]
            if str(last['env']) != 'Walker2d-v5':
                continue
            n_walker += 1
            method = str(last['method'])
            seed = int(float(last['seed']))
            max_step = float(df['global_step'].max())
            if method in methods and max_step >= min_steps and (Path(f).parent / 'eval.csv').exists():
                n_formal += 1
                pairs.add((method, seed))
        if n_walker:
            candidates.append((len(pairs), n_formal, n_walker, p))

    if candidates:
        print('\n[preflight] Walker2d data found in other repository clones:')
        for unique_pairs, n_formal, n_walker, p in sorted(candidates, reverse=True):
            print(f'  {p}  formal_pairs={unique_pairs}/20  formal_files={n_formal}  walker_files={n_walker}')
        best = max(candidates, key=lambda x: x[0])
        if best[0] > 0:
            print('\nRun the analysis with the actual run root, for example:')
            print(f'  ROOT="{best[3]}" bash scripts/analyze_walker2d_full.sh')

print('\nIf no candidate above contains the runs, locate them with:')
print("  find ~/Private_encrypted -type f -path '*/runs/*/episodes.csv' | grep -i Walker2d")
sys.exit(2)
PY
}

preflight

"$PYTHON" -m experiments.aggregate \
  --root "$ROOT" \
  --out "$OUT/summary.csv" \
  --env Walker2d-v5 \
  --methods mpc_only action_residual planning_residual zprl_style \
  --min-steps "$MIN_STEPS" \
  --require-eval \
  --expected-seeds 0 1 2 3 4 \
  --require-complete \
  --tail 20

"$PYTHON" -m experiments.summarize_eval \
  --root "$ROOT" \
  --out "$OUT/eval_summary" \
  --env Walker2d-v5 \
  --methods mpc_only action_residual planning_residual zprl_style \
  --min-steps "$MIN_STEPS" \
  --expected-seeds 0 1 2 3 4 \
  --require-complete

"$PYTHON" -m experiments.plot_results \
  --root "$ROOT" \
  --outdir "$OUT/figures" \
  --env Walker2d-v5 \
  --methods mpc_only action_residual planning_residual zprl_style \
  --min-steps "$MIN_STEPS" \
  --expected-methods mpc_only action_residual planning_residual zprl_style \
  --expected-seeds 0 1 2 3 4 \
  --require-complete \
  --smooth-window 10 \
  --points 400 \
  --error-band std

echo
echo "Walker2d final analysis completed."
echo "Root:         $ROOT"
echo "Summary:      $OUT/summary.csv"
echo "Per seed:     $OUT/summary_per_seed.csv"
echo "Completeness: $OUT/summary_completeness.csv"
echo "Eval summary: $OUT/eval_summary/comparison_eval.csv"
echo "Eval coverage:$OUT/eval_summary/eval_coverage.csv"
echo "Figures:      $OUT/figures"
