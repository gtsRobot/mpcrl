from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_METHODS = ['mpc_only', 'action_residual', 'planning_residual', 'zprl_style']


def first_success_step(g):
    if 'success' not in g:
        return np.nan
    s = pd.to_numeric(g['success'], errors='coerce')
    idx = s[s > 0.5].index
    return float(g.loc[idx[0], 'global_step']) if len(idx) else np.nan


def load_runs(root, env_filter=None, method_filter=None):
    records = []
    env_filter = set(env_filter or [])
    method_filter = set(method_filter or [])
    for f in glob.glob(os.path.join(root, '**', 'episodes.csv'), recursive=True):
        try:
            df = pd.read_csv(f)
        except Exception:
            continue
        if df.empty or not {'env', 'method', 'seed', 'global_step'}.issubset(df.columns):
            continue
        df = df[df['env'].notna() & df['method'].notna() & df['seed'].notna()].copy()
        if df.empty:
            continue
        df['global_step'] = pd.to_numeric(df['global_step'], errors='coerce')
        df = df[df['global_step'].notna()]
        if df.empty:
            continue
        last = df.iloc[-1]
        env = str(last['env'])
        method = str(last['method'])
        if env_filter and env not in env_filter:
            continue
        if method_filter and method not in method_filter:
            continue
        run_dir = Path(f).parent
        records.append({
            'path': f,
            'run_dir': run_dir,
            'env': env,
            'method': method,
            'seed': int(float(last['seed'])),
            'max_step': float(df['global_step'].max()),
            'mtime': Path(f).stat().st_mtime,
            'df': df,
        })
    return records


def choose_one_run_per_seed(records):
    """Avoid double-counting repeated/restarted runs."""
    best = {}
    for r in records:
        key = (r['env'], r['method'], r['seed'])
        score = (r['max_step'], r['mtime'])
        if key not in best or score > (best[key]['max_step'], best[key]['mtime']):
            best[key] = r
    return list(best.values())


def read_eval(run_dir):
    path = Path(run_dir) / 'eval.csv'
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    return df if len(df) else None


def completeness_table(records, envs, methods, seeds, min_steps):
    by_key = {(r['env'], r['method'], r['seed']): r for r in records}
    rows = []
    for env in envs:
        for method in methods:
            for seed in seeds:
                r = by_key.get((env, method, seed))
                eval_exists = bool(r is not None and read_eval(r['run_dir']) is not None)
                max_step = r['max_step'] if r is not None else np.nan
                rows.append({
                    'env': env,
                    'method': method,
                    'seed': seed,
                    'run_found': r is not None,
                    'max_step': max_step,
                    'step_complete': bool(r is not None and max_step >= min_steps),
                    'eval_exists': eval_exists,
                    'complete': bool(r is not None and max_step >= min_steps and eval_exists),
                    'run_dir': str(r['run_dir']) if r is not None else '',
                })
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', default='runs')
    p.add_argument('--out', default='results/summary.csv')
    p.add_argument('--tail', type=int, default=20)
    p.add_argument('--env', action='append', help='exact environment id; repeatable')
    p.add_argument('--methods', nargs='+', default=DEFAULT_METHODS)
    p.add_argument('--min-steps', type=int, default=0)
    p.add_argument('--require-eval', action='store_true')
    p.add_argument('--expected-seeds', nargs='+', type=int, default=[0, 1, 2, 3, 4])
    p.add_argument('--require-complete', action='store_true')
    a = p.parse_args()

    records = choose_one_run_per_seed(load_runs(a.root, a.env, a.methods))
    records = [r for r in records if r['max_step'] >= a.min_steps]
    if not records:
        raise SystemExit('No valid runs matched the requested filters.')

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    envs = list(dict.fromkeys(a.env or sorted({r['env'] for r in records})))
    completeness = completeness_table(records, envs, a.methods, a.expected_seeds, a.min_steps)
    completeness_path = out.with_name(out.stem + '_completeness.csv')
    completeness.to_csv(completeness_path, index=False)

    missing = completeness.loc[~completeness['complete']]
    if len(missing):
        print('\nIncomplete expected runs:')
        print(missing[['env', 'method', 'seed', 'run_found', 'max_step', 'step_complete', 'eval_exists']].to_string(index=False))
        if a.require_complete:
            raise SystemExit('Experiment is incomplete; final aggregation aborted.')

    cols = [
        'episode_return', 'success', 'mpc_ms', 'prediction_mse', 'action_d1',
        'action_d2', 'residual_norm', 'effective_residual_norm', 'gate',
        'adaptive_gate', 'gate_z', 'residual_ramp', 'mpc_cache_hit_rate',
        'safeguard_scale', 'safeguard_gain', 'episode_length',
    ]
    rows = []
    for r in records:
        eval_df = read_eval(r['run_dir'])
        if eval_df is not None:
            g = eval_df.reset_index(drop=True)
            source = 'deterministic_final_eval'
        else:
            if a.require_eval:
                raise SystemExit(f"Missing eval.csv for {r['env']} {r['method']} seed={r['seed']}: {r['run_dir']}")
            g = r['df'].tail(max(1, int(a.tail))).reset_index(drop=True)
            source = f'training_tail_{max(1, int(a.tail))}'
        row = {
            'env': r['env'], 'method': r['method'], 'seed': r['seed'],
            'global_step': r['max_step'], 'episodes_used': len(g),
            'performance_source': source, 'run_path': str(r['run_dir']),
        }
        for col in cols:
            row[col] = pd.to_numeric(g[col], errors='coerce').mean() if col in g else np.nan
        row['sample_efficiency_step'] = first_success_step(r['df'].reset_index(drop=True))
        rows.append(row)

    per_seed = pd.DataFrame(rows).sort_values(['env', 'method', 'seed'])
    per_seed_path = out.with_name(out.stem + '_per_seed.csv')
    per_seed.to_csv(per_seed_path, index=False)
    numeric = [c for c in per_seed.columns if c not in {'env', 'method', 'performance_source', 'run_path'} and pd.api.types.is_numeric_dtype(per_seed[c])]
    summary = per_seed.groupby(['env', 'method'], dropna=False)[numeric].agg(['mean', 'std']).reset_index()
    summary.to_csv(out, index=False)

    print(per_seed[['env', 'method', 'seed', 'global_step', 'performance_source', 'episode_return', 'episode_length']].to_string(index=False))
    print('\nmean/std summary:\n', summary.to_string(index=False))
    print('saved', out)
    print('saved', per_seed_path)
    print('saved', completeness_path)


if __name__ == '__main__':
    main()
