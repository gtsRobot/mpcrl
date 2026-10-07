from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd

from mpcrl.paired_stats import paired_summary, t_quantile_975


DEFAULT_METHODS = ['mpc_only', 'action_residual', 'planning_residual', 'zprl_style']


def discover_run_dirs(root=None, pattern=None):
    if pattern:
        candidates = [Path(x) for x in glob.glob(pattern, recursive=True)]
    else:
        candidates = [p.parent for p in Path(root or 'runs').glob('**/eval.csv')]
    return sorted({p if p.is_dir() else p.parent for p in candidates})


def read_run_meta(run_dir):
    ep = run_dir / 'episodes.csv'
    ev = run_dir / 'eval.csv'
    if not ev.exists():
        return None
    try:
        eval_df = pd.read_csv(ev)
    except Exception:
        return None
    if eval_df.empty:
        return None

    env = str(eval_df['env'].dropna().iloc[-1]) if 'env' in eval_df and eval_df['env'].notna().any() else ''
    method = (
        str(eval_df['method'].dropna().iloc[-1])
        if 'method' in eval_df and eval_df['method'].notna().any()
        else ''
    )
    seed_col = 'train_seed' if 'train_seed' in eval_df else 'seed'
    seed = (
        int(float(eval_df[seed_col].dropna().iloc[-1]))
        if seed_col in eval_df and eval_df[seed_col].notna().any()
        else None
    )

    max_step = np.nan
    if ep.exists():
        try:
            ep_df = pd.read_csv(ep, usecols=lambda c: c in {'global_step', 'env', 'method', 'seed'})
            if 'global_step' in ep_df:
                s = pd.to_numeric(ep_df['global_step'], errors='coerce')
                if s.notna().any():
                    max_step = float(s.max())
            if not env and 'env' in ep_df and ep_df['env'].notna().any():
                env = str(ep_df['env'].dropna().iloc[-1])
            if not method and 'method' in ep_df and ep_df['method'].notna().any():
                method = str(ep_df['method'].dropna().iloc[-1])
            if seed is None and 'seed' in ep_df and ep_df['seed'].notna().any():
                seed = int(float(ep_df['seed'].dropna().iloc[-1]))
        except Exception:
            pass

    if not env or not method or seed is None:
        return None
    return {
        'run_dir': run_dir,
        'env': env,
        'method': method,
        'seed': seed,
        'max_step': max_step,
        'mtime': ev.stat().st_mtime,
        'eval': eval_df,
    }


def choose_best(runs):
    best = {}
    for r in runs:
        key = (r['env'], r['method'], r['seed'])
        step = -np.inf if not np.isfinite(r['max_step']) else r['max_step']
        score = (step, r['mtime'])
        if key not in best:
            best[key] = r
            continue
        old = best[key]
        old_step = -np.inf if not np.isfinite(old['max_step']) else old['max_step']
        if score > (old_step, old['mtime']):
            best[key] = r
    return list(best.values())


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', default='runs')
    p.add_argument('--glob', dest='pattern', help='legacy quoted run-directory glob')
    p.add_argument('--out', required=True)
    p.add_argument('--env', action='append', help='exact environment id; repeatable')
    p.add_argument('--methods', nargs='+', default=DEFAULT_METHODS)
    p.add_argument('--min-steps', type=int, default=0)
    p.add_argument('--expected-seeds', nargs='+', type=int, default=[0, 1, 2, 3, 4])
    p.add_argument('--require-complete', action='store_true')
    a = p.parse_args()

    env_filter = set(a.env or [])
    method_filter = set(a.methods or [])
    runs = []
    for d in discover_run_dirs(a.root, a.pattern):
        r = read_run_meta(d)
        if r is None:
            continue
        if env_filter and r['env'] not in env_filter:
            continue
        if method_filter and r['method'] not in method_filter:
            continue
        if np.isfinite(r['max_step']) and r['max_step'] < a.min_steps:
            continue
        runs.append(r)

    runs = choose_best(runs)
    if not runs:
        raise SystemExit('No deterministic eval runs matched the requested filters.')

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    envs = list(dict.fromkeys(a.env or sorted({r['env'] for r in runs})))
    by_key = {(r['env'], r['method'], r['seed']): r for r in runs}
    coverage_rows = []
    for env in envs:
        for method in a.methods:
            for seed in a.expected_seeds:
                r = by_key.get((env, method, seed))
                coverage_rows.append({
                    'env': env,
                    'method': method,
                    'seed': seed,
                    'eval_found': r is not None,
                    'max_step': r['max_step'] if r is not None else np.nan,
                    'complete': bool(
                        r is not None
                        and (not np.isfinite(r['max_step']) or r['max_step'] >= a.min_steps)
                    ),
                    'run_dir': str(r['run_dir']) if r is not None else '',
                })
    coverage = pd.DataFrame(coverage_rows)
    coverage.to_csv(out / 'eval_coverage.csv', index=False)
    missing = coverage.loc[~coverage['complete']]
    if len(missing):
        print('\nIncomplete deterministic evaluation coverage:')
        print(missing[['env', 'method', 'seed', 'eval_found', 'max_step']].to_string(index=False))
        if a.require_complete:
            raise SystemExit('Deterministic final evaluation is incomplete.')

    all_frames = []
    per_seed_rows = []
    for r in runs:
        q = r['eval'].copy()
        q['run_dir'] = str(r['run_dir'])
        all_frames.append(q)

        def col_mean(name):
            if name not in q:
                return np.nan
            v = pd.to_numeric(q[name], errors='coerce')
            return float(v.mean()) if v.notna().any() else np.nan

        ret = pd.to_numeric(q['episode_return'], errors='coerce').dropna()
        if not len(ret):
            continue
        per_seed_rows.append({
            'env': r['env'],
            'method': r['method'],
            'train_seed': r['seed'],
            'max_step': r['max_step'],
            'eval_return_mean': float(ret.mean()),
            'eval_return_std_within_seed': float(ret.std(ddof=1)) if len(ret) > 1 else 0.0,
            'eval_episodes': int(len(ret)),
            'eval_length_mean': col_mean('episode_length'),
            'eval_action_d1_mean': col_mean('action_d1'),
            'eval_action_d2_mean': col_mean('action_d2'),
            'eval_effective_residual_norm_mean': col_mean('effective_residual_norm'),
            'eval_gate_mean': col_mean('gate'),
            'run_dir': str(r['run_dir']),
        })

    df = pd.concat(all_frames, ignore_index=True)
    per_seed = pd.DataFrame(per_seed_rows).sort_values(['env', 'method', 'train_seed'])

    comparison_rows = []
    for env, eg in per_seed.groupby('env'):
        mpc = eg.loc[eg.method == 'mpc_only', ['train_seed', 'eval_return_mean']].rename(
            columns={'eval_return_mean': 'mpc_return'}
        )
        baseline_means = (
            eg.loc[eg.method != 'zprl_style']
            .groupby('method')['eval_return_mean'].mean()
        )
        best_baseline_mean = float(baseline_means.max()) if len(baseline_means) else np.nan

        for method, g in eg.groupby('method'):
            vals = g['eval_return_mean'].astype(float)
            mean = float(vals.mean())
            std = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
            # Student-t, not the normal 1.96: at n=5 the correct multiplier is
            # 2.776, so the normal approximation understated this interval by 42%.
            ci95 = (t_quantile_975(len(vals) - 1) * std / np.sqrt(len(vals))
                    if len(vals) > 1 else np.nan)

            paired = g[['train_seed', 'eval_return_mean']].merge(mpc, on='train_seed')
            paired_deltas = (paired['eval_return_mean'] - paired['mpc_return']).tolist()
            paired_delta = float(np.mean(paired_deltas)) if len(paired) else np.nan
            pstats = paired_summary(paired_deltas) if len(paired) else {}
            mpc_mean = float(mpc['mpc_return'].mean()) if len(mpc) else np.nan
            comparison_rows.append({
                'env': env,
                'method': method,
                'seeds': int(len(vals)),
                'eval_return_mean': mean,
                'eval_return_std_across_seeds': std,
                'eval_return_ci95_half_width': ci95,
                'vs_mpc_pct': (
                    (mean - mpc_mean) / max(abs(mpc_mean), 1e-9) * 100.0
                    if np.isfinite(mpc_mean) else np.nan
                ),
                'paired_delta_vs_mpc_mean': paired_delta,
                # NOTE: these runs do NOT share a world model. The "paired" delta
                # here only pairs on seed index, so it still carries the
                # world-model-quality confound. Use experiments.train_paired +
                # experiments.summarize_paired for a genuinely paired comparison.
                'paired_is_shared_world_model': False,
                'paired_delta_ci95_low': pstats.get('ci95_low', np.nan),
                'paired_delta_ci95_high': pstats.get('ci95_high', np.nan),
                'paired_n_positive_seeds': pstats.get('n_positive', np.nan),
                'paired_wilcoxon_p': pstats.get('wilcoxon_p', np.nan),
                'paired_wilcoxon_p_floor': pstats.get('wilcoxon_p_floor', np.nan),
                'vs_best_non_zprl_pct': (
                    (mean - best_baseline_mean) / max(abs(best_baseline_mean), 1e-9) * 100.0
                    if method == 'zprl_style' and np.isfinite(best_baseline_mean)
                    else np.nan
                ),
            })

    comparison = pd.DataFrame(comparison_rows).sort_values(
        ['env', 'eval_return_mean'], ascending=[True, False]
    )

    print('\nPer-seed deterministic evaluation:')
    print(per_seed.to_string(index=False))
    print('\nComparison:')
    print(comparison.to_string(index=False))

    df.to_csv(out / 'eval_all.csv', index=False)
    per_seed.to_csv(out / 'eval_per_seed.csv', index=False)
    comparison.to_csv(out / 'comparison_eval.csv', index=False)
    print('saved', out)


if __name__ == '__main__':
    main()
