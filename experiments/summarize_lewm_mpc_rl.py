from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


METHODS = ['lewm_mpc', 'lewm_mpc_rl']


def discover_eval(root):
    rows = []
    for f in Path(root).glob('**/eval.csv'):
        try:
            q = pd.read_csv(f)
        except Exception:
            continue
        if q.empty or 'method' not in q or 'env' not in q:
            continue
        method = str(q['method'].dropna().iloc[-1]) if q['method'].notna().any() else ''
        if method not in METHODS:
            continue
        seed_col = 'train_seed' if 'train_seed' in q else 'seed'
        if seed_col not in q or not q[seed_col].notna().any():
            continue
        seed = int(float(q[seed_col].dropna().iloc[-1]))
        env = str(q['env'].dropna().iloc[-1])
        ret = pd.to_numeric(q['episode_return'], errors='coerce').dropna()
        if not len(ret):
            continue
        success = pd.to_numeric(q.get('success'), errors='coerce') if 'success' in q else pd.Series(dtype=float)
        success = success[np.isfinite(success)] if len(success) else success
        d1 = pd.to_numeric(q.get('action_d1'), errors='coerce') if 'action_d1' in q else pd.Series(dtype=float)
        d2 = pd.to_numeric(q.get('action_d2'), errors='coerce') if 'action_d2' in q else pd.Series(dtype=float)
        mpc_ms = pd.to_numeric(q.get('mpc_ms'), errors='coerce') if 'mpc_ms' in q else pd.Series(dtype=float)
        er = pd.to_numeric(q.get('effective_rank'), errors='coerce') if 'effective_rank' in q else pd.Series(dtype=float)
        rows.append({
            'env': env,
            'method': method,
            'seed': seed,
            'eval_return_mean': float(ret.mean()),
            'eval_return_std_within_seed': float(ret.std(ddof=1)) if len(ret) > 1 else 0.0,
            'eval_episodes': int(len(ret)),
            'success_rate_pct': 100.0 * float(success.mean()) if len(success) else np.nan,
            'action_d1_mean': float(d1.mean()) if len(d1) and d1.notna().any() else np.nan,
            'action_d2_mean': float(d2.mean()) if len(d2) and d2.notna().any() else np.nan,
            'mpc_ms_mean': float(mpc_ms.mean()) if len(mpc_ms) and mpc_ms.notna().any() else np.nan,
            'effective_rank': float(er.mean()) if len(er) and er.notna().any() else np.nan,
            'run_dir': str(f.parent),
        })
    return pd.DataFrame(rows)


def completeness(per_seed, envs, seeds):
    keys = set(zip(per_seed.env, per_seed.method, per_seed.seed)) if len(per_seed) else set()
    rows = []
    for env in envs:
        for method in METHODS:
            for seed in seeds:
                rows.append({
                    'env': env,
                    'method': method,
                    'seed': seed,
                    'complete': (env, method, seed) in keys,
                })
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    p.add_argument('--outdir', required=True)
    p.add_argument('--envs', nargs='+', default=['Hopper-v5', 'Walker2d-v5', 'HalfCheetah-v5', 'Reacher-v5'])
    p.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2, 3, 4])
    p.add_argument('--require-complete', action='store_true')
    a = p.parse_args()

    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    per_seed = discover_eval(a.root)
    per_seed = per_seed[per_seed.env.isin(a.envs)].copy() if len(per_seed) else per_seed
    if per_seed.empty:
        raise SystemExit('No LeWM-MPC / LeWM-MPC+RL eval.csv files found.')

    # Keep the newest discovered run if duplicated for the same env/method/seed.
    per_seed['_mtime'] = per_seed.run_dir.map(lambda x: Path(x, 'eval.csv').stat().st_mtime)
    per_seed = (
        per_seed.sort_values('_mtime')
        .drop_duplicates(['env', 'method', 'seed'], keep='last')
        .drop(columns=['_mtime'])
        .sort_values(['env', 'method', 'seed'])
    )
    per_seed.to_csv(out / 'direct_per_seed.csv', index=False)

    cov = completeness(per_seed, a.envs, a.seeds)
    cov.to_csv(out / 'direct_coverage.csv', index=False)
    missing = cov.loc[~cov.complete]
    if len(missing):
        print('\nMissing direct-comparison runs:')
        print(missing.to_string(index=False))
        if a.require_complete:
            raise SystemExit('Direct LeWM experiment is incomplete.')

    summary_rows = []
    paired_rows = []
    for env in a.envs:
        e = per_seed[per_seed.env == env]
        if e.empty:
            continue
        for method in METHODS:
            g = e[e.method == method]
            if g.empty:
                continue
            vals = g.eval_return_mean.astype(float)
            std = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
            ci95 = 1.96 * std / np.sqrt(len(vals)) if len(vals) > 1 else np.nan
            succ = g.success_rate_pct.astype(float)
            summary_rows.append({
                'env': env,
                'method': method,
                'seeds': int(len(g)),
                'eval_return_mean': float(vals.mean()),
                'eval_return_std_across_seeds': std,
                'eval_return_ci95_half_width': ci95,
                'success_rate_mean_pct': float(succ.mean()) if np.isfinite(succ).any() else np.nan,
                'success_rate_std_pct': float(succ.std(ddof=1)) if np.isfinite(succ).sum() > 1 else np.nan,
                'action_d1_mean': float(g.action_d1_mean.mean()),
                'action_d2_mean': float(g.action_d2_mean.mean()),
                'mpc_ms_mean': float(g.mpc_ms_mean.mean()),
                'effective_rank_mean': float(g.effective_rank.mean()),
            })

        b = e[e.method == 'lewm_mpc'][['seed', 'eval_return_mean', 'success_rate_pct']].rename(
            columns={'eval_return_mean': 'baseline_return', 'success_rate_pct': 'baseline_success'}
        )
        r = e[e.method == 'lewm_mpc_rl'][['seed', 'eval_return_mean', 'success_rate_pct']].rename(
            columns={'eval_return_mean': 'rl_return', 'success_rate_pct': 'rl_success'}
        )
        paired = b.merge(r, on='seed', how='inner')
        for _, row in paired.iterrows():
            delta = float(row.rl_return - row.baseline_return)
            paired_rows.append({
                'env': env,
                'seed': int(row.seed),
                'baseline_return': float(row.baseline_return),
                'rl_return': float(row.rl_return),
                'delta_return': delta,
                'improvement_pct': delta / max(abs(float(row.baseline_return)), 1e-9) * 100.0,
                'baseline_success_rate_pct': float(row.baseline_success) if np.isfinite(row.baseline_success) else np.nan,
                'rl_success_rate_pct': float(row.rl_success) if np.isfinite(row.rl_success) else np.nan,
                'success_rate_delta_pct': (
                    float(row.rl_success - row.baseline_success)
                    if np.isfinite(row.baseline_success) and np.isfinite(row.rl_success)
                    else np.nan
                ),
            })

    summary = pd.DataFrame(summary_rows)
    paired = pd.DataFrame(paired_rows)
    summary.to_csv(out / 'direct_summary.csv', index=False)
    paired.to_csv(out / 'paired_improvements.csv', index=False)

    pair_summary = (
        paired.groupby('env')
        .agg(
            paired_seeds=('seed', 'count'),
            mean_delta_return=('delta_return', 'mean'),
            std_delta_return=('delta_return', 'std'),
            mean_improvement_pct=('improvement_pct', 'mean'),
            mean_success_rate_delta_pct=('success_rate_delta_pct', 'mean'),
        )
        .reset_index()
        if len(paired) else pd.DataFrame()
    )
    pair_summary.to_csv(out / 'paired_summary.csv', index=False)

    print('\nDirect LeWM comparison:')
    print(summary.to_string(index=False))
    print('\nPaired improvements (same shared LeWM and seed):')
    print(pair_summary.to_string(index=False) if len(pair_summary) else 'none')
    print('\nsaved:', out)


if __name__ == '__main__':
    main()
