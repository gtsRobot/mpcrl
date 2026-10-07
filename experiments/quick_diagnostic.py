from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mpcrl.config import load_yaml
from mpcrl.plotting import outside_legend, set_paper_style


METHODS = ('mpc_only', 'action_residual', 'planning_residual')


def run(cmd):
    print('RUN', ' '.join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def numeric_mean(g, col):
    return float(pd.to_numeric(g[col], errors='coerce').mean()) if col in g else np.nan


def numeric_last(g, col):
    if col not in g:
        return np.nan
    s = pd.to_numeric(g[col], errors='coerce').dropna()
    return float(s.iloc[-1]) if len(s) else np.nan


def build_summary(train_df, eval_df, requested_steps):
    rows = []
    if not len(train_df) or 'method' not in train_df:
        return pd.DataFrame(columns=['method', 'seed', 'episodes'])
    for (method, seed), g in train_df.groupby(['method', 'seed'], dropna=False):
        g = g.sort_values('global_step')
        ret = pd.to_numeric(g['episode_return'], errors='coerce')
        final_logged_step = int(pd.to_numeric(g['global_step'], errors='coerce').max())
        row = {
            'method': method,
            'seed': int(seed),
            'episodes': int(len(g)),
            'requested_steps': int(requested_steps),
            'last_completed_episode_step': final_logged_step,
            'completion_pct': 100.0,
            'return_mean': float(ret.mean()),
            'return_last5_mean': float(ret.tail(5).mean()),
            'return_best': float(ret.max()),
            'mpc_ms_mean': numeric_mean(g, 'mpc_ms'),
            'prediction_mse_last': numeric_last(g, 'prediction_mse'),
            'action_d1_mean': numeric_mean(g, 'action_d1'),
            'action_d2_mean': numeric_mean(g, 'action_d2'),
            'residual_norm_mean': numeric_mean(g, 'residual_norm'),
            'effective_residual_norm_mean': numeric_mean(g, 'effective_residual_norm'),
            'gate_mean': numeric_mean(g, 'gate'),
            'adaptive_gate_mean': numeric_mean(g, 'adaptive_gate'),
            'residual_ramp_mean': numeric_mean(g, 'residual_ramp'),
            'mpc_cache_hit_rate_mean': numeric_mean(g, 'mpc_cache_hit_rate'),
            'wm_loss_last': numeric_last(g, 'wm_loss'),
        }

        if len(eval_df):
            e = eval_df[
                (eval_df['method'] == method)
                & (pd.to_numeric(eval_df['train_seed'], errors='coerce') == int(seed))
            ]
            er = pd.to_numeric(e.get('episode_return'), errors='coerce').dropna()
            if len(er):
                row.update({
                    'eval_episodes': int(len(er)),
                    'eval_return_mean': float(er.mean()),
                    'eval_return_std': float(er.std(ddof=1)) if len(er) > 1 else 0.0,
                    'eval_return_min': float(er.min()),
                    'eval_return_max': float(er.max()),
                    'eval_length_mean': numeric_mean(e, 'episode_length'),
                    'eval_prediction_mse_mean': numeric_mean(e, 'prediction_mse'),
                    'eval_action_d1_mean': numeric_mean(e, 'action_d1'),
                    'eval_gate_mean': numeric_mean(e, 'gate'),
                })
        rows.append(row)
    return pd.DataFrame(rows)


def build_comparison(per_seed):
    if not len(per_seed) or 'method' not in per_seed:
        return pd.DataFrame(columns=['method', 'performance_source', 'seeds',
                                     'return_mean', 'return_std_across_seeds', 'vs_mpc_pct'])
    if 'eval_return_mean' in per_seed and per_seed['eval_return_mean'].notna().any():
        metric = 'eval_return_mean'
        source = 'deterministic_final_eval'
    else:
        metric = 'return_last5_mean'
        source = 'training_last5_fallback'

    rows = []
    base = pd.to_numeric(
        per_seed.loc[per_seed['method'] == 'mpc_only', metric],
        errors='coerce',
    ).dropna()
    base_mean = float(base.mean()) if len(base) else np.nan

    for method, g in per_seed.groupby('method'):
        vals = pd.to_numeric(g[metric], errors='coerce').dropna()
        mean = float(vals.mean()) if len(vals) else np.nan
        std = float(vals.std(ddof=1)) if len(vals) > 1 else np.nan
        rel = (
            (mean - base_mean) / max(abs(base_mean), 1e-9) * 100.0
            if np.isfinite(base_mean) and np.isfinite(mean)
            else np.nan
        )
        rows.append({
            'method': method,
            'performance_source': source,
            'seeds': int(len(vals)),
            'return_mean': mean,
            'return_std_across_seeds': std,
            'vs_mpc_pct': rel,
        })
    return pd.DataFrame(rows).sort_values('return_mean', ascending=False)


def plot_metrics(df, outdir):
    set_paper_style()
    outdir.mkdir(parents=True, exist_ok=True)
    metrics = [
        ('episode_return', 'Episode return'),
        ('mpc_ms', 'MPC planning time / ms'),
        ('prediction_mse', 'One-step prediction MSE'),
        ('action_d1', 'First-order action variation'),
        ('effective_residual_norm', 'Effective residual norm'),
        ('gate', 'Effective gate'),
    ]
    for metric, ylabel in metrics:
        if metric not in df:
            continue
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        for method, g in df.groupby('method'):
            q = g[['global_step', metric]].copy()
            q['global_step'] = pd.to_numeric(q['global_step'], errors='coerce')
            q[metric] = pd.to_numeric(q[metric], errors='coerce')
            q = (
                q.dropna()
                .groupby('global_step', as_index=False)[metric]
                .mean()
                .sort_values('global_step')
            )
            if len(q):
                ax.plot(q['global_step'], q[metric], marker='o', label=method)
        ax.set_xlabel('Environment steps')
        ax.set_ylabel(ylabel)
        ax.grid(alpha=.25)
        outside_legend(ax)
        fig.tight_layout()
        fig.savefig(outdir / f'{metric}.png', dpi=300, bbox_inches='tight')
        plt.close(fig)


def plot_eval(eval_df, outdir):
    if not len(eval_df):
        return
    set_paper_style()
    outdir.mkdir(parents=True, exist_ok=True)
    q = (
        eval_df.groupby(['method', 'train_seed'], as_index=False)['episode_return']
        .mean()
    )
    methods = list(q['method'].drop_duplicates())
    x = np.arange(len(methods))
    means, stds = [], []
    for method in methods:
        vals = q.loc[q.method == method, 'episode_return'].astype(float)
        means.append(vals.mean())
        stds.append(vals.std(ddof=1) if len(vals) > 1 else 0.0)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.bar(x, means, yerr=stds, capsize=4)
    ax.set_xticks(x, methods, rotation=15)
    ax.set_ylabel('Deterministic final evaluation return')
    ax.grid(axis='y', alpha=.25)
    fig.tight_layout()
    fig.savefig(outdir / 'final_eval_return.png', dpi=300, bbox_inches='tight')
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='configs/rtx5090/halfcheetah.yaml')
    p.add_argument('--steps', type=int, default=10000)
    p.add_argument('--seeds', type=int, nargs='+', default=[0])
    p.add_argument('--eval-episodes', type=int, default=3)
    p.add_argument('--tag')
    a = p.parse_args()

    cfg = load_yaml(a.config)
    root = Path(cfg['logging'].get('root', 'runs'))
    env_id = cfg['env']['id']
    tag = a.tag or datetime.now().strftime('%Y%m%d_%H%M%S')
    outdir = Path('results') / 'diagnostic_10k' / tag
    outdir.mkdir(parents=True, exist_ok=True)
    run_dirs = []

    for method in METHODS:
        for seed in a.seeds:
            name = f'diagnostic10k__{env_id}__{method}__seed{seed}__{tag}'
            run([
                sys.executable, '-m', 'experiments.train',
                '--config', a.config,
                '--method', method,
                '--steps', str(a.steps),
                '--seed', str(seed),
                '--run-name', name,
                '--eval-episodes', str(a.eval_episodes),
            ])
            run_dirs.append(root / name)

    train_frames = []
    eval_frames = []
    for d in run_dirs:
        f = d / 'episodes.csv'
        q = pd.read_csv(f)
        q['run_dir'] = str(d)
        train_frames.append(q)

        ef = d / 'eval.csv'
        if ef.exists():
            e = pd.read_csv(ef)
            e['run_dir'] = str(d)
            eval_frames.append(e)

    train_df = pd.concat(train_frames, ignore_index=True)
    eval_df = pd.concat(eval_frames, ignore_index=True) if eval_frames else pd.DataFrame()

    train_df.to_csv(outdir / 'episodes_all.csv', index=False)
    if len(eval_df):
        eval_df.to_csv(outdir / 'eval_all.csv', index=False)

    per_seed = build_summary(train_df, eval_df, a.steps)
    per_seed.to_csv(outdir / 'summary_per_seed.csv', index=False)

    if not len(per_seed):
        # Every episode in these envs is 1000 steps (Reacher 50), so a --steps
        # budget smaller than one episode legitimately completes zero episodes.
        # Say so instead of raising KeyError on a missing column.
        print(
            f'\nNo episode completed in {a.steps} steps, so there is nothing to summarise.\n'
            f'HalfCheetah/Hopper/Walker2d episodes are up to 1000 steps (Reacher 50), and '
            f'train.warmup_steps is consumed before the counter starts.\n'
            f'Raise --steps above one episode length, or use a shorter-episode env.'
        )
        print('Saved:', outdir)
        return

    comparison = build_comparison(per_seed)
    comparison.to_csv(outdir / 'comparison.csv', index=False)

    numeric_cols = per_seed.select_dtypes(include=[np.number]).columns
    aggregate = per_seed.groupby('method')[numeric_cols].agg(['mean', 'std'])
    aggregate.to_csv(outdir / 'summary_mean_std.csv')

    plot_metrics(train_df, outdir / 'figures')
    plot_eval(eval_df, outdir / 'figures')

    print('\nPer-seed summary:')
    print(per_seed.to_string(index=False))
    print('\nFinal comparison:')
    print(comparison.to_string(index=False))
    print('Saved:', outdir)


if __name__ == '__main__':
    main()
