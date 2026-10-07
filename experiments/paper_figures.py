from __future__ import annotations

import argparse
import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager

METHODS = ('mpc_only', 'action_residual', 'planning_residual', 'zprl_style')
LABELS = {
    'mpc_only': 'MPC',
    'action_residual': 'Action Residual',
    'planning_residual': 'Planning Residual',
    'zprl_style': 'ZPRL-style',
}


def set_style():
    names = {f.name for f in font_manager.fontManager.ttflist}
    plt.rcParams['font.family'] = 'Times New Roman' if 'Times New Roman' in names else 'DejaVu Serif'
    plt.rcParams['font.size'] = 10.5
    plt.rcParams['axes.unicode_minus'] = False


def _num(s):
    return pd.to_numeric(s, errors='coerce')


def _save(fig, out: Path, stem: str):
    fig.tight_layout()
    fig.savefig(out / f'{stem}.png', dpi=300, bbox_inches='tight')
    fig.savefig(out / f'{stem}.pdf', bbox_inches='tight')
    plt.close(fig)


def load_summary(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {'env', 'method', 'seed', 'episode_return'}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f'{path} missing columns: {sorted(missing)}')
    for c in df.columns:
        if c not in {'env', 'method', 'performance_source', 'run_path'}:
            df[c] = _num(df[c])
    return df


def main_return_figures(df: pd.DataFrame, out: Path):
    rows = []
    for env, g in df.groupby('env'):
        stats = g.groupby('method')['episode_return'].agg(['mean', 'std', 'count']).reindex(METHODS)
        fig, ax = plt.subplots(figsize=(7.2, 4.8))
        x = np.arange(len(METHODS))
        ax.bar(x, stats['mean'].values, yerr=stats['std'].values, capsize=5)
        for i, method in enumerate(METHODS):
            vals = g.loc[g.method == method, 'episode_return'].dropna().values
            if len(vals):
                jitter = np.linspace(-0.10, 0.10, len(vals))
                ax.scatter(np.full(len(vals), i) + jitter, vals, s=24, zorder=3)
        ax.set_xticks(x, [LABELS[m] for m in METHODS], rotation=15, ha='right')
        ax.set_ylabel('Deterministic evaluation return')
        ax.set_title(env)
        ax.grid(axis='y', alpha=.25)
        _save(fig, out, f'return_mean_std_{env.replace("-v5", "")}')

        base = float(stats.loc['mpc_only', 'mean'])
        for method in METHODS:
            mean = float(stats.loc[method, 'mean'])
            std = float(stats.loc[method, 'std'])
            n = int(stats.loc[method, 'count'])
            rows.append({
                'env': env,
                'method': method,
                'mean_return': mean,
                'std_return': std,
                'seeds': n,
                'ci95_half_width': 1.96 * std / np.sqrt(n) if n > 1 else np.nan,
                'vs_mpc_pct': 100.0 * (mean - base) / max(abs(base), 1e-12),
            })

    table = pd.DataFrame(rows)
    table.to_csv(out / 'table_main_return.csv', index=False)

    rel = table[table.method != 'mpc_only']
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    envs = list(dict.fromkeys(table.env.tolist()))
    for method in ('action_residual', 'planning_residual', 'zprl_style'):
        gm = rel[rel.method == method].set_index('env')
        ax.plot(envs, [gm.loc[e, 'vs_mpc_pct'] for e in envs], marker='o', label=LABELS[method])
    ax.axhline(0, linewidth=1)
    ax.set_ylabel('Return change vs MPC (%)')
    ax.set_xlabel('Environment')
    ax.tick_params(axis='x', rotation=15)
    ax.grid(alpha=.25)
    ax.legend()
    _save(fig, out, 'relative_return_vs_mpc')


def paired_planning_delta(df: pd.DataFrame, out: Path):
    rows = []
    for env, g in df.groupby('env'):
        p = g[g.method == 'planning_residual'][['seed', 'episode_return']].rename(
            columns={'episode_return': 'planning_return'}
        )
        b = g[g.method == 'mpc_only'][['seed', 'episode_return']].rename(
            columns={'episode_return': 'mpc_return'}
        )
        q = p.merge(b, on='seed').sort_values('seed')
        if q.empty:
            continue
        q['delta_return'] = q['planning_return'] - q['mpc_return']
        q['env'] = env
        rows.append(q)

        values = q.delta_return.to_numpy(float)
        mean = float(np.mean(values))
        ci = 1.96 * float(np.std(values, ddof=1)) / np.sqrt(len(values)) if len(values) > 1 else 0.0
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        x = np.arange(len(q))
        ax.bar(x, values)
        ax.axhline(0.0, linewidth=1.0)
        ax.axhline(mean, linestyle='--', linewidth=1.2, label=f'Mean = {mean:.2f}')
        ax.fill_between([-0.5, len(q) - 0.5], mean - ci, mean + ci, alpha=0.12, label='95% CI of paired mean')
        ax.set_xticks(x, [f"seed {int(s)}" for s in q.seed])
        ax.set_ylabel('Planning Residual - MPC return')
        ax.set_title(f'{env}: paired final-evaluation difference')
        ax.grid(axis='y', alpha=.25)
        ax.legend(fontsize=8)
        _save(fig, out, f'paired_planning_vs_mpc_{env.replace("-v5", "")}')

    if rows:
        pd.concat(rows, ignore_index=True).to_csv(out / 'paired_planning_vs_mpc.csv', index=False)


def diagnostic_figures(df: pd.DataFrame, out: Path):
    metrics = [
        ('prediction_mse', 'One-step prediction MSE'),
        ('action_d1', 'First-order action variation'),
        ('action_d2', 'Second-order action variation'),
        ('effective_residual_norm', 'Effective residual norm'),
        ('gate', 'Effective gate'),
        ('safeguard_scale', 'Accepted residual scale'),
        ('safeguard_gain', 'Predicted return gain from safeguard'),
        ('mpc_ms', 'MPC planning time (ms)'),
        ('episode_length', 'Episode length'),
    ]
    envs = sorted(df.env.dropna().unique())
    x = np.arange(len(envs))
    width = 0.18
    for metric, ylabel in metrics:
        if metric not in df.columns or not pd.to_numeric(df[metric], errors='coerce').notna().any():
            continue
        fig, ax = plt.subplots(figsize=(8.0, 4.8))
        for j, method in enumerate(METHODS):
            means, stds = [], []
            for env in envs:
                vals = df.loc[(df.env == env) & (df.method == method), metric].dropna()
                means.append(vals.mean() if len(vals) else np.nan)
                stds.append(vals.std(ddof=1) if len(vals) > 1 else 0.0)
            ax.bar(x + (j - 1.5) * width, means, width=width, yerr=stds, capsize=3, label=LABELS[method])
        ax.set_xticks(x, envs, rotation=15, ha='right')
        ax.set_ylabel(ylabel)
        ax.grid(axis='y', alpha=.25)
        ax.legend(fontsize=8)
        _save(fig, out, f'{metric}_mean_std')


def _choose_runs(root: Path):
    best = {}
    for f in glob.glob(str(root / '**' / 'episodes.csv'), recursive=True):
        try:
            d = pd.read_csv(f)
        except Exception:
            continue
        if d.empty or not {'env', 'method', 'seed', 'global_step', 'episode_return'}.issubset(d.columns):
            continue
        d = d[d.env.notna() & d.method.notna() & d.seed.notna()].copy()
        if d.empty:
            continue
        d['global_step'] = _num(d.global_step)
        d['episode_return'] = _num(d.episode_return)
        d = d.dropna(subset=['global_step', 'episode_return'])
        if d.empty:
            continue
        last = d.iloc[-1]
        key = (str(last.env), str(last.method), int(float(last.seed)))
        score = (float(d.global_step.max()), Path(f).stat().st_mtime)
        if key not in best or score > best[key][0]:
            best[key] = (score, d)
    return [v[1] for v in best.values()]


def learning_curves(root: Path, out: Path, smooth_episodes: int = 20, grid_n: int = 101):
    runs = _choose_runs(root)
    if not runs:
        return
    envs = sorted({str(d.iloc[-1].env) for d in runs})
    for env in envs:
        fig, ax = plt.subplots(figsize=(7.2, 4.8))
        plotted = False
        for method in METHODS:
            curves = []
            max_step = 0.0
            selected = []
            for d in runs:
                last = d.iloc[-1]
                if str(last.env) == env and str(last.method) == method:
                    q = d.sort_values('global_step')[['global_step', 'episode_return']].copy()
                    q['smooth'] = q.episode_return.rolling(smooth_episodes, min_periods=1).mean()
                    selected.append(q)
                    max_step = max(max_step, float(q.global_step.max()))
            if not selected or max_step <= 0:
                continue
            grid = np.linspace(0, max_step, grid_n)
            for q in selected:
                x = q.global_step.to_numpy(float)
                y = q.smooth.to_numpy(float)
                curves.append(np.interp(grid, x, y, left=y[0], right=y[-1]))
            a = np.asarray(curves, dtype=float)
            mean = np.nanmean(a, axis=0)
            std = np.nanstd(a, axis=0, ddof=1) if a.shape[0] > 1 else np.zeros_like(mean)
            ax.plot(grid, mean, label=LABELS[method])
            ax.fill_between(grid, mean - std, mean + std, alpha=.18)
            plotted = True
        if plotted:
            ax.set_xlabel('Environment steps')
            ax.set_ylabel(f'{smooth_episodes}-episode smoothed return')
            ax.set_title(env)
            ax.grid(alpha=.25)
            ax.legend()
            _save(fig, out, f'learning_curve_{env.replace("-v5", "")}')
        else:
            plt.close(fig)

    auc_rows = []
    for d in runs:
        last = d.iloc[-1]
        q = d.sort_values('global_step')[['global_step', 'episode_return']].copy()
        q['smooth'] = q.episode_return.rolling(smooth_episodes, min_periods=1).mean()
        x = q.global_step.to_numpy(float)
        y = q.smooth.to_numpy(float)
        if len(x) < 2 or x[-1] <= 0:
            continue
        xnorm = x / x[-1]
        auc_rows.append({
            'env': str(last.env),
            'method': str(last.method),
            'seed': int(float(last.seed)),
            'learning_curve_auc': float(np.trapezoid(np.r_[y[0], y], np.r_[0.0, xnorm])),
            'max_step': float(x[-1]),
        })
    if auc_rows:
        auc_df = pd.DataFrame(auc_rows)
        auc_df.to_csv(out / 'sample_efficiency_auc.csv', index=False)
        for env, g in auc_df.groupby('env'):
            stats = g.groupby('method')['learning_curve_auc'].agg(['mean', 'std']).reindex(METHODS)
            fig, ax = plt.subplots(figsize=(7.2, 4.8))
            x = np.arange(len(METHODS))
            ax.bar(x, stats['mean'], yerr=stats['std'].fillna(0), capsize=5)
            ax.set_xticks(x, [LABELS[m] for m in METHODS], rotation=15, ha='right')
            ax.set_ylabel('Learning-curve AUC (same step budget)')
            ax.set_title(f'{env}: sample efficiency')
            ax.grid(axis='y', alpha=.25)
            _save(fig, out, f'sample_efficiency_{env.replace("-v5", "")}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--summary', default='results/summary_per_seed.csv')
    p.add_argument('--runs', default='runs')
    p.add_argument('--out', default='results/paper_figures')
    p.add_argument('--smooth-episodes', type=int, default=20)
    a = p.parse_args()

    set_style()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df = load_summary(Path(a.summary))
    main_return_figures(df, out)
    paired_planning_delta(df, out)
    diagnostic_figures(df, out)
    learning_curves(Path(a.runs), out, a.smooth_episodes)
    print('paper figures saved to', out)


if __name__ == '__main__':
    main()
