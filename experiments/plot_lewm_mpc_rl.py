from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager


def set_style():
    names = {f.name for f in font_manager.fontManager.ttflist}
    plt.rcParams['font.family'] = 'Times New Roman' if 'Times New Roman' in names else 'DejaVu Serif'
    plt.rcParams['font.size'] = 10.5
    plt.rcParams['axes.unicode_minus'] = False


def save(fig, stem):
    fig.tight_layout()
    fig.savefig(str(stem) + '.png', dpi=300, bbox_inches='tight')
    fig.savefig(str(stem) + '.pdf', bbox_inches='tight')
    plt.close(fig)


def plot_final_bars(summary, outdir):
    methods = ['lewm_mpc', 'lewm_mpc_rl']
    labels = ['LeWM-MPC', 'LeWM-MPC+RL']
    for env, g in summary.groupby('env'):
        g = g.set_index('method')
        vals, errs = [], []
        for m in methods:
            if m in g.index:
                vals.append(float(g.loc[m, 'eval_return_mean']))
                errs.append(float(g.loc[m, 'eval_return_std_across_seeds']))
            else:
                vals.append(np.nan)
                errs.append(0.0)
        fig, ax = plt.subplots(figsize=(5.4, 4.0))
        x = np.arange(2)
        ax.bar(x, vals, yerr=errs, capsize=4)
        ax.set_xticks(x, labels)
        ax.set_ylabel('Deterministic evaluation return')
        ax.set_title(env)
        ax.grid(axis='y', alpha=0.25)
        save(fig, Path(outdir) / f'{env}__return')

        if 'success_rate_mean_pct' in g.columns:
            succ = [
                float(g.loc[m, 'success_rate_mean_pct']) if m in g.index else np.nan
                for m in methods
            ]
            if np.isfinite(succ).any():
                serr = [
                    float(g.loc[m, 'success_rate_std_pct'])
                    if m in g.index and np.isfinite(g.loc[m, 'success_rate_std_pct'])
                    else 0.0
                    for m in methods
                ]
                fig, ax = plt.subplots(figsize=(5.4, 4.0))
                ax.bar(x, succ, yerr=serr, capsize=4)
                ax.set_xticks(x, labels)
                ax.set_ylim(0, 100)
                ax.set_ylabel('Success rate (%)')
                ax.set_title(env)
                ax.grid(axis='y', alpha=0.25)
                save(fig, Path(outdir) / f'{env}__success_rate')


def plot_paired(paired, outdir):
    if paired.empty:
        return
    for env, g in paired.groupby('env'):
        g = g.sort_values('seed')
        values = g['delta_return'].to_numpy(dtype=float)
        mean = float(np.mean(values))
        ci95 = 1.96 * float(np.std(values, ddof=1)) / np.sqrt(len(values)) if len(values) > 1 else 0.0
        fig, ax = plt.subplots(figsize=(5.8, 4.0))
        x = np.arange(len(g))
        ax.bar(x, values)
        ax.axhline(0.0, linewidth=1.0)
        ax.axhline(mean, linestyle='--', linewidth=1.2, label=f'Mean = {mean:.2f}')
        ax.fill_between([-0.5, len(g) - 0.5], mean - ci95, mean + ci95, alpha=0.12, label='95% CI')
        ax.set_xticks(x, [f"seed {int(s)}" for s in g['seed']])
        ax.set_ylabel('Return improvement (RL - MPC)')
        ax.set_title(f'{env}: paired improvement')
        ax.grid(axis='y', alpha=0.25)
        ax.legend(fontsize=8)
        save(fig, Path(outdir) / f'{env}__paired_delta')


def discover_rl_training(root):
    rows = []
    for f in Path(root).glob('**/episodes.csv'):
        try:
            q = pd.read_csv(f)
        except Exception:
            continue
        if q.empty or 'method' not in q or 'env' not in q or 'seed' not in q:
            continue
        method = str(q['method'].dropna().iloc[-1]) if q['method'].notna().any() else ''
        if method != 'lewm_mpc_rl':
            continue
        q = q.copy()
        q['global_step'] = pd.to_numeric(q['global_step'], errors='coerce')
        q['episode_return'] = pd.to_numeric(q['episode_return'], errors='coerce')
        if 'safeguard_scale' in q:
            q['safeguard_scale'] = pd.to_numeric(q['safeguard_scale'], errors='coerce')
        if 'safeguard_gain' in q:
            q['safeguard_gain'] = pd.to_numeric(q['safeguard_gain'], errors='coerce')
        q = q.dropna(subset=['global_step', 'episode_return'])
        if q.empty:
            continue
        env = str(q['env'].dropna().iloc[-1])
        seed = int(float(q['seed'].dropna().iloc[-1]))
        rows.append((env, seed, q))
    return rows


def plot_learning_curves(root, per_seed, outdir, points=300, smooth=5):
    records = discover_rl_training(root)
    for env in sorted({x[0] for x in records}):
        r = [(seed, q) for e, seed, q in records if e == env]
        if not r:
            continue
        xmax = max(float(q.global_step.max()) for _, q in r)
        grid = np.linspace(0, xmax, points)
        curves = []
        for seed, q in r:
            q = q.sort_values('global_step')
            y = q['episode_return'].rolling(smooth, min_periods=1, center=True).mean().to_numpy(float)
            x = q['global_step'].to_numpy(float)
            yi = np.interp(grid, x, y)
            yi[(grid < x.min()) | (grid > x.max())] = np.nan
            curves.append(yi)
        values = np.asarray(curves)
        mean = np.nanmean(values, axis=0)
        std = np.nanstd(values, axis=0, ddof=1) if len(curves) > 1 else np.zeros_like(mean)

        b = per_seed[(per_seed.env == env) & (per_seed.method == 'lewm_mpc')]
        baseline = float(b.eval_return_mean.mean()) if len(b) else np.nan

        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        ax.plot(grid, mean, label='LeWM-MPC+RL training')
        if len(curves) > 1:
            ax.fill_between(grid, mean - std, mean + std, alpha=0.18)
        if np.isfinite(baseline):
            ax.axhline(baseline, linestyle='--', label='LeWM-MPC final eval mean')
        ax.set_xlabel('RL environment steps')
        ax.set_ylabel('Episode return')
        ax.set_title(env)
        ax.grid(alpha=0.25)
        ax.legend()
        save(fig, Path(outdir) / f'{env}__rl_learning_curve')

        if any('safeguard_scale' in q and q.safeguard_scale.notna().any() for _, q in r):
            scale_curves = []
            for _, q in r:
                if 'safeguard_scale' not in q or not q.safeguard_scale.notna().any():
                    continue
                q = q.sort_values('global_step').dropna(subset=['safeguard_scale'])
                if q.empty:
                    continue
                x = q.global_step.to_numpy(float)
                y = q.safeguard_scale.to_numpy(float)
                yi = np.interp(grid, x, y)
                yi[(grid < x.min()) | (grid > x.max())] = np.nan
                scale_curves.append(yi)
            if scale_curves:
                values = np.asarray(scale_curves)
                mean = np.nanmean(values, axis=0)
                std = np.nanstd(values, axis=0, ddof=1) if len(values) > 1 else np.zeros_like(mean)
                fig, ax = plt.subplots(figsize=(6.4, 4.2))
                ax.plot(grid, mean)
                if len(values) > 1:
                    ax.fill_between(grid, mean - std, mean + std, alpha=0.18)
                ax.set_ylim(-0.02, 1.02)
                ax.set_xlabel('RL environment steps')
                ax.set_ylabel('Accepted residual scale')
                ax.set_title(f'{env}: model safeguard')
                ax.grid(alpha=0.25)
                save(fig, Path(outdir) / f'{env}__safeguard_scale')


def plot_rank(summary, outdir):
    if 'effective_rank_mean' not in summary:
        return
    q = (
        summary.groupby('env', as_index=False)['effective_rank_mean']
        .mean()
        .sort_values('env')
    )
    if q.empty:
        return
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    x = np.arange(len(q))
    ax.bar(x, q['effective_rank_mean'].to_numpy(float))
    ax.set_xticks(x, q['env'].tolist(), rotation=15)
    ax.set_ylabel('LeWM latent effective rank')
    ax.set_title('Shared representation diagnostic')
    ax.grid(axis='y', alpha=0.25)
    save(fig, Path(outdir) / 'shared_lewm_effective_rank')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    p.add_argument('--summary-dir', required=True)
    p.add_argument('--outdir', required=True)
    p.add_argument('--points', type=int, default=300)
    p.add_argument('--smooth-window', type=int, default=5)
    a = p.parse_args()

    set_style()
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    summary_dir = Path(a.summary_dir)
    summary = pd.read_csv(summary_dir / 'direct_summary.csv')
    paired = pd.read_csv(summary_dir / 'paired_improvements.csv')
    per_seed = pd.read_csv(summary_dir / 'direct_per_seed.csv')

    plot_final_bars(summary, out)
    plot_paired(paired, out)
    plot_learning_curves(a.root, per_seed, out, a.points, a.smooth_window)
    plot_rank(summary, out)
    print('LeWM MPC+RL paper figures saved to:', out)


if __name__ == '__main__':
    main()
