from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager

from mpcrl.plotting import outside_legend


METHOD_ORDER = ['mpc_only', 'action_residual', 'planning_residual', 'zprl_style']
METHOD_LABEL = {
    'mpc_only': 'MPC',
    'action_residual': 'Action residual',
    'planning_residual': 'Planning residual',
    'zprl_style': 'ZPRL',
}
METRICS = [
    ('episode_return', 'Episode return'),
    ('mpc_ms', 'MPC planning time / ms'),
    ('action_d1', 'First-order action variation'),
    ('action_d2', 'Second-order action variation'),
    ('prediction_mse', 'One-step prediction MSE'),
    ('effective_residual_norm', 'Effective residual norm'),
    ('gate', 'Effective gate'),
]


def set_paper_style():
    """Headless-safe paper style; all labels are ASCII to avoid CJK glyph issues."""
    names = {f.name for f in font_manager.fontManager.ttflist}
    for name in ('Times New Roman', 'Liberation Serif', 'DejaVu Serif'):
        if name in names:
            plt.rcParams['font.family'] = name
            break
    plt.rcParams.update({
        'font.size': 10.5,
        'axes.labelsize': 10.5,
        'axes.titlesize': 10.5,
        'legend.fontsize': 9.0,
        'axes.linewidth': 0.8,
        'lines.linewidth': 1.2,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'axes.unicode_minus': False,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    })


def method_sort_key(method):
    try:
        return METHOD_ORDER.index(method)
    except ValueError:
        return len(METHOD_ORDER)


def load_best_runs(root, env_filter=None, method_filter=None):
    """Keep only the longest/newest run for each (env, method, seed)."""
    best = {}
    env_filter = set(env_filter or [])
    method_filter = set(method_filter or [])
    for f in glob.glob(os.path.join(root, '**', 'episodes.csv'), recursive=True):
        try:
            df = pd.read_csv(f)
        except Exception:
            continue
        required = {'env', 'method', 'seed', 'global_step'}
        if df.empty or not required.issubset(df.columns):
            continue

        df = df.copy()
        df['global_step'] = pd.to_numeric(df['global_step'], errors='coerce')
        df = df.dropna(subset=['env', 'method', 'seed', 'global_step'])
        if df.empty:
            continue

        last = df.iloc[-1]
        env = str(last['env'])
        method = str(last['method'])
        if env_filter and env not in env_filter:
            continue
        if method_filter and method not in method_filter:
            continue

        seed = int(float(last['seed']))
        key = (env, method, seed)
        score = (float(df['global_step'].max()), Path(f).stat().st_mtime)
        if key not in best or score > best[key]['score']:
            best[key] = {
                'env': env,
                'method': method,
                'seed': seed,
                'run_dir': Path(f).parent,
                'df': df,
                'score': score,
            }
    return list(best.values())


def write_completeness(records, outdir, expected_methods, expected_seeds, min_steps):
    rows = []
    envs = sorted({r['env'] for r in records})
    by_key = {(r['env'], r['method'], r['seed']): r for r in records}
    for env in envs:
        for method in expected_methods:
            for seed in expected_seeds:
                r = by_key.get((env, method, seed))
                max_step = r['score'][0] if r is not None else np.nan
                eval_exists = bool(r is not None and (r['run_dir'] / 'eval.csv').exists())
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
    q = pd.DataFrame(rows)
    path = Path(outdir) / 'experiment_completeness.csv'
    q.to_csv(path, index=False)
    return q


def smooth_run(df, metric, window):
    if metric not in df.columns:
        return None
    q = df[['global_step', metric]].copy()
    q['global_step'] = pd.to_numeric(q['global_step'], errors='coerce')
    q[metric] = pd.to_numeric(q[metric], errors='coerce')
    q = (
        q.dropna()
        .groupby('global_step', as_index=False)[metric].mean()
        .sort_values('global_step')
    )
    if len(q) < 2:
        return None
    q[metric] = q[metric].rolling(
        max(1, int(window)), min_periods=1, center=True
    ).mean()
    return q


def align_seed_curves(runs, metric, points, window):
    prepared = []
    for r in runs:
        q = smooth_run(r['df'], metric, window)
        if q is not None:
            prepared.append((r, q))
    if not prepared:
        return None

    xmin = min(float(q['global_step'].min()) for _, q in prepared)
    xmax = max(float(q['global_step'].max()) for _, q in prepared)
    grid = np.linspace(xmin, xmax, max(20, int(points)))
    values, seeds = [], []
    for r, q in prepared:
        x = q['global_step'].to_numpy(dtype=float)
        y = q[metric].to_numpy(dtype=float)
        yi = np.interp(grid, x, y)
        yi[(grid < x.min()) | (grid > x.max())] = np.nan
        values.append(yi)
        seeds.append(r['seed'])
    return grid, np.asarray(values, dtype=float), seeds


def curve_stats(values):
    count = np.sum(np.isfinite(values), axis=0)
    mean = np.nanmean(values, axis=0)
    std = np.full_like(mean, np.nan, dtype=float)
    ci95 = np.full_like(mean, np.nan, dtype=float)
    valid = count >= 2
    if np.any(valid):
        with np.errstate(invalid='ignore', divide='ignore'):
            all_std = np.nanstd(values, axis=0, ddof=1)
        std[valid] = all_std[valid]
        ci95[valid] = 1.96 * std[valid] / np.sqrt(count[valid])
    return count, mean, std, ci95


def save_curve_data(path, grid, values, seeds):
    count, mean, std, ci95 = curve_stats(values)
    data = {'global_step': grid}
    for seed, y in zip(seeds, values):
        data[f'seed_{seed}'] = y
    data.update({
        'n_seeds': count,
        'mean': mean,
        'std': std,
        'ci95_half_width': ci95,
    })
    pd.DataFrame(data).to_csv(path, index=False)


def plot_training_curves(records, outdir, points, window, min_steps, error_band):
    data_dir = Path(outdir) / 'data'
    data_dir.mkdir(parents=True, exist_ok=True)

    envs = sorted({r['env'] for r in records})
    for env in envs:
        env_runs = [r for r in records if r['env'] == env and r['score'][0] >= min_steps]
        if not env_runs:
            continue
        safe_env = env.replace('/', '_')

        methods = sorted({r['method'] for r in env_runs}, key=method_sort_key)
        for metric, ylabel in METRICS:
            fig, ax = plt.subplots(figsize=(6.4, 4.2))
            plotted = False
            for method in methods:
                mruns = [r for r in env_runs if r['method'] == method]
                aligned = align_seed_curves(mruns, metric, points, window)
                if aligned is None:
                    continue

                grid, values, seeds = aligned
                count, mean, std, ci95 = curve_stats(values)
                band = std if error_band == 'std' else ci95
                valid = count >= 2

                label = f"{METHOD_LABEL.get(method, method)} (n={len(seeds)})"
                line, = ax.plot(grid, mean, label=label)
                if np.any(valid):
                    ax.fill_between(
                        grid,
                        mean - band,
                        mean + band,
                        where=valid,
                        alpha=0.18,
                        color=line.get_color(),
                    )

                save_curve_data(
                    data_dir / f'{safe_env}__{method}__{metric}.csv',
                    grid, values, seeds,
                )
                plotted = True

            if not plotted:
                plt.close(fig)
                continue

            ax.set_xlabel('Environment steps')
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.25)
            outside_legend(ax)
            fig.tight_layout()

            stem = Path(outdir) / f'{safe_env}__{metric}'
            fig.savefig(str(stem) + '.png', dpi=300, bbox_inches='tight')
            fig.savefig(str(stem) + '.pdf', bbox_inches='tight')
            plt.close(fig)


def plot_final_eval(records, outdir, min_steps):
    rows = []
    for r in records:
        if r['score'][0] < min_steps:
            continue
        f = r['run_dir'] / 'eval.csv'
        if not f.exists():
            continue
        try:
            e = pd.read_csv(f)
        except Exception:
            continue

        vals = pd.to_numeric(e.get('episode_return'), errors='coerce').dropna()
        if not len(vals):
            continue

        row = {
            'env': r['env'],
            'method': r['method'],
            'seed': r['seed'],
            'eval_return': float(vals.mean()),
            'eval_episode_std': float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
            'eval_episodes': int(len(vals)),
        }
        for col in ('episode_length', 'action_d1', 'action_d2',
                    'effective_residual_norm', 'gate'):
            if col in e:
                v = pd.to_numeric(e[col], errors='coerce')
                row[col] = float(v.mean())
        rows.append(row)

    if not rows:
        return

    q = pd.DataFrame(rows).sort_values(['env', 'method', 'seed'])
    q.to_csv(Path(outdir) / 'final_eval_per_seed.csv', index=False)

    stats = (
        q.groupby(['env', 'method'])['eval_return']
        .agg(['mean', 'std', 'count'])
        .reset_index()
    )
    stats['ci95_half_width'] = np.where(
        stats['count'] >= 2,
        1.96 * stats['std'] / np.sqrt(stats['count']),
        np.nan,
    )
    stats.to_csv(Path(outdir) / 'final_eval_mean_std_ci95.csv', index=False)

    for env, g in stats.groupby('env'):
        g = g.copy()
        g['order'] = g['method'].map(lambda m: method_sort_key(m))
        g = g.sort_values(['order', 'method']).reset_index(drop=True)

        x = np.arange(len(g))
        yerr = g['std'].fillna(0.0).to_numpy(dtype=float)
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        bars = ax.bar(x, g['mean'].to_numpy(dtype=float), yerr=yerr, capsize=4)

        labels = [
            f"{METHOD_LABEL.get(m, m)}\n(n={int(n)})"
            for m, n in zip(g['method'], g['count'])
        ]
        ax.set_xticks(x, labels)
        ax.set_ylabel('Deterministic final evaluation return')
        ax.grid(axis='y', alpha=0.25)

        for bar, value in zip(bars, g['mean']):
            ax.annotate(
                f'{value:.1f}',
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 4),
                textcoords='offset points',
                ha='center',
                va='bottom',
                fontsize=8.5,
            )

        fig.tight_layout()
        safe_env = env.replace('/', '_')
        stem = Path(outdir) / f'{safe_env}__final_eval_return'
        fig.savefig(str(stem) + '.png', dpi=300, bbox_inches='tight')
        fig.savefig(str(stem) + '.pdf', bbox_inches='tight')
        plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', default='runs')
    p.add_argument('--outdir', default='results/figures')
    # Accepted for compatibility with README / scripts/run_main_suite.sh, which
    # have always passed it. This script reads the per-run episodes.csv/eval.csv
    # under --root directly and does not need the aggregated summary, so the
    # value is only used to locate --root when it is left at its default.
    p.add_argument(
        '--summary',
        help='path to results/summary.csv from experiments.aggregate (optional; '
             'only used to infer --root when --root is not given)',
    )
    p.add_argument('--env', action='append', help='exact environment id; repeatable')
    p.add_argument('--methods', nargs='+', default=METHOD_ORDER)
    p.add_argument('--smooth-window', type=int, default=10)
    p.add_argument('--points', type=int, default=300)
    p.add_argument('--min-steps', type=int, default=0)
    p.add_argument('--expected-methods', nargs='+', default=METHOD_ORDER)
    p.add_argument('--expected-seeds', nargs='+', type=int, default=[0, 1, 2, 3, 4])
    p.add_argument('--require-complete', action='store_true')
    p.add_argument('--error-band', choices=['std', 'ci95'], default='std')
    a = p.parse_args()

    if a.summary:
        summary_path = Path(a.summary)
        if not summary_path.exists():
            print(f'warning: --summary {summary_path} does not exist; ignoring it.')
        else:
            # summary_per_seed.csv records the source run dir per row; use it to
            # locate runs when the caller did not override --root.
            per_seed = summary_path.with_name(summary_path.stem + '_per_seed.csv')
            if a.root == 'runs' and per_seed.exists():
                try:
                    paths = pd.read_csv(per_seed).get('run_path')
                    if paths is not None and len(paths.dropna()):
                        common = os.path.commonpath([str(x) for x in paths.dropna()])
                        if Path(common).is_dir():
                            a.root = common
                            print(f'--summary: resolved --root to {a.root}')
                except Exception as exc:
                    print(f'warning: could not read {per_seed}: {exc}')

    set_paper_style()
    Path(a.outdir).mkdir(parents=True, exist_ok=True)

    records = load_best_runs(a.root, a.env, a.methods)
    if not records:
        raise SystemExit('No valid runs matched the requested filters.')

    completeness = write_completeness(
        records,
        a.outdir,
        a.expected_methods,
        a.expected_seeds,
        a.min_steps,
    )
    missing = completeness.loc[~completeness['complete']]
    if len(missing):
        print('\nIncomplete expected runs:')
        print(
            missing[
                ['env', 'method', 'seed', 'run_found', 'max_step',
                 'step_complete', 'eval_exists']
            ].to_string(index=False)
        )
        if a.require_complete:
            raise SystemExit(
                'Experiment is incomplete; figures marked as final were not generated.'
            )

    plot_training_curves(
        records, a.outdir, a.points, a.smooth_window, a.min_steps, a.error_band
    )
    plot_final_eval(records, a.outdir, a.min_steps)
    print(f'Paper figures saved to: {a.outdir}')


if __name__ == '__main__':
    main()
