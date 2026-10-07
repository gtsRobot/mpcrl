from __future__ import annotations

import argparse
import glob
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from mpcrl.config import load_yaml
from mpcrl.plotting import outside_legend


VARIANT_ORDER = [
    'full',
    'no_dagger',
    'no_consistency',
    'no_latent_residual',
    'latent_dim_16',
    'latent_dim_64',
    'residual_scale_half',
    'residual_scale_1p5x',
]


def ci95(std, n):
    if n <= 1 or not np.isfinite(std):
        return 0.0
    # Normal approximation is reported as a descriptive interval. With n=3,
    # the raw std is also preserved in the CSV and should be shown in the paper.
    return 1.96 * float(std) / math.sqrt(int(n))


def load_rows(root):
    rows = []
    for eval_file in glob.glob(str(Path(root) / '**' / 'eval.csv'), recursive=True):
        run_dir = Path(eval_file).parent
        config_file = run_dir / 'config.yaml'
        if not config_file.exists():
            continue
        try:
            cfg = load_yaml(config_file)
            variant = str(cfg.get('ablation', {}).get('name', 'unknown'))
            env_id = str(cfg['env']['id'])
            seed = int(cfg['env']['seed'])
            df = pd.read_csv(eval_file)
        except Exception:
            continue
        if df.empty or 'episode_return' not in df:
            continue
        returns = pd.to_numeric(df['episode_return'], errors='coerce').dropna()
        if returns.empty:
            continue
        row = {
            'env': env_id,
            'variant': variant,
            'seed': seed,
            'eval_return': float(returns.mean()),
            'eval_episode_std': float(returns.std(ddof=1)) if len(returns) > 1 else 0.0,
            'eval_episodes': int(len(returns)),
            'run_dir': str(run_dir),
        }
        for metric in [
            'episode_length',
            'action_d1',
            'action_d2',
            'effective_residual_norm',
            'residual_norm',
            'gate',
        ]:
            if metric in df:
                values = pd.to_numeric(df[metric], errors='coerce').dropna()
                row[metric] = float(values.mean()) if len(values) else np.nan
            else:
                row[metric] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def summarize(per_seed):
    metrics = [
        'eval_return',
        'episode_length',
        'action_d1',
        'action_d2',
        'effective_residual_norm',
        'residual_norm',
        'gate',
    ]
    rows = []
    for (env_id, variant), g in per_seed.groupby(['env', 'variant']):
        row = {
            'env': env_id,
            'variant': variant,
            'seeds': int(g['seed'].nunique()),
        }
        for metric in metrics:
            vals = pd.to_numeric(g[metric], errors='coerce').dropna()
            mean = float(vals.mean()) if len(vals) else np.nan
            std = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
            row[f'{metric}_mean'] = mean
            row[f'{metric}_std'] = std
            row[f'{metric}_ci95_half_width'] = ci95(std, len(vals))
        rows.append(row)
    return pd.DataFrame(rows)


def ordered(g):
    rank = {name: i for i, name in enumerate(VARIANT_ORDER)}
    return g.assign(
        _rank=g['variant'].map(lambda x: rank.get(x, 999))
    ).sort_values(['_rank', 'variant']).drop(columns=['_rank'])


def plot_metric(summary, outdir, metric, ylabel):
    for env_id, g in summary.groupby('env'):
        g = ordered(g).reset_index(drop=True)
        x = np.arange(len(g))
        mean = g[f'{metric}_mean'].to_numpy(dtype=float)
        std = g[f'{metric}_std'].fillna(0.0).to_numpy(dtype=float)

        fig, ax = plt.subplots(figsize=(8.2, 4.6))
        ax.bar(x, mean, yerr=std, capsize=4)
        ax.set_xticks(x, g['variant'].tolist(), rotation=25, ha='right')
        ax.set_ylabel(ylabel)
        ax.set_title(f'{env_id} ZPRL ablation')
        ax.grid(axis='y', alpha=0.25)
        fig.tight_layout()

        safe_env = env_id.replace('/', '_')
        stem = Path(outdir) / f'{safe_env}__ablation__{metric}'
        fig.savefig(str(stem) + '.png', dpi=300, bbox_inches='tight')
        fig.savefig(str(stem) + '.pdf', bbox_inches='tight')
        plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    p.add_argument('--outdir', required=True)
    p.add_argument('--expected-seeds', nargs='*', type=int, default=[0, 1, 2])
    p.add_argument('--require-complete', action='store_true')
    a = p.parse_args()

    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    per_seed = load_rows(a.root)
    if per_seed.empty:
        raise SystemExit('No ablation eval.csv files found.')

    per_seed = per_seed.sort_values(['env', 'variant', 'seed'])
    per_seed.to_csv(outdir / 'ablation_per_seed.csv', index=False)
    summary = summarize(per_seed)
    summary.to_csv(outdir / 'ablation_summary.csv', index=False)

    coverage_rows = []
    expected = set(a.expected_seeds or [])
    for (env_id, variant), g in per_seed.groupby(['env', 'variant']):
        found = set(int(x) for x in g['seed'].unique())
        missing = sorted(expected - found)
        coverage_rows.append({
            'env': env_id,
            'variant': variant,
            'found_seeds': ','.join(map(str, sorted(found))),
            'missing_seeds': ','.join(map(str, missing)),
            'complete': len(missing) == 0,
        })
    coverage = pd.DataFrame(coverage_rows)
    coverage.to_csv(outdir / 'ablation_coverage.csv', index=False)

    if a.require_complete and not bool(coverage['complete'].all()):
        print(coverage.to_string(index=False))
        raise SystemExit('Ablation suite is incomplete; see ablation_coverage.csv')

    plot_metric(summary, outdir, 'eval_return', 'Deterministic evaluation return')
    plot_metric(summary, outdir, 'action_d1', 'First-order action variation')
    plot_metric(summary, outdir, 'effective_residual_norm', 'Effective residual norm')

    print('\nAblation summary:')
    print(
        summary[
            ['env', 'variant', 'seeds', 'eval_return_mean', 'eval_return_std']
        ].to_string(index=False)
    )
    print('saved:', outdir)


if __name__ == '__main__':
    main()
