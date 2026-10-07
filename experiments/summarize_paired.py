"""Seed-level paired summary for experiments.train_paired runs.

The headline statistic is the per-seed paired difference (Arm B - Arm A), because
the seed is the independent unit: episodes within a seed share a world model.
Reports mean, a Student-t 95% CI (not the normal 1.96, which understates the
interval by 42% at n=5), the number of positive seeds, an exact Wilcoxon
signed-rank p, and the minimum detectable effect for the n actually used.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from mpcrl.paired_stats import minimum_detectable_effect, paired_summary


def load_pairs(root):
    """Collect (env, seed, armA mean, armB mean) from pair_summary.json files."""
    rows = []
    for path in glob.glob(os.path.join(root, '**', 'pair_summary.json'), recursive=True):
        try:
            with open(path, encoding='utf-8') as f:
                s = json.load(f)
        except Exception as exc:
            print(f'warning: skipping {path}: {exc}')
            continue
        if 'armA_mpc_return_mean' not in s:
            continue  # LeWM pair_summary.json has a different schema
        rows.append({
            'env': s.get('env'),
            'seed': s.get('seed'),
            'armA': float(s['armA_mpc_return_mean']),
            'armB': float(s['armB_residual_return_mean']),
            'delta': float(s['delta_mean_within_seed']),
            'wm_fingerprint': s.get('world_model_fingerprint', '')[:16],
            'safeguard_mode': s.get('safeguard_mode'),
            'cem_execute': s.get('cem_execute'),
            'residual_hold_steps': s.get('residual_hold_steps'),
            'eval_episodes': s.get('eval_episodes'),
            'total_env_steps': s.get('total_env_steps'),
            'crn': s.get('common_random_numbers'),
            'path': str(Path(path).parent),
        })
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', default='runs')
    p.add_argument('--out', default='results/paired_summary.csv')
    p.add_argument('--min-seeds', type=int, default=6,
                   help='warn below this. n=5 cannot reach Wilcoxon p<0.05 (floor 0.0625)')
    a = p.parse_args()

    df = load_pairs(a.root)
    if df.empty:
        raise SystemExit(f'no paired runs with pair_summary.json found under {a.root}')

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Guard against averaging seeds that did not actually share a protocol.
    for env, g in df.groupby('env'):
        for col in ('safeguard_mode', 'cem_execute', 'residual_hold_steps', 'total_env_steps'):
            vals = set(map(str, g[col].dropna().tolist()))
            if len(vals) > 1:
                print(f'WARNING [{env}]: seeds disagree on {col}: {sorted(vals)}. '
                      'These runs are not comparable; the aggregate below mixes protocols.')
        dupes = g['seed'][g['seed'].duplicated()].tolist()
        if dupes:
            print(f'WARNING [{env}]: duplicate seeds {sorted(set(dupes))}; '
                  'keeping the last of each.')

    df = df.drop_duplicates(subset=['env', 'seed'], keep='last').sort_values(['env', 'seed'])
    df.to_csv(out.with_name(out.stem + '_per_seed.csv'), index=False)

    report = []
    for env, g in df.groupby('env'):
        deltas = g['delta'].tolist()
        s = paired_summary(deltas)
        base = float(np.mean(g['armA']))
        row = {
            'env': env,
            'n_seeds': s['n_seeds'],
            'armA_mpc_mean': base,
            'armB_residual_mean': float(np.mean(g['armB'])),
            'paired_delta_mean': s['mean_delta'],
            'paired_delta_sd': s['sd_delta'],
            'ci95_low': s['ci95_low'],
            'ci95_high': s['ci95_high'],
            'ci95_half_width': s['ci95_half_width'],
            'relative_pct': (s['mean_delta'] / max(abs(base), 1e-9) * 100.0),
            'n_positive_seeds': s['n_positive'],
            'n_negative_seeds': s['n_negative'],
            'wilcoxon_p': s['wilcoxon_p'],
            'wilcoxon_p_floor': s['wilcoxon_p_floor'],
            'ci_excludes_zero': s['significant_at_95'],
            'mde_at_this_n': minimum_detectable_effect(s['sd_delta'], s['n_seeds']),
        }
        report.append(row)

    rep = pd.DataFrame(report).sort_values('env')
    rep.to_csv(out, index=False)

    pd.set_option('display.width', 200)
    print('\nPer-seed paired differences (Arm B - Arm A):')
    print(df[['env', 'seed', 'armA', 'armB', 'delta']].to_string(index=False))
    print('\nSeed-level paired summary:')
    cols = ['env', 'n_seeds', 'armA_mpc_mean', 'armB_residual_mean', 'paired_delta_mean',
            'ci95_low', 'ci95_high', 'n_positive_seeds', 'wilcoxon_p', 'ci_excludes_zero',
            'mde_at_this_n']
    print(rep[cols].to_string(index=False))

    print('\nInterpretation guard rails:')
    for _, r in rep.iterrows():
        n = int(r['n_seeds'])
        msgs = []
        if n < a.min_seeds:
            floor = r['wilcoxon_p_floor']
            if floor != floor:
                msgs.append(f'n={n} is too small for any inferential statement')
            else:
                msgs.append(
                    f"n={n} < {a.min_seeds}; the exact Wilcoxon floor is {floor:.4f}, so "
                    f"p<0.05 is {'unreachable' if floor > 0.05 else 'barely reachable'} "
                    'at this sample size regardless of effect size')
        if not r['ci_excludes_zero']:
            msgs.append(
                f"95% CI [{r['ci95_low']:.2f}, {r['ci95_high']:.2f}] includes 0 -- "
                'no detectable difference')
        if r['mde_at_this_n'] == r['mde_at_this_n'] and abs(r['paired_delta_mean']) < r['mde_at_this_n']:
            msgs.append(
                f"|delta|={abs(r['paired_delta_mean']):.2f} is below the minimum detectable "
                f"effect {r['mde_at_this_n']:.2f} at n={n}")
        print(f"  {r['env']}: " + ('; '.join(msgs) if msgs else 'CI excludes zero.'))

    print('\nsaved', out)
    print('saved', out.with_name(out.stem + '_per_seed.csv'))


if __name__ == '__main__':
    main()
