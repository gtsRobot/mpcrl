from __future__ import annotations

import argparse
import copy
import subprocess
import sys
from pathlib import Path

from tqdm.auto import tqdm

from mpcrl.config import load_yaml, save_yaml


def deep_merge(base, override):
    out = copy.deepcopy(base)
    for key, value in override.items():
        if key == 'derived':
            continue
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def apply_derived(cfg, variant):
    derived = variant.get('derived', {}) if isinstance(variant, dict) else {}
    if 'residual_scale_multiplier' in derived:
        scale = float(cfg.get('zprl_style', {}).get('residual_scale', 0.05))
        cfg.setdefault('zprl_style', {})['residual_scale'] = (
            scale * float(derived['residual_scale_multiplier'])
        )
    return cfg


def build_jobs(suite, run_root, result_root):
    jobs = []
    steps = int(suite.get('steps', 50000))
    eval_episodes = int(suite.get('eval_episodes', 5))
    variants = suite.get('variants', {'full': {}})

    config_dir = Path(result_root) / 'configs'
    config_dir.mkdir(parents=True, exist_ok=True)

    for config_path in suite['tasks']:
        base = load_yaml(config_path)
        env_id = str(base['env']['id'])
        safe_env = env_id.replace('/', '_')
        for variant_name, variant_override in variants.items():
            for seed in suite['seeds']:
                cfg = deep_merge(base, variant_override or {})
                cfg = apply_derived(cfg, variant_override or {})
                cfg['env']['seed'] = int(seed)
                cfg['ablation'] = {
                    'name': str(variant_name),
                    'suite': 'zprl_ablation',
                }
                cfg.setdefault('logging', {})['root'] = str(
                    Path(run_root) / safe_env / str(variant_name)
                )

                generated = config_dir / f'{safe_env}__{variant_name}__seed{seed}.yaml'
                save_yaml(cfg, generated)
                cmd = [
                    sys.executable,
                    '-m',
                    'experiments.train_zprl_style',
                    '--config',
                    str(generated),
                    '--steps',
                    str(steps),
                    '--seed',
                    str(seed),
                    '--eval-episodes',
                    str(eval_episodes),
                ]
                jobs.append((env_id, variant_name, int(seed), cmd))
    return jobs


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--suite', default='configs/rtx5090/ablation_suite.yaml')
    p.add_argument('--run-root', required=True)
    p.add_argument('--result-root', required=True)
    p.add_argument('--start-job', type=int, default=1)
    a = p.parse_args()

    suite = load_yaml(a.suite)
    jobs = build_jobs(suite, a.run_root, a.result_root)
    start = max(1, int(a.start_job))
    if start > len(jobs):
        raise SystemExit(f'--start-job={start} exceeds total jobs={len(jobs)}')

    print(
        f'ZPRL ablation suite: {len(jobs)} jobs; start={start}; '
        f"steps={int(suite.get('steps', 50000))}; "
        f"eval={int(suite.get('eval_episodes', 5))} episodes"
    )

    bar = tqdm(
        total=len(jobs),
        initial=start - 1,
        desc='ZPRL ablations',
        unit='job',
        dynamic_ncols=True,
    )
    for index, (env_id, variant, seed, cmd) in enumerate(jobs, start=1):
        if index < start:
            continue
        bar.set_postfix_str(f'#{index} {env_id}/{variant}/seed{seed}')
        tqdm.write(
            f'\n=== ABLATION {index}/{len(jobs)}: '
            f'{env_id} | {variant} | seed={seed} ==='
        )
        subprocess.run(cmd, check=True)
        bar.update(1)
    bar.close()
    print('ZPRL ablation suite completed.')


if __name__ == '__main__':
    main()
