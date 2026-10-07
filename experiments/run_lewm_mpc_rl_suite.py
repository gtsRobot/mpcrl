from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from mpcrl.config import load_yaml


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--suite', default='configs/rtx5090/lewm_mpc_rl_suite.yaml')
    p.add_argument('--run-root', required=True)
    p.add_argument('--results-dir', default=None)
    p.add_argument('--start-job', type=int, default=1)
    p.add_argument('--no-postprocess', action='store_true')
    a = p.parse_args()

    suite = load_yaml(a.suite)
    jobs = []
    for cfg in suite['tasks']:
        for seed in suite['seeds']:
            jobs.append((cfg, int(seed)))

    start = max(1, int(a.start_job))
    if start > len(jobs):
        raise SystemExit(f'--start-job={start} exceeds total jobs={len(jobs)}')

    print(
        f"Direct LeWM suite: {len(jobs)} paired jobs; each job produces "
        f"LeWM-MPC and LeWM-MPC+RL using one shared frozen LeWM."
    )
    print(f"run root: {a.run_root}")

    for idx, (cfg, seed) in enumerate(jobs, start=1):
        if idx < start:
            print(f'SKIP job {idx}/{len(jobs)}: {Path(cfg).stem} seed={seed}')
            continue
        print('\n' + '=' * 88)
        print(f'JOB {idx}/{len(jobs)} | {Path(cfg).stem} | seed={seed}')
        print('=' * 88)
        cmd = [
            sys.executable, '-m', 'experiments.train_lewm_mpc_rl',
            '--config', cfg,
            '--seed', str(seed),
            '--rl-steps', str(int(suite.get('rl_steps', 50000))),
            '--eval-episodes', str(int(suite.get('eval_episodes', 10))),
            '--run-root', a.run_root,
        ]
        subprocess.run(cmd, check=True)

    print('\nDirect LeWM MPC+RL suite completed.')
    if a.no_postprocess:
        return

    results = Path(a.results_dir or (Path('results') / Path(a.run_root).name))
    summary_dir = results / 'summary'
    figure_dir = results / 'paper_figures'
    summary_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    envs = [load_yaml(cfg)['env']['id'] for cfg in suite['tasks']]
    seeds = [str(int(x)) for x in suite['seeds']]
    subprocess.run([
        sys.executable, '-m', 'experiments.summarize_lewm_mpc_rl',
        '--root', a.run_root,
        '--outdir', str(summary_dir),
        '--envs', *envs,
        '--seeds', *seeds,
        '--require-complete',
    ], check=True)
    subprocess.run([
        sys.executable, '-m', 'experiments.plot_lewm_mpc_rl',
        '--root', a.run_root,
        '--summary-dir', str(summary_dir),
        '--outdir', str(figure_dir),
    ], check=True)
    print('summary:', summary_dir)
    print('paper figures:', figure_dir)


if __name__ == '__main__':
    main()
