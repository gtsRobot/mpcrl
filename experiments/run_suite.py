from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

from tqdm.auto import tqdm

from mpcrl.config import load_yaml


def build_jobs(suite, include_lewm=False):
    jobs = []
    eval_episodes = int(suite.get('eval_episodes', 5))
    for cfg in suite['tasks']:
        env_name = Path(cfg).stem
        for method in suite['methods']:
            for seed in suite['seeds']:
                if method == 'zprl_style':
                    cmd = [
                        sys.executable, '-m', 'experiments.train_zprl_style',
                        '--config', cfg,
                        '--seed', str(seed),
                        '--steps', str(suite['steps']),
                        '--eval-episodes', str(eval_episodes),
                    ]
                else:
                    cmd = [
                        sys.executable, '-m', 'experiments.train',
                        '--config', cfg,
                        '--method', method,
                        '--seed', str(seed),
                        '--steps', str(suite['steps']),
                        '--eval-episodes', str(eval_episodes),
                    ]
                jobs.append((env_name, method, seed, cmd))

    if include_lewm or suite.get('run_lewm', False):
        for cfg in suite.get('lewm_configs', []):
            env_name = Path(cfg).stem.replace('lewm_', '')
            for seed in suite['seeds']:
                cmd = [
                    sys.executable, '-m', 'experiments.train_lewm',
                    '--config', cfg,
                    '--seed', str(seed),
                    '--episodes', str(eval_episodes),
                ]
                jobs.append((env_name, 'lewm_mpc', seed, cmd))
    return jobs


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--suite', default='configs/rtx5090/paper_suite.yaml')
    p.add_argument('--include-lewm', action='store_true',
                   help='add the LeWM jobs. Note: a suite with run_lewm: true already '
                        'includes them, so this flag is a no-op for those suites.')
    p.add_argument(
        '--start-job',
        type=int,
        default=1,
        help='1-based job index; useful after an interrupted suite',
    )
    p.add_argument('--keep-going', action='store_true', default=True,
                   help='record failures and continue (default). One bad job used to '
                        'abort the entire 100-job suite.')
    p.add_argument('--fail-fast', dest='keep_going', action='store_false',
                   help='stop at the first failing job')
    p.add_argument('--failure-log', default='results/suite_failures.csv')
    a = p.parse_args()

    suite = load_yaml(a.suite)
    jobs = build_jobs(suite, include_lewm=a.include_lewm)
    total_jobs = len(jobs)
    start = max(1, int(a.start_job))
    if start > total_jobs:
        raise SystemExit(f'--start-job={start} exceeds total jobs={total_jobs}')

    print(
        f"Paper suite: {total_jobs} jobs, starting from job {start}, "
        f"final eval={int(suite.get('eval_episodes', 5))} episodes/run"
    )
    bar = tqdm(
        total=total_jobs,
        initial=start - 1,
        desc='paper suite',
        unit='job',
        dynamic_ncols=True,
    )

    failures = []
    for job_idx, (env_name, method, seed, cmd) in enumerate(jobs, start=1):
        if job_idx < start:
            continue
        bar.set_postfix_str(f'#{job_idx} {env_name}/{method}/seed{seed}')
        tqdm.write(
            f'\n=== JOB {job_idx}/{total_jobs}: '
            f'{env_name} | {method} | seed={seed} ==='
        )
        result = subprocess.run(cmd)
        if result.returncode != 0:
            failures.append({'job': job_idx, 'env': env_name, 'method': method,
                             'seed': seed, 'returncode': result.returncode,
                             'cmd': ' '.join(cmd)})
            tqdm.write(f'JOB {job_idx} FAILED with exit code {result.returncode}')
            if not a.keep_going:
                bar.close()
                _write_failures(a.failure_log, failures)
                raise SystemExit(f'aborting at job {job_idx} (--fail-fast)')
        bar.update(1)

    bar.close()
    _write_failures(a.failure_log, failures)
    if failures:
        print(f'\nPaper suite finished with {len(failures)}/{total_jobs} FAILED jobs:')
        for f in failures:
            print(f"  job {f['job']}: {f['env']} | {f['method']} | seed={f['seed']} "
                  f"(exit {f['returncode']})")
        print(f'\nRe-run just those with --start-job, or inspect {a.failure_log}.')
        raise SystemExit(1)
    print('Paper suite completed with no failures.')


def _write_failures(path, failures):
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['job', 'env', 'method', 'seed', 'returncode', 'cmd'])
        w.writeheader()
        for row in failures:
            w.writerow(row)


if __name__ == '__main__':
    main()
