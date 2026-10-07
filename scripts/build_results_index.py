"""Build a category-aware navigation table from packaged raw evaluations."""
from pathlib import Path
import csv
import itertools
import statistics


def write_rows(path, rows):
    if not rows:
        raise SystemExit('No archived evaluation rows found')
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    root = Path(__file__).resolve().parents[1]
    archive = root / 'results/archived_runs'
    rows = []
    for path in sorted(archive.rglob('eval.csv')):
        with path.open() as handle:
            data = list(csv.DictReader(handle))
        if not data:
            continue
        first = data[0]
        values = [float(row['episode_return']) for row in data]
        step = None
        episodes = path.parent / 'episodes.csv'
        if episodes.exists():
            with episodes.open() as handle:
                steps = [float(row['global_step']) for row in csv.DictReader(handle)
                         if row.get('global_step')]
            if steps:
                step = max(steps)
        rows.append({
            'category': path.relative_to(archive).parts[0],
            'env': first['env'], 'method': first['method'],
            'seed': first.get('train_seed', first.get('seed')),
            'evaluation_episodes': len(values), 'eval_return_mean': statistics.mean(values),
            'max_logged_training_step': step,
            'run_dir': path.parent.relative_to(root).as_posix(),
            'checkpoint_in_package': False,
        })
    write_rows(root / 'results/experiment_index.csv', rows)
    summary = []
    group_key = lambda row: (row['category'], row['env'], row['method'])
    for key, group in itertools.groupby(sorted(rows, key=group_key), key=group_key):
        group = list(group)
        if len({row['seed'] for row in group}) != len(group):
            raise SystemExit(f'Duplicate seed within {key}; resolve archive selection explicitly.')
        values = [row['eval_return_mean'] for row in group]
        summary.append(dict(zip(['category', 'env', 'method'], key),
            seeds=len(group), return_mean=statistics.mean(values),
            return_std_across_seeds=statistics.stdev(values) if len(values) > 1 else '',
            note='Single seed: across-seed std unavailable' if len(values) == 1 else ''))
    write_rows(root / 'results/overview.csv', summary)
    print(f'Indexed {len(rows)} evaluation records without mixing experiment categories.')


if __name__ == '__main__':
    main()
