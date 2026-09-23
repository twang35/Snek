"""A reads pass as one table: every alternative read of a cell against its `mean:fixed` control.

    PYTHONPATH=. python -m tools.read_compare b40e-mqrdqnlocal-seed5 b40f-... --prefix reads- [--control meanfixed]

Reads `runs/<arm>_checkpoint_evals_<prefix><read>.json` for every arm named and every label with the
prefix, pairs each read's rows with the control's by (arm, step), and prints per read: how many
checkpoints it has in common with the control, the pooled perfect rate, the mean paired difference in
perfect rate with a bootstrap 95% interval over checkpoints, and the death / starve / perfect split of
its episodes. `plans/quantile-reads.md` §3 is the design this reports.

Death and starve come from the row's own `deaths` / `starves` counts (the engine records the env's
flags since 2026-09-22); rows from before that report the perfect rate only.
"""

import argparse
import glob
import os
import random
import re
import statistics

from env import constants
from tools import results


def labels_with_prefix(arms, prefix):
    """Every pass label with the prefix that at least one arm has a merged file for, sorted."""
    found = set()
    for arm in arms:
        stem = '{0}_checkpoint_evals_{1}'.format(results.run_name(arm), prefix)
        exact = re.compile(re.escape(stem) + r'([^-][^/]*)\.json$')
        for path in glob.glob(os.path.join(constants.RUNS_DIR, stem + '*.json')):
            match = exact.search(os.path.basename(path))
            if match and not re.search(r'-s\d+of\d+$', match.group(1)):
                found.add(match.group(1))
    return sorted(found)


def rows_by_step(arm, label):
    payload = results.read(results.stage_b_path(arm, label))
    if payload is None:
        return {}
    return {int(row['step']): row for row in results.rows_of(payload)}


def outcome_counts(row):
    """`(perfect, deaths, starves, episodes)` for one row, or None if the row predates the counts."""
    if row.get('deaths') is None or row.get('starves') is None:
        return None
    return int(row['perfect_games']), int(row['deaths']), int(row['starves']), int(row['episodes'])


def paired(control, variant):
    """`[(step, control_rate, variant_rate)]` over the steps both have."""
    return [(step, float(control[step]['perfect_percent']), float(variant[step]['perfect_percent']))
            for step in sorted(set(control) & set(variant))]


def bootstrap_interval(differences, draws=2000, seed=0):
    if len(differences) < 2:
        return None
    rng = random.Random(seed)
    means = []
    for _ in range(draws):
        sample = [rng.choice(differences) for _ in differences]
        means.append(statistics.fmean(sample))
    means.sort()
    return means[int(0.025 * draws)], means[int(0.975 * draws) - 1]


def compare(arms, prefix, control_label):
    control = {}
    for arm in arms:
        for step, row in rows_by_step(arm, prefix + control_label).items():
            control[(arm, step)] = row
    table = []
    for read in labels_with_prefix(arms, prefix):
        rows = {}
        for arm in arms:
            for step, row in rows_by_step(arm, prefix + read).items():
                rows[(arm, step)] = row
        pairs = [(key, float(control[key]['perfect_percent']), float(rows[key]['perfect_percent']))
                 for key in sorted(set(control) & set(rows))]
        differences = [v - c for _, c, v in pairs]
        counts = [outcome_counts(rows[key]) for key in rows]
        counts = [c for c in counts if c is not None]
        totals = [sum(c[i] for c in counts) for i in range(4)] if counts else None
        table.append({
            'read': read,
            'checkpoints': len(rows),
            'paired': len(pairs),
            'perfect': statistics.fmean(float(r['perfect_percent']) for r in rows.values()) if rows else None,
            'delta': statistics.fmean(differences) if differences else None,
            'interval': bootstrap_interval(differences),
            'wins': len([d for d in differences if d > 0]),
            'losses': len([d for d in differences if d < 0]),
            'death_pct': 100.0 * totals[1] / totals[3] if totals and totals[3] else None,
            'starve_pct': 100.0 * totals[2] / totals[3] if totals and totals[3] else None,
        })
    return table


def render(table, control_label):
    def num(value, digits=2):
        return '-' if value is None else '{0:.{1}f}'.format(value, digits)
    lines = ['| read | ckpts | perfect % | Δ vs {0} (pp) | 95% CI | better / worse | death % | starve % |'.format(control_label),
             '|---|---:|---:|---:|---|---:|---:|---:|']
    for entry in sorted(table, key=lambda e: (e['delta'] is None, -(e['delta'] or 0.0))):
        interval = '-' if entry['interval'] is None else '[{0}, {1}]'.format(num(entry['interval'][0]), num(entry['interval'][1]))
        lines.append('| `{0}` | {1} | {2} | {3} | {4} | {5} / {6} | {7} | {8} |'.format(
            entry['read'], entry['checkpoints'], num(entry['perfect']),
            '-' if entry['read'] == control_label else num(entry['delta'], 2) if entry['delta'] is not None else '-',
            '-' if entry['read'] == control_label else interval,
            entry['wins'], entry['losses'], num(entry['death_pct']), num(entry['starve_pct'])))
    return '\n'.join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('arms', nargs='+', help='the cell\'s arms')
    parser.add_argument('--prefix', default='reads-', help='the pass labels\' shared prefix')
    parser.add_argument('--control', default='meanfixed', help='the control read\'s label after the prefix')
    args = parser.parse_args(argv)
    table = compare(args.arms, args.prefix, args.control)
    if not table:
        parser.error('no pass files with prefix {0!r} for {1}'.format(args.prefix, ', '.join(args.arms)))
    print(render(table, args.control))


if __name__ == '__main__':
    main()
