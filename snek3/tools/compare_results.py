"""Compares two stage-B result files row for row, and says whether the difference is noise.

    PYTHONPATH=. python -m tools.compare_results <file-a> <file-b> [--label-a A] [--label-b B]

**The point is the expected spread, not the observed one.** Two independent 100-episode measurements
of the *same* policy differ by a standard deviation of `sqrt(2 p (1-p) / n)` — 2.2 pp at p=0.975,
n=100 — so a per-row disagreement of 3 or 4 points is what agreement *looks like*, and a report that
only printed the differences would read as a failure. What distinguishes agreement from a real
difference is the **mean** across rows, whose standard error shrinks as `sd / sqrt(rows)`.

This is the tool that closes phase 2's gate: snek3's wave against snek2's own close-out file over the
3,222 checkpoints of `b45a-lowlr8-b29b`, two independent stacks measuring the same weights.
"""

import argparse
import json
import math
import os
import sys

import numpy as np


def rows_of(path):
    """`{step: row}` from a snek3 or a snek2 result file.

    snek2 keys its rows `results` and snek3 keys them `rows`. Both are read, because the whole
    purpose of this tool is comparing one against the other.
    """
    if not os.path.exists(path):
        raise SystemExit('no such file: {0}'.format(path))
    with open(path) as handle:
        payload = json.load(handle)
    rows = payload.get('rows')
    if rows is None:
        rows = payload.get('results')
    if rows is None:
        raise SystemExit('{0} has neither `rows` nor `results`'.format(path))
    return {int(row['step']): row for row in rows}


def expected_sd(rate, episodes_a, episodes_b):
    """Standard deviation of the *difference* of two independent binomial rate estimates, in pp.

    The null hypothesis this whole comparison is against: the two files measured the same policy and
    differ only by their episode samples.
    """
    variance = rate * (1.0 - rate) * (1.0 / episodes_a + 1.0 / episodes_b)
    return 100.0 * math.sqrt(max(variance, 0.0))


def compare(path_a, path_b, label_a=None, label_b=None, show=12):
    a, b = rows_of(path_a), rows_of(path_b)
    label_a = label_a or os.path.basename(path_a)
    label_b = label_b or os.path.basename(path_b)

    shared = sorted(set(a) & set(b))
    only_a, only_b = sorted(set(a) - set(b)), sorted(set(b) - set(a))
    if not shared:
        raise SystemExit('the two files share no steps')

    percent_a = np.array([a[step]['perfect_percent'] for step in shared], dtype=float)
    percent_b = np.array([b[step]['perfect_percent'] for step in shared], dtype=float)
    episodes_a = np.array([a[step]['episodes'] for step in shared], dtype=float)
    episodes_b = np.array([b[step]['episodes'] for step in shared], dtype=float)
    difference = percent_a - percent_b

    pooled_rate = float((percent_a.sum() + percent_b.sum()) / (100.0 * 2 * len(shared)))
    predicted = expected_sd(pooled_rate, float(episodes_a.mean()), float(episodes_b.mean()))
    observed = float(difference.std(ddof=1)) if len(shared) > 1 else 0.0
    mean = float(difference.mean())
    standard_error = observed / math.sqrt(len(shared)) if len(shared) > 1 else 0.0

    print('{0}\n  vs {1}\n'.format(label_a, label_b))
    print('{0:<34}{1}'.format('rows compared', len(shared)))
    if only_a or only_b:
        print('{0:<34}{1} only in A, {2} only in B'.format('rows not shared', len(only_a),
                                                           len(only_b)))
    print('{0:<34}{1:.0f} vs {2:.0f}'.format('episodes per row', episodes_a.mean(),
                                             episodes_b.mean()))
    print('{0:<34}{1:.2f}% vs {2:.2f}%'.format('mean perfect', percent_a.mean(), percent_b.mean()))
    print()
    print('{0:<34}{1:+.3f} pp'.format('mean difference (A - B)', mean))
    print('{0:<34}{1:.3f} pp'.format('standard error of that mean', standard_error))
    print('{0:<34}{1:+.2f}'.format('so, in standard errors', mean / standard_error
                                   if standard_error else 0.0))
    print()
    print('{0:<34}{1:.2f} pp'.format('per-row sd, observed', observed))
    print('{0:<34}{1:.2f} pp'.format('per-row sd, predicted by sampling', predicted))
    print('{0:<34}{1:.2f}'.format('observed / predicted', observed / predicted
                                  if predicted else float('nan')))
    print()
    print('{0:<34}{1:.2f} / {2:.2f} pp'.format('max |difference|, A high / B high',
                                               difference.max(), -difference.min()))

    if show and len(shared) > show:
        worst = np.argsort(-np.abs(difference))[:show]
        print('\nthe {0} largest disagreements:'.format(show))
        print('  {0:>10}  {1:>7}  {2:>7}  {3:>7}'.format('step', 'A', 'B', 'A - B'))
        for index in sorted(worst, key=lambda i: shared[i]):
            print('  {0:>10}  {1:>7.1f}  {2:>7.1f}  {3:>+7.1f}'.format(
                shared[index], percent_a[index], percent_b[index], difference[index]))

    return {'rows': len(shared), 'mean_difference': mean, 'standard_error': standard_error,
            'observed_sd': observed, 'predicted_sd': predicted,
            'mean_a': float(percent_a.mean()), 'mean_b': float(percent_b.mean())}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('file_a')
    parser.add_argument('file_b')
    parser.add_argument('--label-a', default=None)
    parser.add_argument('--label-b', default=None)
    parser.add_argument('--show', type=int, default=12,
                        help='how many of the largest disagreements to list')
    args = parser.parse_args(argv)
    compare(args.file_a, args.file_b, args.label_a, args.label_b, args.show)
    return 0


if __name__ == '__main__':
    sys.exit(main())
