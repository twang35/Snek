"""strong_eval_fraction across the 10 -> 20 episode graph-eval boundary.

`strong_eval_fraction` is the share of an arm's graph evals at >= 80% perfect, and it is this
project's primary metric. **It is not comparable across the 2026-08-19 boundary**, where
`training.num_eval_episodes` went 10 -> 20 (batches 1-44 vs 45-onward), and the bias is large and
one-directional: a *threshold-crossing* fraction rewards noise, so fewer episodes per eval means
more crossings for a policy of identical quality.

The size of it, for a policy whose true perfect rate is p:

    p       P(>=80% of 10)   P(>=80% of 20)   ratio
    0.50        0.0547           0.0059       9.3x
    0.55        0.0996           0.0189       5.3x
    0.65        0.2616           0.1182       2.2x
    0.80        0.6778           0.6296       1.08x
    0.90        0.9298           0.9568       0.97x

So the bias is huge where arms spend their middle band and vanishes once they are genuinely strong,
which is why no single point estimate resolves it -- an arm's `sef` is a mixture over its own
trajectory. **That is what this script is for.** It puts a 20-episode arm on the 10-episode footing
exactly, with no simulation: an eval that scored k perfect out of 20 would, if only 10 of those
episodes had been run, have scored X ~ Hypergeometric(N=20, K=k, n=10) perfect, and counts as strong
iff X >= 8. Summing that probability over the arm's evals is its expected `sef` at 10 episodes.

Measured on b46a vs b38 at step 1428000, this recovered more than half of an apparent deficit:
-11.3 pp raw became -4.8 pp on the common footing, with the sign test unchanged at 0 of 4. So the
correction changed the *size* of the finding and not its direction -- which is the outcome to hope
for, and the reason to run it rather than argue about it.

**Direction of the correction is always the same:** a 20-episode arm's raw `sef` is an
*underestimate* of what it would have posted at 10 episodes, so an older arm always looks better than
it is. Never compare raw `sef` across the boundary; either run this, or use banded mean perfect rate,
which is unbiased because it estimates the same p either way.

Usage:
    PYTHONPATH=. python hyperparamTuning/perDiagnostics/sef_common_footing.py \
        --new runs/b46a-c51batch512seed1_evals.json \
        --old runs/b38a-c51fc320eps3125seed1_evals.json \
        --cap 1428000

    # or a whole batch against its control, paired by seed
    PYTHONPATH=. python hyperparamTuning/perDiagnostics/sef_common_footing.py \
        --new-glob 'runs/b46a-c51batch512seed[0-9]_evals.json' \
        --old-glob 'runs/b38?-c51fc320eps3125seed[0-9]_evals.json' --cap 1428000
"""
import argparse
import glob
import json
import os
import re
from math import comb

STRONG_FRACTION = 0.80   # `run_report`'s definition of a "strong" eval
OLD_EPISODES = 10        # batches 1-44
NEW_EPISODES = 20        # batches 45-onward


def rows_upto(path, cap):
    rows = json.load(open(path))['evals']
    return [r for r in rows if cap is None or r['step'] <= cap]


def raw_sef(rows):
    """The reported figure: share of evals at or above the strong threshold."""
    if not rows:
        return None
    hits = sum(1 for r in rows if r['perfect_percent'] >= 100 * STRONG_FRACTION)
    return 100.0 * hits / len(rows)


def p_strong_subsample(k, n_have, n_want):
    """P(a random `n_want`-episode subsample of an eval that scored k/`n_have` is >= the threshold).

    Hypergeometric, exact. `k` perfect episodes among `n_have`; the subsample needs at least
    ceil(n_want * STRONG_FRACTION) of them.
    """
    need = int(-(-(n_want * STRONG_FRACTION) // 1))   # ceil
    total = comb(n_have, n_want)
    good = 0
    for x in range(need, n_want + 1):
        if x > k or n_want - x > n_have - k:
            continue
        good += comb(k, x) * comb(n_have - k, n_want - x)
    return good / total


def sef_on_old_footing(rows, n_have=NEW_EPISODES, n_want=OLD_EPISODES):
    """Expected `sef` had each eval been measured at `n_want` episodes instead of `n_have`."""
    if not rows:
        return None
    total = 0.0
    for r in rows:
        k = int(round(r['perfect_percent'] * n_have / 100.0))
        total += p_strong_subsample(k, n_have, n_want)
    return 100.0 * total / len(rows)


def mean_perfect(rows):
    """Banded mean perfect rate -- unbiased across the boundary, printed as the control column."""
    if not rows:
        return None
    return sum(r['perfect_percent'] for r in rows) / len(rows)


def seed_of(path):
    m = re.search(r'seed(\d+)', os.path.basename(path))
    return int(m.group(1)) if m else None


def report_pair(new_path, old_path, cap):
    new, old = rows_upto(new_path, cap), rows_upto(old_path, cap)
    return dict(seed=seed_of(new_path), n_new=len(new), n_old=len(old),
                new_raw=raw_sef(new), new_adj=sef_on_old_footing(new), old_raw=raw_sef(old),
                new_pp=mean_perfect(new), old_pp=mean_perfect(old))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--new', action='append', default=[], help='a 20-episode arm\'s _evals.json')
    ap.add_argument('--old', action='append', default=[], help='its 10-episode control')
    ap.add_argument('--new-glob', default=None)
    ap.add_argument('--old-glob', default=None)
    ap.add_argument('--cap', type=int, default=None,
                    help='compare only evals at or below this step -- REQUIRED for a running arm, '
                         'since sef is a fraction of each arm\'s own evals')
    args = ap.parse_args()

    new = sorted(args.new) + sorted(glob.glob(args.new_glob) if args.new_glob else [])
    old = sorted(args.old) + sorted(glob.glob(args.old_glob) if args.old_glob else [])
    if not new or not old:
        raise SystemExit('need at least one --new and one --old (or their --*-glob forms)')
    by_seed = {seed_of(p): p for p in old}
    if len(by_seed) != len(old):
        raise SystemExit('two control files claim the same seed: {0}'.format(old))

    print('strong_eval_fraction on a common {0}-episode footing'.format(OLD_EPISODES))
    print('cap: {0}\n'.format(args.cap if args.cap is not None else 'none (whole run)'))
    head = '{0:>5s} {1:>7s} {2:>10s} {3:>12s} {4:>10s} {5:>9s} {6:>9s} {7:>9s}'
    print(head.format('seed', 'evals', 'new raw', 'new as-if/10', 'old raw', 'delta', 'new mean', 'old mean'))
    deltas = []
    for p in new:
        seed = seed_of(p)
        if seed not in by_seed:
            print('  no control for seed {0} ({1}) -- skipped'.format(seed, os.path.basename(p)))
            continue
        r = report_pair(p, by_seed[seed], args.cap)
        if r['n_new'] != r['n_old']:
            print('  seed {0}: {1} vs {2} evals in the window -- NOT a matched horizon'.format(
                seed, r['n_new'], r['n_old']))
        d = r['new_adj'] - r['old_raw']
        deltas.append(d)
        print(head.format(str(seed), str(r['n_new']), '{0:.1f}%'.format(r['new_raw']),
                          '{0:.1f}%'.format(r['new_adj']), '{0:.1f}%'.format(r['old_raw']),
                          '{0:+.1f}'.format(d), '{0:.1f}'.format(r['new_pp']),
                          '{0:.1f}'.format(r['old_pp'])))
    if deltas:
        print('\nmean delta on the common footing: {0:+.1f} pp   sign test {1} of {2} for the new arm'
              .format(sum(deltas) / len(deltas), sum(1 for x in deltas if x > 0), len(deltas)))
        print('(a raw comparison would have read {0:+.1f} pp -- the difference is the artefact)'
              .format(sum(report_pair(p, by_seed[seed_of(p)], args.cap)['new_raw'] -
                          report_pair(p, by_seed[seed_of(p)], args.cap)['old_raw']
                          for p in new if seed_of(p) in by_seed) / len(deltas)))


if __name__ == '__main__':
    main()
