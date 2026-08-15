"""Rank the C51 pilot's learning rates, pick one, and regenerate the pilot's doc sections.

    cd snek2
    PYTHONPATH=. python -u hyperparamTuning/pick_c51_lr.py            # report only
    PYTHONPATH=. python -u hyperparamTuning/pick_c51_lr.py --write-docs --batch b31 --lr 1e-4

Written so the pilot can hand off to a real batch **unattended**: `launch_c51_batch.sh` calls this,
takes the winner off stdout, launches, and calls it again to write the docs. Nothing here starts a
training run, kills a process, or touches `savedPolicies/`.

**The rule is pre-registered, and it is the docs' own rule.** Rates are compared on mean
`best_perfect30` over their seeds at a **common step horizon**, because `best_perfect30` is the
primary metric near the ceiling (between-seed sd 0.67 against `strong_eval_fraction`'s 5.59) and
because both metrics are cumulative over an arm's own evals, so an arm that trained further would
otherwise win on horizon alone. Ties fall through `strong_eval_fraction` and then `peak_trailing`.

Four guards, and three of them exist because a dry run against 13k-step data got the answer wrong:

- **`peak_trailing` is the third tie-break.** At an early horizon every arm reads `best_perfect30`
  0.0 *and* `sef` 0.0, so a two-level rule picked whichever rate came first out of a dict — it
  chose the slowest rate over the fastest. `peak_trailing` is never degenerate.
- **A short arm does not set the horizon.** The horizon is the lowest final step among arms that
  reached at least `SHORT_FRACTION` of the longest one; an arm that died at 13k of 600k is a
  *failure*, and letting it judge the other seven at 13k would have thrown away 98% of the data. It
  still appears in the ranking at its own numbers, flagged — dropping it would hide the failure.
- an arm with no eval series is **excluded and named**, rather than counted as zero — a launch that
  failed at startup would otherwise silently vote against its own rate;
- if fewer than two rates survive, it **refuses to pick** and exits non-zero. The caller then falls
  back to its own default rather than launching on one arm's evidence.

When the winner's `best_perfect30` and `sef` are both 0, the choice rests on `peak_trailing` alone.
That is a weak basis and the generated doc block says so rather than presenting it as a result.
"""
import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, '..', 'runs')
sys.path.insert(0, os.path.join(HERE, '..'))

import run_report  # noqa: E402  (after sys.path)

# `c51pilot-lr5e5seed2` / `c51pilotB-lr25e4seed1` -> ('5e5', '2') / ('25e4', '1')
ARM_RE = re.compile(r'^c51pilot[A-Za-z]*-lr([0-9]+e[0-9]+)seed([0-9]+)$')
TIE_PP = 1.0
# An arm below this share of the longest arm's final step is treated as having *failed*, not as
# defining the comparison horizon. Every healthy pilot arm stops at the same cap, so a large spread
# means a death, not a design.
SHORT_FRACTION = 0.8

# The compact form in an arm name back to a number: `5e5` -> 5e-05, `25e4` -> 2.5e-04. Every rate in
# this project is written as one or two significant digits times a negative power of ten, so the
# mantissa's digits after the first are decimals.
def rate_of(compact):
    mantissa, _, exponent = compact.partition('e')
    value = float(mantissa[0] + ('.' + mantissa[1:] if len(mantissa) > 1 else ''))
    return value * 10 ** -int(exponent)


def rate_label(compact):
    rate = rate_of(compact)
    return '{0:g}'.format(rate)


def load_arms():
    """Every pilot arm's eval rows, keyed by policy name. Missing series are reported, not counted."""
    arms, missing = {}, []
    for path in sorted(glob.glob(os.path.join(RUNS, 'c51pilot*_evals.json'))):
        policy = os.path.basename(path)[:-len('_evals.json')]
        if not ARM_RE.match(policy):
            continue
        try:
            rows = json.load(open(path))['evals']
        except Exception as exc:  # a truncated write, or a run that died before its first eval
            missing.append('{0} ({1})'.format(policy, type(exc).__name__))
            continue
        if not rows:
            missing.append(policy + ' (no evals)')
            continue
        arms[policy] = rows
    return arms, missing


def pick_horizon(arms):
    """The common step horizon, and the arms that were too short to set it.

    The lowest final step among arms that reached `SHORT_FRACTION` of the longest — so every rate is
    judged over the same span, without one dead arm collapsing that span to its own death step.
    """
    finals = {policy: rows[-1]['step'] for policy, rows in arms.items()}
    longest = max(finals.values())
    healthy = {p: s for p, s in finals.items() if s >= SHORT_FRACTION * longest}
    short = sorted(p for p in finals if p not in healthy)
    return min(healthy.values()) if healthy else min(finals.values()), short


def summarise(arms, horizon=None):
    """Per-arm numbers at a common horizon, plus the horizon used and any short (failed) arms.

    A short arm is still summarised — at its own final step, which is below the horizon — because
    excluding it would hide a rate that killed its own run.
    """
    short = []
    if horizon is None:
        horizon, short = pick_horizon(arms)
    out = []
    for policy, rows in sorted(arms.items()):
        kept = [row for row in rows if row['step'] <= horizon]
        summary = run_report.build_summary(kept)
        match = ARM_RE.match(policy)
        first = next((row['step'] for row in kept if row['perfect_percent'] > 0), None)
        out.append({
            'policy': policy,
            'compact': match.group(1),
            'rate': rate_of(match.group(1)),
            'seed': int(match.group(2)),
            'step': kept[-1]['step'],
            'final_step': rows[-1]['step'],
            'best_perfect30': summary['best_perfect30']['value'],
            'best_perfect30_step': summary['best_perfect30']['step'],
            'strong_eval_fraction': summary['strong_eval_fraction'],
            'peak_trailing': summary['peak_trailing']['value'],
            'first_perfect_step': first,
            'zero_since': summary.get('zero_since'),
            'short': policy in short,
        })
    return out, horizon, short


def rank(rows):
    """Rates ordered best first, on mean `best_perfect30` with `strong_eval_fraction` as tie-break."""
    by_rate = {}
    for row in rows:
        by_rate.setdefault(row['compact'], []).append(row)
    table = []
    for compact, group in by_rate.items():
        table.append({
            'compact': compact,
            'rate': rate_of(compact),
            'label': rate_label(compact),
            'seeds': len(group),
            'best_perfect30': sum(r['best_perfect30'] for r in group) / len(group),
            'strong_eval_fraction': sum(r['strong_eval_fraction'] for r in group) / len(group),
            'peak_trailing': sum(r['peak_trailing'] for r in group) / len(group),
        })
    table.sort(key=lambda r: (-r['best_perfect30'], -r['strong_eval_fraction'],
                              -r['peak_trailing']))
    return table


def choose(table):
    """The winning rate, and the reason in words — including when the reason is weak.

    The table is already sorted through all three keys, so `table[0]` is the winner; this only has to
    say *which* key decided it, because a caller running unattended has to record that.
    """
    if len(table) < 2:
        return None, 'fewer than two rates have usable data'
    best, second = table[0], table[1]
    if best['best_perfect30'] - second['best_perfect30'] >= TIE_PP:
        return best, 'best_perfect30 {0:.1f} against {1:.1f} for the next rate ({2})'.format(
            best['best_perfect30'], second['best_perfect30'], second['label'])
    if abs(best['strong_eval_fraction'] - second['strong_eval_fraction']) > 1e-9:
        return best, ('within {0} pp on best_perfect30 ({1:.1f} vs {2:.1f}), so decided on '
                      'strong_eval_fraction ({3:.1f} vs {4:.1f})'.format(
                          TIE_PP, best['best_perfect30'], second['best_perfect30'],
                          best['strong_eval_fraction'], second['strong_eval_fraction']))
    return best, ('**a weak choice**: best_perfect30 and strong_eval_fraction are tied at '
                  '{0:.1f} / {1:.1f} for every rate at this horizon, so it rests on mean peak '
                  'trailing alone ({2:.2f} against {3:.2f} for {4}). Treat the rate as unresolved '
                  'and read the arms, not this line'.format(
                      best['best_perfect30'], best['strong_eval_fraction'],
                      best['peak_trailing'], second['peak_trailing'], second['label']))


# --------------------------------------------------------------------------- markdown

def arm_table(rows):
    lines = ['| arm | lr | seed | step | best-30 | `sef` | peak trail | first perfect |',
             '|---|---|---|---|---|---|---|---|']
    for row in sorted(rows, key=lambda r: (-r['best_perfect30'], r['policy'])):
        lines.append('| `{0}` | {1} | {2} | {3}k | {4:.1f} | {5:.1f} | {6:.2f} | {7} |'.format(
            row['policy'], rate_label(row['compact']), row['seed'], row['step'] // 1000,
            row['best_perfect30'], row['strong_eval_fraction'], row['peak_trailing'],
            '{0}k'.format(row['first_perfect_step'] // 1000) if row['first_perfect_step'] else 'none'))
    return '\n'.join(lines)


def rate_table(table, winner):
    lines = ['| lr | seeds | mean best-30 | mean `sef` | mean peak trail |', '|---|---|---|---|---|']
    for row in table:
        mark = ' **← chosen**' if winner and row['compact'] == winner['compact'] else ''
        lines.append('| {0}{1} | {2} | {3:.1f} | {4:.1f} | {5:.2f} |'.format(
            row['label'], mark, row['seeds'], row['best_perfect30'],
            row['strong_eval_fraction'], row['peak_trailing']))
    return '\n'.join(lines)


def image_list(rows):
    """One `![]()` plus caption per arm, best first.

    Generated because `refresh_charts.sh` copies PNGs and never writes captions — an arm with an
    image and no entry is the exact drift that reached 12 undocumented arms across batches 5-7, and
    a wave launched unattended would land straight in it.
    """
    out = []
    for row in sorted(rows, key=lambda r: (-r['best_perfect30'], r['policy'])):
        out.append('![{0}](charts/{0}.png)'.format(row['policy']))
        out.append('**{0}** — lr {1}, best-30 {2:.1f}, first perfect {3}{4}'.format(
            row['policy'], rate_label(row['compact']), row['best_perfect30'],
            '{0}k'.format(row['first_perfect_step'] // 1000) if row['first_perfect_step'] else 'none',
            ', **stopped short**' if row['short'] else ''))
        out.append('')
    return '\n'.join(out)


def pilot_block(rows, table, winner, why, horizon, missing, short=(), batch=None,
                batch_lr=None, with_images=False):
    """The generated region for both charts.md and runs.md. Machine-written on purpose, and it says so."""
    stamp = time.strftime('%Y-%m-%d %H:%M')
    parts = ['*Generated by `pick_c51_lr.py` at {0}, when the last pilot arm stopped — the numbers '
             'below are read straight off the eval series, and the prose around this block is '
             'hand-written.*'.format(stamp), '',
             '**Compared at a common horizon of {0}k steps**, the lowest final step any arm reached, '
             'because both metrics accumulate over an arm\'s own evals and a longer arm would '
             'otherwise win on horizon alone.'.format(horizon // 1000), '',
             rate_table(table, winner), '', arm_table(rows), '']
    if short:
        parts += ['**Stopped short of the cap, so they do not set the horizon:** '
                  + ', '.join('`%s`' % p for p in short)
                  + ' — an arm that stopped well before the others failed, and letting it define the '
                    'comparison span would discard the rest of the data. It is still ranked, at its '
                    'own final step.', '']
    if missing:
        parts += ['**Excluded, no usable eval series:** ' + ', '.join('`%s`' % m for m in missing)
                  + ' — excluded rather than counted as zero, so a failed launch cannot vote against '
                    'its own rate.', '']
    if winner:
        parts += ['**Chosen: `{0}`** — {1}.'.format(winner['label'], why), '']
    else:
        parts += ['**No rate chosen** — {0}. The batch fell back to its caller\'s default.'.format(why),
                  '']
    if batch:
        parts += ['**Batch `{0}` launched at {1}** on this rate, 4 seeds, 2M cap, `fc 200,100,100`, '
                  'otherwise b25\'s config — so `b25a-d` is the seed-matched control.'.format(
                      batch, stamp), '']
    if with_images:
        parts += [image_list(rows)]
    return '\n'.join(parts)


MARKER = 'C51-PILOT-STATUS'


def replace_region(path, block):
    """Swap the text between the markers. Absent markers are a loud failure, not a silent no-op."""
    begin, end = '<!-- {0}:BEGIN -->'.format(MARKER), '<!-- {0}:END -->'.format(MARKER)
    text = open(path).read()
    if text.count(begin) != 1 or text.count(end) != 1:
        raise SystemExit('{0} needs exactly one {1} / {2} pair'.format(path, begin, end))
    head, _, rest = text.partition(begin)
    _, _, tail = rest.partition(end)
    open(path, 'w').write(head + begin + '\n' + block + end + tail)
    return path


def refresh_charts():
    subprocess.run(['zsh', os.path.join(HERE, 'refresh_charts.sh')], cwd=HERE,
                   stdout=subprocess.DEVNULL, check=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--write-docs', action='store_true')
    parser.add_argument('--batch', default=None, help='batch prefix to name as launched')
    parser.add_argument('--lr', default=None, help='the rate actually launched, if not the winner')
    parser.add_argument('--horizon', type=int, default=None)
    args = parser.parse_args()

    arms, missing = load_arms()
    if not arms:
        raise SystemExit('no pilot eval series under {0}'.format(RUNS))
    rows, horizon, short = summarise(arms, args.horizon)
    table = rank(rows)
    winner, why = choose(table)

    print(rate_table(table, winner))
    print()
    print(arm_table(rows))
    if missing:
        print('\nexcluded (no usable series):', ', '.join(missing))
    print('\nhorizon: {0} steps'.format(horizon))
    if winner:
        print('chosen: {0}  ({1})'.format(winner['label'], why))
    else:
        print('no choice: ' + why)

    if args.write_docs:
        refresh_charts()
        for name in ('charts.md', 'runs.md'):
            # charts.md is the one that carries the graphs, so only it gets the image list.
            block = pilot_block(rows, table, winner, why, horizon, missing, short,
                                batch=args.batch, batch_lr=args.lr,
                                with_images=(name == 'charts.md'))
            print('wrote', replace_region(os.path.join(HERE, name), block))
        json.dump({'horizon': horizon, 'rates': table, 'arms': rows, 'missing': missing,
                   'short': short,
                   'chosen': winner['label'] if winner else None, 'why': why,
                   'batch': args.batch, 'written_at': time.time()},
                  open(os.path.join(RUNS, 'c51pilot_lr_choice.json'), 'w'), indent=2)

    # stdout's last line is what the launcher reads, so keep it machine-parseable.
    print('CHOSEN_LR=' + (winner['label'] if winner else ''))
    print('CHOSEN_COMPACT=' + (winner['compact'] if winner else ''))
    return 0 if winner else 1


if __name__ == '__main__':
    sys.exit(main())
