"""The four-panel figure behind the drawdown investigation.

Reads the `input_sensitivity_over_time.py` output files for the arms named below plus each arm's
`runs/<policy>_evals.json`, and writes one PNG. Nothing here recomputes a measurement -- it is only
the drawing, so it is cheap to re-run after a longer ladder.

    cd snek2
    PYTHONPATH=. /opt/miniconda3/envs/snek/bin/python -u \
        hyperparamTuning/perDiagnostics/drawdown_chart.py <sens_dir> <out.png>

`<sens_dir>` holds `sens_<policy>.json` and `churn_b23b.json`. Built through the OO matplotlib API
rather than pyplot, matching `under_the_hood.display_progress` -- see the note in `CLAUDE.md` about
the pyplot figure-manager leak that OOM'd the desktop.
"""
import json
import os
import sys

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

B23 = ['b23a-beta01seed1', 'b23b-beta01seed2', 'b23c-beta01seed3', 'b23d-beta01seed4']
B18 = ['b18a-tgt1000seed1', 'b18b-tgt1000seed2', 'b18c-tgt1000seed3', 'b18d-tgt1000seed4']
DRAWDOWN = (217000, 242000)      # b23b's collapse, from its own eval series
COLOURS = ['#c44e52', '#4c72b0', '#dd8452', '#55a868']


def evals(policy):
    rows = json.load(open(os.path.join('runs', policy + '_evals.json')))['evals']
    step = np.array([r['step'] for r in rows])
    perfect = np.array([r['perfect_percent'] for r in rows], float)
    score = np.array([r['avg_score'] for r in rows], float)
    trailing = np.array([perfect[max(0, i - 29):i + 1].mean() for i in range(len(perfect))])
    return step, score, perfect, trailing


def sens(sens_dir, name):
    path = os.path.join(sens_dir, name + '.json')
    if not os.path.exists(path):
        return None
    return json.load(open(path))['rows']


def panel_a(ax, sens_dir):
    """b23b: the collapse against what the network was doing through it."""
    step, score, _, trailing = evals('b23b-beta01seed2')
    ax.axvspan(DRAWDOWN[0] / 1e3, DRAWDOWN[1] / 1e3, color='#f2d1d1', zorder=0,
               label='drawdown')
    ax.plot(step / 1e3, score, color='#4c72b0', lw=0.8, alpha=0.55, label='avg score')
    ax.plot(step / 1e3, trailing, color='#c44e52', lw=1.8, label='trailing-30 perfect %')
    ax.set_ylabel('score / perfect %')
    ax.set_xlabel('thousand steps')
    ax.set_ylim(0, 100)
    ax.set_xlim(100, 560)

    rows = sens(sens_dir, 'churn_b23b')
    if rows:
        twin = ax.twinx()
        s = np.array([r['step'] for r in rows]) / 1e3
        twin.plot(s, [r['d_chase'] for r in rows], color='#8172b3', lw=1.8,
                  label='dQ from "safe to chase food"')
        twin.plot(s, [r['max_q'] for r in rows], color='#937860', lw=1.4, ls='--',
                  label='mean max-Q')
        twin.set_ylabel('dQ  /  Q')
        twin.set_ylim(0, 15)
        handles = ax.get_legend_handles_labels()
        extra = twin.get_legend_handles_labels()
        ax.legend(handles[0] + extra[0], handles[1] + extra[1], fontsize=6.5,
                  loc='center left', bbox_to_anchor=(0.02, 0.32), framealpha=0.9)
    ax.set_title('A. b23b: the network reads the endgame the same way straight through\n'
                 'a collapse from score 94 to 4', fontsize=8.5, loc='left')


def panel_b(ax):
    """The control: every sibling seed makes the same level shift, drawdown or not."""
    for policy, colour in zip(B23, COLOURS):
        step, _, _, trailing = evals(policy)
        label = policy.split('-')[0]
        if policy == 'b23b-beta01seed2':
            label += ' (the drawdown)'
        ax.plot(step / 1e3, trailing, color=colour, lw=1.6, label=label)
    ax.axvspan(DRAWDOWN[0] / 1e3, DRAWDOWN[1] / 1e3, color='#f2d1d1', zorder=0)
    ax.set_xlim(100, 560)
    ax.set_ylim(0, 90)
    ax.set_xlabel('thousand steps')
    ax.set_ylabel('trailing-30 perfect %')
    ax.legend(fontsize=7, loc='upper left')
    ax.set_title('B. b23d rose further with no drawdown at all, so the rise is not\n'
                 'the drawdown\'s doing', fontsize=8.5, loc='left')


def panel_c(ax, sens_dir):
    """Chase-reading grows with training in every arm, and does not rank them by skill."""
    for group, style in ((B18, '-'), (B23, '--')):
        for policy, colour in zip(group, COLOURS):
            rows = sens(sens_dir, 'sens_' + policy)
            if not rows:
                continue
            ax.plot([r['step'] / 1e6 for r in rows], [r['ratio'] for r in rows],
                    style, color=colour, lw=1.5,
                    label='%s %s' % (policy.split('-')[0], 'seed' + policy[-1]))
    ax.set_xlabel('million steps')
    ax.set_ylabel('dQ(chase-safe) / dQ(is-safe)')
    ax.legend(fontsize=6.5, ncol=2, loc='upper left')
    ax.set_title('C. solid b18 (no IS), dashed b23 (IS beta->0.1). It tracks steps and how\n'
                 'much prioritisation survives -- not the perfect rate', fontsize=8.5, loc='left')


def panel_d(ax):
    """The seed, not the config, orders the arms inside a batch."""
    import glob
    import re
    by_seed = {1: [], 2: [], 3: [], 4: []}
    groups = {}
    for path in glob.glob('runs/*_evals.json'):
        policy = os.path.basename(path).replace('_evals.json', '')
        match = re.match(r'^(b\d+)[a-z]-(.*?)seed(\d)$', policy)
        if not match:
            continue
        # Keyed by (batch, config tag), not batch alone: batch 20 ran six waves whose arms all
        # match `b20[a-z]-`, so grouping by batch silently kept one arbitrary wave per seed.
        groups.setdefault((match.group(1), match.group(2)), {})[int(match.group(3))] = policy
    horizon = 550000
    kept = 0
    for key in sorted(groups, key=lambda k: (int(k[0][1:]), k[1])):
        seeds = groups[key]
        if len(seeds) < 4:
            continue
        values = {}
        for seed, policy in seeds.items():
            step, _, perfect, _ = evals(policy)
            if step[-1] < horizon * 0.98:
                values = {}
                break
            values[seed] = 100 * float((perfect[step <= horizon] >= 80).mean())
        if not values:
            continue
        kept += 1
        for seed, value in values.items():
            by_seed[seed].append(value)
    positions = [1, 2, 3, 4]
    ax.boxplot([by_seed[s] for s in positions], positions=positions, widths=0.6)
    for seed in positions:
        jitter = np.random.default_rng(seed).uniform(-0.12, 0.12, len(by_seed[seed]))
        ax.plot(seed + jitter, by_seed[seed], 'o', ms=3, color=COLOURS[seed - 1], alpha=0.75)
    ax.set_xlabel('seed')
    ax.set_ylabel('strong_eval_fraction % at 550k')
    ax.set_title('D. seed 2 or 4 is the best arm in all %d config waves measured, and the\n'
                 'gap is still there at 2M steps (exact p=0.00005)' % kept,
                 fontsize=8.5, loc='left')


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    sens_dir, out_path = sys.argv[1], sys.argv[2]
    figure = Figure(figsize=(13, 9), dpi=130)
    FigureCanvasAgg(figure)
    axes = figure.subplots(2, 2)
    panel_a(axes[0][0], sens_dir)
    panel_b(axes[0][1])
    panel_c(axes[1][0], sens_dir)
    panel_d(axes[1][1])
    figure.suptitle('Is a drawdown how a policy escapes a local minimum? '
                    'b23b and batch 18, measured', fontsize=11)
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    figure.savefig(out_path)
    print('wrote ' + out_path)


if __name__ == '__main__':
    main()
