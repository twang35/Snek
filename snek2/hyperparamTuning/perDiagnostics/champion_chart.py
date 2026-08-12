"""The figure behind "what do the record checkpoints do differently".

Panels A and B come from `behaviour_profile.py` output; C and D from the project's own close-out
measurements in `runs/*_checkpoint_evals*.json`, so C and D need no new compute.

    cd snek2
    PYTHONPATH=. /opt/miniconda3/envs/snek/bin/python -u \
        hyperparamTuning/perDiagnostics/champion_chart.py <bp_dir> <measured.json> <out.png>

Built through the OO matplotlib API, not pyplot -- see the leak note in the root `CLAUDE.md`.
"""
import glob
import json
import os
import re
import sys

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

ELITE = ['elite-b17b1190', 'elite-b18b1588', 'elite-b15b3245', 'elite-b11b0855',
         'elite-b14a3702', 'strong-b13d0986', 'strong-b11a0671']
OTHER = ['b20d-peak', 'b20b-peak', 'b20c-peak', 'b20a-peak', 'b23b-now']
BANDS = ('10-49', '50-84', '85-94', '95-99')
STEP_METRICS = ('hug', 'tail_reach', 'chase_safe', 'regions', 'forward')


def load(bp_dir):
    merged = {}
    for path in sorted(glob.glob(os.path.join(bp_dir, '*.json'))):
        tag = re.sub(r'_\d+\.json$', '', os.path.basename(path))
        merged.setdefault(tag, []).append(json.load(open(path)))
    return merged


def weighted(shards, band, metric):
    key = 'n_steps' if metric in STEP_METRICS else 'n_meals'
    values, weights = [], []
    for shard in shards:
        row = shard['bands'].get(band, {})
        if row.get(metric) is None:
            continue
        values.append(row[metric])
        weights.append(row.get(key) or 0)
    if not values or sum(weights) == 0:
        return None
    return float(np.average(values, weights=weights))


def perfect_of(shards):
    counts = {}
    for shard in shards:
        for key, value in shard['outcomes'].items():
            counts[key] = counts.get(key, 0) + value
    return 100.0 * counts.get('perfect', 0) / sum(counts.values())


def panel_a(ax, merged):
    for tags, colour, label in ((ELITE, '#4c72b0', 'hall of fame'),
                                (OTHER, '#c44e52', 'batch 20 peaks / b23b')):
        xs = [weighted(merged[t], '95-99', 'steps_per_food_p90') for t in tags if t in merged]
        ys = [perfect_of(merged[t]) for t in tags if t in merged]
        ax.plot(xs, ys, 'o', ms=8, color=colour, label=label, alpha=0.85)
    for tag in ELITE + OTHER:
        if tag not in merged:
            continue
        ax.annotate(tag.replace('elite-', '').replace('strong-', '').replace('-peak', ''),
                    (weighted(merged[tag], '95-99', 'steps_per_food_p90'),
                     perfect_of(merged[tag])), fontsize=6,
                    xytext=(4, -3), textcoords='offset points')
    ax.set_xscale('log')
    ax.set_xlabel('p90 steps between meals at length 95-99  (log scale)')
    ax.set_ylabel('perfect % over the same 100 games')
    ax.set_title('A. The endgame difference is hunting speed: the mediocre policies\n'
                 'wander for 85-226 moves between meals, the records for 5-13',
                 fontsize=8.5, loc='left')
    ax.legend(fontsize=7, loc='lower left')
    ax.grid(alpha=0.25)


def panel_b(ax, merged):
    width = 0.35
    positions = np.arange(len(BANDS))
    for offset, tags, colour, label in ((-width / 2, ELITE, '#4c72b0', 'hall of fame'),
                                        (width / 2, OTHER, '#c44e52', 'batch 20 peaks / b23b')):
        means, errs = [], []
        for band in BANDS:
            values = [weighted(merged[t], band, 'headroom_p10') for t in tags if t in merged]
            values = [v for v in values if v is not None]
            means.append(np.mean(values) if values else 0)
            errs.append(np.std(values) if values else 0)
        ax.bar(positions + offset, means, width, yerr=errs, capsize=3, color=colour, label=label)
    ax.axhline(500, color='#555555', ls=':', lw=1)
    ax.annotate('the 500-step starve budget', (0.02, 505), fontsize=6.5, color='#555555')
    ax.set_xticks(positions)
    ax.set_xticklabels(BANDS)
    ax.set_xlabel('snake length')
    ax.set_ylabel('starve budget left at the 10th percentile meal')
    ax.set_ylim(0, 560)
    ax.set_title('B. and it costs them the starve clock: by length 95-99 the worst\n'
                 'tenth of their hunts leaves 270-410 steps against the records\' 486-495',
                 fontsize=8.5, loc='left')
    ax.legend(fontsize=7, loc='lower left')


def panel_c(ax, measured_path):
    rows = json.load(open(measured_path))
    best = {}
    for row in rows:
        key = (row['policy'], row['step'])
        if key not in best or row['episodes'] > best[key]['episodes']:
            best[key] = row
    by_policy = {}
    for row in best.values():
        by_policy.setdefault(row['policy'], []).append(row)
    points = []
    for policy, got in by_policy.items():
        if len(got) < 8:
            continue
        values = np.array([r['perfect'] for r in got])
        points.append((float(np.median(values)), values.max(), policy, len(got)))
    points.sort()
    ax.plot([p[0] for p in points], [p[1] for p in points], 'o', ms=6, color='#8172b3', alpha=0.8)
    for median, best_value, policy, count in points:
        if best_value >= 96 or median < 30:
            ax.annotate(policy.split('-')[0], (median, best_value), fontsize=6,
                        xytext=(4, -3), textcoords='offset points')
    limits = [0, 100]
    ax.plot(limits, limits, ':', color='#999999', lw=1)
    ax.set_xlabel("arm's MEDIAN measured checkpoint, perfect %")
    ax.set_ylabel("arm's BEST measured checkpoint, perfect %")
    ax.set_title('C. No mediocre arm ever produced a great checkpoint. Best is set by\n'
                 'median (r=+0.97): b10b measured 624 of them and never cleared 90',
                 fontsize=8.5, loc='left')
    ax.grid(alpha=0.25)


def panel_d(ax):
    rng = np.random.default_rng(0)
    ks = np.array([1, 2, 3, 5, 8, 12, 20, 35, 60, 100])
    for rate, colour in ((0.90, '#4c72b0'), (0.80, '#dd8452')):
        highs = []
        for k in ks:
            draws = rng.binomial(100, rate, size=(20000, int(k))) / 100.0 * 100
            highs.append(draws.max(axis=1).mean() - rate * 100)
        ax.plot(ks, highs, '-o', ms=4, color=colour,
                label='every checkpoint truly at %d%%' % (rate * 100))
    ax.axhspan(5.0, 6.5, color='#c44e52', alpha=0.15)
    ax.annotate('the project\'s own documented shrinkage on\n'
                're-measurement: 5.05-5.2 pp, four batches',
                (1.3, 6.7), fontsize=6.5, color='#8b2b30')
    ax.set_xscale('log')
    ax.set_xlabel('checkpoints screened, k  (log scale)')
    ax.set_ylabel('pp by which the best of k reads high')
    ax.set_title('D. Selection on 100-episode noise is enough to explain the whole\n'
                 'shrinkage pattern: the max of ~20-50 reads inflates by 5-6 pp',
                 fontsize=8.5, loc='left')
    ax.legend(fontsize=7, loc='lower right')
    ax.grid(alpha=0.25)


def main():
    if len(sys.argv) < 4:
        sys.exit(__doc__)
    bp_dir, measured_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
    merged = load(bp_dir)
    figure = Figure(figsize=(13, 9), dpi=130)
    FigureCanvasAgg(figure)
    axes = figure.subplots(2, 2)
    panel_a(axes[0][0], merged)
    panel_b(axes[0][1], merged)
    panel_c(axes[1][0], measured_path)
    panel_d(axes[1][1])
    figure.suptitle('What the record checkpoints do differently, and why their headline '
                    'numbers are too high', fontsize=11)
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    figure.savefig(out_path)
    print('wrote ' + out_path)


if __name__ == '__main__':
    main()
