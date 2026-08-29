"""A stage-B result file as a picture and a text block: where an arm's record region is.

This is the surviving half of snek2's `eval_progress.py`, and it is a fifth of the size. That script
had three panels and 1,387 lines, and two of the three answered questions snek3 does not have:

- **In-flight convergence, one line per process.** snek2 measured a checkpoint in ~10 rounds of
  pooled episodes, so a checkpoint had a *partial* rate worth watching. A snek3 shard measures a
  checkpoint straight through and writes one row, so there is nothing to converge.
- **A per-stage progress breakdown with the screen's cut rate.** One stage, no screen, no cut.
- **How much is left.** `tools/eval_wave.py` already prints it, on one line, read off the shard files —
  `[ 2167/3222] 4 shard(s) alive, 9m elapsed  eta 5m  [547, 539, 544, 537]`.

What is left is the panel that mattered: **every measured checkpoint as a point against step**, so
the *shape* of the good region is visible rather than a count of rows above a line. That shape is
what [`../docs/protocol.md`](../docs/protocol.md) asks a stage-B comparison to lead with, because a
best row is a selected high and a wide plateau is not.

    PYTHONPATH=. python -m tools.stage_b_chart b45a-import --label ab3222
    PYTHONPATH=. python -m tools.stage_b_chart b45a-import --label ab3222 --watch 30

`--watch` re-reads and redraws, which works on a wave still in flight because shards rewrite their
own files after every row. With no merged file yet, the shard files are pooled instead.
"""

import argparse
import os
import sys
import time

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.ticker import FuncFormatter
import imageio

from env import constants
from tools import progress_chart, results

# One deliberately wider figure than `progress_chart`'s: a stage-B pass has one series and up to a
# few thousand points along the step axis, where an arm's chart has two axes and a trend line.
FIGSIZE = (5.0, 2.4)
REGION_LEVEL = 98.0
POINT_COLOR = 'tab:red'
REGION_COLOR = 'tab:green'
TREND_COLOR = 'darkred'
# Wide enough that the trend is not itself noise: a 100-episode row at p=0.973 has an sd of 1.6 pp,
# so a 40-row mean has 0.26 pp — narrower than the quantisation of a single row.
TREND_WINDOW = 40
LABEL_SIZE = 7
TICK_SIZE = 6
TOP_N = 5


def load(policy, label=None):
    """Every row of a pass, ascending by step — from the merged file, or the shards if it is absent.

    Pooling the shards is what makes this usable on a wave in flight. A duplicate step keeps the
    longer sample, matching `results.merge`, so switching from shards to merged never changes a row.
    """
    merged = results.read(results.stage_b_path(policy, label))
    payloads = [merged] if merged else [results.read(path)
                                        for path in results.shard_paths(policy, label)]
    best = {}
    for payload in payloads:
        for row in results.rows_of(payload):
            step = int(row['step'])
            if step not in best or row['episodes'] > best[step]['episodes']:
                best[step] = row
    return [best[step] for step in sorted(best)]


def widest_region(rows, level=REGION_LEVEL):
    """The longest run of *adjacent measured* checkpoints at or above `level`, as `(count, lo, hi)`.

    Adjacency is in the row list, not in step number: a pass measures the checkpoints its selector
    chose, and asking for consecutive step numbers would report width 1 for every arm whose screen
    skipped one checkpoint in the middle of an otherwise solid plateau.
    """
    best = (0, None, None)
    run_start = None
    for index, row in enumerate(rows + [None]):
        if row is not None and row['perfect_percent'] >= level:
            if run_start is None:
                run_start = index
            continue
        if run_start is not None:
            span = (index - run_start, int(rows[run_start]['step']), int(rows[index - 1]['step']))
            best = max(best, span)
            run_start = None
    return best


def summarise(rows, level=REGION_LEVEL):
    """The numbers the text block prints, as a dict. Empty rows give an empty dict."""
    if not rows:
        return {}
    percents = [row['perfect_percent'] for row in rows]
    episodes = sum(row['episodes'] for row in rows)
    perfect = sum(row['perfect_games'] for row in rows)
    count, region_lo, region_hi = widest_region(rows, level)
    best = max(percents)
    return {'rows': len(rows),
            'step_lo': int(rows[0]['step']), 'step_hi': int(rows[-1]['step']),
            'episodes_per_row': sorted({row['episodes'] for row in rows}),
            'episodes': episodes, 'perfect_games': perfect,
            'pooled_percent': 100.0 * perfect / episodes,
            'best_percent': best,
            'best_steps': [int(row['step']) for row in rows if row['perfect_percent'] == best],
            'at_or_above': {threshold: sum(1 for p in percents if p >= threshold)
                            for threshold in (95.0, 98.0, 99.0, 100.0)},
            'widest_region': count, 'region_lo': region_lo, 'region_hi': region_hi}


def text_summary(rows, name, level=REGION_LEVEL):
    facts = summarise(rows, level)
    if not facts:
        return '{0}: no stage-B rows'.format(name)
    per_row = '/'.join(str(count) for count in facts['episodes_per_row'])
    lines = ['{0}  stage B'.format(name),
             '  rows              {0:>10,}   steps {1:,} - {2:,}'.format(
                 facts['rows'], facts['step_lo'], facts['step_hi']),
             '  episodes per row  {0:>10}'.format(per_row),
             '  pooled perfect    {0:>9.2f}%   ({1:,} / {2:,})'.format(
                 facts['pooled_percent'], facts['perfect_games'], facts['episodes']),
             '  best row          {0:>9.1f}%   @ step {1:,}{2}'.format(
                 facts['best_percent'], facts['best_steps'][0],
                 '' if len(facts['best_steps']) == 1
                 else '  ({0} rows tie)'.format(len(facts['best_steps'])))]
    for threshold in (95.0, 98.0, 99.0, 100.0):
        lines.append('  at or above {0:>4.0f}%  {1:>10,}'.format(
            threshold, facts['at_or_above'][threshold]))
    if facts['widest_region']:
        lines.append('  widest >={0:.0f}% run  {1:>10,}   steps {2:,} - {3:,}'.format(
            level, facts['widest_region'], facts['region_lo'], facts['region_hi']))
    else:
        lines.append('  widest >={0:.0f}% run  {1:>10}'.format(level, 'none'))

    ranked = sorted(rows, key=lambda row: (-row['perfect_percent'], int(row['step'])))[:TOP_N]
    lines.append('  top {0}:'.format(len(ranked)))
    for row in ranked:
        # `perfect_ci95` is stored in percent, like `perfect_percent` beside it.
        low, high = row.get('perfect_ci95') or (float('nan'), float('nan'))
        lines.append('    {0:>10,}  {1:6.1f}%  [{2:.1f}, {3:.1f}]  {4}/{5}'.format(
            int(row['step']), row['perfect_percent'], low, high,
            row['perfect_games'], row['episodes']))
    return '\n'.join(lines)


def build_figure(rows, name=None, level=REGION_LEVEL):
    """The figure, built through the object API. Returns `(figure, axis)`.

    `Figure` + `FigureCanvasAgg` rather than `pyplot`, for the same reason as `progress_chart`:
    pyplot's global figure manager keeps every artist alive and leaked ~0.45 MB an eval in snek2.
    """
    figure = Figure(figsize=FIGSIZE, dpi=100 * progress_chart.CHART_SCALE)
    FigureCanvasAgg(figure)
    axis = figure.add_subplot(1, 1, 1)
    if not rows:
        axis.set_title('{0} — no stage-B rows'.format(name or ''), fontsize=LABEL_SIZE + 1)
        figure.tight_layout(pad=0.4)
        return figure, axis

    steps = np.array([int(row['step']) for row in rows], dtype=np.int64)
    percents = np.array([row['perfect_percent'] for row in rows], dtype=np.float64)

    # Small dots, no line: adjacent checkpoints are a thousand training steps apart and their
    # measurements are independent samples, so joining them draws sampling noise as a trajectory.
    axis.plot(steps, percents, marker='.', markersize=1.6, linestyle='none',
              color=POINT_COLOR, alpha=0.55)

    # Pinned before the rug is drawn, not after. Autoscale would move the floor the rug sits on as
    # soon as the rug extended it, leaving the marks hanging above the axis by a hair.
    floor = max(0.0, min(float(percents.min()), level) - 2.0)
    axis.set_ylim(floor, 100.6)

    above = percents >= level
    if above.any():
        # Marks *where* the region is along the arm, which is the thing a count of rows cannot say.
        # A rug on the floor rather than a shaded span, so it can never hide a point.
        axis.plot(steps[above], np.full(int(above.sum()), floor + 0.25), marker='|',
                  markersize=3, linestyle='none', color=REGION_COLOR, alpha=0.5)
    axis.axhline(level, color=REGION_COLOR, linestyle=(0, (4, 3)), linewidth=0.6)

    # Without this the panel is a cloud. A single 100-episode row is quantised to whole percent and
    # carries 1.6 pp of sampling noise, which is larger than any difference between two neighbouring
    # checkpoints — so the *shape* of the arm is only visible as an average over rows.
    if len(rows) >= TREND_WINDOW:
        axis.plot(steps, progress_chart.trailing_average(percents.tolist(), TREND_WINDOW),
                  color=TREND_COLOR, linewidth=0.8)

    pooled = 100.0 * sum(row['perfect_games'] for row in rows) / sum(row['episodes'] for row in rows)
    axis.axhline(pooled, color='gray', linestyle=(0, (1, 2)), linewidth=0.6)
    # Anchored on the left: at the right it lands on top of the region guide for any arm whose
    # pooled rate is near the threshold, which is every arm worth charting.
    axis.annotate('{0:.2f}% pooled'.format(pooled), xy=(0.0, pooled),
                  xycoords=('axes fraction', 'data'), xytext=(2, -2), textcoords='offset points',
                  ha='left', va='top', fontsize=TICK_SIZE, color='gray')

    axis.set_ylabel('Perfect game %', fontsize=LABEL_SIZE)
    axis.set_xlabel('Training step', fontsize=LABEL_SIZE)
    axis.tick_params(labelsize=TICK_SIZE)
    # Same reason as the arm chart: matplotlib's default offset text renders a `1e6` in a font size
    # it was never given, and at these figure sizes it overlaps the axis label.
    axis.xaxis.set_major_formatter(FuncFormatter(lambda step, _: '{0:,.0f}k'.format(step / 1000)))
    axis.grid(True, linewidth=0.3, alpha=0.3)
    if name:
        facts = summarise(rows, level)
        axis.set_title('{0} — {1:,} rows, {2:,} ep each, best {3:.1f}%, {4:,} at >={5:.0f}%'.format(
            name, facts['rows'], facts['episodes_per_row'][0], facts['best_percent'],
            facts['at_or_above'][level], level), fontsize=LABEL_SIZE)
    figure.tight_layout(pad=0.4)
    return figure, axis


def chart_path(policy, label=None):
    stem = os.path.basename(results.stage_b_path(policy, label))[:-len('.json')]
    return os.path.join(constants.RUNS_DIR, stem + '.png')


def render(rows, path, name=None, level=REGION_LEVEL):
    """Draws and writes the PNG atomically. Returns the pixels."""
    figure, _ = build_figure(rows, name=name, level=level)
    figure.canvas.draw()
    image = np.asarray(figure.canvas.buffer_rgba())[:, :, :3]
    if path:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        staging = path + '.partial.png'
        imageio.imwrite(staging, image)
        os.replace(staging, path)
    return image


def redraw(policy, label=None, out=None, level=REGION_LEVEL):
    """Rebuilds a pass's PNG. Returns `(path, rows)`; the path is None when there is nothing."""
    rows = load(policy, label)
    if not rows:
        return None, rows
    path = out or chart_path(policy, label)
    render(rows, path, name=results.run_name(policy), level=level)
    return path, rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('policy')
    parser.add_argument('--label', default=None, help='names the pass, as passed to the wave')
    parser.add_argument('--out', default=None, help='PNG path; defaults beside the result file')
    parser.add_argument('--level', type=float, default=REGION_LEVEL,
                        help='the region threshold, in percent (default 98)')
    parser.add_argument('--watch', type=float, default=0.0, metavar='SECONDS',
                        help='redraw every SECONDS; works on a wave still running')
    args = parser.parse_args(argv)

    while True:
        path, rows = redraw(args.policy, args.label, args.out, args.level)
        print(text_summary(rows, results.run_name(args.policy), args.level))
        print('chart: {0}'.format(path or 'not written'))
        if not args.watch:
            return 0 if rows else 1
        sys.stdout.flush()
        time.sleep(args.watch)


if __name__ == '__main__':
    sys.exit(main())
