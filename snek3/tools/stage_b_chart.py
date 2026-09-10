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
from tools import eta, progress_chart, results

# One deliberately wider figure than `progress_chart`'s: a stage-B pass has one series and up to a
# few thousand points along the step axis, where an arm's chart has two axes and a trend line.
FIGSIZE = (5.0, 2.4)
# The dotted guide, the rug beneath it and the title's count all read one number, **the gate of the pass
# drawn** (user, 2026-09-09): a stage-B and a hof5000 chart draw it at the hof5000 cut (99.2, the cut a
# stage-B row is judged by and the cut that produced a hof5000 row) and a hof30k chart at the hof30k cut
# (99.6, what produced its rows). Before that it was one constant for every pass: 99 from 2026-08-30 and
# 98 before, and each time the rows outgrew it the guide sat under the cloud marking a region that was no
# longer the interesting one; reading the gate from `eta` means it moves when the gate does. `--level`
# still overrides it, and `summarise` reports 95/98/99/100 whatever it is set to. `REGION_LEVEL` stays
# as the fallback for a label no pass owns.
REGION_LEVEL = eta.HOF_THRESHOLD
REGION_LEVELS = {None: eta.HOF_THRESHOLD, 'hof5000': eta.HOF_THRESHOLD, 'hof30k': eta.HOF30K_THRESHOLD}


def region_level(label=None):
    """The guide level for a pass's chart: the gate its rows are read against (`REGION_LEVELS`)."""
    return REGION_LEVELS.get(label, REGION_LEVEL)
POINT_COLOR = 'tab:red'
REGION_COLOR = 'tab:green'
TREND_COLOR = '#4fa3e0'
TREND_HALF_WINDOW = 1_000_000
TREND_LINEWIDTH = 0.8
# **The trend line is a pooled rate over a step window, added 2026-09-09 (user's call), and it replaces a
# design that was removed on 2026-09-01 for two reasons that still hold.** The removed line was a 40-row
# trailing mean: (1) it averaged row *percentages*, and (2) on a re-measure pass (`above:` selector, e.g.
# `hof5000`) the rows are a selected subset -- 274 of ~14,000 checkpoints for b4, separated by tens of
# millions of transitions -- so a trailing mean over N rows joined points that were not neighbours and
# drew a trend in the selection as a trend in the policy; its 40-row gate also put it on one arm of a
# batch and not the rest. This one is defined on the *step* axis: at each row, the perfect games over the
# episodes of every row within `TREND_HALF_WINDOW` transitions either side -- the basin mean `HOF.md`
# quotes by hand, drawn along the arm -- so it only ever pools neighbours, is episode-weighted rather
# than an average of percentages, and is broken (not bridged) wherever two consecutive rows are further
# apart than the window. It appears on every pass with two or more full rows, so every arm of a batch
# carries it. Window: 1M transitions is ~30 rows of a hof5000 pass (150k episodes, sd ~0.02 pp) and ~55
# rows of a stage-B pass (27k episodes, sd ~0.05 pp).
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


def full_rows(rows):
    """The rows measured to full length. A row stopped early (`abandoned`, `plans/early-stop.md`) is a
    short, downward-biased sample that is only ever "below the target": it is drawn, counted as
    stopped, and left out of every pooled or threshold statistic here."""
    return [row for row in rows if not row.get('abandoned')]


def summarise(rows, level=REGION_LEVEL):
    """The numbers the text block prints, as a dict. Empty rows give an empty dict. The pooled rate,
    the thresholds and the region are over full rows only; `stopped` says how many were not."""
    if not rows:
        return {}
    full = full_rows(rows)
    percents = [row['perfect_percent'] for row in full]
    episodes = sum(row['episodes'] for row in full)
    perfect = sum(row['perfect_games'] for row in full)
    count, region_lo, region_hi = widest_region(full, level) if full else (0, None, None)
    best = max(percents) if percents else None
    return {'rows': len(rows), 'stopped': len(rows) - len(full),
            'step_lo': int(rows[0]['step']), 'step_hi': int(rows[-1]['step']),
            'episodes_per_row': sorted({row.get('episodes_planned', row['episodes']) for row in rows}),
            'episodes': episodes, 'perfect_games': perfect,
            'pooled_percent': 100.0 * perfect / episodes if episodes else None,
            'best_percent': best,
            'best_steps': [int(row['step']) for row in full if row['perfect_percent'] == best],
            'at_or_above': {threshold: sum(1 for p in percents if p >= threshold)
                            for threshold in sorted({95.0, 98.0, 99.0, 100.0, float(level)})},
            'widest_region': count, 'region_lo': region_lo, 'region_hi': region_hi}


def text_summary(rows, name, level=REGION_LEVEL):
    facts = summarise(rows, level)
    if not facts:
        return '{0}: no stage-B rows'.format(name)
    per_row = '/'.join(str(count) for count in facts['episodes_per_row'])
    lines = ['{0}  stage B'.format(name),
             '  rows              {0:>10,}   steps {1:,} - {2:,}{3}'.format(
                 facts['rows'], facts['step_lo'], facts['step_hi'],
                 '   ({0:,} stopped early; the numbers below are over the full rows)'.format(facts['stopped'])
                 if facts['stopped'] else ''),
             '  episodes per row  {0:>10}'.format(per_row)]
    if facts['pooled_percent'] is not None:
        lines += ['  pooled perfect    {0:>9.2f}%   ({1:,} / {2:,})'.format(
                      facts['pooled_percent'], facts['perfect_games'], facts['episodes']),
                  '  best row          {0:>9.1f}%   @ step {1:,}{2}'.format(
                      facts['best_percent'], facts['best_steps'][0],
                      '' if len(facts['best_steps']) == 1
                      else '  ({0} rows tie)'.format(len(facts['best_steps'])))]
    else:
        lines.append('  every row stopped early; nothing to pool')
    for threshold in sorted(facts['at_or_above']):
        lines.append('  at or above {0:>5g}%  {1:>10,}'.format(
            threshold, facts['at_or_above'][threshold]))
    if facts['widest_region']:
        lines.append('  widest >={0:g}% run  {1:>10,}   steps {2:,} - {3:,}'.format(
            level, facts['widest_region'], facts['region_lo'], facts['region_hi']))
    else:
        lines.append('  widest >={0:g}% run  {1:>10}'.format(level, 'none'))

    ranked = sorted(full_rows(rows), key=lambda row: (-row['perfect_percent'], int(row['step'])))[:TOP_N]
    lines.append('  top {0}:'.format(len(ranked)))
    for row in ranked:
        # `perfect_ci95` is stored in percent, like `perfect_percent` beside it.
        low, high = row.get('perfect_ci95') or (float('nan'), float('nan'))
        lines.append('    {0:>10,}  {1:6.1f}%  [{2:.1f}, {3:.1f}]  {4}/{5}'.format(
            int(row['step']), row['perfect_percent'], low, high,
            row['perfect_games'], row['episodes']))
    return '\n'.join(lines)


def pooled_trend(rows, half_window=TREND_HALF_WINDOW):
    """`(steps, trend)` in step order: at each row, the pooled perfect rate (games / episodes) over every
    row within `half_window` transitions either side, as a percentage. A row further than `half_window`
    from its predecessor starts a new segment: the value before it is NaN, so a line through the result
    breaks across the gap instead of bridging it. Empty for fewer than two rows."""
    if len(rows) < 2:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float64)
    order = sorted(rows, key=lambda row: int(row['step']))
    steps = np.array([int(row['step']) for row in order], dtype=np.int64)
    games = np.array([row['perfect_games'] for row in order], dtype=np.float64)
    episodes = np.array([row['episodes'] for row in order], dtype=np.float64)
    cum_games = np.concatenate([[0.0], np.cumsum(games)])
    cum_episodes = np.concatenate([[0.0], np.cumsum(episodes)])
    low = np.searchsorted(steps, steps - half_window, side='left')
    high = np.searchsorted(steps, steps + half_window, side='right')
    trend = 100.0 * (cum_games[high] - cum_games[low]) / (cum_episodes[high] - cum_episodes[low])
    gap = np.diff(steps) > half_window
    if gap.any():
        # NaN on the row *before* each gap: matplotlib lifts the pen there and the next row starts a new segment.
        out_steps = []
        out_trend = []
        for index in range(len(steps)):
            out_steps.append(steps[index])
            out_trend.append(trend[index])
            if index < len(gap) and gap[index]:
                out_steps.append(steps[index])
                out_trend.append(np.nan)
        return np.array(out_steps, dtype=np.int64), np.array(out_trend, dtype=np.float64)
    return steps, trend


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

    # A stopped row (`abandoned`) is drawn hollow and grey at the rate it had when it stopped -- it is
    # a real reading of "below the target", not a measurement to pool -- and every statistic on the
    # chart is over the full rows. `percents` is the full rows' from here on.
    stopped = [row for row in rows if row.get('abandoned')]
    rows = full_rows(rows)
    if stopped:
        axis.plot([int(row['step']) for row in stopped], [row['perfect_percent'] for row in stopped],
                  marker='o', markersize=1.8, markerfacecolor='none', markeredgewidth=0.4,
                  linestyle='none', color='gray', alpha=0.5)
    if not rows:
        axis.set_title('{0} — every row stopped early'.format(name or ''), fontsize=LABEL_SIZE + 1)
        figure.tight_layout(pad=0.4)
        return figure, axis
    steps = np.array([int(row['step']) for row in rows], dtype=np.int64)
    percents = np.array([row['perfect_percent'] for row in rows], dtype=np.float64)

    # Small dots, no line: adjacent checkpoints are a thousand training steps apart and their
    # measurements are independent samples, so joining them draws sampling noise as a trajectory.
    axis.plot(steps, percents, marker='.', markersize=1.6, linestyle='none',
              color=POINT_COLOR, alpha=0.55)
    trend_steps, trend = pooled_trend(rows)
    if trend.size:
        axis.plot(trend_steps, trend, color=TREND_COLOR, linewidth=TREND_LINEWIDTH, alpha=0.95)
        # Pinned to the axes just above the rug at the bottom right, opposite the pooled label: on any
        # arm the line itself ends inside the cloud, where a label on it would sit on the dots.
        axis.annotate('\u2014 pooled rate, \u00b1{0:g}M window'.format(TREND_HALF_WINDOW / 1e6),
                      xy=(1.0, 0.0), xycoords='axes fraction', xytext=(-3, 9), textcoords='offset points',
                      ha='right', va='bottom', fontsize=TICK_SIZE, color=TREND_COLOR)

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
        axis.set_title('{0} — {1:,} rows, {2:,} ep each, best {3:.1f}%, {4:,} at >={5:g}%'.format(
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


def redraw(policy, label=None, out=None, level=None):
    """Rebuilds a pass's PNG at the pass's gate (`region_level`), or `level`. Returns `(path, rows)`;
    the path is None when there is nothing."""
    level = region_level(label) if level is None else level
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
    parser.add_argument('--level', type=float, default=None,
                        help='the region threshold, in percent (default: the pass\'s gate -- 99.2 for '
                             'stage B and hof5000, 99.6 for hof30k)')
    parser.add_argument('--watch', type=float, default=0.0, metavar='SECONDS',
                        help='redraw every SECONDS; works on a wave still running')
    args = parser.parse_args(argv)

    while True:
        level = region_level(args.label) if args.level is None else args.level
        path, rows = redraw(args.policy, args.label, args.out, level)
        print(text_summary(rows, results.run_name(args.policy), level))
        print('chart: {0}'.format(path or 'not written'))
        if not args.watch:
            return 0 if rows else 1
        sys.stdout.flush()
        time.sleep(args.watch)


if __name__ == '__main__':
    sys.exit(main())
