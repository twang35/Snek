"""One PNG per arm: average score in blue on the left axis, perfect-game rate in red on the right.

    PYTHONPATH=. python -m tools.progress_chart <policy>      # redraw from runs/<name>_evals.json

`docs/charts.md` links `runs/<name>.png` directly, so this is the only place the image is written and
there is no copy step to keep in sync.

**Built through matplotlib's object API, never `pyplot`.** `pyplot` registers every figure in a
process-global manager and a callback registry, and `plt.close()` does not fully release them — so a
figure per eval leaked artists and grew snek2's trainer ~1 MB per eval, worse at high dpi, where it
eventually exhausted the desktop's memory. A bare `Figure` is never registered, so it is collected
when it goes out of scope and there is nothing to close.

**Every visual choice here was made against a real ~3,000-eval arm**, which is what the comments
record: at that density the default stroke width smears thousands of points into a solid band with no
readable texture, so the traces are 0.3 pt with a darker trailing average over the top.
"""

import argparse
import os
import sys

import imageio.v2 as imageio
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter

from env import constants
from tools import results
from tools import run_report

# dpi is `100 * scale`. The default keeps the PNG crisp when a viewer magnifies it ~1.5-2x, which is
# what `chart_viewer` does — a low-dpi render looks blurry blown up. Scaling dpi rather than figsize
# keeps text and line weights proportional.
CHART_SCALE = float(os.environ.get('SNEK_CHART_SCALE', '2.0'))

FIGSIZE = (3.65, 2.25)
LABEL_SIZE = 6
TICK_SIZE = 5
SCORE_COLOR = 'tab:blue'
PERCENT_COLOR = 'tab:red'
TREND_COLOR = 'darkred'
TREND_WINDOW = 10

# The perfect-score guide only appears once an arm has been within this fraction of it. Before that
# the line sits far above every point and its only effect is to stretch the y axis and squash the
# score trace into the bottom of the plot. Gated on the arm's *maximum* rather than its latest point
# so it does not flicker on and off between renders.
GUIDE_FRACTION = 0.8


def trailing_average(values, window):
    """Element `i` is the mean of the last `window` values up to and including `i`.

    Trailing rather than centred, because the chart is read live: a centred average would pull the
    newest point toward values that do not exist yet.
    """
    averaged = []
    for index in range(len(values)):
        chunk = values[max(0, index - window + 1):index + 1]
        averaged.append(sum(chunk) / len(chunk))
    return averaged


def build_figure(eval_rows, name=None, resume_steps=()):
    """`(figure, score_axis, percent_axis)` for an arm's whole history, across however many runs
    made it.

    Separate from `render` so a test can assert what is on the axes — a title, a guide line — instead
    of asserting about pixels or reaching in and replacing `Figure`. Takes explicit rows rather than
    assuming evenly spaced evals from a starting step, because a resumed arm's history has a gap
    wherever it was stopped.
    """
    figure = Figure(figsize=FIGSIZE, dpi=100 * CHART_SCALE)
    FigureCanvasAgg(figure)
    score_axis = figure.add_subplot(1, 1, 1)

    steps = [row['step'] for row in eval_rows]
    scores = [row['avg_score'] for row in eval_rows]
    percents = [row['perfect_percent'] for row in eval_rows]

    # Thin on purpose: a long run packs thousands of points into ~300 px of plot width, so the
    # default 1.5 pt stroke overlaps itself into a near-solid band. 0.3 keeps individual excursions
    # readable, chosen by comparing 0.3/0.5/0.8/1.2 side by side on a real arm.
    score_axis.plot(steps, scores, color=SCORE_COLOR, linewidth=0.3)
    score_axis.set_ylabel('Average score', color=SCORE_COLOR, fontsize=LABEL_SIZE)
    # The latest step goes in the x label because late in training the traces fill the plot and the
    # axis's own "1e6" offset is too coarse to say where the run is now.
    score_axis.set_xlabel(
        'Iterations ({0:,}k steps)'.format(steps[-1] // 1000) if steps else 'Iterations',
        fontsize=LABEL_SIZE)
    score_axis.tick_params(axis='y', labelcolor=SCORE_COLOR, labelsize=TICK_SIZE)
    score_axis.tick_params(axis='x', labelsize=TICK_SIZE)
    # Ticks in thousands of steps, which also removes matplotlib's "1e6" offset label. That offset
    # renders at the default font size — three times the tick labels here — and lands in the
    # bottom-right corner competing with the axis label that already states the step count.
    score_axis.xaxis.set_major_formatter(FuncFormatter(lambda step, _: '{0:,.0f}k'.format(step / 1000)))

    # The arm's name is burned into the image because neither consumer can add it: a chart viewer
    # renders each arm as a bare panel with the axes off, and `charts.md` embeds the PNG with only a
    # markdown caption. An untitled figure is unidentifiable in a four-panel wave window, which is
    # exactly where identifying it matters. Black rather than either trace's colour, so it is not
    # mistaken for belonging to one axis.
    if name:
        score_axis.set_title(name, fontsize=LABEL_SIZE + 1, color='black', pad=2)

    percent_axis = score_axis.twinx()
    percent_axis.plot(steps, percents, color=PERCENT_COLOR, linewidth=0.3)
    percent_axis.set_ylabel('Perfect game %', color=PERCENT_COLOR, fontsize=LABEL_SIZE)
    percent_axis.tick_params(axis='y', labelcolor=PERCENT_COLOR, labelsize=TICK_SIZE)
    percent_axis.set_ylim(bottom=0, top=100)

    # Guides on the perfect-game axis. Without them the red trace cannot be read off the right-hand
    # axis by eye, since the left axis's ticks are on a different scale. In the percent axis's own
    # colour so it is unambiguous which axis they belong to; zorder 1 sits above the axes patch and
    # below both traces, so they never hide data. Alpha below ~0.4 is invisible at this figure size.
    percent_axis.set_yticks([0, 20, 40, 60, 80, 100])
    for level in (20, 40, 60, 80):
        percent_axis.axhline(level, color=PERCENT_COLOR, linestyle=(0, (4, 3)),
                             linewidth=0.5, alpha=0.55, zorder=1)

    # The smoothed version of the red trace, not a new quantity — so the same colour family, darker
    # so it reads as the level through the noise. Only a hair wider than the raw trace: a 10-eval
    # average still wiggles at this density, and a bold line just reprints the noise heavier.
    if percents:
        percent_axis.plot(steps, trailing_average(percents, TREND_WINDOW), color=TREND_COLOR,
                          linewidth=0.4, alpha=0.9, zorder=3)

    # What a perfect game scores, so the score trace answers "how much is actually left". On the
    # score axis, so it reads against the left-hand ticks.
    if scores and max(scores) > GUIDE_FRACTION * constants.MAX_POSSIBLE_SCORE:
        score_axis.axhline(constants.MAX_POSSIBLE_SCORE, color='tab:green', linestyle=(0, (4, 3)),
                           linewidth=0.5, alpha=0.65, zorder=1)

    # One line per restart, so a dip or a jump can be tied to a resume rather than to the policy.
    for resume_step in resume_steps:
        score_axis.axvline(resume_step, color='gray', linestyle='--', linewidth=0.6)

    figure.tight_layout()
    return figure, score_axis, percent_axis


def render(eval_rows, path, name=None, resume_steps=()):
    """Draws the arm and writes the PNG. Returns the pixels."""
    figure, _, _ = build_figure(eval_rows, name=name, resume_steps=resume_steps)
    figure.canvas.draw()
    image = np.asarray(figure.canvas.buffer_rgba())[:, :, :3]

    if path:
        # Written beside the target and renamed, so anything reading it mid-eval never sees a
        # half-written PNG. The trainer rewrites this every eval while a viewer may be polling it.
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        staging = path + '.partial.png'
        imageio.imwrite(staging, image)
        os.replace(staging, path)
    # No `close()`: this Figure was never registered with pyplot, so it is collected when this
    # function returns. That is the whole point of the object API above.
    return image


def chart_path(policy):
    return os.path.join(constants.RUNS_DIR, '{0}.png'.format(results.run_name(policy)))


def redraw(policy):
    """Rebuilds an arm's PNG from its stage-A file. Returns the path, or None if there is nothing."""
    eval_rows, resume_steps = run_report.load_history(results.stage_a_path(policy))
    if not eval_rows:
        return None
    path = chart_path(policy)
    render(eval_rows, path, name=results.run_name(policy), resume_steps=resume_steps)
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('policy', nargs='+')
    args = parser.parse_args(argv)
    for policy in args.policy:
        path = redraw(policy)
        print('{0}: {1}'.format(policy, path or 'no evals in {0}'.format(
            results.stage_a_path(policy))))
    return 0


if __name__ == '__main__':
    sys.exit(main())
