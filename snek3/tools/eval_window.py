"""The box's stage-B window: one panel per arm of a close-out, live while the pass runs.

The training window's counterpart, and deliberately the *same* window mechanism —
[`chart_window.py`](chart_window.py) opens both, [`chart_viewer.py`](chart_viewer.py) draws both.
Only two things differ, and they are the two arguments below:

- **what it is pointed at.** The training window follows the box's live-arm registry, because arms
  come and go. A close-out's arms are known when it starts, so this passes the list of stage-B chart
  paths outright — the panels are the batch, in a fixed order, whether or not a file exists yet
- **when it closes.** A training window closes when the registry has been empty for a while; this one
  closes when the close-out process it is watching is gone (`--watch-pid`)

**Nothing produces those PNGs on its own** — `tools/closeout.py` redraws them as the pass runs, off
the shard files, which is the same thing `stage_b_chart --watch` does by hand. A window opened while
nothing is redrawing shows the last state of each pass, which is correct but static.

It takes **its own slot**, so a box can hold a training window and this one at once. That is not
hypothetical on the laptop, where nothing separates a training from an eval.

    PYTHONPATH=. python -m tools.eval_window                     # whatever stage-B charts are newest
    PYTHONPATH=. python -m tools.eval_window p3a p3b --label ab  # a named batch

Killing it is free, exactly as for the training window: no pass reads it, waits on it, or reopens it.
"""

import argparse
import os
import sys

from env import constants
from tools import chart_window
from tools import stage_b_chart

VIEWER_MODULE = 'tools.chart_viewer'

# Its own slot, beside the training window's in `runs/.live/`.
SLOT_NAME = '.evalwindow'

TITLE = 'snek3 — stage B'

# A shard rewrites its file after each 500-episode row, ~10 s of its own time, and the redraw behind
# this is on its own clock anyway. Faster than this is CPU taken off the measurement.
REFRESH_SECONDS = 20

# Every stage-B chart, for a hand relaunch that does not want to type the batch out. The viewer's
# `max_panels` keeps the newest 8 by mtime, which is the batch being written.
GLOB = '*_checkpoint_evals*.png'


def chart_paths(policies, label=None):
    """The PNG each arm's pass is drawn into, in the order the close-out measures them."""
    return [stage_b_chart.chart_path(policy, label) for policy in policies]


def command(paths=(), glob_pattern=None, watch_pids=(), scale=None, python=None):
    """The argv for the stage-B window. Pure, so the spelling is testable without a display."""
    argv = [python or sys.executable, '-u', '-m', VIEWER_MODULE]
    argv += list(paths)
    if glob_pattern:
        argv += ['--glob', glob_pattern]
    if watch_pids:
        argv += ['--watch-pid', ','.join(str(pid) for pid in watch_pids)]
    return argv + ['--slot', SLOT_NAME,
                   '--interval', str(REFRESH_SECONDS),
                   '--scale', str(chart_window.DEFAULT_SCALE if scale is None else scale),
                   '--title', TITLE]


def holder(runs_dir=None):
    """The pid of the stage-B window that is up, or None. **Advisory**, as on the training side."""
    return chart_window.holder(runs_dir, SLOT_NAME)


def ensure(paths=(), glob_pattern=None, watch_pids=(), runs_dir=None, env=None, scale=None):
    """Asks for the stage-B window. Returns the Popen of the process started, or None.

    None means nothing was started — windows are off (`SNEK_CHART_WINDOW=0`, the same switch as the
    training window, because "no window on this box" is one decision and not two), one is already up,
    or it would not start. A close-out must not care either way.
    """
    if scale is None and env is not None:
        scale = env.get('SNEK_CHART_WINDOW_SCALE')
    argv = command(paths, glob_pattern, watch_pids,
                   float(scale) if scale else None)
    return chart_window.ensure(runs_dir, env, argv, SLOT_NAME, label='stage-B window')


reap = chart_window.reap


def main(argv=None):
    """`python -m tools.eval_window` — the supported way to open one by hand.

    With no policies it shows whatever stage-B charts are newest, which is what a window relaunched
    mid-pass wants and saves typing eight arm names.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('policies', nargs='*', help='the batch; default is the newest charts')
    parser.add_argument('--label', default=None, help='names the pass, as passed to the close-out')
    args = parser.parse_args(argv)

    current = holder()
    if current:
        print('a stage-B window is already up: pid {0}'.format(current))
        return 0
    paths = chart_paths(args.policies, args.label) if args.policies else ()
    glob_pattern = None if args.policies else os.path.join(constants.RUNS_DIR, GLOB)
    # The child says whether it got the slot, on this terminal. Nothing to add.
    return 0 if chart_window.spawn(argv=command(paths, glob_pattern),
                                   label='stage-B window') is not None else 1


if __name__ == '__main__':
    sys.exit(main())
