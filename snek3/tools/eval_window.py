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
    PYTHONPATH=. python -m tools.eval_window b6a b6b --label ab  # a named batch

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


def command(paths=(), glob_pattern=None, watch_pids=(), scale=None, python=None,
            max_width_px=None):
    """The argv for the stage-B window. Pure, so the spelling is testable without a display.

    The size flags come from `chart_window.size_argv`, not from here: the two windows differ in what
    they point at and when they close, and **must not** differ in how they are sized.
    """
    argv = [python or sys.executable, '-u', '-m', VIEWER_MODULE]
    argv += list(paths)
    if glob_pattern:
        argv += ['--glob', glob_pattern]
    if watch_pids:
        argv += ['--watch-pid', ','.join(str(pid) for pid in watch_pids)]
    return argv + ['--slot', SLOT_NAME,
                   '--interval', str(REFRESH_SECONDS),
                   '--title', TITLE] + chart_window.size_argv(scale, max_width_px)


def holder(runs_dir=None):
    """The pid of the stage-B window that is up, or None. **Advisory**, as on the training side."""
    return chart_window.holder(runs_dir, SLOT_NAME)


def ensure(paths=(), glob_pattern=None, watch_pids=(), runs_dir=None, env=None, scale=None,
           max_width_px=None):
    """Asks for the stage-B window. Returns the Popen of the process started, or None.

    None means nothing was started — windows are off (`SNEK_CHART_WINDOW=0`, the same switch as the
    training window, because "no window on this box" is one decision and not two) or it would not
    start. A close-out must not care either way.

    **Unlike the training window, a holder does not stop the spawn.** `chart_window.ensure` skips a
    spawn when the slot's pid is alive, which is right for a wave of arms joining one window and
    wrong here: the live holder is usually the *previous* pass's window in its closing grace, and a
    close-out asks exactly once. So the viewer is always started, and it is the viewer that waits for
    the slot and takes it when the holder leaves (`chart_viewer.stand_by_for_slot`). 2026-09-04.
    """
    if not chart_window.wanted(env):
        return None
    if scale is None and max_width_px is None:
        scale, max_width_px = chart_window.sizing(env)
    argv = command(paths, glob_pattern, watch_pids, scale, None, max_width_px)
    return chart_window.spawn(runs_dir, env, argv, label='stage-B window')


reap = chart_window.reap


def main(argv=None):
    """`python -m tools.eval_window` — the supported way to open one by hand.

    With no policies it shows whatever stage-B charts are newest, which is what a window relaunched
    mid-pass wants and saves typing eight arm names.

    **Give it the close-out's pid with `--watch-pid`**, and it closes when the pass does, exactly like
    the window the close-out would have opened. Without one it has no way to know when the pass ends
    — the viewer only closes on watched pids or on the training registry — so it stays up until
    closed by hand, and says so. Until 2026-09-04 it said nothing, and a relaunched window sat on the
    slot after its pass, which is the very thing the relaunch was fixing.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('policies', nargs='*', help='the batch; default is the newest charts')
    parser.add_argument('--label', default=None, help='names the pass, as passed to the close-out')
    parser.add_argument('--watch-pid', default='',
                        help="the close-out's pid (comma-separated for several); the window closes "
                             'once none is alive. Without it the window stays until closed by hand')
    args = parser.parse_args(argv)
    watch_pids = [int(token) for token in args.watch_pid.split(',') if token.strip()]

    current = holder()
    if current:
        print('a stage-B window is already up: pid {0}'.format(current))
        return 0
    paths = chart_paths(args.policies, args.label) if args.policies else ()
    glob_pattern = None if args.policies else os.path.join(constants.RUNS_DIR, GLOB)
    if not watch_pids:
        print('no --watch-pid: this window will stay up until it is closed by hand', flush=True)
    # The child says whether it got the slot, on this terminal. Nothing to add.
    return 0 if chart_window.spawn(argv=command(paths, glob_pattern, watch_pids),
                                   label='stage-B window') is not None else 1


if __name__ == '__main__':
    sys.exit(main())
