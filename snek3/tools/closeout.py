"""A batch's stage-B pass: every arm of a wave, measured in sequence, with a window over it.

    PYTHONPATH=. python -u -m tools.closeout b6a b6b b6c --shards 8

**This is the close-out, and both boxes run this file.** The desktop's daemon dispatches one `eval`
job per batch and that job is this command; on the laptop the same command is typed by hand. Before
2026-08-30 the desktop built a `sh -c` string that chained one `evaluate.py` per arm and the laptop
used whatever an agent had written that day — two sequencers, one of which was a scratch shell script
whose `| tail` swallowed the wave's progress lines. There is one now.

**The arms go one at a time**, which is the 2026-08-29 decision this inherits rather than revisits:
`tools/eval_wave.py` measures a single policy, so N arms are N waves. It costs a lane drain at each
arm's end — ~11 s, against hours for the batch — and what it buys is that a batch is one process, one
log and one thing to look at.

**A failing arm never drops the arms behind it.** Every arm runs; the exit status is the last
failure's. That is the `;`-not-`&&` rule of the shell version it replaces, and the snek2 incident
behind it is a wave that lost a whole batch's measurement to one bad arm.

**Progress is visible three ways**, and none of them is this file doing work per episode: the wave's
own `[ 2167/3222] 4 shard(s) alive, 9m elapsed  eta 5m` line, each arm's stage-B PNG — redrawn here
off the shard files while the wave runs — and the window over those PNGs
([`eval_window.py`](eval_window.py)), which closes itself when this process ends.
"""

import argparse
import os
import sys
import threading
import time

from tools import eval_wave
from tools import eval_window
from tools import stage_b_chart

# How often an arm's PNG is rebuilt from its shard files. A shard writes a row every ~10 s and the
# window redraws on its own clock; faster than this is CPU taken off the measurement.
REDRAW_SECONDS = 20


class Drawer(object):
    """Keeps one arm's stage-B PNG fresh while its wave runs, on a daemon thread.

    A thread and not a subprocess, deliberately: a `stage_b_chart --watch` child would have to be
    reaped and could outlive a killed close-out, and this project has spent hours on orphaned
    workers. A daemon thread cannot outlive its process.

    **Every failure here is swallowed.** A chart is a readout; a pass that dies because a PNG could
    not be written would be the tail wagging the dog.
    """

    def __init__(self, policy, label=None, interval=REDRAW_SECONDS):
        self.policy, self.label, self.interval = policy, label, interval
        self.stop = threading.Event()
        self.thread = None

    def draw_once(self):
        try:
            return stage_b_chart.redraw(self.policy, self.label)[0]
        except Exception as error:                    # noqa: BLE001 - a readout, never the pass
            sys.stderr.write('chart redraw failed for {0}: {1}\n'.format(self.policy, error))
            return None

    def _loop(self):
        while not self.stop.wait(self.interval):
            self.draw_once()

    def __enter__(self):
        self.thread = threading.Thread(target=self._loop, name='redraw', daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *exc_info):
        self.stop.set()
        if self.thread is not None:
            self.thread.join(timeout=5.0)
        # The last word on the arm is drawn from the merged file the wave has just written, not from
        # the shard files a mid-pass redraw pooled.
        self.draw_once()
        return False


def measure_one(policy, episodes=500, width=None, seed=0):
    """A single checkpoint, in this process — the `one` selector.

    `one` is not a close-out; it is the spot check and the record re-measurement, and it goes through
    here only so that **every eval the desktop can dispatch has one entry point**. The alternative was
    a second branch in `desktop/runner/launch.py` choosing between two scripts, which is the shape
    this change exists to remove.

    Imported where it is used rather than at the top: `evaluate.py` is a script, and a `tools/` module
    importing one is the wrong direction to make permanent — this is a delegation, not a dependency.
    """
    import evaluate
    return evaluate.measure_one(policy, None, episodes, width, seed)


def run(policies, selector='screen', episodes=500, shards=4, label=None, width=None, seed=0,
        resume=True, merge=True, window=True, redraw_interval=REDRAW_SECONDS):
    """Measures every arm in turn. Returns the last failing arm's status, or 0.

    The keyword defaults are `evaluate.py`'s, which are the protocol — `screen:97` at 500 episodes.
    Nothing here has an opinion about them; a caller that wants the protocol passes nothing.
    """
    # No window for `one`: it writes no pass and there is nothing to draw.
    if window and selector != 'one':
        eval_window.ensure(eval_window.chart_paths(policies, label), watch_pids=[os.getpid()])

    status = 0
    started = time.time()
    print('=== close-out: {0} arm(s), selector {1}, {2} episodes, {3} shard(s)'.format(
        len(policies), selector, episodes, shards), flush=True)
    for index, policy in enumerate(policies, 1):
        print('\n--- [{0}/{1}] {2}  {3}'.format(
            index, len(policies), policy, time.strftime('%H:%M:%S')), flush=True)
        try:
            if selector == 'one':
                measure_one(policy, episodes, width, seed)
                code = 0
            else:
                with Drawer(policy, label, redraw_interval):
                    code = eval_wave.run(policy, selector, episodes, shards, label, width, seed,
                                         resume, merge)
        except Exception as error:                    # noqa: BLE001 - see the class docstring
            code = 1
            print('--- {0} raised {1}: {2}'.format(policy, type(error).__name__, error), flush=True)
        if code:
            status = code
            print('--- {0} exited {1}; the arms behind it still run'.format(policy, code),
                  flush=True)
    print('\n=== close-out done in {0:.1f}m, status {1}'.format(
        (time.time() - started) / 60.0, status), flush=True)
    return status


def build_parser():
    """This module's command-line contract, as an object.

    Extracted for the same reason `evaluate.build_parser` is: the desktop daemon runs on base python
    and cannot import this file, so `desktop/runner/launch.py` spells this interface out by hand.
    That duplication has drifted before — a `--selector` flag against a positional, and several
    policies against a single one, which made every wave the box dispatched exit 2 — so
    `tests/test_desktop_runner.py` hands what the daemon builds to this parser.

    The selector is a **flag** here and positional in `evaluate.py`. It has to be: `policies` is
    `nargs='+'`, and a trailing positional behind it is ambiguous.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('policies', nargs='+')
    parser.add_argument('--selector', default='screen',
                        help='a step selector; the default is the protocol, screen:97')
    parser.add_argument('--episodes', type=int, default=500)
    parser.add_argument('--shards', type=int, default=4)
    parser.add_argument('--label', default=None, help='names the pass, so two do not collide')
    parser.add_argument('--width', type=int, default=None, help='games in lockstep, per process')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--no-resume', action='store_true')
    parser.add_argument('--no-merge', action='store_true')
    parser.add_argument('--no-window', action='store_true')
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    return run(args.policies, args.selector, args.episodes, args.shards, args.label, args.width,
               args.seed, not args.no_resume, not args.no_merge, not args.no_window)


if __name__ == '__main__':
    sys.exit(main())
