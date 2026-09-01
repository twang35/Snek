"""One chart window per box, showing every training running on it.

Every training asks for the window — on the laptop and on the desktop alike, with nobody launching it
by hand — and asking is safe to do from every arm at once, because **this module enforces nothing.**
`tools/chart_viewer.py` takes an exclusive `flock` on the slot as it starts and the losers exit before
they draw anything, so eight arms launched in the same second give **one** window with eight panels.
Their panels all appear because the window reads the live-arm registry (`tools/live_runs.py`) rather
than a fixed list of files, so an arm that starts an hour later joins that same window.

**The exclusion moved into the viewer on 2026-08-30, and the version before it was wrong.** This file
used to hold an `O_EXCL` claim with a "the slot's holder is dead, so take it over" fallback, and the
fallback had no exclusion of its own: it wrote a pid and read it back, which every racer can win. Two
ways in, neither rare — the slot file exists but is not yet written, so it reads as unheld; or it
holds a dead pid, which is the state at the start of every batch, since nothing deleted it. Measured
over 20 trials of 8 concurrent arms: a mean of **6.6 windows**, and the desktop opened 5 that day.
The lesson is not that the protocol needed a third case. It is that a lock the kernel keeps — one that
cannot be held twice, is released when its holder dies for any reason, and leaves no state behind to
be recognised as stale — retires all three cases instead of handling them.

What is left here is a launcher and an optimisation. `holder` is **advisory only**: it skips a spawn
that would obviously lose, and being wrong costs one 0.3 s process that exits on its own.

The contract is that **the window is disposable and the training is not**:

- it is a **separate process**, started with `start_new_session`, so it shares no process group and no
  controlling terminal with the trainer that opened it. A signal aimed at either cannot reach the
  other, and it outlives that trainer if other arms are still running
- if it dies — closed, killed, an X server that went away — no training notices. Nothing in the
  training loop reads from it or waits on it
- it is **never respawned mid-run**. An agent that kills it to fix something, or relaunches it with
  `python -m tools.chart_window`, is not fighting a trainer for control of the window
- the trainer never draws. It writes `runs/<policy>.png` as it always did; the window reads that
  file. The one-way dependency is what makes the three properties above true rather than hoped for

The window closes itself when the last arm finishes, so a finished batch does not leave one behind.
"""

import os
import subprocess
import sys

from env import constants
from tools import live_runs

# The project root, so `-m tools.chart_viewer` resolves whatever directory the trainer was started
# from. Taken from this file rather than `os.getcwd()`: a window is not something to get wrong
# because a caller had wandered.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

VIEWER_MODULE = 'tools.chart_viewer'

# Slower than the tool's own 2 s default: the trainers rewrite their PNGs once per report, and
# redrawing faster than the files change is pure CPU on a box whose job is training.
REFRESH_SECONDS = 15

# The whole screen, because there is one window for the box. `SNEK_CHART_WINDOW_SCALE` overrides it.
DEFAULT_SCALE = 1.0

# **The size knobs, and both windows on both boxes read them through `sizing` below.** There is one
# implementation because there is one behaviour wanted: the laptop and the desktop differ in the
# monitor they probe, not in how they decide. Duplicating the env read once per window is how the two
# drift apart, and it had already started — `chart_window` and `eval_window` each parsed the scale.
SCALE_ENV = 'SNEK_CHART_WINDOW_SCALE'
MAX_WIDTH_PX_ENV = 'SNEK_CHART_WINDOW_MAX_PX'


def sizing(env=None):
    """`(scale, max_width_px)` from the environment; either may be None, meaning "no override".

    Pure apart from the read, so both windows' argv spelling is testable without a display.
    A value that is not a number is ignored rather than fatal: a window is disposable and a typo in
    an env var must not stop a training from starting.
    """
    env = os.environ if env is None else env

    def number(name, cast):
        raw = env.get(name)
        if not raw:
            return None
        try:
            return cast(raw)
        except (TypeError, ValueError):
            sys.stderr.write('{0}={1!r} is not a number; ignoring it\n'.format(name, raw))
            return None

    return number(SCALE_ENV, float), number(MAX_WIDTH_PX_ENV, int)


def size_argv(scale=None, max_width_px=None):
    """The size flags both windows pass to the viewer, so neither can grow its own dialect."""
    argv = ['--scale', str(DEFAULT_SCALE if scale is None else scale)]
    if max_width_px:
        argv += ['--max-width-px', str(max_width_px)]
    return argv


def wanted(env=None):
    """Whether to open a window at all. `SNEK_CHART_WINDOW=0` says no.

    On by default, because the request was for one to appear without anyone asking. Off is for the
    cases where a window is wrong rather than unwanted: a test that runs the loop, a benchmark whose
    numbers a window would land in, a machine with no display.
    """
    env = os.environ if env is None else env
    return env.get('SNEK_CHART_WINDOW', '1') not in ('0', '', 'false', 'False', 'no')


def holder(runs_dir=None, slot=None):
    """The pid of the window that is up, or None if there is none. **Advisory.**

    `slot` names which window — this box's training window by default; `tools/eval_window.py` passes
    the eval window's slot and reads the answer the same way.

    A hint, not a gate. The viewer writes its own pid under the lock it holds, so a live pid here does
    mean a window is up — but the answer can be stale in the microseconds around a batch launch, and
    nothing may depend on it being right. It exists so the ordinary case (an arm joining a wave that
    already has a window, which is most arms) costs a file read instead of a process.

    The zombie check is what makes a *hand* relaunch work: a killed window is an unreaped child of the
    trainer that opened it, and `os.kill(pid, 0)` calls a zombie alive — so for as long as that
    trainer had not reaped it, `python -m tools.chart_window` reported a window that was visibly gone.
    """
    pid = live_runs.read(live_runs.window_lock_path(runs_dir, slot))
    if pid is None or not live_runs.alive(pid) or live_runs.zombie(pid):
        return None
    return pid


def command(runs_dir=None, scale=None, python=None, max_width_px=None):
    """The argv for the box's window. Pure, so the spelling is testable without a display."""
    return [python or sys.executable, '-u', '-m', VIEWER_MODULE,
            '--runs-dir', runs_dir or constants.RUNS_DIR,
            '--interval', str(REFRESH_SECONDS),
            '--title', 'snek3 — training'] + size_argv(scale, max_width_px)


def spawn(runs_dir=None, env=None, argv=None, label='chart window'):
    """Starts a window and returns its Popen, or None if it would not start.

    `argv` overrides the training window's own command, which is how the eval window reuses this:
    **the two windows differ only in what they are pointed at**, and everything below — the new
    session, the closed stdin, the swallowed `OSError` — is the part that has to be the same in both.

    Starting one is not the same as getting one: the child takes the box's slot itself and exits
    quietly if another window holds it, so this returns a live Popen for a process that may be about
    to become the window or may be about to stand down. The caller cannot tell and does not need to —
    it holds the handle only to reap it.

    Failing to open a window is never an error for the caller. A box with no display, no Tk, or a
    `DISPLAY` pointing at a dead server all land here, and a training must not care — so this reports
    the reason on stderr and returns None.
    """
    env = os.environ if env is None else env
    if argv is None:
        argv = command(runs_dir, *sizing(env))
    try:
        # `start_new_session` is the load-bearing argument. It puts the window in its own session and
        # process group, so a Ctrl-C or a `kill` to a trainer's group leaves it alone, and killing the
        # window cannot signal a trainer. stdout goes to the opener's log, which is where the window's
        # own "chart window: pid N" line and any traceback from it are worth having; stdin is closed
        # so it can never inherit a terminal.
        return subprocess.Popen(argv, cwd=ROOT, start_new_session=True,
                                stdin=subprocess.DEVNULL,
                                stdout=sys.stderr, stderr=subprocess.STDOUT)
    except OSError as error:
        sys.stderr.write('{0} did not open ({1}); the run continues\n'.format(label, error))
        return None


def ensure(runs_dir=None, env=None, argv=None, slot=None, label='chart window'):
    """Asks for the box's window. Returns the Popen of the process started, or None.

    Safe to call from every arm of a wave at once — that is the point of the design, and the reason
    this function has no locking in it. None means nothing was started: the window is off, or one is
    already up, or it would not start.
    """
    if not wanted(env):
        return None
    if holder(runs_dir, slot):
        return None
    return spawn(runs_dir, env, argv, label)


def reap(process):
    """Clears the zombie if the window has exited. Returns whether it is still up.

    Called on the opener's own cadence rather than by a handler: a trainer is a long-lived parent, so
    a window that exits and is never waited on stays a zombie for the rest of a 7-hour arm. That now
    covers the arms whose spawn *lost* the slot and exited in under a second, which is most of a wave.
    One `poll()` is the whole fix, and it must not be a `wait()` — that would block the training on
    the window, which is the one thing this module exists to prevent.
    """
    if process is None:
        return False
    return process.poll() is None


def main(argv=None):
    """`python -m tools.chart_window` — the supported way to open one by hand.

    This exists so that fixing a window never means guessing at the command a trainer would have run.
    There is no `--force`: a second window on the same registry is not something the viewer will
    agree to any more, so the way to replace one is to kill it, which costs nothing.
    """
    current = holder()
    if current:
        print('a chart window is already up: pid {0}'.format(current))
        return 0
    # The child says whether it got the slot, on this terminal. Nothing to add.
    return 0 if spawn() is not None else 1


if __name__ == '__main__':
    sys.exit(main())
