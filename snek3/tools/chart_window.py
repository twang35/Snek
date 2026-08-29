"""One chart window per box, showing every training running on it.

Every training ensures the window exists — on the laptop and on the desktop alike, with nobody
launching it by hand — and the first arm to start is the one that opens it. Later arms find the slot
taken and do nothing; their panels appear anyway, because the window reads the live-arm registry
(`tools/live_runs.py`) rather than a fixed list of files. So four arms give **one** window with four
panels, and an arm that starts an hour later joins that same window.

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

**snek2 owned this window from the trainer too**, and the reason snek3's is 1/5th the size is that
the pid registry answers the two questions its 500 lines of `pgrep` machinery could not answer
reliably — see `tools/live_runs.py`.
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

LOCK_NAME = '.window'


def lock_path(runs_dir=None):
    """Where the window's pid lives. Inside the registry directory, hidden from `live()`."""
    return os.path.join(live_runs.directory(runs_dir), LOCK_NAME)


def wanted(env=None):
    """Whether to open a window at all. `SNEK_CHART_WINDOW=0` says no.

    On by default, because the request was for one to appear without anyone asking. Off is for the
    cases where a window is wrong rather than unwanted: a test that runs the loop, a benchmark whose
    numbers a window would land in, a machine with no display.
    """
    env = os.environ if env is None else env
    return env.get('SNEK_CHART_WINDOW', '1') not in ('0', '', 'false', 'False', 'no')


def holder(runs_dir=None):
    """The pid of the window that is up, or None if there is none."""
    pid = live_runs.read(lock_path(runs_dir))
    if pid is None or not live_runs.alive(pid) or live_runs.zombie(pid):
        return None
    return pid


def hold(runs_dir=None, pid=None):
    """Records `pid` as the window, atomically. Returns whether the record is ours."""
    pid = os.getpid() if pid is None else int(pid)
    path = lock_path(runs_dir)
    temp = '{0}.{1}'.format(path, os.getpid())
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(temp, 'w') as handle:
            handle.write('{0}\n'.format(pid))
        os.replace(temp, path)
    except OSError:
        return False
    return live_runs.read(path) == pid


def claim(runs_dir=None):
    """Takes the window slot for this process, or returns False if a live window already holds it.

    `O_EXCL` settles the common case — four arms launched in the same second, one window — without a
    lock file protocol, because creating the file *is* the claim. The `FileExistsError` branch is the
    other case: a previous window that was closed or killed left its pid behind, and the slot has to
    be re-claimable or the box would never draw again.

    Two trainers taking over a *dead* holder within the same few microseconds can both win, and the
    cost of that is a second window that closes itself with the batch. Worth strictly less than a
    lock protocol that could fail closed and leave a running box with no window at all.
    """
    path = lock_path(runs_dir)
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        return False if holder(runs_dir) else hold(runs_dir)
    except OSError:
        return False
    with os.fdopen(descriptor, 'w') as handle:
        handle.write('{0}\n'.format(os.getpid()))
    return True


def command(runs_dir=None, scale=None, python=None):
    """The argv for the box's window. Pure, so the spelling is testable without a display."""
    scale = DEFAULT_SCALE if scale is None else scale
    return [python or sys.executable, '-u', '-m', VIEWER_MODULE,
            '--runs-dir', runs_dir or constants.RUNS_DIR,
            '--interval', str(REFRESH_SECONDS),
            '--scale', str(scale),
            '--title', 'snek3 — training']


def spawn(runs_dir=None, env=None):
    """Starts the window and returns its Popen, or None if it would not start.

    Failing to open a window is never an error for the caller. A box with no display, no Tk, or a
    `DISPLAY` pointing at a dead server all land here, and a training must not care — so this reports
    the reason on stderr and returns None.
    """
    env = os.environ if env is None else env
    scale = env.get('SNEK_CHART_WINDOW_SCALE')
    argv = command(runs_dir, float(scale) if scale else None)
    try:
        # `start_new_session` is the load-bearing argument. It puts the window in its own session and
        # process group, so a Ctrl-C or a `kill` to a trainer's group leaves it alone, and killing the
        # window cannot signal a trainer. stdout goes to the opener's log, which is where a traceback
        # from it is worth having; stdin is closed so it can never inherit a terminal.
        return subprocess.Popen(argv, cwd=ROOT, start_new_session=True,
                                stdin=subprocess.DEVNULL,
                                stdout=sys.stderr, stderr=subprocess.STDOUT)
    except OSError as error:
        sys.stderr.write('chart window did not open ({0}); training continues\n'.format(error))
        return None


def ensure(runs_dir=None, env=None):
    """Opens the box's window if it is wanted and not already up. Returns its Popen, or None.

    None is the ordinary answer, not a failure: it is what every arm but the first gets.
    """
    if not wanted(env):
        return None
    if holder(runs_dir) or not claim(runs_dir):
        return None
    process = spawn(runs_dir, env)
    if process is None:
        return None
    hold(runs_dir, process.pid)
    print('chart window: pid {0}, showing every training on this box. Killing it does not affect '
          'any run, and no run reopens it.'.format(process.pid), flush=True)
    return process


def reap(process):
    """Clears the zombie if the window has exited. Returns whether it is still up.

    Called on the opener's own cadence rather than by a handler: a trainer is a long-lived parent, so
    a window that exits and is never waited on stays a zombie for the rest of a 7-hour arm. One
    `poll()` is the whole fix, and it must not be a `wait()` — that would block the training on the
    window, which is the one thing this module exists to prevent.
    """
    if process is None:
        return False
    return process.poll() is None


def main(argv=None):
    """`python -m tools.chart_window` — the supported way to open one by hand.

    Takes the slot when it is free, and says who has it when it is not. This exists so that fixing a
    window never means guessing at the command a trainer would have run.
    """
    force = '--force' in (argv if argv is not None else sys.argv[1:])
    current = holder()
    if current and not force:
        print('a chart window is already up: pid {0}. --force opens another.'.format(current))
        return 0
    process = spawn()
    if process is None:
        return 1
    hold(None, process.pid)
    print('chart window: pid {0}'.format(process.pid))
    return 0


if __name__ == '__main__':
    sys.exit(main())
