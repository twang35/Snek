"""The box's one chart window, owned by the scheduler. Nothing else opens one.

    PYTHONPATH=. python -m tools.scheduler --reopen-window      # ask the running scheduler for a fresh one

**The process that knows what is running owns the window.** Until 2026-09-05 the window was opened by
the *work*: every `train.py` and every `tools/closeout.py` spawned a viewer, an `flock` in the viewer
picked the survivor, and every lifecycle question — is there one, what does it show, when does it
close — was answered by N processes that each knew only about themselves. Every gap between them grew
a rule (a pid watch, three negative checks, a stand-by loop, a zombie check), and the incidents behind
those rules are in `plans/scheduler.md` §0: five windows on the desktop, a chart-less window holding
the slot for 15 hours, a pass running an hour with no window because the previous pass's window was
still in its closing grace.

Here the scheduler spawns **one** viewer, holds its `Popen`, and "is the window up" is
`popen.poll() is None`. No lock, no slot file, no registry read, no pid to recognise as stale. The
viewer follows the scheduler's own `runs/.live/status.json` — its `panels` list — so what it shows is
whatever the scheduler says is running, and it exits on its own if the scheduler that started it is
gone (`chart_viewer.py`, `--follow`).

The rules, all of them:

- **opened on a launch.** The scheduler calls `open()` when it starts a wave or a pass. If the window
  is up, that is a no-op; if it was closed by hand, it comes back at the next launch and not before.
- **closed by the scheduler**, when it exits. Between waves of a batch it stays up showing the last wave.
- **reopened on request**: a `runs/.live/reopen-window` file (the daemon's own unlink-is-the-test
  pattern) makes the scheduler close its viewer and spawn a fresh one at its next poll.
- **a stale window is killed, not adopted.** On start the scheduler reads the pid its previous life
  wrote into `status.json` and kills it if it is still a `tools.chart_viewer` — one flicker per deploy,
  one code path.

The window is still **disposable and the training is not**: its own session, so a signal to either
cannot reach the other; nothing in a training loop reads from it or waits on it; killing it cannot
touch a run.
"""

import os
import subprocess
import sys
import time

from tools import live_runs

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIEWER_MODULE = 'tools.chart_viewer'
TITLE = 'snek3'

# Slower than the viewer's own 2 s default: the trainers rewrite their PNGs once per report and a
# shard every ~10 s, so redrawing faster than that is pure CPU on a box whose job is training.
REFRESH_SECONDS = 15

# `SNEK_CHART_WINDOW=0` opens no window; the size knobs are read here and nowhere else.
SWITCH_ENV = 'SNEK_CHART_WINDOW'
SCALE_ENV = 'SNEK_CHART_WINDOW_SCALE'
MAX_WIDTH_PX_ENV = 'SNEK_CHART_WINDOW_MAX_PX'
DEFAULT_SCALE = 1.0

# How long `close()` gives the viewer to leave on SIGTERM before SIGKILL.
CLOSE_GRACE_SECONDS = 5.0


def wanted(env=None):
    """Whether this box wants a window at all. Off is for a test, a benchmark, a box with no display."""
    env = os.environ if env is None else env
    return env.get(SWITCH_ENV, '1') not in ('0', '', 'false', 'False', 'no')


def sizing(env=None):
    """`(scale, max_width_px)` from the environment; either None means "no override".

    A value that is not a number is ignored rather than fatal: a window is disposable and a typo in an
    env var must not stop a batch.
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
    argv = ['--scale', str(DEFAULT_SCALE if scale is None else scale)]
    if max_width_px:
        argv += ['--max-width-px', str(max_width_px)]
    return argv


def command(status_path, env=None, python=None):
    """The viewer's argv. Pure, so the spelling is testable without a display."""
    scale, max_width_px = sizing(env)
    return [python or sys.executable, '-u', '-m', VIEWER_MODULE, '--follow', status_path,
            '--interval', str(REFRESH_SECONDS), '--title', TITLE] + size_argv(scale, max_width_px)


def is_viewer(pid, run=subprocess.run):
    """Whether `pid` is a live `tools.chart_viewer`. A known pid's own command line, never a scan."""
    try:
        result = run(['ps', '-o', 'command=', '-p', str(int(pid))],
                     stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    except (OSError, ValueError):
        return False
    if result.returncode != 0:
        return False
    return VIEWER_MODULE in result.stdout.decode('utf-8', 'replace')


class Window(object):
    """The scheduler's handle on the box's window. Every process call is injectable for tests."""

    def __init__(self, runs_dir=None, env=None, python=None, popen=subprocess.Popen,
                 kill=os.kill, is_viewer=is_viewer, log=None, sleep=time.sleep):
        self.runs_dir = runs_dir
        self.env = os.environ if env is None else env
        self.python = python or sys.executable
        self.popen, self.kill, self.is_viewer, self.sleep = popen, kill, is_viewer, sleep
        self.log = log or (lambda message: sys.stderr.write(message + '\n'))
        self.process = None

    @property
    def status_path(self):
        return live_runs.status_path(self.runs_dir)

    def up(self):
        return self.process is not None and self.process.poll() is None

    def pid(self):
        return self.process.pid if self.up() else None

    def open(self):
        """Spawns the viewer unless one is up or windows are off. Returns whether one was started."""
        if not wanted(self.env) or self.up():
            return False
        argv = command(self.status_path, self.env, self.python)
        try:
            # `start_new_session` is the load-bearing argument: the viewer shares no process group and
            # no terminal with the scheduler, so a Ctrl-C or a `kill` to either leaves the other alone.
            # stdout goes to the scheduler's log, where the viewer's own "pid N" line and any traceback
            # from it are worth having.
            self.process = self.popen(argv, cwd=ROOT, env=dict(self.env), start_new_session=True,
                                      stdin=subprocess.DEVNULL, stdout=sys.stderr,
                                      stderr=subprocess.STDOUT)
        except OSError as error:
            self.log('chart window did not open ({0}); the batch continues'.format(error))
            self.process = None
            return False
        self.log('chart window: pid {0}, following {1}'.format(self.process.pid, self.status_path))
        return True

    def close(self):
        """Ends the viewer if it is up. Never blocks for long, never raises."""
        if not self.up():
            self.process = None
            return
        try:
            self.process.terminate()
            self.process.wait(CLOSE_GRACE_SECONDS)
        except subprocess.TimeoutExpired:
            try:
                self.process.kill()
                self.process.wait(CLOSE_GRACE_SECONDS)
            except (OSError, subprocess.TimeoutExpired):
                pass
        except OSError:
            pass
        self.process = None

    def take_reopen(self):
        """Consumes a reopen request if one is waiting. The unlink is the test, as for the trigger."""
        try:
            os.unlink(live_runs.reopen_path(self.runs_dir))
        except OSError:
            return False
        return True

    def poll(self):
        """One tick: reaps an exited viewer and honours a reopen request. Returns whether it reopened."""
        if not self.up():
            self.process = None
        if not self.take_reopen():
            return False
        self.log('reopen requested; replacing the chart window')
        self.close()
        return self.open()

    def kill_stale(self, pid):
        """Kills the window a previous scheduler left, if `pid` is still a viewer. Returns whether it did.

        The pid comes from the previous life's `status.json`, written by the scheduler itself, and is
        checked against its own command line before anything is sent — so a pid recycled since then is
        left alone, and no pattern is ever matched against a process list.
        """
        if not pid or int(pid) == os.getpid() or not self.is_viewer(pid):
            return False
        try:
            self.kill(int(pid), 15)
        except OSError:
            return False
        self.log('killed the previous scheduler\'s chart window (pid {0})'.format(pid))
        return True


def request_reopen(runs_dir=None):
    """`python -m tools.scheduler --reopen-window`: asks the running scheduler for a fresh window."""
    path = live_runs.reopen_path(runs_dir)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as handle:
        handle.write('{0}\n'.format(os.getpid()))
    return path
