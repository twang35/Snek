"""A live grid of chart PNGs in one window. Reads them; never writes and never trains.

Its own process, so if it dies — a Tk error, an OOM kill — nothing it was watching notices. That
separation is the whole design: snek2's in-process cv2 window took down all four desktop arms at once
with one fatal XIO error.

    PYTHONPATH=. python -m tools.chart_viewer runs/b48*.png
    PYTHONPATH=. python -m tools.chart_viewer --glob 'runs/b48*.png' --watch-pid 8123,8124,8125
    PYTHONPATH=. python -m tools.chart_viewer --glob 'runs/*_checkpoint_evals_*.png' --interval 5

**The launcher opens this, with an explicit file list.** snek2's version was 1,010 lines and ~500 of
them existed because four peer trainers each tried to open one shared window while knowing nothing
about each other: a process registry, an `O_EXCL` claim lock, a grace period, zombie detection and a
dedupe, every line a dated scar. None of that is here, because the requirement is gone — one process
starts the arms, so one process starts the window.

`--watch-pid` is the same idea applied to the exit condition. snek2 asked `pgrep -f <pattern>`
whether training was still running, which is unsafe twice over: the invoking shell's own command line
contains the pattern, and `pgrep` exits 0 on a match, 1 on no match and **>=2 on an error** while all
three print nothing — so a *failed* check read as the strongest possible "nothing is running" and
closed a live window with five hours left. Here the launcher passes the pids it started. `ps -p` on a
known set needs no pattern, cannot match itself, and an error is distinguishable from an empty
answer; on top of that a negative answer must repeat `NEGATIVE_CHECKS` times before the window
closes.
"""

import argparse
import glob as globmod
import os
import signal
import subprocess
import sys
import time

import imageio.v2 as imageio
import matplotlib

# Interactive by necessity — this module *is* the window. Everything that writes a PNG uses the
# Figure/FigureCanvasAgg object API instead, because pyplot's global manager keeps artists alive.
import matplotlib.pyplot as plt

DEFAULT_INTERVAL = 2.0
DEFAULT_MAX_PANELS = 8
# A negative liveness answer has to repeat before it is believed. One is a race against a launcher
# that has not spawned its trainers yet, or a `ps` that lost.
NEGATIVE_CHECKS = 3
# Never `plt.pause`: it runs a nested event loop and re-enters the draw, which on the Tk backend
# deadlocks against a resize. `flush_events` plus a plain sleep does the same job.
FLUSH_SLICE = 0.05


def panels(paths, glob_pattern, max_panels):
    """The PNGs to show, newest first, capped.

    The glob is re-expanded every refresh so a chart that appears mid-run shows up without a
    restart. **The cap is hard**, and it is load-bearing: nothing in this project sweeps old chart
    PNGs, so a pattern that matched four arms last month matches forty now, and forty panels in one
    window is unreadable at any size. mtime picks which four, which is the same as "the ones being
    written".
    """
    found = list(paths)
    if glob_pattern:
        found.extend(globmod.glob(glob_pattern))
    unique = sorted({os.path.abspath(path) for path in found if os.path.exists(path)})
    unique.sort(key=lambda path: os.path.getmtime(path), reverse=True)
    return unique[:max_panels]


def grid_shape(count):
    """Rows and columns for `count` panels, wide rather than tall — charts are wide."""
    if count <= 1:
        return 1, 1
    if count <= 2:
        return 1, 2
    if count <= 4:
        return 2, 2
    if count <= 6:
        return 2, 3
    return 3, 3


def pids_alive(pids):
    """Whether any of `pids` is still running: True, False, or None for "could not tell".

    Three outcomes, not two. A `ps` that failed prints nothing on stdout, exactly like a `ps` that
    found nothing, and treating the first as the second is what closed a live window in snek2.

    **A full `ps -A` scan, not `ps -p <list>`.** macOS `ps -p` rejects the entire request if any one
    pid in the list is invalid — `ps -p 999991,<live pid>` exits 1 with no output — so a wave whose
    first shard had finished would have read as "all four are gone" and closed the window on the
    three still running. One scan of every pid has no such edge, and it costs the same.

    A zombie does not count as alive. `os.kill(pid, 0)` would have been shorter and would have said
    it was: an unreaped child answers signal 0 for as long as its parent ignores it.
    """
    if not pids:
        return None
    wanted = {str(int(pid)) for pid in pids}
    try:
        result = subprocess.run(['ps', '-Ao', 'pid=,stat='],
                                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    except OSError:
        return None
    if result.returncode != 0:
        return None
    for line in result.stdout.decode('utf-8', 'replace').splitlines():
        fields = line.split()
        if len(fields) >= 2 and fields[0] in wanted and not fields[1].startswith('Z'):
            return True
    return False


class Viewer(object):
    """One window, one figure, and one image artist per panel, reused across refreshes.

    The artists are created once and their data replaced, so a viewer left open for a day holds the
    same memory as one just started. Rebuilding the figure each refresh would also flicker and lose
    the window's size.
    """

    def __init__(self, title, interval, max_panels):
        self.interval = interval
        self.max_panels = max_panels
        self.figure = None
        self.axes = []
        self.images = {}
        self.shown = []
        self.mtimes = {}
        self.title = title

    def _build(self, count):
        rows, columns = grid_shape(count)
        if self.figure is not None:
            plt.close(self.figure)
        self.figure, axes = plt.subplots(rows, columns, figsize=(6.0 * columns, 3.0 * rows))
        self.axes = list(axes.flat) if hasattr(axes, 'flat') else [axes]
        for axis in self.axes:
            axis.set_axis_off()
        self.images = {}
        self.figure.canvas.manager.set_window_title(self.title)
        self.figure.tight_layout(pad=0.2)

        # Installed *after* subplots(), not before. Tk installs its own SIGTERM handler inside the
        # first subplots() call and overwrites anything set earlier — 5 of 5 kills still aborted the
        # process and popped a macOS crash dialog before this was moved down here.
        signal.signal(signal.SIGTERM, lambda *_: self.exit_now(0))
        signal.signal(signal.SIGINT, lambda *_: self.exit_now(0))

    def exit_now(self, status):
        """Closes the figures first, then leaves without unwinding.

        Interpreter shutdown with a live Tk window aborts the process and pops a macOS crash
        dialog, so the window has to go before the interpreter does — and `os._exit` skips the
        atexit handlers that would otherwise touch it again.
        """
        try:
            plt.close('all')
        except Exception:
            pass
        sys.stdout.flush()
        os._exit(status)

    def refresh(self, paths):
        """Redraws whatever changed. Returns False if there is nothing to show."""
        if not paths:
            return False
        if paths != self.shown or self.figure is None:
            self._build(len(paths))
            self.shown = paths
            self.mtimes = {}

        for index, path in enumerate(paths):
            mtime = os.path.getmtime(path)
            if self.mtimes.get(path) == mtime:
                continue
            try:
                pixels = imageio.imread(path)
            except Exception:
                # A PNG caught mid-write is a torn read, not an error worth exiting on. Writers here
                # use `.partial.png` + os.replace, so the next refresh sees a whole file.
                continue
            self.mtimes[path] = mtime
            axis = self.axes[index]
            if path in self.images:
                self.images[path].set_data(pixels)
            else:
                self.images[path] = axis.imshow(pixels)
                axis.set_axis_off()
        for axis in self.axes[len(paths):]:
            axis.set_axis_off()
        self.figure.canvas.draw_idle()
        return True

    def sleep(self):
        """Waits one interval while keeping the window responsive."""
        deadline = time.time() + self.interval
        while time.time() < deadline:
            self.figure.canvas.flush_events()
            time.sleep(FLUSH_SLICE)


def run(paths, glob_pattern=None, watch_pids=(), interval=DEFAULT_INTERVAL,
        max_panels=DEFAULT_MAX_PANELS, title='snek3 charts'):
    viewer = Viewer(title, interval, max_panels)
    plt.ion()
    negatives = 0
    while True:
        found = panels(paths, glob_pattern, max_panels)
        if not viewer.refresh(found):
            if viewer.figure is None:
                print('nothing to show yet: {0}'.format(glob_pattern or paths))
                time.sleep(interval)
                continue

        if watch_pids:
            alive = pids_alive(watch_pids)
            if alive is False:
                negatives += 1
                if negatives >= NEGATIVE_CHECKS:
                    print('watched pids gone after {0} checks; closing'.format(negatives))
                    viewer.exit_now(0)
            else:
                negatives = 0

        if not plt.get_fignums():
            # The user closed the window. That is an instruction, not a failure.
            viewer.exit_now(0)
        viewer.sleep()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('paths', nargs='*', help='PNG paths to show')
    parser.add_argument('--glob', dest='glob_pattern', default=None,
                        help='pattern re-expanded every refresh, so new charts appear')
    parser.add_argument('--watch-pid', default='',
                        help='comma-separated pids; the window closes once none of them is alive')
    parser.add_argument('--interval', type=float, default=DEFAULT_INTERVAL)
    parser.add_argument('--max-panels', type=int, default=DEFAULT_MAX_PANELS)
    parser.add_argument('--title', default='snek3 charts')
    args = parser.parse_args(argv)
    if not args.paths and not args.glob_pattern:
        parser.error('give PNG paths or --glob')
    pids = [int(token) for token in args.watch_pid.split(',') if token.strip()]
    run(args.paths, args.glob_pattern, pids, args.interval, args.max_panels, args.title)
    return 0


if __name__ == '__main__':
    sys.exit(main())
