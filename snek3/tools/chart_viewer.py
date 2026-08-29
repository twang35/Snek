"""A live grid of chart PNGs in one window. Reads them; never writes and never trains.

Its own process, so if it dies — a Tk error, an OOM kill — nothing it was watching notices. That
separation is the whole design: snek2's in-process cv2 window took down all four desktop arms at once
with one fatal XIO error.

    PYTHONPATH=. python -m tools.chart_viewer runs/b48*.png
    PYTHONPATH=. python -m tools.chart_viewer --glob 'runs/b48*.png' --watch-pid 8123,8124,8125
    PYTHONPATH=. python -m tools.chart_viewer --glob 'runs/*_checkpoint_evals_*.png' --interval 5
    PYTHONPATH=. python -m tools.chart_viewer runs/b1*.png --scale 0.6   # a smaller window

The window fills the display it opens on. `--scale` is a fraction of that, not a number of inches —
see `fit_dims`.

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

# The window is sized from the **screen**, not from a fixed number of inches, and that is the whole
# fix for "the window is very small". A fixed size cannot be right on a monitor nobody measured: the
# first version asked for 6x3 inches a panel and opened 1201x650 on the desktop's 3840x2160 display,
# 31% of its width. snek2 tuned fixed caps for years — 14.0x8.0 to 18.2x13.0 inches over three
# revisions, each one a clipped bottom row — and even the last would reach only 47% here. Asking the
# display is one call and adapts to any of them.
SCREEN_WIDTH_FRACTION = 0.95
SCREEN_HEIGHT_FRACTION = 0.88        # leaves the title bar and the shell's top panel
# 1.0 fills that budget. Lower it for a window you want to put something beside.
DEFAULT_SCALE = 1.0
# Used only when Tk cannot be asked — a non-Tk backend, or a probe that raised. Deliberately sized
# for a laptop rather than a 4K panel: a window that is too small is a nuisance, while one taller
# than the display hides its bottom row and reads as missing charts.
FALLBACK_PANEL_WIDTH_IN = 6.5
# A fallback only. The real aspect is read from the image, because `imshow` preserves the image's own
# aspect inside its box, so any disagreement between the two becomes letterboxing — once per row,
# which is what puts a band between the panels and shrinks the charts inside them.
DEFAULT_PANEL_ASPECT = 730.0 / 450.0
# Darwin's built-in panels are 2x Retina while the conda Tk build reports scaling 1.0, so matplotlib
# renders at 1x and the compositor upsamples — soft charts however sharp the PNG. Rendering at 2x
# gives the figure the pixels the panel really has. The screen probe divides by the same dpi, so the
# window still comes out the same physical size.
HIDPI_DPI = 200


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
    unique = {os.path.abspath(path) for path in found if os.path.exists(path)}
    chosen = sorted(unique, key=lambda path: os.path.getmtime(path), reverse=True)[:max_panels]
    # **Returned in path order, and that is load-bearing rather than tidy.** mtime decides *which*
    # panels survive the cap; ordering the *output* by it permutes the list every time an arm writes
    # its PNG, and `refresh` rebuilds the figure whenever the panel set changes. Three arms writing
    # on their own clocks therefore closed and reopened the window every few seconds — a visible
    # flash per refresh on the desktop, taking any resize with it. Measured 2026-08-29.
    return sorted(chosen)


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


def viewer_dpi(platform, override=None):
    """The dpi to render at: 2x on darwin (see `HIDPI_DPI`), 100 elsewhere. `--dpi` wins.

    Pure, so the platform rule is testable without a display.
    """
    if override:
        return override
    return HIDPI_DPI if platform == 'darwin' else 100


def panel_aspect(path, fallback=DEFAULT_PANEL_ASPECT):
    """A PNG's width/height, or `fallback` when it cannot be read.

    Read rather than assumed because the two chart shapes here differ a lot — a training chart is
    1.62 and an eval chart 2.08 — and a panel box that disagrees with its image letterboxes it.
    """
    try:
        pixels = imageio.imread(path)
        height, width = pixels.shape[0], pixels.shape[1]
    except Exception:
        return fallback
    return float(width) / float(height) if height and width else fallback


def screen_inches(figure):
    """The display's size in inches, or None if Tk cannot be asked.

    Inches rather than pixels so the caller never has to think about dpi: this divides by the
    figure's own, which is what keeps a 2x darwin render the same physical size.
    """
    try:
        window = figure.canvas.manager.window
        dpi = float(figure.get_dpi())
        return (window.winfo_screenwidth() / dpi, window.winfo_screenheight() / dpi)
    except Exception:
        return None


def fit_dims(rows, columns, aspect, scale=DEFAULT_SCALE, screen=None):
    """Figure size in inches: the panel grid's own aspect, filled into the screen budget.

    **Grows as well as shrinks**, which is the difference from snek2's version and the reason the
    window is now the size of the monitor. A shrink-only fit cannot exceed the inches it was asked
    for, so the fixed request stayed the binding constraint on any large display.

    Uniform in both axes — the grid's aspect is preserved, so a panel never distorts. `screen=None`
    falls back to fixed inches.
    """
    grid = (columns * float(aspect)) / rows          # width / height of the whole grid
    if screen is None:
        width = columns * FALLBACK_PANEL_WIDTH_IN * scale
        return width, width / grid
    width = min(screen[0] * SCREEN_WIDTH_FRACTION * scale,
                screen[1] * SCREEN_HEIGHT_FRACTION * scale * grid)
    return width, width / grid


def apply_tight_grid(figure, gap=0.01):
    """Pushes the panels out to the figure edges with a hairline between them.

    Replaces `tight_layout`, which reserves room for the titles, labels and ticks that axes holding
    a bare `imshow` do not have — about 8% of the height on a 2x2 grid, all of it visible as a band
    between the rows, and all of it taken off the charts.
    """
    try:
        figure.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0, wspace=gap, hspace=gap)
    except Exception:
        pass


class Viewer(object):
    """One window, one figure, and one image artist per panel, reused across refreshes.

    The artists are created once and their data replaced, so a viewer left open for a day holds the
    same memory as one just started. Rebuilding the figure each refresh would also flicker and lose
    the window's size.
    """

    def __init__(self, title, interval, max_panels, scale=DEFAULT_SCALE, dpi=None):
        self.interval = interval
        self.max_panels = max_panels
        self.scale = scale
        self.dpi = dpi
        self.aspect = DEFAULT_PANEL_ASPECT
        self.figure = None
        self.axes = []
        self.images = {}
        self.shown = []
        self.mtimes = {}
        self.title = title

    def _build(self, count):
        """Makes the window, then resizes it to the display.

        Two steps because of a chicken and egg: the screen can only be read through a live Tk
        window, and there is no window until `subplots` has made one. So it opens at the fallback
        size and is resized before anything is drawn into it.
        """
        rows, columns = grid_shape(count)
        if self.figure is not None:
            plt.close(self.figure)
        dpi = viewer_dpi(sys.platform, self.dpi)
        self.figure, axes = plt.subplots(
            rows, columns, dpi=dpi,
            figsize=fit_dims(rows, columns, self.aspect, self.scale, None))
        self.axes = list(axes.flat) if hasattr(axes, 'flat') else [axes]
        for axis in self.axes:
            axis.set_axis_off()
        self.images = {}
        self.figure.canvas.manager.set_window_title(self.title)
        self.figure.set_size_inches(
            *fit_dims(rows, columns, self.aspect, self.scale, screen_inches(self.figure)))
        apply_tight_grid(self.figure)

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
        # **The panel SET, not the list.** The artists are keyed by path and stay on the axis they
        # were created on, so a reordering needs no rebuild — and rebuilding on one is exactly what
        # made the window flash. `panels` returns a stable order now, so this is the second guard on
        # the same bug rather than the fix, and it states the invariant the fix relies on.
        if set(paths) != set(self.shown) or self.figure is None:
            # The panel box matches the image, so its aspect has to be known before the figure
            # exists. One extra read per rebuild, and rebuilds are rare now.
            self.aspect = panel_aspect(paths[0], self.aspect)
            self._build(len(paths))
            self.shown = list(paths)
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
        max_panels=DEFAULT_MAX_PANELS, title='snek3 charts', scale=DEFAULT_SCALE, dpi=None):
    viewer = Viewer(title, interval, max_panels, scale, dpi)
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
    parser.add_argument('--scale', type=float, default=DEFAULT_SCALE,
                        help='fraction of the screen to fill; 1.0 is as large as it goes')
    parser.add_argument('--dpi', type=int, default=None,
                        help='figure render dpi; default 200 on darwin (Retina), 100 elsewhere')
    args = parser.parse_args(argv)
    if not args.paths and not args.glob_pattern:
        parser.error('give PNG paths or --glob')
    pids = [int(token) for token in args.watch_pid.split(',') if token.strip()]
    run(args.paths, args.glob_pattern, pids, args.interval, args.max_panels, args.title,
        args.scale, args.dpi)
    return 0


if __name__ == '__main__':
    sys.exit(main())
