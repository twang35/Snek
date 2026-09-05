"""A live grid of chart PNGs in one window. Reads them; never writes and never trains.

Its own process, so if it dies — a Tk error, an OOM kill — nothing it was watching notices. That
separation is the whole design: snek2's in-process cv2 window took down all four desktop arms at once
with one fatal XIO error.

    PYTHONPATH=. python -m tools.chart_viewer --follow runs/.live/.status.json   # what the scheduler opens
    PYTHONPATH=. python -m tools.chart_viewer --glob 'runs/*_checkpoint_evals_*.png' --interval 5
    PYTHONPATH=. python -m tools.chart_viewer runs/b1*.png --scale 0.6            # a smaller window

**`--follow` is the box's window, and the scheduler is the only thing that opens it** (2026-09-05).
The viewer re-reads that JSON file every refresh and draws its `panels` — the PNG paths the scheduler
says are running — so the panels change as the box moves from a training wave to a stage-B pass with
no restart and no rule of this file's own about what is live. It exits when the scheduler that
started it is gone (the parent pid changes), or when the scheduler ends it, or when the user closes
it. That replaces an `flock` slot, a live-arm registry, a pid watch with three negative checks and a
stand-by loop, each of which was a rule this process had to get right about processes it could not
see; `tools/window.py` says what went wrong with each.

Explicit paths and `--glob` remain for looking at arbitrary charts. Neither claims anything, so an
agent looking at old charts is never refused because a window is up.

The window fills the display it opens on. `--scale` is a fraction of that, not a number of inches —
see `fit_dims`.
"""

import argparse
import glob as globmod
import json
import os
import signal
import sys
import time

import imageio.v2 as imageio
import matplotlib

# Interactive by necessity — this module *is* the window. Everything that writes a PNG uses the
# Figure/FigureCanvasAgg object API instead, because pyplot's global manager keeps artists alive.
import matplotlib.pyplot as plt

DEFAULT_INTERVAL = 5.0
DEFAULT_MAX_PANELS = 8
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
# **A panel is never drawn wider than the PNG behind it.** The screen budget alone is a floor as well
# as a ceiling — it filled the display whatever was in it, so one eval chart (1000 px wide) opened a
# 2858 px panel on this laptop and a 3648 px one on the desktop, upscaling 2.9x and 3.6x. The charts
# went soft and a single arm took the whole screen to say very little. Capping at the source width
# is what makes the window *shrink* when there is little to show, which no fraction of the screen can
# express. Grid aspect is still preserved, so panels never distort.
NATURAL_WIDTH_CAP = True
# An absolute ceiling in logical pixels, for either box, on top of the two above.
# `SNEK_CHART_WINDOW_MAX_PX` sets it; None means only the screen and the charts bound the window.
DEFAULT_MAX_WIDTH_PX = None
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


def follow_panels(status_path):
    """The `panels` the scheduler's status file names, or None if the file cannot be read right now.

    None rather than an empty list, so a file caught mid-write (the scheduler replaces it atomically,
    but a reader on an unrelated clock can still miss) keeps the previous panels up instead of
    blanking the window for one refresh. A file that *says* no panels is an empty list.
    """
    try:
        with open(status_path) as handle:
            status = json.load(handle)
    except (OSError, ValueError):
        return None
    if not isinstance(status, dict):
        return None
    return [str(path) for path in (status.get('panels') or [])]


def parent_gone(original_ppid):
    """Whether the process that started this window has exited. The parent pid changes on reparenting."""
    return os.getppid() != original_ppid


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


def viewer_dpi(platform, override=None):
    """The dpi to render at: 2x on darwin (see `HIDPI_DPI`), 100 elsewhere. `--dpi` wins.

    Pure, so the platform rule is testable without a display.
    """
    if override:
        return override
    return HIDPI_DPI if platform == 'darwin' else 100


def panel_pixels(path):
    """A PNG's `(width, height)` in pixels, or None when it cannot be read.

    Both callers want the same read: the aspect keeps the panel box from letterboxing its image, and
    the width caps how far the window grows. Returning pixels rather than a ratio is what lets one
    `imread` answer both.
    """
    try:
        pixels = imageio.imread(path)
        height, width = pixels.shape[0], pixels.shape[1]
    except Exception:
        return None
    return (int(width), int(height)) if height and width else None


def panel_aspect(path, fallback=DEFAULT_PANEL_ASPECT):
    """A PNG's width/height, or `fallback` when it cannot be read.

    Read rather than assumed because the two chart shapes here differ a lot — a training chart is
    1.62 and an eval chart 2.08 — and a panel box that disagrees with its image letterboxes it.
    """
    size = panel_pixels(path)
    return float(size[0]) / float(size[1]) if size else fallback


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


def fit_dims(rows, columns, aspect, scale=DEFAULT_SCALE, screen=None,
             panel_width=None, max_width=None):
    """Figure size in inches: the panel grid's own aspect, under whichever cap binds first.

    **Grows as well as shrinks**, which is the difference from snek2's version and the reason the
    window is the size of the monitor. A shrink-only fit cannot exceed the inches it was asked for,
    so the fixed request stayed the binding constraint on any large display.

    Three caps, in the same units (inches at the figure's dpi), smallest wins:

    | cap | comes from | what it stops |
    |---|---|---|
    | the screen budget | the Tk probe | a window taller or wider than the display |
    | `panel_width` x columns | the source PNG's own width | upscaling a small chart to fill a big screen |
    | `max_width` | `SNEK_CHART_WINDOW_MAX_PX` | growth past a size the user chose, on either box |

    Uniform in both axes — the grid's aspect is preserved, so a panel never distorts. `screen=None`
    falls back to fixed inches, and that fallback is capped too: a failed probe is not a reason to
    upscale.
    """
    grid = (columns * float(aspect)) / rows          # width / height of the whole grid
    if screen is None:
        width = columns * FALLBACK_PANEL_WIDTH_IN * scale
    else:
        width = min(screen[0] * SCREEN_WIDTH_FRACTION * scale,
                    screen[1] * SCREEN_HEIGHT_FRACTION * scale * grid)
    if panel_width:
        width = min(width, columns * float(panel_width))
    if max_width:
        width = min(width, float(max_width))
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

    def __init__(self, title, interval, max_panels, scale=DEFAULT_SCALE, dpi=None,
                 max_width_px=DEFAULT_MAX_WIDTH_PX):
        self.interval = interval
        self.max_panels = max_panels
        self.scale = scale
        self.dpi = dpi
        self.max_width_px = max_width_px
        self.aspect = DEFAULT_PANEL_ASPECT
        # The source PNG's pixel width, so the panel is never drawn larger than its image. None
        # until the first refresh has read a chart; the caps that need it simply do not apply yet.
        self.panel_px = None
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
        # **Both caps are pixels and `fit_dims` works in inches, so they convert here** — dpi is not
        # known any earlier, and `screen_inches` divides by the same figure dpi, which is what keeps
        # a 2x darwin render the same physical size as a 1x one.
        panel_in = (self.panel_px / float(dpi)) if (self.panel_px and NATURAL_WIDTH_CAP) else None
        max_in = (self.max_width_px / float(dpi)) if self.max_width_px else None
        self.figure, axes = plt.subplots(
            rows, columns, dpi=dpi,
            figsize=fit_dims(rows, columns, self.aspect, self.scale, None,
                             panel_width=panel_in, max_width=max_in))
        self.axes = list(axes.flat) if hasattr(axes, 'flat') else [axes]
        for axis in self.axes:
            axis.set_axis_off()
        self.images = {}
        self.figure.canvas.manager.set_window_title(self.title)
        # No key handler at all: see `disable_keyboard_shortcuts`.
        handler_id = getattr(self.figure.canvas.manager, 'key_press_handler_id', None)
        if handler_id is not None:
            self.figure.canvas.mpl_disconnect(handler_id)
        self.figure.set_size_inches(
            *fit_dims(rows, columns, self.aspect, self.scale, screen_inches(self.figure),
                      panel_width=panel_in, max_width=max_in))
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
            # The panel box matches the image, so its aspect *and* its pixel width have to be
            # known before the figure exists. One read per rebuild answers both, and rebuilds are
            # rare now.
            size = panel_pixels(paths[0])
            if size:
                self.aspect = float(size[0]) / float(size[1])
                self.panel_px = size[0]
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


def disable_keyboard_shortcuts(rcparams=matplotlib.rcParams):
    """Makes the window ignore the keyboard.

    matplotlib binds a dozen bare keys and ctrl/cmd chords by default — `s` saves, `f` goes
    fullscreen, `q`/`cmd+w` quits, `l`/`k` flip the axes to log, `cmd+c` copies — and acts on them
    whenever this window has focus. The window shows PNGs and none of those do anything useful to it,
    while a keystroke meant for something else (a monitor-input switch, 2026-09-05) lands here
    instead. Every `keymap.*` entry is emptied, so the default handler matches nothing; `_build` also
    disconnects that handler, so the window has no key handler at all.
    """
    for name in list(rcparams):
        if name.startswith('keymap.'):
            rcparams[name] = []


def run(paths, glob_pattern=None, follow=None, interval=DEFAULT_INTERVAL,
        max_panels=DEFAULT_MAX_PANELS, title='snek3 charts', scale=DEFAULT_SCALE, dpi=None,
        max_width_px=DEFAULT_MAX_WIDTH_PX):
    disable_keyboard_shortcuts()
    viewer = Viewer(title, interval, max_panels, scale, dpi, max_width_px)
    plt.ion()
    parent = os.getppid()
    followed = []
    while True:
        if follow is not None:
            if parent_gone(parent):
                print('the scheduler that opened this window is gone; closing', flush=True)
                viewer.exit_now(0)
            fresh = follow_panels(follow)
            if fresh is not None:
                followed = fresh
        found = list(paths) + list(followed)

        if not viewer.refresh(panels(found, glob_pattern, max_panels)):
            if viewer.figure is None:
                print('nothing to show yet: {0}'.format(glob_pattern or follow or paths), flush=True)
                time.sleep(interval)
                continue

        if not plt.get_fignums():
            # The user closed the window. That is an instruction, not a failure.
            viewer.exit_now(0)
        viewer.sleep()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('paths', nargs='*', help='PNG paths to show')
    parser.add_argument('--glob', dest='glob_pattern', default=None,
                        help='pattern re-expanded every refresh, so new charts appear')
    parser.add_argument('--follow', default=None, metavar='STATUS_JSON',
                        help="draw the `panels` this file names, re-read every refresh; exit when the "
                             'process that opened this window is gone. What the scheduler passes')
    parser.add_argument('--interval', type=float, default=DEFAULT_INTERVAL)
    parser.add_argument('--max-panels', type=int, default=DEFAULT_MAX_PANELS)
    parser.add_argument('--title', default='snek3 charts')
    parser.add_argument('--scale', type=float, default=DEFAULT_SCALE,
                        help='fraction of the screen to fill; 1.0 is as large as it goes')
    parser.add_argument('--dpi', type=int, default=None,
                        help='figure render dpi; default 200 on darwin (Retina), 100 elsewhere')
    parser.add_argument('--max-width-px', type=int, default=DEFAULT_MAX_WIDTH_PX,
                        help='hard ceiling on window width in logical pixels, for either box. The '
                             'window is already capped at the source charts own width, so this is '
                             'for holding it smaller than that')
    args = parser.parse_args(argv)
    if not args.paths and not args.glob_pattern and args.follow is None:
        parser.error('give PNG paths, --glob or --follow')
    if args.follow is not None:
        print('{0}: pid {1}, following {2}. Killing it does not affect any run; the scheduler '
              'reopens it at its next launch, or now with `tools.scheduler --reopen-window`.'.format(
                  args.title, os.getpid(), args.follow), flush=True)
    run(args.paths, args.glob_pattern, args.follow, args.interval, args.max_panels, args.title,
        args.scale, args.dpi, max_width_px=args.max_width_px)
    return 0


if __name__ == '__main__':
    sys.exit(main())
