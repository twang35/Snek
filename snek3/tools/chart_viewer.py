"""A live grid of chart PNGs in one window. Reads them; never writes and never trains.

Its own process, so if it dies — a Tk error, an OOM kill — nothing it was watching notices. That
separation is the whole design: snek2's in-process cv2 window took down all four desktop arms at once
with one fatal XIO error.

    PYTHONPATH=. python -m tools.chart_viewer --runs-dir runs        # every training on this box
    PYTHONPATH=. python -m tools.chart_viewer --glob 'runs/b48*.png' --watch-pid 8123,8124,8125
    PYTHONPATH=. python -m tools.chart_viewer --glob 'runs/*_checkpoint_evals_*.png' --interval 5
    PYTHONPATH=. python -m tools.chart_viewer runs/b1*.png --scale 0.6   # a smaller window

The window fills the display it opens on. `--scale` is a fraction of that, not a number of inches —
see `fit_dims`.

**Every training asks for this, and there is one per box.** Asking is idempotent because *this*
process takes an exclusive `flock` on the slot at startup and the losers exit before they draw — see
`take_window_slot`, and `tools/chart_window.py` for the launcher side, which now owns no exclusion at
all. snek2 spent ~500 lines on that problem (a process registry, a grace period, an `O_EXCL` claim
lock, `pgrep` corroboration, zombie detection and a dedupe, every line a dated scar) and snek3's
first attempt spent ~60 and opened five windows on the desktop on 2026-08-29. The kernel does it in
one call that cannot be won twice and needs no cleanup.

**The normal mode is `--runs-dir`**, which shows the trainings registered in `runs/.live/`. Panels
appear as arms start and **stay for the rest of the wave** once they do, so a batch with one arm left
still shows all four; the window closes itself, panels and all, once the box has been idle for
`IDLE_CLOSE_SECONDS`. `tools/chart_window.py` is what starts it that way. Explicit paths and
`--glob` remain for looking at arbitrary charts.

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
import fcntl
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

from tools import live_runs

DEFAULT_INTERVAL = 2.0
DEFAULT_MAX_PANELS = 8
# A negative liveness answer has to repeat before it is believed. One is a race against a launcher
# that has not spawned its trainers yet, or a `ps` that lost.
NEGATIVE_CHECKS = 3
# How long the window stays up after the last training on the box finishes. It is not zero, because a
# batch's next wave starts seconds to minutes after the previous one drains and closing the window in
# that gap would mean reopening it — and it is not never, because a box that has stopped training
# should not be holding a window open on a chart that stopped moving.
IDLE_CLOSE_SECONDS = 300
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


def live_panels(runs_dir, prune=True):
    """`(arms, chart_paths)` for the trainings registered in `runs_dir`.

    Two values because they answer different questions and the difference matters: the **arms** are
    what decides whether the window should still be up, and the **paths** are what it draws. An arm
    that has just started is in the first and not yet in the second, which is why the exit condition
    is never "no panels" — that would close the window on a wave whose trainers were still importing
    torch.
    """
    arms = live_runs.live(runs_dir, prune)
    paths = []
    for policy, _ in arms:
        path = os.path.join(runs_dir, policy + '.png')
        if os.path.exists(path):
            paths.append(path)
    return arms, paths


def wave_panels(known, live_paths, new_wave):
    """The panels to show: the live arms **plus the ones that already finished**.

    Sticky on purpose. A batch is read as a batch — with one arm left of four, a glance should still
    show all four, because what the three finished ones did is most of the answer. So a panel is
    added when an arm appears and never removed while the wave lasts.

    `new_wave` is what stops that from accumulating forever: when the registry has been empty and an
    arm appears again, the previous wave's panels go. Without it a batch launched inside the idle
    grace would draw its predecessor's charts beside its own — which is how snek2 came to open a
    window with **eight panels for four arms**, by a different route (a 12 h TTL) but the same
    mistake of never deciding when one wave ends.

    The order is append order and does not matter: `panels` sorts by path, because ordering the panel
    list by anything that moves is what made the window flash.
    """
    known = [] if new_wave else list(known)
    for path in live_paths:
        if path not in known:
            known.append(path)
    return known


def idle_close(arms, seen_arms, empty_since, now):
    """`(seen_arms, empty_since, close)` for one refresh of a `--runs-dir` window.

    The state machine is three lines and one of them is the whole point: **`close` is never True
    before an arm has been seen.** A window opened by hand a minute before a batch launches, or one
    whose first trainer is still importing torch, has an empty registry for reasons that have nothing
    to do with the box being idle — and closing for that would make the window useless exactly when
    someone is watching for a run to start.
    """
    if arms:
        return True, None, False
    if not seen_arms:
        return False, None, False
    empty_since = now if empty_since is None else empty_since
    return True, empty_since, now - empty_since >= IDLE_CLOSE_SECONDS


def take_window_slot(runs_dir, slot=None):
    """Takes a window slot, or returns None if another window already holds it.

    `slot` names which one — the training window's by default, and the eval window passes its own.

    **This is the whole of the mutual exclusion, and the kernel is what enforces it.** An `flock` is
    held by one open file description at a time, is released by the kernel when its holder dies for
    any reason, and leaves nothing behind that a later window has to recognise as stale. So the three
    cases that broke the launcher-side claim protocol it replaced — a slot file created but not yet
    written, a slot file holding a dead pid, and two launchers taking over the same dead slot in the
    same microsecond — stop being cases at all.

    Measured on 2026-08-29, before this existed: eight arms launched together opened a mean of 6.6
    windows, and the desktop opened five. Eight racers against this open one, every time.

    The pid is written *after* the lock is won, so it is always a real window's pid — that is what
    `chart_window.holder` reads to skip a spawn it does not need. Losing is not a failure and neither
    is a filesystem that cannot lock: a box that cannot take the slot still gets its window, which is
    the failure direction to prefer for something disposable.
    """
    path = live_runs.window_lock_path(runs_dir, slot)
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o644)
    except OSError:
        return True                       # cannot lock, so draw rather than refuse
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        os.close(descriptor)
        return None
    except (AttributeError, NameError):   # pragma: no cover - no flock on this platform
        return True
    try:
        os.truncate(descriptor, 0)
        os.write(descriptor, '{0}\n'.format(os.getpid()).encode('utf-8'))
    except OSError:
        pass
    # The descriptor is deliberately never closed. The lock lives as long as it is open, which is as
    # long as this process, and `Viewer.exit_now` leaves via `os._exit` — so there is no close path to
    # get wrong and nothing to release by hand.
    return descriptor


def run(paths, glob_pattern=None, watch_pids=(), interval=DEFAULT_INTERVAL,
        max_panels=DEFAULT_MAX_PANELS, title='snek3 charts', scale=DEFAULT_SCALE, dpi=None,
        runs_dir=None, now=time.time, max_width_px=DEFAULT_MAX_WIDTH_PX):
    viewer = Viewer(title, interval, max_panels, scale, dpi, max_width_px)
    plt.ion()
    negatives = 0
    seen_arms = False
    empty_since = None
    known = []
    # Whether the registry was empty last time round, which is how a *new* wave is told from an arm
    # joining the one already running.
    was_idle = False
    while True:
        found = list(paths)
        if runs_dir is not None:
            arms, live_paths = live_panels(runs_dir)
            known = wave_panels(known, live_paths, new_wave=bool(arms) and was_idle)
            was_idle = not arms
            # The finished arms stay up, including through the idle grace before the window closes:
            # the last thing a batch shows is all of its arms, not none of them.
            found.extend(known)
            seen_arms, empty_since, close = idle_close(arms, seen_arms, empty_since, now())
            if close:
                print('no training running for {0:.0f}s; closing'.format(now() - empty_since))
                viewer.exit_now(0)

        if not viewer.refresh(panels(found, glob_pattern, max_panels)):
            if viewer.figure is None:
                print('nothing to show yet: {0}'.format(glob_pattern or runs_dir or paths))
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
    parser.add_argument('--runs-dir', default=None,
                        help='show the trainings registered in this runs directory: a panel appears '
                             'when an arm starts and stays for the rest of the wave, and the window '
                             'closes once none has been running for a while')
    parser.add_argument('--watch-pid', default='',
                        help='comma-separated pids; the window closes once none of them is alive')
    parser.add_argument('--interval', type=float, default=DEFAULT_INTERVAL)
    parser.add_argument('--max-panels', type=int, default=DEFAULT_MAX_PANELS)
    parser.add_argument('--title', default='snek3 charts')
    parser.add_argument('--slot', default=None,
                        help='claim this window slot, so a second one stands down rather than '
                             'opening beside it. Implied by --runs-dir')
    parser.add_argument('--scale', type=float, default=DEFAULT_SCALE,
                        help='fraction of the screen to fill; 1.0 is as large as it goes')
    parser.add_argument('--dpi', type=int, default=None,
                        help='figure render dpi; default 200 on darwin (Retina), 100 elsewhere')
    parser.add_argument('--max-width-px', type=int, default=DEFAULT_MAX_WIDTH_PX,
                        help='hard ceiling on window width in logical pixels, for either box. The '
                             'window is already capped at the source charts own width, so this is '
                             'for holding it smaller than that')
    args = parser.parse_args(argv)
    if not args.paths and not args.glob_pattern and args.runs_dir is None:
        parser.error('give PNG paths, --glob or --runs-dir')
    pids = [int(token) for token in args.watch_pid.split(',') if token.strip()]
    # **Only a window that claims a slot is a singleton**, and there are exactly two kinds: the
    # training window, which shows the box's live registry (`--runs-dir`), and a window that names a
    # slot outright (`--slot`, which is how the eval window gets one of its own). An agent looking at
    # arbitrary charts with `--glob` or a path list is neither, and must never be refused because a
    # window is up.
    slot = args.slot or (live_runs.WINDOW_LOCK_NAME if args.runs_dir is not None else None)
    if slot:
        if take_window_slot(args.runs_dir, slot) is None:
            # Every arm of a wave asks for the training window, so this is the ordinary outcome for
            # all but one of them and not a failure. Said out loud anyway: it lands in the log of the
            # arm whose spawn lost, where the alternative is a viewer that vanished with no
            # explanation.
            print('a window is already up on slot {0} (pid {1}); nothing to do'.format(
                slot, live_runs.read(live_runs.window_lock_path(args.runs_dir, slot))), flush=True)
            return 0
        # Printed here rather than by the launcher, because only the process holding the slot knows
        # that it is the window. It goes to the stdout of whoever spawned it, which is the log of the
        # arm that opened it.
        print('{0}: pid {1}. Killing it does not affect any run, and no run reopens '
              'it.'.format(args.title, os.getpid()), flush=True)
    run(args.paths, args.glob_pattern, pids, args.interval, args.max_panels, args.title,
        args.scale, args.dpi, args.runs_dir, max_width_px=args.max_width_px)
    return 0


if __name__ == '__main__':
    sys.exit(main())
