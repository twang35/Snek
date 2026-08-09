"""Decoupled live chart viewer — shows a grid of policy chart PNGs and refreshes.

Runs as its **own process**, entirely separate from training: it only *reads* the
PNGs training writes every eval (`runs/<policy>.png`). So if the viewer dies — an X
error, or the OS OOM-kills it — training is untouched; and when training stops, the
viewer notices via `--watch` and exits on its own. This is the safe replacement for
the in-process cv2 window whose fatal XIO error took down all four desktop arms at
once on 2026-08-09.

Memory is trivial (matplotlib plus a few small PNGs, ~100 MB) regardless of how many
arms there are, so it can neither cause nor be worsened by an OOM.

On the laptop a training launch starts one of these itself — `snek2.main()` calls
`spawn_for_policy()`, so an arm shows its chart without anyone remembering to open a
window. See that function for the gate and the dedupe.

Usage:
  python chart_viewer.py runs/b20e-*.png runs/b20f-*.png ...   # explicit files, or
  python chart_viewer.py --glob 'runs/b20*.png'                # a glob, re-read each refresh
    [--watch 'snek2.py b20']   pgrep -f pattern; exit once nothing matches (training stopped)
    [--interval 1]             refresh seconds
    [--scale 2]                window size multiplier
    [--title 'snek training']  window title
"""
import argparse
import glob as globmod
import os
import re
import subprocess
import sys
import tempfile
import time


def _training_alive(pattern):
    """True while any process *other than this viewer* matches `pattern` (pgrep -f).

    The watched pattern is itself part of the viewer's own command line -- it is an
    argument -- so a naive pgrep matches the viewer and it would never exit. Drop our
    own pid and any chart_viewer.py process from the match. Uses `ps` for the cmdline
    check so it works on macOS (no /proc) as well as Linux. If we cannot tell, keep
    showing rather than exit prematurely."""
    if not pattern:
        return True
    try:
        res = subprocess.run(['pgrep', '-f', pattern],
                             stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        mine = str(os.getpid())
        pids = [p for p in res.stdout.decode().split() if p and p != mine]
        if not pids:
            return False
        args = ['ps', '-o', 'pid=,command=']
        for p in pids:
            args += ['-p', p]
        ps = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        for line in ps.stdout.decode('utf-8', 'replace').splitlines():
            if line.strip() and 'chart_viewer.py' not in line:
                return True   # a real watched (training/eval) process
        return False
    except Exception:
        return True


BATCH_RE = re.compile(r'^(b\d+)[a-z]?-')


def batch_prefix(policy_name):
    """The batch an arm belongs to: `b20a-fc50seed1` -> `b20`.

    One window per *batch*, not per arm, because a wave launches four arms within
    seconds of each other and four windows would be unreadable. A name that is not a
    `b<n><letter>-` arm (`train`, `eval`) is its own prefix and gets a window of its own."""
    m = BATCH_RE.match(policy_name)
    return m.group(1) if m else policy_name


def viewer_enabled():
    """On by default on the laptop (darwin), off elsewhere; `SNEK_CHART_VIEWER` overrides.

    Off elsewhere because the desktop's runner daemon owns the viewer there
    (`desktop/runner/runner.py::_ensure_viewer`) — it has to inject the graphical
    session's DISPLAY/XAUTHORITY, which a trainer launched by systemd does not have, so a
    trainer-side launch would fail there anyway."""
    default = '1' if sys.platform == 'darwin' else '0'
    return os.environ.get('SNEK_CHART_VIEWER', default) not in ('0', '', 'false', 'False')


def viewer_running_for(glob_pattern):
    """True if some viewer process is already showing `glob_pattern`.

    Matches on the glob rather than on `chart_viewer.py` alone, so a second batch on the
    same host still gets its own window while the first one's stays up. If the check
    itself fails we answer True: skipping a window is recoverable by hand, whereas
    stacking one per arm on the display is what this dedupe exists to prevent."""
    try:
        res = subprocess.run(['pgrep', '-f', 'chart_viewer.py'],
                             stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        pids = [p for p in res.stdout.decode().split() if p and p != str(os.getpid())]
        if not pids:
            return False
        args = ['ps', '-o', 'command=']
        for p in pids:
            args += ['-p', p]
        ps = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        return any(glob_pattern in line
                   for line in ps.stdout.decode('utf-8', 'replace').splitlines())
    except Exception:
        return True


def spawn_for_policy(policy_name):
    """Best-effort live chart window for a training launch. Returns the Popen, or None.

    A *separate* process in its own session — never a child of training in any way that
    matters — so it cannot take an arm down, which is the whole reason the in-process cv2
    window was removed. Every failure path is swallowed with a note on stdout: a chart is
    never worth a training run. Smoke runs are skipped, since they are verification, not
    an arm anyone watches."""
    if not viewer_enabled() or policy_name.startswith('smoke'):
        return None
    prefix = batch_prefix(policy_name)
    pattern = 'runs/{0}*.png'.format(prefix)
    if viewer_running_for(pattern):
        return None
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        # Not under runs/ or evals/: both are protected from deletion, and a log is
        # throwaway. tempdir keeps it findable without adding clutter to either.
        log_path = os.path.join(tempfile.gettempdir(),
                                'snek_chart_viewer_{0}.log'.format(prefix))
        log = open(log_path, 'ab')
        argv = [sys.executable, '-u', os.path.join(here, 'chart_viewer.py'),
                '--glob', pattern,
                '--watch', 'snek2.py {0}'.format(prefix),
                '--title', 'snek {0} — live'.format(prefix)]
        proc = subprocess.Popen(argv, cwd=here, stdout=log, stderr=log,
                                start_new_session=True, close_fds=True)
        print('chart viewer: pid {0} watching {1} (log {2}; SNEK_CHART_VIEWER=0 to disable)'
              .format(proc.pid, pattern, log_path))
        return proc
    except Exception as error:
        print('chart viewer launch failed ({0}: {1}) — training continues'
              .format(type(error).__name__, error))
        return None


def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument('files', nargs='*', help='PNG files to show (or use --glob)')
    ap.add_argument('--glob', default=None, help='glob for PNGs, re-evaluated each refresh')
    ap.add_argument('--watch', default=None, help='pgrep -f pattern; exit when no match remains')
    # 1s refresh and 2x size are the laptop defaults: a chart is rewritten once per eval,
    # so re-reading every second costs a stat plus a small PNG decode and shows a new point
    # as soon as it exists. The desktop runner passes its own --interval/--scale.
    ap.add_argument('--interval', type=float, default=1.0)
    ap.add_argument('--scale', type=float, default=2.0, help='multiply the window size')
    ap.add_argument('--title', default='snek charts')
    return ap


def main():
    args = build_parser().parse_args()

    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    plt.ion()

    fig = None
    axes = []
    absent_checks = 0
    # Startup grace: training (and its first PNG) may not exist for a few refreshes,
    # so require the watch pattern to be absent on several consecutive checks first.
    GRACE = 6

    while True:
        files = sorted(args.files) if args.files else (
            sorted(globmod.glob(args.glob)) if args.glob else [])
        n = max(1, len(files))
        cols = 2 if n > 1 else 1
        rows = (n + cols - 1) // cols
        if fig is None or len(axes) != rows * cols:
            if fig is not None:
                plt.close(fig)
            fig, grid = plt.subplots(rows, cols, squeeze=False,
                                     figsize=(cols * 4.2 * args.scale, rows * 3.0 * args.scale))
            try:
                fig.canvas.manager.set_window_title(args.title)
            except Exception:
                pass
            axes = [ax for row in grid for ax in row]
        for ax in axes:
            ax.clear()
            ax.axis('off')
        for i, path in enumerate(files):
            try:
                axes[i].imshow(mpimg.imread(path))
                axes[i].set_title(os.path.basename(path).replace('.png', ''), fontsize=8)
            except Exception:
                axes[i].set_title(os.path.basename(path) + ' (waiting…)', fontsize=8)
        try:
            fig.tight_layout()
            fig.canvas.draw_idle()
            plt.pause(0.2)
        except Exception as error:
            # A draw failure never matters to anything but this window.
            sys.stderr.write('viewer draw error ({0}: {1})\n'.format(type(error).__name__, error))

        if args.watch:
            if _training_alive(args.watch):
                absent_checks = 0
            else:
                absent_checks += 1
                if absent_checks >= GRACE:
                    break
        for _ in range(int(max(1, args.interval))):
            time.sleep(1)

    print('chart_viewer: watched training gone, exiting')


if __name__ == '__main__':
    main()
