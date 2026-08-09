"""Decoupled live chart viewer — shows a grid of policy chart PNGs and refreshes.

Runs as its **own process**, entirely separate from training: it only *reads* the
PNGs training writes every eval (`runs/<policy>.png`). So if the viewer dies — an X
error, or the OS OOM-kills it — training is untouched; and when training stops, the
viewer notices via `--watch` and exits on its own. This is the safe replacement for
the in-process cv2 window whose fatal XIO error took down all four desktop arms at
once on 2026-08-09.

Memory is trivial (matplotlib plus a few small PNGs, ~100 MB) regardless of how many
arms there are, so it can neither cause nor be worsened by an OOM.

Usage:
  python chart_viewer.py runs/b20e-*.png runs/b20f-*.png ...   # explicit files, or
  python chart_viewer.py --glob 'runs/b20*.png'                # a glob, re-read each refresh
    [--watch 'snek2.py b20']   pgrep -f pattern; exit once nothing matches (training stopped)
    [--interval 10]            refresh seconds
    [--title 'snek training']  window title
"""
import argparse
import glob as globmod
import os
import subprocess
import sys
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('files', nargs='*', help='PNG files to show (or use --glob)')
    ap.add_argument('--glob', default=None, help='glob for PNGs, re-evaluated each refresh')
    ap.add_argument('--watch', default=None, help='pgrep -f pattern; exit when no match remains')
    ap.add_argument('--interval', type=float, default=10.0)
    ap.add_argument('--scale', type=float, default=1.0, help='multiply the window size')
    ap.add_argument('--title', default='snek charts')
    args = ap.parse_args()

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
