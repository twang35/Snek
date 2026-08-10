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
  python chart_viewer.py --arms b20            # the b20 arms that are actually running
  python chart_viewer.py runs/b20e-*.png ...   # explicit files, or
  python chart_viewer.py --glob 'runs/b20*.png'   # a glob, re-read each refresh
    [--watch 'snek2.py b20']   pgrep -f pattern; exit once nothing matches (training stopped)
    [--interval 1]             refresh seconds
    [--scale 2]                window size multiplier
    [--title 'snek training']  window title
"""
import argparse
import glob as globmod
import os
import re
import signal
import subprocess
import sys
import tempfile
import time


def _matching_commands(pattern):
    """Command lines of processes matching `pattern` (pgrep -f), excluding viewers.

    Our own pid and every other `chart_viewer.py` are dropped, because the watched pattern
    is an *argument* on the viewer's own command line -- a naive pgrep matches the viewer
    itself. Uses `ps` for the cmdline text so it works on macOS (no /proc) as well as Linux.
    Raises on failure; callers decide what an unanswerable check means, which differs
    between "is training alive" (keep showing) and "which arms are live" (show none new).
    """
    res = subprocess.run(['pgrep', '-f', pattern],
                         stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    mine = str(os.getpid())
    pids = [p for p in res.stdout.decode().split() if p and p != mine]
    if not pids:
        return []
    args = ['ps', '-o', 'pid=,command=']
    for p in pids:
        args += ['-p', p]
    ps = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    return [line for line in ps.stdout.decode('utf-8', 'replace').splitlines()
            if line.strip() and 'chart_viewer.py' not in line]


def live_arms(prefix):
    """Policy names of running trainers belonging to batch `prefix`.

    This is what lets a window show *the arms that were launched* rather than every chart
    the batch has ever produced. Globbing `runs/b20*.png` was the alternative and it ages
    badly: by wave 2 that glob matched eight finished arms plus four live ones, which at 2x
    scale is a window taller than the screen.

    Membership is `batch_prefix(policy) == prefix`, not `startswith` -- `'b20a'.startswith('b2')`
    is True, so a prefix test would quietly merge b2 into b20.

    A name is only taken from something that is actually **running python**: the line must
    contain a `python` token before the script, and the script token must be `snek2.py` or
    `.../snek2.py`. Merely mentioning the file is not enough, because plenty of commands do --
    `git ls-files snek2/snek2.py b20i-fc200x50seed1` would otherwise donate its next argument
    as an arm name, and Airbnb git telemetry fires curl processes whose payload lists repo
    paths. That class of false positive has already misread trainer counts twice here.
    """
    arms = []
    try:
        for line in _matching_commands('snek2.py'):
            parts = line.split()
            for i, token in enumerate(parts):
                if not (token == 'snek2.py' or token.endswith('/snek2.py')):
                    continue
                if not any('python' in earlier for earlier in parts[:i]):
                    break       # not a python invocation — this line names no arm
                if i + 1 < len(parts):
                    policy = parts[i + 1]
                    if batch_prefix(policy) == prefix and policy not in arms:
                        arms.append(policy)
                    break
    except Exception:
        return []       # unknown means "nothing new to add", never "drop what we have"
    return sorted(arms)


def wave_files(prefix, known):
    """PNG paths for batch `prefix`'s arms, adding any newly-seen live arm to `known`.

    Sticky on purpose: `known` is mutated and never pruned, so an arm that reaches its step
    cap keeps its panel while its siblings run. That is the point of watching a wave — the
    finished curve is what the others get compared against — and it also means a transient
    `ps` failure cannot blank the window.
    """
    for policy in live_arms(prefix):
        if policy not in known:
            known.append(policy)
    return [os.path.join('runs', policy + '.png') for policy in sorted(known)]


def policy_from_png(path):
    """The policy a chart PNG belongs to: `runs/b20q-fc25.png` -> `b20q-fc25`, and
    `evals/b20q-fc25_eval_progress.png` -> `b20q-fc25`. The two layouts are the trainer's
    chart and the eval's chart; both encode the policy as the stem so a panel can be tied
    back to the process that produces it."""
    base = os.path.basename(path)
    if base.endswith('_eval_progress.png'):
        return base[:-len('_eval_progress.png')]
    if base.endswith('.png'):
        return base[:-len('.png')]
    return base


def running_policies():
    """Policy names that currently have a live trainer OR eval process, or None if unknown.

    This is what lets a panel be tagged `(completed)`: the viewer keeps a wave's finished
    charts on screen (sticky `wave_files`, and the desktop runner unions the set too), so it
    needs a live signal for which of those arms are still going. A policy counts as running
    when its name appears as a whitespace token on the command line of a `snek2.py` (trainer)
    or `eval_checkpoints.py` (eval) process -- both put the policy right after the script.

    Token match, never substring, so `...seed1` is not read as running because `...seed11`
    is; policy names here are long and unique enough that a token collision does not occur.
    Returns None if the process list cannot be read, and the caller then marks nothing
    completed -- a false `(completed)` on a live arm is worse than a missing one on a dead
    arm, which the panel's frozen curve already shows."""
    tokens = set()
    for pattern in ('snek2.py', 'eval_checkpoints.py'):
        try:
            for line in _matching_commands(pattern):
                tokens.update(line.split())
        except Exception:
            return None
    return tokens


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
        return bool(_matching_commands(pattern))
    except Exception:
        return True


BATCH_RE = re.compile(r'^(b\d+)[a-z]*-')


def batch_prefix(policy_name):
    """The batch an arm belongs to: `b20a-fc50seed1` -> `b20`.

    One window per *batch*, not per arm, because a wave launches four arms within
    seconds of each other and four windows would be unreadable. A name that is not a
    `b<n><letters>-` arm (`train`, `eval`) is its own prefix and gets a window of its own.

    The arm suffix is `[a-z]*`, not a single letter: a batch with more than 26 arms rolls
    into double letters (`b20aa-`, `b20ab-`, ...), and those must still group under `b20`,
    not be read as their own batch. Batch 20 does exactly this — nine shapes at four seeds is
    36 arms."""
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


# Where the per-batch claim lock lives. A module constant rather than an inline
# `tempfile.gettempdir()` so tests can point it somewhere private: they exercise real
# prefixes like `b20`, and a test that wrote the real lock could suppress or steal the
# window of a wave that is actually training.
LOCK_DIR = tempfile.gettempdir()


def lock_path(prefix):
    return os.path.join(LOCK_DIR, 'snek_chart_viewer_{0}.lock'.format(prefix))


def claim_viewer_slot(prefix):
    """Atomically claim the right to open batch `prefix`'s window. True if we won it.

    **A check-then-spawn dedupe is not enough, and this shipped broken.** A wave's four
    trainers launch in the same second, so all four ran `viewer_running_for` before any of
    them had spawned anything, all four saw nothing, and all four opened a window. `O_EXCL`
    is the fix because the create and the test are one operation.

    A lock left behind by a dead holder is taken over rather than honoured, so nothing
    suppresses the window forever. The pid written here is only a placeholder — the caller
    replaces it with the *viewer's* pid via `hold_viewer_slot` once the spawn succeeds, and
    calls `release_viewer_slot` if it fails. That ordering matters: the question the lock
    answers is "is a viewer up for this batch", so pointing it at a trainer would keep the
    claim alive after the viewer had been killed, and drop it when the trainer merely
    finished."""
    lock = lock_path(prefix)
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        return True
    except FileExistsError:
        pass
    except Exception:
        return True     # cannot lock (odd tmpdir) -> fall back to the pgrep check alone
    try:
        with open(lock) as fh:
            holder = int(fh.read().strip() or 0)
    except Exception:
        holder = 0
    if holder:
        try:
            os.kill(holder, 0)      # signal 0 only tests existence
            return False            # a live claimant owns this batch's window
        except OSError:
            pass
    try:                            # stale: the claimant is gone, so take it over
        with open(lock, 'w') as fh:
            fh.write(str(os.getpid()))
        return True
    except Exception:
        return False


def hold_viewer_slot(prefix, pid):
    """Point the claim at the viewer we just started, so liveness tracks the window."""
    try:
        with open(lock_path(prefix), 'w') as fh:
            fh.write(str(pid))
    except Exception:
        pass        # the claim still holds under our own pid; not worth failing over


def release_viewer_slot(prefix):
    """Give the claim back, so another arm of the same wave can try after a failed spawn."""
    try:
        os.remove(lock_path(prefix))
    except Exception:
        pass


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
    pattern = '--arms {0}'.format(prefix)
    # Both checks are needed: pgrep catches a viewer started by hand or by an earlier wave
    # whose lock has since been cleaned out of tmp, and the lock closes the race pgrep loses.
    if viewer_running_for(pattern) or not claim_viewer_slot(prefix):
        return None
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        # Not under runs/ or evals/: both are protected from deletion, and a log is
        # throwaway. tempdir keeps it findable without adding clutter to either.
        log_path = os.path.join(tempfile.gettempdir(),
                                'snek_chart_viewer_{0}.log'.format(prefix))
        log = open(log_path, 'ab')
        argv = [sys.executable, '-u', os.path.join(here, 'chart_viewer.py'),
                '--arms', prefix,
                '--watch', 'snek2.py {0}'.format(prefix),
                '--title', 'snek {0} — live'.format(prefix)]
        proc = subprocess.Popen(argv, cwd=here, stdout=log, stderr=log,
                                start_new_session=True, close_fds=True)
        hold_viewer_slot(prefix, proc.pid)
        print('chart viewer: pid {0} showing the live {1} arms '
              '(log {2}; SNEK_CHART_VIEWER=0 to disable)'.format(proc.pid, prefix, log_path))
        return proc
    except Exception as error:
        release_viewer_slot(prefix)
        print('chart viewer launch failed ({0}: {1}) — training continues'
              .format(type(error).__name__, error))
        return None


def exit_now(plt, code=0, exit_fn=None):
    """Close the Tk windows, then leave *without* running interpreter shutdown.

    Both halves matter. Tk delivers events for as long as its windows exist and each one
    can call back into Python; if the interpreter has already cleared its thread state,
    that callback aborts the process — `_Py_FatalError_TstateNULL` under `PythonCmd` inside
    `Tk_HandleEvent`, which macOS reports as "python quit unexpectedly" with a crash-report
    dialog. Seen twice on 2026-08-09 (pids 30193, 76128). Closing the figures first removes
    the windows while the interpreter is fully alive, and `os._exit` then skips the
    finalisation window in which a late event could arrive. It is race-dependent, so a
    clean exit on one run does not mean the ordering is safe.

    Skipping cleanup costs nothing here: the viewer owns no state, it only reads PNGs.
    """
    try:
        plt.close('all')
    except Exception:
        pass
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()   # os._exit does not flush, and the exit line matters
        except Exception:
            pass
    (exit_fn or os._exit)(code)


def install_signal_exit(plt):
    """Route SIGTERM/SIGINT through `exit_now`.

    **Call this after the first figure exists, and after every figure rebuild.** Tk installs
    its own OS-level SIGTERM handler when it creates its interpreter — which happens inside
    the first `subplots()`, not at import — and that overwrites ours. Tcl's handler runs
    `Tcl_Exit` straight off the signal trampoline: it finalises the Tcl thread, destroys the
    windows, fires their `<Destroy>` bindings, and those call back into Python with no thread
    state, which is the abort described in `exit_now`. Installing before the figure looks
    right and does nothing — measured, 5 of 5 kills still aborted."""
    def handle(_sig, _frame):
        exit_now(plt)
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, handle)
        except (ValueError, OSError):   # not the main thread — not worth failing over
            pass
    return handle


def make_figure(plt, rows, cols, scale, title):
    """Build the chart grid, then (re)install the signal handler — one operation, in that
    order, because separating them silently breaks the clean exit. See
    `install_signal_exit`: Tk overwrites the OS-level handler while creating this window,
    so any install that does not follow a `subplots()` call is dead code."""
    fig, grid = plt.subplots(rows, cols, squeeze=False,
                             figsize=(cols * 4.2 * scale, rows * 3.0 * scale))
    try:
        fig.canvas.manager.set_window_title(title)
    except Exception:
        pass
    install_signal_exit(plt)
    return fig, [ax for row in grid for ax in row]


def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument('files', nargs='*', help='PNG files to show (or use --glob)')
    ap.add_argument('--glob', default=None, help='glob for PNGs, re-evaluated each refresh')
    ap.add_argument('--watch', default=None, help='pgrep -f pattern; exit when no match remains')
    ap.add_argument('--arms', default=None, metavar='PREFIX',
                    help='show only the arms of batch PREFIX that are actually running '
                         '(re-checked each refresh, and remembered once seen)')
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
    fignum = None
    axes = []
    wave_arms = []
    absent_checks = 0
    # Startup grace: training (and its first PNG) may not exist for a few refreshes,
    # so require the watch pattern to be absent on several consecutive checks first.
    GRACE = 6

    while True:
        if args.arms:
            files = wave_files(args.arms, wave_arms)
        else:
            files = sorted(args.files) if args.files else (
                sorted(globmod.glob(args.glob)) if args.glob else [])
        n = max(1, len(files))
        cols = 2 if n > 1 else 1
        rows = (n + cols - 1) // cols
        if fig is None or len(axes) != rows * cols:
            if fig is not None:
                plt.close(fig)
            fig, axes = make_figure(plt, rows, cols, args.scale, args.title)
            fignum = fig.number
        for ax in axes:
            ax.clear()
            ax.axis('off')
        live = running_policies()
        for i, path in enumerate(files):
            label = policy_from_png(path)
            # A panel we still show but whose arm is gone is a finished run of the wave --
            # kept on screen as the reference the live ones are compared against, so it must
            # read as done rather than look like a stalled arm. `live is None` means the
            # process list was unreadable this refresh; tag nothing then.
            if live is not None and label and label not in live:
                label += ' (completed)'
            try:
                axes[i].imshow(mpimg.imread(path))
                axes[i].set_title(label, fontsize=8)
            except Exception:
                axes[i].set_title(label + ' (waiting…)', fontsize=8)
        try:
            fig.tight_layout()
            fig.canvas.draw_idle()
            plt.pause(0.2)
        except Exception as error:
            # A draw failure never matters to anything but this window.
            sys.stderr.write('viewer draw error ({0}: {1})\n'.format(type(error).__name__, error))

        # Closed by hand: without this the loop would spin on a destroyed figure, and
        # every redraw is another chance to abort in Tk. Closing the window means done.
        if fignum is not None and not plt.fignum_exists(fignum):
            print('chart_viewer: window closed, exiting')
            exit_now(plt)

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
    exit_now(plt)


if __name__ == '__main__':
    main()
