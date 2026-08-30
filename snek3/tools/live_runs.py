"""Which trainings are running on this box, stated by the trainings themselves.

One file per arm in `runs/.live/`, named for the policy and holding the trainer's pid. A trainer
writes it before its first step and removes it on the way out; `live()` reads the directory and drops
any entry whose pid is gone.

**A registry rather than a process scan**, and snek2 paid for both halves of that lesson. Its viewer
asked `ps` which arms were running, which loses a wave launched inside one second — an arm is not
visible to a scan until its `exec` lands, and Python is still importing at that point — so a window
opened on 3 of 4 arms with no repair path. The fix there was a registry *plus* a 120 s grace period
*plus* a liveness check, because its entries carried a timestamp and a name but no pid, so nothing in
the file could be asked whether it was still true; a 12 h TTL then let a relaunched batch draw eight
panels for four arms.

Recording the **pid** removes all of that. A trainer registers its own pid, so there is no window in
which the entry exists and the process does not, and no grace period to tune. `os.kill(pid, 0)` on a
pid we were handed cannot match the wrong thing the way a `pgrep` pattern can, and a dead entry is
self-evidently dead rather than merely old.

Best-effort throughout: this directory is a convenience for a window, and **no failure here is worth
a training run**.
"""

import errno
import os
import subprocess

from env import constants

DIR_NAME = '.live'
# The chart window's slot, beside the arms it draws. A dot, so `live()` never reads it as an arm.
WINDOW_LOCK_NAME = '.window'


def directory(runs_dir=None):
    """The registry directory: `.live` inside the runs directory the charts are written to.

    Beside the PNGs on purpose. The window needs both, they move together when a test redirects
    `RUNS_DIR`, and the registry is scratch in the same way a chart is regenerable.
    """
    return os.path.join(runs_dir or constants.RUNS_DIR, DIR_NAME)


def path_for(policy, runs_dir=None):
    return os.path.join(directory(runs_dir), str(policy))


def window_lock_path(runs_dir=None, name=None):
    """The file a window holds an `flock` on, and records its pid in.

    Here rather than in `chart_window` because the *viewer* is what holds the lock and the launcher
    is what reads the pid, so neither of them owns the path. Inside the registry directory, where
    `live()` already skips it for starting with a dot.

    `name` picks *which* window, and there are two: the training window on `WINDOW_LOCK_NAME`, and
    the eval window on its own slot. They are separate locks rather than one because a box can hold
    both at once — the wave barrier keeps trainings and evals apart on the desktop, but nothing does
    on the laptop, and a stage-B pass must never be refused a window because a training window is up.
    """
    return os.path.join(directory(runs_dir), name or WINDOW_LOCK_NAME)


def alive(pid):
    """Whether `pid` exists. A pid belonging to another user counts as alive, because it is.

    `os.kill(pid, 0)` and not a `ps` scan: the pid came from the process itself, so there is no
    pattern to match and nothing to mis-match. The one thing it gets wrong is a **zombie**, which
    answers yes for as long as its parent has not waited on it — see `zombie()`, which the window
    slot needs and an arm does not.
    """
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError as error:            # pragma: no cover - not reachable on darwin or linux
        return error.errno != errno.ESRCH
    return True


def zombie(pid):
    """Whether `pid` is an exited process nobody has waited on yet. Unanswerable counts as no.

    **The chart window needs this and it cost a wasted verification to find out.** Killing the window
    leaves a zombie child of the trainer that opened it, and a trainer only reaps on its report
    cadence — so for minutes afterwards `alive()` says the window is up, and a hand relaunch is
    refused with "a chart window is already up" for a window that is visibly gone.

    An arm's own entry does not need it: a trainer's parent is a shell or the daemon, neither of which
    leaves it unreaped for long, and a zombie trainer is finished either way.

    `ps -p` with a **single** pid, which is safe — the rule about `ps -p` rejecting a whole list on one
    bad pid is about lists. A `ps` that cannot answer returns False rather than True, so the failure
    mode is "refuse to open a second window", never "open one on top of a live one".
    """
    try:
        result = subprocess.run(['ps', '-o', 'stat=', '-p', str(int(pid))],
                                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    except OSError:
        return False
    if result.returncode != 0:
        return False
    return result.stdout.decode('utf-8', 'replace').strip().startswith('Z')


def register(policy, pid=None, runs_dir=None):
    """Records `policy` as running under `pid` (default: this process). Returns the path, or None.

    Called before the first step, so the window a launcher opens a moment later already has this
    arm's panel in it.
    """
    pid = os.getpid() if pid is None else pid
    path = path_for(policy, runs_dir)
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as handle:
            handle.write('{0}\n'.format(int(pid)))
    except OSError:
        return None
    return path


def unregister(policy, runs_dir=None):
    """Drops `policy`'s entry. Missing is success — `live()` prunes what a `kill -9` leaves."""
    try:
        os.remove(path_for(policy, runs_dir))
    except OSError:
        pass


def read(path):
    """The pid in one entry, or None if it is unreadable or not a pid."""
    try:
        with open(path) as handle:
            text = handle.read()
    except OSError:
        return None
    try:
        return int(text.split()[0])
    except (IndexError, ValueError):
        return None


def live(runs_dir=None, prune=True):
    """`[(policy, pid)]` for the trainings running here, sorted by policy.

    Entries whose process is gone are dropped, and by default deleted — `kill -9` and a crash both
    leave one behind, and the directory is scratch. Deletion failures are ignored: the entry is
    already excluded from the answer, which is the part that matters.
    """
    found = []
    try:
        names = sorted(os.listdir(directory(runs_dir)))
    except OSError:
        return found
    for name in names:
        if name.startswith('.'):
            continue
        path = os.path.join(directory(runs_dir), name)
        pid = read(path)
        if pid is not None and alive(pid):
            found.append((name, pid))
        elif prune:
            try:
                os.remove(path)
            except OSError:
                pass
    return found


def chart_paths(runs_dir=None, prune=True):
    """`runs/<policy>.png` for every live arm that has written its chart yet.

    An arm that has not reached its first report has no PNG, so it contributes no panel until it
    does. That is the only reason a live arm can be missing from the window, and it lasts one report
    interval.
    """
    paths = []
    for policy, _ in live(runs_dir, prune):
        path = os.path.join(runs_dir or constants.RUNS_DIR, policy + '.png')
        if os.path.exists(path):
            paths.append(path)
    return paths
