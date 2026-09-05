"""Which trainings are running on this box, stated by the trainings themselves — and the scheduler's
own two files beside them.

One file per arm in `runs/.live/`, named for the policy and holding the trainer's pid. A trainer
writes it before its first step and removes it on the way out; `live()` reads the directory and drops
any entry whose pid is gone. The scheduler reads it to adopt an arm a killed predecessor left running
and to count a hand-launched trainer against the cap.

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

**The chart window no longer reads this directory** (2026-09-05). It follows `.status.json` here, which
the scheduler writes — see `tools/window.py` for why the owner of the window is the scheduler and not
the arms. The dot-files -- `.status.json`, `.reopen-window`, `.republish`, `.paused`, `.pass-<label>` -- are the scheduler's;
`live()` skips anything starting with a dot.

Best-effort throughout: this directory is a convenience, and **no failure here is worth a training
run**.
"""

import errno
import json
import os
import subprocess
import time

from env import constants

DIR_NAME = '.live'
# The scheduler's status, single-writer, and what the chart window follows.
STATUS_NAME = '.status.json'
# Dropped by `tools.scheduler --reopen-window`; the scheduler unlinks it and replaces its viewer.
REOPEN_NAME = '.reopen-window'
# Dropped by `tools.scheduler --republish`; the scheduler unlinks it and publishes its status at once
# instead of at its next event or ten-minute refresh (a moved batch shows up within a poll).
REPUBLISH_NAME = '.republish'
# While this exists the scheduler launches nothing new: the desktop daemon writes it for
# `runtime.json`'s `paused`/`drain`, a human touches it on the laptop. What is running finishes.
HOLD_NAME = '.paused'
# A pass the scheduler is running: `.pass-<label>` holding the close-out's pid, written by the scheduler
# (it holds the Popen; the close-out never learns its pass id). A dot-file, so `live()` never counts it
# as a trainer. What lets a restarted scheduler wait for a pass instead of launching it twice.
PASS_PREFIX = '.pass-'
# What the scheduler measured on this box: how long each finished pass took, the last `DURATIONS_KEEP`
# per pass kind. `tools/eta.py` reads it to estimate the passes queued; a box with no ledger yet
# estimates from defaults. Per box by construction, since it lives in the box's runs dir. (Arms need
# no ledger: their `arch.json` and `_evals.json` mtimes are their wall clock.)
DURATIONS_NAME = '.durations.json'
DURATIONS_KEEP = 16


def directory(runs_dir=None):
    """The registry directory: `.live` inside the runs directory the charts are written to.

    Beside the PNGs on purpose. The window needs both, they move together when a test redirects
    `RUNS_DIR`, and the registry is scratch in the same way a chart is regenerable.
    """
    return os.path.join(runs_dir or constants.RUNS_DIR, DIR_NAME)


def path_for(policy, runs_dir=None):
    return os.path.join(directory(runs_dir), str(policy))


def status_path(runs_dir=None):
    return os.path.join(directory(runs_dir), STATUS_NAME)


def reopen_path(runs_dir=None):
    return os.path.join(directory(runs_dir), REOPEN_NAME)


def republish_path(runs_dir=None):
    return os.path.join(directory(runs_dir), REPUBLISH_NAME)


def request_republish(runs_dir=None):
    """`python -m tools.scheduler --republish`: asks the running scheduler to publish its status now."""
    path = republish_path(runs_dir)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as handle:
        handle.write('{0}\n'.format(os.getpid()))
    return path


def take_republish(runs_dir=None):
    """Consumes a republish request if one is waiting. The unlink is the test, as for the reopen."""
    try:
        os.unlink(republish_path(runs_dir))
    except OSError:
        return False
    return True


def hold_path(runs_dir=None):
    return os.path.join(directory(runs_dir), HOLD_NAME)


def pass_entry(label):
    """The registry name for a running pass: `.pass-b17-stageb-w2`. Use with `register`/`read`/`path_for`."""
    return PASS_PREFIX + str(label)


def held(runs_dir=None):
    """Whether the box is paused: the hold marker exists."""
    return os.path.exists(hold_path(runs_dir))


def durations_path(runs_dir=None):
    return os.path.join(directory(runs_dir), DURATIONS_NAME)


def durations(runs_dir=None):
    """The ledger: `kind -> [entry]`, oldest first, each entry `{'seconds', 'ts', 'arms', 'label'}`.
    `{}` when there is none yet."""
    try:
        with open(durations_path(runs_dir)) as handle:
            loaded = json.load(handle)
    except (OSError, ValueError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def record_duration(kind, seconds, runs_dir=None, **fields):
    """Appends one measured duration under `kind`, keeping the last `DURATIONS_KEEP`. Returns the
    path, or None: a readout, never worth a run, so a failure to write is swallowed."""
    ledger = durations(runs_dir)
    entries = [entry for entry in (ledger.get(kind) or []) if isinstance(entry, dict)]
    entries.append(dict(fields, seconds=float(seconds), ts=time.time()))
    ledger[kind] = entries[-DURATIONS_KEEP:]
    path = durations_path(runs_dir)
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        staging = '{0}.{1}.partial'.format(path, os.getpid())
        with open(staging, 'w') as handle:
            json.dump(ledger, handle, indent=1)
        os.replace(staging, path)
    except OSError:
        return None
    return path


def alive(pid):
    """Whether `pid` exists. A pid belonging to another user counts as alive, because it is.

    `os.kill(pid, 0)` and not a `ps` scan: the pid came from the process itself, so there is no
    pattern to match and nothing to mis-match. The one thing it gets wrong is a **zombie**, which
    answers yes for as long as its parent has not waited on it — see `zombie()`.
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

    The eval-worker slots need this: a worker that exited under a trainer that has not reaped it yet
    answers `alive()` for minutes, and a slot held by a zombie would read as a live worker while
    nothing drained the queue.

    `ps -p` with a **single** pid, which is safe — the rule about `ps -p` rejecting a whole list on one
    bad pid is about lists. A `ps` that cannot answer returns False rather than True, so the failure
    mode is "treat it as live", never "treat a live one as dead".
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
    """Records `policy` as running under `pid` (default: this process). Returns the path, or None."""
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
