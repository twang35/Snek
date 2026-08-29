"""Force the daemon to run a git cycle now, from one ssh call.

    ssh the-claw-den 'Snek/snek3/desktop/trigger'

**Why this exists.** The daemon's network half — one fetch of the bus branches, one status push —
runs every `git_seconds` (600 s), not every poll, so the box makes ~144 fetches and ~144 pushes a day
instead of ~2,880 of each. The cost is inbound latency: a batch committed to `ops` is not seen until
the next network cycle. This closes that without giving the traffic back, and it is the only reason
the interval can be as long as it is.

It drops a `trigger` file beside the ledger and then watches two things, so the **exit code says
which half of the round trip worked**:

| exit | means |
|---:|---|
| 0 | consumed **and** `status.json` republished — the cycle completed |
| 2 | consumed, but no new `status.json` inside the timeout: alive and mid-cycle (a dispatch can take a while), or the publish path is broken |
| 1 | never consumed — the daemon is not running the loop. The printed `systemctl is-active` says whether the unit is even up |

That ladder is the reason `CLAUDE.md` names `trigger` as step two of diagnosing the box: it
distinguishes "not polling" from "polling but not publishing", which a stale `status.json` alone
cannot.

**A pre-existing trigger file is a diagnostic, not an error.** It means an earlier trigger was never
consumed — the same evidence as exit 1 — so it is reported rather than silently overwritten.

Stdlib only, and it reads the same `host.env` the daemon does, so there is nothing to keep in sync.
"""

import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from runner import config as config_module
from runner import runner as runner_module

# Its own unit, not snek2's `snek-runner`. The two must never both be active — they would fight over
# the same worktrees and the same `--force-with-lease` — and a distinct name makes the handover an
# explicit disable/enable rather than an overwrite that leaves no way back.
UNIT_NAME = 'snek3-runner'

# Long enough to cover a cycle that lands mid-dispatch, short enough that whoever is waiting on the
# ssh call gets an answer. A cycle with nothing to do finishes in well under a second.
DEFAULT_TIMEOUT = 90


def status_mtime(host):
    """mtime of the `status.json` the daemon writes, or None.

    Read from the **worktree**, not from git, so it advances even when the push itself fails — which
    is the point: it proves the daemon ran a cycle. Whether the push landed is a separate question,
    and the laptop's own fetch is what answers that one.
    """
    try:
        return os.path.getmtime(os.path.join(host['STATUS_WORKTREE'], 'status.json'))
    except OSError:
        return None


def _wait_until(predicate, timeout, step=0.25):
    """True as soon as `predicate()` holds, False if `timeout` passes first."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(step)
    return False


def _unit_state():
    """`systemctl is-active <unit>`, or a note when systemctl is not usable here."""
    try:
        result = subprocess.run(['systemctl', 'is-active', UNIT_NAME], text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=10)
        return (result.stdout or '').strip() or 'unknown'
    except (OSError, subprocess.SubprocessError) as error:
        return 'could not ask systemctl: {0}'.format(error)


def _print_glance(host):
    """The `at_a_glance` block out of the freshly written `status.json`.

    So one ssh call both starts the cycle and reports what the box decided to do with it — otherwise
    the caller has to fetch `ops-status` afterwards and wait for the push that may not have happened.
    """
    try:
        with open(os.path.join(host['STATUS_WORKTREE'], 'status.json')) as handle:
            status = json.load(handle)
    except (OSError, ValueError) as error:
        print('(could not read status.json: {0})'.format(error))
        return
    print('at {0}  counts={1}'.format(status.get('iso'), status.get('counts')))
    glance = status.get('at_a_glance') or {}
    # {'running': [str], 'queued': [str], 'attention': [str]} — a hold notice arrives as the first
    # `queued` line, and anything needing a human arrives under `attention`.
    for key in ('running', 'queued', 'attention'):
        for line in glance.get(key) or []:
            print('  {0}: {1}'.format(key, line))


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    timeout = float(argv[0]) if argv else DEFAULT_TIMEOUT

    host_env = os.environ.get(
        'SNEK_RUNNER_HOST_ENV',
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     'config', 'host.env'))
    host = config_module.load_host_config(host_env)
    path = runner_module.trigger_path(host)

    if os.path.exists(path):
        print('note: a trigger from {0} was never consumed'.format(
            time.strftime('%H:%M:%S', time.localtime(os.path.getmtime(path)))))

    before = status_mtime(host)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as handle:
        handle.write(str(time.time()))
    print('trigger written: {0}'.format(path))

    if not _wait_until(lambda: not os.path.exists(path), timeout):
        print('NOT consumed in {0:.0f}s — the daemon is not polling. systemctl is-active {1}: '
              '{2}'.format(timeout, UNIT_NAME, _unit_state()))
        return 1
    print('consumed; the daemon is fetching and dispatching')

    if not _wait_until(lambda: status_mtime(host) not in (None, before), timeout):
        print('no new status.json in {0:.0f}s — cycle still running, or publish is broken'.format(
            timeout))
        return 2
    _print_glance(host)
    return 0


if __name__ == '__main__':
    sys.exit(main())
