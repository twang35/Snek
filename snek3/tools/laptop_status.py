"""Publish the laptop's queue to the `laptop-status` branch, so both boxes read at a glance.

The desktop daemon publishes `status.json` to `ops-status`; until 2026-09-05 the laptop's queue
(`tools.laptop_batch --queue`) was visible only in `logs/laptop-queue.log` on the laptop itself. Now
the driver publishes the same shape -- `at_a_glance.running` and `.queued`, built by the daemon's own
`build_at_a_glance` -- to a fourth bus branch, **`laptop-status`, written only by the laptop**, and the
daemon folds it into `ops-status` as `at_a_glance.laptop_running`, `.laptop_queued` and `.laptop_iso`
on every network cycle. One `git show origin/ops-status:status.json` then shows every task on both
boxes, from anywhere.

**One writer per branch, as the bus demands.** The laptop must not push into `ops-status`: two writers
on one branch is the merge-and-`force-with-lease` failure `desktop/runner/gitbus.py` is shaped to
avoid. So this is a branch of its own, published through gitbus's own `publish_status` with a laptop
host dict -- one implementation of "write and push a status branch" for both boxes.

**Published on events, not on a clock -- and the last publish of a driver is empty.** The driver
publishes when it launches a wave, when an arm exits, when a pass starts or ends, every ten minutes
while it waits (so the percentages move), and once more, with nothing running and nothing queued, as
it exits. So "empty" means the laptop is idle, and a `laptop_iso` hours old *with* a running line
means the driver died: the two read differently by design (user, 2026-09-04).

**The worktree is under `~/.snek3-laptop/status`**, outside the checkout, like the desktop's
`~/snek-bus/status`; `ensure_worktree` creates the branch (an empty root commit) and the worktree the
first time. A publish that fails never stops the driver -- it is logged and the next event retries,
and a local-only commit is pushed by the next successful publish's `--force-with-lease`.
"""

import os
import subprocess
import sys
import time

# `desktop/` shells out and imports nothing from this project; this direction -- a tool importing
# the daemon's pure helpers -- is the one the "one implementation for both boxes" rule wants, rather
# than a second copy of `build_at_a_glance` here. `desktop` is a namespace package (no `__init__.py`).
from desktop.runner import gitbus
from desktop.runner import runner as daemon

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(ROOT)
BRANCH = os.environ.get('SNEK_LAPTOP_STATUS_BRANCH', 'laptop-status')
WORKTREE = os.environ.get('SNEK_LAPTOP_STATUS_WORKTREE', os.path.expanduser('~/.snek3-laptop/status'))
REMOTE = 'origin'
REPUBLISH_SECONDS = 600     # the desktop's `git_seconds`: how often a waiting driver refreshes the percentages


def host(repo=REPO, worktree=WORKTREE, branch=BRANCH, remote=REMOTE):
    """The host dict `gitbus.publish_status` needs, for this box."""
    return {'REPO_PATH': repo, 'STATUS_WORKTREE': worktree, 'STATUS_BRANCH': branch,
            'GIT_REMOTE': remote}


def build(running, queued, now=None):
    """The laptop's `status.json`: the daemon's `at_a_glance` shape over the driver's job dicts.

    `running` and `queued` are lists of the job dicts `build_at_a_glance` reads -- `id`, `type`,
    `policy`/`policies`, `label`, and `step`/`max_steps` for a running trainer -- which
    `tools.laptop_batch.Driver.jobs` produces from its specs.
    """
    now = time.time() if now is None else now
    return {
        'iso': time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime(now)),
        'ts': now,
        'box': 'laptop',
        'project': 'snek3',
        'at_a_glance': daemon.build_at_a_glance(running, queued, {}),
        'running': [{key: job.get(key) for key in ('id', 'type', 'policy', 'policies', 'step', 'max_steps')}
                    for job in running],
        'queued_ids': [job.get('id') for job in queued],
    }


def _git(args, cwd):
    result = subprocess.run(['git'] + args, cwd=cwd, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return result.returncode, result.stdout.strip()


def ensure_worktree(host_config):
    """Makes sure the status branch exists and is checked out in the status worktree.

    Idempotent. The branch starts from the remote when it is already there, else from an empty root
    commit, so the worktree never holds a copy of the source tree. Returns the worktree path.
    """
    repo, worktree = host_config['REPO_PATH'], host_config['STATUS_WORKTREE']
    branch, remote = host_config['STATUS_BRANCH'], host_config['GIT_REMOTE']
    if os.path.exists(os.path.join(worktree, '.git')):
        return worktree
    _git(['fetch', remote, branch], repo)                     # may fail: the branch may not exist yet
    code, _ = _git(['rev-parse', '--verify', '--quiet', 'refs/heads/' + branch], repo)
    if code != 0:
        remote_code, _ = _git(['rev-parse', '--verify', '--quiet', '{0}/{1}'.format(remote, branch)], repo)
        if remote_code == 0:
            start = '{0}/{1}'.format(remote, branch)
        else:
            _, tree = _git(['hash-object', '-t', 'tree', os.devnull], repo)
            _, start = _git(['commit-tree', tree, '-m', '{0}: empty root'.format(branch)], repo)
        code, out = _git(['branch', branch, start], repo)
        if code != 0:
            raise RuntimeError('could not create branch {0}: {1}'.format(branch, out))
    os.makedirs(os.path.dirname(worktree), exist_ok=True)
    _git(['worktree', 'prune'], repo)
    code, out = _git(['worktree', 'add', worktree, branch], repo)
    if code != 0:
        raise RuntimeError('could not add worktree {0}: {1}'.format(worktree, out))
    return worktree


class Publisher(object):
    """Writes a status dict to the branch. `publish` returns True if the push landed.

    Failures are reported on stderr and swallowed: the driver's job is the arms, and a github outage
    must not stop a wave. Every process call goes through `publish_status` so a test can stand in.
    """

    def __init__(self, host_config=None, publish_status=gitbus.publish_status,
                 ensure=ensure_worktree, log=None):
        self.host = host_config or host()
        self._publish_status, self._ensure = publish_status, ensure
        self._ready = False
        self.log = log or (lambda message: sys.stderr.write(message + '\n'))

    def publish(self, status):
        try:
            if not self._ready:
                self._ensure(self.host)
                self._ready = True
            return self._publish_status(self.host, daemon.status_json(status))
        except Exception as error:      # noqa: BLE001 -- anything here is "could not publish", never fatal
            self.log('laptop-status publish failed (the driver continues): {0}'.format(error))
            return False


def main(argv=None):
    """`python -m tools.laptop_status` publishes an *empty* status: the laptop is idle. For the case
    where a driver was killed and left its last publish standing."""
    publisher = Publisher()
    landed = publisher.publish(build([], []))
    print('published empty laptop status to {0}: {1}'.format(
        publisher.host['STATUS_BRANCH'], 'pushed' if landed else 'NOT pushed (commit is local)'))
    return 0 if landed else 1


if __name__ == '__main__':
    sys.exit(main())
