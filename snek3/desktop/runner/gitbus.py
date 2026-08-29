"""All git interaction, over three single-writer branches.

| branch | writer | read here | payload |
|---|---|---|---|
| `ops` | laptop | yes, read-only | `queue/pending/*.json` specs, `config/runtime.json` |
| `ops-status` | **desktop** | no | `status.json` |
| `results` | **desktop** | no | `results/<job-id>/*` artifacts |

`ops` is read straight from the fetched ref with `git show` / `git ls-tree`, so it is never checked
out and never risks a working-tree conflict. The two written branches go through dedicated
worktrees, so the main checkout is untouched. One writer per branch means `push --force-with-lease`
always applies cleanly and never needs a merge.

## A local-only commit is the failure this module is shaped around

snek2's `publish_results` pushed once and ignored the result. A failed push left the commit on the
box while the ledger said `done` — and "job finished, no files on `results`" is **indistinguishable
from a pass that legitimately found nothing**. It once hid four 500-episode result files, one of them
a 98.2% checkpoint, for hours.

So publishing is split in two here. `publish_results` always commits locally and then *reports*
whether the push landed; `push_unpushed` retries the branches that have commits ahead of their
remote. The daemon calls the second on every network cycle, so a push that fails during a router
reboot lands on the next pass without anything being lost or double-counted — the commit is already
made, so a retry is idempotent.
"""

import os
import shutil
import subprocess
import sys
import time

# A lock older than this is debris from a killed git, not a live one. Real git holds `index.lock`
# for the duration of one command — milliseconds here, since these worktrees hold a single small
# file each — so a minute is orders of magnitude of headroom. The age gate is what keeps this from
# being a footgun: it can never delete a lock a running git still owns.
LOCK_MAX_AGE_SECONDS = 60

# The locks a killed `git add`/`commit` leaves behind that block every later write. `index.lock` is
# the one that bites (held longest, taken by both add and commit); `HEAD.lock` is cheap to cover by
# the same mechanism. Ref locks under `refs/` are held for microseconds and are not worth sweeping.
LOCK_NAMES = ('index.lock', 'HEAD.lock')

# Retries inside one publish call, for a push that fails on a blip rather than an outage. Small on
# purpose: the durable fix is `push_unpushed` on the next cycle, not a long spin here, because this
# runs on the daemon's single thread and a job cannot be dispatched while it waits.
PUSH_ATTEMPTS = 3
PUSH_BACKOFF_SECONDS = 2.0


def clear_stale_locks(worktree, max_age=LOCK_MAX_AGE_SECONDS):
    """Removes leftover git lock files in `worktree` older than `max_age`. Returns what it removed.

    **Why this exists.** `publish_status` commits and pushes on every network cycle, so the daemon is
    inside a git write for a slice of every `git_seconds`. Kill it there — a reboot, an OOM — and the
    lock outlives the process. Every later `git add` then fails with "Unable to create index.lock:
    File exists", the publish raises, the daemon logs it and carries on: it keeps running and keeps
    dispatching jobs, but **`status.json` never updates again.** The laptop sees a frozen heartbeat
    while real work proceeds invisibly, which reads exactly like a dead daemon, and it needed a human
    with ssh to delete a file.

    Resolved through `git rev-parse --git-path`, because a worktree's `.git` is a *file* pointing at
    `<repo>/.git/worktrees/<name>` and the lock lives there, not at `<worktree>/.git/`.
    """
    removed = []
    for name in LOCK_NAMES:
        path = _git(['rev-parse', '--git-path', name], cwd=worktree).strip()
        if not path:
            continue
        if not os.path.isabs(path):
            path = os.path.join(worktree, path)
        try:
            age = time.time() - os.path.getmtime(path)
        except OSError:
            continue                      # not there, which is the normal case
        if age <= max_age:
            continue                      # young enough that a live git may still own it
        try:
            os.unlink(path)
            removed.append(path)
        except OSError as error:
            sys.stderr.write('could not clear stale lock {0}: {1}\n'.format(path, error))
    if removed:
        sys.stderr.write('cleared stale git lock(s): {0}\n'.format(', '.join(removed)))
    return removed


def _git(args, cwd, check=False):
    result = subprocess.run(['git'] + args, cwd=cwd, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if check and result.returncode != 0:
        raise RuntimeError('git {0} failed: {1}'.format(' '.join(args), result.stderr.strip()))
    return result.stdout


def _ops_ref(host):
    return '{0}/{1}'.format(host['GIT_REMOTE'], host['OPS_BRANCH'])


def fetch(host):
    _git(['fetch', host['GIT_REMOTE'], host['OPS_BRANCH'],
          host['STATUS_BRANCH'], host['RESULTS_BRANCH']], cwd=host['REPO_PATH'])


def read_pending_jobs(host):
    """`[(filename, text)]` for every `*.json` under `queue/pending/` on `origin/ops`.

    Read directly from the ref, so no checkout happens and a malformed spec cannot leave the working
    tree dirty. The directory comes from the host config rather than a literal, because the whole
    point of the `project` field is that two eras' queues can coexist on one branch.
    """
    ref = _ops_ref(host)
    base = host['QUEUE_DIR']
    listing = _git(['ls-tree', '-r', '--name-only', ref, '--', base], cwd=host['REPO_PATH'])
    jobs = []
    for path in listing.splitlines():
        path = path.strip()
        if not path.endswith('.json'):
            continue
        text = _git(['show', '{0}:{1}'.format(ref, path)], cwd=host['REPO_PATH'])
        jobs.append((os.path.basename(path), text))
    return jobs


def read_runtime_text(host):
    """The `runtime.json` committed on `origin/ops`, or `''`.

    An empty string is what an unreadable or absent file produces, and the parser treats that as
    invalid and keeps the last known-good config — so a bad commit on `ops` cannot reconfigure the
    box, and cannot stop it either.
    """
    return _git(['show', '{0}:{1}'.format(_ops_ref(host), host['RUNTIME_PATH'])],
                cwd=host['REPO_PATH'])


def publish_status(host, status_json_text):
    """Writes and pushes `status.json`. Returns True if the push landed."""
    worktree = host['STATUS_WORKTREE']
    clear_stale_locks(worktree)
    with open(os.path.join(worktree, 'status.json'), 'w') as handle:
        handle.write(status_json_text)
    _git(['add', 'status.json'], cwd=worktree)
    return _commit_and_push(worktree, host['STATUS_BRANCH'], host, 'status update')


def publish_results(host, job, artifact_paths):
    """Copies a job's artifacts onto the `results` branch. Returns True if the push landed.

    **False does not mean the results are lost** — the commit is local and `push_unpushed` will carry
    it. It means the caller must not yet report the job as published, which is the distinction snek2
    collapsed.
    """
    worktree = host['RESULTS_WORKTREE']
    clear_stale_locks(worktree)
    destination = os.path.join(worktree, 'results', job.id)
    os.makedirs(destination, exist_ok=True)
    copied = 0
    for source in artifact_paths:
        if os.path.exists(source):
            shutil.copy2(source, destination)
            copied += 1
    _git(['add', '-A', os.path.join('results', job.id)], cwd=worktree)
    pushed = _commit_and_push(worktree, host['RESULTS_BRANCH'], host, 'results for ' + job.id)
    if not copied:
        # Worth a line: a job that produced nothing and a job whose artifacts moved look the same
        # afterwards, and only one of them is a bug.
        sys.stderr.write('publish_results: {0} named {1} artifact path(s), none existed\n'.format(
            job.id, len(artifact_paths)))
    return pushed


def unpushed_branches(host):
    """Which of the two written branches have local commits their remote does not have.

    Asked of git rather than remembered in the ledger, because the ledger does not survive a reboot
    in the middle of a push and git's own refs do.
    """
    behind = []
    for worktree, branch in ((host['STATUS_WORKTREE'], host['STATUS_BRANCH']),
                             (host['RESULTS_WORKTREE'], host['RESULTS_BRANCH'])):
        remote = '{0}/{1}'.format(host['GIT_REMOTE'], branch)
        ahead = _git(['rev-list', '--count', '{0}..HEAD'.format(remote)], cwd=worktree).strip()
        try:
            if int(ahead or '0') > 0:
                behind.append((worktree, branch))
        except ValueError:
            # No such remote ref yet — a first push. Treat it as needing one.
            behind.append((worktree, branch))
    return behind


def push_unpushed(host):
    """Retries every branch with local-only commits. Returns the branches that are still behind.

    Called on each network cycle. Idempotent: the commits already exist, so a retry either lands
    them or leaves them exactly as they were.
    """
    stuck = []
    for worktree, branch in unpushed_branches(host):
        if not _push(worktree, branch, host):
            stuck.append(branch)
        else:
            sys.stderr.write('pushed a local-only commit on {0}\n'.format(branch))
    return stuck


def _commit_and_push(worktree, branch, host, message):
    """Commits whatever is staged and pushes. Returns True if the *push* landed.

    A missing git identity is the one failure worth raising for, because it makes every publish fail
    forever and no retry can fix it — so the commit uses `check=True` while the push does not.
    """
    if _git(['status', '--porcelain'], cwd=worktree).strip():
        _git(['commit', '-q', '-m', message], cwd=worktree, check=True)
    return _push(worktree, branch, host)


def _push(worktree, branch, host):
    """`--force-with-lease` because there is exactly one writer. Returns True on success."""
    for attempt in range(PUSH_ATTEMPTS):
        result = subprocess.run(
            ['git', 'push', '--force-with-lease', host['GIT_REMOTE'], 'HEAD:' + branch],
            cwd=worktree, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode == 0:
            return True
        if attempt + 1 < PUSH_ATTEMPTS:
            time.sleep(PUSH_BACKOFF_SECONDS * (attempt + 1))
    sys.stderr.write('push to {0} failed after {1} attempt(s); the commit is local and will be '
                     'retried: {2}\n'.format(branch, PUSH_ATTEMPTS, result.stderr.strip()))
    return False
