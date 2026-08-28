"""All git interaction, over three single-writer branches:

  ops         -- laptop writes (job specs in queue/pending/, runtime.json). Read-only here.
  ops-status  -- desktop writes (status.json). Laptop reads only.
  results     -- desktop writes (per-job artifacts at completion). Laptop reads only.

The desktop reads `ops` straight from the fetched ref (`git show` / `git ls-tree`)
so it never checks it out and never risks a working-tree conflict. It writes its
two branches through dedicated worktrees so the main checkout is untouched. One
writer per branch means `push --force-with-lease` always applies cleanly without
a merge.
"""
import os
import shutil
import subprocess
import sys
import time

# A lock older than this is treated as debris from a killed git, not as a live one. Real git
# holds `index.lock` for the duration of one command -- milliseconds here, since these worktrees
# hold a single small file each -- so a minute is orders of magnitude of headroom. The age gate
# is what keeps this from being a footgun: it can never delete a lock a running git still owns.
LOCK_MAX_AGE_SECONDS = 60

# The locks a killed `git add`/`commit` leaves behind that block every later write. `index.lock`
# is the one that actually bites (held longest, and taken by both add and commit); `HEAD.lock` is
# cheap to cover by the same mechanism. Ref locks under refs/ are held for microseconds during
# the final update and are not worth sweeping.
LOCK_NAMES = ('index.lock', 'HEAD.lock')


def clear_stale_locks(worktree, max_age=LOCK_MAX_AGE_SECONDS):
    """Removes leftover git lock files in `worktree` older than `max_age`. Returns what it removed.

    **Why this exists.** `publish_status` commits and pushes on every *network* cycle, so the
    daemon is inside a git write for a slice of every `git_seconds` (every 30 s before that cycle
    was split out from the poll on 2026-08-27). Kill it there -- a reboot, an OOM -- and the
    lock outlives the process. Every later `git add` then fails with "Unable to create
    index.lock: File exists", `_commit_and_push` raises, `_publish` logs it and carries on: the
    daemon keeps running and keeps dispatching jobs, but **status.json never updates again**. The
    laptop sees a frozen heartbeat while real work proceeds invisibly, which reads exactly like a
    dead daemon. It needed a human with ssh to clear a file.

    These worktrees have exactly one writer -- this daemon, single-threaded -- so a lock present
    when we are about to write is already anomalous. The age gate makes the removal safe anyway.
    Resolved through `git rev-parse --git-path`, because a worktree's `.git` is a *file* pointing
    at `<repo>/.git/worktrees/<name>` and the lock lives there, not at `<worktree>/.git/`."""
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
        except OSError as e:
            sys.stderr.write('could not clear stale lock {0}: {1}\n'.format(path, e))
    if removed:
        sys.stderr.write('cleared stale git lock(s): {0}\n'.format(', '.join(removed)))
    return removed


def _git(args, cwd, check=False):
    r = subprocess.run(['git'] + args, cwd=cwd, text=True,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if check and r.returncode != 0:
        raise RuntimeError('git {0} failed: {1}'.format(' '.join(args), r.stderr.strip()))
    return r.stdout


def _ops_ref(host):
    return '{0}/{1}'.format(host['GIT_REMOTE'], host['OPS_BRANCH'])


def fetch(host):
    _git(['fetch', host['GIT_REMOTE'], host['OPS_BRANCH'],
          host['STATUS_BRANCH'], host['RESULTS_BRANCH']], cwd=host['REPO_PATH'])


def read_pending_jobs(host):
    """Returns [(filename, text)] for every *.json under queue/pending/ on
    origin/ops, read directly from the ref (no checkout)."""
    ref = _ops_ref(host)
    base = 'snek2/desktop/queue/pending'
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
    """The runtime.json committed on origin/ops, or '' if it cannot be read
    (which the parser treats as invalid -> keep last-known-good)."""
    ref = _ops_ref(host)
    return _git(['show', '{0}:snek2/desktop/config/runtime.json'.format(ref)],
                cwd=host['REPO_PATH'])


def publish_status(host, status_json_text):
    wt = host['STATUS_WORKTREE']
    clear_stale_locks(wt)
    with open(os.path.join(wt, 'status.json'), 'w') as fh:
        fh.write(status_json_text)
    _git(['add', 'status.json'], cwd=wt)
    _commit_and_push(wt, host['STATUS_BRANCH'], host, 'status update')


def publish_results(host, job, artifact_paths):
    wt = host['RESULTS_WORKTREE']
    clear_stale_locks(wt)
    dest = os.path.join(wt, 'results', job.id)
    os.makedirs(dest, exist_ok=True)
    for src in artifact_paths:
        if os.path.exists(src):
            shutil.copy2(src, dest)
    _git(['add', '-A', os.path.join('results', job.id)], cwd=wt)
    _commit_and_push(wt, host['RESULTS_BRANCH'], host, 'results for ' + job.id)


def _commit_and_push(worktree, branch, host, message):
    # Nothing staged -> nothing to do (git commit would error out).
    if not _git(['status', '--porcelain'], cwd=worktree).strip():
        return
    # check=True so a failure (e.g. missing git identity) raises instead of being
    # swallowed -- the caller logs it to the journal rather than going silent.
    _git(['commit', '-q', '-m', message], cwd=worktree, check=True)
    # Single writer, so force-with-lease is safe and never needs a merge.
    _git(['push', '--force-with-lease', host['GIT_REMOTE'], 'HEAD:' + branch], cwd=worktree, check=True)
