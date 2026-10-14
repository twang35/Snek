"""All git interaction, over the single-writer branches of the bus.

| branch | writer | read here | payload |
|---|---|---|---|
| `ops` | laptop | yes, read-only | `queue/pending/*.json` specs, `config/runtime.json` |
| `ops-status` | **desktop daemon** | no | `status.json` |
| `results` | **desktop scheduler** (`tools/results_feed.py`) | no | `results/<job-id>/*` -- each finished arm's and pass's files, and every live arm's picture every ten minutes |
| `laptop-results` | laptop scheduler (`tools/results_feed.py`) | by `tools/site_build.py` | the same, for the laptop's work |
| `laptop-status` | laptop scheduler (`tools/laptop_status.py`) | yes, read-only | the laptop's `status.json`, folded into ours as `at_a_glance.laptop_*` |
| `site` | **desktop daemon** (`tools/site_build.py`) | no | the GitHub Pages viewer: a snapshot, one commit rewritten per build |
| `claims` | **both schedulers** (`tools/claims.py`) | yes | `claims/<batch>/w<N>.json`: which box holds which wave. The one multi-writer branch, and it is a lock, not a feed: a plain fast-forward push, and the loser of a race discards its commit and recomputes (`push_fast_forward`) |

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

## Every written branch is a snapshot: a short chain on the remote's tip, collapsed to a root periodically

`commit_and_push` used to append a commit per publish (the `results` pack reached 372 MB), then from
2026-09-10 wrote every publish as a **new parentless root commit**. Nothing read the history, and the tree
still carried every earlier job, so the branch stayed one commit. The cost showed on 2026-10-13: **git only
skips the objects reachable from the parents of what it pushes**, and an unrelated root on the remote
excludes nothing, so every push re-uploaded the whole 345 MB feed (`rev-list --objects HEAD ^<remote>`
listed all 7,645 objects for a six-file change). At the box's usual 8 Mbit/s that was six minutes a push;
the day the uplink fell to 1 Mbit/s the daemon sat inside one push for over an hour, consumed no triggers
and looked dead.

So a publish now (1) fetches the branch, so the remote's tip is current -- it is the commit's parent and the
lease, and a push that landed server-side while the client died no longer leaves the lease stale forever;
(2) writes the staged tree as a commit **whose parent is that tip**, so the pack is the delta; (3) once the
chain is `CHAIN_COLLAPSE_AFTER` commits long, writes a root instead -- one full upload a day, bounded growth
on the remote, the earlier chain left for its gc; and (4) if the tree already equals the tip's, pushes nothing.
`push` has a timeout and kills the whole push process group when it fires, so a slow link costs the daemon
`PUSH_TIMEOUT_SECONDS`, not an afternoon.
"""

import os
import shutil
import signal
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

# A push that runs longer than this is killed, process group and all (`git push` forks `ssh` and
# `pack-objects`, and `subprocess.run`'s own timeout kills only the parent). Twenty minutes carries a
# full collapse upload of the 345 MB `results` feed at 8 Mbit/s with headroom; a routine delta push is
# seconds. The daemon runs this on its single thread, which is why it is bounded at all (2026-10-13).
PUSH_TIMEOUT_SECONDS = 1200

# How many commits a feed's chain may reach before the next publish writes a root instead. 144 live
# publishes is a day at ten minutes; the collapse is the one full upload, so this is "once a day".
CHAIN_COLLAPSE_AFTER = 144


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


def fetch_branch(repo, remote, branch):
    """Fetches one branch on its own; False when it does not exist yet (a feed nobody has written)."""
    return subprocess.run(['git', 'fetch', remote, branch], cwd=repo, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE).returncode == 0


def ensure_worktree(repo, worktree, branch, remote):
    """Makes sure `branch` exists and is checked out in `worktree`, outside the main checkout.

    Idempotent. The branch starts from the remote when it is already there, else from an empty root
    commit, so the worktree never holds a copy of the source tree. Every writer of the bus -- the
    laptop's status and results, the desktop's site -- gets its worktree this way; the desktop's own
    status and results worktrees predate it and are made by hand in `desktop/README.md`.
    """
    if os.path.exists(os.path.join(worktree, '.git')):
        return worktree
    fetch_branch(repo, remote, branch)                                   # may fail: not there yet
    if _git(['rev-parse', '--verify', '--quiet', 'refs/heads/' + branch], cwd=repo).strip() == '':
        remote_ref = '{0}/{1}'.format(remote, branch)
        if _git(['rev-parse', '--verify', '--quiet', remote_ref], cwd=repo).strip():
            start = remote_ref
        else:
            tree = _git(['hash-object', '-t', 'tree', os.devnull], cwd=repo).strip()
            start = _git(['commit-tree', tree, '-m', '{0}: empty root'.format(branch)], cwd=repo).strip()
        _git(['branch', branch, start], cwd=repo, check=True)
    os.makedirs(os.path.dirname(worktree), exist_ok=True)
    _git(['worktree', 'prune'], cwd=repo)
    _git(['worktree', 'add', worktree, branch], cwd=repo, check=True)
    return worktree


def laptop_status_branch(host):
    """`LAPTOP_STATUS_BRANCH` from `host.env`, defaulting rather than required: a new required key
    would crash `load_host_config` on the box the moment the deploy landed (see `trigger_path`)."""
    return host.get('LAPTOP_STATUS_BRANCH') or 'laptop-status'


def fetch_laptop_status(host):
    """Fetches the laptop's status branch on its own, so its absence -- before the laptop has ever
    published -- cannot fail the fetch of the three branches the box needs."""
    _git(['fetch', host['GIT_REMOTE'], laptop_status_branch(host)], cwd=host['REPO_PATH'])


def read_laptop_status(host):
    """The laptop's `status.json` as text from the fetched ref, or `''` when there is none yet."""
    return _git(['show', '{0}/{1}:status.json'.format(host['GIT_REMOTE'], laptop_status_branch(host))],
                cwd=host['REPO_PATH'])


def read_pending_jobs(host):
    """`[(filename, text)]` for every `*.json` under `queue/pending/` on `origin/ops`.

    Read directly from the ref, so no checkout happens and a malformed spec cannot leave the working
    tree dirty. The directory comes from the host config rather than a literal, because the whole
    point of the `project` field is that two eras' queues can coexist on one branch. **Two git calls
    for the whole directory**, not one per file: `ls-tree` for the blob ids, then `cat-file --batch`
    for every content -- the scheduler reads this at every wave boundary and every claim, and a
    `git show` per spec was a minute of process starts on a laptop whose git carries a wrapper.
    """
    ref = _ops_ref(host)
    base = host['QUEUE_DIR']
    listing = _git(['ls-tree', '-r', ref, '--', base], cwd=host['REPO_PATH'])
    wanted = []            # (sha, filename), json files only, in listing order
    for line in listing.splitlines():
        meta, _, path = line.partition('\t')
        if not path.endswith('.json'):
            continue
        parts = meta.split()
        if len(parts) >= 3:
            wanted.append((parts[2], os.path.basename(path.strip())))
    if not wanted:
        return []
    result = subprocess.run(['git', 'cat-file', '--batch'], cwd=host['REPO_PATH'], input=''.join(
        sha + '\n' for sha, _ in wanted).encode(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    data = result.stdout
    jobs, offset = [], 0
    for sha, name in wanted:
        end = data.find(b'\n', offset)
        if end < 0:
            break
        header = data[offset:end].decode('utf-8', 'replace').split()
        offset = end + 1
        if len(header) < 3 or header[1] == 'missing':
            jobs.append((name, ''))
            continue
        size = int(header[2])
        jobs.append((name, data[offset:offset + size].decode('utf-8', 'replace')))
        offset += size + 1                 # the trailing newline cat-file adds
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


def publish_jobs(host, jobs, message):
    """Copies several jobs' files onto the results branch -- `{job_id: [paths]}`, each into
    `results/<job-id>/` -- as one commit and one push. Returns `(pushed, files copied)`. The live
    pictures of a wave go through here: eight arms, one push."""
    worktree = host['RESULTS_WORKTREE']
    clear_stale_locks(worktree)
    copied = 0
    for job_id, paths in sorted(jobs.items()):
        destination = os.path.join(worktree, 'results', job_id)
        os.makedirs(destination, exist_ok=True)
        for source in paths:
            if os.path.exists(source):
                shutil.copy2(source, destination)
                copied += 1
        _git(['add', '-A', os.path.join('results', job_id)], cwd=worktree)
    return _commit_and_push(worktree, host['RESULTS_BRANCH'], host, message), copied


def publish_results(host, job, artifact_paths):
    """Copies a job's artifacts onto the `results` branch. Returns True if the push landed.

    **False does not mean the results are lost** — the commit is local and `push_unpushed` will carry
    it. It means the caller must not yet report the job as published, which is the distinction snek2
    collapsed.
    """
    pushed, copied = publish_jobs(host, {job.id: list(artifact_paths)}, 'results for ' + job.id)
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
        if not push(worktree, branch, host):
            stuck.append(branch)
        else:
            sys.stderr.write('pushed a local-only commit on {0}\n'.format(branch))
    return stuck


def commit_and_push(worktree, branch, host, message):
    """Writes whatever is staged as the branch's next snapshot commit and pushes. Returns True if the
    *push* landed, or if the remote already holds exactly this tree (nothing to push).

    The branch is fetched first so the remote's tip is current: the new commit's parent (the push then
    carries only the delta) and the lease. The chain collapses to a root every `CHAIN_COLLAPSE_AFTER`
    commits. A missing git identity is the one failure worth raising for, because it makes every publish
    fail forever and no retry can fix it -- so the commit uses `check=True` while the push does not.
    """
    remote = host['GIT_REMOTE']
    fetch_branch(worktree, remote, branch)                 # may fail offline: the last fetched tip is used
    tip = _git(['rev-parse', '--verify', '--quiet', '{0}/{1}'.format(remote, branch)], cwd=worktree).strip() or None
    tree = _git(['write-tree'], cwd=worktree, check=True).strip()
    if tip and _git(['rev-parse', '--verify', '--quiet', tip + '^{tree}'], cwd=worktree).strip() == tree:
        _git(['reset', '-q', '--soft', tip], cwd=worktree, check=True)   # the remote has it; align HEAD
        return True
    parent = tip if tip and chain_length(worktree, tip) < CHAIN_COLLAPSE_AFTER else None
    snapshot_commit(worktree, message, parent=parent, tree=tree)
    return push(worktree, branch, host)


_commit_and_push = commit_and_push      # the name the callers and docs used until 2026-10-14


def chain_length(worktree, tip):
    """Commits on the branch since its last root, as the remote has it."""
    try:
        return int(_git(['rev-list', '--count', tip], cwd=worktree).strip() or '0')
    except ValueError:
        return 0


def snapshot_commit(worktree, message, parent=None, tree=None):
    """Replaces HEAD with a commit of the index: on `parent` when given, else a new root."""
    tree = tree or _git(['write-tree'], cwd=worktree, check=True).strip()
    args = ['commit-tree', tree, '-m', message] + (['-p', parent] if parent else [])
    commit = _git(args, cwd=worktree, check=True).strip()
    _git(['reset', '-q', '--soft', commit], cwd=worktree, check=True)
    return commit


def ref_head(repo, ref):
    """The commit `ref` points at, or None when it does not exist (a branch nobody has written yet)."""
    return _git(['rev-parse', '--verify', '--quiet', ref], cwd=repo).strip() or None


def push_fast_forward(worktree, branch, remote):
    """A plain push of `HEAD` to `branch`: `'pushed'`, `'rejected'` (not a fast-forward -- someone else
    pushed first, which on the `claims` branch means the race was lost and the caller recomputes) or
    `'failed'` (anything else: offline, auth). No lease and no force: the server's fast-forward check
    is the compare-and-swap the shared queue is built on (`tools/claims.py`)."""
    result = subprocess.run(['git', 'push', remote, 'HEAD:' + branch], cwd=worktree, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode == 0:
        return 'pushed'
    text = (result.stderr or '') + (result.stdout or '')
    if 'rejected' in text or 'non-fast-forward' in text or 'fetch first' in text or 'stale info' in text:
        return 'rejected'
    sys.stderr.write('push to {0} failed: {1}\n'.format(branch, text.strip()[-500:]))
    return 'failed'


def push(worktree, branch, host):
    """`--force-with-lease` because there is exactly one writer. Returns True on success.

    Bounded by `PUSH_TIMEOUT_SECONDS`: the push runs in its own process group and the whole group is
    killed when the clock runs out, because `git push` does its work in `ssh` and `pack-objects` children
    that would otherwise go on uploading. A timeout is an outage, not a blip, so it is not retried here;
    `push_unpushed` carries the commit on a later cycle.
    """
    stderr = ''
    for attempt in range(PUSH_ATTEMPTS):
        proc = _start_push(['git', 'push', '--force-with-lease', host['GIT_REMOTE'], 'HEAD:' + branch], worktree)
        try:
            _, stderr = proc.communicate(timeout=PUSH_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            _kill_group(proc)
            stderr = 'timed out after {0:.0f} s and was killed'.format(PUSH_TIMEOUT_SECONDS)
            break
        if proc.returncode == 0:
            return True
        if attempt + 1 < PUSH_ATTEMPTS:
            time.sleep(PUSH_BACKOFF_SECONDS * (attempt + 1))
    sys.stderr.write('push to {0} failed after {1} attempt(s); the commit is local and will be '
                     'retried: {2}\n'.format(branch, PUSH_ATTEMPTS, (stderr or '').strip()))
    return False


def _start_push(command, cwd):
    """The push process, in its own group so a timeout can take its children with it. Replaced by tests."""
    return subprocess.Popen(command, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            start_new_session=True)


def _kill_group(proc):
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        proc.kill()
    try:
        proc.communicate(timeout=10)
    except subprocess.TimeoutExpired:
        pass
