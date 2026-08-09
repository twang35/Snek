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
    with open(os.path.join(wt, 'status.json'), 'w') as fh:
        fh.write(status_json_text)
    _git(['add', 'status.json'], cwd=wt)
    _commit_and_push(wt, host['STATUS_BRANCH'], host, 'status update')


def publish_results(host, job, artifact_paths):
    wt = host['RESULTS_WORKTREE']
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
    _git(['commit', '-q', '-m', message], cwd=worktree)
    # Single writer, so force-with-lease is safe and never needs a merge.
    _git(['push', '--force-with-lease', host['GIT_REMOTE'], 'HEAD:' + branch], cwd=worktree)
