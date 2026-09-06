"""Builds the GitHub Pages chart viewer from both boxes' results feeds and pushes it as the `site` branch.

    PYTHONPATH=. python -m tools.site_build            # build if a feed or a local chart moved, then push
    PYTHONPATH=. python -m tools.site_build --force    # build regardless

The desktop daemon runs this on every network cycle (`--force` on a trigger); the same command by hand
on the box is the same writer with the same worktree and lease. Pages serves branch `site`, path `/`.

| step | what |
|---|---|
| feeds | `git fetch` `results`, `laptop-results` and `ops-status`; each feed's tree is flattened into the build directory -- `results/<job-id>/<file>` becomes `<file>` -- incrementally, by `git diff --name-only` since the commit last flattened (`.feeds.json` beside the files). **Only snek3 arms' files** (`b<n><letters>-<what>-seed<N>`): `results` is shared with snek2's era and holds its arms, smoke and sweep jobs, and the old p-names of b3-b6 |
| this box | every file in `SNEK_RUNS_DIR` newer than the build directory's copy is copied over it: the box's own live pictures and measurements win over the feeds' |
| status | `origin/ops-status:status.json` lands at `.live/desktop/status.json`, so the manifest knows which pass is running or queued |
| site | `viewer_manifest.build(build_dir)` and `publish_pages.publish(...)` into the `site` worktree: `index.html`, `manifest.js`, `charts/*.png`, `.nojekyll` |
| push | one commit, **amended** each build so the branch is a snapshot with no history, `--force-with-lease` (`gitbus.push`) |

Nothing here is a source of truth: the build directory and the worktree are rebuilt from the feeds and
the branch is rewritten every time. `SNEK_SITE_BRANCH` (default `site`) and `SNEK_SITE_WORKTREE`
(default `~/.snek3-laptop/site`; the daemon sets it beside its other worktrees) say where.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time

from desktop.daemon import gitbus
from env import constants
from tools import publish_pages
from tools import viewer_manifest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(ROOT)
REMOTE = 'origin'
BRANCH = os.environ.get('SNEK_SITE_BRANCH', 'site')
WORKTREE = os.environ.get('SNEK_SITE_WORKTREE', os.path.expanduser('~/.snek3-laptop/site'))
FEEDS = ('results', 'laptop-results')
STATUS_BRANCH = 'ops-status'
SHARD = re.compile(r'-s\d+of\d+\.json$|\.partial')
# A snek3 arm: `b<n><letters>-<what>-seed<N>`. The `results` branch is shared with snek2's era (b22-b47,
# named `b46a-c51batch512seed1`), and holds smoke and worker-sweep jobs and the p0-p2 copies of b3-b6
# under their old names; none of those are the viewer's (2026-09-05: 45 batches showed where 17 belonged).
ARM = re.compile(r'^b\d+[a-z]+-.+-seed\d+$')


def is_arm_file(name):
    """Whether a runs-style file name belongs to a snek3 arm: `<policy>.png`, `<policy>_evals.json`, ..."""
    stem = name.split('_', 1)[0]
    stem = stem[:-len('.png')] if stem.endswith('.png') else stem[:-len('.md')] if stem.endswith('.md') else stem
    return bool(ARM.match(stem))
FEEDS_STATE = '.feeds.json'


def _git(args, cwd, check=True):
    result = subprocess.run(['git'] + args, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if check and result.returncode != 0:
        raise RuntimeError('git {0} failed: {1}'.format(' '.join(args), result.stderr.strip()))
    return result.stdout


def _rev(repo, ref):
    return _git(['rev-parse', '--verify', '--quiet', ref], cwd=repo, check=False).strip() or None


def flatten_feed(repo, remote, feed, build_dir, since):
    """Copies every non-shard file the feed added or changed since `since` (a commit, or None for all)
    into `build_dir`, flattened. Returns `(new head or None, files written)`. Within one update a later
    job directory wins, and arm directories sort before pass directories, so a pass's copy of a file
    replaces the arm's -- the daemon's pre-2026-09-05 feeds hold both."""
    head = _rev(repo, '{0}/{1}'.format(remote, feed))
    if head is None:
        return None, 0
    if since and since != head and _rev(repo, since):
        listing = _git(['diff', '--name-only', '--diff-filter=AMR', since, head], cwd=repo)
    elif since == head:
        return head, 0
    else:
        listing = _git(['ls-tree', '-r', '--name-only', head], cwd=repo)
    paths = [p for p in listing.splitlines() if p.count('/') == 2 and p.startswith('results/')
             and not SHARD.search(p) and is_arm_file(p.split('/')[2])]
    written = 0
    for path in sorted(paths, key=lambda p: ('-stageb' in p or '-hof' in p, p)):
        target = os.path.join(build_dir, path.split('/')[2])
        with open(target + '.partial', 'wb') as handle:
            handle.write(subprocess.run(['git', 'show', '{0}:{1}'.format(head, path)], cwd=repo,
                                        stdout=subprocess.PIPE, check=True).stdout)
        os.replace(target + '.partial', target)
        written += 1
    return head, written


def overlay_runs(runs_dir, build_dir):
    """Copies this box's own files over the build directory where they are newer or absent. Returns the count."""
    copied = 0
    if not os.path.isdir(runs_dir):
        return 0
    for entry in os.scandir(runs_dir):
        if not entry.is_file() or entry.name.startswith('.') or SHARD.search(entry.name) or not is_arm_file(entry.name):
            continue
        target = os.path.join(build_dir, entry.name)
        source_stat = entry.stat()
        if os.path.exists(target):
            have = os.stat(target)
            if have.st_size == source_stat.st_size and int(have.st_mtime) >= int(source_stat.st_mtime):
                continue
        shutil.copy2(entry.path, target)
        copied += 1
    return copied


def prune_build_dir(build_dir):
    """Removes files that are not a snek3 arm's from the build directory -- what an earlier, unfiltered
    flatten left, or a feed's stray file. Returns the count."""
    removed = 0
    for entry in os.scandir(build_dir):
        if entry.is_file() and not entry.name.startswith('.') and (SHARD.search(entry.name) or not is_arm_file(entry.name)):
            os.remove(entry.path)
            removed += 1
    return removed


def place_status(repo, remote, build_dir):
    """The desktop's published status, for the manifest's pass states. Missing is fine."""
    text = _git(['show', '{0}/{1}:status.json'.format(remote, STATUS_BRANCH)], cwd=repo, check=False)
    if not text.strip():
        return False
    path = os.path.join(build_dir, viewer_manifest.DESKTOP_STATUS)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as handle:
        handle.write(text)
    return True


def build(repo=REPO, remote=REMOTE, branch=BRANCH, worktree=WORKTREE, feeds=FEEDS, runs_dir=None,
          build_dir=None, viewer_dir=None, force=False, log=print, push=True):
    """One build. Returns a dict: `built` (bool), `pushed`, and the counts. Raises on a git failure the
    caller should see (a missing identity, a broken worktree); the daemon turns that into `attention`."""
    runs_dir = runs_dir or constants.RUNS_DIR
    build_dir = build_dir or worktree.rstrip('/') + '-build'
    os.makedirs(build_dir, exist_ok=True)
    gitbus.ensure_worktree(repo, worktree, branch, remote)
    for name in tuple(feeds) + (STATUS_BRANCH,):
        gitbus.fetch_branch(repo, remote, name)
    state_path = os.path.join(build_dir, FEEDS_STATE)
    try:
        with open(state_path) as handle:
            state = json.load(handle)
    except (OSError, ValueError):
        state = {}
    flattened = 0
    for feed in feeds:
        head, written = flatten_feed(repo, remote, feed, build_dir, state.get(feed))
        flattened += written
        if head:
            state[feed] = head
    overlaid = overlay_runs(runs_dir, build_dir)
    pruned = prune_build_dir(build_dir)
    place_status(repo, remote, build_dir)
    with open(state_path, 'w') as handle:
        json.dump(state, handle)
    if not (force or flattened or overlaid or pruned):
        log('site: nothing moved since the last build; skipped')
        return {'built': False, 'pushed': None, 'flattened': 0, 'overlaid': 0}
    manifest = viewer_manifest.build(build_dir)
    copied, removed, total = publish_pages.publish(runs_dir=build_dir, viewer_dir=viewer_dir, docs_dir=worktree,
                                                   manifest=manifest)
    gitbus.clear_stale_locks(worktree)
    _git(['add', '-A'], cwd=worktree)
    changed = bool(_git(['status', '--porcelain'], cwd=worktree).strip())
    if changed:
        _git(['commit', '-q', '--amend', '-m', 'site {0}: {1} arms, {2} charts'.format(
            time.strftime('%Y-%m-%dT%H:%M:%S'), len(manifest['arms']), total)], cwd=worktree)
    pushed = None
    if push and (changed or _rev(repo, '{0}/{1}'.format(remote, branch)) != _rev(worktree, 'HEAD')):
        pushed = gitbus.push(worktree, branch, {'GIT_REMOTE': remote})
    log('site: {0} arms, {1} charts ({2} copied, {3} removed); {4} feed file(s) flattened, {5} local file(s) '
        'overlaid; {6}'.format(len(manifest['arms']), total, copied, removed, flattened, overlaid,
                               'pushed' if pushed else ('push FAILED (commit is local)' if pushed is False else 'nothing to push')))
    return {'built': True, 'pushed': pushed, 'flattened': flattened, 'overlaid': overlaid,
            'arms': len(manifest['arms']), 'charts': total}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--force', action='store_true', help='build even if nothing moved')
    parser.add_argument('--no-push', action='store_true', help='build and commit, do not push')
    args = parser.parse_args(argv)
    result = build(force=args.force, push=not args.no_push)
    return 1 if result.get('pushed') is False else 0


if __name__ == '__main__':
    sys.exit(main())
