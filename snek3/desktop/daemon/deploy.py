"""The box's side of a deploy: fetch, clear the run-artifact collisions, fast-forward.

`git merge --ff-only origin/master` aborts with "untracked working tree files would be overwritten"
whenever the laptop has committed a `runs/` file the box also holds untracked — and since 2026-09-02
every progress update commits the charts of every arm, *including the ones the box is still training*,
so that collision is now the normal state of a deploy rather than a mistake. This module settles it by
what each file **is**, not by whether it collides:

| colliding `runs/` file | what the box does | why |
|---|---|---|
| `*.png`, `*.md` — untracked, or tracked and locally modified | **keeps its own bytes**: saves them, lets the merge write the committed copy, writes its own back | the box drew every picture the laptop has ever committed, so the committed copy is always an older snapshot of the one the box holds — a finished arm's final chart must not be replaced by a mid-training one |
| any other file, identical to the incoming blob | stages it at that hash, then merges | a closed batch imported from `results`: same bytes, so the merge has nothing to overwrite |
| any other file that **differs** | **stops, exit 3, touches nothing** | a live `_evals.json` is the trainer's history and read on resume; a differing stage-B file is a live pass. Neither is anyone's to overwrite from here |

The JSON rule is the important half. `runs/<policy>_evals.json` is single-writer (the trainer) and is
what the arm's chart and report are rebuilt from across restarts, so clobbering it with the laptop's
older copy would silently truncate an arm's history.

Keeping the pictures leaves them as *modified tracked files* in the box's checkout after the first
deploy that carries them, and that is fine: the daemon publishes from its own worktrees
(`gitbus.py`), so the main checkout's status is nobody's input, and the next deploy saves and restores
them again before the fast-forward would otherwise refuse over "local changes".

Runs on base python on the box, like the rest of `daemon/`, so stdlib only.
"""

import argparse
import os
import subprocess
import sys

# Every directory a measurement or chart can land in on both boxes. snek2's two were added 2026-09-03,
# when the frozen era's stragglers were committed on the laptop and the box held untracked copies of
# them: the fast-forward refused on 75 files this script had not looked at, exit 4.
RUNS_PREFIXES = ('snek3/runs/', 'snek2/runs/', 'snek2/evals/')
PICTURES = ('.png', '.md')
EXIT_DIFFERS = 3
EXIT_MERGE_FAILED = 4


def git(repo, *args, check=True):
    result = subprocess.run(['git'] + list(args), cwd=repo, capture_output=True, text=True)
    if check and result.returncode != 0:
        raise RuntimeError('git {0} failed: {1}'.format(' '.join(args), result.stderr.strip()))
    return result.stdout


def repo_root(start=None):
    return git(start or os.getcwd(), 'rev-parse', '--show-toplevel').strip()


def incoming_blobs(repo, ref):
    """`{path: blob hash}` for every `runs/` file the incoming commit carries."""
    out = git(repo, 'ls-tree', '-r', ref, '--format=%(objectname) %(path)', '--', *RUNS_PREFIXES)
    blobs = {}
    for line in out.splitlines():
        sha, _, path = line.partition(' ')
        blobs[path] = sha
    return blobs


def untracked(repo):
    out = git(repo, 'ls-files', '--others', '--exclude-standard', '--', *RUNS_PREFIXES)
    return [line for line in out.splitlines() if line]


def modified_pictures(repo):
    """Tracked `runs/` pictures the box has changed since they were committed — kept on an earlier deploy."""
    out = git(repo, 'diff', '--name-only', 'HEAD', '--', *RUNS_PREFIXES)
    return [line for line in out.splitlines() if line.endswith(PICTURES)]


def plan(repo, ref='origin/master'):
    """Sorts the colliding files into `keep`, `stage` and `differs` without touching anything.

    `keep` holds pictures: untracked ones the incoming commit also carries, and tracked ones the box
    has redrawn since. Either would make the fast-forward refuse, and in both the box's copy is the
    one to end up with.
    """
    blobs = incoming_blobs(repo, ref)
    decision = {'keep': [], 'stage': [], 'differs': []}
    for path in untracked(repo):
        if path not in blobs:
            continue
        if path.endswith(PICTURES):
            decision['keep'].append(path)
            continue
        local = git(repo, 'hash-object', path).strip()
        decision['stage' if local == blobs[path] else 'differs'].append(path)
    decision['keep'].extend(p for p in modified_pictures(repo) if p not in decision['keep'])
    return decision


def apply(repo, decision, ref='origin/master', dry_run=False, out=sys.stdout):
    """Carries the plan out and fast-forwards. Returns the process exit code.

    The `differs` check comes first, before a single file is touched: a stop has to leave the box
    exactly as it found it, pictures included.
    """
    if decision['differs']:
        for path in decision['differs']:
            out.write('DIFFERS  {0}\n'.format(path))
        out.write('{0} untracked file(s) differ from what {1} carries and are not pictures. '
                  'Not merging, nothing touched: look at them first (a live arm\'s JSON must not be '
                  'committed from the laptop; `git rm --cached` it on master).\n'.format(
                      len(decision['differs']), ref))
        return EXIT_DIFFERS
    saved = {}
    tracked = set(git(repo, 'ls-files', '--', *RUNS_PREFIXES).splitlines())
    for path in decision['keep']:
        out.write('keep     {0}\n'.format(path))
        if dry_run:
            continue
        full = os.path.join(repo, path)
        with open(full, 'rb') as handle:
            saved[path] = handle.read()
        if path in tracked:
            git(repo, 'checkout', '--', path)   # back to HEAD so the fast-forward has nothing to refuse
        else:
            os.remove(full)
    for path in decision['stage']:
        out.write('stage    {0}\n'.format(path))
    if decision['stage'] and not dry_run:
        git(repo, 'add', '--', *decision['stage'])
    before = git(repo, 'rev-parse', '--short', 'HEAD').strip()
    if dry_run:
        out.write('would merge {0} -> {1}\n'.format(before, git(repo, 'rev-parse', '--short', ref).strip()))
        return 0
    result = subprocess.run(['git', 'merge', '--ff-only', ref], cwd=repo, capture_output=True, text=True)
    after = git(repo, 'rev-parse', '--short', 'HEAD').strip()
    for path, data in saved.items():          # the box's pictures back, whatever the merge did
        full = os.path.join(repo, path)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, 'wb') as handle:
            handle.write(data)
    if result.returncode != 0:
        out.write(result.stdout + result.stderr)
        out.write('merge failed; HEAD still {0}\n'.format(before))
        return EXIT_MERGE_FAILED
    out.write('HEAD {0} -> {1}; kept {2} of the box\'s own pictures\n'.format(before, after, len(saved)))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--dry-run', action='store_true', help='report the plan, change nothing')
    parser.add_argument('--ref', default='origin/master')
    parser.add_argument('--no-fetch', action='store_true', help='use the ref as already fetched')
    args = parser.parse_args(argv)
    repo = repo_root()
    if not args.no_fetch:
        remote, _, branch = args.ref.partition('/')
        git(repo, 'fetch', remote, branch)
    return apply(repo, plan(repo, args.ref), args.ref, dry_run=args.dry_run)


if __name__ == '__main__':
    sys.exit(main())
