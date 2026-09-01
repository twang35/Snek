"""Reclaim disk from finished work, without losing a measurement.

    PYTHONPATH=. python -m tools.prune_runs shards                     # what it would delete
    PYTHONPATH=. python -m tools.prune_runs shards --apply
    PYTHONPATH=. python -m tools.prune_runs arrays --apply
    PYTHONPATH=. python -m tools.prune_runs checkpoints b6a-... --keep-above 97.5 --apply

**Dry run is the default and `--apply` is the only thing that deletes.** Every subcommand prints
what it would do and the bytes it would free.

Three things accumulate, in ascending order of how much thought deleting them needs:

| subcommand | what goes | what is lost |
|---|---|---|
| `shards` | a pass's `-sNofM.json` files, once its merged file provably covers every row | nothing — exact duplicates |
| `arrays` | `episode_perfect` and `episode_rewards` from stored rows | nothing — one is derivable, the other has no reader (`tools/eval_plan.py`) |
| `checkpoints` | `ckpt-*.pt` whose stage-B row is below a threshold | the ability to re-measure or re-watch **that** checkpoint. Its measurement stays in `runs/` |

**`checkpoints` is the only one that loses anything, and it is also the one worth the most** — an arm
keeps a checkpoint per rollout, so a 100M-transition arm holds ~14,000 files at 109 KB each. It
refuses to touch an arm that is running, or one with no stage-B pass on disk, and it always keeps the
arm's best row whatever the threshold. What it cannot give back is re-screening at a *lower*
threshold than the one you keep, so keep a margin below the record region rather than exactly it.
"""

import argparse
import glob
import json
import os
import re
import subprocess

from env import constants
from tools import checkpoints, live_runs, results

DEAD_ARRAYS = ('episode_perfect', 'episode_rewards')
SHARD_SUFFIX = re.compile(r'-s(\d+)of(\d+)\.json$')


def _mb(n):
    return '{0:.1f} MB'.format(n / 1e6)


def _runs(*parts):
    return os.path.join(constants.RUNS_DIR, *parts)


# ------------------------------------------------------------------ shards

def shard_groups():
    """`{merged_path: [shard_path, ...]}` for every pass with shard files on disk."""
    groups = {}
    for path in sorted(glob.glob(_runs('*-s*of*.json'))):
        if not SHARD_SUFFIX.search(path):
            continue
        groups.setdefault(SHARD_SUFFIX.sub('.json', path), []).append(path)
    return groups


def covered(merged_path, shard_paths):
    """Whether the merged file holds every shard row at a sample at least as long.

    The check is per row rather than per file, because that is the property that makes deleting the
    shards safe: a wave that was killed mid-pass has rows in its shards that never reached the merge.
    """
    if not os.path.exists(merged_path):
        return False, 'no merged file'
    merged = {row['step']: row for row in results.rows_of(results.read(merged_path))}
    for path in shard_paths:
        for row in results.rows_of(results.read(path)):
            held = merged.get(row['step'])
            if held is None:
                return False, 'step {0} is not in the merge'.format(row['step'])
            if held.get('episodes', 0) < row.get('episodes', 0):
                return False, 'step {0} is shorter in the merge'.format(row['step'])
    return True, 'every row covered'


def prune_shards(apply=False):
    freed = kept = 0
    for merged_path, shard_paths in sorted(shard_groups().items()):
        ok, why = covered(merged_path, shard_paths)
        size = sum(os.path.getsize(path) for path in shard_paths)
        name = os.path.basename(merged_path)
        if not ok:
            kept += size
            print('  KEEP  {0:<64} {1} ({2})'.format(name, _mb(size), why))
            continue
        freed += size
        print('  {0}  {1:<64} {2} in {3} shard(s)'.format(
            'DELETE' if apply else '  would', name, _mb(size), len(shard_paths)))
        if apply:
            for path in shard_paths:
                os.remove(path)
    print('shards: {0} {1}{2}'.format(
        'freed' if apply else 'would free', _mb(freed),
        '; kept {0} in passes the merge does not cover'.format(_mb(kept)) if kept else ''))
    return freed


# ------------------------------------------------------------------ arrays

def tracked_paths():
    """The `runs/` files git is tracking, as absolute paths. Empty if git cannot answer.

    Rewriting a *tracked* file to shrink it does not shrink the repository — git keeps the old blob
    in history and the rewrite adds a new one, so it trades working-tree bytes for permanent `.git`
    growth. Measured 2026-09-01: 51 tracked result files hold 129.5 MB of the 970 MB, against a
    263 MB repo. So tracked files are skipped by default, and `--include-tracked` is the deliberate
    choice to take that trade anyway.
    """
    try:
        # `--full-name` because without it the names come back relative to the cwd, and joining
        # those to the repo root silently produces paths that match nothing.
        listed = subprocess.run(['git', 'ls-files', '-z', '--full-name', constants.RUNS_DIR],
                                capture_output=True, text=True, timeout=30,
                                cwd=os.path.dirname(constants.RUNS_DIR))
    except (OSError, subprocess.SubprocessError):
        return set()
    root = subprocess.run(['git', 'rev-parse', '--show-toplevel'], capture_output=True, text=True,
                          cwd=os.path.dirname(constants.RUNS_DIR)).stdout.strip()
    return {os.path.realpath(os.path.join(root, name))
            for name in listed.stdout.split('\0') if name}


def prune_arrays(apply=False, include_tracked=False):
    freed = skipped = 0
    tracked = set() if include_tracked else tracked_paths()
    for path in sorted(glob.glob(_runs('*_checkpoint_evals*.json'))):
        if path.endswith('.partial.json'):
            continue
        if os.path.realpath(path) in tracked:
            skipped += os.path.getsize(path)
            continue
        payload = results.read(path)
        rows = results.rows_of(payload)
        if not any(key in row for row in rows for key in DEAD_ARRAYS):
            continue
        before = os.path.getsize(path)
        for row in rows:
            for key in DEAD_ARRAYS:
                row.pop(key, None)
        if apply:
            results.write(path, payload)
            after = os.path.getsize(path)
        else:
            after = len(json.dumps(payload))
        freed += before - after
        print('  {0}  {1:<64} {2} -> {3}'.format(
            'REWROTE' if apply else '  would', os.path.basename(path), _mb(before), _mb(after)))
    print('arrays: {0} {1}{2}'.format(
        'freed' if apply else 'would free', _mb(freed),
        '; skipped {0} in git-tracked files (--include-tracked takes them too)'.format(_mb(skipped))
        if skipped else ''))
    return freed


# ------------------------------------------------------------- checkpoints

def checkpoint_plan(policy, keep_above, label=None):
    """`(keep, drop, reason)` — the checkpoint steps to keep and to delete for one arm.

    Keeps: every step whose stage-B row is at or above `keep_above`, the best row's step whatever the
    threshold, and any checkpoint the pass never measured is **dropped** — it failed stage A's screen,
    which is the same judgement, made earlier and on 100 episodes rather than 500.
    """
    directory = os.path.join(constants.POLICY_DIR, policy)
    if not os.path.isdir(directory):
        return set(), set(), 'no such policy directory'
    if policy in {name for name, _pid in live_runs.live()}:
        return set(), set(), 'the arm is running'
    rows = results.rows_of(results.read(results.stage_b_path(policy, label)))
    if not rows:
        return set(), set(), 'no stage-B pass on disk, so nothing says which checkpoints matter'

    # `tools/checkpoints.py` owns the naming, both directions, so this cannot drift from it.
    on_disk = {step for step in (checkpoints.step_of(name) for name in os.listdir(directory))
               if step is not None}
    scored = {row['step']: row['perfect_percent'] for row in rows}
    best = max(scored, key=lambda step: scored[step])
    keep = {step for step, percent in scored.items() if percent >= keep_above} | {best}
    return keep & on_disk, on_disk - keep, '{0} measured, {1} on disk'.format(len(scored), len(on_disk))


def prune_checkpoints(policies, keep_above, label=None, apply=False):
    freed = 0
    for policy in policies:
        keep, drop, reason = checkpoint_plan(policy, keep_above, label=label)
        if not keep and not drop:
            print('  SKIP  {0:<40} {1}'.format(policy, reason))
            continue
        directory = os.path.join(constants.POLICY_DIR, policy)
        size = 0
        for step in sorted(drop):
            path = checkpoints.path(directory, step)
            size += os.path.getsize(path)
            if apply:
                os.remove(path)
        freed += size
        print('  {0}  {1:<40} keep {2:>6}  drop {3:>6}  {4:>10}  ({5})'.format(
            'PRUNED' if apply else ' would', policy, len(keep), len(drop), _mb(size), reason))
    print('checkpoints: {0} {1} at >={2}%'.format(
        'freed' if apply else 'would free', _mb(freed), keep_above))
    return freed


# ------------------------------------------------------------------- entry

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--apply', action='store_true', help='actually delete; default is a dry run')
    sub = parser.add_subparsers(dest='what', required=True)
    sub.add_parser('shards', help='merged passes\' duplicate shard files')
    arrays = sub.add_parser('arrays', help='the two dead per-episode arrays in stored rows')
    arrays.add_argument('--include-tracked', action='store_true',
                        help='rewrite git-tracked files too; see tracked_paths() on the trade')
    checkpoints = sub.add_parser('checkpoints', help='a closed arm\'s unwanted checkpoints')
    checkpoints.add_argument('policies', nargs='+')
    checkpoints.add_argument('--keep-above', type=float, default=97.5,
                             help='keep checkpoints whose stage-B row is >= this (default 97.5)')
    checkpoints.add_argument('--label', default=None, help='which stage-B pass to read')
    args = parser.parse_args(argv)

    if not args.apply:
        print('DRY RUN — nothing is deleted. Add --apply.\n')
    if args.what == 'shards':
        prune_shards(apply=args.apply)
    elif args.what == 'arrays':
        prune_arrays(apply=args.apply, include_tracked=args.include_tracked)
    else:
        prune_checkpoints(args.policies, args.keep_above, label=args.label, apply=args.apply)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
