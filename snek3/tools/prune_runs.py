"""Reclaim disk from finished work, without losing a measurement.

    PYTHONPATH=. python -m tools.prune_runs shards                     # what it would delete
    PYTHONPATH=. python -m tools.prune_runs shards --apply
    PYTHONPATH=. python -m tools.prune_runs arrays --apply
    PYTHONPATH=. python -m tools.prune_runs histogram --apply --include-tracked   # episode_scores -> score_counts
    PYTHONPATH=. python -m tools.prune_runs columns --apply --include-tracked     # stage-A rows -> columns
    PYTHONPATH=. python -m tools.prune_runs checkpoints b6a-... --keep-above 97.5 --apply

**Dry run is the default and `--apply` is the only thing that deletes.** Every subcommand prints
what it would do and the bytes it would free.

Three things accumulate, in ascending order of how much thought deleting them needs:

| subcommand | what goes | what is lost |
|---|---|---|
| `shards` | a pass's `-sNofM.json` files, once its merged file provably covers every row | nothing — exact duplicates |
| `arrays` | `episode_perfect` and `episode_rewards` from stored rows | nothing — one is derivable, the other has no reader (`tools/eval_plan.py`) |
| `histogram` | the `episode_scores` array from every pass row, replaced by its `score_counts` histogram (2026-09-11) | the order of episodes within a row, which nothing reads. Every summary field is checked against the histogram first and a file with one disagreement is left alone |
| `columns` | a stage-A `_evals.json`'s list of row dicts, rewritten as columns (`results.stage_a_payload`) | nothing — the round trip is checked before the write. A live arm's file is skipped: its trainer is the single writer |
| `checkpoints` | `ckpt-*.pt` whose stage-B row is below a threshold | the ability to re-measure or re-watch **that** checkpoint. Its measurement stays in `runs/` |

**`checkpoints` is the only one that loses anything, and it is also the one worth the most** — an arm
keeps a checkpoint per rollout, so a 100M-transition arm holds ~14,000 files at 109 KB each. It
refuses to touch an arm that is running, or one with no stage-B pass on disk, and it always keeps the
arm's best row whatever the threshold. What it cannot give back is re-screening at a *lower*
threshold than the one you keep, so keep a margin below the record region rather than exactly it.
"""

import argparse
import datetime
import glob
import json
import os
import re
import subprocess

import numpy as np

from env import constants
from env.observations import is_perfect_score
from tools import checkpoints, eval_plan, live_runs, results

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
                                cwd=constants.ROOT)
    except (OSError, subprocess.SubprocessError):
        return set()
    root = subprocess.run(['git', 'rev-parse', '--show-toplevel'], capture_output=True, text=True,
                          cwd=constants.ROOT).stdout.strip()
    return {os.path.realpath(os.path.join(root, name))
            for name in listed.stdout.split('\0') if name}


def in_flight(path):
    """Whether `path` is a shard file of a pass that has not merged yet.

    Rewriting one is a race with the shard process still writing it: `results.write` is atomic, so
    the loser is silently discarded rather than the file corrupted — but the loser can be this pass's
    rewrite *or* the shard's newest rows. The desktop runs evals unattended, so the tool cannot
    assume a quiet box.
    """
    if not SHARD_SUFFIX.search(path):
        return False
    return not os.path.exists(SHARD_SUFFIX.sub('.json', path))


def _without_arrays(payload):
    """`payload` with the two dead arrays gone from every row. Mutates, and returns it."""
    for row in results.rows_of(payload):
        for key in DEAD_ARRAYS:
            row.pop(key, None)
    return payload


def prune_arrays(apply=False, include_tracked=False):
    freed = skipped = 0
    tracked = set() if include_tracked else tracked_paths()
    for path in sorted(glob.glob(_runs('*_checkpoint_evals*.json'))):
        if path.endswith('.partial.json'):
            continue
        if in_flight(path):
            print('  KEEP  {0:<64} (a shard of a pass that has not merged)'.format(
                os.path.basename(path)))
            continue
        payload = results.read(path)
        rows = results.rows_of(payload)
        if not any(key in row for row in rows for key in DEAD_ARRAYS):
            continue
        before = os.path.getsize(path)
        if os.path.realpath(path) in tracked:
            # The droppable bytes, not the file's size — reporting the size overstated the trade by
            # 46% on the laptop and read as "39.4 MB skipped" on a box where nothing was skippable.
            skipped += before - len(json.dumps(_without_arrays(payload)))
            continue
        payload = _without_arrays(payload)
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
        '; skipped {0} of droppable bytes in git-tracked files '
        '(--include-tracked takes them too)'.format(_mb(skipped))
        if skipped else ''))
    return freed


# ---------------------------------------------------------------- histogram

def _histogram_rows(payload):
    """`payload` with every row's `episode_scores` replaced by `score_counts` (and the dead arrays gone).

    Mutates and returns `(payload, mismatches)`: a row whose stored `episodes`, `perfect_games` or
    `median_score` disagrees with its own histogram counts as a mismatch, and the caller leaves such
    a file untouched -- the check is what makes the conversion lossless by construction rather than by
    assumption.
    """
    mismatches = 0
    for row in results.rows_of(payload):
        for key in DEAD_ARRAYS:
            row.pop(key, None)
        scores = row.pop('episode_scores', None)
        if scores is None:
            continue
        counts = eval_plan.score_counts(scores)
        perfect = sum(n for score, n in counts.items() if is_perfect_score(int(score)))
        expected_median = round(float(np.median(scores)), 1) if scores else None
        if (sum(counts.values()) != row.get('episodes')
                or perfect != row.get('perfect_games')
                or ('median_score' in row and row['median_score'] != expected_median)):
            mismatches += 1
        row['score_counts'] = counts
    return payload, mismatches


def prune_histogram(apply=False, include_tracked=False):
    """Rewrites every pass file whose rows still carry `episode_scores`. Returns the bytes freed."""
    freed = skipped = 0
    tracked = set() if include_tracked else tracked_paths()
    for path in sorted(glob.glob(_runs('*_checkpoint_evals*.json'))):
        if path.endswith('.partial.json'):
            continue
        if in_flight(path):
            print('  KEEP  {0:<64} (a shard of a pass that has not merged)'.format(
                os.path.basename(path)))
            continue
        payload = results.read(path)
        rows = results.rows_of(payload)
        if not any('episode_scores' in row for row in rows):
            continue
        before = os.path.getsize(path)
        converted, mismatches = _histogram_rows(payload)
        after_text = len(json.dumps(converted))
        if mismatches:
            print('  KEEP  {0:<64} ({1} row(s) whose summary disagrees with its scores)'.format(
                os.path.basename(path), mismatches))
            continue
        if os.path.realpath(path) in tracked:
            skipped += before - after_text
            continue
        if apply:
            results.write(path, converted)
            after = os.path.getsize(path)
        else:
            after = after_text
        freed += before - after
        print('  {0}  {1:<64} {2} -> {3}'.format(
            'REWROTE' if apply else '  would', os.path.basename(path), _mb(before), _mb(after)))
    print('histogram: {0} {1}{2}'.format(
        'freed' if apply else 'would free', _mb(freed),
        '; skipped {0} of droppable bytes in git-tracked files '
        '(--include-tracked takes them too)'.format(_mb(skipped)) if skipped else ''))
    return freed


# ------------------------------------------------------------------ columns

def _without_nones(rows):
    return [{key: value for key, value in row.items() if value is not None} for row in rows]


def _stamp_from_disk(policy, path, raw):
    """Fills a stage-A summary's `started` / `finished` / `wall_seconds` from what the disk knows -- the
    arm's `arch.json` mtime and the file's own -- for a file written before the stamps existed (2026-09-11).
    Only when the checkpoint directory is on this box, and only into a summary with no `started`.
    Returns whether anything was filled."""
    summary = raw.get('summary')
    if not isinstance(summary, dict) or summary.get('started'):
        return False
    arch = os.path.join(constants.POLICY_DIR, policy, 'arch.json')
    if not os.path.exists(arch):
        return False
    started = datetime.datetime.fromtimestamp(os.stat(arch).st_mtime).isoformat(timespec='seconds')
    finished = datetime.datetime.fromtimestamp(os.stat(path).st_mtime).isoformat(timespec='seconds')
    if finished < started:
        return False
    summary.update({'started': started, 'finished': finished,
                    'wall_seconds': results.seconds_between(started, finished), 'stamps_from': 'disk mtimes'})
    return True


def prune_columns(apply=False, include_tracked=False):
    """Rewrites every stage-A `_evals.json` still stored as a list of rows in the column form.

    A live arm's file is left alone (the trainer rewrites it on every eval and is its single writer),
    and a file is written only if `from_columns(to_columns(rows))` gives the rows back. Returns the
    bytes freed.
    """
    freed = skipped = 0
    tracked = set() if include_tracked else tracked_paths()
    live = {policy for policy, _ in live_runs.live(constants.RUNS_DIR, prune=False)}
    for path in sorted(glob.glob(_runs('*_evals.json'))):
        name = os.path.basename(path)
        if '_checkpoint_evals' in name or name.endswith('.partial.json'):
            continue
        policy = name[:-len('_evals.json')]
        if policy in live:
            print('  KEEP  {0:<64} (the arm is live; its trainer owns the file)'.format(name))
            continue
        with open(path) as handle:
            raw = json.load(handle)
        if isinstance(raw, dict) and 'columns' in raw:
            if _stamp_from_disk(policy, path, raw):
                if apply:
                    results.write(path, raw)
                print('  {0}  {1:<64} (started/finished from arch.json and the file\'s own mtime)'.format(
                    'STAMPED' if apply else '  would', name))
            continue
        if not isinstance(raw, dict) or 'evals' not in raw:
            continue
        rows = raw.get('evals') or []
        _stamp_from_disk(policy, path, raw)
        converted = results.stage_a_payload(raw.get('summary') or {}, rows, raw.get('resumes') or [])
        if results.from_columns(converted['columns']) != _without_nones(rows):
            print('  KEEP  {0:<64} (the column round trip does not give the rows back)'.format(name))
            continue
        # Anything else the file carried rides along; the rows themselves are the columns now.
        for key, value in raw.items():
            if key not in converted and key != 'evals':
                converted[key] = value
        before = os.path.getsize(path)
        after_text = len(json.dumps(converted))
        if os.path.realpath(path) in tracked:
            skipped += before - after_text
            continue
        if apply:
            results.write(path, converted)
            after = os.path.getsize(path)
        else:
            after = after_text
        freed += before - after
        print('  {0}  {1:<64} {2} -> {3}'.format(
            'REWROTE' if apply else '  would', name, _mb(before), _mb(after)))
    print('columns: {0} {1}{2}'.format(
        'freed' if apply else 'would free', _mb(freed),
        '; skipped {0} of droppable bytes in git-tracked files '
        '(--include-tracked takes them too)'.format(_mb(skipped)) if skipped else ''))
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
    # `--apply` belongs to each subcommand, not to the top level: argparse accepts an option only
    # before the subcommand it is declared on, so a top-level flag makes `prune_runs shards --apply`
    # an "unrecognized arguments" error — which is the form this module's own docstring documents and
    # the form anyone would type. A parent parser puts it on all three.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument('--apply', action='store_true', help='actually delete; default is a dry run')
    sub = parser.add_subparsers(dest='what', required=True)
    sub.add_parser('shards', parents=[common], help='merged passes\' duplicate shard files')
    arrays = sub.add_parser('arrays', parents=[common],
                            help='the two dead per-episode arrays in stored rows')
    arrays.add_argument('--include-tracked', action='store_true',
                        help='rewrite git-tracked files too; see tracked_paths() on the trade')
    histogram = sub.add_parser('histogram', parents=[common],
                               help='episode_scores arrays in pass rows, replaced by score_counts')
    histogram.add_argument('--include-tracked', action='store_true',
                           help='rewrite git-tracked files too; see tracked_paths() on the trade')
    columns = sub.add_parser('columns', parents=[common],
                             help='stage-A _evals.json files stored as row lists, rewritten as columns')
    columns.add_argument('--include-tracked', action='store_true',
                         help='rewrite git-tracked files too; see tracked_paths() on the trade')
    checkpoints = sub.add_parser('checkpoints', parents=[common],
                                 help='a closed arm\'s unwanted checkpoints')
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
    elif args.what == 'histogram':
        prune_histogram(apply=args.apply, include_tracked=args.include_tracked)
    elif args.what == 'columns':
        prune_columns(apply=args.apply, include_tracked=args.include_tracked)
    else:
        prune_checkpoints(args.policies, args.keep_above, label=args.label, apply=args.apply)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
