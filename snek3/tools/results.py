"""Where result files live, and what they are called.

One module so the wave, the shards, the selectors and the report agree on a path without any of them
building it by hand. snek2 had this spelled out in four places and they drifted — a shard writing
`_s1of4` beside a controller looking for `-s1of4` is a wave that reports zero progress and finishes
with an empty file.

| file | written by | holds |
|---|---|---|
| `runs/<name>_evals.json` | the trainer | stage A: one 100-episode eval per checkpoint, plus a `summary` |
| `runs/<name>_checkpoint_evals[_<label>].json` | the wave | stage B: one 500-episode row per selected checkpoint |
| `runs/<name>_checkpoint_evals[_<label>]-s<i>of<n>.json` | one shard | that shard's slice of the above |

**Shards own their files and nothing merges into them while a wave runs.** That is a deliberate
difference from snek2, where one controller banked every lane's episodes and re-serialised the whole
result file 125 times per measurement — 58 s of single-threaded bookkeeping against the 46 s four
lanes needed to produce one, so the controller overtook its own workers and folded a 90-minute
backlog with 16 of them idle. Here there is no central row bookkeeping to overtake: each shard
appends to its own file and `merge` runs once, after.
"""

import glob
import json
import os
import re

from env import constants

def run_name(policy):
    """The name a policy's result files are keyed by — its directory's basename.

    A policy is named either as a bare arm name or as a path to a directory outside
    `savedPolicies/`, and both have to key the same files.
    """
    return os.path.basename(os.path.normpath(policy))


def stage_a_path(policy):
    return os.path.join(constants.RUNS_DIR, '{0}_evals.json'.format(run_name(policy)))


def _stage_b_stem(policy, label=None):
    stem = '{0}_checkpoint_evals'.format(run_name(policy))
    return stem + '_{0}'.format(label) if label else stem


def stage_b_path(policy, label=None, shard=None, shards=None):
    """The stage-B file, or one shard of it.

    `label` names a pass, so a re-measurement does not overwrite the one it is being compared with —
    which is the whole point of the phase-2 A/B. `shard` is zero-based and the filename is
    one-based, because `-s1of8` reads better in a log than `-s0of8`.
    """
    stem = _stage_b_stem(policy, label)
    if shard is not None:
        stem += '-s{0}of{1}'.format(shard + 1, shards)
    return os.path.join(constants.RUNS_DIR, stem + '.json')


def shard_paths(policy, label=None):
    """Every shard file for a pass, ascending by shard index. Missing ones simply are not there.

    The regex is what keeps `_checkpoint_evals_ab3222-s1of8.json` out of the unlabelled pass's list:
    the glob alone would match it, since `_ab3222` looks like part of a name.
    """
    stem = _stage_b_stem(policy, label)
    exact = re.compile(re.escape(stem) + r'-s(\d+)of(\d+)\.json$')
    found = []
    for path in glob.glob(os.path.join(constants.RUNS_DIR, stem + '-s*of*.json')):
        match = exact.search(os.path.basename(path))
        if match:
            found.append((int(match.group(1)), path))
    return [path for _, path in sorted(found)]


def read(path):
    """A result file, or None if it is absent. An unreadable one raises.

    Absent and corrupt are different things: a wave that has not started yet is normal, and a
    truncated file is a bug that should not be silently treated as "no results".
    """
    if not os.path.exists(path):
        return None
    with open(path) as handle:
        return json.load(handle)


def write(path, payload):
    """Writes a result file atomically, so a reader never sees a half-serialised one.

    A wave's progress readout polls these files while the shards write them, and a partial JSON
    parses as a corrupt file rather than as an older one.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    staging = path + '.partial'
    with open(staging, 'w') as handle:
        json.dump(payload, handle)
    os.replace(staging, path)
    return path


def rows_of(payload):
    return list(payload.get('rows', ())) if payload else []


def merge(policy, label=None, delete_shards=False):
    """Combines every shard of a pass into the pass's own file. Returns `(path, rows)`.

    Rows are sorted by step and de-duplicated, keeping the longer sample when two shards measured
    the same checkpoint — which should not happen, and did in snek2 when a re-dispatched shard
    overlapped the slice it was replacing.
    """
    by_step = {}
    for path in shard_paths(policy, label):
        payload = read(path)
        for row in rows_of(payload):
            existing = by_step.get(row['step'])
            if existing is None or row['episodes'] > existing['episodes']:
                by_step[row['step']] = row
    rows = [by_step[step] for step in sorted(by_step)]

    header = {}
    paths = shard_paths(policy, label)
    if paths:
        first = read(paths[0]) or {}
        header = {key: value for key, value in first.items() if key != 'rows'}
    header.update({'policy': run_name(policy), 'label': label, 'shards': len(paths),
                   'rows': rows})
    written = write(stage_b_path(policy, label), header)
    if delete_shards:
        for path in paths:
            os.remove(path)
    return written, rows
