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

import datetime
import glob
import json
import os
import re

from env import constants

# Stage-A files are stored as columns since 2026-09-11 (`plans/runs-archive-compaction.md`, version D):
# `{'format': COLUMNS_FORMAT, 'summary', 'resumes', 'columns': {name: [value per row]}}` instead of a
# list of row dicts. Every reader goes through `read`, which hands back the row form either way, so the
# on-disk shape is this module's business alone. Measured on a 200M arm: 3.3 MB -> 1.06 MB, lossless.
COLUMNS_FORMAT = 'columns-1'


def iso_now():
    """The wall clock as an ISO-8601 string to the second, local time -- the stamp every result file carries."""
    return datetime.datetime.now().isoformat(timespec='seconds')


def seconds_between(started, finished):
    """`finished - started` in whole seconds for two `iso_now` stamps, or None if either is missing or odd."""
    try:
        delta = datetime.datetime.fromisoformat(finished) - datetime.datetime.fromisoformat(started)
    except (TypeError, ValueError):
        return None
    return int(round(delta.total_seconds()))


def to_columns(rows):
    """Rows as `{column: [value per row]}`, in first-seen key order.

    A dict-valued field one level deep (a PPO row's `ppo` block) flattens to `ppo.<key>` columns; a row
    without the field reads as None in every one of them. Lossless with `from_columns` except for a
    field explicitly stored as None, which comes back absent -- every reader uses `.get`, and a
    top-level key must not contain a dot.
    """
    names, seen = [], set()
    for row in rows:
        for key, value in row.items():
            if isinstance(value, dict):
                for sub in value:
                    name = '{0}.{1}'.format(key, sub)
                    if name not in seen:
                        seen.add(name)
                        names.append(name)
            elif key not in seen:
                seen.add(key)
                names.append(key)
    columns = {}
    for name in names:
        parent, dot, sub = name.partition('.')
        if dot:
            columns[name] = [row[parent].get(sub) if isinstance(row.get(parent), dict) else None
                             for row in rows]
        else:
            columns[name] = [row.get(name) for row in rows]
    return columns


def from_columns(columns):
    """The inverse of `to_columns`: a list of row dicts, `parent.sub` columns regrouped into a dict."""
    length = max((len(values) for values in columns.values()), default=0)
    rows = [{} for _ in range(length)]
    for name, values in columns.items():
        parent, dot, sub = name.partition('.')
        for index, value in enumerate(values):
            if value is None:
                continue
            if dot:
                rows[index].setdefault(parent, {})[sub] = value
            else:
                rows[index][name] = value
    return rows


def stage_a_payload(summary, rows, resumes=()):
    """The on-disk shape of a stage-A file: its summary, resume steps and the eval rows as columns."""
    return {'format': COLUMNS_FORMAT, 'summary': summary, 'resumes': list(resumes),
            'columns': to_columns(rows)}


def expand(payload):
    """A payload with its stage-A rows under `evals` as row dicts, whichever shape the file stored."""
    if isinstance(payload, dict) and 'columns' in payload and 'evals' not in payload:
        payload = dict(payload)
        payload['evals'] = from_columns(payload.pop('columns'))
    return payload


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
        return expand(json.load(handle))


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


def stage_a_step(policy):
    """The arm's last evaluated step from its `_evals.json` summary, or None (no file, or not an arm)."""
    try:
        return int(read(stage_a_path(policy))['summary']['step'])
    except (OSError, KeyError, TypeError, ValueError):
        return None


def merged_steps(policy, label=None):
    """The steps already in the arm's merged pass file, or an empty set if there is none or it is
    unreadable (a torn file is not a reason to skip the measurement)."""
    try:
        payload = read(stage_b_path(policy, label))
    except (OSError, ValueError):
        return set()
    return {int(row['step']) for row in rows_of(payload) if 'step' in row}


def merge(policy, label=None, delete_shards=False):
    """Combines every shard of a pass into the pass's own file. Returns `(path, rows)`.

    Rows are sorted by step and de-duplicated, keeping the longer sample when two shards measured
    the same checkpoint — which should not happen, and did in snek2 when a re-dispatched shard
    overlapped the slice it was replacing.

    **The pass's existing merged file is folded in**, so a pass re-run over an arm that trained on past
    its close-out (a cap raised after stage B: b52 i-l, 1.8M -> 2.4M, 2026-10-12) keeps the rows it has
    and the shards measure only the new checkpoints (`tools/shard.py` skips `merged_steps`). The header
    records `through_step`, the arm's stage-A step at the merge, which is how `tools/scheduler.pass_done`
    tells a pass that covered the arm from one that predates its extension.

    **Shards measured under different `stop_target`s are refused when any row was stopped.** A stopped
    row means something only against the target it was stopped under (`plans/archive/early-stop.md`), and a
    merged file has one header, so one file cannot hold two targets' stopped rows. snek2's result files
    had four gate eras that every reader had to know about; here the merge refuses instead. Full rows
    are full under any target, so files with none stopped merge whatever their headers say.
    """
    by_step = {}                            # step -> (row, the stop target it was measured under)
    payloads = [read(path) or {} for path in shard_paths(policy, label)]
    try:
        previous = read(stage_b_path(policy, label)) or {}
    except ValueError:
        previous = {}                       # a torn merged file holds nothing worth keeping
    for payload in payloads + ([previous] if previous else []):
        for row in rows_of(payload):
            existing = by_step.get(row['step'])
            if existing is None or row['episodes'] > existing[0]['episodes']:
                by_step[row['step']] = (row, payload.get('stop_target'))
    rows = [by_step[step][0] for step in sorted(by_step)]
    # Judged over the rows the file will hold: a stopped row a longer re-measurement replaced is gone,
    # and so is its target.
    any_stopped = any(bool(row.get('abandoned')) for row in rows)
    targets = {target for _, target in by_step.values()}
    if any_stopped and len(targets) > 1:
        raise ValueError(
            '{0}: shards were measured under different stop targets {1} and some rows were stopped; '
            'a merged file has one target. Re-measure the odd shards (`--no-resume`) under one.'.format(
                _stage_b_stem(policy, label), sorted(targets, key=lambda t: (t is None, t))))

    header = {}
    paths = shard_paths(policy, label)
    base = payloads[0] if payloads else previous
    if base:
        header = {key: value for key, value in base.items() if key not in ('rows', 'through_step')}
    # When the pass ran: the earliest shard start to the latest shard write (2026-09-11), so a batch's
    # measuring time is on the file rather than in a scheduler log.
    spans = payloads + ([previous] if previous else [])
    starts = [p['started'] for p in spans if p.get('started')]
    ends = [p['finished'] for p in spans if p.get('finished')]
    if starts and ends:
        header.update({'started': min(starts), 'finished': max(ends),
                       'wall_seconds': seconds_between(min(starts), max(ends))})
    header.update({'policy': run_name(policy), 'label': label, 'shards': len(paths),
                   'through_step': stage_a_step(policy), 'rows': rows})
    written = write(stage_b_path(policy, label), header)
    if delete_shards:
        for path in paths:
            os.remove(path)
    return written, rows
