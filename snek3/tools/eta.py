"""Time estimates for the scheduler's status: how long a running arm or pass has left, how long a
queued one will take. What puts `| ~40m left` and `| ~4.5h` on `status.json`'s lines and the box's
`remaining` total under them (user, 2026-09-05: to see at a glance how much time is queued on each
box, and whether to shift a batch).

**An arm's wall clock is two files' mtimes.** `savedPolicies/<arm>/arch.json` is written once, as the
trainer starts (matches the scheduler's `launched` line to the second), and `runs/<arm>_evals.json`
is rewritten at every eval, so for a finished arm the gap is its life and for a running arm it is its
life so far. That is the rate a queued arm is estimated at -- the median over the batch's finished
arms, else the box's most recently finished arms -- and it needs no ledger and no bookkeeping.

**A running arm's own `steps_per_second` is not a wall rate, and the difference is 2x.** The trainer
measures it between evals, so it misses the checkpoint write and the wait on the stage-A queue: b16's
arms accounted for 42 minutes of rows over an 85-minute life. So what is left of a running arm is
its remaining steps at its recent row rate (PPO slows as the games get longer -- 75k steps/s at the
start, 20k at the end -- so the whole-life rate would read low), plus the overhead it has shown so
far scaled to the steps left.

**Passes have no rate file**, so the scheduler records each one it finishes -- seconds and arms -- in
`runs/.live/.durations.json` (`tools/live_runs.py`), and a pass is estimated at that ledger's median
seconds per arm times the wave's arms. Per box by construction (the desktop's 16 shards and the
laptop's 12 give different times); before a box has run one, `DEFAULT_PASS_SECONDS_PER_ARM`.

Every function returns None rather than guess when it has nothing, and `build_at_a_glance` leaves a
line without an estimate as it was.
"""

import glob
import json
import os
import statistics

from env import constants
from tools import live_runs

RECENT_ROWS = 10
# The box-wide fallback rate reads this many of the most recently touched arms.
RECENT_ARMS = 8
# An arm younger than this has no wall rate worth reading.
MIN_WALL_SECONDS = 60.0
# Before this box has measured a pass of its own. b16 on the laptop, 2026-09-05, 8-arm waves at 12
# shards: stage B 55 m, hof5000 9.5 m, hof30k 4.5 m.
DEFAULT_PASS_SECONDS_PER_ARM = {'stageb': 55 * 60 / 8.0, 'hof5000': 9.5 * 60 / 8.0, 'hof30k': 4.5 * 60 / 8.0}
# A running pass is never shown as less than this: its estimate is a median, and half of them run over.
MIN_RUNNING_PASS_SECONDS = 60.0

_rows_cache = {}    # path -> ((mtime, size), [(step, steps_per_second)]); the files are megabytes


def evals_path(policy, runs_dir=None):
    return os.path.join(runs_dir or constants.RUNS_DIR, '{0}_evals.json'.format(policy))


def arch_path(policy, policy_dir=None):
    return os.path.join(policy_dir or constants.POLICY_DIR, str(policy), 'arch.json')


def _rows(policy, runs_dir=None):
    """`[(step, steps_per_second)]` for every row of the arm's `_evals.json` that has both, or `[]`."""
    path = evals_path(policy, runs_dir)
    try:
        stat = os.stat(path)
    except OSError:
        return []
    key = (stat.st_mtime, stat.st_size)
    cached = _rows_cache.get(path)
    if cached is not None and cached[0] == key:
        return cached[1]
    rows = []
    try:
        with open(path) as handle:
            loaded = json.load(handle)
        for row in loaded.get('evals') or []:
            if isinstance(row, dict) and row.get('steps_per_second') and row.get('step') is not None:
                rows.append((int(row['step']), float(row['steps_per_second'])))
    except (OSError, ValueError, AttributeError, TypeError):
        rows = []
    _rows_cache[path] = (key, rows)
    return rows


def recent_rate(policy, runs_dir=None):
    """Steps per second the arm's loop is doing now: the median of its last `RECENT_ROWS` rows, or None."""
    rows = _rows(policy, runs_dir)[-RECENT_ROWS:]
    return statistics.median(rate for _, rate in rows) if rows else None


def accounted_seconds(policy, runs_dir=None):
    """The seconds the arm's rows account for -- each row's step gap at that row's rate -- and its
    last step: `(seconds, step)`. `(0.0, 0)` before its first row."""
    seconds, previous = 0.0, 0
    for step, rate in _rows(policy, runs_dir):
        if rate > 0 and step > previous:
            seconds += (step - previous) / rate
        previous = max(previous, step)
    return seconds, previous


def wall_seconds(policy, runs_dir=None, policy_dir=None, now=None):
    """Seconds from the arm's start (`arch.json`) to its last eval (`_evals.json`); for an arm still
    running, to `now` when given. None without both files or under `MIN_WALL_SECONDS`."""
    try:
        started = os.stat(arch_path(policy, policy_dir)).st_mtime
        ended = os.stat(evals_path(policy, runs_dir)).st_mtime if now is None else float(now)
    except OSError:
        return None
    seconds = ended - started
    return seconds if seconds >= MIN_WALL_SECONDS else None


def wall_rate(policy, runs_dir=None, policy_dir=None):
    """Steps per wall-clock second over the arm's life (so far): its last step over `wall_seconds`."""
    seconds = wall_seconds(policy, runs_dir, policy_dir)
    _, step = accounted_seconds(policy, runs_dir)
    return step / seconds if seconds and step > 0 else None


def recent_arms_rate(runs_dir=None, policy_dir=None, count=RECENT_ARMS):
    """The box's own rate: the median `wall_rate` of the `count` most recently touched arms here."""
    # `*_evals.json` also matches a pass's `<arm>_checkpoint_evals[_<label>].json`; those are not arms.
    paths = [path for path in glob.glob(os.path.join(runs_dir or constants.RUNS_DIR, '*_evals.json'))
             if '_checkpoint_evals' not in os.path.basename(path)]
    paths.sort(key=lambda path: os.stat(path).st_mtime if os.path.exists(path) else 0, reverse=True)
    rates = []
    for path in paths[:count]:
        policy = os.path.basename(path)[:-len('_evals.json')]
        rate = wall_rate(policy, runs_dir, policy_dir)
        if rate:
            rates.append(rate)
    return statistics.median(rates) if rates else None


def reference_rate(policies, runs_dir=None, policy_dir=None):
    """A wall rate for an arm that has none of its own: the median over `policies` that have one,
    else the box's `recent_arms_rate`. None on a box that has never run an arm."""
    rates = [rate for rate in (wall_rate(policy, runs_dir, policy_dir) for policy in policies) if rate]
    return statistics.median(rates) if rates else recent_arms_rate(runs_dir, policy_dir)


def arm_seconds(step, max_steps, rate):
    """Seconds from `step` to `max_steps` at a wall `rate`, or None without a rate or a cap."""
    if not rate or max_steps is None:
        return None
    return max(0, int(max_steps) - int(step or 0)) / float(rate)


def running_arm_seconds(policy, step, max_steps, runs_dir=None, policy_dir=None, now=None, fallback_rate=None):
    """What is left of a running arm: its remaining steps at its recent loop rate, plus the overhead
    (wall clock its rows do not account for) it has shown so far, scaled to the steps left. Before its
    first row, its whole cap at `fallback_rate`; None when neither is available."""
    if max_steps is None:
        return None
    remaining = max(0, int(max_steps) - int(step or 0))
    recent = recent_rate(policy, runs_dir)
    if not recent:
        return arm_seconds(step, max_steps, fallback_rate)
    accounted, done = accounted_seconds(policy, runs_dir)
    seconds = remaining / recent
    wall = wall_seconds(policy, runs_dir, policy_dir, now=now)
    if wall and done > 0:
        seconds += max(0.0, wall - accounted) * remaining / float(done)
    return seconds


def pass_seconds(kind, arms, runs_dir=None):
    """Seconds a pass of `kind` over `arms` arms takes on this box: the ledger's median per arm, or the
    default per arm, times the arms. None for a kind neither knows."""
    entries = [entry for entry in live_runs.durations(runs_dir).get(kind) or []
               if isinstance(entry, dict) and entry.get('seconds') and entry.get('arms')]
    if entries:
        per_arm = statistics.median(entry['seconds'] / entry['arms'] for entry in entries)
    else:
        per_arm = DEFAULT_PASS_SECONDS_PER_ARM.get(kind)
    return None if per_arm is None else per_arm * max(1, int(arms))


def running_pass_seconds(kind, arms, elapsed, runs_dir=None):
    """What is left of a pass that has run `elapsed` seconds, never below `MIN_RUNNING_PASS_SECONDS`."""
    total = pass_seconds(kind, arms, runs_dir)
    return None if total is None else max(MIN_RUNNING_PASS_SECONDS, total - float(elapsed))
