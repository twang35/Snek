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

**A pass costs what its selector picked, not how many arms it covers.** The scheduler records each
pass it finishes -- seconds, arms and the checkpoints its merged files hold -- in
`runs/.live/.durations.json` (`tools/live_runs.py`), and a pass is estimated at that ledger's pooled
seconds per checkpoint (every pass's seconds over every pass's checkpoints) times the checkpoints the pass will measure, which its selector's input file
already says before it starts (`selected_checkpoints`): stage B reads the arm's stage-A rows above the
screen, hof5000 the stage-B rows above 99.2, hof30k the hof5000 rows above 99.6. Per box by construction
(the desktop's 16 shards and the laptop's 12 give different times); before a box has run one,
`DEFAULT_PASS_SECONDS_PER_CHECKPOINT`. Only when the input file is not there yet (a hof5000 queued
behind a stage B still running) does the estimate fall back to the ledger's seconds per arm.

Why per checkpoint (2026-09-07): b19's arms put ~20 checkpoints each into hof5000 and b25's, ladder-top
arms at 200M steps that spend most of their life above 99%, put 811. The per-arm median said 12
minutes for a pass that took five hours, and `running_pass_seconds` then sat at its floor for the
rest of the pass: `~1m left` for four hours and fifty minutes.

**A running pass is estimated from its own progress.** Every arm's merged file, and the shard files
of the arm in flight, say how many checkpoints are measured so far (`pass_progress`); what is left is
the rest at the pass's own seconds per checkpoint so far. Before its first row the ledger estimate
less the elapsed time stands in.

Every function returns None rather than guess when it has nothing, and `build_at_a_glance` leaves a
line without an estimate as it was.
"""

import glob
import json
import os
import re
import statistics

from env import constants
from tools import live_runs

RECENT_ROWS = 10
# The box-wide fallback rate reads this many of the most recently touched arms.
RECENT_ARMS = 8
# An arm younger than this has no wall rate worth reading.
MIN_WALL_SECONDS = 60.0
# Before this box has measured a pass of its own. b16 on the laptop, 2026-09-05, 8-arm waves at 12
# shards: stage B 55 m, hof5000 9.5 m, hof30k 4.5 m. Used only when the pass's checkpoint count is
# not yet knowable (its selector's input file is not written).
DEFAULT_PASS_SECONDS_PER_ARM = {'stageb': 55 * 60 / 8.0, 'hof5000': 9.5 * 60 / 8.0, 'hof30k': 4.5 * 60 / 8.0}
# Seconds per selected checkpoint before this box has measured a pass of its own: the laptop's
# ledger over b18-b25 (12 shards), 2026-09-07 -- stage B 0.14-0.40 s/row at 500 episodes, hof5000
# 1.5-4.8 at 5,000, hof30k 22-90 at 30,000 (a per-arm floor dominates when a wave selects a handful).
DEFAULT_PASS_SECONDS_PER_CHECKPOINT = {'stageb': 0.35, 'hof5000': 3.7, 'hof30k': 30.0}
# The ledger's per-checkpoint rate is trusted once its measured passes hold this many checkpoints
# between them; fewer is a handful of rows timing the pass's startup (`_per_checkpoint`).
MIN_LEDGER_CHECKPOINTS = 50
# The hall-of-fame cuts, as percents: what a stage-B row needs over 500 episodes to reach `hof5000`,
# and what a hof5000 row needs over 5,000 to reach `hof30k`. `closeout.PASSES` spells its selectors
# from these.
# 99.2 for hof5000 (user, 2026-09-07; 99 before): the lowest rate over 500 episodes whose 95% Wilson
# interval reaches the record, 99.65%/30,000 (496/500 has an upper bound of 99.69%, 495 falls short),
# so what is dropped could not have placed. The 500-episode reading is too noisy to cut harder: the
# ten checkpoints that ever read >=99.5/30k entered stage B between 99.0 and 99.8.
# 99.6 for hof30k (user, 2026-09-08; 99.2 before): over the 1,392 rows of b24-b27 with both a 5,000
# and a 30,000 reading, nothing below 99.6/5,000 ever read 99.6/30,000, and 99.5 no longer places.
# The 99.2-99.3 band was 54% of the rows sent and topped out at 99.4; a 30k row costs 5.3x a 5k one,
# and the move-history arms (b27) put ~7,500 rows a cell through the old cut (195 h on the laptop).
HOF_THRESHOLD = 99.2
HOF30K_THRESHOLD = 99.6
# The early stops (user, 2026-09-09; `plans/early-stop.md`): a hof5000 checkpoint is retired once it can
# no longer read 99.6 -- the hof30k cut, so nothing it could have been promoted for is lost -- and a
# hof30k checkpoint once it can no longer read 99.8, the record the pass exists to find. Stage B has no
# stop: density98 counts the rows a stop would retire. Modelled on b27 + b28's rows: 88 -> 60 h with
# the 99.7 stop, ~50 h with 99.8. A stop must never sit below the next pass's cut (asserted in
# `closeout.PASSES`), or a stopped row could be selected.
HOF5000_STOP = 99.6
HOF30K_STOP = 99.8
# Which file each pass's selector reads and the threshold it applies: `(label of the input pass,
# threshold)`, with None the arm's stage-A `_evals.json`. Spelled here rather than read from
# `closeout.PASSES` so this module imports no torch (the daemon reads it too); `closeout` imports
# the cut from here and a test holds the two in step.
PASS_INPUTS = {'stageb': (None, 97.0), 'hof5000': ('', HOF_THRESHOLD), 'hof30k': ('hof5000', HOF30K_THRESHOLD)}
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
    """The box's own rate: the median `wall_rate` of the `count` most recently touched **finished** arms
    here -- arms live in the `runs/.live/` registry are skipped, unless nothing else has ever run.

    Finished, because the most recently touched arms are the wave training now, and one odd wave then
    forecasts everything queued behind it: 2026-09-05 the laptop's b20 lanes-32/64 wave ran at 2,400-4,800
    wall steps/s (eval-queue bound) against b19's 9,800, and b18's three base-config waves read 12 h of
    training instead of ~4 h. A finished arm's wall rate is also its true one."""
    live = {policy for policy, _ in live_runs.live(runs_dir, prune=False)}
    # `*_evals.json` also matches a pass's `<arm>_checkpoint_evals[_<label>].json`; those are not arms.
    paths = [path for path in glob.glob(os.path.join(runs_dir or constants.RUNS_DIR, '*_evals.json'))
             if '_checkpoint_evals' not in os.path.basename(path)]
    paths.sort(key=lambda path: os.stat(path).st_mtime if os.path.exists(path) else 0, reverse=True)
    policies = [os.path.basename(path)[:-len('_evals.json')] for path in paths]
    finished = [policy for policy in policies if policy not in live]
    rates = []
    for policy in (finished or policies)[:count]:
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


def stage_b_path(policy, label, runs_dir=None):
    """`runs/<arm>_checkpoint_evals[_<label>].json`, the merged file of the pass labelled `label`
    ('' or None for the main stage-B pass)."""
    name = '{0}_checkpoint_evals{1}.json'.format(policy, '_' + label if label else '')
    return os.path.join(runs_dir or constants.RUNS_DIR, name)


def _pass_rows(path):
    """The rows of a stage-B file (merged or one shard), `[]` when it is absent or unreadable."""
    try:
        with open(path) as handle:
            loaded = json.load(handle)
        return [row for row in loaded.get('rows') or [] if isinstance(row, dict)]
    except (OSError, ValueError, AttributeError):
        return []


def _shard_paths(policy, label, runs_dir=None):
    stem = stage_b_path(policy, label, runs_dir)[:-len('.json')]
    exact = re.compile(re.escape(os.path.basename(stem)) + r'-s(\d+)of(\d+)\.json$')
    return [path for path in glob.glob(stem + '-s*of*.json') if exact.search(os.path.basename(path))]


def selected_checkpoints(kind, policy, runs_dir=None):
    """How many checkpoints the pass `kind` will measure for `policy`: the rows of its selector's input
    file at or above the threshold. None when that file is not there yet, or for a kind not in
    `PASS_INPUTS`. For an arm still training, stage B's count is the count so far."""
    if kind not in PASS_INPUTS:
        return None
    label, threshold = PASS_INPUTS[kind]
    path = evals_path(policy, runs_dir) if label is None else stage_b_path(policy, label, runs_dir)
    if not os.path.exists(path):
        return None
    try:
        with open(path) as handle:
            loaded = json.load(handle)
        rows = loaded.get('evals' if label is None else 'rows') or []
    except (OSError, ValueError, AttributeError):
        return None
    return sum(1 for row in rows if isinstance(row, dict) and row.get('perfect_percent') is not None
               and float(row['perfect_percent']) >= threshold)


def known_checkpoints(kind, policies, runs_dir=None, pending=()):
    """`(checkpoints, unknown_arms)`: the checkpoints the pass over `policies` will measure for the arms
    whose count is knowable -- an arm's merged file counts exactly when it already exists, else its
    selector's input file -- and how many arms have neither yet. An arm in `pending` is still
    training: its stage-A file holds only the rows so far, so its count is unknown rather than an
    undercount (at launch it would read 0)."""
    total, unknown = 0, 0
    for policy in policies:
        merged = stage_b_path(policy, _pass_label(kind), runs_dir)
        if policy not in pending and os.path.exists(merged):
            total += len(_pass_rows(merged))
            continue
        count = None if policy in pending else selected_checkpoints(kind, policy, runs_dir)
        if count is None:
            unknown += 1
        else:
            total += count
    return total, unknown


def pass_checkpoints(kind, policies, runs_dir=None, pending=()):
    """The checkpoints a pass over `policies` will measure, or None when any arm's is not knowable yet
    (`known_checkpoints`)."""
    total, unknown = known_checkpoints(kind, policies, runs_dir, pending)
    return None if unknown else total


def _pass_label(kind):
    """The label the pass `kind` writes under: '' for stage B (its file is the unlabelled one)."""
    return '' if kind == 'stageb' else kind


def pass_progress(kind, policies, runs_dir=None):
    """`(done, total)` checkpoints of a pass in flight: an arm's merged file when it has one, its shard
    files' rows while it is being measured, nothing before its turn. None when `total` is unknown."""
    total = pass_checkpoints(kind, policies, runs_dir)
    if total is None:
        return None
    done = 0
    for policy in policies:
        merged = stage_b_path(policy, _pass_label(kind), runs_dir)
        if os.path.exists(merged):
            done += len(_pass_rows(merged))
        else:
            done += sum(len(_pass_rows(path)) for path in _shard_paths(policy, _pass_label(kind), runs_dir))
    return done, total


def _per_arm(kind, entries):
    by_arm = [entry for entry in entries if entry.get('arms')]
    if by_arm:
        return statistics.median(entry['seconds'] / entry['arms'] for entry in by_arm)
    return DEFAULT_PASS_SECONDS_PER_ARM.get(kind)


def _per_checkpoint(kind, entries):
    """The box's seconds per checkpoint for `kind`: every measured pass's seconds pooled over every
    measured pass's checkpoints, not a median of per-pass rates. A pass that selected a handful of
    rows is startup plus one row's wall time spread over that handful -- the laptop's one hof30k entry
    with a count, b26's 2 rows in 0.1 h, read 102 s per row against the 22 s a full 12-shard pass
    does, and put hist4's queued hof30k at 195 h (2026-09-08). Pooling weights a pass by its rows, so
    the small ones cannot set the rate -- and until the ledger holds `MIN_LEDGER_CHECKPOINTS` of them
    the default stands, since pooling one small pass is that pass."""
    measured = [entry for entry in entries if entry.get('checkpoints')]
    checkpoints = sum(entry['checkpoints'] for entry in measured)
    if checkpoints >= MIN_LEDGER_CHECKPOINTS:
        return sum(entry['seconds'] for entry in measured) / checkpoints
    return DEFAULT_PASS_SECONDS_PER_CHECKPOINT.get(kind)


def pass_seconds(kind, arms, runs_dir=None, checkpoints=None, unknown_arms=0):
    """Seconds a pass of `kind` takes on this box. With `checkpoints` (what its selector picked,
    `known_checkpoints`): the ledger's median seconds per checkpoint, else the default per checkpoint,
    times the checkpoints -- plus, for the `unknown_arms` whose count is not knowable yet, the median
    per arm times each. Without `checkpoints`: the median per arm, else the default per arm, times
    `arms`. None for a kind neither knows."""
    entries = [entry for entry in live_runs.durations(runs_dir).get(kind) or []
               if isinstance(entry, dict) and entry.get('seconds')]
    per_arm = _per_arm(kind, entries)
    if checkpoints is not None:
        per_checkpoint = _per_checkpoint(kind, entries)
        if per_checkpoint is not None:
            seconds = per_checkpoint * max(0, int(checkpoints))
            if unknown_arms and per_arm is not None:
                seconds += per_arm * int(unknown_arms)
            return seconds
    return None if per_arm is None else per_arm * max(1, int(arms))


def running_pass_seconds(kind, arms, elapsed, runs_dir=None, policies=None):
    """What is left of a pass that has run `elapsed` seconds, never below `MIN_RUNNING_PASS_SECONDS`.

    With `policies`, from the pass's own progress: the checkpoints not yet measured at the seconds per
    checkpoint it has shown so far. Before its first measured checkpoint, or when its total is not
    knowable, the ledger's estimate for its checkpoints (else its arms) less `elapsed`."""
    progress = pass_progress(kind, policies, runs_dir) if policies else None
    if progress is not None:
        done, total = progress
        if done > 0 and elapsed > 0:
            return max(MIN_RUNNING_PASS_SECONDS, (total - done) * float(elapsed) / done)
    total = pass_seconds(kind, arms, runs_dir, checkpoints=progress[1] if progress else None)
    return None if total is None else max(MIN_RUNNING_PASS_SECONDS, total - float(elapsed))
