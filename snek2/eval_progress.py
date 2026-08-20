"""Live progress view for one or more running eval_checkpoints.py runs.

Answers the two questions you actually have while an eval is running — *how is it doing* and
*how much is left* — without tailing six logs.

    cd snek2
    PYTHONPATH=. python -u eval_progress.py b8f-disc9975seed2
    PYTHONPATH=. python -u eval_progress.py            # every policy with eval results
    EVAL_PROGRESS_WATCH=30 PYTHONPATH=. python -u eval_progress.py b8f-disc9975seed2
    EVAL_PROGRESS_ALL=1 PYTHONPATH=. python -u eval_progress.py b8f-disc9975seed2
    EVAL_PROGRESS_WINDOW_MODE=1 EVAL_PROGRESS_WATCH=20 PYTHONPATH=. python -u \
        eval_progress.py b9a-disc9975a b9c-disc995a b9d-disc995b

By default it reports the **current job** — every result file written within
`EVAL_PROGRESS_WINDOW` seconds (default 3600) of the most recent one, which groups the parallel
chunks of one run and excludes earlier sessions. `EVAL_PROGRESS_ALL=1` pools every file for this
policy instead, which is a lifetime view and will double-count any checkpoint measured twice.

It reads `runs/<policy>_checkpoint_evals*.json`, which `eval_checkpoints.py` rewrites after
every round, and renders `evals/<policy>_eval_progress.png` plus a text summary on stdout.

**eval_checkpoints.py now opens this chart itself**, via live_frame() below, so the common
case needs no second command. This script stays useful for the cases that one cannot cover:
attaching to an eval already in flight, the text summary, the lifetime view
(`EVAL_PROGRESS_ALL=1`), and watching several arms at once.

`EVAL_PROGRESS_WINDOW_MODE=1` opens one window per policy from here, which is how you attach to
an eval that is **already running** — started before the built-in chart existed, or by something
that could not open a window itself. Combine it with `EVAL_PROGRESS_WATCH` to keep them live, and
pass several policy names to follow a whole batch from one process.

The original objection to building the chart into eval_checkpoints.py was that splitting one arm
across several `EVAL_OUT_SUFFIX` processes — six were used for the batch-8 close-out — would give
six partial pictures and six writers racing on one PNG. load_runs() answers that: it pools every
result file for the policy, so each window shows the whole job, and render() writes via a
temporary file and os.replace so a racing writer cannot produce a torn PNG. What is left is
duplicate windows in that one case, which is the accepted price of an eval never silently having
no chart at all.

The PNG has three parts, **always all three, at a fixed size**:

1. **In-flight convergence** — running perfect rate against round number, one line per
   process, with the current rate written next to the latest point. A checkpoint takes ~5 minutes
   and 10 rounds, so this is where you see whether the one being measured now is heading somewhere
   good, and how many rounds are left. When nothing is in flight the panel stays, empty and
   labelled — it used to be dropped, which shrank the figure from 9.0in to 6.6in between every pair
   of checkpoints and made the live window jump several times a minute.
2. **Completed checkpoints by step** — every finished measurement as a point, so the shape of
   the arm's good region is visible, with the best and the pooled mean marked. Under the screening
   protocol the points are **split by depth**: solid for full-length measurements, small hollow grey
   for 20-episode screens, because the two are not the same kind of number and drawing them alike
   invites reading a lucky 19/20 as a result.
3. **A text block** — a per-stage progress breakdown, top 5 checkpoints, percent complete, and an
   ETA computed from this run's own measured pace.

**The stage block is the thing to read while a three-stage close-out runs.** One undifferentiated
bar reads as a hung job, because the stages run at wildly different speeds: the 100%-graph-point
tier is 100 episodes a checkpoint and crawls, screening is 20 and races, confirming crawls again. A
finished stage collapses to `done` rather than keeping a 100% bar, and the screening line reports how
much of its pool was **cut** — left at screen length rather than promoted — which is the number that
says what the protocol bought on that arm.

**Two numbers here are deliberately filtered, and both used to be wrong.** `best` and `top 5` rank
over full-length rows only (`deep_rows`): across a few hundred 20-episode screens several land on
20/20, and an unfiltered top 5 showed five lucky screens at 100.0% sitting directly under a `best`
of 95%. And `pooled` prefers the writer's `pooled_equal_effort` — every checkpoint truncated to the
same prefix — because pooling rows of different depths weights the full-length ones, which are the
arm's best by construction, and reads high however good the policy is.

Text output goes to stdout as well, and that is the better one to read when checking progress
programmatically — it carries the same numbers without needing to open an image.
"""
import glob
import json
import os
import sys
import time

import imageio
import numpy as np

import matplotlib
matplotlib.use('Agg')  # no display needed; several of these may run over ssh or headless
import matplotlib.pyplot as plt
from matplotlib import gridspec

from snake_constants import EVALS_DIR, RUNS_DIR

# Results files that are aggregates rather than a running process, so including them would
# double-count finished work and invent in-flight state that no longer exists.
MERGED_SUFFIXES = ('_merged', '_closeout')


def load_runs(policy_name, window=None, include_all=False, suffixes=None):
    """Result files belonging to the *current* job for this policy.

    An arm accumulates result files across sessions — `b8f-disc9975seed2` has eight — and
    summing all of them answers the wrong question: it reports lifetime totals and
    double-counts any checkpoint measured more than once. Progress means "this job".

    A job is identified one of two ways. When the caller knows the job's `suffixes` — the
    running eval knows its own `EVAL_OUT_SUFFIX` and any it is resuming — it passes them and
    only those files count. Otherwise the job is guessed by write time: every file whose mtime
    is within `window` seconds of the newest one. The guess groups a run's parallel chunks
    correctly but misfires when a *different* eval on the same arm finished less than `window`
    ago — a close-out just before an HOF re-measurement — because its stale, much larger file
    lands inside the window and inflates the live chart. The explicit set is exact and is why
    the running eval passes it. `include_all` overrides both for the lifetime view.
    """
    if window is None:
        window = float(os.environ.get('EVAL_PROGRESS_WINDOW', 3600))
    pattern = os.path.join(RUNS_DIR, '{0}_checkpoint_evals*.json'.format(policy_name))

    candidates = []
    for path in sorted(glob.glob(pattern)):
        suffix = os.path.basename(path).split('_checkpoint_evals')[1][:-len('.json')]
        if suffix in MERGED_SUFFIXES:
            continue
        if suffixes is not None and suffix not in suffixes:
            continue
        try:
            mtime = os.path.getmtime(path)
            with open(path) as handle:
                payload = json.load(handle)
        except (ValueError, OSError):
            # A read that lands mid-write is possible in principle; skip it this pass.
            continue
        payload['suffix'] = suffix or '(none)'
        payload['path'] = path
        payload['mtime'] = mtime
        # Files written before `updated_at` and `complete` existed: fall back to the file's
        # own mtime, and treat them as finished since the old code only wrote once, at the end.
        payload.setdefault('updated_at', mtime)
        if payload.get('updated_at') is None:
            payload['updated_at'] = mtime
        if 'complete' not in payload:
            payload['complete'] = True
        candidates.append(payload)

    if not candidates or include_all or suffixes is not None:
        return candidates
    newest = max(run['mtime'] for run in candidates)
    return [run for run in candidates if newest - run['mtime'] <= window]


def discover_policies():
    pattern = os.path.join(RUNS_DIR, '*_checkpoint_evals*.json')
    names = set()
    for path in glob.glob(pattern):
        names.add(os.path.basename(path).split('_checkpoint_evals')[0])
    return sorted(names)


def deep_rows(rows):
    """Rows measured to at least half the deepest length on offer.

    The shared definition of "a measurement worth ranking". A 20-episode screen has a ~5x wider
    interval than a 100-episode confirmation, so across hundreds of screens some land on 20/20 by
    luck; anything that reports a maximum or a ranking has to exclude them or it reports noise.
    With one uniform episode count this is every row, which is why every protocol before screening
    is unaffected.
    """
    if not rows:
        return []
    deepest = max(r['episodes'] for r in rows)
    return [r for r in rows if r['episodes'] >= deepest / 2.0]


def best_of(rows):
    """The best row among those measured to at least half the deepest length on offer.

    Rates from different episode counts are not comparable as a maximum: a 20-episode screen
    has a ~5x wider interval than a 100-episode confirmation, so across hundreds of screened
    checkpoints some will read 19/20 or 20/20 on luck alone and beat every honestly-measured
    row. Restricting the comparison to the deep rows keeps "best checkpoint" meaning what it
    has always meant. With one uniform episode count — every protocol before screening existed
    — this returns exactly what max() over everything would.
    """
    deep = deep_rows(rows)
    if not deep:
        return None
    return max(deep, key=lambda r: r['perfect_percent'])


STAGE_LABELS = {
    # Shortened from 'measuring every 100% graph point at full length' when the text went to 10pt:
    # at 65 characters that line alone ran into the right column, and it is the widest line either
    # column produces.
    'full': 'every 100% graph point, full length',
    'screen': 'screening the rest',
    'confirm': 'confirming the best of the screened',
    'done': 'complete',
}


def flight_workers(run, flight):
    """How many workers the in-flight checkpoint is being measured on.

    Prefer the writer's own `num_workers`, which is EVAL_WORKERS read once at startup and fixed
    for the life of the process. It is a static property of the run and was never something to
    derive per round.

    The derivation is kept only for files written before the field existed, and it has to use
    `episodes_this_pass`: `episodes_so_far` is the checkpoint's *cumulative* sample, so on a
    screened checkpoint being topped up from 20 to 100 it starts 20 ahead of the rounds counting
    it, and `episodes_so_far // round` walks 30, 20, 16, 15, 14, 13, 12, 12 across the pass
    instead of sitting at 10. Files older still than `episodes_this_pass` fall through to the
    original arithmetic, which is exact for a fresh checkpoint and wrong the same way for a
    topped-up one — there is nothing in those files to do better with.
    """
    reported = run.get('num_workers')
    if reported:
        return max(1, int(reported))
    episodes = flight.get('episodes_this_pass')
    if episodes is None:
        episodes = flight.get('episodes_so_far', 0)
    return max(1, episodes // max(1, flight.get('round') or 1))


def stage_summary(runs):
    """Per-stage planned/done totals across every process on this policy, or None for a flat run.

    Exists because a three-stage close-out shown as one bar reads as a hung job: the full tier is
    100 episodes a checkpoint and crawls, then screening is 20 and races, then confirming crawls
    again. Three separate counts make that shape legible.

    `current` is the first stage that is not finished rather than whatever the writer last set,
    so it stays right when several processes are at different points and when a stage has nothing
    planned at all — an arm with no 100% graph points has an empty full tier and should read as
    screening from the start, not as "stage 1, 0/0".
    """
    reported = [r.get('stages') for r in runs if r.get('stages')]
    if not reported:
        return None
    out = {}
    for name in ('full', 'screen', 'confirm'):
        out[name] = {'planned': sum(s[name]['planned'] for s in reported),
                     'done': sum(s[name]['done'] for s in reported)}
    out['current'] = next((name for name in ('full', 'screen', 'confirm')
                           if out[name]['done'] < out[name]['planned']), 'done')
    out['order'] = ['full', 'screen', 'confirm']
    return out


def whole_rounds(episodes, workers):
    """`episodes` rounded up to a whole number of rounds, the way the planner counts them.

    `evaluate` runs one episode per worker per round and cannot stop part-way through one, so a
    100-episode request on 12 workers really runs 108. Mirrors `eval_checkpoints.plan_stages`'
    local function of the same name; an estimate that skipped it would read 7% low on a worker
    count that does not divide the request.
    """
    if not workers or workers <= 0:
        return episodes
    return -(-episodes // workers) * workers


def expected_run_fraction(runs, min_measurements=3):
    """The share of the episodes still *planned* that will actually be run, from this run's own
    abandonment rate. `1.0` when there is not enough evidence, or when nothing is being abandoned.

    `remaining_episodes` prices the plan ahead at full length and says so: "nothing here predicts
    future abandonment, which can only make the real figure smaller". That was written when the gate
    was a trim -- and on a close-out it is. **On a HOF pass at gate 98 it is the dominant term**, and
    calling it a small bias was wrong: `b43d-lowlr-b29c` had abandoned **23 of 23** measurements at a
    mean of 262 of 500 episodes, so the 60 checkpoints ahead were priced at 30,000 episodes when they
    will cost about 15,700 -- the chart read **4h25m** against a real ~2h.

    **Measured off the rows, not off the session counters, and that is the whole subtlety.** The
    obvious source is the identity `session_episodes + episodes_saved` = the planned work for this
    session, which `eval_checkpoints` itself prints. It is wrong across a **resume**: a resumed row
    that the gate re-abandons from its stored samples runs ~0 new episodes while still reporting the
    whole shortfall as saved, so the ratio collapses. On b43a mid-resume it read **0.025**, which
    would have turned a 4-hour estimate into six minutes. A row's own `episodes` against the target
    every row shares is immune -- resumed and fresh rows are priced the same way.

    Only for the **flat** protocol, where every measurement's target is `episodes_per_checkpoint`. A
    three-stage close-out mixes 20-episode screens with full-length rows and the payload does not say
    which a given row was, so there is no honest per-row target and this returns 1.0 -- today's
    behaviour, and conservative, which is the right direction for an ETA. `remaining_episodes`
    already handles the larger stage-mix error there by pricing each stage at its own length.

    `min_measurements` holds it at 1.0 until a few rows exist, so one unlucky first checkpoint cannot
    halve the ETA.
    """
    measured, actual, target_total = 0, 0, 0
    for run in runs:
        if run.get('stages'):
            return 1.0
        target = run.get('episodes_per_checkpoint')
        rows = [r for r in (run.get('results') or []) if r.get('episodes')]
        if not target or not rows:
            continue
        measured += len(rows)
        actual += sum(r['episodes'] for r in rows)
        target_total += target * len(rows)
    if measured < min_measurements or not actual or not target_total:
        return 1.0
    return min(1.0, float(actual) / target_total)


def remaining_episodes(runs):
    """Episodes still to run, counted off the stage plan. None when no run says enough to tell.

    **Not `episodes_planned - episodes_done`, which is what this replaced and which reads high by
    exactly the work the abandonment gate declined to run.** `episodes_planned` is the plan as it
    stood before any measurement — every full-tier row at the full length, every screen at the screen
    length — while `episodes_done` is what was actually run, and a row the gate abandoned at 44 of 100
    contributes 100 to the first and 44 to the second. The 56 it never ran then sits in the difference
    as though it were still ahead. `eval_checkpoints` names the same identity when it reports savings:
    the planned work for a session is `session_episodes + episodes_saved`.

    On `b43c-lowlr-b40b`, mid-confirm, this was a factor of 3.3: planned 101,700 against 77,883 done
    reads as 23,817 episodes remaining and **3h16m**, when the actual remainder was 89 confirmations
    of 80 episodes = 7,120 and **59m**. 16,697 of the gap was `episodes_saved`.

    Counting the plan forward instead of the shortfall backward is immune to that, and to two other
    things. It needs no savings bookkeeping, so it is right on a **resumed** run, where `progress`
    starts fresh and `episodes_saved` covers only the current session while `episodes_planned` covers
    the resumed rows too. And each stage is priced at *its own* length, so the estimate does not
    assume the remaining work looks like the average of what has run.

    Two known biases, both small and both upward, which is the right direction for an ETA: the
    in-flight checkpoint's episodes so far are not deducted (at most one checkpoint's worth), and
    nothing here predicts future abandonment, which can only make the real figure smaller.
    """
    total, known = 0, False
    for run in runs:
        target = run.get('episodes_per_checkpoint')
        if not target:
            continue
        workers = run.get('num_workers')
        stages, screen = run.get('stages'), run.get('screen_episodes')
        if stages and screen:
            # A confirmation tops a screen up to full length, so it costs the difference, not the
            # whole thing -- the same split `plan_stages` uses to build `episodes_planned`.
            for name, episodes in (('full', target), ('screen', screen),
                                   ('confirm', target - screen)):
                entry = stages.get(name) or {}
                left = max(0, (entry.get('planned') or 0) - (entry.get('done') or 0))
                total += left * whole_rounds(episodes, workers)
            known = True
            continue
        # The flat one-pass protocol has no stages, and every measurement is full length. Both
        # counts include resumed rows, so the difference is right for a resumed run too.
        planned, done = run.get('measurements_planned'), run.get('measurements_done')
        if planned is None or done is None:
            continue
        total += max(0, planned - done) * whole_rounds(target, workers)
        known = True
    return total if known else None


def stage_lines(stages):
    """The stage block for the text summary: one line per stage, detail only where it is useful.

    A finished stage collapses to `done` — its percentage is noise once the next stage is running,
    which is what was asked for. The screening line additionally reports how much of its pool was
    *cut*, i.e. left at the screen length rather than promoted, because that is the number that
    says what the protocol actually bought on this arm.
    """
    if not stages:
        return []
    current = stages['current']
    position = stages['order'].index(current) + 1 if current != 'done' else len(stages['order'])
    lines = ['  stage {0} of 3 — {1}'.format(position, STAGE_LABELS[current])]
    for name in stages['order']:
        planned, done = stages[name]['planned'], stages[name]['done']
        if planned == 0:
            lines.append('    {0:<7} {1:>4}       none for this arm'.format(name, '-'))
            continue
        counts = '{0}/{1}'.format(done, planned)
        if done >= planned:
            detail = 'done'
            if name == 'screen':
                promoted = stages['confirm']['planned']
                cut = planned - promoted
                # The cut share, not the cut count: '385 (74%) left at screen length' made this the
                # widest line the left column could produce -- 78 characters, which overlapped the
                # right column even at the old 8.5pt. 385 is `planned - promoted` and both are on
                # the line already, so only the percentage is not derivable.
                detail = 'done — {0} promoted, {1:.0f}% cut'.format(
                    promoted, 100.0 * cut / planned)
        elif done == 0 and name != current:
            detail = 'pending'
        else:
            filled = int(20 * done / planned)
            detail = '{0:>3.0f}%  {1}{2}'.format(100.0 * done / planned,
                                                 '#' * filled, '.' * (20 - filled))
        # `<7`, not `<8`: 'confirm' is exactly seven characters, so the wider field spent one on
        # nothing, and this block holds the left column's widest lines.
        lines.append('    {0:<7} {1:>9}  {2}'.format(name, counts, detail))
    return lines


def summarize(runs, stale_after=180):
    """Aggregate state across every process working on one policy.

    `stale_after` is how many seconds without a write before a run is treated as no longer
    live. A round takes ~30s, so 180 is well clear of normal jitter while still catching a
    process that was killed — which matters because a dead process leaves its last `in_flight`
    behind and would otherwise be drawn forever as though it were still working.
    """
    now = time.time()
    completed, active, stale = [], [], []
    requested = 0
    for run in runs:
        results = run.get('results', [])
        completed.extend(results)
        requested += run.get('checkpoints_requested') or len(results)
        age = now - (run.get('updated_at') or 0)
        in_flight = run.get('in_flight')
        if run.get('complete'):
            continue
        if age > stale_after:
            stale.append((run, age))
        elif in_flight:
            active.append((run, in_flight))

    rates = [r['perfect_percent'] for r in completed]
    total_perfect = sum(r['perfect_games'] for r in completed)
    total_episodes = sum(r['episodes'] for r in completed)
    times = [r['seconds'] for r in completed if r.get('seconds')]

    # ETA from this run's own pace. Checkpoints are measured sequentially within a process and
    # the processes run in parallel, so the wall-clock estimate divides remaining work by the
    # number of live processes rather than assuming one worker.
    #
    # Priced in episodes rather than checkpoints wherever the writer reports them, because a
    # screening protocol measures some checkpoints twice at different lengths: a 20-episode
    # screen and a 100-episode confirmation are not interchangeable units of work, so counting
    # checkpoints would put the ETA out by up to 5x. Files written before those fields existed
    # fall back to the old per-checkpoint arithmetic.
    # Pace from what the *live* processes have done this session, not from every row on file. A
    # resumed close-out carries rows measured under the earlier run's settings, and EVAL_WORKERS
    # alone moves seconds-per-episode by 3x — averaging the two put b10b's ETA at 7h10m against
    # a real ~2h. Falls back through session totals, then all rows, then per-checkpoint counting,
    # so files from any earlier version still produce an estimate.
    # Live *processes*, not workers: seconds-per-episode is already measured across a process's
    # whole worker pool, so the parallelism still to divide out is the number of processes. Named
    # explicitly because calling this `workers` is what let a genuine worker-count bug hide in
    # this file (see flight_workers).
    processes = max(1, len(active))
    session_seconds = sum(r.get('session_seconds') or 0 for r in runs)
    session_episodes = sum(r.get('session_episodes') or 0 for r in runs)
    session_measurements = sum(r.get('session_measurements') or 0 for r in runs)
    if session_episodes and session_seconds:
        seconds_per_episode = session_seconds / session_episodes
        mean_seconds = (session_seconds / session_measurements) if session_measurements else None
    elif total_episodes and times:
        seconds_per_episode = sum(times) / total_episodes
        mean_seconds = sum(times) / len(times)
    else:
        seconds_per_episode = None
        mean_seconds = (sum(times) / len(times)) if times else None

    ahead = remaining_episodes(runs)
    # Deflated by the gate's observed savings: `ahead` is the plan at full length, and on a gated
    # pass most of it is never run. See `expected_run_fraction` -- b43d's chart read 4h25m against
    # a real ~2h without this.
    run_fraction = expected_run_fraction(runs)
    planned_episodes = sum(r.get('episodes_planned') or 0 for r in runs)
    if ahead is not None and seconds_per_episode:
        eta_seconds = ahead * run_fraction * seconds_per_episode / processes
    elif planned_episodes and seconds_per_episode:
        # Files predating `episodes_per_checkpoint` / `stages`. `episodes_saved` is subtracted for
        # the reason in `remaining_episodes`' docstring — it is planned work that will not be run —
        # and is simply absent, hence 0, on the oldest files.
        saved = sum(r.get('episodes_saved') or 0 for r in runs)
        eta_seconds = max(0, planned_episodes - total_episodes - saved) * \
            run_fraction * seconds_per_episode / processes
    else:
        remaining = max(0, requested - len(completed))
        eta_seconds = (remaining * mean_seconds / processes) if mean_seconds else None

    # In a wave the controller's own figure wins, because none of the arithmetic above can be right
    # from one arm's file: its arms share lanes, so what that produces is this arm's remaining
    # *lane*-time over a process count that is not the arm's actual share of the box. `eval_wave`
    # measures the share instead -- wall clock between this arm's last ~10 completions -- which also
    # re-prices itself when a sibling finishes and hands its lane over. Both fields are absent from a
    # single-policy run, which is the common case, and `arm_eta_seconds` is absent for the first two
    # measurements of a wave, where the episode arithmetic above is the fallback.
    arm_eta = next((r.get('arm_eta_seconds') for r in runs
                    if r.get('arm_eta_seconds') is not None), None)
    if arm_eta is not None:
        eta_seconds = arm_eta
    # The whole wave's finish time is a different question -- when is the box free -- so it is shown
    # next to the arm's rather than instead of it. It was *replacing* it for a day, which made the
    # four panels agree at the cost of the number each one was being read for.
    wave_eta = next((r.get('wave_eta_seconds') for r in runs
                     if r.get('wave_eta_seconds') is not None), None)

    # Progress in measurements, which counts a confirmation pass separately from the screen that
    # earned it. Without it a screening run reads 100% done the moment stage 1 ends.
    planned = sum(r.get('measurements_planned') or 0 for r in runs)
    done = sum(r.get('measurements_done') or 0 for r in runs)
    if not planned:
        planned, done = requested, len(completed)

    return {
        'completed': completed,
        'active': active,
        'stale': stale,
        'requested': planned,
        'done': done,
        'percent_done': (100.0 * done / planned) if planned else 0.0,
        # Best over the deeply-measured rows only. Under a screening protocol a checkpoint that
        # got 20 episodes and went 19/20 would otherwise outrank a confirmed 88/100 and be shown
        # as the arm's best — a lucky screen the protocol deliberately declined to pursue. The
        # half-of-deepest floor rather than an exact match tolerates a worker count that rounds
        # 100 episodes up to 108; a 20-episode screen is nowhere near it either way.
        'best': best_of(completed),
        # Stage-aware label: 'measurements' when some checkpoints are measured twice, since then
        # the count is no longer one per checkpoint.
        'unit': ('measurements' if any(r.get('screen_episodes') for r in runs) else 'checkpoints'),
        'stages': stage_summary(runs),
        'screen_episodes': next((r.get('screen_episodes') for r in runs
                                 if r.get('screen_episodes')), None),
        # The run's own full-length target, so the completed-checkpoints chart can mark a row solid
        # only when it actually reached full depth. Read from the payload rather than the
        # environment, for the same reason as `min_achievable` below.
        'target_episodes': next((r.get('episodes_per_checkpoint') for r in runs
                                 if r.get('episodes_per_checkpoint')), None),
        # The abandonment gate, for the threshold line on the perfect-% charts. Taken from the
        # payload rather than the environment so a chart rendered later, or by `report`, shows the
        # gate the *file* was measured under instead of whatever EVAL_MIN_ACHIEVABLE happens to be
        # set to now. None for files that predate the gate.
        'min_achievable': next((r.get('min_achievable') for r in runs
                                if r.get('min_achievable')), None),
        # Prefer the writer's equal-effort figure when it exists. Pooling the rows themselves is
        # only an arm rate when every row has the same depth; under screening the deep rows are the
        # arm's best and pooling them reads high by construction.
        'pooled': next((r['pooled_equal_effort'] for r in runs
                        if r.get('pooled_equal_effort') is not None),
                       (100.0 * total_perfect / total_episodes) if total_episodes else 0.0),
        'pooled_is_equal_effort': any(r.get('pooled_equal_effort') is not None for r in runs),
        'episodes': total_episodes,
        'mean_seconds': mean_seconds,
        # Per-episode cost and the gate's savings, for the metrics block. `episodes_saved` is the
        # writer's own count of episodes the abandonment gate never ran; multiplied by the measured
        # seconds-per-episode it becomes the wall clock the gate bought, which is the form the
        # number is actually useful in.
        'seconds_per_episode': seconds_per_episode,
        'episodes_saved': sum(r.get('episodes_saved') or 0 for r in runs),
        # What the ETA is priced on, published so the text block can show it. Without it the ETA is
        # unfalsifiable on the face of the chart, and the obvious check -- remaining checkpoints
        # times `pace` -- disagrees with it whenever the remaining work is not average work.
        'episodes_ahead': ahead,
        'eta_seconds': eta_seconds,
        # The gate's observed run fraction, published for the same reason as `episodes_ahead`: it is
        # the difference between the episodes the plan names and the episodes the ETA prices.
        'run_fraction': run_fraction,
        # The wave's own total, shown beside the arm's ETA when the file is a wave's. Not a flag: the
        # arm ETA is always the arm's now, so what the text block needs is the other number itself.
        'wave_eta_seconds': wave_eta,
        # How many completions the arm's ETA was averaged over, or None when it came from the episode
        # arithmetic instead. Published so a suspicious ETA can be traced to its evidence -- and tied
        # to `arm_eta` rather than read on its own, so it can never describe an ETA it did not price.
        'eta_window': (next((r.get('arm_eta_window') for r in runs if r.get('arm_eta_window')), None)
                       if arm_eta is not None else None),
        'wave_arms': next((r.get('wave_arms') for r in runs if r.get('wave_arms')), None),
        'rates': rates,
    }


def full_length_rows(state):
    """(rows measured to the run's own target depth, that target).

    The same rule the chart's solid/hollow split uses -- `target_episodes` from the payload, falling
    back to the deepest row for files predating the field -- so the text block and the picture cannot
    disagree about which rows are comparable. Anything short of the target is a 20-episode screen or
    a row the abandonment gate stopped, and neither is a measurement of the same thing.
    """
    rows = state['completed']
    if not rows:
        return [], 0
    deepest = max(r['episodes'] for r in rows)
    target = state.get('target_episodes') or deepest
    return [r for r in rows if r['episodes'] >= target], target


# 98 is the hall-of-fame selection gate (`desktop/runner/runner.py::HOF_THRESHOLD`, and the laptop
# chain copies it), so these are the bands promotion is decided in: never lost a game, lost about
# one in a hundred, lost about two, not a candidate.
HOF_BAND = 98.0
PERFECT_BANDS = (('= 100%', 100.0, None),
                 ('99-100%', 99.0, 100.0),
                 ('98-99%', HOF_BAND, 99.0),
                 ('below 98%', None, HOF_BAND))


def band_counts(rows):
    """Rows per promotion band, as [(label, count), ...], highest band first.

    Half-open ranges rather than exact integers because the depth is not fixed. At 100 episodes
    "99-100%" can only ever be 99/100, but the HOF re-measure runs 500 episodes and 498/500 = 99.6%
    has to land in the same band; a set of `== 99` tests would drop it silently.
    """
    counted = []
    for label, low, high in PERFECT_BANDS:
        count = 0
        for row in rows:
            percent = row['perfect_percent']
            if low is not None and percent < low:
                continue
            if high is not None and percent >= high:
                continue
            count += 1
        counted.append((label, count))
    return counted


def longest_band_run(rows, target, threshold=HOF_BAND):
    """The longest consecutive stretch of decisive checkpoints all at or above `threshold`.

    Returns (length, first_step, last_step), or (0, None, None).

    This is the plateau test, and it is the difference between a champion and a lucky checkpoint: a
    98% row inside a stretch of twenty is a policy that holds, a lone one between two abandoned
    neighbours is a coin flip that landed. Consecutive means consecutive *among the rows that settle
    the question*, in step order rather than file order -- the close-out measures stage 1 before
    stage 2, so the file is not sorted.

    Three kinds of row, and only two of them are decisive:

    - measured to `target` -- counts if it clears `threshold`, breaks the stretch if it does not;
    - **abandoned** -- always breaks it. Abandonment is a proof of being below the close-out gate,
      hence below this one, and an abandoned row is short, so a depth test alone would skip it and
      let it silently bridge two separate stretches;
    - a screen short of `target` and not abandoned -- **skipped entirely**. 19/20 is equally
      consistent with a 98% policy and a 90% one, so counting it would invent a plateau and
      breaking on it would erase a real one.

    Unselected steps are not gaps either: nobody looked at them.
    """
    best, best_span = 0, (None, None)
    current, start = 0, None
    for row in sorted(rows, key=lambda r: r['step']):
        if not row.get('abandoned') and row['episodes'] < target:
            continue
        if row['perfect_percent'] >= threshold and not row.get('abandoned'):
            current += 1
            if start is None:
                start = row['step']
            if current > best:
                best, best_span = current, (start, row['step'])
        else:
            current, start = 0, None
    return (best,) + best_span


def half_split_pooled(rows):
    """(first-half rate, second-half rate) over the measured *step* range, or (None, None).

    Split by step rather than by row index, and pooled by episodes so a deep row counts for more
    than a screen. This is the whole question for a continuation arm -- b42-b45 exist to find out
    whether a low learning rate holds a champion or slowly decays it -- and reading it off the
    scatter by eye is exactly the judgement call that produced a retracted conclusion once already.
    """
    if len(rows) < 4:
        return None, None
    steps = sorted(r['step'] for r in rows)
    midpoint = steps[len(steps) // 2]
    halves = []
    for keep in (lambda s: s < midpoint, lambda s: s >= midpoint):
        chosen = [r for r in rows if keep(r['step'])]
        episodes = sum(r['episodes'] for r in chosen)
        if not episodes:
            return None, None
        halves.append(100.0 * sum(r['perfect_games'] for r in chosen) / episodes)
    return halves[0], halves[1]


def metrics_lines(state):
    """The right-hand column: the distribution and the derived numbers, not the ranking.

    Deliberately none of it duplicates the left column or the two charts above -- the in-flight
    lines came out of the text block entirely on 2026-08-19 because the top panel already draws
    them, per-round, which is strictly more than the text said.
    """
    rows = state['completed']
    if not rows:
        return ['  nothing measured yet']
    full, target = full_length_rows(state)
    pool, note = full, ''
    if not pool:
        # Every row stopped short of the target, which at a 98 gate is the normal case for a weak
        # arm. Rank what there is and label the depth, rather than printing four zeroes.
        pool = deep_rows(rows)
        # Its own line, not a suffix. As a suffix this ran to 53 characters and was **clipped by the
        # right edge of the frame** on `b43d`'s live chart -- the caveat that no row reached full
        # length is the one thing on the line that must not be the part that gets cut. The column's
        # width is what it is (see SUMMARY_FONTSIZE), so a second line is the only fix that scales.
        note = '    deepest {0} of {1} ep, none full length'.format(
            max(r['episodes'] for r in rows), target)
    lines = ['  perfect % of {0} {1} rows'.format(
        len(pool), 'full-length ({0} ep)'.format(target) if not note else 'deep')]
    if note:
        lines.append(note)
    for label, count in band_counts(pool):
        lines.append('    {0:<10} {1:>5}   {2:>4.0f}%'.format(
            label, count, 100.0 * count / len(pool)))

    candidates = [r for r in pool if r['perfect_percent'] >= HOF_BAND]
    if candidates:
        lines.append('  >= 98%: {0} rows over {1}-{2}'.format(
            len(candidates), format_step_k(min(r['step'] for r in candidates)),
            format_step_k(max(r['step'] for r in candidates))))
        length, first, last = longest_band_run(rows, target)
        if length:
            lines.append('    longest run {0} in a row ({1}-{2})'.format(
                length, format_step_k(first), format_step_k(last)))
    else:
        lines.append('  >= 98%: none -- no hall-of-fame candidate')

    early, late = half_split_pooled(rows)
    if early is not None:
        # Zeroed under half a tenth, because '{:+.1f}' renders a delta of -0.02 as '-0.0' — a minus
        # sign on a number that is not negative, which is exactly the direction this line exists to
        # report.
        delta = late - early
        lines.append('  drift {0:.1f}% -> {1:.1f}%  ({2:+.1f} pp, first/last half)'.format(
            early, late, delta if abs(delta) >= 0.05 else 0.0))

    # `.get` on both: an older file may carry neither, and the whole line is optional rather than
    # worth a KeyError on a chart that is a run's only live status display.
    avg_scores = [r['avg_score'] for r in pool if r.get('avg_score') is not None]
    worst = [r['min_score'] for r in pool if r.get('min_score') is not None]
    if avg_scores:
        lines.append('  score  mean {0:.1f}{1}'.format(
            sum(avg_scores) / len(avg_scores),
            '   worst episode {0:.0f}'.format(min(worst)) if worst else ''))

    # How well the training self-eval predicted the real measurement. The graph point is what
    # selects a checkpoint in the first place, so this is the selection protocol grading itself:
    # a low hit rate means the close-out is spending its full-length budget on noise.
    graph_perfect = [r for r in rows if (r.get('graph_single_eval') or 0) >= 100]
    if graph_perfect:
        hits = sum(1 for r in graph_perfect if r['perfect_percent'] >= HOF_BAND)
        lines.append('  graph 100% -> >= 98%: {0} of {1} ({2:.0f}%)'.format(
            hits, len(graph_perfect), 100.0 * hits / len(graph_perfect)))

    abandoned = sum(1 for r in rows if r.get('abandoned'))
    saved = state.get('episodes_saved') or 0
    if abandoned or saved:
        per_episode = state.get('seconds_per_episode')
        clock = ' ~{0}'.format(format_duration(saved * per_episode)) if per_episode else ''
        lines.append('  gate cut {0} rows, saved {1:,} ep{2}'.format(abandoned, saved, clock))
    lines.append('  {0} rows measured = {1} full + {2} partial'.format(
        len(rows), len(full), len(rows) - len(full)))
    # The wave's own total, in the right column because it is context rather than this arm's result --
    # and on the chart at all because the ETA on the left is now the arm's, so without this nothing
    # says when the box comes free. Absent for a single-policy run.
    if state.get('wave_eta_seconds') is not None:
        lines.append('  wave of {0}: all done in {1}'.format(
            state.get('wave_arms') or '?', format_duration(state['wave_eta_seconds'])))
    return lines


def format_duration(seconds):
    if seconds is None:
        return '?'
    seconds = int(seconds)
    if seconds < 90:
        return '{0}s'.format(seconds)
    if seconds < 5400:
        return '{0}m'.format(round(seconds / 60))
    return '{0}h{1:02d}m'.format(seconds // 3600, (seconds % 3600) // 60)


def format_step_k(step):
    """A checkpoint step as comma-grouped thousands: 1320000 -> '1,320k'."""
    return '{0:,}k'.format(step // 1000)


def in_flight_title(active):
    """Title for the in-flight panel, naming the checkpoint step(s) actually running.

    'In flight' used to report only counts -- "1 checkpoint, 1 process" -- which never said *which*
    checkpoint. Include the step(s): a single one reads '@1,320k'; several distinct steps (a shard
    set, or two processes on different checkpoints) join with '/'. The checkpoint and process counts
    stay as they were -- one entry per live process.
    """
    count = len(active)
    steps = sorted({flight['step'] for _, flight in active})
    steps_label = '/'.join(format_step_k(step) for step in steps)
    return 'In flight: running perfect rate by round ({0} checkpoint{1} @{2}, {3} process{4})'.format(
        count, '' if count == 1 else 's', steps_label,
        count, '' if count == 1 else 'es')


def progress_lines(policy_name, state):
    """Name, the progress bar and the stage block -- where the run is, not what it found."""
    lines = []
    done, requested = state['done'], state['requested']
    # 16, not the original 28: the bar is decoration and the line it sits on is the widest in the
    # left column, so those characters are worth more as column clearance at 10pt. Sized for the
    # worst case of five-digit counts on both sides of the slash, which is 51 characters.
    bar_width = 16
    filled = int(bar_width * state['percent_done'] / 100.0)
    lines.append('{0}'.format(policy_name))
    lines.append('  [{0}{1}] {2}/{3} {4} ({5:.0f}%)'.format(
        '#' * filled, '.' * (bar_width - filled), done, requested, state['unit'],
        state['percent_done']))

    lines.extend(stage_lines(state.get('stages')))
    return lines


def ranking_lines(state):
    """The headline number and the top 5: which checkpoints this arm produced."""
    lines = []
    if state['completed']:
        # 'ep', not 'episodes': with an eight-digit step and a six-digit episode total this is the
        # widest line the left column can produce, and at 10pt the long form ran into the right
        # column. `tests/test_eval_progress.py::test_the_left_column_keeps_clear_of_the_right_one`
        # measures it on exactly that worst case.
        pooled_note = (' (equal effort)' if state.get('pooled_is_equal_effort')
                       else ' over {0} ep'.format(state['episodes']))
        lines.append('  best {0:.1f}% @{1}   pooled {2:.1f}%{3}'.format(
            state['best']['perfect_percent'], state['best']['step'],
            state['pooled'], pooled_note))
        # The episode count is shown because `pace` is a blended per-measurement average and the
        # ETA is not priced on it: at the end of a screening close-out every remaining measurement is
        # an 80-episode confirmation, so `remaining x pace` reads ~30% low against a correct ETA and
        # looks like the ETA is wrong. The two numbers reconcile through this one. On a *flat* gated
        # pass the two now agree by construction, since `expected_run_fraction` is exactly the ratio
        # that turns the planned episodes ahead into the mean measurement's actual cost.
        ahead = state.get('episodes_ahead')
        lines.append('  pace {0}/checkpoint   ETA {1}{2}'.format(
            format_duration(state['mean_seconds']),
            format_duration(state['eta_seconds']),
            ' ({0:,} ep left)'.format(ahead) if ahead else ''))
        # Ranked over the deeply-measured rows only, the same rule best_of() applies. Without it
        # the list contradicted the `best` line directly above it: across a few hundred 20-episode
        # screens several land on 20/20, so an unfiltered top 5 was five lucky screens at 100.0%
        # sitting under a `best` of 95% — the filtered answer.
        deep = deep_rows(state['completed'])
        # Two different caveats, and depth outranks filtering.
        #
        # A flat "(full-length rows only)" was wrong whenever the abandonment gate stopped every
        # row short of the target — at gate 95 that is the normal case, and batches 19 and 20
        # both closed out with *no* full-length row in any arm. It read as a 100-episode
        # guarantee sitting on top of a 58-episode measurement.
        #
        # Depth is checked first and regardless of whether anything was filtered, because the
        # filtering test alone stays silent in the worst case: when every row is shallow *and*
        # within half the deepest, nothing is excluded, so b20b printed no caveat at all over a
        # 36-episode 83.3%. Absence of a note has to mean full length or it means nothing.
        #
        # `target_episodes` is the payload's episodes_per_checkpoint, the same source the chart's
        # full/partial split uses; it falls back to the deepest row for files predating it.
        deepest = max(row['episodes'] for row in state['completed'])
        target = state.get('target_episodes') or deepest
        if deepest < target:
            # 'ep' and no em dash: at 10pt the long form ran **57 characters** and crossed into the
            # right column on `b43d`'s live chart, printing `none full length)below 98%` as one word.
            # The left column's budget is `RIGHT_COLUMN_X` minus two characters of clearance, and
            # this branch -- every row abandoned short of the target -- is the widest thing it draws.
            note = '  (deepest {0}/{1} ep, none full)'.format(deepest, target)
        elif len(deep) != len(state['completed']):
            note = '  (full-length rows only)'
        else:
            note = ''
        lines.append('  top 5:{0}'.format(note))
        top = sorted(deep, key=lambda r: -r['perfect_percent'])[:5]
        for rank, row in enumerate(top, 1):
            lines.append('    {0}. {1:>8}  {2:>5.1f}%  (CI {3:.0f}-{4:.0f})  avg score {5}'.format(
                rank, row['step'], row['perfect_percent'],
                row['perfect_ci95'][0], row['perfect_ci95'][1], row['avg_score']))
    else:
        lines.append('  no checkpoint finished yet')
    return lines


def flight_lines(state):
    """In-flight progress as text.

    **Not drawn on the chart** since 2026-08-19 -- the top panel plots the same checkpoints' running
    rate per round, which is the same information at higher resolution, so the text block was
    spending five of its lines restating the panel above it. Kept for `report`, which prints to a
    terminal and has no panel.
    """
    lines = []
    if state['active']:
        lines.append('  in flight:')
        for run, flight in sorted(state['active'], key=lambda p: p[1]['step']):
            # Worker count included here since 2026-08-07, when the chart's worker axis came out:
            # it is the only remaining place the number appears other than the x-axis label, and it
            # is what makes "round 3/10" convertible into episodes.
            lines.append('    {0:<9} {1:>8}  round {2}/{3} x {4}w  running {5:.0f}%  '
                         '({6} elapsed)'.format(
                             run['suffix'], flight['step'], flight['round'], flight['rounds_total'],
                             flight_workers(run, flight), flight['running_percent'],
                             format_duration(max(0, time.time() - flight['started_at']))))
    elif state['done'] < state['requested']:
        lines.append('  nothing in flight')

    return lines


def stale_lines(state):
    """One line per process that stopped writing while incomplete."""
    return ['  STALE: {0} silent {1}, incomplete — likely dead'.format(
        run['suffix'], format_duration(age)) for run, age in state['stale']]


def text_summary(policy_name, state):
    """The whole summary as one block, for `report`'s terminal output."""
    return '\n'.join(progress_lines(policy_name, state)
                     + ranking_lines(state)
                     + metrics_lines(state)
                     + flight_lines(state)
                     + stale_lines(state))


def fit_summary_panel(boxes, needed_in, figure_height_in):
    """Re-lay the three panels so the summary block is `needed_in` tall, keeping everything else.

    `boxes` is `[(y0, height), ...]` bottom-up -- summary, lower chart, upper chart -- in figure
    fraction. Returns the same shape.

    The frame stays **exactly** 9.5 x 8.56 in; only the split moves. That is the property the ‡ note
    on `savefig` protects (a PNG whose pixel size changes makes the viewer's window resize every
    frame, and now also makes it rebuild its figure, since the panel box is sized from the image
    aspect). So the height the summary block gives up goes to the two charts, evenly, and the gaps
    and margins are preserved: the top of the upper chart lands exactly where it was.

    Never grows the block past what the gridspec gave it -- the capacity is the worst case and the
    charts should keep anything the text does not need.
    """
    (t0, th), (c2_0, c2h), (c1_0, c1h) = boxes
    if figure_height_in <= 0 or th <= 0:
        return list(boxes)
    shrink = th - (needed_in / figure_height_in)
    if shrink <= 0:
        return list(boxes)
    gap_lower = c2_0 - (t0 + th)
    gap_upper = c1_0 - (c2_0 + c2h)
    share = shrink / 2.0
    new_t = (t0, th - shrink)
    new_c2 = (new_t[0] + new_t[1] + gap_lower, c2h + share)
    new_c1 = (new_c2[0] + new_c2[1] + gap_upper, c1h + share)
    return [new_t, new_c2, new_c1]


def summary_columns(policy_name, state):
    """(left, right) text blocks for the chart's bottom panel.

    Two columns rather than one because the charts above lost 20% of their height on 2026-08-19 and
    the freed space is all width, not depth -- a single column would have been the same ~15 lines
    with more air under them. The split is by kind, not by length: **left is where this arm got to
    and what its best checkpoints are, right is the distribution and what follows from it**, so a
    glance at one column answers one question.

    In-flight lines are deliberately absent; the top panel draws them.
    """
    left = progress_lines(policy_name, state) + ranking_lines(state)
    right = metrics_lines(state) + stale_lines(state)
    return '\n'.join(left), '\n'.join(right)


def draw_threshold(axis, state):
    """Marks the abandonment gate on a perfect-% axis, so the target is visible rather than implied.

    The value comes from the payload's `min_achievable`, never a literal: the gate has been 0, 85, 90
    and 95 across this project's files, and a line hardcoded at 95 would quietly mislabel every
    earlier arm's chart. Nothing is drawn for a file with no gate, which is how batches 11 and 13
    were measured.

    Worth drawing because the gate is not a display preference — it is the rule that decides which
    checkpoints get measured at full length, so a point below the line has been abandoned and a point
    on or above it is a full-length measurement.
    """
    threshold = state.get('min_achievable')
    if not threshold:
        return
    axis.axhline(threshold, color='#2ca02c', linestyle=(0, (6, 3)), linewidth=1.1, alpha=0.9,
                 zorder=1, label='gate {0:.0f}%'.format(threshold))


# Both perfect-% panels ran 0-100 until 2026-08-19. That spent half the panel on a range nothing
# worth reading has been in for a year: the live question is whether a run sits at 93 or at 98, and
# 5 pp of difference was 5% of the axis. 50 doubles the resolution of the half that is used.
PERFECT_AXIS_FLOOR = 50.0

# Point size of the two text columns, raised from 8.5 on 2026-08-19 (chart titles are 10pt, so this
# now matches them). **It is bounded by width, not by height**: the columns are monospace blocks at
# fixed x, so the widest line in the left column has to stay clear of where the right one starts, and
# a font bump is a proportional widening of every line. Raising it needed the stage label and the
# progress bar shortened to fit; `tests/test_eval_progress.py` measures the rendered extents and
# fails if the columns would touch, which is the only reliable check -- character-width arithmetic on
# a font metric is a guess.
SUMMARY_FONTSIZE = 10.0

# matplotlib's default line spacing for a multi-line `text` artist is 1.2 x fontsize, so one text
# line of the summary block is this many inches tall. Used to size the block to its content.
SUMMARY_LINE_IN = SUMMARY_FONTSIZE * 1.2 / 72.0

# Blank lines left under the last line of the summary block. The block's *capacity* is the worst
# case -- 16 lines, sized for a screened close-out with a four-line stage block and stale processes --
# and a flat HOF pass uses about 10, so the leftover was rendering as **324 px, 17% of the frame's
# height**, at the bottom of every chart. In `chart_viewer`'s 2x2 that lands between the rows and
# reads as a broken layout.
#
# One line, not three, because the gap a reader sees is the sum of four things and only two of them
# are here: this slack, the figure's bottom margin, the last line's own descender space (~1 line
# tall, since the text box is taller than its ink), and the next panel's top margin. Three slack
# lines measured out as ~7 rendered lines between the rows.
SUMMARY_SLACK_LINES = 1

# The frame is a fixed size on every render, whatever the state -- see the ‡ note on `savefig`.
FIGURE_WIDTH_IN = 9.5
FIGURE_HEIGHT_IN = 8.56
# Where the right column starts, in axes fraction. The left column gets everything before it, and
# the measured clearance at 10pt is ~4 characters on the left and ~2 on the right -- enough for a
# step number that grows to eight digits, not enough to absorb a new line wider than the ones there.
RIGHT_COLUMN_X = 0.52


def note_clipped(axis, values, unit):
    """Writes how many points fall below the axis floor, in the panel, if any do.

    The floor is **fixed**, not adaptive, for the same reason the figure size is: an axis that
    rescales when one bad screen lands is an axis the eye has to re-read on every frame, and the
    panels are side by side precisely so two runs can be compared by height. A floor that gave way
    to the lowest point would be set by the worst 20-episode screen in the file.

    Fixed means points *are* hidden, though -- an early screen at 0%, a collapsed arm, a first round
    that misses -- and silently dropping data is the failure this project keeps paying for. So the
    count is written where the data would have been. An empty panel with "138 rows below 50%" in the
    corner says something; an empty panel says nothing.
    """
    count = sum(1 for value in values if value is not None and value < PERFECT_AXIS_FLOOR)
    if not count:
        return
    axis.text(0.008, 0.04, '{0} {1} below {2:.0f}%'.format(count, unit, PERFECT_AXIS_FLOOR),
              transform=axis.transAxes, ha='left', va='bottom', fontsize=7, color='#b0504a')


def render(policy_name, state, out_path):
    # ‡ The layout is deliberately CONSTANT: three panels at a fixed figure size, whether or not
    # anything is in flight. It used to drop to two panels and shrink from 9.0in to 6.6in as soon as
    # a checkpoint finished, which happens between every pair of checkpoints — so the live window
    # resized and both remaining charts jumped several times a minute, and reading a trend off a
    # target that keeps moving is most of what makes a live chart useless.
    #
    # The empty in-flight panel is the cost, and it is the right trade: dead space in a known place
    # beats live content in a moving one.
    #
    # Sizes as of 2026-08-19, in inches of drawn panel: the two charts gave up 20% of their height
    # (2.10 -> ~1.68 each) and the text panel gained 37% (2.00 -> ~2.75), with hspace down from 0.42
    # to 0.35. The figure is 8.56in rather than 9.0, so it is still shorter than it was even after
    # the text panel took two rises -- the first because two columns roughly halve the line count,
    # the second when the text went from 8.5pt to 10pt.
    #
    # ~2.75in is ~16 monospace lines at 10pt, plus the 0.39in bottom margin under it. That is the
    # capacity to keep: the left column runs to 13 (name, bar, a four-line stage block, best, pace,
    # and a top-5 with its header) and the right to 12 plus one line per stale process.
    #
    # None of this touches the constant-size property above: every frame is 9.5x8.56in whatever the
    # state. matplotlib takes hspace as a fraction of the *mean* panel height, so the panel inches
    # here are figure_height * (1 - top_margin - bottom_margin) / (1 + 2 * hspace / 3), split by the
    # ratios -- change one of the four numbers and the others move.
    has_flight = any((flight.get('per_round_perfect') or []) for _, flight in state['active'])
    grid = gridspec.GridSpec(3, 1, height_ratios=[1.68, 1.68, 2.75], hspace=0.35)
    figure = plt.figure(figsize=(FIGURE_WIDTH_IN, FIGURE_HEIGHT_IN))
    next_row = 0

    # --- 1. in-flight convergence: running perfect rate vs round -------------------------
    top = figure.add_subplot(grid[next_row])
    next_row += 1
    if has_flight:
        for run, flight in sorted(state['active'], key=lambda p: p[1]['step']):
            per_round = flight.get('per_round_perfect') or []
            if not per_round:
                continue
            workers = flight_workers(run, flight)
            running, seen, won = [], 0, 0
            for count in per_round:
                won += count
                seen += workers
                running.append(100.0 * won / seen)
            rounds = range(1, len(running) + 1)
            # Strip the leading underscore: matplotlib treats any label starting with '_' as
            # private and silently leaves it out of the legend. Every suffix used here is of
            # the form '_r0' / '_midrun', so labelling them raw produced an empty legend and a
            # "No artists with labels" warning on every render.
            label = run['suffix'].lstrip('_') or 'run'
            line, = top.plot(rounds, running, marker='o', markersize=3.5, linewidth=1.4,
                             label='{0} @{1}'.format(label, flight['step']))
            # The current rate, written next to the last point. Reading it off the y axis is
            # awkward on a 2.1in-tall panel — the gridlines are 20 pp apart and the whole question
            # while a checkpoint is in flight is "what is it running at right now".
            top.annotate('{0:.0f}%'.format(running[-1]),
                         xy=(len(running), running[-1]),
                         xytext=(5, 0), textcoords='offset points',
                         va='center', ha='left', fontsize=8, fontweight='bold',
                         color=line.get_color(), zorder=5)
            # +1.5 rather than +0.5: the running-rate annotation is drawn to the right of the
            # last point and would be clipped at the axes edge on a nearly finished checkpoint.
            top.set_xlim(0.5, flight['rounds_total'] + 1.5)
        # Worker counts are still needed for the x-axis formula below, but they are no longer drawn.
        # There used to be a second y axis here carrying a step line of how many workers were still
        # contributing at each round. It came out on 2026-08-07: for a single process — which is
        # every run except a deliberate shard set — it is a flat line at EVAL_WORKERS that steps down
        # once at the end, so it spent a whole axis, a colour and a legend entry restating a constant
        # that the x-axis label already names. The count is in the in-flight text lines too.
        depths_and_workers = []
        for run, flight in state['active']:
            per_round = flight.get('per_round_perfect') or []
            if not per_round:
                continue
            depths_and_workers.append((len(per_round), flight_workers(run, flight)))

        draw_threshold(top, state)

        if top.lines:
            top.legend(fontsize=7, loc='lower right', ncol=2, framealpha=0.85)
        top.set_title(in_flight_title(state['active']), fontsize=10)
        # EVAL_WORKERS varies run to run (2 and 10 have both been used this project), so the old
        # hardcoded "10 rounds x 10 workers = 100 episodes" was only ever right by coincidence.
        worker_counts = sorted({workers for _, workers in depths_and_workers})
        if len(worker_counts) == 1:
            episode_formula = 'rounds x {0} worker{1} = episodes'.format(
                worker_counts[0], '' if worker_counts[0] == 1 else 's')
        else:
            episode_formula = 'rounds x workers = episodes (worker count varies by process)'
        top.set_xlabel('round  (one episode per worker, so {0})'.format(episode_formula),
                       fontsize=8)
        top.set_ylabel('running perfect %', fontsize=8)
    else:
        # Nothing in flight: keep the panel, its axes and its scale so the figure does not move.
        # Titled and labelled the same way, so the eye lands in the same place when work resumes.
        top.set_title('In flight: running perfect rate by round', fontsize=10)
        top.set_xlabel('round  (one episode per worker)', fontsize=8)
        top.set_ylabel('running perfect %', fontsize=8)
        top.set_xlim(0.5, 11.5)
        top.text(0.5, 0.5, 'no checkpoint in flight', transform=top.transAxes,
                 ha='center', va='center', fontsize=10, color='#999999')
    # Every running rate drawn in the panel, so a checkpoint running below the floor still appears.
    # The idle panel passes nothing and lands on the plain floor, which keeps the two identical --
    # the eye must not have to re-read the axis when work resumes.
    running_rates = []
    for run, flight in state['active']:
        per_round = flight.get('per_round_perfect') or []
        if not per_round:
            continue
        workers = flight_workers(run, flight)
        won = seen = 0
        for count in per_round:
            won += count
            seen += workers
            running_rates.append(100.0 * won / seen)
    note_clipped(top, running_rates, 'round' if len(running_rates) == 1 else 'rounds')
    top.set_ylim(PERFECT_AXIS_FLOOR, 100)
    top.grid(alpha=0.25, linestyle=(0, (4, 3)), linewidth=0.5)
    top.tick_params(labelsize=7)

    # --- 2. completed checkpoints by step ------------------------------------------------
    middle = figure.add_subplot(grid[next_row])
    next_row += 1
    if state['completed']:
        # ‡ Split at **full length**, not at half the deepest row. Solid blue means "measured to the
        # target"; hollow grey means anything less, whether that is a 20-episode screen or a row the
        # abandonment gate stopped at 60.
        #
        # The old rule was `episodes >= deepest / 2`, which was written when the only two depths were
        # a 20-episode screen and a 100-episode confirmation, and half-of-deepest separated them while
        # tolerating a worker count that rounded 100 up to 108. Under `EVAL_MIN_ACHIEVABLE` that stopped
        # being true: the gate leaves rows at every depth from the abandon floor upward, so a row
        # abandoned at 60 of 100 drew as a solid confirmed point. On b18a, 43 of 100 rows sat between
        # 50 and 96 episodes — every one of them mismarked as fully measured.
        #
        # `target_episodes` comes from the payload; `deepest` is the fallback for files that predate
        # the field, where it reproduces the old behaviour for a flat run and is still right.
        target = state.get('target_episodes')
        deepest = max(r['episodes'] for r in state['completed'])
        full_length = target or deepest
        full = [r for r in state['completed'] if r['episodes'] >= full_length]
        partial = [r for r in state['completed'] if r['episodes'] < full_length]
        if partial:
            shallowest = min(r['episodes'] for r in partial)
            deepest_partial = max(r['episodes'] for r in partial)
            middle.scatter([r['step'] / 1000.0 for r in partial],
                           [r['perfect_percent'] for r in partial],
                           s=9, alpha=0.35, facecolors='none', edgecolors='#8c8c8c',
                           linewidths=0.7, zorder=2,
                           label=('screened/abandoned, {0}-{1} ep'.format(
                               shallowest, deepest_partial) if deepest_partial != shallowest
                               else 'screened, {0} ep'.format(shallowest)))
        middle.scatter([r['step'] / 1000.0 for r in full],
                       [r['perfect_percent'] for r in full],
                       s=22, alpha=0.75, color='#1f77b4', zorder=3,
                       label='full, {0} ep'.format(full_length) if partial else None)
        middle.axhline(state['pooled'], color='#666666', linestyle=(0, (5, 3)), linewidth=1.0,
                       label='pooled {0:.1f}%{1}'.format(
                           state['pooled'],
                           ' eq' if state.get('pooled_is_equal_effort') else ''), zorder=2)
        best = state['best']
        middle.scatter([best['step'] / 1000.0], [best['perfect_percent']], s=90, zorder=4,
                       facecolors='none', edgecolors='#d62728', linewidths=1.6,
                       label='best {0:.0f}% @{1}k'.format(best['perfect_percent'], best['step'] // 1000))
        draw_threshold(middle, state)
        middle.legend(fontsize=7, loc='lower right', framealpha=0.85)
    else:
        # Otherwise matplotlib autoscales to 0.0-1.0 "thousands of steps", which reads as real
        # data at a glance and is not.
        middle.text(0.5, 0.5, 'no checkpoint finished yet', fontsize=9, color='#888888',
                    ha='center', va='center', transform=middle.transAxes)
        middle.set_xticks([])
    stages = state.get('stages')
    if stages and stages['current'] != 'done':
        title = 'Completed {0} ({1} of {2}) — stage {3}/3, {4}'.format(
            state['unit'], state['done'], state['requested'],
            stages['order'].index(stages['current']) + 1, stages['current'])
    else:
        title = 'Completed {0} ({1} of {2})'.format(
            state['unit'], state['done'], state['requested'])
    middle.set_title(title, fontsize=10)
    middle.set_xlabel('checkpoint step (thousands)', fontsize=8)
    middle.set_ylabel('measured perfect %', fontsize=8)
    note_clipped(middle, [r['perfect_percent'] for r in state['completed']],
                 'row' if len(state['completed']) == 1 else 'rows')
    middle.set_ylim(PERFECT_AXIS_FLOOR, 100)
    middle.grid(alpha=0.25, linestyle=(0, (4, 3)), linewidth=0.5)
    middle.tick_params(labelsize=7)

    # --- 3. the same numbers as text, so the image is self-contained ---------------------
    bottom = figure.add_subplot(grid[next_row])
    bottom.axis('off')
    # Two independent text artists rather than one string with padding: a monospace column is only
    # aligned if nothing before it on the line can change width, and the left column's step numbers
    # and percentages do change width between frames. Separate artists at fixed x cannot drift.
    left_text, right_text = summary_columns(policy_name, state)
    bottom.text(0.0, 1.0, left_text, fontsize=SUMMARY_FONTSIZE, family='monospace',
                va='top', ha='left', transform=bottom.transAxes)
    bottom.text(RIGHT_COLUMN_X, 1.0, right_text, fontsize=SUMMARY_FONTSIZE, family='monospace',
                va='top', ha='left', transform=bottom.transAxes)

    figure.suptitle('{0} — eval progress {1}'.format(
        policy_name, time.strftime('%m-%d %H:%M:%S')), fontsize=11)
    partial = out_path + '.partial.png'
    # ‡ No bbox_inches='tight' here, deliberately. Tight crops to the drawn content, so the PNG's
    # pixel dimensions changed whenever a label, legend or the text block changed width — and the
    # live window sizes itself from the image, so it kept resizing even at a fixed figsize. Fixed
    # margins instead: every frame is exactly 9.5x8.56in, so the window holds still and each panel
    # stays where it was in the previous frame.
    # top=0.925 (was 0.945) leaves room between the suptitle and the top panel's own title — at
    # 0.945 the "In flight: ..." title crowded right up against the "— eval progress" suptitle.
    # bottom=0.02 (was 0.045): nothing is drawn below the summary block -- it is a bare text axes --
    # so this margin is pure gap in `chart_viewer`'s grid. What is left is the cushion for the one
    # state `fit_summary_panel` cannot help, a block whose content exceeds the gridspec's 16-line
    # capacity and overflows downward; 0.02 of 8.56in is about one line of it.
    figure.subplots_adjust(left=0.07, right=0.985, top=0.925, bottom=0.02)
    # SNEK_EVAL_CHART_DPI (default 110) is the render resolution. chart_viewer.py only *magnifies*
    # this PNG, so on a HiDPI (Retina) panel 110 dpi is blown up ~2x and looks soft — the training
    # chart is crisp on the same laptop because under_the_hood renders it at 200 dpi. eval_checkpoints
    # raises this to 220 on the laptop so the source carries enough pixels to stay sharp at the *same*
    # window size (chart_viewer's window is fixed by figure_dims, not by the source px). The dpi is
    # still one constant per process, so a standalone window is the same size on every frame — only
    # larger. The 1x desktop display keeps the 110 default (smaller PNGs, no memory cost).
    # The text panel has no y axis, so the 0.07 left margin the two charts need is 0.66in of dead
    # width it can spend on the columns instead. Set *after* subplots_adjust, which would otherwise
    # overwrite it, and to constants rather than anything state-dependent so the block stays put.
    #
    # Its *height* is fitted to the text it actually holds, and hands the difference to the two
    # charts. The gridspec ratio is the worst case (16 lines: a four-line stage block, a top-5, and a
    # line per stale process), a flat HOF pass writes about 10, and the leftover was rendering as
    # 324 px of blank -- 17% of the frame -- under the last line. In a 2x2 viewer window that is the
    # band between the rows. `fit_summary_panel` keeps the frame size, the gaps and the top of the
    # upper chart fixed, so nothing about the constant-size property changes.
    lines = max(left_text.count('\n'), right_text.count('\n')) + 1 + SUMMARY_SLACK_LINES
    boxes = [(box.y0, box.height) for box in (bottom.get_position(), middle.get_position(),
                                              top.get_position())]
    fitted = fit_summary_panel(boxes, lines * SUMMARY_LINE_IN, FIGURE_HEIGHT_IN)
    for axes, (y0, height) in zip((bottom, middle, top), fitted):
        box = axes.get_position()
        axes.set_position([box.x0, y0, box.width, height])
    text_box = bottom.get_position()
    bottom.set_position([0.02, text_box.y0, 0.985 - 0.02, text_box.height])

    dpi = int(os.environ.get('SNEK_EVAL_CHART_DPI', '110'))
    figure.savefig(partial, dpi=dpi)
    plt.close(figure)
    os.replace(partial, out_path)


def live_frame(policy_name, out_path=None, include_all=False, suffixes=None):
    """Renders the progress chart and returns it as an RGB array, or None if there is nothing yet.

    For the live window eval_checkpoints.py opens. Costs ~0.1s a frame, which is why it can be
    refreshed every round rather than once per checkpoint.

    Reads the PNG back instead of pulling the figure canvas, so the window and the saved chart are
    guaranteed to be the same image.

    render() writes a **fixed-size** frame — three panels at 9.5x8.56in and 110dpi with explicit
    margins, no bbox_inches='tight' — so every frame has identical pixel dimensions and the window
    does not resize between refreshes. That matters more than the trimmed whitespace tight used to
    save: the panel count used to drop from three to two between checkpoints, and the window jumped
    several times a minute.
    """
    runs = load_runs(policy_name, include_all=include_all, suffixes=suffixes)
    if not runs:
        return None
    if out_path is None:
        out_path = os.path.join(EVALS_DIR, '{0}_eval_progress.png'.format(policy_name))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    render(policy_name, summarize(runs), out_path)
    return imageio.imread(out_path)[:, :, :3]


def report(policy_names, include_all=False):
    for policy_name in policy_names:
        runs = load_runs(policy_name, include_all=include_all)
        if not runs:
            print('{0}: no eval result files'.format(policy_name))
            continue
        state = summarize(runs)
        print(text_summary(policy_name, state))
        print('  from {0} file(s): {1}'.format(
            len(runs), ', '.join(sorted(r['suffix'] for r in runs))))
        out_path = os.path.join(EVALS_DIR, '{0}_eval_progress.png'.format(policy_name))
        os.makedirs(EVALS_DIR, exist_ok=True)
        render(policy_name, state, out_path)
        print('  chart: {0}'.format(out_path))
        print()


def main(argv):
    policy_names = argv[1:] or discover_policies()
    if not policy_names:
        print('no policies with eval results in {0}'.format(RUNS_DIR))
        return 1

    include_all = os.environ.get('EVAL_PROGRESS_ALL', '0') not in ('0', '', 'false', 'False')
    watch = float(os.environ.get('EVAL_PROGRESS_WATCH', 0))
    # One window per policy, for attaching to evals that are already running — an eval started
    # before this existed, or one launched by something that could not open a window itself.
    # eval_checkpoints.py draws its own window now, so this is the catch-up path, not the norm.
    window_mode = os.environ.get('EVAL_PROGRESS_WINDOW_MODE', '0') not in ('0', '', 'false', 'False')
    screens = {}

    def draw_windows():
        if not window_mode:
            return
        import pyformulas
        for name in policy_names:
            try:
                frame = live_frame(name, include_all=include_all)
                if frame is None:
                    continue
                if name not in screens:
                    screens[name] = pyformulas.screen(
                        np.zeros(frame.shape[:2], dtype=np.uint8),
                        '{0} eval progress'.format(name))
                # cv2 reads three channels as BGR; matplotlib produced RGB.
                screens[name].update(frame[:, :, ::-1])
            except Exception as error:
                print('  {0}: window unavailable ({1}: {2})'.format(
                    name, type(error).__name__, error))
                screens[name] = None

    if not watch:
        report(policy_names, include_all)
        draw_windows()
        return 0

    # Watch mode exits on its own once every policy is complete, so it can be left running
    # without becoming a process someone has to remember to kill.
    while True:
        report(policy_names, include_all)
        draw_windows()
        outstanding = False
        for name in policy_names:
            state = summarize(load_runs(name, include_all=include_all))
            if state['done'] < state['requested'] or state['active']:
                outstanding = True
        if not outstanding:
            print('all runs complete')
            return 0
        time.sleep(watch)


if __name__ == '__main__':
    sys.exit(main(sys.argv))
