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
    'full': 'measuring every 100% graph point at full length',
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
            lines.append('    {0:<8} {1:>4}       none for this arm'.format(name, '-'))
            continue
        counts = '{0}/{1}'.format(done, planned)
        if done >= planned:
            detail = 'done'
            if name == 'screen':
                promoted = stages['confirm']['planned']
                cut = planned - promoted
                detail = 'done — {0} promoted, {1} ({2:.0f}%) left at screen length'.format(
                    promoted, cut, 100.0 * cut / planned)
        elif done == 0 and name != current:
            detail = 'pending'
        else:
            filled = int(20 * done / planned)
            detail = '{0:>3.0f}%  {1}{2}'.format(100.0 * done / planned,
                                                 '#' * filled, '.' * (20 - filled))
        lines.append('    {0:<8} {1:>9}  {2}'.format(name, counts, detail))
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

    planned_episodes = sum(r.get('episodes_planned') or 0 for r in runs)
    if planned_episodes and seconds_per_episode:
        remaining_episodes = max(0, planned_episodes - total_episodes)
        eta_seconds = remaining_episodes * seconds_per_episode / processes
    else:
        remaining = max(0, requested - len(completed))
        eta_seconds = (remaining * mean_seconds / processes) if mean_seconds else None

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
        'eta_seconds': eta_seconds,
        'rates': rates,
    }


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


def text_summary(policy_name, state):
    lines = []
    done, requested = state['done'], state['requested']
    bar_width = 28
    filled = int(bar_width * state['percent_done'] / 100.0)
    lines.append('{0}'.format(policy_name))
    lines.append('  [{0}{1}] {2}/{3} {4}  ({5:.0f}%)'.format(
        '#' * filled, '.' * (bar_width - filled), done, requested, state['unit'],
        state['percent_done']))

    lines.extend(stage_lines(state.get('stages')))

    if state['completed']:
        pooled_note = (' (equal effort)' if state.get('pooled_is_equal_effort')
                       else ' over {0} episodes'.format(state['episodes']))
        lines.append('  best {0:.1f}% @{1}   pooled {2:.1f}%{3}'.format(
            state['best']['perfect_percent'], state['best']['step'],
            state['pooled'], pooled_note))
        lines.append('  pace {0}/checkpoint   ETA {1}'.format(
            format_duration(state['mean_seconds']), format_duration(state['eta_seconds'])))
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
            note = '  (best depth {0} of {1} episodes — none full length)'.format(deepest, target)
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
    elif done < requested:
        lines.append('  nothing in flight')

    for run, age in state['stale']:
        lines.append('  STALE: {0} last wrote {1} ago and is incomplete — process likely died'.format(
            run['suffix'], format_duration(age)))
    return '\n'.join(lines)


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


def render(policy_name, state, out_path):
    # ‡ The layout is deliberately CONSTANT: three panels at a fixed figure size, whether or not
    # anything is in flight. It used to drop to two panels and shrink from 9.0in to 6.6in as soon as
    # a checkpoint finished, which happens between every pair of checkpoints — so the live window
    # resized and both remaining charts jumped several times a minute, and reading a trend off a
    # target that keeps moving is most of what makes a live chart useless.
    #
    # The empty in-flight panel is the cost, and it is the right trade: dead space in a known place
    # beats live content in a moving one.
    has_flight = any((flight.get('per_round_perfect') or []) for _, flight in state['active'])
    grid = gridspec.GridSpec(3, 1, height_ratios=[2.1, 2.1, 2.0], hspace=0.42)
    figure = plt.figure(figsize=(9.5, 9.0))
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
    top.set_ylim(0, 100)
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
    middle.set_ylim(0, 100)
    middle.grid(alpha=0.25, linestyle=(0, (4, 3)), linewidth=0.5)
    middle.tick_params(labelsize=7)

    # --- 3. the same numbers as text, so the image is self-contained ---------------------
    bottom = figure.add_subplot(grid[next_row])
    bottom.axis('off')
    bottom.text(0.0, 1.0, text_summary(policy_name, state), fontsize=8.5, family='monospace',
                va='top', ha='left', transform=bottom.transAxes)

    figure.suptitle('{0} — eval progress {1}'.format(
        policy_name, time.strftime('%m-%d %H:%M:%S')), fontsize=11)
    partial = out_path + '.partial.png'
    # ‡ No bbox_inches='tight' here, deliberately. Tight crops to the drawn content, so the PNG's
    # pixel dimensions changed whenever a label, legend or the text block changed width — and the
    # live window sizes itself from the image, so it kept resizing even at a fixed figsize. Fixed
    # margins instead: every frame is exactly 9.5x9.0in, so the window holds still and each panel
    # stays where it was in the previous frame.
    # top=0.925 (was 0.945) leaves room between the suptitle and the top panel's own title — at
    # 0.945 the "In flight: ..." title crowded right up against the "— eval progress" suptitle.
    figure.subplots_adjust(left=0.07, right=0.985, top=0.925, bottom=0.045)
    # SNEK_EVAL_CHART_DPI (default 110) is the render resolution. chart_viewer.py only *magnifies*
    # this PNG, so on a HiDPI (Retina) panel 110 dpi is blown up ~2x and looks soft — the training
    # chart is crisp on the same laptop because under_the_hood renders it at 200 dpi. eval_checkpoints
    # raises this to 220 on the laptop so the source carries enough pixels to stay sharp at the *same*
    # window size (chart_viewer's window is fixed by figure_dims, not by the source px). The dpi is
    # still one constant per process, so a standalone window is the same size on every frame — only
    # larger. The 1x desktop display keeps the 110 default (smaller PNGs, no memory cost).
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

    render() writes a **fixed-size** frame — three panels at 9.5x9.0in and 110dpi with explicit
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
