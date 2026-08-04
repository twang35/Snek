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

The PNG has three parts:

1. **In-flight convergence** — running perfect rate against round number, one line per
   process. A checkpoint takes ~5 minutes and 10 rounds, so this is where you see whether the
   one being measured now is heading somewhere good, and how many rounds are left.
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


def load_runs(policy_name, window=None, include_all=False):
    """Result files belonging to the *current* job for this policy.

    An arm accumulates result files across sessions — `b8f-disc9975seed2` has eight — and
    summing all of them answers the wrong question: it reports lifetime totals and
    double-counts any checkpoint measured more than once. Progress means "this job".

    A job is identified by write time: every file whose mtime is within `window` seconds of
    the newest one. That groups the parallel chunks of one run correctly, because they all
    keep writing while they work, and excludes runs from previous days. `include_all` overrides
    it for the lifetime view.
    """
    if window is None:
        window = float(os.environ.get('EVAL_PROGRESS_WINDOW', 3600))
    pattern = os.path.join(RUNS_DIR, '{0}_checkpoint_evals*.json'.format(policy_name))

    candidates = []
    for path in sorted(glob.glob(pattern)):
        suffix = os.path.basename(path).split('_checkpoint_evals')[1][:-len('.json')]
        if suffix in MERGED_SUFFIXES:
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

    if not candidates or include_all:
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
    workers = max(1, len(active))
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
        eta_seconds = remaining_episodes * seconds_per_episode / workers
    else:
        remaining = max(0, requested - len(completed))
        eta_seconds = (remaining * mean_seconds / workers) if mean_seconds else None

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
        note = '' if len(deep) == len(state['completed']) else '  (full-length rows only)'
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
            lines.append('    {0:<9} {1:>8}  round {2}/{3}  running {4:.0f}%  ({5} elapsed)'.format(
                run['suffix'], flight['step'], flight['round'], flight['rounds_total'],
                flight['running_percent'],
                format_duration(max(0, time.time() - flight['started_at']))))
    elif done < requested:
        lines.append('  nothing in flight')

    for run, age in state['stale']:
        lines.append('  STALE: {0} last wrote {1} ago and is incomplete — process likely died'.format(
            run['suffix'], format_duration(age)))
    return '\n'.join(lines)


def render(policy_name, state, out_path):
    # The in-flight panel is the point of this chart while a job is running and dead space once
    # it finishes, so it is only drawn when there is something to draw. A finished job gets a
    # two-panel figure with the room going to the results instead.
    has_flight = any((flight.get('per_round_perfect') or []) for _, flight in state['active'])
    rows = 3 if has_flight else 2
    heights = [2.1, 2.1, 2.0] if has_flight else [2.6, 2.0]
    figure = plt.figure(figsize=(9.5, 9.0 if has_flight else 6.6))
    grid = gridspec.GridSpec(rows, 1, height_ratios=heights, hspace=0.42)
    next_row = 0

    # --- 1. in-flight convergence: running perfect rate vs round -------------------------
    if has_flight:
        top = figure.add_subplot(grid[next_row])
        next_row += 1
        for run, flight in sorted(state['active'], key=lambda p: p[1]['step']):
            per_round = flight.get('per_round_perfect') or []
            if not per_round:
                continue
            workers = max(1, flight['episodes_so_far'] // max(1, flight['round']))
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
            top.plot(rounds, running, marker='o', markersize=3.5, linewidth=1.4,
                     label='{0} @{1}'.format(label, flight['step']))
            top.set_xlim(0.5, flight['rounds_total'] + 0.5)
        # How many workers are contributing at each round, as a step line on its own axis. Not
        # a section of its own — it is a one-line answer to "how much of the machine is on this
        # right now", which the perfect-rate lines cannot show on their own: several lines at
        # round 3 could be one process at 2 workers each still on its third round, or a second
        # process at 10 workers just getting started.
        #
        # Deliberately workers, not process count — a process is essentially always 1 here (see
        # EVAL_OUT_SUFFIX below for the sharded exception), while EVAL_WORKERS varies run to run,
        # so a process count is almost always a flat, uninformative line at 1. A process's
        # workers count toward round r once it has reported r rounds, so the line starts at
        # however many are working and steps down as quicker processes finish their checkpoint.
        depths_and_workers = []
        for _, flight in state['active']:
            per_round = flight.get('per_round_perfect') or []
            if not per_round:
                continue
            workers = max(1, flight['episodes_so_far'] // max(1, flight['round']))
            depths_and_workers.append((len(per_round), workers))
        if depths_and_workers:
            deepest = max(depth for depth, _ in depths_and_workers)
            per_round_workers = [sum(workers for depth, workers in depths_and_workers if depth >= r)
                                  for r in range(1, deepest + 1)]
            worker_axis = top.twinx()
            worker_axis.step(range(1, deepest + 1), per_round_workers, where='mid',
                             color='tab:gray', linewidth=1.2, linestyle=(0, (5, 2)),
                             alpha=0.8, label='workers')
            worker_axis.set_ylabel('workers evaluating', color='tab:gray', fontsize=8)
            worker_axis.tick_params(axis='y', labelcolor='tab:gray', labelsize=7)
            # Integer ticks only: a count of 2.5 workers is meaningless.
            worker_axis.set_ylim(0, max(per_round_workers) + 1)
            worker_axis.set_yticks(range(0, max(per_round_workers) + 2))
            worker_axis.grid(False)

        if top.lines:
            top.legend(fontsize=7, loc='lower right', ncol=2, framealpha=0.85)
        top.set_title('In flight: running perfect rate by round ({0} checkpoint{1}, {2} process{3})'.format(
            len(state['active']), '' if len(state['active']) == 1 else 's',
            len(state['active']), '' if len(state['active']) == 1 else 'es'), fontsize=10)
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
        top.set_ylim(0, 100)
        top.grid(alpha=0.25, linestyle=(0, (4, 3)), linewidth=0.5)
        top.tick_params(labelsize=7)
        # Keep the perfect-rate lines drawn over the worker step line, and let the twin axis
        # show through: twinx() puts the new axes on top with an opaque patch by default.
        top.set_zorder(2)
        top.patch.set_visible(False)

    # --- 2. completed checkpoints by step ------------------------------------------------
    middle = figure.add_subplot(grid[next_row])
    next_row += 1
    if state['completed']:
        # Split by measurement depth, because the two are not the same kind of number: a screened
        # point is 20 episodes with a ~5x wider interval, and drawing it identically to a
        # 100-episode point invites reading a lucky 19/20 as a real result. Hollow small markers
        # for screens, solid for full-length. With one uniform depth — every protocol before
        # screening — everything lands in the "full" series and the chart looks as it always did.
        deepest = max(r['episodes'] for r in state['completed'])
        full = [r for r in state['completed'] if r['episodes'] >= deepest / 2.0]
        screened = [r for r in state['completed'] if r['episodes'] < deepest / 2.0]
        if screened:
            middle.scatter([r['step'] / 1000.0 for r in screened],
                           [r['perfect_percent'] for r in screened],
                           s=9, alpha=0.35, facecolors='none', edgecolors='#8c8c8c',
                           linewidths=0.7, zorder=2,
                           label='screened, {0} ep'.format(min(r['episodes'] for r in screened)))
        middle.scatter([r['step'] / 1000.0 for r in full],
                       [r['perfect_percent'] for r in full],
                       s=22, alpha=0.75, color='#1f77b4', zorder=3,
                       label='full, {0} ep'.format(deepest) if screened else None)
        middle.axhline(state['pooled'], color='#666666', linestyle=(0, (5, 3)), linewidth=1.0,
                       label='pooled {0:.1f}%{1}'.format(
                           state['pooled'],
                           ' eq' if state.get('pooled_is_equal_effort') else ''), zorder=2)
        best = state['best']
        middle.scatter([best['step'] / 1000.0], [best['perfect_percent']], s=90, zorder=4,
                       facecolors='none', edgecolors='#d62728', linewidths=1.6,
                       label='best {0:.0f}% @{1}k'.format(best['perfect_percent'], best['step'] // 1000))
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

    figure.suptitle('{0} — eval progress   {1}'.format(
        policy_name, time.strftime('%m-%d %H:%M:%S')), fontsize=11)
    partial = out_path + '.partial.png'
    figure.savefig(partial, dpi=110, bbox_inches='tight')
    plt.close(figure)
    os.replace(partial, out_path)


def live_frame(policy_name, out_path=None, include_all=False):
    """Renders the progress chart and returns it as an RGB array, or None if there is nothing yet.

    For the live window eval_checkpoints.py opens. Costs ~0.1s a frame, which is why it can be
    refreshed every round rather than once per checkpoint.

    Reads the PNG back instead of pulling the figure canvas: render() saves with
    bbox_inches='tight', which trims whitespace the raw canvas keeps, so going through the file
    guarantees the window and the saved chart are the same image.
    """
    runs = load_runs(policy_name, include_all=include_all)
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
