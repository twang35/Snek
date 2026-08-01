"""Live progress view for one or more running eval_checkpoints.py runs.

Answers the two questions you actually have while an eval is running — *how is it doing* and
*how much is left* — without tailing six logs.

    cd snek2
    PYTHONPATH=. python -u eval_progress.py b8f-disc9975seed2
    PYTHONPATH=. python -u eval_progress.py            # every policy with eval results
    EVAL_PROGRESS_WATCH=30 PYTHONPATH=. python -u eval_progress.py b8f-disc9975seed2
    EVAL_PROGRESS_ALL=1 PYTHONPATH=. python -u eval_progress.py b8f-disc9975seed2

By default it reports the **current job** — every result file written within
`EVAL_PROGRESS_WINDOW` seconds (default 3600) of the most recent one, which groups the parallel
chunks of one run and excludes earlier sessions. `EVAL_PROGRESS_ALL=1` pools every file for this
policy instead, which is a lifetime view and will double-count any checkpoint measured twice.

It reads `runs/<policy>_checkpoint_evals*.json`, which `eval_checkpoints.py` rewrites after
every round, and renders `runs/<policy>_eval_progress.png` plus a text summary on stdout.

**Why this is a separate script rather than a chart inside eval_checkpoints.py.** Using more
than one core per arm means splitting its checkpoints across several processes, each with its
own `EVAL_OUT_SUFFIX` — six were used for the batch-8 close-out. Six processes each drawing
their own window would be six partial pictures of one job, and if they wrote one shared PNG
they would overwrite each other. Reading the result files from outside gives one consolidated
view and costs the eval nothing.

The PNG has three parts:

1. **In-flight convergence** — running perfect rate against round number, one line per
   process. A checkpoint takes ~5 minutes and 10 rounds, so this is where you see whether the
   one being measured now is heading somewhere good, and how many rounds are left.
2. **Completed checkpoints by step** — every finished measurement as a point, so the shape of
   the arm's good region is visible, with the best and the pooled mean marked.
3. **A text block** — top 5 checkpoints, percent complete, and an ETA computed from this run's
   own measured pace.

Text output goes to stdout as well, and that is the better one to read when checking progress
programmatically — it carries the same numbers without needing to open an image.
"""
import glob
import json
import os
import sys
import time

import matplotlib
matplotlib.use('Agg')  # no display needed; several of these may run over ssh or headless
import matplotlib.pyplot as plt
from matplotlib import gridspec

from snake_constants import RUNS_DIR

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
    remaining = max(0, requested - len(completed))
    mean_seconds = (sum(times) / len(times)) if times else None
    workers = max(1, len(active))
    eta_seconds = (remaining * mean_seconds / workers) if mean_seconds else None

    return {
        'completed': completed,
        'active': active,
        'stale': stale,
        'requested': requested,
        'done': len(completed),
        'percent_done': (100.0 * len(completed) / requested) if requested else 0.0,
        'best': max(completed, key=lambda r: r['perfect_percent']) if completed else None,
        'pooled': (100.0 * total_perfect / total_episodes) if total_episodes else 0.0,
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
    lines.append('  [{0}{1}] {2}/{3} checkpoints  ({4:.0f}%)'.format(
        '#' * filled, '.' * (bar_width - filled), done, requested, state['percent_done']))

    if state['completed']:
        lines.append('  best {0:.1f}% @{1}   pooled {2:.1f}% over {3} episodes'.format(
            state['best']['perfect_percent'], state['best']['step'],
            state['pooled'], state['episodes']))
        lines.append('  pace {0}/checkpoint   ETA {1}'.format(
            format_duration(state['mean_seconds']), format_duration(state['eta_seconds'])))
        lines.append('  top 5:')
        top = sorted(state['completed'], key=lambda r: -r['perfect_percent'])[:5]
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
                flight['running_percent'], format_duration(time.time() - flight['started_at'])))
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
        if top.lines:
            top.legend(fontsize=7, loc='lower right', ncol=2, framealpha=0.85)
        top.set_title('In flight: running perfect rate by round ({0} checkpoint{1})'.format(
            len(state['active']), '' if len(state['active']) == 1 else 's'), fontsize=10)
        top.set_xlabel('round', fontsize=8)
        top.set_ylabel('running perfect %', fontsize=8)
        top.set_ylim(0, 100)
        top.grid(alpha=0.25, linestyle=(0, (4, 3)), linewidth=0.5)
        top.tick_params(labelsize=7)

    # --- 2. completed checkpoints by step ------------------------------------------------
    middle = figure.add_subplot(grid[next_row])
    next_row += 1
    if state['completed']:
        steps = [r['step'] / 1000.0 for r in state['completed']]
        rates = [r['perfect_percent'] for r in state['completed']]
        middle.scatter(steps, rates, s=22, alpha=0.75, color='#1f77b4', zorder=3)
        middle.axhline(state['pooled'], color='#666666', linestyle=(0, (5, 3)), linewidth=1.0,
                       label='pooled {0:.1f}%'.format(state['pooled']), zorder=2)
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
    middle.set_title('Completed checkpoints ({0} of {1})'.format(state['done'], state['requested']),
                     fontsize=10)
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
        out_path = os.path.join(RUNS_DIR, '{0}_eval_progress.png'.format(policy_name))
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
    if not watch:
        report(policy_names, include_all)
        return 0

    # Watch mode exits on its own once every policy is complete, so it can be left running
    # without becoming a process someone has to remember to kill.
    while True:
        report(policy_names, include_all)
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
