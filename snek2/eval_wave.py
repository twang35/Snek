"""One controller for a whole wave of evaluations.

    PYTHONPATH=. python -u eval_wave.py top50 <policy> [<policy> ...]
    PYTHONPATH=. python -u eval_wave.py top50 <batch>            # b44 -> its four arms
    PYTHONPATH=. python -u eval_wave.py above:98 <policy> [<policy> ...]
    PYTHONPATH=. python -u eval_wave.py 1000 2000 3000 <policy>
    PYTHONPATH=. python -u eval_wave.py --chain top50 <batch>    # close-out, then the HOF re-measure

## What this replaces, and why

A close-out or a HOF re-measure has always been four `eval_checkpoints.py` processes started at the
same moment, one per arm — on the laptop by a bash loop, on the desktop by four synthesized jobs.
Nothing represented the *wave*, so nothing could move work between arms, and an arm that ran out
early left its share of the machine idle. Measured over every result file in `runs/`: a HOF wave
gives up **1.7-2.8x** to that (b40's arms had 16, 63, 9 and 2 candidates), and a close-out 1.0-1.2x
normally but up to 3x when arms diverge.

So the unit of work here is one **(policy, checkpoint, stage)** measurement, and lanes take whichever
unit is next regardless of which arm it belongs to. Per-unit cost is ~86 s for a HOF checkpoint and
~35 s for a close-out one, so greedy claiming lands within one unit of perfect balance.

## Shape

    eval_wave.py  (this process — no TensorFlow)
      |- lane 0  thread -> IndependentWorkerPool(EVAL_WORKERS) -> spawned TF workers
      |- lane 1  thread -> ...

A lane is a **thread**, not a process, because `eval_workers.IndependentWorkerPool` already *is* the
"group of independent workers with load/run" abstraction — so there is no IPC protocol to design and
the controller stays the single writer of every file. The thread only ever blocks on a
`multiprocessing.Queue`, so the GIL is irrelevant. This process imports no TensorFlow: planning,
row construction and the charts need only `json`, `numpy` and `matplotlib`, which is what keeps the
controller a light parent instead of a fifth ~230 MB arena.

**Nothing is built at module scope, and that is load-bearing.** `spawn` re-imports `__main__` in
every worker, so a module-level pool construction starts building pools *inside* the workers —
measured as a two-minute hang leaving eight orphaned `spawn_main` helpers, with the real error
visible only in the children.

## What it does not change

Every artifact stays exactly where it was, per policy and byte-compatible:
`runs/<policy>_checkpoint_evals<suffix>.json` with the same payload keys (built by the one
definition, `eval_plan.build_payload`) and `evals/<policy>_eval_progress.png`. That is what keeps
`eval_progress`, `select_checkpoints_above`, `run_report`, `refresh_charts.sh`, the desktop's publish
globs and every tuning doc untouched — and it is what makes this reviewable, because the numbers must
not move. The gate, the stage plan and the selection thresholds are `eval_plan`'s, unchanged.

`EVAL_RENDER` and the batched `ParallelPyEnvironment` path stay in `eval_checkpoints.py`, which
remains the way to watch a single policy or debug one by hand.

## `--chain`

Runs the HOF re-measure in this same process once the close-out is done: `above:98` out of each arm's
own close-out file, 500 episodes, flat, into `_hof500`. The recipe and the gate come from
`eval_plan.hof_settings`, which is now their only definition — the desktop daemon and the laptop's
bash script each had a copy, and the whole point of stage B living here is that the copies go away
along with the `training -> closeout -> hof` job graph. An arm whose close-out did not come out
`complete` is skipped rather than re-measured out of a truncated file; an arm with no checkpoint at
98% contributes no work and is not an error, which is the normal outcome for most arms.

Design and phases: [`plans/eval-wave-controller.md`](plans/eval-wave-controller.md).
"""
import collections
import json
import os
import queue as queuemod
import sys
import threading
import time

os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')   # nothing here draws; workers set their own
os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')   # a bare pygame.init() would open CoreAudio

import policy_arch
from eval_plan import (
    ALWAYS_FULL_SINGLE,
    CHART_MIN_INTERVAL,
    DEFAULT_ABANDON_FLOOR,
    DEFAULT_ABOVE_THRESHOLD,
    DEFAULT_CONFIRM_COUNT,
    DEFAULT_COUNT,
    DEFAULT_MIN_ACHIEVABLE,
    HOF_EPISODES,
    HOF_GATE,
    PayloadSpec,
    RowCache,
    WriteGate,
    archive_existing_eval_pngs,
    backup_previous_results,
    best_full_length_row,
    build_payload,
    build_row,
    equal_effort_pooled,
    hof_settings,
    load_finished_results,
    make_abandon_test,
    pick_finalists,
    plan_stages,
    protocol_from_sources,
    resolve_screen_episodes,
    resume_suffixes,
    select_checkpoints_above,
    select_top_checkpoints,
    wilson_interval,
    write_payload,
)
from snake_constants import EVALS_DIR, POLICY_DIR, RUNS_DIR

# Stage order, and it is the *only* thing that orders the queue: a `confirm` ranks the screens it
# follows, so the stages cannot interleave. Within one stage the arms rotate instead — see
# `interleave_by_arm`, and note that this is a change from the single-policy CLI, where a lone arm's
# rows necessarily arrive in step order. `confirm` units do not exist until that policy's screens
# have all landed (see Wave.issue_confirms). `flat` is the whole plan when screening is off, so it
# sits where the screens would.
STAGES = ('full', 'screen', 'flat', 'confirm')

# How many finished measurements a per-arm ETA is priced on. **Wall clock between completions**, not
# lane-seconds per measurement, and that is the whole point: an arm holding one of four lanes
# completes a measurement every unit-time, and the same arm holding all four completes one every
# quarter of that. So the ratio the old arithmetic had to guess -- how much of the box is this arm
# actually getting -- is measured rather than assumed, and it re-prices itself as siblings finish and
# hand their lanes over. Ten is long enough to average out one slow checkpoint and short enough to
# follow a lane handover within a few minutes of it happening.
ETA_WINDOW = 10

# Seconds between recomputations of the **wave** ETA. Far coarser than `WRITE_MIN_INTERVAL` because
# this is the one number in the payload whose inputs are the *plans* rather than the samples: it
# answers "when is the box free" in hours, and a value 30 s old is indistinguishable from a fresh
# one. It was also the expensive half of a write -- `wave_eta_seconds` builds **all four** arms'
# payloads to price the wave, so it paid four row-rebuild passes on top of the one `write_payload`
# needs, and it paid them once per round.
ETA_MIN_INTERVAL = 30.0


def parse_selector(tokens):
    """`(kind, value, rest)` from the leading argv tokens, in the same spelling as the CLI.

    Spelled `top50` / `above:98` rather than `--top 50`, deliberately identical to
    `eval_checkpoints.py`'s: that one avoids `--flags` because tf_agents routes argv through absl,
    and having the two CLIs read the same is worth more than this one being free to differ.
    """
    if not tokens:
        raise SystemExit(__doc__)
    head = tokens[0]
    if head.startswith('top'):
        rest = head[len('top'):].lstrip(':=')
        return 'top', (int(rest) if rest else DEFAULT_COUNT), tokens[1:]
    if head.startswith('above'):
        rest = head[len('above'):].lstrip(':=')
        return 'above', (float(rest) if rest else DEFAULT_ABOVE_THRESHOLD), tokens[1:]
    steps, index = [], 0
    while index < len(tokens) and tokens[index].isdigit():
        steps.append(int(tokens[index]))
        index += 1
    if not steps:
        raise SystemExit('unrecognised selector {0!r}: expected top<N>, above:<pct> or a list of '
                         'checkpoint steps'.format(head))
    return 'explicit', steps, tokens[index:]


def available_steps(ckpt_dir):
    return {int(f[len('ckpt-'):].split('.')[0])
            for f in os.listdir(ckpt_dir) if f.startswith('ckpt-') and f.endswith('.index')}


def arms_for_prefix(prefix):
    """Every policy directory belonging to `prefix`, the tested replacement for the chain script's
    `ls -d savedPolicies/<prefix>[a-z]-*`.

    Matches on the batch id rather than `startswith`, because `'b44a-x'.startswith('b4')` is true —
    the same trap already fixed once in `chart_viewer.live_arms`.
    """
    found = []
    for name in sorted(os.listdir(POLICY_DIR)):
        if not os.path.isdir(os.path.join(POLICY_DIR, name)):
            continue
        if batch_of(name) == prefix:
            found.append(name)
    return found


def batch_of(policy_name):
    """`b44a-lowlr7-b29b` -> `b44`; anything unnumbered -> the text up to the first `-`."""
    head = policy_name.split('-')[0]
    digits = ''
    for char in head[1:]:
        if not char.isdigit():
            break
        digits += char
    return head[0] + digits if head[:1].isalpha() and digits else head


class Arm:
    """One policy's plan, samples and output file. The controller owns every one of these; a lane
    only ever hands back a finished measurement, so this class needs no locking of its own beyond
    the one the controller holds while folding a row in."""

    def __init__(self, policy_name, suffix, num_episodes, num_workers, screen_episodes,
                 confirm_count, min_achievable, abandon_floor, resume_spec, screen_requested):
        self.policy_name = policy_name
        self.ckpt_dir = POLICY_DIR + policy_name
        self.arch = policy_arch.require_arch(self.ckpt_dir)
        self.suffix = suffix
        self.num_episodes = num_episodes
        self.out_path = os.path.join(
            RUNS_DIR, '{0}_checkpoint_evals{1}.json'.format(policy_name, suffix))
        self.chart_path = os.path.join(EVALS_DIR, '{0}_eval_progress.png'.format(policy_name))
        self.chart_suffixes = {suffix} | set(resume_suffixes(resume_spec, suffix))

        rows, resumed_steps, source_screens, partial = load_finished_results(
            policy_name, resume_suffixes(resume_spec, suffix), num_episodes)
        # A row the gate already settled is a **final** measurement, not a partial one, so it joins
        # the resumed set rather than the work list. `load_finished_results` cannot make that call --
        # it knows nothing about gates -- and without it a resumed run re-planned every abandoned
        # checkpoint: b43's HOF resume made 30 units that each cost a checkpoint restore and a round
        # of worker setup (~3 s) to re-derive an answer the controller already held, and the progress
        # bar counted work that would never be done. Applied with this run's own threshold, so a
        # resume under a *lower* gate correctly re-opens what the stricter one gave up on.
        self.settled = {}
        gate = make_abandon_test(min_achievable, num_episodes, abandon_floor)
        if gate:
            for step in sorted(partial):
                held = partial[step]
                have = len(held.get('scores') or ())
                if have and gate(int(sum(held.get('perfect') or ())), have):
                    self.settled[step] = partial.pop(step)
                    resumed_steps.add(step)
        self.resumed_by_step = {row['step']: row for row in rows}
        self.resumed_steps = resumed_steps
        self.source_screens = source_screens
        self.partial = partial
        self.screen_episodes = screen_episodes
        self.screen_requested = screen_requested
        self.confirm_count = confirm_count
        self.min_achievable = min_achievable
        self.abandon_floor = abandon_floor
        self.num_workers = num_workers
        self.samples = {}
        self.in_flight = {}
        # Almost every entry here is placed by `Wave.on_done` from the row it has already built for
        # the log line, so a completed measurement costs no rebuild at all; the lazy path in
        # `RowCache.rows` covers only the partial samples `finalise_plan` seeds from a resumed file.
        self.row_cache = RowCache()
        # Per arm rather than per wave, so a busy arm's rounds cannot starve a quiet sibling's file
        # of updates.
        self.write_gate = WriteGate()
        # Wall-clock stamps of this session's finished measurements, newest last, `ETA_WINDOW`
        # intervals' worth. Session-only on purpose: a resumed row's original timestamp says nothing
        # about the pace of the run that is now going.
        self.completions = collections.deque(maxlen=ETA_WINDOW + 1)
        self.progress = {'measurements': len(resumed_steps),
                         'session_measurements': 0, 'session_episodes': 0,
                         'session_seconds': 0.0, 'stage': 'full' if screen_episodes else 'flat',
                         'full_done': 0, 'screen_done': 0, 'confirm_done': 0,
                         'abandoned': 0, 'episodes_saved': 0}
        self.spec = None
        self.plan = None
        self.requested_steps = []
        self.selected_by = {}
        self.all_steps = []
        self.confirm_issued = False

    # ---------------------------------------------------------------- planning

    def adopt_protocol(self, skipped):
        """Continue under the protocol the *source files* record, never one inferred from the depth
        of the resumed rows — the batch-18 failure `protocol_from_sources` exists for. A mix of 20-
        and 100-episode rows is not comparable with itself, let alone with the arm it is meant to be
        compared against."""
        if not (self.screen_episodes and not self.screen_requested and self.resumed_steps):
            return
        keep, recorded = protocol_from_sources(self.source_screens)
        if keep:
            if recorded != self.screen_episodes:
                print('    {0}: continuing at the recorded screen depth of {1} rather than {2}'
                      .format(self.policy_name, recorded, self.screen_episodes))
                self.screen_episodes = recorded
                self.abandon_floor = max(self.abandon_floor, recorded)
            if len(self.source_screens) > 1:
                print('    {0}: warning: the source files disagree about the protocol ({1}); '
                      'continuing screened at {2}'.format(
                          self.policy_name, sorted(str(v) for v in self.source_screens),
                          self.screen_episodes))
            print('    {0}: resuming a screened arm at {1} episodes ({2} full-length rows carried '
                  'over)'.format(self.policy_name, self.screen_episodes, len(skipped)))
        else:
            print('    {0}: screening off — the source files record a flat run{1}'.format(
                self.policy_name, '' if keep is False else ' (or do not record one)'))
            self.screen_episodes = 0

    def finalise_plan(self, requested_steps, selected_by):
        """Fixes the arm's step list and stage plan, and builds the immutable payload spec."""
        skipped = [s for s in requested_steps if s in self.resumed_steps]
        self.requested_steps = [s for s in requested_steps if s not in self.resumed_steps]
        self.selected_by = selected_by
        self.adopt_protocol(skipped)
        self.partial = {step: held for step, held in self.partial.items()
                        if step in set(self.requested_steps)}
        self.samples = {step: dict(held) for step, held in self.partial.items()}
        # Settled rows are not work, but they are still results: `rows()` unions the samples with the
        # resumed rows, and `load_finished_results` returned these as partials rather than rows, so
        # dropping them here would delete them from the file on the next write.
        self.samples.update({step: dict(held) for step, held in self.settled.items()})
        # `samples` was just replaced wholesale, so nothing cached against the old one is valid.
        # Empty in practice — no write has happened yet — but the cache's invariant is that it is
        # cleared wherever `samples` is rebound and updated wherever a sample is mutated, and half an
        # invariant is how a cache goes quietly stale.
        self.row_cache.clear()
        self.all_steps = sorted(set(self.resumed_steps) | set(self.requested_steps))
        self.plan = plan_stages(self.requested_steps, selected_by, self.screen_episodes,
                                self.confirm_count, self.num_episodes, self.num_workers,
                                resumed=len(self.resumed_steps))
        # One confirmation's episode cost, rounded to whole rounds the way plan_stages does it, for
        # the correction applied once the screens are in and the finalists are known.
        self.whole_confirm_rounds = (
            -(-(self.num_episodes - self.screen_episodes) // self.num_workers) * self.num_workers
            if self.screen_episodes else 0)
        self.screens_expected = 0
        self.screens_done = 0
        self.spec = PayloadSpec(
            policy_name=self.policy_name, num_episodes=self.num_episodes,
            all_steps=self.all_steps, num_workers=self.num_workers,
            screen_episodes=self.screen_episodes, confirm_count=self.confirm_count,
            min_achievable=self.min_achievable, abandon_floor=self.abandon_floor,
            measurements_planned=self.plan['measurements_planned'],
            episodes_planned=self.plan['episodes_planned'],
            full_planned=len(self.plan['full']), screen_planned=len(self.plan['screened']),
            confirm_planned=self.plan['confirmed'])
        return skipped

    def _units_for(self, steps, target, stage):
        """Units for `steps`, each asking only for the episodes it still owes.

        The top-up is not an optimisation: a checkpoint the gate stopped at 60 of 100, or one a kill
        interrupted, would otherwise be restarted and throw away real episodes. Same rule the
        single-policy path applies in all three of its stages.

        **A checkpoint the gate has already settled is not a unit at all.** The stopping rule is
        arithmetic on (perfect, episodes, target, threshold) -- `make_abandon_test` -- so the
        controller can evaluate it against the stored samples without a worker. Before this, a
        resumed run made a *unit* out of every abandoned checkpoint, and the lane paid a checkpoint
        restore and a round of setup to be told what the controller already knew: b43's HOF resumed
        30 abandoned checkpoints at ~3 s each. It is small here, and it is not small on the 607-arm.
        The answer cannot change by running more episodes, which is exactly what makes the row final.

        Evaluated fresh rather than read off the row's `abandoned` flag, so a run resumed under a
        *lower* gate correctly re-opens a checkpoint the old gate had given up on.
        """
        settled = make_abandon_test(self.min_achievable, target, self.abandon_floor)
        units = []
        for step in steps:
            sample = self.samples.get(step) or {}
            scores = sample.get('scores') or ()
            have = len(scores)
            if have >= target:
                continue
            if settled and have and settled(int(sum(sample.get('perfect') or ())), have):
                continue
            units.append(Unit(self, step, target - have, stage))
        return units

    def initial_units(self):
        """The units known before anything is measured: the full-length tier, then the screens (or
        one flat pass per checkpoint when screening is off).

        Confirm units do not exist yet — `pick_finalists` ranks the screened rows against each
        other, so they cannot be issued until every screen for this arm has landed.
        """
        if not self.screen_episodes:
            return self._units_for(self.requested_steps, self.num_episodes, 'flat')
        units = self._units_for(self.plan['full'], self.num_episodes, 'full')
        screens = self._units_for(self.plan['screened'], self.screen_episodes, 'screen')
        self.screens_expected = len(screens)
        return units + screens

    def confirm_units(self):
        """Top-ups for the best screened checkpoints, and the plan correction they imply.

        A per-arm barrier by nature, which is exactly why a wave is one process: while this arm
        waits for its last screen, its lanes are on somebody else's work instead of idling.

        Ranked among the screened only — the full tier and any resumed full-length rows already have
        the measurement a confirmation slot would buy, so spending one there spends it on finished
        work. A mandatory 20/20 screen can take the count past `confirm_count`, so the plan totals
        are corrected here; `plan_stages` could not know the number, because it depends on how the
        screens actually came out.
        """
        already_full = set(self.resumed_steps) | set(self.plan['full'])
        finalists = pick_finalists(self.rows(), self.confirm_count, already_full=already_full)
        overshoot = max(0, len(finalists) - self.plan['confirmed'])
        if overshoot:
            self.plan['confirmed'] = len(finalists)
            self.spec = self.spec._replace(
                confirm_planned=len(finalists),
                measurements_planned=self.spec.measurements_planned + overshoot,
                episodes_planned=(self.spec.episodes_planned
                                  + overshoot * self.whole_confirm_rounds))
            print('    {0}: plan raised by {1} confirmation{2} for the perfect screens'.format(
                self.policy_name, overshoot, '' if overshoot == 1 else 's'))
        return self._units_for([row['step'] for row in finalists], self.num_episodes, 'confirm')

    # ---------------------------------------------------------------- results

    def rows(self):
        """Every checkpoint with episodes banked. One in flight is skipped until its episodes
        land — its running state travels in the payload's `in_flight` block instead.

        Memoised per step, so **`self.samples[step]` may be mutated only by `Wave.on_done`**, which
        pairs the mutation with a `row_cache.put`. See `eval_plan.RowCache` for why a cache here is
        worth the invariant, and what it looks like when the invariant is broken.
        """
        return self.row_cache.rows(self.samples, self.resumed_by_step, self.selected_by.get)

    def payload(self, complete):
        # One in-flight block, because that is what the format holds: a policy worked by two lanes
        # at once reports the older unit. Conservative for the ETA, and it keeps the file
        # byte-compatible with the single-policy path.
        oldest = None
        if self.in_flight:
            oldest = min(self.in_flight.values(), key=lambda state: state['queued'])
            # `queued` and the `_`-prefixed tallies are this controller's own bookkeeping; the
            # payload's block holds exactly the keys the single-policy path puts there.
            oldest = {k: v for k, v in oldest.items()
                      if k != 'queued' and not k.startswith('_')}
        return build_payload(self.spec, self.progress, self.samples, self.rows(), complete, oldest)

    def summarise(self):
        """The per-arm tail of the log, in the same shape the single-policy path prints.

        Kept identical on purpose: a wave's output is read next to the close-outs that came before
        it, and a different summary would be read as a different measurement.
        """
        results = self.rows()
        print('\n=== {0} ==='.format(self.policy_name))
        print('{0:>9}  {1:>11}  {2:>11}  {3:>8}  {4:>16}  {5:>9}'.format(
            'step', 'graph eval', 'surrounding', 'perfect', '95% CI', 'avg score'))
        for row in results:
            graph = row.get('graph_single_eval')
            near = row.get('graph_surrounding')
            print('{0:>9}  {1:>11}  {2:>11}  {3:>7}%  {4:>7}-{5:<7}  {6:>9}{7}'.format(
                row['step'],
                '-' if graph is None else '{0:.0f}%'.format(graph),
                '-' if near is None else '{0:.1f}%'.format(near),
                row['perfect_percent'],
                row['perfect_ci95'][0], row['perfect_ci95'][1], row['avg_score'],
                '  abandoned at {0}'.format(row['episodes']) if row.get('abandoned') else ''))
        if self.min_achievable and self.progress.get('abandoned'):
            planned = self.progress['session_episodes'] + self.progress['episodes_saved']
            print('abandoned {0} checkpoints that could no longer reach {1}%, saving {2} of {3} '
                  'episodes ({4:.0f}% of this arm\'s planned work in this session)'.format(
                      self.progress['abandoned'], self.min_achievable,
                      self.progress['episodes_saved'], planned,
                      100.0 * self.progress['episodes_saved'] / max(1, planned)))
            print('    None of them could have reached the gate arithmetically, so no ranking '
                  'changed.')
        if self.screen_episodes:
            perfect, episodes, count = equal_effort_pooled(self.samples, self.screen_episodes)
        else:
            perfect = sum(r['perfect_games'] for r in results)
            episodes = sum(r['episodes'] for r in results)
            count = len(results)
        if episodes:
            low, high = wilson_interval(perfect, episodes)
            print('pooled{0}: {1}/{2} = {3:.1f}%  (95% CI {4:.1f}-{5:.1f}%)  over {6} '
                  'checkpoints'.format(
                      ' (first {0} episodes of every checkpoint, equal effort)'.format(
                          self.screen_episodes) if self.screen_episodes else '',
                      perfect, episodes, 100.0 * perfect / episodes,
                      100.0 * low, 100.0 * high, count))
        if results:
            best = best_full_length_row(results, self.num_episodes)
            print('best checkpoint: {0} at {1}% (95% CI {2}-{3}%) over {4} episodes{5}'.format(
                best['step'], best['perfect_percent'], best['perfect_ci95'][0],
                best['perfect_ci95'][1], best['episodes'],
                '  [truncated — no checkpoint reached the abandonment gate]'
                if best['episodes'] < self.num_episodes else ''))
        print('wrote {0}'.format(self.out_path))


class Unit:
    """One (policy, checkpoint, stage) measurement — what a lane claims and runs to completion.

    Whole-checkpoint rather than a slice of one, which is what keeps `EVAL_MIN_ACHIEVABLE` exact:
    the gate counts failures against this unit's own target, the same arithmetic the single-policy
    path uses, so the property that no ranking among rows reaching the gate can change survives.
    """

    __slots__ = ('arm', 'step', 'episodes', 'stage')

    def __init__(self, arm, step, episodes, stage):
        self.arm = arm
        self.step = step
        self.episodes = episodes
        self.stage = stage

    @property
    def label(self):
        return {'full': 'full length', 'screen': 'screen', 'confirm': 'confirm',
                'flat': 'flat'}[self.stage]

    def __repr__(self):
        return '<{0} {1} {2} x{3}>'.format(self.arm.policy_name, self.step, self.stage,
                                           self.episodes)


def interleave_by_arm(units):
    """`units` ordered stage-major, then one unit per arm in rotation.

    Stage order is the part that is not negotiable: a confirmation depends on the screens it ranks,
    so `STAGES.index` leads the key. **Within a stage the order is free**, and rotating over the arms
    is strictly better than taking one arm at a time.

    Two reasons, and the second is the one that made this a bug rather than a preference.

    A wave's arms have very unequal amounts of work -- b43's HOF selected 166 / 607 / 133 / 83
    checkpoints -- so arm-major order spends its first hours entirely inside the first arm. The
    window then shows **one chart filling and three saying "nothing measured yet"**, for hours, which
    reads as three broken arms. It is also the wrong thing to have on disk: stop the run early and
    arm-major leaves one arm complete and three unmeasured, where rotation leaves four comparable
    partial measurements. Comparing the arms is the entire point of the batch.

    It costs nothing. A lane crossing to another arm's checkpoint is one `checkpoint.restore` into
    the same network, the same call a lane already makes between two checkpoints of one arm -- the
    `arch.json` check is cached per directory in the worker, so a switch measures 0.00s. Rotation
    does not change how many units exist or how long each takes; it changes only which order they
    come back in.

    Ranking is done over a *deterministically sorted* copy rather than over insertion order, so the
    result does not depend on the order `Wave` happened to build its arms in, and re-ranking a queue
    that has already had units taken from it stays stable.
    """
    ordered = sorted(units, key=lambda u: (STAGES.index(u.stage), u.arm.policy_name, u.step))
    rank, ranked = {}, []
    for unit in ordered:
        group = (unit.stage, unit.arm.policy_name)
        position = rank.get(group, 0)
        rank[group] = position + 1
        ranked.append(((STAGES.index(unit.stage), position, unit.arm.policy_name, unit.step), unit))
    ranked.sort(key=lambda row: row[0])
    return [unit for _, unit in ranked]


class WaveQueue:
    """The units waiting to be measured, handed out stage-major and then **round-robin over arms**.

    One lock, held only while a list is inspected — a lane spends ~35-90 s per unit, so contention
    is not a consideration and a simple structure is worth more than a clever one.

    Eligibility is `(restore_signature, eval-relevant env)`, not the arch alone: a worker builds one
    `SnakeEnvironment` from its process's `SNEK_*` at startup, so a lane can only measure arms whose
    observation-affecting configuration matches what it was built with. See `lane_key`.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._pending = []          # in stage-major order
        self._issued = set()        # (policy, step, stage), so no unit is ever handed out twice
        self._closed = False

    def add(self, units):
        with self._lock:
            for unit in units:
                key = (unit.arm.policy_name, unit.step, unit.stage)
                if key in self._issued:
                    continue
                self._issued.add(key)
                self._pending.append(unit)
            self._pending = interleave_by_arm(self._pending)

    def take(self, key):
        """The next unit this lane may run, or None. `key` is the lane's eligibility key."""
        with self._lock:
            for index, unit in enumerate(self._pending):
                if lane_key(unit.arm) == key:
                    return self._pending.pop(index)
            return None

    def close(self):
        with self._lock:
            self._closed = True

    @property
    def closed(self):
        with self._lock:
            return self._closed

    def __len__(self):
        with self._lock:
            return len(self._pending)


# The `SNEK_*` an eval actually depends on. `SNEK_ZERO_OBS` changes what the policy *sees*, so two
# arms that disagree about it cannot share a lane at all. The shaping and reward knobs move
# `avg_reward` only — `perfect_percent` is immune because a perfect game is identified by score,
# never by reward — but a row whose reward was measured under another arm's configuration is a number
# that has quietly stopped meaning what it says, so they partition lanes too. `SNEK_TILE_PIXELS` is
# cosmetic (verified by a fixed-seed observation hash) and `SNEK_V_MAX` never reaches eval, because
# the support comes from `arch.json` via `support_from_arch`.
EVAL_RELEVANT_ENV = ('SNEK_ZERO_OBS', 'SNEK_CHASE_SAFE_SHAPING', 'SNEK_CHASE_SAFE_GATE',
                     'SNEK_FREE_SPACE_SHAPING', 'SNEK_FREE_SPACE_GATE',
                     'SNEK_FOOD_DISTANCE_REWARD', 'SNEK_PERFECT_GAME_REWARD')


def lane_key(arm):
    """What must match for a lane to measure this arm: its restore signature, plus the process-wide
    eval-relevant environment.

    The env half is constant within one controller process today — every lane inherits the same
    `os.environ` — so it contributes nothing yet and is here to make the *rule* explicit rather than
    implicit. A wave assembled by hand out of two batches with different shaping is the case it will
    catch once per-arm env arrives with the desktop's wave job.
    """
    env = tuple((name, os.environ.get(name)) for name in EVAL_RELEVANT_ENV)
    return (policy_arch.restore_signature(arm.arch), env)


class Lane:
    """One worker pool and the thread that feeds it.

    The pool is built on the main thread and only *driven* from here: `multiprocessing` is happy
    either way, but building four pools from four threads while each imports TensorFlow in eight
    children is a needless variable in the one place this design cannot afford surprises.
    """

    def __init__(self, index, pool, policy_name, queue, key, outbox):
        self.index = index
        self.pool = pool
        self.queue = queue
        self.key = key
        # Every result leaves through here and is folded in by the main thread. Lanes never touch an
        # Arm's samples or its file: one writer per file, by construction rather than by lock.
        self._outbox = outbox
        self.units_run = 0
        self.busy_seconds = 0.0
        self.switches = 0
        self._current_policy = policy_name
        self._thread = threading.Thread(target=self._loop, name='lane-%d' % index, daemon=True)

    def start(self):
        self._thread.start()

    def join(self):
        self._thread.join()

    def alive(self):
        return self._thread.is_alive()

    def _loop(self):
        while True:
            unit = self.queue.take(self.key)
            if unit is None:
                if self.queue.closed:
                    return
                time.sleep(0.2)     # a per-arm confirm barrier: work may still appear
                continue
            try:
                self._measure(unit)
            except Exception:                                # noqa: BLE001 - reported, not hidden
                import traceback
                self._outbox.put(('error', unit, self.index, traceback.format_exc()))
                return

    def _measure(self, unit):
        arm = unit.arm
        if arm.policy_name != self._current_policy:
            # Free: the workers restore into the network they already built, guarded by
            # policy_arch.assert_same_network. Measured at 0.00s against 3.2s to build a pool.
            self.switches += 1
            self._current_policy = arm.policy_name
        started = time.time()
        restored = self.pool.load(unit.step, ckpt_dir=arm.ckpt_dir)
        if restored != unit.step:
            print('    warning: {0} global_step reads {1}, expected {2}'.format(
                arm.policy_name, restored, unit.step))

        # The stopping rule reasons about the checkpoint's *whole* sample against its target length,
        # while the pool counts only this pass — the two differ by whatever is already banked, so
        # fold the held tally in here, where it is known. Read once, before the run: the main thread
        # cannot add episodes to a step that is in flight.
        held = arm.samples.get(unit.step, {})
        already = len(held.get('scores') or ())
        held_perfect = int(sum(held.get('perfect') or ()))
        abandon = make_abandon_test(arm.min_achievable, arm.num_episodes, arm.abandon_floor)

        def should_abandon(perfect, done):
            return abandon(held_perfect + perfect, already + done)

        def on_progress(*args):
            self._outbox.put(('round', unit, self.index, args))

        scores, perfect, rewards, elapsed, abandoned = self.pool.run(
            unit.episodes, on_progress=on_progress,
            should_abandon=should_abandon if abandon else None)
        self.units_run += 1
        self.busy_seconds += time.time() - started
        self._outbox.put(('done', unit, self.index,
                          (scores, perfect, rewards, elapsed, abandoned, already, held_perfect)))


class Wave:
    """The controller: owns every arm's plan and file, the unit queue, and the lanes.

    All folding and all writing happen on this thread. Lanes only measure and report, so each
    result file has exactly one writer without a lock — which is the property that makes a shared
    wave safe to run against files the whole investigation reads.
    """

    def __init__(self, arms, lanes_wanted, workers_per_lane):
        self.arms = arms
        self.queue = WaveQueue()
        self.outbox = queuemod.Queue()
        self.lanes = []
        self.lanes_wanted = lanes_wanted
        self.workers_per_lane = workers_per_lane
        self.chart = {'last': 0.0, 'off': False}
        # The wave ETA, recomputed on `ETA_MIN_INTERVAL` rather than once per write. `None` is both
        # the initial value and what `wave_eta_seconds` returns before there is a pace to price
        # with, so a not-yet-computed value and an unknowable one are already the same thing here.
        self.eta = {'last': 0.0, 'value': None}
        self.issued = 0
        self.done = 0
        self.failures = []
        self.started = time.time()

    # ---------------------------------------------------------------- setup

    def groups(self):
        """Arms bucketed by lane eligibility, largest bucket first.

        Normally one bucket: a batch's arms share an architecture and a launch environment, so every
        lane can take anything. Two buckets means a wave was assembled out of arms that disagree,
        which is legal and simply costs the balancing.
        """
        buckets = {}
        for arm in self.arms:
            buckets.setdefault(lane_key(arm), []).append(arm)
        ordered = sorted(buckets.values(),
                         key=lambda group: -sum(len(a.requested_steps) for a in group))
        return ordered

    def lane_split(self):
        """How many lanes each eligibility group gets: proportional to the work it holds, at least
        one each, never more than `lanes_wanted` in total."""
        groups = self.groups()
        total = sum(len(a.requested_steps) for arm_group in groups for a in arm_group) or 1
        counts = []
        for index, group in enumerate(groups):
            units = sum(len(a.requested_steps) for a in group)
            left = self.lanes_wanted - sum(counts)
            spare = left - (len(groups) - index - 1)        # keep one for every group still to come
            counts.append(max(1, min(spare, int(round(self.lanes_wanted * units / float(total))))))
        return list(zip(groups, counts))

    def build_lanes(self):
        """One pool per lane, built here on the main thread (see `Lane`)."""
        from eval_workers import IndependentWorkerPool          # imports TF in its children only
        split = self.lane_split()
        if len(split) > 1:
            print('{0} eligibility groups (these arms disagree about their network or their '
                  'observation config), lanes split {1}'.format(
                      len(split), [count for _, count in split]))
        for group, count in split:
            for _ in range(count):
                first = group[0]
                pool = IndependentWorkerPool(first.policy_name, first.ckpt_dir,
                                             self.workers_per_lane)
                self.lanes.append(Lane(len(self.lanes), pool, first.policy_name, self.queue,
                                       lane_key(first), self.outbox))
        print('{0} lanes x {1} workers'.format(len(self.lanes), self.workers_per_lane))

    def enqueue_initial(self):
        for arm in self.arms:
            self.add_units(arm.initial_units())
            # A file exists from the first moment, so a wave killed in its first minute still says
            # what it was going to do rather than leaving no trace at all. Forced, because that
            # guarantee is exactly the kind a throttle would quietly withdraw.
            self.write(arm, force=True)
            if arm.screen_episodes and not arm.screens_expected:
                # Every screen came in from a resumed file, so the barrier is already satisfied and
                # nothing will arrive to trip it.
                self.issue_confirms(arm)

    def add_units(self, units):
        self.issued += len(units)
        self.queue.add(units)

    # ---------------------------------------------------------------- the loop

    def run(self):
        for lane in self.lanes:
            lane.start()
        while self.done < self.issued:
            try:
                kind, unit, lane_index, payload = self.outbox.get(timeout=5.0)
            except queuemod.Empty:
                self.refresh_charts()
                if not any(lane.alive() for lane in self.lanes):
                    print('every lane has exited with {0} of {1} measurements done'.format(
                        self.done, self.issued))
                    break
                continue
            if kind == 'round':
                self.on_round(unit, *payload)
            elif kind == 'done':
                self.on_done(unit, *payload)
            else:
                self.on_error(unit, lane_index, payload)
        self.queue.close()
        for lane in self.lanes:
            lane.join()
        for lane in self.lanes:
            lane.pool.close()

    def on_round(self, unit, round_index, rounds_total, perfect_so_far, episodes_so_far,
                 per_round):
        """A lane's progress report, folded into the arm's `in_flight` block and written out.

        Identical keys to the single-policy path's, because `eval_progress.py` draws the live ETA
        from them by name. `queued` is this controller's own bookkeeping and is stripped before the
        payload is built.
        """
        arm = unit.arm
        state = arm.in_flight.get(unit.step)
        already = state['_already'] if state else len(
            arm.samples.get(unit.step, {}).get('scores') or ())
        held_perfect = state['_held_perfect'] if state else int(
            sum(arm.samples.get(unit.step, {}).get('perfect') or ()))
        total_perfect = held_perfect + perfect_so_far
        total_episodes = already + episodes_so_far
        arm.in_flight[unit.step] = {
            'queued': state['queued'] if state else time.time(),
            '_already': already,
            '_held_perfect': held_perfect,
            'step': unit.step,
            'round': round_index,
            'rounds_total': rounds_total,
            'perfect_so_far': total_perfect,
            'episodes_so_far': total_episodes,
            # Just this pass, where `episodes_so_far` is the checkpoint's whole sample. The two
            # differ by `already` whenever a screened checkpoint is being topped up, and only this
            # one shares a denominator with `round`.
            'episodes_this_pass': episodes_so_far,
            'running_percent': round(100.0 * total_perfect / total_episodes, 1),
            'per_round_perfect': per_round,
            'started_at': state['started_at'] if state else time.time(),
        }
        self.write(arm)

    def on_done(self, unit, scores, perfect, rewards, elapsed, abandoned, already, held_perfect):
        arm = unit.arm
        held = arm.samples.setdefault(unit.step, {'scores': [], 'perfect': [], 'rewards': [],
                                                  'seconds': 0.0})
        if abandoned:
            held['abandoned'] = True
            arm.progress['abandoned'] += 1
            arm.progress['episodes_saved'] += max(0, unit.episodes - len(scores))
        held['scores'].extend(scores)
        held['perfect'].extend(perfect)
        held['rewards'].extend(rewards)
        held['seconds'] += elapsed
        arm.in_flight.pop(unit.step, None)

        arm.progress['stage'] = unit.stage
        arm.progress['measurements'] += 1
        key = unit.stage + '_done'
        arm.progress[key] = arm.progress.get(key, 0) + 1
        arm.progress['session_measurements'] += 1
        arm.progress['session_episodes'] += len(scores)
        arm.progress['session_seconds'] += elapsed
        arm.completions.append(time.time())
        self.done += 1

        row = build_row(unit.step, held, arm.selected_by.get(unit.step))
        # Built from the sample this method just extended, so it *is* what `rows()` would produce for
        # this step, and this is the `put` that `Arm.rows`'s invariant requires: `on_done` is the only
        # place a sample is mutated, so it is the only place the cache can go stale.
        arm.row_cache.put(unit.step, row)
        print('[{0:>4}/{1:<4}] {2:<26} {3:>8}  {4:<11} {5:>5.1f}%  {6:>4} eps  {7:>4.0f}s{8}'.format(
            self.done, self.issued, arm.policy_name, unit.step, unit.label,
            row['perfect_percent'], row['episodes'], elapsed,
            '  ABANDONED' if abandoned else ''))
        self.note_stage_done(unit)
        self.write(arm, force=True)

    def on_error(self, unit, lane_index, traceback_text):
        self.failures.append((unit, traceback_text))
        self.done += 1
        print('\nlane {0} failed on {1} and has stopped:\n{2}'.format(
            lane_index, unit, traceback_text))
        # The barrier still has to advance, or this arm's siblings wait for a screen that will never
        # arrive and the wave hangs with the other lanes idle.
        self.note_stage_done(unit)

    def note_stage_done(self, unit):
        arm = unit.arm
        if unit.stage != 'screen':
            return
        arm.screens_done += 1
        if arm.screens_done >= arm.screens_expected:
            self.issue_confirms(arm)

    def issue_confirms(self, arm):
        if arm.confirm_issued:
            return
        arm.confirm_issued = True
        units = arm.confirm_units()
        if units:
            self.add_units(units)
            print('    {0}: screens complete, confirming the best {1}'.format(
                arm.policy_name, len(units)))

    # ---------------------------------------------------------------- output

    def arm_eta_seconds(self, arm):
        """Seconds until **this arm** finishes, from its own last `ETA_WINDOW` completions.

        The measured quantity is wall clock between finished measurements, so an arm's share of the
        lanes is observed rather than assumed -- see `ETA_WINDOW`. That is what the episode-priced
        arithmetic could not do: it divides an arm's remaining lane-seconds by a process count, which
        in a wave is neither 1 nor the lane count but whatever the rotation is currently giving it, so
        `b43b-lowlr-b29a` read **37.9 h** on a wave with ~13 h left while its three siblings each read
        low. Here the same handover shows up as the intervals getting shorter, within a window.

        `None` until two measurements have landed this session, which is when there is an interval to
        average. The caller leaves whatever the file already carried in that case rather than
        publishing a number with no evidence behind it.

        Counts *measurements*, not episodes, because the interval already prices what a measurement
        costs -- including the abandonment gate cutting most of them short, which the episode path has
        to model with a separate deflator. The remaining count includes the in-flight unit, so an arm
        never reads 0 while a lane is still working on it.
        """
        stamps = list(arm.completions)
        if len(stamps) < 2 or arm.spec is None:
            return None
        left = max(0, arm.spec.measurements_planned - arm.progress['measurements'])
        if not left:
            return 0.0
        return left * (stamps[-1] - stamps[0]) / (len(stamps) - 1)

    def wave_eta_seconds(self):
        """Seconds until the **whole wave** finishes, or None before there is a pace to price with.

        Not a substitute for the per-arm number, which is on every panel -- this is the answer to a
        different question, "when is the box free", and it is the one that cannot be read off any
        single arm's file. Remaining lane-seconds is summed across the arms -- each priced at its own
        stage lengths by `remaining_episodes` and deflated by its own observed abandonment rate --
        then divided by the lane count.

        Priced on the wave's pooled seconds-per-episode rather than per arm, because the arms differ
        only in how much work they have, not in what an episode costs; pooling is the larger sample.

        Episode-priced rather than window-priced, unlike `arm_eta_seconds`: the whole wave keeps every
        lane busy until the tail, so there is no lane-share to discover, and the episode plan is the
        more direct measure of what is left.
        """
        import eval_progress
        try:
            per_episode = [(arm.progress['session_seconds'], arm.progress['session_episodes'])
                           for arm in self.arms]
            seconds = sum(s for s, _ in per_episode)
            episodes = sum(e for _, e in per_episode)
            if not seconds or not episodes:
                return None
            ahead = 0
            for arm in self.arms:
                if arm.spec is None:
                    continue
                run = arm.payload(complete=False)
                left = eval_progress.remaining_episodes([run])
                if left is None:
                    return None
                ahead += left * eval_progress.expected_run_fraction([run])
            lanes = max(1, len(self.lanes))
            return ahead * (seconds / float(episodes)) / lanes
        except Exception:
            return None      # an ETA is never worth losing a measurement over

    def wave_eta_cached(self):
        """`wave_eta_seconds` behind a wall-clock gate. See `ETA_MIN_INTERVAL` for why it needs one."""
        now = time.time()
        if now - self.eta['last'] >= ETA_MIN_INTERVAL:
            self.eta['last'] = now
            self.eta['value'] = self.wave_eta_seconds()
        return self.eta['value']

    def write(self, arm, force=False):
        """`runs/<policy>_checkpoint_evals<suffix>.json`, rewritten in full.

        Rewriting is cheap next to a checkpoint's runtime and it is what makes a killed wave
        resumable: a 15-hour close-out that wrote only at the end would throw all of it away.
        `write_payload` does the `.partial` + `os.replace`, so a reader never sees a half-written
        file.

        **Rewriting it once per round is not cheap**, which is what `WRITE_MIN_INTERVAL` is for: a
        progress call from `on_round` is dropped when this arm was written less than that ago, while
        `force=True` — every completed measurement, and the plan's first write — always goes through.
        So the in-flight block advances on wall clock instead of on episode count, and **no durable
        result ever waits on the gate**, which is the property that makes the throttle safe: a
        dropped write would otherwise be a dropped measurement on a killed wave.
        """
        if not (force or arm.write_gate.due()):
            return
        arm.write_gate.record()
        eta = self.wave_eta_cached()
        for other in self.arms:
            # Stamped on every arm, not only the one being written: the charts are refreshed as a set
            # and a stale wave total on three of four panels reads as four disagreeing answers.
            other.progress['wave_eta_seconds'] = eta
            other.progress['wave_lanes'] = len(self.lanes)
            other.progress['wave_arms'] = len(self.arms)
            # The arm's own ETA, on the other hand, is per arm by definition. Left alone when there is
            # not yet an interval to average, so the previous frame's number stands rather than the
            # line disappearing and coming back.
            own = self.arm_eta_seconds(other)
            if own is not None:
                other.progress['arm_eta_seconds'] = own
                other.progress['arm_eta_window'] = min(ETA_WINDOW, len(other.completions) - 1)
        write_payload(arm.out_path, arm.payload(complete=False))
        self.refresh_charts()

    def refresh_charts(self, force=False):
        """One PNG per arm, same renderer the single-policy path uses. Throttled across the whole
        wave rather than per arm, so four arms do not multiply the cost by four.

        No live window here: `chart_viewer.py` is the way to watch, and an in-process Tk window is a
        fatal-XIO liability under memory pressure (see `training.py`).
        """
        if self.chart['off']:
            return
        now = time.time()
        if not force and now - self.chart['last'] < CHART_MIN_INTERVAL:
            return
        self.chart['last'] = now
        try:
            import eval_progress
            for arm in self.arms:
                eval_progress.live_frame(arm.policy_name, arm.chart_path,
                                         suffixes=arm.chart_suffixes)
        except Exception as error:      # a chart is never worth losing an eval over
            self.chart['off'] = True
            print('    progress charts off ({0}: {1})'.format(type(error).__name__, error))

    def finish(self):
        """Marks every arm complete that has nothing outstanding, then prints the wave's own tail.

        `complete` is the flag `select_checkpoints_above` refuses to select from when it is false, so
        an arm whose lane died must not carry it — the HOF pass would otherwise select out of a
        truncated close-out.
        """
        failed_arms = {unit.arm.policy_name for unit, _ in self.failures}
        for arm in self.arms:
            complete = arm.policy_name not in failed_arms and not arm.in_flight
            write_payload(arm.out_path, arm.payload(complete=complete))
        self.refresh_charts(force=True)
        for arm in self.arms:
            arm.summarise()

        elapsed = time.time() - self.started
        busy = sum(lane.busy_seconds for lane in self.lanes)
        print('\n{0} measurements over {1} arms in {2:.0f}s wall clock on {3} lanes'.format(
            self.done, len(self.arms), elapsed, len(self.lanes)))
        # The number this whole design exists to move: lane-seconds spent measuring against
        # lane-seconds available. Four separate processes cannot report it, because none of them can
        # see the others' idle time.
        if self.lanes and elapsed:
            print('lane utilisation {0:.0f}%  ({1} arm switches, {2})'.format(
                100.0 * busy / (elapsed * len(self.lanes)),
                sum(lane.switches for lane in self.lanes),
                ', '.join('{0}:{1}'.format(lane.index, lane.units_run) for lane in self.lanes)))
        print('\nPooled rates only compare across arms when the selection rule matches.')
        if self.failures:
            print('\n{0} measurement{1} failed; {2} left incomplete'.format(
                len(self.failures), '' if len(self.failures) == 1 else 's',
                ', '.join(sorted(failed_arms))))
        return not self.failures


def resolve_policies(tokens):
    """Policy names from the trailing argv, expanding a bare batch id into its arms.

    So `eval_wave.py top50 b44` and `eval_wave.py top50 b44a-x b44b-y ...` mean the same thing,
    which is what lets the desktop's job spec carry a batch rather than a list — and what retires
    `chain_closeout_after_training.sh`'s `ls -d savedPolicies/<prefix>[a-z]-*` glob.
    """
    policies = []
    for token in tokens:
        if token.startswith('-'):
            raise SystemExit('unrecognised argument {0!r} (this CLI takes no --flags: tf_agents '
                             'routes argv through absl, which rejects them)'.format(token))
        if os.path.isdir(os.path.join(POLICY_DIR, token)):
            policies.append(token)
            continue
        arms = arms_for_prefix(token)
        if not arms:
            raise SystemExit('no policy directory and no arms for {0!r} in {1}'.format(
                token, POLICY_DIR))
        print('{0} expands to {1} arms: {2}'.format(token, len(arms), ' '.join(arms)))
        policies.extend(arms)
    seen, unique = set(), []
    for name in policies:
        if name not in seen:
            seen.add(name)
            unique.append(name)
    return unique


def select_for(arm, kind, value, available, source_suffix=''):
    """This arm's checkpoints under the wave's one selector. Returns `(steps, selected_by)`.

    Every arm is selected by the same rule — that is what makes their pooled rates comparable, and
    the reason the selector is a wave-level argument rather than a per-arm one.

    `source_suffix` is which result file `above:` reads its candidates out of, and it is not the file
    being written: stage B of a chain selects from stage A's close-out and writes `_hof500`.
    """
    if kind == 'top':
        return select_top_checkpoints(arm.policy_name, available, value)
    if kind == 'above':
        return select_checkpoints_above(arm.policy_name, available, value,
                                        source_suffix=source_suffix)
    missing = [step for step in value if step not in available]
    if missing:
        raise SystemExit('no checkpoint for step(s) {0} in {1}'.format(missing, arm.ckpt_dir))
    return list(value), {step: {'selected_by': 'explicit'} for step in value}


def build_arms(policies, kind, value, settings):
    """One `Arm` per policy, planned and ready to enqueue. Arms with nothing to do are dropped.

    An `above:` selector finding nothing is the common case rather than a failure — most arms have
    no checkpoint that good — so it drops the arm quietly and the wave runs on whatever is left.
    """
    arms = []
    for policy_name in policies:
        arm = Arm(policy_name, settings['suffix'], settings['num_episodes'],
                  settings['num_workers'], settings['screen_episodes'],
                  settings['confirm_count'], settings['min_achievable'],
                  settings['abandon_floor'], settings['resume'], settings['screen_requested'])
        available = available_steps(arm.ckpt_dir)
        requested, selected_by = select_for(arm, kind, value, available,
                                            settings.get('source_suffix', ''))
        if not requested:
            print('{0}: nothing selected — skipping'.format(policy_name))
            continue
        skipped = arm.finalise_plan(requested, selected_by)
        if not arm.requested_steps:
            print('{0}: all {1} selected checkpoints are already measured — skipping'.format(
                policy_name, len(skipped)))
            continue
        if skipped:
            note = ('' if not arm.settled else
                    ' ({0} of them settled below the gate)'.format(len(arm.settled)))
            print('{0}: resuming, {1} already measured{2}, {3} left'.format(
                policy_name, len(skipped), note, len(arm.requested_steps)))
        if arm.partial:
            carried = sum(len(held['scores']) for held in arm.partial.values())
            print('    {0}: reusing {1} partial sample{2} ({3} episodes) rather than '
                  're-measuring them'.format(policy_name, len(arm.partial),
                                             '' if len(arm.partial) == 1 else 's', carried))
        backup_previous_results(arm.out_path)
        arms.append(arm)
    return arms


def describe(arms):
    total_units = 0
    for arm in arms:
        plan = arm.plan
        if arm.screen_episodes:
            print('{0}: {1} at full length ({2:.0f}% graph point or explicit), {3} screened at {4}, '
                  'best {5} confirmed — {6} episodes against {7} flat ({8:.2f}x)'.format(
                      arm.policy_name, len(plan['full']), ALWAYS_FULL_SINGLE,
                      len(plan['screened']), arm.screen_episodes, plan['confirmed'],
                      plan['episodes_planned'], plan['flat_episodes'],
                      plan['flat_episodes'] / plan['episodes_planned']
                      if plan['episodes_planned'] else 0))
        else:
            print('{0}: {1} checkpoints x {2} episodes, flat'.format(
                arm.policy_name, len(arm.requested_steps), arm.num_episodes))
        total_units += plan['measurements_planned']
    print('wave: {0} arms, ~{1} measurements planned'.format(len(arms), total_units))


def describe_selector(kind, value):
    if kind == 'top':
        return 'top{0}'.format(value)
    if kind == 'above':
        return 'above:{0:g}'.format(value)
    return '{0} explicit step{1}'.format(len(value), '' if len(value) == 1 else 's')


def run_stage(name, kind, value, policies, settings, lanes):
    """One wave: plan the arms, build the lanes, measure, write. Returns `(exit_code, arms)`.

    Stage A and stage B of a chain are the same function with different settings, which is the point
    of having one — the desktop's chain was two job kinds with two recipes, and the second copy is
    what drifted.
    """
    print('\n== {0} =='.format(name))
    arms = build_arms(policies, kind, value, settings)
    if not arms:
        # Clean exit, not a failure: an `above:` wave where no arm produced a good enough checkpoint
        # is the normal outcome, and the desktop must mark that job done rather than retry it.
        print('nothing left to measure')
        return 0, []
    describe(arms)
    wave = Wave(arms, lanes, settings['num_workers'])
    wave.build_lanes()
    wave.enqueue_initial()
    wave.run()
    return (0 if wave.finish() else 1), arms


def completed_policies(policies, suffix):
    """The policies whose stage-A file came out `complete`, in the order they were given.

    Read from the file rather than from this process's own arms, because an arm the wave skipped —
    everything already measured, so nothing to do — is *finished*, not failed, and its HOF stage is
    owed all the same. `complete` is the same flag the bash chain script checked by hand, and the
    same one `select_checkpoints_above` refuses to select from when it is false.
    """
    ready = []
    for policy_name in policies:
        path = os.path.join(RUNS_DIR, '{0}_checkpoint_evals{1}.json'.format(policy_name, suffix))
        try:
            with open(path) as handle:
                if json.load(handle).get('complete'):
                    ready.append(policy_name)
                    continue
        except (IOError, OSError, ValueError):
            pass
        print('    {0}: no complete result at {1} — skipping the HOF stage'.format(
            policy_name, os.path.basename(path)))
    return ready


def parse_options(tokens):
    """`(chain, rest)`. Only one option exists, and it has to be read before the selector.

    Split out from `main` so it can be tested without building a wave — and kept to `--chain` alone
    on purpose: `eval_checkpoints.py` takes no `--flags` because tf_agents routes its argv through
    absl, and every *selector* spelling here matches that CLI exactly so the two read the same. This
    controller does not go through absl, so one leading option is safe.
    """
    chain = False
    rest = list(tokens)
    while rest and rest[0] == '--chain':
        chain = True
        rest = rest[1:]
    return chain, rest


def main(argv):
    chain, tokens = parse_options(argv[1:])
    if not tokens:
        print(__doc__)
        return 1
    kind, value, rest = parse_selector(tokens)
    policies = resolve_policies(rest)
    if not policies:
        raise SystemExit('no policies given')

    # Before anything else, exactly as the single-policy path does it: this moves whatever charts are
    # at the top level of `evals/` into `evals/archive/<timestamp>/`, and it happens whether or not
    # the wave goes on to measure anything. A finished arm's chart does not come back by itself --
    # which is why the batches this wave is about to measure are exempt: a batch closed out in two
    # or three waves used to archive its own earlier waves' charts here.
    import chart_viewer as _cv
    archive_existing_eval_pngs(keep_batches={_cv.batch_prefix(p) for p in policies})
    if sys.platform == 'darwin':
        # HiDPI: chart_viewer only magnifies the PNG, so 110 dpi looks soft on a Retina panel.
        os.environ.setdefault('SNEK_EVAL_CHART_DPI', '220')
        try:
            import chart_viewer
            chart_viewer.spawn_for_eval(policies[0], watch='eval_wave.py .*{prefix}')
        except Exception as error:
            print('chart viewer skipped ({0}: {1})'.format(type(error).__name__, error))

    num_episodes = int(os.environ.get('EVAL_EPISODES', 100))
    num_workers = int(os.environ.get('EVAL_WORKERS', 4))
    # Four, because that is the measured throughput point on both hosts: 4 x 4 fills the 14 cores at
    # ~12.7 busy, and 16 spawned workers hold ~3.7 GB. The knob exists so a box with more cores can
    # say so, not because 4 needs tuning.
    lanes = int(os.environ.get('EVAL_LANES', 4))
    screen_requested = os.environ.get('EVAL_SCREEN_EPISODES')
    screen_episodes, screen_note = resolve_screen_episodes(screen_requested, num_episodes)
    if screen_note:
        print(screen_note)
    min_achievable = float(os.environ.get('EVAL_MIN_ACHIEVABLE', DEFAULT_MIN_ACHIEVABLE))
    if min_achievable and not 0 < min_achievable <= 100:
        raise SystemExit('EVAL_MIN_ACHIEVABLE={0} must be a percentage in (0, 100], or 0 to '
                         'disable early abandonment.'.format(min_achievable))
    # Never below the screen depth: equal_effort_pooled truncates to it and drops shorter rows, so a
    # lower floor would quietly delete checkpoints from the one arm-level figure meant to be
    # comparable across arms.
    abandon_floor = max(int(os.environ.get('EVAL_ABANDON_FLOOR', DEFAULT_ABANDON_FLOOR)),
                        screen_episodes)
    if min_achievable:
        print('abandoning any checkpoint that can no longer reach {0}%, once it has run {1}+ '
              'episodes (EVAL_MIN_ACHIEVABLE=0 to disable)'.format(min_achievable, abandon_floor))
    settings = {'suffix': os.environ.get('EVAL_OUT_SUFFIX', ''), 'source_suffix': '',
                'num_episodes': num_episodes, 'num_workers': num_workers,
                'screen_episodes': screen_episodes, 'screen_requested': screen_requested,
                'confirm_count': int(os.environ.get('EVAL_CONFIRM_COUNT', DEFAULT_CONFIRM_COUNT)),
                'min_achievable': min_achievable, 'abandon_floor': abandon_floor,
                'resume': os.environ.get('EVAL_RESUME')}

    code, arms = run_stage('stage A: {0}'.format(describe_selector(kind, value)),
                           kind, value, policies, settings, lanes)
    if not chain:
        return code

    # Stage B, in this same process. That is what retires the desktop's `training -> closeout -> hof`
    # job graph: the box queues one eval wave and the re-measure happens inside it, so there is no
    # `hof: pending` marker to set, nothing to forecast separately, and no second recipe to keep in
    # step with this one.
    hof = hof_settings(settings)
    ready = completed_policies(policies, settings['suffix'])
    if not ready:
        print('\nno arm produced a complete close-out — nothing to re-measure')
        return code
    hof_code, _ = run_stage(
        'stage B: above:{0:g} at {1} episodes, flat, into {2}'.format(
            HOF_GATE, HOF_EPISODES, hof['suffix'] or '<no suffix>'),
        'above', HOF_GATE, ready, hof, lanes)
    return code or hof_code


if __name__ == '__main__':
    # The guard is mandatory, not stylistic: `spawn` re-imports `__main__` in every worker, so
    # without it each of them re-runs this and starts building its own pools. Measured once as a
    # two-minute hang leaving eight orphaned `spawn_main` helpers, with the real error visible only
    # in the children.
    sys.exit(main(sys.argv))
