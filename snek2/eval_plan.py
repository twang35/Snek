"""Everything about *which* checkpoints to measure and *how to record* the result — with no
TensorFlow anywhere in it.

## Why this module exists

`eval_checkpoints.py` imports TensorFlow at module scope, so importing any part of it costs a
~230 MB TF arena. But selection, the stage plan, the abandonment arithmetic, row construction and
the result-file payload are all pure functions of JSON and numpy. `eval_wave.py` — the wave
controller — needs exactly those and none of the TF: it plans the work, hands whole-checkpoint
units to `eval_workers.IndependentWorkerPool` lanes (which import their own TF, in their own
processes) and writes the files. Keeping this half TF-free is what lets the controller stay a
light parent instead of a fifth arena. See
[`plans/eval-wave-controller.md`](plans/eval-wave-controller.md).

**It is also the single definition of the protocol.** The thresholds and gates below were
duplicated in three places — here, `desktop/runner/runner.py`'s `CLOSEOUT_THRESHOLD`/
`HOF_THRESHOLD`, and `hyperparamTuning/scripts/chain_closeout_after_training.sh`'s
`CLOSEOUT_GATE`/`HOF_GATE`, whose own comment read *"copied from runner.py … if that changes,
change this too"*. Anything that needs a threshold imports it from here.

## This module is a verbatim extraction

Every function and constant below was moved out of `eval_checkpoints.py` unchanged, in one commit
that changed no behaviour: the suite read 29 modules / 736 tests / 0 failed before and after.
`eval_checkpoints.py` re-imports all 29 names, so it and its 90 tests are unaffected and
`eval_checkpoints.build_row` still resolves. **Do not "tidy up" that re-export** — it is the
compatibility surface, and `tests/test_eval_checkpoints.py` and `tests/test_selection_tiers.py`
both reach through it.
"""
import collections
import glob
import json
import os
import shutil
import time

import numpy as np

from snake_constants import EVALS_ARCHIVE_DIR, EVALS_DIR, RUNS_DIR


def backup_previous_results(out_path):
    """Copies an existing *complete* result file to `<out_path>.previous` before the first write.

    `write_results()` rewrites the whole file from the very first round — seconds in, not at the
    end — so a run killed early destroys a prior complete measurement at the same
    `EVAL_OUT_SUFFIX` with no warning. That cost a 246-checkpoint close-out once. A throwaway
    suffix is the real protection; this is the safety net for when that is forgotten.

    One rolling backup, not history: a second overwrite replaces the same `.previous`.
    """
    if not os.path.exists(out_path):
        return
    try:
        with open(out_path) as handle:
            payload = json.load(handle)
    except (json.JSONDecodeError, OSError):
        return
    if not payload.get('complete'):
        return
    shutil.copy2(out_path, out_path + '.previous')


def resume_suffixes(spec, own_suffix):
    """Which result-file suffixes EVAL_RESUME asks us to treat as already-measured work.

    `1`/`true` means "this run's own output file", which is the case that matters: a killed run
    is continued by relaunching the identical command with EVAL_RESUME=1. Anything else is read
    as a comma-separated list of explicit suffixes, for pulling in work done under a different
    EVAL_OUT_SUFFIX.
    """
    if spec is None or spec in ('', '0', 'false', 'False'):
        return []
    if spec in ('1', 'true', 'True'):
        return [own_suffix]
    return [s.strip() for s in spec.split(',') if s.strip()]


def resolve_screen_episodes(requested, num_episodes):
    """How long the screening stage runs, from EVAL_SCREEN_EPISODES. Returns (episodes, note).

    0 means no screening: one flat pass at `num_episodes`, which is how every arm before batch 10
    was measured. Screening is the default, so `requested` is None in the normal case.

    A screen at least as long as the full measurement has nothing left to confirm. **An explicit
    request for that is an error; the default running into it is not** — a short run, say
    `EVAL_EPISODES=20` to sanity-check one checkpoint, has nothing to gain from screening, and a
    default that made such a run fail would be a trap rather than a safeguard. So the default
    stands down with a note and the explicit request raises.
    """
    if requested is not None and requested != '':
        screen = int(requested)
        if screen and screen >= num_episodes:
            raise SystemExit(
                'EVAL_SCREEN_EPISODES={0} must be below EVAL_EPISODES={1} — a screen that already '
                'runs the full length has nothing left to confirm'.format(screen, num_episodes))
        return screen, None
    if DEFAULT_SCREEN_EPISODES >= num_episodes:
        return 0, ('screening off: EVAL_EPISODES={0} is not longer than the default {1}-episode '
                   'screen'.format(num_episodes, DEFAULT_SCREEN_EPISODES))
    return DEFAULT_SCREEN_EPISODES, None


def load_finished_results(policy_name, suffixes, num_episodes):
    """Rows from earlier runs that are already measured to this run's full episode count.

    Returns `(rows, steps, source_screens, partial)`.

    `steps` is the set to skip outright — rows already measured to `num_episodes`.

    `partial` maps step -> a `held` sample for rows measured to *less* than `num_episodes` that carry
    their per-episode results. Those are **reused, not discarded**: a resumed screen counts as
    screened, and stage 3 tops it up to full length. Before per-episode storage existed this was not
    possible — topping up meant pooling summary statistics, and the median cannot be pooled — so a run
    killed mid-screening lost every screen it had done (192 rows and 7,534 episodes on `b18a` in one
    incident). Rows from files predating the fields yield None from `held_from_row` and fall back to
    being re-measured.

    `source_screens` is the set of `screen_episodes` values the source files **recorded** — the
    protocol each was actually run under. It exists because inferring the protocol from row depths
    is wrong, and was wrong in production: see `protocol_from_sources`.

    A step appearing in two source files is loaded once — first file listed wins — because these
    are alternative records of the same frozen checkpoint, not extra samples to combine. Use
    merge_checkpoint_evals() when pooling repeat measurements is what you actually want.
    """
    rows, steps, source_screens = [], set(), set()
    partial = {}
    for suffix in suffixes:
        path = os.path.join(RUNS_DIR, '{0}_checkpoint_evals{1}.json'.format(policy_name, suffix))
        if not os.path.exists(path):
            continue
        with open(path) as handle:
            payload = json.load(handle)
        contributed = False
        for row in payload.get('results', []):
            step = row['step']
            if step in steps:
                continue
            if row.get('episodes', 0) >= num_episodes:
                rows.append(row)
                steps.add(step)
                # A full-length row supersedes any partial one carried from an earlier file.
                partial.pop(step, None)
                contributed = True
                continue
            # Shorter than full length: reusable as a partial sample when the row carries its
            # per-episode results. Deeper wins if two files hold the same step, since more episodes
            # is strictly more information about the same frozen checkpoint.
            held = held_from_row(row)
            if held is None:
                continue
            if len(held['scores']) > len(partial.get(step, {}).get('scores', [])):
                partial[step] = held
                contributed = True
        if contributed:
            # Files written before this field existed record nothing; `None` is the honest answer
            # and `protocol_from_sources` treats it as "unknown", not as "flat".
            source_screens.add(payload.get('screen_episodes'))
    return sorted(rows, key=lambda r: r['step']), steps, source_screens, partial


def protocol_from_sources(source_screens):
    """Whether a resume should keep screening, from what the source files recorded.

    **The guard.** Resume used to decide this by looking at the resumed rows: they were at full
    length, so the arm "must have been" measured flat, so screening was switched off for the rest of
    it. That inference is unsound, and it misfired on batch 18 — `b18a` and `b18d` were resumed from
    3 and 2 full-length rows that were the **stage-1 tier** of the three-stage protocol, where any
    checkpoint whose graph point read 100% is measured at full length immediately. The heuristic read
    them as a flat run and turned screening off, which cost ~3x the episodes and left those two arms
    with no `pooled_equal_effort` while their siblings had one — half the batch's seeds unusable on
    the metric that compares arms.

    The fix is to stop inferring: the payload has recorded `screen_episodes` all along.

    | recorded in sources | meaning | return |
    |---|---|---|
    | any value > 0 | the arm was screened | `True` — keep screening, at that depth |
    | only 0 | genuinely a flat run | `False` |
    | only None | predates the field | `None` — unknown, caller decides |
    | a mix of 0 and >0 | the arm is already inconsistent | `True`, and the caller warns |

    Returns `(keep_screening, depth)`, where `depth` is the screen depth to continue at when a
    source recorded one, so a resumed arm cannot silently change its own screen length either.
    """
    screened = {value for value in source_screens if value}
    if screened:
        return True, max(screened)
    if source_screens and source_screens != {None}:
        return False, 0
    return None, 0


def archive_existing_eval_pngs(keep_batches=()):
    """Moves whatever is currently in EVALS_DIR into a timestamped EVALS_ARCHIVE_DIR
    subfolder, so a new eval or batch starts from an empty folder and evals/ always shows
    only the most recently completed work.

    Safe when several processes start at once, which is the normal case for a batch: the
    move happens before this process writes anything of its own, so whichever process gets
    here first archives the previous batch's leftovers and the rest find nothing left to
    move. A FileNotFoundError from a sibling winning that race is swallowed rather than
    raised.

    `keep_batches` names batch prefixes whose charts stay where they are, and the caller passes
    the batches it is *about to measure*. Without it a batch measured in two waves erased its own
    first wave: `b45`'s closeout ran as three waves, and wave 2 archived `b45a`'s and `b45c`'s
    finished charts on startup, so the window they had just filled went blank and nothing ever
    rewrote them. Keeping them is safe because each arm rewrites its own file by name -- a re-run
    overwrites, and an arm the re-run leaves out keeps the chart it had, which is the point.
    Anything from an *earlier* batch is archived exactly as before.
    """
    os.makedirs(EVALS_DIR, exist_ok=True)
    pngs = [name for name in os.listdir(EVALS_DIR) if name.endswith('.png')]
    keep = set(keep_batches or ())
    if keep and pngs:
        # Imported here rather than at module scope: `chart_viewer` owns both the batch regex and
        # the `<policy>_eval_progress.png` naming, and this keeps eval_plan's import surface --
        # which every eval process pays for -- exactly as it was. It is stdlib-only at module
        # level, so this costs nothing and cannot cycle.
        import chart_viewer
        pngs = [name for name in pngs
                if chart_viewer.batch_prefix(chart_viewer.policy_from_png(name)) not in keep]
    if not pngs:
        return
    dest = os.path.join(EVALS_ARCHIVE_DIR, time.strftime('%Y%m%d-%H%M%S'))
    os.makedirs(dest, exist_ok=True)
    for name in pngs:
        try:
            shutil.move(os.path.join(EVALS_DIR, name), os.path.join(dest, name))
        except FileNotFoundError:
            pass


def wilson_interval(successes, trials, z=1.96):
    """95% confidence interval for a rate. Normal approximation breaks down for
    the small counts and near-0 rates here, so use Wilson's score interval."""
    if trials == 0:
        return 0.0, 0.0
    p = successes / trials
    denom = 1.0 + z * z / trials
    centre = (p + z * z / (2 * trials)) / denom
    spread = z * ((p * (1 - p) / trials + z * z / (4 * trials * trials)) ** 0.5) / denom
    return max(0.0, centre - spread), min(1.0, centre + spread)


def build_row(step, held, meta=None):
    """One result row from a checkpoint's accumulated episodes.

    `held` carries the raw per-episode lists rather than running totals, so a screening pass
    that is later topped up to full length recomputes the median and the extremes exactly
    instead of approximating them from two summaries.
    """
    scores = held['scores']
    perfect = int(sum(held['perfect']))
    meta = meta or {'selected_by': 'explicit'}
    low, high = wilson_interval(perfect, len(scores))
    return {
        'step': step,
        'selected_by': meta.get('selected_by', 'explicit'),
        'graph_single_eval': meta.get('single_eval'),
        'graph_surrounding': meta.get('surrounding'),
        'episodes': len(scores),
        'perfect_games': perfect,
        'perfect_percent': round(100.0 * perfect / len(scores), 1),
        'perfect_ci95': [round(100.0 * low, 1), round(100.0 * high, 1)],
        'avg_score': round(float(np.mean(scores)), 2),
        'median_score': round(float(np.median(scores)), 1),
        'min_score': round(float(np.min(scores)), 1),
        'max_score': round(float(np.max(scores)), 1),
        'avg_reward': round(float(np.mean(held['rewards'])), 2),
        # True when EVAL_MIN_ACHIEVABLE stopped this checkpoint short because it could no longer
        # reach the threshold. Such a row is a valid but *shorter* sample whose rate is always
        # below the threshold, so it can be read as "below the bar" but not compared on equal
        # footing with a full-length row. `equal_effort_pooled` is unaffected — see
        # `make_abandon_test` for why the floor guarantees that.
        'abandoned': bool(held.get('abandoned')),
        # Wall-clock per checkpoint, so eval_progress.py can give an ETA from this run's own
        # throughput rather than a hardcoded guess. Strong policies play longer episodes and
        # measure slower, so a fixed estimate is wrong in both directions.
        'seconds': round(held['seconds'], 1),
        # The raw per-episode results, from 2026-08-08. Everything above is derivable from
        # these, and storing them is what makes a screen *resumable*: load_finished_results can
        # seed a checkpoint's sample from a killed run and let stage 3 top it up, instead of
        # discarding the screen and measuring it again from zero.
        #
        # Leaving them out was not free. A run killed mid-screening lost every screen it had
        # done - 192 rows and 7,534 episodes on b18a, 193 and 6,865 on b18d, in one incident.
        # Summaries cannot substitute: perfect_games, episodes, min and max pool exactly and the
        # averages pool from sums, but the *median does not*, so a topped-up row rebuilt from
        # summaries would carry a median that is quietly wrong.
        #
        # Cost is ~1.6 KB a row, so ~1 MB on a 600-row arm against a 145 KB payload. Scores are
        # whole food counts and stored as ints; perfect flags as 0/1 rather than true/false,
        # which is smaller and is what sum() already treats them as.
        'episode_scores': [int(score) for score in scores],
        'episode_perfect': [int(bool(flag)) for flag in held['perfect']],
        'episode_rewards': [round(float(reward), 2) for reward in held['rewards']],
    }


def held_from_row(row):
    '''A `held` sample rebuilt from a stored row, or None if the row predates per-episode
    storage.

    The inverse of the three `episode_*` fields `build_row` writes. Returning None rather than
    raising is deliberate: every result file written before 2026-08-08 lacks them, and a resume
    against one of those has to fall back to re-measuring rather than fail.
    '''
    scores = row.get('episode_scores')
    perfect = row.get('episode_perfect')
    rewards = row.get('episode_rewards')
    if not scores or perfect is None or rewards is None:
        return None
    if not (len(scores) == len(perfect) == len(rewards) == row.get('episodes', -1)):
        # A truncated or hand-edited row. Re-measuring is cheap; pooling a mismatched sample is
        # a silent wrong answer, which is the trade this whole module is written around.
        return None
    return {'scores': list(scores), 'perfect': list(perfect), 'rewards': list(rewards),
            'seconds': float(row.get('seconds') or 0.0),
            'abandoned': bool(row.get('abandoned'))}


class RowCache:
    """`build_row` output per step, so assembling a result list costs a lookup per row, not a build.

    Both writers assemble the whole list on **every** write — `eval_wave.Arm.rows` and
    `eval_checkpoints`'s `current_results` — while at most one step's sample can have changed since
    the last one. Measured on `b43b-lowlr-b29a` at 513 rows: **97 ms** a pass, run five times per
    write in the wave and once per write in the single-policy path, against writes arriving 125 times
    per 500-episode measurement.

    **The invariant is the whole of it: whoever mutates a sample must `put` the rebuilt row.** A
    cache that is merely never invalidated does not look broken — it looks like a checkpoint whose
    row froze at its screening depth while the file kept being rewritten around it. Both call sites
    mutate a sample in exactly one place and both already build the row there for their own log line,
    so `put` costs nothing and is the natural pairing; `clear` is for the one place a samples dict is
    rebound wholesale (`eval_wave.Arm.finalise_plan`).

    Rows are handed out by reference. That is safe because `build_row` copies the three `episode_*`
    lists out of the sample rather than aliasing them, and every consumer downstream — the payload's
    sort, the summaries, `eval_progress` — only reads.
    """

    def __init__(self):
        self._rows = {}

    def put(self, step, row):
        self._rows[step] = row

    def clear(self):
        self._rows.clear()

    def rows(self, samples, resumed_by_step, meta_for):
        """Result rows for every step with episodes banked, in step order.

        A step whose sample is still empty is skipped — that is the checkpoint in flight, whose
        running state travels in the payload's `in_flight` block instead. `meta_for(step)` supplies
        the selection metadata, which is fixed for the run and so never invalidates a row.
        """
        rows = dict(resumed_by_step)
        for step, held in samples.items():
            if not held['scores']:
                continue
            row = self._rows.get(step)
            if row is None:
                row = build_row(step, held, meta_for(step))
                self._rows[step] = row
            rows[step] = row
        return [rows[step] for step in sorted(rows)]


class WriteGate:
    """A wall-clock gate on *progress* writes. See `WRITE_MIN_INTERVAL` for the cost it bounds.

    Split into a pure `due()` and an explicit `record()` so the two call sites read the way they
    actually behave: the wave asks whether a round's write is due, while the single-policy path's
    `write_results` records every write it makes — including the unconditional per-checkpoint one,
    which has to reset the gate or the next round would write again immediately.

    `now` is injectable so the fixtures need no sleeps.
    """

    def __init__(self, interval=None):
        # Looked up here rather than as a default argument: `WRITE_MIN_INTERVAL` lives with the other
        # protocol constants further down the file, so a default would be evaluated before it exists.
        # A late read is also what lets a fixture patch the constant and have it take effect.
        self.interval = WRITE_MIN_INTERVAL if interval is None else interval
        self.last = 0.0

    def due(self, now=None):
        return (time.time() if now is None else now) - self.last >= self.interval

    def record(self, now=None):
        self.last = time.time() if now is None else now


def skips_screening(meta, threshold=None):
    """True when a checkpoint goes straight to a full-length measurement, no screen first.

    Two cases qualify. A **graph point at `threshold`** — `ALWAYS_FULL_SINGLE`, 95% since
    2026-08-19, so 19/20 or 20/20 of a 20-episode graph eval — is the strongest selector this
    project has, and measuring every one of them completely is a coverage guarantee: the strongest
    tier is never represented by a 20-episode row. It used to be 100% only, which left several
    hundred 19/20 checkpoints per continuation arm able to finish on a screen. An
    **explicitly named step** qualifies too, because naming one is a request to measure it — the
    docs already say explicit steps bypass the selection thresholds, and screening one to 20
    episodes and possibly leaving it there would quietly break that.

    This tier is uncapped and on a strong arm it is large — hundreds of checkpoints.

    `threshold` defaults to ALWAYS_FULL_SINGLE, resolved in the body rather than the signature
    because the constants block sits below the helpers here and a default would be evaluated at
    import time.
    """
    if threshold is None:
        threshold = ALWAYS_FULL_SINGLE
    if meta is None:
        return True
    single = meta.get('single_eval')
    if single is None:
        return True
    return single >= threshold


def equal_effort_pooled(samples, episodes):
    """(perfect, episodes, checkpoints) over the first `episodes` of every measured checkpoint.

    The arm-level pooled rate, and the only version of it that means anything once checkpoints are
    measured to different depths. Pooling the finished rows instead weights the deeply-measured
    ones — which are, by construction, the arm's best — five times as heavily as the rest, so it
    reads high by an amount that depends on the protocol rather than on the policy.

    Truncating every checkpoint to the same prefix restores equal effort. It is a valid sample of
    each: episodes are i.i.d., so the first 20 of a 100-episode measurement are as good a
    20-episode sample as a screen that stopped there. That also lets the 100%-tier checkpoints
    count, which a snapshot taken at the end of a screening stage could not — they never had a
    screening stage.
    """
    perfect = total = count = 0
    for held in samples.values():
        flags = held['perfect'][:episodes]
        if len(flags) < episodes:
            continue
        perfect += int(sum(flags))
        total += len(flags)
        count += 1
    return perfect, total, count


def plan_stages(requested_steps, selected_by, screen_episodes, confirm_count, num_episodes,
                num_workers, resumed=0):
    """Splits the work into stages and prices it. Returns a dict; see the keys below.

    Separated from main() so it can be tested without loadable checkpoints — which matters more
    than usual here, because an observation change makes every existing checkpoint unloadable and
    there is then no arm in the repository to rehearse a protocol against.

    Episode counts are rounded up to whole rounds, because `evaluate` runs one episode per worker
    per round and cannot stop part-way through one. That is why a worker count dividing
    `num_episodes` is worth preferring: at 12 workers a 100-episode request really runs 108.
    """
    def whole_rounds(episodes):
        return -(-episodes // num_workers) * num_workers

    if not screen_episodes:
        total = resumed + len(requested_steps)
        return {'full': list(requested_steps), 'screened': [], 'confirmed': 0,
                'measurements_planned': total,
                'episodes_planned': total * whole_rounds(num_episodes),
                'flat_episodes': total * whole_rounds(num_episodes)}

    full = [s for s in requested_steps if skips_screening(selected_by.get(s))]
    screened = [s for s in requested_steps if not skips_screening(selected_by.get(s))]
    # A floor, not a cap: `pick_finalists` confirms every 20/20 screen even past the quota, and how
    # many of those there will be is unknowable until the screens have run. main() raises the totals
    # once the finalists are known, so a plan printed here can read slightly low.
    confirmed = min(confirm_count, len(screened))
    return {
        'full': full,
        'screened': screened,
        'confirmed': confirmed,
        'measurements_planned': resumed + len(full) + len(screened) + confirmed,
        'episodes_planned': (resumed * whole_rounds(num_episodes)
                             + len(full) * whole_rounds(num_episodes)
                             + len(screened) * whole_rounds(screen_episodes)
                             + confirmed * whole_rounds(num_episodes - screen_episodes)),
        'flat_episodes': (resumed + len(requested_steps)) * whole_rounds(num_episodes),
    }


def pick_finalists(rows, count, already_full=None):
    """The screened checkpoints that earn a full-length measurement.

    Ranked on the screen rate, then on the surrounding graph rate, then on step. The
    tie-break is doing real work rather than making the order deterministic: a 20-episode
    screen can only return 21 distinct rates, so dozens of checkpoints arrive on the same
    value and something has to separate them. The surrounding rate is the right something —
    across 88 checkpoints that had all already cleared 80%, it correlated +0.48 with the true
    rate while the graph point itself managed +0.10 (see select_top_checkpoints).

    **A perfect screen is mandatory and ignores `count`.** Any checkpoint that went 20/20 is
    confirmed even if that takes the confirm stage past `EVAL_CONFIRM_COUNT`, mirroring the rule
    `select_top_checkpoints` already applies one stage earlier, where a graph point of >=90% is
    measured however many slots are left. The reason is the same in both places: **the quota exists
    to ration a large middling pool, and the whole point of the close-out is not to miss the best
    checkpoint.** A 20/20 screen is the strongest signal screening can produce, and dropping one
    because the quota filled would be selection working directly against the goal — a checkpoint that
    would have read 95+ never gets the episodes that would show it.

    It is also cheap in the case that matters. Rates are quantised at screen depth, so 20/20 is rare:
    across batch 15's four arms it was **2 of 596 screens**, and on `b17b` — the arm that produced the
    project record — **11 of 186**. A dead arm produces none at all. So the cost is a handful of extra
    measurements on the arms where an extra measurement is most likely to find something.

    `already_full` names steps that came in from a resumed run at full length. They are
    excluded because they already have the measurement this stage would buy them, so letting
    them occupy a slot would spend it on work that is finished.
    """
    already_full = already_full or {}
    pool = [r for r in rows if r['step'] not in already_full]
    ranked = sorted(pool, key=lambda r: (-r['perfect_games'] / r['episodes'],
                                         -(r.get('graph_surrounding') or 0.0), r['step']))
    chosen = ranked[:count]
    # Perfect screens past the quota, appended in the same ranked order. Kept as a separate pass
    # rather than a bigger slice so the intent survives a future edit to `count`. The `chosen_steps`
    # guard is what makes this idempotent: scanning `ranked` instead of `ranked[count:]` is an
    # equivalent mutant precisely because of it, which is the guard doing its job rather than a gap.
    chosen_steps = {r['step'] for r in chosen}
    mandatory = [r for r in ranked[count:]
                 if r['perfect_games'] == r['episodes'] and r['step'] not in chosen_steps]
    if mandatory:
        print('    {0} checkpoint{1} screened 100% past the confirm quota of {2} and will be '
              'confirmed anyway: {3}'.format(
                  len(mandatory), '' if len(mandatory) == 1 else 's', count,
                  ', '.join(str(r['step']) for r in mandatory)))
    return chosen + mandatory


def achievable_percent(perfect_so_far, episodes_so_far, target_episodes):
    """The best perfect rate a checkpoint can still reach, if every remaining episode is perfect.

    Monotonically non-increasing as episodes land, which is what makes it usable as a stopping
    rule: once it drops below a threshold it can never come back.
    """
    remaining = max(0, target_episodes - episodes_so_far)
    return 100.0 * (perfect_so_far + remaining) / target_episodes


def make_abandon_test(min_achievable, target_episodes, floor_episodes):
    """Stopping rule: give up on a checkpoint that can no longer reach `min_achievable`.

    Returns None when disabled, so the caller can skip the mechanism entirely.

    Two guards. **`floor_episodes`** never stops before this many episodes, however hopeless,
    because `equal_effort_pooled` truncates to `screen_episodes` and *skips shorter rows* —
    abandoning above the floor leaves that arm-level figure identical to the un-gated protocol,
    abandoning below it would silently delete rows from the one number meant to compare across
    arms. **`episodes_so_far >= target_episodes`** means a completed checkpoint is never
    "abandoned".

    The rule is exact rather than predictive: it fires only when the remaining episodes cannot
    arithmetically reach the threshold. So a checkpoint that would have finished at or above
    `min_achievable` is never stopped, and an abandoned row's rate is always below the threshold —
    it can never outrank a kept one.
    """
    if not min_achievable:
        return None

    def should_abandon(perfect_total, episodes_total):
        if episodes_total < floor_episodes or episodes_total >= target_episodes:
            return False
        return achievable_percent(perfect_total, episodes_total, target_episodes) < min_achievable

    return should_abandon


# The immutable half of a result file: everything decided before the first checkpoint is measured.
# Split from `progress` (which mutates every round) so `build_payload` cannot accidentally depend on
# a value that changed under it, and so the two callers — `eval_checkpoints.main` and
# `eval_wave.py` — provably pass the same thing.
PayloadSpec = collections.namedtuple('PayloadSpec', [
    'policy_name', 'num_episodes', 'all_steps', 'num_workers', 'screen_episodes',
    'confirm_count', 'min_achievable', 'abandon_floor', 'measurements_planned',
    'episodes_planned', 'full_planned', 'screen_planned', 'confirm_planned'])


def build_payload(spec, progress, samples, results, complete, in_flight=None):
    """The result-file payload, and the **only** definition of it.

    Both the single-policy CLI and the wave controller write
    `runs/<policy>_checkpoint_evals<suffix>.json`, and everything downstream — `eval_progress`,
    `select_checkpoints_above`, `run_report`, the desktop's publish globs — reads it by key. Two
    builders that drift is the failure this project has already paid for twice in a different guise
    (two `build_eval_agent`-shaped constructions, where `expect_partial()` hid the mismatch), so
    there is one.

    `complete` distinguishes a finished run from a partial one, so a later reader does not quietly
    treat 6 checkpoints as the arm's full measurement — and `select_checkpoints_above` refuses to
    select from a file that is not complete.
    """
    return {'policy_name': spec.policy_name,
            'episodes_per_checkpoint': spec.num_episodes,
            'checkpoints_requested': len(spec.all_steps),
            # Recorded rather than inferred from episodes/rounds, which is wrong for any
            # checkpoint being topped up — the episode count includes the screen already on
            # file while the round count does not.
            'num_workers': spec.num_workers,
            'complete': complete,
            'requested_steps': spec.all_steps,
            # Screening protocol, or None for the flat one-pass measurement. Recorded
            # because a pooled rate only compares across arms when the selection rule
            # matches, and this is part of the rule.
            'screen_episodes': spec.screen_episodes or None,
            'confirm_count': spec.confirm_count if spec.screen_episodes else None,
            # Also part of the selection rule, so also recorded: a file measured with the
            # gate on holds shorter rows below the threshold than one measured without it,
            # and pooling the two sets of raw rows would compare different protocols.
            'min_achievable': spec.min_achievable or None,
            'abandon_floor': spec.abandon_floor if spec.min_achievable else None,
            'abandoned': progress.get('abandoned', 0),
            'episodes_saved': progress.get('episodes_saved', 0),
            # Progress in measurements and in episodes. Both are needed once checkpoints
            # can be measured twice at different lengths: the first tracks the stage plan,
            # the second is what an ETA can actually be built from.
            'measurements_planned': spec.measurements_planned,
            'measurements_done': progress['measurements'],
            # This process's own throughput, for the ETA. See `eval_checkpoints.main`'s `progress`
            # for why the pace cannot be averaged over resumed rows.
            'session_measurements': progress['session_measurements'],
            'session_episodes': progress['session_episodes'],
            'session_seconds': round(progress['session_seconds'], 1),
            # Wave-level, and absent from a single-policy run, which is why these are `.get`.
            # `arm_eta_seconds` is this arm's own finish time measured as wall clock between its last
            # few completions -- the only way to price an arm whose share of the lanes changes as its
            # siblings finish -- and `eval_progress` prefers it over its own episode arithmetic.
            # `wave_eta_seconds` is the whole wave's, which no single arm's file could carry.
            'arm_eta_seconds': progress.get('arm_eta_seconds'),
            'arm_eta_window': progress.get('arm_eta_window'),
            'wave_eta_seconds': progress.get('wave_eta_seconds'),
            'wave_lanes': progress.get('wave_lanes'),
            'wave_arms': progress.get('wave_arms'),
            # Which pass is running and how far through each one is, so the chart can show
            # the shape of a three-stage close-out rather than one bar that stalls.
            'stage': progress['stage'],
            # The arm-level rate: every checkpoint truncated to its first `screen_episodes`,
            # because pooling rows of different depths weights the full-length ones — the
            # arm's best by construction — 5x. None for a flat run, already equal effort.
            'pooled_equal_effort': (
                (lambda t: round(100.0 * t[0] / t[1], 2) if t[1] else None)(
                    equal_effort_pooled(samples, spec.screen_episodes))
                if spec.screen_episodes else None),
            'stages': {
                'full': {'planned': spec.full_planned, 'done': progress['full_done']},
                'screen': {'planned': spec.screen_planned, 'done': progress['screen_done']},
                'confirm': {'planned': spec.confirm_planned, 'done': progress['confirm_done']},
            } if spec.screen_episodes else None,
            'episodes_planned': spec.episodes_planned,
            'episodes_done': sum(r['episodes'] for r in results),
            # The checkpoint being measured right now, updated every round, or None
            # between checkpoints. eval_progress.py renders this; without it a
            # ~5-minute checkpoint is invisible until it lands.
            'in_flight': in_flight,
            'updated_at': time.time(),
            # Step order, not measurement order: a resumed run appends new checkpoints
            # after the rows it loaded, which would otherwise interleave arbitrarily.
            'results': sorted(results, key=lambda r: r['step'])}


def write_payload(out_path, payload):
    """Writes a result payload atomically: `.partial` then `os.replace`.

    Part of the protocol rather than a detail. A long run rewrites the whole file after every
    checkpoint — writing only at the end means an interruption throws all of it away — and a reader
    polling the file (`eval_progress.live_frame`, the desktop's publish) must never see a
    half-written one.
    """
    partial_path = out_path + '.partial'
    with open(partial_path, 'w') as handle:
        json.dump(payload, handle, indent=2)
    os.replace(partial_path, out_path)


def best_full_length_row(results, num_episodes):
    """The best checkpoint, ranked over rows deep enough for a maximum to mean anything.

    A screened-out checkpoint has a 20-episode rate whose interval is ~5x wider than a
    100-episode one, so across hundreds of screens some read 19/20 or 20/20 on luck alone.
    Letting one of those win would crown a checkpoint the protocol deliberately declined to
    measure, so ranking is over rows at `num_episodes` whenever any exist.

    **The fallback is why this is a function.** At `EVAL_MIN_ACHIEVABLE=95` most arms have *no*
    full-length row, since few checkpoints clear 95%. Falling back to all rows would hand the
    title to a 20-episode screen on a lucky 20/20 — the outcome the deep-row rule exists to
    prevent, reintroduced exactly when the arm is too weak to defend itself. So the fallback
    relaxes the depth requirement rather than dropping it: rows at half `num_episodes` or better,
    matching `eval_progress.deep_rows`, and only then everything.
    """
    if not results:
        return None
    for minimum in (num_episodes, num_episodes / 2.0, 0):
        deep = [r for r in results if r['episodes'] >= minimum]
        if deep:
            return max(deep, key=lambda r: r['perfect_percent'])
    return None


# Selection thresholds, in single-eval percentage points. **A graph point is `training.
# num_eval_episodes` episodes — 20 since 2026-08-19, 10 before it — so these are coarser than they
# look and the granularity is load-bearing.** At 20 episodes the reportable values are multiples of
# 5, so the mandatory tier is {95, 100} and the fill band is exactly {90}. At the old 10 they were
# multiples of 10, which is why the thresholds used to be 90/60: a 95 tier would have collapsed to
# {100} and been indistinguishable from ALWAYS_FULL_SINGLE. **If num_eval_episodes changes again,
# re-derive these** rather than assuming the bands survive.
ALWAYS_EVAL_SINGLE = 95.0   # every checkpoint at or above this is measured, even past `count`
MIN_EVAL_SINGLE = 90.0      # below this, a checkpoint is not worth 100 episodes
DEFAULT_COUNT = 50          # target total; the mandatory tier may exceed it

# `above:N` with no N given. A HOF re-measure of the checkpoints a close-out already found
# excellent; the desktop's auto-HOF passes `above:98` explicitly. This is a *measured* rate on
# the close-out's 100-episode result, not a graph single-eval, so it is on a different scale from
# the constants above and never mixes with them.
DEFAULT_ABOVE_THRESHOLD = 98.0

# Screen depth: every selected checkpoint gets this many episodes, then only the best
# DEFAULT_CONFIRM_COUNT go on to EVAL_EPISODES. 0 gives the flat one-pass protocol.
DEFAULT_SCREEN_EPISODES = 20

# Abandon a checkpoint once it can no longer reach this perfect rate even if every remaining
# episode is perfect. `EVAL_MIN_ACHIEVABLE=0` turns it off.
#
# 97 (raised from 95 on 2026-08-19) because that is the bar a checkpoint has to clear to be
# interesting at all; anything lower is not a hall-of-fame candidate and does not need 100 episodes
# to be ruled out. At 100 episodes this stops a run once more than 3 have failed, which is most of
# them — and it must stay strictly BELOW the HOF selection gate of 98, since the HOF pass reads
# `above:98` out of the close-out's own file and only rows reaching the close-out gate are measured
# full length. Both the desktop runner and the laptop chain script assert that invariant.
#
# The cost is that **most arms will have no full-length row**, since few checkpoints clear 95%.
# best_full_length_row handles that by relaxing to half-depth rows; pooled_equal_effort is
# unaffected at any gate. Cross-batch best-checkpoint stays valid for the question that matters
# here — "did this arm produce a >=97% checkpoint" — because any checkpoint at or above the gate is
# measured full length under it.
DEFAULT_MIN_ACHIEVABLE = 97.0

# Never abandon before this many episodes, so an abandoned row is always long enough to count in
# equal_effort_pooled. Raised to the screen depth at startup — see make_abandon_test.
DEFAULT_ABANDON_FLOOR = 20

# Graph single-eval at or above which a checkpoint skips the screen and goes straight to full
# length. **Equal to ALWAYS_EVAL_SINGLE since 2026-08-19, so the mandatory tier *is* the
# full-length tier** — at 20 episodes that is 19/20 and 20/20. The collapse is deliberate.
#
# It looks like it should be expensive, since the tier is uncapped and full-length work is the
# dominant cost of a close-out. It is not, and the reason is `DEFAULT_MIN_ACHIEVABLE`: a 19/20
# checkpoint sent straight to full length is abandoned as soon as 97% becomes unreachable — after
# 4 failures, often at the 20-episode floor — so a mediocre one costs about what a screen would
# have cost, and one that survives deserved the full measurement anyway. Simulated on b43/b44's own
# curves at 20-episode graph evals: moving this from 100 to 95 changed total close-out episodes by
# **-1%**, every arm within ±3%. **That result depends on the gate.** At a loose gate (or
# EVAL_MIN_ACHIEVABLE=0) this threshold becomes the whole bill again, so do not lower the gate and
# leave this at 95.
#
# What it buys is coverage. Under the old 100 the tier below it was screened to 20 episodes and only
# `EVAL_CONFIRM_COUNT` of them promoted, so on b43/b44 **427-575 checkpoints per arm** sat at 19/20
# on the graph and could finish with a 20-episode row. Now every one of them gets a real
# full-length attempt, bounded by the gate rather than by a quota.
ALWAYS_FULL_SINGLE = 95.0

# How many screened checkpoints get promoted to full length. 100 is the knee of the recall curve:
# at 30 the arm's true best checkpoint made the cut only 57% of the time, at 100 it is 97%, for
# ~23% more episodes. A 20-episode screen cannot rank a population clustered between 60% and 80%.
DEFAULT_CONFIRM_COUNT = 100

# Seconds between live-chart refreshes. A frame is ~0.1s and a round ~4s, so this only bites
# early in a run when episodes are short and rounds finish fast.
CHART_MIN_INTERVAL = 2.0

# Seconds between full rewrites of a result file *while a measurement is in flight*. A completed
# measurement always writes immediately, so this bounds only the progress refresh.
#
# ‡ It exists because the write is O(banked rows) and the round cadence is O(episodes), so the two
# multiply. `on_round` fires once per `num_workers` episodes -- **125 times** for a 500-episode
# measurement -- and each call rebuilt every banked row and re-serialised the whole file. Measured on
# `b43b-lowlr-b29a` at 513 rows: 97 ms to rebuild the rows, 0.52 s for one of `eval_wave`'s writes
# (which built five payloads), **65 s of single-threaded bookkeeping per measurement** against the
# 46 s of wall clock four lanes needed to produce one. So the controller thread overtook the lanes
# partway through the arm and then became the only thing running: b43's HOF pass finished all 767
# measurements at ~07:29 and spent the next 90 minutes folding a backlog out of its own outbox at
# 45 s a row, with 4 lanes and 16 workers idle and the machine 95% free.
#
# A wall-clock gate makes the cost independent of episode count, which is the property that was
# missing. It does not make the write cheap -- see `eval_wave.Arm.rows` for that half.
WRITE_MIN_INTERVAL = 2.0

# ---------------------------------------------------------------- the HOF re-measure

# 500 episodes, flat, gate 98, into its own file. Selected `above:98` from the close-out's own
# result file, so it re-measures exactly the checkpoints the close-out already found excellent.
#
# A distinct suffix is not cosmetic: it keeps this off the close-out's file so neither clobbers the
# other, and so the close-out's 100-episode rows can never be mistaken for finished HOF work.
HOF_EPISODES = 500
HOF_SUFFIX = '_hof500'
HOF_GATE = DEFAULT_ABOVE_THRESHOLD

# **The one place this ordering is asserted.** It was previously asserted in two — `runner.py` and
# the laptop's chain script — which is how two copies of a recipe start to drift. The invariant: the
# HOF pass reads `above:HOF_GATE` out of the close-out's own file, and only rows that *reach* the
# close-out gate are measured full length, so a close-out gate at or above the HOF gate would
# abandon precisely the rows the re-measure needs and starve it silently. At 97 against 98 there is
# exactly one point of headroom, so this matters more than it did at 96.
assert DEFAULT_MIN_ACHIEVABLE < HOF_GATE, (
    'the close-out gate ({0}) must stay strictly below the HOF selection gate ({1})'.format(
        DEFAULT_MIN_ACHIEVABLE, HOF_GATE))


def hof_settings(closeout_settings):
    """The stage-B settings, derived from stage A's — the single definition of the recipe.

    Takes the close-out's settings rather than being a constant dict so the parts that are *not* the
    recipe (worker count, `EVAL_RESUME`) carry over from however the wave was launched, while
    everything the recipe fixes is overridden here. `source_suffix` is what `select_checkpoints_above`
    reads the candidates out of, which is stage A's file, not stage B's.
    """
    return dict(closeout_settings,
                suffix=closeout_settings['suffix'] + HOF_SUFFIX,
                source_suffix=closeout_settings['suffix'],
                num_episodes=HOF_EPISODES,
                screen_episodes=0,
                screen_requested='0',
                min_achievable=HOF_GATE,
                abandon_floor=DEFAULT_ABANDON_FLOOR)


def select_top_checkpoints(policy_name, available, count=DEFAULT_COUNT, window=10):
    """Every checkpoint at >=`ALWAYS_EVAL_SINGLE`, then the best of >=`MIN_EVAL_SINGLE` to `count`.

    1. Everything at `ALWAYS_EVAL_SINGLE` or better is measured, even past `count`.
    2. Remaining slots go to the highest single evals down to `MIN_EVAL_SINGLE`, ordered by
       the surrounding perfect rate within an equal-eval tier.
    3. Nothing below `MIN_EVAL_SINGLE` is measured at all.

    Fewer than `count` is a normal outcome, not an error.

    Two measured results this rests on. **Rank on the raw single eval, not a smoothed
    region** — raw correlates +0.64 with the 100-episode measurement across the full range
    where a smoothed rate correlates -0.40. **Inside the high band that reverses**, which is
    why the surrounding rate orders the fill tier: over 88 checkpoints already past 80%, the
    surrounding rate correlated +0.48 against the graph value's +0.10.

    An outlier eval is not luck. Against the checkpoints 1000 steps either side, outliers won
    3 of 3 by 9.0, 11.5 and 27.5 points; if a policy's true rate were 27%, a 10-episode eval
    reading 7+ perfect has probability 0.006.
    """
    path = os.path.join(RUNS_DIR, '{0}_evals.json'.format(policy_name))
    with open(path) as handle:
        evals = json.load(handle)['evals']

    rates = [e['perfect_percent'] for e in evals]
    candidates = []
    for index, entry in enumerate(evals):
        if entry['step'] not in available:
            continue
        lo = max(0, index - window // 2)
        smoothed = sum(rates[lo:lo + window]) / len(rates[lo:lo + window])
        candidates.append({'step': entry['step'],
                           'single': entry['perfect_percent'],
                           'smoothed': smoothed})

    if not candidates:
        raise SystemExit('no eval steps in {0} have checkpoints in savedPolicies'.format(path))

    # Primary: the single-eval spike. Secondary: the surrounding perfect rate.
    def rank(entry):
        return (-entry['single'], -entry['smoothed'], entry['step'])

    mandatory = sorted([c for c in candidates if c['single'] >= ALWAYS_EVAL_SINGLE], key=rank)
    fill_pool = sorted([c for c in candidates
                        if MIN_EVAL_SINGLE <= c['single'] < ALWAYS_EVAL_SINGLE], key=rank)
    excluded = len(candidates) - len(mandatory) - len(fill_pool)

    for entry in mandatory:
        entry['selected_by'] = 'threshold{0:g}'.format(ALWAYS_EVAL_SINGLE)
    fill = fill_pool[:max(0, count - len(mandatory))]
    for entry in fill:
        entry['selected_by'] = 'outlier'
    chosen = mandatory + fill

    if not chosen:
        best = max(candidates, key=lambda c: c['single'])
        raise SystemExit(
            'no checkpoint in {0} reached {1:.0f}% on its graph point (best was '
            '{2:.0f}% at step {3}), so there is nothing worth measuring for this arm'.format(
                path, MIN_EVAL_SINGLE, best['single'], best['step']))

    # The band is printed half-open (`>=lo, <hi`) rather than as `lo-(hi-10)`: the old form
    # hardcoded a 10-episode graph point, and the granularity is now 20 episodes.
    print('selected {0} of {1} available checkpoints: {2} at >={3:.0f}% (all measured), '
          '{4} filled from the >={5:.0f}% and <{6:.0f}% band, {7} skipped below {5:.0f}%'.format(
              len(chosen), len(candidates), len(mandatory), ALWAYS_EVAL_SINGLE,
              len(fill), MIN_EVAL_SINGLE, ALWAYS_EVAL_SINGLE, excluded))
    if len(chosen) < count:
        print('    only {0} of {1} slots filled — everything else was below {2:.0f}%, '
              'which is not worth 100 episodes'.format(
                  len(chosen), count, MIN_EVAL_SINGLE))
    if len(mandatory) > count:
        print('    {0} checkpoints at >={1:.0f}% exceeds the {2}-slot target on purpose'.format(
            len(mandatory), ALWAYS_EVAL_SINGLE, count))
    if len(fill_pool) > len(fill):
        print('    {0} more in the >={1:.0f}% and <{2:.0f}% band were not measured (cap is {3})'.format(
            len(fill_pool) - len(fill), MIN_EVAL_SINGLE, ALWAYS_EVAL_SINGLE, count))
    for entry in sorted(chosen, key=lambda c: c['step']):
        print('    {0:>8}  single eval {1:>5.1f}%   surrounding {2:>5.1f}%   {3}'.format(
            entry['step'], entry['single'], entry['smoothed'], entry['selected_by']))
    return [c['step'] for c in sorted(chosen, key=lambda c: c['step'])], \
        {c['step']: {'selected_by': c['selected_by'],
                     'single_eval': c['single'],
                     'surrounding': round(c['smoothed'], 1)} for c in chosen}


def select_checkpoints_above(policy_name, available, threshold, source_suffix=''):
    """Checkpoints whose *measured* perfect rate in the close-out result file is >= `threshold`.

    Unlike `select_top_checkpoints`, which ranks on the 10-episode graph point, this reads the
    close-out's own `<policy>_checkpoint_evals<source_suffix>.json` — the 100-episode measurement.
    A HOF re-measure exists to reconfirm the checkpoints the close-out already found excellent, so
    the strong measured rate is the right selector, not the noisy graph value the close-out itself
    selected on. The desktop close-out writes no `EVAL_OUT_SUFFIX`, so `source_suffix=''` names the
    file it produced; the HOF re-measure writes under its own suffix and so never reads its own output.

    Every close-out row at or above the threshold is taken — including 20-episode screens that read
    high — because "test everything the close-out flagged" is the whole point, and the re-measure is
    what settles a lucky screen. Returns `([], {})` when nothing qualifies, which is the normal
    outcome for most arms and must not be an error: the caller exits 0. An `abandoned` row can never
    qualify (its rate sits below the close-out gate, which is <= threshold) and is skipped for clarity.

    A meta dict without a `single_eval` key makes each step skip screening (`skips_screening`) and go
    straight to full length — exactly right here, and moot when the HOF pass runs `EVAL_SCREEN_EPISODES=0`.
    """
    path = os.path.join(RUNS_DIR, '{0}_checkpoint_evals{1}.json'.format(policy_name, source_suffix))
    if not os.path.exists(path):
        raise SystemExit('no close-out result file to select from: {0}'.format(path))
    with open(path) as handle:
        payload = json.load(handle)
    results = payload.get('results', [])
    chosen = sorted((r for r in results
                     if not r.get('abandoned')
                     and r.get('perfect_percent', 0) >= threshold
                     and r['step'] in available),
                    key=lambda r: r['step'])
    print('HOF selection from {0}: {1} of {2} close-out checkpoints scored >= {3:g}%'.format(
        os.path.basename(path), len(chosen), len(results), threshold))
    for r in chosen:
        print('    {0:>8}  close-out {1:>5.1f}%  ({2} episodes)'.format(
            r['step'], r['perfect_percent'], r.get('episodes')))
    selected_by = {r['step']: {'selected_by': 'above{0:g}'.format(threshold),
                               'closeout_percent': r['perfect_percent']} for r in chosen}
    return [r['step'] for r in chosen], selected_by


def merge_checkpoint_evals(policy_name, suffixes=None, out_suffix='_merged'):
    """Combines several result files for one policy into one, and writes it.

    Parallel processes on one arm each need their own EVAL_OUT_SUFFIX or they overwrite each
    other; this puts the pieces back together.

    **A step measured more than once is combined, not deduplicated.** Repeat measurements of a
    frozen checkpoint are independent samples of the same quantity, so summing episodes and
    perfect games is correct and tightens the interval. `perfect_percent` and the Wilson interval
    are recomputed from the combined counts.

    Pass `suffixes` explicitly, or leave it None to pick up every
    `<policy>_checkpoint_evals*.json` except previous merges. Returns the merged payload.
    """
    pattern = os.path.join(RUNS_DIR, '{0}_checkpoint_evals*.json'.format(policy_name))
    if suffixes is None:
        paths = [p for p in sorted(glob.glob(pattern)) if not p.endswith(out_suffix + '.json')]
    else:
        paths = [os.path.join(RUNS_DIR, '{0}_checkpoint_evals{1}.json'.format(policy_name, s))
                 for s in suffixes]

    by_step, episodes_per, sources, incomplete = {}, set(), [], []
    for path in paths:
        if not os.path.exists(path):
            raise SystemExit('no such result file: {0}'.format(path))
        with open(path) as handle:
            payload = json.load(handle)
        sources.append(os.path.basename(path))
        episodes_per.add(payload.get('episodes_per_checkpoint'))
        if payload.get('complete') is False:
            incomplete.append(os.path.basename(path))
        for row in payload.get('results', []):
            existing = by_step.get(row['step'])
            if existing is None:
                by_step[row['step']] = dict(row)
                continue
            # Same checkpoint measured twice: pool the episodes.
            total_episodes = existing['episodes'] + row['episodes']
            total_perfect = existing['perfect_games'] + row['perfect_games']
            weight, other = existing['episodes'], row['episodes']
            existing['avg_score'] = round(
                (existing['avg_score'] * weight + row['avg_score'] * other) / total_episodes, 2)
            existing['episodes'] = total_episodes
            existing['perfect_games'] = total_perfect
            existing['perfect_percent'] = round(100.0 * total_perfect / total_episodes, 1)
            low, high = wilson_interval(total_perfect, total_episodes)
            existing['perfect_ci95'] = [round(100.0 * low, 1), round(100.0 * high, 1)]
            existing['measurements'] = existing.get('measurements', 1) + 1
            # min/max are order statistics, so take the extremes across both runs.
            existing['min_score'] = min(existing['min_score'], row['min_score'])
            existing['max_score'] = max(existing['max_score'], row['max_score'])

    results = [by_step[step] for step in sorted(by_step)]
    payload = {'policy_name': policy_name,
               'episodes_per_checkpoint': (episodes_per.pop() if len(episodes_per) == 1 else
                                           sorted(e for e in episodes_per if e is not None)),
               'checkpoints_requested': len(results),
               'complete': not incomplete,
               'merged_from': sources,
               'incomplete_sources': incomplete,
               'results': results}

    out_path = os.path.join(RUNS_DIR, '{0}_checkpoint_evals{1}.json'.format(policy_name, out_suffix))
    partial_path = out_path + '.partial'
    with open(partial_path, 'w') as handle:
        json.dump(payload, handle, indent=2)
    os.replace(partial_path, out_path)

    repeats = sum(1 for r in results if r.get('measurements', 1) > 1)
    print('merged {0} files -> {1} checkpoints ({2} measured more than once), wrote {3}'.format(
        len(sources), len(results), repeats, out_path))
    if incomplete:
        print('    note: {0} source(s) were incomplete: {1}'.format(len(incomplete), ', '.join(incomplete)))
    return payload
