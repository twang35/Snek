"""Measures a saved checkpoint's true perfect-game rate over many episodes.

The graph in runs/<policy>.png plots one 10-episode eval per point, so its value moves in
10-point jumps and is far too coarse to quote. This script reloads specific checkpoints and
evaluates each over hundreds of episodes. The graph point is still a good *selector* even
though it is a bad measurement — see select_top_checkpoints.

Protocol rationale, measured costs and cross-batch comparability rules live in
hyperparamTuning/hyperparamTuning.md. This docstring covers how to run it and the traps.

Usage:

    cd snek2
    PYTHONPATH=. python -u eval_checkpoints.py <policy_name> <step> [<step> ...]

    # the normal close-out
    EVAL_WORKERS=10 PYTHONPATH=. python -u eval_checkpoints.py <policy_name> top20

    # flat one-pass protocol (every arm before batch 10 was measured this way)
    EVAL_SCREEN_EPISODES=0 ... top20

    # continue an interrupted close-out
    EVAL_RESUME=1 ... top20

Selection (`top20`, `top`, `top:20`) ranks on the single 10-episode graph eval, breaking ties
on the surrounding rate. Selection (`above:98`, `above`) instead reads a *prior close-out's*
100-episode measurement and takes every checkpoint whose `perfect_percent` is at or above the
threshold — the HOF re-measure path, which reconfirms already-excellent checkpoints rather than
re-discovering them from the noisy graph (see select_checkpoints_above):

- every checkpoint at **>=90%** is measured, even past N
- remaining slots go to the best of the rest down to **>=60%**
- nothing below 60% is measured

N is a target, not a quota — a weak arm may run 1 or 0. Because a graph point is 10 episodes,
the mandatory tier is exactly {90, 100} and the fill band is {60, 70, 80}. Adjacent steps are
allowed through on purpose: 1000 train steps can change the perfect rate by tens of points, so
neighbours are separate policies rather than repeat samples.

Three stages when screening is on (the default):

1. every checkpoint whose graph point is **100%**, plus any explicitly named step, gets the
   full EVAL_EPISODES immediately. Uncapped.
2. everything else selected gets EVAL_SCREEN_EPISODES (20)
3. the best EVAL_CONFIRM_COUNT **of those screened** get the remaining 80

A promoted checkpoint ends with exactly EVAL_EPISODES, so its number is comparable with arms
measured flat. The 100% tier is excluded from confirmation slots — it already has the
measurement a slot would buy. **That tier is coverage, not a shortlist of champions**; the
larger 90% tier holds more of the actual best checkpoints, and finding them is stage 3's job.

Early abandonment (EVAL_MIN_ACHIEVABLE, default 95) stops a checkpoint once its ceiling — its
rate if every remaining episode were perfect — falls below the gate. At 100 episodes and a 95%
gate that is "stop once more than 5 have failed". The rule is arithmetic, not predictive, which
is what makes it safe: a checkpoint that would reach the gate is never stopped, and an abandoned
row's own rate is always below the gate, so it can never outrank a kept one.

What the gate costs, and it is not nothing:

- **best-checkpoint degrades on an arm that never clears the gate**, which at 95 is most arms.
  Such an arm has no full-length row at all; best_full_length_row falls back to half-depth rows
  and the printed line is marked `[truncated]`.
- **abandoned rows are shorter, noisier and not comparable with full-length rows.** They carry
  `abandoned: true`; the payload records `min_achievable` so files measured under different
  gates can be told apart. Do not pool raw rows across them.
- **pooled_equal_effort is exact at any gate** — it truncates to screen depth, and the abandon
  floor is never below that. It is the arm-level figure to use.

Take the arm-level rate from the equal-effort figure the run prints, never by pooling the rows
in the output file: those have different depths and the deep ones are by construction the arm's
best, so pooling weights the winners 5x.

Results are written after every checkpoint, so an interrupted run keeps what it measured;
`complete` stays false until the last one lands. EVAL_RESUME=1 relaunches the identical command
and skips whatever the file already holds at full length. Part-measured checkpoints are redone
from scratch, so no run pools two summaries of one checkpoint.

Environment:
    EVAL_EPISODES     episodes per checkpoint, rounded up to whole rounds (default 100)
    EVAL_WORKERS      parallel envs inside this process (default 10)
    EVAL_OUT_SUFFIX   appended to the output filename
    EVAL_SCREEN_EPISODES  screen depth before promotion (default 20; 0 turns screening off)
    EVAL_CONFIRM_COUNT    how many screened checkpoints get promoted (default 100)
    EVAL_MIN_ACHIEVABLE   abandon once this rate is unreachable (default 95; 0 disables)
    EVAL_ABANDON_FLOOR    never abandon before this many episodes (default 20, raised to
                      EVAL_SCREEN_EPISODES if larger)
    EVAL_RESUME       1 to skip checkpoints already measured at full length, or a
                      comma-separated list of suffixes to take them from
    EVAL_PERFECT_WAIT_MS  window pause on a perfect game (default 400)
    EVAL_RENDER       1 to show a game in a window (default 0, all workers headless)

Traps, all of them learned the hard way:

**Use a throwaway EVAL_OUT_SUFFIX for anything exploratory.** The first write happens seconds
in and unconditionally overwrites whatever is at that path, so reusing the suffix of a complete
measurement destroys it immediately, killed early or not. backup_previous_results() keeps one
rolling `.previous` copy, but a distinct suffix is what actually prevents this.

**Prefer a worker count that divides EVAL_EPISODES.** Episodes round up to whole rounds, so 12
workers turn a 100-episode request into 108 and the rows stop matching the rest of the arm.
More workers is close to free; fewer is actively slower *and* costs more CPU per episode. XLA
(jit_compile=True) measured *worse* here, 0.38 s/episode against 0.32, and is not used.

**Rendering is the most expensive thing in an eval** — 163us per game step headless against
6050us in a window — and because ParallelPyEnvironment waits for the slowest worker, one
rendering worker paces all of them. Hence headless by default. EVAL_RENDER=1 puts a window on
worker 0 for watching by hand; it cannot affect the numbers.

Two things about that window look like bugs and are not, both cosmetic because only each
worker's *first* episode of a round is counted:

1. **It stops mid-game and closes.** Workers that finish early keep being stepped and
   auto-reset into uncounted episodes, so the visible worker is usually part-way through a
   throwaway game when the round ends.
2. **It used to freeze for seconds.** snake_constants.PERFECT_GAME_WAIT_MS blocks the whole
   round via pygame.time.wait(); EVAL_PERFECT_WAIT_MS overrides it.

**evals/ holds only the latest work.** Before writing anything this script moves whatever is
in evals/ into a timestamped evals/archive/ folder. Nothing is lost, but **any** eval launched
for any reason displaces every chart there, including a one-checkpoint verification run — and
EVAL_OUT_SUFFIX does not protect it, since the chart path has no suffix in it. Simultaneous
processes do not archive each other (the archive step runs before any of them writes a chart);
starting one *more* while another runs archives the live one's chart, which reappears within a
round.

A live progress window always opens and cannot be suppressed; it disables itself if no window
can be created, so unattended runs need no configuration. It pools every result file for the
policy, so several processes on one arm each show the same consolidated view. To attach a
window to an eval already running, use
`EVAL_PROGRESS_WINDOW_MODE=1 EVAL_PROGRESS_WATCH=20 python -u eval_progress.py <policy>`.

Results go to runs/<policy_name>_checkpoint_evals<suffix>.json. Two levels of parallelism
compose: EVAL_WORKERS spreads one checkpoint's episodes, and several copies of this script can
run on different checkpoints. Give each copy its own EVAL_OUT_SUFFIX or they overwrite each
other; merge afterwards with merge_checkpoint_evals().
"""
import glob
import json
import os
import shutil
import sys
import time

os.environ['SDL_VIDEODRIVER'] = 'dummy'  # must precede any pygame import
# Belt-and-braces against audio. Snake.Game inits only display+font, but a bare
# pygame.init() anywhere would open a CoreAudio stream per worker and spin coreaudiod
# (measured 15% CPU for 10 idle workers). Unlike the video driver, this is never
# unset for the visible worker — nothing in this project plays sound.
os.environ['SDL_AUDIODRIVER'] = 'dummy'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import numpy as np
import tensorflow as tf
from tf_agents.environments import parallel_py_environment
from tf_agents.environments import tf_py_environment
from tf_agents.system import system_multiprocessing
from tf_agents.utils import common

import snake_constants
from eval_agent import build_eval_agent
from eval_workers import IndependentWorkerPool
from snake_constants import EVALS_ARCHIVE_DIR, EVALS_DIR, POLICY_DIR, RUNS_DIR
from snake_environment import SnakeEnvironment
from state_helpers import is_perfect_score


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


def archive_existing_eval_pngs():
    """Moves whatever is currently in EVALS_DIR into a timestamped EVALS_ARCHIVE_DIR
    subfolder, so a new eval or batch starts from an empty folder and evals/ always shows
    only the most recently completed work.

    Safe when several processes start at once, which is the normal case for a batch: the
    move happens before this process writes anything of its own, so whichever process gets
    here first archives the previous batch's leftovers and the rest find nothing left to
    move. A FileNotFoundError from a sibling winning that race is swallowed rather than
    raised.
    """
    os.makedirs(EVALS_DIR, exist_ok=True)
    pngs = [name for name in os.listdir(EVALS_DIR) if name.endswith('.png')]
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



def skips_screening(meta, threshold=None):
    """True when a checkpoint goes straight to a full-length measurement, no screen first.

    Two cases qualify. A **graph point at `threshold`** (100% — ten perfect games out of ten) is
    the strongest selector this project has, and measuring every one of them completely is a
    coverage guarantee: the strongest tier is never represented by a 20-episode row. An
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


def run_round(parallel_env, policy_action, worker_envs):
    """One episode per worker, every worker run to completion.

    Deliberately *not* "collect N finished episodes and stop". Truncating mid-flight throws away
    the episodes still running, and which ones those are is correlated with the outcome, so the
    surviving sample is not a random one. Running whole rounds costs some idle time on the fast
    workers and keeps the estimate unbiased.

    Which way the bias runs is worth knowing if you reason about a variant of this loop:
    **truncation would read high, not low.** Perfect games average ~1780 steps against ~2200 for
    non-perfect ones — a win ends the moment the board fills, while a policy about to fail circles
    until the starve budget runs out — so truncating drops *failures* preferentially.

    The idle time is real: across 20-40 workers, 20-35% of the env steps in a round belong to
    workers that already finished. Giving each worker a fixed back-to-back quota instead of a
    barrier every episode would recover ~1.2x and stay unbiased. Not implemented.
    """
    num_workers = len(worker_envs)
    scores = np.zeros(num_workers, dtype=np.float64)
    rewards = np.zeros(num_workers, dtype=np.float64)
    perfect = np.zeros(num_workers, dtype=bool)
    done = np.zeros(num_workers, dtype=bool)
    steps = 0

    time_step = parallel_env.reset()
    while not np.all(done):
        action_step = policy_action(time_step)
        time_step = parallel_env.step(action_step.action)
        step_rewards = time_step.reward.numpy()
        is_last = time_step.is_last().numpy()

        active = ~done
        rewards[active] += step_rewards[active]
        steps += int(np.sum(active))

        # A finished worker auto-resets on its next step, so read get_score now,
        # before that reset overwrites current_score.
        newly_done = active & is_last
        if np.any(newly_done):
            indices = np.flatnonzero(newly_done)
            promises = [worker_envs[i].call('get_score') for i in indices]
            for i, promise in zip(indices, promises):
                scores[i] = promise()
                # From the score, not `step_rewards[i]`: a shaping term moves the winning step's
                # reward off `PERFECT_GAME_REWARD`. See `state_helpers.is_perfect_score`.
                perfect[i] = is_perfect_score(scores[i])
        done |= is_last

    return scores.tolist(), perfect.tolist(), rewards.tolist(), steps


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


def evaluate(parallel_env, policy_action, num_episodes, on_round=None, should_abandon=None):
    """Collects at least num_episodes, in whole rounds of one episode per worker.

    `on_round(round_index, rounds_total, perfect_so_far, episodes_so_far, per_round_perfect)`
    is called after each round if given. It exists so the running perfect rate can be
    persisted while a checkpoint is still in flight — a checkpoint takes ~5 minutes and
    without this the only observable state is "started" until it finishes.

    `should_abandon(perfect_so_far, episodes_so_far)` is checked after each round, and stops this
    pass early when it returns True. The counts passed to it are **this pass's**, so a caller
    topping up a screened checkpoint has to fold in the episodes already held — see `measure`. The
    predicate is the caller's business rather than this function's because only the caller knows
    the checkpoint's target length; a round is the smallest unit that can be stopped at, since a
    round runs until every worker's episode ends.

    Returns `(scores, perfect_flags, rewards, elapsed, abandoned)`.
    """
    worker_envs = parallel_env.pyenv.envs
    num_workers = len(worker_envs)
    rounds = -(-num_episodes // num_workers)  # ceil

    scores, perfect_flags, rewards = [], [], []
    per_round_perfect = []
    steps = 0
    abandoned = False
    start = time.time()
    for index in range(rounds):
        round_scores, round_perfect, round_rewards, round_steps = run_round(
            parallel_env, policy_action, worker_envs)
        scores.extend(round_scores)
        perfect_flags.extend(round_perfect)
        rewards.extend(round_rewards)
        steps += round_steps
        per_round_perfect.append(int(sum(round_perfect)))
        print('    round {0}/{1}: {2} episodes, {3} perfect'.format(
            index + 1, rounds, len(round_scores), sum(round_perfect)))
        if on_round is not None:
            on_round(index + 1, rounds, int(sum(perfect_flags)), len(scores), list(per_round_perfect))
        if should_abandon is not None and should_abandon(int(sum(perfect_flags)), len(scores)):
            abandoned = True
            print('    abandoned after {0} episodes: cannot reach the threshold from here'.format(
                len(scores)))
            break

    elapsed = time.time() - start
    print('    {0} episodes in {1}s ({2} env steps/s)'.format(
        len(scores), round(elapsed, 1), round(steps / elapsed)))
    return scores, perfect_flags, rewards, elapsed, abandoned


# Selection thresholds, in single-eval (10-episode) percentage points. A graph point is 10
# episodes, so these are coarser than they look: the mandatory tier is {90, 100} and the fill
# band is exactly {60, 70, 80}.
ALWAYS_EVAL_SINGLE = 90.0   # every checkpoint at or above this is measured, even past `count`
MIN_EVAL_SINGLE = 60.0      # below this, a checkpoint is not worth 100 episodes
DEFAULT_COUNT = 20          # target total; the mandatory tier may exceed it

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
# 95 because that is the bar a checkpoint has to clear to be interesting at all; anything lower is
# not a hall-of-fame candidate and does not need 100 episodes to be ruled out. At 100 episodes this
# stops a run once more than 5 have failed, which is most of them.
#
# The cost is that **most arms will have no full-length row**, since few checkpoints clear 95%.
# best_full_length_row handles that by relaxing to half-depth rows; pooled_equal_effort is
# unaffected at any gate. Cross-batch best-checkpoint stays valid for the question that matters
# here — "did this arm produce a >=95% checkpoint" — because any checkpoint at or above the gate is
# measured full length under it.
DEFAULT_MIN_ACHIEVABLE = 95.0

# Never abandon before this many episodes, so an abandoned row is always long enough to count in
# equal_effort_pooled. Raised to the screen depth at startup — see make_abandon_test.
DEFAULT_ABANDON_FLOOR = 20

# Graph single-eval at or above which a checkpoint skips the screen and goes straight to full
# length. 100% is ten perfect games out of ten, and the tier is uncapped — see skips_screening.
# It is a coverage guarantee, not a champion shortlist: the larger 90% tier holds more of the
# actual best checkpoints. Finding those is EVAL_CONFIRM_COUNT's job.
ALWAYS_FULL_SINGLE = 100.0

# How many screened checkpoints get promoted to full length. 100 is the knee of the recall curve:
# at 30 the arm's true best checkpoint made the cut only 57% of the time, at 100 it is 97%, for
# ~23% more episodes. A 20-episode screen cannot rank a population clustered between 60% and 80%.
DEFAULT_CONFIRM_COUNT = 100

# Seconds between live-chart refreshes. A frame is ~0.1s and a round ~4s, so this only bites
# early in a run when episodes are short and rounds finish fast.
CHART_MIN_INTERVAL = 2.0


def select_top_checkpoints(policy_name, available, count=DEFAULT_COUNT, window=10):
    """Every checkpoint at >=90% single eval, then the best of >=60% up to `count` total.

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
        entry['selected_by'] = 'threshold90'
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

    print('selected {0} of {1} available checkpoints: {2} at >={3:.0f}% (all measured), '
          '{4} filled from the {5:.0f}-{6:.0f}% band, {7} skipped below {5:.0f}%'.format(
              len(chosen), len(candidates), len(mandatory), ALWAYS_EVAL_SINGLE,
              len(fill), MIN_EVAL_SINGLE, ALWAYS_EVAL_SINGLE - 10, excluded))
    if len(chosen) < count:
        print('    only {0} of {1} slots filled — everything else was below {2:.0f}%, '
              'which is not worth 100 episodes'.format(
                  len(chosen), count, MIN_EVAL_SINGLE))
    if len(mandatory) > count:
        print('    {0} checkpoints at >={1:.0f}% exceeds the {2}-slot target on purpose'.format(
            len(mandatory), ALWAYS_EVAL_SINGLE, count))
    if len(fill_pool) > len(fill):
        print('    {0} more in the {1:.0f}-{2:.0f}% band were not measured (cap is {3})'.format(
            len(fill_pool) - len(fill), MIN_EVAL_SINGLE, ALWAYS_EVAL_SINGLE - 10, count))
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


def main(argv):
    if len(argv) < 3:
        print(__doc__)
        return 1
    archive_existing_eval_pngs()
    policy_name = argv[1]
    # Live chart window on the laptop only. `viewer_enabled()` is darwin-gated, so this is a no-op
    # on the desktop, where the runner daemon owns the viewer (`desktop/runner/runner.py`) — two
    # owners would open two windows per wave. It runs after archive_existing_eval_pngs() so the
    # viewer globs the fresh charts, and it is best-effort: a chart is never worth an eval.
    if sys.platform == 'darwin':
        # HiDPI: chart_viewer only magnifies the PNG, and 110 dpi looks soft blown up on a Retina
        # panel while the 200-dpi training chart stays crisp. 220 gives the source enough pixels to
        # match, at the same window size. setdefault so an explicit SNEK_EVAL_CHART_DPI still wins.
        os.environ.setdefault('SNEK_EVAL_CHART_DPI', '220')
        try:
            import chart_viewer
            chart_viewer.spawn_for_eval(policy_name)
        except Exception as error:
            print('chart viewer skipped ({0}: {1})'.format(type(error).__name__, error))
    num_episodes = int(os.environ.get('EVAL_EPISODES', 100))
    # 4, lowered from 10 on 2026-08-08. Measured on the real close-out shape (4 parallel eval
    # processes, 800 episodes each): with independent workers 4 gives 117s against 118s at 5 and
    # 134s at 10, because 4 processes x 4 workers already saturates 14 cores at ~12.7 busy. More
    # workers past that add cost, not throughput. See eval_workers.py for the full table.
    num_workers = int(os.environ.get('EVAL_WORKERS', 4))
    screen_requested = os.environ.get('EVAL_SCREEN_EPISODES')
    screen_episodes, screen_note = resolve_screen_episodes(screen_requested, num_episodes)
    confirm_count = int(os.environ.get('EVAL_CONFIRM_COUNT', DEFAULT_CONFIRM_COUNT))
    if screen_note:
        print(screen_note)
    min_achievable = float(os.environ.get('EVAL_MIN_ACHIEVABLE', DEFAULT_MIN_ACHIEVABLE))
    if min_achievable and not 0 < min_achievable <= 100:
        raise SystemExit('EVAL_MIN_ACHIEVABLE={0} must be a percentage in (0, 100], or 0 to '
                         'disable early abandonment.'.format(min_achievable))
    # Never below the screen depth: equal_effort_pooled truncates to it and drops shorter rows, so
    # a lower floor would quietly delete checkpoints from the one arm-level figure meant to be
    # comparable across arms. Raised silently rather than rejected, because the two knobs are
    # independent and a screen deeper than the floor is a reasonable thing to ask for.
    abandon_floor = max(int(os.environ.get('EVAL_ABANDON_FLOOR', DEFAULT_ABANDON_FLOOR)),
                        screen_episodes)
    if min_achievable:
        print('abandoning any checkpoint that can no longer reach {0}%, once it has run {1}+ '
              'episodes (EVAL_MIN_ACHIEVABLE=0 to disable)'.format(min_achievable, abandon_floor))
    perfect_wait_ms = int(os.environ.get('EVAL_PERFECT_WAIT_MS', 400))
    # Off by default: 163us per game step headless against 6050us in a window, and because
    # ParallelPyEnvironment waits for the slowest worker, one rendering worker paces all of them.
    render_worker = os.environ.get('EVAL_RENDER', '0') not in ('0', '', 'false', 'False')
    # On by default from 2026-08-08. Each worker owns its env and its own network copy, so there is
    # no batched inference to idle through and no per-episode barrier: 1.41x at 4 workers, 1.91x at
    # 5, 2.33x at 10. EVAL_INDEPENDENT=0 restores the ParallelPyEnvironment path, which is what
    # every measurement before this date used and which EVAL_RENDER still needs.
    independent_workers = os.environ.get('EVAL_INDEPENDENT', '1') not in ('0', '', 'false', 'False')

    ckpt_dir = POLICY_DIR + policy_name
    available = {int(f[len('ckpt-'):].split('.')[0])
                 for f in os.listdir(ckpt_dir) if f.startswith('ckpt-') and f.endswith('.index')}

    # Spelled `top` rather than `--top`: tf_agents' handle_main routes argv through absl,
    # which rejects any unregistered `--flag` before main() is reached.
    if argv[2].startswith('top'):
        rest = argv[2][len('top'):].lstrip(':=')
        count = int(rest) if rest else (int(argv[3]) if len(argv) > 3 else DEFAULT_COUNT)
        requested_steps, selected_by = select_top_checkpoints(policy_name, available, count)
    elif argv[2].startswith('above'):
        rest = argv[2][len('above'):].lstrip(':=')
        threshold = float(rest) if rest else (
            float(argv[3]) if len(argv) > 3 else DEFAULT_ABOVE_THRESHOLD)
        requested_steps, selected_by = select_checkpoints_above(policy_name, available, threshold)
        # No qualifying checkpoint is the common case, not a failure: exit clean so the job is
        # marked done rather than failed (and never re-tried on the desktop).
        if not requested_steps:
            print('no close-out checkpoint reached {0:g}% — nothing to re-measure'.format(threshold))
            return 0
    else:
        requested_steps = [int(a) for a in argv[2:]]
        selected_by = {step: {'selected_by': 'explicit'} for step in requested_steps}
        missing = [s for s in requested_steps if s not in available]
        if missing:
            raise SystemExit('no checkpoint for step(s) {0} in {1}'.format(missing, ckpt_dir))

    # Resume before the expensive setup, so a run that has nothing left to do says so in
    # seconds rather than after building an agent and 20 worker processes.
    suffix = os.environ.get('EVAL_OUT_SUFFIX', '')
    resumed_rows, resumed_steps, source_screens, resumed_partial = load_finished_results(
        policy_name, resume_suffixes(os.environ.get('EVAL_RESUME'), suffix), num_episodes)
    if resumed_steps:
        skipped = [s for s in requested_steps if s in resumed_steps]
        requested_steps = [s for s in requested_steps if s not in resumed_steps]
        print('resuming: {0} of the selected checkpoints are already measured at >={1} '
              'episodes, {2} left to do'.format(len(skipped), num_episodes, len(requested_steps)))
        if not requested_steps:
            print('nothing left to measure')
            return 0
        if screen_episodes and not screen_requested:
            # Resume means continue: an arm that started under the flat protocol finishes under it,
            # because a mix of 20- and 100-episode rows is not comparable with itself, let alone with
            # the arm it is meant to be compared against.
            #
            # Which protocol it started under is READ from the source files, never inferred from the
            # depth of the resumed rows — see protocol_from_sources for the batch-18 failure that
            # rule replaced.
            keep_screening, recorded_depth = protocol_from_sources(source_screens)
            if keep_screening:
                if recorded_depth != screen_episodes:
                    print('    continuing at the recorded screen depth of {0} rather than {1}, so '
                          'this arm keeps one protocol throughout'.format(
                              recorded_depth, screen_episodes))
                    screen_episodes = recorded_depth
                    abandon_floor = max(abandon_floor, screen_episodes)
                if len(source_screens) > 1:
                    print('    warning: the source files disagree about the protocol ({0}); '
                          'continuing screened at {1}'.format(
                              sorted(str(v) for v in source_screens), screen_episodes))
                print('    resuming a screened arm: keeping screening on at {0} episodes '
                      '({1} full-length rows carried over)'.format(screen_episodes, len(skipped)))
            elif keep_screening is False:
                print('    screening off: the source files record a flat run, so the rest of the '
                      'arm will be too (EVAL_SCREEN_EPISODES={0} to override)'.format(
                          screen_episodes))
                screen_episodes = 0
            else:
                # Pre-dates `screen_episodes` in the payload. Keep the old behaviour rather than
                # guess, but say which branch was taken and why.
                print('    screening off: these {0} resumed rows come from a file that does not '
                      'record its protocol, so it is assumed flat '
                      '(EVAL_SCREEN_EPISODES={1} to override)'.format(
                          len(skipped), screen_episodes))
                screen_episodes = 0

    # Screens carried over from a killed run. Restricted to this run's candidate set, because a
    # partial sample for a checkpoint nobody selected is not work this run owes.
    resumed_partial = {step: held for step, held in resumed_partial.items()
                       if step in set(requested_steps)}
    if resumed_partial:
        carried = sum(len(held['scores']) for held in resumed_partial.values())
        print('    reusing {0} completed screen{1} ({2} episodes) from the earlier run rather than '
              're-measuring them'.format(len(resumed_partial),
                                         '' if len(resumed_partial) == 1 else 's', carried))

    print('policy {0}: evaluating {1} checkpoints x {2} episodes on {3} workers'.format(
        policy_name, len(requested_steps), num_episodes, num_workers))

    # Same env, network and agent as training, so the restored weights line up.
    spec_env = SnakeEnvironment(discount=0.99, display=False, policy_name=policy_name)
    spec_env.reset()
    spec_tf_env = tf_py_environment.TFPyEnvironment(spec_env)

    # One definition, shared with every independent worker in eval_workers.py. A second copy of
    # this construction is the failure mode this project has hit twice: expect_partial() hides a
    # mismatch, so two builders that drift produce a policy that loads silently and plays badly.
    agent, checkpoint, global_step = build_eval_agent(spec_tf_env, spec_env, ckpt_dir)

    def make_headless_worker():
        os.environ['SDL_VIDEODRIVER'] = 'dummy'
        return SnakeEnvironment(discount=0.99, display=False, policy_name=policy_name)

    def make_visible_worker():
        # Each worker is its own process, so clearing the dummy driver here gives
        # exactly one real window per eval process — several checkpoints evaluated
        # in parallel each get their own window to watch.
        os.environ.pop('SDL_VIDEODRIVER', None)
        # The game's 5000ms perfect-game celebration is a *blocking* pygame.time.wait(), which
        # stalls every worker in the round and leaves the window unresponsive. Override it.
        snake_constants.PERFECT_GAME_WAIT_MS = perfect_wait_ms
        return SnakeEnvironment(discount=0.99, display=True, policy_name=policy_name)

    # Independent workers (the default) never build a ParallelPyEnvironment at all: each worker
    # owns its env *and* its own network copy, so there is no batched inference to share and no
    # per-episode barrier. Measured 1.41x at 4 workers, 1.91x at 5 and 2.33x at 10 — see
    # eval_workers.py for the full table and for why ~10 workers is the optimum.
    #
    # EVAL_RENDER forces the batched path: rendering needs one visible worker among headless ones,
    # which is a property of the shared ParallelPyEnvironment. Watching a game is interactive and
    # already ~37x off the critical path, so it does not need the faster collector.
    pool = None
    parallel_env = None
    policy_action = None
    if independent_workers and not render_worker:
        print('    collecting with {0} independent workers '
              '(EVAL_INDEPENDENT=0 for the batched path)'.format(num_workers))
        pool = IndependentWorkerPool(policy_name, ckpt_dir, num_workers)
    else:
        if independent_workers and render_worker:
            print('    EVAL_RENDER=1 forces the batched path, which is the one that can show a '
                  'window')
        # All headless unless EVAL_RENDER=1, in which case worker 0 renders. Episodes are i.i.d.
        # across workers, so which worker produced one carries no information either way — the
        # window is purely for watching, and it costs ~37x on the critical path (see EVAL_RENDER
        # above), so a close-out nobody is watching should not pay for it.
        if render_worker:
            constructors = [make_visible_worker] + [make_headless_worker] * (num_workers - 1)
        else:
            constructors = [make_headless_worker] * num_workers
        parallel_env = tf_py_environment.TFPyEnvironment(
            parallel_py_environment.ParallelPyEnvironment(constructors))

        # Inference in a tf.function rather than eager: 208us per call against 1421us — 6.8x — for
        # byte-identical actions. Traced once and reused, since restoring weights writes into the
        # same variables and does not force a retrace.
        policy_action = common.function(agent.policy.action)

    out_path = os.path.join(RUNS_DIR, '{0}_checkpoint_evals{1}.json'.format(policy_name, suffix))
    backup_previous_results(out_path)

    # Everything this arm is accountable for, resumed work included, so the output file and the
    # progress chart describe the whole job rather than only what this process happened to run.
    all_steps = sorted(resumed_steps.union(requested_steps))

    # Planned work, in the two units a reader cares about: how many checkpoint measurements will
    # happen, and how many episodes they add up to. A screening protocol measures some
    # checkpoints twice, so "checkpoints" alone no longer tracks progress and the episode count
    # is what gives eval_progress.py an honest ETA across stages of different lengths.
    # The screening split: a graph point at ALWAYS_FULL_SINGLE (or an explicitly named step) is
    # measured at full length straight away, everything else is screened first and only the best
    # `confirm_count` of *those* are taken the rest of the way.
    plan = plan_stages(requested_steps, selected_by, screen_episodes, confirm_count,
                       num_episodes, num_workers, resumed=len(resumed_steps))
    full_steps, screen_steps = plan['full'], plan['screened']
    measurements_planned = plan['measurements_planned']
    episodes_planned = plan['episodes_planned']
    # One confirmation's episode cost, rounded to whole rounds the way plan_stages does it, for the
    # correction applied after the screens run.
    whole_confirm_rounds = (-(-(num_episodes - screen_episodes) // num_workers) * num_workers
                            if screen_episodes else 0)
    if screen_episodes:
        print('plan: {0} at full length ({1:.0f}% graph point or explicit), {2} screened at {3}, '
              'of which the best {4} confirmed — {5} episodes against {6} for a flat pass '
              '({7:.2f}x)'.format(
                  len(full_steps), ALWAYS_FULL_SINGLE, len(screen_steps), screen_episodes,
                  plan['confirmed'], episodes_planned, plan['flat_episodes'],
                  plan['flat_episodes'] / episodes_planned if episodes_planned else 0))

    # Resumed rows arrive already measured, so they start the count rather than being re-earned.
    # `session_*` covers only what this process has done, which is what an ETA has to be built
    # from: resumed rows carry the pace of whatever settings the earlier run used, and a
    # close-out relaunched at a different EVAL_WORKERS can differ by 3x. Averaging over every
    # row put b10b's ETA out by nearly 3x — 7h10m against a real ~2h.
    #
    # `stage` and the per-stage counts exist so the progress chart can say which of the three
    # passes is running and how far through it is. Without them a screening close-out shows one
    # undifferentiated bar that stalls for an hour on the full tier and then races, which reads
    # as a hung run.
    progress = {'measurements': len(resumed_steps),
                'session_measurements': 0, 'session_episodes': 0, 'session_seconds': 0.0,
                'stage': 'full' if screen_episodes else 'flat',
                'full_done': 0, 'screen_done': 0, 'confirm_done': 0}

    def write_results(results, complete, in_flight=None):
        """Rewrites the whole file after every checkpoint, and after every round.

        A long run is expensive — 20 checkpoints x 100 episodes is well over an hour, and a
        63-checkpoint run took four — so writing only at the end means any interruption
        throws all of it away. The numbers would still be in the log, but nothing
        machine-readable would survive. Rewriting is cheap next to a checkpoint's runtime.

        `complete` distinguishes a finished run from a partial one, so a later reader does
        not quietly treat 6 checkpoints as the arm's full measurement. The write goes via
        .partial + os.replace, so a reader never sees a half-written file even if the
        process dies mid-write.
        """
        payload = {'policy_name': policy_name,
                   'episodes_per_checkpoint': num_episodes,
                   'checkpoints_requested': len(all_steps),
                   # Recorded rather than inferred from episodes/rounds, which is wrong for any
                   # checkpoint being topped up — the episode count includes the screen already on
                   # file while the round count does not.
                   'num_workers': num_workers,
                   'complete': complete,
                   'requested_steps': all_steps,
                   # Screening protocol, or None for the flat one-pass measurement. Recorded
                   # because a pooled rate only compares across arms when the selection rule
                   # matches, and this is part of the rule.
                   'screen_episodes': screen_episodes or None,
                   'confirm_count': confirm_count if screen_episodes else None,
                   # Also part of the selection rule, so also recorded: a file measured with the
                   # gate on holds shorter rows below the threshold than one measured without it,
                   # and pooling the two sets of raw rows would compare different protocols.
                   'min_achievable': min_achievable or None,
                   'abandon_floor': abandon_floor if min_achievable else None,
                   'abandoned': progress.get('abandoned', 0),
                   'episodes_saved': progress.get('episodes_saved', 0),
                   # Progress in measurements and in episodes. Both are needed once checkpoints
                   # can be measured twice at different lengths: the first tracks the stage plan,
                   # the second is what an ETA can actually be built from.
                   'measurements_planned': measurements_planned,
                   'measurements_done': progress['measurements'],
                   # This process's own throughput, for the ETA. See `progress` above for why
                   # the pace cannot be averaged over resumed rows.
                   'session_measurements': progress['session_measurements'],
                   'session_episodes': progress['session_episodes'],
                   'session_seconds': round(progress['session_seconds'], 1),
                   # Which pass is running and how far through each one is, so the chart can show
                   # the shape of a three-stage close-out rather than one bar that stalls.
                   'stage': progress['stage'],
                   # The arm-level rate: every checkpoint truncated to its first `screen_episodes`,
                   # because pooling rows of different depths weights the full-length ones — the
                   # arm's best by construction — 5x. None for a flat run, already equal effort.
                   'pooled_equal_effort': (
                       (lambda t: round(100.0 * t[0] / t[1], 2) if t[1] else None)(
                           equal_effort_pooled(samples, screen_episodes))
                       if screen_episodes else None),
                   'stages': {
                       'full': {'planned': len(full_steps), 'done': progress['full_done']},
                       'screen': {'planned': len(screen_steps), 'done': progress['screen_done']},
                       'confirm': {'planned': plan['confirmed'],
                                   'done': progress['confirm_done']},
                   } if screen_episodes else None,
                   'episodes_planned': episodes_planned,
                   'episodes_done': sum(r['episodes'] for r in results),
                   # The checkpoint being measured right now, updated every round, or None
                   # between checkpoints. eval_progress.py renders this; without it a
                   # ~5-minute checkpoint is invisible until it lands.
                   'in_flight': in_flight,
                   'updated_at': time.time(),
                   # Step order, not measurement order: a resumed run appends new checkpoints
                   # after the rows it loaded, which would otherwise interleave arbitrarily.
                   'results': sorted(results, key=lambda r: r['step'])}
        partial_path = out_path + '.partial'
        with open(partial_path, 'w') as handle:
            json.dump(payload, handle, indent=2)
        os.replace(partial_path, out_path)
        update_chart()

    # Live progress window, same mechanism as training's graph. Refreshed from write_results()
    # so it advances every round; a frame is ~0.1s against a ~4s round.
    #
    # Always on, deliberately — update_chart() disables itself if no window can be opened, so an
    # unattended run needs no configuration. The chart is scoped to this job's suffixes — its own
    # EVAL_OUT_SUFFIX plus any it resumes — so parallel EVAL_OUT_SUFFIX processes still show the
    # whole job, but a *different* eval that left a result file on this arm is not merged in. That
    # last case is why the scoping is explicit: load_runs' mtime-window guess pulled a close-out
    # finished <1h earlier into an HOF re-measurement's live chart, reading ~1700 checkpoints.
    chart_suffixes = {suffix} | set(resume_suffixes(os.environ.get('EVAL_RESUME'), suffix))
    chart_path = os.path.join(EVALS_DIR, '{0}_eval_progress.png'.format(policy_name))
    # 'off' disables everything after a real error. 'window_off' disables only the live
    # window (SNEK_CHART_WINDOW=0, the default) while live_frame() keeps writing the PNG on
    # every update -- conflating the two once meant a headless eval wrote the chart a single
    # time and it looked frozen (the decoupled chart_viewer.py then showed a stale snapshot).
    chart = {'screen': None, 'last': 0.0, 'off': False, 'window_off': False}

    def update_chart(force=False):
        if chart['off']:
            return
        now = time.time()
        if not force and now - chart['last'] < CHART_MIN_INTERVAL:
            return
        chart['last'] = now
        try:
            import eval_progress
            frame = eval_progress.live_frame(policy_name, chart_path, suffixes=chart_suffixes)
            if frame is None:
                return
            if chart['window_off']:
                return  # the PNG was written by live_frame() above; only the window is off
            if chart['screen'] is None:
                # Window off by default (the decoupled chart_viewer.py is the way to
                # watch); live_frame() above still writes the PNG every update. Set
                # SNEK_CHART_WINDOW=1 to opt back in. See training.py for why an
                # in-process live window is a fatal-XIO liability under memory pressure.
                if os.environ.get('SNEK_CHART_WINDOW', '0') in ('0', '', 'false', 'False'):
                    chart['window_off'] = True
                    return
                import pyformulas
                chart['screen'] = pyformulas.screen(
                    np.zeros(frame.shape[:2], dtype=np.uint8),
                    '{0} eval progress'.format(policy_name))
            # cv2 reads a three-channel array as BGR and matplotlib produces RGB, so the window
            # needs the channels reversed. The PNG keeps RGB and is written by live_frame().
            chart['screen'].update(frame[:, :, ::-1])
        except Exception as error:
            # A chart is never worth losing an eval over — no display, no cv2, a closed window.
            chart['off'] = True
            print('    progress chart off ({0}: {1})'.format(type(error).__name__, error))

    # Episodes measured in this process, kept raw per step so a screening pass can be topped up
    # to full length later without pooling summary statistics. 660 checkpoints x 100 floats is
    # nothing next to the agent itself.
    # Seeded with any screens carried over from a killed run, so `measure` tops them up to full
    # length exactly the way it tops up a screen measured in this session — same code path, and the
    # median and extremes are recomputed from the pooled raw episodes rather than approximated.
    samples = {step: dict(held) for step, held in resumed_partial.items()}
    resumed_by_step = {row['step']: row for row in resumed_rows}

    def current_results():
        """Result rows for every checkpoint with episodes banked.

        The checkpoint in flight is skipped until its episodes land: `on_round` writes progress
        while `evaluate` is still running, so its sample is still empty at that point and there
        is no rate to report. Its running state travels in the payload's `in_flight` block
        instead, which is what eval_progress.py draws it from.
        """
        rows = dict(resumed_by_step)
        for step, held in samples.items():
            if held['scores']:
                rows[step] = build_row(step, held, selected_by.get(step))
        return [rows[step] for step in sorted(rows)]

    def measure(step, episodes, label, stage=None):
        """Restores one checkpoint and adds `episodes` more episodes to its sample."""
        if stage:
            progress['stage'] = stage
        print('\ncheckpoint {0} ({1})'.format(step, label))
        # With independent workers the parent holds no weights that matter — each worker restores
        # into its own network and reports the global_step it read, which is a stronger check than
        # the parent's own restore was: it confirms all N agree.
        if pool is not None:
            restored = pool.load(step)
        else:
            checkpoint.restore(os.path.join(ckpt_dir, 'ckpt-{0}'.format(step))).expect_partial()
            restored = int(global_step.numpy())
        if restored != step:
            print('    warning: global_step reads {0}, expected {1}'.format(restored, step))

        held = samples.setdefault(step, {'scores': [], 'perfect': [], 'rewards': [], 'seconds': 0.0})
        already = len(held['scores'])
        started_at = time.time()

        def on_round(round_index, rounds_total, perfect_so_far, episodes_so_far, per_round):
            # Report the checkpoint's whole sample, screening episodes included, so a topped-up
            # finalist shows its true running rate rather than restarting from zero.
            total_perfect = sum(held['perfect']) + perfect_so_far
            total_episodes = already + episodes_so_far
            write_results(current_results(), complete=False, in_flight={
                'step': step,
                'round': round_index,
                'rounds_total': rounds_total,
                'perfect_so_far': total_perfect,
                'episodes_so_far': total_episodes,
                # Just this pass, where `episodes_so_far` is the checkpoint's whole sample. The
                # two differ by `already` whenever a screened checkpoint is being topped up, and
                # only this one shares a denominator with `round`.
                'episodes_this_pass': episodes_so_far,
                'running_percent': round(100.0 * total_perfect / total_episodes, 1),
                'per_round_perfect': per_round,
                'started_at': started_at,
            })

        # The stopping rule reasons about the checkpoint's *whole* sample against its target
        # length, but `evaluate` only counts this pass — the two differ by `already` whenever a
        # screened checkpoint is being topped up. Fold the held tally in here, where it is known.
        held_perfect = int(sum(held['perfect']))
        abandon_test = make_abandon_test(min_achievable, num_episodes, abandon_floor)
        should_abandon = (None if abandon_test is None else
                         lambda perfect, done: abandon_test(held_perfect + perfect, already + done))

        if pool is not None:
            scores, perfect_flags, rewards, elapsed, abandoned = pool.run(
                episodes, on_progress=on_round, should_abandon=should_abandon)
        else:
            scores, perfect_flags, rewards, elapsed, abandoned = evaluate(
                parallel_env, policy_action, episodes, on_round=on_round,
                should_abandon=should_abandon)
        if abandoned:
            held['abandoned'] = True
            progress['abandoned'] = progress.get('abandoned', 0) + 1
            progress['episodes_saved'] = (progress.get('episodes_saved', 0)
                                          + (episodes - len(scores)))
        held['scores'].extend(scores)
        held['perfect'].extend(perfect_flags)
        held['rewards'].extend(rewards)
        held['seconds'] += elapsed
        progress['measurements'] += 1
        progress[progress['stage'] + '_done'] = progress.get(progress['stage'] + '_done', 0) + 1
        progress['session_measurements'] += 1
        progress['session_episodes'] += len(scores)
        progress['session_seconds'] += elapsed

        row = build_row(step, held, selected_by.get(step))
        write_results(current_results(), complete=False)
        print('    perfect {0}/{1} = {2}%  (95% CI {3}-{4}%)'.format(
            row['perfect_games'], row['episodes'], row['perfect_percent'],
            row['perfect_ci95'][0], row['perfect_ci95'][1]))
        print('    score mean {0}  median {1}  min {2}  max {3}'.format(
            row['avg_score'], row['median_score'], row['min_score'], row['max_score']))

    if screen_episodes:
        print('\nstage 1: {0} checkpoints at the full {1} episodes (graph point '
              '{2:.0f}%, or explicitly named)'.format(len(full_steps), num_episodes,
                                                      ALWAYS_FULL_SINGLE))
        for index, step in enumerate(full_steps, 1):
            # Same top-up logic as stage 2: a checkpoint the gate stopped at 60 of 100, or one a kill
            # interrupted, needs only the remainder. Restarting it would throw away real episodes for
            # nothing, which is what the old resume did to every partial row it found.
            have = len(samples.get(step, {}).get('scores', []))
            if have >= num_episodes:
                continue
            label = ('full {0} of {1}'.format(index, len(full_steps)) if not have
                     else 'full {0} of {1}, topping up {2} to {3}'.format(
                         index, len(full_steps), have, num_episodes))
            measure(step, num_episodes - have, label, stage='full')

        # A carried screen already has its episodes, so it is not screened again. One that was cut
        # short of the screen depth — the abandonment gate, or a kill mid-checkpoint — is topped up by
        # the difference rather than restarted, which is the whole point of storing raw episodes.
        to_screen = []
        for step in screen_steps:
            have = len(samples.get(step, {}).get('scores', []))
            if have >= screen_episodes:
                continue
            to_screen.append((step, screen_episodes - have, have))
        reused = len(screen_steps) - len(to_screen)
        topped = sum(1 for _, _, have in to_screen if have)
        note = ''
        if reused or topped:
            note = ' ({0} already screened, {1} topped up)'.format(reused, topped)
        print('\nstage 2: screening {0} of {1} checkpoints at {2} episodes each{3}'.format(
            len(to_screen), len(screen_steps), screen_episodes, note))
        for index, (step, needed, have) in enumerate(to_screen, 1):
            label = ('screen {0} of {1}'.format(index, len(to_screen)) if not have
                     else 'screen {0} of {1}, topping up {2} to {3}'.format(
                         index, len(to_screen), have, screen_episodes))
            measure(step, needed, label, stage='screen')

        # Ranked among the screened only. The full tier is excluded because it already has the
        # measurement a confirmation slot would buy — spending one there would spend it on
        # finished work, and the whole point of the split is that these slots go to checkpoints
        # the graph did *not* already flag.
        finalists = pick_finalists(current_results(), confirm_count,
                                   already_full=set(resumed_steps) | set(full_steps))
        # Mandatory 20/20 screens can take this past the planned confirm count, so the totals are
        # corrected here rather than left to report >100% done. plan_stages could not know the
        # number: it depends on how the screens actually came out.
        overshoot = max(0, len(finalists) - plan['confirmed'])
        if overshoot:
            plan['confirmed'] = len(finalists)
            measurements_planned += overshoot
            episodes_planned += overshoot * whole_confirm_rounds
            print('    plan raised by {0} confirmation{1} for the perfect screens'.format(
                overshoot, '' if overshoot == 1 else 's'))
        print('\nstage 3: confirming the best {0} of {1} screened checkpoints at {2} episodes '
              '({3} more each)'.format(len(finalists), len(screen_steps), num_episodes,
                                       num_episodes - screen_episodes))
        for entry in finalists:
            print('    {0:>8}  screen {1:>5.1f}%   graph {2:>5.0f}%   surrounding {3}'.format(
                entry['step'], entry['perfect_percent'],
                entry.get('graph_single_eval') or 0,
                '-' if entry.get('graph_surrounding') is None
                else '{0:.1f}%'.format(entry['graph_surrounding'])))
        for index, entry in enumerate(finalists, 1):
            measure(entry['step'], num_episodes - screen_episodes,
                    'confirm {0} of {1}'.format(index, len(finalists)), stage='confirm')
    else:
        for index, step in enumerate(requested_steps, 1):
            measure(step, num_episodes,
                    '{0} of {1}'.format(len(resumed_by_step) + index, len(all_steps)),
                    stage='flat')

    results = current_results()
    write_results(results, complete=True)
    update_chart(force=True)
    print('\nwrote {0}'.format(out_path))

    print('\n{0:>9}  {1:>11}  {2:>11}  {3:>8}  {4:>16}  {5:>9}'.format(
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
            # Flagged because the rate on this row is over fewer episodes than its neighbours', so
            # it reads as "provably below the gate" rather than as a comparable measurement.
            '  abandoned at {0}'.format(row['episodes']) if row.get('abandoned') else ''))

    if min_achievable and progress.get('abandoned'):
        planned = progress['session_episodes'] + progress['episodes_saved']
        print('\nabandoned {0} checkpoints that could no longer reach {1}%, saving {2} of {3} '
              'episodes ({4:.0f}% of this session\'s planned work)'.format(
                  progress['abandoned'], min_achievable, progress['episodes_saved'], planned,
                  100.0 * progress['episodes_saved'] / max(1, planned)))
        print('    None of them could have reached the gate arithmetically, so no ranking changed.')

    if screen_episodes:
        perfect, episodes, count = equal_effort_pooled(samples, screen_episodes)
        low, high = wilson_interval(perfect, episodes)
        print('\npooled (first {0} episodes of every checkpoint, equal effort): {1}/{2} = {3:.1f}%'
              '  (95% CI {4:.1f}-{5:.1f}%)  over {6} checkpoints'.format(
                  screen_episodes, perfect, episodes, 100.0 * perfect / episodes,
                  100.0 * low, 100.0 * high, count))
        print('    this is the arm-level rate. Do not pool the rows in the output file instead:'
              '\n    the full-length rows hold {0}x the episodes of the screened ones and are the'
              '\n    arm\'s best by construction, so that figure reads high however good the '
              'policy is.'.format(num_episodes // max(1, screen_episodes)))
    else:
        perfect = sum(r['perfect_games'] for r in results)
        episodes = sum(r['episodes'] for r in results)
        low, high = wilson_interval(perfect, episodes)
        print('\npooled: {0}/{1} = {2:.1f}%  (95% CI {3:.1f}-{4:.1f}%)  over {5} checkpoints'.format(
            perfect, episodes, 100.0 * perfect / episodes,
            100.0 * low, 100.0 * high, len(results)))

    best = best_full_length_row(results, num_episodes)
    print('best checkpoint: {0} at {1}% (95% CI {2}-{3}%) over {4} episodes{5}'.format(
        best['step'], best['perfect_percent'], best['perfect_ci95'][0],
        best['perfect_ci95'][1], best['episodes'],
        '  [truncated — no checkpoint reached the abandonment gate]'
        if best['episodes'] < num_episodes else ''))
    print('\nPooled rates only compare across arms when the selection rule matches.')
    if pool is not None:
        pool.close()
    return 0


if __name__ == '__main__':
    system_multiprocessing.handle_main(lambda argv: sys.exit(main(argv)))
