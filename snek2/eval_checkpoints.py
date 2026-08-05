"""Measures a saved checkpoint's true perfect-game rate over many episodes.

The graph in runs/<policy>.png plots a 10-episode eval per point, so a single point
moves in 10-percentage-point jumps and its *value* is far too coarse to quote. This
script reloads specific checkpoints and evaluates each over a few hundred episodes,
which is the only way to state a policy's real rate.

Note the graph point is still a good *selector* even though it is a bad measurement:
a high 10-episode eval reliably marks a genuinely better checkpoint. See
select_top_checkpoints below.

Usage:

    cd snek2
    PYTHONPATH=. python -u eval_checkpoints.py <policy_name> <step> [<step> ...]
    PYTHONPATH=. python -u eval_checkpoints.py <policy_name> top[N]

    # the normal close-out. Screening is on by default: 20 episodes on every selected
    # checkpoint, then 80 more on the best 30 -- 3.6x cheaper than 100 on all of them
    EVAL_WORKERS=10 PYTHONPATH=. python -u eval_checkpoints.py <policy_name> top20

    # the flat one-pass protocol, which every arm before batch 10 was measured under
    EVAL_SCREEN_EPISODES=0 PYTHONPATH=. python -u eval_checkpoints.py <policy_name> top20

    # continue a close-out that was interrupted
    EVAL_RESUME=1 EVAL_WORKERS=10 PYTHONPATH=. python -u \
        eval_checkpoints.py <policy_name> top20

`top20` (or `top`, `top:20`) is the normal way to close out an arm. It selects on the
**single 10-episode eval** from the graph, using the surrounding perfect rate to rank within
a tier, under three rules:

- every checkpoint at **>=90%** is measured, even if that alone exceeds the N asked for
- remaining slots go to the best of the rest down to **>=60%**, up to N total
- nothing below 60% is measured at all

So the count is a target, not a quota: a strong arm runs N and a weak one may run 1 or 0.
Fewer than N is a normal outcome, and padding with sub-60% checkpoints would only depress
the pooled rate without adding a candidate for best-checkpoint.

Because a graph point is 10 episodes, `perfect_percent` only takes the values 0, 10, ... 100,
so these thresholds are coarser than they look: the mandatory tier is {90, 100} and the fill
band is exactly {60, 70, 80}.

**Why these numbers.** An earlier version measured everything at >=80%, which on a strong arm
meant 63 checkpoints and four hours. Restricting the mandatory tier to >=90% and capping the
total at 20 covers the most promising checkpoints at a fraction of the cost — measurement
showed 90% and 80% points have indistinguishable means (57.9% vs 58.6%), so an unbounded 80%
tier bought volume rather than information.

Adjacent steps are allowed through on purpose: 1000 train steps is enough to change the
perfect rate by tens of points, so neighbours are separate policies, not repeat samples.

**Pooled rates only compare across arms when the selection rule matches**, and this rule has
now changed twice. A variable checkpoint count, the 60% floor and the 20-checkpoint cap all
move the pooled number on their own. Compare best-checkpoint instead, or re-measure both arms
under the current rule.

**Results are written after every checkpoint**, so an interrupted run keeps everything measured
so far. The payload carries `complete: false` until the last checkpoint lands. `EVAL_RESUME=1`
picks such a run back up: relaunch the identical command and it skips whatever the output file
already holds at full length. A checkpoint that was only part-measured is redone from scratch
rather than topped up, so no run ever has to pool two summaries of the same checkpoint.

## Screening: EVAL_SCREEN_EPISODES

Screening is **on by default** (`EVAL_SCREEN_EPISODES=20`), which runs the close-out in three
stages instead of one:

1. every checkpoint whose **graph point is 100%** — ten perfect games out of ten — gets the full
   `EVAL_EPISODES` straight away. Uncapped, like the >=90% tier in select_top_checkpoints, and on
   a strong arm it is large: 47, 142, 7 and 146 checkpoints across batch 10's four arms.
   Explicitly named steps join this tier too, since naming one is a request to measure it.
2. everything else selected gets 20 episodes
3. the best `EVAL_CONFIRM_COUNT` (default 30) **of those screened** get 80 more, reaching the
   full 100

A promoted checkpoint therefore ends with exactly `EVAL_EPISODES` episodes — the screen counts
toward the total rather than being thrown away — so its number is directly comparable with every
arm measured under the flat protocol. Checkpoints that do not make the cut keep their 20-episode
row, whose much wider Wilson interval says plainly how little it is worth.

**Confirmation slots deliberately exclude the 100% tier**, which already has the measurement a
slot would buy. The point of the split is that those 30 slots go to checkpoints the training graph
did *not* already flag.

**This costs ~2.4x fewer episodes than measuring everything at 100** on a batch-10-shaped arm
(2.34x, 2.38x, 1.21x, 2.42x). Screening *everything* and confirming 30 was 3.6x, so the uncapped
full tier is what the extra episodes buy — a completeness guarantee on the strongest graph tier.

**Know what the 100% tier is and is not.** It is coverage, not a shortlist of likely champions.
Measured across batch 10, graph-100% checkpoints average 73.0% against 71.2% for graph-90% ones,
and the 90% tier holds the higher maximum (93% against 89%) *and* all four arms' best checkpoint,
because it is roughly four times larger. Finding the best checkpoint is what stage 3 is for, and
`EVAL_CONFIRM_COUNT=30` is thin for it: simulated on b10d, the arm's true best non-100% checkpoint
reaches the top 30 of 369 screened candidates only **57%** of the time. Raising the count to 100
takes that to **97%** and costs only 2.71x -> 2.20x. Consider it on any arm you care about.

**Rank the pool on the screen rate, break ties on the surrounding graph rate.** 20 episodes
admit only 21 distinct values, so dozens of checkpoints arrive on the same one and the tie-break
does real work — see pick_finalists.

**Take the arm-level pooled rate from the equal-effort figure the run prints, not from the output
file.** The rows in the file have different episode counts, and the deep ones are by construction
the best of the arm, so pooling them weights the winners 5x and reads high however good the policy
is. equal_effort_pooled() truncates every checkpoint to its first 20 episodes instead, which is a
valid sample of each and lets the 100% tier count too. Same care for best-checkpoint: it is taken
over the full-length rows only, because across hundreds of screens some will read 19/20 on luck.

**An early-abandonment gate was measured and rejected.** The obvious alternative — start every
checkpoint at 100 episodes and abandon it once it is clearly weak — barely helps here. Cutting
anything below 12/20 at the 20-episode mark is statistically safe (of the 157 checkpoints that
finished at >=80%, the expected number wrongly cut is 0.24, or 0.15%) but saves only 14% of the
episodes. The reason is the shape of the selected population: it is a tight blob between 60% and
80%, only 9.7% of it finishes below 60%, and a gate lenient enough to keep an 80% checkpoint
necessarily keeps almost everything above 60% too. Screening beats gating because it spends its
savings on the many mediocre checkpoints rather than the few bad ones.

## Early abandonment: EVAL_MIN_ACHIEVABLE

**On by default at 85%.** After every round, a checkpoint's *ceiling* — its rate if every remaining
episode were perfect — is compared against the threshold, and the measurement stops the moment the
ceiling falls below it. At 100 episodes that means "stop once more than 15 episodes have failed",
since 15 failures already cap the run at 85%.

This is free rather than a trade-off, and the reason is that the test is arithmetic rather than
predictive:

| property | why it holds |
|---|---|
| a checkpoint that would reach the gate is never stopped | the ceiling is monotone, and it only fires once the gate is unreachable |
| an abandoned row can never outrank a kept one | firing implies the row's own rate is below the gate |
| best-checkpoint is unaffected | it is already taken from full-length rows only |
| `pooled_equal_effort` is unaffected | the floor is never below `screen_episodes`, the depth it truncates to |

**Measured saving: full-length work drops to 70%** on batch 13's first 505 full-length rows, where
439 were already out of contention before their 100th episode. A 90% gate would drop it to 52% but
would leave the 85-89% band — which holds real hall-of-fame candidates — as truncated rows.

What it does cost: a row below the gate is shorter, so its own rate is noisier and is *not*
comparable with a full-length row. Such rows carry `abandoned: true` and print with an
`abandoned at N` marker, and the payload records `min_achievable` so a file measured under the gate
can be told apart from one measured without it. **Do not pool raw rows across the two**, which was
already true for the screened/full split and is now true for one more reason.

The floor exists because `equal_effort_pooled` skips rows shorter than `screen_episodes`, so
abandoning below it would silently delete checkpoints from the one arm-level figure meant to be
comparable across arms. At the shipped defaults the floor is slack — 100 episodes cannot reach a
sub-85% ceiling in the first 10 — but it binds as soon as either knob moves.

## Worker count: the throughput knob that matters

**`EVAL_WORKERS` is close to free, and low values are actively expensive.** Measured on one
checkpoint of `b10d`, seconds per episode: 1.03 at 2 workers, 0.33 at 10, 0.30 at 20. TensorFlow's
thread pool burns about a core whether the batch it is given has 2 rows or 20, so a small worker
count pays the full inference overhead for a fraction of the work — 2 workers is 3x slower *and*
worse per unit of CPU. Two processes at 10 workers each measure a 100-episode checkpoint in ~37s
at ~50% of a 14-core machine; the same pair at 2 workers took ~103s.

Prefer a worker count that divides `EVAL_EPISODES`. Episodes are rounded up to a whole number of
rounds, so 12 workers turn a 100-episode request into 108 and the rows stop matching the rest of
the arm.

XLA (`jit_compile=True`) was measured and is *worse* here — 0.38 s/episode against 0.32 — and
pinning TensorFlow to one thread made no reliable difference. Neither is used.

Environment:
    EVAL_EPISODES     episodes per checkpoint, rounded up to a whole number of
                      rounds (default 100)
    EVAL_WORKERS      parallel envs inside this process (default 10)
    EVAL_OUT_SUFFIX   appended to the output filename
    EVAL_SCREEN_EPISODES  screen every checkpoint at this many episodes first, then measure
                      only the best of them at full length (default 20; 0 turns screening off)
    EVAL_CONFIRM_COUNT    how many screened checkpoints get promoted (default 30)
    EVAL_MIN_ACHIEVABLE   stop measuring a checkpoint once it can no longer reach this perfect
                      rate even if every remaining episode is perfect (default 85; 0 disables)
    EVAL_ABANDON_FLOOR    never abandon before this many episodes (default 20, raised to
                      EVAL_SCREEN_EPISODES if that is larger)
    EVAL_RESUME       1 to skip checkpoints this run's own output file already measured at
                      full length, or a comma-separated list of suffixes to take them from
    EVAL_PERFECT_WAIT_MS  how long the visible window pauses on a perfect game
                      (default 400; the game default stalls the whole round)
    EVAL_RENDER       1 to show a game in a window (default 0, all workers headless)

**All workers are headless by default.** Rendering is the most expensive thing in an
eval by a wide margin — 163us per game step headless against 6050us in a real window —
and because ParallelPyEnvironment steps every worker together and waits for the slowest,
a single rendering worker paced all ten while the other nine idled. Turning it off made a
30-episode eval 5x faster (70.1s -> 14.0s). It cannot affect the numbers: render() only
draws, and with display=False it returns before doing even that.

**`EVAL_RENDER=1` puts the window back**, on worker 0 only, which is what you want when
watching a policy by hand. Running four checkpoints in parallel then gives four windows.

Two things about that window look like bugs and are not. Both are cosmetic: the
recorded results are unaffected, because only each worker's *first* episode of a
round is counted.

1. **It stops mid-game and the window closes.** A round runs until every worker has
   finished one episode. Workers that finish early keep being stepped by
   ParallelPyEnvironment (it steps all envs together) and auto-reset into fresh
   episodes that are *not* counted. So the visible worker is often part-way through a
   throwaway episode when the round ends, and the process exits after the last
   checkpoint, closing the window at whatever point it had reached.

2. **It used to freeze for seconds at a time.** snake_constants.PERFECT_GAME_WAIT_MS
   pauses on a win so a human can see it, and Snake.render() implements it with a
   blocking pygame.time.wait(). In an eval that blocks the whole round — every other
   worker waits on the parallel step — and with no event pumping during the wait macOS
   marks the window unresponsive. This script overrides it with EVAL_PERFECT_WAIT_MS
   (default 400ms) so wins are still visible without stalling.

**A live progress window always opens** — there is no flag to suppress it. It uses the same
cv2-via-pyformulas mechanism as training's graph: completed checkpoints, the one in flight
converging round by round, and a text block with the top 5 and an ETA. It refreshes every round,
costs ~0.1s a frame against a ~4s round, and writes the same picture to
evals/<policy_name>_eval_progress.png. If a window cannot be opened at all — headless, no cv2 —
it disables itself on the first failure and the eval carries on, so an unattended run needs no
configuration.

**`evals/` holds only the latest work.** Before writing anything, this script moves whatever
is already in `evals/` into a timestamped folder under `evals/archive/`, so the top-level
folder always shows just what the current eval or batch produced rather than accumulating
every chart from every arm ever measured. History is not lost, only moved — check
`evals/archive/` for anything earlier.

**Several processes starting together, as a batch does, do not archive each other.** The
archive step runs before any setup — before the checkpoint restore, before the first eval
round — so every process in a simultaneous batch clears out the *previous* batch before any
of them has written a chart of its own; there is nothing from the current batch yet for a
sibling to sweep up. Verified directly: four processes launched at once each end with their
own chart at the top level and no cross-archiving.

**Starting one *more* eval while an earlier one is still running is different.** The new
process's archive call moves the still-running one's current chart into `evals/archive/`
too — it has no way to tell "mid-run" from "finished". That is harmless rather than lossy:
the running process keeps writing to the same top-level path every round regardless of
whether anything is there, so its chart reappears within one round (a few seconds). Expect a
brief window where a genuinely active arm's chart looks archived rather than current.

It shows the *whole* job, pooling every result file for this policy, so running several processes
on one arm with different EVAL_OUT_SUFFIX gives each window the same consolidated view rather
than its own slice. Duplicate windows in that one case are the accepted price of never silently
having none.

To attach a window to an eval **already running**, use
`EVAL_PROGRESS_WINDOW_MODE=1 EVAL_PROGRESS_WATCH=20 python -u eval_progress.py <policy> [...]`,
which draws the identical chart from outside and can follow several arms at once.

Results go to runs/<policy_name>_checkpoint_evals<suffix>.json so they survive and
can be compared across sessions.

Two levels of parallelism are available and they compose: EVAL_WORKERS spreads
one checkpoint's episodes across worker envs, and several copies of this script
can run at once on different checkpoints. Give each copy its own
EVAL_OUT_SUFFIX in that case, or they overwrite each other's results; merge
afterwards with merge_checkpoint_evals().

**Use a throwaway EVAL_OUT_SUFFIX for anything exploratory** — a timing check, a CPU-load
calibration probe, or just watching what the `top20` selection picks before committing to a
full run. The first write to disk happens seconds in, at the very first round of the very
first checkpoint, and it unconditionally overwrites whatever was already at that path. A run
sharing its suffix with a prior complete measurement destroys that measurement the moment it
starts, whether or not it is later killed early. backup_previous_results() below keeps one
rolling `.previous` copy as a safety net, but a distinct suffix is the thing that actually
prevents this.
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
from tf_agents.agents.dqn import dqn_agent
from tf_agents.environments import parallel_py_environment
from tf_agents.environments import tf_py_environment
from tf_agents.specs import tensor_spec
from tf_agents.system import system_multiprocessing
from tf_agents.utils import common

import snake_constants
from snake_constants import EVALS_ARCHIVE_DIR, EVALS_DIR, POLICY_DIR, RUNS_DIR
from snake_environment import SnakeEnvironment
from snek2 import build_q_net


def backup_previous_results(out_path):
    """Copies an existing *complete* result file to `<out_path>.previous` before this run's
    first write can overwrite it.

    `write_results()` below rewrites the whole file starting from the very first round of the
    very first checkpoint — seconds into the run, not at the end — so a run killed early (a
    timing check, a CPU-load calibration probe, second-guessing a `top20` selection once the
    checkpoint count turns out huge) destroys a prior complete measurement with no warning if
    it happens to share this run's `EVAL_OUT_SUFFIX`. That cost this project a 246-checkpoint
    close-out once already. Prefer a throwaway `EVAL_OUT_SUFFIX` for anything exploratory —
    this backup is the safety net for when that doesn't happen, not a reason to skip it.

    A single rolling backup, not history: a second overwrite in a row replaces the same
    `.previous` file, so it protects the last complete run at this path, not every run ever
    made with this suffix.
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

    Returns (rows, steps) where `steps` is the set to skip. Only rows with at least
    `num_episodes` episodes count: a checkpoint that a killed run had only partly measured is
    re-measured from scratch rather than topped up, which keeps this a plain skip-list and
    avoids having to pool a prior row's summary statistics with fresh episodes. Re-measuring one
    checkpoint is cheap next to getting the arithmetic subtly wrong.

    A step appearing in two source files is loaded once — first file listed wins — because these
    are alternative records of the same frozen checkpoint, not extra samples to combine. Use
    merge_checkpoint_evals() when pooling repeat measurements is what you actually want.
    """
    rows, steps = [], set()
    for suffix in suffixes:
        path = os.path.join(RUNS_DIR, '{0}_checkpoint_evals{1}.json'.format(policy_name, suffix))
        if not os.path.exists(path):
            continue
        with open(path) as handle:
            payload = json.load(handle)
        for row in payload.get('results', []):
            if row.get('episodes', 0) >= num_episodes and row['step'] not in steps:
                rows.append(row)
                steps.add(row['step'])
    return sorted(rows, key=lambda r: r['step']), steps


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
    }


def skips_screening(meta, threshold=None):
    """True when a checkpoint goes straight to a full-length measurement, no screen first.

    Two cases qualify. A **graph point at `threshold`** (100% — ten perfect games out of ten) is
    the strongest selector this project has, and measuring every one of them completely is a
    coverage guarantee: the strongest tier is never represented by a 20-episode row. An
    **explicitly named step** qualifies too, because naming one is a request to measure it — the
    docs already say explicit steps bypass the selection thresholds, and screening one to 20
    episodes and possibly leaving it there would quietly break that.

    Note this tier is uncapped, like the >=90% tier in select_top_checkpoints, and on a strong arm
    it is large: 47, 142, 7 and 146 checkpoints across batch 10's four arms.

    `threshold` defaults to ALWAYS_FULL_SINGLE, resolved in the body rather than in the signature
    because the constants block sits below the helpers in this file and a default argument would
    be evaluated at import time.
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

    `already_full` names steps that came in from a resumed run at full length. They are
    excluded because they already have the measurement this stage would buy them, so letting
    them occupy a slot would spend it on work that is finished.
    """
    already_full = already_full or {}
    pool = [r for r in rows if r['step'] not in already_full]
    ranked = sorted(pool, key=lambda r: (-r['perfect_games'] / r['episodes'],
                                         -(r.get('graph_surrounding') or 0.0), r['step']))
    return ranked[:count]


def run_round(parallel_env, policy_action, worker_envs):
    """One episode per worker, every worker run to completion.

    Deliberately *not* "collect N finished episodes and stop". Truncating mid-flight throws away
    the episodes still running, and which ones those are is correlated with the outcome, so the
    surviving sample is not a random one. Running whole rounds costs some idle time on the fast
    workers and keeps the estimate unbiased.

    **The bias runs the opposite way to what this comment used to claim.** It said perfect games
    are the longest episodes there are, so truncation would drop them preferentially and read
    low. Measured over 11 samples of a strong policy, perfect games average ~1780 steps and
    non-perfect ones ~2200: a win ends the moment the board fills, while a policy that is about
    to fail circles until the starve budget runs out. So truncation would drop *failures*
    preferentially and read **high**. The conclusion is unchanged — do not truncate — but the
    direction matters to anyone reasoning about a variant of this loop.

    The idle time is real and measurable: across 20-40 workers, 20-35% of the env steps executed
    in a round belong to workers that had already finished and were being stepped into
    uncounted episodes. Giving each worker a fixed quota of episodes to run back to back,
    instead of a barrier every episode, recovers ~1.2x of that and stays unbiased, since every
    worker still contributes exactly the same number of counted episodes. Not implemented.
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
                perfect[i] = step_rewards[i] == snake_constants.PERFECT_GAME_REWARD
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

    Returns None when disabled, so the caller can skip the whole mechanism rather than pass a
    predicate that always says no.

    Two guards, and both matter:

    * **`floor_episodes`** — never stop before this many episodes, however hopeless. The floor is
      raised to the screen depth by the caller because `equal_effort_pooled` truncates every
      checkpoint to its first `screen_episodes` and *skips rows shorter than that*. Abandoning at
      or above the floor therefore leaves the arm-level pooled figure bit-for-bit identical to what
      the un-gated protocol would have produced. Abandoning below it would silently delete rows
      from that statistic, which is the one number in the output designed to be comparable
      across arms.
    * **`episodes_so_far >= target_episodes`** — a completed checkpoint is never "abandoned".

    The rule is exact, not predictive: it fires only when the remaining episodes *cannot*
    arithmetically carry the checkpoint to the threshold. So a checkpoint that would have finished
    at or above `min_achievable` is never stopped, and an abandoned row's own rate is always below
    the threshold — an abandoned row can never outrank a kept one.
    """
    if not min_achievable:
        return None

    def should_abandon(perfect_total, episodes_total):
        if episodes_total < floor_episodes or episodes_total >= target_episodes:
            return False
        return achievable_percent(perfect_total, episodes_total, target_episodes) < min_achievable

    return should_abandon


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


# Selection thresholds, in single-eval (10-episode) percentage points.
#
# A graph point is 10 episodes, so perfect_percent only ever takes the values 0, 10,
# ... 100. That makes these two numbers coarser than they look: the mandatory tier is the
# set {90, 100}, the fill band is exactly {60, 70, 80}, and everything at 50 or below is
# excluded.
ALWAYS_EVAL_SINGLE = 90.0   # every checkpoint at or above this is measured, even past `count`
MIN_EVAL_SINGLE = 60.0      # below this, a checkpoint is not worth 100 episodes
DEFAULT_COUNT = 20          # target total; the mandatory tier may exceed it

# Screening is the default close-out shape: every selected checkpoint gets this many episodes,
# then only the best DEFAULT_CONFIRM_COUNT are taken to EVAL_EPISODES. Set EVAL_SCREEN_EPISODES=0
# for the flat one-pass protocol every arm before batch 10 was measured under.
DEFAULT_SCREEN_EPISODES = 20

# Abandon a checkpoint mid-measurement once it can no longer reach this perfect rate, even if
# every remaining episode is perfect. `EVAL_MIN_ACHIEVABLE=0` turns it off.
#
# 85 rather than something closer to the record because the rule has to be *arithmetically* certain
# to be free: it fires only when `perfect + remaining < threshold`, so a checkpoint that would have
# finished at or above 85% is never stopped, and no ranking changes. Measured on batch 13's first
# 505 full-length rows, an 85% gate leaves the top tier untouched and cuts full-length work to
# **70%** — 439 of those 505 rows were already arithmetically out of contention before their 100th
# episode. A 90% gate cuts it to 52% but would leave the 85-89% band, which holds real
# hall-of-fame candidates, as truncated rows.
#
# What it costs: rows below the gate are shorter, so their own rates are noisier and cannot be
# compared on equal footing with full-length ones. The pooled arm-level figure is untouched, since
# the floor below is never lower than the screen depth `equal_effort_pooled` truncates to.
DEFAULT_MIN_ACHIEVABLE = 85.0

# Never abandon before this many episodes. Raised to the screen depth at startup, so an abandoned
# row is always long enough to still count in equal_effort_pooled — see make_abandon_test.
DEFAULT_ABANDON_FLOOR = 20

# Graph single-eval at or above which a checkpoint skips the screen and is measured at full length
# straight away. 100% means ten perfect games out of ten, the strongest signal the training graph
# produces, and the tier is uncapped — see skips_screening.
#
# Worth knowing what this tier is and is not. It is a **coverage** guarantee: every checkpoint the
# graph called perfect gets a real number. It is *not* where the champion usually lives — measured
# across batch 10, graph-100% checkpoints average 73.0% against 71.2% for graph-90% ones, and the
# 90% tier holds both the higher maximum (93% against 89%) and all four arms' best checkpoint,
# because it is roughly four times larger. Finding the best checkpoint is what EVAL_CONFIRM_COUNT
# is for.
ALWAYS_FULL_SINGLE = 100.0

# How many screened checkpoints EVAL_SCREEN_EPISODES promotes to a full-length measurement.
#
# 100, raised from 30 on 2026-08-03, because 30 was losing the champion outright. Simulated on
# b10d's 369 screened candidates, the arm's genuinely best checkpoint reached the top 30 only
# **57%** of the time — a 20-episode screen simply cannot rank a population clustered between 60%
# and 80%. Recall against cost, with the uncapped 100%-graph tier included in both:
#
#     confirm  recall  episodes  vs a flat pass
#          30     57%    24,380           2.71x
#          50     85%    25,980           2.54x
#         100     97%    29,980           2.20x
#         150     99%    33,980           1.94x
#
# 100 is the knee: it converts a coin-flip on the headline number into near-certainty for ~23%
# more episodes. Going further trades real cost for a percentage point.
DEFAULT_CONFIRM_COUNT = 100

# Seconds between live-chart refreshes. A frame is ~0.1s and a round ~4s, so this only bites
# early in a run when episodes are short and rounds finish fast.
CHART_MIN_INTERVAL = 2.0


def select_top_checkpoints(policy_name, available, count=DEFAULT_COUNT, window=10):
    """Every checkpoint at >=90% single eval, then the best of >=60% up to `count` total.

    Three rules, in order:

    1. **Every** checkpoint whose single eval was `ALWAYS_EVAL_SINGLE` or better is
       measured, even if that alone exceeds `count`. A 10-episode eval reading 9+ perfect
       is the strongest signal available and there is no reason to drop one because the
       last slot ran out.
    2. Remaining slots go to the highest single evals at or above `MIN_EVAL_SINGLE`,
       ranked by the surrounding perfect rate within an equal-eval tier.
    3. **Nothing** below `MIN_EVAL_SINGLE` is measured at all. Below 60%, 100 episodes buys
       a precise number for a checkpoint that was never going to be the arm's best, and it
       displaces a slot that could go to a real candidate.

    Fewer than `count` checkpoints is a normal outcome, not an error: a weak arm may have
    only two or three points above the floor, and padding the list with 30% checkpoints
    would only drag its pooled rate down while telling us nothing.

    **The tiers used to be >=80% mandatory with a cap of 10.** That made a strong arm
    ruinously expensive to measure — `b8f-disc9975seed2` presented 63 checkpoints at >=80%,
    four hours of evaluation — while measurement showed 90% and 80% points have
    indistinguishable mean true rates (57.9% vs 58.6% over 88 checkpoints). So the wide
    mandatory tier was buying volume, not information. Restricting it to >=90% and raising
    the cap to 20 keeps every strongest signal and a deep fill band at a bounded cost.

    Ranking on the raw single eval rather than a smoothed region is a measured result, not
    an assumption: raw correlates +0.64 with the 100-episode measurement where the smoothed
    region rate correlates -0.40 *as a selector across the full range*.

    Within the high band the picture reverses, which is why the surrounding rate is used to
    order the fill tier rather than merely break exact ties. Across 88 checkpoints that had
    all already cleared 80%, the surrounding rate correlated **+0.48** with the true rate
    while the graph value itself managed only **+0.10**. Once a checkpoint has spiked, the
    region it sits in is the better guide to whether the spike will hold up.

    An outlier eval is **not luck** — those checkpoints really are better than their
    neighbours. Measured against the checkpoints 1000 steps either side, outliers won 3 of
    3 comparisons by 9.0, 11.5 and 27.5 points, one of them reading 8% / 35% / 7% across
    three consecutive checkpoints. The binomial agrees: if a policy's true rate were 27%,
    a 10-episode eval showing 7+ perfect games has probability 0.006.
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


def merge_checkpoint_evals(policy_name, suffixes=None, out_suffix='_merged'):
    """Combines several result files for one policy into one, and writes it.

    Splitting an arm's checkpoints across parallel processes is the only way to use more than
    one core per arm, and each process needs its own EVAL_OUT_SUFFIX or they overwrite each
    other. This puts the pieces back together.

    **A step measured more than once is combined, not deduplicated.** Repeat measurements of
    one frozen checkpoint are independent samples of the same quantity, so summing episodes and
    perfect games is the statistically correct treatment and tightens the interval — dropping
    one would throw away half the data. `perfect_percent` and the Wilson interval are
    recomputed from the combined counts.

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
    num_episodes = int(os.environ.get('EVAL_EPISODES', 100))
    num_workers = int(os.environ.get('EVAL_WORKERS', 10))
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
    # Rendering is off by default because it was the single slowest thing in an eval, by a
    # wide margin. Measured per game step: 163us headless, 2449us with display=True on the
    # dummy driver, 6050us in a real window. ParallelPyEnvironment steps every worker
    # together and does not return until the slowest finishes, so one rendering worker set
    # the pace for all ten while the other nine idled — a ~37x penalty on the critical path.
    # EVAL_RENDER=1 brings the window back for watching a game by hand.
    render_worker = os.environ.get('EVAL_RENDER', '0') not in ('0', '', 'false', 'False')

    ckpt_dir = POLICY_DIR + policy_name
    available = {int(f[len('ckpt-'):].split('.')[0])
                 for f in os.listdir(ckpt_dir) if f.startswith('ckpt-') and f.endswith('.index')}

    # Spelled `top` rather than `--top`: tf_agents' handle_main routes argv through absl,
    # which rejects any unregistered `--flag` before main() is reached.
    if argv[2].startswith('top'):
        rest = argv[2][len('top'):].lstrip(':=')
        count = int(rest) if rest else (int(argv[3]) if len(argv) > 3 else DEFAULT_COUNT)
        requested_steps, selected_by = select_top_checkpoints(policy_name, available, count)
    else:
        requested_steps = [int(a) for a in argv[2:]]
        selected_by = {step: {'selected_by': 'explicit'} for step in requested_steps}
        missing = [s for s in requested_steps if s not in available]
        if missing:
            raise SystemExit('no checkpoint for step(s) {0} in {1}'.format(missing, ckpt_dir))

    # Resume before the expensive setup, so a run that has nothing left to do says so in
    # seconds rather than after building an agent and 20 worker processes.
    suffix = os.environ.get('EVAL_OUT_SUFFIX', '')
    resumed_rows, resumed_steps = load_finished_results(
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
            # Resume means continue, so an arm that started under the flat protocol finishes under
            # it. Otherwise switching the *default* would silently leave one arm holding some rows
            # at 100 episodes and some at 20 — not comparable with each other, let alone with the
            # arm it is meant to be compared against. Ask for it explicitly to override.
            print('    screening off: these {0} resumed rows were measured at full length, so the '
                  'rest of the arm will be too (EVAL_SCREEN_EPISODES=20 to override)'.format(
                      len(skipped)))
            screen_episodes = 0

    print('policy {0}: evaluating {1} checkpoints x {2} episodes on {3} workers'.format(
        policy_name, len(requested_steps), num_episodes, num_workers))

    # Same env, network and agent as training, so the restored weights line up.
    spec_env = SnakeEnvironment(discount=0.99, display=False, policy_name=policy_name)
    spec_env.reset()
    spec_tf_env = tf_py_environment.TFPyEnvironment(spec_env)

    action_tensor_spec = tensor_spec.from_spec(spec_env.action_spec())
    num_actions = action_tensor_spec.maximum - action_tensor_spec.minimum + 1
    # Shared with training and watch.py, and it reads SNEK_FC_LAYERS like training does. This
    # used to hardcode (50, 100, 50) while training took the override, so a run with
    # SNEK_FC_LAYERS set would have been measured against the wrong network — silently, since
    # restore() below uses expect_partial().
    q_net = build_q_net(num_actions)

    global_step = tf.compat.v1.train.get_or_create_global_step()
    agent = dqn_agent.DdqnAgent(
        spec_tf_env.time_step_spec(),
        spec_tf_env.action_spec(),
        q_network=q_net,
        epsilon_greedy=0.0,  # eval is greedy; epsilon only affects the collect policy
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        td_errors_loss_fn=common.element_wise_huber_loss,
        target_update_period=8,
        train_step_counter=global_step)
    agent.initialize()

    def make_headless_worker():
        os.environ['SDL_VIDEODRIVER'] = 'dummy'
        return SnakeEnvironment(discount=0.99, display=False, policy_name=policy_name)

    def make_visible_worker():
        # Each worker is its own process, so clearing the dummy driver here gives
        # exactly one real window per eval process — several checkpoints evaluated
        # in parallel each get their own window to watch.
        os.environ.pop('SDL_VIDEODRIVER', None)
        # Snake.render() implements the perfect-game celebration with a *blocking*
        # pygame.time.wait(), and the game default is 5000ms. Inside an eval that stalls
        # the entire round, because parallel_env.step() does not return until every worker
        # has stepped — so one visible win froze all 10 workers for 5 seconds, and with no
        # event pumping during the wait the window also went unresponsive. On a 40%-perfect
        # policy that is ~20s wasted per checkpoint and a window that looks hung.
        snake_constants.PERFECT_GAME_WAIT_MS = perfect_wait_ms
        return SnakeEnvironment(discount=0.99, display=True, policy_name=policy_name)

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

    # Mirrors the keys common.Checkpointer uses in snek2.py, so a specific
    # ckpt-<step> can be restored instead of only the latest.
    checkpoint = tf.train.Checkpoint(agent=agent, policy=agent.policy, global_step=global_step)

    # Run inference inside a tf.function instead of eager. Training already does this for its
    # collect policy (PyTFEagerPolicy(..., use_tf_function=True) in snek2.py) but this script
    # called agent.policy.action() directly, paying full eager dispatch on every step of every
    # episode. For this network, batched across 10 workers, that is 1421us per call eager
    # against 208us wrapped — 6.8x — for byte-identical actions.
    #
    # Traced once and reused across every checkpoint: restoring new weights writes into the
    # same variables, so it does not force a retrace.
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
                   # EVAL_WORKERS, recorded once at startup because it is fixed for the life of
                   # the process — the ParallelPyEnvironment is built once. The chart used to
                   # infer it as episodes_so_far // round, which is wrong for any checkpoint
                   # being topped up: the numerator counts the screen episodes already on file
                   # and the denominator only counts this pass's rounds, so a 20-episode screen
                   # growing to 100 reported 30, 20, 16, 15, 14, 13, 12, 12 workers over its
                   # eight rounds. Read this instead.
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
                   # The arm-level rate, computed the only way that means anything once rows have
                   # different depths: every checkpoint truncated to its first `screen_episodes`.
                   # Pooling the rows instead weights the full-length ones — the arm's best by
                   # construction — five times as heavily, so the chart was displaying a figure its
                   # own documentation says not to use. None for a flat run, where pooling the rows
                   # already gives equal effort.
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

    # Live progress window, the same mechanism training uses for its own graph: render a
    # matplotlib figure and push the pixels into a cv2 window via pyformulas. Refreshed from
    # write_results(), so it advances every round rather than once a checkpoint — a frame costs
    # ~0.1s against a ~4s round, and CHART_MIN_INTERVAL keeps that bounded if rounds are quick.
    #
    # Always on. There is no switch, deliberately: an eval you cannot follow is the thing this
    # was built to fix, and a flag to suppress it only ever got used by mistake. If the window
    # cannot be opened at all — headless, no cv2 — update_chart() disables itself on the first
    # failure and the eval carries on, so nothing needs configuring for an unattended run.
    #
    # eval_progress.live_frame() reads *every* result file for this policy, not just this
    # process's, so with several EVAL_OUT_SUFFIX processes on one arm each window shows the whole
    # job rather than its own slice, and they will be near-identical. Duplicate windows are the
    # price of never silently having none.
    chart_path = os.path.join(EVALS_DIR, '{0}_eval_progress.png'.format(policy_name))
    chart = {'screen': None, 'last': 0.0, 'off': False}

    def update_chart(force=False):
        if chart['off']:
            return
        now = time.time()
        if not force and now - chart['last'] < CHART_MIN_INTERVAL:
            return
        chart['last'] = now
        try:
            import eval_progress
            frame = eval_progress.live_frame(policy_name, chart_path)
            if frame is None:
                return
            if chart['screen'] is None:
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
    samples = {}
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
            measure(step, num_episodes, 'full {0} of {1}'.format(index, len(full_steps)),
                    stage='full')

        print('\nstage 2: screening the other {0} checkpoints at {1} episodes each'.format(
            len(screen_steps), screen_episodes))
        for index, step in enumerate(screen_steps, 1):
            measure(step, screen_episodes, 'screen {0} of {1}'.format(index, len(screen_steps)),
                    stage='screen')

        # Ranked among the screened only. The full tier is excluded because it already has the
        # measurement a confirmation slot would buy — spending one there would spend it on
        # finished work, and the whole point of the split is that these slots go to checkpoints
        # the graph did *not* already flag.
        finalists = pick_finalists(current_results(), confirm_count,
                                   already_full=set(resumed_steps) | set(full_steps))
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

    # Best-checkpoint over the full-length rows only. A screened-out checkpoint has a 20-episode
    # rate whose interval is ~5x wider, so letting one win on a lucky 18/20 would crown a
    # checkpoint the protocol deliberately declined to measure.
    full_length = [r for r in results if r['episodes'] >= num_episodes] or results
    best = max(full_length, key=lambda r: r['perfect_percent'])
    print('best checkpoint: {0} at {1}% (95% CI {2}-{3}%) over {4} episodes'.format(
        best['step'], best['perfect_percent'], best['perfect_ci95'][0],
        best['perfect_ci95'][1], best['episodes']))
    print('\nPooled rates only compare across arms when the selection rule matches.')
    return 0


if __name__ == '__main__':
    system_multiprocessing.handle_main(lambda argv: sys.exit(main(argv)))
