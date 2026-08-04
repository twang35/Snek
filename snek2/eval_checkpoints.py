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

    # the normal close-out: screen wide, confirm the best 30, 3.6x cheaper than measuring
    # every selected checkpoint at 100 episodes
    EVAL_SCREEN_EPISODES=20 EVAL_WORKERS=10 PYTHONPATH=. python -u \
        eval_checkpoints.py <policy_name> top20

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

`EVAL_SCREEN_EPISODES=20` runs the close-out in two stages instead of one:

1. every selected checkpoint gets 20 episodes
2. the best `EVAL_CONFIRM_COUNT` (default 30) get 80 more, reaching the full 100

A finalist therefore ends with exactly `EVAL_EPISODES` episodes — the screen counts toward the
total rather than being thrown away — so its number is directly comparable with every arm
measured under the flat protocol. Checkpoints that do not make the cut keep their 20-episode row,
whose much wider Wilson interval says plainly how little it is worth.

**This costs 3.6x fewer episodes than measuring everything at 100, for the same answer.**
Simulated against the 937 checkpoints batch 10 measured at 100 episodes: crowning the arm's best
checkpoint under screening lands 3.04pp below its true best, against 2.82pp for the flat
protocol — a difference far smaller than either number. What makes that possible is that the
regret is dominated by 100 episodes being too few to rank a tight population, not by which
checkpoints got measured.

**Rank the pool on the screen rate, break ties on the surrounding graph rate.** 20 episodes
admit only 21 distinct values, so dozens of checkpoints arrive on the same one and the tie-break
does real work — see pick_finalists.

**Take the arm-level pooled rate from stage 1, not from the output file.** The rows in the file
have different episode counts, and the deep ones are by construction the best of the arm, so
pooling them weights the winners 5x and reads high. The stage-1 figure the run prints gives every
checkpoint identical effort and is the honest one. Same for best-checkpoint: it is taken over the
full-length rows only, because across hundreds of screens some will read 19/20 on luck alone.

**An early-abandonment gate was measured and rejected.** The obvious alternative — start every
checkpoint at 100 episodes and abandon it once it is clearly weak — barely helps here. Cutting
anything below 12/20 at the 20-episode mark is statistically safe (of the 157 checkpoints that
finished at >=80%, the expected number wrongly cut is 0.24, or 0.15%) but saves only 14% of the
episodes. The reason is the shape of the selected population: it is a tight blob between 60% and
80%, only 9.7% of it finishes below 60%, and a gate lenient enough to keep an 80% checkpoint
necessarily keeps almost everything above 60% too. Screening beats gating because it spends its
savings on the many mediocre checkpoints rather than the few bad ones.

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
                      only the best of them at full length (default 0, off)
    EVAL_CONFIRM_COUNT    how many screened checkpoints get promoted (default 30)
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
        # Wall-clock per checkpoint, so eval_progress.py can give an ETA from this run's own
        # throughput rather than a hardcoded guess. Strong policies play longer episodes and
        # measure slower, so a fixed estimate is wrong in both directions.
        'seconds': round(held['seconds'], 1),
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


def evaluate(parallel_env, policy_action, num_episodes, on_round=None):
    """Collects at least num_episodes, in whole rounds of one episode per worker.

    `on_round(round_index, rounds_total, perfect_so_far, episodes_so_far, per_round_perfect)`
    is called after each round if given. It exists so the running perfect rate can be
    persisted while a checkpoint is still in flight — a checkpoint takes ~5 minutes and
    without this the only observable state is "started" until it finishes.
    """
    worker_envs = parallel_env.pyenv.envs
    num_workers = len(worker_envs)
    rounds = -(-num_episodes // num_workers)  # ceil

    scores, perfect_flags, rewards = [], [], []
    per_round_perfect = []
    steps = 0
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

    elapsed = time.time() - start
    print('    {0} episodes in {1}s ({2} env steps/s)'.format(
        len(scores), round(elapsed, 1), round(steps / elapsed)))
    return scores, perfect_flags, rewards, elapsed


# Selection thresholds, in single-eval (10-episode) percentage points.
#
# A graph point is 10 episodes, so perfect_percent only ever takes the values 0, 10,
# ... 100. That makes these two numbers coarser than they look: the mandatory tier is the
# set {90, 100}, the fill band is exactly {60, 70, 80}, and everything at 50 or below is
# excluded.
ALWAYS_EVAL_SINGLE = 90.0   # every checkpoint at or above this is measured, even past `count`
MIN_EVAL_SINGLE = 60.0      # below this, a checkpoint is not worth 100 episodes
DEFAULT_COUNT = 20          # target total; the mandatory tier may exceed it

# How many screened checkpoints EVAL_SCREEN_EPISODES promotes to a full-length measurement.
# 30 came out of simulating protocols against the 937 checkpoints batch 10 measured at 100
# episodes: screening everything at 20 and confirming the top 30 costs 3.6x fewer episodes than
# measuring all of them at 100, for indistinguishable accuracy in the checkpoint it crowns
# (regret 3.04pp against 2.82pp for the arm's true best). Raising it buys very little, because
# the selected population is a tight blob between 60% and 80% where extra slots go to
# checkpoints that were never going to win.
DEFAULT_CONFIRM_COUNT = 30

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
    screen_episodes = int(os.environ.get('EVAL_SCREEN_EPISODES', 0))
    confirm_count = int(os.environ.get('EVAL_CONFIRM_COUNT', DEFAULT_CONFIRM_COUNT))
    if screen_episodes >= num_episodes:
        raise SystemExit(
            'EVAL_SCREEN_EPISODES={0} must be below EVAL_EPISODES={1} — a screen that already '
            'runs the full length has nothing left to confirm'.format(screen_episodes, num_episodes))
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

    def whole_rounds(episodes):
        """Episodes actually run for a request of `episodes`, rounded up to a whole round."""
        return -(-episodes // num_workers) * num_workers

    # Planned work, in the two units a reader cares about: how many checkpoint measurements will
    # happen, and how many episodes they add up to. A screening protocol measures some
    # checkpoints twice, so "checkpoints" alone no longer tracks progress and the episode count
    # is what gives eval_progress.py an honest ETA across stages of different lengths.
    if screen_episodes:
        confirmed = min(confirm_count, len(requested_steps))
        measurements_planned = len(resumed_steps) + len(requested_steps) + confirmed
        episodes_planned = (len(resumed_steps) * whole_rounds(num_episodes)
                            + len(requested_steps) * whole_rounds(screen_episodes)
                            + confirmed * whole_rounds(num_episodes - screen_episodes))
    else:
        measurements_planned = len(all_steps)
        episodes_planned = len(all_steps) * whole_rounds(num_episodes)

    # Resumed rows arrive already measured, so they start the count rather than being re-earned.
    # `session_*` covers only what this process has done, which is what an ETA has to be built
    # from: resumed rows carry the pace of whatever settings the earlier run used, and a
    # close-out relaunched at a different EVAL_WORKERS can differ by 3x. Averaging over every
    # row put b10b's ETA out by nearly 3x — 7h10m against a real ~2h.
    progress = {'measurements': len(resumed_steps),
                'session_measurements': 0, 'session_episodes': 0, 'session_seconds': 0.0}

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
                   'complete': complete,
                   'requested_steps': all_steps,
                   # Screening protocol, or None for the flat one-pass measurement. Recorded
                   # because a pooled rate only compares across arms when the selection rule
                   # matches, and this is part of the rule.
                   'screen_episodes': screen_episodes or None,
                   'confirm_count': confirm_count if screen_episodes else None,
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

    def measure(step, episodes, label):
        """Restores one checkpoint and adds `episodes` more episodes to its sample."""
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
                'running_percent': round(100.0 * total_perfect / total_episodes, 1),
                'per_round_perfect': per_round,
                'started_at': started_at,
            })

        scores, perfect_flags, rewards, elapsed = evaluate(
            parallel_env, policy_action, episodes, on_round=on_round)
        held['scores'].extend(scores)
        held['perfect'].extend(perfect_flags)
        held['rewards'].extend(rewards)
        held['seconds'] += elapsed
        progress['measurements'] += 1
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
        print('\nstage 1: screening {0} checkpoints at {1} episodes each'.format(
            len(requested_steps), screen_episodes))
        for index, step in enumerate(requested_steps, 1):
            measure(step, screen_episodes, 'screen {0} of {1}'.format(index, len(requested_steps)))

        # Snapshot the pooled rate now, while every screened checkpoint has had exactly the same
        # number of episodes. After stage 2 the finalists carry five times the weight of
        # everything else and, being the best of the arm, drag a pooled figure upward — so the
        # end-of-run pooled number over mixed lengths is not an arm-level rate at all. This one
        # is: equal effort per checkpoint, selected the same way, no promotion applied yet.
        screened = [r for r in current_results() if r['step'] not in resumed_by_step]
        screen_pooled = (sum(r['perfect_games'] for r in screened),
                         sum(r['episodes'] for r in screened), len(screened))

        finalists = pick_finalists(current_results(), confirm_count, resumed_by_step)
        print('\nstage 2: confirming the top {0} of {1} screened checkpoints at {2} episodes '
              '({3} more each)'.format(len(finalists), len(requested_steps), num_episodes,
                                       num_episodes - screen_episodes))
        for entry in finalists:
            print('    {0:>8}  screen {1:>5.1f}%   surrounding {2}'.format(
                entry['step'], entry['perfect_percent'],
                '-' if entry.get('graph_surrounding') is None
                else '{0:.1f}%'.format(entry['graph_surrounding'])))
        for index, entry in enumerate(finalists, 1):
            measure(entry['step'], num_episodes - screen_episodes,
                    'confirm {0} of {1}'.format(index, len(finalists)))
    else:
        for index, step in enumerate(requested_steps, 1):
            measure(step, num_episodes,
                    '{0} of {1}'.format(len(resumed_by_step) + index, len(all_steps)))

    results = current_results()
    write_results(results, complete=True)
    update_chart(force=True)
    print('\nwrote {0}'.format(out_path))

    print('\n{0:>9}  {1:>11}  {2:>11}  {3:>8}  {4:>16}  {5:>9}'.format(
        'step', 'graph eval', 'surrounding', 'perfect', '95% CI', 'avg score'))
    for row in results:
        graph = row.get('graph_single_eval')
        near = row.get('graph_surrounding')
        print('{0:>9}  {1:>11}  {2:>11}  {3:>7}%  {4:>7}-{5:<7}  {6:>9}'.format(
            row['step'],
            '-' if graph is None else '{0:.0f}%'.format(graph),
            '-' if near is None else '{0:.1f}%'.format(near),
            row['perfect_percent'],
            row['perfect_ci95'][0], row['perfect_ci95'][1], row['avg_score']))

    if screen_episodes:
        perfect, episodes, count = screen_pooled
        low, high = wilson_interval(perfect, episodes)
        print('\npooled (stage 1, equal effort per checkpoint): {0}/{1} = {2:.1f}%  '
              '(95% CI {3:.1f}-{4:.1f}%)  over {5} checkpoints'.format(
                  perfect, episodes, 100.0 * perfect / episodes,
                  100.0 * low, 100.0 * high, count))
        print('    this is the arm-level rate. Do not pool the rows in the output file instead:'
              '\n    the confirmed finalists hold {0}x the episodes of everything else and are '
              'the\n    best of the arm, so that figure reads high by construction.'.format(
                  num_episodes // max(1, screen_episodes)))
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
