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
so far. The payload carries `complete: false` until the last checkpoint lands.

Environment:
    EVAL_EPISODES     episodes per checkpoint, rounded up to a whole number of
                      rounds (default 100)
    EVAL_WORKERS      parallel envs inside this process (default 10)
    EVAL_OUT_SUFFIX   appended to the output filename
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


def run_round(parallel_env, policy_action, worker_envs):
    """One episode per worker, every worker run to completion.

    Deliberately *not* "collect N finished episodes and stop": stopping mid-flight
    discards the episodes still running, and perfect games are the longest episodes
    there are, so truncation drops them preferentially and biases the perfect rate
    down. Running whole rounds costs a little idle time on the fast workers and
    keeps the sample unbiased.
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

    suffix = os.environ.get('EVAL_OUT_SUFFIX', '')
    out_path = os.path.join(RUNS_DIR, '{0}_checkpoint_evals{1}.json'.format(policy_name, suffix))

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
                   'checkpoints_requested': len(requested_steps),
                   'complete': complete,
                   'requested_steps': requested_steps,
                   # The checkpoint being measured right now, updated every round, or None
                   # between checkpoints. eval_progress.py renders this; without it a
                   # ~5-minute checkpoint is invisible until it lands.
                   'in_flight': in_flight,
                   'updated_at': time.time(),
                   'results': results}
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

    results = []
    for step in requested_steps:
        path = os.path.join(ckpt_dir, 'ckpt-{0}'.format(step))
        print('\ncheckpoint {0} ({1} of {2})'.format(
            step, len(results) + 1, len(requested_steps)))
        checkpoint.restore(path).expect_partial()
        restored = int(global_step.numpy())
        if restored != step:
            print('    warning: global_step reads {0}, expected {1}'.format(restored, step))

        started_at = time.time()

        def on_round(round_index, rounds_total, perfect_so_far, episodes_so_far, per_round):
            write_results(results, complete=False, in_flight={
                'step': step,
                'round': round_index,
                'rounds_total': rounds_total,
                'perfect_so_far': perfect_so_far,
                'episodes_so_far': episodes_so_far,
                'running_percent': round(100.0 * perfect_so_far / episodes_so_far, 1),
                'per_round_perfect': per_round,
                'started_at': started_at,
            })

        scores, perfect_flags, rewards, elapsed = evaluate(
            parallel_env, policy_action, num_episodes, on_round=on_round)
        perfect = sum(perfect_flags)
        low, high = wilson_interval(perfect, len(scores))
        meta = selected_by.get(step, {'selected_by': 'explicit'})
        row = {
            'step': step,
            'selected_by': meta['selected_by'],
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
            'avg_reward': round(float(np.mean(rewards)), 2),
            # Wall-clock per checkpoint, so eval_progress.py can give an ETA from this run's
            # own throughput rather than a hardcoded guess. Strong policies play longer
            # episodes and measure slower, so a fixed estimate is wrong in both directions.
            'seconds': round(elapsed, 1),
        }
        results.append(row)
        write_results(results, complete=len(results) == len(requested_steps))
        print('    perfect {0}/{1} = {2}%  (95% CI {3}-{4}%)'.format(
            perfect, len(scores), row['perfect_percent'], row['perfect_ci95'][0], row['perfect_ci95'][1]))
        print('    score mean {0}  median {1}  min {2}  max {3}'.format(
            row['avg_score'], row['median_score'], row['min_score'], row['max_score']))

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

    perfect = sum(r['perfect_games'] for r in results)
    episodes = sum(r['episodes'] for r in results)
    low, high = wilson_interval(perfect, episodes)
    print('\npooled: {0}/{1} = {2:.1f}%  (95% CI {3:.1f}-{4:.1f}%)  over {5} checkpoints'.format(
        perfect, episodes, 100.0 * perfect / episodes,
        100.0 * low, 100.0 * high, len(results)))
    best = max(results, key=lambda r: r['perfect_percent'])
    print('best checkpoint: {0} at {1}% (95% CI {2}-{3}%)'.format(
        best['step'], best['perfect_percent'], best['perfect_ci95'][0], best['perfect_ci95'][1]))
    print('\nPooled rates only compare across arms when the selection rule matches.')
    return 0


if __name__ == '__main__':
    system_multiprocessing.handle_main(lambda argv: sys.exit(main(argv)))
