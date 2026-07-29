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

`top10` (or `top`, `top:10`) picks the N biggest **single-eval outliers** — highest
10-episode eval on the graph — using the surrounding perfect rate only to break ties.
This is the normal way to close out an arm: 10 checkpoints x 100 episodes, ~30 minutes on
an idle machine.

Outliers are **not luck**. Measured against the checkpoints 1000 steps either side, they
won 3 of 3 comparisons by 9.0, 11.5 and 27.5 points, and raw single-eval rate correlates
+0.64 with the true 100-episode rate where the smoothed region rate correlates -0.40.
Selecting on smoothed rate, as an earlier version did, systematically picked worse
checkpoints.

Adjacent steps are allowed through on purpose: 1000 train steps is enough to change the
perfect rate by tens of points, so neighbours are separate policies, not repeat samples.

Environment:
    EVAL_EPISODES     episodes per checkpoint, rounded up to a whole number of
                      rounds (default 100)
    EVAL_WORKERS      parallel envs inside this process (default 10)
    EVAL_OUT_SUFFIX   appended to the output filename
    EVAL_PERFECT_WAIT_MS  how long the visible window pauses on a perfect game
                      (default 400; the game default of 5000 stalls the whole round)

**Worker 0 renders a visible window**, so each eval process shows one game as it
plays; the remaining workers are headless. Running four checkpoints in parallel
therefore gives four windows to watch.

Two things about that window look like bugs and are not. Both are cosmetic: the
recorded results are unaffected, because only each worker's *first* episode of a
round is counted.

1. **It stops mid-game and the window closes.** A round runs until every worker has
   finished one episode. Workers that finish early keep being stepped by
   ParallelPyEnvironment (it steps all envs together) and auto-reset into fresh
   episodes that are *not* counted. So the visible worker is often part-way through a
   throwaway episode when the round ends, and the process exits after the last
   checkpoint, closing the window at whatever point it had reached.

2. **It used to freeze for ~5s at a time.** snake_constants.PERFECT_GAME_WAIT_MS
   defaults to 5000 so a human can see a win, and Snake.render() implements it with a
   blocking pygame.time.wait(). In an eval that blocks the whole round — every other
   worker waits on the parallel step — and with no event pumping during the wait macOS
   marks the window unresponsive. This script now sets it to EVAL_PERFECT_WAIT_MS
   (default 400ms) so wins are still visible without stalling.

Results go to runs/<policy_name>_checkpoint_evals<suffix>.json so they survive and
can be compared across sessions.

Two levels of parallelism are available and they compose: EVAL_WORKERS spreads
one checkpoint's episodes across worker envs, and several copies of this script
can run at once on different checkpoints. Give each copy its own
EVAL_OUT_SUFFIX in that case, or they overwrite each other's results; merge
afterwards with merge_checkpoint_evals().
"""
import json
import os
import sys
import time

os.environ['SDL_VIDEODRIVER'] = 'dummy'  # must precede any pygame import
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import numpy as np
import tensorflow as tf
from tf_agents.agents.dqn import dqn_agent
from tf_agents.environments import parallel_py_environment
from tf_agents.environments import tf_py_environment
from tf_agents.networks import sequential
from tf_agents.specs import tensor_spec
from tf_agents.system import system_multiprocessing
from tf_agents.utils import common

import snake_constants
from snake_constants import POLICY_DIR, RUNS_DIR
from snake_environment import SnakeEnvironment
from snek2 import dense_layer


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


def run_round(parallel_env, policy, worker_envs):
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
        action_step = policy.action(time_step)
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


def evaluate(parallel_env, policy, num_episodes):
    """Collects at least num_episodes, in whole rounds of one episode per worker."""
    worker_envs = parallel_env.pyenv.envs
    num_workers = len(worker_envs)
    rounds = -(-num_episodes // num_workers)  # ceil

    scores, perfect_flags, rewards = [], [], []
    steps = 0
    start = time.time()
    for index in range(rounds):
        round_scores, round_perfect, round_rewards, round_steps = run_round(
            parallel_env, policy, worker_envs)
        scores.extend(round_scores)
        perfect_flags.extend(round_perfect)
        rewards.extend(round_rewards)
        steps += round_steps
        print('    round {0}/{1}: {2} episodes, {3} perfect'.format(
            index + 1, rounds, len(round_scores), sum(round_perfect)))

    elapsed = time.time() - start
    print('    {0} episodes in {1}s ({2} env steps/s)'.format(
        len(scores), round(elapsed, 1), round(steps / elapsed)))
    return scores, perfect_flags, rewards


def select_top_checkpoints(policy_name, available, count=10, window=10):
    """The `count` biggest single-eval outliers, ties broken by surrounding perfect rate.

    Selection is built on a measured result rather than an assumption. Measuring `b4c`
    both ways showed the **raw single 10-episode eval is the better predictor** of a
    checkpoint's true rate, correlating +0.64 with the 100-episode measurement against
    **-0.40** for the smoothed region rate. Smoothing is not merely weaker, it is
    anti-predictive, because a window describes the *region* while the thing being measured
    is the *checkpoint*.

    An outlier eval is **not luck** — those checkpoints really are better than their
    neighbours. Measured against the checkpoints 1000 steps either side, outliers won 3 of
    3 comparisons by 9.0, 11.5 and 27.5 points, one of them reading 8% / 35% / 7% across
    three consecutive checkpoints. The binomial agrees: if a policy's true rate were 27%,
    a 10-episode eval showing 7+ perfect games has probability 0.006.

    So rank on the single eval and use the surrounding rate only to break ties. Among
    checkpoints that spiked equally, the one sitting in a stronger region is the better bet,
    but the spike leads.
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
    ranked = sorted(candidates, key=lambda c: (-c['single'], -c['smoothed'], c['step']))
    chosen = ranked[:count]

    print('selected {0} of {1} available checkpoints, biggest single-eval outliers '
          '(ties broken by surrounding rate):'.format(len(chosen), len(candidates)))
    for entry in sorted(chosen, key=lambda c: c['step']):
        print('    {0:>8}  single eval {1:>5.1f}%   surrounding {2:>5.1f}%'.format(
            entry['step'], entry['single'], entry['smoothed']))
    return [c['step'] for c in sorted(chosen, key=lambda c: c['step'])], \
        {c['step']: {'selected_by': 'outlier',
                     'single_eval': c['single'],
                     'surrounding': round(c['smoothed'], 1)} for c in chosen}


def main(argv):
    if len(argv) < 3:
        print(__doc__)
        return 1
    policy_name = argv[1]
    num_episodes = int(os.environ.get('EVAL_EPISODES', 100))
    num_workers = int(os.environ.get('EVAL_WORKERS', 10))
    perfect_wait_ms = int(os.environ.get('EVAL_PERFECT_WAIT_MS', 400))

    ckpt_dir = POLICY_DIR + policy_name
    available = {int(f[len('ckpt-'):].split('.')[0])
                 for f in os.listdir(ckpt_dir) if f.startswith('ckpt-') and f.endswith('.index')}

    # Spelled `top` rather than `--top`: tf_agents' handle_main routes argv through absl,
    # which rejects any unregistered `--flag` before main() is reached.
    if argv[2].startswith('top'):
        rest = argv[2][len('top'):].lstrip(':=')
        count = int(rest) if rest else (int(argv[3]) if len(argv) > 3 else 10)
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
    fc_layer_params = (50, 100, 50)
    dense_layers = [dense_layer(units) for units in fc_layer_params]
    q_values_layer = tf.keras.layers.Dense(
        num_actions,
        activation=None,
        kernel_initializer=tf.keras.initializers.RandomUniform(minval=-0.03, maxval=0.03),
        bias_initializer=tf.keras.initializers.Constant(0.0))
    q_net = sequential.Sequential(dense_layers + [q_values_layer])

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

    # Worker 0 renders; the rest are headless. A rendering worker is slower, which
    # only means it and its round-mates take a little longer — episodes are i.i.d.
    # across workers, so which worker produced one carries no information.
    constructors = [make_visible_worker] + [make_headless_worker] * (num_workers - 1)
    parallel_env = tf_py_environment.TFPyEnvironment(
        parallel_py_environment.ParallelPyEnvironment(constructors))

    # Mirrors the keys common.Checkpointer uses in snek2.py, so a specific
    # ckpt-<step> can be restored instead of only the latest.
    checkpoint = tf.train.Checkpoint(agent=agent, policy=agent.policy, global_step=global_step)

    results = []
    for step in requested_steps:
        path = os.path.join(ckpt_dir, 'ckpt-{0}'.format(step))
        print('\ncheckpoint {0}'.format(step))
        checkpoint.restore(path).expect_partial()
        restored = int(global_step.numpy())
        if restored != step:
            print('    warning: global_step reads {0}, expected {1}'.format(restored, step))

        scores, perfect_flags, rewards = evaluate(parallel_env, agent.policy, num_episodes)
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
        }
        results.append(row)
        print('    perfect {0}/{1} = {2}%  (95% CI {3}-{4}%)'.format(
            perfect, len(scores), row['perfect_percent'], row['perfect_ci95'][0], row['perfect_ci95'][1]))
        print('    score mean {0}  median {1}  min {2}  max {3}'.format(
            row['avg_score'], row['median_score'], row['min_score'], row['max_score']))

    suffix = os.environ.get('EVAL_OUT_SUFFIX', '')
    out_path = os.path.join(RUNS_DIR, '{0}_checkpoint_evals{1}.json'.format(policy_name, suffix))
    payload = {'policy_name': policy_name, 'episodes_per_checkpoint': num_episodes, 'results': results}
    partial = out_path + '.partial'
    with open(partial, 'w') as handle:
        json.dump(payload, handle, indent=2)
    os.replace(partial, out_path)
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
