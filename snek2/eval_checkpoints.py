"""Measures a saved checkpoint's true perfect-game rate over many episodes.

The graph in runs/<policy>.png plots a 10-episode eval per point, so a single
point moves in 10-percentage-point jumps and a lucky eval reads far above the
policy's real rate. This script reloads specific checkpoints and evaluates each
over a few hundred episodes instead, which is the only way to compare against a
remembered figure like theSchlong's 76%.

Usage:

    cd snek2
    PYTHONPATH=. python -u eval_checkpoints.py <policy_name> <step> [<step> ...]
    PYTHONPATH=. python -u eval_checkpoints.py <policy_name> top[N]

`top10` (or `top`, `top:10`) picks the N most promising checkpoints automatically, ranked
by perfect rate smoothed over a centred 10-eval window rather than by a single eval,
so the selection isn't just the luckiest 10-episode reads. **This is the normal way to
close out an arm** — 10 checkpoints x 100 episodes, ~8 minutes.

Adjacent steps are allowed through on purpose. Checkpoints 1000 train steps apart
*should* score alike, so when they don't, that spread is the checkpoint-to-checkpoint
variance, which is worth measuring rather than designing around.

Environment:
    EVAL_EPISODES     episodes per checkpoint, rounded up to a whole number of
                      rounds (default 100)
    EVAL_WORKERS      parallel envs inside this process (default 10)
    EVAL_OUT_SUFFIX   appended to the output filename

**Worker 0 renders a visible window**, so each eval process shows one game as it
plays; the remaining workers are headless. Running four checkpoints in parallel
therefore gives four windows to watch.

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


def select_top_checkpoints(policy_name, available, count, window=10):
    """The `count` most promising checkpoints, by smoothed perfect rate.

    Ranking on a single eval's perfect_percent would just pick the luckiest 10-episode
    evals, which is the winner's curse in its purest form. Ranking on a centred window
    finds regions where the policy was genuinely good and then evaluates the individual
    checkpoints inside them.

    Adjacent steps are allowed through deliberately. Neighbouring checkpoints are 1000
    train steps apart and *should* behave alike, so when they don't, that spread is the
    checkpoint-to-checkpoint variance — which is exactly what a 100-episode eval is for
    measuring. Spacing the picks out would hide it.
    """
    path = os.path.join(RUNS_DIR, '{0}_evals.json'.format(policy_name))
    with open(path) as handle:
        evals = json.load(handle)['evals']

    rates = [e['perfect_percent'] for e in evals]
    ranked = []
    for index, entry in enumerate(evals):
        if entry['step'] not in available:
            continue
        lo = max(0, index - window // 2)
        smoothed = sum(rates[lo:lo + window]) / len(rates[lo:lo + window])
        ranked.append((smoothed, entry['perfect_percent'], entry['step']))

    if not ranked:
        raise SystemExit('no eval steps in {0} have checkpoints in savedPolicies'.format(path))

    ranked.sort(reverse=True)
    chosen = sorted(step for _, _, step in ranked[:count])
    print('selected {0} of {1} available checkpoints by smoothed perfect rate:'.format(
        len(chosen), len(ranked)))
    for smoothed, single, step in sorted(ranked[:count], key=lambda r: r[2]):
        print('    {0:>8}  smoothed {1:>5.1f}%  this eval {2:>5.1f}%'.format(step, smoothed, single))
    return chosen


def main(argv):
    if len(argv) < 3:
        print(__doc__)
        return 1
    policy_name = argv[1]
    num_episodes = int(os.environ.get('EVAL_EPISODES', 100))
    num_workers = int(os.environ.get('EVAL_WORKERS', 10))

    ckpt_dir = POLICY_DIR + policy_name
    available = {int(f[len('ckpt-'):].split('.')[0])
                 for f in os.listdir(ckpt_dir) if f.startswith('ckpt-') and f.endswith('.index')}

    # Spelled `top` rather than `--top`: tf_agents' handle_main routes argv through absl,
    # which rejects any unregistered `--flag` before main() is reached.
    if argv[2].startswith('top'):
        rest = argv[2][len('top'):].lstrip(':=')
        count = int(rest) if rest else (int(argv[3]) if len(argv) > 3 else 10)
        requested_steps = select_top_checkpoints(policy_name, available, count)
    else:
        requested_steps = [int(a) for a in argv[2:]]
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
        row = {
            'step': step,
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

    print('\n{0:>9}  {1:>8}  {2:>16}  {3:>9}'.format('step', 'perfect', '95% CI', 'avg score'))
    for row in results:
        print('{0:>9}  {1:>7}%  {2:>7}-{3:<7}  {4:>9}'.format(
            row['step'], row['perfect_percent'], row['perfect_ci95'][0], row['perfect_ci95'][1],
            row['avg_score']))
    return 0


if __name__ == '__main__':
    system_multiprocessing.handle_main(lambda argv: sys.exit(main(argv)))
