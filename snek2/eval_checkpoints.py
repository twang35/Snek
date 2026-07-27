"""Measures a saved checkpoint's true perfect-game rate over many episodes.

The graph in runs/<policy>.png plots a 10-episode eval per point, so a single
point moves in 10-percentage-point jumps and a lucky eval reads far above the
policy's real rate. This script reloads specific checkpoints and evaluates each
over a few hundred episodes instead, which is the only way to compare against a
remembered figure like theSchlong's 76%.

Usage:

    cd snek2
    PYTHONPATH=. python -u eval_checkpoints.py <policy_name> <step> [<step> ...]

Environment:
    EVAL_EPISODES     episodes per checkpoint (default 100)
    EVAL_WORKERS      parallel headless envs inside this process (default 10)
    EVAL_OUT_SUFFIX   appended to the output filename

Everything runs headless. Results go to
runs/<policy_name>_checkpoint_evals<suffix>.json so they survive and can be
compared across sessions.

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


def evaluate(parallel_env, policy, num_episodes):
    """Runs num_episodes to completion across the parallel workers.

    Workers auto-reset when an episode ends, so a worker that finishes early
    starts a fresh episode. Rather than fight that, this collects a fixed number
    of *completed* episodes and stops reading a worker's later ones.
    """
    worker_envs = parallel_env.pyenv.envs
    num_workers = len(worker_envs)

    scores, perfect_flags, rewards = [], [], []
    time_step = parallel_env.reset()
    running_reward = np.zeros(num_workers, dtype=np.float64)
    steps = 0
    start = time.time()

    while len(scores) < num_episodes:
        action_step = policy.action(time_step)
        time_step = parallel_env.step(action_step.action)
        step_rewards = time_step.reward.numpy()
        running_reward += step_rewards
        steps += num_workers

        finished = np.flatnonzero(time_step.is_last().numpy())
        if finished.size:
            # get_score has to be read before the worker's auto-reset clears it
            promises = [worker_envs[i].call('get_score') for i in finished]
            for i, promise in zip(finished, promises):
                if len(scores) >= num_episodes:
                    break
                scores.append(float(promise()))
                perfect_flags.append(bool(step_rewards[i] == snake_constants.PERFECT_GAME_REWARD))
                rewards.append(float(running_reward[i]))
            running_reward[finished] = 0.0

    elapsed = time.time() - start
    print('    {0} episodes in {1}s ({2} env steps/s)'.format(
        len(scores), round(elapsed, 1), round(steps / elapsed)))
    return scores, perfect_flags, rewards


def main(argv):
    if len(argv) < 3:
        print(__doc__)
        return 1
    policy_name = argv[1]
    requested_steps = [int(a) for a in argv[2:]]
    num_episodes = int(os.environ.get('EVAL_EPISODES', 100))
    num_workers = int(os.environ.get('EVAL_WORKERS', 10))

    ckpt_dir = POLICY_DIR + policy_name
    available = {int(f[len('ckpt-'):].split('.')[0])
                 for f in os.listdir(ckpt_dir) if f.startswith('ckpt-') and f.endswith('.index')}
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

    def make_worker():
        os.environ['SDL_VIDEODRIVER'] = 'dummy'
        return SnakeEnvironment(discount=0.99, display=False, policy_name=policy_name)

    parallel_env = tf_py_environment.TFPyEnvironment(
        parallel_py_environment.ParallelPyEnvironment([make_worker] * num_workers))

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
