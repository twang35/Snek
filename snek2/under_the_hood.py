from __future__ import absolute_import, division, print_function

import snake_constants
from snake_constants import *

import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import time

import tensorflow as tf
from tf_agents.networks import sequential


# Define a helper function to create Dense layers configured with the right
# activation and kernel initializer.
def dense_layer(num_units):
    return tf.keras.layers.Dense(
        num_units,
        activation=tf.keras.activations.relu,
        kernel_initializer=tf.keras.initializers.VarianceScaling(
            scale=2.0, mode='fan_in', distribution='truncated_normal'))


def eval_fc_layer_params():
    """Layer widths for rebuilding a trained network, honouring SNEK_FC_LAYERS.

    Training reads its widths from the same variable, so anything that restores a checkpoint
    has to read it too. eval_checkpoints.py used to hardcode (50, 100, 50), which was right
    for every run so far only because nobody has set the override — and a mismatch would not
    have been loud, because restore() is called with expect_partial() and would simply not
    populate the layers it could not match.
    """
    raw = os.environ.get('SNEK_FC_LAYERS')
    if raw is None:
        return 50, 100, 50
    return tuple(int(width) for width in raw.split(','))


def build_q_net(num_actions, fc_layer_params=None):
    """The Q-network training builds, so restored weights line up.

    Shared by eval_checkpoints.py and watch.py. Keeping a second copy of the architecture is
    how the two quietly drift apart.
    """
    if fc_layer_params is None:
        fc_layer_params = eval_fc_layer_params()
    dense_layers = [dense_layer(num_units) for num_units in fc_layer_params]
    q_values_layer = tf.keras.layers.Dense(
        num_actions,
        activation=None,
        kernel_initializer=tf.keras.initializers.RandomUniform(minval=-0.03, maxval=0.03),
        bias_initializer=tf.keras.initializers.Constant(0.0))
    return sequential.Sequential(dense_layers + [q_values_layer])


def compute_avg_return(parallel_environment, policy, metrics, eval_only, num_episodes=10):
    """Runs num_episodes greedy episodes, one per worker, and folds them into metrics.

    Every episode runs in parallel. This used to play the first episode by itself on a
    separate single environment in the main process, because that was the only episode that
    could be drawn in a window — pygame allows one display per process and the workers are
    separate processes. Watching moved to watch.py, so the reason went away and with it the
    whole serial path, the extra environment and the display switch.

    Flattening it is worth about 24% of an eval: at champion skill the split shape measured
    5.95s (1.67s serial + 4.28s round) against 4.55s for a single round of 10. Adding the
    tenth worker costs only ~0.27s, because a round ends with its slowest episode and the
    slowest of ten is barely worse than the slowest of nine.

    Statistically unchanged: still num_episodes independent greedy episodes, and
    run_parallel_eval_episodes already counts only each worker's first episode.
    """
    start_time = time.time()

    rewards, scores, last_rewards_array, total_steps = run_parallel_eval_episodes(
        parallel_environment, policy, num_episodes)
    episode_rewards = rewards.tolist()
    episode_scores = scores.tolist()
    last_rewards = last_rewards_array.tolist()

    for episode_reward in episode_rewards:
        if metrics.min_reward > episode_reward:
            metrics.min_reward = episode_reward
        if metrics.max_reward < episode_reward:
            metrics.max_reward = episode_reward

    for episode_score in episode_scores:
        if metrics.min_score > episode_score:
            metrics.min_score = episode_score
        if metrics.max_score < episode_score:
            metrics.max_score = episode_score

    perfect_games = sum(1 for reward in last_rewards if reward == snake_constants.PERFECT_GAME_REWARD)

    if snake_constants.DEBUG_LOGGING:
        print('eval steps/second: ', round(total_steps / (time.time() - start_time), 2))

    metrics.last_eval_perfect_percent = perfect_games / num_episodes
    if eval_only:
        metrics.append_perfect_percent(metrics.last_eval_perfect_percent)

    avg_reward = sum(episode_rewards) / num_episodes
    avg_score = sum(episode_scores) / num_episodes
    return avg_reward, avg_score


def run_parallel_eval_episodes(parallel_environment, policy, num_parallel):
    worker_envs = parallel_environment.pyenv.envs

    time_step = parallel_environment.reset()
    episode_rewards = np.zeros(num_parallel, dtype=np.float32)
    episode_scores = np.zeros(num_parallel, dtype=np.float32)
    last_rewards = np.zeros(num_parallel, dtype=np.float32)
    done = np.zeros(num_parallel, dtype=bool)
    total_steps = 0

    while not np.all(done):
        action_step = policy.action(time_step)
        time_step = parallel_environment.step(action_step.action)
        rewards = time_step.reward.numpy()
        is_last = time_step.is_last().numpy()

        active = ~done
        episode_rewards[active] += rewards[active]
        total_steps += int(np.sum(active))

        # an already-finished worker auto-resets into a new episode on its
        # next step; `active` keeps that from being double-counted. Scores
        # must be fetched now, before that reset overwrites current_score.
        newly_done = active & is_last
        if np.any(newly_done):
            finished_indices = np.flatnonzero(newly_done)
            score_promises = [worker_envs[i].call('get_score') for i in finished_indices]
            for i, promise in zip(finished_indices, score_promises):
                episode_scores[i] = promise()
            last_rewards[newly_done] = rewards[newly_done]
        done = done | newly_done

    return episode_rewards, episode_scores, last_rewards, total_steps


def display_progress(eval_rows, resume_steps, screen, graph_path=None):
    """Draws the whole history of a policy, across however many runs made it.

    Takes explicit (step, score, percent) rows rather than assuming evenly spaced
    evals from a starting step, because a resumed policy's history has gaps
    wherever it was stopped and restarted.
    """
    fig, score_axis = plt.subplots(figsize=(3.65, 2.25))
    steps = [row['step'] for row in eval_rows]
    scores = [row['avg_score'] for row in eval_rows]
    perfect_percents = [row['perfect_percent'] for row in eval_rows]

    label_size = 6
    tick_size = 5

    score_color = 'tab:blue'
    score_axis.plot(steps, scores, color=score_color)
    score_axis.set_ylabel('Average Score', color=score_color, fontsize=label_size)
    score_axis.set_xlabel('Iterations', fontsize=label_size)
    score_axis.tick_params(axis='y', labelcolor=score_color, labelsize=tick_size)
    score_axis.tick_params(axis='x', labelsize=tick_size)
    # score_axis.set_ylim(top=250)

    percent_color = 'tab:red'
    percent_axis = score_axis.twinx()
    percent_axis.plot(steps, perfect_percents, color=percent_color)
    percent_axis.set_ylabel('Perfect Game %', color=percent_color, fontsize=label_size)
    percent_axis.tick_params(axis='y', labelcolor=percent_color, labelsize=tick_size)
    percent_axis.set_ylim(bottom=0, top=100)

    # Horizontal guides at 20/40/60/80 on the perfect-game axis. Without them the red
    # trace can't be read off the right-hand axis by eye — the left axis ticks are on a
    # different scale and give no help. Faint, dashed and in the percent axis's colour so
    # it is unambiguous which axis they belong to, and thin enough not to compete with
    # either trace. zorder keeps them behind the red line they annotate.
    # zorder 1 sits above the axes patch (0) and below both traces (2), so the guides
    # never hide data. Alpha below ~0.4 is invisible at this figure size (3.65x2.25in).
    percent_axis.set_yticks([0, 20, 40, 60, 80, 100])
    for level in (20, 40, 60, 80):
        percent_axis.axhline(level, color=percent_color, linestyle=(0, (4, 3)),
                             linewidth=0.5, alpha=0.55, zorder=1)

    # One dashed line per point where training was picked back up, so a dip or
    # jump can be tied to a restart rather than to the policy itself.
    for resume_step in resume_steps:
        score_axis.axvline(resume_step, color='gray', linestyle='--', linewidth=0.6)

    fig.tight_layout()
    fig.canvas.draw()

    image = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]

    screen.update(image)
    if graph_path is not None:
        # Same array that goes to the window, so the file always matches what's
        # on screen. Written beside the target and renamed so anything reading it
        # mid-eval never sees a half-written PNG.
        os.makedirs(os.path.dirname(graph_path), exist_ok=True)
        partial = graph_path + '.partial.png'
        imageio.imwrite(partial, image)
        os.replace(partial, graph_path)
    plt.close(fig)


def create_policy_eval_video(eval_py_env, eval_env, policy, filename, num_episodes=5, fps=30):
    filename = filename + ".mp4"
    with imageio.get_writer(filename, fps=fps) as video:
        for _ in range(num_episodes):
            time_step = eval_env.reset()
            video.append_data(eval_py_env.render())
            while not time_step.is_last():
                action_step = policy.action(time_step)
                time_step = eval_env.step(action_step.action)
                video.append_data(eval_py_env.render())
