from __future__ import absolute_import, division, print_function

import snake_constants
from snake_constants import *

import imageio
import matplotlib.pyplot as plt
import numpy as np
import time

import tensorflow as tf


# Define a helper function to create Dense layers configured with the right
# activation and kernel initializer.
def dense_layer(num_units):
    return tf.keras.layers.Dense(
        num_units,
        activation=tf.keras.activations.relu,
        kernel_initializer=tf.keras.initializers.VarianceScaling(
            scale=2.0, mode='fan_in', distribution='truncated_normal'))


def compute_avg_return(environment, parallel_environment, policy, metrics, eval_only, num_episodes=10):
    start_time = time.time()
    total_steps = 0

    # first episode: shown displayed on the single environment
    py_env = environment.pyenv
    if hasattr(py_env, 'envs'):
        py_env = py_env.envs[0]
    py_env.set_display(True)

    time_step = environment.reset()
    episode_reward = 0.0
    while not time_step.is_last():
        action_step = policy.action(time_step)
        time_step = environment.step(action_step.action)
        episode_reward += time_step.reward.numpy()[0]
        total_steps += 1

    episode_rewards = [episode_reward]
    episode_scores = [py_env.get_score()]
    last_rewards = [time_step.reward.numpy()[0]]

    # remaining episodes: run in parallel, headless
    num_parallel = num_episodes - 1
    if num_parallel > 0 and parallel_environment is not None:
        parallel_rewards, parallel_scores, parallel_last_rewards, parallel_steps = run_parallel_eval_episodes(
            parallel_environment, policy, num_parallel)
        episode_rewards.extend(parallel_rewards.tolist())
        episode_scores.extend(parallel_scores.tolist())
        last_rewards.extend(parallel_last_rewards.tolist())
        total_steps += parallel_steps

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


def display_progress(starting_step, steps, eval_interval, scores, perfect_percents, screen):
    fig, score_axis = plt.subplots(figsize=(3.65, 2.25))
    steps = range(starting_step, steps + 1, eval_interval)

    score_color = 'tab:blue'
    score_axis.plot(steps, scores, color=score_color)
    score_axis.set_ylabel('Average Score', color=score_color)
    score_axis.set_xlabel('Iterations')
    score_axis.tick_params(axis='y', labelcolor=score_color)
    # score_axis.set_ylim(top=250)

    percent_color = 'tab:red'
    percent_axis = score_axis.twinx()
    percent_axis.plot(steps, [percent * 100 for percent in perfect_percents], color=percent_color)
    percent_axis.set_ylabel('Perfect Game %', color=percent_color)
    percent_axis.tick_params(axis='y', labelcolor=percent_color)
    percent_axis.set_ylim(bottom=0, top=100)

    fig.canvas.draw()

    image = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]

    screen.update(image)
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
