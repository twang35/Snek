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
    episode_return = 0.0
    while not time_step.is_last():
        action_step = policy.action(time_step)
        time_step = environment.step(action_step.action)
        episode_return += time_step.reward.numpy()[0]
        total_steps += 1

    episode_returns = [episode_return]
    last_rewards = [time_step.reward.numpy()[0]]

    # remaining episodes: run in parallel, headless
    num_parallel = num_episodes - 1
    if num_parallel > 0 and parallel_environment is not None:
        parallel_returns, parallel_last_rewards, parallel_steps = run_parallel_eval_episodes(
            parallel_environment, policy, num_parallel)
        episode_returns.extend(parallel_returns.tolist())
        last_rewards.extend(parallel_last_rewards.tolist())
        total_steps += parallel_steps

    for episode_return in episode_returns:
        if metrics.min_score > episode_return:
            metrics.min_score = episode_return
        if metrics.max_score < episode_return:
            metrics.max_score = episode_return

    perfect_games = sum(1 for reward in last_rewards if reward == snake_constants.PERFECT_GAME_REWARD)

    print('eval steps/second: ', round(total_steps / (time.time() - start_time), 2))

    metrics.last_eval_perfect_percent = perfect_games / num_episodes
    if eval_only:
        metrics.append_perfect_percent(metrics.last_eval_perfect_percent)

    return sum(episode_returns) / num_episodes


def run_parallel_eval_episodes(parallel_environment, policy, num_parallel):
    time_step = parallel_environment.reset()
    episode_returns = np.zeros(num_parallel, dtype=np.float32)
    last_rewards = np.zeros(num_parallel, dtype=np.float32)
    done = np.zeros(num_parallel, dtype=bool)
    total_steps = 0

    while not np.all(done):
        action_step = policy.action(time_step)
        time_step = parallel_environment.step(action_step.action)
        rewards = time_step.reward.numpy()
        is_last = time_step.is_last().numpy()

        active = ~done
        episode_returns[active] += rewards[active]
        total_steps += int(np.sum(active))

        # an already-finished worker auto-resets into a new episode on its
        # next step; `active` keeps that from being double-counted
        newly_done = active & is_last
        last_rewards[newly_done] = rewards[newly_done]
        done = done | newly_done

    return episode_returns, last_rewards, total_steps


def compute_trailing_avg_return(trailing_avg_returns):
    total = 0.0
    for avg in trailing_avg_returns:
        total += avg
    return total / len(trailing_avg_returns)


def display_progress(starting_step, steps, eval_interval, returns, screen):
    fig = plt.figure(figsize=(7.3, 4.5))
    steps = range(starting_step, steps + 1, eval_interval)
    plt.clf()
    plt.plot(steps, returns)
    plt.ylabel('Average Return')
    plt.xlabel('Iterations')
    # plt.ylim(top=250)

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
