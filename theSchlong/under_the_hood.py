from __future__ import absolute_import, division, print_function

import imageio
import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf


# Define a helper function to create Dense layers configured with the right
# activation and kernel initializer.
def dense_layer(num_units):
    return tf.keras.layers.Dense(
        num_units,
        activation=tf.keras.activations.relu,
        kernel_initializer=tf.keras.initializers.VarianceScaling(
            scale=2.0, mode='fan_in', distribution='truncated_normal'))


def compute_avg_return(environment, policy, metrics, num_episodes=10):
    total_return = 0.0
    for _ in range(num_episodes):
        time_step = environment.reset()
        episode_return = 0.0

        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = environment.step(action_step.action)
            episode_return += time_step.reward

        if metrics.min_score > episode_return:
            metrics.min_score = episode_return
        if metrics.max_score < episode_return:
            metrics.max_score = episode_return

        total_return += episode_return

    avg_return = total_return / num_episodes
    return avg_return.numpy()[0]


def compute_trailing_avg_return(trailing_avg_returns):
    total = 0.0
    for avg in trailing_avg_returns:
        total += avg
    return total / len(trailing_avg_returns)


def display_progress(steps, eval_interval, returns, screen):
    fig = plt.figure()
    steps = range(0, steps + 1, eval_interval)
    plt.clf()
    plt.plot(steps, returns)
    plt.ylabel('Average Return')
    plt.xlabel('Iterations')
    # plt.ylim(top=250)

    fig.canvas.draw()

    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

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
