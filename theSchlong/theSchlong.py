from __future__ import absolute_import, division, print_function

from snake_environment import SnakeEnvironment
from under_the_hood import *

from time import time

import reverb

import tensorflow as tf

from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import py_driver
from tf_agents.environments import tf_py_environment
from tf_agents.networks import sequential
from tf_agents.policies import py_tf_eager_policy
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import reverb_replay_buffer
from tf_agents.replay_buffers import reverb_utils
from tf_agents.specs import tensor_spec
from tf_agents.utils import common

num_iterations = 1000000000  # 1,000,000,000

initial_collect_steps = 100
collect_steps_per_iteration = 1
replay_buffer_max_length = 10000

log_interval = 200
num_eval_episodes = 10
eval_interval = 1000
display_progress_interval = eval_interval

# lr: avg return: steps
# 1e-3: lr too high, learns then forgets, stuck at -9: 30k
# 1e-4: -7: 10k, jagged learning, but too high lr, -5: 20k, -4: 30k, -5: 40k, -6: 50k
# 5e-5: -1:  10k, slowly stops dying
#       -7: 10k, lr might be too high?
# 1e-5: -9: 10k, more consistent learning but slow, -7: 14k, -5: 20k, -3: 24k, -1: 55k
#     : -8: 10k, consistent learning but slow, -4: 20k, -2: 30k, -1: 40k, -3: 50k
# 1e-6: -10: 10k, lr too low, -10 at 20k and 30k
learning_rate = 1e-5  # next 5e-6

batch_size = 64
# batch_size = 256
# discount = 1.0
discount = 0.99
# agent_target_update_period = 1
agent_target_update_period = 4
initial_priority = 0.5
# display_training = False
display_training = True
display_eval = True
# eval_limit_fps = True
eval_limit_fps = False

train_py_env = SnakeEnvironment(discount=discount, display=display_training)
eval_py_env = SnakeEnvironment(discount=discount, display=display_eval, limit_fps=eval_limit_fps)

train_py_env.reset()
eval_py_env.reset()

train_env = tf_py_environment.TFPyEnvironment(train_py_env)
eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

# fc_layer_params = (100, 50)
fc_layer_params = (100, 50)
action_tensor_spec = tensor_spec.from_spec(train_py_env.action_spec())
num_actions = action_tensor_spec.maximum - action_tensor_spec.minimum + 1


# Define a helper function to create Dense layers configured with the right
# activation and kernel initializer.
def dense_layer(num_units):
    return tf.keras.layers.Dense(
        num_units,
        activation=tf.keras.activations.relu,
        kernel_initializer=tf.keras.initializers.VarianceScaling(
            scale=2.0, mode='fan_in', distribution='truncated_normal'))


# QNetwork consists of a sequence of Dense layers followed by a dense layer
# with `num_actions` units to generate one q_value per available action as
# its output.
dense_layers = [dense_layer(num_units) for num_units in fc_layer_params]
q_values_layer = tf.keras.layers.Dense(
    num_actions,
    activation=None,
    kernel_initializer=tf.keras.initializers.RandomUniform(
        minval=-0.03, maxval=0.03),
    # bias_initializer=tf.keras.initializers.Constant(-0.2))
    bias_initializer=tf.keras.initializers.Constant(0.0))
q_net = sequential.Sequential(dense_layers + [q_values_layer])

agent = dqn_agent.DqnAgent(
    train_env.time_step_spec(),
    train_env.action_spec(),
    q_network=q_net,
    epsilon_greedy=train_py_env.get_updated_epsilon,
    optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
    # td_errors_loss_fn=common.element_wise_squared_loss,
    td_errors_loss_fn=common.element_wise_huber_loss,
    target_update_period=agent_target_update_period,
    train_step_counter=tf.Variable(0))

agent.initialize()

eval_policy = agent.policy
collect_policy = agent.collect_policy

random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(),
                                                train_env.action_spec())

# Replay buffer

table_name = 'uniform_table'
replay_buffer_signature = tensor_spec.from_spec(agent.collect_data_spec)
replay_buffer_signature = tensor_spec.add_outer_dim(replay_buffer_signature)

table = reverb.Table(
    table_name,
    max_size=replay_buffer_max_length,
    sampler=reverb.selectors.Prioritized(priority_exponent=0.8),
    remover=reverb.selectors.Fifo(),
    rate_limiter=reverb.rate_limiters.MinSize(1),
    signature=replay_buffer_signature
)

reverb_server = reverb.Server([table])

replay_buffer = reverb_replay_buffer.ReverbReplayBuffer(
    agent.collect_data_spec,
    table_name=table_name,
    sequence_length=2,
    local_server=reverb_server
)

rb_observer = reverb_utils.ReverbAddTrajectoryObserver(
    replay_buffer.py_client,
    table_name,
    priority=initial_priority,
    sequence_length=2
)

print('Random play to populate replay buffer')
py_driver.PyDriver(
    train_py_env,
    py_tf_eager_policy.PyTFEagerPolicy(
        random_policy, use_tf_function=True),
    [rb_observer],
    max_steps=initial_collect_steps).run(train_py_env.reset())

dataset = replay_buffer.as_dataset(
    num_parallel_calls=3,
    sample_batch_size=batch_size,
    num_steps=2).prefetch(3)

iterator = iter(dataset)

# Training the agent

# (Optional) Optimize by wrapping some of the code in a graph using TF function.
agent.train = common.function(agent.train)

# Reset the train step.
agent.train_step_counter.assign(0)

# Evaluate the agent's policy once before training
avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
returns = [avg_return]
print('before training return: ', returns)

# Reset the environment.
time_step = train_py_env.reset()

# Create a driver to collect experience.
collect_driver = py_driver.PyDriver(
    train_py_env,
    py_tf_eager_policy.PyTFEagerPolicy(
        agent.collect_policy, use_tf_function=False),
    [rb_observer],
    max_steps=collect_steps_per_iteration)

screen = pf.screen(np.zeros((480, 640)), 'Training results')

print('Begin training:')
start_time = time()
for i in range(num_iterations):
    # Collect a few steps and save to replay buffer.
    # To view q_values, breakpoint at line 160 in tf_agents/policies/q_policy.py
    time_step, _ = collect_driver.run(time_step)

    # Sample a batch of data from the buffer and update the agent's network.
    experience, sample_info = next(iterator)
    loss_info = agent.train(experience)

    replay_buffer.update_priorities([element[0] for element in sample_info.key],
                                    tf.cast(loss_info.extra.td_loss, tf.float64))

    step = agent.train_step_counter.numpy()

    if step % log_interval == 0:
        steps_per_second = log_interval / (time() - start_time)
        print('step = {0}: loss = {1}, steps/second = {2}'.format(step, loss_info.loss, steps_per_second))
        start_time = time()

    if step % eval_interval == 0:
        avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
        print('train_py_env high score: ', train_py_env.high_score)
        print('step = {0}: avg_return = {1}'.format(step, avg_return))
        returns.append(avg_return)
        # restart time because compute_avg_return() takes a while and messes up the timing
        start_time = time()

    if step % display_progress_interval == 0:
        display_progress(i + 1, eval_interval, returns, screen)

# todo: fix video creation by using the display surface
# print(create_policy_eval_video(agent.policy, "trained-agent"))

print('done')
