from __future__ import absolute_import, division, print_function

from snake_environment import SnakeEnvironment
from training import *

import reverb

from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import py_driver
from tf_agents.environments import tf_py_environment
from tf_agents.networks import sequential
from tf_agents.policies import py_tf_eager_policy
from tf_agents.replay_buffers import reverb_replay_buffer
from tf_agents.replay_buffers import reverb_utils
from tf_agents.specs import tensor_spec
from tf_agents.utils import common

# --------------------------------------------- Constants ---------------------------------------------
learning_rate = 1e-5  # next 1e-4

batch_size = 64
# batch_size = 256
# discount = 1.0
discount = 0.95
# agent_target_update_period = 1
agent_target_update_period = 4
initial_priority = 1.0

display_training = False
# display_training = True
display_eval = True
# eval_limit_fps = True
eval_limit_fps = False

num_iterations = 1000000000  # 1,000,000,000

initial_collect_steps = 100
collect_steps_per_iteration = 1
replay_buffer_max_length = 10000
# ------------------------------------------- End Constants -------------------------------------------

print(tf.config.list_physical_devices('GPU'))
print('learning_rate: {0}, discount: {1}'.format(learning_rate, discount))

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

random_play(train_env, train_py_env, rb_observer, initial_collect_steps)

dataset = replay_buffer.as_dataset(
    num_parallel_calls=3,
    sample_batch_size=batch_size,
    num_steps=2).prefetch(3)

iterator = iter(dataset)

# Create a driver to collect experience.
collect_driver = py_driver.PyDriver(
    train_py_env,
    py_tf_eager_policy.PyTFEagerPolicy(
        agent.collect_policy, use_tf_function=False),
    [rb_observer],
    max_steps=collect_steps_per_iteration)

train(num_iterations, eval_env, train_py_env, agent, collect_driver, iterator, replay_buffer)

# todo: fix video creation by using the display surface
# print(create_policy_eval_video(agent.policy, "trained-agent"))

print('done')
