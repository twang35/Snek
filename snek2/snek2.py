from __future__ import absolute_import, division, print_function

from snake_environment import SnakeEnvironment
from training import *

import os

from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import py_driver
from tf_agents.environments import parallel_py_environment
from tf_agents.environments import tf_py_environment
from tf_agents.networks import sequential
from tf_agents.policies import py_tf_eager_policy
from tf_agents.replay_buffers import py_uniform_replay_buffer
from tf_agents.specs import tensor_spec
from tf_agents.system import system_multiprocessing
from tf_agents.utils import common
from tf_agents.utils import nest_utils


def main(argv):
    # --------------------------------------------- Constants ---------------------------------------------
    learning_rate = 1e-5  # next 1e-4

    # batch_size = 64
    batch_size = 128
    # discount = 1.0
    discount = 0.99
    # agent_target_update_period = 4
    agent_target_update_period = 8
    # target_update_tau = 0.01

    display_training = False
    # display_training = True
    initialize_with_schmid = False

    display_eval = True
    # eval_limit_fps = True
    eval_limit_fps = False

    eval_only = False
    # eval_only = True

    num_iterations = 1000000000  # 1,000,000,000

    initial_populate_replay_buffer_steps = 1000
    collect_steps_per_iteration = 1
    replay_buffer_max_length = 100000

    # policy_name = 'eval'
    policy_name = 'train'

    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # turns off GPU

    # ------------------------------------------- End Constants -------------------------------------------

    if len(argv) > 1:
        policy_name = argv[1]

    if policy_name == 'eval':
        eval_only = True
        initial_populate_replay_buffer_steps = 10
        snake_constants.PERFECT_GAME_REWARD = 10000
        snake_constants.PERFECT_GAME_WAIT_MS = 500

    print('policy_name: {0}, learning_rate: {1}, discount: {2}, initialize_with_schmid: {3}, steps_left: False, '
          'FOOD_DISTANCE_REWARD: {4}, initial_populate_replay_buffer_steps: {5}, total_groups_obs: True, '
          'DEATH_REWARD: {6}, agent_target_update_period: {7}'
          .format(policy_name, learning_rate, discount, initialize_with_schmid, FOOD_DISTANCE_REWARD,
                  initial_populate_replay_buffer_steps, DEATH_REWARD, agent_target_update_period))
    print(tf.config.list_physical_devices('GPU'))
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=500)])

    train_py_env = SnakeEnvironment(discount=discount, display=display_training, policy_name=policy_name)
    schmid_py_env = None
    schmid_env = None
    if initialize_with_schmid:
        schmid_py_env = SnakeEnvironment(discount=discount, display=True, policy_name=policy_name)
        schmid_py_env.reset()
        schmid_env = tf_py_environment.TFPyEnvironment(schmid_py_env)
    eval_py_env = SnakeEnvironment(discount=discount, display=display_eval, limit_fps=eval_limit_fps,
                                   policy_name=policy_name)

    train_py_env.reset()
    eval_py_env.reset()

    train_env = tf_py_environment.TFPyEnvironment(train_py_env)
    eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

    # Beyond the first (displayed) eval episode, the rest run headless in
    # parallel worker processes to speed up eval.
    num_parallel_eval_envs = num_eval_episodes - 1
    eval_parallel_env = None
    if num_parallel_eval_envs > 0:
        def make_headless_eval_env():
            os.environ['SDL_VIDEODRIVER'] = 'dummy'
            return SnakeEnvironment(discount=discount, display=False, policy_name=policy_name)

        eval_parallel_env = tf_py_environment.TFPyEnvironment(
            parallel_py_environment.ParallelPyEnvironment(
                [make_headless_eval_env] * num_parallel_eval_envs))

    # fc_layer_params = (100, 50)
    fc_layer_params = (50, 100, 50)
    action_tensor_spec = tensor_spec.from_spec(train_py_env.action_spec())
    num_actions = action_tensor_spec.maximum - action_tensor_spec.minimum + 1

    global_step = tf.compat.v1.train.get_or_create_global_step()
    # step = tf.Variable(0, trainable=False, dtype=tf.int32)

    # QNetwork consists of a sequence of Dense layers followed by a dense layer
    # with `num_actions` units to generate one q_value per available action as
    # its output.
    dense_layers = [dense_layer(num_units) for num_units in fc_layer_params]
    q_values_layer = tf.keras.layers.Dense(
        num_actions,
        activation=None,
        kernel_initializer=tf.keras.initializers.RandomUniform(
            minval=-0.03, maxval=0.03),
        bias_initializer=tf.keras.initializers.Constant(0.0))
    q_net = sequential.Sequential(dense_layers + [q_values_layer])

    agent = dqn_agent.DdqnAgent(
        train_env.time_step_spec(),
        train_env.action_spec(),
        q_network=q_net,
        epsilon_greedy=train_py_env.get_updated_epsilon,
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        td_errors_loss_fn=common.element_wise_huber_loss,
        target_update_period=agent_target_update_period,
        # target_update_tau=target_update_tau,
        train_step_counter=global_step)

    agent.initialize()

    eval_policy = agent.policy
    collect_policy = agent.collect_policy

    # Replay buffer
    # Note: reverb has no macOS wheel, so this uses tf_agents' local
    # PyUniformReplayBuffer instead of reverb's prioritized ReverbReplayBuffer.
    # This means experience is sampled uniformly rather than by TD-error priority.

    replay_buffer_data_spec = tensor_spec.to_nest_array_spec(agent.collect_data_spec)
    replay_buffer = py_uniform_replay_buffer.PyUniformReplayBuffer(
        data_spec=replay_buffer_data_spec,
        capacity=replay_buffer_max_length
    )

    def rb_observer(traj):
        replay_buffer.add_batch(nest_utils.batch_nested_array(traj))

    initial_populate_replay_buffer(initialize_with_schmid,
                                   train_env.time_step_spec(),
                                   train_env.action_spec(),
                                   train_py_env,
                                   schmid_py_env,
                                   rb_observer,
                                   initial_populate_replay_buffer_steps)

    dataset = replay_buffer.as_dataset(
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

    # The replay buffer's 100k transitions serialize to ~10 MB, which dwarfs the
    # ~180 KB of agent weights, so keeping it in the 1000-deep history meant
    # ~10 GB per policy. Only the newest buffer is useful (it just warm-starts
    # the next run), so it gets its own single-slot checkpointer. The agent
    # history stays 1000 deep for replaying earlier iterations.
    train_checkpointer = common.Checkpointer(
        ckpt_dir=POLICY_DIR + policy_name,
        max_to_keep=1000,
        agent=agent,
        policy=agent.policy,
        global_step=global_step
    )

    replay_buffer_checkpointer = common.Checkpointer(
        ckpt_dir=os.path.join(POLICY_DIR + policy_name, 'replay_buffer'),
        max_to_keep=1,
        replay_buffer=replay_buffer
    )

    train_checkpointer.initialize_or_restore()
    replay_buffer_checkpointer.initialize_or_restore()
    global_step = tf.compat.v1.train.get_global_step()

    train(num_iterations, eval_env, eval_parallel_env, train_py_env, agent, collect_driver, iterator, replay_buffer,
          train_checkpointer, replay_buffer_checkpointer, global_step, eval_only, policy_name)

    # todo: fix video creation by using the display surface
    # print(create_policy_eval_video(agent.policy, "trained-agent"))

    print('done')


if __name__ == '__main__':
    system_multiprocessing.handle_main(main)
