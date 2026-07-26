from __future__ import absolute_import, division, print_function

from prioritized_replay_buffer import TrajectoryPrioritizedReplayBuffer
from snake_environment import SnakeEnvironment
from training import *

import os

from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import py_driver
from tf_agents.environments import parallel_py_environment
from tf_agents.environments import tf_py_environment
from tf_agents.networks import sequential
from tf_agents.policies import py_tf_eager_policy
from tf_agents.specs import tensor_spec
from tf_agents.system import system_multiprocessing
from tf_agents.utils import common


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
    initial_epsilon = 0.4

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

    # Prioritized replay. 0.8 was what the reverb setup used, but it paired that
    # with Huber loss as the priority, which works out to a far more aggressive
    # exponent than intended; 0.6 against the raw TD error is the usual choice.
    # beta corrects the sampling bias prioritizing introduces and anneals to 1.0,
    # which the reverb version never did -- it prioritized with no importance
    # sampling at all.
    priority_exponent = 0.6
    initial_importance_sampling_beta = 0.4
    beta_anneal_steps = 1000000

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

    # Epsilon has to be a Variable rather than a Python callable: the collect
    # policy runs inside a tf.function (use_tf_function=True below), which would
    # bake a plain float in as a constant at trace time and freeze the decay in
    # maybe_update_epsilon(). A Variable is read on every call instead.
    epsilon = tf.Variable(initial_epsilon, dtype=tf.float32, trainable=False, name='epsilon')

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
        epsilon_greedy=epsilon,
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        td_errors_loss_fn=common.element_wise_huber_loss,
        target_update_period=agent_target_update_period,
        # target_update_tau=target_update_tau,
        train_step_counter=global_step)

    agent.initialize()

    eval_policy = agent.policy
    collect_policy = agent.collect_policy

    # Replay buffer
    # reverb has no macOS wheel, so prioritized replay comes from cpprb's C++ sum
    # tree instead of reverb's ReverbReplayBuffer. See prioritized_replay_buffer.py.
    replay_buffer_data_spec = tensor_spec.to_nest_array_spec(agent.collect_data_spec)
    replay_buffer = TrajectoryPrioritizedReplayBuffer(
        data_spec=replay_buffer_data_spec,
        capacity=replay_buffer_max_length,
        alpha=priority_exponent,
        initial_beta=initial_importance_sampling_beta,
        beta_anneal_steps=beta_anneal_steps
    )

    def rb_observer(traj):
        replay_buffer.add(traj)

    initial_populate_replay_buffer(initialize_with_schmid,
                                   train_env.time_step_spec(),
                                   train_env.action_spec(),
                                   train_py_env,
                                   schmid_py_env,
                                   rb_observer,
                                   initial_populate_replay_buffer_steps)

    # Create a driver to collect experience.
    collect_driver = py_driver.PyDriver(
        train_py_env,
        py_tf_eager_policy.PyTFEagerPolicy(
            agent.collect_policy, use_tf_function=True),
        [rb_observer],
        max_steps=collect_steps_per_iteration)

    # The replay buffer holds 100k transitions, far more than the ~180 KB of agent
    # weights, so keeping it in the 1000-deep history would mean gigabytes per
    # policy. It lives outside the checkpointer in its own single file, which only
    # ever needs the newest copy to warm-start the next run. The agent history
    # stays 1000 deep for replaying earlier iterations.
    train_checkpointer = common.Checkpointer(
        ckpt_dir=POLICY_DIR + policy_name,
        max_to_keep=1000,
        agent=agent,
        policy=agent.policy,
        global_step=global_step
    )

    replay_buffer_dir = os.path.join(POLICY_DIR + policy_name, 'replay_buffer')

    train_checkpointer.initialize_or_restore()
    global_step = tf.compat.v1.train.get_global_step()
    if replay_buffer.restore(replay_buffer_dir):
        print('restored replay buffer:', replay_buffer.size, 'transitions')

    train(num_iterations, eval_env, eval_parallel_env, train_py_env, agent, collect_driver, batch_size, replay_buffer,
          train_checkpointer, replay_buffer_dir, global_step, epsilon, eval_only, policy_name)

    # todo: fix video creation by using the display surface
    # print(create_policy_eval_video(agent.policy, "trained-agent"))

    print('done')


if __name__ == '__main__':
    system_multiprocessing.handle_main(main)
