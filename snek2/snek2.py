from __future__ import absolute_import, division, print_function

import os as _os

_QUIET = _os.environ.get('SNEK_DEBUG', '0') in ('0', '', 'false', 'False')
if _QUIET:
    _os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

# The gym "unmaintained, upgrade to Gymnasium" block is deliberately left alone. It is a raw
# print to stderr from gym's import, not a warnings.warn - it survives
# warnings.filterwarnings('ignore') - so removing it would mean redirecting stderr around the
# tf_agents import chain, which would also hide genuine import failures. It costs ~33 lines
# once per launch, against absl's checkpoint line costing ~2000 per run, so it is not worth
# that trade.

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



def tuned(name, default, cast=float):
    """Reads a hyperparameter from SNEK_<NAME>, falling back to the default.

    Hyperparameter sweeps run several policies side by side from this one file, so
    the values that vary between them come from the environment rather than edits.
    Everything read here lands in run_config, so runs/<policy_name>.md always
    records what the run actually used. See snek2/hyperparamTuning.md.
    """
    raw = os.environ.get('SNEK_' + name)
    if raw is None:
        return default
    value = cast(raw)
    print('hyperparameter override: {0} = {1} (default {2})'.format(name, value, default))
    return value


def main(argv):
    if _QUIET:
        # Set here, not at import: handle_main() routes through absl's app.run(), which
        # initialises absl logging to INFO after module import and would overwrite an earlier
        # call. At INFO, common.Checkpointer logs one "Saved checkpoint" line *per eval* —
        # ~2000 lines on a 2M-step run, the single largest source of log volume. WARNING and
        # above still get through, so real problems surface.
        from absl import logging as absl_logging
        absl_logging.set_verbosity(absl_logging.WARNING)
        tf.get_logger().setLevel('ERROR')

    # --------------------------------------------- Constants ---------------------------------------------
    learning_rate = tuned('LEARNING_RATE', 1e-5)

    # batch_size = 64
    batch_size = tuned('BATCH_SIZE', 128, int)
    # discount = 1.0
    discount = tuned('DISCOUNT', 0.99)
    # agent_target_update_period = 4
    agent_target_update_period = tuned('TARGET_UPDATE_PERIOD', 8, int)
    target_update_tau = tuned('TARGET_UPDATE_TAU', 1.0)
    gradient_clipping = tuned('GRADIENT_CLIPPING', 0.0)
    initial_epsilon = tuned('INITIAL_EPSILON', 0.4)
    # Floor for the decay ladder in maybe_update_epsilon(). The ladder's last rung
    # is 0.0, i.e. a fully greedy collect policy, which makes the replay buffer a
    # closed loop on the policy's own behaviour. Raise this to keep a trickle of
    # exploration forever. See hyperparamTuning/runs.md.
    min_epsilon = tuned('MIN_EPSILON', 0.0)

    display_training = False
    # display_training = True
    initialize_with_schmid = False

    display_eval = True
    # eval_limit_fps = True
    eval_limit_fps = False

    eval_only = False
    # eval_only = True

    num_iterations = 1000000000  # 1,000,000,000

    initial_populate_replay_buffer_steps = tuned('INITIAL_POPULATE_STEPS', 1000, int)
    collect_steps_per_iteration = 1
    replay_buffer_max_length = tuned('REPLAY_BUFFER_MAX_LENGTH', 100000, int)
    n_step_update = tuned('N_STEP_UPDATE', 1, int)

    # Prioritized replay. 0.8 was what the reverb setup used, but it paired that
    # with Huber loss as the priority, which works out to a far more aggressive
    # exponent than intended; 0.6 against the raw TD error is the usual choice.
    # beta corrects the sampling bias prioritizing introduces and anneals to 1.0,
    # which the reverb version never did -- it prioritized with no importance
    # sampling at all.
    priority_exponent = tuned('PRIORITY_EXPONENT', 0.6)
    initial_importance_sampling_beta = tuned('IS_BETA', 0.4)
    beta_anneal_steps = tuned('BETA_ANNEAL_STEPS', 1000000, int)
    # theSchlong -- the version that reached a far higher perfect-game rate than
    # anything measured here -- differed from this file on three PER details at
    # once: alpha 0.8, Huber td_loss as the priority signal, and no importance
    # sampling whatsoever. Each was "corrected" when the buffer was ported, and
    # the corrections were never validated past 30k steps. These two knobs make
    # the old behaviour reachable so the three can be tested rather than assumed.
    priority_signal = tuned('PRIORITY_SIGNAL', 'td_error', str)
    if priority_signal not in ('td_error', 'td_loss'):
        raise ValueError('SNEK_PRIORITY_SIGNAL must be td_error or td_loss, got ' + priority_signal)
    # 0 reproduces theSchlong, which applied no IS correction at all.
    use_is_weights = bool(tuned('IS_WEIGHTS', 1, int))

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

    if snake_constants.DEBUG_LOGGING:
        print('policy_name: {0}, learning_rate: {1}, discount: {2}, initialize_with_schmid: {3}, steps_left: False, '
              'FOOD_DISTANCE_REWARD: {4}, initial_populate_replay_buffer_steps: {5}, total_groups_obs: True, '
              'DEATH_REWARD: {6}, agent_target_update_period: {7}'
              .format(policy_name, learning_rate, discount, initialize_with_schmid, FOOD_DISTANCE_REWARD,
                      initial_populate_replay_buffer_steps, DEATH_REWARD, agent_target_update_period))
        print(tf.config.list_physical_devices('GPU'))
    else:
        # The full config is in runs/<policy>.md; the console only needs enough to confirm
        # the right arm started. Override lines are printed unconditionally by tuned().
        print('{0}: lr {1}, discount {2}, quiet logging (SNEK_DEBUG=1 for full output)'.format(
            policy_name, learning_rate, discount), flush=True)
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
    fc_layer_params = tuned('FC_LAYERS', (50, 100, 50),
                            lambda raw: tuple(int(width) for width in raw.split(',')))
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
        target_update_tau=target_update_tau,
        gradient_clipping=gradient_clipping or None,
        n_step_update=n_step_update,
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
        beta_anneal_steps=beta_anneal_steps,
        # n_step_update=n needs n+1 consecutive steps per sampled item.
        sequence_length=n_step_update + 1
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

    # The replay buffer holds 100k transitions, far more than the ~188 KB of agent
    # weights, so keeping it in the full history would mean gigabytes per policy. It
    # lives outside the checkpointer in its own single file, which only ever needs the
    # newest copy to warm-start the next run.
    #
    # History is 10000 deep because a checkpoint is written every 1000 steps, so 1000
    # deep was a rolling 1M-step window that silently deleted the checkpoint behind an
    # arm's best result: b5c-schlongIS's 17.0% peak at 211k became unmeasurable once the
    # run passed 1.28M steps. At 188 KB each, 10000 is ~1.8 GB per policy, which buys a
    # 10M-step window. Note the legacy train*/ dirs run 9.7 MB per checkpoint because
    # they predate moving the replay buffer out; those would be ~97 GB at this depth.
    train_checkpointer = common.Checkpointer(
        ckpt_dir=POLICY_DIR + policy_name,
        max_to_keep=10000,
        agent=agent,
        policy=agent.policy,
        global_step=global_step
    )

    replay_buffer_dir = os.path.join(POLICY_DIR + policy_name, 'replay_buffer')

    train_checkpointer.initialize_or_restore()
    global_step = tf.compat.v1.train.get_global_step()
    if replay_buffer.restore(replay_buffer_dir):
        print('restored replay buffer:', replay_buffer.size, 'transitions')

    # Recorded verbatim into runs/<policy_name>.md, so the write-up can't drift
    # away from the values that actually produced the run.
    run_config = {
        'policy_name': policy_name,
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        'discount': discount,
        'target_update_period': agent_target_update_period,
        'target_update_tau': target_update_tau,
        'gradient_clipping': gradient_clipping or 'none',
        'n_step_update': n_step_update,
        'initial_epsilon': initial_epsilon,
        'min_epsilon': min_epsilon,
        'fc_layer_params': fc_layer_params,
        'replay_buffer': 'cpprb prioritized, capacity {0}'.format(replay_buffer_max_length),
        'priority_exponent (alpha)': priority_exponent,
        'priority_signal': priority_signal,
        'importance_sampling_beta': '{0} -> 1.0 over {1} steps'.format(
            initial_importance_sampling_beta, beta_anneal_steps) if use_is_weights else 'disabled',
        'initial_populate_steps': initial_populate_replay_buffer_steps,
        'initialize_with_schmid': initialize_with_schmid,
        'eval': '{0} episodes every {1} steps'.format(num_eval_episodes, eval_interval),
        'grid': '{0}x{0}, max possible score {1}'.format(GRID_LENGTH, int(MAX_POSSIBLE_SCORE)),
        'DEATH_REWARD': DEATH_REWARD,
        'FOOD_REWARD': FOOD_REWARD,
        'FOOD_DISTANCE_REWARD': FOOD_DISTANCE_REWARD,
        'eval_only': eval_only,
    }

    train(num_iterations, eval_env, eval_parallel_env, train_py_env, agent, collect_driver, batch_size, replay_buffer,
          train_checkpointer, replay_buffer_dir, global_step, epsilon, min_epsilon, eval_only, policy_name,
          run_config, priority_signal, use_is_weights)

    # todo: fix video creation by using the display surface
    # print(create_policy_eval_video(agent.policy, "trained-agent"))

    print('done')


if __name__ == '__main__':
    system_multiprocessing.handle_main(main)
