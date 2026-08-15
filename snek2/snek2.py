from __future__ import absolute_import, division, print_function

import os as _os

_QUIET = _os.environ.get('SNEK_DEBUG', '0') in ('0', '', 'false', 'False')
if _QUIET:
    _os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

# Must precede any pygame import. Snake.Game inits only display+font, but a bare
# pygame.init() anywhere would open a real CoreAudio stream in every env process and
# spin coreaudiod (measured 15% CPU for 10 idle workers). Nothing here plays sound.
_os.environ['SDL_AUDIODRIVER'] = 'dummy'

# The gym "unmaintained, upgrade to Gymnasium" block is deliberately left alone. It is a raw
# print to stderr from gym's import, not a warnings.warn - it survives
# warnings.filterwarnings('ignore') - so removing it would mean redirecting stderr around the
# tf_agents import chain, which would also hide genuine import failures. It costs ~33 lines
# once per launch, against absl's checkpoint line costing ~2000 per run, so it is not worth
# that trade.

import chart_viewer
import policy_arch
from forking_collector import ForkingCollector, validate_config
from prioritized_replay_buffer import TrajectoryPrioritizedReplayBuffer
from shielded_policy import ShieldedEpsilonGreedyPolicy
from snake_environment import OBS_ERA, SnakeEnvironment
from training import *

import os

from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import py_driver
from tf_agents.environments import parallel_py_environment
from tf_agents.environments import tf_py_environment
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
    # Off by default, which is how every arm up to batch 10 ran: nothing in this project seeded
    # anything, so "seed1".."seed4" in those names were labels rather than controlled conditions.
    # Set it and the base seed is recorded in runs/<policy>.md like any other override. It buys
    # reduced variance and a roughly repeatable run, not bit-identical replay — see seed_process.
    seed = tuned('SEED', None, int)
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
    # Floor for the two-phase schedule in training.epsilon_for(). 0.002 is ~2.4 forced
    # non-greedy moves in a 1780-step perfect game, which is meaningful exploration without
    # wrecking the endgame the collect policy has to play through.
    #
    # **Exactly 0 is no longer accepted.** It made the collect policy fully greedy and the
    # replay buffer a closed loop on the policy's own behaviour, and it was where 96.8% of
    # batches 10-11's training steps actually ran. Rejected rather than silently clamped,
    # because a hyperparameter override that quietly does something else is worse than one
    # that refuses: this project reads `hyperparameter override:` at startup to confirm an
    # arm got its config. See hyperparamTuning/hyperparamTuning.md.
    min_epsilon = tuned('MIN_EPSILON', 0.002)
    if min_epsilon < EPSILON_HARD_FLOOR:
        raise SystemExit(
            'SNEK_MIN_EPSILON={0} is below the hard floor of {1}. Epsilon reaching 0 is no '
            'longer supported — it makes the collect policy fully greedy. Use {1} or higher.'
            .format(min_epsilon, EPSILON_HARD_FLOOR))
    if min_epsilon >= initial_epsilon / (2.0 ** BOOTSTRAP_RUNGS):
        raise SystemExit(
            'SNEK_MIN_EPSILON={0} is at or above where the refinement phase starts ({1}), so '
            'epsilon would never decay. Lower the floor or raise SNEK_INITIAL_EPSILON.'
            .format(min_epsilon, initial_epsilon / (2.0 ** BOOTSTRAP_RUNGS)))

    # Food-distance shaping is read in snake_constants rather than here, because it is consumed in
    # the env worker processes — see the comment on FOOD_DISTANCE_REWARD. Printed here anyway, once
    # from the parent, so that `grep 'hyperparameter override:'` on a log still shows every knob an
    # arm was given; that grep is how a misconfigured control arm was caught before.
    if FOOD_DISTANCE_REWARD != DEFAULT_FOOD_DISTANCE_REWARD:
        print('hyperparameter override: FOOD_DISTANCE_REWARD = {0} (default {1})'.format(
            FOOD_DISTANCE_REWARD, DEFAULT_FOOD_DISTANCE_REWARD))
    # Same reasoning for the chase-safe shaping knobs, and the gate is printed alongside `c`
    # whenever the term is on at all — a gate is meaningless without knowing the coefficient, and
    # this pair is exactly the kind of thing that would otherwise be reconstructed from memory when
    # a batch is read back months later.
    if CHASE_SAFE_SHAPING != DEFAULT_CHASE_SAFE_SHAPING:
        print('hyperparameter override: CHASE_SAFE_SHAPING = {0} (default {1}), '
              'CHASE_SAFE_GATE = {2} (default {3})'.format(
                  CHASE_SAFE_SHAPING, DEFAULT_CHASE_SAFE_SHAPING,
                  CHASE_SAFE_GATE, DEFAULT_CHASE_SAFE_GATE))

    # Fraction of *refinement-phase* episodes in which the epsilon coin's random move is drawn
    # from the non-fatal moves instead of all three. 0.0 reproduces batch 12 exactly. See
    # shielded_policy.py for why only the exploration draw is shielded and never the greedy
    # action, and training.guided_fraction_for() for why it stays 0 until bootstrap stands down.
    #
    # 0.8 rather than the original 0.5: every arm from batch 15 onward has passed 0.8 explicitly,
    # so the default now matches the standing value instead of a setting nothing uses. Note this
    # makes the *default* a value that has never been isolated — 0.8 arrived alongside a discount
    # change in batch 14 and has been carried forward since — so it is the standing choice by
    # convention, not by measurement.
    guided_fraction = tuned('GUIDED_FRACTION', 0.8)
    if not 0.0 <= guided_fraction <= 1.0:
        raise SystemExit(
            'SNEK_GUIDED_FRACTION={0} is not a probability. Use a value in [0.0, 1.0].'
            .format(guided_fraction))

    # Forked endgame collection. 1 is off — "one branch, the main line", i.e. the plain collect
    # loop. See forking_collector.py for what the branches are for; the short version is that the
    # buffer holds the consequence of the action taken at an endgame decision point and never the
    # consequence of the alternative, so `Q(s, a_good)` for the untaken safe action is trained on
    # nothing and the argmax has no reason to flip.
    #
    # **Default raised 1 → 4 on 2026-08-14**, because 1 had stopped describing anything that runs:
    # every arm from batch 17 on passes `SNEK_FORK_BRANCHES=4` explicitly, including the record
    # holder `b24d` and its control `b22`. A default nobody uses is a trap rather than a
    # conservative choice — an arm launched without the knob differed from the batch it was meant
    # to join, silently and in the collector rather than anywhere a metric would show it.
    #
    # Read here rather than in snake_constants, unlike FOOD_DISTANCE_REWARD: these are consumed in
    # this process only — the collector, its environment pool and the training environment all live
    # in the parent — so there is no worker-process copy to go stale, and tuned() gets the
    # `hyperparameter override:` line and the run_config entry for free.
    fork_branches = tuned('FORK_BRANCHES', 4, int)
    fork_prob = tuned('FORK_PROB', 0.5)
    fork_min_length = tuned('FORK_MIN_LENGTH', 85, int)
    fork_max_steps = tuned('FORK_MAX_STEPS', 60, int)
    forking_enabled = validate_config(fork_branches, fork_prob, fork_min_length, fork_max_steps,
                                      ZERO_OBS_INDICES)

    # Training never draws — use watch.py to see a policy play. Game.__init__ still calls
    # pygame.display.set_mode() regardless of its display flag, and reset() blits the
    # background and flips, so without the dummy driver this process would open a real window,
    # paint it white once and never touch it again. An empty white square looks like a broken
    # window rather than an absent one, which is exactly the bug this replaced.
    #
    # setdefault so an explicit SDL_VIDEODRIVER still wins, and inside main() rather than at
    # import so that importing snek2 — which eval_checkpoints.py and watch.py both do, for
    # build_q_net — cannot suppress *their* windows.
    _os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')

    eval_only = False
    # eval_only = True

    # Absolute step at which training stops, so a wave self-terminates and frees its slots
    # instead of needing a human to notice. 10M is deliberately generous — this is "well past
    # useful" rather than a planned horizon, and running over is cheaper than an arm dying
    # unattended. `b9b-disc9975b` ran **10.1M steps past its peak** overnight producing nothing,
    # which is the failure this prevents. Raised from 5M after batch 14, where two of four arms
    # produced their best window past 3.5M and `b14c` was still gaining in its final 4.0-4.5M band —
    # so 5M was inside the useful range rather than past it.
    #
    # Absolute rather than per-run: `global_step` is restored on resume, so a relative count would
    # let an arm resumed at 4M run to 9M. See training.steps_remaining().
    max_steps = tuned('MAX_STEPS', 10000000, int)
    if max_steps < 1:
        raise SystemExit('SNEK_MAX_STEPS={0} must be at least 1.'.format(max_steps))

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
    # The anneal target, 1.0 by default. beta below 1.0 leaves IS *partial*: it holds
    # the effective update exponent at alpha*(1-beta) rather than driving it to 0, so
    # prioritization still reaches the gradient. Measured 2026-08-10: beta=1.0 cancels
    # prioritization outright (ESS/N 0.951 vs uniform's 0.975), which made batches 19-20
    # uniform replay past the anneal. IS_BETA_FINAL=0.5 keeps some bias correction for
    # forgetting while leaving alpha*(1-beta)=0.3 of the priority signal on the update.
    final_importance_sampling_beta = tuned('IS_BETA_FINAL', 1.0)
    # 300k, not 1M: arms in this env do their productive learning well before 1M
    # (batch 19's peaked ~0.9-1.1M), so a 1M anneal reaches the target only near the
    # end of the useful window. 300k puts full IS correction in place during it.
    beta_anneal_steps = tuned('BETA_ANNEAL_STEPS', 300000, int)
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

    # Checkpoints below this average score are not written. max_to_keep is a rolling window,
    # so a dead arm that keeps training evicts the good checkpoints behind it: b8d-disc995clip
    # reached 11.64M steps with its last 4.5M at trailing ~1, hit the 10000 cap, and deleted
    # everything before step 1.64M. Set to 0 to restore the old save-everything behaviour.
    #
    # 40 rather than a token value: measured against 232 checkpoints evaluated at 100 episodes,
    # every one that reached 30% perfect games scored at least 49.8 on max(avg_score, trailing),
    # so 40 has margin and still discards the whole dead-arm range.
    snake_constants.MIN_CHECKPOINT_SCORE = tuned('MIN_CHECKPOINT_SCORE', 40.0)

    if policy_name == 'eval':
        eval_only = True
        initial_populate_replay_buffer_steps = 10
        snake_constants.PERFECT_GAME_REWARD = 10000

    if snake_constants.DEBUG_LOGGING:
        print('policy_name: {0}, learning_rate: {1}, discount: {2}, steps_left: False, '
              'FOOD_DISTANCE_REWARD: {3}, initial_populate_replay_buffer_steps: {4}, total_groups_obs: True, '
              'DEATH_REWARD: {5}, agent_target_update_period: {6}'
              .format(policy_name, learning_rate, discount, FOOD_DISTANCE_REWARD,
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

    # Stream 0 is this process: network initialisation, replay sampling, epsilon draws, and the
    # training environment, which lives here rather than in a worker. Every other stream belongs
    # to a parallel eval worker below. Seeding is off unless SNEK_SEED is set, which is how every
    # arm before batch 11 ran — those runs are not reproducible and never were.
    seed_process(seed, stream=0)

    train_py_env = SnakeEnvironment(discount=discount, display=False, policy_name=policy_name)
    train_py_env.reset()
    train_env = tf_py_environment.TFPyEnvironment(train_py_env)

    # Every eval episode runs in parallel, one per worker. There used to be a second
    # environment here whose only job was to play the first episode by itself so it could be
    # drawn in a window; watch.py renders instead, so the environment, the serial episode and
    # the display switches are all gone. Worth ~24% of an eval — see compute_avg_return().
    # One constructor per worker, each closing over its own stream. Deliberately *not*
    # `[make_headless_eval_env] * num_eval_episodes` with a shared seed: the constructor runs
    # inside the worker process, so a single stream would have every worker deal identical food
    # and turn a 10-episode eval into one episode counted ten times. See derive_seed().
    def make_headless_eval_env(stream):
        def build():
            os.environ['SDL_VIDEODRIVER'] = 'dummy'
            seed_process(seed, stream=stream)
            return SnakeEnvironment(discount=discount, display=False, policy_name=policy_name)
        return build

    eval_parallel_env = tf_py_environment.TFPyEnvironment(
        parallel_py_environment.ParallelPyEnvironment(
            [make_headless_eval_env(stream) for stream in range(1, num_eval_episodes + 1)]))

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

    # QNetwork consists of a sequence of Dense layers followed by a dense layer with `num_actions`
    # units to generate one q_value per available action as its output. Built through the shared
    # build_q_net so training, eval and watch.py cannot drift to different architectures — and so the
    # fc_layer_params written into arch.json below are exactly the ones the net was built from.
    q_net = build_q_net(num_actions, fc_layer_params)

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
    # Scheduled alongside epsilon and for the same reason a Variable is used there: the collect
    # policy runs inside a tf.function, so a plain float would be frozen at trace time. Starts
    # at 0.0 because every run starts in the bootstrap phase.
    guided_fraction_var = tf.Variable(0.0, dtype=tf.float32, trainable=False,
                                      name='guided_fraction')
    # Replaces agent.collect_policy. Wraps agent.policy — the greedy one — because the shield
    # supplies its own epsilon branch and must not stack on top of a second one.
    collect_policy = ShieldedEpsilonGreedyPolicy(agent.policy, epsilon, guided_fraction_var)

    # Replay buffer
    # reverb has no macOS wheel, so prioritized replay comes from cpprb's C++ sum
    # tree instead of reverb's ReverbReplayBuffer. See prioritized_replay_buffer.py.
    replay_buffer_data_spec = tensor_spec.to_nest_array_spec(agent.collect_data_spec)
    replay_buffer = TrajectoryPrioritizedReplayBuffer(
        data_spec=replay_buffer_data_spec,
        capacity=replay_buffer_max_length,
        alpha=priority_exponent,
        initial_beta=initial_importance_sampling_beta,
        final_beta=final_importance_sampling_beta,
        beta_anneal_steps=beta_anneal_steps,
        # n_step_update=n needs n+1 consecutive steps per sampled item.
        sequence_length=n_step_update + 1
    )

    def rb_observer(traj):
        replay_buffer.add(traj)

    random_play(train_env.time_step_spec(),
                train_env.action_spec(),
                train_py_env,
                rb_observer,
                initial_populate_replay_buffer_steps)

    # Hoisted out of the PyDriver call below so the forking collector shares the same wrapped
    # policy object, and therefore the same tf.function trace. Two wrappers would mean two traces
    # and an inference path that is only *probably* identical between the on and off paths.
    eager_collect_policy = py_tf_eager_policy.PyTFEagerPolicy(
        collect_policy, use_tf_function=True)

    # Create a driver to collect experience.
    collect_driver = py_driver.PyDriver(
        train_py_env,
        eager_collect_policy,
        [rb_observer],
        max_steps=collect_steps_per_iteration)

    # Built only when forking is on, and `None` otherwise, so training.py keeps calling PyDriver
    # exactly as it always has. The environment pool is allocated once here rather than on demand:
    # a branch that had to construct its environment mid-run would pay the pygame setup cost inside
    # the training loop, and a pool that can run dry is an IndexError thousands of steps in.
    forking_collector = None
    if forking_enabled and not eval_only:
        branch_envs = [SnakeEnvironment(discount=discount, display=False,
                                        policy_name=policy_name)
                       for _ in range(fork_branches - 1)]
        forking_collector = ForkingCollector(
            train_py_env, branch_envs, eager_collect_policy, replay_buffer,
            max_branches=fork_branches, fork_prob=fork_prob,
            fork_min_length=fork_min_length, fork_max_steps=fork_max_steps,
            guided_flag=collect_policy.guided_episode,
            seed=derive_seed(seed, stream=num_eval_episodes + 1))

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

    # arch.json records the architecture and observation era this checkpoint is built for, so
    # nothing can restore it into a differently-shaped or differently-meaning network without a loud
    # failure (policy_arch.py). A fresh policy writes it; a resume asserts the environment still
    # matches *before* restoring — the guard that catches a resume under a changed SNEK_FC_LAYERS
    # before it loads weights into the wrong net.
    policy_dir = POLICY_DIR + policy_name
    obs_len = int(train_py_env.observation_spec().shape[0])
    if policy_arch.read_arch(policy_dir) is None:
        policy_arch.write_arch(policy_dir, policy_arch.build_arch(
            fc_layer_params, num_actions, obs_len, OBS_ERA))
    else:
        policy_arch.assert_restorable(policy_dir, num_actions, obs_len, OBS_ERA)
        policy_arch.assert_config_matches(policy_dir, fc_layer_params)

    train_checkpointer.initialize_or_restore()
    global_step = tf.compat.v1.train.get_global_step()
    if replay_buffer.restore(replay_buffer_dir):
        print('restored replay buffer:', replay_buffer.size, 'transitions')

    # Recorded verbatim into runs/<policy_name>.md, so the write-up can't drift
    # away from the values that actually produced the run.
    run_config = {
        'policy_name': policy_name,
        # `tuned()` only prints an override to the console; this dict is what reaches
        # runs/<policy>.md, and it is written out by hand. A seed nobody recorded is no better
        # than no seed, so this line is the whole point of the knob.
        'seed': seed if seed is not None else 'unseeded',
        'zeroed_observations': (','.join(str(i) for i in sorted(ZERO_OBS_INDICES))
                                if ZERO_OBS_INDICES else 'none'),
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        'discount': discount,
        'target_update_period': agent_target_update_period,
        'target_update_tau': target_update_tau,
        'gradient_clipping': gradient_clipping or 'none',
        'n_step_update': n_step_update,
        'initial_epsilon': initial_epsilon,
        'min_epsilon': min_epsilon,
        'epsilon_schedule': 'bootstrap on avg_reward {0} then geometric to floor by {1:.0%} '
                            'trailing-{2} perfect'.format(
                                list(BOOTSTRAP_REWARD_THRESHOLDS), REFINE_PERFECT_TARGET,
                                REFINE_TRAILING_WINDOW),
        'guided_fraction': guided_fraction,
        'forking': ('off' if not forking_enabled else
                    'up to {0} live branches including the main line, fork p={1} at length >= {2}, '
                    'branch capped at {3}, one branch advanced per iteration'.format(
                        fork_branches, fork_prob, fork_min_length,
                        '{0} steps'.format(fork_max_steps) if fork_max_steps else 'its terminal '
                                                                                 'state')),
        'exploration_shield': ('off' if guided_fraction == 0.0 else
                               '{0:.0%} of refinement-phase episodes draw the epsilon move from '
                               'non-fatal actions; greedy moves never shielded'
                               .format(guided_fraction)),
        'fc_layer_params': fc_layer_params,
        'replay_buffer': 'cpprb prioritized, capacity {0}'.format(replay_buffer_max_length),
        'priority_exponent (alpha)': priority_exponent,
        'priority_signal': priority_signal,
        'importance_sampling_beta': '{0} -> {1} over {2} steps'.format(
            initial_importance_sampling_beta, final_importance_sampling_beta,
            beta_anneal_steps) if use_is_weights else 'disabled',
        'max_steps': max_steps,
        'initial_populate_steps': initial_populate_replay_buffer_steps,
        'eval': '{0} episodes every {1} steps'.format(num_eval_episodes, eval_interval),
        'grid': '{0}x{0}, max possible score {1}'.format(GRID_LENGTH, int(MAX_POSSIBLE_SCORE)),
        'DEATH_REWARD': DEATH_REWARD,
        'FOOD_REWARD': FOOD_REWARD,
        'FOOD_DISTANCE_REWARD': FOOD_DISTANCE_REWARD,
        'CHASE_SAFE_SHAPING': ('off' if not CHASE_SAFE_SHAPING else
                               'c={0}, potential-based on head/food/tail in one region, '
                               '{1}'.format(CHASE_SAFE_SHAPING,
                                            'ungated' if CHASE_SAFE_GATE <= 0 else
                                            'gated to snake length >= {0}'.format(
                                                CHASE_SAFE_GATE))),
        'eval_only': eval_only,
        'min_checkpoint_score': snake_constants.MIN_CHECKPOINT_SCORE,
    }

    # A live chart window for this batch, in its own process. One per batch, so four arms
    # launched together share it, and it exits by itself when the last of them stops.
    if not eval_only:
        chart_viewer.spawn_for_policy(policy_name)

    train(max_steps, eval_parallel_env, train_py_env, agent, collect_driver, batch_size, replay_buffer,
          train_checkpointer, replay_buffer_dir, global_step, epsilon, initial_epsilon, min_epsilon,
          guided_fraction_var, guided_fraction,
          eval_only, policy_name, run_config, priority_signal, use_is_weights,
          forking_collector)

    # todo: fix video creation by using the display surface
    # print(create_policy_eval_video(agent.policy, "trained-agent"))

    print('done')


if __name__ == '__main__':
    system_multiprocessing.handle_main(main)
