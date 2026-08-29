"""Tests for C51 — `categorical_agent.py`, and the support arithmetic that sizes it.

Design and measurements: [`plans/distributional-c51.md`](../plans/distributional-c51.md).

Two of these carry more weight than the rest.

**The loss is reimplemented, not vendored**, so `test_loss_equals_upstream_when_unweighted_and_single`
is what says the reimplementation is right — and it is also the tripwire for a `tf_agents` upgrade
changing `project_distribution` under us, which a copied block would absorb silently.

**The two upstream defects the override exists for are each pinned by a test that fails on upstream.**
`test_skewed_importance_weights_change_the_loss` and `test_the_priority_signal_reaches_both_extra_fields`
both pass against `SnekCategoricalDqnAgent` and fail against `CategoricalDqnAgent`, which is the only
honest way to state "we override this because upstream is wrong here".

One fixture detail is load-bearing and cost a wrong result in the probe: **a double-selection test must
desync the target network first.** `initialize()` copies the online weights into the target, so
immediately after construction both selections agree exactly and the mutant survives.
"""
import math

import numpy as np
import tensorflow as tf
from tf_agents.agents.categorical_dqn import categorical_dqn_agent
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.trajectories import trajectory

import categorical_agent
import shielded_policy
import snake_constants
import under_the_hood

OBS_SIZE = 30
NUM_ACTIONS = 3
NUM_ATOMS = 51
V_MIN, V_MAX = -5.0, 120.0
FC = (16, 16)
SAFE = shielded_policy.SAFETY_OBS_START


def _specs():
    obs_spec = tensor_spec.TensorSpec((OBS_SIZE,), tf.float32)
    action_spec = tensor_spec.BoundedTensorSpec((), tf.int32, 0, NUM_ACTIONS - 1)
    return obs_spec, action_spec


def make_agent(double=True, priority_signal=categorical_agent.PRIORITY_KL, num_atoms=NUM_ATOMS,
               v_min=V_MIN, v_max=V_MAX, zero_init=False, upstream=False, seed=0):
    """An initialised agent over this project's observation size, and its online network.

    `seed` fixes the network draw, so two agents built with the same seed hold the same weights and
    their losses are comparable — which is what the equal-to-upstream test needs.
    """
    tf.random.set_seed(seed)
    obs_spec, action_spec = _specs()
    net = under_the_hood.build_categorical_q_net(
        obs_spec, action_spec, FC, num_atoms, v_min=v_min, v_max=v_max, zero_init=zero_init)
    kind = (categorical_dqn_agent.CategoricalDqnAgent if upstream
            else categorical_agent.SnekCategoricalDqnAgent)
    extra = {} if upstream else {'double': double, 'priority_signal': priority_signal}
    agent = kind(
        ts.time_step_spec(obs_spec), action_spec,
        categorical_q_network=net,
        min_q_value=v_min, max_q_value=v_max,
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        epsilon_greedy=0.0, target_update_period=1, target_update_tau=1.0,
        train_step_counter=tf.compat.v1.train.get_or_create_global_step(),
        **extra)
    agent.initialize()
    return agent, net


def make_experience(rewards=(1.0, 0.0, 1.0, -5.0), discounts=(0.9975, 0.9975, 0.9975, 0.0),
                    actions=(0, 1, 2, 0), seed=3):
    """A `[batch, 2]` trajectory, which is what `experience_to_transitions` consumes.

    The last row is terminal by default (`discount == 0`, `reward == DEATH_REWARD`), because a batch
    with no terminal transition would let the terminal contract break unnoticed.
    """
    rng = np.random.default_rng(seed)
    batch = len(rewards)
    obs = rng.random((batch, 2, OBS_SIZE)).astype(np.float32)
    pair = lambda values, dtype: tf.constant([[v, v] for v in values], dtype=dtype)
    return trajectory.Trajectory(
        step_type=tf.fill([batch, 2], tf.constant(1, tf.int32)),
        observation=tf.constant(obs),
        action=pair(actions, tf.int32),
        policy_info=(),
        next_step_type=tf.fill([batch, 2], tf.constant(1, tf.int32)),
        reward=pair(rewards, tf.float32),
        discount=pair(discounts, tf.float32))


def _next_time_steps(experience):
    return trajectory.experience_to_transitions(experience, squeeze_time_dim=True)[2]


def desync_target(agent, scale=3.0):
    """Push the target network away from the online one, so the two argmaxes can differ.

    Scaling the head rather than reinitialising it: the two nets stay the same shape and the same
    trunk, so a difference in the selected atoms can only come from the selection rule.
    """
    for variable in agent._target_q_network.trainable_variables:
        variable.assign(variable * scale + 0.7)


# ------------------------------------------------------- the loss, against upstream

def test_loss_equals_upstream_when_unweighted_and_single():
    # Single (target-net) selection and no weights is exactly the configuration upstream implements,
    # so the two numbers must agree. Any error in the reimplemented projection or cross-entropy lands
    # here, and so does a tf_agents change to project_distribution.
    experience = make_experience()
    ours, _ = make_agent(double=False, seed=11)
    theirs, _ = make_agent(upstream=True, seed=11)
    mine = float(ours._loss(experience, gamma=0.9975).loss.numpy())
    upstream = float(theirs._loss(experience, gamma=0.9975).loss.numpy())
    assert abs(mine - upstream) < 1e-5, (mine, upstream)


def test_skewed_importance_weights_change_the_loss():
    # The first upstream defect: `_loss` accepts `weights` and never uses them, so every
    # importance-sampling weight this project computes was being discarded. PER is on in every arm and
    # SNEK_IS_WEIGHTS is a live knob, so that is a whole feature quietly doing nothing.
    experience = make_experience()
    weights = tf.constant([32.0, 1.0, 1.0, 1.0])
    ours, _ = make_agent(double=False, seed=11)
    flat = float(ours._loss(experience, gamma=0.9975).loss.numpy())
    skewed = float(ours._loss(experience, gamma=0.9975, weights=weights).loss.numpy())
    assert abs(skewed - flat) > 0.01, (flat, skewed)

    # And the same call against upstream does not move, which is the evidence for the override.
    theirs, _ = make_agent(upstream=True, seed=11)
    their_flat = float(theirs._loss(experience, gamma=0.9975).loss.numpy())
    their_skewed = float(theirs._loss(experience, gamma=0.9975, weights=weights).loss.numpy())
    assert their_flat == their_skewed


def test_the_priority_signal_reaches_both_extra_fields():
    # The second upstream defect: it returns DqnLossInfo(td_loss=(), td_error=()), so training.py's
    # `signal.numpy()` would raise on an empty tuple and PER would have no priorities at all. Both
    # fields are filled because SNEK_PRIORITY_SIGNAL picks between them and either must work.
    experience = make_experience()
    agent, _ = make_agent()
    extra = agent._loss(experience, gamma=0.9975).extra
    error, loss = extra.td_error.numpy(), extra.td_loss.numpy()
    assert error.shape == (4,), error.shape
    assert np.all(np.isfinite(error)) and np.all(error >= 0.0)
    assert np.array_equal(error, loss)

    theirs, _ = make_agent(upstream=True)
    assert theirs._loss(experience, gamma=0.9975).extra.td_error == ()


def test_the_loss_at_init_is_about_log_num_atoms():
    # A freshly built head is near-uniform over its atoms, so the cross-entropy against any target
    # starts at ~ln(N) — 3.932 at 51 atoms. This is the cheap end-to-end check that the network's atom
    # count and the agent's support are the same size: a mismatch shows up as a wrong constant here
    # long before it shows up as a bad arm.
    for atoms in (51, 81):
        agent, _ = make_agent(num_atoms=atoms)
        loss = float(agent._loss(make_experience(), gamma=0.9975).loss.numpy())
        assert abs(loss - math.log(atoms)) < 0.05, (atoms, loss)


# ------------------------------------------------------------ the priority signal

def test_kl_is_zero_when_the_prediction_matches_the_target_and_ce_is_not():
    # Why the priority is the KL and not the cross-entropy. The projection spreads target mass over
    # two atoms, so H(target) > 0 and the cross-entropy carries an irreducible per-transition floor
    # even at a perfect prediction. That floor compresses the relative spread PER's alpha acts on —
    # the same defect this repo already documented for Huber td_loss.
    agent, _ = make_agent(priority_signal=categorical_agent.PRIORITY_KL)
    # -0.5 (STARVE_REWARD) is deliberately *not* on the 2.5-spaced grid, so the projection splits it
    # 0.20 / 0.80 across two atoms and H(target) is 0.5004 nats rather than 0.
    experience = make_experience(rewards=(-0.5,), discounts=(0.0,), actions=(0,))
    target = agent.target_distribution(_next_time_steps(experience)).numpy()[0]
    # Masked rather than np.where'd: np.where evaluates both branches, so log(0) would warn.
    live = target[target > 0]
    entropy = -float(np.sum(live * np.log(live)))
    assert entropy > 0.4, entropy

    # Make the prediction *equal* the target for the taken action, so KL is 0 by construction.
    bias = under_the_hood.categorical_head_bias(agent._q_network, NUM_ACTIONS, NUM_ATOMS)
    logits = np.log(np.maximum(target, 1e-12)).astype(np.float32)
    bias.assign(np.tile(logits, NUM_ACTIONS))
    for kernel in [v for v in agent._q_network.trainable_variables
                   if v.shape.rank == 2 and int(v.shape[1]) == NUM_ACTIONS * NUM_ATOMS]:
        kernel.assign(tf.zeros_like(kernel))

    kl = float(agent._loss(experience, gamma=0.9975).extra.td_error.numpy()[0])
    assert kl < 1e-4, kl
    # The same state under the CE signal reports the entropy floor instead.
    agent._priority_signal = categorical_agent.PRIORITY_CE
    ce = float(agent._loss(experience, gamma=0.9975).extra.td_error.numpy()[0])
    assert abs(ce - entropy) < 1e-3, (ce, entropy)


def test_priority_signal_for_maps_the_scalar_knob_onto_the_kl():
    # A c51 arm is launched with a control's environment block copied verbatim, so `td_error` — the
    # default, and what every existing launch line says — must resolve rather than refuse.
    for configured in ('td_error', 'td_loss', 'kl'):
        assert categorical_agent.priority_signal_for(configured) == 'kl'
    assert categorical_agent.priority_signal_for('ce') == 'ce'
    try:
        categorical_agent.priority_signal_for('huber')
    except ValueError:
        return
    raise AssertionError('an unknown priority signal must raise rather than defaulting')


# --------------------------------------------------------------- double selection

def test_double_selection_uses_the_online_argmax_on_a_desynced_target():
    # The one fixture rule this file has: **desync the target first.** initialize() copies the online
    # weights into the target, so straight after construction both rules pick the same action and a
    # test that skips this passes whichever branch is live.
    agent, net = make_agent(double=True)
    desync_target(agent)
    next_time_steps = _next_time_steps(make_experience())
    support = np.linspace(V_MIN, V_MAX, NUM_ATOMS)

    online_q = under_the_hood.expected_q(
        net, next_time_steps.observation, support=support,
        step_type=next_time_steps.step_type).numpy()
    target_logits, _ = agent._target_q_network(
        next_time_steps.observation, step_type=next_time_steps.step_type)
    target_probabilities = tf.nn.softmax(target_logits).numpy()
    target_q = (target_probabilities * support).sum(axis=-1)

    # The fixture is only meaningful if the two rules actually disagree somewhere.
    assert not np.array_equal(online_q.argmax(axis=-1), target_q.argmax(axis=-1))

    selected = agent._next_q_distribution(next_time_steps).numpy()
    expected = target_probabilities[np.arange(len(selected)), online_q.argmax(axis=-1)]
    assert np.allclose(selected, expected, atol=1e-6)


def test_single_selection_uses_the_target_argmax():
    # The other branch, so `SNEK_C51_DOUBLE=0` is a measurable ablation rather than dead code.
    agent, _ = make_agent(double=False)
    desync_target(agent)
    next_time_steps = _next_time_steps(make_experience())
    support = np.linspace(V_MIN, V_MAX, NUM_ATOMS)
    target_logits, _ = agent._target_q_network(
        next_time_steps.observation, step_type=next_time_steps.step_type)
    target_probabilities = tf.nn.softmax(target_logits).numpy()
    target_q = (target_probabilities * support).sum(axis=-1)
    selected = agent._next_q_distribution(next_time_steps).numpy()
    expected = target_probabilities[np.arange(len(selected)), target_q.argmax(axis=-1)]
    assert np.allclose(selected, expected, atol=1e-6)


# ------------------------------------------------------------ the terminal contract

def test_a_terminal_transition_puts_all_target_mass_at_the_reward():
    # `discount == 0` is the only thing that stops the bootstrap in this environment
    # (snake_environment.to_tensor_time_step), and at d=0 the target support collapses to the reward.
    # 100.0 is exactly atom 42 on the shipped grid, so the target must be a *point* mass — a `1 - done`
    # spelling, or dropping the discount factor, would spread it over the discounted support and keep
    # training terminal states against a tail that does not exist.
    agent, _ = make_agent()
    experience = make_experience(rewards=(100.0,), discounts=(0.0,), actions=(0,))
    target = agent.target_distribution(_next_time_steps(experience), gamma=0.9975).numpy()[0]
    atom = int(round((100.0 - V_MIN) / ((V_MAX - V_MIN) / (NUM_ATOMS - 1))))
    assert abs(target[atom] - 1.0) < 1e-5, (atom, target[atom])
    assert target.sum() - target[atom] < 1e-5


def test_a_non_terminal_transition_spreads_its_target_over_the_support():
    # The negative control for the test above: with d > 0 the target is the *bootstrapped*
    # distribution, so a point mass here would mean the discount had stopped reaching the support.
    agent, _ = make_agent()
    experience = make_experience(rewards=(1.0,), discounts=(0.9975,), actions=(0,))
    target = agent.target_distribution(_next_time_steps(experience), gamma=0.9975).numpy()[0]
    assert (target > 1e-4).sum() > 5, (target > 1e-4).sum()


# ------------------------------------------------------------------ support guards

def test_the_derived_maximum_return_is_not_hard_coded():
    # 94 * FOOD_REWARD + PERFECT_GAME_REWARD on the shipped constants. The first draft of the plan
    # wrote `PERFECT_GAME_REWARD + FOOD_REWARD` (101), which was *too low* — a near-win state still
    # collects food on the way, which is why the measured max is 104.38. So the bound is derived, and
    # this test moves the constants to prove it.
    assert categorical_agent.theoretical_max_return() == 194.0
    original = snake_constants.PERFECT_GAME_REWARD
    try:
        snake_constants.PERFECT_GAME_REWARD = 200.0
        assert categorical_agent.theoretical_max_return() == 294.0
    finally:
        snake_constants.PERFECT_GAME_REWARD = original


def test_the_minimum_return_is_one_step_of_death():
    # `Snake.step` **assigns** each outcome reward rather than accumulating, so DEATH_REWARD +
    # STARVE_REWARD describes a reward the game cannot pay. Only the shaping term is additive on a
    # terminal step, and it is off in the shipped config.
    assert categorical_agent.min_possible_return() == snake_constants.DEATH_REWARD
    original = snake_constants.CHASE_SAFE_SHAPING
    try:
        snake_constants.CHASE_SAFE_SHAPING = 0.10
        assert (categorical_agent.min_possible_return()
                == snake_constants.DEATH_REWARD - 0.10)
    finally:
        snake_constants.CHASE_SAFE_SHAPING = original


def test_the_chosen_support_starts_and_warns():
    # The configuration this feature is for: v_max = 120 is deliberately below the derived 194, which
    # is a judgement rather than an error. Collapsing the guards into one "must cover the theoretical
    # max" rule would refuse to start it; collapsing them into none would lose the record of the
    # choice, which is why the warning is returned rather than only printed.
    warnings = categorical_agent.check_support(V_MIN, V_MAX, NUM_ATOMS)
    assert len(warnings) == 1
    assert '194' in warnings[0] and '120' in warnings[0]


def test_a_v_max_below_the_measured_maximum_is_refused():
    # Clipping returns real policies demonstrably reach (104.38 over 3 checkpoints, 60 episodes) is a
    # mistake, not a trade-off.
    try:
        categorical_agent.check_support(-5.0, 100.0, 51)
    except SystemExit as exc:
        assert 'SNEK_C51_ALLOW_CLIPPING' in str(exc)
    else:
        raise AssertionError('a v_max below the measured maximum return must refuse to start')
    # And the escape hatch overrides it, still warning.
    assert categorical_agent.check_support(-5.0, 100.0, 51, allow_clipping=True)


def test_a_v_min_above_the_reachable_minimum_is_refused():
    # A clipped death is a wrong terminal target, and satisfying this costs nothing.
    try:
        categorical_agent.check_support(-1.0, 120.0, 51)
    except SystemExit as exc:
        assert 'SNEK_V_MIN' in str(exc)
        return
    raise AssertionError('a v_min above DEATH_REWARD must refuse to start')


def test_a_support_covering_the_derived_bound_warns_about_nothing():
    assert categorical_agent.check_support(-5.0, 200.0, 81) == []


# --------------------------------------------------------------------- zero init

def test_zero_init_moves_the_initial_expected_q_from_the_midpoint_to_zero():
    # A categorical head's initial expected Q is the grid midpoint — 57.5 here — where a scalar head
    # starts at ~0. That is a second difference between a c51 arm and its ddqn control, on top of the
    # algorithm, and it scales with v_max. The mutant this kills is the ramp silently doing nothing,
    # which would look identical in every metric the project records.
    support = np.linspace(V_MIN, V_MAX, NUM_ATOMS)
    observations = tf.constant(np.random.default_rng(7).random((32, OBS_SIZE)), dtype=tf.float32)

    _, plain = make_agent(zero_init=False)
    midpoint = float(under_the_hood.expected_q(plain, observations, support=support).numpy().mean())
    assert abs(midpoint - (V_MIN + V_MAX) / 2.0) < 1.0, midpoint

    _, ramped = make_agent(zero_init=True)
    zeroed = float(under_the_hood.expected_q(ramped, observations, support=support).numpy().mean())
    assert abs(zeroed) < 1.0, zeroed


def test_zero_init_refuses_a_support_that_does_not_straddle_zero():
    try:
        categorical_agent.zero_init_lambda(1.0, 120.0, 51)
    except ValueError:
        return
    raise AssertionError('no bias ramp reaches 0 from a strictly positive support')


# ----------------------------------------------------------- narrowings and shield

def test_n_step_update_above_one_is_refused_at_construction():
    # The n-step target support is ~10 lines and nothing since batch 15 uses n>1, so it is refused
    # rather than shipped untested. A silently-wrong n-step target would look like a bad arm.
    try:
        make_agent_with(n_step_update=3)
    except ValueError as exc:
        assert 'n_step_update' in str(exc)
        return
    raise AssertionError('n_step_update > 1 must raise at construction')


def make_agent_with(**kwargs):
    """`make_agent` with extra constructor arguments, for the narrowing tests."""
    obs_spec, action_spec = _specs()
    net = under_the_hood.build_categorical_q_net(obs_spec, action_spec, FC, NUM_ATOMS,
                                                 v_min=V_MIN, v_max=V_MAX)
    return categorical_agent.SnekCategoricalDqnAgent(
        ts.time_step_spec(obs_spec), action_spec,
        categorical_q_network=net, min_q_value=V_MIN, max_q_value=V_MAX,
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        train_step_counter=tf.compat.v1.train.get_or_create_global_step(), **kwargs)


def test_an_unknown_priority_signal_is_refused_at_construction():
    try:
        make_agent_with(priority_signal='td_error')
    except ValueError as exc:
        assert 'priority_signal' in str(exc)
        return
    raise AssertionError("the constructor takes a resolved signal; map it with priority_signal_for")


def test_the_shield_masks_fatal_moves_over_a_categorical_greedy_policy():
    # The shield reads the safety block straight out of the observation and wraps `agent.policy`,
    # which for c51 is GreedyPolicy(CategoricalQPolicy). Its requirement is `info_spec == ()`; this is
    # the test that says the categorical policy still satisfies it, at the level of the behaviour that
    # matters — an exploration draw never takes the fatal move when a safe one exists.
    agent, _ = make_agent()
    policy = shielded_policy.ShieldedEpsilonGreedyPolicy(
        agent.policy, tf.Variable(1.0, dtype=tf.float32), tf.Variable(1.0, dtype=tf.float32))
    values = np.zeros((1, OBS_SIZE), dtype=np.float32)
    values[0, SAFE:SAFE + NUM_ACTIONS] = [0, 0, 1]
    step = ts.restart(tf.constant(values), batch_size=1)
    chosen = {int(policy.action(step).action.numpy()[0]) for _ in range(50)}
    assert chosen == {2}, chosen
