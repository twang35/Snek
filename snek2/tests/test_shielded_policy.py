"""Tests for the exploration-only safety shield in shielded_policy.py.

The shield exists because batch 12 deadlocked: epsilon pinned at 0.05 for up to 942k steps with
every arm at 0% perfect games, because a random move with a long snake is usually fatal and the
replay buffer never filled with endgame states. These tests pin the properties that failure and
the fix turn on — that the *greedy* action is never overridden, that a guided exploration draw
never kills the snake when it has a choice, and that guided_fraction=0 reproduces the old
behaviour exactly.
"""
import numpy as np
import tensorflow as tf
from tf_agents.networks import sequential
from tf_agents.policies import greedy_policy
from tf_agents.policies import q_policy
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import time_step as ts

import shielded_policy
import training

OBS_SIZE = 30
NUM_ACTIONS = 3
SAFE = shielded_policy.SAFETY_OBS_START


class FixedQNetwork(tf.keras.layers.Layer):
    """A Q-network with hand-picked Q-values, so "the greedy action" is exactly known."""

    def __init__(self, q_values):
        super(FixedQNetwork, self).__init__()
        self._q_values = tf.constant([q_values], dtype=tf.float32)

    def call(self, observation):
        return tf.tile(self._q_values, [tf.shape(observation)[0], 1])


def make_policy(q_values, epsilon, guided_fraction):
    """A shielded policy over a 3-action spec whose greedy choice is argmax(q_values).

    The `GreedyPolicy` wrapper is not decoration: a bare `QPolicy` *samples* from a Categorical
    over the Q-values, so with q=[9, 5, 1] it returns the non-argmax action ~1.8% of the time.
    `DqnAgent` builds `agent.policy` as `GreedyPolicy(QPolicy(...))` for exactly that reason and
    that is what production passes to the shield, so the fixture has to match or every
    "the greedy action is never overridden" test below reads as a flaky shield bug.
    """
    obs_spec = tensor_spec.TensorSpec((OBS_SIZE,), tf.float32)
    action_spec = tensor_spec.BoundedTensorSpec((), tf.int32, 0, NUM_ACTIONS - 1)
    net = sequential.Sequential([FixedQNetwork(q_values)], input_spec=obs_spec)
    greedy = greedy_policy.GreedyPolicy(
        q_policy.QPolicy(ts.time_step_spec(obs_spec), action_spec, q_network=net))
    return shielded_policy.ShieldedEpsilonGreedyPolicy(
        greedy, tf.Variable(epsilon, dtype=tf.float32),
        tf.Variable(guided_fraction, dtype=tf.float32))


def observation(safety):
    """One observation whose safety block (indices 6-8) is `safety`; everything else 0."""
    values = np.zeros((1, OBS_SIZE), dtype=np.float32)
    values[0, SAFE:SAFE + NUM_ACTIONS] = safety
    return tf.constant(values)


def first_step(safety):
    return ts.restart(observation(safety), batch_size=1)


def mid_step(safety):
    """A non-first step, so the per-episode guided flag is not redrawn."""
    return ts.transition(observation(safety), reward=tf.constant([0.0]),
                         discount=tf.constant([1.0]))


def actions_over(policy, step, count=300):
    return [int(policy.action(step).action.numpy()[0]) for _ in range(count)]


# ------------------------------------------------- the greedy action is never shielded

def test_a_fatal_greedy_action_still_plays_out():
    # The core asymmetry. epsilon=0 means every move is the network's own choice, and the
    # network wants action 0, which is fatal. The shield must not save it: that death is the
    # only thing that ever teaches Q(s, 0) about DEATH_REWARD, and evals run unshielded.
    policy = make_policy([9.0, 1.0, 1.0], epsilon=0.0, guided_fraction=1.0)
    assert set(actions_over(policy, first_step([0, 1, 1]))) == {0}


def test_a_fatal_greedy_action_plays_out_even_with_a_safe_alternative():
    # Same property on a mid-episode step, and with two safe moves available, so there is no
    # reading of the code where the shield "had no choice but" to allow it.
    policy = make_policy([9.0, 5.0, 1.0], epsilon=0.0, guided_fraction=1.0)
    assert set(actions_over(policy, mid_step([0, 1, 1]))) == {0}


# ------------------------------------------------- guided exploration avoids fatal moves

def test_guided_exploration_never_takes_a_fatal_move():
    # epsilon=1.0 makes every action an exploration draw. Only action 2 is safe, so a guided
    # episode must return action 2 every single time, however the coin falls.
    policy = make_policy([9.0, 1.0, 1.0], epsilon=1.0, guided_fraction=1.0)
    assert set(actions_over(policy, first_step([0, 0, 1]))) == {2}


def test_guided_exploration_spreads_over_every_safe_move():
    # It must be a *draw* over the safe actions, not a preference for the lowest index: a
    # deterministic pick would silently turn exploration off.
    policy = make_policy([9.0, 1.0, 1.0], epsilon=1.0, guided_fraction=1.0)
    assert set(actions_over(policy, first_step([0, 1, 1]))) == {1, 2}


def test_guided_exploration_can_still_re_pick_the_greedy_move():
    # Epsilon's meaning is preserved: the draw is over safe actions *including* the greedy one,
    # which is what keeps the effective non-greedy rate at epsilon * 2/3 when nothing is
    # masked. Excluding it would silently change what every epsilon number in the project means.
    policy = make_policy([9.0, 1.0, 1.0], epsilon=1.0, guided_fraction=1.0)
    assert set(actions_over(policy, first_step([1, 1, 1]))) == {0, 1, 2}


def test_no_safe_move_lets_the_snake_die():
    # Boxed in. There is nothing to steer to, so the shield must fall through to an ordinary
    # uniform draw rather than crashing, hanging or emitting an out-of-range action.
    policy = make_policy([9.0, 1.0, 1.0], epsilon=1.0, guided_fraction=1.0)
    drawn = set(actions_over(policy, first_step([0, 0, 0])))
    assert drawn <= {0, 1, 2}
    assert len(drawn) > 1, 'a fully boxed-in state should still explore, not freeze on one move'


# ------------------------------------------------- guided_fraction = 0 changes nothing

def test_unguided_exploration_takes_fatal_moves():
    # guided_fraction=0 must reproduce tf_agents' EpsilonGreedyPolicy. With only action 2 safe
    # and epsilon=1, an unshielded draw hits the two fatal actions about two thirds of the time.
    policy = make_policy([9.0, 1.0, 1.0], epsilon=1.0, guided_fraction=0.0)
    drawn = actions_over(policy, first_step([0, 0, 1]))
    fatal = sum(1 for action in drawn if action in (0, 1))
    assert set(drawn) == {0, 1, 2}
    assert fatal > len(drawn) / 3, 'unguided exploration should be killing the snake freely'


def test_unguided_draw_is_uniform_over_all_actions():
    # Not just "hits every action" but hits them evenly, which is what makes epsilon * 2/3 the
    # right effective rate. 300 draws, so a third is 100 +/- ~15 at three sigma.
    policy = make_policy([9.0, 1.0, 1.0], epsilon=1.0, guided_fraction=0.0)
    drawn = actions_over(policy, first_step([0, 0, 1]), count=300)
    for action in range(NUM_ACTIONS):
        assert 60 < drawn.count(action) < 140, 'action {0}: {1}'.format(action, drawn.count(action))


# ------------------------------------------------- the flag is per episode, not per step

def test_the_guided_flag_is_redrawn_only_on_the_first_step():
    # A mid-episode step must not reroll, or a single trajectory would mix shielded and
    # unshielded moves and the buffer would be impossible to reason about.
    policy = make_policy([9.0, 1.0, 1.0], epsilon=1.0, guided_fraction=0.5)
    policy.guided_episode.assign(True)
    policy.action(mid_step([0, 0, 1]))
    assert bool(policy.guided_episode.numpy()) is True

    policy.guided_episode.assign(False)
    policy.action(mid_step([0, 0, 1]))
    assert bool(policy.guided_episode.numpy()) is False


def test_a_mid_episode_step_keeps_shielding_the_whole_episode():
    # The flag surviving is only useful if it still *drives* the shield on later steps.
    policy = make_policy([9.0, 1.0, 1.0], epsilon=1.0, guided_fraction=0.0)
    policy.guided_episode.assign(True)
    assert set(actions_over(policy, mid_step([0, 0, 1]))) == {2}


def test_the_first_step_of_an_episode_redraws_the_flag():
    policy = make_policy([9.0, 1.0, 1.0], epsilon=1.0, guided_fraction=1.0)
    policy.guided_episode.assign(False)
    policy.action(first_step([1, 1, 1]))
    assert bool(policy.guided_episode.numpy()) is True

    policy = make_policy([9.0, 1.0, 1.0], epsilon=1.0, guided_fraction=0.0)
    policy.guided_episode.assign(True)
    policy.action(first_step([1, 1, 1]))
    assert bool(policy.guided_episode.numpy()) is False


def test_half_of_episodes_are_guided_at_the_default_fraction():
    # 0.5 is the configured default, so it has to actually split. 400 episode starts, so a half
    # is 200 +/- ~30 at three sigma.
    policy = make_policy([9.0, 1.0, 1.0], epsilon=1.0, guided_fraction=0.5)
    guided = 0
    for _ in range(400):
        policy.action(first_step([1, 1, 1]))
        guided += int(bool(policy.guided_episode.numpy()))
    assert 150 < guided < 250, 'guided {0} of 400 episodes'.format(guided)


# ------------------------------------------------- the schedule that turns it on

def test_the_shield_is_off_during_bootstrap():
    # Nothing to protect while the snake is a few segments long, and those deaths are the
    # signal. Every bootstrap-band reward must read 0.
    for avg_reward in (-5.0, 0.0, 7.0, 15.0, 20.0):
        assert training.guided_fraction_for(avg_reward, 0.4, 0.5) == 0.0


def test_the_shield_switches_on_at_the_handover():
    # Exactly where bootstrap stands down and the refinement phase takes over at 0.05.
    assert training.bootstrap_epsilon(25.0, 0.4) == 0.0
    assert training.guided_fraction_for(25.0, 0.4, 0.5) == 0.5


def test_a_collapsed_arm_loses_the_shield_with_its_refinement_phase():
    # Stateless, like epsilon_for: one rule, "shielded iff refining". An arm that falls back
    # into the bootstrap band is relearning to survive, which is where dying is informative.
    assert training.guided_fraction_for(4.0, 0.4, 0.5) == 0.0


def test_a_zero_fraction_stays_zero_everywhere():
    # The knob that reproduces batch 12 exactly, so the A/B has a real control arm.
    for avg_reward in (-5.0, 15.0, 25.0, 90.0):
        assert training.guided_fraction_for(avg_reward, 0.4, 0.0) == 0.0


def test_maybe_update_guided_fraction_assigns_and_holds():
    fraction = tf.Variable(0.0, dtype=tf.float32)
    training.maybe_update_guided_fraction(15.0, 0.4, 0.5, fraction)
    assert float(fraction.numpy()) == 0.0

    training.maybe_update_guided_fraction(25.0, 0.4, 0.5, fraction)
    assert round(float(fraction.numpy()), 6) == 0.5

    # float32 round-trip must not make the comparison flap, the way it did for epsilon at 0.2.
    training.maybe_update_guided_fraction(25.0, 0.4, 0.5, fraction)
    assert round(float(fraction.numpy()), 6) == 0.5

    training.maybe_update_guided_fraction(1.0, 0.4, 0.5, fraction)
    assert float(fraction.numpy()) == 0.0


# ------------------------------------------------- the mask reads the right observations

def test_the_shield_reads_the_body_and_wall_block():
    # Indices 6-8 are "is the move safe", per state_helpers.get_observations' layout table, and
    # 1 means safe. Reading the wrong block, or the wrong polarity, would steer the snake
    # *into* walls while every other test above still passed.
    assert (shielded_policy.SAFETY_OBS_START, shielded_policy.SAFETY_OBS_END) == (6, 9)

    policy = make_policy([9.0, 1.0, 1.0], epsilon=1.0, guided_fraction=1.0)
    # Only index 7 set — action 1 is the safe one. Wrong offset or inverted polarity picks
    # something else.
    values = np.zeros((1, OBS_SIZE), dtype=np.float32)
    values[0, 7] = 1.0
    step = ts.restart(tf.constant(values), batch_size=1)
    assert set(int(policy.action(step).action.numpy()[0]) for _ in range(200)) == {1}


def test_the_shield_ignores_neighbouring_observation_blocks():
    # The blocks either side of 6-8 are food distance (0-5) and reach-the-tail (9-14). A shield
    # that read one index off would treat those as safety and produce nonsense; both are set
    # here so an off-by-one is unambiguous rather than merely unlucky.
    policy = make_policy([9.0, 1.0, 1.0], epsilon=1.0, guided_fraction=1.0)
    values = np.zeros((1, OBS_SIZE), dtype=np.float32)
    values[0, 0:6] = 1.0
    values[0, 9:15] = 1.0
    values[0, SAFE + 2] = 1.0
    step = ts.restart(tf.constant(values), batch_size=1)
    assert set(int(policy.action(step).action.numpy()[0]) for _ in range(200)) == {2}
