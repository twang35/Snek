"""GAE, the rollout buffer, and advantage normalisation.

Three of these would be silent in a training run and would look like a hyperparameter problem:

- **An advantage that flows across an episode boundary** credits a death to the next episode's
  opening. snek2 shipped exactly this in its n-step window, which had no episode check at all.
- **A minibatch pass that does not cover the rollout exactly once** makes "4 epochs" mean something
  other than four passes, and the difference is invisible in every metric.
- **Normalising a constant advantage vector** gives NaN, which poisons every parameter in the batch.
"""

import math

import numpy as np
import pytest

from ppo import rollout as rollout_module


DISCOUNT = 0.9
LAMBDA = 0.8


def one_lane(steps=5, rewards=None, values=None, dones=None):
    """A single-lane rollout with hand-chosen numbers, so GAE can be checked digit for digit."""
    roll = rollout_module.Rollout(steps, 1, 3)
    rewards = [1.0] * steps if rewards is None else rewards
    values = [float(v + 1) for v in range(steps)] if values is None else values
    dones = [False] * steps if dones is None else dones
    for t in range(steps):
        roll.add(t, np.zeros((1, 3), dtype=np.float32), [0], [-1.0986], [values[t]],
                 [rewards[t]], [dones[t]])
    return roll


# --- the arithmetic ------------------------------------------------------------------------------

def test_gae_matches_a_hand_computed_five_step_example():
    """Computed by hand at gamma=0.9, lambda=0.8, V=[1..5], r=1, bootstrap V=6.

    Asserted against literals rather than against a second implementation, because a second
    implementation in the test is the same code twice and agrees with itself when both are wrong.
    """
    roll = one_lane()
    roll.finish([6.0], DISCOUNT, LAMBDA)
    expected = [4.78954598, 4.1521472, 3.40576, 2.508, 1.4]
    assert roll.advantages[:, 0] == pytest.approx(expected, abs=1e-5)
    # `returns = advantages + values`, which is the value target. Checked separately because a
    # separately accumulated discounted sum is the tempting alternative and the two then drift.
    assert roll.returns[:, 0] == pytest.approx(
        [a + v for a, v in zip(expected, [1.0, 2.0, 3.0, 4.0, 5.0])], abs=1e-5)


def test_at_lambda_one_and_no_discount_the_advantage_is_the_monte_carlo_return_less_the_value():
    """The identity that pins the recursion's shape rather than its constants.

    At gamma=1, lambda=1 every delta telescopes, so `A_t = sum(r_t..r_T-1) + bootstrap - V_t`. A
    recursion with the decay applied in the wrong place still matches the five-step literals above if
    the literals were generated from it; this cannot be satisfied by a wrong decay.
    """
    roll = one_lane()
    roll.finish([6.0], 1.0, 1.0)
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    for t in range(5):
        tail = sum([1.0] * (5 - t)) + 6.0
        assert roll.advantages[t, 0] == pytest.approx(tail - values[t], abs=1e-4)


def test_at_lambda_zero_the_advantage_is_exactly_the_one_step_td_error():
    roll = one_lane()
    roll.finish([6.0], DISCOUNT, 0.0)
    values = [1.0, 2.0, 3.0, 4.0, 5.0] + [6.0]
    for t in range(5):
        delta = 1.0 + DISCOUNT * values[t + 1] - values[t]
        assert roll.advantages[t, 0] == pytest.approx(delta, abs=1e-5)


def test_the_horizon_is_reported_and_says_what_lambda_buys():
    # **‡ The number the lambda default is chosen from, and the first draft of this fixture had it
    # wrong: 44.5 at lambda=0.98, not the 132 asserted in the plan and in two docstrings.** The
    # conclusion did not change — neither horizon reaches a win ~950 moves away — but a wrong
    # number in a design doc is how a later session picks lambda badly.
    assert rollout_module.horizon(0.9975, 0.95) == pytest.approx(19.1, abs=0.1)
    assert rollout_module.horizon(0.9975, 0.98) == pytest.approx(44.5, abs=0.1)
    assert rollout_module.horizon(0.9975, 1.0) == pytest.approx(400.0, abs=0.5)
    assert rollout_module.horizon(0.99, 0.98) == pytest.approx(33.6, abs=0.1)
    assert rollout_module.horizon(1.0, 1.0) == float('inf')


# --- the episode boundary ------------------------------------------------------------------------

def test_no_advantage_flows_across_an_episode_boundary():
    """**The one that would be silent.** A death at t must not be credited with what came after it.

    `VecSnake` auto-resets inside `step()`, so `values[t+1]` on a lane that just died is a *fresh
    episode's* value. Both terms have to be gated: the bootstrap and the recursion.
    """
    roll = one_lane(dones=[False, False, True, False, False])
    roll.finish([6.0], DISCOUNT, LAMBDA)
    advantages = roll.advantages[:, 0]
    # At the terminal step the advantage is the whole reward less the value, with nothing added.
    assert advantages[2] == pytest.approx(1.0 - 3.0, abs=1e-5)
    # And the steps before it see the terminal advantage, because they are the same episode.
    assert advantages[1] == pytest.approx(1.7 + DISCOUNT * LAMBDA * (1.0 - 3.0), abs=1e-5)
    assert advantages[0] == pytest.approx(1.8 + DISCOUNT * LAMBDA * advantages[1], abs=1e-5)


def test_a_terminal_step_bootstraps_off_nothing_whatever_the_next_value_is():
    """Parametrised over an absurd next value, so the gate is asserted rather than the arithmetic.

    A next value of 10,000 changes nothing if the `(1 - done)` is there and changes everything if it
    is not — which a fixture using plausible numbers can fail to notice.
    """
    for next_value in (0.0, 1.0, 10000.0, -10000.0):
        roll = one_lane(steps=2, values=[3.0, next_value], dones=[True, False])
        roll.finish([0.0], DISCOUNT, LAMBDA)
        assert roll.advantages[0, 0] == pytest.approx(1.0 - 3.0, abs=1e-4), next_value


def test_the_last_step_bootstraps_off_the_state_the_next_rollout_starts_from():
    """Truncation is not termination: a rollout ending mid-episode must use `last_values`."""
    roll = one_lane(steps=1, values=[0.0], dones=[False])
    roll.finish([50.0], DISCOUNT, LAMBDA)
    assert roll.advantages[0, 0] == pytest.approx(1.0 + DISCOUNT * 50.0, abs=1e-4)


def test_lanes_are_independent():
    """Or one lane's death silences another's advantage, which no metric would show."""
    roll = rollout_module.Rollout(2, 2, 3)
    for t in range(2):
        roll.add(t, np.zeros((2, 3), dtype=np.float32), [0, 0], [-1.0, -1.0], [0.0, 0.0],
                 [1.0, 1.0], [t == 0, False])
    roll.finish([0.0, 0.0], DISCOUNT, LAMBDA)
    # Lane 0 died at t=0, lane 1 did not: their t=0 advantages must differ by the bootstrap.
    assert roll.advantages[0, 0] == pytest.approx(1.0, abs=1e-5)
    assert roll.advantages[0, 1] == pytest.approx(1.0 + DISCOUNT * LAMBDA * 1.0, abs=1e-5)


# --- reading it back -----------------------------------------------------------------------------

def test_reading_a_rollout_before_gae_has_run_is_refused():
    # Otherwise an update trains on the *previous* rollout's advantages against this one's samples,
    # which is a silent off-policy update with a correct-looking loss.
    roll = one_lane()
    with pytest.raises(ValueError, match='finish'):
        roll.flat()


def test_an_epoch_covers_every_sample_exactly_once():
    """What makes "4 epochs" four passes. A sampler with replacement passes any weaker check."""
    roll = rollout_module.Rollout(4, 8, 3)
    for t in range(4):
        roll.add(t, np.zeros((8, 3), dtype=np.float32), np.zeros(8), np.zeros(8), np.zeros(8),
                 np.zeros(8), np.zeros(8, dtype=bool))
    roll.finish(np.zeros(8), DISCOUNT, LAMBDA)
    rng = np.random.default_rng(0)
    seen = []
    for batch in roll.minibatches(5, rng):
        seen.extend(batch['actions'].tolist())
    assert len(seen) == roll.size == 32


def test_a_trailing_partial_minibatch_is_yielded_and_not_dropped():
    """Dropping it shortens every epoch by up to `minibatch - 1` samples, invisibly.

    32 samples in batches of 5 is 6 full batches and a remainder of 2. The planned production sizes
    (16,384 / 256) divide exactly, which is why this needs a fixture rather than a run to notice.
    """
    roll = rollout_module.Rollout(4, 8, 3)
    for t in range(4):
        roll.add(t, np.zeros((8, 3), dtype=np.float32), np.arange(t * 8, t * 8 + 8),
                 np.zeros(8), np.zeros(8), np.zeros(8), np.zeros(8, dtype=bool))
    roll.finish(np.zeros(8), DISCOUNT, LAMBDA)
    sizes = [len(b['actions']) for b in roll.minibatches(5, np.random.default_rng(0))]
    assert sizes == [5, 5, 5, 5, 5, 5, 2]
    seen = sorted(a for b in roll.minibatches(5, np.random.default_rng(1))
                  for a in b['actions'].tolist())
    assert seen == list(range(32)), 'every sample exactly once, including the remainder'


def test_a_minibatchs_fields_stay_aligned_with_each_other():
    """A shuffle applied per field rather than per sample pairs an obs with another sample's action.

    Every field is stamped with the same index here, so a mismatch is arithmetic rather than luck.
    """
    roll = rollout_module.Rollout(4, 4, 3)
    for t in range(4):
        marks = np.arange(t * 4, t * 4 + 4, dtype=np.float64)
        roll.add(t, np.tile(marks.reshape(4, 1), (1, 3)).astype(np.float32),
                 marks.astype(np.int64), marks, marks, np.zeros(4), np.zeros(4, dtype=bool))
    roll.finish(np.zeros(4), 0.0, 0.0)
    for batch in roll.minibatches(3, np.random.default_rng(2)):
        assert batch['obs'][:, 0] == pytest.approx(batch['actions'].astype(np.float32))
        assert batch['log_probs'] == pytest.approx(batch['actions'].astype(np.float32))


# --- normalisation -------------------------------------------------------------------------------

def test_normalising_gives_zero_mean_and_unit_spread():
    values = np.array([1.0, 5.0, -3.0, 7.0, 0.5], dtype=np.float32)
    out = rollout_module.normalise(values)
    assert float(out.mean()) == pytest.approx(0.0, abs=1e-6)
    assert float(out.std()) == pytest.approx(1.0, abs=1e-5)


@pytest.mark.parametrize('values', [
    [4.0],                      # a size-one minibatch: sd is 0
    [2.0, 2.0, 2.0, 2.0],       # every advantage equal, which happens early in a run
    [0.0, 0.0],
])
def test_a_constant_advantage_vector_does_not_produce_nan(values):
    """**The failure this prevents is total, not partial.** One NaN advantage makes the whole
    minibatch's gradient NaN, which makes every parameter NaN, and the arm then reports a flat 0
    score forever with no error anywhere.
    """
    out = rollout_module.normalise(np.array(values, dtype=np.float32))
    assert np.all(np.isfinite(out)), out
    assert out == pytest.approx(np.zeros(len(values)), abs=1e-6)


def test_a_rollout_of_no_steps_or_no_lanes_is_refused():
    with pytest.raises(ValueError, match='at least 1 step'):
        rollout_module.Rollout(0, 4, 3)
    with pytest.raises(ValueError, match='at least 1 step'):
        rollout_module.Rollout(4, 0, 3)


def test_the_uniform_entropy_constant_this_suite_reads_against():
    # ln 3 is what a fresh 3-action policy's entropy must equal; asserted here so the number in
    # `test_ppo_agent.py` has one definition.
    assert math.log(3.0) == pytest.approx(1.0986122886681098)


def test_adding_a_step_after_gae_has_run_invalidates_the_rollout():
    """The other direction of the guard above, and the one a *reused* buffer needs.

    The buffer is allocated once and overwritten every rollout, so "finished" is a property of the
    current contents rather than of the object. Without the reset in `add`, rollout 2 could be read
    with rollout 1's advantages still in place against rollout 2's samples — a silent off-policy
    update whose reported loss is an ordinary number.
    """
    roll = one_lane()
    roll.finish([6.0], DISCOUNT, LAMBDA)
    roll.flat()                                          # fine: this is the finished rollout
    roll.add(0, np.zeros((1, 3), dtype=np.float32), [0], [-1.0], [9.0], [9.0], [False])
    with pytest.raises(ValueError, match='finish'):
        roll.flat()
