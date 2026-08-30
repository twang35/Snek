"""The rollout collector: what it stores, when it stores it, and what it bootstraps off.

Driven by stubs rather than by `VecSnake` and a real net, because all three claims here are about
*which* observation reaches *which* slot, and a stub can label an observation with the step it came
from. The real pairing of an action with the log-prob and value of the forward pass that produced it
is asserted in `tests/test_ppo_agent.py`, where the net is real.

Two of these would be silent in a run:

- **Storing the post-step observation** trains `V(s_t)` and `pi(a|s_t)` on `s_{t+1}`. Every metric
  stays plausible and the arm simply never learns.
- **Bootstrapping off the last stored value** instead of the state after it shortens every rollout's
  horizon by one step and biases the value target — worth ~0 at T=128 and everything at T=1.
"""

import numpy as np
import pytest

from ppo import collect as collect_module
from ppo import rollout as rollout_module

OBS_LEN = 3
DISCOUNT = 0.9
LAMBDA = 0.5


class StubVec(object):
    """N lanes whose observation is stamped with the step number it was produced at.

    `observation t` is `[t, t, t]`, so an assertion can name the step a stored row came from. `done`
    and `perfect` are scripted per step, which is what lets the counters be checked exactly.
    """

    def __init__(self, n, dones=None, perfect=None, reward=1.0):
        self.n = n
        self.t = 0
        self.dones = dones or {}
        self.perfect = perfect or {}
        self.reward = reward
        self.actions_seen = []

    def _obs(self):
        return np.full((self.n, OBS_LEN), float(self.t), dtype=np.float32)

    def reset_all(self):
        self.t = 0
        return self._obs()

    def observe(self):
        return self._obs()

    def step(self, actions):
        self.actions_seen.append(np.asarray(actions).copy())
        done = np.zeros(self.n, dtype=bool)
        done[:] = False
        for lane in self.dones.get(self.t, ()):
            done[lane] = True
        perfect = np.zeros(self.n, dtype=bool)
        for lane in self.perfect.get(self.t, ()):
            perfect[lane] = True
        self.t += 1
        rewards = np.full(self.n, self.reward, dtype=np.float32)
        return self._obs(), rewards, done, {'perfect': perfect}


class StubAgent(object):
    """`V(obs) = obs[:, 0]` — the step stamp — so a bootstrap can be traced to the state it came from.

    The action is the lane index modulo 3 and the log-prob is its negative, both arbitrary and both
    distinguishable, so a field written into the wrong slot is visible.
    """

    discount = DISCOUNT
    gae_lambda = LAMBDA

    def __init__(self):
        self.value_calls = []

    def act(self, observations):
        observations = np.asarray(observations)
        lanes = len(observations)
        actions = np.arange(lanes, dtype=np.int64) % 3
        return actions, -actions.astype(np.float32), self.values(observations)

    def values(self, observations):
        observations = np.asarray(observations, dtype=np.float32)
        self.value_calls.append(observations[:, 0].copy())
        return observations[:, 0].copy()


def collector(steps=3, lanes=2, **vec_kwargs):
    vec = StubVec(lanes, **vec_kwargs)
    agent = StubAgent()
    roll = rollout_module.Rollout(steps, lanes, OBS_LEN)
    return collect_module.Collector(vec, agent, roll), vec, agent, roll


# --- which observation goes where ----------------------------------------------------------------

def test_the_stored_observation_is_the_one_the_action_was_chosen_from():
    """`s_t`, not `s_{t+1}`. Off by one here and the arm learns a policy for the state it left."""
    coll, _, _, roll = collector(steps=3, lanes=2)
    coll.collect()
    for t in range(3):
        assert roll.obs[t] == pytest.approx(np.full((2, OBS_LEN), float(t))), t


def test_the_stored_value_is_v_of_the_stored_observation():
    coll, _, _, roll = collector(steps=3, lanes=2)
    coll.collect()
    assert roll.values[:, 0] == pytest.approx([0.0, 1.0, 2.0])


def test_the_reward_and_done_stored_at_t_are_the_outcome_of_the_action_at_t():
    """A death at step 1 must be stamped on row 1, or GAE gates the wrong step."""
    coll, _, _, roll = collector(steps=3, lanes=2, dones={1: (0,)})
    coll.collect()
    assert roll.dones[:, 0].tolist() == [False, True, False]
    assert roll.dones[:, 1].tolist() == [False, False, False]


# --- the bootstrap -------------------------------------------------------------------------------

def test_the_bootstrap_is_v_of_the_state_after_the_last_stored_step():
    """**The mutant this file was written for.**

    `rollout.values[-1]` is `V(s_{T-1})`, which is already stored and looks like the right thing.
    The value GAE needs is `V(s_T)` — the state the *next* rollout starts from — and it is not any
    state the loop scored, which is why it is fetched after the loop. Measured as a survivor of
    `tests/mut_ppo.json` before this fixture existed.
    """
    coll, _, agent, roll = collector(steps=3, lanes=2)
    coll.collect()
    # The last `values` call is the bootstrap, and it saw observation 3 — one past the last stored.
    assert agent.value_calls[-1] == pytest.approx([3.0, 3.0])
    assert roll.values[-1] == pytest.approx([2.0, 2.0]), 'the stored one is a different number'
    # And the advantage at the last step carries the bootstrap, not the stored value.
    expected = 1.0 + DISCOUNT * 3.0 - 2.0
    assert roll.advantages[-1, 0] == pytest.approx(expected, abs=1e-5)


def test_a_terminal_last_step_ignores_the_bootstrap_entirely():
    """The gate `rollout.py` owns, exercised through the collector so the wiring is covered too."""
    coll, _, _, roll = collector(steps=2, lanes=1, dones={1: (0,)})
    coll.collect()
    assert roll.advantages[-1, 0] == pytest.approx(1.0 - 1.0, abs=1e-5)


def test_gae_has_run_by_the_time_collect_returns():
    # The trainer calls `update` straight after, and a rollout read before `finish` raises.
    coll, _, _, roll = collector()
    coll.collect()
    roll.flat()


# --- the counters --------------------------------------------------------------------------------

def test_the_transition_count_is_exactly_steps_times_lanes():
    """Nothing in a PPO rollout is conditional, which is what makes a step a game move."""
    coll, _, _, _ = collector(steps=4, lanes=8, dones={0: (0, 1), 2: (3,)})
    assert coll.collect() == 32
    assert coll.counters['transitions'] == 32
    assert coll.counters['rollouts'] == 1
    assert coll.collect() == 32
    assert coll.counters['transitions'] == 64


def test_episodes_and_perfect_games_are_counted_per_lane():
    coll, _, _, _ = collector(steps=3, lanes=4, dones={0: (0, 1), 2: (2,)},
                              perfect={0: (1,), 2: (2,)})
    coll.collect()
    assert coll.counters['episodes'] == 3
    assert coll.counters['perfect_games'] == 2


def test_the_snapshot_is_a_copy_and_not_the_live_counters():
    # It goes into an eval row that is written later; a live reference would keep growing.
    coll, _, _, _ = collector()
    coll.collect()
    snapshot = coll.snapshot()
    coll.collect()
    assert snapshot['rollouts'] == 1


# --- the wiring ----------------------------------------------------------------------------------

def test_a_lane_count_mismatch_is_refused_at_construction():
    """Numpy would broadcast a `(2,)` into a `(4,)` row without complaint in some shapes, so the
    check is at construction rather than at the first `add`."""
    vec = StubVec(2)
    roll = rollout_module.Rollout(3, 4, OBS_LEN)
    with pytest.raises(ValueError, match='lanes'):
        collect_module.Collector(vec, StubAgent(), roll)


def test_the_second_rollout_continues_from_where_the_first_ended():
    """No reset at a rollout boundary: a rollout boundary is not an episode boundary."""
    coll, vec, _, roll = collector(steps=2, lanes=1)
    coll.collect()
    coll.collect()
    assert roll.obs[0] == pytest.approx(np.full((1, OBS_LEN), 2.0))
    assert vec.t == 4
