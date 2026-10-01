"""The stateful-policy seam: `begin(rows, fresh, prev_reward)` on the engine, `StatefulPolicy` behind it,
and the two scalar entry points' loops (`plans/algoExploration/e-memory.md` §1).

What would be silent without these:

- **a lane that migrates to a new job keeps the old job's state**, which reads as noise across a pass;
- **a reset lane keeps its state across the episode boundary**, so every episode after the first opens
  with a memory of a different game;
- **the previous reward never reaches the policy**, so R2D2's net sees a constant input it was trained on
  varying -- silent, because the engine measures and the shard writes rows either way;
- **a plain callable is treated differently** once the protocol exists.

The policies here count steps per lane rather than running a net, so none of this needs torch except
`StatefulPolicy` itself, which holds its state as tensors.
"""

import numpy as np
import pytest
import torch

from algos.stateful import StatefulPolicy, one_hot_previous
from vectorized import engine


class CountingPolicy(object):
    """A stateful policy that plays a survival heuristic and records, per lane, the steps since its reset,
    the rewards it was told, and every `fresh` flag."""

    def __init__(self, width=4096):
        self.since_reset = np.zeros(width, dtype=np.int64)
        self.max_since_reset = np.zeros(width, dtype=np.int64)
        self.rewards_seen = []
        self.fresh_seen = 0
        self.calls = 0
        self._rows = None

    def begin(self, rows, fresh, prev_reward):
        rows = np.asarray(rows)
        fresh = np.asarray(fresh, dtype=bool)
        self.since_reset[rows[fresh]] = 0
        self.fresh_seen += int(fresh.sum())
        self.rewards_seen.append((rows.copy(), np.asarray(prev_reward, dtype=np.float32).copy(), fresh.copy()))
        self._rows = rows

    def __call__(self, obs):
        assert self._rows is not None, 'the engine must call begin() before every call'
        rows = self._rows
        self._rows = None
        self.calls += 1
        self.since_reset[rows] += 1
        self.max_since_reset[rows] = np.maximum(self.max_since_reset[rows], self.since_reset[rows])
        return np.argmax(obs[:, 6:9] * 100.0 + obs[:, 9:14:2] * 10.0 + obs[:, 0:6:2], axis=1)


def survival_policy(obs):
    return np.argmax(obs[:, 6:9] * 100.0 + obs[:, 9:14:2] * 10.0 + obs[:, 0:6:2], axis=1)


def _run(jobs, episodes, width, seed=0):
    queue = list(jobs)
    out = {}
    stats = engine.measure_stream(lambda: queue.pop(0) if queue else None,
                                  lambda key, held: out.__setitem__(key, held),
                                  episodes, width=width, seed=seed)
    return out, stats


# --- the engine's half -----------------------------------------------------------------------------

def test_a_stateful_policy_is_told_every_episode_start_and_counts_steps_to_its_lengths():
    """Steps since reset, per lane, agree with the episode lengths the engine reports: the engine marks a
    lane fresh at the opening assignment and after every completion, and never in between."""
    policy = CountingPolicy()
    out, stats = _run([('c', policy)], episodes=12, width=6, seed=3)
    assert len(out['c']['scores']) == 12
    # Every started episode began with a fresh flag: 12 episodes, 12 fresh flags (6 opening + 6 refills).
    assert policy.fresh_seen == 12
    # Every policy call was preceded by a begin.
    assert policy.calls == stats['steps']


def test_the_previous_reward_handed_over_is_the_lanes_own_last_reward_and_zero_when_fresh():
    policy = CountingPolicy()
    _run([('c', policy)], episodes=8, width=4, seed=1)
    for rows, rewards, fresh in policy.rewards_seen:
        assert rewards.shape == rows.shape
        assert np.all(rewards[fresh] == 0.0)
    # Something non-zero was handed over at some point: the step penalty and food rewards are not 0.
    assert any(np.any(rewards != 0.0) for _, rewards, _ in policy.rewards_seen)


def test_a_lane_that_migrates_to_another_job_starts_fresh_there():
    """Two stateful jobs share the lanes; a lane handed from one to the other arrives fresh."""
    first, second = CountingPolicy(), CountingPolicy()
    out, _ = _run([('a', first), ('b', second)], episodes=6, width=4, seed=2)
    assert len(out['a']['scores']) == 6 and len(out['b']['scores']) == 6
    # Each job saw exactly as many fresh flags as episodes it played.
    assert first.fresh_seen == 6 and second.fresh_seen == 6


def test_a_plain_callable_is_measured_exactly_as_before():
    """The protocol is duck-typed on `begin`; a function has none and the engine's numbers do not move."""
    held_plain = engine.measure(survival_policy, 20, lanes=10, seed=5)
    stateful = CountingPolicy()
    held_stateful = engine.measure(stateful, 20, lanes=10, seed=5)
    assert held_plain['scores'] == held_stateful['scores']
    assert held_plain['rewards'] == pytest.approx(held_stateful['rewards'])


def test_the_single_checkpoint_helper_runs_a_stateful_policy_too():
    policy = CountingPolicy()
    held = engine.measure(policy, 10, lanes=5, seed=4)
    assert len(held['scores']) == 10
    assert policy.fresh_seen == 10


# --- StatefulPolicy --------------------------------------------------------------------------------

def _echo_step(observations, prev_action, prev_reward, state):
    """A step function whose action is the row's step count so far (mod 3) and whose state counts steps,
    carries the previous reward in its second column and the previous action in its third."""
    count = state[:, 0] + 1.0
    new_state = torch.stack([count, prev_reward, prev_action.to(torch.float32)], dim=1)
    return (count.to(torch.int64) % 3), new_state


def test_state_persists_across_calls_and_is_zeroed_for_fresh_rows():
    policy = StatefulPolicy(_echo_step, state_width=3)
    obs = np.zeros((3, 4), dtype=np.float32)
    policy.begin([0, 1, 2], [True, True, True], [0.0, 0.0, 0.0])
    assert policy(obs).tolist() == [1, 1, 1]
    policy.begin([0, 1, 2], [False, False, False], [0.5, 0.0, 0.0])
    assert policy(obs).tolist() == [2, 2, 2]
    # Row 1 reset: its count restarts; the others carry on.
    policy.begin([0, 1, 2], [False, True, False], [0.0, 0.0, 0.0])
    assert policy(obs).tolist() == [0, 1, 0]
    assert policy.state[:, 0].tolist() == [3.0, 1.0, 3.0]


def test_the_previous_action_and_reward_reach_the_step_function_and_are_cleared_on_fresh():
    policy = StatefulPolicy(_echo_step, state_width=3)
    obs = np.zeros((2, 4), dtype=np.float32)
    policy.begin([0, 1], [True, True], [0.0, 0.0])
    policy(obs)                                         # actions [1, 1]
    policy.begin([0, 1], [False, True], [2.5, 9.0])
    policy(obs)
    # Row 0: previous action 1 and reward 2.5 were fed. Row 1 was fresh: no previous action (-1) and
    # the reward it was handed is overridden to 0.
    assert policy.state[0, 1].item() == pytest.approx(2.5)
    assert policy.state[0, 2].item() == pytest.approx(1.0)
    assert policy.state[1, 1].item() == pytest.approx(0.0)
    assert policy.state[1, 2].item() == pytest.approx(-1.0)


def test_rows_are_absolute_lane_indexes_and_the_arrays_grow_to_fit():
    policy = StatefulPolicy(_echo_step, state_width=3)
    obs = np.zeros((2, 4), dtype=np.float32)
    policy.begin([7, 2], [True, True], [0.0, 0.0])
    policy(obs)
    policy.begin([7], [False], [0.0])
    assert policy(obs[:1]).tolist() == [2]
    assert policy.state.shape[0] == 8
    assert policy.state[2, 0].item() == 1.0 and policy.state[7, 0].item() == 2.0


def test_a_call_without_begin_runs_rows_in_order_fresh_once():
    """The fallback for a hand loop: rows 0..m-1, fresh on the first call only, reward 0."""
    policy = StatefulPolicy(_echo_step, state_width=3)
    obs = np.zeros((2, 4), dtype=np.float32)
    assert policy(obs).tolist() == [1, 1]
    assert policy(obs).tolist() == [2, 2]


def test_a_begin_whose_row_count_disagrees_with_the_call_is_refused():
    policy = StatefulPolicy(_echo_step, state_width=3)
    policy.begin([0, 1], [True, True], [0.0, 0.0])
    with pytest.raises(ValueError, match='named 2 rows'):
        policy(np.zeros((3, 4), dtype=np.float32))
    with pytest.raises(ValueError, match='one fresh flag'):
        policy.begin([0, 1], [True], [0.0, 0.0])


def test_one_hot_previous_is_the_zero_vector_for_none():
    out = one_hot_previous(torch.tensor([-1, 0, 2]), 3)
    assert out.tolist() == [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
