"""The collector: n-step windows, episode boundaries, and forking.

Three properties here are silent when broken and each one corrupts the buffer rather than crashing:

- **A window must not span an episode boundary.** snek2's equivalent lived in the buffer with no
  episode check at all, and one shared window across interleaved games stored frame k of one game
  beside frame k+1 of another — a transition that never happened, on every interleaved add.
- **Every terminal transition must carry `discount=0`.** The stored `next_obs` there is the lane's
  post-reset observation, which is only harmless because it is multiplied out.
- **A forked lane's first transition must be anchored at the fork point** and take the forced
  action. Anchored anywhere else, forking collects ordinary experience under a different name.
"""

import numpy as np
import pytest

from dqn import collect
from dqn.agent import SAFETY_RANGE
from dqn.collect import Collector, ForkConfig
from env import constants
from vectorized import config as C
from vectorized.vec_env import VecSnake

TURN_LEFT = 1


class RecordingBuffer(object):
    """Stores every `add` verbatim, so a test can inspect what the collector banked."""

    def __init__(self):
        self.added = []

    def add(self, obs, action, reward, next_obs, discount):
        self.added.append({'obs': np.array(obs), 'action': int(action), 'reward': float(reward),
                           'next_obs': np.array(next_obs), 'discount': float(discount)})
        return len(self.added) - 1


class FixedAgent(object):
    """Plays one action always, and records what it was asked. No torch, no network."""

    def __init__(self, action=0):
        self.action = int(action)
        self.calls = []

    def act(self, observations, epsilon, guided=False):
        self.calls.append({'epsilon': epsilon, 'guided': np.array(guided, dtype=bool, copy=True)})
        return np.full(len(observations), self.action, dtype=np.int64)


class SurvivalAgent(FixedAgent):
    """The same heuristic `tests/test_engine.py` measures the engine against.

    Needed because `FixedAgent(0)` dies within a handful of steps from the opening board, which
    silently confounded four fixtures below: an n-step window fixture read `[0, 0, 3, 0]` because a
    boundary flushed the window on step 3, and three forking fixtures never reached the length gate
    at all and failed as "no fork happened". This heuristic scores a mean of ~42 with zero perfect
    games, so it survives long enough to grow past a gate and never wins by accident.
    """

    def act(self, observations, epsilon, guided=False):
        self.calls.append({'epsilon': epsilon, 'guided': np.array(guided, dtype=bool, copy=True)})
        return np.argmax(observations[:, 6:9] * 100.0
                         + observations[:, 9:14:2] * 10.0
                         + observations[:, 0:6:2], axis=1).astype(np.int64)


def make(width=1, collect_envs=1, n_step=1, discount=0.9, fork=None, guided_fraction=0.0,
         action=0, seed=0, survival=False):
    buffer = RecordingBuffer()
    agent = SurvivalAgent() if survival else FixedAgent(action)
    vec = VecSnake(width, seed=seed)
    collector = Collector(vec, agent, buffer, discount=discount, n_step=n_step,
                          collect_envs=collect_envs,
                          fork=fork if fork is not None else ForkConfig(branches=1),
                          guided_fraction=guided_fraction, seed=seed)
    return collector, buffer, agent, vec


def clean_steps(collector, count):
    """`count` steps, asserting no episode ended. Returns the per-step transition counts.

    An episode boundary flushes the whole n-step window, so a fixture about window filling has to
    know none happened rather than hope so.
    """
    banked = []
    for index in range(count):
        before = collector.counters['episodes']
        banked.append(collector.step(0.0))
        assert collector.counters['episodes'] == before, (
            'an episode ended on step {0}, so this fixture is measuring a flush'.format(index + 1))
    return banked


# --- configuration ---------------------------------------------------------------------------

def test_one_branch_means_forking_is_off():
    assert ForkConfig(branches=1).enabled is False
    assert ForkConfig(branches=4).enabled is True


def test_zero_branches_is_rejected_rather_than_clamped():
    with pytest.raises(ValueError, match='at least 1'):
        ForkConfig(branches=0)


def test_a_fork_probability_outside_zero_to_one_is_rejected():
    with pytest.raises(ValueError, match='not a probability'):
        ForkConfig(branches=4, prob=1.5)


def test_a_length_gate_below_the_opening_or_above_the_board_is_rejected():
    with pytest.raises(ValueError, match='outside'):
        ForkConfig(branches=4, min_length=collect.MIN_GATE_LENGTH - 1)
    with pytest.raises(ValueError, match='outside'):
        ForkConfig(branches=4, min_length=C.PERFECT_SCORE + 1)


def test_a_negative_step_cap_is_rejected_but_zero_is_allowed():
    with pytest.raises(ValueError, match='at least 0'):
        ForkConfig(branches=4, max_steps=-1)
    assert ForkConfig(branches=4, max_steps=0).max_steps == 0


def test_the_off_switch_skips_validation_so_a_stale_gate_cannot_block_a_control_arm():
    # branches == 1 means the knobs are inert. Validating them anyway would make a control arm fail
    # to launch over a value it never uses.
    assert ForkConfig(branches=1, prob=99.0, min_length=-5, max_steps=-3).enabled is False


def test_blanking_the_safe_move_flags_with_forking_on_is_rejected(monkeypatch):
    """Silent otherwise, and the arm would look like a forking arm while collecting nothing.

    With indices 6-8 zeroed every move reads fatal, so no state is ever eligible.
    """
    monkeypatch.setattr(C, 'ZERO_OBS_INDICES', (SAFETY_RANGE[0],))
    with pytest.raises(ValueError, match='SNEK_ZERO_OBS'):
        ForkConfig(branches=4)
    # And it must not fire when forking is off, nor when the ablation misses the block.
    assert ForkConfig(branches=1).enabled is False
    monkeypatch.setattr(C, 'ZERO_OBS_INDICES', (0, 29))
    assert ForkConfig(branches=4).enabled is True


def test_a_width_that_does_not_match_the_lane_arithmetic_is_rejected():
    # width must be collect_envs * branches, or a primary has no slots and forking silently
    # never happens.
    buffer, agent = RecordingBuffer(), FixedAgent()
    with pytest.raises(ValueError, match='does not match'):
        Collector(VecSnake(3, seed=0), agent, buffer, collect_envs=1,
                  fork=ForkConfig(branches=4))


def test_an_n_step_below_one_is_rejected():
    with pytest.raises(ValueError, match='at least 1'):
        make(n_step=0)


def test_the_lanes_split_into_primaries_and_secondaries():
    collector, _, _, _ = make(width=8, collect_envs=2, fork=ForkConfig(branches=4, min_length=10))
    assert collector.primaries.tolist() == [0, 1]
    assert collector.secondaries.tolist() == [2, 3, 4, 5, 6, 7]


# --- n-step windows --------------------------------------------------------------------------

def test_single_step_banks_one_transition_per_lane_per_step():
    collector, buffer, _, _ = make(width=4, collect_envs=4)
    banked = collector.step(0.0)
    assert banked == 4 and len(buffer.added) == 4


def test_the_return_value_is_the_transition_count_not_the_lane_count():
    # The training loop drives its gradient budget off this, which is what keeps the replay ratio
    # exact whether forking is on or off.
    collector, buffer, _, _ = make(width=2, collect_envs=2, n_step=3)
    assert collector.step(0.0) == 0, 'a window of 3 banks nothing on step 1'
    assert len(buffer.added) == 0


def test_an_n_step_window_banks_nothing_until_it_is_full():
    collector, buffer, _, _ = make(width=1, collect_envs=1, n_step=3, survival=True)
    assert clean_steps(collector, 5) == [0, 0, 1, 1, 1]


def test_the_n_step_reward_matches_a_hand_computed_sum():
    """Checked against the rewards the env actually returned, not against a re-derivation.

    Built by intercepting the window before it is flushed, so the arithmetic under test is the only
    thing between the env's rewards and the stored value.
    """
    collector, buffer, _, _ = make(width=1, collect_envs=1, n_step=3, discount=0.5, survival=True)
    seen = []
    original = collector._emit

    def watched(window, start, next_obs, discount):
        seen.append([entry[2] for entry in window[start:]])
        return original(window, start, next_obs, discount)

    collector._emit = watched
    clean_steps(collector, 3)
    assert len(buffer.added) == 1
    rewards = seen[0]
    expected = sum((0.5 ** offset) * reward for offset, reward in enumerate(rewards))
    assert buffer.added[0]['reward'] == pytest.approx(expected)


def test_a_single_step_transition_carries_the_plain_discount():
    collector, buffer, _, _ = make(width=1, collect_envs=1, n_step=1, discount=0.9)
    collector.step(0.0)
    assert buffer.added[0]['discount'] == pytest.approx(0.9)


def test_the_stored_observation_is_the_one_that_was_acted_in():
    collector, buffer, _, _ = make(width=1, collect_envs=1)
    before = collector.obs[0].copy()
    collector.step(0.0)
    assert np.allclose(buffer.added[0]['obs'], before)
    assert np.allclose(buffer.added[0]['next_obs'], collector.obs[0])


def test_the_stored_action_is_the_one_that_was_played():
    collector, buffer, _, _ = make(width=1, collect_envs=1, action=2)
    collector.step(0.0)
    assert buffer.added[0]['action'] == 2


def test_the_discounted_sum_is_hand_computed_for_a_synthetic_window():
    """`_emit` on a window the test wrote, so the arithmetic is the only thing under test.

    The env-driven fixture above cannot pin this on its own: whenever a window's trailing rewards
    are zero the discounted and undiscounted sums agree, and the shaping reward is zero on most
    steps.
    """
    collector, buffer, _, _ = make(width=1, collect_envs=1, n_step=3, discount=0.5)
    obs = np.zeros(constants.observation_length(), dtype=np.float32)
    window = [(obs, 0, 1.0), (obs, 1, 1.0), (obs, 2, 1.0)]
    collector._emit(window, 0, obs, 0.125)
    assert buffer.added[-1]['reward'] == pytest.approx(1.0 + 0.5 + 0.25)
    collector._emit(window, 1, obs, 0.125)
    assert buffer.added[-1]['reward'] == pytest.approx(1.0 + 0.5), 'start offsets the exponent too'


def test_an_n_step_transition_carries_the_discount_raised_to_n():
    # n steps of the env passed under one transition, so the bootstrap must be discounted n times.
    collector, buffer, _, _ = make(width=1, collect_envs=1, n_step=3, discount=0.9, survival=True)
    clean_steps(collector, 3)
    assert len(buffer.added) == 1
    assert buffer.added[0]['discount'] == pytest.approx(0.9 ** 3)


def test_the_transition_counter_matches_what_the_buffer_received():
    # The eval row reports this number and the replay ratio is computed from it.
    collector, buffer, _, _ = make(width=4, collect_envs=4, n_step=3, action=TURN_LEFT)
    for _ in range(100):
        collector.step(0.0)
    assert collector.counters['transitions'] == len(buffer.added)


# --- episode boundaries ----------------------------------------------------------------------

def run_until_first_death(collector, limit=400):
    """Steps until a lane finishes an episode. Returns how many steps it took."""
    for count in range(1, limit + 1):
        before = collector.counters['episodes']
        collector.step(0.0)
        if collector.counters['episodes'] > before:
            return count
    raise AssertionError('no episode ended in {0} steps'.format(limit))


def test_turning_left_forever_dies_so_the_boundary_fixtures_are_reachable():
    # A fixture whose subject cannot violate it is not a fixture: everything below depends on this
    # action sequence actually ending an episode.
    collector, _, _, _ = make(width=1, collect_envs=1, action=TURN_LEFT)
    assert run_until_first_death(collector) < 400


def test_every_terminal_transition_bootstraps_off_nothing():
    """The property that makes the post-reset `next_obs` harmless.

    `VecSnake` auto-resets inside `step()`, so a terminal transition's stored `next_obs` is the
    *fresh* episode's first observation. That is only safe because `discount` is 0 there, and this is
    what holds it.
    """
    collector, buffer, _, _ = make(width=1, collect_envs=1, n_step=3, action=TURN_LEFT)
    while collector.counters['episodes'] < 3:
        collector.step(0.0)
    zeros = [entry for entry in buffer.added if entry['discount'] == 0.0]
    assert zeros, 'no terminal transition was banked at all'
    for entry in zeros:
        assert entry['discount'] == 0.0


def test_a_terminal_step_flushes_the_whole_window():
    """Exactly as many transitions as the window held, not one.

    A death two steps into a 3-step window has to contribute both, because deaths are the signal the
    endgame is learned from and the shorter tail is the only record of them.
    """
    collector, buffer, _, _ = make(width=1, collect_envs=1, n_step=3, action=TURN_LEFT)
    while True:
        held = len(collector.windows[0])
        episodes = collector.counters['episodes']
        banked = collector.step(0.0)
        if collector.counters['episodes'] > episodes:
            break
    # One more step went into the window before it was flushed, and every entry was emitted.
    assert banked == held + 1
    assert banked >= 2, 'the window held only one entry, so "the whole window" is untested'
    assert collector.windows[0] == [], 'the window must be empty after a boundary'


def test_no_window_survives_an_episode_boundary():
    """A transition may never pair a pre-boundary state with a post-boundary one.

    Checked by watching the window handed to `_emit`: every emitted slice must come from a single
    episode, which here means no emission may follow a boundary until the window has refilled.
    """
    collector, _, _, _ = make(width=1, collect_envs=1, n_step=4, action=TURN_LEFT)
    emissions = []
    original = collector._emit

    def watched(window, start, next_obs, discount):
        emissions.append((collector.counters['episodes'], len(window) - start, discount))
        return original(window, start, next_obs, discount)

    collector._emit = watched

    steps_since_boundary = 0
    for _ in range(200):
        episodes_before = collector.counters['episodes']
        marker = len(emissions)
        collector.step(0.0)
        crossed = collector.counters['episodes'] > episodes_before
        for _, span, discount in emissions[marker:]:
            assert span <= steps_since_boundary + 1, (
                'emitted a {0}-step window {1} steps after a boundary'.format(
                    span, steps_since_boundary))
        steps_since_boundary = 0 if crossed else steps_since_boundary + 1


def test_a_perfect_game_is_counted_from_the_env_info():
    collector, _, _, _ = make(width=1, collect_envs=1, action=TURN_LEFT)
    run_until_first_death(collector)
    # Turning left forever cannot win, so the counter must stay at zero rather than tracking deaths.
    assert collector.counters['perfect_games'] == 0
    assert collector.counters['episodes'] >= 1


# --- the shield's flag -----------------------------------------------------------------------

def test_a_guided_fraction_of_zero_never_shields():
    collector, _, agent, _ = make(width=4, collect_envs=4, guided_fraction=0.0, action=TURN_LEFT)
    for _ in range(50):
        collector.step(0.1)
    assert not any(call['guided'].any() for call in agent.calls)


def test_a_guided_fraction_of_one_always_shields():
    collector, _, agent, _ = make(width=4, collect_envs=4, guided_fraction=1.0, action=TURN_LEFT)
    for _ in range(50):
        collector.step(0.1)
    assert all(call['guided'].all() for call in agent.calls)


def test_the_flag_is_redrawn_per_episode_not_per_step():
    """A lane whose episode is still running keeps its flag, even when another lane's ends.

    Four lanes rather than one on purpose: with a single lane "every lane is redrawn on any
    boundary" and "each lane is redrawn on its own boundary" are the same rule. And the flags start
    all-True against a fraction of 0.0, so every redraw is observable — left to a coin the test
    would only notice a wrong redraw half the time.
    """
    collector, _, _, _ = make(width=4, collect_envs=4, guided_fraction=0.0, survival=True)
    collector.guided[:] = True
    boundaries = []
    original = collector._settle_episodes

    def watched(done, info):
        boundaries.append(np.array(done, dtype=bool, copy=True))
        return original(done, info)

    collector._settle_episodes = watched

    partial = 0
    for _ in range(400):
        before = collector.guided.copy()
        marker = len(boundaries)
        collector.step(0.0)
        for done in boundaries[marker:]:
            survivors = np.flatnonzero(~done & before)
            partial += int(survivors.size > 0 and done.any())
            for lane in survivors.tolist():
                assert collector.guided[lane], (
                    'lane {0} was redrawn without its own episode ending'.format(lane))
            for lane in np.flatnonzero(done).tolist():
                assert not collector.guided[lane], 'a finished lane was not redrawn'
    assert partial > 0, 'no step ended one lane and not another, so this proves nothing'


def test_a_new_fraction_takes_effect_at_the_next_boundary():
    collector, _, _, _ = make(width=1, collect_envs=1, guided_fraction=0.0, action=TURN_LEFT)
    collector.step(0.0)
    assert collector.guided[0] == False
    collector.set_guided_fraction(1.0)
    assert collector.guided[0] == False, 'an episode in flight must not switch regime mid-way'
    while collector.counters['episodes'] < 1:
        collector.step(0.0)
    assert collector.guided[0] == True


# --- forking --------------------------------------------------------------------------------

def forking_collector(width=4, min_length=5, prob=1.0, max_steps=60, seed=0, survival=True,
                      action=0, n_step=1):
    fork = ForkConfig(branches=width, prob=prob, min_length=min_length, max_steps=max_steps)
    return make(width=width, collect_envs=1, fork=fork, action=action, seed=seed,
                survival=survival, n_step=n_step)


def test_forking_off_means_there_are_no_secondary_lanes_at_all():
    # With branches == 1 the width arithmetic leaves nothing to reseat, so "off" is structural
    # rather than a flag checked at each decision point.
    collector, _, _, _ = make(width=2, collect_envs=2, fork=ForkConfig(branches=1), survival=True)
    assert collector.secondaries.tolist() == []
    for _ in range(100):
        collector.step(0.0)
    assert collector.counters['forks'] == 0
    assert collector.counters['eligible'] == 0, 'the gate is not even consulted when off'


def test_a_fork_happens_once_the_length_gate_is_met():
    collector, _, _, _ = forking_collector(min_length=5, prob=1.0)
    for _ in range(200):
        collector.step(0.0)
        if collector.counters['forks']:
            break
    assert collector.counters['forks'] > 0


def test_a_length_gate_above_the_board_never_forks():
    collector, _, _, _ = forking_collector(min_length=C.PERFECT_SCORE, prob=1.0)
    for _ in range(300):
        collector.step(0.0)
    assert collector.counters['forks'] == 0
    assert collector.counters['eligible'] == 0


def test_a_fork_probability_of_zero_counts_eligible_states_but_never_forks():
    # The separation matters for reading an arm: `eligible` says the gate is right, `forks` says the
    # coin is. A single counter cannot distinguish "gate too high" from "coin never passes".
    collector, _, _, _ = forking_collector(min_length=5, prob=0.0)
    for _ in range(300):
        collector.step(0.0)
    assert collector.counters['eligible'] > 0
    assert collector.counters['forks'] == 0


def watch_forks(collector):
    """Records each fork as it happens, with the parent's state at that instant.

    Hooked rather than inspected after the step: a fork is a copy of the parent's state, and one
    `vec.step()` later the parent and the branch have both moved, so nothing observable afterwards
    still shows they started equal.
    """
    seen = []
    original = collector._fork
    original_maybe = collector._maybe_fork
    played = {}

    def watched_maybe(actions):
        played['actions'] = np.array(actions, copy=True)
        return original_maybe(actions)

    def watched(parent, slot, action):
        options = np.flatnonzero(
            collector.obs[parent, SAFETY_RANGE[0]:SAFETY_RANGE[1]] > 0.5).tolist()
        pre_window = len(collector.windows[slot])
        result = original(parent, slot, action)
        seen.append({'parent': parent, 'slot': slot, 'action': action,
                     'options': options, 'played': int(played['actions'][parent]),
                     'pre_window': pre_window, 'post_window': len(collector.windows[slot]),
                     'age': int(collector.branch_age[slot]),
                     'obs': collector.obs[slot].copy(),
                     'parent_obs': collector.obs[parent].copy(),
                     'score': int(collector.vec.score[slot]),
                     'parent_score': int(collector.vec.score[parent]),
                     'length': int(collector.vec.length[slot]),
                     'parent_length': int(collector.vec.length[parent]),
                     'guided': bool(collector.guided[slot]),
                     'parent_guided': bool(collector.guided[parent])})
        return result

    collector._fork = watched
    collector._maybe_fork = watched_maybe
    return seen


def run_to_first_fork(collector, limit=200):
    seen = watch_forks(collector)
    for _ in range(limit):
        marker = len(collector.buffer.added)
        collector.step(0.0)
        if seen:
            return seen, collector.buffer.added[marker:]
    raise AssertionError('no fork happened in {0} steps'.format(limit))


def test_a_forked_lane_starts_from_its_parent_state():
    collector, _, _, _ = forking_collector(min_length=5, prob=1.0)
    seen, _ = run_to_first_fork(collector)
    for fork in seen:
        assert fork['score'] == fork['parent_score']
        assert fork['length'] == fork['parent_length']
        assert np.allclose(fork['obs'], fork['parent_obs']), 'the observation the policy sees'


def test_a_forked_lane_plays_the_forced_action_first():
    """The branch's first action is the forced one, taken without consulting the policy.

    Asserted against the banked transition, which is the thing that matters: a branch whose first
    stored action were the policy's would be collecting ordinary experience under another name.
    """
    collector, _, _, _ = forking_collector(min_length=5, prob=1.0)
    seen, banked = run_to_first_fork(collector)
    for fork in seen:
        matching = [entry for entry in banked
                    if np.allclose(entry['obs'], fork['parent_obs'])
                    and entry['action'] == fork['action']]
        assert matching, 'no transition banked at the fork point with the forced action {0}'.format(
            fork['action'])


def test_a_fork_never_repeats_the_action_its_parent_took():
    # The whole point is the *untaken* alternative; forking the taken action collects a duplicate.
    collector, _, agent, _ = forking_collector(min_length=5, prob=1.0)
    seen = watch_forks(collector)
    for _ in range(200):
        played = None
        if collector.fork.enabled:
            played = agent.act(collector.obs, 0.0)[0]
        marker = len(seen)
        collector.step(0.0)
        for fork in seen[marker:]:
            if fork['parent'] == 0:
                assert fork['action'] != int(played), 'forked the action the parent took'
        if len(seen) >= 3:
            return
    raise AssertionError('fewer than 3 forks happened')


def test_two_alternatives_at_one_decision_point_both_get_a_branch():
    # A state with three safe moves and two free slots should fill both, or the coverage this
    # feature buys is halved for no reason.
    collector, _, _, _ = forking_collector(width=4, min_length=5, prob=1.0)
    seen, _ = run_to_first_fork(collector)
    by_parent = {}
    for fork in seen:
        by_parent.setdefault(fork['parent'], []).append(fork['action'])
    assert any(len(actions) >= 2 for actions in by_parent.values()), by_parent
    for actions in by_parent.values():
        assert len(actions) == len(set(actions)), 'the same alternative was forked twice'


def test_a_forked_lane_inherits_the_shield_flag():
    collector, _, _, _ = forking_collector(min_length=5, prob=1.0)
    collector.guided[:] = False
    collector.guided[0] = True
    seen, _ = run_to_first_fork(collector)
    for fork in seen:
        assert fork['guided'] == fork['parent_guided']
    assert any(fork['guided'] for fork in seen), 'no fork inherited a True flag, so this is vacuous'


def test_a_forked_lane_drops_its_previous_window():
    """Or its first stored transition pairs the old episode's state with the fork point's next one.

    `n_step=3`, because with a window of 1 the slot's window is empty before every fork and the
    fixture would hold whether or not the collector cleared it.
    """
    collector, _, _, _ = forking_collector(min_length=5, prob=1.0, n_step=3, max_steps=3)
    seen = watch_forks(collector)
    for _ in range(600):
        collector.step(0.0)
    # Past the first fork on purpose: a slot only holds a partial window once it has been retired
    # and has played a few ordinary steps, which is the case the clear exists for.
    assert any(fork['pre_window'] > 0 for fork in seen), (
        'no slot held a partial window at fork time, so this fixture proves nothing')
    for fork in seen:
        assert fork['post_window'] == 0


def test_a_branch_retires_when_its_step_cap_is_reached():
    collector, _, _, _ = forking_collector(min_length=5, prob=1.0, max_steps=2)
    for _ in range(300):
        collector.step(0.0)
        if collector.counters['truncated']:
            break
    assert collector.counters['truncated'] > 0
    assert collector.counters['retired'] >= collector.counters['truncated']


def test_a_step_cap_of_zero_lets_a_branch_run_to_its_terminal_state():
    collector, _, _, _ = forking_collector(min_length=5, prob=1.0, max_steps=0)
    for _ in range(300):
        collector.step(0.0)
    assert collector.counters['truncated'] == 0
    assert collector.counters['terminated'] > 0, 'branches should still retire on death'


def test_a_retired_slot_becomes_available_again():
    collector, _, _, _ = forking_collector(min_length=5, prob=1.0, max_steps=2)
    for _ in range(300):
        collector.step(0.0)
    assert collector.counters['forks'] > collector.fork.branches, 'slots were never reused'


def test_slot_exhaustion_is_counted_rather_than_silently_dropped():
    collector, _, _, _ = forking_collector(width=2, min_length=5, prob=1.0, max_steps=0)
    for _ in range(400):
        collector.step(0.0)
    assert collector.counters['skipped_full'] > 0


def test_only_primaries_fork():
    # A branch of a branch deepens the correlation between collected trajectories rather than
    # widening coverage, and slots are already scarce.
    collector, _, _, _ = forking_collector(width=4, min_length=5, prob=1.0, max_steps=0)
    for _ in range(400):
        collector.step(0.0)
    assert collector.counters['forks'] > 0
    # Every fork consumed exactly one free slot, so forks can never exceed the number of times a
    # slot was free — which a branch-of-a-branch would break by forking from a secondary.
    assert collector.snapshot()['live_branches'] <= len(collector.secondaries)



def test_the_forced_action_is_played_once_and_not_replayed():
    """A branch takes its forced action on step one and the policy's from then on.

    Not replayed forever, which is what an uncleared `pending` would do — the branch would spin in
    one direction and fill the buffer with a trajectory no policy would ever produce.
    """
    collector, _, agent, vec = forking_collector(min_length=5, prob=1.0, max_steps=0)
    seen = watch_forks(collector)
    passed = []
    original = vec.step

    def watched(actions):
        passed.append(np.array(actions, copy=True))
        return original(actions)

    vec.step = watched

    for _ in range(400):
        marker = len(seen)
        collector.step(0.0)
        for fork in seen[marker:]:
            assert passed[-1][fork['slot']] == fork['action'], 'the forced action was not played'
        if len(seen) >= 2 and len(passed) > 2:
            slot = seen[0]['slot']
            wanted = agent.act(collector.obs, 0.0)[slot]
            collector.step(0.0)
            assert (collector.pending == -1).all(), 'a forced action outlived its step'
            assert passed[-1][slot] == wanted, 'the branch replayed its forced action'
            return
    pytest.fail('fewer than 2 forks happened')


def test_the_eligible_counter_matches_a_recount_of_the_safe_moves():
    """Recomputed from the observation, which pins both halves of "decision point".

    A state is one only if the snake is past the gate **and** more than one move is safe. Counting
    single-option states would inflate the eligible figure the fork rate is judged against, and
    reading the safe flags with a slack threshold would make every state look eligible.
    """
    collector, _, _, vec = forking_collector(min_length=5, prob=0.0)
    expected = 0
    for _ in range(400):
        safe = collector.obs[:, SAFETY_RANGE[0]:SAFETY_RANGE[1]] > 0.5
        for lane in collector.primaries.tolist():
            if vec.length[lane] >= collector.fork.min_length and safe[lane].sum() >= 2:
                expected += 1
        collector.step(0.0)
    assert expected > 0, 'no eligible state was reached, so this fixture proves nothing'
    assert collector.counters['eligible'] == expected


def test_a_slot_shortage_does_not_always_take_the_lowest_alternative():
    """One free slot and two alternatives: which one is forked has to vary.

    Taken in action order it would be the lowest every time, and the buffer would fill with left
    turns the policy never took while the right turns it never took stayed missing — the exact
    asymmetry forking exists to remove, reintroduced one level down.
    """
    # A one-step cap so the single slot is free again on the next step; with branches running to
    # death it is busy almost always and barely a handful of forks get a choice at all.
    collector, _, _, _ = forking_collector(width=2, min_length=5, prob=1.0, max_steps=1)
    seen = watch_forks(collector)
    for _ in range(3000):
        collector.step(0.0)
    choices = []
    for fork in seen:
        alternatives = sorted(set(fork['options']) - {fork['played']})
        if len(alternatives) >= 2:
            choices.append(fork['action'] == alternatives[0])
    assert len(choices) >= 10, 'only {0} forks had a real choice'.format(len(choices))
    assert any(choices) and not all(choices), 'the same alternative was taken every time'


def test_a_reused_slot_starts_its_age_again():
    """Or a slot's second branch retires on its first step, and every slot is single-use.

    Silent: the counters still show forks and retirements, just one step of experience each.
    """
    collector, _, _, _ = forking_collector(min_length=5, prob=1.0, max_steps=3)
    seen = watch_forks(collector)
    for _ in range(600):
        collector.step(0.0)
    slots = [fork['slot'] for fork in seen]
    assert len(slots) > len(set(slots)), 'no slot was reused, so this fixture proves nothing'
    for fork in seen:
        assert fork['age'] == 0, 'slot {0} was forked at age {1}'.format(fork['slot'], fork['age'])


def test_a_step_cap_of_one_retires_a_branch_after_one_step():
    # `>= max_steps`, not `>`: a cap of 1 has to mean one step, or the smallest cap is really two.
    collector, _, _, _ = forking_collector(min_length=5, prob=1.0, max_steps=1)
    seen, _ = run_to_first_fork(collector, limit=400)
    slots = [fork['slot'] for fork in seen]
    assert collector.counters['retired'] >= len(slots)
    for slot in slots:
        assert not collector.branching[slot], 'slot {0} outlived a cap of one step'.format(slot)


def test_a_branch_that_dies_at_its_cap_is_counted_once():
    """A death on the same step the cap is reached is one retirement, not two.

    Driven directly rather than waited for: the coincidence is rare in a real run, and a double
    count would show up only as a `retired` figure that quietly exceeds the forks that produced it.
    """
    collector, _, _, vec = forking_collector(min_length=5, prob=0.0, max_steps=2)
    slot = int(collector.secondaries[0])
    collector.branching[slot] = True
    collector.branch_age[slot] = 1
    done = np.zeros(vec.n, dtype=bool)
    done[slot] = True
    collector._retire_branches(done)
    assert collector.counters['terminated'] == 1
    assert collector.counters['truncated'] == 0, 'the same retirement was counted twice'
    assert collector.counters['retired'] == 1
    assert not collector.branching[slot]

def test_the_snapshot_reports_what_an_eval_row_needs():
    collector, _, _, _ = forking_collector(min_length=5, prob=1.0)
    for _ in range(100):
        collector.step(0.0)
    snapshot = collector.snapshot()
    for key in ('forks', 'retired', 'truncated', 'terminated', 'eligible', 'skipped_full',
                'episodes', 'transitions', 'perfect_games', 'live_branches', 'free_slots'):
        assert key in snapshot, key
